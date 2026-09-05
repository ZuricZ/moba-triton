"""Benchmark driver.

The headline number is not latency. It is `attainment`: the speedup actually
achieved over dense attention, divided by the speedup the sparsity pattern
implies is available. A kernel reaching 2x where sparsity offers 8x has an
attainment of 0.25, and closing that gap is the point of this project.

Reporting attainment rather than raw speedup matters because raw speedup can
be improved trivially by making the config sparser, which says nothing about
kernel quality. Attainment normalizes that away: it asks how much of the
theoretically available win the implementation actually collects.
"""

import math
from dataclasses import asdict, dataclass

import torch
import triton.testing

from implementations import available_backends, skipped_backends
from moba.config import MoBAConfig
from moba.device import BYTES_PER_MIB, DeviceInfo

DENSE_BASELINE = "sdpa_dense"

# Longest error text kept in a results row. Long enough for a CUDA message to
# stay diagnosable, short enough that one failure cannot swamp a CSV.
MAX_NOTE_CHARS = 160

# Sentinel top_k meaning "select every block", used to price the dense
# baseline's own FLOPs. Any value >= num_blocks works; this is unmistakable.
SELECT_ALL_TOP_K = 10**9


@dataclass(frozen=True)
class BenchRow:
    """One measurement, carrying enough metadata to merge across machines."""

    backend: str
    batch: int
    heads: int
    seq_len: int
    head_dim: int
    block_size: int
    top_k: int
    dtype: str
    latency_ms: float
    tflops: float
    peak_memory_mb: float
    sparsity: float
    speedup_vs_dense: float
    speedup_ceiling: float
    attainment: float
    gpu_name: str
    capability: str
    total_memory_mb: int
    torch_version: str
    triton_version: str
    cuda_version: str
    status: str
    note: str


def sparsity_ratio(seq_len: int, cfg: MoBAConfig) -> float:
    """Fraction of causally-legal keys that MoBA actually attends to.

    Computed in closed form rather than by materializing a mask, so it is
    cheap enough to call per benchmark row. Cross-checked against real
    selection masks in tests/test_bench.py.

    For query i: its own block contributes the causally-visible prefix
    `i - own_block*block_size + 1`, and each selected past block contributes
    all `block_size` of its keys.

    Args:
        seq_len: Sequence length.
        cfg: Configuration supplying `block_size` and `top_k`.

    Returns:
        A value in (0, 1]; 1.0 means no sparsity (dense causal attention).
    """
    bk = cfg.block_size
    num_blocks = cfg.num_blocks(seq_len)
    past_budget = cfg.past_blocks_selected(num_blocks)

    i = torch.arange(seq_len, dtype=torch.float64)
    own_block = torch.div(i, bk, rounding_mode="floor")
    within_own = i - own_block * bk + 1
    n_past = torch.clamp(own_block, max=float(past_budget))

    attended = within_own + n_past * bk
    causal = i + 1
    return (attended.sum() / causal.sum()).item()


def effective_tflops(
    batch: int,
    heads: int,
    seq_len: int,
    head_dim: int,
    cfg: MoBAConfig,
    ms: float,
) -> float:
    """FLOP/s counting only the attention arithmetic actually performed.

    Counts 2 FLOPs per multiply-accumulate, for both QK^T and PV, over
    attended key positions only. Softmax is excluded: it is O(S) per row
    against O(S*D) for the matmuls, and every implementation pays it.

    Args:
        batch: Batch size.
        heads: Number of heads.
        seq_len: Sequence length.
        head_dim: Head dimension.
        cfg: Configuration.
        ms: Measured latency in milliseconds.

    Returns:
        Effective TFLOP/s.
    """
    causal_pairs = seq_len * (seq_len + 1) / 2
    attended_pairs = causal_pairs * sparsity_ratio(seq_len, cfg)
    flops = 2 * 2 * batch * heads * attended_pairs * head_dim
    return flops / (ms * 1e-3) / 1e12


def estimate_memory_mb(
    batch: int,
    heads: int,
    seq_len: int,
    head_dim: int,
    dtype: torch.dtype,
    query_chunk: int | None = None,
) -> float:
    """Estimates peak memory for the naive oracle.

    The oracle's footprint is dominated by its logit matrix. With query
    chunking that is [query_chunk, S] per (batch, head) rather than [S, S],
    which is the difference between reaching S=8192 and not.

    Args:
        batch: Batch size.
        heads: Number of heads.
        seq_len: Sequence length.
        head_dim: Head dimension.
        dtype: Compute dtype.
        query_chunk: Query rows processed at once, or None for all.

    Returns:
        Estimated megabytes.
    """
    itemsize = torch.empty(0, dtype=dtype).element_size()
    rows = seq_len if query_chunk is None else min(query_chunk, seq_len)

    # logits + probs at `itemsize`, plus two boolean masks at 1 byte.
    per_element = 2 * itemsize + 2
    scores = batch * heads * rows * seq_len * per_element
    tensors = 4 * batch * heads * seq_len * head_dim * itemsize  # q, k, v, out
    return (scores + tensors) / BYTES_PER_MIB


def _blank_row(
    backend: str,
    shape: tuple[int, int, int, int],
    gate: tuple[int, int],
    dtype: torch.dtype,
    sparsity: float,
    ceiling: float,
    device_row: dict,
    status: str,
    note: str,
) -> BenchRow:
    """Builds a row for a measurement that did not happen."""
    batch, seq_len, heads, head_dim = shape
    block_size, top_k = gate
    return BenchRow(
        backend=backend, batch=batch, heads=heads, seq_len=seq_len,
        head_dim=head_dim, block_size=block_size, top_k=top_k,
        dtype=str(dtype).replace("torch.", ""),
        latency_ms=math.nan, tflops=math.nan, peak_memory_mb=math.nan,
        sparsity=sparsity, speedup_vs_dense=math.nan, speedup_ceiling=ceiling,
        attainment=math.nan, status=status, note=note, **device_row,
    )


def _measure(backend, q, k, v, cfg) -> tuple[float, float]:
    """Times one backend and reports its peak memory.

    Returns:
        A tuple of (median_ms, peak_mb).
    """
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    ms = triton.testing.do_bench(lambda: backend.forward(q, k, v, cfg))
    return ms, torch.cuda.max_memory_allocated() / BYTES_PER_MIB


def run_suite(
    shapes: list[tuple[int, int, int, int]],
    gates: list[tuple[int, int]],
    dtype: torch.dtype = torch.bfloat16,
    memory_budget_mb: float = 3000.0,
) -> list[BenchRow]:
    """Benchmarks every available backend over the config sweep.

    The dense baseline is measured first so speedups can be filled in as each
    row is produced, rather than by rescanning results afterwards.

    Args:
        shapes: List of (batch, seq_len, heads, head_dim).
        gates: List of (block_size, top_k).
        dtype: Compute dtype.
        memory_budget_mb: Configs estimated above this are skipped.

    Returns:
        One BenchRow per (config, backend), including skipped and errored ones.
    """
    device_row = DeviceInfo.probe(0).as_row()
    rows: list[BenchRow] = []

    for shape in shapes:
        batch, seq_len, heads, head_dim = shape
        for gate in gates:
            block_size, top_k = gate
            cfg = MoBAConfig(block_size=block_size, top_k=top_k, dtype=dtype)
            sparsity = sparsity_ratio(seq_len, cfg)
            ceiling = 1.0 / sparsity

            def blank(name, status, note):
                return _blank_row(
                    name, shape, gate, dtype, sparsity, ceiling,
                    device_row, status, note,
                )

            for name, reason in skipped_backends(cfg):
                rows.append(blank(name, "skipped", reason))

            estimate = estimate_memory_mb(
                batch, heads, seq_len, head_dim, dtype, query_chunk=None)
            if estimate > memory_budget_mb:
                for backend in available_backends(cfg):
                    rows.append(
                        blank(
                            backend.name, "skipped",
                            f"estimated {estimate:.0f} MB exceeds "
                            f"{memory_budget_mb:.0f} MB budget",
                        )
                    )
                continue

            g = torch.Generator(device="cuda").manual_seed(0)
            q, k, v = (
                torch.randn(shape, generator=g, device="cuda", dtype=dtype)
                for _ in range(3)
            )

            # Dense baseline first: every other row's speedup is relative to it.
            dense_ms = None
            backends = available_backends(cfg)
            for backend in backends:
                if backend.name != DENSE_BASELINE:
                    continue
                try:
                    dense_ms, peak = _measure(backend, q, k, v, cfg)
                except (RuntimeError, NotImplementedError) as exc:
                    rows.append(blank(backend.name, "error", str(exc)[:MAX_NOTE_CHARS]))
                    break
                rows.append(
                    BenchRow(
                        backend=backend.name, batch=batch, heads=heads,
                        seq_len=seq_len, head_dim=head_dim,
                        block_size=block_size, top_k=top_k,
                        dtype=str(dtype).replace("torch.", ""),
                        latency_ms=dense_ms,
                        # The baseline attends everything, so its own FLOPs are
                        # the dense count, not the sparse one.
                        tflops=effective_tflops(
                            batch, heads, seq_len, head_dim,
                            MoBAConfig(block_size=block_size, top_k=SELECT_ALL_TOP_K,
                                       dtype=dtype),
                            dense_ms,
                        ),
                        peak_memory_mb=peak, sparsity=1.0,
                        speedup_vs_dense=1.0, speedup_ceiling=1.0,
                        attainment=1.0, status="ok", note="dense roofline",
                        **device_row,
                    )
                )

            for backend in backends:
                if backend.name == DENSE_BASELINE:
                    continue
                try:
                    ms, peak = _measure(backend, q, k, v, cfg)
                except (RuntimeError, NotImplementedError, torch.OutOfMemoryError) as exc:
                    rows.append(blank(backend.name, "error", str(exc)[:MAX_NOTE_CHARS]))
                    continue
                speedup = math.nan if dense_ms is None else dense_ms / ms
                rows.append(
                    BenchRow(
                        backend=backend.name, batch=batch, heads=heads,
                        seq_len=seq_len, head_dim=head_dim,
                        block_size=block_size, top_k=top_k,
                        dtype=str(dtype).replace("torch.", ""),
                        latency_ms=ms,
                        # A dense entry attends to everything, so counting its
                        # FLOPs against the sparse pattern would credit it with
                        # arithmetic it never skipped and overstate its
                        # attainment. Same correction the roofline row makes.
                        tflops=effective_tflops(
                            batch, heads, seq_len, head_dim,
                            cfg if backend.computes_moba else MoBAConfig(
                                block_size=block_size, top_k=SELECT_ALL_TOP_K,
                                dtype=dtype),
                            ms,
                        ),
                        peak_memory_mb=peak, sparsity=sparsity,
                        speedup_vs_dense=speedup, speedup_ceiling=ceiling,
                        attainment=speedup / ceiling, status="ok", note="",
                        **device_row,
                    )
                )

            del q, k, v

    return rows
