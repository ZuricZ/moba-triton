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
import subprocess
import time
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


# Fraction of total VRAM a config may be estimated to need before it is
# skipped. A fraction rather than a constant: the same sweep has to run on a
# 4 GB laptop and an 80 GB A100 without editing a number, and what matters is
# leaving room for the allocator, not an absolute size.
MEMORY_BUDGET_FRACTION = 0.75


def total_memory_mb(device=None) -> float:
    """Total VRAM on the device, in MiB."""
    return torch.cuda.get_device_properties(device).total_memory / BYTES_PER_MIB


def default_memory_budget_mb(device=None) -> float:
    """Memory a config may need before the sweep skips it.

    Was a hardcoded 3000.0, which is 75% of this laptop's 4 GB and 3.7% of an
    80 GB A100 -- so on rented hardware it would have skipped almost the entire
    interesting sweep while looking like it ran fine.
    """
    return MEMORY_BUDGET_FRACTION * total_memory_mb(device)


def default_seq_lens(device=None, longest: int = 131072) -> list[int]:
    """Powers of two the device can plausibly hold, for the dense baselines.

    Sized off the *dense* backends, not ours. Attention memory for a fused
    dense kernel grows linearly in S but its compute grows quadratically, while
    our partial buffers grow linearly and stay small -- 64 MB at S=8192. So the
    limit on how far a comparison sweep can run is set by what the baselines
    survive, and a card that holds more should sweep further.

    Both references sweep powers of two; FlashMoBA runs 4K-512K at batch 2.

    Args:
        device: CUDA device to size against.
        longest: Upper bound regardless of memory, so a very large card does
            not silently start a sweep that takes hours.

    Returns:
        Ascending powers of two from 1024.
    """
    gib = total_memory_mb(device) / 1024
    # Empirical: S=8192 is comfortable on this 4 GB card, and the ceiling rises
    # roughly with memory. Conservative by design -- a skipped row is reported
    # with its reason, an OOM mid-sweep loses the run.
    ceiling = 8192 * max(1, int(gib // 4))
    ceiling = min(ceiling, longest)
    lens, s = [], 1024
    while s <= ceiling:
        lens.append(s)
        s *= 2
    return lens


COOLDOWN_TARGET_C = 76
COOLDOWN_CAP_S = 180


_CLOCKS_LOCKED = False


def lock_clocks() -> bool:
    """Pins the SM clock to its maximum, so timings stop depending on history.

    THE PROPER FIX, WHERE IT IS AVAILABLE. Everything below -- the cooldown,
    the short warm-up, best-of-3 -- exists because a laptop GPU's clock swings
    with its thermal and power state, so a measurement depends on whatever ran
    before it. A datacenter card in a container with the capability lets you
    remove the variable outright: lock the clock and every measurement is taken
    under identical conditions.

    Needs `nvidia-smi -lgc`, which requires elevated permissions and is
    normally refused on a consumer laptop under WDDM. Returns False there, and
    the thermal protocol carries the load instead.

    Returns:
        True if the clock is now pinned.
    """
    global _CLOCKS_LOCKED
    try:
        query = subprocess.run(
            ["nvidia-smi", "--query-gpu=clocks.max.sm", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        max_mhz = int(query.stdout.strip().splitlines()[0])
        locked = subprocess.run(
            ["nvidia-smi", "-lgc", f"{max_mhz},{max_mhz}"],
            capture_output=True, text=True, timeout=20,
        )
        _CLOCKS_LOCKED = locked.returncode == 0
        return _CLOCKS_LOCKED
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        return False


def unlock_clocks() -> None:
    """Returns the GPU to its own clock management. Safe to call unlocked."""
    global _CLOCKS_LOCKED
    if not _CLOCKS_LOCKED:
        return
    try:
        subprocess.run(["nvidia-smi", "-rgc"], capture_output=True, timeout=20)
    except (OSError, subprocess.SubprocessError):
        pass
    _CLOCKS_LOCKED = False


def clocks_are_locked() -> bool:
    """Whether this process pinned the clock; the thermal dance is then moot."""
    return _CLOCKS_LOCKED


def gpu_temperature() -> int | None:
    """Returns the GPU temperature in Celsius, or None if unreadable."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
        return int(out.stdout.strip().splitlines()[0])
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        return None


def cool_down_device(target_c: int = COOLDOWN_TARGET_C,
                     cap_s: float = COOLDOWN_CAP_S) -> None:
    """Idles until the GPU drops to `target_c`, or `cap_s` elapses.

    ON A SATURATED CARD, WARMING UP IS THE WRONG DIRECTION. `warm_up_device`
    exists because a cold GPU sits at a low clock and a short measurement can
    finish before it ramps. The opposite failure is just as large and was found
    the harder way: on this laptop at 84-86 C there is no headroom left, and
    further load *spends* what remains. Measured, `ours` at S=8192, three runs
    each, varying only the warm-up length:

        warm-up      runs (ms)                spread   temp
          2 s        1.583  1.612  1.610       1.02x    84
         10 s        5.414  5.858  5.614       1.08x    85
         20 s       10.847 14.810 11.495       1.37x    86

    Note the spread column: the card is *repeatable* in any given thermal
    state, just not comparable across states. Three runs agreeing within 1.37x
    around a 7x-wrong number is what makes this hard to catch -- tight error
    bars do not indicate a good measurement here.

    Cooling first restores it, and cheaply:

        after 20 s warm-up (cooked)   16.669 ms   86 C
        cooled 26 s to 76 C, warm 2 s  1.617 ms
        immediately again              1.585 ms
        another cooldown cycle         1.620 ms

    So the order is cool, then briefly warm, then measure. One short
    measurement does not re-cook the card; a provider running hundreds of
    milliseconds per rep does, which is why this runs before every one rather
    than once.

    A no-op where the temperature cannot be read -- a container without
    `nvidia-smi`, or a device that does not report one. Benchmarks still run;
    they are simply not protected.
    """
    if _CLOCKS_LOCKED or gpu_temperature() is None:
        return
    deadline = time.perf_counter() + cap_s
    while time.perf_counter() < deadline:
        current = gpu_temperature()
        if current is None or current <= target_c:
            return
        time.sleep(5)


def warm_up_device(seconds: float = 2.0) -> None:
    """Runs throwaway work until the GPU reaches its sustained clock.

    Without this the first backend measured absorbs the clock ramp, and this
    suite measures the dense baseline first for every config -- so the ramp
    landed on the denominator of every speedup it has ever reported. Measured
    at S=4096: the same dense call took 1.182 ms on the first pass and settled
    at 0.741-0.767 ms over the next four, a 1.59x spread. `ours` moved 1.10x
    across the same passes because by then the clocks were already up.

    `do_bench` warms the callable it is given, which is not the same thing: it
    compiles the kernel, allocates its buffers and fills caches, all of which
    belong to the code. The ramp belongs to the *device*.

    CALL THIS BEFORE EVERY MEASUREMENT, NOT ONCE PER PROCESS. The sentence
    above used to end "so only the first measurement in a process pays it",
    which is wrong: the clock is shared state and any neighbouring workload
    resets it. Measured at S=8192 against a cold-box 1.55 ms for `ours`:

        after a 30 s matmul burn, no re-warm       6.73 ms
        after 90 s idle, no re-warm               11.70 ms
        after 90 s idle, re-warmed                 1.55 ms
        repeated                                   1.55 ms

    A pure matmul burn touching no MoBA code reproduces it, so this is not
    something a particular backend leaks. Note the middle row: idling made it
    *worse* than hammering did, so there is no "rest the card" alternative --
    both extremes are bad and only a deliberate ramp lands in the middle.

    Not thermal, despite appearances. `nvidia-smi` reported no throttle flags
    and a byte-identical 1717 MHz / 84 C at three different moments, which
    under WSL2 is a stale reading rather than a stable card, and 180 s of extra
    idle made the number worse rather than better.
    """
    a = torch.randn((2048, 2048), device="cuda", dtype=torch.bfloat16)
    end = time.perf_counter() + seconds
    while time.perf_counter() < end:
        a = a @ a.T if a.shape[0] == a.shape[1] else a
        a = a / a.abs().amax().clamp_min(1e-6)
    torch.cuda.synchronize()


def _measure(backend, q, k, v, cfg) -> tuple[float, float]:
    """Times one backend and reports its peak memory.

    Warms the device itself, every time. The previous version warmed once per
    run and said so here; that is wrong for the reason `warm_up_device`
    records, and it biased whichever backend followed a slow one.

    Returns:
        A tuple of (median_ms, peak_mb).
    """
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    cool_down_device()
    warm_up_device()
    ms = triton.testing.do_bench(lambda: backend.forward(q, k, v, cfg))
    return ms, torch.cuda.max_memory_allocated() / BYTES_PER_MIB


def _selected(cfg, only):
    """Returns the available backends, narrowed to `only` when given."""
    backends = available_backends(cfg)
    if only is None:
        return backends
    return [b for b in backends if b.name in only]


def run_suite(
    shapes: list[tuple[int, int, int, int]],
    gates: list[tuple[int, int]],
    dtype: torch.dtype = torch.bfloat16,
    memory_budget_mb: float | None = None,
    only: set[str] | None = None,
) -> list[BenchRow]:
    """Benchmarks every available backend over the config sweep.

    The dense baseline is measured first so speedups can be filled in as each
    row is produced, rather than by rescanning results afterwards.

    Args:
        shapes: List of (batch, seq_len, heads, head_dim).
        gates: List of (block_size, top_k).
        dtype: Compute dtype.
        memory_budget_mb: Configs estimated above this are skipped. Defaults
            to a fraction of the card's own memory, so the same sweep runs on
            a 4 GB laptop and an 80 GB A100 -- the old hardcoded 3000.0 was
            75% of this laptop and 3.7% of an A100, which would have skipped
            almost every interesting row while appearing to succeed.
        only: Backend names to measure; every other registered backend is left
            out entirely rather than reported as skipped. The estimate above is
            per *config*, not per backend, so at S=65536 the dense reference's
            O(S^2) need refuses the sparse rows too -- and raising the budget
            instead lets the dense backends attempt a 68 GB allocation on a
            40 GB card, whose failure corrupts the CUDA context and takes the
            rest of the sweep with it. Naming the backends is the way to ask
            for the long-context rows without that.

    Returns:
        One BenchRow per (config, backend), including skipped and errored ones.
    """
    if memory_budget_mb is None:
        memory_budget_mb = default_memory_budget_mb()
    device_row = DeviceInfo.probe(0).as_row()
    if lock_clocks():
        print("clocks locked to max -- thermal protocol not needed")
    warm_up_device()
    rows: list[BenchRow] = []

    total = len(shapes) * len(gates)
    done = 0
    started = time.time()

    for shape in shapes:
        batch, seq_len, heads, head_dim = shape
        for gate in gates:
            block_size, top_k = gate
            # Progress, because a sweep that is merely slow and one that has
            # hung look identical from outside. Every measurement here is
            # preceded by `cool_down_device`, which spins until the card drops
            # to its threshold -- so an idle GPU at high host CPU is the
            # *expected* state between timings, not a symptom. A 48-cell sweep
            # at ~40 s a cell runs for the better part of an hour, and a
            # half-finished one was once killed for looking stuck.
            done += 1
            elapsed = time.time() - started
            rate = elapsed / max(1, done - 1) if done > 1 else 0.0
            eta = f", ~{rate * (total - done + 1) / 60:.0f} min left" if rate                 else ""
            print(f"[{done}/{total}] S={seq_len} block={block_size} "
                  f"top_k={top_k}{eta}", flush=True)
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
                for backend in _selected(cfg, only):
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
            backends = _selected(cfg, only)
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
