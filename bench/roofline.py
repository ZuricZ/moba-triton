"""Where each implementation sits against this card's compute and memory limits.

WHY THIS MATTERS FOR THIS KERNEL SPECIFICALLY. The whole design rests on a
memory-traffic argument: a query-major kernel discards up to 88% of its
tensor-core work on MoBA's per-query routing, so we invert the loop and let a
program own a KV block. The price is that a query's output arrives in pieces
and needs `combine_partials`, a pass that is pure bandwidth. Until now we have
asserted that trade is worth it and observed that wall-clock agrees. A roofline
says *why*, and says which ceiling we are actually pinned against.

THE MODEL IS REQUESTED TRAFFIC, NOT MEASURED TRAFFIC. Every byte is counted
once per time the algorithm *asks* for it. That is not the same as DRAM
traffic, and it errs in different directions for different terms:

* **Terms too large to cache are accurate.** This card has 2 MiB of L2. The
  partials are 32 MiB each way at S=8192, so every one of those bytes really
  does cross the bus.
* **Re-read terms are overstated.** `ours` pulls each query row `top_k` times,
  but the whole of Q is only 4 MiB, so L2 absorbs some of it. Real Q traffic is
  at or below the modelled figure.
* **Dense kernels are understated.** A FlashAttention query tile re-reads
  earlier K/V tiles, which is quadratic in S and only partly cached; the four
  tensors are a floor, not the truth.

So the arrows point opposite ways for sparse and dense, which is exactly why
the model cannot settle the comparison on its own. `bench/profile.py` measures
DRAM traffic with Nsight Compute's hardware counters; the gap between modelled
and measured is itself the interesting number.

READ THE PLOT LIKE THIS. A point under the sloped roof is memory bound -- it
cannot go faster without moving fewer bytes. A point under the flat roof is
compute bound. Distance below the roof is headroom the implementation is not
collecting.
"""

import argparse
import dataclasses
import math
import time
from dataclasses import dataclass, field

import torch
import triton.testing

from bench import paths, plots
from bench.suite import (
    BYTES_PER_MIB,
    cool_down_device,
    lock_clocks,
    unlock_clocks,
    warm_up_device,
)
from implementations import all_backends, get_backend
from moba.config import MoBAConfig

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CFG = MoBAConfig(block_size=128, top_k=8, dtype=torch.bfloat16)
BATCH, HEADS, HEAD_DIM = 1, 4, 64
BYTES_PER_GIB = 1 << 30
FP32 = 4

# Big enough that kernel launch and cache effects are noise, small enough for
# a 4 GB card. Both ceilings are measured, never taken from a spec sheet: a
# laptop part rarely reaches its datasheet numbers, and a roofline drawn
# against an unreachable ceiling makes every implementation look bad by the
# same wrong factor.
BANDWIDTH_ELEMS = 1 << 26           # 64 Mi bf16 = 128 MiB per buffer
MATMUL_N = 4096

# Datasheet figures, by architecture. These are the *audit*, never the roof:
# a measured GEMM at 60-70% of spec is ordinary, and a measurement outside that
# band means the harness is wrong rather than the card being unusual. Drawing
# the roof here instead would fold a 35% gap nobody can close into every
# implementation's score.
#
# Dense rates, not the "with sparsity" headline: NVIDIA quotes A100 bf16 as
# "312 TFLOPS | 624 TFLOPS*" where the asterisk is 2:4 structured sparsity,
# which nothing here uses. Bandwidth is the 40 GB PCIe part's 1555 GB/s.
#
# Only architectures whose datasheet has actually been read appear. A missing
# entry prints no audit rather than a guessed one.
SPEC = {
    "sm_80": {"tflops_bf16": 312.0, "bandwidth_gbs": 1555.0},
}

# sm_90 is deliberately absent. Unlike Ampere, the architecture does not
# determine the board: an H100 SXM is 989 TFLOP/s over 3352 GB/s and an
# H100 PCIe is 756 over 2039, a 1.6x spread on the roof. Keying sm_90 to
# either would silently mis-audit the other, so Hopper resolves by name
# below and prints no audit when the name is unrecognised.

# The same figures reachable by device name, for `--from-log`: a log replotted
# on another machine must take its datasheet from the card that produced it,
# not from whatever is plugged into the machine doing the drawing.
SPEC_BY_NAME = {
    # Longest keys first: matching is by substring and the first hit
    # wins, so "H100 PCIe" must be tested before any bare "H100".
    "H100 PCIe": {"tflops_bf16": 756.5, "bandwidth_gbs": 2039.0},
    "H100 80GB HBM3": {"tflops_bf16": 989.4, "bandwidth_gbs": 3352.0},
    "H100 SXM": {"tflops_bf16": 989.4, "bandwidth_gbs": 3352.0},
    "A100": SPEC["sm_80"],
}

# Dense bf16 tensor-core rates, halved from NVIDIA's asterisked figures
# the way the A100 entry is -- their headline doubles for 2:4 structured
# sparsity, which nothing here uses. H100 NVL is absent on purpose: it is
# a dual-GPU board and its quoted rate is for the pair.


def device_spec(device=None, name: str | None = None) -> dict:
    """Returns datasheet figures for a card, or an empty dict.

    Args:
        device: Passed to `torch.cuda.get_device_capability`, when no `name`
            is given.
        name: A device name, as `--from-log` recovers from a stored run. Takes
            precedence, because the log's card is the one the numbers describe.

    Returns:
        A dict with "tflops_bf16" and "bandwidth_gbs", or {} when that
        architecture's datasheet is not recorded. Missing beats guessed: an
        absent entry prints no audit rather than a wrong one.
    """
    live = name is None and torch.cuda.is_available()
    if live:
        # The board, not just the architecture. Hopper needs this: one
        # sm_90 tag covers cards whose roofs differ by 1.6x.
        name = torch.cuda.get_device_name(device)
    if name:
        for key, spec in SPEC_BY_NAME.items():
            if key in name:
                return spec
        if not live:
            return {}
    if not torch.cuda.is_available():
        return {}
    major, minor = torch.cuda.get_device_capability(device)
    return SPEC.get(f"sm_{major}{minor}", {})


# ---------------------------------------------------------------------------
# Device ceilings
# ---------------------------------------------------------------------------


def measure_peak_bandwidth(reps: int = 30) -> float:
    """Achievable HBM bandwidth in bytes/second, measured by a large copy.

    A copy touches each byte exactly twice -- one read, one write -- so the
    traffic is unambiguous, which a triad or a reduction cannot claim. Uses
    `copy_` rather than a Triton kernel so the number is a property of the
    card rather than of our own code generation.

    Returns:
        Bytes per second at the median of `reps` timings.
    """
    src = torch.randn(BANDWIDTH_ELEMS, device="cuda", dtype=torch.bfloat16)
    dst = torch.empty_like(src)
    moved = 2 * src.numel() * src.element_size()
    cool_down_device()
    warm_up_device()
    ms = triton.testing.do_bench(lambda: dst.copy_(src), quantiles=[0.5, 0.2, 0.8])[0]
    return moved / (ms * 1e-3)


def measure_peak_tflops(dtype: torch.dtype = torch.bfloat16) -> float:
    """Achievable tensor-core throughput in FLOP/s, measured by a square GEMM.

    A large square matmul is the friendliest possible shape -- if the card
    cannot reach a rate here it will not reach it in an attention kernel, so
    this is the honest flat roof.

    Returns:
        FLOP per second.
    """
    a = torch.randn(MATMUL_N, MATMUL_N, device="cuda", dtype=dtype)
    b = torch.randn(MATMUL_N, MATMUL_N, device="cuda", dtype=dtype)
    flop = 2 * MATMUL_N ** 3
    cool_down_device()
    warm_up_device()
    ms = triton.testing.do_bench(lambda: a @ b, quantiles=[0.5, 0.2, 0.8])[0]
    return flop / (ms * 1e-3)


@dataclass(frozen=True)
class Ceilings:
    """This card's limits: what it is rated for, and what it actually reaches.

    THREE COMPUTE CEILINGS, NOT ONE, because "% of roof" is only useful when
    every gap inside it is one somebody could close. Stacked:

        spec_flops        what the silicon is rated for. Nobody reaches it.
        peak_flops        what a 4096^3 GEMM reaches -- the best any code does.
        attainable_flops  what this tile shape plus an online softmax reaches.

    On the A100 those are 312, 204 and 164 TFLOP/s, and the two gaps have
    different owners. 312 -> 204 is silicon against best-possible code: memory
    latency, scheduling, clocks -- nobody's fault and nobody's to fix. 204 ->
    164 is a GEMM against attention: the softmax runs on CUDA cores while the
    matmuls own the tensor cores, so it is the algorithm's price rather than
    this implementation's.

    Only the third is the roof a kernel is measured against, because only the
    distance below *it* is headroom anyone can chase. Drawn against 312, a 35%
    gap nobody can close is folded into our number, and the reading barely
    moves whether we optimise well or not at all.

    The other two still earn their place on the plot: they say *why* the roof
    sits where it does, and `spec_flops` audits the measurement. A measured GEMM
    at 65% of spec is ordinary; at 95% or 20% the harness is wrong, not the
    card. The datasheet is the check, never the line.

    Attributes:
        bandwidth: Measured HBM bandwidth, bytes/s.
        peak_flops: Measured large-GEMM throughput, FLOP/s.
        attainable_flops: Measured throughput at this kernel's tile shape with
            its softmax (`bench/tile_ceiling.py`). None until measured, in
            which case `peak_flops` is the roof and the plot says so.
        spec_bandwidth: Datasheet bandwidth, bytes/s, or None.
        spec_flops: Datasheet compute, FLOP/s, or None.
    """

    bandwidth: float                 # bytes/s
    peak_flops: float                # FLOP/s
    attainable_flops: float | None = None
    spec_bandwidth: float | None = None
    spec_flops: float | None = None

    @property
    def roof_flops(self) -> float:
        """The flat roof a kernel is actually measured against."""
        return self.attainable_flops or self.peak_flops

    @property
    def ridge(self) -> float:
        """Arithmetic intensity where the two roofs meet, in FLOP/byte.

        Below it a kernel is memory bound however well it is written; above it
        no amount of extra bandwidth helps.
        """
        return self.roof_flops / self.bandwidth

    def roof(self, intensity: float) -> float:
        """Attainable FLOP/s at a given arithmetic intensity."""
        return min(self.roof_flops, self.bandwidth * intensity)


# ---------------------------------------------------------------------------
# Traffic models
# ---------------------------------------------------------------------------


@dataclass
class Traffic:
    """Compulsory HBM bytes, broken down by what moved them."""

    parts: dict[str, int] = field(default_factory=dict)
    exact: bool = True
    note: str = ""

    @property
    def total(self) -> int:
        return sum(self.parts.values())

    def share(self, key: str) -> float:
        return self.parts.get(key, 0) / max(1, self.total)


def _tensor_bytes(batch, seq_len, heads, head_dim, itemsize) -> int:
    """Bytes in one [B, S, H, D] tensor."""
    return batch * seq_len * heads * head_dim * itemsize


def traffic_ours(batch, seq_len, heads, head_dim, cfg, kv_heads=None,
                 partials_dtype=None) -> Traffic:
    """Compulsory traffic for the KV-block-major kernel plus its combine.

    Four distinct movements, and the interesting result is which dominates.

    * **Q is read `top_k` times, not once.** A program owns a KV block and
      gathers the queries that selected it; a query selected `top_k` blocks, so
      its row is pulled by `top_k` different programs. This is the direct cost
      of the loop inversion and it does not appear in a dense kernel at all.
    * **K and V are read exactly once each.** Each block is loaded by the one
      program that owns it. This is the inversion's payoff, and it is why the
      utilization argument holds.
    * **The partials round-trip.** `part_acc` is [B, H, S, top_k, D], written by
      the attend pass and read back by the combine. Stored at the *compute*
      dtype, not float32: `attend_partials` defaults `partials_dtype` to
      `q.dtype`, which `moba_fwd`'s docstring records as worth ~22% end to end.
      Only `empty_partials` says float32, and only because it has no query to
      read a dtype from. Modelling it as float32 -- as the first version of
      this file did -- doubles the largest term and moves the whole verdict.
      `part_m` and `part_l` stay float32 but are a factor of `head_dim`
      smaller, so they are folded in rather than listed.
    * **O is written once.**
    """
    kv_heads = kv_heads or heads
    itemsize = torch.tensor([], dtype=cfg.dtype).element_size()
    partials_dtype = partials_dtype or cfg.dtype
    pitem = torch.tensor([], dtype=partials_dtype).element_size()
    num_blocks = cfg.num_blocks(seq_len)
    slots = min(cfg.top_k, num_blocks)

    q_row = batch * heads * seq_len * head_dim * itemsize
    partial = batch * heads * seq_len * slots * head_dim * pitem
    # part_m and part_l: [B, H, S, slots] float32 each, both directions.
    partial += 2 * 2 * batch * heads * seq_len * slots * FP32 // (2 * head_dim)
    return Traffic({
        "Q (gathered, x top_k)": slots * q_row,
        "K": _tensor_bytes(batch, seq_len, kv_heads, head_dim, itemsize),
        "V": _tensor_bytes(batch, seq_len, kv_heads, head_dim, itemsize),
        "partials write": partial,
        "partials read": partial,
        "O": _tensor_bytes(batch, seq_len, heads, head_dim, itemsize),
    })


# FlashMoBA's MMA tile is 64 query rows (`kernel_traits.h:88-91`,
# `Tile<64, _16, _16>`), so a gathered query list is processed in chunks of
# 64 and each chunk reloads K and V (`flash_fwd_kernel.h:544-590`).
FLASH_MOBA_CHUNK_ROWS = 64


def flash_moba_window(seq_len: int, cfg) -> int:
    """Reproduces FlashMoBA's `decide_lg_block_m` policy, uncapped.

    Their density is attended keys over causally available keys, which is
    `top_k * block_size / (S / 2)` up to the early-query clamp
    (`flash_moba_interface.py:20-48`). The sm_86/sm_89 cap to 512 is not
    applied, which is correct for every card this roofline is drawn for:
    the cap is theirs for sm_86/sm_89 only, and sm_80 and sm_90 take the
    uncapped branch. Drawing a roofline on a laptop would need the cap.

    Args:
        seq_len: Sequence length.
        cfg: Gate configuration, supplying block_size and top_k.

    Returns:
        The query window (`lg_block_m`) one of their programs owns.
    """
    sparsity = 1.0 - min(1.0, 2 * cfg.top_k * cfg.block_size / seq_len)
    if sparsity <= 0.5:
        return 128
    if sparsity <= 0.7:
        return 256
    if sparsity <= 0.8:
        return 512
    if sparsity <= 0.9:
        return 768
    return 1024


def traffic_flash_moba(batch, seq_len, heads, head_dim, cfg,
                       kv_heads=None) -> Traffic:
    """Modelled traffic for FlashMoBA, from its source.

    Until 2026-09-15 this counted gathered Q, K and V once each, and O -- a
    floor on any gathering kernel, not FlashMoBA's traffic. It put them at
    186 FLOP/byte, right of the ridge, and the A100 doc read that as
    "compute bound at 4% of roof". Two terms were missing, and both are read
    from their kernel (`docs/traffic-derivation.md` section 8.2):

    * **K/V are reloaded once per (query window, KV block, 64-row chunk).**
      Their grid is one program per window of `lg_block_m` queries; inside it
      the loop walks every KV block and gathers the window's queries that
      chose it, so a block is loaded once per window that has a picker,
      per chunk of 64 pickers. The count below is a closed form over the
      mean pickers per pair; it reproduces the routing-measured 4,934 loads
      at S=8192 to within 6%.
    * **The running output round-trips HBM in fp32** once per pair
      (`o_tmp_load`/`o_tmp_store`), plus a zero-fill and a final read.

    Still inexact: the mean-count model ignores the variance that `ceil`
    turns into extra chunks, and the pair count uses `top_k` per query with
    no early-query clamp (about 6% high at S=8192). Both are small next to
    the terms they sit in.
    """
    kv_heads = kv_heads or heads
    del kv_heads                                     # their grid is per query head
    itemsize = torch.tensor([], dtype=cfg.dtype).element_size()
    num_blocks = cfg.num_blocks(seq_len)
    slots = min(cfg.top_k, num_blocks)
    rows = batch * heads * seq_len
    pairs = slots * rows

    window = flash_moba_window(seq_len, cfg)
    loads = 0
    for i in range(math.ceil(seq_len / window)):
        # Blocks causally available to the last query of window i.
        cols = min(num_blocks, math.ceil(window * (i + 1) / cfg.block_size))
        per_pair = window * min(cfg.top_k, cols) / cols
        loads += cols * math.ceil(per_pair / FLASH_MOBA_CHUNK_ROWS)
    loads *= batch * heads
    kv_tile = cfg.block_size * head_dim * itemsize

    o_fp32 = rows * head_dim * 4
    return Traffic({
        "Q (gathered, x top_k)": pairs * head_dim * itemsize,
        "K (per window-block chunk)": loads * kv_tile,
        "V (per window-block chunk)": loads * kv_tile,
        "row indices": pairs * 4,
        "o_tmp fp32 round trip (x top_k)": 2 * pairs * head_dim * 4,
        "o_tmp zero-fill + final read": 2 * o_fp32,
        "O": rows * head_dim * itemsize,
        "LSE": rows * 4,
    }, exact=False,
       note="closed-form window model; docs/traffic-derivation.md s8.2")


def traffic_fused_dense(batch, seq_len, heads, head_dim, cfg) -> Traffic:
    """Compulsory traffic for a fused dense kernel (FlashAttention, SDPA).

    The defining property of FlashAttention is that the S x S score matrix
    never reaches HBM, so compulsory traffic is just the four tensors. Real
    traffic is higher -- a query tile re-reads earlier K/V tiles, which is
    quadratic in S and only partly absorbed by L2 -- so this bound is looser
    here than for the MoBA kernels, and the resulting intensity is famously
    high. That is the correct textbook answer and it is a useful check that
    the model is not nonsense.
    """
    itemsize = torch.tensor([], dtype=cfg.dtype).element_size()
    t = _tensor_bytes(batch, seq_len, heads, head_dim, itemsize)
    return Traffic({"Q": t, "K": t, "V": t, "O": t},
                   exact=False, note="ignores K/V re-reads across query tiles")


def traffic_materializing(batch, seq_len, heads, head_dim, cfg) -> Traffic:
    """Traffic for an implementation that writes the score matrix to HBM.

    The naive PyTorch path and MoonshotAI's reference both build [B, H, S, S]
    tensors. Each elementwise stage over that matrix is a read and a write, and
    at S=8192 one pass alone is 1.07 GB against 4.2 MB for the whole of Q. This
    is the entire reason they are two orders of magnitude slower, and it is
    visible on the roofline as an intensity near the far left.
    """
    itemsize = torch.tensor([], dtype=cfg.dtype).element_size()
    t = _tensor_bytes(batch, seq_len, heads, head_dim, itemsize)
    scores = batch * heads * seq_len * seq_len * FP32
    return Traffic({
        "Q + K + V + O": 4 * t,
        "scores write": scores,
        "scores read (softmax)": scores,
        "softmax write": scores,
        "softmax read (PV)": scores,
    }, exact=False, note="four passes over the S x S matrix, approximate")


MODELS = {
    "ours": traffic_ours,
    "flash_moba": traffic_flash_moba,
    "sdpa_dense": traffic_fused_dense,
    "fa2_minimal": traffic_fused_dense,
    "naive_torch": traffic_materializing,
    "moonshot_naive": traffic_materializing,
    # `moonshot_efficient` used to borrow `traffic_flash_moba`. That was a
    # floor common to any gathering kernel; now that the FlashMoBA model
    # counts FlashMoBA's window reloads and fp32 round trip, it no longer
    # describes a kernel built on FlashAttention's varlen path. No model has
    # been derived for it from its source, so it has none and is omitted
    # from the roofline rather than placed by a model that is known wrong.
}


def model_traffic(name, batch, seq_len, heads, head_dim, cfg) -> Traffic | None:
    """Returns the traffic model for a backend, or None if we have no model."""
    fn = MODELS.get(name)
    if fn is None:
        return None
    if fn in (traffic_ours, traffic_flash_moba):
        return fn(batch, seq_len, heads, head_dim, cfg)
    return fn(batch, seq_len, heads, head_dim, cfg)


# ---------------------------------------------------------------------------
# FLOPs actually performed
# ---------------------------------------------------------------------------


def flops_performed(name, batch, seq_len, heads, head_dim, cfg) -> float:
    """FLOPs the implementation actually issues, forward pass.

    Not the dense-equivalent convention. A roofline divides real work by real
    bytes, so a sparse kernel must be credited only with the work it did --
    crediting it with the dense equivalent would place it to the right of its
    true intensity and flatter it exactly where the plot is meant to be honest.
    """
    if name in ("sdpa_dense", "fa2_minimal", "naive_torch",
                "moonshot_naive"):
        # QK^T and PV, two FLOPs per MAC, halved for causal.
        return 2.0 * 2.0 * batch * heads * seq_len * seq_len * head_dim / 2.0
    attended = min(cfg.top_k, cfg.num_blocks(seq_len)) * cfg.block_size
    return 2.0 * 2.0 * batch * heads * seq_len * attended * head_dim


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------


@dataclass
class Point:
    """One backend at one sequence length, placed on the roofline."""

    name: str
    seq_len: int
    ms: float
    flops: float
    traffic: Traffic

    @property
    def intensity(self) -> float:
        return self.flops / self.traffic.total

    @property
    def achieved(self) -> float:
        return self.flops / (self.ms * 1e-3)

    def fraction_of_roof(self, ceilings: Ceilings) -> float:
        return self.achieved / ceilings.roof(self.intensity)

    def bound_by(self, ceilings: Ceilings) -> str:
        return "memory" if self.intensity < ceilings.ridge else "compute"


def measure(name: str, seq_len: int, cfg: MoBAConfig) -> float | None:
    """Times one backend's forward pass, or returns None if it cannot run."""
    backend = get_backend(name)
    ok, _ = backend.is_available(cfg)
    if not ok:
        return None
    g = torch.Generator(device="cuda").manual_seed(0)
    try:
        q, k, v = (torch.randn(BATCH, seq_len, HEADS, HEAD_DIM, generator=g,
                               device="cuda", dtype=cfg.dtype) for _ in range(3))
        fn = lambda: backend.forward(q, k, v, cfg)      # noqa: E731
        # Settle the device before the first launch, because that launch is
        # what triggers autotuning for our backend -- see bench/compare.py.
        cool_down_device()
        warm_up_device()
        fn()
        cool_down_device()
        warm_up_device()
        # Best of three, for the reason bench/compare.py records: measurement
        # error here is one-sided, so the minimum is the least contaminated
        # sample rather than an optimistic pick.
        return min(triton.testing.do_bench(fn, quantiles=[0.5, 0.2, 0.8])[0]
                   for _ in range(3))
    except (RuntimeError, torch.OutOfMemoryError):
        torch.cuda.empty_cache()
        return None


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def print_ceilings(c: Ceilings, name: str = "") -> None:
    """Prints the ceiling stack, each measured value audited against spec.

    Args:
        c: The ceilings.
        name: The card these describe. Supplied by `--from-log`, which may be
            replotting a run from a different machine entirely.
    """
    name = name or (torch.cuda.get_device_properties(0).name
                    if torch.cuda.is_available() else "device")
    print(f"\n{name} -- ceilings")

    def audit(measured, spec):
        """Renders a measured/spec ratio, or nothing when spec is unknown."""
        return f"  ({measured / spec:.0%} of spec)" if spec else ""

    print(f"  HBM bandwidth       {c.bandwidth / BYTES_PER_GIB:8.1f} GiB/s"
          f"{audit(c.bandwidth, c.spec_bandwidth)}")
    print(f"  bf16, {MATMUL_N}^3 GEMM  {c.peak_flops / 1e12:8.2f} TFLOP/s"
          f"{audit(c.peak_flops, c.spec_flops)}")
    if c.attainable_flops:
        print(f"  bf16, our tile      {c.attainable_flops / 1e12:8.2f} TFLOP/s"
              f"{audit(c.attainable_flops, c.spec_flops)}   <- the roof")
        print(f"    tile shape and softmax cost "
              f"{c.peak_flops / c.attainable_flops:.2f}x against a big GEMM")
    else:
        print("  bf16, our tile         not measured -- run "
              "`python -m bench.tile_ceiling` and pass --attainable-tflops;")
        print("                         the GEMM stands in, and overstates "
              "the roof")
    print(f"  ridge point         {c.ridge:8.1f} FLOP/byte  "
          f"<- below this, memory bound however good the kernel")


def print_table(points: list[Point], c: Ceilings) -> None:
    print(f"\nForward, block_size={CFG.block_size} top_k={CFG.top_k}, "
          f"B={BATCH} H={HEADS} D={HEAD_DIM}")
    print(f"  {'backend':<13} {'S':>6} {'ms':>8} {'GB moved':>9} "
          f"{'FLOP/byte':>10} {'TFLOP/s':>8} {'roof':>8} {'of roof':>8}  bound")
    print("  " + "-" * 88)
    over = [p for p in points if p.fraction_of_roof(c) > 1.0]
    for p in sorted(points, key=lambda p: (p.seq_len, p.name)):
        roof = c.roof(p.intensity) / 1e12
        mark = "" if p.traffic.exact else " ~"
        print(f"  {p.name:<13} {p.seq_len:>6} {p.ms:>8.3f} "
              f"{p.traffic.total / BYTES_PER_GIB:>9.3f} "
              f"{p.intensity:>10.1f}{mark:<2} {p.achieved / 1e12:>6.2f} "
              f"{roof:>8.2f} {p.fraction_of_roof(c):>7.0%}  {p.bound_by(c)}")
    print("  ~ = traffic model is a lower bound; see the module docstring.")
    if over:
        # A free correctness check on the whole figure. Nothing can beat the
        # roof, so a point above it means the *ceiling* is wrong, not the
        # kernel -- and the usual cause is that the ceilings were measured
        # while something else had the GPU. Caught exactly that way on
        # 2026-09-07, when a test suite was still running in another process
        # and peak_flops came out at 8.55 TFLOP/s against 14.16 on a quiet box.
        worst = max(p.fraction_of_roof(c) for p in over)
        print("")
        print(f"  WARNING: {len(over)} point(s) exceed the roof, worst "
              f"{worst:.0%}. Nothing can beat the roof, so the ceiling")
        print("  measurement is wrong -- almost always because another "
              "process had the GPU")
        print("  while it was taken. Re-run on a quiet box before "
              "believing any row.")


def print_breakdown(points: list[Point], seq_len: int) -> None:
    """Prints where the bytes go, which is the actionable part."""
    print(f"\nWhere the bytes go at S={seq_len}")
    for p in sorted(points, key=lambda p: p.name):
        if p.seq_len != seq_len:
            continue
        print(f"\n  {p.name}  ({p.traffic.total / BYTES_PER_MIB:.1f} MiB"
              f"{'' if p.traffic.exact else ', ' + p.traffic.note})")
        for key, value in sorted(p.traffic.parts.items(),
                                 key=lambda kv: -kv[1]):
            bar = "#" * max(1, round(40 * value / p.traffic.total))
            print(f"    {key:<24} {value / BYTES_PER_MIB:>8.1f} MiB "
                  f"{value / p.traffic.total:>5.0%} {bar}")


def plot(points: list[Point], c: Ceilings, path: str) -> str:
    """Draws the roofline: attainable FLOP/s against arithmetic intensity."""
    fig, ax = plots.figure(width=9.0, height=5.8)

    lo, hi = 0.05, max(4096.0, max(p.intensity for p in points) * 3)
    xs = [lo * (hi / lo) ** (i / 400) for i in range(401)]

    # The ceilings a kernel is *not* measured against, drawn faintly above the
    # roof. They are what makes the roof's position legible: without them a
    # reader cannot tell whether 164 TFLOP/s is the card being slow or
    # attention being expensive. Weight encodes authority -- dotted for a
    # datasheet nobody reaches, dashed for a GEMM no attention kernel
    # resembles, solid for the line that actually bounds this work.
    for value, label, dashes in (
        (c.spec_flops, "datasheet peak", (0, (1, 3))),
        (c.peak_flops if c.attainable_flops else None,
         f"{MATMUL_N}^3 GEMM", (0, (6, 3))),
    ):
        if not value:
            continue
        ax.axhline(value / 1e12, color=plots.INK_FAINT, linewidth=1.0,
                   linestyle=dashes, zorder=2)
        ax.text(lo * 1.3, value / 1e12 * 1.06, label, color=plots.INK_FAINT,
                fontsize=8.5, va="bottom")

    ax.plot(xs, [c.roof(x) / 1e12 for x in xs], color=plots.INK,
            linewidth=1.6, zorder=3)
    ax.axvline(c.ridge, color=plots.INK_FAINT, linewidth=0.9,
               linestyle=(0, (3, 3)), zorder=2)


    seen, biggest = set(), max(p.seq_len for p in points)
    for p in sorted(points, key=lambda p: p.name):
        label = p.name if p.name not in seen else None
        seen.add(p.name)
        # Marker area encodes sequence length, so one plot carries the sweep
        # without a second axis.
        size = 26 + 74 * (p.seq_len / biggest)
        ax.scatter(p.intensity, p.achieved / 1e12, s=size,
                   color=plots.color(p.name), label=label, zorder=5,
                   edgecolor=plots.SURFACE, linewidth=1.2)
    tips = [p for p in points if p.seq_len == biggest]
    plots.label_points(ax, [p.intensity for p in tips],
                       [p.achieved / 1e12 for p in tips],
                       [p.name for p in tips], log=True)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    # Headroom above the *tallest* ceiling drawn, not the roof, or the
    # datasheet line lands outside the axes.
    top = max(c.roof_flops, c.peak_flops, c.spec_flops or 0.0)
    ax.set_ylim(top / 1e12 * 1e-4, top / 1e12 * 2.2)

    # Region labels last, and both *under* the roof. The roof is a ceiling, so
    # the area above it is unattainable: a label printed up there names a
    # region no kernel can occupy, and reads as "compute bound means faster
    # than the machine can go". The two regions being named are the halves of
    # the space below the line, split at the ridge.
    #
    # After set_xscale/set_ylim, not before -- the sloped label's angle is read
    # off the axes transform, which is not final until the scales and limits
    # are.
    ax.text(lo * 6.0, c.bandwidth * lo * 6.0 / 1e12 * 0.45, "memory bound",
            color=plots.INK_SOFT, fontsize=9.5,
            rotation=plots.slope_degrees(ax), rotation_mode="anchor",
            ha="left", va="top")
    ax.text(c.ridge * 1.6, c.peak_flops / 1e12 * 0.62, "compute bound",
            color=plots.INK_SOFT, fontsize=9.5, va="top")
    # The ridge label sits at the bottom of its own line rather than partway up
    # it. At mid-height it landed on whichever implementation happened to have
    # an intensity near the ridge -- on the A100 that was flash_moba, and the
    # two strings printed through each other.
    # Just left of its own line, reading upward. Two horizontal placements
    # have now collided with something -- a data point at mid-height, the
    # legend at the foot -- because the ridge is a vertical line through the
    # busiest part of the figure. Setting the text along it avoids competing
    # for horizontal space at all.
    ax.text(c.ridge * 0.88, c.roof_flops / 1e12 * 0.30,
            f"ridge {c.ridge:.0f} FLOP/byte", color=plots.INK_SOFT,
            fontsize=9, rotation=90, va="bottom", ha="center")

    plots.titles(
        ax,
        "Roofline -- MoBA forward pass",
        # One sequence length is the common case and "8192-8192" reads as a
        # bug rather than as a range of one.
        (f"S = {biggest}. " if min(p.seq_len for p in points) == biggest else
         f"Marker size is sequence length, {min(p.seq_len for p in points)} to "
         f"{biggest}; labels mark S={biggest}. ") +
        "Higher is faster. The solid roof is what an attention kernel reaches "
        "here.\nThe faint lines above are a big GEMM and the datasheet: "
        "neither is reachable by this work.",
        "arithmetic intensity (FLOP per byte of compulsory HBM traffic)",
        "achieved throughput (TFLOP/s)")
    plots.legend(ax, loc="lower right", ncol=2)
    return plots.save(fig, path)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _with_extra_ceilings(ceilings: Ceilings,
                         attainable_tflops: float | None,
                         device_name: str = "") -> Ceilings:
    """Attaches the attainable roof and this card's datasheet figures.

    Kept separate from measurement so that `--from-log` gets the same treatment
    as a live run: the datasheet is a constant and the attainable roof is an
    argument, so neither needs a GPU.

    Args:
        ceilings: The measured bandwidth and GEMM figures.
        attainable_tflops: `bench/tile_ceiling.py`'s dots+softmax result, or
            None to leave the GEMM standing in as the roof.

    Returns:
        A new `Ceilings` carrying whatever is known.
    """
    spec = device_spec(name=device_name)
    return dataclasses.replace(
        ceilings,
        attainable_flops=(attainable_tflops * 1e12 if attainable_tflops
                          else None),
        spec_flops=spec.get("tflops_bf16", 0.0) * 1e12 or None,
        spec_bandwidth=spec.get("bandwidth_gbs", 0.0) * 1e9 or None,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seq-lens", default="2048,4096,8192",
                        help="comma separated")
    parser.add_argument("--out", default=None,
                        help="default: this card's figures directory")
    parser.add_argument("--attainable-tflops", type=float, default=None,
                        help="the roof, from `python -m bench.tile_ceiling` "
                             "(its dots+softmax figure). Without it the GEMM "
                             "stands in and overstates the ceiling.")
    parser.add_argument("--from-log", default=None,
                        help="redraw a finished run's log instead of "
                             "measuring; needs no GPU")
    args = parser.parse_args()
    seq_lens = [int(x) for x in args.seq_lens.split(",")]

    if args.from_log:
        points, ceilings, device_name = parse_log(args.from_log)
        ceilings = _with_extra_ceilings(ceilings, args.attainable_tflops,
                                        device_name)
        print_ceilings(ceilings, device_name)
        print_table(points, ceilings)
        print()
        print(f"wrote {plot(points, ceilings, args.out)}")
        return

    if lock_clocks():
        print("clocks locked to max")
    try:
        ceilings = _with_extra_ceilings(
            Ceilings(measure_peak_bandwidth(), measure_peak_tflops()),
            args.attainable_tflops)
        print_ceilings(ceilings)

        names = [b.name for b in all_backends() if b.name in MODELS]
        points: list[Point] = []
        for seq_len in seq_lens:
            for name in names:
                ms = measure(name, seq_len, CFG)
                if ms is None:
                    continue
                traffic = model_traffic(name, BATCH, seq_len, HEADS,
                                        HEAD_DIM, CFG)
                points.append(Point(
                    name, seq_len, ms,
                    flops_performed(name, BATCH, seq_len, HEADS, HEAD_DIM, CFG),
                    traffic))

        if not points:
            print("no backend produced a measurement")
            return
        print_table(points, ceilings)
        print_breakdown(points, max(seq_lens))
        import os
        out = args.out or os.path.join(
            paths.card_dir(kind="figures", create=True), "roofline.png")
        print(f"\nwrote {plot(points, ceilings, out)}")
    finally:
        unlock_clocks()



# ---------------------------------------------------------------------------
# Replotting a finished run
# ---------------------------------------------------------------------------


def parse_log(path: str) -> tuple[list[Point], Ceilings]:
    """Rebuilds the points and ceilings of a finished run from its log.

    WHY THIS EXISTS. The ceilings and every point are measured on the GPU, so
    redrawing the figure used to mean having that GPU. This project's A100 is
    rented by the hour, and three of the reruns it paid for changed nothing but
    the position of a text label. The log already contains every number the
    figure needs -- both ceilings and, per row, arithmetic intensity and
    achieved throughput -- so a run's data can outlive the machine that
    produced it.

    Reconstruction is partial by design. `Point` normally derives intensity
    and throughput from `flops` and a `Traffic` breakdown; the log carries the
    two derived values but not the breakdown, so the totals are back-solved to
    reproduce them exactly. That is enough for `plot`, which reads only
    `intensity` and `achieved`, and not enough for `print_breakdown`, which is
    why this returns points fit for drawing rather than for re-analysis.

    Args:
        path: A log written by a previous `main()` run.

    Returns:
        A tuple of (points, ceilings).

    Raises:
        ValueError: If the ceiling lines are missing, so a caller never gets a
            figure drawn against a roof that was silently defaulted.
    """
    bandwidth = peak = None
    device_name = ""
    points: list[Point] = []

    with open(path, encoding="utf-8", errors="replace") as handle:
        for line in handle:
            fields = line.split()
            if " -- ceilings" in line or " -- measured ceilings" in line:
                # The card that produced the log, so its datasheet and its name
                # travel with the numbers instead of being taken from whatever
                # machine is drawing.
                device_name = line.split(" -- ")[0].strip()
            elif "HBM bandwidth" in line:
                bandwidth = float(fields[2]) * BYTES_PER_GIB
            elif "bf16 peak" in line:
                peak = float(fields[2]) * 1e12
            elif len(fields) >= 8 and fields[0] in MODELS:
                # backend S ms GB FLOP/byte [~] TFLOP/s roof pct bound
                numbers = [f for f in fields[1:] if f not in ("~",)]
                try:
                    seq_len, ms = int(numbers[0]), float(numbers[1])
                    intensity, achieved = float(numbers[3]), float(numbers[4])
                except (ValueError, IndexError):
                    continue
                flops = achieved * 1e12 * ms * 1e-3
                # The log stores the latency, which is a measurement, and the
                # intensity, which is only the traffic model as it stood when
                # the run was made. Re-model from the current code where a
                # model exists, so a corrected model corrects the figure
                # without a GPU (the FlashMoBA model was wrong until
                # 2026-09-15); back-solve the total only where there is none.
                traffic = model_traffic(fields[0], BATCH, seq_len, HEADS,
                                        HEAD_DIM, CFG)
                if traffic is None:
                    traffic = Traffic(parts={"total": round(flops / intensity)},
                                      exact=False, note="reconstructed from log")
                points.append(Point(fields[0], seq_len, ms, flops, traffic))

    if bandwidth is None or peak is None:
        raise ValueError(f"{path} has no ceiling lines; refusing to draw a "
                         "roofline against an assumed roof")
    return points, Ceilings(bandwidth, peak), device_name


if __name__ == "__main__":
    main()
