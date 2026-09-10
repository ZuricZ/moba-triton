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
import time
from dataclasses import dataclass, field

import torch
import triton.testing

from bench import plots
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
    """This card's measured limits."""

    bandwidth: float                 # bytes/s
    peak_flops: float                # FLOP/s

    @property
    def ridge(self) -> float:
        """Arithmetic intensity where the two roofs meet, in FLOP/byte.

        Below it a kernel is memory bound however well it is written; above it
        no amount of extra bandwidth helps.
        """
        return self.peak_flops / self.bandwidth

    def roof(self, intensity: float) -> float:
        """Attainable FLOP/s at a given arithmetic intensity."""
        return min(self.peak_flops, self.bandwidth * intensity)


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


def traffic_flash_moba(batch, seq_len, heads, head_dim, cfg,
                       kv_heads=None) -> Traffic:
    """Lower bound for FlashMoBA.

    Modelled only as far as we have read their kernel: the same gathered-Q,
    once-each-K/V shape, because their attention takes the identical CSC
    routing triple ours produces. Their internal buffers are *not* modelled --
    we have not read that part -- so this is a floor on their traffic and
    therefore a ceiling on their arithmetic intensity. Marked inexact so the
    plot can say so rather than implying a precision we do not have.
    """
    kv_heads = kv_heads or heads
    itemsize = torch.tensor([], dtype=cfg.dtype).element_size()
    slots = min(cfg.top_k, cfg.num_blocks(seq_len))
    q_row = batch * heads * seq_len * head_dim * itemsize
    return Traffic({
        "Q (gathered, x top_k)": slots * q_row,
        "K": _tensor_bytes(batch, seq_len, kv_heads, head_dim, itemsize),
        "V": _tensor_bytes(batch, seq_len, kv_heads, head_dim, itemsize),
        "O": _tensor_bytes(batch, seq_len, heads, head_dim, itemsize),
    }, exact=False, note="internal buffers not modelled; a floor")


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
    "moonshot_efficient": traffic_flash_moba,
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


def print_ceilings(c: Ceilings) -> None:
    props = torch.cuda.get_device_properties(0)
    print(f"\n{props.name} -- measured ceilings, not datasheet")
    print(f"  HBM bandwidth   {c.bandwidth / BYTES_PER_GIB:8.1f} GiB/s")
    print(f"  bf16 peak       {c.peak_flops / 1e12:8.2f} TFLOP/s "
          f"({MATMUL_N}^3 GEMM)")
    print(f"  ridge point     {c.ridge:8.1f} FLOP/byte  "
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
    ax.plot(xs, [c.roof(x) / 1e12 for x in xs], color=plots.INK,
            linewidth=1.6, zorder=3)
    ax.axvline(c.ridge, color=plots.INK_FAINT, linewidth=0.9,
               linestyle=(0, (3, 3)), zorder=2)
    ax.text(c.ridge * 1.1, c.peak_flops / 1e12 * 0.045,
            f"ridge {c.ridge:.0f} FLOP/byte", color=plots.INK_SOFT, fontsize=9)


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
    ax.set_ylim(c.peak_flops / 1e12 * 1e-4, c.peak_flops / 1e12 * 2.2)

    # Region labels last, and both *under* the roof. The roof is a ceiling, so
    # the area above it is unattainable: a label printed up there names a
    # region no kernel can occupy, and reads as "compute bound means faster
    # than the machine can go". The two regions being named are the halves of
    # the space below the line, split at the ridge.
    #
    # After set_xscale/set_ylim, not before -- the sloped label's angle is read
    # off the axes transform, which is not final until the scales and limits
    # are.
    ax.text(lo * 2.4, c.bandwidth * lo * 2.4 / 1e12 * 0.5, "memory bound",
            color=plots.INK_SOFT, fontsize=9.5, rotation=slope_degrees(ax),
            rotation_mode="anchor", ha="left", va="top")
    ax.text(c.ridge * 1.6, c.peak_flops / 1e12 * 0.62, "compute bound",
            color=plots.INK_SOFT, fontsize=9.5, va="top")

    plots.titles(
        ax,
        "Roofline -- MoBA forward pass",
        # One sequence length is the common case and "8192-8192" reads as a
        # bug rather than as a range of one.
        (f"S = {biggest}. " if min(p.seq_len for p in points) == biggest else
         f"Marker size is sequence length, {min(p.seq_len for p in points)} to "
         f"{biggest}; labels mark S={biggest}. ") +
        "Higher is faster; the roof is what this card can do.",
        "arithmetic intensity (FLOP per byte of compulsory HBM traffic)",
        "achieved throughput (TFLOP/s)")
    plots.legend(ax, loc="lower right", ncol=2)
    return plots.save(fig, path)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seq-lens", default="2048,4096,8192",
                        help="comma separated")
    parser.add_argument("--out", default="results/roofline.png")
    args = parser.parse_args()
    seq_lens = [int(x) for x in args.seq_lens.split(",")]

    if lock_clocks():
        print("clocks locked to max")
    try:
        ceilings = Ceilings(measure_peak_bandwidth(), measure_peak_tflops())
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
        os.makedirs("results", exist_ok=True)
        print(f"\nwrote {plot(points, ceilings, args.out)}")
    finally:
        unlock_clocks()


if __name__ == "__main__":
    main()
