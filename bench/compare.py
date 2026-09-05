"""Head-to-head against every available implementation, forward and backward.

Built to the shape the reference repos use, so the numbers can sit beside
theirs without a translation note.

WHAT WAS COPIED, AND FROM WHERE.

* `triton.testing.perf_report` with one line per provider, and `_bwd` lines
  alongside the forward ones -- fla-org/NSA, `benchmarks/benchmark_nsa.py`.
  A backward line times `forward().backward(do)`, i.e. the whole training
  step, which is what NSA reports and what a user actually pays.
* `quantiles=[0.5, 0.2, 0.8]` -- also NSA. `do_bench` then returns median with
  p20/p80, so a noisy row is visible instead of being averaged into a
  confident-looking median. This project spent a session discovering a 1.59x
  spread by hand that error bars would have shown at a glance
  (`docs/reference-comparison.md`).
* Sequence lengths in powers of two -- both references. FlashMoBA sweeps
  4K-512K at batch 2; we stop far earlier because this card has 4 GB.

TWO FLOP CONVENTIONS, REPORTED SIDE BY SIDE. They differ by the sparsity
ratio, which is ~8x at S=8192 with top_k=8, so quoting one against a paper
using the other is off by that factor:

* `flops_dense` is FlashMoBA's (`benchmarks/bench_flash_moba.py`):
  `4 * B * S^2 * H * D / 2` for causal, counted regardless of what the kernel
  skipped. It measures effective throughput against a dense equivalent, so it
  doubles as a speedup proxy and is what their published numbers use.
* `flops_attended` is ours (`bench.suite.effective_tflops`): only the
  multiply-accumulates actually performed. It measures how well the hardware
  is being used on the work that really happened.

A sparse kernel scores higher on the first and lower on the second. Neither is
wrong; mixing them is.
"""

from pathlib import Path

import torch
import triton
import triton.testing

from bench.suite import cool_down_device, warm_up_device
from implementations.base import all_backends
from moba.config import MoBAConfig

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# FlashMoBA's published gate: block_size 128, top_k 8.
CFG = MoBAConfig(block_size=128, top_k=8, dtype=torch.bfloat16)
BATCH, HEADS, HEAD_DIM = 1, 4, 64

# Powers of two, as both references sweep. FlashMoBA runs 4K-512K at batch 2;
# 8192 is where a 4 GB card stops holding the naive implementations.
SEQ_LENS = [1024, 2048, 4096, 8192]

QUANTILES = [0.5, 0.2, 0.8]

# Repeats per measurement; the fastest is kept. See `benchmark` for why the
# minimum rather than the median.
BEST_OF = 3

# provider -> why it produced no number.
SKIPPED: dict[str, str] = {}

# (provider, seq_len) -> median ms, filled by the sweep so the throughput
# table can read them instead of measuring a second time. Re-timing after
# the sweep meant the naive backends ran against a nearly-full 4 GB card,
# which reported moonshot at 17,490 ms where the sweep had it at 980 ms.
MEDIANS: dict[tuple[str, int], float] = {}


# ---------------------------------------------------------------------------
# FLOP counting, in both conventions
# ---------------------------------------------------------------------------


def flops_dense(batch, seq_len, heads, head_dim, causal=True, mode="fwd"):
    """FlashMoBA's convention: dense-equivalent FLOPs, sparsity ignored.

    Copied from `references/flash-moba/benchmarks/bench_flash_moba.py`. The
    backward multiplier of 2.5 is FlashAttention's standard accounting: the
    backward recomputes the forward's scores and runs two more matmuls.

    Args:
        batch: Batch size.
        seq_len: Sequence length.
        heads: Number of query heads.
        head_dim: Head dimension.
        causal: Whether the mask is causal, which halves the pair count.
        mode: "fwd", "bwd", or "fwd_bwd".

    Returns:
        FLOP count.
    """
    f = 4 * batch * seq_len ** 2 * heads * head_dim // (2 if causal else 1)
    return {"fwd": f, "bwd": int(f * 2.5), "fwd_bwd": int(f * 3.5)}[mode]


def flops_attended(batch, seq_len, heads, head_dim, cfg, mode="fwd"):
    """Our convention: only the arithmetic the kernel actually performed.

    Args:
        batch: Batch size.
        seq_len: Sequence length.
        heads: Number of query heads.
        head_dim: Head dimension.
        cfg: Gate configuration, supplying block_size and top_k.
        mode: "fwd", "bwd", or "fwd_bwd".

    Returns:
        FLOP count over attended key positions only.
    """
    num_blocks = max(1, seq_len // cfg.block_size)
    attended = min(cfg.top_k, num_blocks) * cfg.block_size
    # 2 FLOPs per MAC, two matmuls (QK^T and PV).
    f = 2 * 2 * batch * heads * seq_len * attended * head_dim
    return {"fwd": f, "bwd": int(f * 2.5), "fwd_bwd": int(f * 3.5)}[mode]


# ---------------------------------------------------------------------------
# Providers
# ---------------------------------------------------------------------------


def available(cfg: MoBAConfig) -> list:
    """Returns backends that can run this config, in a stable order."""
    out = []
    for b in sorted(all_backends(), key=lambda x: x.name):
        ok, _ = b.is_available(cfg)
        if ok:
            out.append(b)
    return out


_DIFFERENTIABLE: list | None = None


def differentiable(cfg: MoBAConfig) -> list:
    """Returns the backends whose output carries a gradient.

    Probed once with a tiny input rather than hardcoded, so a backend added
    later is classified correctly without editing this file. Two ways to fail
    it: the kernel has no backward at all (`fa2_minimal` is a forward-only
    teaching rebuild), or the adapter calls a bare launcher that builds no
    graph -- which ours did until it grew `forward_grad`.

    A backward plot that carries a line for a backend with no backward is a
    legend entry for a curve of NaN.
    """
    global _DIFFERENTIABLE
    if _DIFFERENTIABLE is None:
        _DIFFERENTIABLE = []
        for b in available(cfg):
            call = getattr(b, "forward_grad", b.forward)
            try:
                probe = [
                    torch.randn((1, 256, HEADS, HEAD_DIM), device="cuda",
                                dtype=cfg.dtype, requires_grad=True)
                    for _ in range(3)
                ]
                if call(*probe, cfg).requires_grad:
                    _DIFFERENTIABLE.append(b)
            except (RuntimeError, NotImplementedError, torch.OutOfMemoryError):
                pass
        torch.cuda.empty_cache()
    return _DIFFERENTIABLE


def _providers(cfg: MoBAConfig) -> list[str]:
    return ([b.name for b in available(cfg)]
            + [f"{b.name}_bwd" for b in differentiable(cfg)])


PROVIDERS = _providers(CFG)
BY_NAME = {b.name: b for b in available(CFG)}


_INPUTS: dict[tuple[int, bool], list[torch.Tensor]] = {}


def _inputs(seq_len, requires_grad):
    """Returns cached q, k, v for this size.

    Cached deliberately. Rebuilding them per measurement, and calling
    `empty_cache` between measurements, made this benchmark report `ours` at
    9.4 ms for S=1024 and 8.2 ms for S=8192 -- slower at the shorter sequence,
    which is not a thing a kernel does. That is allocator churn, the same WDDM
    hazard `docs/gqa-measurements.md` records.
    """
    key = (seq_len, requires_grad)
    if key not in _INPUTS:
        g = torch.Generator(device="cuda").manual_seed(0)
        shape = (BATCH, seq_len, HEADS, HEAD_DIM)
        _INPUTS[key] = [
            torch.randn(shape, generator=g, device="cuda", dtype=CFG.dtype,
                        requires_grad=requires_grad)
            for _ in range(3)
        ]
    return _INPUTS[key]


def _callable(backend, backward: bool):
    """Picks the differentiable entry point when one is needed.

    `Backend.forward` is not required to build an autograd graph, and ours does
    not -- it calls `moba_forward`, the bare launcher. Backends that can be
    differentiated expose `forward_grad`; the rest fall back, and a backward
    line for a forward-only kernel reports as a gap.
    """
    if backward and hasattr(backend, "forward_grad"):
        return backend.forward_grad
    return backend.forward


# What each line *is*, which decides how it is drawn. A dense baseline is not
# a competitor -- it is the reference line the sparse kernels are trying to
# beat -- and our own oracle is a correctness check that happens to be
# runnable. Drawing all three groups identically invites reading the plot as a
# seven-way race, which it is not.
_GROUPS = {
    "ours":        ("#d62728", "-"),    # the subject, solid red
    "flash_moba":  ("#1f77b4", "--"),   # MoBA, CUDA          -- the perf target
    "triton_moba": ("#9467bd", "--"),   # MoBA, Triton+CUDA FA
    "moonshot":    ("#8c564b", "--"),   # MoBA, plain PyTorch -- the reference
    "sdpa_dense":  ("#7f7f7f", ":"),    # dense baseline
    "fa2_minimal": ("#2ca02c", ":"),    # dense baseline, ours
    "naive_torch": ("#bcbd22", "-."),   # our oracle
}


def _bench(mode: str) -> triton.testing.Benchmark:
    """One plot per direction, log-log.

    Split because a forward and a training step are different quantities and
    do not belong on one axis. Log-log because both axes span orders of
    magnitude -- sequence length by construction, and latency because the
    unfused PyTorch reference is ~10^3 times slower than the fused kernels at
    S=8192. On linear axes that single line flattens every other one onto zero.
    """
    backends = available(CFG) if mode == "fwd" else differentiable(CFG)
    names = [b.name for b in backends]
    suffix = "_bwd" if mode == "bwd" else ""
    return triton.testing.Benchmark(
        x_names=["seq_len"],
        x_vals=SEQ_LENS,
        x_log=True,
        line_arg="provider",
        line_vals=[f"{n}{suffix}" for n in names],
        line_names=names,                      # the suffix is the plot, not the line
        styles=[_GROUPS.get(n, ("black", "-")) for n in names],
        xlabel="sequence length",
        ylabel="latency (ms)",
        y_log=True,
        plot_name=f"moba-{mode}",
        args={},
    )


@triton.testing.perf_report([_bench("fwd"), _bench("bwd")])
def benchmark(seq_len, provider):
    """Times one provider at one sequence length.

    A `_bwd` provider times `forward().backward(do)` -- the full step, as NSA
    reports it -- not the backward in isolation.

    Returns:
        (median, p20, p80) in milliseconds, or (nan, nan, nan) if the provider
        cannot run this size on this card.
    """
    backward = provider.endswith("_bwd")
    name = provider[:-4] if backward else provider
    backend = BY_NAME[name]

    call = _callable(backend, backward)
    try:
        q, k, v = _inputs(seq_len, requires_grad=backward)
        if backward:
            out = call(q, k, v, CFG)
            if not out.requires_grad:
                raise NotImplementedError("no differentiable path")
            do = torch.randn_like(out)
            fn = lambda: call(q, k, v, CFG).backward(do)  # noqa: E731
        else:
            fn = lambda: call(q, k, v, CFG)  # noqa: E731
        fn()                      # surface an unsupported path before timing
        # Cool, then briefly warm, before *every* measurement. The device's
        # thermal state is shared, and a sweep hands it from one provider to
        # the next: `moonshot` at 690 ms a rep runs full-tilt for tens of
        # seconds, and whatever follows inherits a saturated card. Measured at
        # S=8192, `ours`, whose true figure here is ~1.59 ms:
        #
        #     after 90 s idle, no warm-up              11.70 ms
        #     after a 30 s matmul burn                  6.73 ms
        #     after a 20 s warm-up (86 C)              16.67 ms
        #     cooled to 76 C, then warmed 2 s           1.62 ms
        #
        # Both extremes are wrong, which is why this is two calls and not one.
        # `cool_down_device` handles a card with no headroom left; the short
        # warm-up handles one that has dropped to its idle clock. See both
        # docstrings in bench/suite.py.
        cool_down_device()
        warm_up_device()
        # Best of BEST_OF, not one measurement. Cooling cuts the contamination
        # but does not remove it: rechecking `ours` three times per size, with
        # a cooldown before each, still produced one bad reading in three --
        #
        #     S=2048   0.504  0.506  1.687
        #     S=4096   0.785  0.797  0.782
        #     S=8192   1.616  1.617  3.182
        #
        # -- and every bad reading is *slow*. Nothing makes a kernel finish
        # faster than its true time, so the error is one-sided and the minimum
        # is the least-contaminated sample. A median over three would have
        # taken 1.687 at S=2048; the minimum takes 0.504, which is the figure
        # two independent runs agree on.
        samples = [triton.testing.do_bench(fn, quantiles=QUANTILES)
                   for _ in range(BEST_OF)]
        result = min(samples, key=lambda r: r[0])
        MEDIANS[(provider, seq_len)] = result[0]
        return result
    except (RuntimeError, NotImplementedError, torch.OutOfMemoryError) as exc:
        # A gap, not a crash. Recorded rather than swallowed: a silent NaN hid
        # the fact that our own backward was unreachable through this adapter.
        SKIPPED.setdefault(provider, str(exc)[:90])
        torch.cuda.empty_cache()
        return float("nan"), float("nan"), float("nan")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def sparsity(seq_len: int) -> tuple[int, int]:
    """Returns (num_blocks, blocks a query actually attends) at this length.

    The gate cannot select more blocks than exist, so `top_k` clamps. When it
    clamps to `num_blocks` the selection is everything and MoBA is dense with
    extra steps -- routing, partial storage and a combine pass bought nothing.
    Reported per row because two of the four lengths this card can reach are
    in exactly that regime, and a table that hides it invites reading an
    honest overhead cost as a loss.
    """
    num_blocks = max(1, seq_len // CFG.block_size)
    return num_blocks, min(CFG.top_k, num_blocks)


def throughput_table() -> None:
    """Prints forward latency and TFLOP/s under both FLOP conventions."""
    print(f"\nForward, block_size={CFG.block_size} top_k={CFG.top_k}, "
          f"B={BATCH} H={HEADS} D={HEAD_DIM}")
    print("  'dense' counts FLOPs the dense equivalent would do (FlashMoBA's "
          "convention);\n  'attended' counts only what ran (ours). They differ "
          "by the sparsity ratio.\n")
    print(f"  {'backend':<13} {'S':>6} {'blocks':>7} {'attends':>8} "
          f"{'ms':>9} {'x dense':>8} {'TF/s dense':>11} {'TF/s attn':>10}")
    print("  " + "-" * 80)
    for seq_len in SEQ_LENS:
        blocks, attends = sparsity(seq_len)
        base = MEDIANS.get(("sdpa_dense", seq_len), float("nan"))
        for backend in available(CFG):
            ms = MEDIANS.get((backend.name, seq_len), float("nan"))
            if ms != ms:                       # nan, or never measured
                print(f"  {backend.name:<13} {seq_len:>6} {blocks:>7} "
                      f"{attends:>8} {'-':>9} {'-':>8} {'-':>11} {'-':>10}")
                continue
            secs = ms * 1e-3
            fd = flops_dense(BATCH, seq_len, HEADS, HEAD_DIM) / secs / 1e12
            fa = flops_attended(BATCH, seq_len, HEADS, HEAD_DIM,
                                CFG) / secs / 1e12
            spd = f"{base / ms:.2f}x" if base == base else "-"
            print(f"  {backend.name:<13} {seq_len:>6} {blocks:>7} "
                  f"{attends:>8} {ms:>9.3f} {spd:>8} {fd:>11.2f} {fa:>10.2f}")
        if attends == blocks:
            print(f"  {'':<13} {seq_len:>6}  <- top_k clamps to num_blocks: "
                  f"every block selected, no sparsity to exploit")


if __name__ == "__main__":
    print(f"providers: {', '.join(PROVIDERS)}\n")
    Path("results").mkdir(exist_ok=True)
    benchmark.run(print_data=True, save_path="results")
    throughput_table()
    if SKIPPED:
        print("\nno number produced:")
        for key, why in sorted(SKIPPED.items()):
            print(f"  {key:<20} {why}")
