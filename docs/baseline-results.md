# Baseline measurements — the problem, quantified

**Date:** 2026-08-15
**Device:** NVIDIA GeForce RTX 3050 Ti Laptop, sm_86, 4095 MiB, 20 SMs
**Stack:** Python 3.13.15, torch 2.13.0+cu126, triton 3.7.1, bf16
**Raw data:** `results/local_rtx3050ti.csv`
**Reproduce:** see "Reproducing" at the bottom.

This is the number the project exists to move. Everything measured here is a
*baseline*: both MoBA implementations are deliberately naive, so being slow is
expected. The point is that the measurement now exists and the gap is
enormous and specific.

## Definitions

- **sparsity** — fraction of causally-legal key positions actually attended.
  Closed-form, and cross-checked against real oracle selection masks in
  `tests/test_bench.py::test_analytic_sparsity_matches_the_oracle`.
- **ceiling** — `1 / sparsity`. The speedup the sparsity pattern makes
  theoretically available.
- **speedup** — measured, against a dense causal `torch SDPA` roofline. SDPA
  dispatches to a fused/Flash kernel, so this is what a *good* dense
  implementation costs, not a strawman.
- **attainment** — `speedup / ceiling`. The fraction of the available win an
  implementation actually collects. **This is the headline number.**

Attainment exists because raw speedup is gameable: you can improve it just by
choosing a sparser config, which says nothing about kernel quality.

## Results, S=1024, top_k=2

| block_size | sparsity | ceiling | naive_torch | moonshot | attainment (moonshot) |
|---:|---:|---:|---:|---:|---:|
| 128 | 0.344 | 2.90x | 3.07 ms | 5.95 ms | 0.008 |
| 64 | 0.180 | 5.54x | 3.10 ms | 6.75 ms | 0.003 |
| 32 | 0.093 | 10.79x | 3.82 ms | 10.01 ms | 0.001 |
| **16** | **0.047** | **21.13x** | 4.43 ms | **15.35 ms** | **0.000** |

Dense SDPA roofline: **0.13 ms** at every block size (it ignores the gate).

## The finding

**As the available speedup rises 2.90x -> 21.13x, the implementation gets
slower, 5.95 ms -> 15.35 ms.**

At `block_size=16` only 4.7% of causal key positions are attended — a 21x
reduction in arithmetic — and Moonshot's implementation takes **2.6x longer**
doing that smaller amount of work. Both effects move the wrong way at once.

Attainment at block=16 is **0.001** for naive_torch and **0.0004** for
moonshot: roughly one tenth of one percent of the theoretically available win.

Effective throughput: dense SDPA reaches **8.2 TFLOP/s**; the MoBA paths
manage **0.003–0.12 TFLOP/s**.

The same pattern holds at S=512, with a lower ceiling (10.7x at block=16),
confirming it is not an artifact of one sequence length.

## Why (mechanism, to be confirmed against FlashMoBA's source)

Three effects compound as `block_size` shrinks, all pushing the same way:

1. **Gate cost grows.** `num_blocks = S / block_size`, so the gate's
   `S x num_blocks` score matrix grows 8x going from block=128 to block=16.
   The router gets more expensive exactly as the routed work gets cheaper.
2. **Routing diverges across a query tile.** Selection is per *query*, not per
   query block — two queries in the same block routinely pick different KV
   blocks (see `src/moba/ref/naive.py::select_blocks`, and the visualization
   in the session log). FlashAttention amortizes one KV tile load across a
   whole query tile; MoBA breaks that assumption, and the divergence worsens
   as blocks shrink.
3. **Tiles get too small for the hardware.** Tensor cores operate on fixed
   tile shapes, 16x16 at minimum and most efficient at 64x64 or larger. A
   16-key block leaves most of each tensor-core operation computing nothing.

Effect 3 is the one that makes this a *kernel* problem rather than an
algorithmic one, and it is what FlashMoBA and FSA both address. Confirming the
mechanism against real CUDA source is Task 10.

## Caveats

- Both MoBA backends are naive references, not optimized kernels. Low
  attainment is expected; the value here is the baseline and the *trend*.
- Measured on a 4 GB laptop sm_86. Per CLAUDE.md, absolute numbers here are
  not the project's benchmark — those need a rented A100/H100. The *shape* of
  the curve is what transfers.
- `flash_moba` and `triton_moba` could not run locally (unbuilt extension;
  missing flash-attn). Their rows appear in the CSV as `skipped` with reasons,
  never silently omitted.
- `top_k=2` under the inclusive convention means the own block plus one past
  block. See `docs/semantics-crosscheck.md`.

## Reproducing

```bash
wsl -e bash -lc 'export PATH="$HOME/.local/bin:$PATH"; \
  export UV_PROJECT_ENVIRONMENT="$HOME/.venvs/moba"; \
  cd "/mnt/c/D/Side Projects/GPU Kernel" && uv run python -c "
from bench.suite import run_suite
from bench.report import to_csv, to_markdown
rows = run_suite(shapes=[(1,4,512,64),(1,8,1024,64)],
                 gates=[(16,2),(32,2),(64,2),(128,2)])
to_csv(rows, \"results/local_rtx3050ti.csv\")
print(to_markdown(rows))
"'
```

## What this means for the kernel

The target is not "be faster than the naive implementation" — that is a low
bar and hitting it would prove little. The target is **attainment**: at
`block_size=16`, sparsity offers 21x and current implementations deliver
effectively none of it. A kernel that reaches even 0.3 attainment there would
be roughly a 6x real speedup over dense attention, at a block size where every
implementation we can measure is currently *slower* than dense.

---

# Kernel v1 — the finished forward pass against the same baselines

**Date:** 2026-08-22
**Device:** RTX 3050 Ti Laptop, sm_86, 4095 MiB, 20 SMs, WSL2 on WDDM
**Raw data:** `results/kernel_v1.csv`
**Config:** `block_size=128, top_k=8`, head_dim 128, bf16 — the configuration
both reference implementations benchmark.

Everything above this line is the *problem statement*: baselines measured
before any kernel existed. This section is the first measurement of the
finished forward pass — fused gate, KV-block-major attention, Triton combine —
against those same baselines.

## Result

| S | moba_triton | sdpa_dense | **vs dense** | naive_torch | moonshot | attainment |
|---:|---:|---:|---:|---:|---:|---:|
| 1024 | 0.55 ms | 0.20 ms | 0.37x | 2.68 ms | 4.17 ms | 0.365 |
| 2048 | 0.90 ms | 0.36 ms | 0.39x | 6.45 ms | 13.50 ms | 0.283 |
| 4096 | 1.73 ms | 1.28 ms | 0.74x | 18.00 ms | 33.74 ms | 0.307 |
| **8192** | **1.78 ms** | 2.70 ms | **1.52x** | 33.37 ms | 63.93 ms | **0.336** |

At `top_k=12` the picture is the same with less sparsity to exploit: 1.05x over
dense at S=8192, attainment 0.343.

`flash_moba` and `triton_moba` could not run here. FlashMoBA's extension targets
sm_80/90/100/120 and this card is sm_86; Triton-MoBA requires `flash_attn`.
The head-to-head against the actual performance target still needs rented
hardware.

## Reading it

**Against the spec's success criteria.** The target was *attainment >= 0.10 at
S=4096*; measured **0.307**, three times over. The threshold criterion
("faster than dense SDPA at S >= 2048") is not met until S ~ 6000, but that
criterion was written for `block_size=16`, the configuration corrected away
from on 2026-08-19. At `block_size=128` there is simply not enough sparsity at
S=2048 to beat dense — sparsity there is 0.719, meaning MoBA still touches 72%
of the pairs dense attention does.

**Where the crossover is.** Between S=4096 and S=8192, and it moves earlier as
context grows because sparsity does: 1.000 -> 0.719 -> 0.414 -> 0.221 as S goes
1024 -> 8192. This is the expected shape. MoBA is a long-context mechanism and
loses to dense at short context by construction, not by implementation defect.

**The decomposition that matters.** Two ratios say different things:

| | dense SDPA | moba_triton |
|---|---:|---:|
| throughput | 12.7 TFLOP/s | 4.3 TFLOP/s |
| FLOPs at S=8192 | 4.5x ours | 1x |
| latency | 2.70 ms | 1.78 ms |

**Per FLOP executed we are three times less efficient than dense**, which is
what a Triton kernel doing scattered gathers should expect against
cuBLAS-quality dense code. We win anyway because at S=8192 sparsity means doing
4.5x fewer FLOPs. Winning by doing less work while executing it worse.

That decomposition locates the next optimization precisely. Attainment is
capped by the 4.3 against 12.7 TFLOP/s gap, so closing it is worth up to
another 3x, and it lies entirely in the attention kernel — the only stage that
has had no optimization work. For contrast, the gate is now 10-13% of the pass
and the combine stage is running at the memory bus's speed
(`docs/stage-timing-measurements.md`).

## Against the reference implementations

| | S=8192 | vs moba_triton |
|---|---:|---:|
| moonshot (the original PyTorch path) | 63.93 ms | **36.0x** |
| naive_torch (our oracle) | 33.37 ms | **18.8x** |

Both are deliberately naive, so this is a floor rather than an achievement. It
is recorded because the problem statement above measured them as *slower than
dense at every configuration*, and the kernel is the first implementation in
this repo that is not.
