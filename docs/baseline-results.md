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
