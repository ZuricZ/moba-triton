# A100-40GB — the head-to-head the project was built to get

**Date:** 2026-09-10
**Device:** NVIDIA A100-PCIE-40GB, sm_80, 108 SMs, 164 KiB shared memory,
driver 595.58.03, CUDA 12.6.77
**Config:** `block_size=128, top_k=8` (FlashMoBA's published gate), bf16,
`[1, S, 4, 64]`
**Source:** `bench/compare.py`, run twice against two flash-attn builds.
Raw logs in `results/sm80-a100/logs/twopass.log` and `results/sm80-a100/logs/pass2.log`; per-pass
CSVs in `results/sm80-a100/data/fa283-*.csv` and `results/sm80-a100/data/fa263-*.csv`.

Every previous number in this repo came from a 20-SM laptop with 4 GB and
99 KiB of shared memory. Those measured a kernel entangled with one card. This
is the first run on hardware the references were designed for, and the first
with **all five** MoBA implementations live at once.

---

## Forward

All five MoBA implementations, plus two dense baselines. Latency in ms; lower is
better.

| S | blocks | **ours** | flash_moba | triton_moba | moonshot_eff | moonshot_naive | sdpa_dense | fa2_minimal |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1024 | 8 | **0.163** | 0.454 | 1.664 | 1.740 | 0.989 | 0.027 | 0.031 |
| 2048 | 16 | **0.207** | 0.507 | 1.672 | 1.734 | 1.987 | 0.043 | 0.067 |
| 4096 | 32 | **0.250** | 0.640 | 1.787 | 1.849 | 5.829 | 0.144 | 0.155 |
| 8192 | 64 | **0.419** | 1.030 | 2.387 | 2.422 | 19.342 | 0.352 | 0.455 |
| 16384 | 128 | **0.692** | 2.108 | 4.669 | 5.063 | 67.685 | 1.182 | 1.561 |
| 32768 | 256 | **1.262** | 5.080 | 13.573 | 17.144 | OOM | 4.062 | 5.694 |
| 65536 | 512 | **2.727** | 12.419 | 47.876 | 57.162 | OOM | 15.180 | 21.579 |

## Backward (`forward().backward(dO)`, the full step)

| S | **ours** | flash_moba | triton_moba | moonshot_eff | sdpa_dense | fa2_minimal |
|---:|---:|---:|---:|---:|---:|---:|
| 1024 | **0.551** | 0.845 | 3.099 | 2.702 | 0.111 | 0.378 |
| 2048 | **0.539** | 0.879 | 3.106 | 2.809 | 0.184 | 0.734 |
| 4096 | **0.502** | 0.977 | 3.585 | 2.863 | 0.400 | 1.523 |
| 8192 | **0.846** | 1.423 | 4.454 | 4.000 | 1.171 | 5.008 |
| 16384 | **1.373** | 2.719 | 10.497 | 8.236 | 3.828 | 15.446 |
| 32768 | **2.654** | 6.311 | 34.289 | 24.516 | 13.821 | 57.896 |
| 65536 | **5.237** | 14.343 | 131.464 | 75.270 | 53.921 | 231.015 |

## The ratios (forward, x times slower than ours)

| S | sdpa_dense | flash_moba | triton_moba | moonshot_eff |
|---:|---:|---:|---:|---:|
| 8192 | 0.84x | 2.46x | 5.70x | 6.16x |
| 16384 | **1.71x** | 3.05x | 6.74x | 7.26x |
| 32768 | **3.22x** | 4.03x | 10.76x | 14.06x |
| 65536 | **5.57x** | 4.55x | **17.56x** | **20.73x** |

**The central claim holds and strengthens with context.** Against dense the
margin runs 0.84x -> 1.71x -> 3.22x -> 5.57x. Interpolated in log-log space,
`ours` overtakes `sdpa_dense` at **S ~ 9,700** forward; the backward crossover
is around S ~ 5,400 but sits inside the harness-dominated region below, so
treat it as a lower bound.

**Read the small-S rows with care.** Below S=8192 a large part of the measured
step is harness rather than kernel -- 47% at S=1024, 21% at 2048, 11% at 4096,
and gone by 8192 (`bench/_small_s_dip.py`, `results/sm80-a100/logs/dip-overhead.log`).
That is why the backward column *falls* from S=1024 to S=4096 while the work
doubles: a rising kernel cost outrun by a faster-falling harness cost. The
figures shade that region. Every headline number above is quoted from S >=
16384, where the overhead is negligible and the ratios are larger anyway.

## The two passes, and what a cross-pass comparison is worth

`moonshot_efficient` needs flash-attn 2.6.x and `triton_moba` needs >= 2.7, so
no single environment runs both (`docs/environment.md`). The sweep therefore ran
twice, switching flash-attn between passes; the `moonshot_eff` column comes from
the second pass and every other column from the first.

Four backends -- `ours`, `flash_moba`, `sdpa_dense`, `fa2_minimal` -- do not
depend on flash-attn and were measured identically in both. Their spread is a
free estimate of run-to-run reproducibility on this box:

| backend | forward median / worst | backward median / worst |
|---|---|---|
| `ours` | 1.1% / **6.5%** | 1.3% / **13.7%** |
| `flash_moba` | 0.2% / 1.2% | 1.2% / 7.9% |
| `sdpa_dense` | 0.1% / 0.8% | 0.3% / 0.8% |
| `fa2_minimal` | 0.0% / 1.0% | 0.0% / 0.8% |

**The worst cases fall on exactly the two implementations that route.** The
dense kernels reproduce to under a percent across runs an hour apart; the sparse
ones do not. That is not thermal, which would hit everything equally -- it is
that a gate near a scoring tie can pick different blocks between runs, changing
column lengths and therefore load balance. `sdpa_dense` does the same arithmetic
every time; `ours` does slightly different arithmetic every time.

So the error bar is **~1% for dense, up to ~14% for anything that routes**, and
it applies only across the pass boundary. Consequences:

* `ours` vs `flash_moba` and `ours` vs `triton_moba` are **within one pass** --
  same process, same card state -- and need no such allowance.
* `triton_moba` vs `moonshot_efficient` **crosses the boundary** and they differ
  by 4-19%. Those two are **tied**, and are reported as tied.

## Why beating FlashMoBA is not suspicious

> **Superseded in part, 2026-09-20.** The mechanism below was narrowed by four
> panels of measurement in `bench/residual.py`: their gate, under-subscription
> and occupancy are all eliminated as explanations, and the window turns out to
> be correctly tuned for sm_80 rather than mistuned from an H100. What survives
> is traffic. The full argument, with the eliminations, is
> `docs/why-we-beat-flashmoba.md`.

A 4.55x win over a CUDA kernel from the lab that published the paper on making
MoBA fast deserves scrutiny before it is written down. Three checks.

### It is computing the same thing

An earlier version of this section inferred "same block choices" from all
three implementations reporting the same error against the oracle, 0.0478 at
S=4096. That inference was wrong. The metric (`bench/_a100_fairness.py`) is one
scalar, max abs error over max abs reference, and one number agreeing proves
nothing about which blocks were chosen; the same log has them *disagreeing* at
S=1024 (`ours` 0.0037, `flash_moba` 0.0019).

What does prove it is comparing the selections directly. Done 2026-09-15 on the
laptop at the sweep's config, `[1, S, 4, 64]`, bf16, seed 0:

| S | (query, block) pairs the oracle selects | ours | flash_moba | ours vs oracle | flash_moba vs oracle |
|---:|---:|---:|---:|---:|---:|
| 4096 | 116,736 | 116,736 | 116,736 | 0 differ | 522 differ |
| 8192 | 247,808 | 247,808 | 247,808 | 0 differ | 1,658 differ |

Both attend exactly the same *number* of pairs under the same convention (the
query's own block counts against `top_k`:
`references/flash-moba/flash_moba/moba_test_utils.py:87-90` and
`src/moba/config.py`). `ours` reproduces the oracle's choices exactly. FlashMoBA
flips about 0.5% of entries on near-tie scores, which is its rounding, not ours.
Nothing is being skipped, and the direction of any divergence is theirs.

No test in `tests/` compares these two backends directly:
`tests/test_disagreement.py` runs at `block_size=32, float32`, which
`flash_moba` refuses (needs a multiple of 64) and `ours` refuses (no fp32
tensor-core path). That gap is still open.

### The gap grows on realistic shapes

The sweep uses `[1, S, 4, 64]`, a small workload. If the win were an artefact
of leaving a 108-SM card idle, a wider shape would close it:

| shape | ours | flash_moba | ratio |
|---|---:|---:|---:|
| B1 S8192 H4 D64 (sweep) | 0.401 | 1.042 | 2.60x |
| B1 S8192 H32 D128 | 2.649 | 8.144 | 3.07x |
| B4 S8192 H32 D128 | 9.618 | 29.984 | 3.12x |
| B1 S16384 H32 D128 | 5.462 | 18.043 | 3.30x |

It widens.

### It is the attention stage, not the gate

Both pipelines split at the same seam -- routing, then attention -- so the
halves can be timed separately:

| shape | stage | ours | theirs | ratio |
|---|---|---:|---:|---:|
| B1 S8192 H4 D64 | gate + routing | 0.187 | 0.390 | 2.08x |
| | attention + combine | 0.217 | 0.686 | **3.16x** |
| B1 S8192 H32 D128 | gate + routing | 0.666 | 1.277 | 1.92x |
| | attention + combine | 2.026 | 6.866 | **3.39x** |

**The win is concentrated in the attention kernel**, which is the stage
FlashMoBA exists to optimise, and the only stage where the two designs differ
structurally.

**The mechanism, corrected 2026-09-15.** An earlier version of this paragraph
said FlashMoBA's `lg_block_m=512` tile "loads the union of every block any of
its queries selected, then masks away the rest", wasting 77% of tensor-core
work. Their kernel does not do that, and the 77% was never a measurement of
FlashMoBA: it comes from `bench/divergence.py`, which models a hypothetical
query-major kernel with 64-query tiles. `docs/traffic-derivation.md` derives
the correction from their source. The summary:

* **Their grid is one program per (query window, batch, head)**
  (`flash_fwd_launch_template.h:51-52`). The window `lg_block_m` is 512 at
  S=8192 and **1024** at S=65536 on sm_80. Inside the program the loop is
  KV-block-major: for each block it binary-searches the block's sorted query
  list for the queries in its window (`index_gather_scatter.h:50-87`) and
  gathers only those. No union, no masking of unselected blocks.
* **The window is what costs them at long context.** Each (window, block) pair
  holds on average `lg_block_m * top_k / num_blocks` queries, padded to a
  64-row MMA tile: 64 at S=8192 (about 22% waste from part-full tiles) but 16
  at S=65536 (about 55% padding). Ours gathers across the whole sequence, so
  every block sees about `top_k * block_size` = 1024 queries at any S. This is
  why the ratio grows with context.
* ~~**One program per SM.**~~ **Wrong, retracted 2026-09-20.** This said
  their 84 KiB of shared memory per program left room for one 4-warp program
  against our two, at about 6% occupancy. The arithmetic was wrong -- 164 / 84
  is 1.95, so two of theirs fit -- and profiling both kernels on the card
  settles it rather than arguing it:

  ```
  kernel                      regs     smem   blk/SM  warps  occ  limited by
  ours   moba_fwd_kernel       230   49,920        2      8   12%  SMEM
  theirs (77.1% of their run)  241   86,016        2      8   12%  SMEM
  ```

  Identical occupancy. It is not a difference between these kernels, and the
  gap has to be explained without it. See `docs/why-we-beat-flashmoba.md`
  section 3, and `results/sm80-a100/logs/occ_fm.log`.
* **The running output lives in HBM.** The gathered query set changes per
  block, so the accumulator cannot stay in registers. It is an fp32 scratch
  buffer (`flash_moba_api.cpp:505`, zeroed every call, 67 MB at this shape),
  loaded and stored once per block per 64-row chunk with blocking copies on
  sm_80 (`kernel_traits.h:178-181`). Modelled traffic puts them about 4x above
  their own DRAM floor, which is what a latency-bound kernel looks like.
* **Host overhead.** `triton_mean_pool.py:140` calls `.item()` on every call,
  a hard device-to-host sync, so iterations cannot overlap; six or more kernel
  launches per call, not two. **Measured 2026-09-20 and it is not the main
  term**: timing their gate and attention apart puts their *attention alone*
  at 8.35x ours at S=65536, a larger ratio than the whole pipeline's 4.62x.
  Real, but not what the gap is made of.

The window is a trade, not a mistake. It lets them merge online-softmax
partials on the fly inside one kernel with a scratch buffer 1x the output.
Ours merges in a second kernel with a buffer `top_k`x the output
(`moba_fwd.py`, `part_acc`). That is why ours is faster here and why theirs
still fits at S=1M where ours, as written, would not (64 GiB at their
`[2, S, 16, 128]`). See `docs/deferred-decisions.md`, DD-2.

That the gate ratio (1.9-2.1x) is *smaller* than the attention ratio (3.2-3.4x)
is itself corroboration. Both pipelines compute the same routing by similar
means; only the attention differs by design. A uniform win across both stages
would have suggested a measurement artefact instead.

---

## What the sweep confirms about the algorithm

**Cost is linear in S, and the measurement says so.** MoBA attends
`top_k * block_size` = 1024 keys per query whatever S is, so doubling S should
roughly double the time: 0.692 -> 1.262 -> 2.727 ms is 1.82x then 2.16x.

**The dense baselines die where predicted.** `naive_torch` and
`moonshot_naive` both OOM at S=32768 trying to allocate 16 GiB -- attention
memory quadratic in S -- while `ours` runs S=65536 in 2.7 ms. That is not a
speed result, it is a feasibility one.

**Read `TF/s attn`, not `TF/s dense`.** The sweep prints both conventions. The
dense-equivalent figure reaches 803 TFLOP/s at S=65536, which is not real
throughput -- it counts FLOPs a dense kernel *would* have done. Attended
throughput, the honest column, is flat across the whole sweep: 20.4, 24.6,
27.3, 25.1. **Flat attended throughput beside an exploding dense-equivalent
number is exactly what sparsity looks like**, and is why the harness refuses to
print one without the other.

**The crossover moved.** On the laptop `ours` overtook dense at S=4096; here it
is between 8192 and 16384. An A100 runs dense attention far faster in absolute
terms (0.361 ms at S=8192 against the laptop's 2.814), so dense holds on longer
before quadratic cost catches it. Same kernel, same algorithm, different
crossover -- which is precisely why one machine could not settle this.

---

## Triton against Triton, which is the cleanest comparison here

`triton_moba` is the fifth implementation and the most directly comparable of
all, because it holds the *language* constant. A sceptic can attribute part of
the FlashMoBA gap to CUDA-versus-Triton, to hand-tuning, to a more mature
compiler path. None of that applies here: both are Triton, both call into the
same ecosystem, both were compiled by the same triton 3.7.1.

It is **5.7x to 17.6x slower than ours**, and the shape of that column is the
argument:

| S | ours | triton_moba | ratio |
|---:|---:|---:|---:|
| 8192 | 0.419 | 2.387 | 5.70x |
| 16384 | 0.692 | 4.669 | 6.74x |
| 32768 | 1.262 | 13.573 | 10.76x |
| 65536 | 2.727 | 47.876 | **17.56x** |

**A constant-factor advantage would show as a flat ratio.** Better tiling, a
luckier autotune, cheaper Python -- all of those scale with the work and leave
the ratio roughly where it started. This ratio *grows*, from 5.7x to 17.6x over
three doublings, which means the two implementations differ in how they scale on
the sparsity rather than in their constants.

That is what the loop-order argument predicts. Triton-MoBA reuses
FlashAttention's varlen kernel for the attention itself, which is query-major: a
tile owns a range of queries and loads the union of every block any of them
selected. As `num_blocks = S / block_size` grows, the queries inside one tile
agree less about which `top_k` blocks they want, the union grows, and the useful
fraction of each loaded tile shrinks. A KV-block-major program owns one block
and gathers exactly the queries that chose it, so its useful fraction is 100% at
every S.

**`triton_moba` is also slower than dense attention at long context** -- 47.9 ms
against `sdpa_dense`'s 15.2 ms at S=65536. A MoBA implementation losing to the
dense kernel it is supposed to replace means its sparsity is not paying for its
routing at all, which is precisely the state of affairs FlashMoBA's paper
describes as the motivation for its own existence.

### The three tiers are loop order, not language

`triton_moba` and `moonshot_efficient` land within 19% of each other at every
sequence length -- 1.664 vs 1.740 at S=1024, 47.9 vs 57.2 at S=65536 -- which is
inside the cross-pass error bar above. They are tied. That is not a coincidence:
**both delegate the attention to FlashAttention's varlen kernel** and differ only
in how they compute the gate. Same attention, same loop order, same ceiling.

So the five sort into three tiers by loop order:

| tier | implementations | at S=65536 |
|---|---|---|
| query-major on FA-varlen | `triton_moba`, `moonshot_efficient` | 17.6x, 20.7x |
| KV-block-major gather, but only inside a 512-1024-query window; hand-written CUDA | `flash_moba` | 4.6x |
| **KV-block-major** | **ours** | **1x** |

Language explains the gap *within* the first two tiers. It does not explain the
gap between them and us.

### The roofline says the same thing independently

`results/sm80-a100/figures/roofline.png` was drawn with `flash_moba` and `moonshot_efficient` at
**186.2 FLOP/byte**, above the A100's measured ridge of 153.5, at **4% and 2%
of roof**. That intensity was an artefact of the traffic model: the version of
`bench/roofline.py::traffic_flash_moba` in use counted only gathered Q, K, V
once each, and O, which is a floor on *any* gathering kernel, not FlashMoBA's
traffic. It omitted their per-(window, block) K/V reloads and the fp32
running-output round trip, which `docs/traffic-derivation.md` §8.2 counts at
2.9x our bytes at S=8192. The model now includes both (2026-09-15): 340 MiB
modelled against 326.5 MiB counted by hand at S=8192, which puts FlashMoBA at
**24 FLOP/byte**, well left of the ridge and memory bound like `ours` at 75.7,
but moving 3-4x the bytes for the same arithmetic. Against that lower roof it
reaches 25% of what its position allows, next to our 21%: both kernels move
bytes at a similar rate, and the one moving more of them takes longer. That is
the roofline agreeing with the stage split and with
`docs/traffic-derivation.md` §9 rather than contradicting them.
`moonshot_efficient` had been borrowing the same model; it now has none until
one is derived from its source, so it is absent from the figure rather than
misplaced. `results/sm80-a100/figures/roofline.png` was regenerated from `results/sm80-a100/logs/roofline.log`
with `--from-log`, which now re-models traffic from the current code rather than
trusting the intensity the log stored; the latencies on it are unchanged.

## Environment: two passes, because flash-attn cannot serve both

`triton_moba` unpacks four values from the private
`_flash_attn_varlen_forward`; `moonshot_efficient` unpacks eight. The signature
changed at flash-attn 2.7, so the two are mutually exclusive and the sweep runs
twice -- `bootstrap_remote.sh --fa=` builds either, `--use-fa=` switches in a
second. `docs/environment.md` carries the API table, why a compatibility shim
was rejected, and why FA3 and FA4 are unavailable on sm_80.

Both backends now gate on the *signature* rather than on whether `flash_attn`
imports. It imports fine against either build and fails at call time, which
previously surfaced as fourteen failing correctness tests instead of one
unavailable backend.

## Caveats

* **`ncu` was refused here too.** The container had neither `CAP_SYS_ADMIN` nor
  `CAP_PERFMON`, and the host driver restricts profiling to admins -- verified
  by compiling a CUDA kernel and profiling it. Occupancy came from
  `torch.profiler` instead; DRAM traffic remains modelled rather than measured.
* **Clocks were not lockable**, so the cool/warm/best-of-3 thermal protocol ran
  here as on the laptop. A racked A100 is far more stable, so this is belt and
  braces rather than load-bearing.
* **Small S measures the harness.** Below S=8192, 11-47% of the backward step
  is Python rather than CUDA. Headline numbers are quoted from S >= 16384.
* **Two passes.** The `moonshot_efficient` column was measured in a separate run
  from the rest; see the reproducibility control above before comparing it
  across that boundary.
* **One config.** `block_size=128, top_k=8` throughout, and `[1, S, 4, 64]` for
  the sweep. The shape check above covers H=32, D=128 and batch 4 for the
  FlashMoBA comparison only.
* **Forward and backward, no end-to-end model.** Nothing here measures a
  training step of a real network.
