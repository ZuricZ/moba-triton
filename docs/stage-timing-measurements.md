# Stage timing — where the forward pass actually spends its time

**Date:** 2026-08-19
**Device:** NVIDIA GeForce RTX 3050 Ti Laptop, sm_86, 4095 MiB, 20 SMs, WDDM
**Stack:** Python 3.13.15, torch 2.13.0+cu126, triton 3.7.1, bf16 inputs
**Reproduce:** `uv run python -m bench.stages`

The design decisions D6–D10 were each justified by a measurement of the
*selection pattern* — utilization, consumers per block, gather cost, load
imbalance. None of them measured the assembled kernel. This does.

The headline is not about the kernel at all.

## Results

`block_size=128, top_k=8, head_dim=128, batch 1, heads 4`. "kernel" is
attention plus combine; the gate is reported separately because it is still
PyTorch by decision D11.

| S | part_acc | size | gate | attn | combine | kernel | gate share | combine share of kernel | rel err vs fp32 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1024 | float32 | 16 MB | 3863 us | 344 us | 116 us | 460 us | 89% | 25% | 0.0e+00 |
| 1024 | bfloat16 | 8 MB | 3638 us | 287 us | 66 us | 354 us | 91% | 19% | 4.0e-03 |
| 2048 | float32 | 32 MB | 3801 us | 735 us | 223 us | 957 us | 80% | 23% | 0.0e+00 |
| 2048 | bfloat16 | 16 MB | 3766 us | 615 us | 123 us | 738 us | 84% | 17% | 4.6e-03 |
| 4096 | float32 | 64 MB | 4250 us | 1514 us | 437 us | 1952 us | 69% | 22% | 0.0e+00 |
| 4096 | bfloat16 | 32 MB | 8187 us | 1246 us | 237 us | 1483 us | 85% | 16% | 2.5e-03 |
| 8192 | float32 | 128 MB | 6704 us | 3063 us | 863 us | 3925 us | 63% | 22% | 0.0e+00 |
| 8192 | bfloat16 | 64 MB | 9418 us | 2539 us | 465 us | 3004 us | 76% | 15% | 5.4e-03 |

**Read the float32 rows for the gate.** The gate never touches `part_acc`, so
its cost must be identical across the two arms. Where it is not — the bfloat16
rows at S=4096 and 8192 — that is measurement noise, not signal. This is a
control, not a coincidence: it was included so noise would be visible rather
than quietly averaged in.

## The D11 trigger has fired

The gate is **63–89% of the forward pass**, and it was never in the design's
critical path on paper. D11 deferred the gate to PyTorch on the explicit
argument that it should be measured before being optimized. That measurement
now exists and the answer is unambiguous: **the gate is the bottleneck, and the
kernel is not.**

The scaling says what kind of cost it is:

| stage | S=1024 | S=2048 | S=4096 | S=8192 | growth per doubling |
|---|---:|---:|---:|---:|---|
| gate | 3863 | 3801 | 4250 | 6704 | ~flat, then ~1.6x |
| attn | 344 | 735 | 1514 | 3063 | 2.0x |
| combine | 116 | 223 | 437 | 863 | 2.0x |

The two kernel stages double exactly when S doubles — linear in the work they
do, which is `S * top_k` block-query pairs. The gate is flat to S=4096 and only
then starts to grow. A cost that does not scale with the problem is not
compute; it is **fixed overhead**, and under WDDM that means kernel launch
latency across the roughly twenty separate PyTorch operations inside
`build_csc` and `moba_selection`.

Two consequences:

1. The 63–89% figure is not a fixed property. It shrinks as S grows, because
   the numerator is flat and the denominator is linear. At the spec's target of
   S >= 16384 the gate share would be materially smaller.
2. Because it is launch overhead rather than arithmetic, the fix is **fusion**,
   not a faster algorithm. Fewer, larger kernels.

Component timings inside the gate at S=4096 (each stable to within 1.2x across
repeats, unlike the composed calls):

| | time |
|---|---:|
| `block_representatives` (mean-pool K) | 341 us |
| `select_blocks` — of which the stable argsort is ~530 us | 579 us |
| `einsum` gate scores | 42 us |
| `nonzero()` in `build_csc` | 707 us |
| `cumsum` for `slot_grid` | 192 us |

No single operation dominates. That is itself the diagnosis: the cost is spread
across many small launches, which is exactly the shape of a fusion problem.

Note also that `select_blocks` uses a full stable `argsort` where a `top_k`
would do, because stability is required for deterministic tie-breaking. On this
device stable argsort costs 532 us against 324 us for `topk(8)` at N=32 — a
real but secondary difference, and the determinism is worth more than 200 us.

## Storing `part_acc` in bfloat16

`part_acc` is the largest single term in the design's memory traffic. At
S=4096, head_dim 128, top_k 12 it is 24 MB written and 24 MB read per
batch-head, against 15 MB for all of Q, K and V combined.

Halving it behaves exactly as the traffic model predicts:

| stage | fp32 | bf16 | change |
|---|---:|---:|---|
| combine (S=4096) | 437 us | 237 us | **-46%** |
| combine (S=8192) | 863 us | 465 us | **-46%** |
| attn (S=4096) | 1514 us | 1246 us | -18% |
| attn (S=8192) | 3063 us | 2539 us | -17% |
| kernel total (S=8192) | 3925 us | 3004 us | **-23%** |

Combine halves, because it is purely memory-bound and its traffic is almost
entirely `part_acc`. Attention improves less because `part_acc` is only its
write traffic; the gathered Q reads are unchanged.

Accuracy against the fp64 oracle under the D5 tolerance policy, all
configurations passing:

| shape | fp32 partials | bf16 partials | budget |
|---|---:|---:|---:|
| (1,4,1024,128) | 1.231e-02 | 1.396e-02 | 1.914e-02 |
| (1,4,2048,128) | 1.407e-02 | 1.407e-02 | 1.537e-02 |
| (1,2,4096,128) | 9.397e-03 | 1.189e-02 | 1.537e-02 |

**Verdict: available, not default.** `moba_forward(..., partials_dtype=...)`
exposes it and it is correct, but float32 remains the default for two reasons.
First, 23% of the kernel is only about 7% end-to-end while the gate dominates,
so the gate is the change worth making first. Second, it consumes roughly a
quarter of the remaining error budget in a forward pass that a backward pass
would later compound. Revisit once the gate is fused and the number matters.

## Measurement notes — this box is hostile to benchmarking

This machine runs **WDDM**: the GPU is shared with the desktop compositor and
work submissions are batched by the display driver. Two consequences bit hard
enough to be worth recording.

**Allocation is not free and must be hoisted out of timed regions.** An earlier
version of `bench/stages.py` allocated the partial buffers inside the timed
call and reported `index` at **120% of total** — arithmetically impossible.
Under WDDM a `cudaMalloc` goes through the display driver and costs
milliseconds; repeated `do_bench` runs fragmented the caching allocator until
it began genuinely calling the driver. That measured the Windows memory
manager, not the kernel. `attend_partials` now accepts pre-allocated buffers
(`empty_partials`) so the benchmark can pay allocation once — which is also a
real improvement to the library, since allocation was 400–900 us of production
time per call.

**`build_csc` cannot be fully hoisted**, because `nonzero()` returns a
data-dependent size that the caching allocator cannot reuse. The gate numbers
above are therefore an *upper bound* on its true compute cost.

**Take the minimum of repeated measurements, not the mean.** Composed calls
were observed to spread 1.4x run to run while individual components held to
1.2x. The minimum is the run least polluted by the compositor; a mean averages
over how busy the desktop happened to be.

**Absolute numbers from this box should not be quoted.** Ratios and scaling
trends survive the noise, and those are what this document reports. The
spec's performance criteria at S >= 16384 need rented hardware.

## What this does not measure

No comparison against dense SDPA or against the reference MoBA
implementations appears here — that is `bench/suite.py`'s job, and our kernel
now runs in it automatically as the `moba_triton` backend. This document is
attribution only: given that the forward pass takes some time, where does it
go.

The query-major control kernel remains unbuilt, so the wall-clock crossover
between the two loop orders is still inferred from the traffic model rather
than measured.
