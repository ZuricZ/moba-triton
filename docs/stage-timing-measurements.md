# Stage timing — where the forward pass spends its time

**Date:** 2026-08-22
**Device:** NVIDIA GeForce RTX 3050 Ti Laptop, sm_86, 4095 MiB, 20 SMs, WSL2 on WDDM
**Stack:** Python 3.13.15, torch 2.13.0+cu126, triton 3.7.1, bf16 inputs
**Reproduce:** `uv run python -m bench.stages`

Decisions D6–D10 were each justified by measuring the *selection pattern* —
utilization, consumers per block, gather cost, load imbalance. None measured
the assembled kernel. This does.

Its first result overturned the project's assumption about where the time went,
and led to rewriting the gate. Both the before and after are recorded here,
because the "before" is what justified the work.

## Result

`block_size=128, top_k=8, head_dim=128, batch 1, heads 4`. "kernel" is
attention plus combine.

| S | part_acc | size | gate | attn | combine | kernel | gate share | combine share of kernel |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1024 | float32 | 16 MB | 182 us | 345 us | 119 us | 465 us | 28% | 26% |
| 1024 | bfloat16 | 8 MB | 178 us | 287 us | 66 us | 353 us | 34% | 19% |
| 2048 | float32 | 32 MB | 195 us | 743 us | 230 us | 973 us | 17% | 24% |
| 2048 | bfloat16 | 16 MB | 193 us | 623 us | 123 us | 746 us | 21% | 16% |
| 4096 | float32 | 64 MB | 232 us | 1526 us | 453 us | 1979 us | 10% | 23% |
| 4096 | bfloat16 | 32 MB | 235 us | 1270 us | 237 us | 1507 us | 13% | 16% |
| 8192 | float32 | 128 MB | 468 us | 3116 us | 904 us | 4020 us | 10% | 22% |
| 8192 | bfloat16 | 64 MB | 469 us | 2572 us | 466 us | 3039 us | 13% | 15% |

**The two `part_acc` arms are a control, not a duplicate.** The gate never
touches those buffers, so its cost must be identical across them — 468 against
469 us at S=8192, 232 against 235 at S=4096. Where the arms disagree, the
measurement is broken and not the code. That check caught two separate
benchmark bugs, both described below.

## What the first run found, and what changed

The original measurement put the **gate at 63–89% of the forward pass**. D11
had deferred the gate to PyTorch on the explicit condition that it be measured
before being optimized; that measurement fired the trigger.

The diagnosis came from the scaling rather than the share:

| stage | S=1024 | S=2048 | S=4096 | growth per doubling |
|---|---:|---:|---:|---|
| gate (PyTorch) | 3863 | 3801 | 4250 | ~flat |
| attn | 344 | 735 | 1514 | 2.0x |
| combine | 116 | 223 | 437 | 2.0x |

The two kernel stages double exactly when S doubles — linear in the
`S * top_k` block-query pairs they process. The gate barely moved. **A cost
that does not grow with the problem is not doing the work**, so the gate was
carrying a large fixed component.

Profiling then identified it, and corrected two guesses along the way (see
"Corrections" below). Per gate call the GPU was busy ~10 ms and the CPU ~12.5 ms
under the profiler, so neither side was idle — the work was real. But its
composition was:

| | share of gate GPU time |
|---|---:|
| `aten::copy_` (21 calls per gate) | 31% |
| `Memcpy DtoD` | 9% |
| `aten::sort` | 15% |
| `aten::sum` | 12% |
| `aten::sub` | 12% |
| `aten::cumsum` | 7% |
| `aten::index` | 6% |

**~40% was moving intermediates, not computing.** No single operation
dominated, which is the signature of unfused eager PyTorch: roughly 35
operations in a chain, each writing a full-size array to memory and reading it
back for the next.

## The rewrite

Three fused Triton kernels, following FlashMoBA's three-stage pipeline
(`arxiv.org/abs/2511.11571`) and fla-org's NSA `parallel_nsa_kernel_topk` for
the Triton form of stage 2.

| stage | what it does | PyTorch | Triton | |
|---|---|---:|---:|---:|
| 1 · mean-pool | one representative per block | 443 us | 42 us | 10.5x |
| 2 · rank | score + top-k, in registers | 551 us | 98 us | 5.6x |
| 3 · transpose | count → prefix sum → atomic scatter | ~2400 us | 103 us | ~23x |

End to end, at `block_size=128, top_k=8, heads 4, head_dim 128`:

| S | PyTorch gate | fused gate | |
|---:|---:|---:|---:|
| 1024 | 4102 us | 184 us | 22.2x |
| 2048 | 3301 us | 173 us | 19.1x |
| 4096 | 3461 us | 232 us | 14.9x |
| 8192 | 5293 us | 464 us | 11.4x |

**The speedup shrinking with S is the point, not a disappointment.** The old
gate's cost was mostly constant, so removing it helps most where there is least
other work. The honest statement is not "22x faster gate" but "a fixed cost of
roughly 3.3 ms was deleted".

Routing is unchanged: selections match the PyTorch gate exactly across the full
shape × gate sweep, and `col_nnz` and `col_offsets` match element for element.

### Host synchronization, which was the largest single item

A device-to-host round trip on this box costs **~230 us**, against **~6 us** for
the same arithmetic kept on device. A 37x penalty for letting the CPU look at a
result, and it is flat regardless of how much data is involved — you pay for
the trip, not the cargo.

The PyTorch gate performed two. `nonzero()` forces one because its output size
is data-dependent, so PyTorch must copy a count back before it can size an
allocation; mean-pooling forced the other by writing the tail block's occupancy
from the host. **The fused gate performs zero**, asserted by
`test_fused_index_never_synchronizes_with_the_host` — a reintroduced sync fails
no correctness test, so it needs a test of its own.

The fused stage 3 needs no count because the gate emits a *list* of selected
block ids rather than a dense mask, and the total is bounded above by
`S * top_k`, known before anything runs.

### One guarantee dropped

The PyTorch builder emitted each column's query indices ascending, free from
the memory order it read the dense mask in. The fused builder has queries claim
slots with an atomic counter, so a column's entries come out in race order.

Safe on both counts, and both were already measured. Results are unchanged
because the attention kernel addresses its output by `row * K_SLOTS + slot` —
neither term depends on position within a column. Speed is unchanged because
`docs/gather-cost-measurements.md` measured a *fully* randomized gather at
1.03x, a query row being one cache line wherever it sits. FlashMoBA accepts the
same shuffling for the same reason.

## Storing `part_acc` in bfloat16

`part_acc` is the largest single term in the design's memory traffic. At
S=4096, head_dim 128, top_k 12 it is 24 MB written and 24 MB read per
batch-head, against 15 MB for all of Q, K and V combined.

Halving it behaves as the traffic model predicts:

| stage | fp32 | bf16 | change |
|---|---:|---:|---|
| combine (S=4096) | 453 us | 237 us | **-48%** |
| combine (S=8192) | 904 us | 466 us | **-48%** |
| attn (S=8192) | 3116 us | 2572 us | -17% |
| kernel total (S=8192) | 4020 us | 3039 us | **-24%** |

Combine halves because it is purely memory-bound and its traffic is almost
entirely `part_acc`. Attention gains less because `part_acc` is only its write
traffic; the gathered Q reads are unchanged.

Accuracy against the fp64 oracle under the D5 policy, all passing:

| shape | fp32 partials | bf16 partials | budget |
|---|---:|---:|---:|
| (1,4,1024,128) | 1.231e-02 | 1.396e-02 | 1.914e-02 |
| (1,4,2048,128) | 1.407e-02 | 1.407e-02 | 1.537e-02 |
| (1,2,4096,128) | 9.397e-03 | 1.189e-02 | 1.537e-02 |

**Verdict: available, not default.** `moba_forward(..., partials_dtype=...)`
exposes it. It spends roughly a quarter of the remaining error budget in a
forward pass that a backward pass would compound, and the combine stage it
helps most is already running at the memory bus's speed (see below), so the
gain is real but bounded. FSA stores its partial outputs in bfloat16, which is
precedent for eventually flipping the default.

## The combine pass is finished

At head_dim 128 with 12 slots and S=8192 the combine pass moves ~221 MB in
1313 us — **168 GB/s**. A bare `torch.zeros` on the same card runs at
**163 GB/s**. The combine kernel is running at this machine's memory bandwidth.

Consequently tile shape is irrelevant to its speed. Measured, every tile from 1
to 64 queries per program lands within 1% of the same time:

| queries per program | time | registers |
|---:|---:|---:|
| 1 | 1316 us | 84 |
| 8 | 1309 us | 168 |
| 16 | 1315 us | 181 |
| 64 | 1311 us | 255 |

So the tile is chosen for robustness rather than speed: 8 uses a quarter of the
register file and cannot spill, where 64 sat exactly on the 255-register
ceiling and stayed off it only by deriving the right warp count. This follows
FSA's `reduce_kernel`, which runs one query per program. The wide tile was
inherited from FlashAttention-2, where it is forced by `tl.dot` needing a matrix
to fill a tensor core; there is no `tl.dot` in this pass, so nothing forced it.

The only remaining lever on this stage is moving fewer bytes — which is exactly
what the bfloat16 result above shows, at -48%.

## Corrections to earlier versions of this document

Two claims here were wrong and are recorded rather than quietly removed.

**"Buffer allocation is expensive because of the Windows driver."** Wrong
cause. `torch.empty` costs ~2 us at any size — the caching allocator just hands
back a pointer. What costs 394 us at 64 MB is `torch.zeros`, and it scales with
size at ~163 GB/s, this card's memory bandwidth. That is the zeroing, and it is
physics, not the OS. (A *genuine* `cudaMalloc` of 64 MB does cost 4.6 ms here,
which is a real WDDM effect — but the caching allocator normally prevents one
from happening.)

**"The gate's fixed cost is kernel launch overhead."** Wrong mechanism. A
launch costs 6 us here; twenty of them is 120 us, about 3% of the gate. The
fixed cost was the two host synchronizations at ~230 us each, plus copy
traffic that is fixed in S because `num_blocks` grows while the per-block work
shrinks. The flatness was correctly observed and then twice misattributed;
the profiler settled it.

## Measurement notes — this box is hostile to benchmarking

WSL2 has no NVIDIA kernel driver. CUDA calls forward through `/dev/dxg` to the
Windows driver, so Windows costs — WDDM submission batching, expensive real
allocations, slow host round trips — apply despite the Linux userspace.

**Allocation must be hoisted out of timed regions.** An early version allocated
inside them and reported `index` at **120% of total**, which is impossible.
Repeated `do_bench` runs had fragmented the caching allocator until it began
calling the driver for real, so the measurement was of the Windows memory
manager. `attend_partials` now accepts pre-allocated buffers via
`empty_partials`, which is also a real improvement to the library — allocation
was 400–900 us of production time per call.

**Do not hold large tensors alive across a timed region.** A later version
computed the bfloat16 accuracy check inside every timing run, holding two extra
full-size outputs beside the timing buffers. At S=8192 that is 128 MB of
partials on a 4 GiB card, and the allocator thrashed: the *gate* was reported
26x slower while the attention stage barely moved — impossible, since the gate
never touches those buffers. The accuracy probe now runs once at a fixed small
shape, which is correct anyway: the error is a property of the storage dtype,
not of the sequence length.

**Take the minimum of repeated measurements, not the mean.** Composed calls
spread 1.4x run to run while individual components held to 1.2x. The minimum is
the run least polluted by the desktop compositor.

**Absolute numbers from this box should not be quoted.** Ratios and scaling
trends survive the noise, and those are what this document reports. The spec's
performance criteria at S >= 16384 need rented hardware.

## What this does not measure

No comparison against dense SDPA or the reference MoBA implementations — that
is `bench/suite.py`'s job, and our kernel runs there automatically as the
`ours` backend (registered as `moba_triton` when this was written).

The query-major control kernel remains unbuilt, so the wall-clock crossover
between the two loop orders is still inferred from the traffic model rather
than measured.
