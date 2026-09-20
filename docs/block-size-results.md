# Shrinking the MoBA block, measured

**Date:** 2026-09-20
**Device:** A100-PCIE-40GB, sm_80 | bf16 | B=1, H=4, head_dim 64
**Source:** `bench/block_sweep.py` (`uv run python -m bench.block_sweep`)
**Data:** `results/sm80-a100/data/block-sweep-budget.csv`, `results/sm80-a100/data/block-sweep-fixed-top-k.csv`
**Confirmation run:** `results/sm86-3050ti/data/block-sweep-*.csv`, same shape on 20 SMs

## What this asked, and what it answered instead

`num_blocks = seq_len / block_size` is the quantity this kernel's advantage is
supposed to track. Two knobs produce a large one: raise the numerator, or
shrink the denominator. Every other result here turns the first knob. This
sweep turns the second, and its stated hypothesis was that the knob should not
matter:

> If the advantage tracks `num_blocks` whichever knob produced it, the
> mechanism is what we say it is rather than something merely correlated with
> long context.

**The knob matters.** At the same `num_blocks`, reached by shrinking the block
rather than lengthening the sequence, the kernel is 29.5x slower rather than
faster. The hypothesis as written is false, and the reason it is false is a
sharper statement of the design than the hypothesis would have been.

Nothing here contradicts the A100 headline. `ours` still beats every sparse
competitor in every row it can be compared in. What it contradicts is the idea
that `num_blocks` alone predicts the result.

## The two panels are a controlled experiment

Both sweep `block_size` over exactly the same values. They differ in one
variable.

* **budget panel** holds `top_k * block_size = 1024`, so every row attends the
  same 1024 keys and `top_k` rises 8 -> 64 as the block shrinks.
* **fixed-top-k panel** holds `top_k = 8`, so the block shrinking means each
  row attends *fewer* keys, 1024 down to 128.

`ours`, S=16384:

| block_size | top_k | keys attended | latency | vs block 128 |
|---:|---:|---:|---:|---:|
| **budget panel** | | | | |
| 128 | 8 | 1024 | 0.721 ms | 1.00x |
| 64 | 16 | 1024 | 1.343 ms | 1.86x |
| 32 | 32 | 1024 | 4.277 ms | 5.93x |
| 16 | 64 | 1024 | **21.291 ms** | **29.52x** |
| **fixed-top-k panel** | | | | |
| 128 | 8 | 1024 | 0.707 ms | 1.00x |
| 64 | 8 | 512 | 0.654 ms | 0.92x |
| 32 | 8 | 256 | 0.813 ms | 1.15x |
| 16 | 8 | 128 | **1.447 ms** | **2.05x** |

The block size is swept identically in both. The penalty is 29.52x in one and
2.05x in the other.

**The cost is `top_k`'s, not `block_size`'s.** Dividing them attributes 14.4x
to `top_k` rising eightfold and 2.05x to the block shrinking eightfold at
constant `top_k`.

## Why `top_k` is what costs

This kernel is KV-block-major: one program owns one key block and loops over
the queries that selected it. Programs handling different blocks run
simultaneously and cannot see each other, so a query's softmax cannot be
completed in any one of them. Each program instead writes a partial result --
`(m, l, acc)` -- and `moba_combine_kernel` merges them in a second pass
(`src/moba/kernels/moba_fwd.py`).

That means the scratch is counted per **(query, block) pair**, not per key:

| block_size | top_k | keys attended | `part_acc` etc. | FlashMoBA running output |
|---:|---:|---:|---:|---:|
| 128 | 8 | 1024 | 34.0 MB | 8.0 MB |
| 64 | 16 | 1024 | 68.0 MB | 8.0 MB |
| 32 | 32 | 1024 | 136.0 MB | 8.0 MB |
| 16 | 64 | 1024 | **272.0 MB** | **8.0 MB** |

(`bench/block_sweep.py`, `partials_mb` and `running_output_mb`, S=8192.)

Identical arithmetic in every row; eight times the bookkeeping in the last.
The right-hand column is flat because FlashMoBA's scratch is O(S) -- it keeps
its running softmax in shared memory and finishes in one kernel, so it holds
one running output per query regardless of how many blocks that query picked.
Ours is O(S * top_k). That single difference is the whole table.

14.4x rather than 8x because the merge is a reduction over `top_k` partials as
well as a write of them, so the second pass grows with the same factor that
grew the first pass's output.

## The same fact, read on its other axis

| | raise `seq_len` 8x | shrink `block_size` 8x (budget panel) |
|---|---|---|
| `num_blocks` | x8 | x8 -- **same** |
| (query, block) pairs | x8 | x8 -- **same** |
| arithmetic | **x8** | **unchanged** |
| pairs per unit of work | **unchanged** | **x8** |

Long context adds pairs and work together, so the per-pair overhead stays a
fixed fraction and the design's advantage survives. Shrinking the block adds
pairs alone, so the overhead grows without bound.

The 4.55x win at S=65536 and the 29.5x loss at `block_size=16` are the same
property of the design measured on its two axes, not a success and a defect.

## Where this leaves the competitors

`ours` against everything else, budget panel, S=16384:

| block_size | ours | flash_moba | triton_moba | sdpa_dense |
|---:|---:|---:|---:|---:|
| 128 | **0.721** | 2.138 | 4.951 | 1.200 |
| 64 | **1.343** | 3.905 | 12.638 | 1.207 |
| 32 | **4.277** | not runnable | 42.664 | 1.206 |
| 16 | **21.291** | not runnable | 92.737 | 1.202 |

Three things worth separating.

**`ours` wins every comparable row**, including the ones where it is slow in
absolute terms: 4.4x over `triton_moba` even at `block_size=16`. The advantage
narrows from 6.9x to 4.4x, so small blocks hurt this design more than they hurt
a query-major one, but not enough to reverse the order.

**FlashMoBA cannot run below 64** -- its block dimension must be a multiple of
64 (`flash_moba_api.cpp:465`), reported as an availability reason rather than
dropped. Those blanks are evidence about where the contribution lies.

**Dense attention wins outright below `block_size=64`.** `sdpa_dense` is flat
at ~1.2 ms across the whole panel, because nothing about it depends on the
gate; at `block_size=16` it is 17.7x faster than `ours`. Any claim about small
blocks has to be read against that row: **at `block_size=16` with a 1024-key
budget, sparse attention of any kind is the wrong tool on this hardware.**

## What this does not say

* **Not a claim about trained models.** The selection comes from this project's
  gate on `randn` inputs, the same distribution every other result here uses.
* **Not a claim about `block_size=16` mattering.** `CLAUDE.md` retired that
  framing on 2026-08-19: neither vendored reference uses a block below 128,
  and both get sparsity from context length. This sweep is a probe of the
  mechanism, not of a configuration anyone ships.
* **Not a limit anyone has hit in practice.** FlashMoBA benchmarks
  `block_size=128, top_k=8`; MoonshotAI defaults to chunk 512/4096.

## The compile ceiling, which was a separate bug

While reading this sweep, `N_BLOCKS >= 2048` was found to fail outright. It is
unrelated to anything above and is recorded here because the two look alike
from the outside.

`offsets_and_order_kernel` ranked columns with a selection sort whose outer
loop was `tl.static_range(N_BLOCKS)`, which unrolls. Emitted code grew as N^2:
2.01 MB of PTX at 512 columns, 5.02 MB at 1024, and at 2048 the compile never
finished -- 11 minutes at 1.3 GB RSS inside the MLIR passes, GPU at 0%, no
`ptxas` process. Fixed 2026-09-19 in 838ca69 by packing the column id into the
low bits of a sort key and calling `tl.sort`, whose bitonic network is
O(log^2 N) stages: 0.31 MB at 2048, and 37x faster at 1024 besides.

**It never touched this sweep.** The largest configuration here is
16384 / 16 = 1024 columns, one step below the ceiling, and the sweep's own
floor of `block_size = 16` means it cannot reach 2048 without a longer
sequence. The crash was in the separate S=65536 run
(`results/sm80-a100/logs/65k.log`), where `block_size=32` gives 2048 columns.

The fix does change these numbers slightly -- the retired sort was 2-6% of each
row, most at `block_size=128` where the row is shortest and least at
`block_size=16` where it is longest. The table above is pre-fix and therefore
pessimistic by that much. It is not re-measured because the effect is smaller
than the cross-pass reproducibility band of 14% that `docs/a100-results.md`
establishes for anything that routes.

## Reproducibility

A confirmation run on an RTX 3050 Ti Laptop (20 SMs against the A100's 108, at
S=8192) reproduces the shape rather than the magnitudes:

| block_size | A100 budget | 3050 Ti budget | A100 fixed | 3050 Ti fixed |
|---:|---:|---:|---:|---:|
| 128 | 1.00x | 1.00x | 1.00x | 1.00x |
| 64 | 1.86x | 10.10x | 0.92x | 0.86x |
| 32 | 5.93x | 6.57x | 1.15x | 0.92x |
| 16 | **29.52x** | **30.97x** | **2.05x** | **1.20x** |

The headline degradation lands within 5% on a card with a fifth of the SMs,
which is what a structural cost rather than an occupancy artefact looks like.

The 3050 Ti's `block_size=64` row is not monotonic -- slower than
`block_size=32` -- and the A100's is. That is a laptop artefact, most likely a
missing entry in `src/moba/kernels/tuned_configs.py` for sm_86 at that shape,
and it is recorded rather than smoothed. It should not be read as a feature of
the curve.
