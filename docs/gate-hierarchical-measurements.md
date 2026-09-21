# Coarse-to-fine block selection: the idea works, the summary doesn't

**Date:** 2026-09-21
**Device:** RTX 3050 Ti Laptop, sm_86 | float32, PyTorch only, no Triton
**Shape:** S=524288, H=2, head_dim 64, block_size 128, top_k 8 → 4096 blocks,
7 past blocks selected per query, 1024 sampled queries
**Source:** a throwaway probe against `gate_select_naive` / `pool_blocks_naive`
**Settles:** whether hierarchical selection can cut the gate's merge count
without changing which blocks get picked.
**Answer:** the hierarchy is sound — a perfect ranker reaches **100%** — but
**mean-pooled super-block representatives lose 5 to 25 points of recall**, and
the loss grows with group size. Fix the summary, not the idea.
**Companion:** `docs/gate-topk-measurements.md` for why merge *count* is the
lever at all.

---

## Why this was measured before any kernel was written

`docs/gate-topk-measurements.md` established that the merge loop is 84% of the
gate and that making each merge faster is worth 1.10x at most. What remains is
merging *fewer times*: group the 4096 blocks into super-blocks of `G`, score
the `4096/G` super-block representatives, open the best `P`, and score only the
`P*G` blocks inside.

That is an approximation. A strong block inside a weak super-block is never
scored at all. This probe measures how often that happens, in PyTorch, for
free, before committing to a kernel.

## Two kinds of data, because the data decides this, not the algorithm

Mean-pooling `G` iid Gaussian vectors gives a vector of norm ~`1/sqrt(G)`
pointing nowhere in particular. Random keys are therefore the **worst case** for
any hierarchy, not a neutral one. Real keys have positional locality, which is
exactly what a super-block summarises, so a random walk along the sequence is
the more representative case. Both are reported; the truth is between them.

## The frontier, on locality-bearing keys

`saving` is the drop in merge count against today's flat gate at `BLOCK_N=32`.
`ceiling` is the recall a **perfect** super-block ranker would reach — one that
ranks super-blocks by their best member's true score, which is unimplementable
(knowing it means having scored every block). The gap between `recall` and
`ceiling` is what a better summary function could win back.

| G | P | merges | saving | recall | ceiling | exact set | top-1 hit |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 8 | 33 | 3.9x | 95.91% | 100.00% | 82.71% | 100.00% |
| 4 | 32 | 36 | 3.6x | 99.32% | 100.00% | 96.83% | 100.00% |
| 8 | 4 | 17 | 7.5x | 90.57% | 99.66% | 68.16% | 100.00% |
| 8 | 32 | 24 | 5.3x | 98.51% | 100.00% | 93.36% | 100.00% |
| 16 | 4 | 10 | **12.8x** | 83.54% | 99.99% | 66.89% | 98.83% |
| 16 | 8 | 12 | **10.7x** | 91.20% | 100.00% | 78.32% | 99.95% |
| 16 | 16 | 16 | 8.0x | 94.25% | 100.00% | 83.94% | 99.95% |
| 32 | 4 | 8 | 16.0x | 76.35% | 100.00% | 59.38% | 94.34% |
| 32 | 8 | 12 | 10.7x | 79.02% | 100.00% | 63.48% | 94.87% |
| 64 | 4 | 10 | 12.8x | 74.86% | 100.00% | 56.40% | 93.36% |
| 64 | 32 | 66 | 1.9x | 94.36% | 100.00% | 85.64% | 99.56% |

On iid random keys the same table reads 6% to 49% recall against the same
100% ceilings — catastrophic, and worth remembering as the floor.

## The result, in one line

**The exact top-7 blocks almost always live inside the best 4 super-blocks
(ceiling 100%). We just cannot tell which 4 those are from mean-pooled
representatives.**

## Why mean-pooling specifically fails, and it is not subtle

The gate scores a super-block by `q · mean(kbar_i)`. Dot products are linear,
so that is **exactly** `mean(q · kbar_i)` — the super-block's *average* score.
But the thing we need to rank by is its *best* score, because we are about to
open it and take the best block out of it.

A super-block holding one excellent block among 31 mediocre ones ranks as
mediocre. That is precisely the case the gate exists to find.

The measurement confirms the mechanism rather than just asserting it: the
recall-to-ceiling gap widens monotonically with `G`, because averaging more
blocks makes the mean a worse proxy for the max.

| G | recall at P=8 | ceiling | gap |
|---:|---:|---:|---:|
| 4 | 95.91% | 100% | 4.1 |
| 8 | 93.56% | 100% | 6.4 |
| 16 | 91.20% | 100% | 8.8 |
| 32 | 79.02% | 100% | 21.0 |
| 64 | 79.22% | 100% | 20.8 |

## What this means for the next step

**Not abandoned, and not ready to build either.** The structural question came
back positive, so the work is now a well-posed and much narrower one: find a
super-block summary that predicts the *maximum* member score rather than the
mean. Candidates, none measured yet:

* **Several representatives per super-block**, scored and maxed. The `G=4` and
  `G=8` rows are already evidence this direction works — they are the same idea
  expressed as a finer partition.
* **An admissible upper bound**, e.g. `|q| * max_i |kbar_i|` per super-block.
  A bound that is never an underestimate permits *exact* pruning with no
  accuracy loss at all; the open question is how loose it is in high dimension,
  which decides how much it prunes.
* **Learned or asymmetric summaries** — out of scope for a kernel project.

The best operating point available today is roughly **10x fewer merges at ~91%
recall and ~78% set agreement** (`G=16, P=8`, locality-bearing keys). Whether
that is acceptable is a model-quality question this repository cannot answer:
nothing here trains or evaluates a model, and recall of selected blocks is a
proxy, not an end metric.

## Caveats

* Synthetic keys. The "clustered" generator is a random walk plus noise, chosen
  because it produces positional locality; it is not text and not a trained
  model's activations. The real figure is somewhere between the two columns
  reported and this probe cannot narrow it further.
* 1024 sampled query positions per head, uniform over `[4*block_size, S)`.
* Recall counts *which blocks were selected*, not attention output error.
* `top-1 hit` stays above 93% almost everywhere, so the single most important
  block usually survives even where recall is poor. That is the most
  encouraging column and also the easiest to over-read.
