# `tl.topk` instead of the merge loop, measured

**Date:** 2026-09-21
**Device:** RTX 3050 Ti Laptop, sm_86 | bf16 | B=1, H=4, head_dim 64,
block_size 128, top_k 8, `GATE_BLOCK_M=32`, `GATE_BLOCK_N=32`
**Source:** a throwaway three-mode ablation of `gate_kernel`, not kept
**Settles:** whether replacing the hand-rolled `TAKE`-round merge with
Triton's `tl.topk` is worth doing. **It is not — 1.04x to 1.10x.**
**Companion:** `docs/gate-streaming-measurements.md` for the streaming form
this measures, and `docs/h100-results.md` for why the gate matters now.

---

## The question

`docs/h100-results.md` found the gate is 87% of our forward pass at S=524288,
so the gate is the remaining lever. Inside `gate_kernel` the obvious suspect is
the merge: `TAKE` unrolled rounds per chunk, each offering the chunk's best
remaining candidate to the running list, and each round costing about five
cross-lane reductions. Triton 3.7.1 ships `tl.topk` -- bitonic top-k, not a
full sort -- so the question is whether a compiler primitive beats the
hand-rolled loop.

## First: how much is the merge even worth?

Before writing a replacement, bound what any replacement could win. Three
modes of the same kernel, identical in every other respect:

* **no-merge** -- dot, causal mask, one running `tl.max` per chunk. The merge
  is gone; one reduction remains so the dot and the loads are not dead code.
* **loop** -- the shipping kernel, verbatim.
* **topk** -- the packed-key `tl.topk` form described below.

| S | blocks | no-merge | loop | topk | merge share | topk speedup | same blocks |
|---:|---:|---:|---:|---:|---:|---:|:--:|
| 8,192 | 64 | 0.043 ms | 0.133 | 0.128 | 67.7% | 1.04x | yes |
| 32,768 | 256 | 0.233 | 1.473 | 1.379 | 84.1% | 1.07x | yes |
| 131,072 | 1,024 | 3.385 | 21.626 | 19.573 | 84.3% | 1.10x | yes |

**The merge is 84% of the gate.** Combined with the gate being 87% of the
forward at long context, this one loop is roughly 73% of our entire forward
pass -- by far the largest single line item left in the project.

**And a better top-k algorithm barely touches it: 1.10x.**

Those two facts together are the result. The merge dominates, and it does *not*
dominate because the top-k algorithm is bad.

## Why the swap wins so little

Both forms do about the same number of cross-lane operations per chunk, and
cross-lane operations are what the merge costs.

| | per chunk, at `TAKE=7`, `K_POW2=8`, `BLOCK_N=32` |
|---|---|
| loop | 7 rounds x ~5 reductions = ~35 reductions |
| topk | `topk(32 -> 8)` is ~15 compare-exchange stages, `topk(16 -> 8)` ~10 |

Roughly 35 against roughly 25. A 1.1x ratio is exactly what that predicts. The
bitonic network's asymptotic advantage needs a wide chunk and a large `k`; at
32 wide and `k=8` there is no asymptotic regime to enter, which is the same
reason `gating.py`'s module docstring already gives for not borrowing NSA's
sorted-merge design.

**The lever is merging less often, not merging better.** The per-chunk cost is
near-irreducible at this shape, so what remains is reducing the number of
(chunk, merge) pairs -- which is the sub-quadratic block-selection work already
flagged as D11 / Task 6, not a change to this loop.

## The packed key, kept for the record

`tl.topk` returns values and no indices ([triton#3698][1], open since April
2024), and the kernel must know *which* block won, with ties broken toward the
lower block id (`tie_break_low_index`, a determinism requirement -- getting it
wrong disagreed with the oracle on 764 of 8.4M decisions). So score and id go
into one sortable int32:

```
order = monotone uint16 image of the bf16 score
        (negatives: ~bits; positives: bits | 0x8000)
key   = (order - 32768) * 65536 + (65535 - block_id)
```

The shipping kernel already rounds the score to bf16 before ranking, so 16 bits
loses nothing that could change a decision. Larger key means better score;
equal scores rank by lower id, so **the tie-break falls out of the ordering
instead of costing a second reduction**. Empty slots sit at `INT32_MIN`; the
`+inf` seed (`always_select_current`) and the retired surplus slots sit at the
maximum order so they survive every merge, the retired ones carrying the
`N_BLOCKS` sentinel so they still emit as `-1` padding.

The merge then becomes two primitive calls and no bookkeeping at all:

```python
keys       = pack_key(s, cols)                  # [BLOCK_M, BLOCK_N]
chunk_best = tl.topk(keys, K_POW2)              # [BLOCK_M, K_POW2]
both       = tl.reshape(tl.join(best_key, chunk_best), [BLOCK_M, 2 * K_POW2])
best_key   = tl.topk(both, K_POW2)
```

No `floor`, no `victim`, no `floor_slot`, no `enter` -- the eviction rule the
shipping kernel spends three reductions computing is implied by the key order.
It is a genuinely nicer kernel. It is also 1.1x, and it costs a 16-bit id field
(`num_blocks <= 65535`, or 8.4M tokens at block_size 128) and a representation
nobody can read in a debugger.

**Verified identical**, not merely close: `torch.equal` on the emitted block
ids against the shipping kernel at all three lengths, ties included.

## Why this is not being merged

1.10x on the gate at 131K is about 1.08x on the forward pass, for a rewrite
that trades readable `best_val`/`best_idx` for packed keys and introduces a
sequence-length ceiling the current kernel does not have. The measurement is
recorded so the idea is not re-proposed from first principles; if the chunk
width or `top_k` ever grow enough to reach the bitonic regime, the packing
above is the starting point and it is known to be correct.

[1]: https://github.com/openai/triton/issues/3698
