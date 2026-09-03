# Learning note 05 — The backward pass

**Why the loop order that made the forward awkward makes the backward easy, and
what has to be carried across between them.**

Prerequisites: `docs/learning/01-loop-order-and-memory-traffic.md` (HBM vs SRAM,
why loop order decides memory traffic) and
`docs/learning/03-gqa-and-the-column-merge.md` (the grid, the lists, "one program
per column"). Attention math assumed; nothing about GPUs assumed beyond note 01.

---

## What it has to produce

The forward took `Q`, `K`, `V` and made `out`. The backward is handed `dOut` --
"how wrong was each output" -- and must produce three things: **`dQ`, `dK`,
`dV`**.

## The problem: we threw away the thing we need

To compute any of them you need `P`, the softmax weights -- "how much did query
*i* attend to key *j*".

The forward computed `P`. Then deliberately threw it away. At S=8192 the full
`P` is 8192 x 8192 x 4 bytes = **256 MB per head**. Not storing it is the entire
point of FlashAttention.

> **So the backward recomputes `P` from scratch. Arithmetic instead of memory --
> the same trade the forward made, paid a second time.**

## The transpose, which explains everything else

Write the forward as one matmul:

```
out  =  P @ V                     P is [queries x keys]

out_i  =  SUM_j  P_ij * V_j       indexed by query, sums over KEYS
```

Differentiate with respect to `V`:

```
dV   =  P^T @ dOut

dV_j  =  SUM_i  P_ij * dOut_i     indexed by key, sums over QUERIES
```

`P` became `P^T`. **Transposing a matmul swaps which axis you sum over.** All
four quantities, side by side:

| | formula | indexed by | sums over |
|---|---|---|---|
| `out` | `P @ V` | query | keys |
| `dV` | **`P^T`** `@ dOut` | **key** | **queries** |
| `dK` | **`dS^T`** `@ Q` | **key** | **queries** |
| `dQ` | `dS @ K` | query | keys |

Two of the three gradients use the transpose. That is not a design choice -- it
is what the chain rule says. What *is* a design choice is that our loop order
happens to be the one that likes it.

## The one idea

Take note 03's grid. Six queries, three blocks, one program per column:

```
         B0  B1  B2
   q0     X   X   .
   q1     X   .   X
   q2     .   X   X
   q3     X   X   .
   q4     X   .   X
   q5     .   X   X
```

The B0 program's list is `[q0, q1, q3, q4]`. **A program is a block of keys and
a stream of queries.**

**Forward -- can it finish `q0`'s answer?** No. Row `q0` has an X under B0 *and*
under B1. Half of the answer is computed here, half somewhere else, and somebody
must add them up later. That somebody is `combine_partials`, and the place the
pieces wait is `part_acc` -- 64 MB at S=8192, the design's largest memory term.

**Backward -- can it finish `dV` for B0's keys?** Yes. `dV[B0]` sums over
*everyone who attended B0*. Read column B0: `q0, q1, q3, q4`. **That is exactly
this program's list.** Nobody else contributes.

```
dV[B0]  =  q0's term + q1's term + q3's term + q4's term
           |______________ all four are in this program ______________|
```

It accumulates into a register as it walks its list and stores once. No buffer,
no second pass, no atomics.

> **The program *is* the block. So anything counted per block is already home;
> anything counted per query is scattered.**

Said from both sides, which is the same fact twice:

* A query's answer is **split across** the blocks it chose. (The forward's
  problem.)
* A block's gradient is **gathered from** the queries that chose it. (The
  backward's gift.)

## What that buys, counted

| | counted per... | forward | backward |
|---|---|---|---|
| `K`, `V` | **block** | input, loaded once | input, loaded once |
| `out` | query | output, **scattered** | -- |
| `dK`, `dV` | **block** | -- | output, **finished in-program** |
| `dQ` | query | -- | output, **scattered** |

Forward: one scattered output, zero free ones.
Backward: one scattered output, **two free ones**.

**The bill moved from two tensors to one.**

## And the tensor that stayed scattered got cheaper

| | forward's `combine` | backward's `reduce_dq` |
|---|---|---|
| what is merged | each block's partial output | each block's `dQ` contribution |
| computed against | that block's **own local** maximum | the query's **global** `lse` |
| so the pieces are | on **different scales** | on the **same scale** |
| the merge is | rescale by `exp2(m_local - m_global)`, two passes | **a plain sum** |

The forward cannot know a query's global maximum until every block has reported,
so it exponentiates against a local one and corrects afterwards. The backward
already has the global number, because the forward saved it. Nothing to correct.

**The bill moved from two tensors to one, and got cheaper on the way.**

## Recompute what is local, save what is global

`lse` is the log of a query's softmax denominator -- one float per query. Why is
it *saved* rather than recomputed, when `P` is recomputed rather than saved?

Sort by orientation again:

| | indexed by | sums over | can one program finish it? |
|---|---|---|---|
| `out` | query | keys | no |
| **`lse`** | **query** | **keys** | **no** |
| `dQ` | query | keys | no |
| `dV`, `dK` | key | queries | yes |

`lse` sits on the same axis as `out`, and is scattered for the same reason:
`lse_i = log SUM_j exp(qk_ij)` sums over every key query *i* attended, which
means every block it selected. A program holding one block holds one term.

That is true in the forward and the backward alike -- both are block-major. So
neither attention kernel can finish `lse`. It is computed in
`combine_partials`, the one query-major pass, where all of a query's blocks
finally meet, and it falls out of a reduction that pass was already doing:

```python
if WRITE_LSE:
    tl.store(Lse + pid_bh * S + offs_m, m_g + tl.log2(l), mask=live)
```

The costs, side by side:

| | store it | recompute it |
|---|---|---|
| `P` | 256 MB per head | one `tl.dot` -- `q` and `k` are in registers |
| `lse` | 256 KB per head | re-run the whole forward attention *and* combine |

> **The question is never "how expensive is this quantity". It is "can this
> program see everything the quantity depends on?" If yes, recompute. If no,
> save.**

And what a program can see is decided by orientation: per-key things are
visible, because the program *is* a block of keys. Per-query things are not.

## The one piece of algebra worth showing

The softmax's derivative gives:

```
dS = P * ( dP - rowsum(P * dP) )
                |_____________|
                a sum over ALL keys
```

That rowsum is over every key the query attended -- the scattered axis again,
which no single program can compute.

But it is **algebraically equal** to:

```
delta_i = rowsum( dO_i * O_i )      a sum over head_dim
```

`head_dim` is 64. Every program has all 64 numbers in front of it.

> **A sum over 8192 things nobody can see became a sum over 64 things everybody
> has.**

That substitution is the only reason `_moba_bwd_preprocess` exists: one tiny
pass, one float per query, run before anything else. FlashAttention-2 makes the
identical move for the identical reason.

## Why the backward is tuned differently

Count matmuls per inner-loop iteration:

```
forward     Q K^T,  P V                                         2
backward    Q K^T (recompute),  P^T dO -> dV,  dO V^T -> dP,
            dS^T Q -> dK,  dS K -> dQ                           5
```

More arithmetic on the same amount of data. So the backward wants **more warps
on a narrower tile**: `BWD_GROUP = 32` over 8 warps, against the forward's 64
over 4. The backward first shipped with the forward's constants and that cost a
factor of two, measured. Nothing about one kernel's tuning transfers to the
other.

## Why a program owns only part of a block

The forward holds `K` and `V` in shared memory -- 64 KiB at `block_size=128,
head_dim=128`.

The backward needs those **and their transposes**, because `dP = dO V^T` and
`dQ = dS K`, and Triton stages each as its own tile. Measured: **96 KiB before a
single query is loaded**, against this card's 99 KiB budget. Room for four query
rows where `tl.dot` needs sixteen. Nothing compiles.

So a program takes a 64-key **sub-tile** rather than a whole 128-key block. Note
that nothing persists between the two passes: the forward's shared memory is
gone when it ends, and the backward is a separate launch that loads `K` and `V`
from HBM again. The 96 KiB is the backward's own footprint.

### The invariant that split broke

The forward guarantees each `(query, slot)` partial has exactly **one** writer.
Splitting a block means both sub-tiles produce a `dQ` contribution for the *same*
`(query, slot)` -- and they **overwrote each other instead of summing**.

`dK` and `dV` were untouched: sub-tiling splits the **key** axis, and they are
key-indexed, so the two programs still write disjoint keys.

**It was precisely the already-scattered tensor that suffered.** The same
asymmetry as everywhere else in this note, in yet another guise. Fixed by
widening the `dQ` slot axis so each sub-tile gets its own address; the measured
error before the fix was 6073% of the tolerance budget, which is the good kind of
bug -- too wrong to miss.

## Why there are no atomics

`dV[B0]` is one sum produced by one program, so it is an ordinary register
accumulation and an ordinary store. Atomics are for when *several* programs add
into the same address.

The guarantee is specifically that **key ranges are disjoint across programs** --
including under sub-tiling, where program A writes keys 0..63 and program B
writes 64..127 of the same block.

`dQ` avoids atomics differently, and it is worth naming the trade. The natural
CSC sparse-matrix-vector product needs `atomicAdd` on its output, because
different columns hit the same row. Instead every `(query, block)` pair gets its
**own slot** in advance, so the scatter has no contention at all and the
reduction is deferred to a second pass. **We traded atomics for memory** --
`part_acc` and `dq_part` are that trade made visible.

## What does not change

| | |
|---|---|
| the loop order | **same** -- block-major, identical to the forward |
| the CSC index | **same** -- reused, never rebuilt |
| `K`/`V` load amortization | **same** -- one trip per block |
| the gate | **not differentiated at all** |
| determinism | bitwise given an index -- which is why `dQ` uses partials, not `atomic_add` |

The gate deserves a line of its own. MoBA's routing is a hard top-k, and
Moonshot's reference treats it as non-differentiable: `gate_top_k_idx` is used
only as an index and no gate score is saved for backward. So gradient flows
through the *selected* attention and not through the *decision to select*. That
is what makes the backward tractable at all -- it sees a fixed sparsity pattern
rather than a differentiable routing function.

## See also

* `docs/backward-measurements.md` -- the tuning, including a case where
  optimizing the stage that is 91% of the time, in isolation, chose wrongly.
* `docs/learning/02-anatomy-of-a-flash-attention-kernel.md` -- the forward's
  online softmax, which `lse` and the rescaled merge come from.
* `src/moba/kernels/moba_bwd.py` -- the module docstring states the same
  ownership argument in one paragraph.
