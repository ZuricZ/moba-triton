# Learning note 03 — GQA and the column merge

**Why grouped-query attention costs this kernel nothing and buys it a lot,
explained with a grid of X's.**

Prerequisites: `docs/learning/01-loop-order-and-memory-traffic.md` (HBM vs SRAM,
arithmetic intensity, why loop order decides memory traffic). Attention math
assumed; GPU architecture is built up as it comes.

---

## What a "column" is

The kernel's index is stored as CSC — compressed sparse column — and that name
hides a very simple picture.

Draw a grid. **Rows = queries. Columns = KV blocks.** Put an `X` where a query
selected that block. For one head, 6 queries, 3 blocks, top-k 2:

```
         B0  B1  B2
   q0     X   X   .
   q1     X   .   X
   q2     .   X   X
   q3     X   X   .
   q4     X   .   X
   q5     .   X   X
```

It is the score matrix's key axis, **coarsened to blocks** instead of individual
keys. Column `B0` is the vertical stripe, and reading it off gives a **list**:
`[q0, q1, q3, q4]`. That list is all "column" ever means here.

**One list = one program's job.** A program grabs `B0`, loads B0's keys and
values into shared memory **once**, then walks its list. That single load, spread
across every query in the list, is the entire reason for the KV-block-major loop
order (decision D6).

## MHA vs GQA

**MHA, 4 heads.** Each head has *its own* keys. Head 0's `B0` and head 1's `B0`
are different bytes in memory. So there are **four separate grids**, side by
side, over four different key sets:

```
   head 0          head 1          head 2          head 3
    B0 B1 B2        B0 B1 B2        B0 B1 B2        B0 B1 B2
q0   X  X  .    q0   .  X  X    q0   X  .  X    q0   X  X  .
q1   X  .  X    q1   X  X  .    q1   .  X  X    q1   X  .  X
q2   .  X  X    q2   X  .  X    q2   X  X  .    q2   .  X  X
```

**12 lists.** Each ~4 entries long. Twelve programs, twelve separate K/V loads.

**GQA, 4 query heads sharing 1 KV head.** There is now only **one** set of keys.
All four heads point at the same `B0`, `B1`, `B2`. So the grids no longer sit
side by side — they **stack on top of each other**, because they share the same
column headings:

```
            B0  B1  B2
  h0.q0      X   X   .
  h0.q1      X   .   X     <- head 0's rows
  h0.q2      .   X   X
  h1.q0      .   X   X
  h1.q1      X   X   .     <- head 1's rows
  h1.q2      X   .   X
  h2.q0      X   .   X
  h2.q1      .   X   X     <- head 2's rows
  h2.q2      X   X   .
  h3.q0      X   X   .
  h3.q1      X   .   X     <- head 3's rows
  h3.q2      .   X   X
```

**3 lists.** Each ~16 entries long.

## The one idea

> **The entries did not multiply. Four lists got glued into one.**

That is the whole of `n -> 4n`. Count the `X`s in both pictures — the same
number, because the same queries picked the same number of blocks. No work was
created. The **containers merged**, because the four things they described
turned out to be one thing.

The list is 4x longer for the same reason there are 4x fewer lists. One fact,
two views.

## Why that is worth anything

Line the two up:

| | MHA | GQA (group 4) |
|---|---:|---:|
| `X`s to process (the maths) | 48 | **48** — same |
| Lists -> programs -> **K/V loads** | 12 | **3** |

Twelve K/V loads became three. The other nine were loading **identical bytes** —
four copies of the same keys, because MHA stores four copies. GQA deletes the
duplicates from the model, and the loads go with them.

In the vocabulary of note 01: the arithmetic term is unchanged and the memory
term is divided by the group size, so **arithmetic intensity rises by roughly
the group size**. On this card the ridge point is 78 FLOP/byte and the attention
kernel sits at 111.5 — already compute-bound. On an H100 the ridge point is
~295, and the same unchanged kernel is memory-bound. Group-4 GQA is what moves
it back across.

## Longer road, same car

A natural worry: if the list is 4x longer, does the program need 4x the
registers?

No — and the reason is a design choice that is easy to make without noticing.
"Serve four query heads" could mean **widening the tile** to hold four heads at
once, which really would multiply register pressure by four and collapse
occupancy. But the kernel already does the other thing:

```python
for start in range(0, n, GROUP):
```

It **already** streams an arbitrary-length list through a fixed-size tile. A
longer list is more trips around the loop. Per iteration nothing changes: still
a `[GROUP, D]` query tile, a `[GROUP, BLOCK_SIZE]` score tile, a `[GROUP, D]`
accumulator. Same registers, same shared memory, same programs per SM.

**Parallel-in-registers costs; serial-in-loop is free.** The kernel was already
written the free way, for unrelated reasons.

Shared memory is likewise untouched: K and V for a block are the same tensors no
matter how many query heads consume them. Loaded once either way.

## Why this is a KV-major dividend specifically

Query-major would collect much less of this. There, a program owns a *tile of
queries* and loads the **union** of blocks that tile selected. Add three more
heads, each with its own independently-chosen top-k, and the union **grows** —
you load more K/V to serve more queries, so intensity barely improves.

The sharing only materializes when **the K/V block is the thing you own and
queries are the thing you stream**. That is exactly the inversion D6 chose, for
reasons that had nothing to do with GQA. Not every design decision pays a
dividend it was not aiming at; this one did.

## What it costs in the code

Almost nothing, and the diff shows why the framing above is the right one:

* A list is keyed by `(batch, kv_head, block)` instead of `(batch, head, block)`
  — because that key identifies a *physical* block of keys.
* That key **no longer determines the query head**. Looking at `h1.q2` in the
  merged list, "q2" alone does not say whose query it was. So every entry
  carries its query head alongside its position, in a `head_indices` array
  parallel to `row_indices`.
* `_moba_combine` and `_moba_dq_reduce` change not at all. Both index everything
  by `(query, slot)`, so their head count simply *becomes* `H_q`.

The rule the whole change reduces to: **anything indexed by a query keeps
`H_q`; anything indexed by a key switches to `H_kv`.** Only the kernels that
touch both sides at once need real edits.

## The honest ceiling

GQA divides K/V traffic. On this card, at S=8192, the attention kernel moves
104 MB — of which `part_acc` is 63.4 MB, gathered queries are 31.7 MB, and *all*
K/V loads together are **8.4 MB**.

GQA divides the 8.4 MB. It does not touch the 63.4 MB.

So the end-to-end gain here will be small, possibly within noise. That is not an
argument against it: the 4x intensity gain is real, and it is precisely what a
higher-ridge-point machine at long context needs. But the payoff lives there,
not here, and saying so **before** measuring is what lets the measurement
falsify the claim rather than decorate it.

---

**See also:** `docs/superpowers/plans/2026-08-23-gqa-support.md` for the
implementation, including the rejected alternatives (packed head+position, group-
shared top-k, wider tiles).
