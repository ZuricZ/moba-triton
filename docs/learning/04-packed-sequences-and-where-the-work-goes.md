# Learning note 04 — Packed sequences, and where the work goes

**What variable-length batching actually changes, and why the smallest piece of
it runs on the CPU.**

Prerequisites: `docs/learning/01-loop-order-and-memory-traffic.md` (HBM vs SRAM,
why loop order decides memory traffic). Nothing else assumed.

---

## The problem: a batch has to be a rectangle

Today every sequence in a batch must be the same length. Real data is not. Three
sequences of length 5, 3 and 4 have to be padded into a rectangle:

```
seq 0:  a  b  c  d  e
seq 1:  f  g  h  .  .     <- 2 fake tokens
seq 2:  i  j  k  l  .     <- 1 fake token
```

15 slots for 12 real tokens. Attention is computed on the fakes and thrown away.

**Packing** stores them end to end instead:

```
       a  b  c  d  e  f  g  h  i  j  k  l
       0  1  2  3  4  5  6  7  8  9 10 11
       |___seq 0___|_seq 1_|___seq 2____|
```

12 slots, 12 tokens, and one small list saying where the cuts are:

```
cut list = [0, 5, 8, 12]
```

That list is the only new input. Everything else in this note exists to serve
it. Its conventional name is `cu_seqlens` -- cumulative sequence lengths -- and
it is the same convention FlashAttention uses, so we copy the name.

## What breaks, and it does not crash

MoBA chops keys into blocks. Today: every `block_size` tokens. On a rectangle
that is fine. On the packed row it is wrong:

```
       a  b | c  d | e  f | g  h | i  j | k  l
                     ^^^^
                     e is sequence 0, f is sequence 1 -- same block
```

A query in sequence 1 that selects that block reads sequence 0's keys. A
different document. The shapes stay right, the numbers stay finite, nothing
raises. The model just quietly attends across a boundary.

So blocks are cut **per sequence**, and each sequence's last block may be short:

```
seq 0:  [a b] [c d] [e]      3 blocks
seq 1:  [f g] [h]            2 blocks
seq 2:  [i j] [k l]          2 blocks
                             = 7 blocks
```

## The one idea

> **Instead of computing where a block starts, the kernel looks it up.**

```
block:         0  1  2  3  4  5  6
starts at:     0  2  4  5  7  8 10
ends at:       2  4  5  7  8 10 12
```

`blk * BLOCK_SIZE` becomes `BlockStart[blk]`. `< S` becomes `< BlockEnd[blk]`.
Two lookups replacing one multiply, in two kernels. The matmul, the softmax, the
masks and every store are untouched.

The short last block is not a new case either: the kernel already handles a
short tail for the final block of a padded sequence. Packing makes that arise
once per *sequence* rather than once per *batch*.

## Why the small table is built on the CPU

Building the table needs three numbers, all from `[5, 3, 4]`:

* total tokens = **12**
* total blocks = **7**
* longest sequence = **5**

You could do it on paper. The question is where.

**The CPU never waits for the GPU.** It fires off work and moves on:

```
CPU:  send-send-send-send-send->   (keeps going)
GPU:      [work][work][work][work]
```

But to *read a number that lives in GPU memory*, the CPU must stop and wait for
everything already queued to finish:

```
CPU:  send-send-[ waiting .................... ]->
GPU:      [work][work][work][work][done]
                 ^^^^^^^^^^^^^^^^^^^^^^ CPU idle
```

Measured on this box: **~230 us** per stop, against ~6 us for the same
arithmetic kept on device. That number is recorded in `index_fused.py`, whose
entire reason for existing is removing stops of this kind.

Three numbers means three stops: **~700 us**, against a forward pass of
~2280 us.

> **We would have spent 30% of the forward pass waiting to read three numbers
> that fit on a napkin.**

And the list starts on the CPU anyway -- whatever loaded the data knew the
lengths before it packed them. Sending it to the GPU only to read three integers
back is a round trip for information already in hand.

So the arithmetic happens on the CPU, where the list already is, and the
finished 7-entry table is transferred **once**.

### The rule this is an instance of

**Counting sequences → CPU. Touching tokens → GPU.**

Here that is 3 sequences against 12 tokens; realistically 32 against 8192.
Anything that scales with 32 is not worth a kernel launch, and worse, putting it
on the device means fetching scalars back off it. The fetch is the expensive
part, never the arithmetic.

The same rule decides a second question. A **query tile** covers 32 queries, and
a tile spanning two sequences would give its lanes different candidate ranges --
but a Triton program has one loop bound for all its lanes. So tiles must never
straddle a boundary.

> **Superseded, 2026-08-24.** This note originally described a CPU-built
> tile-to-sequence map, borrowed from NSA's `token_indices`. It is no longer
> needed. If the caller passes `max_seqlen` -- which FlashAttention already
> requires, for exactly the CPU-waits-for-GPU reason above -- the grid can be
> shaped `(cdiv(max_seqlen, BLOCK_M), batch, heads)` and every program learns
> its sequence from its own program id. Tiles cannot straddle because the grid
> is per-sequence by construction. Sequences shorter than the longest get a few
> programs that return immediately, which on a card measured as grid-starved
> (`docs/gqa-measurements.md`) is not the thing to economise on.
>
> The rule survives its own example: the raggedness is still resolved outside
> the kernel, just by the grid's shape rather than by an array. The plan is
> being rewritten accordingly, and this note will be revised with it.

## The invariant that keeps the causal mask untouched

This line does not change:

```python
qk += tl.where(key_pos[None, :] <= rows[:, None], 0.0, float("-inf"))
```

`key_pos` and `rows` are now packed token indices. Within a sequence, packed
order *is* position order, so the comparison is still exactly right.

Across sequences it would not be -- sequence 1's query at token 5 versus
sequence 0's key at token 4 gives `4 <= 5`, which passes. It never happens
**only because the gate can never offer a query a block from another sequence**.

That is worth stating plainly: the correctness of a line in the attention kernel
is guaranteed by a different file, several stages upstream. It is the one place
in this design where that is true, and it is why there is a test asserting the
property directly rather than hoping a downstream comparison notices.

## Before and after

| | padded | packed | changed? |
|---|---|---|---|
| the matmul, the softmax, the masks | | | **same** |
| tile sizes, registers, shared memory | | | **same** |
| every tuning constant | | | **same** |
| `combine`, `dq_reduce`, `preprocess` kernels | | | **same** |
| tensor shapes | `[B, H, S, D]` | `[1, H, T, D]` | **same family** |
| address arithmetic inside the kernels | | | **same** |
| how a block finds its keys | `blk * BLOCK_SIZE`, `< S` | `BlockStart[blk]`, `< BlockEnd[blk]` | two lookups |
| which blocks the gate may offer | all past blocks | this sequence's past blocks | one offset |
| how query tiles are assigned | `pid * BLOCK_M` | a per-sequence grid dimension | grid shape |

Shapes keep four dimensions with the batch pinned to 1, so `[1, H, T, D]` holds
every token of every sequence. That is deliberate: the kernels address queries
as `pid_bh * S * D + row * D`, and with batch 1 that already reads
`head * T * D + token * D` -- correct, character for character.

FlashAttention's `[T, H, D]` is arguably the better *public* interface, and
switching would cost less than it first appears: about a dozen address
expressions, all of the same shape. We keep head-major because 379 tests and
every benchmark number speak it, not because switching is hard. That is a
provisional choice with a condition attached -- see `docs/deferred-decisions.md`,
DD-1.

## Where each piece lives in the plan

`docs/superpowers/plans/2026-08-24-varlen-support.md`, seven tasks:

| Task | What it does | Which idea above |
|---|---|---|
| **1** | `SeqLayout` — builds the block table and the tile map on the CPU. Plus the oracle. | "the small table", "why on the CPU" |
| **2** | Block representatives (mean-pooled keys) pooled per sequence | short last block, per sequence |
| **3** | The gate: a query may only pick its own sequence's blocks | the invariant that keeps the causal mask untouched |
| **4** | The index (block → which queries). Almost nothing to do. | blocks are numbered globally, so the transpose already works |
| **5** | Forward attention: two lookups | "the one idea" |
| **6** | Backward: the same two lookups | "the one idea" |
| **7** | Benchmark against a packed dense baseline, and write down what it bought | — |

Task 1 is pure PyTorch with no kernels. Tasks 5 and 6 are two lines each. Task 3
is the one that carries the correctness risk.

## What it should buy, stated before measuring

Padding waste is `1 - mean(length) / max(length)`. On a uniform batch that is
zero and packing buys **nothing**; on a ragged one it is whatever the raggedness
is. So the gain is entirely a property of the data, not of the kernel, and the
honest way to report it is against a dense baseline that is *also* packed --
`torch.nested` gives one for free, verified to dispatch to the fused
FlashAttention varlen kernel.

One secondary effect is worth watching. A ragged batch produces *more* blocks
than a padded one produces useful blocks, so the launch grid grows. On this
20-SM card, `docs/gqa-measurements.md` measured grid starvation costing up to
2x -- a bigger grid may be worth more here than the bytes saved.

Head-major is what fla-org's NSA calls `head_first=True`, and it supports both
orderings from one kernel -- the layout is invisible where the compute happens.
Keeping ours is a provisional choice with a condition attached for revisiting
it; see `docs/deferred-decisions.md`, DD-1.

## See also

* `docs/deferred-decisions.md` — DD-1, why head-major and when to reconsider.
* `docs/learning/03-gqa-and-the-column-merge.md` — the previous change to the
  same index, with the same "what did not change" discipline.
* `docs/gqa-measurements.md` — the grid-size effect on this card.
* `src/moba/kernels/index_fused.py` — the module docstring measuring the 230 us
  CPU-waits-for-GPU stop.
