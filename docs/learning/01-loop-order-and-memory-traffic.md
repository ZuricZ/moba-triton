# Learning note 01 — Loop order and memory traffic

**Why MoBA's small blocks break FlashAttention, stated from first principles.**

Prerequisites: attention math (QKV, softmax, causal masking). No GPU
architecture assumed — that gets built up here.

Companion documents: `docs/flashmoba-digest.md` (what FlashMoBA's CUDA
actually does), `docs/baseline-results.md` (the measured curve this explains).

---

## 1. The physical fact everything follows from

A GPU has two kinds of memory that matter for this problem.

**HBM** (High Bandwidth Memory) is the headline capacity — the 4 GB on an
RTX 3050 Ti, the 80 GB on an A100. It is large and, by GPU standards, slow.
Your tensors live here.

**SRAM** (also called *shared memory*) is a small scratchpad physically on the
chip, adjacent to the compute units. This machine has **99 KiB per thread
block** (101376 bytes, measured). It is roughly 10–20× faster than HBM.

The number that drives every decision below: a modern GPU performs arithmetic
far faster than it can fetch operands from HBM — on the order of a hundred
floating-point operations in the time one number arrives from HBM.

So the question for any kernel is never *"how many FLOPs does this do?"* It is:

> **How many times did I cross the HBM boundary, and how much work did I
> extract from each crossing?**

That ratio is **arithmetic intensity**: FLOPs performed per byte moved from
HBM. Low intensity means *memory-bound* — the compute units idle, waiting.
That is the failure mode this entire project is fighting.

> **The reframe.** Our sparse kernel does 21× less arithmetic than dense
> attention, and is currently *slower* (`docs/baseline-results.md`). That is
> not a paradox once you accept that arithmetic was never the cost. We removed
> the cheap thing and left the expensive thing untouched. Sparsity in *math*
> does not automatically become sparsity in *memory traffic*. Converting one
> into the other is the kernel's whole job.

---

## 2. What FlashAttention does about it

Naive attention materializes the full `S×S` score matrix in HBM, softmaxes it,
then multiplies by V. At S=1024 that is a million entries written out and read
back — pure traffic, zero reuse.

FlashAttention's move is to **never materialize it**. Chop Q, K, and V into
tiles small enough to fit in SRAM. For each tile pair, compute the scores
*inside* SRAM, fold them into an output accumulator, and discard them.

The trick making this legal is **online softmax**. Softmax normally needs all
the scores at once (you need the max for numerical stability, and the full sum
to normalize). Online softmax instead keeps a running maximum `m` and a running
sum `ℓ`, and rescales the accumulated output each time a new tile arrives with
a larger max. Algebraically identical, but it never needs the whole row
resident.

Now look at the loop nest, because this is where the speed comes from:

```
for each Q tile (64 queries):              ← outer
    load Q tile
    for each K/V tile (64 keys):           ← inner
        load K/V tile into SRAM            ← ONE trip to HBM
        score ALL 64 queries against ALL 64 keys
        accumulate
```

**One load of that K/V tile feeds 64 queries.** The cost of the HBM trip is
divided across 64 consumers. This amortization — not reduced arithmetic — is
where FlashAttention's speed comes from.

---

## 3. Why kernels are query-major: the field already ran this experiment

Worth knowing before we propose inverting it, because we are proposing to go
backwards.

**FlashAttention-1 was KV-major**: outer loop over K/V blocks, inner over Q
blocks. **FlashAttention-2 swapped it**, putting Q outermost — one of its three
headline changes, alongside better work partitioning and fewer non-matmul
FLOPs.

Why the swap paid off:

With the KV block outermost, every outer iteration touches *every* query block.
Each of those queries carries accumulated output `O`, running max `m`, and
running sum `ℓ` — and none can stay in registers, because you will return to
that query on the next outer iteration. So `O`, `m`, `ℓ` are **read from HBM,
updated, and written back, once per query block per KV block**. That is heavy
traffic in the least valuable currency available: intermediate state, not
operands.

Flip to Q-major and it vanishes. One thread block owns one query tile for its
entire lifetime. `O`, `m`, `ℓ` live in registers through the whole inner loop
and touch HBM exactly once, at the end.

> **The key asymmetry.** In *dense* attention, reuse is guaranteed in both loop
> orders — every query needs every causal KV block, so whichever loop is outer,
> the inner loop is full and the loaded tile is fully consumed. Amortization is
> free either way. Given that, you pick the order that avoids the *other* cost,
> repeatedly spilling accumulator state. That is query-major.

Note carefully what that argument depends on: **reuse being free in both
orders.** Hold onto that.

---

## 4. MoBA breaks the assumption

Our selection tensor is `[B, H, S, num_blocks]` — one row **per query**. Not
per query tile. Per query. From the oracle's own output
(`src/moba/ref/naive.py::select_blocks`):

```
query 24 (blk 3): # # . # . . . .     selects blocks 0, 1
query 28 (blk 3): . # # # . . . .     selects blocks 1, 2
```

Same query tile, different KV blocks. Now re-read the loop nest in §2 and ask
what the inner loop iterates over.

It cannot iterate over "the blocks this tile selected," because the tile
selects nothing — its 64 queries each select their own. Two options remain, and
both are bad.

### Option A — load the union

Take every block any query in the tile wants.

At S=1024, `block_size=16`, `top_k=2`: 64 KV blocks exist, each query wants 2,
and a tile holds 64 queries whose selections diverge. The union covers a large
fraction of all 64 blocks. Suppose it covers 20 — that is 320 keys loaded into
SRAM, of which **each individual query uses 32**. The rest are masked off:
fetched from HBM, occupying SRAM, pushed through the tensor cores, multiplied
by zero.

You paid dense memory traffic to compute sparse attention. The 21× the sparsity
pattern offered is gone.

### Option B — load per query

Give each query only its own 2 blocks. Memory traffic is now genuinely sparse.
But a "tile" is one query, so `QKᵀ` is `1×16` times `16×64`.

**Tensor cores** — the dedicated matrix-multiply units — operate on fixed tile
shapes, 16×16 minimum. A single-row matmul wastes fifteen sixteenths of the
unit. And §2's amortization is gone entirely: one HBM trip now feeds one
consumer instead of 64.

### It gets worse exactly where we care

Finer blocks make the router more discriminating, so queries within a tile
agree *less*, so the union grows relative to what any one query needs.
**Measured** (`docs/divergence-measurements.md`), at `top_k=2`, reporting the
fraction of computed block-query pairs that survive masking:

| seq_len | block_size | BLOCK_M=16 | BLOCK_M=64 | BLOCK_M=128 |
|---:|---:|---:|---:|---:|
| 1024 | 128 | 0.429 | 0.417 | 0.417 |
| 1024 | 16 | 0.165 | 0.075 | 0.059 |
| 4096 | 128 | 0.203 | 0.128 | 0.121 |
| 4096 | **16** | **0.134** | **0.043** | **0.028** |

At S=4096 with `block_size=16` and FlashAttention's conventional tile height of
128, a query-major kernel **discards 97.2% of its tensor-core work.**

> **The counterintuitive part.** In dense attention, raising `BLOCK_M`
> *improves* arithmetic intensity — a larger tile amortizes each KV load across
> more consumers, which is why FA-2 tiles are large. Here raising `BLOCK_M`
> **degrades** utilization, because the union grows faster than the tile does.
> At S=4096, `block_size=16`, mean union goes 14.9 → 46.2 → 72.2 blocks as
> `BLOCK_M` goes 16 → 64 → 128, while each query's need stays fixed at 2.
>
> Query-major is therefore in a vise with no good setting: small `BLOCK_M`
> preserves utilization but cannot fill a tensor core and abandons
> amortization; large `BLOCK_M` fills the tensor core and then discards 97% of
> what it computed.

Utilization also decays with sequence length at fixed configuration, so the
problem worsens exactly in the long-context regime MoBA exists to serve.

The superseded sketch, kept for orientation — a 64-query tile at S=1024,
`top_k=2`:

| block_size | num_blocks | tile spans own-blocks | union vs. per-query need | utilization |
|---:|---:|---:|---|---:|
| 128 | 8 | 1 | small union, high overlap | **high** |
| 64 | 16 | 1 | moderate | falling |
| 32 | 32 | 2 | growing | low |
| 16 | 64 | 4 | union approaches everything | **very low** |

The measurement confirmed the direction and was harsher than the sketch.

This is the measured curve in `docs/baseline-results.md`: available speedup
rises 2.90× → 21.13× while latency rises 5.95 ms → 15.35 ms.

---

## 5. The inversion

Turn the nest inside out:

```
for each KV BLOCK j:                       ← outer  (was inner)
    load block j's K/V into SRAM           ← ONE trip to HBM, ever
    for each QUERY that selected block j:  ← inner  (was outer)
        score it against block j
        accumulate into that query's output
```

Every key loaded is now used by **every** query in the inner loop — because the
group was gathered by the property *"wanted this block."* Utilization returns
to 100%. Divergence stops mattering, because queries are no longer iterated at
the outer level; they are the payload gathered for a block already resident in
SRAM.

### How full is that inner loop? A closed form

Each of `S` queries selects about `top_k` blocks, and there are
`S / block_size` blocks. So the average number of queries per block is

```
S · top_k / (S / block_size)  =  top_k · block_size
```

**`S` cancels.** The inverted kernel's amortization factor is a constant fixed
by the gate configuration, while query-major utilization *decays* with `S`. The
gap between the designs widens with context length.

Measurement confirms it to within a percent
(`docs/divergence-measurements.md`): 31.9 consumers per block at
`block_size=16, top_k=2`, unchanged across S = 1024, 2048, 4096.

And note what that means for the hardest case. At `block_size=16, top_k=2` —
where query-major bottoms out at 2.8% utilization — the inverted kernel still
has **32 queries per loaded block**: two full tensor-core rows, nothing padded,
nothing masked away.

Note what did *not* change: not one FLOP was saved. Same math, same selections,
same result. Only which loop is outer — and therefore how many times each byte
crosses the HBM boundary.

> **The general shape of GPU optimization**: you rarely make the math cheaper.
> You rearrange who reuses what.

### What the inversion costs

Three things, and they are why this is a trade rather than a free win.

1. **The index must be transposed.** Our gate produces "for each query, which
   blocks." The inverted loop needs "for each block, which queries." Building
   that is a counting sort — an entire extra kernel
   (`docs/flashmoba-digest.md` §2).
2. **Query rows must be gathered**, scattered across memory, rather than read
   as a contiguous tile.
3. **We reintroduce FA-1's accumulator problem.** Query 24's output receives
   contributions from block 0 *and* block 1, handled by **different thread
   blocks, running concurrently, in no fixed order.** You cannot accumulate in
   registers. And online softmax makes it sharper: combining two partial
   attention outputs is not addition, because each carries its own `m` and `ℓ`
   that must be reconciled.

Point 3 is precisely the cost that made FA-2 abandon KV-major in the first
place. We are accepting it to buy back the reuse that MoBA destroyed.

### Therefore a crossover exists

The inversion buys reuse and pays overhead. At coarse blocks there is little
reuse to buy back and the overhead can exceed the win — **so the inverted
kernel may well be slower than the naive one at large `block_size`.**

The inversion is not "better." It is a trade that pays off past a crossover
point, and **where that point sits is an empirical question about our kernels
on our hardware.** FSA answers the analogous question for NSA by falling back
to the stock kernel at GQA group size ≥ 8. Whether FlashMoBA has any such
fallback is an open question (`docs/flashmoba-digest.md` §8).

This is the argument for building the query-major kernel too: without it we
cannot locate our own crossover, and any dispatch rule would be guesswork.

---

## 6. Prefill vs decode — and why decoding is not our problem

A natural worry: does going KV-major cost us the ability to use a KV cache?

It does not, and the reason is a useful scoping insight.

- **Prefill / training** — thousands of queries processed at once. This is what
  we are building (decision D3: forward-only, full sequence). All the
  divergence pain lives here.
- **Decode** — one query per sequence against a long KV cache.

**At decode time, MoBA's problem cannot occur.** With a single query there is
no tile for selections to diverge *across*. You take that one query's top-k
blocks and attend to them. No union, no waste, no gather. MoBA decoding is
easy, and it is out of scope for the good reason that it is not hard.

### Decoding hands us a solution template

Decoding is *forced* into KV-major — with one query, the only available loop is
over KV. And with one query against a 100k-token cache, a single thread block
would leave the other SMs idle.

The fix, **FlashDecoding**, splits the KV cache across many thread blocks; each
computes a partial attention output with its own local `m` and `ℓ`, and a small
second kernel combines the partials via the log-sum-exp trick to reconcile the
differing maxima.

That is exactly the cross-block reduction problem the inversion creates for us,
already solved in shipping code in the dense setting. "FlashDecoding-style
two-pass combine" is therefore a concrete candidate when we specify how the
inverted kernel accumulates — with the advantage of being a pattern with known
performance characteristics rather than something we invent.

---

## 7. Summary

1. Attention kernels are memory-bound. Loop order determines HBM traffic.
2. FlashAttention's speed comes from amortizing one KV tile load across a whole
   query tile.
3. Query-major won in the dense setting because reuse was free in both orders,
   so the accumulator-spilling cost became the tie-breaker.
4. MoBA's per-query selection destroys free reuse in query-major. The
   tie-breaker no longer applies.
5. Inverting to KV-block-major restores reuse — at the price of an index
   transpose, gathered reads, and cross-block reduction.
6. Because it is a trade, a crossover exists. Locating it needs both kernels.
7. Decoding is unaffected and easy; FlashDecoding supplies the reduction
   pattern we will need.

---

## Glossary

| Term | Meaning |
|---|---|
| **HBM** | Main GPU memory. Large, comparatively slow. Tensors live here. |
| **SRAM / shared memory** | On-chip scratchpad, ~99 KiB per block here. 10–20× faster than HBM. |
| **Arithmetic intensity** | FLOPs per byte moved from HBM. Low ⇒ memory-bound. |
| **Memory-bound** | Compute units idle waiting on data; adding FLOPs is free, adding loads is not. |
| **Tensor core** | Fixed-shape matrix-multiply unit, 16×16 minimum. Smaller operands waste it. |
| **Online softmax** | Running-max/running-sum formulation letting softmax proceed tile by tile. |
| **Amortization** | Dividing one HBM load's cost across many consumers of that data. |
| **Thread block** | Group of threads sharing one SRAM allocation; the unit of work scheduled to an SM. |
| **SM** | Streaming Multiprocessor — a GPU core cluster. This GPU has 20; an A100 has 108. |
| **CSC** | Compressed Sparse Column. Here: columns are KV blocks, entries are query indices. |
| **Prefill** | Processing a full input sequence at once. Our target. |
| **Decode** | Generating one token at a time against a KV cache. Out of scope. |

---

## Check yourself

Answers are derivable from the text; try before re-reading.

1. Our kernel does 21× less arithmetic than dense attention and runs slower.
   Which sentence in §1 explains this, and what quantity would you measure to
   confirm it?
2. FA-2 moved the Q loop outward and got faster. We are moving it inward and
   expect to get faster. Both cannot be universally true — what distinguishes
   the two situations?
3. At `block_size=128` with only 8 KV blocks, would you expect query-major or
   KV-block-major to win? Why?
4. Why does raising `top_k` reduce the divergence problem, and what does it cost
   in exchange?
5. Two thread blocks finish partial attention for the same query, each with its
   own `m` and `ℓ`. Why can you not simply add the two output vectors?
