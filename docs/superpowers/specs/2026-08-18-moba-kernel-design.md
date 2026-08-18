# MoBA Triton Kernel — Design

**Date:** 2026-08-18
**Follows:** `2026-08-15-moba-foundations-design.md` (decisions D1–D5, semantics,
tolerance policy, backend protocol — all still binding)
**Evidence base:** `docs/divergence-measurements.md`,
`docs/gather-cost-measurements.md`, `docs/load-balance-measurements.md`,
`docs/flashmoba-digest.md`, `docs/learning/01`, `docs/learning/02`

Every decision below cites the measurement that settled it. Where something is
assumed rather than measured, it says so.

---

## Goal

A Triton MoBA forward kernel that retains its advantage at **small block sizes**
(`block_size` 16–32), where every implementation we can measure is currently
*slower* than dense attention despite doing up to 21x less arithmetic.

## Non-goals

- Backward pass. Forward only (D3 stands); the design must not preclude it.
- Beating FlashMoBA on an A100. This machine is for correctness and structure;
  headline numbers need rented hardware.
- Decode / KV-cache inference. With a single query there is no routing
  divergence, so the problem this kernel solves does not arise
  (`docs/learning/01` §6).
- GQA and variable-length batching. Both are mechanical additions afterwards.

---

## Decisions

**D6 — KV-block-major loop order.** A program owns one KV block and processes
the queries that selected it, rather than owning a query tile and looping over
blocks.

*Rationale:* MoBA's selection is per query, so a query-major tile must load the
union of blocks its queries wanted and mask away the rest. Measured utilization
at S=4096, `block_size=16`, `BLOCK_M=128` is **0.028** — 97.2% of tensor-core
work discarded — and it *decays* with sequence length, worsening in exactly the
long-context regime MoBA exists to serve. No tile height escapes it: small tiles
keep utilization up but abandon amortization; large tiles restore amortization
and discard 97%.

Inverted, every key loaded is used by every query held. The consumer count has a
closed form, `top_k x block_size`, **independent of S** (measured 31.9 against a
predicted 32), so the inner loop stays full as context grows.

*Rejected:* query-major with a tuned tile height — no viable setting exists at
`block_size <= 32`. *Known risk:* at `block_size=128` query-major utilization is
0.417, so a crossover exists where the inversion's overheads exceed its benefit.
Unmeasured in wall-clock terms; see Deferred.

**D7 — Gather query rows inside the kernel. Do not materialize a permuted Q.**

*Rationale:* one query row is `D x sizeof(dtype)` bytes — exactly one 128-byte
cache line at D=64 bf16 — so gathering whole rows costs the same transaction
count as reading them in order. Measured scatter cost: **1.02–1.03x** at D=64
and D=128; index loading adds 2%. Materializing `Q_perm` costs an extra write
and read of the same data: roughly **3x the work to buy a pattern worth 3%**.

*Rejected:* materialized permutation (MoonshotAI's efficient path, the standard
MoE dispatch pattern) — correct for PyTorch-level code, unnecessary when writing
the kernel. CUTLASS gather iterators (FlashMoBA, digest §2) — unavailable from
Triton; FSA demonstrates `tl.load` with computed row offsets suffices.

**D8 — Dispatch KV blocks longest-first. No chunking in v1.**

*Rationale:* skew barely predicts cost. Measured makespan penalty against ideal:
**4.9x skew costs 3%; 2.0x skew costs 73%**, because severity is governed by
jobs per processor, not job unevenness. In the target regime (S=4096,
`block_size=16`) there are 2048 programs against ~40 concurrent slots and random
dispatch costs **1.03x**. Where it does bite (256 programs, 1.27x),
longest-first recovers it to **1.05x**.

Sorting is close to free: consumer counts are a by-product of the counting sort
that builds the index, sorting a few thousand int32s costs microseconds, and it
requires **no scatter-back** — it permutes which program handles which block,
not the data, and output addresses derive from query indices.

*Deferred, not rejected:* chunking hot blocks across programs. It wins only at
~1.6 jobs/processor, and it makes each chunk re-load the KV block, spending the
memory reuse D6 exists to obtain. See Deferred for the trigger.

**D9 — Separate local and sparse passes.** The query's own block is handled by a
dedicated block-diagonal kernel; only *past* blocks go through the gathered
path. Results are merged in the combine pass.

*Rationale:* the own block is the only one needing an internal causal mask, and
it is contiguous, always selected, and free of routing divergence. Routing it
through the gather machinery would pay index cost for structure known
statically. Splitting it also removes causal masking from the sparse pass
entirely — every block it sees lies wholly in the past. Matches MoBA Algorithm 1
(`get_self_attn_block` versus `index_select_moba_attn_block`).

*Consequence to note:* the sparse pass's group size becomes
`(top_k - 1) x block_size`, not `top_k x block_size`, since the own block left.
At `top_k=2, block_size=16` that is **16** — exactly the tensor-core minimum.
Tight but workable; see Global constraints.

**D10 — Fixed-slot partial storage with a separate combine pass.** Each sparse
program writes `(m, l, acc)` per query into a slot indexed by the query's
selection rank, not by CSC position. A final kernel merges slots.

*Rationale:* a query receives partials from several programs, and combining them
requires each partial's running max — take-max, rescale, add. `atomicAdd`
performs only simple operations and cannot express that; splitting it into
separate atomics reintroduces the race. Adding raw `acc` values is precisely the
"no rescale" bug measured in `docs/learning/02` §5: **800x error, no crash, no
NaN, output range identical to correct.**

Indexing partials by selection rank rather than CSC position lets the combine
pass read a fixed `K = top_k - 1` slots per query at constant stride —
contiguous and coalesced, no indirection — at the cost of padding rows for early
queries with fewer past blocks. Those are masked by initializing `m` to `-inf`.

**D11 — Gate and index construction in PyTorch for v1.** The Triton work in v1
is the attention passes. The gate (mean-pool, score, top-k) and the CSC build
(count, prefix sum, scatter) stay in PyTorch.

*Rationale:* the project's thesis concerns the attention loop order; that is
where the difficulty and the learning are. The gate is a small wide reduction
and a sort — a separate, easier optimization problem. Building it in PyTorch
first gives a correct reference and a measurement of what it actually costs
before effort is spent on it.

*Risk, acknowledged:* `docs/baseline-results.md` shows gate cost grows as
`block_size` shrinks (`num_blocks = S / block_size`), so it may dominate at
`block_size=16`. That is a measurement to take, not a reason to pre-optimize.
See Deferred.

---

## Global constraints

Additional to the foundations spec.

- **`D x sizeof(dtype) >= 128 bytes`.** D7's measurement holds because a query
  row is at least one cache line. At D=32 bf16 (64 bytes) a gathered row pulls a
  full line to use half of it. D=64 and D=128 are the supported cases; smaller D
  must be validated, not assumed.
- **Sparse-pass group size**
  `GROUP = max(16, round_up_16((top_k - 1) * block_size))`. Derived analytically
  from the measured consumer count, not from a threshold table as FlashMoBA's
  `decide_lg_block_m` uses (digest §4.2).
- **Shared memory budget 99 KiB per program** on sm_86. From learning note 02,
  shared memory for a tiled kernel follows `2*D*BLOCK_M + 8*D*BLOCK_N` (bf16),
  the factor 8 being double-buffering of K and V. Budget explicitly; overflow
  fails the launch rather than degrading.
- **Registers cap at 255 per thread**; exceeding it spills silently to HBM.
  Watch `n_spills`, do not assume.
- Partial storage is fp32. Accumulating in bf16 loses precision that the tensor
  cores do not lose.

---

## Architecture

Four kernels, three stages.

```
Stage 1  gate + index build          (PyTorch, v1)
           -> col_nnz, col_offsets, row_indices, slot_of_pair, order

Stage 2a local pass       (Triton)   block-diagonal causal attention
           -> local_m, local_l, local_acc       [B,H,S], [B,H,S], [B,H,S,D]

Stage 2b sparse pass      (Triton)   gathered, KV-block-major, no causal mask
           -> part_m, part_l, part_acc     [B,H,S,K], [B,H,S,K], [B,H,S,K,D]

Stage 3  combine          (Triton)   log-sum-exp merge of K+1 partials
           -> out                                        [B,H,S,D]
```

`K = top_k - 1` is the number of past blocks a query may select.

### Data structures

Per (batch, head), with `N = num_blocks` and `T` the pair count. `T <= S * K`
strictly: early queries have fewer than `K` past blocks available, and `top_k`
is a ceiling rather than a quota (foundations spec, semantics rule on partial
budgets). Allocate at the `S * K` upper bound so shapes are static, and let
`col_nnz` carry the true counts:

| array | shape | dtype | meaning |
|---|---|---|---|
| `col_nnz` | `[B,H,N]` | int32 | queries selecting each block |
| `col_offsets` | `[B,H,N+1]` | int32 | exclusive prefix sum of `col_nnz` |
| `row_indices` | `[B,H,T]` | int32 | query indices, grouped by block, ascending within a block |
| `slot_of_pair` | `[B,H,T]` | int8 | which of the query's K slots this pair owns |
| `order` | `[B,H,N]` | int32 | block ids sorted by `col_nnz` descending |

Ascending `row_indices` within a column is deliberate: it costs nothing during
construction and preserves whatever DRAM locality is available. The gather
measurement used a fully random permutation, so it is the pessimistic bound.

### Stage 1 — gate and index build (PyTorch)

Follows the oracle exactly (`src/moba/ref/naive.py`), including the stable
descending sort for tie-breaking toward the lower block index — an unstable
tie-break makes the kernel non-deterministic run to run, which is nearly
indistinguishable from a race when debugging.

1. `kbar = block_representatives(k, cfg)` — mean-pool, dividing by real
   occupancy so a short tail block is not shrunk toward zero.
2. `scores = einsum(q, kbar)`, unscaled: `1/sqrt(D)` is a positive monotone
   transform and cannot reorder a ranking.
3. Mask non-past blocks to `-inf`; stable argsort; take `K`.
4. Counting sort into CSC: `bincount` -> `cumsum` -> scatter.
5. `order = argsort(col_nnz, descending=True)`.

Exposed as `moba.kernels.index.build_csc(q, k, cfg)` returning the table above,
so it can be unit-tested against the oracle's boolean mask independently of any
attention kernel.

### Stage 2a — local pass

One program per `BLOCK_M` queries. Loads only the keys and values of the blocks
those queries belong to, applies the causal mask within the block, and emits
`(m, l, acc)` un-normalized.

With `BLOCK_M = block_size` the tile aligns exactly with one block and the mask
is the standard causal triangle. Larger `BLOCK_M` spans several blocks and needs
a block-diagonal mask; both are expressible with `tl.where`, and `BLOCK_M` is an
autotune parameter. At `block_size=16` this is a thin band — small absolute
work, but `S` rows of it, so it is not free.

### Stage 2b — sparse pass

The heart of the kernel.

```
pid_col, pid_bh = program_id(0), program_id(1)
block = order[pid_bh, pid_col]                   # longest-first dispatch (D8)

lo, n = col_offsets[block], col_nnz[block]
kv = load K[block], V[block]                     # once, into SRAM

for chunk in range(0, n, GROUP):
    rows  = row_indices[lo + chunk : lo + chunk + GROUP]   # query ids
    slots = slot_of_pair[lo + chunk : ...]
    q     = load Q[rows]                         # gathered rows (D7)

    qk  = dot(q, trans(kv.k)) * sm_scale         # no causal mask needed (D9)
    m   = max(qk, axis=1)
    p   = exp(qk - m)
    l   = sum(p, axis=1)
    acc = dot(p, kv.v)

    store part_m[rows, slots]   = m
    store part_l[rows, slots]   = l
    store part_acc[rows, slots] = acc            # un-normalized (D10)
```

Notes that matter:

- **No causal masking.** Every block here lies wholly in the past. The
  per-element `tl.where` disappears.
- **No online softmax loop.** A program sees each of its queries against exactly
  one block, so the max and sum are computed in one shot. The online machinery
  moves entirely into stage 3. This is a genuine simplification relative to a
  dense FA-2 inner loop.
- **Uniform branching.** Every query in the program selected this block, so
  "does this query need this block?" is uniform across the warp — the free cell
  of the divergence table in learning note 02 §4.
- `n` is not generally a multiple of `GROUP`; the tail chunk is masked.

### Stage 3 — combine

One program per `BLOCK_M` queries. Reads `K+1` partials per query — `K` sparse
slots at fixed stride plus the local one — and merges:

```
m_g = max over slots of m_i                   # empty slots hold -inf
acc = sum over slots of acc_i * exp(m_i - m_g)
l   = sum over slots of l_i   * exp(m_i - m_g)
out = acc / l
```

This is `alpha` from FA-2's inner loop, applied *across programs* instead of
across loop iterations. Structurally identical to FlashDecoding's reduction and
to MoBA Algorithm 1's `combine_with_online_softmax`.

Empty slots carry `m = -inf`, `l = 0`, `acc = 0`, so `exp(-inf - m_g) = 0`
removes them without a branch.

---

## Numerics

- Partials in fp32; inputs bf16 (fp16 secondary).
- Tolerance policy from D5 is unchanged: routing checked **exactly**, arithmetic
  checked relative to a same-dtype fp64 reference.
- **Known failure fingerprint.** A broken rescale in stage 3 produces plausible,
  bounded, non-crashing output with roughly 800x the expected error
  (`docs/learning/02` §5). Treat a max-abs-error around `3e-1` with a healthy
  output range as a rescale bug, not precision loss; bf16 precision loss looks
  like `1e-2`.

---

## Testing strategy

Extends the four tiers from the foundations spec.

**Index construction (new, unit).** `build_csc` round-trips against the oracle's
boolean selection mask: expanding the CSC back to a dense mask must be exactly
equal. Also assert `col_nnz.sum() == mask.sum()`, `col_offsets` is a valid
exclusive prefix sum, `row_indices` is ascending within every column, and
`slot_of_pair` is a permutation of `range(k_i)` for each query.

**Kernel versus oracle.** Routing compared exactly; outputs compared under the
relative tolerance policy. Full shape x gate sweep.

**Invariants (no reference needed).** These catch bugs a reference cannot:

1. *Unselected blocks do not matter.* Perturb a KV block a query did **not**
   select; that query's output must be **bitwise identical**. Directly tests the
   gather indices.
2. *Causality.* Perturbing keys and values after position `i` must leave output
   rows `<= i` bitwise identical.
3. *Tile-size invariance.* Every `BLOCK_M`/`GROUP` combination must produce the
   same result. Spilling and low occupancy may make a config slow; never wrong.
4. *Dispatch-order invariance.* Results with and without longest-first ordering
   must be identical — a good race detector, since reordering changes which
   program runs when.

**Determinism.** The same inputs must produce bitwise identical outputs across
runs. Stage 3 sums a fixed number of slots in a fixed order, so this should
hold; if it does not, something is racing.

---

## Deferred, with triggers

| item | trigger |
|---|---|
| Chunk hot KV blocks across programs | profiling shows tail stalls in a targeted regime, or jobs/processor approaches 1. Note an A100's 108 SMs cut jobs/processor ~5x versus this machine. |
| Triton gate kernel | gate exceeds ~20% of total forward time in the benchmark |
| Triton CSC build | index build exceeds ~10% of total forward time |
| Query-major control kernel | needed to locate the wall-clock crossover before writing any dispatch or fallback rule |
| Backward pass | after forward attainment targets are met |
| GQA, varlen | when a real model integration needs them |

---

## Risks

**The crossover is unmeasured.** D6 rests on work-efficiency, not latency.
Utilization 0.028 says query-major *cannot* win at small blocks; it does not say
the inverted kernel *will*, because CSC construction, gathered reads and the
combine pass are unpriced. Mitigation: stage-by-stage benchmarking so each
overhead is attributable.

**The combine pass may dominate.** It reads `K+1` partials and writes `S x D`
outputs — traffic proportional to `top_k`. At large `top_k` this could exceed
the attention itself. Mitigation: measure at `top_k` 2, 4, 8; if it dominates,
the fix is to have one program handle several of a query's blocks, reducing the
partial count.

**`GROUP = 16` at the hardest configuration** sits exactly at the tensor-core
minimum, with no margin. If `(top_k - 1) x block_size < 16`, padding waste
begins. Mitigation: the constraint is stated, and grouping multiple KV blocks
per program is the known escape.

**Gate cost may swamp everything at `block_size=16`** (D11). Mitigation:
measured, not assumed; a Triton gate is already scoped in Deferred.

---

## Success criteria

Measured on this machine, `block_size=16`, `top_k=2`, bf16, against dense SDPA:

| level | criterion |
|---|---|
| **Correctness** | exact routing match; arithmetic within the D5 relative tolerance; all four invariants hold |
| **Threshold** | faster than dense SDPA at S >= 2048 — the bar no measured implementation currently clears |
| **Target** | attainment >= 0.10 at S=4096 (currently 0.000) |
| **Stretch** | attainment >= 0.30, roughly a 6x real speedup over dense |

Attainment on 20 SMs with 4 GB is not the project's headline number; the *shape*
of the curve across `block_size` is what transfers to an A100.
