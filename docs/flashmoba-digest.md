# FlashMoBA, in plain English — and why small blocks break the naive loop order

**Date:** 2026-08-15
**Source read:** `references/flash-moba/` at depth-1 clone, 253 stars, C++/CUDA
**Cross-read:** FSA (arXiv 2508.18224), `references/Flash-Sparse-Attention/`
**Companion data:** `docs/baseline-results.md`

Every factual claim about the code cites the file and line it came from.
Anything I could not determine from the source is marked as an open question
rather than guessed. The kernel was **read, not run** — it does not build for
this machine's sm_86 (see "Why we could not run it").

---

## 1. What FlashMoBA computes

The same algorithm our oracle implements (`src/moba/ref/naive.py`), with one
convention we adopted from Moonshot after the Task 8 cross-check: `top_k`
counts the query's own block. See `docs/semantics-crosscheck.md`.

Structurally it is a **fork of FlashAttention-2** — the file layout is FA-2's
(`flash_fwd_kernel.h`, `kernel_traits.h`, `softmax.h`, `mask.h`, `utils.h`),
with MoBA-specific machinery added alongside.

It splits into **two separate CUDA kernels** plus a sort, driven from
`flash_moba/flash_moba_interface.py:688-727`:

```
flash_topk_varlen_func(...)   ->  col_offsets, col_nnz, indices
decide_lg_block_m(...)        ->  lg_block_m
flash_moba_attn_varlen_func(..., col_offsets, col_nnz, indices, lg_block_m)
```

Stage one computes the gate and the top-k selection. Stage two does the
attention over whatever stage one selected. They are separate kernels because
they have completely different shapes: the gate is a small, wide reduction,
the attention is a big tiled matmul.

## 2. The loop order — the key design decision

**The selection is stored as a sparse matrix in CSC (Compressed Sparse
Column) form, with columns being KV blocks and rows being queries.**

The construction is a three-pass counting sort inside
`csrc/flash_topk/src/flash_topk_kernel.h`:

1. **Count** — `atomicAdd(&col_nnz_ptr[lg_col_idx], 1)` at `:309`. For each
   KV block, how many queries selected it?
2. **Exclusive prefix sum** — `:398-400`:
   `col_offsets_ptr[i] = cumsum; cumsum += col_nnz_ptr[i]; col_nnz_ptr[i] = 0;`
   Turn counts into start offsets, and reset the counters for reuse.
3. **Scatter** — `:473-474`:
   `int offset = atomicAdd(&col_nnz_ptr[lg_col_idx], 1);`
   `indices_ptr[col_offsets_ptr[lg_col_idx] + offset] = m_block * kBlockM + i;`
   Write each **query index** into its selected column's slot.

Note what is stored: `m_block * kBlockM + i` is a **query row index**, filed
under a **KV block column**. The attention kernel then walks columns, not
queries. In the attention kernel the same array is named `moba_row_indices`
(`flash_moba_interface.py:186`), confirming rows = queries.

The attention kernel finds the queries belonging to a column via
`IndexGatherScatter::get_lg_block_indices` in
`csrc/flash_moba/src/index_gather_scatter.h:50-88`, which **binary searches**
the sorted index range. Gathering the actual KV rows uses CUTLASS-style
gather iterators (`IndexedGather` at `:159`, `CustomStride` at `:204`).

So the loop nest is, from outside in:

| Level | Iterates over | What lives where |
|---|---|---|
| grid | (**query window** of `lg_block_m`, batch, head) | `dim3 grid(num_lg_m_block, b, h)`, `flash_fwd_launch_template.h:51-52` |
| outer | every KV block column, in order | column's sorted query list, binary-searched for this window's queries (`index_gather_scatter.h:50-87`) |
| inner | 64-row chunks of those gathered queries | K/V tile and gathered Q in shared memory; running max/sum in a shared-memory table indexed by query; running output in an fp32 HBM scratch, loaded and stored per chunk |

**Corrected 2026-09-15.** This table originally put the KV block column in the
grid. It is not: one program owns a *window* of `lg_block_m` queries, and the
KV-block-major loop with the gather runs *inside* that program. The data
structure above is KV-block-major; the launch is not. The consequences of the
window are in `docs/traffic-derivation.md` and `docs/a100-results.md`.

**Within a program, the KV block is the outer loop and the gathered queries
are the inner loop.** That is the inversion of the obvious order, and it is the
whole idea. The window is what bounds it.

## 3. Why small blocks break the naive loop order

The naive order is query-major, and it is what our oracle does and what any
straightforward FlashAttention adaptation does:

```
for each query tile (say 64 queries):
    for each KV block that tile selected:
        load the block, compute, accumulate
```

Three things go wrong as `block_size` shrinks, and they compound.

### 3.1 Selection diverges across a query tile

This is the root cause, and it is visible directly in our own oracle. The
selection mask has shape `[B, H, S, num_blocks]` — **per query, not per query
block** (`src/moba/ref/naive.py::select_blocks`). Two queries sitting in the
same block routinely choose different KV blocks. From the visualization in
`docs/session-log/2026-08-15-foundations-planning.md`:

```
query 24 (blk 3): # # . # . . . .     selects blocks 0, 1
query 28 (blk 3): . # # # . . . .     selects blocks 1, 2
```

FlashAttention's entire performance argument is that one KV tile load is
amortized across a whole query tile. MoBA violates the precondition. If 64
queries each want their own 2 blocks, the union can approach 128 distinct
blocks, and each individual block is wanted by only a handful of them. Load
the union and most of the work is masked away; load per query and the
amortization is gone.

**Smaller blocks make this strictly worse.** Finer granularity means selection
is more discriminating, so the tile's queries agree less.

### 3.2 The matmul tiles get too small for the hardware

Tensor cores — the matrix-multiply units this kernel targets — operate on
fixed tile shapes. FlashMoBA's MMA atom is
`SM80_16x8x16_F32F16F16F32_TN`, and the tiler is
`Tile<Int<16 * kNWarps>, _16, _16>` (`csrc/flash_moba/src/kernel_traits.h:92-93`),
so the hardware wants a 16-element granularity on each matrix dimension.

A `block_size` of 16 gives an inner matmul whose N dimension is 16 — the bare
minimum. Anything smaller must be padded. FSA states this explicitly for the
analogous problem in NSA: the kernel *"must pad data to satisfy hardware
requirements on the matrix dimensions of matrix multiplication"*
(`references/Flash-Sparse-Attention/README.md`). Padded lanes consume
tensor-core throughput and produce nothing.

### 3.3 The gate grows as the attention shrinks

`num_blocks = S / block_size`, so the gate's score matrix is `S x num_blocks`.
Going from `block_size=128` to `16` at `S=1024` multiplies `num_blocks` by 8,
so the router does 8x more work — precisely as the work it routes to gets
cheaper. The overhead fraction rises twice over.

### 3.4 What our measurements show

From `docs/baseline-results.md`, at S=1024, top_k=2, on this sm_86 laptop:

| block_size | sparsity | ceiling | moonshot latency | attainment |
|---:|---:|---:|---:|---:|
| 128 | 0.344 | 2.90x | 5.95 ms | 0.008 |
| 16 | 0.047 | **21.13x** | **15.35 ms** | **0.000** |

As the theoretically available speedup rises 7x, a real implementation gets
**2.6x slower**. That is sections 3.1-3.3 compounding, measured.

## 4. What FlashMoBA does about it

**Two moves, both aimed squarely at restoring amortization.**

### 4.1 Invert the loop: iterate KV blocks, gather their queries

Because the sparse pattern is stored KV-block-major (section 2), each KV block
is loaded **once** and every query that selected it is processed against that
one load. Divergence across queries stops mattering, because queries are no
longer the thing being iterated at the outer level — they are the payload
gathered for a block that is already resident.

This converts "many query tiles each re-loading overlapping block sets" into
"each block loaded once, serving all its consumers".

### 4.2 Grow the query group as sparsity increases

`flash_moba/flash_moba_interface.py:20-48`:

```python
if   sparsity <= 0.5: lg_block_m = 128
elif sparsity <= 0.7: lg_block_m = 256
elif sparsity <= 0.8: lg_block_m = 512
elif sparsity <= 0.9: lg_block_m = 768
else:                 lg_block_m = 1024
```

Counterintuitive until you see the loop order. `lg_block_m` is the **query
window one program owns** (section 2, corrected). Sparser configs mean fewer
queries in a window select any given block, so a small window leaves the loaded
block underused. Enlarging the window raises the expected number of consumers
per loaded block, restoring some of the arithmetic intensity that sparsity took
away. It cannot restore all of it: at S=65536 the policy returns 1024 (0.969
sparsity, top row), and 1024 x 8 / 512 blocks is 16 queries per (window, block)
pair, padded to a 64-row tile.

The cost is capacity, and it is paid at *compile* time. The shared-memory
tables for the window's row indices and running max/sum are sized by
`kMxLgBlockM = 1024` (`kernel_traits.h:110-111, 128-129`) whatever
`decide_lg_block_m` returns at runtime, so a program needs 84 KiB at D=64 and
100 KiB at D=128. An A100 SM has 164 KiB: one program fits, two do not. Which
brings us to the constraint that concerns this machine directly.

### 4.3 They special-case sm_86 for shared memory

`flash_moba/flash_moba_interface.py:43-47`:

```python
# sm86 (A6000, 3090) and sm89 (4090, L40) have smaller shared memory than A100 (sm80)
if major == 8 and minor > 0:
    lg_block_m = min(lg_block_m, 512)
```

This is the 99 KiB vs 164 KiB shared-memory-per-SM gap between sm_86 and
sm_80, hard-coded as a workaround in shipped code. Our device probe measures
101376 bytes per block on this GPU (`docs/session-log/...`). The fix in 4.2
and its ceiling are the same resource.

## 5. What FSA does, and why it is the closest template

FSA solves the structurally identical problem for NSA. Its README:

- *"when GQA group size is not sufficiently large, NSA selected attention
  kernel must pad data to satisfy hardware requirements on the matrix
  dimensions of matrix multiplication"*
- *"FSA exchanges the kernel loop order of original NSA kernel design"*
- it targets *"GQA group sizes smaller than 8"* and falls back to stock NSA
  at group size >= 8.

Map it onto us:

| | NSA / FSA | MoBA / FlashMoBA |
|---|---|---|
| Small quantity | GQA group size | `block_size` |
| Symptom | matmul M too small, padded | matmul N too small, plus divergent selection |
| Fix | exchange kernel loop order | KV-block-major CSC + larger logical query group |
| Fallback | stock NSA when group >= 8 | (open question, see below) |

The important precedent: **FSA is Triton**, and it demonstrates that the
loop-order inversion is expressible in Triton rather than requiring CUTLASS.
FlashMoBA leans on CUTLASS gather iterators
(`index_gather_scatter.h:159-250`), which we cannot use from Triton — so FSA
is the better structural model even though FlashMoBA is the closer algorithm.

## 6. What transfers to our Triton kernel

Phrased so each can become a requirement in the kernel spec.

1. **The sparse pattern must be materialized KV-block-major before the
   attention kernel runs.** Our gate currently produces a `[B,H,S,num_blocks]`
   boolean mask; the kernel wants CSC-style `(col_offsets, col_nnz, indices)`.
   That conversion is itself a kernel (counting sort), and it should be
   benchmarked separately — FlashMoBA spends a whole kernel plus a sort on it.
2. **Selection and attention should be separate kernels.** Their shapes and
   parallelization differ enough that fusing them forces one to run badly.
3. **The logical query group size must be tunable, and chosen from sparsity.**
   Not a fixed constant. `decide_lg_block_m`'s thresholds are a reasonable
   starting point for our autotuner's search space.
4. **Shared memory is the binding constraint, and it is 99 KiB here.** Any
   tiling that assumes an A100's 164 KiB will compile and then fail at launch.
   Budget explicitly; do not discover it.
5. **Keep the matmul N dimension at or above 16.** At `block_size < 16` the
   kernel must group multiple blocks per tile or accept padding waste. This is
   the regime the project cares most about, and it is where a design decision
   is unavoidable.
6. **Follow FSA's structure over FlashMoBA's.** Triton has no CUTLASS gather
   iterator; FSA proves the inversion works in Triton.

## 7. Why we could not run it (superseded)

**Superseded 2026-09-05.** It was built with `FLASH_MOBA_CUDA_ARCHS=86` and
launches at every sequence length from 1024 to 8192 on this card, because
`decide_lg_block_m` chooses the window at runtime rather than compile time
(`implementations/flash_moba.py` docstring). It has since been measured on an
A100 (`docs/a100-results.md`). The paragraph below is kept as the prediction
that turned out wrong.

`csrc` builds via `setup.py`, whose `cuda_archs()` defaults to
`"80;90;100;120"` (`setup.py:55`) — **sm_86 is absent**. Even with 86 added,
section 4.3 shows the authors already treat sm_86 as a constrained target and
cap `lg_block_m` for it, so a build could still hit shared-memory limits at
launch rather than compile time.

Consequences: this digest is grounded in reading, and the `flash_moba` backend
reports itself unavailable with that reason
(`implementations/flash_moba.py`). Nothing in the harness blocks on it.

## 8. Open questions

- **Does FlashMoBA fall back to dense attention** when sparsity is low enough
  that the machinery does not pay? FSA falls back to stock NSA above group
  size 8; I did not find an equivalent switch. Worth locating before we design
  our own fallback policy.
- **How does the backward pass handle the inverted loop order?**
  `flash_bwd_kernel.h` is the largest file at 893 lines and was not read; our
  project is forward-only for now (decision D3).
- ~~**What exactly is a "logical column"** (`lg_col_idx`)?~~ Answered
  2026-09-15: it is one KV block of `n_lg_block_dim` keys, and
  `n_lg_block_dim` is `moba_chunk_size` (`index_gather_scatter.h:52`,
  `flash_moba_interface.py:714-731`), so 128 at the benchmark config. Not
  coarser than a block.
- **Triton-MoBA's gate semantics** remain unverified — its module needs
  flash-attn to import. See `docs/semantics-crosscheck.md`.
