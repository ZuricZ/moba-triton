# MoBA Triton Kernel

A [Triton](https://triton-lang.org) implementation of **MoBA** — Mixture of
Block Attention ([Lu et al., Moonshot AI](https://arxiv.org/abs/2502.13189)) —
written to turn MoBA's arithmetic sparsity into wall-clock speedup at long
context.

MoBA is sparse attention where **each query picks its own handful of key
blocks**. That per-query choice is what makes it cheap on paper and awkward on
a GPU: it breaks the one assumption FlashAttention is built on. This kernel is
organized around that problem — it inverts FlashAttention's loop order so a
program owns a *key* block and gathers the queries that chose it, rather than
owning a query tile and loading the union of whatever its queries wanted.

Forward and backward, grouped-query attention, and variable-length packing are
all implemented and tested against a PyTorch oracle. **440 tests pass.**

> **Scope.** This is a learning project, and the numbers below come from an
> RTX 3050 Ti laptop (20 SMs, 4 GB). Read the *ratios*; the absolute times mean
> nothing. A datacenter head-to-head is still pending.

---

## Results so far

`block_size=128, top_k=8` (FlashMoBA's published configuration), bf16,
shape `[1, S, 4, 64]`, forward pass:

| S | blocks | ours | FlashMoBA (CUDA) | torch SDPA | MoonshotAI (naive) |
|---:|---:|---:|---:|---:|---:|
| 1024 | 8 | 0.293 ms | 1.418 ms | 0.083 ms | 4.386 ms |
| 2048 | 16 | 0.476 ms | 1.655 ms | 0.231 ms | 9.619 ms |
| 4096 | 32 | 0.766 ms | 2.438 ms | 0.776 ms | 31.686 ms |
| 8192 | 64 | **1.596 ms** | 4.698 ms | 2.830 ms | 382.931 ms |

At S=8192 — the only length here with real sparsity — that is **1.77× torch
SDPA on the forward and 2.02× on the backward**.

Three things this table says, and one it does not:

- **Sparsity is not free below the crossover.** At S=1024 there are 8 blocks
  and `top_k=8` selects *all of them* — there is no sparsity to exploit, only
  the routing overhead of pretending there is. Dense attention should win, and
  it does. The kernel overtakes dense between S=4096 and S=8192, and the gap
  widens from there, which is the regime MoBA exists for.
- **Against MoonshotAI's *naive* path it is 15–240× faster** — but read that
  with care. MoonshotAI ship two implementations: `moba_naive.py`, written to
  be read, which materialises the score matrix; and `moba_efficient.py`, which
  fuses onto FlashAttention and is what their library actually runs. Only the
  naive one is measured above, because the fused one needs flash-attn 2.6.3.
  **A 240× number against code written for clarity is not a claim about their
  fast path**, and the honest comparison is still pending.
- **The numbers scale the way the algorithm says they should.** MoBA attends
  `top_k × block_size` = 1024 keys per query whatever S is, so cost is linear
  rather than quadratic — and 0.766 → 1.596 ms across a doubling of S is a
  factor of 2.08.
- It does **not** mean we are faster than FlashMoBA. Their kernel picks a
  512-query tile at S=8192, sized for the 164 KiB of shared memory an A100
  has; this card has 99 KiB and 20 SMs. That comparison needs the rented card.

Full numbers, the four measurement artifacts caught along the way, and the
thermal protocol these required:
[`docs/reference-comparison.md`](docs/reference-comparison.md).

---

## Using it

Requires a CUDA GPU and Linux (or WSL2). Triton has no native-Windows support
here.

```bash
git clone https://github.com/ZuricZ/moba-triton.git
cd moba-triton
uv sync                      # torch is pinned to the cu126 index; see pyproject.toml
uv run pytest -m "not slow"
```

The whole public API is two names:

```python
import torch
from moba import MoBAConfig, moba_attention

B, S, H, D = 1, 8192, 8, 64
q, k, v = (torch.randn(B, S, H, D, device="cuda", dtype=torch.bfloat16,
                       requires_grad=True) for _ in range(3))

cfg = MoBAConfig(block_size=128, top_k=8)
out = moba_attention(q, k, v, cfg)      # [B, S, H, D]
out.sum().backward()                     # dq, dk, dv — the gate gets no gradient
```

Tensors crossing the API are **token-major `[B, S, H, D]`**, matching
FlashAttention and every reference implementation.

Three optional arguments cover the rest:

| Argument | What it does |
|---|---|
| `causal=True` | Masks within the query's own block. Blocks wholly in the past need no mask. |
| `cu_seqlens` | Variable-length packing: pass `[1, T, H, D]` plus cumulative token counts, and each sequence attends independently. |
| `index` | A precomputed routing index, to hold block selection fixed while inputs vary. |

`MoBAConfig` also carries four flags pinning down details the MoBA paper leaves
open — tie-breaking, whether the query's own block consumes a `top_k` slot,
whether pooling respects the causal mask. Defaults match MoonshotAI's reference
bitwise. Reasoning is in [`src/moba/config.py`](src/moba/config.py).

---

## How it works

### 1. What MoBA computes

Dense attention has every query look at every key. MoBA cuts the keys into
fixed-size blocks, builds **one representative key per block** by averaging it,
and has each query score itself against the *representatives* — a much smaller
problem. It keeps the `top_k` highest-scoring blocks and attends only to those.

With `block_size=128, top_k=8` at S=8192, that's 64 blocks of which each query
reads 8: **one eighth of the work** of dense attention.

The catch is in one word: *each*. The choice is per query, not per query
block.

### 2. Why per-query choice breaks FlashAttention

FlashAttention's speed rests on a single trick: load a tile of keys into fast
on-chip memory (SRAM) **once**, then reuse that load across a whole tile of
queries. One expensive trip to HBM, many cheap uses.

MoBA removes the precondition. Two adjacent queries routinely select different
blocks, so a query-major program must load the *union* of everything its
queries asked for, then mask away what each individual query didn't want. The
masked-away part is tensor-core work that gets computed and thrown out.

Measured, at `block_size=128, top_k=8`:

| seq_len | num_blocks | utilization | discarded |
|---:|---:|---:|---:|
| 4096 | 32 | 0.43 | 57% |
| 8192 | 64 | 0.23 | 77% |
| 16384 | 128 | 0.12 | **88%** |

It gets *worse* as context grows — precisely the direction MoBA is meant to
scale in.

**So the kernel inverts the loop: a program owns one KV block and gathers
exactly the queries that selected it.** Every key it loads is used by every
query it holds. Utilization is 100% by construction rather than by tuning.

| | FlashAttention-2 | This kernel |
|---|---|---|
| A program owns | a tile of **queries** | a tile of **keys** |
| It loops over | key blocks | the queries that chose this block |
| Which keys does it load? | a contiguous range | one block, always used in full |
| Wasted tensor-core work | none (dense) | none (by construction) |
| Softmax normalizer | complete when the loop ends | *incomplete* — needs a merge pass |
| Numerics, tiling, SRAM budget | — | **same** |

The last two rows are the trade. Inverting the loop buys perfect utilization
and costs output ownership: a query's answer now arrives in pieces from several
programs.

### 3. The five stages

```
   pooling  ──▶  gating  ──▶  routing  ──▶  moba_fwd  ──▶  combine
   one key       score vs.     transpose      attend        merge the
   per block     reps, top-k   the choice     partials      partials
```

Each is a module under [`src/moba/kernels/`](src/moba/kernels/), named for the
stage rather than the mechanism.

**Routing is the interesting one.** The gate answers *"for each query, which
blocks?"*. The inverted kernel needs the transpose: *"for each block, which
queries?"*. That transpose is stored as **CSC** — compressed sparse column,
where columns are KV blocks and entries are query indices.

Small enough to count by hand — 6 queries, `block_size=2` (so 3 blocks),
`top_k=2` with the query's own block included:

```
  gate output (query → blocks)        CSC transpose (block → queries)

  q0 → B0                             B0 │ q0 q1 q2 q3 q4      (5 queries)
  q1 → B0                             B1 │       q2 q3    q5   (3 queries)
  q2 → B0 B1                          B2 │             q4 q5   (2 queries)
  q3 → B0 B1
  q4 → B0    B2                       col_nnz     = [5, 3, 2]
  q5 →    B1 B2                       col_offsets = [0, 5, 8, 10]
                                      row_indices = [0,1,2,3,4, 2,3,5, 4,5]
```

Read left to right and it's six independent choices. Read top to bottom and
it's three work items, each with a query list attached. **Same decision, two
directions; the right-hand form is the one a KV-block-major kernel can
schedule.**

That picture also shows the kernel's main open problem: B0 has 5 queries and B2
has 2. Early blocks are visible to everyone, late blocks to almost nobody, and
that skew (measured at 6.2× in the worst column) is a load-balancing problem the
scheduler has to absorb.

The transpose is built entirely on the GPU — count, prefix-sum, scatter — with
no `nonzero()` anywhere. `nonzero()` returns a data-dependent number of results,
which forces PyTorch to copy a count back to the CPU before it can allocate. On
this machine that round trip costs ~230 µs against ~6 µs for the same
arithmetic kept on device.

### 4. Merging the partials

Because a query's row is now split across programs, each program produces an
**un-normalized** partial: it cannot divide by a softmax denominator it hasn't
seen. Each writes out its local max `m`, its local sum `l`, and its
accumulator.

`combine_partials` then does exactly what FlashAttention's online softmax does
between loop iterations — rescale by `exp(m_local − m_global)` and sum — except
across *programs* instead of across iterations. Identical algebra, relocated.

This is the one genuinely extra pass the inversion costs. It buys the 88%.

### 5. The backward

Standard FlashAttention-2 recomputation: the softmax is rebuilt from `q`, `k`,
`v` and the saved log-sum-exp rather than stored, so no S×S matrix ever exists.
The two large forward buffers (64 MB at S=8192) are deliberately *not* saved.

**The gate receives no gradient.** MoBA's routing is a hard top-k, and the
backward reuses the forward's selection as a fixed sparsity pattern — matching
Moonshot's reference, which likewise treats block choice as
non-differentiable. So `q` gets gradient through the attention scores, not
through its role in choosing blocks.

---

## Layout

```
src/moba/                 the library, and nothing else
  config.py               MoBAConfig — semantics flags, block arithmetic
  naive.py                the PyTorch oracle: an executable definition of MoBA
  device.py  gqa.py  varlen.py
  kernels/
    pooling.py            block → representative key
    gating.py             query × representatives → top-k blocks
    routing.py            that choice, transposed into CSC
    moba_fwd.py           one program per (KV block, batch, KV head)
    moba_bwd.py           the backward of the same
    tuning.py             host-side launch policy the kernels share
    attention.py          the two halves as one differentiable op

implementations/          uniform adapters for the head-to-head — ours, the
                          three references, and two dense baselines
bench/                    benchmark drivers; run as modules (python -m bench.compare)
tests/                    440 tests, plus the tolerance policy
docs/                     measurements and learning notes
```

`implementations/` and `tests/` sit outside the package on purpose: nothing
under `src/moba` may import either, so the library ships without the harness.

---

## Documentation

Written to be read in order.

| Document | What it covers |
|---|---|
| [learning/01 — loop order and memory traffic](docs/learning/01-loop-order-and-memory-traffic.md) | Why per-query routing breaks FlashAttention, from first principles. No GPU background assumed. |
| [learning/02 — anatomy of a FlashAttention kernel](docs/learning/02-anatomy-of-a-flash-attention-kernel.md) | Reading a real Triton kernel: grids, tiling, occupancy, warps, online softmax — then breaking it on purpose. |
| [learning/03 — GQA and the column merge](docs/learning/03-gqa-and-the-column-merge.md) | How grouped-query attention changes the CSC columns, and what stays the same. |
| [learning/04 — packed sequences and where the work goes](docs/learning/04-packed-sequences-and-where-the-work-goes.md) | Variable-length packing, and what it does to the block boundaries. |
| [learning/05 — the backward pass](docs/learning/05-the-backward-pass.md) | Recomputation, and why dQ needs a different strategy than dK/dV. |
| [flashmoba-digest](docs/flashmoba-digest.md) | What FlashMoBA's CUDA actually does, cited line by line. |
| [reference-comparison](docs/reference-comparison.md) | The head-to-head, and three measurement artifacts that flattered us. |
| [divergence-measurements](docs/divergence-measurements.md) | Routing divergence measured from real selection masks. |
| [gather-cost-measurements](docs/gather-cost-measurements.md) | What gathering scattered query rows actually costs. |
| [load-balance-measurements](docs/load-balance-measurements.md) | What KV-block skew costs after scheduling. |
| [semantics-crosscheck](docs/semantics-crosscheck.md) | Rule-by-rule agreement with MoonshotAI's reference. |

## References

- **MoBA** — Lu et al., Moonshot AI — [arXiv:2502.13189](https://arxiv.org/abs/2502.13189)
- **FlashMoBA** — Xiao, Guo, Mazaheri, Han (MIT Han Lab) — [arXiv:2511.11571](https://arxiv.org/abs/2511.11571) — the performance target
- **FSA** — [arXiv:2508.18224](https://arxiv.org/abs/2508.18224) — closest structural template: the same loop-order problem, for NSA
- **FlashAttention** — [arXiv:2205.14135](https://arxiv.org/abs/2205.14135) · **FlashAttention-2** — [arXiv:2307.08691](https://arxiv.org/abs/2307.08691)

Reference implementations are vendored read-only under `references/`
(gitignored) and are never modified. Fetch them with
`scripts/fetch_references.sh`.
