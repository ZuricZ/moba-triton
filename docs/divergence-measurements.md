# Routing divergence, measured

**Date:** 2026-08-22 (re-measured at realistic configurations; see "Correction")
**Device:** RTX 3050 Ti Laptop, sm_86 | bf16 | 8 heads, head_dim 128
**Source:** `bench/divergence.py` (`uv run python -m bench.divergence`)
**Explains:** `docs/learning/01-loop-order-and-memory-traffic.md` §4–§5

These numbers come from real oracle selection masks, not estimates. They depend
on no kernel existing — divergence is a property of the *selection pattern*, so
it can be measured before a line of Triton is written, and it determines the
tile shapes the kernel should use.

## What is measured

**Query-major utilization.** A query tile must load the union of blocks its
queries selected, then mask away what each individual query did not want.
Utilization is useful block-query pairs over computed ones: the fraction of a
query-major kernel's tensor-core work that survives masking.

**Consumers per KV block.** In the inverted kernel a loaded block serves every
query that selected it. This is how many — the amortization factor of the
KV-block-major design, and the thing that sizes the logical query group.

## Caveat on input data

Inputs are random Gaussian, so gate scores are unstructured and queries select
near-independently. This is the **adversarial** end of the range. A trained
model's neighbouring queries have correlated representations and would agree
more, raising utilization. Treat these as a lower bound for query-major, not an
expected production value. Measuring on real model activations is future work.

The `per query` column (7.12 at `top_k=8`, `num_blocks=32`) is a sanity check:
it should sit just under `top_k`, since early queries have fewer past blocks
available. It does.

---

## Finding 1 — utilization is governed by `num_blocks`, not by seq_len or block_size

This is the organizing fact, and it was obscured in the earlier version of this
document. Utilization depends on how many *choices* each query has, which is
`num_blocks = seq_len / block_size`. It does not matter which way you get
there: `S=4096, block_size=32` and `S=16384, block_size=128` both give 128
blocks and both give the same utilization.

Query-major utilization by `num_blocks` and query tile height `BLOCK_M`:

| top_k | num_blocks | BLOCK_M=16 | BLOCK_M=64 | BLOCK_M=128 |
|---:|---:|---:|---:|---:|
| 4 | 8 | 0.722 | 0.722 | 0.722 |
| 4 | 16 | 0.430 | 0.426 | 0.426 |
| 4 | 32 | 0.254 | 0.231 | 0.231 |
| 4 | 64 | 0.169 | 0.122 | 0.118 |
| 4 | 128 | 0.128 | 0.069 | 0.061 |
| 8 | 8 | 1.000 | 1.000 | 1.000 |
| 8 | 16 | 0.735 | 0.735 | 0.735 |
| 8 | 32 | 0.434 | 0.432 | 0.432 |
| 8 | 64 | 0.249 | 0.233 | 0.229 |
| 8 | **128** | 0.160 | **0.121** | **0.118** |

**Where the project's target sits.** FlashMoBA benchmarks `block_size=128,
top_k=8` and sweeps seq_len from 8K to 512K. At `block_size=128`:

| seq_len | num_blocks | utilization at BLOCK_M=64 | discarded |
|---:|---:|---:|---:|
| 4096 | 32 | 0.432 | 57% |
| 8192 | 64 | 0.233 | 77% |
| **16384** | **128** | **0.121** | **88%** |
| 65536 | 512 | below the sweep | — |

So the loop-order argument holds, and it *strengthens* with context length,
which is the regime MoBA exists to serve. But it is weaker than earlier
versions of this document claimed at the sequence lengths this 4 GB card can
actually run: at S=4096 a query-major kernel discards 57%, not 88%.

> **The unexpected part.** In dense attention, raising `BLOCK_M` *improves*
> arithmetic intensity — a larger tile amortizes each KV load across more
> consumers, which is why FA-2 tiles are large. Here raising `BLOCK_M` never
> helps and eventually hurts, because the union of selected blocks grows faster
> than the tile does.
>
> Note where it bites. At `num_blocks=32` the tile height barely matters
> (0.434 / 0.432 / 0.432) — there are so few blocks that a 16-query tile
> already wants most of them. At `num_blocks=128` it matters a great deal
> (0.160 / 0.121 / 0.118), because the mean union grows 48.7 → 64.3 → 66.0
> while each query's need stays fixed near 7.8.

This puts query-major in a vise with no good setting at long context:

- **Small `BLOCK_M`** keeps utilization up, but wastes tensor cores (below the
  16-row minimum they cannot be filled at all) and abandons amortization.
- **Large `BLOCK_M`** restores tensor-core efficiency and pays for it by
  computing, then discarding, ~88% of the result.

## Finding 2 — the inverted kernel's amortization is `top_k × block_size`

Consumers per KV block at S=4096, mean over blocks with at least one consumer:

| block_size | top_k | consumers/block | max | `top_k × block_size` |
|---:|---:|---:|---:|---:|
| 32 | 4 | 126 | 542 | 128 |
| 32 | 8 | 249 | 1006 | 256 |
| 64 | 4 | 250 | 913 | 256 |
| 64 | 8 | 484 | 1597 | 512 |
| 128 | 4 | 488 | 1453 | 512 |
| 128 | 8 | 912 | 2361 | 1024 |

Measured tracks predicted to within 5% everywhere, the shortfall being early
queries that have fewer past blocks than the budget allows.

**Why it is independent of S.** Each query selects `top_k` blocks, so there are
`S × top_k` selections spread over `S / block_size` blocks. Divide: `top_k ×
block_size` consumers per block, with `S` cancelling. Growing the context does
not thin out a block's audience — it adds proportionally more blocks.

That is the whole case for the inversion. One K/V load serves ~900 queries at
the realistic configuration, against a query-major tile that loads a block to
serve, on average, a third of the 64 queries holding it.

## Finding 3 — the tail is long, and that is what D8 addresses

Maximum consumers runs 2.4–4× the mean (2361 against 912 at the realistic
configuration). One program will do several times the work of the average one.

That skew is why dispatch order matters, and
`docs/load-balance-measurements.md` prices it: greedy longest-first scheduling
absorbs it almost entirely (1.00–1.02× of ideal makespan against 1.07–1.21× for
arbitrary order), because what governs load imbalance is jobs-per-processor
rather than skew.

---

## Correction — the earlier version of this document

The first version was measured at `top_k=2` with `block_size` swept down to 16,
and reported 0.028 utilization — "97% discarded". Both the block size and the
budget were wrong for this project:

- `block_size=16` appears in neither reference implementation. FlashMoBA uses
  128; MoonshotAI uses 512/4096. Sparsity comes from `num_blocks` growing with
  *context*, not from shrinking blocks.
- `top_k=2` is far below either reference's 8–12, and it matters here in a way
  that is easy to miss: **a larger budget raises utilization**, because queries
  that each pick 8 of 32 blocks overlap far more than queries picking 2 of 32.
  At `num_blocks=128`, going from `top_k=4` to `top_k=8` nearly doubles
  utilization, 0.069 to 0.121.

The figure "0.121, 88% discarded" survived the correction but changed meaning.
It was `block_size=128, top_k=2, S=4096`; it is now `top_k=8` at
`num_blocks=128`, which at `block_size=128` means **S=16384**. Documents that
attached 88% to `S=4096, block_size=128` were wrong — that case is 57%.

The conclusion is unchanged and the reasoning is now attached to the right
numbers.
