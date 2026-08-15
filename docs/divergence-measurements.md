# Routing divergence, measured

**Date:** 2026-08-15
**Device:** RTX 3050 Ti Laptop, sm_86 | bf16 | 8 heads, head_dim 64
**Source:** `bench/divergence.py` (`uv run python -m bench.divergence`)
**Explains:** `docs/learning/01-loop-order-and-memory-traffic.md` §4–§5

These numbers come from real oracle selection masks, not estimates. They
depend on no kernel existing — divergence is a property of the *selection
pattern*, so it can be measured before a line of Triton is written, and it
determines the tile shapes the kernel should use.

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

The `per query` column (~1.98 at top_k=2, ~3.91 at top_k=4) is a sanity check:
it should sit just under top_k, since early queries have fewer past blocks
available. It does.

---

## Finding 1 — query-major utilization collapses, and larger tiles make it worse

top_k=2, varying the query tile height `BLOCK_M`:

| seq_len | block_size | BLOCK_M=16 | BLOCK_M=64 | BLOCK_M=128 |
|---:|---:|---:|---:|---:|
| 1024 | 128 | 0.429 | 0.417 | 0.417 |
| 1024 | 64 | 0.278 | 0.229 | 0.215 |
| 1024 | 32 | 0.202 | 0.125 | 0.111 |
| 1024 | **16** | 0.165 | 0.075 | 0.059 |
| 4096 | 128 | 0.203 | 0.128 | 0.121 |
| 4096 | 64 | 0.165 | 0.080 | 0.065 |
| 4096 | 32 | 0.144 | 0.056 | 0.040 |
| 4096 | **16** | **0.134** | **0.043** | **0.028** |

At S=4096, `block_size=16`, with the conventional FlashAttention tile height of
128, a query-major kernel **discards 97.2% of its tensor-core work.**

> **The unexpected part.** In dense attention, raising `BLOCK_M` *improves*
> arithmetic intensity — a larger tile amortizes each KV load across more
> consumers, which is why FA-2 tiles are large. Here raising `BLOCK_M`
> **degrades** utilization monotonically, because the union of selected blocks
> grows faster than the tile does. At S=4096, `block_size=16`, mean union goes
> 14.9 → 46.2 → 72.2 blocks as `BLOCK_M` goes 16 → 64 → 128, while each query's
> need stays fixed at 2.

This puts query-major in a vise with no good setting:

- **Small `BLOCK_M`** keeps utilization up, but wastes tensor cores (below the
  16-row minimum they cannot be filled at all) and abandons amortization.
- **Large `BLOCK_M`** restores tensor-core efficiency and pays for it by
  computing, then discarding, up to 97% of the result.

Utilization also **decays with sequence length** at fixed configuration
(0.059 → 0.038 → 0.028 at `block_size=16`, `BLOCK_M=128`), so the problem
worsens exactly in the long-context regime MoBA exists to serve.

## Finding 2 — the inverted kernel's amortization is `top_k × block_size`, independent of S

Consumers per KV block, mean over blocks with at least one consumer:

| block_size | top_k=2 | top_k=4 | `top_k × block_size` |
|---:|---:|---:|---:|
| 16 | 31.8 / 31.9 / 31.9 | 62.5 / 63.2 / 63.6 | 32 / 64 |
| 32 | 63.0 / 63.5 / 63.8 | 122.0 / 125.0 / 126.5 | 64 / 128 |
| 64 | 124.0 / 126.0 / 127.0 | 232.0 / 244.0 / 250.0 | 128 / 256 |
| 128 | 240.0 / 248.0 / 252.0 | 416.0 / 464.0 / 488.0 | 256 / 512 |

(three values per cell: S = 1024 / 2048 / 4096)

The measurement matches a closed form. Each of `S` queries selects about
`top_k` blocks, and there are `S / block_size` blocks, so

```
mean consumers per block  =  S · top_k / (S / block_size)  =  top_k · block_size
```

**`S` cancels.** Two consequences that shape the design:

1. **The inverted kernel's inner loop stays full as context grows.** Its
   amortization factor is a constant set by the gate configuration, while
   query-major utilization decays with `S`. The gap between the two designs
   widens with sequence length.
2. **Even the hardest case is comfortable.** At `block_size=16`, `top_k=2` —
   the configuration where query-major bottoms out at 0.028 — the inverted
   kernel still has **32 queries per loaded block**. That is two full
   tensor-core rows' worth, with no padding and nothing masked away.

This is the quantitative form of the project's thesis, measured on our own
hardware from our own oracle.

## Finding 3 — load imbalance, and it worsens with context

`max_consumers` versus the mean:

| seq_len | block_size | top_k | mean | max | ratio |
|---:|---:|---:|---:|---:|---:|
| 1024 | 16 | 2 | 31.8 | 109 | 3.4× |
| 2048 | 16 | 2 | 31.9 | 110 | 3.4× |
| 4096 | 16 | 2 | 31.9 | 197 | **6.2×** |
| 4096 | 128 | 4 | 488.0 | 1462 | 3.0× |

The mean is flat in `S` but the maximum is not. Some KV blocks are popular —
selected by six times the average number of queries — and popularity
concentrates as context grows.

This matters because in the inverted design the **outer loop is over KV
blocks** and the inner loop length is the consumer count. Highly variable inner
loops mean thread blocks finish at very different times, so the kernel's
runtime is set by the slowest column while other SMs idle. On a 20-SM GPU that
is severe; even on a 108-SM A100 the tail matters.

The spec must address this. Candidates, not yet decided: split hot columns
across multiple thread blocks (which forces the cross-block reduction anyway,
so it may be nearly free once that machinery exists); sort columns by consumer
count so the long ones start first; or use a persistent-kernel work queue.

---

## What this settles, and what it does not

**Settled.** The inversion is the right structure for small `block_size`. The
evidence is now ours and quantitative, not inherited from FSA and FlashMoBA.
Query-major has no viable tile height at `block_size ≤ 32`, and the inverted
kernel's amortization is provably independent of sequence length.

**Not settled.** Utilization is a *work-efficiency* proxy, not latency. It says
nothing about the inversion's three overheads — building the CSC index,
gathering scattered query rows, and reconciling partial softmax results across
thread blocks. The crossover point in wall-clock terms therefore remains
unmeasured, and finding it still requires both kernels to exist.

**Design inputs this hands the spec.**

1. Logical query group size should be derived from `top_k × block_size`, which
   is known at launch, rather than from a sparsity threshold table as
   FlashMoBA's `decide_lg_block_m` does (`docs/flashmoba-digest.md` §4.2).
2. At `block_size=16`, `top_k=2` the group is 32 — so a 32-row query tile is
   both sufficient and fully occupied. No padding needed at the project's
   hardest target configuration.
3. Load balancing across KV columns is a first-class requirement, not an
   optimization to add later.
