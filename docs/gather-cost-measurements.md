# The cost of gathering, measured

**Date:** 2026-08-17
**Device:** RTX 3050 Ti Laptop, sm_86, 20 SMs | bf16
**Source:** `bench/gather_cost.py` (`uv run python -m bench.gather_cost`)
**Settles:** design question A — where the query permutation lives

---

## The question

In the KV-block-major design, a program owns one KV block and needs the query
rows that selected it. Those rows sit at scattered positions in `Q`. Two ways
to get them:

**A1 — materialize a permuted copy.** Build `Q_perm = Q[row_indices]` as a real
tensor, then have the kernel read it contiguously. This is what MoonshotAI's
efficient path does, and it is the standard MoE token-dispatch pattern.

**A2 — gather inside the kernel.** Pass the index arrays in and load rows
directly from wherever they are. No intermediate tensor.

A1 costs an extra write and read of `S x top_k x D` elements. It buys a
contiguous access pattern. The question is what that pattern is worth.

## The hypothesis

A cache line is 128 bytes. One query row is `D x sizeof(dtype)` bytes —
**exactly 128 at D=64 in bf16**.

The familiar warning that "gathers destroy coalescing" describes gathering
*elements*: 32 lanes each pull a 128-byte line to use 4 bytes of it, wasting
97% of the bandwidth. Gathering whole, line-sized *rows* is a different
operation. 32 scattered rows and 32 contiguous rows are both 32 cache lines,
every byte used.

Predicted cost of scattering: small, from lost DRAM row-buffer locality and
defeated prefetching — not from transaction count.

## Method

Three modes read **exactly the same rows**, so data volume is held fixed and
only the access pattern varies:

| mode | what it does | isolates |
|---|---|---|
| contiguous | rows 0..N-1, no index array | baseline |
| identity | same order, addresses read from an index array | cost of loading indices |
| permuted | a random permutation of the same rows | cost of scattering |

Holding the rows fixed is what makes the ratio meaningful. Varying volume and
pattern together would measure nothing in particular. (Same discipline as the
tolerance policy: hold the discrete thing fixed, vary the continuous one.)

`N = 32768` rows — `B*H*S` at S=4096, H=8. `BLOCK_M = 32`, the expected MoBA
group size `top_k x block_size` at the project's hardest configuration.

## Results

| head_dim | row bytes | working set | contiguous | identity | permuted | index cost | **scatter cost** |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 32 | 64 | 2 MB | 22.8 us | 23.5 us | 23.8 us | 1.03x | **1.04x** |
| 64 | 128 | 4 MB | 36.5 us | 37.2 us | 37.6 us | 1.02x | **1.03x** |
| 128 | 256 | 8 MB | 62.5 us | 63.1 us | 63.6 us | 1.01x | **1.02x** |

**Scattering costs 2–3% at the head dimensions real models use.** Loading the
index array itself costs another 2%.

The prediction was directionally right and quantitatively too pessimistic: the
estimate was 1.2–1.5x at D=64 (measured **1.03x**) and a 2x penalty at D=32
(measured **1.07x**).

### Caveat on the D=32 row

At D=32 the working set is 2 MB, which roughly fits this GPU's L2 cache, so
that row is not fully DRAM-bound and likely understates the penalty for larger
working sets. D=64 (4 MB) and D=128 (8 MB) both exceed L2 and are genuinely
memory-bound — those are the trustworthy numbers, and they agree with each
other.

## What it settles

Composing measured components for the query-fetch step at D=64:

```
A2  gather in kernel:      read rows directly, scattered      ~36 us
A1  materialize first:     index_select reads them scattered  ~36 us
                         + write Q_perm to HBM                ~34 us
                         + kernel reads Q_perm back           ~36 us
                                                              -------
                                                              ~106 us
```

**A1 does roughly 3x the work to buy an access pattern worth 3%.**

> **Decision: gather inside the kernel (A2).** Not on the cache-line argument —
> on 1.03x.

Rejected alongside A1: CUTLASS-style gather iterators, as used by FlashMoBA
(`docs/flashmoba-digest.md` §2). They are unavailable from Triton. FSA's
precedent shows `tl.load` with computed row offsets is sufficient, and these
measurements show it is also cheap.

## What this does not settle

- **Write traffic for partial outputs is untouched by this decision.** Both A1
  and A2 still emit `(m, l, acc)` per (query, program) pair for the combine
  pass. That traffic is inherent to the loop inversion, not to the gather.
- **Load imbalance** (design question B) is independent and still open. See
  `docs/divergence-measurements.md` Finding 3: 6.2x max/mean consumers per KV
  block, worsening with context.
- **Row sortedness.** The permuted case here is fully random. Real CSC indices
  within a column can be kept ascending, which should be no worse and may be
  slightly better. Not separately measured; the random case is the pessimistic
  bound.

## Design constraint this creates

The result depends on rows being at least one cache line:

```
D x sizeof(dtype) >= 128 bytes
```

At D=64 bf16 this holds exactly. At D=32 bf16 a row is half a line, and a
gathered row pulls a full line to use half of it. The measured penalty there
was mild (1.07x, in an L2-resident working set), but the kernel should record
this as a stated assumption rather than an accident of the shapes tested.
