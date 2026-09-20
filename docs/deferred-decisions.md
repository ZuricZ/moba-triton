# Deferred decisions

Choices made deliberately and provisionally, with the condition that should
trigger revisiting them. Distinct from the design specs, which record decisions
considered settled, and from the measurement docs, which record what was
observed.

An entry belongs here when the reasoning is sound *today* but rests on something
that may change — a user base, a benchmark target, a hardware constraint. The
point is that the revisit is triggered by a condition, not by someone
remembering.

---

## DD-1 — Tensor layout — **RESOLVED 2026-09-03, reversed**

**Decided:** 2026-08-24, while planning variable-length support.
**Resolved:** 2026-09-03. We moved to token-major `[B, S, H, D]`, which is the
opposite of what this entry concluded. Measurements in
`docs/layout-measurements.md`; the work is
`docs/superpowers/plans/2026-09-03-token-major-layout.md`.

**What we do now.** Every public entry point takes `[B, S, H, D]`, and the
packed path takes `[1, T, H, D]` — the same ordering with the batch axis pinned
to 1. This is FlashAttention-2's convention, fla-org's NSA's *native* one, and
FlashMoBA's. There is no compatibility flag; adding one later is ~15 lines at
two entry points.

Buffers we allocate for ourselves did not move: `part_m`, `part_l`, `part_acc`,
`dq_part`, `lse`, `delta`, block representatives, selection masks and every
`CSCIndex` field are still head-major. Nothing outside the project reads them.
`moba.heads.expand_kv` takes a `dim` argument because it is the one function
that sees both conventions.

---

### Where this entry's reasoning was wrong

**1. The NSA citation supported the opposite of what it was used for.** This
entry said head-major "is what fla-org's NSA calls `head_first=True`". True, and
misleading. NSA's *kernels* are token-major — `parallel.py:515` addresses
`(bos + i_t) * HQ*K` with `make_block_ptr(..., (HQ, K), (K, 1), ...)`, hardcoded
and `constexpr`-derived. `head_first=True` is a `rearrange` shim at the door
(`parallel.py:1381`), and NSA **refuses it for varlen** outright
(`naive.py:86`: *"Sequences with variable lengths are not supported for
head-first mode"*). We were citing the compatibility hack as precedent for the
design.

Every kernel in this project's reference set — FA-2, NSA, FlashMoBA — is
token-major. We were the only one that was not, and we arrived there by copying
torch SDPA, which has no varlen path and therefore never had to care.

**2. "The kernel is indifferent" was asserted, not measured** — and
under-argued. Our loop order is block-major, so a program owns `BLOCK_SIZE`
consecutive keys, which is exactly the access pattern head-major rewards and
which FA and NSA never have. The conclusion held anyway, at **0.983×–1.024×
with no systematic direction** (`bench/layout_probe.py`), but for a reason this
entry never gave: all of a batch's heads run concurrently, so the bytes
token-major skips are read at the same instant by the sibling program.

**3. The strongest argument was missing, because it predates varlen.**
Head-major's `stride_head` is `S*D` — it *contains the sequence length* — so
packing changes it and the kernels needed a `q_span` branch in their hottest
address computation. No token-major stride contains `S`: heads are `D` apart
and tokens `H*D` apart whether packed or padded. The packed and padded query
addresses collapsed into one expression and `q_span` stopped existing. This is
almost certainly why FA and NSA chose token-major in the first place.

---

### What the reversal cost and bought

**Cost: neutral on time.** Same-session A/B: −1.4% to +1.3% across
`bench/stages.py` and `bench/varlen_sweep.py`, the two benchmarks disagreeing
about the sign. Nothing was generalized — no stride became a runtime value, so
`constexpr` folding is intact. Test count identical, 429 passed / 7 skipped.

**Bought, concretely:**

* `pack` is FlashAttention's row gather. Head-major made it `H` short
  transactions per token plus a permute; a token's `H*D` bytes are now
  contiguous. This entry recorded that as its "concrete cost carried today".
* `from_nested` lost its transpose entirely — `nt.values()` *is* our packed
  form now.
* `_refs.to_varlen` went from permute-reshape-copy to a bare `reshape`.
* `sdpa_dense.forward_packed` got **7–16% faster**, because building a jagged
  nested tensor no longer needs a copy. Our published "vs dense" ratios drop as
  a result — the baseline stopped paying for something it never needed, which
  makes every such ratio conservative rather than flattering.
* The caller-side transpose this entry estimated at ~13% of a forward for a
  HuggingFace-shaped caller is gone.

**Bought, structurally:** a head-to-head against FlashMoBA on rented hardware
no longer has a layout conversion at the boundary — trigger 3 below, satisfied
by construction rather than by measurement.

---

### Original entry, kept for the record

The reasoning below is what we believed on 2026-08-24. Two of its three
arguments for token-major were correct and we discounted them; the argument we
kept — consistency with our own API — turned out to cost nothing to abandon,
because the layouts measured the same and the tests carried the churn.

**Why token-major is genuinely better for a library.**

1. It is what the model produces. `hidden [B, S, E] -> Linear -> [B, S, H*D] ->
   view [B, S, H, D]`. Head-major requires a `.transpose(1, 2)`, which is
   exactly what HuggingFace writes before every SDPA call.
2. Packing becomes a memcpy. With the token axis outermost, one token's `H*D`
   bytes are contiguous, so FlashAttention's `unpad_input` is a row gather.
3. The kernel is indifferent. A program loads a Q tile for one head into shared
   memory either way; head-major strides by `D`, token-major by `H*D`, both
   regular and both coalescing.

**Why we kept head-major anyway.**

* **Consistency with our own API beats consistency with someone else's.** 379
  tests, every published benchmark number, and the `sdpa_dense` baseline all
  speak `[B, H, S, D]`. Changing it invalidates the comparison this project
  exists to make.
* The transpose cost is symmetric — someone always pays it.
* Estimated cost to switch: roughly a dozen address expressions across six
  kernels. Not a rewrite. The larger cost is the test and benchmark churn.

*(Both estimates proved right. It was ~18 expressions across five kernels, and
the churn was indeed the bulk of the work. The claim that changing it
"invalidates the comparison" proved wrong: the layouts measure the same, so the
old numbers remain comparable.)*

**Revisit triggers, and how they resolved.**

| trigger | status at resolution |
|---|---|
| users outside our own benchmark suite | had **not** fired |
| wanting FA's `unpad_input`/`pad_input` | had **not** fired |
| head-to-head showing time in layout conversion | had **not** fired — the head-to-head has not run |
| `pack` showing up in a profile | had **not** fired |

**None of the stated triggers had fired.** The revisit happened because the
user asked, and because reading NSA's source showed the entry's central
citation was backwards. Recorded plainly: a trigger list is a floor, not a
ceiling, and "no trigger fired" is not evidence a decision was right.

**See also:** `docs/layout-measurements.md`;
`docs/learning/04-packed-sequences-and-where-the-work-goes.md` for the layout
table and the packing picture.

---

## DD-2 — Partial-output buffer is `top_k` x the output, unwindowed

**Decided:** 2026-09-15, while auditing the FlashMoBA comparison
(`docs/a100-results.md`, "The mechanism, corrected").

**What we do now.** The attention stage writes one partial output per
(query, selected block) into `part_acc`, shaped `[B, H, S, top_k, D]`
(`src/moba/kernels/moba_fwd.py`, `empty_partials`), and a separate combine
kernel merges the slots. The buffer is allocated for the whole sequence at
once. That is what lets every KV-block program gather across all of S, so a
block sees about `top_k * block_size` queries at any context length and the
tensor-core tiles stay full.

**What it costs.** `top_k` copies of the output, in the compute dtype:

| shape | `part_acc` |
|---|---:|
| `[1, 65536, 4, 64]`, top_k 8 | 268 MB |
| `[2, 1048576, 16, 128]`, top_k 8 (FlashMoBA's benchmark shape at 1M) | 64 GiB |

The second does not fit an 80 GB card. FlashMoBA's window design keeps its
scratch at 1x the output in fp32 (16 GiB there) and pays for it in padding and
occupancy at every S; we made the opposite trade.

**Why not fix it now.** Nothing we can rent runs 1M at that shape, and the fix
does not touch the kernel: chunk the *queries* at the host level (build the CSC
for a 64K-query chunk, attend, combine, reuse the buffer). A 64K window bounded
by HBM keeps 64 queries per (chunk, block) at 1M, so no padding penalty
appears; K/V are re-read once per chunk, about 16x at 1M, which is small next
to the partials traffic. The window FlashMoBA uses is 1024 because theirs is
bounded by shared memory; ours would be bounded by HBM, three orders of
magnitude larger.

**Revisit when:** a target shape's `B * S * H * top_k * D * itemsize` exceeds
about half the card's memory, or a benchmark sweep goes past S=262144 at
H=16, D=128.

**Amended 2026-09-20 — the wall is closer than this entry says.** "Nothing we
can rent runs 1M at that shape" is true and was the wrong thing to measure.
The trigger is the product, not the sequence length, and it is reached by
raising heads and head_dim at a sequence length we already benchmark:

| shape | `part_acc` | share of a 40 GB card |
|---|---:|---:|
| `[1, 65536, 4, 64]`, top_k 8 | 268 MB | 0.7% |
| `[4, 65536, 32, 128]`, top_k 8 | **16.0 GiB** | **43%** |
| `[2, 1048576, 16, 128]`, top_k 8 | 64 GiB | does not fit |

The middle row is the "same, batch 4" entry in `bench/residual.py`'s
`SHAPES`.

**Corrected the same day.** An earlier version of this paragraph said that
row "has already run on the rented A100, where `ours` held 2.67x-2.91x".
It had not. The A100's panel B ran only at S=8192
(`results/sm80-a100/data/residual-b.json`) and the fairness run topped out
at `B1 S16384 H32 D128`; the 2.67x-2.91x figures are from those smaller
shapes. That shape had never run anywhere.

**And when it finally did, it did not hit a memory wall at all.** On an
80 GB H100, where 16 GiB of scratch fits with room to spare, it raised an
illegal memory access — a 32-bit index overflow in the partials
addressing, not exhaustion. See DD-2a below.

So the "revisit when" condition is not a distant 1M-token scenario. It is
within a factor of about 1.2 of a configuration in the benchmark suite today,
and any of batch, heads, head_dim or `top_k` doubling crosses it. The fix is
unchanged and still does not touch the kernel -- chunk the queries host-side
and reuse the buffer -- but it should be costed before a sweep raises one of
those four, not after an out-of-memory error names the shape for us.

## DD-2a -- the partials buffer overflowed int32 before it overflowed memory

**Found and fixed:** 2026-09-20 on the rented H100
(`results/sm90-h100/logs/int32-overflow.log`).

DD-2 costed this buffer in bytes and asked when it stops fitting. The
binding limit turned out to be *elements*, and it arrives four times
sooner.

Triton's offsets are int32 unless something in the expression is widened,
and `part_acc` is `[B, H, S, top_k, D]` -- the largest object this design
allocates, against the output's `[B, S, H, D]`. Two address expressions
multiplied a batch- or head-scaled base by `D` without widening:
`moba_fwd_kernel`'s store to `PartAcc`, and `moba_combine_kernel`'s
`a_base`. The backward had the same pattern in `DQPart`, larger still by
the tile factor.

The boundary follows the *product*, not any one axis, which is what makes
it a 32-bit index rather than a capacity limit:

| `B*H*S*top_k*D` | shape | before the fix |
|---:|---|---|
| 2,147,483,648 (2^31) | B=1 H=32 S=65536 D=128 | ran |
| 2,147,483,648 (2^31) | B=4 H=32 S=16384 D=128 | ran |
| 4,294,967,296 (2^32) | B=2 S=65536 / B=4 S=32768 / B=8 S=16384 | **all three: illegal memory access** |
| 8,589,934,592 (2^33) | B=4 H=32 S=65536 D=128 | illegal memory access |

Three different routes to the same element count fail identically, and two
different shapes sitting exactly on 2^31 both pass. An 8 GiB buffer failing
on an 80 GB card rules out exhaustion.

**Fixed** by widening only the base of each chain to int64; every term
added to an int64 is promoted, so the per-lane offsets stay 32-bit and the
cost is a few address registers per program. Re-measured at S=65536:
forward 1.492 -> 1.485 ms, backward 2.658 -> 2.580 ms, i.e. nothing. All
six shapes above now return finite output.

FlashMoBA never meets this: FA-2 declares `using index_t = int64_t` once
in `csrc/flash_moba/src/flash.h:24` and every stride inherits it, and their
window design allocates no `top_k`-deep partials buffer at all.
`routing.py`'s `PACK64` guard is this project solving the same problem in
the same place by hand.

**Headroom after the fix.** int64 tops out at 9.22e18 elements. FlashMoBA's
1M benchmark shape is 3.44e10 -- four parts per billion of the range.
`part_acc` there is 64 GiB, so memory binds roughly 268 million times before
the index does, which is the correct ordering: exhaustion is catchable, an
overflowed index is an illegal access that poisons the CUDA context and
takes unrelated work down with it. DD-2's host-side query chunking is still
the fix for the memory limit; it is no longer racing an indexing bug to get
there.
