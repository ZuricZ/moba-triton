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
