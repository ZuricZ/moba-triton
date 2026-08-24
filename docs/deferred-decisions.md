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

## DD-1 — Tensor layout: head-major `[B, H, S, D]`

**Decided:** 2026-08-24, while planning variable-length support.

**What we do now.** Every public entry point takes `[B, H, S, D]`, and the
packed variable-length path takes `[1, H, T, D]` — the same ordering with the
batch axis pinned to 1. This is torch SDPA's convention, and it is what
fla-org's NSA calls `head_first=True`.

**The alternative.** FlashAttention uses `[B, S, H, D]` padded and `[T, H, D]`
packed — token-major. NSA supports both from one kernel via a `head_first` flag
(`references/native-sparse-attention/.../ops/naive.py:44`).

**Why token-major is genuinely better for a library.**

1. It is what the model produces. `hidden [B, S, E] -> Linear -> [B, S, H*D] ->
   view [B, S, H, D]`. Sequence outer, heads inner, straight out of the
   projection. Head-major requires a `.transpose(1, 2)`, which is exactly what
   HuggingFace writes before every SDPA call.
2. Packing becomes a memcpy. With the token axis outermost, one token's `H*D`
   bytes are contiguous, so FlashAttention's `unpad_input` is a row gather. In
   head-major a token is `H` separate `D`-sized runs, `S*D` apart — a strided
   gather. This is also why FA's `bert_padding.py` helpers cannot be lifted
   verbatim into a head-major codebase.
3. The kernel is indifferent. A program loads a Q tile for one head into shared
   memory either way; head-major strides by `D`, token-major by `H*D`, both
   regular and both coalescing. The layout is invisible where the compute
   happens, which is why NSA can offer both — and why FA optimised the only
   place it *is* visible, the caller.

**Why we kept head-major anyway.**

* **Consistency with our own API beats consistency with someone else's.** 379
  tests, every published benchmark number, and the `sdpa_dense` baseline all
  speak `[B, H, S, D]`. Changing it invalidates the comparison this project
  exists to make.
* The transpose cost is symmetric — someone always pays it. Head-major charges
  FlashAttention-shaped callers; token-major charges torch-shaped ones. Ours are
  torch-shaped.
* Estimated cost to a token-major caller: one transpose-and-copy of Q, K and V,
  ~25 MB at `T=8192, H=8, D=64, bf16`, ≈300 us against a ~2280 us forward —
  about **13%**.
* Estimated cost to switch: roughly a dozen address expressions across six
  kernels, all of the same shape (`q_row = heads * S + rows` becomes
  `q_row = rows * H_Q + heads`). Not a rewrite. The larger cost is the test and
  benchmark churn, not the kernels.

**Revisit when any of these becomes true.**

* The project acquires users outside its own benchmark suite — anyone
  integrating with a HuggingFace or vLLM-shaped stack will arrive token-major.
* We want FlashAttention's `unpad_input` / `pad_input` rather than writing our
  own packers, which becomes attractive if padded-plus-mask input turns out to
  be the common case.
* A head-to-head against FlashMoBA on rented hardware shows measurable time
  going into layout conversion at the benchmark boundary.

**Do not revisit before** the varlen work is complete and green. Layout is an
API-wide change, and doing it while a feature is half-built would mix two
sources of breakage. The right moment is with everything passing, as one
mechanical commit, ideally behind a `head_first`-style flag so both conventions
are supported rather than one being swapped for the other.

**See also:** `docs/learning/04-packed-sequences-and-where-the-work-goes.md`
for the layout table and the packing picture;
`docs/superpowers/plans/2026-08-24-varlen-support.md` decision D-VL-1 for the
form this took in the varlen plan.
