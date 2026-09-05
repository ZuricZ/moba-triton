# Variable-length (packed) batches — measurements

`bench/varlen_sweep.py`, RTX 3050 Ti Laptop (sm_86, 20 SMs), bf16,
`block_size=128, top_k=8, H_q=8, H_kv=2, head_dim=64`, `max_seqlen=2048`.
Median of 100, partial buffers preallocated.

---

## The predictions, stated before the numbers

**P1 — the gain belongs to the data, not the kernel.** Padding waste is
`1 - mean(len)/max(len)`. A batch whose sequences are all the same length has
nothing to save, and no kernel work will find any. A varlen path that showed a
gain on a uniform batch would be measuring something other than packing.

**P2 — the launch grid does not grow.** The plan predicted it would, on the
theory that a ragged batch yields more real blocks than a padded one yields
useful ones, which per `docs/gqa-measurements.md` would matter a lot on 20 SMs.
Reading the launch says otherwise. `attend_partials` launches
`(index.num_blocks, n_seq · h_kv)`, and `num_blocks` is
`cdiv(max_seqlen, block_size)` packed against `cdiv(S, block_size)` padded —
the same number, because the design packs the data and **pads the index**.

> Packing does not shrink the launch. It empties programs.

A short sequence's high-numbered columns have `col_nnz == 0`, and those
programs return before loading anything.

**P3 — so the gain should be the memory term**, and roughly proportional to the
token count, tracking `part_acc` — the largest buffer in the design.

P1 and P2 held. **P3 was wrong, in our favour**, and the reason is the
interesting part of this document.

---

## The numbers

`waste` is `1 - mean(len)/max(len)`. `gain` is MoBA padded ÷ MoBA packed.

| kind | B | waste | tokens pad→pk | `part_acc` pad→pk | grid | live cols pad→pk | pairs pad→pk | moba pad | moba pk | gain |
|---|---|---|---|---|---|---|---|---|---|---|
| uniform | 4 | 0% | 8192 → 8192 | 64.0 → 64.0 MB | 128 | 128 → 128 | 52.4M → 52.4M | 2.412 | 2.433 | **0.991×** |
| mixed | 4 | 38% | 8192 → 5120 | 64.0 → 40.0 MB | 128 | 128 → 80 | 52.4M → 28.6M | 2.370 | 1.162 | **2.039×** |
| skewed | 4 | 66% | 8192 → 2816 | 64.0 → 22.0 MB | 128 | 128 → 44 | 52.4M → 14.3M | 1.974 | 0.617 | **3.197×** |
| uniform | 8 | 0% | 16384 → 16384 | 128.0 → 128.0 MB | 256 | 256 → 256 | 104.9M → 104.9M | 3.846 | 3.888 | **0.989×** |
| mixed | 8 | 48% | 16384 → 8576 | 128.0 → 67.0 MB | 256 | 256 → 134 | 104.9M → 45.5M | 3.826 | 1.771 | **2.160×** |
| skewed | 8 | 77% | 16384 → 3840 | 128.0 → 30.0 MB | 256 | 256 → 60 | 104.9M → 15.9M | 3.839 | 0.692 | **5.550×** |
| uniform | 16 | 0% | 32768 → 32768 | 256.0 → 256.0 MB | 512 | 512 → 512 | 209.7M → 209.7M | 7.600 | 7.671 | **0.991×** |
| mixed | 16 | 35% | 32768 → 21376 | 256.0 → 167.0 MB | 512 | 512 → 334 | 209.7M → 120.1M | 7.626 | 4.582 | **1.664×** |
| skewed | 16 | 82% | 32768 → 5888 | 256.0 → 46.0 MB | 512 | 512 → 92 | 209.7M → 19.0M | 7.611 | 0.897 | **8.485×** |

`uniform` = every sequence 2048. `mixed` = lengths sampled from multiples of
128 up to 2048. `skewed` = one 2048 and the rest 256.

**P1 confirmed exactly.** Three uniform rows, three times 0.99×. Packing a
batch that has no padding costs about 1% and buys nothing — the cost of
carrying `cu_seqlens` and a `VARLEN` branch, and no more. That row is the
control, and the fact that it is flat across `B=4,8,16` is what makes the other
rows readable.

**P2 confirmed.** The `grid` column is identical packed and padded on every
row; the benchmark asserts it rather than printing it hopefully. Live columns
are what move.

---

## Where P3 went wrong: the gain is superlinear in the tokens saved

If time tracked token count, `gain` would be `1/(1 - waste)`. It is not:

| row | tokens saved | predicted gain `1/(1-waste)` | measured gain |
|---|---|---|---|
| mixed 16 | 1.53× | 1.53× | **1.66×** |
| skewed 8 | 4.27× | 4.35× | **5.55×** |
| skewed 16 | 5.57× | 5.56× | **8.49×** |

The `pairs` column explains it. On `skewed 16`, tokens fall **5.57×** but
scored query-key pairs fall **11.0×**. Time lands between the two, at 8.49×.

**A short sequence is cheaper per token as well as fewer tokens.**

At `block_size=128`, a 256-token sequence contains **two blocks**. `top_k=8`
cannot select eight of two, so it clamps. Each of that sequence's queries
scores ~2 blocks where a padded query scores ~6.25 — below `top_k=8` already,
because causality trims the early queries.

Checked rather than assumed, on `skewed 16` (lengths `[2048] + [256]x15`):

```
long sequence    2048 tok x 8 heads x 6.25 blocks  = 102,400
short sequences  3840 tok x 8 heads x 1.5  blocks  =  46,080
                                            total  = 148,480 entries
measured                                              148,437
```

0.03% apart, so the model of where the work goes is right.

> Padding does not merely add tokens. It manufactures blocks for those tokens
> to select, and MoBA faithfully attends to every one of them.

This is a *sparse*-attention-specific effect and it does not exist in dense
attention, where a padded token costs what its position costs and nothing more.
It is the strongest argument for varlen in this project: on a ragged batch the
padding tax is paid twice, once in tokens and once in `top_k`.

---

## Against a packed dense baseline

The baseline is `sdpa_dense.forward_packed` — a jagged nested tensor, which
torch runs on the fused variable-length FlashAttention kernel. Verified rather
than assumed: `MATH` refuses this exact call ("Nested tensors for query / key
are not supported when `is_causal=True`"), so a padded fallback cannot be what
is being timed. Comparing against a *padded* dense baseline would have credited
us with padding the baseline wasted.

Its K and V are expanded to `H_q` outside the timed region: nested SDPA has no
`enable_gqa` and raises "This path is currently not implemented for jagged
layout NT" when the head counts differ.

**These dense numbers are from the head-major era and are now conservative in
our favour.** Since the move to token-major (`docs/layout-measurements.md`),
building the jagged tensor no longer needs a transpose-and-copy, and the
baseline got 7-16% faster. The `gain` column -- our kernel against our own
kernel -- is unaffected and remains the trustworthy one.

| kind | B | dense packed (ms) | moba packed (ms) | ratio |
|---|---|---|---|---|
| uniform | 4 | 5.946 | 2.433 | 2.44× |
| mixed | 4 | 5.429 | 1.162 | 4.67× |
| skewed | 4 | 5.576 | 0.617 | 9.03× |
| uniform | 8 | 6.357 | 3.888 | 1.64× |
| mixed | 8 | 5.749 | 1.771 | 3.25× |
| skewed | 8 | 5.572 | 0.692 | 8.06× |
| uniform | 16 | 9.478 | 7.671 | 1.24× |
| mixed | 16 | 6.694 | 4.582 | 1.46× |
| skewed | 16 | 5.564 | 0.897 | 6.20× |

**Read this column with suspicion.** The baseline barely responds to the work
it is given: `skewed 16` has **13x less** dense arithmetic than `uniform 16`
(`2048² + 15·256²` against `16·2048²`) and runs only 1.7× faster. A dense
kernel that cannot scale down is launch- and occupancy-bound, not
compute-bound, and on 20 SMs at these sizes that is entirely believable.

So a good part of the 6–9× at high waste is the baseline failing to shrink,
not us. **The trustworthy column is `gain`** — our own kernel against our own
kernel, same code path, same tuning, only the representation differing. That is
the number this document is about.

The dense comparison is also the wrong shape of question at `S=2048`. Per
`CLAUDE.md`, the sparsity MoBA exists to exploit comes from `num_blocks` being
large, which comes from long context. At 2048 there are 16 blocks and `top_k=8`
selects half of them. The real head-to-head needs rented sm_80/90.

---

## What this does not measure

* **Padded index memory.** `row_indices` is `[B, H_kv, G·max_seqlen·top_k]`
  and its padding waste is the same raggedness ratio, applied to metadata
  rather than data. It is small next to `part_acc` — the varlen plan's risk
  section estimates ~16 MB at `B=32` — but it is not zero and it is not in the
  table above.
* **`pack` itself.** It runs once per batch at data-prep time and is excluded,
  as `unpad_input` would be in a FlashAttention benchmark. It is a strided
  gather in our layout; see `docs/deferred-decisions.md`, DD-1.
* **The backward.** Same structure and the same `cu_seqlens`, unmeasured here.
* **Anything about absolute speed.** 4 GB and 20 SMs. This box is for
  correctness and for ratios where both sides run on it.

## See also

* `docs/learning/04-packed-sequences-and-where-the-work-goes.md` — what packing
  is and why blocks must be cut per sequence.
* `docs/gqa-measurements.md` — the grid-starvation result that motivated P2,
  and the WDDM allocator lesson that this benchmark's preallocation obeys.
* `docs/superpowers/plans/2026-08-24-varlen-support.md` — the design decisions
  the numbers here are testing.
