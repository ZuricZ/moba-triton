# Token-major layout — measurements

The move from head-major `[B, H, S, D]` to token-major `[B, S, H, D]`,
resolving `docs/deferred-decisions.md` DD-1. RTX 3050 Ti Laptop (sm_86,
20 SMs), bf16 unless stated, `uv run python bench/…`.

---

## The prediction, stated before the numbers

**The change should be neutral.** Nothing was generalized: the addressing is
still derived from `constexpr` values and still folds at compile time, the same
bytes are read in the same order, and the same kernels run with the same tile
shapes, warp counts and shared-memory budgets. Only two constants moved.

The one place a real *improvement* was expected is `pack`, which went from `H`
short transactions per token to one long one.

The one place a real *regression* was plausible is the K/V load. Our loop order
is block-major, so a program owns `BLOCK_SIZE` consecutive keys, and head-major
gives it one contiguous run where token-major gives it `BLOCK_SIZE` runs of `D`
spaced `H*D` apart. `bench/layout_probe.py` was written to answer exactly that
question before any kernel was touched.

Not testable here: DD-1's third revisit trigger is layout conversion at the
FlashMoBA benchmark boundary, which needs rented sm_80/90.

---

## 1. The probe: does the stride itself cost anything?

`bench/layout_probe.py` loads K and V tiles and does nothing else, holding
bytes, block size, grid, warps and dtype identical between the two arms.

| B | H | S | D | BLK | head-maj | token-maj | ratio |
|---|---|---|---|---|---|---|---|
| 1 | 8 | 8192 | 64 | 128 | 0.1444 | 0.1454 | 1.007× |
| 1 | 8 | 8192 | 128 | 128 | 0.2437 | 0.2396 | 0.983× |
| 1 | 2 | 8192 | 64 | 128 | 0.0420 | 0.0430 | 1.024× |
| 4 | 8 | 2048 | 64 | 128 | 0.1270 | 0.1290 | 1.016× |
| 1 | 8 | 8192 | 64 | 64 | 0.1178 | 0.1183 | 1.004× |
| 1 | 8 | 8192 | 64 | 256 | 0.1403 | 0.1393 | 0.993× |

**0.983×–1.024×, no systematic direction** — token-major wins two and loses
four here, and won three of six on the first run. Repeating the probe moved
individual rows by up to 2.5%, which is the resolution limit and is larger than
any effect present.

**Why the structural argument does not survive contact with the grid.**
Head-major's advantage is meant to be fewer, longer transactions. But every
head's program is launched and resident together, so the bytes token-major
"skips" are being read at that same instant by the sibling program for the next
head. The memory system sees the same contiguous range either way; only the
identity of the issuing program changed.

At `D=64` bf16 a key's row is exactly 128 bytes — one full cache line — so
neither layout over-fetches. `moba.device.check_gathered_head_dim` already
rejects the narrow-row case where that would stop being true.

---

## 2. Same-session A/B on the real kernels

`bench/stages.py`, `master` and `token-major-layout` run back to back in one
sitting. Times are the whole kernel (gate + attention + combine), microseconds.

| S | part_acc | head-major | token-major | Δ |
|---|---|---|---|---|
| 1024 | float32 | 332 | 336 | +1.2% |
| 1024 | bfloat16 | 257 | 260 | +1.2% |
| 2048 | float32 | 710 | 705 | **−0.7%** |
| 2048 | bfloat16 | 540 | 533 | **−1.3%** |
| 4096 | float32 | 1473 | 1465 | **−0.5%** |
| 4096 | bfloat16 | 1105 | 1090 | **−1.4%** |
| 8192 | float32 | 3022 | 3007 | **−0.5%** |
| 8192 | bfloat16 | 2243 | 2223 | **−0.9%** |

Token-major is equal or faster on six of eight, and the two rows where it loses
are the smallest, where launch overhead dominates. Per stage at S=8192 bf16:
gate 425→424 µs, attention 1777→1757 µs, combine 466→466 µs.

`bench/varlen_sweep.py`, same session, MoBA packed times in milliseconds:

| kind | B | head-major | token-major | Δ |
|---|---|---|---|---|
| uniform | 4 | 2.072 | 2.085 | +0.6% |
| mixed | 4 | 1.194 | 1.206 | +1.0% |
| skewed | 4 | 0.629 | 0.634 | +0.8% |
| uniform | 8 | 4.014 | 4.019 | +0.1% |
| mixed | 8 | 1.828 | 1.842 | +0.8% |
| skewed | 8 | 0.700 | 0.709 | +1.3% |
| uniform | 16 | 7.937 | 7.982 | +0.6% |
| mixed | 16 | 4.734 | 4.747 | +0.3% |
| skewed | 16 | 0.921 | 0.925 | +0.4% |

> **Neutral. −1.4% to +1.3%, straddling zero, with the two benchmarks
> disagreeing about the sign.** The prediction held.

The varlen gains themselves are unchanged, which is the more important check:
0.99× uniform and 8.57× at 82% waste on head-major against 0.98× and 8.53× on
token-major. `docs/varlen-measurements.md` stands as written.

### A measurement mistake worth recording

The first comparison used the branch's numbers against `docs/varlen-measurements.md`,
recorded earlier the same day, and showed a consistent **+3% regression across
every row**. It was not real. Re-running `master` in the same session produced
numbers ~3% slower than its own morning figures (uniform-8 padded: 3.846 →
3.963 ms). The machine had drifted; the layout had not.

This is the second time this project has read a machine artifact as a kernel
effect — the first was the WDDM allocator inflating the GQA sweep
(`docs/gqa-measurements.md`). Both produced a plausible number with a
convincing story attached. **A cross-session comparison is not an A/B.**

---

## 3. What actually got faster: the baseline, and packing

`sdpa_dense.forward_packed` builds a jagged nested tensor, which wants exactly
`[T, H, D]`. Head-major needed a transpose-and-copy of Q, K and V to get there;
token-major does not.

| kind | B | dense packed, head-major | token-major | Δ |
|---|---|---|---|---|
| uniform | 4 | 6.600 | 5.516 | **−16.4%** |
| uniform | 8 | 6.687 | 5.853 | **−12.5%** |
| uniform | 16 | 10.409 | 9.670 | **−7.1%** |

So the **"vs dense" column in `docs/varlen-measurements.md` drops** — not
because we slowed down, but because the baseline stopped paying for a copy it
never needed. That makes every ratio in that document conservative rather than
flattering, which is the direction to prefer.

`pack` itself is now FlashAttention's row gather rather than a strided one plus
a permute — DD-1's second argument, delivered. It runs once per batch at
data-prep time and is outside every timed region, so there is no number here;
what changed is the operation, from `H` short transactions per token to one
`H*D`-byte run.

`_refs.to_varlen`, which the vendored backends use, went from
`permute(0, 2, 1, 3).reshape(...).contiguous()` — a full copy — to a bare
`reshape`.

---

## 4. What did not change

| | |
|---|---|
| the loop order | **same** — block-major, KV-major inversion untouched |
| tile shapes, warp counts, chunk heights | **same** — `GROUP`, `BWD_GROUP`, shared-memory budget all unchanged |
| the CSC index | **same** — every field, still head-major |
| `part_m`, `part_l`, `part_acc`, `dq_part`, `lse`, `delta` | **same** — ours, still head-major |
| block representatives and selection masks | **same** — ours |
| numerical results | **same** — the oracle was checked bitwise against its head-major self, including GQA and packed |
| test count | **same** — 429 passed, 7 skipped, before and after |

The addressing is still `constexpr`-derived. **No stride became a runtime
value**, which is the cost the rejected "pass strides in" design would have
carried and this one does not.

---

## 5. What this does not measure

* **Rented sm_80/90.** More SMs and a much larger L2 change the concurrency
  argument in §1, which is the one hardware-specific claim here. Re-run
  `bench/layout_probe.py` there before trusting it.
* **The FlashMoBA head-to-head.** DD-1's third revisit trigger was layout
  conversion at that boundary. It is now zero by construction — FlashMoBA is
  token-major and so are we — but that is an argument, not a measurement.
* **Any absolute number.** 4 GB and 20 SMs. Ratios where both sides ran here
  are the only trustworthy output.

## See also

* `docs/deferred-decisions.md` — DD-1, resolved, including the two places its
  own reasoning was wrong.
* `docs/varlen-measurements.md` — the varlen gains these numbers confirm are
  unchanged.
* `docs/superpowers/plans/2026-09-03-token-major-layout.md` — the plan.
