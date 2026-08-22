# Streaming the gate's top-k, measured

**Date:** 2026-08-22
**Device:** RTX 3050 Ti Laptop, sm_86 | bf16 | 2 batch-heads, head_dim 128,
block_size 128, top_k 8
**Source:** `src/moba/kernels/gate.py`; A/B against the one-shot gate at
commit `7975a63`

## Why the one-shot gate degraded

The previous gate scored every block representative in a single
`[BLOCK_M, num_blocks]` tile. Its documented limit was capacity -- a docstring
claimed the representatives stopped fitting in shared memory around 16K. **That
was wrong.** Compiling the kernel and reading the metadata directly:

| num_blocks | N_POW2 | warps | regs/thread | spills | shared mem |
|---:|---:|---:|---:|---:|---:|
| 32 | 32 | 4 | 56 | 0 | 12 KB |
| 128 | 128 | 4 | 154 | 0 | 12 KB |
| 256 | 256 | 8 | 155 | 0 | 20 KB |
| 512 | 512 | 8 | 187 | 0 | 20 KB |

512 blocks compiles, uses 20 KB of the 99 KB budget, and does not spill. The
guard predicted 131,072 bytes and rejected it. It modelled the representatives
as parked in shared memory all at once; Triton feeds `tl.dot` operands through
shared memory in slices and holds the rest in registers.

What actually degraded was **occupancy**, and the warp column shows it. Register
allocation is a per-SM budget of 65,536, claimed as `regs/thread x threads`:

- 128 blocks: 154 x 128 threads (4 warps) = 19,712 -> **3 programs/SM**
- 256 blocks: 155 x 256 threads (8 warps) = 39,680 -> **1 program/SM**

Per-thread use moved by one register. The *program* got wider, because
`choose_gate_warps` scaled warps with the tile to avoid spilling. It succeeded
-- zero spills everywhere -- by paying in occupancy instead, trading a loud
failure for a silent one. Doubling num_blocks past 128 therefore cost 2x in
arithmetic and 3x in occupancy, and the measured jump was 5.75x.

The arithmetic itself is not the problem: `TAKE = past_blocks_selected` is
constant in num_blocks (7 at top_k 8), so ranking is linear per query.

## The streaming form

Representatives are streamed in `GATE_BLOCK_N`-wide chunks past a resident
query tile, with a running top-k of `K_POW2` entries in registers. The working
tile no longer grows with the sequence, so the warp count is fixed at 4 and
occupancy no longer depends on context length.

Two things beyond the footprint fix:

* **The merge keeps no sort order.** NSA's `parallel_nsa_kernel_topk` merges
  each chunk into a sorted list with a bitonic network. The only thing this
  merge asks of the list is its *minimum* -- the bar a candidate must clear --
  which is one reduction over 8 entries against `log^2` compare-exchange stages
  for the sort. NSA needs the sort because its candidate list is long; MoBA's
  is `top_k`.
* **The loop stops at the causal horizon.** Its trip count is a runtime value,
  so a query tile scores only up to the last block any of its queries can see,
  instead of scoring everything and masking the future afterwards. The one-shot
  form cannot express this at all: it has no loop to shorten, and a tile at
  position 0 did exactly as much work as one at the end.

## Result

Best-of-5 `do_bench`, both kernels in one process on identical inputs, at the
tuned `BLOCK_M=32, BLOCK_N=32, 4 warps`:

| seq_len | num_blocks | one-shot | streaming | speedup |
|---:|---:|---:|---:|---:|
| 4096 | 32 | 54 us | 41 us | 1.34x |
| 8192 | 64 | 117 us | 91 us | 1.28x |
| 16384 | 128 | 385 us | 284 us | 1.36x |
| 32768 | 256 | 2320 us | 944 us | **2.46x** |
| 65536 | 512 | *rejected by the guard* | 3470 us | -- |

The gain grows with context, which is the regime MoBA exists to serve, and the
ceiling is gone.

## Tuning

`BLOCK_N` is a real trade-off, not a free parameter. Useful work per chunk is
`TAKE * BLOCK_N`; merge bookkeeping is `TAKE * 2 * K_POW2` whatever the width.
Overhead is therefore `2 * K_POW2 / BLOCK_N`, favouring wide chunks, while the
causal horizon is only tracked to `BLOCK_N` granularity, favouring narrow ones.

| seq_len | N=16 | N=32 | N=64 |
|---:|---:|---:|---:|
| 4096 | 46 us | **41 us** | 62 us |
| 8192 | 112 us | **91 us** | 128 us |
| 16384 | 346 us | **284 us** | 364 us |
| 32768 | 1278 us | **944 us** | 991 us |
| 65536 | 4947 us | 3470 us | **3436 us** |

`BLOCK_M` went 16 -> 32, worth 1.3-1.9x: halving the program count halves how
many times the representatives are re-read. The one-shot gate could not afford
this, because its tile was already `num_blocks` wide.

An early sweep of these same points disagreed with itself between adjacent
sequence lengths. Single `do_bench` calls on this laptop card vary enough with
clock state to invert a 20% difference; every table here is best-of-5.

## Two bugs this introduced, both about the running list

Neither was reachable in the one-shot form.

**Eviction had no tie-break.** Admission uses strict `>`, so an equal-scoring
later block cannot displace an earlier one -- chunks are visited in ascending
block order, so `tie_break_low_index` falls out for free. But when a stronger
block arrives it evicts *the minimum*, and if two entries tie for minimum the
first version chose between them by slot position, which is just the order they
happened to be written. Measured at block_size 32, seq_len 16384: 764
disagreements with the oracle out of 8.4M decisions, every single one an exact
bf16 score tie, with the kernel keeping the higher block index. The one-shot
gate only ever set bits in a `chosen` mask and never took one back out --
eviction is a concept streaming introduces, and the tie rule had to reach it.
Fixed by evicting the largest block index among tied minima.

**Surplus slots acted as extra capacity.** `K_POW2` is rounded up to a power of
two for `tl.arange`, so top_k 12 gives 16 slots for 12 entries. Empty slots sit
at -inf, which the merge reads as "fill me", so across two chunks the list
admitted 16 blocks against a budget of 12. Invisible at top_k 8, where capacity
is exactly 8, and invisible whenever `num_blocks <= BLOCK_N`, since one chunk
can only admit `TAKE` entries. Only a backend test at seq_len 8192 caught it.
Fixed by retiring surplus slots at +inf: never the minimum, so never a victim,
and their sentinel index emits as padding.
`tests/test_gate.py` now covers both conditions directly.
