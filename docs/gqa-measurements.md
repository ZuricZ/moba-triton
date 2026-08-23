# Grouped-query attention — what it actually cost on this card

**Summary: GQA is correct, free in resources, and on this 20-SM laptop card it
is a small loss rather than a small win.** The prediction written into the plan
before measuring was "the gain will be small, possibly within noise." The
magnitude was right and the sign was wrong, for a reason the prediction did not
consider: grouped-query attention **shrinks the launch grid**, and this card
does not have enough work to spare.

Prediction on record, from
`docs/superpowers/plans/2026-08-23-gqa-support.md`, Task 6:

> GQA divides the 8.4 MB of K/V traffic. It does not touch the 63.4 MB of
> `part_acc`. So the end-to-end gain on this card will be small, possibly
> within noise.

## Method

`H_q` pinned at 8 in every row, so the arithmetic — the number of query·key
pairs scored — is identical throughout. Only the count of distinct K/V heads
varies, which is exactly the memory term.

The `expanded` column is the control: the same shared K/V materialized back out
to 8 heads and run through the ordinary multi-head path. Identical mathematics,
identical routing, only the layout differs. `gain = expanded / narrow`, so
**above 1.0 means GQA won**.

`block_size=128, top_k=8, head_dim=64, bf16`, batch 1, median of 200 reps after
50 warmup.

### The first measurement was garbage, and why

The first attempt timed `moba_forward` directly and produced numbers where the
*expanded* runs were faster than the identical-math narrow ones, non-monotonic
in sequence length. `moba_forward` allocates `part_acc` on every call — 16 to
64 MB at these shapes — and `attend_partials`' own docstring records that on
Windows WDDM a `cudaMalloc` goes through the display driver and costs
milliseconds. The allocator was the measurement.

Fixed by preallocating with `empty_partials` and passing `partials=` on every
call. Median rather than mean for the same reason: WDDM produces occasional
multi-millisecond stalls that a mean absorbs into the signal.

## Results

Times in milliseconds, batch 1.

| seq_len | H_kv | group | attn | attn expanded | gain | K/V MB | FLOP/byte |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2048 | 8 | 1 | 0.589 | 0.400 | 0.680× | 4.00 | 98.5 |
| 2048 | 4 | 2 | 0.421 | 0.401 | 0.954× | 2.00 | 104.9 |
| 2048 | 2 | 4 | 0.438 | 0.403 | 0.922× | 1.00 | 108.5 |
| 2048 | 1 | 8 | 0.811 | 0.400 | **0.494×** | 0.50 | 110.3 |
| 4096 | 8 | 1 | 0.839 | 0.839 | 0.999× | 8.00 | 106.5 |
| 4096 | 4 | 2 | 0.846 | 0.836 | 0.988× | 4.00 | 113.1 |
| 4096 | 2 | 4 | 0.909 | 0.843 | 0.927× | 2.00 | 116.7 |
| 4096 | 1 | 8 | 1.210 | 0.841 | 0.695× | 1.00 | 118.6 |
| 8192 | 8 | 1 | 1.782 | 1.793 | 1.006× | 16.00 | 110.2 |
| 8192 | 4 | 2 | 1.776 | 1.802 | **1.015×** | 8.00 | 116.9 |
| 8192 | 2 | 4 | 1.863 | 1.804 | 0.969× | 4.00 | 120.5 |
| 8192 | 1 | 8 | 1.916 | 1.804 | 0.942× | 2.00 | 122.4 |

Two things are visible before any explanation:

* **The penalty shrinks as sequence length grows.** At group 8 it goes
  0.494 → 0.695 → 0.942 across 2048 → 4096 → 8192.
* **Arithmetic intensity rises exactly as designed**, but only from 98.5 to
  122.4 FLOP/byte — **+24%, not the 4x the K/V saving alone would suggest.**

## Why the intensity gain is 24% and not 4x

The predicted ceiling held. K/V is a small share of what this kernel moves:

| Term | Bytes at S=8192, H_kv=8 | Divided by GQA? |
|---|---:|---|
| `part_acc` writes | 64.0 MB | no |
| gathered queries | 60.5 MB | no |
| K and V loads | 16.0 MB | **yes** |

Dividing 16 MB by 8 removes 14 MB from a 140 MB total. That is a 10% traffic
cut, which is the +24% intensity (`total` also shrinks). The design's own
bookkeeping dominates its memory traffic, and GQA does not touch it.

This is not a defect in GQA. It is the same finding `attend_partials` already
records: the KV-major inversion worked so well that its own partial buffers
became the bottleneck.

## Why GQA *lost*: the grid shrinks

The prediction missed this entirely. One program owns
`(KV block, batch, kv head)`. Grouped-query attention divides the kv-head axis,
so **the grid shrinks by the group size** — that is the same fact as "`group`
times fewer K/V loads", seen from the launch side rather than the memory side.

At `S=2048, block_size=128` there are 16 blocks. With `H_kv=1` and batch 1 the
grid is **16 programs on 20 SMs**: four SMs idle, and no second wave to hide
memory latency behind.

Tested directly by adding batch, which multiplies the grid back without
changing anything else:

| batch | grid (GQA) | grid (expanded) | narrow ms | expanded ms | gain |
|---:|---:|---:|---:|---:|---:|
| 1 | 16 | 128 | 1.299 | 0.399 | 0.307× |
| 2 | 32 | 256 | 0.864 | 0.750 | 0.868× |
| 4 | 64 | 512 | 1.647 | 1.482 | 0.900× |
| 8 | 128 | 1024 | 3.098 | 3.020 | 0.975× |
| 16 | 256 | 2048 | 5.969 | 6.084 | **1.019×** |

`S=2048, H_kv=1, group 8`. The penalty is monotone in grid size and gone by
256 programs. **The loss is starvation, not a cost of sharing.**

It also explains the sequence-length trend in the main table: `num_blocks`
grows with `S`, so the grid grows with it, and the penalty fades.

## What this means, and what it does not

**On this card:** use GQA when the grid stays comfortably above the SM count.
`grid = num_blocks * batch * H_kv`; at 20 SMs, roughly 128 programs is where
the penalty disappears. Group 2 and 4 at S≥4096 are within a few percent either
way; group 8 at short context is the case to avoid.

**Off this card, none of the above transfers, and the direction reverses.**
An H100 has 132 SMs — 6.6x the parallelism to fill, so grid starvation bites
*harder* at small shapes and vanishes just as completely at realistic ones
(64K context, `num_blocks` in the hundreds, batch above 1). Meanwhile its ridge
point is ~295 FLOP/byte against this card's 78, so a kernel at 110-122
FLOP/byte is compute-bound here and **memory-bound there**. Every byte GQA
removes is worth ~3.8x more on that machine than on this one.

So the honest statement is: **this measurement establishes that GQA is correct
and costs no resources, and that its benefit is not visible on 20 SMs at batch
1.** Whether the benefit is real is a question this hardware cannot answer.
That is the rented-GPU run.

## What did not change

Confirming decision D-GQA-4 from the plan. Grouped-query attention makes a
column longer, not a tile wider, so:

* No tuning constant was re-derived. `MAX_GROUP`, `BWD_KEY_TILE`, `BWD_GROUP`,
  `BWD_WARPS`, `COMBINE_BLOCK_M` are all untouched.
* Registers and shared memory per program are identical — the inner loop runs
  more iterations at the same tile shape.
* `_moba_combine` and `_moba_dq_reduce` needed no edit at all (D-GQA-5).
* The inner-loop cost is one `[GROUP]` int32 gather and one multiply-add:
  128 bytes against a 16 KiB K/V tile.

## Correctness

379 tests pass, 7 skipped. GQA-specific coverage is `tests/test_gqa.py` (17
tests) plus `tests/test_heads.py` (8), with GQA cases added to `test_gate.py`,
`test_index.py` and `test_index_fused.py`.

Every one is anchored against `expand_kv` — the same code path on materialized
key/value heads — because the failure mode does not crash. A wrong head mapping
produces a well-formed index of the right shape holding plausible numbers, and
the kernel attends to another head's keys.

One test needed the semantics flag flipped to be reachable at all:
`always_select_current=True` means every block is selected by the queries inside
it, so `col_nnz >= block_size` for every column and an *empty* column cannot
exist. Measured zero empty columns across six GQA configurations with the flag
on, one or two with it off. Without that,
`test_unselected_blocks_still_get_a_zero_gradient_under_gqa` was a permanent
skip — green, and testing nothing.

## See also

* `docs/learning/03-gqa-and-the-column-merge.md` — why a column merge is the
  same fact as a K/V saving, with pictures.
* `docs/superpowers/plans/2026-08-23-gqa-support.md` — the design decisions and
  their rejected alternatives.
* `docs/gather-cost-measurements.md` — why gathered-query order does not matter.
