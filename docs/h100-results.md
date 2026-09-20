# H100-80GB — the A100 story re-measured, plus a bug the A100 could not reach

**Date:** 2026-09-20
**Device:** NVIDIA H100 80GB HBM3, sm_90, 132 SMs, 228 KiB shared memory/SM,
driver 550.163.01
**Data:** `results/sm90-h100/` (CSVs in `data/`, session logs in `logs/`,
`README.md` records which document cites which file)
**Companion:** `docs/a100-results.md` for the same measurements on the A100,
and `docs/why-we-beat-flashmoba.md` for the explanation this run re-tested.

This was the first run on a card past the 40 GB A100. It had two jobs — re-run
the A100's numbers on FlashMoBA's own hardware, and settle the two questions
the A100 could only infer. It did both, and it also found a bug the A100 was
physically too small to reach.

## The headline

| | S | ours | flash_moba | sdpa_dense | vs FM | vs SDPA |
|---|---:|---:|---:|---:|---:|---:|
| fwd | 65,536 | 1.492 ms | 5.694 | 7.405 | 3.82x | 4.96x |
| fwd | 131,072 | 3.942 | 16.691 | 28.435 | 4.23x | 7.21x |
| bwd | 65,536 | 2.658 | 6.822 | 26.544 | 2.57x | 9.99x |
| bwd | 131,072 | 6.181 | 19.022 | 103.933 | 3.08x | 16.82x |

`131072` is new ground — the 40 GB A100 could not reach it, and the ratio
*grows* with length (3.82x -> 4.23x against FlashMoBA), which is what the
traffic argument predicts and what a fixed-cost explanation would not.

The A100 measured 4.62x at S=65536. Hopper narrows that to 3.82x. The next
section is why.

## Why the gap narrowed: the efficiency half closed, the byte half didn't

The two-factor decomposition reproduces almost exactly, and it shows the
narrowing is *entirely* one factor.

| S | bytes ratio | efficiency ratio | predicted | measured (attention-only) |
|---:|---:|---:|---:|---:|
| 8,192 | 3.14x | ~1.3x | 4.05x | 4.05x |
| 16,384 | 3.19x | ~1.5x | 4.78x | 4.78x |
| 65,536 | 4.30x | 1.64x | 7.05x | 7.05x |

The **byte ratio is 4.30x at 65K, identical to the A100** — their extra
traffic comes from what a program owns, and that is unchanged across cards.
What moved is the **efficiency ratio: 1.95x on the A100 to 1.64x here**. Panel
C of `bench/residual.py` puts numbers on it:

```
roof: 2298 GiB/s, measured by copy on this card (the A100's was 1239)
  65536   ours     864.5 MiB   0.559 ms   1508.9 GiB/s   65.7% of roof
  65536 theirs    3718.0 MiB   3.944 ms    920.6 GiB/s   40.1%
```

On the A100 the same column read ours at 61% of roof against theirs at 31%.
Hopper lets their kernel reach further toward the roof — 40% against 31% —
which is the entire narrowing. Ours barely moved (61% -> 66%).

## The bug: a 32-bit index the A100 could not reach

The headline number was never the interesting part of this run. The interesting
part was the shape `B=4 H=32 S=65536 D=128` — 16 GiB of scratch, four times the
A100's capacity, so it had never run *anywhere*. On the 80 GB card it fit with
room to spare and still raised an illegal memory access.

The partial buffer `part_acc` is `[B, H, S, top_k, D]`, and Triton's offsets
are int32 unless something is widened. Its element count crossed 2^31, the
addressing wrapped, and the kernel wrote out of bounds. The boundary follows
the *product*, which is what distinguishes a 32-bit index from a capacity limit:

| `B*H*S*top_k*D` | reached by | before the fix |
|---:|---|---|
| 2^31 | B=1 S=65536, or B=4 S=16384 | ran |
| 2^32 | B=2 S=65536, or B=4 S=32768, or B=8 S=16384 | all three: illegal access |
| 2^33 | B=4 S=65536 | illegal access |

Three routes to the same count fail identically; two shapes sitting on 2^31
pass. An 8 GiB buffer failing on an 80 GB card rules out exhaustion. Fixed by
widening the base of each address chain to int64 — full details, including the
headroom arithmetic, in `docs/deferred-decisions.md` DD-2a. FlashMoBA never
meets it: FA-2 declares `using index_t = int64_t` once and inherits it
everywhere, and their window design allocates no `top_k`-deep buffer.

## The tuning table was wrong, and looked right

`bench/tune_table.py` ran with clocks unpinnable in this container, and its
backward pick (`num_warps=8, num_stages=2`) was **1.29x worse than `DEFAULT`**.
It produced a table that had every property of a correct one — the same answer
at all four brackets, no outliers — because a card that is consistently hot
biases every bracket identically. The uniformity was the bias.

It was caught by an accident worth keeping: the bootstrap runs `compare` before
the table exists, and that throwaway CSV became an untuned control. Our
backward measured 2.68 ms untuned against 3.40 ms "tuned", five times this
kernel's known spread, while FlashMoBA moved 1.5% over the same interval.

A direct A/B (one variable, minimum of three, cool-down between) gave the
corrected table, which is in `src/moba/kernels/tuned_configs.py` (`c20668d`).
Lesson for the next box: **when clocks are not lockable, treat `tune_table`
output as a hypothesis and A/B the winner against `DEFAULT`.**

## The window policy re-test: §3 holds where it matters, narrows elsewhere

`docs/why-we-beat-flashmoba.md` §3 eliminates "it is tuned for an H100" by
narrowing their window on an A100. This run asked the question directly on
their own hardware (`bench/residual.py` panel D):

```
      S  fastest window   their policy   penalty
   8192     128               512         1.57x
  16384     256               768         1.27x
  65536    1024              1024         1.00x  <- still correct here
```

Their policy is up to 1.57x off at short lengths on Hopper, but it is still the
best window at 65K — the length the headline lives at. The elimination survives
where the claim matters and the general statement does not. The §3 text now
carries this caveat.

## What this run could not settle

* **Occupancy is not settled on this card.** The A100's 12% was a *static*
  bound — kernel registers and shared memory divided into the SM's 64 warps.
  This run's `bench.profile` used a different, CUPTI-derived column that
  reports 152% for a memset on sm_90, which is impossible, so the 22.9% it
  prints for `moba_fwd_kernel` is not trustworthy and not comparable to the
  A100's 12% (they are not the same measurement). The sound number from this
  run is tensor-core utilisation — ours 42.7% against theirs 15.5%, the "full
  tiles" argument in a counter. `ncu`, which would settle it, remains refused
  on every machine this project has touched.
* **The byte counts are still modelled.** Measured DRAM traffic for ours is
  1.27x the model (`results/sm90-h100/logs/profile-roofline.log`). For
  FlashMoBA the profiler reports 50x the model, which cannot be real: 17.2 GiB
  in 0.549 ms is 29.8 TiB/s, nine times the H100's 3.35 TiB/s HBM peak, so the
  number is instrumentation, not traffic. `docs/why-we-beat-flashmoba.md` §6
  already flags the byte split as model-dependent; this is the first measurement
  of how far off it is, and it only affects ours (1.27x).

## Caveats

* Clocks were not pinnable in this container, so every timing uses the
  cool/warm protocol and run-to-run spread is the A100's 5.7% class, not
  tighter.
* `triton_moba` is absent here (no flash-attn built), so the six-backend table
  is missing that row; it crashed at 65K on the A100 anyway.
* `naive_torch` and `moonshot_naive` OOM at S >= 32768 by design — dense
  O(S^2) attention on 80 GB.
