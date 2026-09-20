# H100 80GB HBM3 — sm_90

```
NVIDIA H100 80GB HBM3, 81559 MiB, compute capability 9.0, driver 550.163.01
132 SMs, 228 KiB shared memory per SM
rented box, 2026-09-20, ~87 min, ~$4.35, deleted after
```

Recorded from `nvidia-smi` into `logs/device.txt`. The board name `H100 80GB
HBM3` is what `bench/roofline.py`'s `SPEC_BY_NAME` matches, so the roofline
resolves the SXM datasheet (989.4 TFLOP/s, 3352 GB/s) without asking.

This is the first card past the 40 GB A100, and the first time several shapes
could run at all.

## What this session produced

| finding | where |
|---|---|
| headline: 3.82x FlashMoBA / 4.96x SDPA fwd at 65K, 4.23x / 7.21x at 131K | `data/moba-fwd.csv` |
| backward 2.57x / 3.08x | `data/moba-bwd.csv` |
| gap narrowed vs A100 (4.62x): bytes stayed 4.30x, efficiency 1.95x -> 1.64x | `data/residual.json`, `logs/residual-cd.log` |
| gate becomes the bottleneck (87% of our forward at 512K; attention-only 21.5x) | `data/residual-a-long.json`, `logs/residual-a-long.log` |
| headline extended to 262144/524288 (separate run) | `data/moba-fwd-long.csv`, `data/moba-bwd-long.csv` |
| their window policy 1.57x off at S=8192, still correct at 65536 | `logs/residual-cd.log` |
| **int32 overflow in the partials index**, found, fixed, re-verified | `logs/int32-overflow*.log` |
| `tune_table` returned a backward config 1.29x worse than DEFAULT | `logs/tune-table.log`, `logs/ab-backward-config.log` |
| measured/modeled byte ratio 1.27x (ours), occupancy column untrustworthy (152% for memset) | `logs/profile-roofline.log` |

## What the tuning table cost

`tune_table` ran with clocks unpinnable in this container, and its backward
pick was 1.29x worse than `DEFAULT` — caught only because the bootstrap's
pre-table `compare` became an untuned control. The corrected values are in
`src/moba/kernels/tuned_configs.py` (`c20668d`). Lesson for the next box: when
clocks are not lockable, treat `tune_table` output as a hypothesis and A/B the
winner against `DEFAULT` before committing it.

## The int32 bug, in one paragraph

`part_acc` is `[B, H, S, top_k, D]`, the largest object this design allocates,
and its addressing wrapped at 2^31 elements. B=4 H=32 S=65536 D=128 is 8.59e9
elements (16 GiB, fits fine) and raised an illegal memory access. Three shapes
reaching 2^32 by different routes all failed; two shapes sitting on 2^31 both
passed — a 32-bit index, not capacity. Fixed by widening the address base to
int64 (`04236d3`); full details in `docs/deferred-decisions.md` DD-2a.

## Which doc cites what

* `data/moba-fwd.csv`, `data/moba-bwd.csv` — the H100 headline
* `data/residual.json` — panels C and D only; panel C's roof is measured
  per run (2298 GiB/s).
* `data/residual-b.json` — panel B, complete: all nine shapes including
  B=4 H=32 S=65536, the 16 GiB shape that crashed before the int64 fix
  (`04236d3`). Re-run after the fix, it now measures 3.50x.
* Panel A never made it into a JSON; it lives in `logs/residual-ab.log`.
* `figures/roofline.png` — the first roofline with a real Hopper roof
* `logs/int32-overflow.log` / `-after-fix.log` — DD-2a
* `logs/ab-backward-config.log`, `logs/ab-forward-group.log` — the A/B that corrected the table

`logs/bootstrap.log` is present on disk but not tracked (build transcript,
like the A100's). `logs/post-fix-tests.log` is the 486-pass suite after the
int64 fix.

## Known bad / open

* `naive_torch` and `moonshot_naive` OOM at S>=32768 (dense O(S^2)); `NaN` by
  design rather than a crash.
* CUPTI's occupancy column is not trustworthy on sm_90 — it reports >100% for
  a memset. The tensor-core utilisation column (ours 42.7%, theirs 15.5%) is
  the sound number.
