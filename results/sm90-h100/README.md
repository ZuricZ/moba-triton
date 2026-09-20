# H100 — sm_90

**Empty. Nothing has been run on an H100 yet.**

This directory exists so the next session has somewhere to write, and so the
question "which card produced this" has an answer before the first file lands
rather than after.

## Before renting one

Four things are pinned to sm_80 and need a value for sm_90. Three are data,
one is a datasheet row; none is kernel code.

| what | where | if skipped |
|---|---|---|
| no `sm_90` tuned entries | `src/moba/kernels/tuned_configs.py` | every launch silently takes `DEFAULT` — correct but untuned, and the A100 headline *was* measured with the table. Run `bench/tune_table.py`, 8 brackets × 2 passes, **before** any headline number. |
| no `sm_90` datasheet | `bench/roofline.py` `SPEC` | prints no audit rather than a guessed roof; the figure loses its roof line. Needs the SXM/PCIe split — 989 TFLOP/s ÷ 3352 GB/s versus 756 ÷ 2039. |
| measured bandwidth roof | `bench/residual.py` | the "% of roof" column in panel C is A100's 1239 GiB/s. Measured, not datasheet, so it must be re-measured here. |
| `sm8x` window cap | `bench/tile_census.py`, `bench/roofline.py` | already correct for sm_90 (no cap). Docstrings say "the roofline is for an A100". |

## The two questions worth the rent

1. **Their window policy, directly.** `docs/why-we-beat-flashmoba.md` §3
   eliminates "it is tuned for an H100" *indirectly*, by narrowing their
   window on an A100. On an H100 it becomes a direct measurement on the
   hardware their paper reports. It is the only alibi in that document a
   skeptic can currently push on.
2. **Whether occupancy diverges.** On the A100 both kernels measured 2
   blocks/SM, 8 warps, 12%. Shared memory grows 164 → 228 KB but the register
   file does **not** — 65,536 either way. At 230 registers per thread we may
   already be register-limited rather than SMEM-limited, in which case the
   extra shared memory buys nothing. One `bench/profile.py` run settles it.

## Suggested order

```
pytest                                    # confirm 500 pass on sm_90
python -m bench.tune_table                # BEFORE any headline number
python -m bench.compare                   # writes into data/ here, not the A100's
python -m bench.residual                  # re-measure the roof first
python -m bench.profile                   # the occupancy question
```
