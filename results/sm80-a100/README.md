# A100-PCIE-40GB — sm_80

```
NVIDIA A100-PCIE-40GB, 40960 MiB, compute capability 8.0, driver 595.58.03
108 SMs, 164 KiB shared memory per SM
rented box, 2026-09-18 .. 2026-09-20, deleted after
```

Recorded from `nvidia-smi` into `logs/device.txt` at the start of the session.
Every number in this directory came from that card; the instance no longer
exists, so nothing here can be re-measured without renting another.

**Why this file exists.** Eight of these files carried no device string
anywhere in their contents — `moba-fwd.csv` and `moba-bwd.csv`,
`fa263-*.csv`, `fa283-*.csv`, `tabled-*.csv`. Before the card became a
directory they were identifiable only by a filename prefix, and two of them
had no prefix either. `moba-fwd.csv` is the file the headline table is read
from.

## What cites what

| file | cited by |
|---|---|
| `data/moba-fwd.csv` | `docs/why-we-beat-flashmoba.md` §1 — the headline table |
| `data/residual.json` | `docs/why-we-beat-flashmoba.md` §3–4 — the four suspects, the two factors |
| `data/block-sweep-budget.csv`, `data/block-sweep-fixed-top-k.csv` | `docs/block-size-results.md` — the 29.52× vs 2.05× comparison |
| `data/fa283-*.csv`, `data/fa263-*.csv` | `docs/a100-results.md` §1 — per-pass, two flash-attn versions |
| `logs/occ_fm.log` | `docs/why-we-beat-flashmoba.md` §3, `docs/a100-results.md` — the occupancy retraction |
| `logs/final-batch.log` | `docs/why-we-beat-flashmoba.md` — the final measurement batch |
| `logs/dip-stages.log` | `docs/why-we-beat-flashmoba.md` §2 — 247,808 pairs, cross-checking `tile_census` |
| `logs/dip-overhead.log` | `docs/a100-results.md:63` — the small-S dip |
| `logs/twopass.log`, `logs/pass2.log` | `docs/a100-results.md:9` — raw session logs |
| `logs/roofline.log` | `docs/a100-results.md:339`. It no longer draws `figures/roofline.png`: this log predates the FlashMoBA traffic model that counts window reloads, so replaying it places `flash_moba` at the superseded 186.2 FLOP/byte instead of 24.1. The figure is now built by `--from-data`, which recomputes traffic with the current model. |
| `logs/65k.log` | `docs/block-size-results.md:165` — block_size=32 at 2048 columns |
| `logs/block-sweep.log` | `docs/superpowers/plans/2026-09-14-block-size-sweep.md` |

## Not cited by anything

`data/residual-b.json` (second residual run), `data/tabled-*.csv` (tuned-table
confirmation), `figures/roofline-alt.png` (a second render from this session;
`figures/roofline.png` is the one the docs cite, and since the
2026-09-22 rebuild the two are byte-identical, so this one can go
whenever someone decides to), `figures/moba-*.png`,
`figures/results.html` (84 bytes of `<img>` tags `triton.testing` writes next
to its CSVs), and the session logs `last`, `occ`, `passA`, `passB`, `sweep`,
`fairness`, `65k-recovery`.

`logs/bootstrap.log` (332 KB) and `logs/fa283-build.log` (360 KB) are present
on disk but deliberately **not tracked** — they are compiler transcripts
proving a build happened, which no document claims. See the `results/` block
in `.gitignore`.

## Known bad

`triton_moba` crashes with an illegal memory access at S=65536. Our kernel
passes the identical config alone, so it is their bug, not the harness's. It
is why several sweeps here carry `--backends` filters.
