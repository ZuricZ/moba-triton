# results/

Measurements, one directory per card.

```
results/
  sm80-a100/      A100-PCIE-40GB   the performance numbers
  sm86-3050ti/    RTX 3050 Ti      the development laptop
  sm90-h100/      H100             empty, ready
    data/         csv, json  -- the measurements
    figures/      png        -- drawn from that card's data
    logs/         the raw session transcripts
```

Each card directory has a `README.md` recording its device string, which
document cites which file, and what is deliberately not cited.

## Why the card is a directory and not a filename suffix

`triton.testing.perf_report(save_path=...)` writes a **fixed** set of names —
`moba-fwd.csv`, `moba-bwd.csv`, and their plots — into whatever directory it
is handed. Flat, that meant a second card's run overwrote the first card's CSV
in place: no error, no name collision, nothing in `git status` beyond a
modified file. `data/moba-fwd.csv` is the file the headline table is read
from, so an A100 claim would have come to rest on H100 data.

The card being structural makes that collision impossible however fixed the
filenames are. It also means the `-3050ti` suffix is gone: the same plotting
script reads either card, and `--card` chooses.

## Reading someone else's card

Writing scripts ask the driver which GPU they are on. Reading scripts must
not — A100 figures are drawn on a laptop that has never seen one — so every
plotting script takes `--card`:

```bash
python -m bench.block_sweep_plot --card sm80-a100
python -m bench.block_sweep_plot --card sm80-a100 --tag=-65k
```

`--tag` must be joined with `=` when its value starts with `-`, or argparse
reads it as another flag. The resolution lives in `bench/paths.py`.

## What is tracked

CSV, JSON, PNG and Markdown at any depth, plus the session logs under
`*/logs/`. Deliberately excluded: `logs/bootstrap.log` and
`logs/fa283-build.log`, 692 KB of compiler transcript between them that prove
a build happened, which no document claims. The rules are at the end of
`.gitignore`, where the comment explains the two ways that block has silently
broken before.
