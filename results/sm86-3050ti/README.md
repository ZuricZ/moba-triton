# RTX 3050 Ti Laptop — sm_86

```
NVIDIA GeForce RTX 3050 Ti Laptop GPU, 4095 MiB, compute capability 8.6
20 SMs, 99 KiB shared memory per block (101376 B opt-in)
driver 566.07 / CUDA 12.7, WDDM, via WSL2
```

The development machine. **This card is for correctness and learning, not for
headline numbers** — 4 GB of VRAM caps sequence length in the low thousands,
and `CLAUDE.md` says not to over-index on absolute speed measured here.

Three things are only measurable on this card, which is why its results are kept
rather than superseded:

* `decide_lg_block_m` caps FlashMoBA's window at 512 on sm_86/sm_89 (their
  "[A6000 Fix] ... avoid Shared Memory OOM"). sm_80 and sm_90 take no cap, so
  the capped branch is exercised nowhere else.
* Shared memory per block is 99 KiB against the A100's 164 KiB, so a tuned
  entry that fits there may not fit here — `src/moba/kernels/tuning.py:307`.
* The measured tile-shape flat roof (`bench/tile_ceiling.py`) exists only for
  this card. It reads 14.23 TFLOP/s against a 4096³ GEMM's 13.04 — the tile is
  *faster* than the big GEMM here, because on 4 GB the GEMM is itself memory
  bound. Correcting the roof moved no conclusion in any document: every kernel
  the write-ups argue about is memory bound, and a memory-bound point is
  measured against the slanted roof, which the flat roof does not enter.

## What cites what

| file | cited by |
|---|---|
| `data/local_rtx3050ti.csv` | `docs/baseline-results.md` — the pre-kernel baseline |
| `data/kernel_v1.csv` | `docs/baseline-results.md`, `docs/reference-comparison.md:43` — where the reported 1.53× came from |
| `data/block-sweep-budget.csv`, `data/block-sweep-fixed-top-k.csv` | `docs/block-size-results.md:7` — the confirmation run, same shape on 20 SMs |
| `figures/roofline.png`, `logs/roofline.log` | `bench/tile_ceiling.py` docstring — the first run of the measured flat roof, 2026-09-21 |

The two block-sweep files were `block-sweep-*-3050ti.csv` when `results/` was
flat. The `-3050ti` suffix is now the directory, so the filenames match the
A100's exactly — which is the point: the same plot script reads either, and
`--card` chooses.
