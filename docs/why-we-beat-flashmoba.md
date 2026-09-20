# Why this kernel beats FlashMoBA, measured

**Date:** 2026-09-20
**Device:** A100-PCIE-40GB, sm_80 | bf16 | B=1, H=4, head_dim 64 unless stated
**Source:** `bench/residual.py`, `bench/tile_census.py`, `bench/profile.py`,
`bench/compare.py`
**Data:** `results/sm80-a100/data/residual.json`, `results/sm80-a100/logs/final-batch.log`,
`results/sm80-a100/logs/occ_fm.log`, `results/sm80-a100/data/moba-fwd.csv`
**Presentable version:** https://claude.ai/artifact/4fZjJ8nEKS79Mk1D5XcAAz

This is the argument the whole project rests on, with the three explanations
that turned out to be wrong kept in rather than tidied away. One of them was
argued confidently in this repository's own documentation for five days.

## 1. The result

Both implementations compute the same thing: each query scores every block of
keys against a mean-pooled representative, keeps the top `top_k`, and attends
only to those.

| implementation | language | S=65536 | vs ours |
|---|---|---:|---:|
| **ours** | Triton | **2.659 ms** | -- |
| `flash_moba` | CUDA / CUTLASS | 12.287 ms | 4.62x |
| `sdpa_dense` | PyTorch SDPA | 15.414 ms | 5.80x |
| `fa2_minimal` | FlashAttention-2 | 21.529 ms | 8.10x |
| `triton_moba` | Triton | 48.045 ms | 18.07x |

Whole pipeline, gate included, so both sides are measured over the same
stages. The advantage is not an artefact of the benchmark's narrow shape: at
32 heads, head_dim 128 and batch 4 it holds between **2.67x and 2.91x**
(`bench/_a100_fairness.py`).

## 2. What tile padding already accounts for

Both kernels issue query tiles 64 rows tall -- theirs via
`Flash_moba_fwd_kernel_traits<Headdim, 64, ...>`, ours via `GROUP=64`. A tile
costs a full tensor-core issue whether 64 rows are real or nine are, so tile
*count* is a work multiplier, and it is countable on a CPU.

FlashMoBA may only fill a tile from within one query window, at most 1024
consecutive queries. Ours fills from the whole sequence.

| S | their tiles / ours | their tiles full | ours full |
|---:|---:|---:|---:|
| 1,024 | 1.00x | 100% | 100% |
| 8,192 | 1.24x | 78.3% | 97% |
| 65,536 | 2.26x | 42.9% | 97% |

Their fill degrades because the queries selecting any given block number
about `top_k * block_size` = 1024 regardless of `S`, while the sequence they
are spread over grows and the window stays capped. At 65K a window catches
roughly 16 of them for a 64-row tile.

`bench/tile_census.py` cross-checks against measured routing: at S=8192 it
reports 247,808 pairs, matching the `nnz` column of
`results/sm80-a100/logs/dip-stages.log` exactly.

**So 2.26x is accounted for. Measured on the attention kernels alone the gap
at 65K is 8.35x, leaving 3.7x -- and it grows with `S`, which rules out any
fixed cost.**

## 3. Four suspects, three alibis

### Eliminated: their gate is slow

Their routing is three launches, a host sync per call
(`triton_mean_pool.py:140`) and 67-134 MB of `torch.zeros` each time. Timed
apart, using their own two public entry points:

```
attention only, S=65536    ours 1.126 ms   theirs 9.402 ms   8.35x
whole pipeline             ours 2.659 ms   theirs 12.287 ms  4.62x
```

Their attention alone is the slower half. The attention-only ratio is
*larger* than the whole-pipeline ratio, so their gate is relatively the
better part of their pipeline.

### Eliminated: they leave the GPU idle

Their grid is `(ceil(S / lg_block_m), batch, heads)`. At the benchmark's own
shape that is 64 programs on a 108-SM card -- 40% unused before the kernel
starts -- against our 256. Raising `batch * heads` until both oversubscribe:

```
S=8,192    64 programs -> 2.59x      2,048 programs -> 2.67x
S=16,384   88 programs -> 3.21x      2,816 programs -> 2.85x
```

From 40% idle to 19x oversubscribed moves the ratio by nothing.

### Eliminated: it is tuned for an H100

Their kernel takes 84 KiB of shared memory per program. An H100 has 228 KB
per SM, an A100 164 KB. If the footprint were binding here, a *narrower*
window -- less shared memory, more programs resident -- should help. Their
attention takes the window as a parameter, so it can simply be asked for:

```
S=65,536   window 128 -> 31.018 ms   window 512 -> 12.472 ms
           window 768 -> 10.263 ms   window 1024 -> 9.471 ms  <- their policy
```

Narrowing it makes them 3.3x slower. `decide_lg_block_m` picks the best
window at every length measured; it is tuned correctly for sm_80.

**Re-tested on an H100, 2026-09-20** (`results/sm90-h100/logs/residual-cd.log`),
and there the claim narrows rather than failing. Their policy is still the
best window at S=65,536 — the length the headline lives at — but
it is off at shorter ones:

```
      S  fastest window   their policy   penalty
   8192     128               512         1.57x
  16384     256               768         1.27x
  65536    1024              1024         1.00x  <- still correct here
```

So the elimination survives where it matters — the gap grows with `S`,
and the window that produces it is the correctly-chosen one — but the
general statement "tuned correctly at every length" holds only on sm_80. On
Hopper their policy is up to 1.57x off at S=8192.

### Eliminated: they get worse occupancy

This one was asserted in `docs/a100-results.md` from arithmetic rather than
measured. Profiling both kernels on the A100 settles it:

```
kernel                      regs     smem   blk/SM  warps  occupancy  limited by
ours   moba_fwd_kernel       230   49,920        2      8        12%  SMEM
theirs (77.1% of their run)  241   86,016        2      8        12%  SMEM
```

Identical. The claim that they fit one program per SM against our two was
bad arithmetic: 164 / 84 is 1.95, so two of theirs fit, not one.

### Survives: they move far more memory

The only candidate left, and the only one that predicts a gap growing with
`S`, which is what is measured.

## 4. The gap is two factors multiplied

Modelled bytes over measured time gives achieved bandwidth, and the gap
splits cleanly:

| S | bytes ratio | efficiency ratio | predicted | measured |
|---:|---:|---:|---:|---:|
| 8,192 | 3.14x | 1.02x | 3.20x | 3.07x |
| 16,384 | 3.19x | 1.30x | 4.15x | 4.16x |
| 65,536 | 4.30x | 1.95x | 8.39x | 8.35x |

The second column is what makes the gap grow. At S=8192 they achieve
*exactly the bandwidth we do* -- 479 GiB/s against our 468 -- and their extra
time is entirely extra bytes. By S=65536 ours has climbed to 61% of the
measured roof (1239 GiB/s) while theirs has fallen to 31%.

With occupancy measured identical, an efficiency gap at long context can only
be the access pattern.

## 5. The cause: what a program owns

Both kernels loop over key blocks on the inside. **It is ownership, not loop
nesting, that decides how many times K and V cross the memory bus.**

```
FlashMoBA -- a program owns a query window
  program W0 (q0..q1023)     load B0, B1, B2 ... B511
  program W1 (q1024..q2047)  load B0, B1, B2 ... B511   again
  ...                        64 windows at S=65,536

Ours -- a program owns a key block
  program B0                 load B0   once
  program B1                 load B1   once
  ...                        each block read exactly once
```

Their byte budget at S=65536, from the traffic model:

| FlashMoBA, 3,718 MiB total | MiB | share |
|---|---:|---:|
| K re-read per (window, block) | 1,134 | 31% |
| V re-read per (window, block) | 1,134 | 31% |
| fp32 accumulator round-trip | 1,024 | 28% |
| gathered queries, output, indices | 426 | 10% |

The first three are 90% of their traffic and all three follow from the
window. We pay none of the first two.

The third is the price of finishing in one kernel: they keep a running
softmax in shared memory and round-trip an fp32 accumulator once per pair.
Our alternative is not free either -- partial results are 60% of *our* 864
MiB, and a second kernel merges them, which is why our scratch is 4.2x
theirs (`docs/traffic-derivation.md`). It buys full tiles and single reads.

## 6. What this does not say

* **The byte counts are modelled, not measured.** Derived by reading both
  sources and counting what each must move. `ncu` would measure them and has
  been refused on every machine this project has touched -- the laptop by
  driver policy, the rented A100 by container capabilities. The *timings* are
  real; the split between "more bytes" and "worse efficiency" is
  model-dependent.
* **This is one card.** Everything here is sm_80. Their paper reports 2.02x
  over FA2 at 64K on an H100 SXM; we measure 1.22x on this card. Section 3
  tested whether that is a tuning mismatch and found it is not, but their
  kernel has not been run on its own hardware here.
* **The advantage is specific to how sparsity is produced.** This design pays
  per (query, block) pair, so it converts sparsity into speed when sparsity
  comes from length. Produced the other way -- shrinking the block at a fixed
  key budget, which drives `top_k` 8 to 64 -- the same kernel is 112x slower
  at S=65536, and below block_size 64 dense attention wins outright
  (`docs/block-size-results.md`).
* **Their gate is not identical to ours.** Both spend the same budget and
  select the same pair count, but their tie-breaking differs from the oracle
  on about 1% of queries -- every one at the selection cutoff, none before the
  first block where a query has more candidates than slots. Covered by
  `tests/test_flash_moba_parity.py`.
* **Our measurements are noisier than theirs.** Run-to-run spread across a
  process restart is 5.7% for us against 0.5% for them, which is the cost of
  a Triton kernel re-selecting launch configs where a compiled CUDA kernel is
  byte-identical every run. Do not read these ratios to three digits.

## 7. The conclusion

FlashMoBA assigns each program a neighbourhood of queries and makes it walk
every key block, so a block is loaded once per neighbourhood that wants it,
and a 64-row tile can only be filled from 1024 consecutive queries. We assign
each program a key block and let it gather from the whole sequence, so every
block is read once and every tile is full.

**That one choice shows up twice -- as tiles and as bytes -- and it is the
entire difference.**
