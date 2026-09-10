# A100-40GB — the head-to-head the project was built to get

**Date:** 2026-09-10
**Device:** NVIDIA A100-PCIE-40GB, sm_80, 108 SMs, 164 KiB shared memory,
driver 595.58.03, CUDA 12.6.77
**Config:** `block_size=128, top_k=8` (FlashMoBA's published gate), bf16,
`[1, S, 4, 64]`
**Source:** `bench/compare.py`, raw log in `results/a100-sweep.log`

Every previous number in this repo came from a 20-SM laptop with 4 GB and
99 KiB of shared memory. Those measured a kernel entangled with one card. This
is the first run on hardware the references were designed for, and the first
with four MoBA implementations live at once.

---

## Forward

| S | blocks | **ours** | flash_moba | moonshot_efficient | moonshot_naive | sdpa_dense | fa2_minimal |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1024 | 8 | 0.214 | 0.486 | 1.908 | 1.122 | 0.027 | 0.031 |
| 2048 | 16 | 0.220 | 0.540 | 1.919 | 2.178 | 0.044 | 0.067 |
| 4096 | 32 | 0.252 | 0.671 | 2.018 | 6.222 | 0.144 | 0.156 |
| 8192 | 64 | 0.421 | 1.082 | 2.569 | 20.070 | 0.361 | 0.465 |
| 16384 | 128 | **0.699** | 2.126 | 5.280 | 69.432 | 1.210 | 1.613 |
| 32768 | 256 | **1.261** | 5.111 | 17.246 | OOM | 4.171 | 5.871 |
| 65536 | 512 | **2.736** | 12.198 | 57.254 | OOM | 15.456 | 21.789 |

## Backward (`forward().backward(dO)`, the full step)

| S | **ours** | flash_moba | moonshot_efficient | sdpa_dense | fa2_minimal |
|---:|---:|---:|---:|---:|---:|
| 8192 | 0.807 | 1.453 | 4.167 | 1.212 | 5.005 |
| 16384 | **1.416** | 2.779 | 8.342 | 3.923 | 15.400 |
| 32768 | **2.631** | 6.338 | 24.668 | 14.324 | 58.449 |
| 65536 | **5.446** | 14.278 | 76.044 | 55.025 | 233.363 |

## The ratios

| S | vs `sdpa_dense` | vs `flash_moba` | vs `moonshot_efficient` |
|---:|---:|---:|---:|
| 8192 | 0.86x | 2.57x | 6.10x |
| 16384 | **1.73x** | 3.04x | 7.55x |
| 32768 | **3.31x** | 4.05x | 13.7x |
| 65536 | **5.65x** | 4.46x | 20.9x |

**The central claim holds and strengthens with context**, which is what the
loop-inversion argument predicted and what no laptop measurement could confirm:
0.57x -> 0.86x -> 1.73x -> 3.31x -> **5.65x** against dense.

---

## Why beating FlashMoBA is not suspicious

A 4.46x win over a CUDA kernel from the lab that published the paper on making
MoBA fast deserves scrutiny before it is written down. Three checks.

### It is computing the same thing

Relative to the PyTorch oracle at S=4096, error is *identical to four decimals*
across all three: `ours` 0.0478, `flash_moba` 0.0478, `moonshot_efficient`
0.0478. Same block choices, differing only in bf16 rounding. Nothing is being
skipped.

### The gap grows on realistic shapes

The sweep uses `[1, S, 4, 64]`, a small workload. If the win were an artefact
of leaving a 108-SM card idle, a wider shape would close it:

| shape | ours | flash_moba | ratio |
|---|---:|---:|---:|
| B1 S8192 H4 D64 (sweep) | 0.401 | 1.042 | 2.60x |
| B1 S8192 H32 D128 | 2.649 | 8.144 | 3.07x |
| B4 S8192 H32 D128 | 9.618 | 29.984 | 3.12x |
| B1 S16384 H32 D128 | 5.462 | 18.043 | 3.30x |

It widens.

### It is the attention stage, not the gate

Both pipelines split at the same seam -- routing, then attention -- so the
halves can be timed separately:

| shape | stage | ours | theirs | ratio |
|---|---|---:|---:|---:|
| B1 S8192 H4 D64 | gate + routing | 0.187 | 0.390 | 2.08x |
| | attention + combine | 0.217 | 0.686 | **3.16x** |
| B1 S8192 H32 D128 | gate + routing | 0.666 | 1.277 | 1.92x |
| | attention + combine | 2.026 | 6.866 | **3.39x** |

**The win is concentrated in the attention kernel**, which is the stage
FlashMoBA exists to optimise, and the only stage where the two designs differ
structurally.

The mechanism is the project's thesis, measured. Their `decide_lg_block_m`
picks `lg_block_m=512` -- a *query-major* tile, where one program owns 512
queries and loads the union of every block any of them selected, then masks
away the rest. `docs/divergence-measurements.md` puts that waste at **77% of
tensor-core work** at S=8192 with 64 blocks and `top_k=8`. A KV-block-major
program owns one block and gathers exactly the queries that chose it, so it
discards none. 3.2x against 77% waste is the right order of magnitude.

That the gate ratio (1.9-2.1x) is *smaller* than the attention ratio (3.2-3.4x)
is itself corroboration. Both pipelines compute the same routing by similar
means; only the attention differs by design. A uniform win across both stages
would have suggested a measurement artefact instead.

---

## What the sweep confirms about the algorithm

**Cost is linear in S, and the measurement says so.** MoBA attends
`top_k * block_size` = 1024 keys per query whatever S is, so doubling S should
roughly double the time: 0.699 -> 1.261 -> 2.736 ms is 1.80x then 2.17x.

**The dense baselines die where predicted.** `naive_torch` and
`moonshot_naive` both OOM at S=32768 trying to allocate 16 GiB -- attention
memory quadratic in S -- while `ours` runs S=65536 in 2.7 ms. That is not a
speed result, it is a feasibility one.

**Read `TF/s attn`, not `TF/s dense`.** The sweep prints both conventions. The
dense-equivalent figure reaches 803 TFLOP/s at S=65536, which is not real
throughput -- it counts FLOPs a dense kernel *would* have done. Attended
throughput, the honest column, is flat across the whole sweep: 20.4, 24.6,
27.3, 25.1. **Flat attended throughput beside an exploding dense-equivalent
number is exactly what sparsity looks like**, and is why the harness refuses to
print one without the other.

**The crossover moved.** On the laptop `ours` overtook dense at S=4096; here it
is between 8192 and 16384. An A100 runs dense attention far faster in absolute
terms (0.361 ms at S=8192 against the laptop's 2.814), so dense holds on longer
before quadratic cost catches it. Same kernel, same algorithm, different
crossover -- which is precisely why one machine could not settle this.

---

## Triton-MoBA did not run, and cannot alongside MoonshotAI

`triton_moba` is the fifth implementation and the most directly comparable of
all -- Triton against Triton. It failed:

    ValueError: too many values to unpack (expected 4)

Their `triton_moba.py` unpacks **four** values from flash-attn's private
`_flash_attn_varlen_forward`. Version 2.6.3 returns **eight**. Four is the 2.7
signature -- and their own `requirements.txt` pins 2.6.3.

Their repo is internally inconsistent about this: `efficient_moba.py` in the
same package unpacks eight. But that file is not an alternative, because it is
a **byte-identical copy of MoonshotAI's `moba_efficient.py`** apart from the
function name (verified by diff). Benchmarking it would re-measure
`moonshot_efficient` under a different label.

So the versions are mutually exclusive:

| flash-attn | `moonshot_efficient` | `triton_moba` |
|---|---|---|
| 2.6.3 (what we built) | works | fails |
| >= 2.7 | would fail | would work |

Getting both means running the sweep twice against two flash-attn builds. That
is now what happens -- `bootstrap_remote.sh --fa=` builds either version and
`--use-fa=` switches between them in a second, because flash-attn is reached
through a `.pth` file rather than installed, so torch and triton stay
bit-identical across the two passes. `docs/environment.md` has the API table
and why a compatibility shim was rejected.

The first pass, the one in the tables above, used 2.6.3: MoonshotAI's fused path
is the reference implementation the field actually uses, and it was the
comparison pending since `moonshot` was renamed to `moonshot_naive`. The
`triton_moba` column is measured in a second pass and is **not** in the tables
above -- when it lands it goes in its own table, with the flash-attn version
named, because a number measured under a different dependency does not belong in
the same row as one that was not.

---

## Caveats

* **`ncu` was refused here too.** The container had neither `CAP_SYS_ADMIN` nor
  `CAP_PERFMON`, and the host driver restricts profiling to admins -- verified
  by compiling a CUDA kernel and profiling it. Occupancy came from
  `torch.profiler` instead; DRAM traffic remains modelled rather than measured.
* **Clocks were not lockable**, so the cool/warm/best-of-3 thermal protocol ran
  here as on the laptop. A racked A100 is far more stable, so this is belt and
  braces rather than load-bearing.
* **One config.** `block_size=128, top_k=8` throughout, and `[1, S, 4, 64]` for
  the sweep. The shape check above covers H=32, D=128 and batch 4 for the
  FlashMoBA comparison only.
* **Forward and backward, no end-to-end model.** Nothing here measures a
  training step of a real network.
