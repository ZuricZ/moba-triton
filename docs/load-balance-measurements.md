# Load imbalance, measured after scheduling

**Date:** 2026-08-18
**Device:** RTX 3050 Ti Laptop, sm_86, 20 SMs | bf16 | 8 heads, head_dim 64
**Source:** `bench/load_balance.py` (`uv run python -m bench.load_balance`)
**Settles:** design question B — how hot KV blocks are handled

---

## The question

In the KV-block-major design a program owns one KV block, and its workload is
the number of queries that selected it. `docs/divergence-measurements.md`
Finding 3 reports **6.2x max/mean consumers per block**, worsening with context.
Taken alone that suggests one program runs 6x longer than its neighbours while
other SMs idle.

But skew is not cost. A GPU dispatches programs to processors as they free up,
so variation between programs can be absorbed — provided there are enough of
them. This measures what the skew costs *after* scheduling.

## Method

Job costs come from real oracle selection masks. A program loads its KV block
once, then processes consumers in tiles of 32 queries, so work is counted in
tiles:

```
tiles(block) = ceil(consumers / GROUP),  GROUP = 32
```

Tiling is not a modelling artifact — it is what the kernel does for
tensor-core reasons — and it damps skew for free, since `ceil` compresses the
distribution. Skew in tiles (4.9x) is lower than skew in consumers (6.2x).

Scheduling is greedy list scheduling, which is what "dispatch to the next free
processor" means. Four dispatch orders are compared:

| order | meaning |
|---|---|
| ascending | shortest first — adversarial, longest job starts last |
| random | no ordering; the default |
| descending | longest first (LPT); the cheap fix under consideration |
| chunked | every tile its own program; perfect balance by construction |

**40 concurrent slots** = 20 SMs x 2 resident programs, the occupancy measured
in `docs/learning/02-anatomy-of-a-flash-attention-kernel.md`.

## Results

Makespan divided by the perfectly balanced ideal. `1.00x` is perfect.

| config | programs | jobs/proc | skew | ascending | **random** | **descending** | chunked |
|---:|---:|---:|---:|---:|---:|---:|---:|
| S=4096 blk=16 k=2 | 2048 | 51.2 | 4.9x | 1.07x | **1.03x** | **1.00x** | 1.00x |
| S=4096 blk=32 k=2 | 1024 | 25.6 | 3.1x | 1.06x | **1.04x** | **1.01x** | 1.01x |
| S=4096 blk=128 k=2 | 256 | 6.4 | 2.7x | 1.25x | **1.27x** | **1.05x** | 1.01x |
| S=4096 blk=128 k=4 | 256 | 6.4 | 2.9x | 1.27x | **1.28x** | **1.03x** | 1.01x |
| S=1024 blk=128 k=2 | 64 | 1.6 | 2.0x | 1.73x | **1.73x** | **1.26x** | 1.02x |
| S=1024 blk=128 k=4 | 64 | 1.6 | 1.9x | 1.63x | **1.68x** | **1.21x** | 1.03x |

## Finding 1 — skew barely predicts cost; jobs per processor does

Read the `skew` and `random` columns together. **4.9x skew costs 3%. 2.0x skew
costs 73%.** The relationship is inverted, because severity is governed by
`jobs/proc`, not by how uneven the jobs are.

With 51 jobs per slot, unevenness averages out on its own — a processor that
draws a long job draws short ones afterwards. With 1.6 jobs per slot there is
no averaging left to do, and one unlucky draw sets the runtime.

> This is why the alarming 6.2x figure turned out to be harmless in the regime
> the project targets, and why a mild 2.0x is severe in a regime it does not.

## Finding 2 — in the target regime there is nothing to fix

At `block_size` 16 and 32 with long context — small blocks, the entire point of
this project — random dispatch costs **3–4%**. The grid is
`num_blocks x batch x heads`, which at S=4096, `block_size=16`, H=8 gives 2048
programs against 40 slots.

Sorting removes even that, taking it to 1.00–1.01x.

## Finding 3 — sorting recovers most of the penalty everywhere else

Longest-first dispatch:

- 256 programs: **1.27x → 1.05x**
- 64 programs: **1.73x → 1.26x**

Chunking does better in the 64-program case (1.02x), and only there.

## Decision

> **Sort KV blocks by consumer count, descending, and dispatch longest-first.**
> **Do not chunk in v1.**

Sorting is close to free:

- Job sizes are a **by-product of building the index**. The counting-sort pass
  that produces the CSC layout must count consumers per block anyway; that
  count *is* the job size.
- Sorting a few thousand int32s costs microseconds.
- It requires **no scatter-back**. Sorting permutes *which program handles
  which block*, not the data. A program reads `block_id = order[program_id]`
  and writes results to that block's queries, at addresses derived from query
  indices — which never depended on the program id. One extra indirection at
  the top of the kernel, and nothing to undo afterwards.

Chunking is deferred rather than rejected:

- Splitting a hot block across several programs makes each of them **load that
  block's keys and values separately**, trading away the memory reuse the
  inversion exists to obtain.
- It produces more partial results for the combine pass.
- It requires a data-dependent grid, or a static upper bound with early-exit
  tiles.
- It wins only at ~1.6 jobs per processor — coarse blocks and short sequences,
  which is precisely where query-major utilization is 0.417
  (`docs/divergence-measurements.md`) and the inversion is least justified. We
  may not route that regime to this kernel at all.

**Trigger for revisiting:** if profiling shows tail stalls in a regime we
actually target, or if program count approaches slot count on the eventual
benchmark hardware. Note an A100 has 108 SMs against this machine's 20, so
`jobs/proc` falls by ~5x for the same problem — the 64-program case becomes
relatively worse there, not better.

## Caveats

All of these make the table the **pessimistic** reading:

- Fixed per-program cost (loading the KV block) is ignored. Including it adds a
  constant to every job and reduces relative imbalance.
- Bandwidth contention between concurrent programs is ignored.
- Greedy list scheduling approximates the hardware dispatcher.
- Occupancy of 2 programs/SM is inherited from the FA-2 measurement at 64x64
  tiles; the MoBA kernel's occupancy will differ. Lower occupancy means fewer
  slots, hence *more* jobs per slot, hence less imbalance.
