# Load imbalance, measured after scheduling

**Date:** 2026-08-22 (re-measured at realistic configurations)
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
tiles(block) = ceil(consumers / GROUP),  GROUP from choose_group()
```

Tiling is not a modelling artifact — it is what the kernel does for
tensor-core reasons — and it damps skew for free, since `ceil` compresses the
distribution. Skew in tiles (2.3–3.7x) is lower than skew in consumers
(2.4–4x, `docs/divergence-measurements.md`).

`GROUP` is read from `choose_group()` rather than hardcoded. An earlier version
of this file fixed it at 32 with a comment asserting it equalled
`top_k * block_size`; once the kernel began deriving it from the shared-memory
budget, this simulation was silently predicting a kernel that no longer
existed.

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
| S=4096 blk=32 k=4 | 1024 | 25.6 | 3.7x | 1.07x | **1.07x** | **1.01x** | 1.01x |
| S=4096 blk=32 k=8 | 1024 | 25.6 | 3.7x | 1.09x | **1.08x** | **1.00x** | 1.00x |
| S=4096 blk=64 k=8 | 512 | 12.8 | 3.1x | 1.13x | **1.11x** | **1.01x** | 1.01x |
| S=4096 blk=128 k=8 | 256 | 6.4 | 2.6x | 1.20x | **1.21x** | **1.02x** | 1.00x |
| S=4096 blk=128 k=12 | 256 | 6.4 | 2.3x | 1.18x | **1.18x** | **1.01x** | 1.00x |
| S=1024 blk=32 k=4 | 256 | 6.4 | 2.5x | 1.25x | **1.19x** | **1.06x** | 1.06x |

## Finding 1 — skew barely predicts cost; jobs per processor does

Read the `skew` and `random` columns together. **3.7x skew costs 7%. 2.6x skew
costs 21%.** The relationship is inverted, because severity is governed by
`jobs/proc`, not by how uneven the jobs are.

With 25.6 jobs per slot, unevenness averages out on its own — a processor that
draws a long job draws short ones afterwards. With 6.4 jobs per slot there is
much less averaging left to do, and one unlucky draw moves the runtime.

> This is why the alarming skew figures turned out to be harmless in the regime
> the project targets, and why milder skew is worse in a regime it does not.

## Finding 2 — in the target regime there is nothing to fix

At small `block_size` with long context, random dispatch costs **7–8%**. The
grid is `num_blocks x batch x heads`, which at S=4096, `block_size=32`, H=8
gives 1024 programs against 40 slots — 25.6 jobs each, enough for the law of
averages to do the work.

Sorting removes even that, taking it to 1.00–1.01x.

## Finding 3 — sorting recovers most of the penalty everywhere else

Longest-first dispatch:

- 512 programs (blk=64, k=8): **1.11x → 1.01x**
- 256 programs (blk=128, k=8): **1.21x → 1.02x**
- 256 programs at S=1024: **1.19x → 1.06x**

Chunking matches longest-first everywhere in this sweep and beats it nowhere,
which is a change from the earlier `block_size=128, top_k=2` configuration
where it helped at 64 programs. Fewer, larger blocks made the tail matter more;
the realistic `top_k` makes columns more uniform and removes the need.

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
