# Occupancy, and why raising it does not help

**Date:** 2026-09-10
**Device:** RTX 3050 Ti Laptop, sm_86, 20 SMs, 48 warps/SM | bf16
**Source:** `bench/profile.py --tier torch`, and a `maxnreg` sweep
**Settles:** whether low occupancy explains the roofline gap. **It does not.**

---

## The question

`bench/roofline.py` puts `ours` at 75.8 FLOP/byte against a ridge of 82 --
barely memory bound -- while reaching only ~50% of the throughput that
intensity allows. Neither ceiling explains the other half. Occupancy was the
leading suspect: a kernel that cannot keep enough warps resident has nothing to
switch to while it waits on memory, and would sit below both ceilings exactly
as observed.

## Occupancy is indeed low

`ncu` is refused on every machine this project has access to, but CUPTI's
*activity* records are not gated and carry the occupancy calculation.
Measured at S=4096:

| kernel | % time | regs | smem | blocks/SM | warps/SM | occupancy | limited by |
|---|---|---|---|---|---|---|---|
| `moba_fwd_kernel` | 60.8% | 218 | 49920 | 2 | 8 | **17%** | registers |
| `moba_combine_kernel` | 14.2% | 40 | 0 | 12 | 48 | 100% | warps |
| `gate_kernel` | 8.0% | 56 | 12544 | 7 | 28 | 58% | shared memory |
| everything else | <7% | | | 12 | 48 | 100% | warps |

The kernel taking 60% of the time keeps 8 of a possible 48 warps resident.
Count it by hand: 218 registers x 128 threads = 27,904 registers per block, and
an SM has 65,536, so two blocks fit and no more.

## Raising it makes things worse

`maxnreg` caps registers per thread, trading spills for residency. Attend
kernel alone, S=8192, best of three under the thermal protocol:

| maxnreg | regs | spill | blocks/SM | warps/SM | ms |
|---|---|---|---|---|---|
| none | 218 | 0 | 2 | 8 | **0.834** |
| 200 | 200 | 0 | 2 | 8 | 0.899 |
| 168 | 168 | 0 | 3 | 12 | 0.882 |
| 128 | 128 | 54 | 4 | 16 | 1.251 |
| 96 | 96 | 190 | 5 | 20 | 3.603 |
| 64 | 64 | 376 | 8 | 32 | 7.525 |

Two separate readings, and the second is the one that matters.

**Spilling is catastrophic, as expected.** At `maxnreg=128` occupancy doubles
and the kernel is 1.5x slower; at 64 it quadruples and the kernel is 9x slower.
A spilled register lives in local memory, which is DRAM behind an alias -- so
the trade buys latency-hiding capacity by adding the very latency it hides.

**`maxnreg=168` is the decisive row.** Zero spills, three blocks per SM instead
of two, 12 warps instead of 8 -- 50% more residency at no cost -- and it is
*still slower* (0.882 against 0.834). Free occupancy, no benefit.

## What that rules out

**Occupancy is not the binding constraint.** If the kernel were stalling for
want of another warp to run, 12 warps would beat 8. It does not, so the missing
throughput is not latency-hiding capacity.

The register count is also not worth attacking for its own sake. 218 with zero
spills is the best point on this curve; every attempt to reduce it either
spills or slows the kernel down by lengthening its instruction stream.

## What remains

The next hypothesis is the gathered load. A program collects the query rows
that selected its KV block, and those rows are scattered -- 128 separate
addresses per tile rather than one contiguous burst. `docs/gather-cost-
measurements.md` measured a fully randomised permutation at 1.03x, but that
measured a *copy*, where the address stream is the whole workload and the
hardware can run far ahead. Inside an attention kernel the gather alternates
with dependent arithmetic, and a memory-level-parallelism limit would look
exactly like this: below both ceilings, and indifferent to occupancy.

Distinguishing it needs either `ncu`'s memory counters (refused everywhere so
far) or an experiment that changes the address pattern while holding the
arithmetic fixed -- for instance, timing the kernel against a routing index
whose columns have been made artificially contiguous. That is a real
measurement rather than a reading, and is not yet done.

## Reproducing

```bash
uv run python -m bench.profile --seq-len 4096 --tier torch
```

Two traps in the trace, both silent, both worth knowing before trusting any
occupancy number from this source:

* The field named `occupancy` is a nested object, not a percentage. Read as a
  float it yields `0.0` for every kernel, which looks like data.
* CUPTI reports `blockLimitSharedMem = 0` for any kernel that opts in past the
  48 KiB static limit -- its calculator does not model the opt-in Triton
  performs. Its own `est. achieved occupancy %` is therefore `0` for precisely
  the kernel under investigation. `bench/profile.py` ignores those zeros and
  computes residency from the block limits instead.
