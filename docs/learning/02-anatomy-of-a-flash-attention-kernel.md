# Learning note 02 — Anatomy of a FlashAttention kernel

**Reading a real Triton kernel line by line, then breaking it on purpose.**

Prerequisites: `docs/learning/01-loop-order-and-memory-traffic.md` (HBM vs SRAM,
arithmetic intensity, why loop order decides memory traffic).

Code under discussion: `src/moba/kernels/fa2_minimal.py`. Every number below was
measured on the project GPU (RTX 3050 Ti Laptop, sm_86, 20 SMs, 99 KiB shared
memory per block), not estimated.

---

## 1. You write one worker; Triton launches thousands

The largest conceptual jump from PyTorch. In PyTorch, `q @ k.T` means the whole
tensor. In Triton you write what **one small piece of the work** does, and
declare how many pieces exist.

Each copy is a **program** (elsewhere: a thread block). It runs on one of the
GPU's **SMs** (Streaming Multiprocessors — the physical core clusters).

```python
grid = (triton.cdiv(s, block_m), b * h)     # how many copies
```

With `S=1024`, `BLOCK_M=64`, `B=1`, `H=8`, that is `(16, 8)` = **128 programs**.
Inside the kernel each copy asks which one it is:

```python
pid_m  = tl.program_id(0)     # which query tile
pid_bh = tl.program_id(1)     # which flattened (batch, head)
```

Program `(3, 5)` computes output rows 192–255 for the sixth (batch, head) slab.
That is its whole job: read what it needs, compute, write 64 rows, exit.

```
        queries 0-63   64-127   128-191   192-255   ...
head 0     P(0,0)      P(1,0)    P(2,0)    P(3,0)
head 1     P(0,1)      P(1,1)    P(2,1)    P(3,1)
                                     ↑
                          one program, one tile of output
```

**Programs never communicate.** No locks, no ordering, no shared state. That
independence is why dense attention parallelizes so cleanly — and it is exactly
what the MoBA inversion gives up (§6).

`BLOCK_M` is a tuning knob, not a property of the problem. The mathematics is
identical at 32 or 128; only the hardware efficiency changes.

---

## 2. The `S × S` matrix never exists

The loop, stripped to its skeleton:

```python
q = tl.load(...)                          # once, before the loop

for start_n in range(0, hi, BLOCK_N):
    k  = tl.load(...)                     # one K tile
    qk = tl.dot(q, tl.trans(k))           # [BLOCK_M, BLOCK_N] scores
    ...                                   # fold into running result
    v  = tl.load(...)
    acc += tl.dot(p, v)
```

`qk` is a `64 × 64` **window of the attention matrix**, computed on-chip, used,
and discarded before the next iteration. It never reaches HBM.

```
full S x S at S=1024, fp32:  1024 x 1024 x 4  =  4 MB    -- cannot fit in 99 KiB
one 64x64 tile, fp32:          64 x   64 x 4  = 16 KB    -- fits comfortably
```

**Causal attention saves by not looking, not by masking.**

```python
hi = (pid_m + 1) * BLOCK_M if CAUSAL else S
```

Program `(2, ·)` covering rows 256–383 loops keys `0 → 384` and stops. Later
tiles are never loaded. Note the side effect: program 0 finishes almost
immediately while the last program grinds through everything — a first taste of
load imbalance.

---

## 3. What a tile actually costs — measured

`BLOCK_M`/`BLOCK_N` sweep at `B=1, H=8, S=2048, D=64`, fp16, causal:

| BLOCK_M | BLOCK_N | shared KB | regs | spills | progs/SM | ms |
|---:|---:|---:|---:|---:|---:|---:|
| 32 | 32 | 22.0 | 95 | 0 | 4 | 0.752 |
| 64 | 32 | 24.0 | 128 | 0 | 4 | 0.589 |
| **64** | **64** | **40.0** | **168** | **0** | **2** | **0.408** |
| 128 | 64 | 48.0 | 255 | 2 | 2 | 0.509 |
| 64 | 128 | 72.0 | 204 | 0 | 1 | 0.445 |
| 128 | 128 | 80.0 | 255 | 50 | 1 | 0.602 |

### Shared memory follows an exact formula

```
shared = 2·D·BLOCK_M  +  8·D·BLOCK_N        (bytes, fp16)
         └─ q, one buffer   └─ k and v, two buffers each
```

Verified against every row: 64×64 → `8 + 2×(8+8) = 40 KB`; 128×128 →
`16 + 2×(16+16) = 80 KB`; 64×128 → `8 + 2×(16+16) = 72 KB`.

The ×2 on `k` and `v` is **software pipelining**: while the tensor cores work on
tile `n`, Triton is already fetching tile `n+1` into a second buffer, so the
loop does not stall for hundreds of cycles on every iteration. It costs double
the buffer space.

**Consequence: `BLOCK_N` costs 4× as much shared memory per element as
`BLOCK_M`.** When short on shared memory, shrink `BLOCK_N` first.

### Two different resources, two different failure modes

- **Registers** — fastest storage, private per thread, hard cap of **255 per
  thread** on Ampere. `acc`, `qk`, `m_i`, `l_i` live here. Exceed it and the
  compiler **spills** to "local memory", which despite the name is backed by
  HBM. It still runs; it just quietly gets slower. Visible above as
  `spills = 50` at 128×128.
- **Shared memory** — the 99 KiB scratchpad, staging operands for the tensor
  cores. Exceed it and the launch **fails outright** with
  `out of resource: shared memory`. No graceful degradation.

A common early mistake (made while writing this note) is to budget `acc` and
`qk` as shared memory. They are registers. The predicted 144 KB for 128×128 was
really 80 KB of shared plus register pressure.

### Occupancy is why the biggest tile loses

**Occupancy** is how many programs are resident on one SM simultaneously. An SM
holds several and switches instantly between them: when program A stalls on an
HBM fetch, the SM runs program B. That is how GPUs hide memory latency — not by
being fast, but by always having other work ready.

At 40 KB per program, two fit in 99 KiB. At 80 KB, one fits, and there is
nothing to switch to during a stall.

> **The fastest configuration is 64×64 — neither the smallest nor the largest.**
> Smaller (32×32) fits four programs per SM but re-loads K/V too often.
> Larger (128×128) amortizes well but drops to one program per SM *and* spills
> 50 registers to HBM. The optimum is a compromise between reuse (wants big
> tiles) and occupancy plus register pressure (want small tiles), and it moves
> with dtype, head dimension, and GPU. This is why Triton ships an autotuner.

This is also the mechanism behind FlashMoBA's sm_86 special case
(`docs/flashmoba-digest.md` §4.3): capacity, not arithmetic.

---

## 4. Warps, divergence, and why `tl.where` is not an `if`

A **warp** is 32 threads executing in lockstep. The GPU never schedules
individual threads; the warp is the atomic scheduling unit, and all 32 lanes
issue the same instruction each cycle on different data (**SIMT** — Single
Instruction, Multiple Threads).

You do not write warps in Triton. `num_warps` (default 4, so 128 threads per
program) determines how your tile work is spread across them.

**Warp divergence.** If lanes in a warp take different branches, the hardware
runs *both* paths serially, masking off inactive lanes each time.

```
lanes:      0  1  2  3  ...  31
condition:  T  T  F  T  ...  F
            pass 1: run "if"   -- lanes 2, 31 idle
            pass 2: run "else" -- lanes 0, 1, 3 idle
```

**Registers are per thread.** The 255-register cap applies to each of the 128
threads, so a program at 255 registers uses 32,640 of the SM's 65,536 — which
independently caps occupancy at 2, agreeing with the shared-memory arithmetic.

**`tl.where` is a predicated select, not a branch.** Both values are computed
and the hardware picks per lane with a single instruction. There is no jump, so
nothing serializes — you pay only for computing both sides.

| | cheap branch bodies | expensive branch bodies |
|---|---|---|
| **condition varies within warp** | `tl.where` — pay both, no serialization | genuinely bad; restructure the problem |
| **condition uniform across warp** | either | real `if` — skip the work entirely |

Our kernel uses both cells correctly. The causal mask uses `tl.where` (cheap:
a computed value versus a constant). The real saving uses a **uniform** branch:

```python
hi = (pid_m + 1) * BLOCK_M if CAUSAL else S
```

`pid_m` is identical for every lane, so this costs nothing and skips whole tiles
rather than masking them.

> **The principle that carries into MoBA:** make expensive decisions uniform per
> program, and use `where` only for cheap per-element cleanup. In a query-major
> MoBA kernel, "does this query need block *j*?" is a non-uniform condition
> guarding an expensive body — the bad cell, with no `where` trick available.
> Gather-by-destination moves it to the bottom-right cell: the program *is*
> block `j`, and every query it holds selected `j`, so the condition is uniform.

---

## 5. Online softmax — two safeguards that look alike and are unrelated

The inner-loop update:

```python
m_new = tl.maximum(m_i, tl.max(qk, axis=1))
alpha = tl.exp(m_i - m_new)          # correction factor
p     = tl.exp(qk - m_new[:, None])

l_i = l_i * alpha + tl.sum(p, axis=1)
acc = acc * alpha[:, None]
acc += tl.dot(p, v)
```

Three variants were run at `B=1, H=2, S=512, D=64`, fp16, causal
(`scratchpad/break_softmax.py`), against `torch` SDPA:

**Normal logits**

| variant | max abs err | NaN | Inf | output range |
|---|---:|---:|---:|---|
| correct | 4.883e-04 | 0 | 0 | [-2.91, 2.88] |
| no rescale | **3.831e-01** | 0 | 0 | [-2.91, 2.88] |
| no running max | 1.953e-03 | 0 | 0 | [-2.91, 2.88] |

**Large logits (`q × 30`)**

| variant | max abs err | NaN | Inf | output range |
|---|---:|---:|---:|---|
| correct | 1.953e-03 | 0 | 0 | [-4.79, 3.81] |
| no rescale | **3.583e+00** | 0 | 0 | [-3.84, 3.35] |
| no running max | **nan** | 64527 | 753 | [-2.07, 2.68] |

### The rescale is mathematics

Removing `acc = acc * alpha` gives **800× the error, with no crash, no NaN, and
an output range identical to the correct answer.** The result is still a
weighted average of `V` rows, so it looks entirely healthy; only the weights are
wrong. Tiles processed early were exponentiated against a smaller running max
and never corrected, so they are over-weighted.

The damage grows with logit spread — `3.8e-1` at normal scale, `3.6e+0` at 30×.

> **This is the failure fingerprint to hunt for in the MoBA kernel: plausible,
> bounded, non-crashing, and wrong.** In a training run it appears only as
> "converges worse than expected."

### The running max is arithmetic self-defence

Algebraically, subtracting the max cancels exactly between numerator and
denominator — `softmax` is unchanged. It exists *solely* because `exp()`
overflows. Hence MODE 2 matching the correct answer at normal scale
(`1.9e-3`) and producing **64,527 NaNs** once logits are large: `exp(120) → inf`,
then `inf / inf → NaN`.

Opposite hazards: the max is a bug that hides in testing and explodes on real
data; the rescale is a bug that never explodes and quietly degrades everything.

---

## 6. The cross-program combine, and why atomics cannot do it

In this dense kernel, one program owns a query row for its whole lifetime, so
`m`, `ℓ`, `acc` stay in registers and touch HBM once. In the MoBA inversion a
query's contributions are computed by **different programs** (one per selected
KV block), and they must meet somewhere.

The obvious idea is `atomicAdd`. An **atomic** is a hardware instruction making
a read-modify-write indivisible: without it, two programs doing
`total = total + mine` can both read the old value and one update is silently
lost. Atomics are how GPU programs safely accumulate into a shared destination.

**It cannot work here.** Adding two partial `acc` values raw is precisely the
"no rescale" bug above — summing terms normalized against different maxima.
Correct combination requires, per query: every program's `(m, ℓ, acc)`, then a
global max, then rescaling each partial by `exp(m_local − m_global)` before
summing. That is `alpha` again, applied *across programs* rather than *across
loop iterations*.

Atomics only perform simple operations — add, max, min, exchange. Take-the-max-
then-rescale-then-add is three dependent steps needing each participant's `m`,
and splitting it into separate atomics reintroduces the race.

**Therefore each program must emit three things per query, and normalization is
deferred to a separate pass:**

| emitted | meaning |
|---|---|
| `m` | which exponent scale this partial is on |
| `ℓ` | this program's share of the softmax denominator |
| `acc` | un-normalized contribution |

A program must *not* divide by its own `ℓ` first. That would discard the
information needed to reweight it.

This is the same structure FlashDecoding uses for long-context inference, and
what MoBA's Algorithm 1 calls `combine_with_online_softmax`. Independent designs
converging is a good sign the constraint is hardware, not taste.

---

## 7. Summary

1. You write one program; the grid says how many run. They are independent.
2. The `S × S` matrix is never materialized — one on-chip tile at a time.
3. Causal attention saves by *not loading* future tiles, not by masking them.
4. `shared = 2·D·BLOCK_M + 8·D·BLOCK_N`; `BLOCK_N` is 4× more expensive.
5. Registers spill silently to HBM; shared memory overflow fails the launch.
6. The fastest tile balances reuse against occupancy — here 64×64, not 128×128.
7. Warps are 32 lanes in lockstep; `tl.where` is a select, not a branch; make
   expensive decisions uniform per program.
8. The rescale is mathematics; the running max is overflow protection.
9. Cross-program combination needs `(m, ℓ, acc)` and a separate pass. Atomics
   are structurally incapable of it.

---

## Glossary

| Term | Meaning |
|---|---|
| **Program / thread block** | One instance of your kernel; the unit scheduled to an SM. |
| **Grid** | How many program instances to launch. |
| **Warp** | 32 threads executing in lockstep; the atomic scheduling unit. |
| **SIMT** | Single Instruction, Multiple Threads — one instruction, 32 lanes of data. |
| **Warp divergence** | Lanes in a warp taking different branches; both paths run serially. |
| **Occupancy** | How many programs are resident per SM; the source of latency hiding. |
| **Register spilling** | Registers overflowing to HBM-backed "local memory". Silent slowdown. |
| **Software pipelining** | Prefetching tile `n+1` while computing tile `n`; costs double buffers. |
| **Predicated select** | Compute both values, pick per lane — no branch, no serialization. |
| **Atomic** | Indivisible read-modify-write; prevents lost updates, but only simple ops. |
| **Online softmax** | Running max/sum formulation allowing softmax tile by tile. |

---

## Check yourself

1. Doubling `BLOCK_N` from 64 to 128 costs four times as much shared memory as
   doubling `BLOCK_M`. Why?
2. A kernel compiles, runs, produces no NaNs, and is slower than expected. Which
   column of the resource table would you check first?
3. Why can a program not divide by its own `ℓ` before writing its partial out?
4. Why does removing the running max pass every test at normal input scale?
5. Which cell of the divergence table does query-major MoBA fall into, and why
   is there no `tl.where` fix for it?
