# The backward pass, measured

**Date:** 2026-08-23
**Device:** RTX 3050 Ti Laptop, sm_86 | bf16 | 1x4 heads, head_dim 128,
block_size 128, top_k 8
**Source:** `src/moba/kernels/moba_bwd.py`

## Shape of the problem

The loop inversion pays off in the opposite direction from the forward. `dK`
and `dV` are indexed by *key*, so a program owning a key range owns both
gradients outright -- it accumulates them in registers across every query that
selected the block and stores each once. No atomics, no partials, no
contention, for two full tensors. Only `dQ` scatters, and its merge is a plain
sum rather than the forward's rescaled two-pass, because the softmax is
recomputed against the query's global `lse` instead of a per-block maximum.

Two constraints the forward does not have:

* **A program cannot own a whole MoBA block.** The backward needs K, V and both
  of their transposes at once. Measured at block_size 128, head_dim 128, the
  fixed cost before a single query loads is 98,304 bytes against a 101,376
  budget -- room for four query rows where `tl.dot` needs sixteen. Nothing
  compiles. Hence `BWD_KEY_TILE`.
* **Sub-tiling breaks the forward's one-writer invariant.** Every sub-tile
  produces a `dQ` contribution for the same (query, slot) address, and they
  overwrite rather than sum. `dK` and `dV` are untouched, being key-indexed, so
  it damages precisely the tensor that was already scattered. Fixed by widening
  the `dQ` slot axis to `top_k * tiles`.

## Where the time goes

| seq_len | total | delta | main kernel | reduce_dq |
|---:|---:|---:|---:|---:|
| 2048 | 2090 us | 36 us (2%) | 1841 us (88%) | 221 us (11%) |
| 4096 | 4754 us | 63 us (1%) | 4261 us (90%) | 434 us (9%) |
| 8192 | 10176 us | 112 us (1%) | 9241 us (91%) | 857 us (8%) |

Measured before tuning. The main kernel is the only stage worth working on.

## Tuning, and a trap

The backward first shipped with the forward's chunk height and Triton's default
warp count. Neither transfers: the backward runs four matmuls per iteration
where the forward runs two, so it wants more warps on a narrower tile.

Main kernel alone, S=4096, best of 60 compiled configurations:

| KEY_TILE | GROUP | warps | stages | time | smem |
|---:|---:|---:|---:|---:|---:|
| 32 | 16 | 4 | 3 | **1762 us** | 36 KB |
| 32 | 32 | 4 | 2 | 1791 us | 48 KB |
| 64 | 32 | 8 | 3 | 1899 us | 72 KB |
| 64 | 64 | 4 | 3 | 4256 us | *shipped* |

On this evidence `KEY_TILE=32` wins, and the reason is legible: half the shared
memory, so two programs per SM instead of one.

**It is the wrong answer.** `KEY_TILE` is not local to the kernel being
measured. Halving it doubles the `dQ` partial buffer's slot axis, so
`reduce_dq` reads twice as much and the allocation doubles. Measured on the
whole backward:

| seq_len | kt64/g32/w8 | kt32/g16/w4 | kt16/g16/w4 |
|---:|---:|---:|---:|
| 2048 | **1075 us**, 32 MB | 1178 us, 64 MB | 1922 us, 128 MB |
| 4096 | **2363 us**, 64 MB | 2626 us, 128 MB | 4310 us, 256 MB |
| 8192 | **5101 us**, 128 MB | 5658 us, 256 MB | 9149 us, 512 MB |

`KEY_TILE=64` is 10% faster end to end and uses half the memory. Tuning the
stage that is 91% of the time, in isolation, chose wrongly -- and it chose
wrongly with a plausible mechanism attached, which is what made it convincing.

The actual win was never the key tile. It was `GROUP=32` at 8 warps instead of
`GROUP=64` at 4, at the same key tile the kernel already used.

## Result

Against `torch.nn.functional.scaled_dot_product_attention`'s backward:

| seq_len | dense bwd | ours, shipped | ours, tuned | vs dense | bwd / our fwd |
|---:|---:|---:|---:|---:|---:|
| 2048 | 915 us | 2092 us | 1085 us | 0.84x | 2.01x |
| 4096 | 3445 us | 4723 us | 2385 us | **1.44x** | 2.18x |
| 8192 | 13172 us | 10092 us | 5105 us | **2.58x** | 2.29x |

The backward now costs 2.0-2.3x its own forward, against a theoretical ratio
near 2.5x -- the structural overhead is essentially gone, and what remains
tracks the extra matmuls the mathematics requires.

Below 4K we are still slower than dense, for the same reason the forward is
least impressive there: at S=2048 with block_size 128 there are 16 blocks and
top_k 8, so half of them are selected and there is little sparsity to exploit.

## Correctness

Gradients are held to `moba.tolerance`'s policy, unchanged from the forward:
within `TOLERANCE_FACTOR` of what the same algorithm in bfloat16 already costs,
against an fp64 oracle. Measured 31-68% of that budget across the gate sweep at
two shapes. Routing is held fixed by passing the kernel's own selection mask to
the oracle, so a near-tie cannot be mistaken for a numerical difference.

The kernel is deterministic *given an index*, not across index builds:
`build_csc_fused` lets queries claim column positions with an atomic cursor, so
column order varies, and `dK`/`dV` are sums down the query axis where floating
point addition is not associative. `dQ` is per-query and stays bitwise exact.
`tests/test_backward.py` states both halves.
