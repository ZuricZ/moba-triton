# Why our kernel beats FlashMoBA, derived from the source

**Date:** 2026-09-13
**Shape:** B=1, S=8192, H=4 (H_kv=4), D=64, bf16, `block_size=128`, `top_k=8`, causal

Every number below is either read from source (file and line given), measured
by a script printed at the end, or quoted from a paper. Where something is
inference rather than fact, it says so.

---

## 1. What this answers, and why it was needed

We measured our attention stage at **0.217 ms** and FlashMoBA's at **0.686 ms**
on an A100 at this shape (`docs/a100-results.md:133-138`,
`bench/_a100_stage_split.py`). That is a 3.16x gap against a hand-written CUDA
kernel from the group that published the paper on making MoBA fast.

`docs/a100-results.md:144-150` explained the gap by saying FlashMoBA
"loads the union of every block any of [its queries] selected, then masks away
the rest", discarding 77% of its tensor-core work. **That explanation is wrong**
— their kernel does not do this (section 7). So the result had no valid
explanation. This document derives one.

**Why B=1 appears everywhere.** It is not an assumption. It is the shape the
benchmark actually ran (`bench/compare.py:60`: `BATCH, HEADS, HEAD_DIM = 1, 4, 64`).
Batch multiplies every term in both byte counts by the same factor, so every
*ratio* here is independent of B. Batch does change how much of the working set
fits in cache, which section 8 covers. FlashMoBA's own benchmark uses B=2, 16
heads, head dim 128, fp16 — a shape we have never run (section 9).

---

## 2. The problem both kernels solve

MoBA gives each query its own list of `top_k` key blocks. Queries sitting next
to each other pick different blocks. Any kernel has to decide what one program
(one unit of work on one processor) owns.

**Own a group of queries.** Then that program must fetch every block any of its
queries wanted. If it computes every query against every fetched block and masks
off the pairs nobody asked for, most of the arithmetic is wasted. That waste is
what `docs/divergence-measurements.md` measures, and it is severe (section 7).

**Own a key block.** Then the program fetches one block and serves exactly the
queries that picked it, so nothing is wasted. The cost is that a query's answer
is now produced by `top_k` different programs, none of which can see the others.
Those pieces have to be written somewhere and added up later.

**FlashMoBA does something in between, and this is the key fact.** One program
owns a group of 512 queries (outer loop), walks the key blocks (inner loop), and
for each block fetches **only the queries in its group that picked that block**
(`index_gather_scatter.h:50-75`). So it gets the gather benefit: it does not
compute masked-off pairs. What it gives up is reuse — a block wanted by queries
in 12 different groups is loaded 12 times, once per group.

Group size is the dial. Set it to the whole sequence and you get our K/V reuse
but only `B x H` programs, far too few to fill a GPU. Set it small and you get
parallelism but reload K/V constantly. They picked 512
(`decide_lg_block_m`, `flash_moba_interface.py:20-48`).

**Our design gets full K/V reuse and full parallelism at the same time, and pays
for both with a scratch buffer and a second kernel.** That buffer is the whole
trade, and section 3 prices it.

---

## 3. Why FlashMoBA is built this way

Their paper does not justify this choice; Algorithm 1 introduces
`O_tmp ∈ R^(N×d)` without comment, and never mentions split-KV or FSA-style
designs. So what follows is inference from what each design costs, measured —
not a claim about their reasoning.

Each design needs scratch memory in HBM:

| | scratch buffer | grows with `top_k`? |
|---|---|---|
| ours | `[B, H, S, top_k, D]` bf16 plus two `[B, H, S, top_k]` fp32 (`empty_partials`, `moba_fwd.py:503-536`) | **yes, linearly** |
| FlashMoBA | `[N, H, D]` fp32 (`flash_moba_api.cpp:505`) | **no** |

Evaluated (script in section 11):

| shape | ours | FlashMoBA | ratio |
|---|---:|---:|---:|
| our sweep, B1 S8192 H4 D64 | 0.03 GiB | 0.01 GiB | 4.2x |
| **their benchmark, B2 S512K H16 D128** | **33.00 GiB** | **8.00 GiB** | **4.1x** |
| same, `top_k=12` (MoBA's own default) | **49.50 GiB** | 8.00 GiB | 6.2x |

Their published sweep runs to 524,288 tokens on an 80 GB H100
(`benchmarks/bench_flash_moba.py:69-70`). At that shape our design's scratch
alone takes 33 GiB, and 49.5 GiB at `top_k=12`. Ours has never been run past
65,536 tokens. FSA, which uses our design, reports the same problem: its
intermediate buffer reaches **11 GB** at 64K (arXiv 2508.18224, Appendix E), and
it calls this "more buffer overhead" traded against lower memory access.
FlashMoBA's paper advertises **6.1x less memory** than the original MoBA
(§5.3) — memory is something they sell.

Two further costs of our design, both real:

- **A second kernel.** Our combine pass is 39 MiB of the 113 MiB we move (34%),
  and a separate launch. FlashMoBA has neither; its softmax statistics stay in
  shared memory for the whole group (`flash_fwd_kernel.h:406-434`).
- **Load imbalance.** One program per key block means programs with wildly
  different amounts of work: `docs/load-balance-measurements.md` measures
  **6.2x max/mean consumers per block**. After scheduling it costs about 7%, and
  we mitigate it by dispatching longest-first. FlashMoBA's programs each own
  exactly 512 queries, so they are balanced by construction.

**So this is a trade-off with a clear rationale on both sides, not an oversight.**
We win at this shape on this hardware. Their design is the one that still runs at
512K tokens.

---

## 4. How to count FLOPs and bytes (the method, and its source)

**FLOPs.** Count the multiply-adds in the two matrix multiplies (Q·Kᵀ and P·V) at
2 FLOPs each. This is FlashAttention's convention and FlashMoBA's own benchmark
uses it: `benchmarks/bench_flash_moba.py:23-25`,
`4 * batch * seqlen**2 * nheads * headdim`, halved for causal.

**Bytes.** FlashAttention (arXiv 2205.14135, §3.2, Theorem 2) counts HBM accesses
by walking the algorithm's loop nest: every line that loads or stores a tile
contributes *(tile bytes) x (times that line runs)*. FSA (arXiv 2508.18224,
Appendix E) counts its kernels the same way and sums them. Sections 6 and 7 do
exactly this for both kernels.

This counts bytes **requested** from global memory. The roofline model (Williams,
Waterman, Patterson, CACM 2009) is defined on DRAM traffic *after* caches. The two
differ when a re-read is served by the L2 cache. We cannot measure DRAM traffic:
`ncu` is refused on both machines. Section 8 covers what that leaves open.

---

## 5. Padding: what it is, and why both kernels have it

A **tensor core** is a hardware unit that multiplies small matrices of a fixed
size. It cannot be asked for a fractional tile. FlashMoBA's is
`SM80_16x8x16_F32BF16BF16F32_TN` (`kernel_traits.h:34-35`) and its tiler is
`Tile<Int<16 * kNWarps>, _16, _16>` (`kernel_traits.h:93`), which with 4 warps is
64 rows wide. That is where their `kBlockM = 64` comes from.

The consequence is visible in their inner loop. The accumulator is declared with
a **compile-time** shape and cleared, then the matmul runs over all of it:

```cpp
Tensor acc_s = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{});
clear(acc_s);                                    // flash_fwd_kernel.h:559-560
... gemm(acc_s, tSrQ, tSrK, ...)                 // :574   full 64 x 128 tile
... mask(acc_s, ..., cur_idx_len, ...)           // :582   masking happens after
```

`cur_idx_len` is how many queries this tile actually has. It masks the **result**
and the loads. The multiply itself always runs 64 rows. If only 50 queries picked
this block in this group, 14 rows of tensor-core work are computed and thrown
away. **That is padding.**

Ours works the same way — `GROUP` is a compile-time constant, `qk` is a full
`GROUP x BLOCK_SIZE` tile, and `live = offs < n` masks loads and stores, not the
`tl.dot` (`moba_fwd.py`, attend kernel).

**Measured**, over the real routing at this shape (script in section 11):

| | tiles | rows per tile | row-slots issued | useful | **padding** |
|---|---:|---:|---:|---:|---:|
| FlashMoBA (chunks of 64) | 4,934 | 64 | 315,776 | 247,808 | **21.5%** |
| ours, GROUP=16 | 15,603 | 16 | 249,648 | 247,808 | 0.7% |
| ours, GROUP=32 | 7,862 | 32 | 251,584 | 247,808 | 1.5% |
| ours, GROUP=64 | 3,994 | 64 | 255,616 | 247,808 | **3.1%** |
| ours, GROUP=128 | 2,060 | 128 | 263,680 | 247,808 | 6.0% |

Which GROUP the autotuner picked on the A100 was not recorded, so all four are
shown; the worst is 6.0%.

**Why theirs is higher at the same tile size.** Ours fills tiles from one block's
entire list of queries — about 968 per block, so few tiles are part-full. Theirs
fills tiles from one group's slice of that list — about 114 per (group, block),
so most tiles are part-full. Splitting queries into 512-groups fragments the tile
fill and the K/V reuse at once. **They are the same fact.**

---

## 6. FLOPs (the same useful work for both)

Per (query, block) pair:

```
Q row (1 x 64)  times  K tileᵀ (64 x 128)  =  128 x 64 = 8,192 multiply-adds
P row (1 x 128) times  V tile  (128 x 64)  =  128 x 64 = 8,192 multiply-adds
                                             total      16,384 multiply-adds
                                                      = 32,768 FLOP
```

Measured pair count **P = 247,808** (7.56 blocks per query; early queries have
fewer past blocks to choose from). Both kernels compute the same selection: the
fairness check found identical error against the oracle, 0.0478 for both
(`docs/a100-results.md:109-112`).

Useful work = 247,808 x 32,768 = **8.12 GFLOP**.

This counts the query's own block as a full tile although causal masking removes
about half of it, overstating by at most 6.6%. It is the same overstatement for
both kernels, so it cancels in the comparison.

Including padding from section 5, FlashMoBA issues 315,776 row-tiles of matmul
against our 255,616 at GROUP=64 — **23.5% more matmul than we issue**. That is
real but small next to the byte gap below.

---

## 7. Where the "77%" came from, and why it does not apply

`docs/divergence-measurements.md:67` reports utilization 0.233 at S=8192, i.e.
77% discarded. It is a real measurement, computed by `bench/divergence.py`:

```python
tiles = selected.view(b, h, s // block_m, block_m, n)
union = tiles.any(dim=3)                          # blocks ANY query in the tile wanted
computed = union.sum(dim=-1) * block_m            # every query x every block in the union
useful   = tiles.sum(dim=(-1, -2))                # pairs actually selected
utilization = useful.sum() / computed.sum()
```

Read the third line. It models a kernel that computes **every query in the tile
against every block in the tile's union**. That is the cost of a design that
fetches the union and masks.

**FlashMoBA does not do that.** For each key block it binary-searches the block's
sorted query list for the queries inside its own group
(`index_gather_scatter.h:50-75`) and processes only those. A query that did not
pick the block never enters the tile. Its only waste is the part-full tile
padding of section 5: **21.5%, not 77%**.

The 77% figure still describes something real — a query-major kernel that masks
rather than gathers. It does not describe FlashMoBA, and
`docs/a100-results.md:144-150` attaches it to FlashMoBA.

---

## 8. Bytes moved

**Units used throughout:** one query row is `D x 2 = 128 B` in bf16 and `256 B`
in fp32; one K (or V) block tile is `128 keys x 64 x 2 B = 16,384 B`;
query rows `S x H = 32,768`; `P = 247,808`.

### 8.1 Ours — attend, then combine (`src/moba/kernels/moba_fwd.py`)

Grid is one program per (head, key block): 4 x 64 = **256 programs**, and all 256
blocks were picked by someone (measured). Each loads K and V **once**, before the
query loop, then for each query that picked the block gathers its Q row and
stores its piece of the answer.

| step | bytes per run | runs | MiB |
|---|---:|---:|---:|
| fill `part_m` with −inf, fp32, 8 slots (`:618`) | 4 | 262,144 | 1.00 |
| load K tile | 16,384 | 256 | 4.00 |
| load V tile | 16,384 | 256 | 4.00 |
| read routing indices (3 x int32) | 12 | P | 2.84 |
| gather Q row | 128 | P | 30.25 |
| store `m`, `l` (fp32) | 8 | P | 1.89 |
| store `acc` (bf16) | 128 | P | 30.25 |
| combine: read `m`, two passes | 8 | 262,144 | 2.00 |
| combine: read `l`, all 8 slots | 4 | 262,144 | 1.00 |
| combine: read `acc`, all 8 slots | 128 | 262,144 | 32.00 |
| combine: write O (bf16) | 128 | 32,768 | 4.00 |
| **total** | | | **113.2** |

262,144 = 32,768 query rows x 8 slots. The combine reads every slot, including
those no program wrote.

### 8.2 FlashMoBA — one kernel (`csrc/flash_moba/src/flash_fwd_kernel.h`)

Grid is one program per (head, group of 512 queries):
`dim3 grid(num_lg_m_block, b, h)` (`flash_fwd_launch_template.h:52`). Per key
block it gathers its group's queries in chunks of 64, and per chunk it loads K
and V, gathers Q rows, **loads the running output from HBM**, computes, and
**stores the running output back** (`:544-590`). A final pass converts the
running output and writes O and LSE.

K/V loads are therefore once per chunk:

```
loads = sum over (group, block) of ceil(queries of that group that picked it / 64)
```

Worked by hand: a block picked by 150 queries, 100 in group A and 50 in group B.
Ours: one program owns the block, **1 load**. Theirs: `ceil(100/64) = 2` plus
`ceil(50/64) = 1`, **3 loads**.

Measured over the real routing:

| | K/V tile loads | vs ours |
|---|---:|---:|
| ours | 256 | 1x |
| theirs, if chunks were unlimited (one per (group, block) with any picker) | 2,176 | 8.5x |
| **theirs, actual (chunks of 64)** | **4,934** | **19.3x** |

| step | bytes per run | runs | MiB |
|---|---:|---:|---:|
| zero-fill `o_tmp`, fp32 (`flash_moba_api.cpp:505`) | 256 | 32,768 | 8.00 |
| load K tile (`:548`) | 16,384 | 4,934 | 77.09 |
| load V tile (inner `n_block` loop) | 16,384 | 4,934 | 77.09 |
| read row indices (int32) | 4 | P | 0.95 |
| gather Q row (`make_gather_tensor_Q`) | 128 | P | 30.25 |
| load running output, fp32 (`o_tmp_load`) | 256 | P | 60.50 |
| store running output, fp32 (`o_tmp_store`) | 256 | P | 60.50 |
| final pass: read `o_tmp` | 256 | 32,768 | 8.00 |
| final pass: write O (bf16) | 128 | 32,768 | 4.00 |
| final pass: write LSE (fp32) | 4 | 32,768 | 0.13 |
| **total** | | | **326.5** |

Their per-query softmax statistics stay in shared memory
(`flash_fwd_kernel.h:406-434`) and cost no HBM traffic.

---

## 9. Does the count explain the measured time?

| | bytes requested | measured time | bytes / time |
|---|---:|---:|---:|
| ours | 113.2 MiB | 0.217 ms | **510 GiB/s** |
| FlashMoBA | 326.5 MiB | 0.686 ms | **465 GiB/s** |
| ratio | **2.88x** | **3.16x** | |

**Both kernels move bytes at about the same rate, and the one requesting 2.88x
the bytes takes 3.16x as long.** The remaining ~10% is not explained here;
their extra 23.5% of matmul (section 6) is a candidate but was not isolated.

That both sustain a similar rate is also the answer to "CUDA should beat Triton":
their CUDA does not move bytes more cheaply than our Triton. It moves more of
them, because of the design trade in section 3.

Arithmetic intensity, useful FLOP per requested byte:

```
ours       8.12 GFLOP / 113.2 MiB =  68.4 FLOP/byte
FlashMoBA  8.12 GFLOP / 326.5 MiB =  23.7 FLOP/byte
A100 ridge 204 TFLOP/s / 1239 GiB/s (both measured) = 153.5 FLOP/byte
```

Both far below the ridge, so **both are memory bound**. `bench/roofline.py`
prints 186.2 for FlashMoBA because `traffic_flash_moba` omits their running
output (137 MiB) and their K/V reloads (146 MiB).

---

## 10. What this does not establish

- **Requested bytes are not DRAM bytes.** The A100 has 40 MB of L2 (NVIDIA A100
  whitepaper). At B=1 here, K+V is 8 MiB and their `o_tmp` is 8 MiB, so many of
  their re-reads may be served by cache and never reach DRAM; our 32 MiB buffer is
  near L2's size. True DRAM traffic is lower than both tables by an unmeasured
  amount. What holds is that the requested-byte ratio predicts the time ratio here.
- **One shape, one batch.** H=32, D=128 uses different FlashMoBA tile traits and
  has not been derived. Larger B or H makes everything exceed cache, which would
  strengthen the count, but that is untested.
- **We have never run their benchmark shape** (B=2, 16 heads, D=128, fp16, up to
  512K, forward+backward, vs real FlashAttention-2). Their paper reports FlashMoBA
  2.0x faster than FA-2 at 64K (49 vs 99 ms, Figure 4); we measured 1.22x against
  torch SDPA on the A100. Until that 1.6x discrepancy is explained, the headline
  speedup should be stated as "at `[1, S, 4, 64]` on an A100".
- **MoonshotAI and Triton-MoBA are not derived here.**

---

## 11. Reproducing every measured number

Runs on the laptop; needs only the routing, not either kernel.

```python
"""Pairs, K/V tile loads, padding, and scratch memory for both designs."""
import torch
from moba.config import MoBAConfig
from moba.naive import moba_selection

cfg = MoBAConfig(block_size=128, top_k=8, dtype=torch.bfloat16)
S, H, D, GROUP_Q, THEIR_CHUNK = 8192, 4, 64, 512, 64
g = torch.Generator(device="cuda").manual_seed(0)
q, k = (torch.randn(1, S, H, D, generator=g, device="cuda", dtype=cfg.dtype)
        for _ in range(2))
sel = moba_selection(q, k, cfg)                      # [1, H, S, num_blocks]
P = int(sel.sum())                                                    # 247808

per_block = sel.sum(dim=2).double()                                   # ours
per_gb = sel.view(1, H, S // GROUP_Q, GROUP_Q, -1).sum(dim=3).double()  # theirs
print(int((sel.sum(dim=2) > 0).sum()))                    # ours loads     256
print(int(torch.ceil(per_gb / THEIR_CHUNK).sum()))        # theirs loads  4934
for group in (16, 32, 64, 128):                           # padding, ours
    rows = int(torch.ceil(per_block / group).sum()) * group
    print(group, f"{(rows - P) / rows * 100:.1f}%")

GIB = 1024 ** 3                                           # scratch memory
for B, s, h, d, tk in ((1, 8192, 4, 64, 8), (2, 524288, 16, 128, 8),
                       (2, 524288, 16, 128, 12)):
    ours = (B * h * s * tk * d * 2 + 2 * B * h * s * tk * 4) / GIB
    print(f"ours {ours:.2f} GiB   theirs {B * s * h * d * 4 / GIB:.2f} GiB")
```

## 12. Corrections owed to other files

| file | claim | status |
|---|---|---|
| `docs/a100-results.md:144-150` | FlashMoBA loads the union and discards 77% | **wrong** (section 7). It gathers; its waste is 21.5% padding. The cost is K/V reloads and the fp32 running-output round trip. |
| `docs/flashmoba-digest.md` §2 | grid is (batch, head, KV block) | **wrong**: `dim3 grid(num_lg_m_block, b, h)`, `flash_fwd_launch_template.h:52` |
| `bench/roofline.py` `traffic_flash_moba` | 44 MiB, 186.2 FLOP/byte, "compute bound" | **wrong** (section 9); also reused for `moonshot_efficient` |
| every measured latency, the fairness checks, all kernel code | — | **unchanged** |

## Sources

- Dao et al., *FlashAttention*, arXiv 2205.14135, §3.2, Theorem 2, Algorithm 1
- Xiao, Guo, Mazaheri, Han, *Optimizing Mixture of Block Attention* (FlashMoBA),
  arXiv 2511.11571, §4.2 Algorithm 1, §5.3. Gives no traffic or FLOP analysis of
  its kernel, and no justification for `O_tmp`.
- *FSA*, arXiv 2508.18224, §3.1-3.2 and Appendix E (same loop order as ours;
  reports the buffer cost)
- Williams, Waterman, Patterson, *Roofline*, CACM 52(4), 2009, doi:10.1145/1498765.1498785
- NVIDIA A100 Tensor Core GPU Architecture whitepaper (L2 = 40 MB)
