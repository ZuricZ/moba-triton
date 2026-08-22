"""Builds the CSC index on device, from selected block ids, without a sync.

The PyTorch builder takes a dense `[B, H, S, num_blocks]` boolean mask and
*searches* it for the selected entries with `nonzero()`. That search is what
forces a host synchronization: the number of results is data-dependent, so
PyTorch must copy a count back to the CPU before it can allocate room for them.
Measured on this WSL box, one such round trip costs ~230 us against ~6 us for
the same arithmetic kept on device -- a 37x penalty for letting the CPU look.

Nothing here needs that. The fused gate already emits a *list* of selected
block ids, so the entries are known and counted; and the total is bounded above
by `S * top_k`, which is known before anything runs. Three kernels follow
FlashMoBA's stage 3 (its Algorithm 4): count, prefix-sum into offsets, then
scatter. All on device, no `nonzero`, no round trip, no dense mask.

ONE GUARANTEE IS DROPPED. The PyTorch builder emitted each column's query
indices in ascending order, which fell out for free from the memory order it
read the dense mask in. Here queries claim their slot with an atomic counter,
so a column's entries come out in whatever order the programs raced. That is
safe, and both halves of why are already measured:

* Results do not change. The attention kernel addresses its output by
  `row * K_SLOTS + slot` -- who the query is and which of its slots this is.
  Neither depends on position within a column, so permuting a column moves no
  value.
* Speed does not change. Scattered order means the gathered query rows are no
  longer ascending. `docs/gather-cost-measurements.md` measured a *fully*
  randomized permutation at 1.03x, because a query row at these head dims is
  one cache line either way. That measurement was taken for this question.

FlashMoBA accepts the same shuffling for the same reason.
"""

import torch
import triton
import triton.language as tl

from moba.config import MoBAConfig
from moba.device import TENSOR_CORE_TILE
from moba.kernels.gate import block_representatives_triton, gate_select
from moba.kernels.index import CSCIndex

# Queries per program for the count and scatter passes. Both are pure
# scatter-with-atomics over `S * top_k` entries, so this only trades program
# count against per-program work; it is not a tuning-sensitive number.
INDEX_BLOCK_M = 128


@triton.jit
def _count_columns(
    BlockIdx, ColNnz,
    S,
    TOP_K: tl.constexpr, N_BLOCKS: tl.constexpr, BLOCK_M: tl.constexpr,
):
    """Counts how many queries selected each block.

    Pass 1 of 3. `ColNnz` must be zeroed by the caller.
    """
    offs = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    pid_bh = tl.program_id(1)
    live = offs < S

    for j in tl.static_range(TOP_K):
        blk = tl.load(BlockIdx + pid_bh * S * TOP_K + offs * TOP_K + j,
                      mask=live, other=-1)
        ok = live & (blk >= 0)
        # A padding slot carries -1, which would address before the array.
        # Clamping keeps the pointer in range; the mask keeps the write out.
        tl.atomic_add(ColNnz + pid_bh * N_BLOCKS + tl.where(ok, blk, 0), 1,
                      mask=ok)


@triton.jit
def _offsets_and_order(
    ColNnz, ColOffsets, Order,
    N_BLOCKS: tl.constexpr, N_POW2: tl.constexpr,
):
    """Turns counts into column offsets, and ranks columns by size.

    Pass 2 of 3, one program per (batch, head). Both outputs are reductions
    over `num_blocks` -- 32 at S=4096 -- so a single program does the whole
    thing and the prefix sum needs no multi-block scan.
    """
    pid_bh = tl.program_id(0)
    cols = tl.arange(0, N_POW2)
    valid = cols < N_BLOCKS

    cnt = tl.load(ColNnz + pid_bh * N_BLOCKS + cols, mask=valid, other=0)

    # Exclusive prefix sum: column j starts where every earlier column ended.
    offsets = tl.cumsum(cnt, axis=0) - cnt
    tl.store(ColOffsets + pid_bh * (N_BLOCKS + 1) + cols, offsets, mask=valid)
    tl.store(ColOffsets + pid_bh * (N_BLOCKS + 1) + N_BLOCKS, tl.sum(cnt))

    # Longest-first dispatch (D8): the heaviest columns start first, so the
    # long pole is not left until the end. Repeated maximum extraction rather
    # than a sort -- num_blocks is small, and the explicit lowest-index
    # tie-break makes the order reproducible run to run. An unstable order here
    # would vary the kernel's dispatch between runs, which is exactly what a
    # race condition looks like from the outside.
    #
    # Invalid columns sit at -1 so every real column, including empty ones,
    # outranks them and the loop never runs out of candidates.
    work = tl.where(valid, cnt, -1)
    for r in tl.static_range(N_BLOCKS):
        best = tl.max(work, axis=0)
        pick = tl.min(tl.where(work == best, cols, N_POW2), axis=0)
        tl.store(Order + pid_bh * N_BLOCKS + r, pick)
        work = tl.where(cols == pick, -1, work)


@triton.jit
def _scatter_pairs(
    BlockIdx, ColOffsets, Cursor, RowIdx, SlotIdx,
    S, T,
    TOP_K: tl.constexpr, N_BLOCKS: tl.constexpr, BLOCK_M: tl.constexpr,
):
    """Writes each (query, slot) pair into its column's region.

    Pass 3 of 3. `Cursor` must be zeroed by the caller; `atomic_add` returns
    the pre-increment value, which is the pair's position within its column.
    """
    offs = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    pid_bh = tl.program_id(1)
    live = offs < S

    for j in tl.static_range(TOP_K):
        blk = tl.load(BlockIdx + pid_bh * S * TOP_K + offs * TOP_K + j,
                      mask=live, other=-1)
        ok = live & (blk >= 0)
        safe = tl.where(ok, blk, 0)

        within = tl.atomic_add(Cursor + pid_bh * N_BLOCKS + safe, 1, mask=ok)
        start = tl.load(ColOffsets + pid_bh * (N_BLOCKS + 1) + safe, mask=ok,
                        other=0)
        dst = start + within

        tl.store(RowIdx + pid_bh * T + dst, offs.to(tl.int32), mask=ok)
        tl.store(SlotIdx + pid_bh * T + dst, j, mask=ok)


def build_csc_from_indices(
    block_idx: torch.Tensor, num_blocks: int, cfg: MoBAConfig
) -> CSCIndex:
    """Assembles a CSC index from per-query selected block ids.

    Args:
        block_idx: Selected block ids shaped [B, H, S, top_k], int32, -1
            padded, as `gate_select` returns.
        num_blocks: Number of KV blocks.
        cfg: Configuration supplying `top_k`.

    Returns:
        A populated CSCIndex. Query indices within a column are in arbitrary
        order; see the module docstring.
    """
    b, h, s, top_k = block_idx.shape
    device = block_idx.device
    bh = b * h
    total = s * cfg.top_k

    block_idx = block_idx.contiguous()
    col_nnz = torch.zeros(b, h, num_blocks, dtype=torch.int32, device=device)
    col_offsets = torch.zeros(b, h, num_blocks + 1, dtype=torch.int32,
                              device=device)
    order = torch.zeros(b, h, num_blocks, dtype=torch.int32, device=device)
    cursor = torch.zeros(b, h, num_blocks, dtype=torch.int32, device=device)
    row_indices = torch.zeros(b, h, total, dtype=torch.int32, device=device)
    slot_of_pair = torch.zeros(b, h, total, dtype=torch.int32, device=device)

    grid = (triton.cdiv(s, INDEX_BLOCK_M), bh)
    _count_columns[grid](
        block_idx, col_nnz, s,
        TOP_K=top_k, N_BLOCKS=num_blocks, BLOCK_M=INDEX_BLOCK_M,
    )
    _offsets_and_order[(bh,)](
        col_nnz, col_offsets, order,
        N_BLOCKS=num_blocks,
        N_POW2=max(TENSOR_CORE_TILE, triton.next_power_of_2(num_blocks)),
    )
    _scatter_pairs[grid](
        block_idx, col_offsets, cursor, row_indices, slot_of_pair, s, total,
        TOP_K=top_k, N_BLOCKS=num_blocks, BLOCK_M=INDEX_BLOCK_M,
    )

    return CSCIndex(
        col_nnz=col_nnz,
        col_offsets=col_offsets,
        row_indices=row_indices,
        slot_of_pair=slot_of_pair,
        order=order,
        num_blocks=num_blocks,
        max_slots=cfg.top_k,
    )


def build_csc_fused(
    q: torch.Tensor, k: torch.Tensor, cfg: MoBAConfig
) -> CSCIndex:
    """Builds the index entirely on device, replacing `build_csc`.

    All three stages are Triton: mean-pool the keys into block
    representatives, rank them per query, then transpose into CSC. No
    intermediate leaves the GPU, so the host synchronizations the PyTorch
    path needed -- ~230 us each on WSL -- are gone entirely.

    Args:
        q: Queries shaped [B, H, S, D].
        k: Keys shaped [B, H, S, D].
        cfg: Configuration.

    Returns:
        A populated CSCIndex.
    """
    kbar = block_representatives_triton(k, cfg)
    block_idx = gate_select(q, kbar, cfg)
    return build_csc_from_indices(block_idx, kbar.shape[2], cfg)
