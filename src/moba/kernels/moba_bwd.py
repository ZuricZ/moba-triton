"""KV-block-major MoBA backward.

The same inversion as the forward, and it pays off in the opposite direction.

In the forward, one program owning a KV block makes the K and V loads cheap and
leaves a query's *output* scattered across programs, which is the bill
`combine_partials` pays. In the backward the ownership is worth more. `dK` and
`dV` are indexed by *key*, so the program that owns a block owns both of its
gradients outright: it accumulates them in registers across every query that
selected the block and stores each exactly once. No atomics, no partials, no
contention -- for two full tensors.

`dQ` is the one that scatters now, because a query collects gradient from every
block it selected. So the bill moves from two tensors to one, and gets cheaper
on the way: the forward's merge had to rescale each partial by
`exp2(m_local - m_global)` because every block exponentiated against its own
maximum. Here the softmax is recomputed against the query's *global* `lse`,
saved by the forward, so every `dQ` partial is already on the right scale and
the merge is a plain sum.

FSA reaches the same split -- `dq_compute_kernel` followed by
`dq_reduce_kernel`, with `dK`/`dV` handled block-locally.

WHAT IS NOT DIFFERENTIATED. The gate. MoBA's routing is a hard top-k, and
Moonshot's reference treats it as such: `gate_top_k_idx` is used only as an
index and no gate score is saved for backward, so gradient flows through the
selected attention and not through the decision to select. That is what makes
this tractable -- the backward sees a fixed sparsity pattern and reuses the
forward's CSC index unchanged.

`dQ` partials are merged with the forward's `part_m` as the written-marker,
exactly as `combine_partials` does, rather than by zeroing the partial buffer.
`part_m` is a factor of `head_dim` smaller than the buffer it guards -- 1 MB
against 64 MB at S=8192 -- so carrying it from the forward is far cheaper than
writing the accumulator twice.
"""

import os

import torch
import triton
import triton.language as tl

from moba.config import MoBAConfig
from moba.gqa import head_group_size
from moba.kernels.routing import CSCIndex
from moba.device import shared_memory_budget
from moba.kernels.moba_fwd import _best_of_three
from moba.kernels.tuning import (
    LOG2_E,
    choose_combine_tile,
    estimate_shared_memory,
    launch_policy,
)

# Queries per program for the preprocess and dQ-reduce passes. Both are pure
# streaming over [B, H, S, ...] with no matmul, so this only trades program
# count against per-program work.
BWD_BLOCK_M = 64

# Keys per backward program. The forward gives one program a whole MoBA
# block; the backward cannot, and the reason is shared memory rather than
# anything about the algorithm.
#
# The forward holds K and V, 64 KiB at block_size 128, head_dim 128. The
# backward additionally needs their *transposes* -- dP = dO V^T and dQ = dS K
# -- and Triton stages each as its own tile. Measured, the fixed cost before a
# single query is loaded is exactly 96 KiB against this GPU's 99 KiB budget,
# leaving room for four query rows where `tl.dot` needs sixteen. Nothing
# compiles.
#
# So a program owns a *sub-tile* of a block's keys instead of all of them.
# dK and dV stay owned outright -- each program writes a disjoint key range --
# and the cost is that Q and dO are gathered once per sub-tile rather than once
# per block. Some of that is bought back in parallelism: splitting multiplies
# the program count, and 20 SMs are easy to leave idle otherwise.
#
# 64 rather than 32, and the two do not disagree by much, but *why* 32 loses is
# worth recording. Measured on the main kernel alone, 32 wins: it halves shared
# memory, which lifts occupancy from one program per SM to two. Measured on the
# whole backward, 32 loses -- because this constant is not local to that kernel.
# Halving it doubles the slot axis of the dQ partial buffer, so `reduce_dq`
# reads twice as much and the allocation doubles, 128 MB to 256 MB at S=8192.
# Tuning the stage that is 91% of the time, in isolation, chose wrongly.
BWD_KEY_TILE = 64

# Queries per iteration, and the launch shape. All measured, none inherited:
# the backward first shipped with the forward's chunk height and Triton's
# default warp count, which cost a factor of two.
#
# 32 queries over 8 warps, against the forward's 64 over 4. The backward does
# four matmuls per iteration where the forward does two, so it wants more warps
# on a narrower tile; the forward wants the opposite. Nothing about one kernel's
# tuning transfers to the other.
# Chunk heights offered to the autotuner. Capped below the forward's 128: this
# kernel gathers two tiles an iteration (q and dO) where the forward gathers
# one, so the same height needs more shared memory here.
BWD_AUTOTUNE_GROUPS = (64, 32, 16)

# Warp counts and pipeline depths. Both are tuned rather than fixed, and both
# were fixed until 2026-09-07 at values that measurement no longer supports.
#
# The comment above used to justify `32 queries over 8 warps` as a measured
# result. It was measured -- before the thermal protocol existed, on a card
# whose clock depended on whatever had just run. Re-measured under cool, warm,
# best-of-three, forward plus backward at S=8192:
#
#     GROUP 64, warps 4, stages 2 -> 3.895 ms      <- best
#     GROUP 32, warps 4, stages 3 -> 3.993 ms
#     GROUP 32, warps 8, stages 3 -> 4.649 ms      <- what shipped
#     GROUP 64, warps 8, stages 3 -> 5.637 ms
#
# 19% was sitting in three constants. `num_warps` dominates -- 4 beats 8 at
# every height -- which is the opposite of what the old comment argued from the
# matmul count, and a good reason to let the machine decide.
#
# `num_stages` is tuned too because it interacts: 2 wins at GROUP=64 and loses
# at GROUP=32, an 8% swing either way.
BWD_AUTOTUNE_WARPS = (4, 8)

# Pipeline depth is fixed, not tuned, and that is a cost decision. It swings
# ~8% and it interacts -- 2 wins at GROUP=64, 3 wins at GROUP=32 -- but 64 is
# the height that wins overall, so 2 is the right constant to freeze. Tuning it
# as well doubled the configs and took the test suite from 162 s to 651 s,
# because compilation rather than benchmarking is what a config list costs.
BWD_STAGES = 2


# -------------------------------------------------------------------------
# Device code
# -------------------------------------------------------------------------


@triton.jit
def moba_bwd_preprocess_kernel(O, DO, Delta, S, D: tl.constexpr,
                               BLOCK_M: tl.constexpr, H_Q: tl.constexpr):
    """Computes `delta_i = rowsum(dO_i * O_i)` for every query.

    This is the term the softmax Jacobian contributes: `dS = P * (dP - delta)`,
    where `delta_i = sum_j P_ij dP_ij`. That sum is over keys, which no single
    backward program sees all of -- but it is algebraically equal to
    `sum_d O_id dO_id`, a sum over the head dimension, which every program can
    read directly. FlashAttention-2 makes the same substitution for the same
    reason, and it is why a preprocess pass exists at all.
    """
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    live = offs_m < S
    offs_d = tl.arange(0, D)
    # O and DO are the caller's, so token-major; Delta below is ours.
    base = ((pid_bh // H_Q) * S * H_Q * D + (pid_bh % H_Q) * D
            + offs_m[:, None] * (H_Q * D) + offs_d[None, :])

    o = tl.load(O + base, mask=live[:, None], other=0.0)
    do = tl.load(DO + base, mask=live[:, None], other=0.0)

    delta = tl.sum(o.to(tl.float32) * do.to(tl.float32), axis=1)
    tl.store(Delta + pid_bh * S + offs_m, delta, mask=live)


def _bwd_prune_by_shared_memory(configs, named_args, **kwargs):
    """Drops chunk heights that cannot fit, before anything is compiled.

    Same role as the forward's pruner, with a larger per-height cost: this
    kernel holds q and dO tiles at once. `estimate_shared_memory` covers one
    gathered tile, so the height is charged twice.

    Args:
        configs: Every `triton.Config` offered.
        named_args: Positional kernel arguments, unused.
        **kwargs: The constexprs, where `BLOCK_SIZE` and `D` arrive.

    Returns:
        Surviving configs, never empty -- the smallest is kept if the estimate
        rejects everything, so the compiler gives the verdict.
    """
    del named_args
    block_size, head_dim = kwargs["BLOCK_SIZE"], kwargs["D"]
    budget = shared_memory_budget()
    kept = [c for c in configs
            if 2 * estimate_shared_memory(block_size, c.kwargs["GROUP"],
                                          head_dim, 2) <= budget]
    if not kept:
        return [min(configs, key=lambda c: c.kwargs["GROUP"])]

    # Only the two largest surviving heights are worth benchmarking. Measured,
    # the gap between the best height and the next is ~5% while the gap between
    # warp counts is ~45%, so a third height buys little and costs two more
    # compilations per shape. The smaller heights stay in the config list
    # because they are the fallback when nothing larger fits -- they just are
    # not raced when something larger does.
    heights = sorted({c.kwargs["GROUP"] for c in kept}, reverse=True)[:2]
    return [c for c in kept if c.kwargs["GROUP"] in heights]


# Table by default, autotuner on MOBA_AUTOTUNE=1; see `moba_fwd` and
# `moba/kernels/tuned_configs.py`. `key_tile` is passed because the backward
# has an axis the forward does not, and two kernels keyed the same way would
# share one entry.
@launch_policy("bwd", triton.autotune(
    configs=[triton.Config({"GROUP": group}, num_warps=warps,
                           num_stages=BWD_STAGES)
             for group in BWD_AUTOTUNE_GROUPS
             for warps in BWD_AUTOTUNE_WARPS],
    # `CAUSAL` and `VARLEN` are absent for the reason `moba_fwd` records: they
    # change which branches compile, not the size of a tile, and including them
    # doubles the shapes to tune for no gain.
    key=["BLOCK_SIZE", "KEY_TILE", "D", "K_SLOTS", "H_Q", "H_KV"],
    prune_configs_by={"early_config_prune": _bwd_prune_by_shared_memory},
    do_bench=_best_of_three,
    cache_results=os.environ.get("MOBA_TUNE_CACHE", "0") == "1",
), key_tile=BWD_KEY_TILE)
@triton.jit
def moba_bwd_kernel(
    Q, K, V, DO, Lse, Delta,
    RowIdx, HeadIdx, SlotIdx, ColOffsets, ColNnz, Order, CuSeqlens,
    DK, DV, DQPart,
    S, T, TOKENS, softmax_scale, sm_scale, num_blocks,
    BLOCK_SIZE: tl.constexpr, KEY_TILE: tl.constexpr, GROUP: tl.constexpr,
    D: tl.constexpr, K_SLOTS: tl.constexpr, DQ_SLOTS: tl.constexpr,
    H_Q: tl.constexpr, H_KV: tl.constexpr,
    CAUSAL: tl.constexpr, VARLEN: tl.constexpr,
):
    """One program per (key sub-tile, batch, key/value head).

    Writes its sub-tile's `dK` and `dV` in full, and one `dQ` partial per
    (query, slot) pair for `reduce_dq` to sum. See `BWD_KEY_TILE` for why the
    unit of work is a sub-tile rather than a whole block.

    Grouped-query attention needs no atomics and no wider accumulator here, and
    the reason is worth stating because it looks like it should. The chain rule
    says a shared key/value head's gradient is the *sum* over the query heads
    consuming it -- normally an explicit reduction across the head axis. Here
    the group's query rows are already merged into this program's column, and
    `dk`/`dv` are reductions down that column, so the sum happens on the way
    past. The accumulators are indexed by *key*, so their shape cannot depend
    on how many queries feed them; and the program still owns a disjoint key
    range, because grouped-query attention removed competing programs rather
    than adding them.

    Packed mode changes only the key range and the query-side bases, exactly as
    in the forward. Every key this program owns belongs to one sequence, so its
    dK and dV cannot receive another document's gradient -- provided no foreign
    query ever entered its column, which the gate guarantees.
    """
    pid = tl.program_id(0)
    pid_bh = tl.program_id(1)

    # Keys are shared, queries are not. See `moba_fwd_kernel` -- the same split,
    # and at H_Q == H_KV both reduce to `pid_bh` exactly.
    pid_b = pid_bh // H_KV
    if VARLEN:
        # The query-side tensors carry a pinned batch of 1, so there is no batch
        # stride; the sequence enters only through the key range below.
        q_base = 0
        q_lin = 0
        part_span = TOKENS
    else:
        # Unchanged by the move: H_Q*S*D and S*H_Q*D are the same number.
        # int64: see moba_fwd_kernel. DQPart is [B, H, S, K*TILES, D],
        # larger than the forward's partials by the tile factor, so it
        # crosses 2^31 elements sooner than anything else here.
        q_base = pid_b.to(tl.int64) * H_Q * S * D
        q_lin = pid_b.to(tl.int64) * H_Q * S
        part_span = S

    tiles: tl.constexpr = BLOCK_SIZE // KEY_TILE
    # Longest-first dispatch is preserved: the block index still comes from
    # `Order`, and the sub-tiles of one block dispatch adjacently.
    blk = tl.load(Order + pid_bh * num_blocks + pid // tiles)
    sub = pid % tiles
    n = tl.load(ColNnz + pid_bh * num_blocks + blk)
    lo = tl.load(ColOffsets + pid_bh * (num_blocks + 1) + blk)

    offs_d = tl.arange(0, D)

    if VARLEN:
        bos = tl.load(CuSeqlens + pid_b)
        eos = tl.load(CuSeqlens + pid_b + 1)
        lo_k = bos + blk * BLOCK_SIZE
        hi_k = tl.minimum(lo_k + BLOCK_SIZE, eos)
        kv_base = (pid_bh % H_KV) * D
    else:
        lo_k = blk * BLOCK_SIZE
        hi_k = S
        kv_base = pid_b * S * H_KV * D + (pid_bh % H_KV) * D

    key_pos = lo_k + sub * KEY_TILE + tl.arange(0, KEY_TILE)
    key_ok = key_pos < hi_k

    k = tl.load(K + kv_base + key_pos[:, None] * (H_KV * D) + offs_d[None, :],
                mask=key_ok[:, None], other=0.0)
    v = tl.load(V + kv_base + key_pos[:, None] * (H_KV * D) + offs_d[None, :],
                mask=key_ok[:, None], other=0.0)

    # Owned outright by this program, so they live in registers for the whole
    # kernel and are stored once at the end. This is the payoff of the loop
    # inversion in the backward direction.
    dk = tl.zeros([KEY_TILE, D], dtype=tl.float32)
    dv = tl.zeros([KEY_TILE, D], dtype=tl.float32)

    # No early return when n == 0. A block nobody selected still has a
    # gradient -- zero -- and the store below is the only thing that writes it,
    # so returning early would leave the caller's buffer uninitialized. The
    # loop simply runs zero times.
    for start in range(0, n, GROUP):
        offs = start + tl.arange(0, GROUP)
        live = offs < n

        rows = tl.load(RowIdx + pid_bh * T + lo + offs, mask=live, other=0)
        heads = tl.load(HeadIdx + pid_bh * T + lo + offs, mask=live, other=0)
        slots = tl.load(SlotIdx + pid_bh * T + lo + offs, mask=live, other=0)

        # `rows` stays a raw position so the causal compare below is untouched;
        # the head rides alongside it. Same trade as the forward -- and the
        # same split: `part_row` names *which query*, addressing our head-major
        # Lse, Delta and dQ partials; the loads name *where it lives*, in the
        # caller's token-major tensors.
        part_row = heads * part_span + rows

        q_addr = (q_base + rows[:, None] * (H_Q * D)
                  + heads[:, None] * D + offs_d[None, :])
        q = tl.load(Q + q_addr, mask=live[:, None], other=0.0)
        do = tl.load(DO + q_addr, mask=live[:, None], other=0.0)
        lse = tl.load(Lse + q_lin + part_row, mask=live, other=0.0)
        delta = tl.load(Delta + q_lin + part_row, mask=live, other=0.0)

        # Same additive construction as the forward, for the same reason: the
        # masks do not depend on the matmul, so they can issue while it is in
        # flight.
        qk = tl.zeros((GROUP, KEY_TILE), dtype=tl.float32)
        qk += tl.where(key_ok[None, :], 0.0, float("-inf"))
        if CAUSAL:
            qk += tl.where(key_pos[None, :] <= rows[:, None], 0.0,
                           float("-inf"))
        # Two scales, because they are used in different bases.
        # `sm_scale` carries the log2(e) that puts the scores in base 2
        # for `exp2`; `softmax_scale` is the plain 1/sqrt(D), which is
        # what the chain rule actually multiplies dQ and dK by. Folding
        # log2(e) into those would scale the gradients by 1.44.
        qk += tl.dot(q, tl.trans(k)) * sm_scale

        # The softmax, recomputed rather than stored -- the central
        # FlashAttention trade, memory for arithmetic. `lse` is global and in
        # log2 units, so this is the true probability, not a per-block one.
        p = tl.exp2(qk - lse[:, None])

        # Dead lanes must be zeroed, and this is the one place in either kernel
        # where masking the *stores* is not enough. `dk` and `dv` are reductions
        # down the query axis, so a padding lane at the end of the last group
        # contributes to them. Its q and do are zero, but its qk is therefore
        # zero too rather than -inf, and exp2(0 - 0) is 1: a padding lane would
        # add a full uniform row of attention weight to this block's dV.
        p = tl.where(live[:, None], p, 0.0)

        dv += tl.dot(tl.trans(p).to(do.dtype), do)

        dp = tl.dot(do, tl.trans(v))
        ds = p * (dp - delta[:, None])

        dk += tl.dot(tl.trans(ds).to(q.dtype), q) * softmax_scale

        # The scattered half. The forward's invariant was that each
        # (query, slot) partial is written by exactly one program; splitting a
        # block into key sub-tiles breaks it, because every sub-tile produces a
        # dQ contribution for the same (query, slot) and they would overwrite
        # rather than sum. dK and dV are untouched by the split, being indexed
        # by key, so it is exactly the already-scattered tensor that suffers.
        #
        # Restored by widening the slot axis to give each sub-tile its own
        # address. `reduce_dq` sums over all of them.
        dq = tl.dot(ds.to(k.dtype), k) * softmax_scale
        dst = part_row * DQ_SLOTS + slots * (BLOCK_SIZE // KEY_TILE) + sub
        tl.store(
            DQPart + q_lin * DQ_SLOTS * D + dst[:, None] * D
            + offs_d[None, :],
            dq.to(DQPart.dtype.element_ty),
            mask=live[:, None],
        )

    # `kv_base`, not `pid_bh * S * D`: under packing the grid enumerates
    # `n_seq * H_KV` while the tensor holds one pinned batch, so the program id
    # is not a stride into it. Getting this wrong writes past the end of dK and
    # dV -- which does not fault, it corrupts whatever the allocator put next.
    kv_out = kv_base + key_pos[:, None] * (H_KV * D) + offs_d[None, :]
    tl.store(DK + kv_out, dk.to(DK.dtype.element_ty), mask=key_ok[:, None])
    tl.store(DV + kv_out, dv.to(DV.dtype.element_ty), mask=key_ok[:, None])


@triton.jit
def moba_dq_reduce_kernel(
    DQPart, PartM, DQ,
    S,
    K_SLOTS: tl.constexpr, TILES: tl.constexpr, D: tl.constexpr,
    BLOCK_M: tl.constexpr, H_Q: tl.constexpr,
):
    """Sums each query's per-slot `dQ` partials.

    A plain sum, unlike `moba_combine_kernel`. Each partial was computed against the
    query's global `lse`, so they are already on a common scale and there is no
    `exp2(m_local - m_global)` correction to apply and no two-pass structure.

    `PartM` is the forward's per-slot maximum, reused here purely as a
    written-marker: a slot no program wrote still holds -inf. That is what lets
    `DQPart` be allocated uninitialized -- selecting on the marker means an
    unwritten slot's contents are never read, so the buffer is written once
    instead of twice.

    One marker covers a slot's `TILES` sub-tile entries. If the forward wrote a
    slot then the query selected that block, and every sub-tile program for
    that block writes it, so the sub-tiles of a written slot are all written.
    """
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    live = offs_m < S
    offs_d = tl.arange(0, D)

    dq_slots: tl.constexpr = K_SLOTS * TILES
    m_base = PartM + pid_bh * S * K_SLOTS + offs_m * K_SLOTS
    # int64 for the reason moba_fwd_kernel records.
    p_base = (DQPart + pid_bh.to(tl.int64) * S * dq_slots * D
              + offs_m[:, None] * dq_slots * D + offs_d[None, :])

    acc = tl.zeros([BLOCK_M, D], dtype=tl.float32)
    for slot in tl.static_range(K_SLOTS):
        written = tl.load(m_base + slot, mask=live,
                          other=float("-inf")) > float("-inf")
        for sub in tl.static_range(TILES):
            part = tl.load(p_base + (slot * TILES + sub) * D,
                           mask=live[:, None], other=0.0)
            acc += tl.where(written[:, None], part.to(tl.float32), 0.0)

    # DQPart and PartM above are ours and head-major; DQ goes back to the
    # caller, so it is token-major.
    tl.store(DQ + (pid_bh // H_Q) * S * H_Q * D + (pid_bh % H_Q) * D
             + offs_m[:, None] * (H_Q * D) + offs_d[None, :],
             acc.to(DQ.dtype.element_ty), mask=live[:, None])


# -------------------------------------------------------------------------
# Launchers
# -------------------------------------------------------------------------


def compute_delta(out: torch.Tensor, dout: torch.Tensor) -> torch.Tensor:
    """Computes the softmax-Jacobian term `rowsum(dO * O)` per query.

    Args:
        out: Forward output shaped [B, S, H, D].
        dout: Output gradient, same shape.

    Returns:
        Delta shaped [B, H, S], float32.
    """
    b, s, h, d = out.shape            # token-major
    delta = torch.empty((b, h, s), device=out.device, dtype=torch.float32)
    moba_bwd_preprocess_kernel[(triton.cdiv(s, BWD_BLOCK_M), b * h)](
        out.contiguous(), dout.contiguous(), delta, s,
        D=d, BLOCK_M=BWD_BLOCK_M, H_Q=h,
    )
    return delta


def reduce_dq(
    dq_part: torch.Tensor, part_m: torch.Tensor, out_dtype: torch.dtype
) -> torch.Tensor:
    """Sums per-slot `dQ` partials into one gradient per query.

    Args:
        dq_part: Per-slot gradients shaped [B, H, S, K * tiles, D], the
            slot axis widened by the key sub-tile count.
        part_m: Forward per-slot maxima [B, H, S, K], used as a written-marker.
        out_dtype: Dtype of the returned tensor.

    Returns:
        Query gradient shaped [B, S, H, D].
    """
    b, h, s, wide, d = dq_part.shape
    slots = part_m.shape[-1]
    # `dq_part` is ours and head-major; `dq` goes to the caller token-major.
    dq = torch.empty((b, s, h, d), device=dq_part.device, dtype=out_dtype)
    block_m, warps = choose_combine_tile(d)

    moba_dq_reduce_kernel[(triton.cdiv(s, block_m), b * h)](
        dq_part, part_m, dq, s,
        K_SLOTS=slots, TILES=wide // slots, D=d, BLOCK_M=block_m, H_Q=h,
        num_warps=warps,
    )
    return dq


# -------------------------------------------------------------------------
# Public entry point
# -------------------------------------------------------------------------


def moba_backward(
    dout: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    part_m: torch.Tensor,
    cfg: MoBAConfig,
    index: CSCIndex,
    causal: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Computes gradients for the MoBA forward pass.

    Args:
        dout: Gradient of the loss w.r.t. the forward output, [B, S, H, D].
        q: Queries as passed to the forward, [B, S, H, D].
        k: Keys, [B, S, H_kv, D], where `H_kv` divides `H`.
        v: Values, same shape as `k`.
        out: Forward output, shaped like `q`.
        lse: Per-query global log-sum-exp in log2 units, [B, H, S], as
            `combine_partials(..., return_lse=True)` returns.
        part_m: Forward per-slot maxima [B, H, S, K], the written-marker.
        cfg: Configuration.
        index: The CSC index the forward used. Routing is not differentiated,
            so this is the same fixed sparsity pattern, not a rebuilt one. It
            also carries `cu_seqlens` when the batch is packed, so the backward
            needs no separate argument and cannot be run in the wrong mode.
        causal: Whether the forward applied causal masking.

    Returns:
        A tuple of (dq, dk, dv), each shaped like its input. Under grouped-query
        attention `dk` and `dv` carry `H_kv` heads, each already summed over the
        query heads that shared it.

    Raises:
        ValueError: If no chunk height compiles within shared memory.
    """
    b, s, h_q, d = q.shape
    h_kv = k.shape[2]
    head_group_size(h_q, h_kv)
    slots = index.max_slots
    varlen = index.cu_seqlens is not None
    n_seq = index.col_nnz.shape[0]
    q, k, v, dout = (t.contiguous() for t in (q, k, v, dout))

    delta = compute_delta(out, dout)

    dk = torch.empty_like(k)
    dv = torch.empty_like(v)
    key_tile = min(cfg.block_size, BWD_KEY_TILE)
    tiles = cfg.block_size // key_tile
    dq_part = torch.empty((b, h_q, s, slots * tiles, d), device=q.device,
                          dtype=q.dtype)

    # GROUP, num_warps and num_stages come from `triton.autotune` on the
    # decorator; the launcher passes none of them. The backward tunes
    # separately from the forward and lands somewhere else -- 64 queries over
    # 4 warps here against the forward's 64 over 4 with different pipelining --
    # which is the argument for measuring each kernel rather than sharing a
    # constant between them.
    moba_bwd_kernel[(index.num_blocks * tiles, n_seq * h_kv)](
        q, k, v, dout, lse, delta,
        index.row_indices, index.head_indices, index.slot_of_pair,
        index.col_offsets, index.col_nnz, index.order,
        index.cu_seqlens if varlen else q,
        dk, dv, dq_part,
        s, index.row_indices.shape[-1], s,
        d ** -0.5, d ** -0.5 * LOG2_E, index.num_blocks,
        BLOCK_SIZE=cfg.block_size, KEY_TILE=key_tile,
        D=d, K_SLOTS=slots, DQ_SLOTS=slots * tiles,
        H_Q=h_q, H_KV=h_kv, CAUSAL=causal, VARLEN=varlen,
    )
    return reduce_dq(dq_part, part_m, q.dtype), dk, dv
