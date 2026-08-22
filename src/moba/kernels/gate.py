"""Fused MoBA gate: score blocks and pick the top-k without materializing scores.

The PyTorch gate computes an `[B, H, S, num_blocks]` score matrix, writes it to
HBM, reads it back to mask it, writes it again, sorts it, and writes a dense
boolean mask. Profiling put ~40% of the gate's GPU time in `aten::copy_` and
device-to-device memcpy -- moving intermediates, not computing anything.

This kernel keeps a query tile's scores in registers, ranks them there, and
emits only the answer: for each query, the block ids it selected, ascending.
The score matrix never exists in memory.

This is the same trick as FlashAttention itself. FA never writes the `[S, S]`
score matrix because it only ever needs a running summary of it -- the running
max and sum. Here we only ever need a running top-k. In both cases the large
intermediate is scaffolding, not the answer, and scaffolding you can keep on
chip is scaffolding you never pay to move.

Pattern borrowed from FlashMoBA's "Flash TopK" (its stage 2, in CUDA, which
reports 22x over the unfused gate) and from fla-org's NSA
`parallel_nsa_kernel_topk`, which does the same in Triton. Both stream a
centroid matrix past a resident query tile and keep the running top-k on chip.

What is different here: NSA merges each streamed chunk into a sorted running
list with a bitonic network, because its candidate count is large. MoBA's
candidate count is `num_blocks = S / block_size` -- 32 at S=4096, 128 at
S=16384. Small enough to hold the whole score row in registers and rank it in
place by repeated maximum extraction, which is both cheaper at this size and
far easier to make deterministic. The ceiling that buys is set by the
representative tile, and `check_gate_fits` enforces it.
"""

import torch
import triton
import triton.language as tl

from moba.config import MoBAConfig
from moba.device import (
    TENSOR_CORE_TILE,
    WARP_SIZE,
    round_up_to_tensor_core_tile,
    shared_memory_budget,
)

# Queries per gate program. Small for the same reason the combine tile is
# small: the score row `[BLOCK_M, N_BLOCKS]` lives in registers, so the tile
# height multiplies the register cost of every extraction round.
GATE_BLOCK_M = 16

# Score elements one thread may hold before the compiler spills, mirroring
# `COMBINE_ELEMENTS_PER_THREAD`. The gate holds two full-width tiles at once
# (the working scores and the selection mask), so this is half the combine
# budget.
GATE_ELEMENTS_PER_THREAD = 16

MIN_GATE_WARPS = 4
MAX_GATE_WARPS = 8


@triton.jit
def _moba_gate_fwd(
    Q, Kbar, BlockIdx,
    S, sm_scale,
    BLOCK_SIZE: tl.constexpr, BLOCK_M: tl.constexpr, D: tl.constexpr,
    N_BLOCKS: tl.constexpr, N_POW2: tl.constexpr,
    TAKE: tl.constexpr, TOP_K: tl.constexpr,
    ALWAYS_CURRENT: tl.constexpr,
):
    """One program per (query tile, batch-head). Emits selected block ids.

    Writes `BlockIdx[b, h, s, j] = block id of the query's j-th selection`,
    ascending in `j`, padded with -1. Nothing else is written: no scores, no
    dense mask.
    """
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    live = offs_m < S
    offs_d = tl.arange(0, D)
    cols = tl.arange(0, N_POW2)

    q = tl.load(Q + pid_bh * S * D + offs_m[:, None] * D + offs_d[None, :],
                mask=live[:, None], other=0.0)
    kbar = tl.load(
        Kbar + pid_bh * N_BLOCKS * D + cols[:, None] * D + offs_d[None, :],
        mask=(cols < N_BLOCKS)[:, None], other=0.0,
    )

    # `tl.dot` accumulates in float32. The PyTorch gate multiplies bf16 by
    # bf16, and cuBLAS also accumulates in float32 but rounds its *output* back
    # to bf16. Rounding here reproduces that exactly, which matters because the
    # routing test compares selections for equality and a near-tie decided
    # differently is a mismatch, not an approximation.
    scores = tl.dot(q, tl.trans(kbar)) * sm_scale
    scores = scores.to(kbar.dtype).to(tl.float32)

    # A block is a candidate only if it lies wholly in the query's past. The
    # query's own block is handled separately below: it needs an internal
    # causal mask, so it is never ranked against the others.
    own_block = offs_m // BLOCK_SIZE
    is_past = (cols[None, :] < own_block[:, None]) & (cols < N_BLOCKS)[None, :]
    work = tl.where(is_past, scores, float("-inf"))

    # --- Rank by repeated maximum extraction. --------------------------------
    # TAKE rounds, each finding the best remaining candidate and retiring it.
    # An alternative was a bitonic sort of the whole row, as NSA uses; at
    # num_blocks this small, TAKE passes is less work than log^2(N) stages, and
    # the tie-break below is explicit rather than a property of the network.
    chosen = cols[None, :] < 0        # all false, [BLOCK_M, N_POW2]
    for _ in tl.static_range(TAKE):
        best = tl.max(work, axis=1)

        # Lowest index wins a tie. This is `tie_break_low_index`, and it is a
        # determinism requirement rather than an accuracy one: an unstable
        # tie-break makes the kernel select different blocks on different runs,
        # which is almost indistinguishable from a race when debugging. Taking
        # the minimum column index among the maxima states the rule outright
        # instead of trusting `argmax` to resolve ties in any order.
        at_best = work == best[:, None]
        pick = tl.min(tl.where(at_best, cols[None, :], N_POW2), axis=1)

        # A round finds nothing when the query has fewer past blocks than the
        # budget. top_k is a ceiling, not a quota.
        found = best > float("-inf")
        hit = (cols[None, :] == pick[:, None]) & found[:, None]
        chosen = chosen | hit
        work = tl.where(hit, float("-inf"), work)

    if ALWAYS_CURRENT:
        chosen = chosen | ((cols[None, :] == own_block[:, None])
                           & (cols < N_BLOCKS)[None, :])

    # --- Emit ascending. -----------------------------------------------------
    # Slot j is the rank of a block among the query's selections ordered by
    # ascending block index -- the same rule the PyTorch index used, so the
    # combine pass's fixed-slot addressing is unchanged. Repeatedly taking the
    # lowest remaining selection produces exactly that order.
    remaining = chosen
    for j in tl.static_range(TOP_K):
        nxt = tl.min(tl.where(remaining, cols[None, :], N_POW2), axis=1)
        valid = nxt < N_BLOCKS
        tl.store(
            BlockIdx + pid_bh * S * TOP_K + offs_m * TOP_K + j,
            tl.where(valid, nxt, -1).to(tl.int32),
            mask=live,
        )
        remaining = remaining & ~(cols[None, :] == nxt[:, None])


def choose_gate_warps(block_m: int, n_pow2: int) -> int:
    """Picks a warp count that keeps the gate's score tile out of spills.

    Args:
        block_m: Queries per program.
        n_pow2: Score-row width, rounded to a power of two.

    Returns:
        A warp count between `MIN_GATE_WARPS` and `MAX_GATE_WARPS`.
    """
    per_thread = WARP_SIZE * GATE_ELEMENTS_PER_THREAD
    needed = (block_m * n_pow2 + per_thread - 1) // per_thread
    return max(MIN_GATE_WARPS, min(MAX_GATE_WARPS, needed))


def check_gate_fits(num_blocks: int, head_dim: int, itemsize: int,
                    device=None) -> tuple[bool, str]:
    """Reports whether the block representatives fit in one program's SRAM.

    This kernel loads every block representative at once, which is what lets it
    rank a query's whole score row in registers without a streaming merge. The
    cost is a ceiling on `num_blocks`: at head_dim 128 in bfloat16, 128 blocks
    need 32 KiB and fit, 512 need 128 KiB and do not. Since
    `num_blocks = seq_len / block_size`, that ceiling is a sequence length --
    roughly 16K at block_size 128 on sm_86.

    Beyond it the representatives must be streamed in chunks with a running
    top-k merged per chunk, as NSA does. That is a different kernel, not a
    tuning change, so this returns a reason rather than silently degrading.

    Args:
        num_blocks: Number of MoBA blocks.
        head_dim: Head dimension.
        itemsize: Bytes per element of the gate dtype.
        device: CUDA device, for the shared-memory budget.

    Returns:
        A tuple of (ok, reason). `reason` is empty when ok.
    """
    budget = shared_memory_budget(device)
    needed = round_up_to_tensor_core_tile(num_blocks) * head_dim * itemsize
    if needed > budget:
        return False, (
            f"{num_blocks} blocks at head_dim {head_dim} need {needed} bytes "
            f"of shared memory for the representative tile, above this GPU's "
            f"{budget}. This kernel holds all representatives at once; longer "
            "sequences need a streaming top-k."
        )
    return True, ""


def gate_select(
    q: torch.Tensor, kbar: torch.Tensor, cfg: MoBAConfig
) -> torch.Tensor:
    """Selects blocks for every query, returning ids rather than a dense mask.

    Args:
        q: Queries shaped [B, H, S, D], contiguous.
        kbar: Block representatives shaped [B, H, num_blocks, D], contiguous.
        cfg: Configuration supplying `block_size`, `top_k` and the selection
            conventions.

    Returns:
        Selected block ids shaped [B, H, S, top_k], int32, ascending within
        each query and padded with -1.

    Raises:
        ValueError: If the shapes disagree or the representatives do not fit.
    """
    b, h, s, d = q.shape
    if kbar.shape[:2] != (b, h) or kbar.shape[-1] != d:
        raise ValueError(
            f"representative shape {tuple(kbar.shape)} does not match queries "
            f"{tuple(q.shape)}"
        )
    n = kbar.shape[2]
    ok, reason = check_gate_fits(n, d, q.element_size(), q.device)
    if not ok:
        raise ValueError(reason)

    q, kbar = q.contiguous(), kbar.contiguous()
    n_pow2 = max(TENSOR_CORE_TILE, triton.next_power_of_2(n))
    block_m = max(TENSOR_CORE_TILE, GATE_BLOCK_M)

    out = torch.empty((b, h, s, cfg.top_k), device=q.device, dtype=torch.int32)

    _moba_gate_fwd[(triton.cdiv(s, block_m), b * h)](
        q, kbar, out,
        s,
        # The gate score is deliberately unscaled: softmax's 1/sqrt(D) is a
        # positive monotone transform and cannot reorder a ranking, so applying
        # it would be arithmetic that changes no decision.
        1.0,
        BLOCK_SIZE=cfg.block_size, BLOCK_M=block_m, D=d,
        N_BLOCKS=n, N_POW2=n_pow2,
        TAKE=cfg.past_blocks_selected(n), TOP_K=cfg.top_k,
        ALWAYS_CURRENT=cfg.always_select_current,
        num_warps=choose_gate_warps(block_m, n_pow2),
    )
    return out
