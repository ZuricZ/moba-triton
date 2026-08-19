"""KV-block-major MoBA forward.

One program owns one KV block, loads it into SRAM once, and streams every query
that selected it past that single load. This is decision D6: the loop order is
inverted relative to FlashAttention-2 because MoBA's per-query routing destroys
the amortization a query-major tile depends on. Measured query-major
utilization at S=4096, block_size=16 is 0.028 -- 97% of tensor-core work
discarded (`docs/divergence-measurements.md`).

Three things about this kernel differ from a dense FA-2 inner loop, all of them
simplifications:

1. **The online-softmax rescale is relocated, not removed.** In FA-2 one
   program folds many KV tiles into one query, so the `exp(m_old - m_new)`
   correction is loop-carried inside the kernel. Here a program sees each of
   its queries against exactly *one* block, so there is no sequence of tiles
   to fold and no loop-carried rescale in this kernel.

   The query's full row still spans several blocks, computed by different
   programs, and those partials still require the identical correction --
   `exp(m_local - m_global)` -- before they can be summed. That is
   `combine_partials`, and it is mandatory (D10). Same algebra, applied across
   programs instead of across loop iterations.
2. **No branch on selection.** Every query the program holds selected this
   block, so "does this query need this block?" is uniform across the warp.
3. **Causality is one predicated select** that is a no-op for any block wholly
   in the past, and compiles away entirely when CAUSAL is False (D9).

What it gives up is output ownership: a query's contributions come from several
programs and must be merged afterwards (D10).
"""

import torch
import triton
import triton.language as tl

from moba.config import MoBAConfig
from moba.device import (
    TENSOR_CORE_TILE,
    check_gathered_head_dim,
    round_up_to_tensor_core_tile,
    shared_memory_budget,
)
from moba.kernels.index import CSCIndex, build_csc


@triton.jit
def _moba_attn_fwd(
    Q, K, V, RowIdx, SlotIdx, ColOffsets, ColNnz, Order,
    PartM, PartL, PartAcc,
    S, T, sm_scale, num_blocks,
    BLOCK_SIZE: tl.constexpr, GROUP: tl.constexpr, D: tl.constexpr,
    K_SLOTS: tl.constexpr, CAUSAL: tl.constexpr,
):
    """One program per (KV block, batch-head).

    Writes un-normalized partials; nothing here is divided by a softmax
    denominator, because this program does not know the other programs'
    running maxima.
    """
    pid_col = tl.program_id(0)
    pid_bh = tl.program_id(1)

    # Longest-first dispatch (D8). Only the block id changes -- results are
    # written to query addresses, which never depended on the program id, so
    # there is nothing to undo afterwards.
    blk = tl.load(Order + pid_bh * num_blocks + pid_col)
    n = tl.load(ColNnz + pid_bh * num_blocks + blk)
    if n == 0:
        return

    lo = tl.load(ColOffsets + pid_bh * (num_blocks + 1) + blk)

    offs_d = tl.arange(0, D)
    key_pos = blk * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    key_ok = key_pos < S

    kv_base = pid_bh * S * D

    # The single HBM trip this whole design exists to amortize.
    k = tl.load(K + kv_base + key_pos[:, None] * D + offs_d[None, :],
                mask=key_ok[:, None], other=0.0)
    v = tl.load(V + kv_base + key_pos[:, None] * D + offs_d[None, :],
                mask=key_ok[:, None], other=0.0)

    for start in range(0, n, GROUP):
        offs = start + tl.arange(0, GROUP)
        live = offs < n

        rows = tl.load(RowIdx + pid_bh * T + lo + offs, mask=live, other=0)
        slots = tl.load(SlotIdx + pid_bh * T + lo + offs, mask=live, other=0)

        # Gathered rows (D7). One row is D*itemsize bytes -- exactly one
        # 128-byte cache line at D=64 bf16 -- so this costs the same
        # transaction count as a contiguous read. Measured at 1.03x.
        q = tl.load(Q + kv_base + rows[:, None] * D + offs_d[None, :],
                    mask=live[:, None], other=0.0)

        qk = tl.dot(q, tl.trans(k)) * sm_scale
        qk = tl.where(key_ok[None, :], qk, float("-inf"))
        if CAUSAL:
            # A no-op for every block wholly in the past; trims only the
            # query's own block (D9). Predicated select, not a branch, so it
            # does not reintroduce warp divergence.
            qk = tl.where(key_pos[None, :] <= rows[:, None], qk, float("-inf"))

        # This is the online-softmax state, computed against this block only.
        # `m` and `l` are LOCAL -- they know nothing of the other blocks this
        # query selected. They are written out precisely so `combine_partials`
        # can rescale them by exp(m_local - m_global) and sum them. Dividing
        # acc by l here would discard the information that merge needs.
        #
        # Every row has at least one live entry: a query only reaches a block
        # it selected, and it can only select its own block if that block
        # starts at or before it. So `m` is never -inf and exp() is safe.
        m = tl.max(qk, axis=1)
        p = tl.exp(qk - m[:, None])
        l = tl.sum(p, axis=1)
        acc = tl.dot(p.to(v.dtype), v)

        # Fixed-slot destination (D10). Task 1 guarantees each (query, slot)
        # pair is written by exactly one program, so these scattered stores
        # never collide.
        dst = rows * K_SLOTS + slots
        tl.store(PartM + pid_bh * S * K_SLOTS + dst, m, mask=live)
        tl.store(PartL + pid_bh * S * K_SLOTS + dst, l, mask=live)
        tl.store(
            PartAcc + pid_bh * S * K_SLOTS * D
            + dst[:, None] * D + offs_d[None, :],
            acc,
            mask=live[:, None],
        )


# Absolute ceiling on the inner-loop chunk height, before the shared-memory
# budget is applied. Beyond this, register pressure bites even when shared
# memory allows more: measured 255 registers with 8 spills at D=64,
# BLOCK_SIZE=128, GROUP=128 (`docs/learning/02`).
MAX_GROUP = 128

# Pipelining depth Triton allocates for the gathered q tile. Measured 2 or 3
# depending on configuration; 3 is used so the estimate is an upper bound and
# never predicts a launch that then fails.
PIPELINE_STAGES = 3


def estimate_shared_memory(
    block_size: int, group: int, head_dim: int, itemsize: int
) -> int:
    """Estimates shared memory for one attention program, in bytes.

    The K and V tiles load once outside the inner loop, so they are
    single-buffered at `block_size x head_dim` each. The gathered q tile loads
    every iteration and Triton pipelines it, costing `PIPELINE_STAGES` copies
    of `group x head_dim`.

    Fitted against measurements: at head_dim 64, block_size 128 the observed
    values are 44.2 / 56.5 / 80.0 KiB for group 32 / 64 / 128, against
    44.2 / 56.2 / 80.2 predicted.

    Args:
        block_size: KV block size.
        group: Inner-loop chunk height.
        head_dim: Head dimension.
        itemsize: Bytes per element of the input dtype.

    Returns:
        Estimated bytes of shared memory.
    """
    return (2 * block_size + PIPELINE_STAGES * group) * head_dim * itemsize


def choose_group(
    cfg: MoBAConfig, head_dim: int, itemsize: int, device=None
) -> int:
    """Picks the largest inner-loop chunk height that fits in shared memory.

    `group` is how many queries are processed per iteration -- NOT how many
    selected this block. The loop iterates as many times as needed either way,
    and K/V load once *outside* it, so a larger group buys no extra
    amortization; it only trims loop overhead. What it costs is shared memory,
    because the gathered q tile is pipelined.

    This is why an earlier `GROUP = top_k * block_size` was wrong: it sized a
    tile from a consumer count. Uncapped it asked for 112 KiB against a 99 KiB
    limit, and shared-memory overflow fails the launch outright rather than
    degrading.

    Public rather than private because `bench/load_balance.py` must simulate
    the tile height the kernel actually uses. It previously hardcoded 32 and
    silently went stale.

    Args:
        cfg: Configuration supplying `block_size` and `top_k`.
        head_dim: Head dimension.
        itemsize: Bytes per element of the input dtype.
        device: CUDA device, for the shared-memory budget.

    Returns:
        A multiple of `TENSOR_CORE_TILE` that fits the budget.

    Raises:
        ValueError: If no chunk height down to `TENSOR_CORE_TILE` fits, which
            means the K and V tiles alone exceed shared memory.
    """
    budget = shared_memory_budget(device)
    group = min(MAX_GROUP, round_up_to_tensor_core_tile(cfg.top_k * cfg.block_size))

    while group >= TENSOR_CORE_TILE:
        if estimate_shared_memory(cfg.block_size, group, head_dim, itemsize) <= budget:
            return group
        group //= 2

    needed = estimate_shared_memory(
        cfg.block_size, TENSOR_CORE_TILE, head_dim, itemsize
    )
    raise ValueError(
        f"block_size {cfg.block_size} with head_dim {head_dim} needs {needed} "
        f"bytes of shared memory at the minimum chunk height, exceeding this "
        f"GPU's {budget}. Reduce block_size."
    )


def combine_partials(part_m, part_l, part_acc, out_dtype):
    """Merges per-slot partial softmaxes. Temporary PyTorch reference.

    Replaced by a Triton kernel in Task 3. Present so the attention kernel can
    be tested before the merge exists.

    Args:
        part_m: Per-slot maxima, [B, H, S, K] fp32.
        part_l: Per-slot denominators, [B, H, S, K] fp32.
        part_acc: Per-slot un-normalized outputs, [B, H, S, K, D] fp32.
        out_dtype: Dtype of the returned tensor.

    Returns:
        Attention output shaped [B, H, S, D].
    """
    m_g = part_m.max(dim=-1, keepdim=True).values
    w = torch.exp(part_m - m_g)
    l = (part_l * w).sum(dim=-1)
    acc = (part_acc * w.unsqueeze(-1)).sum(dim=-2)
    return (acc / l.unsqueeze(-1)).to(out_dtype)


def moba_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cfg: MoBAConfig,
    index: CSCIndex | None = None,
    causal: bool = True,
) -> torch.Tensor:
    """Computes MoBA attention with the KV-block-major kernel.

    Args:
        q: Queries shaped [B, H, S, D], contiguous.
        k: Keys shaped [B, H, S, D], contiguous.
        v: Values shaped [B, H, S, D], contiguous.
        cfg: Configuration.
        index: Precomputed CSC index; built from `q` and `k` when omitted.
            Passing one explicitly holds routing fixed while inputs vary,
            which is what the invariant tests need.
        causal: Whether to apply causal masking within the query's own block.

    Returns:
        Attention output shaped [B, H, S, D], same dtype as `q`.

    Raises:
        ValueError: If shapes disagree, or head_dim or block_size is
            unsupported.
    """
    if not (q.shape == k.shape == v.shape):
        raise ValueError(f"shape mismatch: {q.shape}, {k.shape}, {v.shape}")
    b, h, s, d = q.shape
    ok, reason = check_gathered_head_dim(d, q.element_size())
    if not ok:
        raise ValueError(reason)
    if cfg.block_size < TENSOR_CORE_TILE:
        raise ValueError(
            f"block_size {cfg.block_size} below the tl.dot minimum of "
            f"{TENSOR_CORE_TILE}"
        )

    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    if index is None:
        index = build_csc(q, k, cfg)

    slots = index.max_slots
    group = choose_group(cfg, d, q.element_size(), q.device)

    # part_m starts at -inf so a slot no program writes contributes nothing:
    # exp(-inf - m_global) is 0, which removes it without a branch.
    part_m = torch.full((b, h, s, slots), float("-inf"),
                        device=q.device, dtype=torch.float32)
    part_l = torch.zeros((b, h, s, slots), device=q.device,
                         dtype=torch.float32)
    part_acc = torch.zeros((b, h, s, slots, d), device=q.device,
                           dtype=torch.float32)

    _moba_attn_fwd[(index.num_blocks, b * h)](
        q, k, v,
        index.row_indices, index.slot_of_pair,
        index.col_offsets, index.col_nnz, index.order,
        part_m, part_l, part_acc,
        s, index.row_indices.shape[-1], d ** -0.5, index.num_blocks,
        BLOCK_SIZE=cfg.block_size, GROUP=group, D=d,
        K_SLOTS=slots, CAUSAL=causal,
    )

    return combine_partials(part_m, part_l, part_acc, q.dtype)
