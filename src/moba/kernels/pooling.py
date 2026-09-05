"""Stage 1 of MoBA: pool each key block into one representative key.

MoBA's arithmetic saving starts here. Rather than scoring a query against every
key, the gate scores it against one representative per block -- `num_blocks`
dot products instead of `S`. This module produces those representatives, and
`gating.py` consumes them.

The representative is the block's mean key. That choice is MoBA's, not ours;
`docs/flashmoba-digest.md` records what the alternatives would cost.
"""

import torch
import triton
import triton.language as tl

from moba.config import MoBAConfig
from moba.packing import validate_cu_seqlens


@triton.jit
def pool_blocks_kernel(
    K, Kbar, CuSeqlens,
    S, T, N_BLOCKS,
    BLOCK_SIZE: tl.constexpr, BLOCK_POW2: tl.constexpr, D: tl.constexpr,
    H_KV: tl.constexpr, VARLEN: tl.constexpr,
):
    """One program per (block, batch, key/value head). Mean-pools a block's keys.

    The PyTorch version pads the sequence to a block multiple with `F.pad`,
    which copies the entire key tensor, and then writes the final block's
    occupancy from the host -- a device-to-host round trip costing ~230 us on
    WSL. A kernel needs neither: it masks the tail instead of padding it, and
    computes the occupancy from the sequence length it was already passed.

    Packed mode changes only *where* a block's keys are. Blocks are cut per
    sequence, so a block straddling a boundary would mix two documents into one
    representative -- accepted by every downstream stage, finite, and wrong. The
    batch comes from the grid, so the range is two arithmetic ops on values
    already loaded rather than a lookup table.
    """
    blk = tl.program_id(0)
    pid_bh = tl.program_id(1)

    offs_d = tl.arange(0, D)

    # Token-major: heads sit D apart and tokens H_KV*D apart. Neither stride
    # contains the sequence length, so the packed and padded loads differ only
    # by a batch term that is zero when packed.
    b = pid_bh // H_KV
    h = pid_bh % H_KV

    if VARLEN:
        bos = tl.load(CuSeqlens + b)
        eos = tl.load(CuSeqlens + b + 1)
        if blk * BLOCK_SIZE >= eos - bos:
            return          # a padded index slot; this block does not exist
        lo = bos + blk * BLOCK_SIZE
        hi = tl.minimum(lo + BLOCK_SIZE, eos)
        kbase = h * D                           # K is [1, T, H_kv, D]
    else:
        lo = blk * BLOCK_SIZE
        hi = tl.minimum(lo + BLOCK_SIZE, S)
        kbase = b * S * H_KV * D + h * D        # K is [B, S, H_kv, D]

    rows = lo + tl.arange(0, BLOCK_POW2)
    live = (rows < hi) & (tl.arange(0, BLOCK_POW2) < BLOCK_SIZE)

    # `Kbar`, the output, is ours and stays head-major -- see `moba.heads`.
    k = tl.load(K + kbase + rows[:, None] * (H_KV * D) + offs_d[None, :],
                mask=live[:, None], other=0.0)

    # Accumulate in float32 regardless of input dtype, matching what torch's
    # reduction does internally, then round to the input dtype before dividing.
    # The rounding point is not cosmetic: the gate ranks these representatives,
    # and rounding at a different step would flip near-ties, which the routing
    # test compares for exact equality.
    total = tl.sum(k.to(tl.float32), axis=0)
    total = total.to(K.dtype.element_ty).to(tl.float32)

    # A short final block averages only the positions that exist. Dividing by
    # BLOCK_SIZE instead would shrink a 1-key tail block's representative
    # toward zero and make it almost never selected -- a bug invisible whenever
    # seq_len divides evenly by block_size.
    # Under packing every *sequence* has such a tail, not just the batch, so
    # this stops being a rare edge case and becomes one block in every row.
    occupancy = hi - lo
    mean = total / occupancy.to(tl.float32)

    tl.store(Kbar + pid_bh * N_BLOCKS * D + blk * D + offs_d,
             mean.to(Kbar.dtype.element_ty))


def pool_blocks(
    k: torch.Tensor,
    cfg: MoBAConfig,
    cu_seqlens: torch.Tensor | None = None,
    max_seqlen: int | None = None,
) -> torch.Tensor:
    """Mean-pools each block's keys into one representative.

    Args:
        k: Keys shaped [B, S, H, D], contiguous, or [1, T, H, D] when packed.
        cfg: Configuration supplying `block_size`.
        cu_seqlens: Cumulative token counts, [num_sequences + 1] int32. When
            given, `k` holds one packed batch and `B` must be 1.
        max_seqlen: Longest sequence, required with `cu_seqlens`. It sizes the
            output's block axis and the launch grid; deriving it from
            `cu_seqlens` would cost a host synchronization.

    Returns:
        Block representatives, same dtype as `k`: `[B, H, num_blocks, D]`
        padded, `[num_sequences, H, cdiv(max_seqlen, block_size), D]` packed.
        The packed form is the *padded index* of a packed batch -- a sequence
        with fewer blocks leaves the tail of its row unwritten, and the gate
        never reads past that sequence's own block count.

    Raises:
        ValueError: If packed input has a batch axis above 1, or `max_seqlen`
            is missing.
    """
    b, s, h, d = k.shape
    k = k.contiguous()

    if cu_seqlens is None:
        n = cfg.num_blocks(s)
        out = torch.empty((b, h, n, d), device=k.device, dtype=k.dtype)
        pool_blocks_kernel[(n, b * h)](
            k, out, k, s, s, n,
            BLOCK_SIZE=cfg.block_size,
            BLOCK_POW2=triton.next_power_of_2(cfg.block_size),
            D=d, H_KV=h, VARLEN=False,
        )
        return out

    if b != 1:
        raise ValueError(
            f"packed input must have batch 1, got {b}; every sequence lives on "
            f"the token axis and cu_seqlens says where"
        )
    if max_seqlen is None:
        raise ValueError("max_seqlen is required when cu_seqlens is given")
    n_seq = validate_cu_seqlens(cu_seqlens, total_tokens=s,
                                max_seqlen=max_seqlen)
    n = cfg.num_blocks(max_seqlen)

    out = torch.empty((n_seq, h, n, d), device=k.device, dtype=k.dtype)
    pool_blocks_kernel[(n, n_seq * h)](
        k, out, cu_seqlens, s, s, n,
        BLOCK_SIZE=cfg.block_size,
        BLOCK_POW2=triton.next_power_of_2(cfg.block_size),
        D=d, H_KV=h, VARLEN=True,
    )
    return out
