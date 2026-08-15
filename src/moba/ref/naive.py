"""Naive PyTorch MoBA -- the project's oracle.

This is deliberately slow and deliberately obvious. It materializes the full
[S, S] attention matrix, which is exactly what a real kernel exists to avoid,
and that is the point: this is the executable *definition* of what the kernel
must compute, not a competitor to it. Every line should map to one line of the
algorithm, so that when the Triton kernel disagrees the question is always
"which line?" and never "what was this supposed to do?".

Semantics are specified in
`docs/superpowers/specs/2026-08-15-moba-foundations-design.md`, section
"MoBA semantics". The nine numbered rules there correspond to the sections
below.
"""

import torch
import torch.nn.functional as F

from moba.config import MoBAConfig


def block_representatives(k: torch.Tensor, cfg: MoBAConfig) -> torch.Tensor:
    """Computes the mean-pooled key representing each block.

    This is the gate's whole reason for being cheap. Scoring a query against
    an entire block costs one dot product against this summary rather than
    `block_size` dot products against the block's keys.

    A short final block averages only the positions that exist. Padding to a
    block multiple and taking a plain mean would divide a 1-key tail block by
    `block_size`, shrinking its representative toward zero and making it
    almost never selected -- a bug invisible whenever seq_len divides evenly.

    Args:
        k: Keys shaped [B, H, S, D].
        cfg: Configuration supplying `block_size`.

    Returns:
        Block representatives shaped [B, H, num_blocks, D].
    """
    b, h, s, d = k.shape
    bk = cfg.block_size
    num_blocks = cfg.num_blocks(s)
    pad = num_blocks * bk - s

    # Pad the sequence axis so the reshape below is well-formed. F.pad's
    # padding spec runs last-axis-first, so (0, 0, 0, pad) pads D by nothing
    # and S by `pad` on the right.
    k_padded = F.pad(k, (0, 0, 0, pad))
    sums = k_padded.view(b, h, num_blocks, bk, d).sum(dim=3)

    # Divide by the real occupancy of each block, not by bk.
    counts = torch.full((num_blocks,), bk, device=k.device, dtype=k.dtype)
    counts[-1] = bk - pad
    return sums / counts.view(1, 1, num_blocks, 1)


def select_blocks(
    q: torch.Tensor, kbar: torch.Tensor, cfg: MoBAConfig
) -> torch.Tensor:
    """Selects which blocks each query attends to.

    A block is a candidate only if it lies wholly in the query's past. The
    query's own block is always selected and does not consume a top_k slot:
    it holds the most recent context and is the only block requiring an
    internal causal mask.

    Args:
        q: Queries shaped [B, H, S, D].
        kbar: Block representatives shaped [B, H, num_blocks, D].
        cfg: Configuration supplying `block_size`, `top_k`, and semantics flags.

    Returns:
        Boolean selection mask shaped [B, H, S, num_blocks].
    """
    b, h, s, _ = q.shape
    num_blocks = kbar.shape[2]
    device = q.device

    # Gate scores. Deliberately unscaled: softmax's 1/sqrt(D) is a positive
    # monotone transform, and a monotone transform cannot reorder a ranking.
    # Applying it here would be arithmetic that changes no decision.
    scores = torch.einsum("bhsd,bhnd->bhsn", q, kbar)

    pos = torch.arange(s, device=device)
    own_block = pos // cfg.block_size                       # [S]
    blocks = torch.arange(num_blocks, device=device)        # [num_blocks]
    is_past = blocks.view(1, -1) < own_block.view(-1, 1)    # [S, num_blocks]
    is_current = blocks.view(1, -1) == own_block.view(-1, 1)

    # Non-candidates score -inf so they sort last and can never be chosen.
    masked = scores.masked_fill(~is_past.view(1, 1, s, num_blocks), float("-inf"))

    # A *stable* descending sort makes equal scores resolve toward the lower
    # block index. torch.topk offers no such guarantee, and an unstable
    # tie-break makes the kernel non-deterministic run to run -- which is
    # almost indistinguishable from a race condition when debugging.
    if cfg.tie_break_low_index:
        order = torch.argsort(masked, dim=-1, descending=True, stable=True)
    else:
        order = torch.argsort(masked, dim=-1, descending=True)

    take = min(cfg.top_k, num_blocks)
    chosen = order[..., :take]

    selected = torch.zeros_like(masked, dtype=torch.bool)
    selected.scatter_(-1, chosen, True)

    # An early query may have fewer than `take` past blocks, in which case the
    # first `take` sorted slots include -inf entries. Drop them: top_k is a
    # ceiling, not a quota.
    selected &= is_past.view(1, 1, s, num_blocks)

    if cfg.always_select_current:
        selected |= is_current.view(1, 1, s, num_blocks)
    return selected


def moba_attention_naive(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, cfg: MoBAConfig
) -> torch.Tensor:
    """Computes MoBA attention the slow, obvious way.

    Materializes the full [S, S] logit matrix. Memory is O(B*H*S^2), so this
    is for correctness on modest shapes, never for speed.

    Args:
        q: Queries shaped [B, H, S, D].
        k: Keys shaped [B, H, S, D].
        v: Values shaped [B, H, S, D].
        cfg: Configuration.

    Returns:
        Attention output shaped [B, H, S, D], same dtype as `q`.
    """
    b, h, s, d = q.shape
    scale = d ** -0.5

    kbar = block_representatives(k, cfg)
    selected = select_blocks(q, kbar, cfg)           # [B, H, S, num_blocks]

    # Expand block-level selection to token-level visibility. Indexing the
    # block axis with a per-token block id broadcasts each block's decision
    # across all of its keys.
    pos = torch.arange(s, device=q.device)
    block_of_token = pos // cfg.block_size            # [S]
    visible = selected[..., block_of_token]           # [B, H, S, S]

    # Causal masking. Strictly, only the current block needs it -- every other
    # selected block lies wholly in the past. Applying it across the board is
    # equivalent and removes a special case, which matters in an oracle whose
    # job is to be obviously right rather than fast.
    causal = pos.view(-1, 1) >= pos.view(1, -1)       # [S, S]
    mask = visible & causal.view(1, 1, s, s)

    logits = torch.einsum("bhid,bhjd->bhij", q, k) * scale
    logits = logits.masked_fill(~mask, float("-inf"))
    probs = torch.softmax(logits, dim=-1)
    return torch.einsum("bhij,bhjd->bhid", probs, v)
