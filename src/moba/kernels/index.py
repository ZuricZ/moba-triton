"""Turns per-query routing into the KV-block-major arrays the kernel needs.

The gate answers "for each query, which blocks?". The inverted kernel asks the
transpose: "for each block, which queries?". This module is that transpose,
stored as CSC -- columns are KV blocks, entries are query indices. See
`docs/superpowers/specs/2026-08-18-moba-kernel-design.md` decisions D6 and D8.

Kept separate from the kernels because D11 schedules the gate for replacement
by a Triton implementation; the boundary means that swap touches one file.

In v1 the gate is the oracle's own `moba_selection`, so routing agrees with the
oracle by construction and only the *conversion* is novel. That is deliberate:
it isolates the new failure mode (a wrong transpose) from the old one (wrong
routing), and a wrong transpose does not crash -- it makes the kernel attend to
the wrong keys and return plausible numbers.
"""

from dataclasses import dataclass

import torch

from moba.config import MoBAConfig
from moba.ref.naive import moba_selection


@dataclass(frozen=True)
class CSCIndex:
    """KV-block-major view of the routing decision.

    Attributes:
        col_nnz: Queries selecting each block, [B, H, num_blocks] int32.
        col_offsets: Exclusive prefix sum of `col_nnz`, [B, H, num_blocks + 1]
            int32. Column `j` occupies `row_indices[..., col_offsets[j] :
            col_offsets[j] + col_nnz[j]]`.
        row_indices: Query indices grouped by block and ascending within each
            block, [B, H, T] int32. Allocated at the static upper bound
            `T = S * top_k`; entries past a column's extent are undefined.
        slot_of_pair: Which of the query's `top_k` output slots each pair owns,
            [B, H, T] int32, aligned with `row_indices`.
        order: Block ids sorted by descending `col_nnz`, [B, H, num_blocks]
            int32. Longest-first dispatch (D8).
        num_blocks: Number of KV blocks.
        max_slots: `top_k`; the number of partial slots reserved per query.
    """

    col_nnz: torch.Tensor
    col_offsets: torch.Tensor
    row_indices: torch.Tensor
    slot_of_pair: torch.Tensor
    order: torch.Tensor
    num_blocks: int
    max_slots: int


def build_csc(q: torch.Tensor, k: torch.Tensor, cfg: MoBAConfig) -> CSCIndex:
    """Builds the KV-block-major index for one batch of queries and keys.

    Args:
        q: Queries shaped [B, H, S, D].
        k: Keys shaped [B, H, S, D].
        cfg: Configuration supplying `block_size` and `top_k`.

    Returns:
        A populated CSCIndex.
    """
    mask = moba_selection(q, k, cfg)                    # [B, H, S, N] bool
    b, h, s, n = mask.shape
    device = mask.device
    slots = cfg.top_k
    total = s * slots

    # Slot number is the rank of this block among the query's selections,
    # ordered by ascending block index. Any fixed rule works; having *a* fixed
    # rule is what makes the combine pass deterministic, since it fixes the
    # order in which partial sums are added.
    slot_grid = (mask.cumsum(dim=-1) - 1).to(torch.int32)

    # Block-major flattening. Reading [B, H, N, S] in memory order visits all
    # of column 0, then column 1, and so on -- which *is* CSC ordering, with
    # query indices ascending inside each column at no extra cost.
    #
    # The alternative was argsort on a `block * S + query` key. At S=4096 that
    # sorts ~1M elements per (batch, head); this costs one scan and no sort,
    # because the layout already is the sorted order.
    bm = mask.permute(0, 1, 3, 2).reshape(b * h, n * s)

    flat = bm.reshape(-1).nonzero(as_tuple=False).squeeze(1)
    bh_of = torch.div(flat, n * s, rounding_mode="floor")
    within = flat - bh_of * (n * s)
    n_of = torch.div(within, s, rounding_mode="floor")
    s_of = within - n_of * s

    # Position within each (batch, head)'s own segment of the global pair list.
    counts = bm.sum(dim=1)                              # [B*H]
    starts = torch.zeros_like(counts)
    starts[1:] = counts.cumsum(0)[:-1]
    pos = torch.arange(flat.numel(), device=device) - starts[bh_of]

    row_indices = torch.zeros(b * h, total, dtype=torch.int32, device=device)
    slot_of_pair = torch.zeros(b * h, total, dtype=torch.int32, device=device)
    row_indices[bh_of, pos] = s_of.to(torch.int32)
    slot_of_pair[bh_of, pos] = slot_grid.reshape(b * h, s, n)[bh_of, s_of, n_of]

    col_nnz = mask.sum(dim=2).to(torch.int32)           # [B, H, N]
    col_offsets = torch.zeros(b, h, n + 1, dtype=torch.int32, device=device)
    col_offsets[..., 1:] = col_nnz.cumsum(dim=-1)

    # Longest-first dispatch (D8). Stable, so the order is reproducible run to
    # run -- an unstable sort here would make the kernel's dispatch order vary,
    # which is indistinguishable from a race when debugging.
    order = torch.argsort(
        col_nnz.to(torch.int64), dim=-1, descending=True, stable=True
    ).to(torch.int32)

    return CSCIndex(
        col_nnz=col_nnz,
        col_offsets=col_offsets,
        row_indices=row_indices.view(b, h, total),
        slot_of_pair=slot_of_pair.view(b, h, total),
        order=order,
        num_blocks=n,
        max_slots=slots,
    )


def csc_to_mask(index: CSCIndex, seq_len: int) -> torch.Tensor:
    """Expands a CSC index back to a dense boolean selection mask.

    Test support only. Deliberately slow and obvious: it is the inverse the
    round-trip test needs, and an inverse that shared the forward direction's
    cleverness could share its bugs and agree while both were wrong.

    Args:
        index: The index to expand.
        seq_len: Sequence length.

    Returns:
        Boolean mask shaped [B, H, seq_len, num_blocks].
    """
    b, h, n = index.col_nnz.shape
    out = torch.zeros(
        b, h, seq_len, n, dtype=torch.bool, device=index.col_nnz.device
    )
    for bi in range(b):
        for hi in range(h):
            for blk in range(n):
                lo = int(index.col_offsets[bi, hi, blk])
                cnt = int(index.col_nnz[bi, hi, blk])
                rows = index.row_indices[bi, hi, lo:lo + cnt].long()
                out[bi, hi, rows, blk] = True
    return out
