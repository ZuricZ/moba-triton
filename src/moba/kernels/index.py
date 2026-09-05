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
from moba.heads import head_group_size
from moba.ref.naive import moba_selection


@dataclass(frozen=True)
class CSCIndex:
    """KV-block-major view of the routing decision.

    Attributes:
        col_nnz: Queries selecting each block, [B, H_kv, num_blocks] int32.
        col_offsets: Exclusive prefix sum of `col_nnz`, [B, H_kv, num_blocks + 1]
            int32. Column `j` occupies `row_indices[..., col_offsets[j] :
            col_offsets[j] + col_nnz[j]]`.
        row_indices: Query positions grouped by block, [B, H_kv, T] int32.
            Allocated at the static upper bound `T = G * S * top_k`; entries
            past a column's extent are undefined.

            Ordering within a column is lexicographic by (query head, position)
            -- ascending within each head's run, not across the column. Do not
            try to restore a global ascending order: what mattered was never
            the ordering but the *locality* of the kernel's gather, and
            head-major is the layout that has it. Consecutive entries then step
            one query row; position-major would step `S` rows, 2 MB apart at
            S=8192. `index_fused` drops ordering entirely and measured 1.03x
            (`docs/gather-cost-measurements.md`).
        head_indices: Which query head each entry came from, [B, H_kv, T]
            int32, aligned with `row_indices`. Required because a column is
            keyed by (batch, kv_head, block), which under grouped-query
            attention no longer determines the query head: every query head
            sharing that kv head lands in the same column.

            Held separately rather than packed into `row_indices` as
            `head * S + pos` so the kernels' causal comparison
            `key_pos <= rows` stays a raw position compare, with no runtime
            integer modulo in the inner loop.
        slot_of_pair: Which of the query's `top_k` output slots each pair owns,
            [B, H_kv, T] int32, aligned with `row_indices`.
        order: Block ids sorted by descending `col_nnz`, [B, H_kv, num_blocks]
            int32. Longest-first dispatch (D8).
        num_blocks: Number of KV blocks.
        max_slots: `top_k`; the number of partial slots reserved per query.
        num_query_heads: H_q. Equal to `col_nnz.shape[1]` for multi-head
            attention, a multiple of it under grouped-query attention.
        cu_seqlens: Cumulative token counts when this index describes a packed
            batch, else None. Carried so the attention and backward kernels need
            no extra argument, and so a mismatched index cannot silently be used
            with the wrong mode.
        max_seqlen: Longest sequence of a packed batch, else None. `num_blocks`
            is `cdiv(max_seqlen, block_size)` in that case -- the *padded index*
            of a packed batch, one row per sequence.
    """

    col_nnz: torch.Tensor
    col_offsets: torch.Tensor
    row_indices: torch.Tensor
    head_indices: torch.Tensor
    slot_of_pair: torch.Tensor
    order: torch.Tensor
    num_blocks: int
    max_slots: int
    num_query_heads: int
    cu_seqlens: torch.Tensor | None = None
    max_seqlen: int | None = None


def build_csc(q: torch.Tensor, k: torch.Tensor, cfg: MoBAConfig) -> CSCIndex:
    """Builds the KV-block-major index for one batch of queries and keys.

    Args:
        q: Queries shaped [B, S, H, D].
        k: Keys shaped [B, S, H_kv, D].
        cfg: Configuration supplying `block_size` and `top_k`.

    Returns:
        A populated CSCIndex.
    """
    mask = moba_selection(q, k, cfg)                # [B, H_q, S, N] bool
    b, h_q, s, n = mask.shape
    h_kv = k.shape[2]                               # token-major head axis
    group = head_group_size(h_q, h_kv)
    device = mask.device
    slots = cfg.top_k
    total = group * s * slots

    # Slot number is the rank of this block among the query's selections,
    # ordered by ascending block index. Any fixed rule works; having *a* fixed
    # rule is what makes the combine pass deterministic, since it fixes the
    # order in which partial sums are added.
    slot_grid = (mask.cumsum(dim=-1) - 1).to(torch.int32)

    # Block-major flattening. Reading [B, H_kv, N, G, S] in memory order visits
    # all of column 0, then column 1, and so on -- which *is* CSC ordering, at
    # no extra cost.
    #
    # The alternative was argsort on a `block * S + query` key. At S=4096 that
    # sorts ~1M elements per (batch, head); this costs one scan and no sort,
    # because the layout already is the sorted order.
    #
    # Grouped-query attention stacks the group's query heads onto the *row*
    # axis of a shared grid, because the columns they index are now the same
    # physical blocks. `view` splits H_q into (H_kv, G) with the group index
    # fastest, matching the convention that query head `qh` reads kv head
    # `qh // G`. Nothing is copied; the permute reorders strides only.
    #
    # Order within a column is then lexicographic by (head, position). See
    # `CSCIndex.row_indices` for why that is preferred to a globally ascending
    # order rather than merely tolerated.
    stacked = mask.view(b, h_kv, group, s, n)           # [B, H_kv, G, S, N]
    bm = stacked.permute(0, 1, 4, 2, 3).reshape(b * h_kv, n * group * s)

    flat = bm.reshape(-1).nonzero(as_tuple=False).squeeze(1)
    bh_of = torch.div(flat, n * group * s, rounding_mode="floor")
    within = flat - bh_of * (n * group * s)
    n_of = torch.div(within, group * s, rounding_mode="floor")
    rem = within - n_of * (group * s)
    g_of = torch.div(rem, s, rounding_mode="floor")
    s_of = rem - g_of * s

    # Position within each (batch, head)'s own segment of the global pair list.
    counts = bm.sum(dim=1)                              # [B*H_kv]
    starts = torch.zeros_like(counts)
    starts[1:] = counts.cumsum(0)[:-1]
    pos = torch.arange(flat.numel(), device=device) - starts[bh_of]

    row_indices = torch.zeros(b * h_kv, total, dtype=torch.int32, device=device)
    head_indices = torch.zeros(b * h_kv, total, dtype=torch.int32, device=device)
    slot_of_pair = torch.zeros(b * h_kv, total, dtype=torch.int32, device=device)

    row_indices[bh_of, pos] = s_of.to(torch.int32)

    # The *absolute* query head, not its index within the group, so the kernels
    # address Q directly without having to know which group they are in.
    kv_of = bh_of % h_kv
    head_indices[bh_of, pos] = (kv_of * group + g_of).to(torch.int32)

    sg = slot_grid.reshape(b * h_kv, group, s, n)
    slot_of_pair[bh_of, pos] = sg[bh_of, g_of, s_of, n_of]

    col_nnz = stacked.sum(dim=(2, 3)).to(torch.int32)   # [B, H_kv, N]
    col_offsets = torch.zeros(b, h_kv, n + 1, dtype=torch.int32, device=device)
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
        row_indices=row_indices.view(b, h_kv, total),
        head_indices=head_indices.view(b, h_kv, total),
        slot_of_pair=slot_of_pair.view(b, h_kv, total),
        order=order,
        num_blocks=n,
        max_slots=slots,
        num_query_heads=h_q,
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
        Boolean mask shaped [B, H_q, seq_len, num_blocks], or
        [1, H_q, T, num_blocks] when the index describes a packed batch --
        rows are packed token indices and block ids are sequence-local, the
        same convention `moba.ref.naive.moba_selection` produces. Columns are
        keyed by kv head, but the mask is per query head, so `head_indices` is
        what routes each entry back.
    """
    b, h_kv, n = index.col_nnz.shape
    packed = index.cu_seqlens is not None

    # Packed: one batch axis of 1, because rows are packed token indices and
    # every sequence shares the token axis. The block axis stays sequence-local
    # and `n` wide, matching `moba.ref.naive.moba_selection`'s convention.
    out = torch.zeros(
        1 if packed else b, index.num_query_heads, seq_len, n,
        dtype=torch.bool, device=index.col_nnz.device
    )
    for bi in range(b):
        for kv in range(h_kv):
            for blk in range(n):
                lo = int(index.col_offsets[bi, kv, blk])
                cnt = int(index.col_nnz[bi, kv, blk])
                rows = index.row_indices[bi, kv, lo:lo + cnt].long()
                heads = index.head_indices[bi, kv, lo:lo + cnt].long()
                out[0 if packed else bi, heads, rows, blk] = True
    return out
