"""CSC index construction: does the transpose of the routing survive intact?

The gate answers "for each query, which blocks?". The kernel needs "for each
block, which queries?". These tests check that transpose, which is the part
that can silently be wrong -- a bad index does not crash, it makes the kernel
attend to the wrong keys and return plausible numbers.

In v1 `build_csc` calls the oracle's own `moba_selection`, so routing agrees by
construction and only the conversion is under test. When a Triton gate replaces
that call (spec D11, deferred), the round-trip becomes a genuine routing check
without changing.
"""

import pytest
import torch

from conftest import GATE_SWEEP, SHAPE_SWEEP, case_id, make_qkv
from moba.config import MoBAConfig
from moba.kernels.routing import build_csc, csc_to_mask
from moba.ref.naive import moba_selection

_CASES = [(s, g) for s in SHAPE_SWEEP for g in GATE_SWEEP]
_IDS = [case_id(s, g) for s in SHAPE_SWEEP for g in GATE_SWEEP]


@pytest.mark.parametrize("shape,gate", _CASES, ids=_IDS)
def test_csc_round_trips_to_the_oracle_mask(device, shape, gate):
    """Expanding the CSC back to a dense mask must reproduce it exactly."""
    block_size, top_k = gate
    cfg = MoBAConfig(block_size=block_size, top_k=top_k, dtype=torch.bfloat16)
    q, k, _ = make_qkv(shape, dtype=torch.bfloat16)

    expected = moba_selection(q, k, cfg)
    index = build_csc(q, k, cfg)
    actual = csc_to_mask(index, shape[1])   # token-major: seq_len is axis 1

    assert torch.equal(actual, expected), (
        "CSC does not round-trip; the index and the routing disagree"
    )


def test_col_offsets_is_a_valid_exclusive_prefix_sum(device):
    """Offsets must start at zero and step by exactly each column's count."""
    cfg = MoBAConfig(block_size=32, top_k=4, dtype=torch.bfloat16)
    q, k, _ = make_qkv((1, 512, 4, 64), dtype=torch.bfloat16)
    ix = build_csc(q, k, cfg)

    assert torch.equal(
        ix.col_offsets[..., 0], torch.zeros_like(ix.col_offsets[..., 0])
    )
    assert torch.equal(
        ix.col_offsets[..., 1:] - ix.col_offsets[..., :-1], ix.col_nnz
    )
    assert torch.equal(ix.col_offsets[..., -1], ix.col_nnz.sum(-1))


def test_row_indices_ascend_within_every_column(device):
    """Ascending rows preserve whatever DRAM locality the gather can get.

    The gather benchmark used a fully random permutation, so it is a
    pessimistic bound; keeping columns sorted costs nothing to produce.
    """
    cfg = MoBAConfig(block_size=32, top_k=4, dtype=torch.bfloat16)
    q, k, _ = make_qkv((1, 512, 2, 64), dtype=torch.bfloat16)
    ix = build_csc(q, k, cfg)

    for bi in range(ix.col_nnz.shape[0]):
        for hi in range(ix.col_nnz.shape[1]):
            for blk in range(ix.num_blocks):
                lo = int(ix.col_offsets[bi, hi, blk])
                cnt = int(ix.col_nnz[bi, hi, blk])
                rows = ix.row_indices[bi, hi, lo:lo + cnt]
                assert torch.all(rows[1:] > rows[:-1]), (
                    f"column {blk} is not strictly ascending"
                )


def test_slots_are_dense_and_unique_per_query(device):
    """Each query's slots must be exactly range(k_i).

    If two pairs for one query shared a slot they would race in the kernel, and
    if slots were sparse the combine pass would read uninitialised entries.
    """
    cfg = MoBAConfig(block_size=32, top_k=4, dtype=torch.bfloat16)
    q, k, _ = make_qkv((1, 512, 2, 64), dtype=torch.bfloat16)
    ix = build_csc(q, k, cfg)
    mask = moba_selection(q, k, cfg)

    b, h, n = ix.col_nnz.shape
    for bi in range(b):
        for hi in range(h):
            per_query: dict[int, list[int]] = {}
            for blk in range(n):
                lo = int(ix.col_offsets[bi, hi, blk])
                cnt = int(ix.col_nnz[bi, hi, blk])
                rows = ix.row_indices[bi, hi, lo:lo + cnt].tolist()
                slots = ix.slot_of_pair[bi, hi, lo:lo + cnt].tolist()
                for r, sl in zip(rows, slots):
                    per_query.setdefault(r, []).append(sl)
            for r, sl in per_query.items():
                expected = int(mask[bi, hi, r].sum())
                assert sorted(sl) == list(range(expected)), (
                    f"query {r} has slots {sorted(sl)}, expected "
                    f"{list(range(expected))}"
                )


def test_order_is_longest_first(device):
    """Dispatch order must put the heaviest columns first (spec D8)."""
    cfg = MoBAConfig(block_size=16, top_k=2, dtype=torch.bfloat16)
    q, k, _ = make_qkv((1, 512, 4, 64), dtype=torch.bfloat16)
    ix = build_csc(q, k, cfg)

    sorted_counts = torch.gather(ix.col_nnz, -1, ix.order.long())
    assert torch.all(sorted_counts[..., 1:] <= sorted_counts[..., :-1]), (
        "order does not sort columns by descending consumer count"
    )


def test_total_pairs_matches_the_mask(device):
    """Nothing may be dropped or duplicated in the transpose."""
    cfg = MoBAConfig(block_size=16, top_k=2, dtype=torch.bfloat16)
    q, k, _ = make_qkv((1, 512, 4, 64), dtype=torch.bfloat16)
    ix = build_csc(q, k, cfg)
    mask = moba_selection(q, k, cfg)

    assert int(ix.col_nnz.sum()) == int(mask.sum())


def _gqa_qkv(device, b=1, h_q=8, h_kv=2, s=512, d=64, seed=0):
    """Returns (q, k, v) with fewer key/value heads than query heads."""
    gen = torch.Generator(device=device).manual_seed(seed)
    q = torch.randn(b, s, h_q, d, generator=gen, device=device,
                    dtype=torch.float32)
    k = torch.randn(b, s, h_kv, d, generator=gen, device=device,
                    dtype=torch.float32)
    v = torch.randn(b, s, h_kv, d, generator=gen, device=device,
                    dtype=torch.float32)
    return q, k, v


def test_columns_merge_across_the_query_heads_that_share_a_kv_head(device):
    """Four query heads sharing one KV head produce one column, not four.

    This is the whole mechanism. The total number of (query, block) pairs is
    unchanged -- the same queries picked the same number of blocks -- but they
    are redistributed over `group` times fewer columns. A column is one
    program's job and loads K and V once, so `group` times fewer columns means
    `group` times fewer K/V loads for identical arithmetic. See
    `docs/learning/03-gqa-and-the-column-merge.md`.
    """
    cfg = MoBAConfig(block_size=64, top_k=4)
    b, h_q, h_kv, s = 1, 8, 2, 512
    q, k, _ = _gqa_qkv(device, b=b, h_q=h_q, h_kv=h_kv, s=s)

    index = build_csc(q, k, cfg)
    n = cfg.num_blocks(s)

    assert index.col_nnz.shape == (b, h_kv, n)
    assert index.head_indices.shape == index.row_indices.shape
    assert index.num_query_heads == h_q

    # Pair count is conserved: containers merged, entries did not multiply.
    assert int(index.col_nnz.sum()) == int(moba_selection(q, k, cfg).sum())


def test_every_entry_names_the_query_head_it_came_from(device):
    """A column no longer determines the head, so each entry must carry it.

    Checked against the dense mask directly: for every stored (head, row,
    block) triple the mask must be set, and the counts must agree. A
    `head_indices` that were all zero would still produce a well-formed index.
    """
    cfg = MoBAConfig(block_size=64, top_k=4)
    b, h_q, h_kv, s = 1, 8, 2, 512
    q, k, _ = _gqa_qkv(device, b=b, h_q=h_q, h_kv=h_kv, s=s)

    index = build_csc(q, k, cfg)
    mask = moba_selection(q, k, cfg)
    group = h_q // h_kv

    for bi in range(b):
        for kv in range(h_kv):
            for blk in range(cfg.num_blocks(s)):
                lo = int(index.col_offsets[bi, kv, blk])
                cnt = int(index.col_nnz[bi, kv, blk])
                rows = index.row_indices[bi, kv, lo:lo + cnt].long()
                heads = index.head_indices[bi, kv, lo:lo + cnt].long()
                assert bool(mask[bi, heads, rows, blk].all()), (
                    f"column ({kv}, {blk}) names an unselected pair"
                )
                # Every head in the entry must belong to this kv head's group.
                assert bool(((heads // group) == kv).all()), (
                    f"column ({kv}, {blk}) names a head outside its group"
                )


def test_round_trip_reproduces_the_selection_under_gqa(device):
    """The transpose must be lossless with the head axis merged away."""
    cfg = MoBAConfig(block_size=64, top_k=4)
    q, k, _ = _gqa_qkv(device, s=512)

    index = build_csc(q, k, cfg)
    got = csc_to_mask(index, 512)
    assert got.shape == (1, 8, 512, cfg.num_blocks(512))
    assert torch.equal(got, moba_selection(q, k, cfg))


def test_slots_still_identify_a_query_s_own_selection_rank_under_gqa(device):
    """`slot_of_pair` is per query, so merging columns must not disturb it.

    Slots address the partial buffers; two entries for the same query with the
    same slot would collide and one output would be lost. Checked by pairing
    (head, row, slot) across the whole index and requiring uniqueness.
    """
    cfg = MoBAConfig(block_size=64, top_k=4)
    b, h_q, h_kv, s = 1, 8, 2, 512
    q, k, _ = _gqa_qkv(device, b=b, h_q=h_q, h_kv=h_kv, s=s)

    index = build_csc(q, k, cfg)
    seen = set()
    for bi in range(b):
        for kv in range(h_kv):
            for blk in range(cfg.num_blocks(s)):
                lo = int(index.col_offsets[bi, kv, blk])
                cnt = int(index.col_nnz[bi, kv, blk])
                for i in range(lo, lo + cnt):
                    key = (bi,
                           int(index.head_indices[bi, kv, i]),
                           int(index.row_indices[bi, kv, i]),
                           int(index.slot_of_pair[bi, kv, i]))
                    assert key not in seen, f"duplicate (query, slot): {key}"
                    seen.add(key)
                    assert key[3] < cfg.top_k, "slot outside the budget"
