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
from moba.kernels.index import build_csc, csc_to_mask
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
    actual = csc_to_mask(index, shape[2])

    assert torch.equal(actual, expected), (
        "CSC does not round-trip; the index and the routing disagree"
    )


def test_col_offsets_is_a_valid_exclusive_prefix_sum(device):
    """Offsets must start at zero and step by exactly each column's count."""
    cfg = MoBAConfig(block_size=32, top_k=4, dtype=torch.bfloat16)
    q, k, _ = make_qkv((1, 4, 512, 64), dtype=torch.bfloat16)
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
    q, k, _ = make_qkv((1, 2, 512, 64), dtype=torch.bfloat16)
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
    q, k, _ = make_qkv((1, 2, 512, 64), dtype=torch.bfloat16)
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
    q, k, _ = make_qkv((1, 4, 512, 64), dtype=torch.bfloat16)
    ix = build_csc(q, k, cfg)

    sorted_counts = torch.gather(ix.col_nnz, -1, ix.order.long())
    assert torch.all(sorted_counts[..., 1:] <= sorted_counts[..., :-1]), (
        "order does not sort columns by descending consumer count"
    )


def test_total_pairs_matches_the_mask(device):
    """Nothing may be dropped or duplicated in the transpose."""
    cfg = MoBAConfig(block_size=16, top_k=2, dtype=torch.bfloat16)
    q, k, _ = make_qkv((1, 4, 512, 64), dtype=torch.bfloat16)
    ix = build_csc(q, k, cfg)
    mask = moba_selection(q, k, cfg)

    assert int(ix.col_nnz.sum()) == int(mask.sum())
