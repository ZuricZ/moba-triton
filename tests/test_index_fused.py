"""The device-side index build against the PyTorch build it replaces.

`build_csc` stays in the tree as the oracle. It is a good one for this job:
it reaches the same answer by a completely different route -- a dense mask,
`nonzero`, and a cumulative sum -- so a bug would have to occur twice, in two
unrelated pieces of arithmetic, to escape both.

What is *not* compared is the order of query indices within a column. The
PyTorch builder produces them ascending; this one lets queries claim slots with
an atomic counter, so they come out in whatever order the programs raced. Every
test here is therefore written to be order-insensitive, and
`test_column_membership_is_correct` checks the *set* rather than the sequence.
"""

import warnings

import pytest
import torch

from conftest import GATE_SWEEP, SHAPE_SWEEP, make_qkv
from moba.config import MoBAConfig
from moba.kernels.index import build_csc, csc_to_mask
from moba.kernels.index_fused import build_csc_fused


@pytest.mark.parametrize("shape", SHAPE_SWEEP)
@pytest.mark.parametrize("gate", GATE_SWEEP)
def test_fused_index_routes_identically(device, shape, gate):
    """The same queries must reach the same blocks, exactly."""
    block_size, top_k = gate
    cfg = MoBAConfig(block_size=block_size, top_k=top_k, dtype=torch.bfloat16)
    q, k, _ = make_qkv(shape, dtype=torch.bfloat16, device=device)
    seq_len = shape[2]

    expected = build_csc(q, k, cfg)
    actual = build_csc_fused(q, k, cfg)

    assert torch.equal(csc_to_mask(expected, seq_len),
                       csc_to_mask(actual, seq_len)), "routing differs"
    assert torch.equal(expected.col_nnz, actual.col_nnz), "counts differ"
    assert torch.equal(expected.col_offsets, actual.col_offsets), (
        "column offsets differ"
    )


@pytest.mark.parametrize("gate", GATE_SWEEP)
def test_column_membership_is_correct(device, gate):
    """Each column holds exactly the queries that selected it -- as a set.

    Order within a column is deliberately unspecified here, so this compares
    sorted membership. A scatter that dropped or duplicated a pair would show
    up as a membership difference even though the counts still matched, which
    is why this exists alongside the count check.
    """
    block_size, top_k = gate
    cfg = MoBAConfig(block_size=block_size, top_k=top_k, dtype=torch.bfloat16)
    q, k, _ = make_qkv((1, 2, 1024, 128), dtype=torch.bfloat16, device=device)

    expected, actual = build_csc(q, k, cfg), build_csc_fused(q, k, cfg)

    for b in range(expected.col_nnz.shape[0]):
        for h in range(expected.col_nnz.shape[1]):
            for blk in range(expected.num_blocks):
                lo = int(expected.col_offsets[b, h, blk])
                n = int(expected.col_nnz[b, h, blk])
                want = expected.row_indices[b, h, lo:lo + n].sort().values
                got = actual.row_indices[b, h, lo:lo + n].sort().values
                assert torch.equal(want, got), (
                    f"column {blk} membership differs at batch {b}, head {h}"
                )


@pytest.mark.parametrize("gate", GATE_SWEEP)
def test_slot_of_pair_agrees_with_the_query_it_belongs_to(device, gate):
    """Every pair's slot must match the block it sits in.

    `slot_of_pair` is what the attention kernel uses to address its partial
    output. A wrong slot does not crash and does not change which keys are
    read -- it writes a correct partial into the wrong mailbox, and the merge
    then combines the wrong things. Silent and plausible, so it is checked
    directly rather than only through the final numbers.
    """
    block_size, top_k = gate
    cfg = MoBAConfig(block_size=block_size, top_k=top_k, dtype=torch.bfloat16)
    q, k, _ = make_qkv((1, 2, 1024, 128), dtype=torch.bfloat16, device=device)

    ref, ix = build_csc(q, k, cfg), build_csc_fused(q, k, cfg)
    ref_mask = csc_to_mask(ref, 1024)

    for b in range(ix.col_nnz.shape[0]):
        for h in range(ix.col_nnz.shape[1]):
            for blk in range(ix.num_blocks):
                lo = int(ix.col_offsets[b, h, blk])
                n = int(ix.col_nnz[b, h, blk])
                rows = ix.row_indices[b, h, lo:lo + n]
                slots = ix.slot_of_pair[b, h, lo:lo + n]
                for row, slot in zip(rows.tolist(), slots.tolist()):
                    # Slot j means "the j-th block this query chose, counting
                    # up by block index".
                    rank = int(ref_mask[b, h, row, :blk].sum())
                    assert slot == rank, (
                        f"query {row} in block {blk} has slot {slot}, "
                        f"expected {rank}"
                    )


def test_dispatch_order_is_longest_first_and_repeatable(device):
    """D8 needs the heaviest columns first, and needs the same order twice.

    Repeatability is the load-bearing half. An unstable order would vary which
    program handles which block between runs, and a kernel whose scheduling
    changes run to run is nearly indistinguishable from one with a race.
    """
    cfg = MoBAConfig(block_size=32, top_k=4, dtype=torch.bfloat16)
    q, k, _ = make_qkv((1, 2, 1024, 128), dtype=torch.bfloat16, device=device)

    ix = build_csc_fused(q, k, cfg)
    counts = torch.gather(ix.col_nnz, -1, ix.order.long())

    assert bool((counts[..., 1:] <= counts[..., :-1]).all()), (
        "columns are not ordered by descending consumer count"
    )
    assert torch.equal(ix.order, build_csc_fused(q, k, cfg).order), (
        "dispatch order is not repeatable"
    )


def test_fused_index_never_synchronizes_with_the_host(device):
    """The whole point of the rewrite, asserted so it cannot quietly return.

    A host round trip costs ~230 us on this WSL box against ~6 us for the same
    arithmetic kept on device -- a 37x penalty. The PyTorch builder needed two,
    because `nonzero` returns a data-dependent count that must reach the CPU
    before an allocation can be sized, and mean-pooling wrote the tail block's
    occupancy from the host.

    Nothing about a reintroduced sync fails a correctness test; it just makes
    everything slower. So it is tested as a property.
    """
    cfg = MoBAConfig(block_size=128, top_k=8, dtype=torch.bfloat16)
    q, k, _ = make_qkv((1, 2, 1024, 128), dtype=torch.bfloat16, device=device)

    build_csc_fused(q, k, cfg)          # compile before measuring
    torch.cuda.synchronize()

    torch.cuda.set_sync_debug_mode("warn")
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            build_csc_fused(q, k, cfg)
    finally:
        torch.cuda.set_sync_debug_mode("default")

    syncs = [w for w in caught if "synchron" in str(w.message).lower()]
    assert not syncs, (
        f"build_csc_fused performed {len(syncs)} host synchronization(s): "
        f"{[str(w.message) for w in syncs]}"
    )


def test_fused_index_handles_a_sequence_that_is_not_a_block_multiple(device):
    """A short tail block must be pooled and routed like any other.

    Padding to a block multiple and taking a plain mean would divide a short
    tail by the full block size, shrinking its representative toward zero so it
    is almost never selected. The bug is invisible whenever seq_len divides
    evenly, which every other test in this file does.
    """
    cfg = MoBAConfig(block_size=128, top_k=4, dtype=torch.bfloat16)
    q, k, _ = make_qkv((1, 2, 1000, 128), dtype=torch.bfloat16, device=device)

    expected, actual = build_csc(q, k, cfg), build_csc_fused(q, k, cfg)

    assert actual.num_blocks == 8, f"expected 8 blocks, got {actual.num_blocks}"
    assert torch.equal(csc_to_mask(expected, 1000), csc_to_mask(actual, 1000))


def _gqa_qk(device, b=1, h_q=8, h_kv=2, s=512, d=64, seed=0):
    """Returns (q, k) with fewer key/value heads than query heads."""
    gen = torch.Generator(device=device).manual_seed(seed)
    q = torch.randn(b, h_q, s, d, generator=gen, device=device,
                    dtype=torch.bfloat16)
    k = torch.randn(b, h_kv, s, d, generator=gen, device=device,
                    dtype=torch.bfloat16)
    return q, k


def test_fused_index_merges_columns_across_a_head_group(device):
    """Same pairs, fewer columns -- the whole point of grouped-query support.

    A column is one program's job and loads K and V once, so `group` times
    fewer columns is `group` times fewer K/V loads for identical arithmetic.
    See `docs/learning/03-gqa-and-the-column-merge.md`.
    """
    cfg = MoBAConfig(block_size=64, top_k=4, dtype=torch.bfloat16)
    b, h_q, h_kv, s = 1, 8, 2, 512
    q, k = _gqa_qk(device, b=b, h_q=h_q, h_kv=h_kv, s=s)

    index = build_csc_fused(q, k, cfg)
    assert index.col_nnz.shape == (b, h_kv, cfg.num_blocks(s))
    assert index.head_indices.shape == index.row_indices.shape
    assert index.num_query_heads == h_q

    mha_pairs = int(build_csc(q, k, cfg).col_nnz.sum())
    assert int(index.col_nnz.sum()) == mha_pairs


def test_fused_index_agrees_with_the_pytorch_builder_under_gqa(device):
    """Order-insensitive comparison, as everywhere else in this file.

    The atomic cursor permutes each column, so the round-trip mask is the
    comparable object rather than the raw arrays.
    """
    cfg = MoBAConfig(block_size=64, top_k=4, dtype=torch.bfloat16)
    q, k = _gqa_qk(device, s=512, seed=1)

    ours = csc_to_mask(build_csc_fused(q, k, cfg), 512)
    ref = csc_to_mask(build_csc(q, k, cfg), 512)
    assert ours.shape == (1, 8, 512, cfg.num_blocks(512))
    assert torch.equal(ours, ref)


def test_fused_entries_name_a_head_inside_their_own_group(device):
    """A wrong kv-head divisor would file entries under the wrong column.

    It would not crash and the shapes would be right; the kernel would simply
    attend to another head's keys. Checked directly on the stored heads.
    """
    cfg = MoBAConfig(block_size=64, top_k=4, dtype=torch.bfloat16)
    b, h_q, h_kv, s = 1, 8, 2, 512
    q, k = _gqa_qk(device, b=b, h_q=h_q, h_kv=h_kv, s=s)
    group = h_q // h_kv

    index = build_csc_fused(q, k, cfg)
    for bi in range(b):
        for kv in range(h_kv):
            for blk in range(cfg.num_blocks(s)):
                lo = int(index.col_offsets[bi, kv, blk])
                cnt = int(index.col_nnz[bi, kv, blk])
                heads = index.head_indices[bi, kv, lo:lo + cnt]
                assert bool(((heads // group) == kv).all()), (
                    f"column ({kv}, {blk}) holds a head outside its group"
                )
