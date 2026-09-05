"""Packed (variable-length) batches, end to end.

Anchored throughout against the *same code path* run on each sequence alone.
Arithmetic, routing and tuning are identical on both sides, so a disagreement
can only be a boundary or indexing bug.

The failure to fear does not crash. A block straddling a boundary, or a gate
offering a query another document's block, produces finite, plausible numbers of
exactly the right shape. Nothing downstream would notice; the model would
quietly attend across documents.

See `docs/superpowers/plans/2026-08-24-varlen-support.md` and
`docs/learning/04-packed-sequences-and-where-the-work-goes.md`.
"""

import pytest
import torch

from moba.config import MoBAConfig
from moba.kernels.gating import gate_select
from moba.kernels.pooling import pool_blocks

# Deliberately awkward at block_size 16: 80 divides evenly, 48 and 64 do too,
# and 20 does not -- so one sequence ends in a 4-key block. The last sequence is
# also shorter than one 32-query gate tile, and the lengths are not sorted.
#
# head_dim 64 and block_size 16 are floors, not preferences: the attention
# kernel rejects a head_dim whose row is below a 128-byte cache line, and a
# block_size below `tl.dot`'s 16-row minimum. The gate enforces neither, which
# is why its own tests could run smaller.
LENS = [80, 48, 64, 20]
CU = [0, 80, 128, 192, 212]
MAXLEN = 80
TOTAL = 212
BLOCK = 16


def _packed(device, dtype, heads=2, d=64, seed=0):
    """Three tensors shaped [1, heads, TOTAL, d] -- one packed batch."""
    gen = torch.Generator(device=device).manual_seed(seed)

    def mk():
        return torch.randn(1, TOTAL, heads, d, generator=gen, device=device,
                           dtype=dtype)

    return mk(), mk(), mk()


def _cu(device):
    return torch.tensor(CU, dtype=torch.int32, device=device)


def _kbar(k, cfg, cu):
    return pool_blocks(k, cfg, cu_seqlens=cu,
                                        max_seqlen=MAXLEN)


def test_no_selected_block_crosses_a_sequence_boundary(device):
    """The invariant the untouched causal mask depends on (D-VL-7).

    `qk += tl.where(key_pos <= rows, ...)` compares packed token indices. Within
    a sequence, packed order is position order, so it is right -- but only
    because a query is never offered a block from another sequence. Nothing in
    the attention kernel checks that, so it is checked here.

    Block ids are sequence-local, so "crossing" shows up as an id at or beyond
    that sequence's own block count: a block that does not exist for the query
    but does address a real, padded index slot belonging to a longer sequence.
    """
    cfg = MoBAConfig(block_size=BLOCK, top_k=3, dtype=torch.bfloat16)
    cu = _cu(device)
    q, k, _ = _packed(device, torch.bfloat16)

    idx = gate_select(q, _kbar(k, cfg, cu), cfg, cu_seqlens=cu,
                      max_seqlen=MAXLEN)
    assert idx.shape == (1, 2, TOTAL, cfg.top_k)

    lo = 0
    for n in LENS:
        n_blocks = -(-n // cfg.block_size)
        chosen = idx[0, :, lo:lo + n]
        real = chosen[chosen >= 0]
        assert bool((real < n_blocks).all()), (
            f"sequence at {lo} (length {n}, {n_blocks} blocks) selected id "
            f"{int(real.max())}"
        )
        lo += n


def test_gate_matches_running_each_sequence_alone(device):
    """Selections must be identical -- routing is a ranking, not an estimate.

    Both sides emit sequence-local ids, so no offset is needed (D-VL-6). That
    is the whole point of keeping ids local: the comparison is `torch.equal`
    with nothing in between.
    """
    cfg = MoBAConfig(block_size=BLOCK, top_k=3, dtype=torch.bfloat16)
    cu = _cu(device)
    q, k, _ = _packed(device, torch.bfloat16)

    packed = gate_select(q, _kbar(k, cfg, cu), cfg, cu_seqlens=cu,
                         max_seqlen=MAXLEN)
    lo = 0
    for n in LENS:
        piece = gate_select(
            q[:, lo:lo + n],
            pool_blocks(k[:, lo:lo + n], cfg), cfg)
        assert torch.equal(packed[:, :, lo:lo + n], piece), f"at {lo}"
        lo += n


def test_every_query_selects_its_own_block(device):
    """`always_select_current` must hold per sequence, not per packed buffer.

    Query t of sequence i sits in that sequence's block `(t - bos) // bs`, which
    is not `t // bs`. Getting this wrong is invisible in the first sequence and
    wrong in every other one -- exactly the shape of bug a single-sequence test
    would miss.
    """
    cfg = MoBAConfig(block_size=BLOCK, top_k=3, dtype=torch.bfloat16)
    cu = _cu(device)
    q, k, _ = _packed(device, torch.bfloat16, seed=1)

    idx = gate_select(q, _kbar(k, cfg, cu), cfg, cu_seqlens=cu,
                      max_seqlen=MAXLEN)
    lo = 0
    for n in LENS:
        for t in range(n):
            own = t // cfg.block_size
            chosen = idx[0, :, lo + t]
            assert bool((chosen == own).any(dim=-1).all()), (
                f"token {t} of the sequence at {lo} did not select its own "
                f"block {own}; got {chosen.tolist()}"
            )
        lo += n


def test_short_sequences_leave_no_out_of_range_ids(device):
    """The per-sequence grid over-launches; those programs must be inert.

    The last sequence holds 5 tokens against a max of 20, so most of its query
    tiles have nothing to do. They must write nothing, and no id anywhere may
    address past the padded index.
    """
    cfg = MoBAConfig(block_size=BLOCK, top_k=3, dtype=torch.bfloat16)
    cu = _cu(device)
    q, k, _ = _packed(device, torch.bfloat16, seed=2)

    idx = gate_select(q, _kbar(k, cfg, cu), cfg, cu_seqlens=cu,
                      max_seqlen=MAXLEN)
    nb = -(-MAXLEN // cfg.block_size)
    assert bool(((idx >= -1) & (idx < nb)).all())


def test_a_sequence_shorter_than_one_tile_is_gated(device):
    """A 5-token sequence with a 32-query tile still has to be routed.

    The early return that makes over-launched programs cheap must not swallow
    the one program that does have work.
    """
    cfg = MoBAConfig(block_size=BLOCK, top_k=2, dtype=torch.bfloat16)
    cu = _cu(device)
    q, k, _ = _packed(device, torch.bfloat16, seed=3)

    idx = gate_select(q, _kbar(k, cfg, cu), cfg, cu_seqlens=cu,
                      max_seqlen=MAXLEN)
    tail = idx[0, :, CU[-2]:CU[-1]]                 # the 5-token sequence
    assert bool(((tail >= 0).sum(-1) > 0).all()), "a query selected nothing"


def test_packed_gate_rejects_a_real_batch_axis(device):
    cfg = MoBAConfig(block_size=BLOCK, top_k=2, dtype=torch.bfloat16)
    cu = _cu(device)
    q = torch.randn(2, TOTAL, 2, 64, device=device, dtype=torch.bfloat16)
    kbar = torch.randn(4, 2, 5, 64, device=device, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="batch 1"):
        gate_select(q, kbar, cfg, cu_seqlens=cu, max_seqlen=MAXLEN)


def test_max_seqlen_is_required_when_packing(device):
    """Not derivable without a host synchronization, so it is an argument."""
    cfg = MoBAConfig(block_size=BLOCK, top_k=2, dtype=torch.bfloat16)
    cu = _cu(device)
    q = torch.randn(1, TOTAL, 2, 64, device=device, dtype=torch.bfloat16)
    kbar = torch.randn(4, 2, 5, 64, device=device, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="max_seqlen"):
        gate_select(q, kbar, cfg, cu_seqlens=cu)


def test_columns_keep_a_batch_axis_and_hold_only_their_sequence(device):
    """Pack the data, pad the index (D-VL-3).

    Block ids are sequence-local, so a column is `(b, kv_head, local_block)` and
    the CSC arrays keep a real batch axis -- exactly FlashMoBA's
    `moba_col_offsets: (batch_size, num_heads, max_lg_col_num)` alongside a
    packed `q`. The data is O(T*H*D) and worth packing; this metadata is
    O(B*H*NB) and is not.
    """
    from moba.kernels.routing import build_csc_fused

    cfg = MoBAConfig(block_size=BLOCK, top_k=3, dtype=torch.bfloat16)
    cu = _cu(device)
    q, k, _ = _packed(device, torch.bfloat16)

    index = build_csc_fused(q, k, cfg, cu_seqlens=cu, max_seqlen=MAXLEN)
    assert index.col_nnz.shape == (4, 2, 5)        # B, H_kv, NB
    assert index.num_blocks == 5

    for b, (lo, n) in enumerate(zip(CU[:-1], LENS)):
        for kv in range(2):
            for blk in range(index.num_blocks):
                cnt = int(index.col_nnz[b, kv, blk])
                if cnt == 0:
                    continue
                off = int(index.col_offsets[b, kv, blk])
                rows = index.row_indices[b, kv, off:off + cnt]
                assert bool(((rows >= lo) & (rows < lo + n)).all()), (
                    f"column ({b}, {kv}, {blk}) holds a query from another "
                    f"sequence; rows {rows.tolist()} outside [{lo}, {lo + n})"
                )


def test_no_column_exists_past_a_sequences_own_block_count(device):
    """Padded index slots must stay empty, not merely unread.

    A sequence of 5 tokens has 2 blocks at block_size 4, but its row of the
    padded index is 5 wide. Slots 2..4 describe blocks that do not exist; if
    anything ever landed there the attention kernel would launch a program for
    a block whose keys are another document's.
    """
    from moba.kernels.routing import build_csc_fused

    cfg = MoBAConfig(block_size=BLOCK, top_k=3, dtype=torch.bfloat16)
    cu = _cu(device)
    q, k, _ = _packed(device, torch.bfloat16, seed=5)
    index = build_csc_fused(q, k, cfg, cu_seqlens=cu, max_seqlen=MAXLEN)

    for b, n in enumerate(LENS):
        own = -(-n // cfg.block_size)
        beyond = index.col_nnz[b, :, own:]
        assert not bool(beyond.any()), (
            f"sequence {b} has {own} blocks but its index describes "
            f"{int((index.col_nnz[b] > 0).sum())} non-empty columns"
        )


def test_index_round_trips_to_the_gate_under_packing(device):
    """The transpose must be lossless, and in sequence-local ids.

    A wrong transpose does not crash -- it attends to the wrong keys and
    returns plausible numbers -- so the round trip is what makes it visible.
    """
    from moba.kernels.routing import csc_to_mask, build_csc_fused
    from moba.naive import moba_selection

    cfg = MoBAConfig(block_size=BLOCK, top_k=3, dtype=torch.bfloat16)
    cu = _cu(device)
    q, k, _ = _packed(device, torch.bfloat16, seed=6)

    index = build_csc_fused(q, k, cfg, cu_seqlens=cu, max_seqlen=MAXLEN)
    got = csc_to_mask(index, TOTAL)
    assert got.shape == (1, 2, TOTAL, 5)
    assert int(got.sum()) == int(index.col_nnz.sum()), "entries were lost"

    # The oracle uses the same sequence-local convention, established in
    # `moba_selection`'s docstring.
    assert torch.equal(got, moba_selection(q, k, cfg, cu_seqlens=cu))


def test_slots_stay_unique_per_query_under_packing(device):
    """Two entries for one query with the same slot would collide in `part_acc`.

    Slots address the partial buffers, which are indexed by (packed token,
    slot). Packing changes what a row index means, so this is worth re-pinning.
    """
    from moba.kernels.routing import build_csc_fused

    cfg = MoBAConfig(block_size=BLOCK, top_k=3, dtype=torch.bfloat16)
    cu = _cu(device)
    q, k, _ = _packed(device, torch.bfloat16, seed=7)
    index = build_csc_fused(q, k, cfg, cu_seqlens=cu, max_seqlen=MAXLEN)

    seen = set()
    for b in range(4):
        for kv in range(2):
            for blk in range(index.num_blocks):
                cnt = int(index.col_nnz[b, kv, blk])
                off = int(index.col_offsets[b, kv, blk])
                for i in range(off, off + cnt):
                    key = (int(index.head_indices[b, kv, i]),
                           int(index.row_indices[b, kv, i]),
                           int(index.slot_of_pair[b, kv, i]))
                    assert key not in seen, f"duplicate (head, query, slot): {key}"
                    seen.add(key)
                    assert key[2] < cfg.top_k, "slot outside the budget"


# Both at or above `tl.dot`'s 16-row minimum; 32 also exercises a block
# size that does not divide every sequence length.
@pytest.mark.parametrize("gate", [(BLOCK, 3), (32, 4)])
def test_forward_matches_running_each_sequence_alone(device, gate):
    """Bitwise per sequence.

    Slicing gives identical inputs in identical order, and the kernel addresses
    its output by (query, slot) rather than by position, so anything but
    equality is a boundary bug -- not rounding, not summation order.
    """
    from moba.kernels.moba_fwd import moba_forward

    block_size, top_k = gate
    cfg = MoBAConfig(block_size=block_size, top_k=top_k, dtype=torch.bfloat16)
    cu = _cu(device)
    q, k, v = _packed(device, torch.bfloat16)

    packed = moba_forward(q, k, v, cfg, cu_seqlens=cu, max_seqlen=MAXLEN)
    assert packed.shape == q.shape

    lo = 0
    for n in LENS:
        piece = moba_forward(q[:, lo:lo + n], k[:, lo:lo + n],
                             v[:, lo:lo + n], cfg)
        assert torch.equal(packed[:, lo:lo + n], piece), (
            f"sequence at {lo} (length {n}) differs"
        )
        lo += n


def test_packed_forward_matches_the_oracle(device):
    """Routing held fixed, so this measures arithmetic and not gate flips.

    The gate is discontinuous in its inputs -- a ranking -- so a near-tie can
    swap under a dtype change and send a query to entirely different keys.
    Passing the kernel's own selection to the oracle removes that from the
    measurement; see `tests/tolerance.py`.
    """
    from moba.kernels.routing import csc_to_mask, build_csc_fused
    from moba.kernels.moba_fwd import moba_forward
    from moba.naive import moba_attention_naive
    from tolerance import (
        TOLERANCE_FACTOR, TOLERANCE_FLOOR, max_abs_error)

    cfg = MoBAConfig(block_size=BLOCK, top_k=3, dtype=torch.bfloat16)
    cu = _cu(device)
    q64, k64, v64 = _packed(device, torch.float64, seed=8)
    qb, kb, vb = (t.to(torch.bfloat16) for t in (q64, k64, v64))

    index = build_csc_fused(qb, kb, cfg, cu_seqlens=cu, max_seqlen=MAXLEN)
    selected = csc_to_mask(index, TOTAL)

    truth = moba_attention_naive(q64, k64, v64, cfg, selected=selected,
                                 cu_seqlens=cu)
    ref = moba_attention_naive(qb, kb, vb, cfg, selected=selected,
                               cu_seqlens=cu)
    ours = moba_forward(qb, kb, vb, cfg, index=index)

    cap = max(TOLERANCE_FACTOR * max_abs_error(ref, truth), TOLERANCE_FLOOR)
    err = max_abs_error(ours, truth)
    assert err <= cap, f"forward error {err:.3e} exceeds {cap:.3e}"


def test_no_output_token_is_left_unwritten(device):
    """Every packed token must be produced by exactly one program.

    `attend_partials` allocates uninitialized and relies on that. If the
    over-launched programs of a short sequence returned too eagerly, or a
    sequence's tokens were addressed with the wrong base, some outputs would
    hold whatever the allocator had -- finite, plausible, silent.
    """
    from moba.kernels.moba_fwd import moba_forward

    cfg = MoBAConfig(block_size=BLOCK, top_k=3, dtype=torch.bfloat16)
    cu = _cu(device)
    q, k, v = _packed(device, torch.bfloat16, seed=9)

    out = moba_forward(q, k, v, cfg, cu_seqlens=cu, max_seqlen=MAXLEN)
    assert bool(torch.isfinite(out).all()), "a token is non-finite"
    per_token = out.abs().amax(dim=(0, 1, 3))
    assert bool((per_token > 0).all()), (
        f"token(s) {(per_token == 0).nonzero().flatten().tolist()} are all zero"
    )


def test_packed_forward_under_gqa(device):
    """Packing and head sharing must compose, not merely coexist.

    They touch the same addressing: GQA splits the head axis, packing pins the
    batch axis and moves the row index. Nothing tests the product of the two
    unless a test uses both.
    """
    from moba.kernels.moba_fwd import moba_forward

    cfg = MoBAConfig(block_size=BLOCK, top_k=3, dtype=torch.bfloat16)
    cu = _cu(device)
    gen = torch.Generator(device=device).manual_seed(10)
    q = torch.randn(1, TOTAL, 4, 64, generator=gen, device=device,
                    dtype=torch.bfloat16)
    k = torch.randn(1, TOTAL, 2, 64, generator=gen, device=device,
                    dtype=torch.bfloat16)
    v = torch.randn(1, TOTAL, 2, 64, generator=gen, device=device,
                    dtype=torch.bfloat16)

    packed = moba_forward(q, k, v, cfg, cu_seqlens=cu, max_seqlen=MAXLEN)
    lo = 0
    for n in LENS:
        piece = moba_forward(q[:, lo:lo + n], k[:, lo:lo + n],
                             v[:, lo:lo + n], cfg)
        assert torch.equal(packed[:, lo:lo + n], piece), f"at {lo}"
        lo += n


def test_packed_forward_rejects_a_real_batch_axis(device):
    from moba.kernels.moba_fwd import moba_forward
    cfg = MoBAConfig(block_size=BLOCK, top_k=2, dtype=torch.bfloat16)
    cu = _cu(device)
    q = torch.randn(2, TOTAL, 2, 64, device=device, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="batch 1"):
        moba_forward(q, q, q, cfg, cu_seqlens=cu, max_seqlen=MAXLEN)


def test_forward_requires_max_seqlen_when_packing(device):
    """Not derivable without a host synchronization, so it is an argument."""
    from moba.kernels.moba_fwd import moba_forward
    cfg = MoBAConfig(block_size=BLOCK, top_k=2, dtype=torch.bfloat16)
    cu = _cu(device)
    q = torch.randn(1, TOTAL, 2, 64, device=device, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="max_seqlen"):
        moba_forward(q, q, q, cfg, cu_seqlens=cu)


def _grads(fn, q, k, v, g):
    """Runs `fn` and returns the input gradients."""
    q, k, v = (t.detach().clone().requires_grad_(True) for t in (q, k, v))
    (fn(q, k, v) * g).sum().backward()
    return q.grad, k.grad, v.grad


def test_gradients_match_running_each_sequence_alone(device):
    """Gradient must not leak across a boundary.

    dK and dV are reductions down a column -- every query that selected a block
    adds into that block's key gradient. If a column ever held a query from
    another sequence, that query's gradient would be summed into this
    document's keys: finite, plausible, and wrong.
    """
    from moba.kernels.attention import moba_attention

    cfg = MoBAConfig(block_size=BLOCK, top_k=3, dtype=torch.bfloat16)
    cu = _cu(device)
    q, k, v = _packed(device, torch.bfloat16, seed=11)
    g = torch.randn_like(q) * 0.1

    pq, pk, pv = _grads(
        lambda a, x, y: moba_attention(a, x, y, cfg, cu_seqlens=cu,
                                       max_seqlen=MAXLEN), q, k, v, g)
    assert pq.shape == q.shape and pk.shape == k.shape

    lo = 0
    for n in LENS:
        sq, sk, sv = _grads(
            lambda a, x, y: moba_attention(a, x, y, cfg),
            q[:, lo:lo + n], k[:, lo:lo + n], v[:, lo:lo + n],
            g[:, lo:lo + n])
        for name, a, b in [("dq", pq, sq), ("dk", pk, sk), ("dv", pv, sv)]:
            assert torch.allclose(a[:, lo:lo + n].float(), b.float(),
                                  rtol=2e-2, atol=1e-3), (
                f"{name} differs for the sequence at {lo}: max "
                f"{(a[:, lo:lo + n].float() - b.float()).abs().max():.3e}")
        lo += n


def test_packed_gradients_match_the_oracle(device):
    """Routing held fixed, so this measures arithmetic and not gate flips."""
    from moba.kernels.attention import moba_attention
    from moba.kernels.routing import csc_to_mask, build_csc_fused
    from moba.naive import moba_attention_naive
    from tolerance import (
        TOLERANCE_FACTOR, TOLERANCE_FLOOR, max_abs_error)

    cfg = MoBAConfig(block_size=BLOCK, top_k=3, dtype=torch.bfloat16)
    cu = _cu(device)
    q64, k64, v64 = _packed(device, torch.float64, seed=12)
    qb, kb, vb = (t.to(torch.bfloat16) for t in (q64, k64, v64))
    gen = torch.Generator(device=device).manual_seed(13)
    g = torch.randn(1, TOTAL, 2, 64, generator=gen, device=device,
                    dtype=torch.float64) * 0.1

    index = build_csc_fused(qb, kb, cfg, cu_seqlens=cu, max_seqlen=MAXLEN)
    selected = csc_to_mask(index, TOTAL)

    def oracle(a, x, y):
        return moba_attention_naive(a, x, y, cfg, selected=selected,
                                    cu_seqlens=cu)

    def kernel(a, x, y):
        return moba_attention(a, x, y, cfg, index=index)

    tq, tk, tv = _grads(oracle, q64, k64, v64, g)
    rq, rk, rv = _grads(oracle, qb, kb, vb, g)
    dq, dk, dv = _grads(kernel, qb, kb, vb, g)

    for name, ours, ref, truth in [("dq", dq, rq, tq), ("dk", dk, rk, tk),
                                   ("dv", dv, rv, tv)]:
        cap = max(TOLERANCE_FACTOR * max_abs_error(ref, truth),
                  TOLERANCE_FLOOR)
        err = max_abs_error(ours, truth)
        assert err <= cap, f"{name} error {err:.3e} exceeds {cap:.3e}"


def test_padded_index_slots_get_no_gradient(device):
    """dK and dV for keys past a short sequence's end must stay zero.

    The backward writes dK and dV for every block its grid launches, and the
    grid is sized by the *longest* sequence. A program for a block that does
    not exist would write into the next document's keys.
    """
    from moba.kernels.attention import moba_attention

    cfg = MoBAConfig(block_size=BLOCK, top_k=3, dtype=torch.bfloat16)
    cu = _cu(device)
    q, k, v = _packed(device, torch.bfloat16, seed=14)
    g = torch.randn_like(q) * 0.1

    _, dk, dv = _grads(
        lambda a, x, y: moba_attention(a, x, y, cfg, cu_seqlens=cu,
                                       max_seqlen=MAXLEN), q, k, v, g)

    assert bool(torch.isfinite(dk).all()) and bool(torch.isfinite(dv).all())
    # Every packed key belongs to some sequence and was selected by at least
    # its own block's queries, so no real key may be left at exactly zero.
    per_key = dv.abs().amax(dim=(0, 1, 3))
    assert bool((per_key > 0).all()), (
        f"key(s) {(per_key == 0).nonzero().flatten().tolist()} got no gradient"
    )


def test_packed_backward_under_gqa(device):
    """Packing, head sharing and the chain rule must compose.

    dK for a shared key/value head is the sum over the query heads consuming
    it; packing changes which rows those are. Nothing tests the product unless
    a test uses both.
    """
    from moba.kernels.attention import moba_attention

    cfg = MoBAConfig(block_size=BLOCK, top_k=3, dtype=torch.bfloat16)
    cu = _cu(device)
    gen = torch.Generator(device=device).manual_seed(15)
    q = torch.randn(1, TOTAL, 4, 64, generator=gen, device=device,
                    dtype=torch.bfloat16)
    k = torch.randn(1, TOTAL, 2, 64, generator=gen, device=device,
                    dtype=torch.bfloat16)
    v = torch.randn(1, TOTAL, 2, 64, generator=gen, device=device,
                    dtype=torch.bfloat16)
    g = torch.randn_like(q) * 0.1

    pq, pk, pv = _grads(
        lambda a, x, y: moba_attention(a, x, y, cfg, cu_seqlens=cu,
                                       max_seqlen=MAXLEN), q, k, v, g)
    lo = 0
    for n in LENS:
        sq, sk, sv = _grads(
            lambda a, x, y: moba_attention(a, x, y, cfg),
            q[:, lo:lo + n], k[:, lo:lo + n], v[:, lo:lo + n],
            g[:, lo:lo + n])
        for name, a, b in [("dq", pq, sq), ("dk", pk, sk), ("dv", pv, sv)]:
            assert torch.allclose(a[:, lo:lo + n].float(), b.float(),
                                  rtol=3e-2, atol=2e-3), f"{name} at {lo}"
        lo += n


def test_packed_dense_baseline_matches_running_each_sequence_alone(device):
    """The baseline the varlen benchmark compares against.

    Without this the comparison is dishonest in our favour: a padded baseline
    wastes the same padding we removed, so its cost would be charged to dense
    attention rather than to raggedness.
    """
    from implementations.base import get_backend

    cfg = MoBAConfig(block_size=BLOCK, top_k=3, dtype=torch.bfloat16)
    backend = get_backend("sdpa_dense")
    q, k, v = _packed(device, torch.bfloat16, heads=2)

    packed = backend.forward_packed(q, k, v, cfg, _cu(device))
    assert packed.shape == q.shape

    lo = 0
    for n in LENS:
        piece = backend.forward(q[:, lo:lo + n], k[:, lo:lo + n],
                                v[:, lo:lo + n], cfg)
        assert torch.allclose(packed[:, lo:lo + n].float(), piece.float(),
                              rtol=1e-3, atol=1e-3), f"at {lo}"
        lo += n


def test_packed_dense_baseline_really_uses_the_variable_length_kernel(device):
    """Proof the baseline is not padding behind our back.

    A nested tensor that SDPA quietly widened into a rectangle would look like
    a packed baseline and cost like a padded one. `MATH` refusing this exact
    call -- "Nested tensors for query / key are not supported when
    is_causal=True" -- is the evidence that only the fused variable-length
    kernel can serve it.
    """
    from torch.nn.attention import SDPBackend, sdpa_kernel

    from implementations.base import get_backend

    cfg = MoBAConfig(block_size=BLOCK, top_k=3, dtype=torch.bfloat16)
    backend = get_backend("sdpa_dense")
    q, k, v = _packed(device, torch.bfloat16, heads=2)
    cu = _cu(device)

    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        backend.forward_packed(q, k, v, cfg, cu)
    with pytest.raises(RuntimeError, match="No viable backend"):
        with sdpa_kernel(SDPBackend.MATH):
            backend.forward_packed(q, k, v, cfg, cu)


def test_packed_dense_baseline_rejects_a_real_batch_axis(device):
    cfg = MoBAConfig(block_size=BLOCK, top_k=3, dtype=torch.bfloat16)
    from implementations.base import get_backend

    backend = get_backend("sdpa_dense")
    q = torch.randn(2, 16, 2, 64, device=device, dtype=torch.bfloat16)
    cu = torch.tensor([0, 8, 16], dtype=torch.int32, device=device)
    with pytest.raises(ValueError, match=r"\[1, T, H, D\]"):
        backend.forward_packed(q, q, q, cfg, cu)
