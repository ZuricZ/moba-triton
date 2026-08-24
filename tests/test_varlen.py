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
from moba.kernels.gate import block_representatives_triton, gate_select

# Deliberately awkward: a length that is not a block multiple (20 at block 4 is,
# 5 is not), a sequence shorter than one tile, and a longest-first ordering that
# is not the natural one.
LENS = [20, 12, 16, 5]
CU = [0, 20, 32, 48, 53]
MAXLEN = 20
TOTAL = 53


def _packed(device, dtype, heads=2, d=16, seed=0):
    """Three tensors shaped [1, heads, TOTAL, d] -- one packed batch."""
    gen = torch.Generator(device=device).manual_seed(seed)

    def mk():
        return torch.randn(1, heads, TOTAL, d, generator=gen, device=device,
                           dtype=dtype)

    return mk(), mk(), mk()


def _cu(device):
    return torch.tensor(CU, dtype=torch.int32, device=device)


def _kbar(k, cfg, cu):
    return block_representatives_triton(k, cfg, cu_seqlens=cu,
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
    cfg = MoBAConfig(block_size=4, top_k=3, dtype=torch.bfloat16)
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
    cfg = MoBAConfig(block_size=4, top_k=3, dtype=torch.bfloat16)
    cu = _cu(device)
    q, k, _ = _packed(device, torch.bfloat16)

    packed = gate_select(q, _kbar(k, cfg, cu), cfg, cu_seqlens=cu,
                         max_seqlen=MAXLEN)
    lo = 0
    for n in LENS:
        piece = gate_select(
            q[:, :, lo:lo + n],
            block_representatives_triton(k[:, :, lo:lo + n], cfg), cfg)
        assert torch.equal(packed[:, :, lo:lo + n], piece), f"at {lo}"
        lo += n


def test_every_query_selects_its_own_block(device):
    """`always_select_current` must hold per sequence, not per packed buffer.

    Query t of sequence i sits in that sequence's block `(t - bos) // bs`, which
    is not `t // bs`. Getting this wrong is invisible in the first sequence and
    wrong in every other one -- exactly the shape of bug a single-sequence test
    would miss.
    """
    cfg = MoBAConfig(block_size=4, top_k=3, dtype=torch.bfloat16)
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
    cfg = MoBAConfig(block_size=4, top_k=3, dtype=torch.bfloat16)
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
    cfg = MoBAConfig(block_size=4, top_k=2, dtype=torch.bfloat16)
    cu = _cu(device)
    q, k, _ = _packed(device, torch.bfloat16, seed=3)

    idx = gate_select(q, _kbar(k, cfg, cu), cfg, cu_seqlens=cu,
                      max_seqlen=MAXLEN)
    tail = idx[0, :, CU[-2]:CU[-1]]                 # the 5-token sequence
    assert bool(((tail >= 0).sum(-1) > 0).all()), "a query selected nothing"


def test_packed_gate_rejects_a_real_batch_axis(device):
    cfg = MoBAConfig(block_size=4, top_k=2, dtype=torch.bfloat16)
    cu = _cu(device)
    q = torch.randn(2, 2, TOTAL, 16, device=device, dtype=torch.bfloat16)
    kbar = torch.randn(4, 2, 5, 16, device=device, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="batch 1"):
        gate_select(q, kbar, cfg, cu_seqlens=cu, max_seqlen=MAXLEN)


def test_max_seqlen_is_required_when_packing(device):
    """Not derivable without a host synchronization, so it is an argument."""
    cfg = MoBAConfig(block_size=4, top_k=2, dtype=torch.bfloat16)
    cu = _cu(device)
    q = torch.randn(1, 2, TOTAL, 16, device=device, dtype=torch.bfloat16)
    kbar = torch.randn(4, 2, 5, 16, device=device, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="max_seqlen"):
        gate_select(q, kbar, cfg, cu_seqlens=cu)
