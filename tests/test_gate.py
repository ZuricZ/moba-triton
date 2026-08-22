"""The fused gate against the PyTorch gate it replaces.

The PyTorch gate is the oracle here, and it is a strong one: the routing rule
is a ranking, so agreement is checked for *exact equality*, never within a
tolerance. A gate that selects a different block is not slightly wrong, it is
attending somewhere else.

The score comparison is the part that makes exactness achievable. `tl.dot`
accumulates in float32 while the PyTorch path rounds its output back to
bfloat16, so the kernel rounds too. Without that, near-ties would resolve
differently and these tests would fail intermittently on some shapes and pass
on others -- the worst possible signal.
"""

import pytest
import torch

from conftest import GATE_SWEEP, SHAPE_SWEEP, make_qkv
from moba.config import MoBAConfig
from moba.kernels.gate import check_gate_fits, gate_select
from moba.ref.naive import block_representatives, select_blocks


def indices_to_mask(idx: torch.Tensor, num_blocks: int) -> torch.Tensor:
    """Expands selected block ids back to a dense boolean mask.

    Padding entries are routed to a scratch column that is then dropped. The
    obvious alternative -- clamping -1 to 0 -- silently corrupts the result,
    because `scatter_` lets the last write to a duplicated index win, so a
    padding slot would overwrite a genuine selection of block 0.

    Args:
        idx: Selected block ids shaped [B, H, S, top_k], -1 padded.
        num_blocks: Total blocks.

    Returns:
        Boolean mask shaped [B, H, S, num_blocks].
    """
    b, h, s, _ = idx.shape
    mask = torch.zeros(b, h, s, num_blocks + 1, dtype=torch.bool,
                       device=idx.device)
    target = torch.where(idx >= 0, idx,
                         torch.full_like(idx, num_blocks)).long()
    mask.scatter_(-1, target, True)
    return mask[..., :num_blocks]


@pytest.mark.parametrize("shape", SHAPE_SWEEP)
@pytest.mark.parametrize("gate", GATE_SWEEP)
def test_gate_selects_exactly_what_the_oracle_selects(device, shape, gate):
    """Routing is a ranking, so it is compared exactly."""
    block_size, top_k = gate
    cfg = MoBAConfig(block_size=block_size, top_k=top_k, dtype=torch.bfloat16)
    q, k, _ = make_qkv(shape, dtype=torch.bfloat16, device=device)
    kbar = block_representatives(k, cfg)

    expected = select_blocks(q, kbar, cfg)
    actual = indices_to_mask(gate_select(q, kbar, cfg), kbar.shape[2])

    disagreements = int((expected ^ actual).sum())
    assert disagreements == 0, (
        f"{disagreements} block selections differ from the oracle at "
        f"block_size {block_size}, top_k {top_k}"
    )


@pytest.mark.parametrize("gate", GATE_SWEEP)
def test_gate_emits_ascending_ids_with_padding_last(device, gate):
    """Slot order is the combine pass's addressing, so it is part of the API.

    A query's slot `j` must be its `j`-th selection *by ascending block index*,
    because that is the rule the CSC index and the fixed-slot partial buffers
    were built on. Emitting by descending score instead would still select the
    right blocks and would still pass the test above, while silently permuting
    every query's partial slots.
    """
    block_size, top_k = gate
    cfg = MoBAConfig(block_size=block_size, top_k=top_k, dtype=torch.bfloat16)
    q, k, _ = make_qkv((1, 2, 1024, 128), dtype=torch.bfloat16, device=device)

    idx = gate_select(q, block_representatives(k, cfg), cfg)
    valid = idx >= 0
    both_valid = valid[..., 1:] & valid[..., :-1]

    assert bool(((idx[..., 1:] > idx[..., :-1]) | ~both_valid).all()), (
        "selected block ids are not strictly ascending within a query"
    )
    assert bool((valid[..., 1:] <= valid[..., :-1]).all()), (
        "padding is not a suffix; a -1 appears before a real selection"
    )


def test_gate_breaks_ties_toward_the_lower_block_index(device):
    """Equal scores must resolve the same way on every run.

    This is a determinism requirement, not an accuracy one. An unstable
    tie-break makes the kernel attend to different blocks on different runs,
    which looks exactly like a race condition and would send a future debugging
    session in entirely the wrong direction.

    Identical representatives force exact ties, which random data essentially
    never produces -- so without constructing them the tie-break is untested.
    """
    cfg = MoBAConfig(block_size=64, top_k=2, dtype=torch.bfloat16)
    q, k, _ = make_qkv((1, 1, 512, 64), dtype=torch.bfloat16, device=device)
    kbar = block_representatives(k, cfg)

    # Every block gets the same representative, so all scores tie exactly and
    # only the tie-break rule decides. With top_k 2 and the own block free,
    # exactly one past block is selected: it must be block 0 every time.
    kbar[:] = kbar[:, :, :1, :]

    idx = gate_select(q, kbar, cfg)
    late = idx[0, 0, 400]                     # a query with many past blocks
    assert int(late[0]) == 0, (
        f"tie resolved to block {int(late[0])}, expected the lowest index"
    )
    assert torch.equal(idx, gate_select(q, kbar, cfg)), "gate is not repeatable"


def test_gate_matches_the_oracle_when_the_budget_exceeds_the_blocks(device):
    """top_k is a ceiling, not a quota.

    An early query has fewer past blocks than the budget allows, and a short
    sequence can have fewer blocks than top_k outright. Both cases must select
    what exists rather than padding with a repeated or bogus id.
    """
    cfg = MoBAConfig(block_size=128, top_k=12, dtype=torch.bfloat16)
    q, k, _ = make_qkv((1, 2, 512, 64), dtype=torch.bfloat16, device=device)
    kbar = block_representatives(k, cfg)          # only 4 blocks exist

    idx = gate_select(q, kbar, cfg)
    expected = select_blocks(q, kbar, cfg)

    assert torch.equal(indices_to_mask(idx, kbar.shape[2]), expected)
    assert int(idx.max()) < kbar.shape[2], "emitted a block id out of range"


def test_gate_refuses_a_representative_tile_that_will_not_fit(device):
    """The ceiling is reported, not silently degraded.

    This kernel holds every block representative in SRAM at once, which is what
    lets it rank a whole score row in registers. Past that point the
    representatives must be streamed with a per-chunk merge -- a different
    kernel, not a tuning change -- so the limit raises rather than falling back.
    """
    ok, reason = check_gate_fits(num_blocks=8192, head_dim=128, itemsize=2)
    assert not ok
    assert "streaming top-k" in reason

    ok, _ = check_gate_fits(num_blocks=64, head_dim=128, itemsize=2)
    assert ok
