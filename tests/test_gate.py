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
from moba.heads import expand_kv
from moba.kernels.gate import (
    GATE_BLOCK_N,
    pool_blocks,
    gate_select,
)
from moba.ref.naive import pool_blocks_naive, gate_select_naive


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
    kbar = pool_blocks_naive(k, cfg)

    expected = gate_select_naive(q, kbar, cfg)
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
    q, k, _ = make_qkv((1, 1024, 2, 128), dtype=torch.bfloat16, device=device)

    idx = gate_select(q, pool_blocks_naive(k, cfg), cfg)
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
    q, k, _ = make_qkv((1, 512, 1, 64), dtype=torch.bfloat16, device=device)
    kbar = pool_blocks_naive(k, cfg)

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
    q, k, _ = make_qkv((1, 512, 2, 64), dtype=torch.bfloat16, device=device)
    kbar = pool_blocks_naive(k, cfg)          # only 4 blocks exist

    idx = gate_select(q, kbar, cfg)
    expected = gate_select_naive(q, kbar, cfg)

    assert torch.equal(indices_to_mask(idx, kbar.shape[2]), expected)
    assert int(idx.max()) < kbar.shape[2], "emitted a block id out of range"


def test_gate_handles_more_blocks_than_a_single_tile_could_hold(device):
    """The streaming form has no ceiling on num_blocks, so this must just work.

    The earlier one-shot gate scored every representative in one tile, and was
    guarded by a check that rejected 512 blocks as needing 128 KiB of shared
    memory. The guard was wrong twice over: the compiler reported 20 KiB, and
    what actually degraded was occupancy, not capacity. This configuration --
    512 blocks, four times what the guard permitted -- is here so that a
    reintroduced ceiling fails loudly instead of at some untested length.

    It earns its keep for a second reason. Exact bf16 score ties get common
    at this many blocks, and ties are where the tie-break rule is actually
    exercised; this configuration caught an eviction-order bug that every
    shape at seq_len 2048 or below missed.
    """
    cfg = MoBAConfig(block_size=32, top_k=8, dtype=torch.bfloat16)
    q, k, _ = make_qkv((1, 16384, 1, 128), dtype=torch.bfloat16, device=device)
    kbar = pool_blocks_naive(k, cfg)
    assert kbar.shape[2] == 512, "expected 512 blocks for this configuration"

    expected = gate_select_naive(q, kbar, cfg)
    actual = indices_to_mask(gate_select(q, kbar, cfg), kbar.shape[2])

    assert torch.equal(expected, actual), (
        f"{int((expected ^ actual).sum())} routing disagreements at 512 blocks"
    )


def test_gate_respects_the_budget_when_capacity_is_not_a_power_of_two(device):
    """A running list with surplus slots must not use them as extra capacity.

    `K_POW2` is rounded up for `tl.arange`, so top_k 12 gives 16 slots holding
    12 entries. If the surplus is left looking empty the merge fills it, and
    the query ends up attending to more blocks than its budget allows.

    Two conditions are needed to see it, and no other test here has both: a
    top_k whose capacity is not a power of two, and num_blocks above
    `GATE_BLOCK_N` so more than one chunk is merged. Only a backend test at
    seq_len 8192 caught it originally.
    """
    cfg = MoBAConfig(block_size=128, top_k=12, dtype=torch.bfloat16)
    q, k, _ = make_qkv((1, 8192, 1, 128), dtype=torch.bfloat16, device=device)
    kbar = pool_blocks_naive(k, cfg)
    assert kbar.shape[2] > GATE_BLOCK_N, "needs more than one streamed chunk"

    idx = gate_select(q, kbar, cfg)
    selected = (idx >= 0).sum(-1)

    assert int(selected.max()) <= cfg.top_k, (
        f"a query selected {int(selected.max())} blocks, over the budget of "
        f"{cfg.top_k}"
    )
    assert torch.equal(gate_select_naive(q, kbar, cfg),
                       indices_to_mask(idx, kbar.shape[2]))



def test_gate_scores_each_query_head_against_its_shared_representatives(device):
    """A query head must rank against its own KV head's blocks, and no other.

    The failure this guards against is silent and plausible: with the wrong
    divisor the gate ranks head 5 against head 5's representatives instead of
    head 1's, which produces a perfectly well-formed index that attends to the
    wrong keys. Comparing against the expanded oracle catches it; a shape check
    cannot.

    Note the grid this implies. The gate runs per *query* head, because each
    query head ranks independently -- same representatives, different `q`, so
    different winners. Only the thing being ranked is shared. See
    `docs/learning/03-gqa-and-the-column-merge.md`.
    """
    cfg = MoBAConfig(block_size=64, top_k=4, dtype=torch.bfloat16)
    b, h_q, h_kv, s, d = 1, 8, 2, 512, 64
    gen = torch.Generator(device=device).manual_seed(0)
    q = torch.randn(b, s, h_q, d, generator=gen, device=device,
                    dtype=torch.bfloat16)
    k = torch.randn(b, s, h_kv, d, generator=gen, device=device,
                    dtype=torch.bfloat16)

    kbar = pool_blocks(k, cfg)
    assert kbar.shape == (b, h_kv, cfg.num_blocks(s), d)

    narrow = gate_select(q, kbar, cfg)
    wide = gate_select(q, expand_kv(kbar, h_q, dim=1), cfg)
    assert narrow.shape == (b, h_q, s, cfg.top_k)
    assert torch.equal(narrow, wide)


def test_gate_agrees_with_the_pytorch_oracle_under_gqa(device):
    """Exact equality against the PyTorch gate, as everywhere else in this file."""
    cfg = MoBAConfig(block_size=64, top_k=4, dtype=torch.bfloat16)
    b, h_q, h_kv, s, d = 1, 8, 2, 512, 64
    gen = torch.Generator(device=device).manual_seed(1)
    q = torch.randn(b, s, h_q, d, generator=gen, device=device,
                    dtype=torch.bfloat16)
    k = torch.randn(b, s, h_kv, d, generator=gen, device=device,
                    dtype=torch.bfloat16)

    kbar = pool_blocks_naive(k, cfg)
    idx = gate_select(q, kbar, cfg)
    assert torch.equal(gate_select_naive(q, kbar, cfg),
                       indices_to_mask(idx, kbar.shape[2]))


def test_every_query_head_is_gated_under_gqa(device):
    """No query head may be left unwritten.

    Sizing the gate's grid by kv heads instead of query heads would run the
    gate for 2 heads of 8 and leave the other 6 holding whatever the allocator
    returned -- uninitialized memory that transposes into a well-formed index
    and attends to blocks nobody chose. It would not crash, so it is checked.
    """
    cfg = MoBAConfig(block_size=64, top_k=4, dtype=torch.bfloat16)
    b, h_q, h_kv, s, d = 1, 8, 2, 256, 64
    gen = torch.Generator(device=device).manual_seed(2)
    q = torch.randn(b, s, h_q, d, generator=gen, device=device,
                    dtype=torch.bfloat16)
    k = torch.randn(b, s, h_kv, d, generator=gen, device=device,
                    dtype=torch.bfloat16)

    idx = gate_select(q, pool_blocks(k, cfg), cfg)
    n = cfg.num_blocks(s)

    # Every query selects its own block at minimum, so every row has an entry
    # in range. An ungated head would fail this on uninitialized memory.
    assert bool(((idx >= -1) & (idx < n)).all()), "an id is out of range"
    assert bool(((idx >= 0).sum(-1) > 0).all()), "a query selected nothing"


def test_gate_rejects_indivisible_head_counts(device):
    cfg = MoBAConfig(block_size=64, top_k=4, dtype=torch.bfloat16)
    q = torch.randn(1, 256, 8, 64, dtype=torch.bfloat16, device=device)
    kbar = torch.randn(1, 3, 4, 64, dtype=torch.bfloat16, device=device)
    with pytest.raises(ValueError, match="divisible"):
        gate_select(q, kbar, cfg)


def test_pool_blocks_naive_are_pooled_per_sequence(device):
    """A block averages its own sequence's keys and stops at the boundary.

    The bug this catches is a block straddling two sequences: its mean would mix
    documents, every downstream stage would accept it, and the numbers stay
    finite. Compared against per-sequence pooling of the same keys, bitwise --
    the inputs are identical, so rounding cannot explain a difference.
    """
    cfg = MoBAConfig(block_size=4, top_k=2, dtype=torch.bfloat16)
    cu = torch.tensor([0, 20, 32, 48], dtype=torch.int32, device=device)
    gen = torch.Generator(device=device).manual_seed(0)
    k = torch.randn(1, 48, 2, 16, generator=gen, device=device,
                    dtype=torch.bfloat16)

    got = pool_blocks(k, cfg, cu_seqlens=cu, max_seqlen=20)
    assert got.shape == (3, 2, 5, 16)      # B=3, H_kv=2, NB=cdiv(20,4)=5

    lo = 0
    for i, n in enumerate([20, 12, 16]):
        want = pool_blocks(k[:, lo:lo + n], cfg)
        nb = want.shape[2]
        assert torch.equal(got[i:i + 1, :, :nb], want), f"sequence {i}"
        lo += n


def test_a_short_final_block_divides_by_its_real_occupancy(device):
    """Every sequence now has a tail, not just the batch.

    A sequence of 6 with block_size 4 ends in a 2-key block. Dividing that by 4
    would shrink its representative toward zero and make it almost never
    selected -- the bug the padded path already guards against, now arising once
    per sequence rather than once per batch.
    """
    cfg = MoBAConfig(block_size=4, top_k=2, dtype=torch.float32)
    cu = torch.tensor([0, 6, 10], dtype=torch.int32, device=device)
    k = torch.ones(1, 10, 1, 8, device=device, dtype=torch.float32)
    k[0, 4, 0] = 3.0
    k[0, 5, 0] = 5.0            # sequence 0's tail block holds 3 and 5

    got = pool_blocks(k, cfg, cu_seqlens=cu, max_seqlen=6)
    assert got.shape == (2, 1, 2, 8)
    assert torch.allclose(got[0, 0, 1], torch.full((8,), 4.0, device=device)), (
        f"tail block averaged to {got[0, 0, 1, 0].item()}, expected 4.0"
    )


def test_packed_representatives_reject_a_real_batch_axis(device):
    cfg = MoBAConfig(block_size=4, top_k=2, dtype=torch.bfloat16)
    cu = torch.tensor([0, 8, 16], dtype=torch.int32, device=device)
    k = torch.randn(2, 16, 2, 16, device=device, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="batch 1"):
        pool_blocks(k, cfg, cu_seqlens=cu, max_seqlen=8)
