"""Properties the kernel must satisfy regardless of any reference.

Comparing against an oracle only catches bugs the oracle does not share, and
in v1 the oracle shares a great deal: `build_csc` calls `moba_selection`, so
routing is identical to the oracle by construction. A test suite made only of
oracle comparisons would therefore be blind to a whole class of errors -- most
importantly a wrong *transpose*, which makes the kernel attend to the wrong
keys and return entirely plausible numbers.

These four tests need no reference at all. Each asserts a property the
algorithm must have on its own terms:

1. locality of the gather -- unselected keys cannot reach a query's output;
2. causality -- future keys cannot reach a past query's output;
3. dispatch-order independence -- which program runs when is irrelevant;
4. determinism -- the same inputs give the same bits.

All four compare **bitwise** with `torch.equal` rather than within a
tolerance. That is deliberate and is not the fp32-shift situation from
`test_combine_preserves_softmax_shift_invariance`: nothing here perturbs the
arithmetic a query actually performs. The claim under test is that a query's
result is *the same computation*, so any difference at all -- even one ulp --
means a value leaked across a boundary it should not have crossed. A relative
tolerance would let exactly the small leaks these tests exist to find slip
past.
"""

import dataclasses

import pytest
import torch

from conftest import GATE_SWEEP, gate_is_sparse, make_qkv
from moba.config import MoBAConfig
from moba.kernels.moba_fwd import moba_forward
from moba.kernels.routing import build_csc
from moba.ref.naive import moba_selection

# One configuration, drawn from the shared sweep rather than invented here, so
# these tests exercise a gate the correctness sweep also covers. GATE_SWEEP[0]
# is (block_size, top_k) = (32, 4): the smallest block in the sweep, which
# maximizes `num_blocks` and so gives the gather the most opportunity to read
# the wrong column.
BLOCK_SIZE, TOP_K = GATE_SWEEP[0]

# (batch, seq_len, heads, head_dim). Deliberately smaller than SHAPE_SWEEP:
# these are property tests, not accuracy tests, and each one runs the kernel
# two to four times. seq_len 1024 at block_size 32 gives 32 blocks against a
# budget of 4, comfortably clear of the sparsity floor `gate_is_sparse`
# enforces, while `part_acc` costs only 2 MiB of the 4 GiB budget.
SHAPE = (1, 1024, 2, 64)

CFG = MoBAConfig(block_size=BLOCK_SIZE, top_k=TOP_K, dtype=torch.bfloat16)

# Added to keys and values to perturb them. Inputs are standard normal, so
# +10 moves a logit by roughly ten standard deviations: if a perturbed key
# ever reaches a softmax it should not, it does not merely nudge the result,
# it dominates it. A perturbation of the same order as the data could be lost
# in bfloat16's 8-bit mantissa and let a real leak pass unnoticed.
PERTURBATION = 10.0

# Repeats for the determinism check. Three is enough to expose a race that
# resolves the same way most of the time without making the test slow.
DETERMINISM_REPEATS = 3


def _declined_block(mask: torch.Tensor, block_size: int) -> int:
    """Finds a block some later query could have selected but did not.

    Requiring a *later* query is what makes the resulting test meaningful. A
    query before the block is protected by causality as well as by routing, so
    it would pass even if the gather ignored the index entirely. A query after
    the block is protected by routing alone.

    Args:
        mask: Selection mask for one (batch, head), shaped [S, num_blocks].
        block_size: Key positions per block.

    Returns:
        The index of a block at least one later query declined.

    Raises:
        pytest.skip.Exception: If no such block exists, which means the
            configuration is dense and the test would prove nothing.
    """
    seq_len, num_blocks = mask.shape
    for blk in range(num_blocks):
        first_later_query = (blk + 1) * block_size
        if first_later_query >= seq_len:
            break
        if bool((~mask[first_later_query:, blk]).any()):
            return blk
    pytest.skip("no query declines a block it could have selected; gate is dense")


def test_unselected_blocks_do_not_affect_output(device):
    """Catches a gather that reads rows outside the column it was handed.

    The inverted loop order (D6) replaces FlashAttention's contiguous query
    tile with an indirect read through `row_indices`, and an off-by-one in
    `col_offsets`, a stale `lo`, or a mask that lets a dead lane load garbage
    all produce output that is wrong but entirely plausible. Perturbing only
    one block isolates that: every query that did not select it must be
    bit-identical afterwards, and if any moved, the program owning that block
    touched a query outside its column.

    The assertion is a set equality over *every* query, not a spot check on a
    sampled one, and it runs in both directions: the queries whose output
    moves must be exactly the queries that selected the block. Both halves
    earn their place, verified by mutating the index and re-running:

    | mutation on column `blk`     | caught by            |
    |------------------------------|----------------------|
    | `col_offsets` -= 1 (stale lo)| a selected row did not move |
    | `col_nnz` -= 1 (short read)  | a selected row did not move |
    | `col_nnz` += 1 (over-read)   | not here -- see below |

    A single-row spot check caught none of the three, because each disturbs
    exactly one row out of `S`. The over-read is the interesting miss: the
    extra entry belongs to the neighbouring column, so the program does not
    read a foreign *key*, it writes a foreign *slot* that the rightful program
    also writes. That is a race, and it is
    `test_dispatch_order_does_not_change_results` that sees it -- which is
    why both tests exist rather than one.

    Routing is held fixed by passing `index`, because perturbing the keys
    would otherwise move the gate too and the test would be measuring the
    router rather than the attention path.
    """
    q, k, v = make_qkv(SHAPE, dtype=torch.bfloat16, device=device)
    mask = moba_selection(q, k, CFG)
    index = build_csc(q, k, CFG)
    blk = _declined_block(mask[0, 0], CFG.block_size)

    before = moba_forward(q, k, v, CFG, index=index)

    k2, v2 = k.clone(), v.clone()
    lo = blk * CFG.block_size
    # Token-major: [batch, token, head, dim], so the head index comes last.
    k2[0, lo:lo + CFG.block_size, 0] += PERTURBATION
    v2[0, lo:lo + CFG.block_size, 0] += PERTURBATION

    after = moba_forward(q, k2, v2, CFG, index=index)

    # Only (batch 0, head 0) was perturbed, so only its rows may move -- and
    # within it, only the rows that selected `blk`.
    selected = mask[0, 0, :, blk]
    moved = (before[0, :, 0] != after[0, :, 0]).any(dim=-1)

    leaked = (moved & ~selected).nonzero().squeeze(1)
    assert leaked.numel() == 0, (
        f"{leaked.numel()} queries that did not select block {blk} changed "
        f"when it was perturbed, first at row {int(leaked[0])} -- the gather "
        "is reading rows outside its column"
    )

    # The converse is the control, and is exact rather than a mere "something
    # changed": a query that selected the block and did not move means the
    # program skipped a row it owned.
    missing = (selected & ~moved).nonzero().squeeze(1)
    assert missing.numel() == 0, (
        f"{missing.numel()} queries selected block {blk} but did not change "
        f"when it was perturbed, first at row {int(missing[0])} -- the gather "
        "is skipping rows inside its column"
    )
    assert torch.equal(before[0, :, 1], after[0, :, 1]), (
        "an unperturbed head changed -- the kernel is indexing across heads"
    )


def test_causality(device):
    """Catches future keys leaking into earlier rows through the block mask.

    Two mechanisms could leak here and this covers both. The gate could offer
    a query a block that starts after it, and the kernel's intra-block causal
    select (D9) could be dropped or inverted, letting the tail of a query's own
    block into its softmax. The cut is block-aligned so that neither effect is
    confounded with a partially-perturbed block.

    The second assertion is the control. Without it a kernel that ignored `k`
    and `v` entirely -- returning zeros, say -- would pass the first assertion
    perfectly.
    """
    q, k, v = make_qkv(SHAPE, dtype=torch.bfloat16, device=device)
    index = build_csc(q, k, CFG)
    before = moba_forward(q, k, v, CFG, index=index)

    cut = SHAPE[1] // 2   # token-major: seq_len is axis 1
    assert cut % CFG.block_size == 0, "cut must be block-aligned"

    k2, v2 = k.clone(), v.clone()
    k2[:, cut:] += PERTURBATION
    v2[:, cut:] += PERTURBATION
    after = moba_forward(q, k2, v2, CFG, index=index)

    assert torch.equal(before[:, :cut], after[:, :cut]), (
        "future keys leaked into earlier output rows"
    )
    assert not torch.equal(before[:, cut:], after[:, cut:]), (
        "nothing changed at all -- the perturbation never reached the kernel"
    )


def test_dispatch_order_does_not_change_results(device):
    """Catches two programs writing the same partial slot.

    Longest-first dispatch (D8) exists purely to smooth the tail of a wave; it
    is a permutation of program ids and must be arithmetically invisible,
    because every result is written to a *query* address that never depended
    on the program id. Running the identical index twice under two orders
    checks that claim directly.

    It doubles as a race detector, which is the reason it is worth its runtime.
    If `slot_of_pair` ever assigns one (query, slot) pair to two programs, the
    surviving write is decided by which program happens to finish last, and
    that ordering is precisely what changing `order` perturbs. A race like that
    is invisible to the oracle comparison, which would simply see a small
    error and blame precision.
    """
    q, k, v = make_qkv(SHAPE, dtype=torch.bfloat16, device=device)
    index = build_csc(q, k, CFG)
    sorted_out = moba_forward(q, k, v, CFG, index=index)

    identity = (
        torch.arange(index.num_blocks, device=q.device, dtype=torch.int32)
        .expand_as(index.order)
        .contiguous()
    )
    unsorted = dataclasses.replace(index, order=identity)
    unsorted_out = moba_forward(q, k, v, CFG, index=unsorted)

    assert torch.equal(sorted_out, unsorted_out), (
        "dispatch order changed the result -- likely a write race on a "
        "partial slot; check slot_of_pair uniqueness"
    )


def test_determinism(device):
    """Catches non-reproducibility from uninitialized or contended memory.

    Every buffer the kernel writes is allocated fresh per call, and the combine
    pass reads slots no program wrote -- those must carry the `-inf` / 0 / 0
    initialization, not whatever the allocator handed back. If `part_m` were
    ever left uninitialized, or two programs contended for a slot, results
    would differ run to run while every tolerance-based test still passed.
    Bitwise repetition is the only check that sees it.
    """
    q, k, v = make_qkv(SHAPE, dtype=torch.bfloat16, device=device)
    index = build_csc(q, k, CFG)
    first = moba_forward(q, k, v, CFG, index=index)

    for repeat in range(DETERMINISM_REPEATS):
        again = moba_forward(q, k, v, CFG, index=index)
        assert torch.equal(first, again), (
            f"run {repeat + 1} differs from the first on identical inputs"
        )


def test_the_chosen_configuration_is_sparse():
    """Guards the premise the other four tests rest on.

    If `SHAPE` and the gate ever drift into a regime where every query selects
    every block, `test_unselected_blocks_do_not_affect_output` skips and the
    rest degenerate into dense-attention tests that still pass. That failure is
    silent, so it is asserted here rather than left to inspection.
    """
    assert gate_is_sparse(SHAPE, (BLOCK_SIZE, TOP_K)), (
        f"shape {SHAPE} with block_size {BLOCK_SIZE}, top_k {TOP_K} is not "
        "sparse enough for these invariants to mean anything"
    )
