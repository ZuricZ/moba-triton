"""Grouped-query attention end to end, forward and backward.

Every test here is anchored the same way: the GQA kernel must agree with the
*same code path* run on expanded key/value heads. That comparison isolates the
head-sharing logic from everything else -- arithmetic, routing and tuning are
identical on both sides, so a disagreement can only be an indexing bug.

The failure mode this is defending against does not crash. A wrong head mapping
produces a well-formed index of the right shape holding plausible numbers, and
the kernel attends to another head's keys. Shape assertions cannot see it.

See `docs/superpowers/plans/2026-08-23-gqa-support.md` and
`docs/learning/03-gqa-and-the-column-merge.md`.
"""

import pytest
import torch

from moba.config import MoBAConfig
from moba.gqa import expand_kv
from moba.kernels.moba_fwd import moba_forward
from moba.kernels.routing import csc_to_mask, build_csc_fused
from moba.ref.naive import moba_attention_naive
from tolerance import TOLERANCE_FACTOR, TOLERANCE_FLOOR, max_abs_error

# (batch, query heads, kv heads, seq_len, head_dim). Small on purpose: 4 GiB of
# VRAM, and grouped-query attention lengthens every column, so a shape that fit
# under multi-head attention may not fit here.
GQA_SHAPES = [
    (1, 4, 1, 512, 64),     # group 4 -- the multi-query extreme
    (1, 8, 2, 512, 64),     # group 4
    (1, 4, 4, 512, 64),     # group 1 -- must reproduce plain MHA exactly
    (1, 4, 2, 1024, 128),   # group 2, head_dim 128
]
GQA_GATES = [(64, 4), (128, 8)]


def _shapes(shape, dtype, device, seed=0):
    """Returns (q, k, v) with the query and key/value head counts of `shape`."""
    b, h_q, h_kv, s, d = shape
    gen = torch.Generator(device=device).manual_seed(seed)
    q = torch.randn(b, s, h_q, d, generator=gen, device=device, dtype=dtype)
    k = torch.randn(b, s, h_kv, d, generator=gen, device=device, dtype=dtype)
    v = torch.randn(b, s, h_kv, d, generator=gen, device=device, dtype=dtype)
    return q, k, v


@pytest.mark.parametrize("shape", GQA_SHAPES)
@pytest.mark.parametrize("gate", GQA_GATES)
def test_forward_matches_the_oracle_under_gqa(device, shape, gate):
    """Routing held fixed, so this measures arithmetic and not gate flips.

    The gate is discontinuous in its inputs -- a ranking, so a near-tie can
    swap under a dtype change and send a query to entirely different keys.
    Passing the kernel's own selection to the oracle removes that from the
    measurement; see `tests/tolerance.py`.
    """
    block_size, top_k = gate
    cfg = MoBAConfig(block_size=block_size, top_k=top_k, dtype=torch.bfloat16)
    b, h_q, h_kv, s, d = shape

    q64, k64, v64 = _shapes(shape, torch.float64, device)
    qb, kb, vb = (t.to(torch.bfloat16) for t in (q64, k64, v64))

    index = build_csc_fused(qb, kb, cfg)
    selected = csc_to_mask(index, s)
    assert selected.shape == (b, h_q, s, cfg.num_blocks(s))

    truth = moba_attention_naive(q64, k64, v64, cfg, selected=selected)
    ref = moba_attention_naive(qb, kb, vb, cfg, selected=selected)
    ours = moba_forward(qb, kb, vb, cfg, index=index)

    assert ours.shape == (b, s, h_q, d)
    cap = max(TOLERANCE_FACTOR * max_abs_error(ref, truth), TOLERANCE_FLOOR)
    err = max_abs_error(ours, truth)
    assert err <= cap, (
        f"forward error {err:.3e} exceeds {cap:.3e} "
        f"({TOLERANCE_FACTOR}x the bfloat16 oracle's)"
    )


@pytest.mark.parametrize("gate", GQA_GATES)
def test_gqa_forward_equals_the_kernel_on_expanded_heads(device, gate):
    """The sharing must be free of arithmetic consequences.

    Expanding K/V and running the ordinary multi-head path poses the identical
    mathematical problem. Results differ only by summation order within a
    column, whose entries an atomic cursor permutes -- hence allclose rather
    than bitwise. `test_index_fused` pins the routing itself exactly.
    """
    block_size, top_k = gate
    cfg = MoBAConfig(block_size=block_size, top_k=top_k, dtype=torch.bfloat16)
    q, k, v = _shapes((1, 8, 2, 512, 64), torch.bfloat16, device)

    narrow = moba_forward(q, k, v, cfg)
    wide = moba_forward(q, expand_kv(k, 8), expand_kv(v, 8), cfg)
    assert torch.allclose(narrow, wide, rtol=2e-2, atol=1e-3), (
        f"max {max_abs_error(narrow, wide):.3e}"
    )


def test_every_query_head_receives_output_under_gqa(device):
    """No query head may be left unwritten.

    `attend_partials` allocates uninitialized and relies on every (query, slot)
    pair being written by exactly one program. If the head axis were dropped
    anywhere in the chain, some heads would return whatever the allocator held
    -- which is finite, plausible, and silent.
    """
    cfg = MoBAConfig(block_size=64, top_k=4, dtype=torch.bfloat16)
    q, k, v = _shapes((1, 8, 2, 512, 64), torch.bfloat16, device, seed=3)

    out = moba_forward(q, k, v, cfg)
    assert bool(torch.isfinite(out).all()), "a head returned non-finite output"
    per_head = out.abs().amax(dim=(0, 2, 3))
    assert bool((per_head > 0).all()), (
        f"a query head produced an all-zero output: {per_head.tolist()}"
    )


def test_indivisible_head_counts_are_rejected_at_the_boundary(device):
    cfg = MoBAConfig(block_size=64, top_k=4, dtype=torch.bfloat16)
    q = torch.randn(1, 256, 8, 64, dtype=torch.bfloat16, device=device)
    k = torch.randn(1, 256, 3, 64, dtype=torch.bfloat16, device=device)
    with pytest.raises(ValueError, match="divisible"):
        moba_forward(q, k, k, cfg)


def test_mismatched_key_and_value_shapes_are_rejected(device):
    """K and V must still agree with each other, GQA or not."""
    cfg = MoBAConfig(block_size=64, top_k=4, dtype=torch.bfloat16)
    q = torch.randn(1, 256, 8, 64, dtype=torch.bfloat16, device=device)
    k = torch.randn(1, 256, 2, 64, dtype=torch.bfloat16, device=device)
    v = torch.randn(1, 256, 4, 64, dtype=torch.bfloat16, device=device)
    with pytest.raises(ValueError, match="key/value"):
        moba_forward(q, k, v, cfg)


def _grads(fn, q, k, v, g):
    """Runs `fn` and returns the input gradients."""
    q, k, v = (t.detach().clone().requires_grad_(True) for t in (q, k, v))
    out = fn(q, k, v)
    (out * g.to(out.dtype)).sum().backward()
    return q.grad, k.grad, v.grad


@pytest.mark.parametrize("gate", GQA_GATES)
def test_gradients_under_gqa_are_within_the_precision_budget(device, gate):
    """dk and dv must accumulate every sharing head's contribution.

    The chain rule says a shared key/value head's gradient is the *sum* over
    the query heads consuming it. The kernel gets that for free: the group's
    query rows are already in the same column, and summing down a column is
    what the dK/dV accumulation was already doing. This test is what makes
    "for free" a claim rather than a hope.

    The failure to fear is an off-by-`group`: a dk that is right in shape and
    plausible in magnitude but sums only one head's contribution.
    """
    from moba.kernels.autograd import moba_attention

    block_size, top_k = gate
    cfg = MoBAConfig(block_size=block_size, top_k=top_k, dtype=torch.bfloat16)
    shape = (1, 8, 2, 512, 64)
    b, h_q, h_kv, s, d = shape

    q64, k64, v64 = _shapes(shape, torch.float64, device)
    qb, kb, vb = (t.to(torch.bfloat16) for t in (q64, k64, v64))
    gen = torch.Generator(device=device).manual_seed(7)
    g = torch.randn(b, s, h_q, d, generator=gen, device=device,
                    dtype=torch.float64) * 0.1

    index = build_csc_fused(qb, kb, cfg)
    selected = csc_to_mask(index, s)

    def oracle(a, x, y):
        return moba_attention_naive(a, x, y, cfg, selected=selected)

    def kernel(a, x, y):
        return moba_attention(a, x, y, cfg, index=index)

    tq, tk, tv = _grads(oracle, q64, k64, v64, g)
    rq, rk, rv = _grads(oracle, qb, kb, vb, g)
    dq, dk, dv = _grads(kernel, qb, kb, vb, g)

    # Token-major. `selected` above is the head-major mask, which did
    # not move -- it is ours, not part of the API surface.
    assert dq.shape == (b, s, h_q, d)
    assert dk.shape == (b, s, h_kv, d)
    assert dv.shape == (b, s, h_kv, d)

    for name, ours, ref, truth in [("dq", dq, rq, tq), ("dk", dk, rk, tk),
                                   ("dv", dv, rv, tv)]:
        cap = max(TOLERANCE_FACTOR * max_abs_error(ref, truth),
                  TOLERANCE_FLOOR)
        err = max_abs_error(ours, truth)
        assert err <= cap, (
            f"{name} error {err:.3e} exceeds {cap:.3e} "
            f"({TOLERANCE_FACTOR}x the bfloat16 oracle's {max_abs_error(ref, truth):.3e})"
        )


def test_gqa_backward_equals_the_kernel_on_expanded_heads(device):
    """Sharing must be free of arithmetic consequences in the backward too.

    The expanded run produces one dk per *query* head; summing each group
    recovers what the shared run computes in one pass. That sum is the
    comparison, and it is the chain rule written out.
    """
    from moba.kernels.autograd import moba_attention

    cfg = MoBAConfig(block_size=64, top_k=4, dtype=torch.bfloat16)
    b, h_q, h_kv, s, d = 1, 8, 2, 512, 64
    q, k, v = _shapes((b, h_q, h_kv, s, d), torch.bfloat16, device, seed=5)
    gen = torch.Generator(device=device).manual_seed(9)
    g = torch.randn(b, s, h_q, d, generator=gen, device=device,
                    dtype=torch.bfloat16) * 0.1
    group = h_q // h_kv

    nq, nk, nv = _grads(lambda a, x, y: moba_attention(a, x, y, cfg), q, k, v, g)
    wq, wk, wv = _grads(lambda a, x, y: moba_attention(a, x, y, cfg),
                        q, expand_kv(k, h_q), expand_kv(v, h_q), g)

    # Token-major: the head axis is 2, and `expand_kv` interleaves there, so
    # query head g belongs to kv head g // group and the group is contiguous.
    folded_k = wk.view(b, s, h_kv, group, d).sum(dim=3)
    folded_v = wv.view(b, s, h_kv, group, d).sum(dim=3)

    for name, a, bb in [("dq", nq, wq), ("dk", nk, folded_k),
                        ("dv", nv, folded_v)]:
        assert torch.allclose(a.float(), bb.float(), rtol=3e-2, atol=2e-3), (
            f"{name} differs by more than summation order explains: "
            f"max {max_abs_error(a, bb):.3e}"
        )


def test_unselected_blocks_still_get_a_zero_gradient_under_gqa(device):
    """A merged column that nobody filled must still write zeros.

    The backward deliberately does not return early on an empty column, because
    its store is the only writer of that block's dK and dV; returning would
    leave whatever the allocator handed back. Grouped-query attention makes the
    case rarer -- every head in the group must decline the block -- so it needs
    reaching on purpose rather than hoping a sweep wanders into it.

    `always_select_current=False` is what makes it reachable at all. With the
    default, a block is always selected by the queries inside it, so `col_nnz`
    is at least `block_size` for every column and an empty one cannot exist.
    Measured: zero empty columns across six GQA configurations with the flag
    on, one or two with it off.
    """
    from moba.kernels.autograd import moba_attention

    cfg = MoBAConfig(block_size=64, top_k=2, dtype=torch.bfloat16,
                     always_select_current=False)
    b, h_q, h_kv, s, d = 1, 4, 1, 1024, 64
    q, k, v = _shapes((b, h_q, h_kv, s, d), torch.bfloat16, device, seed=11)
    gen = torch.Generator(device=device).manual_seed(13)
    g = torch.randn(b, s, h_q, d, generator=gen, device=device,
                    dtype=torch.bfloat16) * 0.1

    index = build_csc_fused(q, k, cfg)
    empty = (index.col_nnz == 0)
    if not bool(empty.any()):
        pytest.skip("no unselected block at this configuration")

    _, dk, dv = _grads(
        lambda a, x, y: moba_attention(a, x, y, cfg, index=index), q, k, v, g)

    for bi, kv, blk in empty.nonzero().tolist():
        lo = blk * cfg.block_size
        hi = min(lo + cfg.block_size, s)
        assert not bool(dk[bi, kv, lo:hi].any()), f"block {blk} has nonzero dk"
        assert not bool(dv[bi, kv, lo:hi].any()), f"block {blk} has nonzero dv"
