"""Gradients against the fp64 oracle, under the same policy as the forward.

`tests/tolerance.py` asks whether our error is comparable to what the number format
already costs, rather than whether it is small in absolute terms. That question
extends to gradients unchanged: run the oracle in fp64 for truth, run the same
oracle in bfloat16 for the price of the format, and require our kernel to stay
within `TOLERANCE_FACTOR` of the latter.

Routing is held fixed throughout by passing the kernel's own selection mask to
the oracle. Without that a near-tie could route the two implementations
differently, and a routing difference is not a numerical difference -- it would
show up as a large gradient error with an entirely wrong cause.
"""

import pytest
import torch

from conftest import GATE_SWEEP, make_qkv
from moba.config import MoBAConfig
from moba.kernels.autograd import moba_attention
from moba.kernels.routing import csc_to_mask, build_csc_fused
from moba.ref.naive import moba_attention_naive
from tolerance import TOLERANCE_FACTOR, TOLERANCE_FLOOR, max_abs_error

SHAPES = [(1, 512, 2, 64), (1, 1024, 2, 128)]


def _oracle_grads(q, k, v, cfg, selected, g):
    """Runs the naive oracle and returns its output and input gradients."""
    q, k, v = (t.detach().clone().requires_grad_(True) for t in (q, k, v))
    out = moba_attention_naive(q, k, v, cfg, selected=selected)
    (out * g.to(out.dtype)).sum().backward()
    return out.detach(), q.grad, k.grad, v.grad


def _kernel_grads(q, k, v, cfg, index, g):
    """Runs the kernel through autograd and returns its input gradients."""
    q, k, v = (t.detach().clone().requires_grad_(True) for t in (q, k, v))
    out = moba_attention(q, k, v, cfg, index=index)
    (out * g.to(out.dtype)).sum().backward()
    return out.detach(), q.grad, k.grad, v.grad


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("gate", GATE_SWEEP)
def test_gradients_are_within_the_precision_budget(device, shape, gate):
    """dq, dk and dv must each be no worse than bfloat16 itself already is."""
    block_size, top_k = gate
    cfg = MoBAConfig(block_size=block_size, top_k=top_k, dtype=torch.bfloat16)
    q, k, v = make_qkv(shape, dtype=torch.float64, device=device)
    qb, kb, vb = (t.to(torch.bfloat16) for t in (q, k, v))
    g = torch.randn(shape, device=device, dtype=torch.float64) * 0.1

    index = build_csc_fused(qb, kb, cfg)
    selected = csc_to_mask(index, shape[1])

    _, tq, tk, tv = _oracle_grads(q, k, v, cfg, selected, g)
    _, rq, rk, rv = _oracle_grads(qb, kb, vb, cfg, selected, g)
    _, dq, dk, dv = _kernel_grads(qb, kb, vb, cfg, index, g)

    for name, ours, ref, truth in [("dq", dq, rq, tq), ("dk", dk, rk, tk),
                                   ("dv", dv, rv, tv)]:
        ref_err = max_abs_error(ref, truth)
        cap = max(TOLERANCE_FACTOR * ref_err, TOLERANCE_FLOOR)
        err = max_abs_error(ours, truth)
        assert err <= cap, (
            f"{name} error {err:.3e} exceeds {cap:.3e} "
            f"({TOLERANCE_FACTOR}x the bfloat16 oracle's {ref_err:.3e})"
        )


@pytest.mark.parametrize("gate", GATE_SWEEP)
def test_every_key_receives_gradient_including_unselected_blocks(device, gate):
    """A block nobody selected must get a zero gradient, not a stale one.

    The backward kernel deliberately does not return early when a block has no
    consumers, because its store is the only thing that writes that block's dK
    and dV. Returning early would leave whatever the allocator handed back --
    plausible-looking numbers from a previous iteration, and silent.

    Runs with `always_select_current=False`, which is the only way to reach the
    case at all: under the default every block is selected by the queries
    inside it, so `col_nnz` is never zero and there is nothing to check. With
    the own block optional, the last block goes unselected -- it is causally
    visible only to its own queries, and they prefer earlier ones.
    """
    block_size, top_k = gate
    cfg = MoBAConfig(block_size=block_size, top_k=top_k, dtype=torch.bfloat16,
                     always_select_current=False)
    shape = (1, 1024, 2, 64)
    q, k, v = make_qkv(shape, dtype=torch.bfloat16, device=device)
    g = torch.randn(shape, device=device, dtype=torch.bfloat16) * 0.1

    index = build_csc_fused(q, k, cfg)
    empty = (index.col_nnz == 0)
    # Asserted, not skipped. This premise held silently for months while the
    # body below was unreachable; a skip here would hide that again.
    assert bool(empty.any()), "no unselected block, so nothing is being checked"

    _, _, dk, dv = _kernel_grads(q, k, v, cfg, index, g)

    for b, h, blk in empty.nonzero().tolist():
        lo = blk * cfg.block_size
        hi = min(lo + cfg.block_size, shape[1])
        assert not bool(dk[b, lo:hi, h].any()), (
            f"unselected block {blk} has a nonzero dk"
        )
        assert not bool(dv[b, lo:hi, h].any()), (
            f"unselected block {blk} has a nonzero dv"
        )


def test_rebuilding_the_index_changes_gradients_only_by_summation_order(
    device
):
    """Routing is stable across index builds; dk and dv shift in the last bits.

    Worth pinning down because it is easy to expect bitwise equality here and
    be alarmed when it is absent. `build_csc_fused` lets queries claim their
    position within a column with an atomic cursor, so a column's order varies
    run to run -- documented in `moba.kernels.routing`.

    The forward is untouched by that: `part_acc` is addressed by (query, slot),
    so permuting a column moves no value. `dk` and `dv` are sums *down* the
    query axis, and floating-point addition is not associative, so their order
    shows. `dq` is per-query and stays exact.

    The guarantee this kernel offers is therefore determinism given an index,
    not across index builds; `test_backward_is_deterministic` covers the
    former.
    """
    cfg = MoBAConfig(block_size=64, top_k=4, dtype=torch.bfloat16)
    shape = (1, 512, 2, 64)
    q, k, v = make_qkv(shape, dtype=torch.bfloat16, device=device)
    g = torch.randn(shape, device=device, dtype=torch.bfloat16) * 0.1

    first, second = build_csc_fused(q, k, cfg), build_csc_fused(q, k, cfg)
    assert torch.equal(csc_to_mask(first, shape[1]),
                       csc_to_mask(second, shape[1])), "routing is unstable"

    _, aq, ak, av = _kernel_grads(q, k, v, cfg, first, g)
    _, bq, bk, bv = _kernel_grads(q, k, v, cfg, second, g)

    assert torch.equal(aq, bq), (
        "dq is addressed per (query, slot) and must not depend on column order"
    )
    for name, a, b in [("dk", ak, bk), ("dv", av, bv)]:
        assert torch.allclose(a, b, rtol=2e-2, atol=1e-3), (
            f"{name} differs by more than summation order explains: "
            f"max {max_abs_error(a, b):.3e}"
        )


def test_backward_is_deterministic(device):
    """Repeated backward passes must give bitwise identical gradients.

    This is why `dQ` is merged through partials rather than `atomic_add`. An
    atomic accumulation would be correct in expectation and different every
    run, which makes any later numerical investigation much harder and is
    exactly the property the gate's tie-break rule exists to protect.
    """
    cfg = MoBAConfig(block_size=128, top_k=8, dtype=torch.bfloat16)
    shape = (1, 1024, 2, 128)
    q, k, v = make_qkv(shape, dtype=torch.bfloat16, device=device)
    g = torch.randn(shape, device=device, dtype=torch.bfloat16) * 0.1
    index = build_csc_fused(q, k, cfg)

    _, aq, ak, av = _kernel_grads(q, k, v, cfg, index, g)
    _, bq, bk, bv = _kernel_grads(q, k, v, cfg, index, g)

    assert torch.equal(aq, bq), "dq is not reproducible"
    assert torch.equal(ak, bk), "dk is not reproducible"
    assert torch.equal(av, bv), "dv is not reproducible"
