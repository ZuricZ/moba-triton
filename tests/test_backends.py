"""Tier 2: every available backend measured against the oracle.

Two independent checks per case, matching the smooth/discrete split the
tolerance policy enforces:

  routing   -- exact. A backend that sends a query to different blocks is
               computing a different function, not a less accurate one.
  output    -- relative tolerance, with routing held fixed so a discrete
               flip cannot contaminate a precision measurement.

Right now the only MoBA backend is the oracle itself, so these largely verify
that the harness machinery works and that the tolerance budget is satisfiable.
Their value arrives with Task 7's vendored backends and with our Triton
kernel, at which point nothing about this file needs to change -- which is the
point of the Backend protocol.
"""

import pytest
import torch

from moba.backends import available_backends, skipped_backends
from moba.config import MoBAConfig
from moba.ref.naive import moba_selection
from tolerance import (
    assert_selection_matches,
    assert_within_tolerance,
    reference_error,
)

from conftest import QUERY_CHUNK, make_qkv, sweep_cases

# sdpa_dense is a roofline baseline, not a MoBA implementation. It attends to
# everything by design, so comparing it against the oracle would fail for a
# reason that is not a defect.
NOT_A_MOBA_BACKEND = {"sdpa_dense"}


@pytest.mark.parametrize("shape,gate", sweep_cases())
def test_backend_matches_oracle(shape, gate, device):
    b, s, h, d = shape
    block_size, top_k = gate
    cfg = MoBAConfig(block_size=block_size, top_k=top_k, dtype=torch.bfloat16)

    if top_k >= cfg.num_blocks(s):
        pytest.skip("degenerates to dense; covered exactly by Tier 1")

    q, k, v = make_qkv(shape, dtype=torch.float64, device=device)
    truth, ref_err = reference_error(q, k, v, cfg, query_chunk=QUERY_CHUNK)

    ql, kl, vl = (t.to(cfg.dtype) for t in (q, k, v))
    expected_routing = moba_selection(ql, kl, cfg)

    checked = 0
    for backend in available_backends(cfg):
        if backend.name in NOT_A_MOBA_BACKEND:
            continue
        out = backend.forward(ql, kl, vl, cfg)

        # Shape, dtype and finiteness are contract violations for anyone.
        assert out.shape == (b, s, h, d), f"{backend.name} returned {out.shape}"
        assert out.dtype is cfg.dtype, f"{backend.name} returned {out.dtype}"
        assert torch.isfinite(out).all(), f"{backend.name} produced non-finite values"

        # Accuracy is asserted only for implementations we own. Per design
        # decision D1 our oracle is the spec, so a vendored reference
        # disagreeing is evidence about *their* semantics, not a defect in
        # ours. Those divergences are quantified in Tier 3 and resolved in
        # docs/semantics-crosscheck.md, not raised as failures here.
        if getattr(backend, "vendored", False):
            continue

        assert_within_tolerance(out, truth, ref_err, name=backend.name)
        checked += 1

    assert checked >= 1, "no first-party MoBA backend was available to check"


@pytest.mark.parametrize("shape,gate", sweep_cases(include_long=False))
def test_backend_routing_matches_oracle(shape, gate, device):
    """Routing is checked separately and exactly.

    Restricted to the fast shapes: routing is a property of the gate, which
    depends on block_size and top_k rather than on sequence length, so long
    shapes add cost without adding coverage.
    """
    _, s, _, _ = shape
    block_size, top_k = gate
    cfg = MoBAConfig(block_size=block_size, top_k=top_k, dtype=torch.bfloat16)

    if top_k >= cfg.num_blocks(s):
        pytest.skip("degenerates to dense; covered exactly by Tier 1")

    q, k, _ = make_qkv(shape, dtype=torch.float64, device=device)
    ql, kl = q.to(cfg.dtype), k.to(cfg.dtype)
    expected = moba_selection(ql, kl, cfg)

    exercised = 0
    for backend in available_backends(cfg):
        if backend.name in NOT_A_MOBA_BACKEND:
            continue
        routing = getattr(backend, "selection", None)
        if routing is None:
            continue  # backend does not expose routing; output check covers it
        assert_selection_matches(routing(ql, kl, cfg), expected, name=backend.name)
        exercised += 1

    # For the oracle this comparison is tautological -- it checks a function
    # against itself. The assertion below is what keeps the test honest: if no
    # backend exposes `selection`, this file has silently become dead
    # scaffolding, and that should fail loudly rather than pass vacuously.
    assert exercised >= 1, "no backend exposed a `selection` method to check"


def test_sparsity_actually_reduces_attended_keys(device):
    """Guards the premise. If selection is not sparse, nothing else matters.

    Also demonstrates the effect that motivates the project: at fixed top_k,
    shrinking block_size sharply reduces the number of keys attended.
    """
    s = 1024
    q, k, _ = make_qkv((1, s, 4, 64), dtype=torch.float64, device=device)

    previous = None
    for block_size in (128, 64, 32, 16):
        cfg = MoBAConfig(block_size=block_size, top_k=2)
        sel = moba_selection(q, k, cfg)
        keys = sel.float().sum(-1).mean().item() * block_size
        assert keys < s, f"blk={block_size} attends {keys:.0f} of {s} keys"
        if previous is not None:
            assert keys < previous, "smaller blocks must attend fewer keys"
        previous = keys


def test_skip_reasons_are_reported(device, capsys):
    """Surfaces why any backend is unavailable. Always passes; run with -s."""
    cfg = MoBAConfig(block_size=32, top_k=2, dtype=torch.bfloat16)
    with capsys.disabled():
        skipped = skipped_backends(cfg)
        if skipped:
            print("\n  unavailable backends:")
            for name, reason in skipped:
                print(f"    {name}: {reason}")
