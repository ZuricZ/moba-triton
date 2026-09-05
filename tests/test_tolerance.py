"""Tests for the relative tolerance policy.

A tolerance policy makes two claims: it accepts correct low-precision
implementations, and it rejects subtly wrong ones. Both halves are tested
here. The second half matters more -- a harness that has never been shown to
fail on a wrong answer is not evidence of correctness, it is decoration.

Note the split this module enforces. MoBA has a smooth part (attention
arithmetic) and a discrete part (top_k routing). Only the smooth part gets a
tolerance; routing is checked exactly.
"""

import pytest
import torch

from moba.config import MoBAConfig
from moba.ref.naive import moba_attention_naive, moba_selection
from tolerance import (
    TOLERANCE_FACTOR,
    TOLERANCE_FLOOR,
    assert_selection_matches,
    assert_within_tolerance,
    max_abs_error,
    reference_error,
)

DEVICE = "cuda"


def make_qkv(b, s, h, d, dtype=torch.float64, seed=0):
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    shape = (b, s, h, d)
    return tuple(
        torch.randn(shape, generator=g, device=DEVICE, dtype=dtype) for _ in range(3)
    )


def run_in(dtype, q, k, v, cfg, selected=None):
    """Runs the oracle at a given precision, optionally with fixed routing."""
    return moba_attention_naive(
        q.to(dtype), k.to(dtype), v.to(dtype), cfg, selected=selected
    )


# --------------------------------------------------------------------------
# Measurement primitives
# --------------------------------------------------------------------------


def test_max_abs_error_promotes_before_comparing():
    """Comparing bf16 against fp64 must not round the fp64 side down."""
    a = torch.zeros(4, device=DEVICE, dtype=torch.bfloat16)
    b = torch.full((4,), 0.5, device=DEVICE, dtype=torch.float64)
    assert max_abs_error(a, b) == pytest.approx(0.5)


def test_reference_error_is_positive_and_sane_for_bf16():
    q, k, v = make_qkv(1, 128, 2, 32)
    cfg = MoBAConfig(block_size=32, top_k=2, dtype=torch.bfloat16)
    truth, ref_err = reference_error(q, k, v, cfg)

    assert truth.dtype is torch.float64
    assert ref_err > 0, "bf16 must lose precision relative to fp64"


def test_reference_error_measures_arithmetic_not_routing_flips():
    """The bug this module was rewritten to fix.

    With routing held fixed, ref_error must reflect only accumulation and
    rounding, so it stays within the order of magnitude bf16's ~8 mantissa
    bits imply. Before the fix a single flipped query out of 2048 pushed this
    to ~0.9, which made the budget both unstable and uselessly loose.
    """
    for seq_len in (128, 512, 1024):
        q, k, v = make_qkv(1, seq_len, 2, 64)
        cfg = MoBAConfig(block_size=32, top_k=2, dtype=torch.bfloat16)
        _, ref_err = reference_error(q, k, v, cfg)
        assert ref_err < 0.05, (
            f"seq_len={seq_len}: ref_error {ref_err:.3e} is too large to be "
            f"precision loss; routing is probably leaking into the measurement"
        )


def test_bf16_is_less_precise_than_fp16():
    """Sanity-checks the premise: error tracks mantissa width.

    bf16 has 7 explicit mantissa bits, fp16 has 10. bf16 trades precision for
    fp32's exponent range. If this ordering inverted, the methodology would be
    measuring something other than format precision.
    """
    q, k, v = make_qkv(1, 128, 2, 32)
    _, err_bf16 = reference_error(
        q, k, v, MoBAConfig(block_size=32, top_k=2, dtype=torch.bfloat16)
    )
    _, err_fp16 = reference_error(
        q, k, v, MoBAConfig(block_size=32, top_k=2, dtype=torch.float16)
    )
    assert err_bf16 > err_fp16


# --------------------------------------------------------------------------
# The policy accepts correct implementations
# --------------------------------------------------------------------------


def test_correct_bf16_implementation_passes():
    q, k, v = make_qkv(1, 128, 2, 32)
    cfg = MoBAConfig(block_size=32, top_k=2, dtype=torch.bfloat16)
    truth, ref_err = reference_error(q, k, v, cfg)

    actual = run_in(torch.bfloat16, q, k, v, cfg)
    assert_within_tolerance(actual, truth, ref_err, name="bf16_oracle")


def test_a_more_precise_implementation_passes():
    """An fp32 implementation, routed identically, is well inside a bf16 budget."""
    q, k, v = make_qkv(1, 128, 2, 32)
    cfg = MoBAConfig(block_size=32, top_k=2, dtype=torch.bfloat16)
    truth, ref_err = reference_error(q, k, v, cfg)

    selected = moba_selection(q.to(cfg.dtype), k.to(cfg.dtype), cfg)
    actual = run_in(torch.float32, q, k, v, cfg, selected=selected)
    assert_within_tolerance(actual, truth, ref_err, name="fp32_oracle")


def test_floor_saves_the_degenerate_case():
    """When ref_error is ~0, a pure multiple of it would reject everything."""
    truth = torch.zeros(16, device=DEVICE, dtype=torch.float64)
    actual = torch.full((16,), TOLERANCE_FLOOR / 2, device=DEVICE)
    assert_within_tolerance(actual, truth, ref_error=0.0, name="degenerate")


# --------------------------------------------------------------------------
# The policy rejects wrong implementations
# --------------------------------------------------------------------------


def test_perturbed_implementation_is_rejected():
    q, k, v = make_qkv(1, 128, 2, 32)
    cfg = MoBAConfig(block_size=32, top_k=2, dtype=torch.bfloat16)
    truth, ref_err = reference_error(q, k, v, cfg)

    poisoned = run_in(torch.bfloat16, q, k, v, cfg) + (
        TOLERANCE_FACTOR * ref_err * 100 + 1.0
    )
    with pytest.raises(AssertionError, match="exceeds tolerance"):
        assert_within_tolerance(poisoned, truth, ref_err, name="poisoned")


def test_wrong_top_k_is_rejected():
    """A semantic bug, not a numeric one, must also be caught."""
    q, k, v = make_qkv(1, 256, 2, 32)
    cfg = MoBAConfig(block_size=32, top_k=2, dtype=torch.bfloat16)
    truth, ref_err = reference_error(q, k, v, cfg)

    wrong = MoBAConfig(block_size=32, top_k=4, dtype=torch.bfloat16)
    actual = run_in(torch.bfloat16, q, k, v, wrong)

    with pytest.raises(AssertionError, match="exceeds tolerance"):
        assert_within_tolerance(actual, truth, ref_err, name="wrong_top_k")


def test_failure_message_reports_the_numbers():
    """A failure must be diagnosable from a log, without a debugger."""
    truth = torch.zeros(8, device=DEVICE, dtype=torch.float64)
    actual = torch.ones(8, device=DEVICE, dtype=torch.float64)
    with pytest.raises(AssertionError) as exc:
        assert_within_tolerance(actual, truth, ref_error=1e-4, name="mybackend")
    msg = str(exc.value)
    assert "mybackend" in msg
    assert "1.000e+00" in msg
    assert str(TOLERANCE_FACTOR) in msg


# --------------------------------------------------------------------------
# Routing is checked exactly, never with a tolerance
# --------------------------------------------------------------------------


def test_identical_routing_passes():
    q, k, _ = make_qkv(1, 256, 2, 32)
    cfg = MoBAConfig(block_size=32, top_k=2, dtype=torch.bfloat16)
    sel = moba_selection(q.to(cfg.dtype), k.to(cfg.dtype), cfg)
    assert_selection_matches(sel, sel.clone(), name="self")


def test_different_tie_break_is_caught_as_a_routing_bug():
    """The subtlest semantics flag surfaces as a routing failure, not a
    tolerance failure -- which is the diagnosis a reader actually needs."""
    b, s, h, d = 1, 256, 1, 32
    g = torch.Generator(device=DEVICE).manual_seed(0)
    q = torch.randn(b, s, h, d, generator=g, device=DEVICE, dtype=torch.float64)
    k = torch.ones(b, s, h, d, device=DEVICE, dtype=torch.float64)

    low = MoBAConfig(block_size=32, top_k=2, tie_break_low_index=True)
    high = MoBAConfig(block_size=32, top_k=2, tie_break_low_index=False)

    with pytest.raises(AssertionError, match="routing bug"):
        assert_selection_matches(
            moba_selection(q, k, high), moba_selection(q, k, low), name="flipped"
        )


def test_routing_failure_message_locates_the_first_bad_query():
    cfg = MoBAConfig(block_size=32, top_k=2)
    a = torch.zeros(1, 1, 4, 3, dtype=torch.bool, device=DEVICE)
    b = a.clone()
    b[0, 0, 2, 1] = True

    with pytest.raises(AssertionError) as exc:
        assert_selection_matches(b, a, name="bk")
    msg = str(exc.value)
    assert "1/4" in msg
    assert "[0, 0, 2]" in msg


def test_routing_shape_mismatch_is_reported():
    a = torch.zeros(1, 1, 4, 3, dtype=torch.bool, device=DEVICE)
    b = torch.zeros(1, 1, 4, 5, dtype=torch.bool, device=DEVICE)
    with pytest.raises(AssertionError, match="selection shape"):
        assert_selection_matches(b, a, name="bk")
