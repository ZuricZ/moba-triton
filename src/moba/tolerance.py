"""Relative tolerance policy for correctness comparisons.

An absolute tolerance is a magic number: it gets tuned until the tests pass
and thereafter means nothing. Worse, it does not survive a change of shape,
dtype, or hardware -- the threshold that was strict at S=128 is vacuous at
S=8192, because more accumulation steps means more rounding.

This module asks a better question. Not "is the error small?" but "is our
error comparable to what ANY correct implementation at this precision would
produce?"

Procedure, following the FlashAttention paper's methodology:

  1. Run the oracle in fp64. Call it `truth`.
  2. Run the SAME oracle in the target dtype. Its deviation from `truth` is
     `ref_error` -- the price of the number format itself, with the algorithm
     held constant.
  3. A backend passes if its own deviation from `truth` is within
     TOLERANCE_FACTOR * ref_error.

Because the algorithm is identical in steps 1 and 2, `ref_error` isolates
precision loss from logic. A backend that reorders accumulation (as every
real kernel does) will differ slightly from the oracle, but a *correct* one
differs by the same order of magnitude the format already imposes.
"""

import torch

from moba.config import MoBAConfig
from moba.ref.naive import moba_attention_naive, moba_selection

# How much worse than a same-dtype reference a backend may be. 2.0 follows
# FlashAttention's convention: it absorbs differences in accumulation order
# while staying far below the gap a genuine logic bug produces.
TOLERANCE_FACTOR = 2.0

# Guards the degenerate case where ref_error is ~0 (tiny or trivial inputs),
# where a pure multiple of it would reject even exact agreement.
TOLERANCE_FLOOR = 1e-6


def max_abs_error(a: torch.Tensor, b: torch.Tensor) -> float:
    """Returns the maximum absolute elementwise difference.

    Both tensors are promoted to float64 before subtracting. Without this,
    comparing a bf16 result against an fp64 truth would round the truth down
    to bf16 first, which would measure zero error for any bf16 value and make
    the whole policy vacuous.

    Max rather than mean: attention outputs are averages over many keys, so a
    single badly-handled position is easily lost in a mean. Mask bugs tend to
    show up in a few rows, not uniformly.

    Args:
        a: First tensor.
        b: Second tensor, same shape as `a`.

    Returns:
        The maximum absolute difference as a Python float.
    """
    return (a.to(torch.float64) - b.to(torch.float64)).abs().max().item()


def reference_error(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cfg: MoBAConfig,
    query_chunk: int | None = None,
) -> tuple[torch.Tensor, float]:
    """Establishes ground truth and the precision cost of `cfg.dtype`.

    Selection is computed ONCE, from the low-precision inputs -- the same
    inputs a real backend will see -- and then held fixed for both the fp64
    and low-precision attention. Without this, `ref_error` would measure the
    gate's discrete routing flips rather than the arithmetic's precision loss,
    and a single flipped query out of 2048 inflates the measured error by two
    orders of magnitude. Selection correctness is checked separately and
    exactly, by `assert_selection_matches`.

    Args:
        q: Queries shaped [B, H, S, D], float64.
        k: Keys shaped [B, H, S, D], float64.
        v: Values shaped [B, H, S, D], float64.
        cfg: Configuration; `cfg.dtype` is the precision being characterized.
        query_chunk: Passed through to the oracle to bound peak memory. Does
            not change results at realistic sizes.

    Returns:
        A tuple of (truth, ref_error) where `truth` is the fp64 oracle output
        under the low-precision selection, and `ref_error` is the max absolute
        deviation of the same oracle evaluated in `cfg.dtype`.
    """
    selected = moba_selection(q.to(cfg.dtype), k.to(cfg.dtype), cfg)

    truth = moba_attention_naive(
        q, k, v, cfg, selected=selected, query_chunk=query_chunk
    )
    low = moba_attention_naive(
        q.to(cfg.dtype),
        k.to(cfg.dtype),
        v.to(cfg.dtype),
        cfg,
        selected=selected,
        query_chunk=query_chunk,
    )
    return truth, max_abs_error(low, truth)


def assert_selection_matches(
    actual: torch.Tensor, expected: torch.Tensor, name: str
) -> None:
    """Asserts two backends route every query to exactly the same blocks.

    This is an exact check by design. Selection is a boolean mask, and the
    gate is discontinuous: a backend that routes one query differently is not
    "slightly less accurate", it computes a different function for that query.
    Tolerance has no meaning here.

    Args:
        actual: Backend's selection mask, shaped [B, H, S, num_blocks].
        expected: Oracle's selection mask, same shape.
        name: Backend name, used in the failure message.

    Raises:
        AssertionError: If any query selects a different set of blocks.
    """
    if actual.shape != expected.shape:
        raise AssertionError(
            f"{name}: selection shape {tuple(actual.shape)} != "
            f"oracle {tuple(expected.shape)}"
        )
    mismatched = actual != expected
    if mismatched.any():
        queries = int(mismatched.any(dim=-1).sum())
        total = int(mismatched.any(dim=-1).numel())
        first = torch.nonzero(mismatched.any(dim=-1))[0].tolist()
        raise AssertionError(
            f"{name}: block selection differs from oracle for {queries}/{total} "
            f"queries; first at [b,h,s]={first}. This is a routing bug, not a "
            f"precision issue -- check tie-breaking and causal candidacy."
        )


def assert_within_tolerance(
    actual: torch.Tensor,
    truth: torch.Tensor,
    ref_error: float,
    name: str,
) -> None:
    """Asserts that `actual` is no less accurate than a same-dtype oracle.

    Args:
        actual: The backend's output.
        truth: The fp64 oracle output.
        ref_error: Deviation of the same-dtype oracle from `truth`.
        name: Backend name, used in the failure message.

    Raises:
        AssertionError: If the deviation exceeds the allowed band. The message
            carries every number needed to judge the failure, so a CI log is
            enough to diagnose it without reproducing locally.
    """
    budget = max(TOLERANCE_FACTOR * ref_error, TOLERANCE_FLOOR)
    err = max_abs_error(actual, truth)
    if err > budget:
        raise AssertionError(
            f"{name}: max abs error {err:.3e} exceeds tolerance {budget:.3e} "
            f"(reference error {ref_error:.3e} x factor {TOLERANCE_FACTOR})"
        )
