"""Tier 3: pairwise comparison across available backends.

This tier reports; it does not assert correctness. A disagreement between two
backends is a finding to resolve in docs/semantics-crosscheck.md, not a test
failure -- per design decision D1, our oracle is the spec and a reference
differing from it is evidence about their semantics.

One thing here *is* asserted: that every divergence currently in the matrix is
one we have already investigated and written down. That turns the document
from a snapshot into a regression check -- if a new gap appears, or a resolved
one reopens, this fails and someone has to explain it.
"""

import itertools

import pytest
import torch

from implementations import available_backends, skipped_backends
from moba.config import MoBAConfig
from tolerance import max_abs_error

from conftest import make_qkv



# Pairs known to agree after the Task 8 cross-check, with the threshold each
# is expected to meet. Moonshot forces fp32 internally (moba_naive.py:84-86),
# so an fp32 comparison is exact while anything finer is bounded by fp32
# epsilon rather than by the input dtype.
EXPECTED_AGREEMENT = {
    ("moonshot", "naive_torch"): 1e-6,
}


def disagreement_matrix(q, k, v, cfg):
    """Computes max absolute difference between every pair of backends.

    Args:
        q: Queries shaped [B, H, S, D].
        k: Keys shaped [B, H, S, D].
        v: Values shaped [B, H, S, D].
        cfg: Configuration.

    Returns:
        A dict mapping (name_a, name_b) sorted alphabetically to their max
        absolute difference.
    """
    outputs = {}
    for backend in available_backends(cfg):
        if not backend.computes_moba:
            continue
        try:
            outputs[backend.name] = backend.forward(q, k, v, cfg)
        except (RuntimeError, NotImplementedError) as exc:
            print(f"    {backend.name}: excluded -- {exc}")

    return {
        (a, b): max_abs_error(outputs[a], outputs[b])
        for a, b in itertools.combinations(sorted(outputs), 2)
    }


def test_report_disagreements(device, capsys):
    """Prints the matrix. Run with -s to see it."""
    cfg = MoBAConfig(block_size=32, top_k=2, dtype=torch.float32)
    q, k, v = make_qkv((1, 512, 4, 64), dtype=cfg.dtype, device=device)

    with capsys.disabled():
        print(
            f"\n  === disagreement matrix "
            f"(dtype={cfg.dtype}, blk={cfg.block_size}, top_k={cfg.top_k}) ==="
        )
        for name, reason in skipped_backends(cfg):
            print(f"    skipped: {name}: {reason}")
        matrix = disagreement_matrix(q, k, v, cfg)
        if not matrix:
            print("    only one MoBA backend available; no pairs to compare")
        for (a, b), err in sorted(matrix.items()):
            print(f"    {a:>14} vs {b:<14} max abs diff {err:.3e}")


@pytest.mark.parametrize("top_k", [1, 2, 4])
def test_known_agreements_still_hold(top_k, device):
    """Regression guard on the Task 8 cross-check.

    Every pair listed in EXPECTED_AGREEMENT was investigated and reconciled.
    If one drifts apart again, the semantics changed and
    docs/semantics-crosscheck.md needs revisiting.
    """
    cfg = MoBAConfig(block_size=32, top_k=top_k, dtype=torch.float32)
    q, k, v = make_qkv((1, 256, 2, 64), dtype=cfg.dtype, device=device)

    matrix = disagreement_matrix(q, k, v, cfg)
    checked = 0
    for pair, threshold in EXPECTED_AGREEMENT.items():
        if pair not in matrix:
            continue
        assert matrix[pair] <= threshold, (
            f"{pair[0]} and {pair[1]} now differ by {matrix[pair]:.3e}, above "
            f"the reconciled threshold {threshold:.3e}. See "
            f"docs/semantics-crosscheck.md -- a semantics rule has drifted."
        )
        checked += 1

    if checked == 0:
        pytest.skip("no reconciled backend pair is currently available")


def test_every_pair_in_the_matrix_is_accounted_for(device):
    """No silent disagreements.

    A pair that neither agrees nor appears in the cross-check document is an
    unexamined divergence, which is exactly the state this project exists to
    avoid.
    """
    cfg = MoBAConfig(block_size=32, top_k=2, dtype=torch.float32)
    q, k, v = make_qkv((1, 256, 2, 64), dtype=cfg.dtype, device=device)

    for pair, err in disagreement_matrix(q, k, v, cfg).items():
        if pair in EXPECTED_AGREEMENT:
            continue
        pytest.fail(
            f"{pair[0]} vs {pair[1]} differ by {err:.3e} and this pair is not "
            f"recorded in EXPECTED_AGREEMENT. Investigate and document it in "
            f"docs/semantics-crosscheck.md."
        )
