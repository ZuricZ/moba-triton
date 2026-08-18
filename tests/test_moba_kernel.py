"""The KV-block-major kernel against the oracle.

The single-slot test comes first and matters most during bring-up: with
`top_k=1` every query selects exactly its own block, so there is one partial
slot and no merging happens. That isolates the attention kernel from the
combine pass, which is the difference between "the gather is wrong" and "the
merge is wrong" when a number comes out bad.

Ground truth follows the D5 policy: inputs are built in fp64, `reference_error`
returns the fp64 oracle output plus the precision cost of running the same
oracle in the low dtype, and the kernel must be no less accurate than that.
Routing is checked separately and exactly.
"""

import pytest
import torch

from conftest import QUERY_CHUNK, make_qkv, sweep_cases
from moba.config import MoBAConfig
from moba.kernels.index import build_csc, csc_to_mask
from moba.kernels.moba_fwd import moba_forward
from moba.ref.naive import moba_selection
from moba.tolerance import (
    assert_selection_matches,
    assert_within_tolerance,
    reference_error,
)


def test_single_slot_matches_the_oracle(device):
    """top_k=1 means one block per query, so no cross-slot merging occurs."""
    cfg = MoBAConfig(block_size=32, top_k=1, dtype=torch.bfloat16)
    q, k, v = make_qkv((1, 4, 512, 64), dtype=torch.float64, device=device)
    truth, ref_err = reference_error(q, k, v, cfg, query_chunk=QUERY_CHUNK)

    ql, kl, vl = (t.to(cfg.dtype) for t in (q, k, v))
    actual = moba_forward(ql, kl, vl, cfg)

    assert_within_tolerance(actual, truth, ref_err, name="single-slot kernel")


@pytest.mark.parametrize("shape,gate", sweep_cases(include_long=False))
def test_kernel_matches_the_oracle(device, shape, gate):
    """Full shape x gate sweep, with merging active."""
    block_size, top_k = gate
    cfg = MoBAConfig(block_size=block_size, top_k=top_k, dtype=torch.bfloat16)
    q, k, v = make_qkv(shape, dtype=torch.float64, device=device)
    truth, ref_err = reference_error(q, k, v, cfg, query_chunk=QUERY_CHUNK)

    ql, kl, vl = (t.to(cfg.dtype) for t in (q, k, v))
    actual = moba_forward(ql, kl, vl, cfg)

    assert_within_tolerance(
        actual, truth, ref_err, name=f"moba_triton-blk{block_size}-top{top_k}"
    )


@pytest.mark.parametrize("shape,gate", sweep_cases(include_long=False))
def test_kernel_routing_matches_the_oracle_exactly(device, shape, gate):
    """The gate is a ranking, so its output is checked exactly, never by norm."""
    block_size, top_k = gate
    cfg = MoBAConfig(block_size=block_size, top_k=top_k, dtype=torch.bfloat16)
    q, k, _ = make_qkv(shape, dtype=torch.float64, device=device)
    ql, kl = q.to(cfg.dtype), k.to(cfg.dtype)

    expected = moba_selection(ql, kl, cfg)
    actual = csc_to_mask(build_csc(ql, kl, cfg), shape[2])

    assert_selection_matches(actual, expected, name="moba_triton")


def test_rejects_mismatched_shapes(device):
    """Validation fires before any launch."""
    cfg = MoBAConfig(block_size=32, top_k=2, dtype=torch.bfloat16)
    q, _, _ = make_qkv((1, 2, 128, 64), dtype=torch.bfloat16, device=device)
    k, _, _ = make_qkv((1, 2, 256, 64), dtype=torch.bfloat16, device=device)

    with pytest.raises(ValueError, match="shape mismatch"):
        moba_forward(q, k, k, cfg)


def test_rejects_block_size_below_tensor_core_minimum(device):
    """block_size < 16 cannot fill a tl.dot operand and is refused, not padded."""
    cfg = MoBAConfig(block_size=8, top_k=2, dtype=torch.bfloat16)
    q, k, v = make_qkv((1, 2, 128, 64), dtype=torch.bfloat16, device=device)

    with pytest.raises(ValueError, match="below the tl.dot minimum"):
        moba_forward(q, k, v, cfg)
