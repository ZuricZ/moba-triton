"""Tests for the backend protocol and registry.

The registry's contract is narrow but load-bearing: every implementation is
reachable uniformly, and anything that cannot run says why. Most of these
tests guard the *reporting* half, because a harness that silently omits a
backend produces a results table that looks complete and is not.
"""

import pytest
import torch

from moba.backends import (
    Backend,
    all_backends,
    available_backends,
    get_backend,
    skipped_backends,
)
from moba.config import MoBAConfig


def test_baseline_backends_are_registered():
    names = {b.name for b in all_backends()}
    assert {"naive_torch", "sdpa_dense"} <= names


def test_registered_backends_satisfy_the_protocol():
    for backend in all_backends():
        assert isinstance(backend, Backend), f"{backend.name} is not a Backend"


def test_naive_torch_is_always_available():
    cfg = MoBAConfig(block_size=32, top_k=2)
    ok, reason = get_backend("naive_torch").is_available(cfg)
    assert ok, reason


def test_available_and_skipped_partition_the_registry():
    cfg = MoBAConfig(block_size=32, top_k=2)
    assert len(available_backends(cfg)) + len(skipped_backends(cfg)) == len(
        all_backends()
    )


def test_skipped_backends_always_carry_a_reason():
    cfg = MoBAConfig(block_size=32, top_k=2)
    for name, reason in skipped_backends(cfg):
        assert reason, f"{name} was skipped with an empty reason"


def test_is_available_never_raises_on_absurd_config():
    """is_available is a predicate for skipping work, not a validator.

    Callers rely on it to decide whether to run something; if it could throw,
    every call site would need a try/except, and what gets written in practice
    is `except: pass` -- the silent omission this design exists to prevent.
    """
    cfg = MoBAConfig(block_size=1, top_k=999999, dtype=torch.float64)
    for backend in all_backends():
        ok, reason = backend.is_available(cfg)
        assert isinstance(ok, bool)
        assert isinstance(reason, str)


def test_unknown_backend_raises_with_available_names():
    with pytest.raises(KeyError) as exc:
        get_backend("does_not_exist")
    assert "naive_torch" in str(exc.value), "error should list what IS available"


def test_naive_and_sdpa_agree_when_top_k_covers_all_blocks():
    """Cross-checks the two always-available backends against each other.

    With every block selectable the gate is a no-op, so the MoBA backend and
    the dense baseline must agree exactly. This is the keystone identity
    again, routed through the registry rather than called directly -- it
    verifies the adapters do not mangle shapes or arguments.
    """
    b, h, s, d = 1, 2, 64, 32
    g = torch.Generator(device="cuda").manual_seed(0)
    q, k, v = (
        torch.randn(b, h, s, d, generator=g, device="cuda", dtype=torch.float64)
        for _ in range(3)
    )
    cfg = MoBAConfig(block_size=16, top_k=s // 16, dtype=torch.float64)

    got = get_backend("naive_torch").forward(q, k, v, cfg)
    want = get_backend("sdpa_dense").forward(q, k, v, cfg)

    torch.testing.assert_close(got, want, rtol=1e-10, atol=1e-10)


def test_backends_preserve_shape_and_dtype():
    b, h, s, d = 1, 2, 64, 32
    cfg = MoBAConfig(block_size=16, top_k=2, dtype=torch.bfloat16)
    q, k, v = (
        torch.randn(b, h, s, d, device="cuda", dtype=torch.bfloat16) for _ in range(3)
    )
    for backend in available_backends(cfg):
        out = backend.forward(q, k, v, cfg)
        assert out.shape == (b, h, s, d), backend.name
        assert out.dtype is torch.bfloat16, backend.name


def test_duplicate_registration_is_rejected():
    """Silent overwrite would make results depend on import order."""
    from moba.backends.base import register
    from moba.backends.naive_torch import NaiveTorchBackend

    with pytest.raises(ValueError, match="already registered"):
        register(NaiveTorchBackend())
