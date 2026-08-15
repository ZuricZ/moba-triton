"""Optional backends must degrade gracefully, never explode.

The vendored references have real, expected reasons to be unavailable: an
unbuilt CUDA extension, a missing flash-attn, an un-cloned checkout. The
contract is that every one of those is a reported reason, not a traceback.
"""

import pytest
import torch

from moba.backends import all_backends, available_backends, get_backend
from moba.backends._refs import cu_seqlens_for, from_varlen, to_varlen
from moba.config import MoBAConfig

OPTIONAL = ["moonshot", "triton_moba", "flash_moba"]


# --------------------------------------------------------------------------
# Availability contract
# --------------------------------------------------------------------------


@pytest.mark.parametrize("name", OPTIONAL)
def test_optional_backend_is_registered(name):
    assert name in {b.name for b in all_backends()}


@pytest.mark.parametrize("name", OPTIONAL)
def test_availability_is_reported_without_raising(name):
    cfg = MoBAConfig(block_size=32, top_k=2, dtype=torch.bfloat16)
    ok, reason = get_backend(name).is_available(cfg)
    assert isinstance(ok, bool)
    if not ok:
        assert reason, f"{name} unavailable with no reason given"


@pytest.mark.parametrize("name", OPTIONAL)
def test_unavailable_backend_forward_raises_clearly(name):
    cfg = MoBAConfig(block_size=32, top_k=2, dtype=torch.bfloat16)
    backend = get_backend(name)
    if backend.is_available(cfg)[0]:
        pytest.skip(f"{name} is available; nothing to check here")
    q = k = v = torch.zeros(1, 1, 32, 16, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match="not available"):
        backend.forward(q, k, v, cfg)


def test_flash_moba_reason_names_the_architecture_gap():
    """The most likely reason is specific and worth stating precisely."""
    cfg = MoBAConfig(block_size=32, top_k=2, dtype=torch.bfloat16)
    ok, reason = get_backend("flash_moba").is_available(cfg)
    if ok:
        pytest.skip("flash_moba built; the gap closed")
    assert "sm_" in reason or "not built" in reason


# --------------------------------------------------------------------------
# varlen conversion -- our code, so it must be exact
# --------------------------------------------------------------------------


@pytest.mark.parametrize("batch,heads,seq_len,dim", [(1, 4, 32, 16), (3, 2, 8, 4)])
def test_varlen_roundtrip_is_exact(batch, heads, seq_len, dim):
    x = torch.randn(batch, heads, seq_len, dim, device="cuda", dtype=torch.float64)
    assert torch.equal(from_varlen(to_varlen(x), batch, seq_len), x)


def test_varlen_packing_puts_sequences_contiguously():
    """Sequence b must occupy rows [b*S, (b+1)*S) in the packed layout."""
    batch, heads, seq_len, dim = 3, 2, 4, 5
    x = torch.zeros(batch, heads, seq_len, dim, device="cuda")
    for b in range(batch):
        x[b] = b + 1
    packed = to_varlen(x)
    for b in range(batch):
        assert (packed[b * seq_len : (b + 1) * seq_len] == b + 1).all()


def test_cu_seqlens_matches_flash_attn_convention():
    cu = cu_seqlens_for(3, 4, torch.device("cuda"))
    assert cu.dtype is torch.int32
    assert cu.tolist() == [0, 4, 8, 12]


# --------------------------------------------------------------------------
# Behaviour of whichever references did load
# --------------------------------------------------------------------------


def test_available_reference_reduces_to_dense_causal():
    """Any correct MoBA reduces to dense causal attention when every block is
    selectable. Run in fp64 so a failure means semantics, not precision.

    The tolerance here is loose on purpose: it is a sanity check that the
    varlen adapter maps shapes correctly, not a correctness assertion about
    the reference. Moonshot's naive path lands ~6.7e-7 from dense even in
    fp64, which suggests an internal fp32 step; that is a Task 8 question.
    """
    import torch.nn.functional as F

    b, h, s, d = 1, 2, 256, 64
    g = torch.Generator(device="cuda").manual_seed(0)
    q, k, v = (
        torch.randn(b, h, s, d, generator=g, device="cuda", dtype=torch.float64)
        for _ in range(3)
    )
    dense = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    cfg = MoBAConfig(block_size=32, top_k=s // 32, dtype=torch.float64)

    checked = 0
    for backend in available_backends(cfg):
        if backend.name in ("sdpa_dense", "naive_torch"):
            continue
        out = backend.forward(q, k, v, cfg)
        err = (out - dense).abs().max().item()
        assert err < 1e-4, (
            f"{backend.name} differs from dense causal by {err:.3e} even with "
            f"every block selectable -- the varlen adapter is probably wrong"
        )
        checked += 1

    if checked == 0:
        pytest.skip("no vendored reference is currently available")
