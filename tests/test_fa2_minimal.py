"""Correctness of the minimal FA-2 kernel against torch SDPA.

SDPA is the right oracle here rather than `moba.naive`: this kernel
computes *dense* attention, and SDPA dispatches to a fused Flash implementation
that is independently written. Agreement between our Triton kernel and NVIDIA's
CUDA one is a genuinely independent check.

Tolerance is absolute and generous (2e-2 for fp16). The two implementations
accumulate in a different order, and reassociating thousands of floating-point
additions moves the last bits. This kernel is not part of the routing-sensitive
path, so the strict machinery in `tests/tolerance.py` -- which exists to separate
discrete gate flips from smooth arithmetic error -- does not apply.
"""

import pytest
import torch
import torch.nn.functional as F

from implementations.fa2_minimal import fa2_forward

# (batch, seq_len, heads, head_dim). Includes a sequence length that is not a
# multiple of any tile size, to exercise the masked tail.
SHAPES = [
    (1, 128, 2, 64),
    (1, 512, 4, 64),
    (2, 384, 2, 64),
    (1, 200, 2, 64),
]

# Tile sizes probed in docs/learning/02. 128x128 spills registers and 64x128
# drops to one program per SM; both must still be *correct*.
TILES = [(32, 32), (64, 64), (128, 64), (64, 128), (128, 128)]

ATOL = 2e-2


def _reference(q, k, v, causal):
    """torch SDPA on the same inputs, converted to its head-major convention."""
    out = F.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
        is_causal=causal,
    )
    return out.transpose(1, 2)


@pytest.mark.parametrize("shape", SHAPES, ids=lambda s: "b%ds%dh%dd%d" % s)
@pytest.mark.parametrize("causal", [True, False], ids=["causal", "full"])
def test_matches_sdpa(device, shape, causal):
    """The kernel agrees with torch SDPA across shapes and masking modes."""
    g = torch.Generator(device=device).manual_seed(0)
    q, k, v = (
        torch.randn(shape, generator=g, device=device, dtype=torch.float16)
        for _ in range(3)
    )

    out = fa2_forward(q, k, v, causal=causal)
    ref = _reference(q, k, v, causal)

    err = (out.float() - ref.float()).abs().max().item()
    assert err < ATOL, f"max abs error {err:.3e} exceeds {ATOL:.0e}"


@pytest.mark.parametrize("block_m,block_n", TILES, ids=lambda t: str(t))
def test_tile_sizes_all_correct(device, block_m, block_n):
    """Tile size is a performance knob and must never change the result.

    Register spilling and reduced occupancy are allowed to make a
    configuration slow. They are not allowed to make it wrong.
    """
    shape = (1, 512, 4, 64)
    g = torch.Generator(device=device).manual_seed(0)
    q, k, v = (
        torch.randn(shape, generator=g, device=device, dtype=torch.float16)
        for _ in range(3)
    )

    out = fa2_forward(q, k, v, causal=True, block_m=block_m, block_n=block_n)
    ref = _reference(q, k, v, causal=True)

    err = (out.float() - ref.float()).abs().max().item()
    assert err < ATOL, f"max abs error {err:.3e} exceeds {ATOL:.0e}"


def test_causal_mask_actually_masks(device):
    """A future-only perturbation must not change any earlier output row.

    Comparing against SDPA cannot catch a causal bug that both implementations
    share. This checks the property directly: editing key/value rows at the end
    of the sequence must leave the first half of the output untouched.
    """
    shape = (1, 256, 2, 64)
    g = torch.Generator(device=device).manual_seed(0)
    q, k, v = (
        torch.randn(shape, generator=g, device=device, dtype=torch.float16)
        for _ in range(3)
    )

    before = fa2_forward(q, k, v, causal=True)

    k2, v2 = k.clone(), v.clone()
    k2[:, 128:] += 10.0
    v2[:, 128:] += 10.0
    after = fa2_forward(q, k2, v2, causal=True)

    assert torch.equal(before[:, :128], after[:, :128]), (
        "rows 0-127 changed when only keys/values at positions 128+ were "
        "modified -- causal masking is leaking future information"
    )
    assert not torch.equal(before[:, 128:], after[:, 128:]), (
        "rows 128+ did not change at all -- the perturbation never reached "
        "the kernel, so this test proves nothing"
    )


def test_rejects_mismatched_shapes(device):
    """Shape validation fires before any kernel launch."""
    g = torch.Generator(device=device).manual_seed(0)
    q = torch.randn(1, 128, 2, 64, generator=g, device=device, dtype=torch.float16)
    k = torch.randn(1, 256, 2, 64, generator=g, device=device, dtype=torch.float16)

    with pytest.raises(ValueError, match="shape mismatch"):
        fa2_forward(q, k, k)


# ---------------------------------------------------------------------------
# Backward
# ---------------------------------------------------------------------------


BWD_SHAPES = [
    (1, 256, 2, 64),
    (1, 512, 4, 64),
    (1, 1000, 2, 64),      # not a multiple of any tile: masked tail
    (1, 512, 2, 128),      # steps the backward tile down to fit shared memory
]


@pytest.mark.parametrize("shape", BWD_SHAPES, ids=lambda s: "b%ds%dh%dd%d" % s)
def test_fa2_gradients_match_sdpa(shape, device):
    """Our backward must add no error beyond what bfloat16 itself imposes.

    Measured against the right floor. Comparing a bf16 kernel to a float32
    truth and calling 2e-2 "close" says nothing -- bf16 has eight mantissa
    bits, so 2e-2 may be the format's own error or may be our bug. So torch
    SDPA is run twice: once in float32 for truth, once in bfloat16 for the
    floor. We are allowed `TOLERANCE_FACTOR` times the floor.

    At head_dim 128 dq lands at ~1.8x the floor rather than 1.0x, because the
    backward tile steps down to fit shared memory, so dQ accumulates over twice
    as many atomic additions and the summation order changes. That is
    reassociation, not error.
    """
    from tolerance import TOLERANCE_FACTOR, max_abs_error
    from implementations.fa2_minimal import fa2_attention

    b, s, h, d = shape
    gen = torch.Generator(device=device).manual_seed(0)
    f32 = [torch.randn(b, s, h, d, generator=gen, device=device,
                       dtype=torch.float32, requires_grad=True)
           for _ in range(3)]
    g = torch.randn(b, s, h, d, generator=gen, device=device,
                    dtype=torch.float32)

    def sdpa_grads(tensors, grad):
        qh, kh, vh = (t.transpose(1, 2) for t in tensors)
        out = F.scaled_dot_product_attention(
            qh, kh, vh, is_causal=True).transpose(1, 2)
        out.backward(grad)
        return [t.grad for t in tensors]

    truth = sdpa_grads(f32, g)

    bf16 = [t.detach().to(torch.bfloat16).requires_grad_(True) for t in f32]
    floor = sdpa_grads(bf16, g.to(torch.bfloat16))

    ours = [t.detach().to(torch.bfloat16).requires_grad_(True) for t in f32]
    fa2_attention(*ours, causal=True).backward(g.to(torch.bfloat16))

    for i, name in enumerate(("dq", "dk", "dv")):
        ref_err = max_abs_error(floor[i].float(), truth[i])
        our_err = max_abs_error(ours[i].grad.float(), truth[i])
        budget = max(TOLERANCE_FACTOR * ref_err, 1e-6)
        assert our_err <= budget, (
            f"{name} error {our_err:.3e} exceeds {budget:.3e} "
            f"({TOLERANCE_FACTOR}x the bfloat16 SDPA floor {ref_err:.3e})"
        )


def test_fa2_forward_is_unchanged_by_returning_lse(device):
    """Adding the log-sum-exp output must not perturb the forward.

    `WRITE_LSE` is a constexpr, so the two forms compile to different kernels;
    this asserts the store is genuinely additive rather than something the
    compiler reordered around.
    """
    from implementations.fa2_minimal import fa2_forward as fwd

    gen = torch.Generator(device=device).manual_seed(0)
    q, k, v = (torch.randn(1, 512, 2, 64, generator=gen, device=device,
                           dtype=torch.bfloat16) for _ in range(3))
    plain = fwd(q, k, v, causal=True)
    with_lse, lse = fwd(q, k, v, causal=True, return_lse=True)

    assert torch.equal(plain, with_lse)
    assert lse.shape == (1, 2, 512)
    assert torch.isfinite(lse).all()
