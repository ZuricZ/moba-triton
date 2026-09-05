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
