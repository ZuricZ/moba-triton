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
import triton

from conftest import QUERY_CHUNK, make_qkv, sweep_cases
from moba.config import MoBAConfig
from moba.kernels.index import build_csc, csc_to_mask
from moba.kernels.moba_fwd import (
    _moba_combine,
    choose_combine_tile,
    combine_partials,
    moba_forward,
)
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


# --- The combine pass -----------------------------------------------------
#
# These compare the Triton merge against the obvious PyTorch one on synthetic
# partials rather than on kernel output. That is deliberate: a bug in Task 2's
# gather would otherwise corrupt the inputs to this test and make a correct
# merge look wrong. Feeding it random partials keeps the two failures separable.


def _reference_combine(part_m, part_l, part_acc):
    """The textbook merge, in float64, used as ground truth.

    Args:
        part_m: Per-slot maxima, [B, H, S, K].
        part_l: Per-slot denominators, [B, H, S, K].
        part_acc: Per-slot un-normalized outputs, [B, H, S, K, D].

    Returns:
        Merged output shaped [B, H, S, D], float64.
    """
    m, l, acc = (t.double() for t in (part_m, part_l, part_acc))
    w = torch.exp(m - m.max(dim=-1, keepdim=True).values)
    return (acc * w.unsqueeze(-1)).sum(-2) / (l * w).sum(-1).unsqueeze(-1)


def _random_partials(device, b=1, h=2, s=256, k=4, d=64, seed=0):
    """Builds plausible partial-softmax buffers with one empty slot.

    The spread on `part_m` is wide on purpose. Partials whose maxima differ by
    only a little would let an unrescaled sum pass; a spread of several units
    makes `exp(m_i - m_g)` range over orders of magnitude, so omitting it is
    unmissable.

    Args:
        device: CUDA device.
        b, h, s, k, d: Batch, heads, sequence, slots, head dim.
        seed: RNG seed.

    Returns:
        A tuple of (part_m, part_l, part_acc), all float32.
    """
    g = torch.Generator(device=device).manual_seed(seed)
    part_m = torch.randn(b, h, s, k, generator=g, device=device) * 5
    part_l = torch.rand(b, h, s, k, generator=g, device=device) + 0.1
    part_acc = torch.randn(b, h, s, k, d, generator=g, device=device)

    # A slot no program wrote keeps the initialization from `moba_forward`:
    # m = -inf, l = 0, acc = 0. It must contribute exactly nothing. Skipped at
    # k=1, where blanking the only slot would leave a query with no partials
    # at all -- a state `build_csc` cannot produce, since every query selects
    # at least its own block.
    if k > 1:
        part_m[..., -1] = float("-inf")
        part_l[..., -1] = 0.0
        part_acc[..., -1] = 0.0
    return part_m, part_l, part_acc


def test_combine_matches_the_reference_merge(device):
    """The Triton merge must agree with the textbook one."""
    part_m, part_l, part_acc = _random_partials(device)

    actual = combine_partials(part_m, part_l, part_acc, torch.float32)
    expected = _reference_combine(part_m, part_l, part_acc)

    err = (actual.double() - expected).abs().max().item()
    assert err < 1e-5, f"combine mismatch {err:.3e}"


@pytest.mark.parametrize("slots", [1, 2, 3, 5, 7, 12])
def test_combine_handles_slot_counts_that_are_not_powers_of_two(device, slots):
    """`max_slots` is whatever the index produced, so 3, 5 and 12 must work.

    This is why the kernel loops with `tl.static_range` over a constexpr
    instead of loading a `[BLOCK_M, K_SLOTS]` tile: `tl.arange` requires a
    power-of-two length and would reject most real slot counts.
    """
    part_m, part_l, part_acc = _random_partials(device, k=slots, seed=slots)

    actual = combine_partials(part_m, part_l, part_acc, torch.float32)
    expected = _reference_combine(part_m, part_l, part_acc)

    err = (actual.double() - expected).abs().max().item()
    assert err < 1e-5, f"combine mismatch at {slots} slots: {err:.3e}"


def test_combine_is_dominated_by_the_largest_maximum(device):
    """A slot far above the rest must swamp them -- the rescale, isolated.

    Drop the `exp(m_i - m_g)` weights and every slot contributes equally, so
    the answer moves toward the mean of the accumulators. This is the 800x
    failure mode from learning note 02, which produces plausible output and no
    crash, so it needs a test that fails loudly rather than merely imprecisely.
    """
    part_m, part_l, part_acc = _random_partials(device, k=4, seed=7)
    part_m[..., 0] = 100.0          # dominant
    part_m[..., 1:-1] = 0.0         # negligible: exp(-100) underflows to zero

    actual = combine_partials(part_m, part_l, part_acc, torch.float32)
    expected = part_acc[..., 0, :] / part_l[..., 0].unsqueeze(-1)

    err = (actual - expected).abs().max().item()
    assert err < 1e-5, f"dominant slot not isolated: {err:.3e}"


def test_combine_preserves_softmax_shift_invariance(device):
    """Adding a constant to every maximum must not move the output.

    Softmax is invariant to a shift in its logits, and that invariance is
    exactly what the per-slot maxima encode. If the merge leaked an absolute
    max anywhere -- an `exp(m_i)` rather than an `exp(m_i - m_g)` -- this
    diverges or overflows.

    Tolerance is relative, and looser than the other combine tests, because the
    shift itself costs precision: numbers near 50 are spaced about 4e-6 apart
    in fp32, so storing `m + 50` rounds `m`, and `exp` carries that error
    through. Measured ~1e-5 relative, which is the predicted size. The bug this
    guards against -- exponentiating an absolute maximum -- overflows to inf
    rather than drifting, so a relative bound still catches it.
    """
    part_m, part_l, part_acc = _random_partials(device, seed=3)
    shifted = part_m + 50.0         # -inf stays -inf, so empty slots are safe

    base = combine_partials(part_m, part_l, part_acc, torch.float32)
    moved = combine_partials(shifted, part_l, part_acc, torch.float32)

    assert torch.isfinite(moved).all(), "shifted merge produced inf or NaN"
    rel = ((base - moved).abs().max() / base.abs().max()).item()
    assert rel < 1e-4, f"shift invariance broken: {rel:.3e} relative"


@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("slots", [4, 12])
def test_combine_does_not_spill_registers(device, head_dim, slots):
    """The chosen tile and warp count must compile with zero spills.

    This has to be asserted rather than benchmarked because register spilling
    is silent: unlike shared-memory overflow it does not fail the launch, and
    every correctness test still passes. Triton's default of 4 warps spilled
    220 values at head_dim 128 with 12 slots and ran 58% slower, and nothing in
    the suite noticed until the compiled kernel was inspected directly.
    """
    s = 256
    block_m, warps = choose_combine_tile(head_dim)
    part_m = torch.zeros((1, 2, s, slots), device=device)
    part_l = torch.ones((1, 2, s, slots), device=device)
    part_acc = torch.zeros((1, 2, s, slots, head_dim), device=device)
    out = torch.empty((1, 2, s, head_dim), device=device)

    kernel = _moba_combine[(triton.cdiv(s, block_m), 2)](
        part_m, part_l, part_acc, out, s,
        K_SLOTS=slots, D=head_dim, BLOCK_M=block_m, num_warps=warps,
    )

    assert kernel.n_spills == 0, (
        f"combine spills {kernel.n_spills} values at head_dim {head_dim}, "
        f"{slots} slots (block_m {block_m}, {warps} warps, "
        f"{kernel.n_regs} registers)"
    )


def test_combine_rejects_a_tile_that_would_spill(device):
    """An explicit oversized `block_m` is refused, not silently honoured."""
    part_m, part_l, part_acc = _random_partials(device, d=128)

    with pytest.raises(ValueError, match="without spilling registers"):
        combine_partials(part_m, part_l, part_acc, torch.float32, block_m=128)
