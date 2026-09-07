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

from conftest import GATE_SWEEP, QUERY_CHUNK, make_qkv, sweep_cases
from moba.config import MoBAConfig
from moba.device import shared_memory_budget
from moba.kernels.routing import build_csc, csc_to_mask, build_csc_fused
from triton.runtime.errors import OutOfResources

from moba.kernels import moba_fwd
from moba.kernels.moba_fwd import (
    _GROUP_CACHE,
    moba_fwd_kernel,
    moba_combine_kernel,
    attend_partials,
    combine_partials,
    empty_partials,
    moba_forward,
)
from moba.kernels.tuning import choose_combine_tile, group_candidates
from moba.naive import moba_selection
from tolerance import (
    assert_selection_matches,
    assert_within_tolerance,
    reference_error,
)


def test_single_slot_matches_the_oracle(device):
    """top_k=1 means one block per query, so no cross-slot merging occurs."""
    cfg = MoBAConfig(block_size=32, top_k=1, dtype=torch.bfloat16)
    q, k, v = make_qkv((1, 512, 4, 64), dtype=torch.float64, device=device)
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
        actual, truth, ref_err, name=f"ours-blk{block_size}-top{top_k}"
    )


@pytest.mark.parametrize("shape,gate", sweep_cases(include_long=False))
def test_kernel_routing_matches_the_oracle_exactly(device, shape, gate):
    """The gate is a ranking, so its output is checked exactly, never by norm."""
    block_size, top_k = gate
    cfg = MoBAConfig(block_size=block_size, top_k=top_k, dtype=torch.bfloat16)
    q, k, _ = make_qkv(shape, dtype=torch.float64, device=device)
    ql, kl = q.to(cfg.dtype), k.to(cfg.dtype)

    expected = moba_selection(ql, kl, cfg)
    actual = csc_to_mask(build_csc(ql, kl, cfg), shape[1])

    assert_selection_matches(actual, expected, name="ours")


def test_rejects_mismatched_shapes(device):
    """Validation fires before any launch."""
    cfg = MoBAConfig(block_size=32, top_k=2, dtype=torch.bfloat16)
    q, _, _ = make_qkv((1, 128, 2, 64), dtype=torch.bfloat16, device=device)
    k, _, _ = make_qkv((1, 256, 2, 64), dtype=torch.bfloat16, device=device)

    with pytest.raises(ValueError, match="shape mismatch"):
        moba_forward(q, k, k, cfg)


def test_rejects_block_size_below_tensor_core_minimum(device):
    """block_size < 16 cannot fill a tl.dot operand and is refused, not padded."""
    cfg = MoBAConfig(block_size=8, top_k=2, dtype=torch.bfloat16)
    q, k, v = make_qkv((1, 128, 2, 64), dtype=torch.bfloat16, device=device)

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

    Uses `exp2`, not `exp`, because the kernel carries its maxima in log2
    units: the attention kernel folds log2(e) into its scale factor so it can
    use the hardware's native 2^x instruction, which measured 7% faster. The
    merged result is identical either way -- softmax does not care which base
    its logits are expressed in -- but a reference in the wrong base disagrees
    with the kernel by exp((ln 2 - 1) * (m_i - m_g)), which looks like a merge
    bug and is not one.

    Args:
        part_m: Per-slot maxima in log2 units, [B, H, S, K].
        part_l: Per-slot denominators, [B, H, S, K].
        part_acc: Per-slot un-normalized outputs, [B, H, S, K, D].

    Returns:
        Merged output shaped [B, S, H, D], float64 -- token-major, matching
        what `combine_partials` returns. The partials themselves stay
        head-major: they are ours, and only the merged output crosses the API
        boundary.
    """
    m, l, acc = (t.double() for t in (part_m, part_l, part_acc))
    w = torch.exp2(m - m.max(dim=-1, keepdim=True).values)
    merged = (acc * w.unsqueeze(-1)).sum(-2) / (l * w).sum(-1).unsqueeze(-1)
    return merged.transpose(1, 2)          # [B, H, S, D] -> [B, S, H, D]


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
    expected = (part_acc[..., 0, :]
                / part_l[..., 0].unsqueeze(-1)).transpose(1, 2)

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
    out = torch.empty((1, s, 2, head_dim), device=device)

    lse = torch.empty((1, 2, s), device=device)

    # Both arms: writing the log-sum-exp costs an extra live value per query,
    # and the backward pass always asks for it, so the arm the training path
    # uses has to be the one that is checked.
    for write_lse in (False, True):
        kernel = moba_combine_kernel[(triton.cdiv(s, block_m), 2)](
            part_m, part_l, part_acc, out, lse, s,
            K_SLOTS=slots, D=head_dim, BLOCK_M=block_m, H_Q=2,
            WRITE_LSE=write_lse, num_warps=warps,
        )

        assert kernel.n_spills == 0, (
            f"combine spills {kernel.n_spills} values at head_dim "
            f"{head_dim}, {slots} slots, write_lse={write_lse} (block_m "
            f"{block_m}, {warps} warps, {kernel.n_regs} registers)"
        )


def test_combine_rejects_a_tile_that_would_spill(device):
    """An explicit oversized `block_m` is refused, not silently honoured."""
    part_m, part_l, part_acc = _random_partials(device, d=128)

    with pytest.raises(ValueError, match="without spilling registers"):
        combine_partials(part_m, part_l, part_acc, torch.float32, block_m=128)


def test_combine_ignores_garbage_in_slots_no_program_wrote(device):
    """Unwritten accumulator slots may hold anything, including NaN.

    `attend_partials` deliberately does not zero `part_acc` -- it is the
    largest buffer in the design, 128 MB at S=8192, and zeroing it means
    writing it once before the kernel writes it again. Only the per-slot
    maximum is initialized, and the combine pass selects on that marker.

    The hole this closes is specific: multiplying by a zero weight kills a
    finite leftover value, but `0 * NaN` is NaN, so arithmetic alone is not
    enough. `tl.where` picks a branch instead of evaluating both, which is why
    it works. Filling the untouched slots with NaN here is not a pathological
    case -- it is what uninitialized device memory is allowed to contain.
    """
    part_m, part_l, part_acc = _random_partials(device, k=4)
    part_l[..., -1] = float("nan")
    part_acc[..., -1, :] = float("nan")

    actual = combine_partials(part_m, part_l, part_acc, torch.float32)

    assert torch.isfinite(actual).all(), "NaN leaked from an unwritten slot"

    # The answer must equal the one obtained with those slots properly zeroed.
    part_l[..., -1] = 0.0
    part_acc[..., -1, :] = 0.0
    expected = _reference_combine(part_m, part_l, part_acc)
    err = (actual.double() - expected).abs().max().item()
    assert err < 1e-5, f"garbage changed the result by {err:.3e}"


@pytest.mark.parametrize("gate", GATE_SWEEP)
@pytest.mark.parametrize("head_dim", [64, 128])
def test_the_chunk_height_that_ran_fits_shared_memory(device, gate, head_dim):
    """Whatever height was chosen, the kernel it compiled has to fit.

    An earlier version compared `estimate_shared_memory` against the kernel's
    reported `metadata.shared`, and immediately showed why that comparison is
    the wrong one: Triton pipelines the gathered q tile 2 deep at head_dim 128
    and 4 deep at 64, so no constant depth is both safe and tight. Modelling it
    conservatively is what cost 12% by rejecting GROUP=64.

    So this asserts the property that survives any selection policy. Shared
    memory overflow fails a launch outright rather than degrading, which makes
    this the invariant worth guarding.
    """
    block_size, top_k = gate
    cfg = MoBAConfig(block_size=block_size, top_k=top_k, dtype=torch.bfloat16)
    q, k, v = make_qkv((1, 1024, 1, head_dim), dtype=torch.bfloat16,
                       device=device)
    index = build_csc_fused(q, k, cfg)
    budget = shared_memory_budget(q.device)

    _GROUP_CACHE.clear()
    attend_partials(q, k, v, cfg, index)
    key = (block_size, head_dim, q.element_size(), index.max_slots, True, 1,
           torch.cuda.get_device_capability(q.device))
    used = _GROUP_CACHE[key]

    kernel = _compile_attn(q, k, v, cfg, index, used)
    assert kernel.metadata.shared <= budget, (
        f"chunk height {used} allocates {kernel.metadata.shared} bytes, over "
        f"the {budget} budget"
    )


@pytest.mark.parametrize("gate", GATE_SWEEP)
@pytest.mark.parametrize("head_dim", [64, 128])
def test_without_autotuning_the_height_is_the_largest_that_compiles(
    device, gate, head_dim, monkeypatch
):
    """The fallback policy, which MOBA_AUTOTUNE=0 selects, is still exact.

    This test used to be unconditional and was named "...and is the largest
    that does". That stopped being true on 2026-09-07: `attend_partials` now
    times the candidates that compile and keeps the fastest, which on sm_86 is
    64 where the largest that fits is 128 -- worth up to 15%, because a tile
    that large spills registers and leaves an SM with nothing resident to hide
    memory latency behind.

    The old contract is still the fallback, and a fallback nobody tests is a
    fallback that quietly rots, so it is asserted here with autotuning off
    rather than deleted.
    """
    monkeypatch.setattr(moba_fwd, "_AUTOTUNE", False)
    block_size, top_k = gate
    cfg = MoBAConfig(block_size=block_size, top_k=top_k, dtype=torch.bfloat16)
    q, k, v = make_qkv((1, 1024, 1, head_dim), dtype=torch.bfloat16,
                       device=device)
    index = build_csc_fused(q, k, cfg)

    _GROUP_CACHE.clear()
    attend_partials(q, k, v, cfg, index)
    key = (block_size, head_dim, q.element_size(), index.max_slots, True, 1,
           torch.cuda.get_device_capability(q.device))
    used = _GROUP_CACHE[key]

    bigger = [g for g in group_candidates(cfg, head_dim, q.element_size(),
                                          q.device) if g > used]
    for g in bigger:
        with pytest.raises(OutOfResources):
            _compile_attn(q, k, v, cfg, index, g)


@pytest.mark.parametrize("gate", GATE_SWEEP)
def test_autotuning_moves_the_result_by_far_less_than_the_dtype_does(
    device, gate
):
    """The autotuner's freedom has to be numerically negligible, and is.

    This asserted `torch.equal` when first written, and the hardware disagreed:
    at block_size 32 the heights 16 and 32 agree bitwise, 64 and 128 agree
    bitwise, and the two pairs differ by 1.2e-4. Chunk height decides how many
    queries a program handles per iteration, not what each query computes --
    but `tl.dot` selects different instruction sequences for different tile
    shapes, so the accumulation order inside the dot product moves with it.

    1.2e-4 against bfloat16's epsilon of ~8e-3 is a difference the storage
    format cannot represent as a distinction, which is the standard this
    project already applies elsewhere: error is judged against what the dtype
    already costs, not against zero. The stricter guarantee that *does* hold --
    that a given choice is stable run to run -- is enforced by
    `_AUTOTUNE_MARGIN` rather than by numerics.
    """
    block_size, top_k = gate
    cfg = MoBAConfig(block_size=block_size, top_k=top_k, dtype=torch.bfloat16)
    q, k, v = make_qkv((1, 1024, 2, 64), dtype=torch.bfloat16, device=device)
    index = build_csc_fused(q, k, cfg)

    _GROUP_CACHE.clear()
    tuned = combine_partials(*attend_partials(q, k, v, cfg, index), q.dtype)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(moba_fwd, "_AUTOTUNE", False)
        _GROUP_CACHE.clear()
        plain = combine_partials(*attend_partials(q, k, v, cfg, index), q.dtype)

    scale = plain.float().abs().max().item()
    err = (tuned.float() - plain.float()).abs().max().item()
    epsilon = torch.finfo(torch.bfloat16).eps
    assert err <= epsilon * scale, (
        f"chunk height moved the result by {err:.2e} against a bfloat16 "
        f"epsilon of {epsilon * scale:.2e}; it should be far smaller"
    )


def _compile_attn(q, k, v, cfg, index, group):
    """Launches the attention kernel at a given chunk height."""
    b, s, h_q, d = q.shape
    h_kv = k.shape[2]
    part_m, part_l, part_acc = empty_partials(
        q.shape, index.max_slots, q.device, q.dtype)
    part_m.fill_(float("-inf"))
    return moba_fwd_kernel[(index.num_blocks, b * h_kv)](
        q, k, v, index.row_indices, index.head_indices, index.slot_of_pair,
        index.col_offsets, index.col_nnz, index.order, q,
        part_m, part_l, part_acc,
        s, index.row_indices.shape[-1], s, d ** -0.5 * 1.4426950408889634,
        index.num_blocks, BLOCK_SIZE=cfg.block_size, GROUP=group, D=d,
        K_SLOTS=index.max_slots, H_Q=h_q, H_KV=h_kv, CAUSAL=True,
        VARLEN=False,
    )


@pytest.mark.parametrize("block_size,top_k", [(16, 3), (16, 5), (32, 3),
                                              (64, 7), (128, 8)])
def test_every_chunk_height_offered_is_a_power_of_two(block_size, top_k, device):
    """`tl.arange(0, GROUP)` is illegal otherwise, and the launch will not compile.

    `group_candidates` used to round `top_k * block_size` up to a multiple of
    TENSOR_CORE_TILE, which is not the same thing. Every configuration in the
    sweep happened to land on a power of two -- block_size 32 and above with
    top_k 4 and above gives at least 128, which MAX_GROUP caps -- so the defect
    was unreachable until a varlen test used top_k 3 at block_size 16 and got
    48.
    """
    cfg = MoBAConfig(block_size=block_size, top_k=top_k, dtype=torch.bfloat16)
    for group in group_candidates(cfg, 64, 2, device):
        assert group & (group - 1) == 0, (
            f"chunk height {group} at block_size {block_size}, top_k {top_k} "
            f"is not a power of two"
        )
