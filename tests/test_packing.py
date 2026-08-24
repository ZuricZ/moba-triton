"""Converting between a padded batch and the packed form, in our layout.

FlashAttention ships `unpad_input`/`pad_input` for this
(`references/flash-moba/flash_moba/bert_padding.py:98`), but they cannot be
lifted verbatim: they gather *rows*, which works because FA's token axis is
outermost. Ours is head-major, so one token's data is `H` separate runs of `D`
rather than a contiguous block, and packing gathers along the sequence axis
instead. Same idea, different index arithmetic, slightly worse memory pattern --
recorded in `docs/deferred-decisions.md`, DD-1.

`validate_cu_seqlens` is the part that runs on every forward call, so it is
tested for what it must *not* do as well as what it must: no host
synchronization. `pack` runs once at data-prep time and may synchronize, exactly
as FA's `unpad_input` does.
"""

import pytest
import torch

from moba.packing import from_nested, pack, unpack, validate_cu_seqlens


def test_pack_keeps_only_real_tokens(device):
    """Lengths 5, 3, 4 in a batch padded to 5 -- countable by hand."""
    x = torch.arange(3 * 2 * 5 * 4, dtype=torch.float32,
                     device=device).view(3, 2, 5, 4)
    mask = torch.tensor([[1, 1, 1, 1, 1],
                         [1, 1, 1, 0, 0],
                         [1, 1, 1, 1, 0]], device=device)

    packed, cu, max_seqlen = pack(x, mask)
    assert packed.shape == (1, 2, 12, 4)
    assert cu.tolist() == [0, 5, 8, 12]
    assert max_seqlen == 5
    assert cu.dtype == torch.int32

    for i, (lo, n) in enumerate([(0, 5), (5, 3), (8, 4)]):
        assert torch.equal(packed[0, :, lo:lo + n], x[i, :, :n]), f"seq {i}"


def test_unpack_inverts_pack_on_the_real_tokens(device):
    """Round trip. Padding comes back zeroed, not as whatever was there."""
    x = torch.randn(3, 2, 5, 4, device=device)
    mask = torch.tensor([[1, 1, 1, 1, 1],
                         [1, 1, 1, 0, 0],
                         [1, 1, 1, 1, 0]], device=device)
    packed, cu, _ = pack(x, mask)
    back = unpack(packed, cu, batch=3, seq_len=5)

    m = mask.bool()[:, None, :, None].expand_as(x)
    assert torch.equal(back[m], x[m]), "a real token changed"
    assert not bool(back[~m].any()), "padding must come back zeroed"


def test_pack_handles_a_batch_with_no_padding(device):
    """A uniform batch must still pack correctly, buying nothing.

    Worth pinning: the gain from packing is `1 - mean(len)/max(len)`, which is
    zero here. The code path must not special-case it into being wrong.
    """
    x = torch.randn(3, 2, 5, 4, device=device)
    mask = torch.ones(3, 5, device=device)
    packed, cu, max_seqlen = pack(x, mask)

    assert packed.shape == (1, 2, 15, 4)
    assert cu.tolist() == [0, 5, 10, 15]
    assert max_seqlen == 5
    assert torch.equal(unpack(packed, cu, batch=3, seq_len=5), x)


def test_from_nested_unwraps_a_jagged_tensor(device):
    """A jagged nested tensor is already packed plus cu_seqlens.

    `nt.values()` is the packed data and `nt.offsets()` *is* cu_seqlens; the
    only work is the transpose into our head-major layout, which is a copy.
    """
    nt = torch.nested.nested_tensor(
        [torch.randn(5, 2, 4), torch.randn(3, 2, 4), torch.randn(4, 2, 4)],
        layout=torch.jagged, device=device, dtype=torch.float32)
    packed, cu, max_seqlen = from_nested(nt)

    assert packed.shape == (1, 2, 12, 4)
    assert cu.tolist() == [0, 5, 8, 12]
    assert max_seqlen == 5
    assert packed.is_contiguous()
    # Head-major: token t of the packed buffer must equal row t of nt.values().
    assert torch.equal(packed[0].transpose(0, 1), nt.values())


def test_validate_returns_the_sequence_count(device):
    cu = torch.tensor([0, 5, 8, 12], dtype=torch.int32, device=device)
    assert validate_cu_seqlens(cu, total_tokens=12, max_seqlen=5) == 3


@pytest.mark.parametrize("cu,total,mx,match", [
    ([0, 5, 8, 12], 11, 5, "total"),         # last entry disagrees with T
    ([0, 5, 8, 12], 12, 4, "max_seqlen"),    # max too small for the longest
    ([5, 8, 12], 12, 5, "start at 0"),
    ([0, 8, 5], 5, 8, "increasing"),
    ([0, 0, 5], 5, 5, "increasing"),         # an empty sequence
])
def test_malformed_values_are_rejected_on_a_cpu_tensor(cu, total, mx, match):
    """Value checks are eager on the CPU, where reading them is free.

    Deliberately not parametrized over `device`. On a CUDA tensor these checks
    would each cost a host synchronization -- ~230 us, on every forward call --
    so they are skipped there by default. See `strict` below and the module
    docstring.
    """
    t = torch.tensor(cu, dtype=torch.int32)
    with pytest.raises(ValueError, match=match):
        validate_cu_seqlens(t, total_tokens=total, max_seqlen=mx)


def test_structural_errors_are_caught_on_any_device(device):
    """Rank, length and dtype are host metadata, so they cost nothing to check."""
    with pytest.raises(ValueError, match="at least two"):
        validate_cu_seqlens(torch.tensor([0], dtype=torch.int32, device=device),
                            total_tokens=0, max_seqlen=1)
    with pytest.raises(ValueError, match="1-D"):
        validate_cu_seqlens(torch.zeros(2, 2, dtype=torch.int32, device=device),
                            total_tokens=0, max_seqlen=1)
    with pytest.raises(ValueError, match="positive"):
        validate_cu_seqlens(
            torch.tensor([0, 5], dtype=torch.int32, device=device),
            total_tokens=5, max_seqlen=0)


def test_strict_catches_a_bad_value_on_the_device_by_paying_the_sync(device):
    """The debugging escape hatch, opt-in and documented as costly."""
    bad = torch.tensor([0, 8, 5], dtype=torch.int32, device=device)
    validate_cu_seqlens(bad, total_tokens=5, max_seqlen=8)      # silent
    with pytest.raises(ValueError, match="increasing"):
        validate_cu_seqlens(bad, total_tokens=5, max_seqlen=8, strict=True)


def test_validation_does_not_synchronize_with_the_host(device):
    """Checks must not read a scalar back; that costs ~230 us on this box.

    `index_fused` exists to remove synchronizations of this kind. Validation
    that reintroduced one on every forward call would be a silent tax that
    looked like a kernel regression -- the exact trap this project keeps
    hitting, a plausible number with a convincing story attached.
    """
    cu = torch.tensor([0, 5, 8, 12], dtype=torch.int32, device=device)
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        validate_cu_seqlens(cu, total_tokens=12, max_seqlen=5)
    finally:
        torch.cuda.set_sync_debug_mode("default")


def test_validation_rejects_a_wrong_dtype(device):
    """int32 specifically: the kernels index with it and Triton will not cast."""
    cu = torch.tensor([0, 5, 8, 12], dtype=torch.int64, device=device)
    with pytest.raises(ValueError, match="int32"):
        validate_cu_seqlens(cu, total_tokens=12, max_seqlen=5)


def test_packed_oracle_equals_running_each_sequence_alone(device):
    """The definition of varlen, checked bitwise.

    Slicing the packed buffer gives the same inputs in the same order, so
    anything but equality is a boundary bug rather than rounding.
    """
    from moba.config import MoBAConfig
    from moba.ref.naive import moba_attention_naive

    cfg = MoBAConfig(block_size=4, top_k=2, dtype=torch.float64)
    cu = torch.tensor([0, 20, 32, 48], dtype=torch.int32, device=device)
    gen = torch.Generator(device=device).manual_seed(0)

    def mk():
        return torch.randn(1, 2, 48, 16, generator=gen, device=device,
                           dtype=torch.float64)

    q, k, v = mk(), mk(), mk()

    packed = moba_attention_naive(q, k, v, cfg, cu_seqlens=cu)
    lo = 0
    for n in [20, 12, 16]:
        piece = moba_attention_naive(q[:, :, lo:lo + n], k[:, :, lo:lo + n],
                                     v[:, :, lo:lo + n], cfg)
        assert torch.equal(packed[:, :, lo:lo + n], piece), f"at {lo}"
        lo += n


def test_packed_selection_uses_sequence_local_block_ids(device):
    """One convention, established here and reproduced by `csc_to_mask`.

    The mask is [1, H, T, NB] with NB the *longest* sequence's block count, and
    a column index means "block j of my own sequence". A global numbering would
    need a mask `total_blocks` wide, almost entirely false, growing with the
    batch for no information.
    """
    from moba.config import MoBAConfig
    from moba.ref.naive import moba_selection

    cfg = MoBAConfig(block_size=4, top_k=2, dtype=torch.float64)
    cu = torch.tensor([0, 20, 32, 48], dtype=torch.int32, device=device)
    gen = torch.Generator(device=device).manual_seed(1)
    q = torch.randn(1, 2, 48, 16, generator=gen, device=device,
                    dtype=torch.float64)
    k = torch.randn(1, 2, 48, 16, generator=gen, device=device,
                    dtype=torch.float64)

    sel = moba_selection(q, k, cfg, cu_seqlens=cu)
    assert sel.shape == (1, 2, 48, 5)      # NB = cdiv(20, 4)

    lo = 0
    for n in [20, 12, 16]:
        piece = moba_selection(q[:, :, lo:lo + n], k[:, :, lo:lo + n], cfg)
        nb = piece.shape[-1]
        assert torch.equal(sel[:, :, lo:lo + n, :nb], piece), f"at {lo}"
        # Columns past this sequence's own block count must be empty.
        assert not bool(sel[:, :, lo:lo + n, nb:].any()), (
            f"sequence at {lo} has {nb} blocks but selected beyond them")
        lo += n


def test_packed_oracle_rejects_a_real_batch_axis(device):
    from moba.config import MoBAConfig
    from moba.ref.naive import moba_attention_naive

    cfg = MoBAConfig(block_size=4, top_k=2, dtype=torch.float64)
    cu = torch.tensor([0, 8, 16], dtype=torch.int32, device=device)
    q = torch.randn(2, 2, 16, 16, device=device, dtype=torch.float64)
    with pytest.raises(ValueError, match="batch 1"):
        moba_attention_naive(q, q, q, cfg, cu_seqlens=cu)
