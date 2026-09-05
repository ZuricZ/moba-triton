"""Converting between a padded batch and the packed (variable-length) form.

A packed batch stores sequences of different lengths end to end in one buffer,
with `cu_seqlens` -- cumulative sequence lengths -- marking the boundaries. This
module is the boundary between that representation and the rectangular one, and
holds nothing else: the kernels take `cu_seqlens` directly and derive everything
they need from their own `program_id`.

WHY WE STILL SHIP OUR OWN. FA's `unpad_input`/`pad_input`
(`references/flash-moba/flash_moba/bert_padding.py:98`) now do exactly what
these do, in the same layout: since the move to token-major `[B, S, H, D]`, one
token's `H*D` elements are contiguous, so packing is a plain row gather on both
sides. DD-1 recorded the head-major version as a *strided* gather -- `H` short
transactions per token rather than one long one -- and that cost is gone.

What remains is a dependency argument, not a correctness one. FA's helpers carry
an autograd `Function` and the rest of `bert_padding`, for an operation that is
three lines of advanced indexing. Reimplementing is cheaper than importing.

WHAT MAY SYNCHRONIZE, AND WHAT MAY NOT. `pack` and `from_nested` run once when a
batch is built, and both must read a scalar -- the longest sequence -- so they
synchronize. FA does the same, with `.item()` on `max_seqlen_in_batch`.
`validate_cu_seqlens` runs on *every forward call* and must not: a host
synchronization costs ~230 us on this box (measured in `moba.kernels.index_fused`,
whose whole purpose is removing them), which against a ~2280 us forward would be
a silent 10% tax that looked like a kernel regression.
"""

import torch
import torch.nn.functional as F


def validate_cu_seqlens(
    cu_seqlens: torch.Tensor,
    total_tokens: int,
    max_seqlen: int,
    strict: bool = False,
) -> int:
    """Checks packed metadata and returns the sequence count.

    Structural checks -- rank, length, dtype -- are free: they read tensor
    metadata that already lives on the host, and they run always. *Value* checks
    -- starts at zero, strictly increasing, ends at `total_tokens`, no sequence
    longer than `max_seqlen` -- need the data, so they run eagerly only when
    `cu_seqlens` is on the CPU. On a device tensor they would each cost a host
    synchronization on every forward call.

    That is the same trade FlashAttention makes, more explicitly: it validates
    nothing and trusts the caller. `pack` and `from_nested` construct
    `cu_seqlens` correctly by definition, so the untrusted path is only a caller
    building it by hand.

    Args:
        cu_seqlens: Cumulative token counts, [num_sequences + 1] int32.
        total_tokens: `q.shape[1]`. The last entry must equal this, and the
            caller knows it without a transfer.
        max_seqlen: Longest sequence in the batch, as a plain int. Required
            rather than derived for the reason FlashAttention requires it:
            deriving it means reading `cu_seqlens.max()` back to the host.
        strict: Run the value checks even on a device tensor, paying the
            synchronization. For debugging a suspected bad index, not for
            production calls.

    Returns:
        Number of sequences.

    Raises:
        ValueError: If anything checked is wrong.
    """
    if cu_seqlens.ndim != 1:
        raise ValueError(
            f"cu_seqlens must be 1-D, got shape {tuple(cu_seqlens.shape)}"
        )
    if cu_seqlens.numel() < 2:
        raise ValueError(
            f"cu_seqlens needs at least two entries (one sequence), got "
            f"{cu_seqlens.numel()}"
        )
    if cu_seqlens.dtype != torch.int32:
        raise ValueError(
            f"cu_seqlens must be int32 -- the kernels index with it and Triton "
            f"will not cast -- got {cu_seqlens.dtype}"
        )
    if max_seqlen <= 0:
        raise ValueError(f"max_seqlen must be positive, got {max_seqlen}")

    num_sequences = cu_seqlens.numel() - 1

    if cu_seqlens.device.type == "cpu" or strict:
        cu = cu_seqlens.cpu()
        if int(cu[0]) != 0:
            raise ValueError(f"cu_seqlens must start at 0, got {int(cu[0])}")
        lengths = cu[1:] - cu[:-1]
        if bool((lengths <= 0).any()):
            raise ValueError(
                f"cu_seqlens must be strictly increasing (no empty sequence), "
                f"got lengths {lengths.tolist()}"
            )
        if int(cu[-1]) != total_tokens:
            raise ValueError(
                f"cu_seqlens ends at {int(cu[-1])} but the total token count is "
                f"{total_tokens}"
            )
        longest = int(lengths.max())
        if longest > max_seqlen:
            raise ValueError(
                f"max_seqlen {max_seqlen} is below the longest sequence "
                f"{longest}"
            )

    return num_sequences


def pack(
    x: torch.Tensor, attention_mask: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Packs a padded batch into one buffer of real tokens.

    Synchronizes once, to read the longest sequence. That is deliberate and
    matches FlashAttention's `unpad_input`: this runs when a batch is built, not
    on the hot path.

    Args:
        x: Padded tensor shaped [B, S, H, D].
        attention_mask: Shaped [B, S], nonzero where a token is real.

    Returns:
        A tuple of (packed, cu_seqlens, max_seqlen), where `packed` is
        [1, T, H, D] contiguous, `cu_seqlens` is [B + 1] int32 on `x`'s device,
        and `max_seqlen` is a plain int.

    Raises:
        ValueError: If the shapes disagree.
    """
    if x.ndim != 4:
        raise ValueError(f"x must be [B, S, H, D], got {tuple(x.shape)}")
    b, s, _, _ = x.shape
    if attention_mask.shape != (b, s):
        raise ValueError(
            f"attention_mask must be [B, S] = {(b, s)}, got "
            f"{tuple(attention_mask.shape)}"
        )

    mask = attention_mask.bool()
    lengths = mask.sum(dim=1, dtype=torch.int32)
    cu_seqlens = F.pad(lengths.cumsum(0, dtype=torch.int32), (1, 0))
    max_seqlen = int(lengths.max())         # the one synchronization

    # A row gather, and nothing else: indexing the two leading axes with 1-D
    # index tensors yields a contiguous [T, H, D], because one token's H*D
    # elements are already adjacent. This is FlashAttention's `unpad_input`
    # written out. Under head-major it needed a permute and a second copy.
    b_idx, s_idx = mask.nonzero(as_tuple=True)
    packed = x[b_idx, s_idx].unsqueeze(0)             # [1, T, H, D]
    return packed, cu_seqlens, max_seqlen


def unpack(
    packed: torch.Tensor,
    cu_seqlens: torch.Tensor,
    batch: int,
    seq_len: int,
) -> torch.Tensor:
    """Scatters a packed buffer back into a padded rectangle.

    Padding positions come back zeroed rather than uninitialized, so a caller
    that forgets to re-apply a mask gets a wrong-but-obvious answer instead of
    plausible garbage.

    Args:
        packed: Packed tensor shaped [1, T, H, D].
        cu_seqlens: Cumulative token counts, [batch + 1] int32.
        batch: Sequence count of the rectangle to rebuild.
        seq_len: Padded length of the rectangle to rebuild.

    Returns:
        Padded tensor shaped [batch, seq_len, H, D].

    Raises:
        ValueError: If the shapes disagree.
    """
    if packed.ndim != 4 or packed.shape[0] != 1:
        raise ValueError(
            f"packed must be [1, T, H, D], got {tuple(packed.shape)}"
        )
    _, t, h, d = packed.shape
    if cu_seqlens.numel() != batch + 1:
        raise ValueError(
            f"cu_seqlens has {cu_seqlens.numel()} entries, expected "
            f"{batch + 1} for batch {batch}"
        )

    lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).to(torch.int64)
    positions = torch.arange(seq_len, device=packed.device)
    mask = positions[None, :] < lengths[:, None]      # [batch, seq_len]

    out = torch.zeros(batch, seq_len, h, d, device=packed.device,
                      dtype=packed.dtype)
    b_idx, s_idx = mask.nonzero(as_tuple=True)
    out[b_idx, s_idx] = packed[0]                      # [T, H, D], a row scatter
    return out


def from_nested(nt: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Unwraps a jagged nested tensor into our packed form.

    A jagged nested tensor is already packed plus `cu_seqlens`: `nt.values()` is
    the packed data as [T, H, D] and `nt.offsets()` *is* `cu_seqlens`. In
    token-major that is our packed form exactly, so the only work is a rank
    change -- no transpose and no copy. Head-major needed one, which was the
    concrete cost of DD-1 at this boundary.

    Args:
        nt: A nested tensor with `layout=torch.jagged`, ragged over the token
            axis, shaped [B, j, H, D].

    Returns:
        A tuple of (packed, cu_seqlens, max_seqlen), as `pack` returns.

    Raises:
        ValueError: If `nt` is not a jagged nested tensor of rank 4.
    """
    if not nt.is_nested:
        raise ValueError("from_nested expects a nested tensor")
    values = nt.values()
    if values.ndim != 3:
        raise ValueError(
            f"expected a [B, j, H, D] jagged tensor; its values are "
            f"{tuple(values.shape)}, not [T, H, D]"
        )

    offsets = nt.offsets()
    lengths = offsets[1:] - offsets[:-1]
    max_seqlen = int(lengths.max())          # the one synchronization
    packed = values.unsqueeze(0).contiguous()
    return packed, offsets.to(torch.int32), max_seqlen
