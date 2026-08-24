"""Converting between a padded batch and the packed (variable-length) form.

A packed batch stores sequences of different lengths end to end in one buffer,
with `cu_seqlens` -- cumulative sequence lengths -- marking the boundaries. This
module is the boundary between that representation and the rectangular one, and
holds nothing else: the kernels take `cu_seqlens` directly and derive everything
they need from their own `program_id`.

WHY NOT FLASHATTENTION'S HELPERS. FA ships `unpad_input`/`pad_input`
(`references/flash-moba/flash_moba/bert_padding.py:98`) which do exactly this,
and they cannot be lifted verbatim. They gather *rows* -- `torch.gather` along
axis 0 of a `(b s) ...`-flattened tensor -- which works because FA's token axis
is outermost, so one token is one contiguous `H*D`-byte run. Ours is head-major
`[B, H, S, D]`, where a token's data is `H` separate runs of `D` elements, `S*D`
apart. The gather is therefore strided: `H` short transactions per token instead
of one long one. That cost is recorded in `docs/deferred-decisions.md`, DD-1,
along with the condition that would make us reconsider the layout.

It is a once-per-batch cost at data-prep time, not a per-forward one.

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
        total_tokens: `q.shape[2]`. The last entry must equal this, and the
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
        x: Padded tensor shaped [B, H, S, D].
        attention_mask: Shaped [B, S], nonzero where a token is real.

    Returns:
        A tuple of (packed, cu_seqlens, max_seqlen), where `packed` is
        [1, H, T, D] contiguous, `cu_seqlens` is [B + 1] int32 on `x`'s device,
        and `max_seqlen` is a plain int.

    Raises:
        ValueError: If the shapes disagree.
    """
    if x.ndim != 4:
        raise ValueError(f"x must be [B, H, S, D], got {tuple(x.shape)}")
    b, _, s, _ = x.shape
    if attention_mask.shape != (b, s):
        raise ValueError(
            f"attention_mask must be [B, S] = {(b, s)}, got "
            f"{tuple(attention_mask.shape)}"
        )

    mask = attention_mask.bool()
    lengths = mask.sum(dim=1, dtype=torch.int32)
    cu_seqlens = F.pad(lengths.cumsum(0, dtype=torch.int32), (1, 0))
    max_seqlen = int(lengths.max())         # the one synchronization

    # Advanced indexing on dims 0 and 2 with a slice between them puts the
    # gathered axis first, giving [T, H, D]; the permute then puts it back into
    # head-major. Both steps copy. See the module docstring for why this cannot
    # be FlashAttention's single row gather.
    b_idx, s_idx = mask.nonzero(as_tuple=True)
    packed = x[b_idx, :, s_idx]                       # [T, H, D]
    packed = packed.permute(1, 0, 2).unsqueeze(0).contiguous()
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
        packed: Packed tensor shaped [1, H, T, D].
        cu_seqlens: Cumulative token counts, [batch + 1] int32.
        batch: Sequence count of the rectangle to rebuild.
        seq_len: Padded length of the rectangle to rebuild.

    Returns:
        Padded tensor shaped [batch, H, seq_len, D].

    Raises:
        ValueError: If the shapes disagree.
    """
    if packed.ndim != 4 or packed.shape[0] != 1:
        raise ValueError(
            f"packed must be [1, H, T, D], got {tuple(packed.shape)}"
        )
    _, h, t, d = packed.shape
    if cu_seqlens.numel() != batch + 1:
        raise ValueError(
            f"cu_seqlens has {cu_seqlens.numel()} entries, expected "
            f"{batch + 1} for batch {batch}"
        )

    lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).to(torch.int64)
    positions = torch.arange(seq_len, device=packed.device)
    mask = positions[None, :] < lengths[:, None]      # [batch, seq_len]

    out = torch.zeros(batch, h, seq_len, d, device=packed.device,
                      dtype=packed.dtype)
    b_idx, s_idx = mask.nonzero(as_tuple=True)
    out[b_idx, :, s_idx] = packed[0].permute(1, 0, 2)  # [T, H, D]
    return out


def from_nested(nt: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Unwraps a jagged nested tensor into our packed form.

    A jagged nested tensor is already packed plus `cu_seqlens`: `nt.values()` is
    the packed data as [T, H, D] and `nt.offsets()` *is* `cu_seqlens`. The only
    work is the transpose into head-major, which is a copy -- the concrete cost
    of DD-1 at this boundary.

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
    packed = values.permute(1, 0, 2).unsqueeze(0).contiguous()
    return packed, offsets.to(torch.int32), max_seqlen
