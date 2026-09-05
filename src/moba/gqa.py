"""Reconciling a query head count with a smaller key/value head count.

Grouped-query attention lets several query heads share one key/value head. The
group size is not configured anywhere: it is `H_q // H_kv`, read off the
tensors. A `MoBAConfig` field would be a second source of truth for a fact the
shapes already state, and the two could disagree; `MoBAConfig` exists for
genuinely ambiguous *semantics* -- tie breaks, own-block handling -- not for
restating shapes.

`expand_kv` is oracle and test support only. Materializing the repeated heads is
precisely the memory traffic grouped-query attention exists to avoid, so no
kernel path may call it: the kernels share a K/V tile by *owning* it for the
whole group, never by copying it. Doing the expansion explicitly here is what
makes the kernels' avoidance of it checkable --
`docs/learning/03-gqa-and-the-column-merge.md` explains why the KV-block-major
loop order collects that saving without widening a single tile.
"""

import torch


def head_group_size(num_query_heads: int, num_kv_heads: int) -> int:
    """Returns how many query heads share each key/value head.

    Args:
        num_query_heads: Number of query heads.
        num_kv_heads: Number of key/value heads.

    Returns:
        The group size, 1 for plain multi-head attention.

    Raises:
        ValueError: If either count is not positive, or if the query head count
            is not a multiple of the key/value head count.
    """
    if num_query_heads <= 0 or num_kv_heads <= 0:
        raise ValueError(
            f"head counts must be positive, got q={num_query_heads} "
            f"kv={num_kv_heads}"
        )
    if num_query_heads % num_kv_heads != 0:
        raise ValueError(
            f"query heads {num_query_heads} must be divisible by key/value "
            f"heads {num_kv_heads}; a fractional group has no meaning"
        )
    return num_query_heads // num_kv_heads


def expand_kv(t: torch.Tensor, num_query_heads: int, dim: int = 2) -> torch.Tensor:
    """Repeats each key/value head across the query heads that share it.

    Test and oracle support only; see the module docstring.

    The axis is an argument because this codebase holds two conventions on
    purpose, split on a clean line: tensors crossing the API boundary are
    token-major `[B, S, H, D]`, so their head axis is 2, while buffers we
    allocate for ourselves -- block representatives, selection masks, partials,
    every `CSCIndex` field -- stayed head-major, where it is 1. Nothing outside
    the project reads the latter, so moving them would have been churn with no
    beneficiary. This function is the only one that sees both.

    Args:
        t: Tensor whose head axis is `dim`, shaped [B, S, H_kv, D] by default.
        num_query_heads: Target head count.
        dim: Axis holding the key/value heads. 2 for token-major inputs, 1 for
            the head-major buffers named above.

    Returns:
        The same shape with `H_kv` replaced by `num_query_heads`. Returns `t`
        itself when the head counts already match, so the multi-head path
        copies nothing.

    Raises:
        ValueError: If the head counts are incompatible.
    """
    group = head_group_size(num_query_heads, t.shape[dim])
    if group == 1:
        return t

    # `repeat_interleave` rather than `repeat`: head g must map to head g//group,
    # so the copies of one source head have to be adjacent. `repeat` would tile
    # the whole head axis instead, mapping head g to g % H_kv -- a permutation
    # that is wrong in a way no shape check can see.
    return t.repeat_interleave(group, dim=dim)
