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


def expand_kv(t: torch.Tensor, num_query_heads: int) -> torch.Tensor:
    """Repeats each key/value head across the query heads that share it.

    Test and oracle support only; see the module docstring.

    Args:
        t: Tensor shaped [B, H_kv, S, D].
        num_query_heads: Target head count.

    Returns:
        Tensor shaped [B, num_query_heads, S, D]. Returns `t` itself when the
        head counts already match, so the multi-head path copies nothing.

    Raises:
        ValueError: If the head counts are incompatible.
    """
    group = head_group_size(num_query_heads, t.shape[1])
    if group == 1:
        return t

    # `repeat_interleave` rather than `repeat`: head g must map to head g//group,
    # so the copies of one source head have to be adjacent. `repeat` would tile
    # the whole head axis instead, mapping head g to g % H_kv -- a permutation
    # that is wrong in a way no shape check can see.
    return t.repeat_interleave(group, dim=1)
