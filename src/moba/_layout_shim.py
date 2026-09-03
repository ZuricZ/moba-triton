"""TEMPORARY scaffolding for the token-major migration. Delete in Task 5.

The public surface, the oracle and the tests speak token-major `[B, S, H, D]`
from Task 1 of `docs/superpowers/plans/2026-09-03-token-major-layout.md`. The
Triton kernels still speak head-major and are converted one group at a time in
Tasks 2-4. This module bridges the two so the suite stays green throughout
instead of going red for four tasks.

Every call here is a real transpose and a real copy. **No benchmark number taken
while this module exists is meaningful.**

Deletion check, once Tasks 2-4 are done:

    grep -rn "_layout_shim" src/ tests/ bench/     # expect no output
"""

import torch


def to_head_major(*tensors: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """Converts token-major [B, S, H, D] inputs to head-major [B, H, S, D].

    Differentiable, so autograd routes gradients back through it without the
    caller doing anything. That is why the shim sits outside
    `_MoBAAttention.apply` rather than inside its `forward`.

    Args:
        *tensors: Token-major tensors.

    Returns:
        The same tensors, head-major and contiguous.
    """
    return tuple(t.transpose(1, 2).contiguous() for t in tensors)


def to_token_major(t: torch.Tensor) -> torch.Tensor:
    """Converts a head-major [B, H, S, D] result back to [B, S, H, D].

    Args:
        t: Head-major tensor.

    Returns:
        The same tensor, token-major and contiguous.
    """
    return t.transpose(1, 2).contiguous()
