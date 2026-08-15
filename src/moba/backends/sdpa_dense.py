"""Dense causal attention baseline.

This is deliberately NOT a MoBA implementation, and the harness excludes it
from correctness comparisons. It is the roofline: the cost of attending to
everything, computed by a heavily optimized kernel that PyTorch ships.

Its job is to anchor the benchmark. Every MoBA speedup is measured against it,
and the ratio between the speedup achieved and the speedup that sparsity
implies is available -- the `attainment` column -- is the number this project
is ultimately trying to move.

Using torch SDPA rather than a hand-written dense attention matters: SDPA
dispatches to FlashAttention or a fused kernel where it can, so the baseline
represents what a *good* dense implementation costs, not what a naive one
does. Beating a naive baseline would prove nothing.
"""

import torch
import torch.nn.functional as F

from moba.backends.base import register
from moba.config import MoBAConfig


class SdpaDenseBackend:
    """torch.nn.functional.scaled_dot_product_attention, causal, dense."""

    name = "sdpa_dense"

    def is_available(self, cfg: MoBAConfig) -> tuple[bool, str]:
        if not torch.cuda.is_available():
            return False, "no CUDA device"
        return True, ""

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cfg: MoBAConfig,
    ) -> torch.Tensor:
        """Ignores cfg's gate parameters by design: this attends to everything."""
        return F.scaled_dot_product_attention(q, k, v, is_causal=True)


register(SdpaDenseBackend())
