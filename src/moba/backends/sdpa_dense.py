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

The same argument applies to packed batches, which is why `forward_packed`
exists. Timing our variable-length path against a baseline that pads would
credit us with work the baseline wasted, not work we saved. Handing SDPA a
jagged nested tensor makes it skip the padding too, so the comparison isolates
sparsity from raggedness.
"""

import torch
import torch.nn.functional as F
from torch.nested import nested_tensor_from_jagged

from moba.backends.base import register
from moba.config import MoBAConfig


class SdpaDenseBackend:
    """torch.nn.functional.scaled_dot_product_attention, causal, dense."""

    name = "sdpa_dense"
    vendored = False

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

    def forward_packed(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cfg: MoBAConfig,
        cu_seqlens: torch.Tensor,
    ) -> torch.Tensor:
        """Dense causal attention over a packed batch, without padding it.

        A jagged nested tensor is our packed form plus `cu_seqlens`, so the
        conversion is a relabelling of the same bytes: `nested_tensor_from_jagged`
        takes the [T, H, D] values and the offsets and returns [B, j, H, D]. The
        transposes into and out of head-major are the concrete cost of DD-1 at
        this boundary; both are copies, and both are inside the timed region so
        the baseline is not flattered.

        Causality is per sequence, enforced by the layout rather than by a mask.
        Verified on this box: SDPA runs this on `FLASH_ATTENTION` and refuses it
        on `MATH` -- "Nested tensors for query / key are not supported when
        is_causal=True" -- which proves the fused variable-length kernel is doing
        the work rather than a padded fallback.

        `max_seqlen` is deliberately absent. Our kernels need it to size a grid
        on the host; SDPA reads its own offsets, so requiring it here would be a
        parameter that exists only for symmetry.

        Args:
            q: Packed queries shaped [1, H, T, D].
            k: Packed keys, same shape.
            v: Packed values, same shape.
            cfg: Configuration. Unused, as in `forward`.
            cu_seqlens: Cumulative token counts, [num_sequences + 1] int32.

        Returns:
            Packed output shaped [1, H, T, D], same dtype as `q`.

        Raises:
            ValueError: If the inputs are not a packed batch.
        """
        if q.ndim != 4 or q.shape[0] != 1:
            raise ValueError(
                f"packed inputs must be [1, H, T, D], got {tuple(q.shape)}"
            )
        offsets = cu_seqlens.to(torch.int64)

        def jag(t: torch.Tensor) -> torch.Tensor:
            # [1, H, T, D] -> [T, H, D] -> nested [B, j, H, D] -> [B, H, j, D].
            values = t[0].transpose(0, 1).contiguous()
            return nested_tensor_from_jagged(values, offsets=offsets).transpose(
                1, 2
            )

        out = F.scaled_dot_product_attention(
            jag(q), jag(k), jag(v), is_causal=True
        )
        return out.transpose(1, 2).values().transpose(0, 1).unsqueeze(0)


register(SdpaDenseBackend())
