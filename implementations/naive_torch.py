"""The oracle, exposed as a backend."""

import torch

from implementations.base import register
from moba.config import MoBAConfig
from moba.naive import moba_attention_naive, moba_selection


class NaiveTorchBackend:
    """Pure-PyTorch MoBA. Always available; the correctness ground truth.

    Reports available unconditionally -- it is plain tensor ops, so it runs
    wherever torch runs, including CPU. That matters because it is the one
    backend the harness cannot do without.
    """

    name = "naive_torch"
    vendored = False

    def is_available(self, cfg: MoBAConfig) -> tuple[bool, str]:
        return True, ""

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cfg: MoBAConfig,
    ) -> torch.Tensor:
        return moba_attention_naive(q, k, v, cfg)

    def selection(
        self, q: torch.Tensor, k: torch.Tensor, cfg: MoBAConfig
    ) -> torch.Tensor:
        """Exposes routing so it can be checked exactly, separate from output.

        Optional part of the Backend interface. A backend that implements it
        gets its gate verified directly; one that does not is covered only
        indirectly, through output comparison. Direct is far better: a routing
        bug and a precision bug look identical in the output but need
        completely different fixes.

        Args:
            q: Queries shaped [B, S, H, D].
            k: Keys shaped [B, S, H, D].
            cfg: Configuration.

        Returns:
            Boolean selection mask shaped [B, H, S, num_blocks].
        """
        return moba_selection(q, k, cfg)


register(NaiveTorchBackend())
