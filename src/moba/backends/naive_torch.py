"""The oracle, exposed as a backend."""

import torch

from moba.backends.base import register
from moba.config import MoBAConfig
from moba.ref.naive import moba_attention_naive


class NaiveTorchBackend:
    """Pure-PyTorch MoBA. Always available; the correctness ground truth.

    Reports available unconditionally -- it is plain tensor ops, so it runs
    wherever torch runs, including CPU. That matters because it is the one
    backend the harness cannot do without.
    """

    name = "naive_torch"

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


register(NaiveTorchBackend())
