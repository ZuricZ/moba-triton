"""mit-han-lab/flash-moba as a backend -- the performance target.

Unlike the other two references this is a compiled CUDA extension, so it
cannot be loaded from source: it must be built and installed. Two obstacles
are expected on this machine, both reported here as availability reasons
rather than crashes.

1. Its `setup.py` defaults to CUDA archs `80;90;100;120`. sm_86 is absent.
2. Even with 86 added, its shared-memory tiling is sized for the 164 KiB per
   SM an A100 provides. This machine's sm_86 offers 99 KiB, so a kernel that
   compiles may still fail at *launch* with a shared-memory error.

The FlashMoBA digest (Task 10) depends on *reading* `references/flash-moba/`,
not on running it, so an unavailable backend does not block that work.
"""

import importlib.util

import torch

from moba.backends._refs import cu_seqlens_for, from_varlen, to_varlen
from moba.backends.base import register
from moba.config import MoBAConfig
from moba.device import supports_dtype


class FlashMoBABackend:
    """FlashMoBA (CUDA), from MIT Han Lab."""

    name = "flash_moba"
    vendored = True

    def is_available(self, cfg: MoBAConfig) -> tuple[bool, str]:
        if not torch.cuda.is_available():
            return False, "no CUDA device"

        capability = torch.cuda.get_device_capability(0)
        ok, reason = supports_dtype(cfg.dtype, capability)
        if not ok:
            return False, reason

        if importlib.util.find_spec("flash_moba") is None:
            return False, (
                "extension not built; setup.py targets archs 80;90;100;120 "
                f"and this GPU is sm_{capability[0]}{capability[1]} "
                "(see references/flash-moba)"
            )
        return True, ""

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cfg: MoBAConfig,
    ) -> torch.Tensor:
        ok, reason = self.is_available(cfg)
        if not ok:
            raise RuntimeError(f"flash_moba backend not available: {reason}")

        from flash_moba import flash_moba_attn_varlen_func  # type: ignore

        b, h, s, d = q.shape
        out = flash_moba_attn_varlen_func(
            to_varlen(q),
            to_varlen(k),
            to_varlen(v),
            cu_seqlens_for(b, s, q.device),
            cu_seqlens_for(b, s, q.device),
            s,
            s,
            cfg.block_size,
            cfg.top_k,
            causal=True,
        )
        return from_varlen(out, b, s)


register(FlashMoBABackend())
