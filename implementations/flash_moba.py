"""mit-han-lab/flash-moba as a backend -- the performance target.

Unlike the other two references this is a compiled CUDA extension, so it
cannot be loaded from source: it must be built and installed. Its `setup.py`
defaults to CUDA archs `80;90;100;120`, so sm_86 needs
`FLASH_MOBA_CUDA_ARCHS=86` set before the build.

A SECOND OBSTACLE WAS PREDICTED AND DID NOT HAPPEN. This docstring used to
warn that their tiling is sized for the 164 KiB of shared memory an A100
provides, and that a kernel compiling on this card's 99 KiB might still fail
at *launch*. Built and measured 2026-09-05: it launches, at every sequence
length from 1024 to 8192. The reason is `decide_lg_block_m`, which picks the
query tile from predicted sparsity at call time -- 128 at S=1024 rising to 512
at S=8192 -- rather than fixing it at compile time. A runtime policy fits
whatever SRAM the card has; a compile-time constant would not have.

Recorded because the prediction was load-bearing: it is why this backend went
unbuilt for so long.
"""

import importlib.util

import torch

from implementations.base import register
from implementations.layout import cu_seqlens_for, from_varlen, to_varlen
from moba.config import MoBAConfig
from moba.device import supports_dtype


class FlashMoBABackend:
    """FlashMoBA (CUDA), from MIT Han Lab."""

    name = "flash_moba"
    vendored = True
    computes_moba = True

    def is_available(self, cfg: MoBAConfig) -> tuple[bool, str]:
        if not torch.cuda.is_available():
            return False, "no CUDA device"

        capability = torch.cuda.get_device_capability(0)
        ok, reason = supports_dtype(cfg.dtype, capability)
        if not ok:
            return False, reason

        # Their kernel requires `n_lg_block_dim` to be a multiple of 64 --
        # raised from C++ as a RuntimeError, after the launch has been set up.
        # Declared here instead so the grid's block_size=32 rows skip with a
        # reason rather than erroring, which is this project's policy for a
        # backend that cannot run a configuration.
        if cfg.block_size % 64 != 0:
            return False, (
                f"block_size={cfg.block_size} is not a multiple of 64, which "
                "flash-moba requires (n_lg_block_dim)"
            )

        if importlib.util.find_spec("flash_moba") is None:
            return False, (
                "extension not built; setup.py targets archs 80;90;100;120 "
                f"and this GPU is sm_{capability[0]}{capability[1]}. Build it "
                f"with FLASH_MOBA_CUDA_ARCHS={capability[0]}{capability[1]} "
                "outside /mnt (see docs/reference-comparison.md)"
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

        # `flash_moba_varlen_func`, not `flash_moba_attn_varlen_func`. The
        # latter is the attention kernel alone: it takes the routing as three
        # tensors (col_offsets, col_nnz, row_indices) and does no gating. This
        # one is their whole pipeline -- gate, then attend -- which is what
        # `ours` does too, so the two are measured over the same stages.
        from flash_moba import flash_moba_varlen_func  # type: ignore

        b, s = q.shape[:2]          # token-major [B, S, H, D]
        cu = cu_seqlens_for(b, s, q.device)
        out = flash_moba_varlen_func(
            to_varlen(q),
            to_varlen(k),
            to_varlen(v),
            cu,
            cu,
            s,
            s,
            cfg.block_size,
            cfg.top_k,
            causal=True,
        )
        return from_varlen(out, b, s)


register(FlashMoBABackend())
