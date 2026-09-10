"""ReyJerry/Triton-MoBA as a backend -- the community Triton port.

Closest in spirit to what this project is building, so its disagreements with
us are the most informative of the three.

Its `triton_moba.py` imports `flash_attn` at module scope (it reuses
FlashAttention's varlen kernels for the dense-within-block work) and `einops`.
flash-attn 2.6.3 is a 30-60 minute source build that predates CUDA 12.7, so
this backend is expected to be unavailable locally. That is a designed-for
outcome, not a failure: the availability gate reports it and the suite carries
on.
"""

import importlib.util

import torch

from implementations.layout import (
    cu_seqlens_for,
    flash_attn_generation,
    from_varlen,
    load_reference_module,
    to_varlen,
)
from implementations.base import register
from moba.config import MoBAConfig

_MODULE_ALIAS = "_reyjerry_triton_moba"
_RELATIVE_PATH = "Triton-MoBA/triton_moba/triton_moba.py"


class TritonMoBABackend:
    """Community Triton port of MoBA, adapted from varlen to [B, S, H, D]."""

    name = "triton_moba"
    vendored = True
    computes_moba = True

    def is_available(self, cfg: MoBAConfig) -> tuple[bool, str]:
        if not torch.cuda.is_available():
            return False, "no CUDA device"
        for dep in ("flash_attn", "einops"):
            if importlib.util.find_spec(dep) is None:
                return False, f"requires {dep} (see references/Triton-MoBA)"
        try:
            load_reference_module(_MODULE_ALIAS, _RELATIVE_PATH)
        except FileNotFoundError as exc:
            return False, f"not vendored: run scripts/fetch_references.sh ({exc})"
        except Exception as exc:  # noqa: BLE001 - must never raise
            return False, f"import failed: {type(exc).__name__}: {exc}"
        # The mirror of the check in `moonshot_efficient`: this one unpacks
        # four values where 2.6.x returns eight. Their own requirements.txt
        # pins 2.6.3, which their code cannot actually run against.
        generation, detail = flash_attn_generation()
        if generation != 27:
            return False, f"needs the flash-attn >= 2.7 signature: {detail}"
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
            raise RuntimeError(f"triton_moba backend not available: {reason}")

        module = load_reference_module(_MODULE_ALIAS, _RELATIVE_PATH)
        b, s, h, d = q.shape        # token-major
        out = module.moba_attn_varlen_triton(
            to_varlen(q),
            to_varlen(k),
            to_varlen(v),
            cu_seqlens=cu_seqlens_for(b, s, q.device),
            max_seqlen=s,
            moba_chunk_size=cfg.block_size,
            moba_topk=cfg.top_k,
        )
        return from_varlen(out, b, s)


register(TritonMoBABackend())
