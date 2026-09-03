"""MoonshotAI/MoBA as a backend -- the algorithm's reference implementation.

We load `moba_naive.py` directly by path rather than importing their package.
Their package is also named `moba`, and their `__init__.py` pulls in
`transformers` and `flash_attn`. The naive module itself imports only torch
and math, so loading just that file gives us a runnable third-party MoBA with
no heavy dependencies and no name collision. See `_refs.load_reference_module`.

Their efficient path (`moba_efficient.py`) does need flash-attn 2.6.3 and is
deliberately not wired up here; the naive path is what matters for checking
semantics against ours.
"""

import torch

from moba.backends._refs import (
    cu_seqlens_for,
    from_varlen,
    load_reference_module,
    to_varlen,
)
from moba.backends.base import register
from moba.config import MoBAConfig

_MODULE_ALIAS = "_moonshot_moba_naive"
_RELATIVE_PATH = "MoBA/moba/moba_naive.py"


def _load():
    """Returns the reference module, or raises with a legible reason."""
    return load_reference_module(_MODULE_ALIAS, _RELATIVE_PATH)


class MoonshotBackend:
    """MoonshotAI's naive MoBA, adapted from varlen to [B, H, S, D]."""

    name = "moonshot"
    vendored = True

    def is_available(self, cfg: MoBAConfig) -> tuple[bool, str]:
        if not torch.cuda.is_available():
            return False, "no CUDA device"
        try:
            _load()
        except FileNotFoundError as exc:
            return False, f"not vendored: run scripts/fetch_references.sh ({exc})"
        except Exception as exc:  # noqa: BLE001 - must never raise
            return False, f"import failed: {type(exc).__name__}: {exc}"
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
            raise RuntimeError(f"moonshot backend not available: {reason}")

        module = _load()
        b, s, h, d = q.shape        # token-major
        out = module.moba_attn_varlen_naive(
            to_varlen(q),
            to_varlen(k),
            to_varlen(v),
            cu_seqlens=cu_seqlens_for(b, s, q.device),
            max_seqlen=s,
            moba_chunk_size=cfg.block_size,
            moba_topk=cfg.top_k,
        )
        return from_varlen(out, b, s)


register(MoonshotBackend())
