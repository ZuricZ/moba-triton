"""MoonshotAI/MoBA, naive path -- the algorithm's readable reference.

We load `moba_naive.py` directly by path rather than importing their package.
Their package is also named `moba`, and their `__init__.py` pulls in
`transformers` and `flash_attn`. The naive module itself imports only torch
and math, so loading just that file gives us a runnable third-party MoBA with
no heavy dependencies and no name collision. See `layout.load_reference_module`.

WHY THE NAME CARRIES `_naive`. This backend was called plain `moonshot` until
2026-09-07, which quietly claimed it was MoonshotAI's MoBA. It is not: they
ship two, and this is the one written to be *read*. `moba_efficient.py` is the
one written to be fast, and it lives in `moonshot_efficient.py` beside this.
Reporting a speedup against the readable one under the unqualified name of its
authors overstated our result, and the fix belongs in the name rather than in a
footnote nobody reads.

Keep this backend anyway. It materializes the score matrix, so it is the
clearest statement of what MoBA means, and it needs no compiled dependency --
which makes it the one that always runs.
"""

import torch

from implementations.layout import (
    cu_seqlens_for,
    from_varlen,
    load_reference_module,
    to_varlen,
)
from implementations.base import register
from moba.config import MoBAConfig

_MODULE_ALIAS = "_moonshot_moba_naive"
_RELATIVE_PATH = "MoBA/moba/moba_naive.py"


def _load():
    """Returns the reference module, or raises with a legible reason."""
    return load_reference_module(_MODULE_ALIAS, _RELATIVE_PATH)


class MoonshotNaiveBackend:
    """MoonshotAI's naive MoBA, adapted from varlen to [B, S, H, D]."""

    name = "moonshot_naive"
    vendored = True
    computes_moba = True

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
            raise RuntimeError(f"moonshot_naive backend not available: {reason}")

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


register(MoonshotNaiveBackend())
