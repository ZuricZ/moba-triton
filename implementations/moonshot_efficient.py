"""MoonshotAI/MoBA, efficient path -- the authors' own fast implementation.

This is the backend that makes a "faster than MoonshotAI" claim mean something.
Its sibling, `moonshot_naive`, materializes the score matrix and exists to be
read; this one fuses the same algorithm onto FlashAttention and is what a user
of their library actually runs. Comparing against only the naive path and
calling the result "MoonshotAI MoBA" flattered us by roughly two orders of
magnitude, which is why both are now registered under explicit names.

Structurally it is the closest thing to us in the reference set: Triton kernels
for the gate, FlashAttention for the attention itself, and an online-softmax
merge to recombine. The same three stages our pipeline has. Where we differ is
that our attention is also Triton, so a gap between us and this backend is a
statement about the attention kernel rather than about the algorithm.

WHY IT IS OFTEN UNAVAILABLE. It imports `flash_attn` at module scope, and that
is a compiled CUDA extension pinned to a torch ABI. `Triton-MoBA` needs the
same one, so a single flash-attn 2.6.3 build brings up two backends rather than
one -- worth knowing before deciding it is not worth an hour of compile time.
See `docs/reference-comparison.md`.

Loaded by path rather than imported, for the reason `moonshot_naive` records:
their package is also called `moba` and would shadow ours. Loading the single
file skips their `__init__.py`, so `transformers` is never pulled in -- but
`flash_attn` and `einops` are imported by this module itself and really are
required.
"""

import torch

from implementations.base import register
from implementations.layout import (
    cu_seqlens_for,
    from_varlen,
    load_reference_module,
    to_varlen,
)
from moba.config import MoBAConfig

_MODULE_ALIAS = "_moonshot_moba_efficient"
_RELATIVE_PATH = "MoBA/moba/moba_efficient.py"


def _load():
    """Returns the reference module, or raises with a legible reason."""
    return load_reference_module(_MODULE_ALIAS, _RELATIVE_PATH)


class MoonshotEfficientBackend:
    """MoonshotAI's fused MoBA, adapted from varlen to [B, S, H, D]."""

    name = "moonshot_efficient"
    vendored = True
    computes_moba = True

    def is_available(self, cfg: MoBAConfig) -> tuple[bool, str]:
        """Reports availability without raising, per the registry contract.

        The import is attempted rather than probed with `find_spec`, because
        `flash_attn` can be present and still fail to load: the extension is
        built against a specific torch ABI, and a mismatch surfaces only when
        the `.so` is actually opened.
        """
        if not torch.cuda.is_available():
            return False, "no CUDA device"
        try:
            _load()
        except FileNotFoundError as exc:
            return False, f"not vendored: run scripts/fetch_references.sh ({exc})"
        except ImportError as exc:
            return False, (
                f"requires flash-attn 2.6.3 and einops ({exc}); one build also "
                "brings up triton_moba"
            )
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
        """Runs their fused path on token-major inputs.

        Args:
            q: Queries shaped [B, S, H, D].
            k: Keys, same shape.
            v: Values, same shape.
            cfg: Configuration; `block_size` is their `moba_chunk_size` and
                `top_k` their `moba_topk`.

        Returns:
            Output shaped [B, S, H, D].

        Raises:
            RuntimeError: If the backend is unavailable.
        """
        ok, reason = self.is_available(cfg)
        if not ok:
            raise RuntimeError(f"moonshot_efficient backend not available: {reason}")

        module = _load()
        b, s, _, _ = q.shape        # token-major
        out = module.moba_attn_varlen(
            to_varlen(q),
            to_varlen(k),
            to_varlen(v),
            cu_seqlens=cu_seqlens_for(b, s, q.device),
            max_seqlen=s,
            moba_chunk_size=cfg.block_size,
            moba_topk=cfg.top_k,
        )
        return from_varlen(out, b, s)


register(MoonshotEfficientBackend())
