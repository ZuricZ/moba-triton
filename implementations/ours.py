"""Our KV-block-major Triton kernel, behind the common Backend interface.

Registering here is what puts the kernel into the Tier-2 correctness harness
and the benchmark suite without either of them learning its name. Because
`vendored` is False, `tests/test_backends.py` *asserts* on its accuracy rather
than merely reporting divergence -- ours is held to the oracle, references are
only compared against it (D1).

The one judgement call in this file is how much `is_available` can honestly
promise. It sees a `MoBAConfig` and a GPU, not a tensor shape, so it can rule
out a wrong architecture, a dtype with no tensor-core path, or a block size
below the MMA atom -- but not a head dimension, which arrives only with the
call. That check stays in `moba_forward`, where the shape exists.
"""

import torch

from implementations.base import register
from moba.config import MoBAConfig
from moba.device import MIN_CAPABILITY, TENSOR_CORE_TILE, supports_dtype
from moba.kernels.moba_fwd import moba_forward
from moba.kernels.routing import build_csc, csc_to_mask

# Dtypes with a tensor-core `tl.dot` path on Ampere. `supports_dtype` is the
# broader question of what the *architecture* can represent -- it admits fp32
# and fp64, which this kernel cannot feed to `tl.dot` at all. Both checks are
# needed: one is about the GPU, this one is about the kernel.
TENSOR_CORE_DTYPES = (torch.bfloat16, torch.float16)


class OursBackend:
    """Adapter for `moba.kernels.moba_fwd.moba_forward`."""

    name = "ours"
    vendored = False

    def is_available(self, cfg: MoBAConfig) -> tuple[bool, str]:
        """Reports whether this backend can run `cfg` on this machine.

        Never raises, per the protocol: every rejection below is a return
        value, so a caller deciding whether to skip work does not need a
        try/except around the predicate.

        Args:
            cfg: The configuration to check.

        Returns:
            A tuple of (available, reason). `reason` is empty when available.
        """
        if not torch.cuda.is_available():
            return False, "no CUDA device"

        capability = torch.cuda.get_device_capability(0)
        if capability < MIN_CAPABILITY:
            return False, (
                f"needs sm_{MIN_CAPABILITY[0]}{MIN_CAPABILITY[1]}+, found "
                f"sm_{capability[0]}{capability[1]}"
            )

        ok, reason = supports_dtype(cfg.dtype, capability)
        if not ok:
            return False, reason
        if cfg.dtype not in TENSOR_CORE_DTYPES:
            return False, (
                f"dtype {cfg.dtype} has no tensor-core tl.dot path; this "
                "kernel computes in bfloat16 or float16"
            )

        # Ampere's MMA atom is 16x8x16, so a KV tile narrower than this cannot
        # fill a tensor core -- `tl.dot` rejects the shape outright rather
        # than running slowly.
        if cfg.block_size < TENSOR_CORE_TILE:
            return False, (
                f"block_size {cfg.block_size} is below the tl.dot minimum of "
                f"{TENSOR_CORE_TILE}"
            )

        return True, ""

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cfg: MoBAConfig,
    ) -> torch.Tensor:
        """Computes MoBA attention with our kernel.

        The CSC index is rebuilt on every call rather than cached. That is
        deliberate for now: the gate is still PyTorch (D11), and Task 6 exists
        to price it. A cache here would hide the cost the measurement is
        supposed to expose.

        Args:
            q: Queries shaped [B, S, H, D].
            k: Keys shaped [B, S, H, D].
            v: Values shaped [B, S, H, D].
            cfg: Configuration.

        Returns:
            Output shaped [B, S, H, D], same dtype as `q`.

        Raises:
            RuntimeError: If this backend is not available for `cfg`.
            ValueError: If the shapes or head dimension are unsupported.
        """
        ok, reason = self.is_available(cfg)
        if not ok:
            raise RuntimeError(f"ours backend not available: {reason}")
        return moba_forward(q, k, v, cfg)

    def selection(
        self, q: torch.Tensor, k: torch.Tensor, cfg: MoBAConfig
    ) -> torch.Tensor:
        """Exposes routing so it is checked exactly, separate from output.

        Round-tripping the CSC index back to a dense mask -- rather than
        calling `moba_selection` directly -- is what makes this check worth
        anything. The kernel reads the index, not the mask, so the index is
        what has to be right. When a Triton gate replaces the PyTorch one
        (D11), this becomes a genuine routing check with no edit here.

        Args:
            q: Queries shaped [B, S, H, D].
            k: Keys shaped [B, S, H, D].
            cfg: Configuration.

        Returns:
            Boolean selection mask shaped [B, H, S, num_blocks].
        """
        return csc_to_mask(build_csc(q, k, cfg), q.shape[1])


register(OursBackend())
