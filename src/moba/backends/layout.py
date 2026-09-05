"""Helpers for adapting vendored reference implementations.

Two problems this module solves.

**Layout.** Every reference uses FlashAttention's `varlen` convention:
q/k/v packed as [total_tokens, H, D] with a `cu_seqlens` offset tensor,
rather than our [B, S, H, D]. For uniform sequence lengths the conversion is
a permute and a reshape.

**Name collision.** `references/MoBA/moba/` is a package named `moba` -- the
same name as ours. Putting it on sys.path would make `import moba` resolve by
path order, which is a bug whose behavior depends on how pytest was invoked.
We therefore load individual reference *files* by absolute path under
distinct module names, which bypasses package resolution entirely. As a bonus
it skips their `__init__.py`, so heavy optional imports (transformers) are
never executed.
"""

import importlib.util
import os
import sys
from pathlib import Path
from types import ModuleType

import torch


def references_dir() -> Path:
    """Returns the directory holding vendored reference checkouts.

    Returns:
        Path to `references/`, overridable with MOBA_REFERENCES_DIR.
    """
    override = os.environ.get("MOBA_REFERENCES_DIR")
    if override:
        return Path(override)
    # src/moba/backends/layout.py -> repo root is four parents up.
    return Path(__file__).resolve().parents[3] / "references"


_loaded: dict[str, ModuleType] = {}


def load_reference_module(alias: str, relative_path: str) -> ModuleType:
    """Loads a single reference source file as a standalone module.

    Args:
        alias: Module name to register it under. Must not collide with any
            real package name.
        relative_path: Path to the .py file, relative to `references/`.

    Returns:
        The imported module.

    Raises:
        FileNotFoundError: If the file is not present.
    """
    if alias in _loaded:
        return _loaded[alias]

    path = references_dir() / relative_path
    if not path.is_file():
        raise FileNotFoundError(path)

    spec = importlib.util.spec_from_file_location(alias, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot build a module spec for {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module
    spec.loader.exec_module(module)
    _loaded[alias] = module
    return module


def to_varlen(x: torch.Tensor) -> torch.Tensor:
    """Converts [B, S, H, D] to the varlen packing [B*S, H, D].

    A reshape, not a permute: token-major already puts the axes in this order,
    so no data moves. Under head-major this was a permute and a full copy.

    Args:
        x: Tensor shaped [B, S, H, D].

    Returns:
        Tensor shaped [B*S, H, D], contiguous whenever `x` is.
    """
    b, s, h, d = x.shape
    return x.reshape(b * s, h, d)


def from_varlen(x: torch.Tensor, batch: int, seq_len: int) -> torch.Tensor:
    """Converts varlen packing [B*S, H, D] back to [B, S, H, D].

    The inverse of `to_varlen`, and equally free.

    Args:
        x: Tensor shaped [B*S, H, D].
        batch: Batch size.
        seq_len: Sequence length (uniform across the batch).

    Returns:
        Tensor shaped [B, S, H, D].
    """
    _, h, d = x.shape
    return x.reshape(batch, seq_len, h, d)


def cu_seqlens_for(batch: int, seq_len: int, device: torch.device) -> torch.Tensor:
    """Builds the cumulative sequence-length tensor for uniform lengths.

    FlashAttention's varlen APIs take offsets rather than lengths: for batch=3
    and seq_len=4 this is [0, 4, 8, 12], so sequence i occupies
    [cu[i], cu[i+1]).

    Args:
        batch: Batch size.
        seq_len: Sequence length, uniform across the batch.
        device: Device to allocate on.

    Returns:
        int32 tensor of shape [batch + 1].
    """
    return torch.arange(
        0, (batch + 1) * seq_len, seq_len, device=device, dtype=torch.int32
    )
