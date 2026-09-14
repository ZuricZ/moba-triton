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

    Searches upward for the repo root rather than counting parent directories.
    A hardcoded depth is correct only for one location of this file, and it
    fails *open*: a wrong root means `references/` is simply absent, so every
    vendored backend reports itself unavailable and its tests skip. That is a
    green suite with less coverage, which is the failure mode this project has
    now hit three times.

    Returns:
        Path to `references/`, overridable with MOBA_REFERENCES_DIR.

    Raises:
        RuntimeError: If no ancestor directory holds both `pyproject.toml` and
            `references/`.
    """
    override = os.environ.get("MOBA_REFERENCES_DIR")
    if override:
        return Path(override)
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").is_file() and (parent / "references").is_dir():
            return parent / "references"
    raise RuntimeError(
        f"no repo root above {here}: looked for a directory holding both "
        f"pyproject.toml and references/. Set MOBA_REFERENCES_DIR to override."
    )


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


def flash_attn_generation() -> tuple[int, str]:
    """Reports which flash-attn API generation is installed, if any.

    Two MoBA references call the *private* `_flash_attn_varlen_forward` and
    unpack its return tuple positionally, so they are pinned to opposite sides
    of a signature change made at 2.7:

        2.6.x   window_size=(left, right)     forward returns 8 values
        >= 2.7  window_size_left/right ints   forward returns 4

    `moonshot_efficient` unpacks eight and `triton_moba` unpacks four, so
    exactly one of them works against any given build. Checking that
    `flash_attn` merely imports is not enough -- it imports fine either way and
    then raises `ValueError: too many values to unpack` at call time, which
    presents as a wall of failing correctness tests rather than as one
    unavailable backend.

    The parameter names are inspected rather than the version string, because
    the signature is the thing that actually breaks and a fork or a patched
    build can carry either shape under any version number.

    Returns:
        A tuple of (generation, detail), where generation is 26 for the 2.6.x
        shape, 27 for the >= 2.7 shape, and 0 if flash-attn is absent or its
        signature matches neither. `detail` is a human-readable reason,
        suitable for an `is_available` message.
    """
    import inspect

    try:
        from flash_attn.flash_attn_interface import _flash_attn_varlen_forward
    except Exception as exc:                    # noqa: BLE001 - never raises
        return 0, f"flash-attn not importable ({type(exc).__name__}: {exc})"

    version = getattr(__import__("flash_attn"), "__version__", "unknown")

    try:
        params = set(inspect.signature(_flash_attn_varlen_forward).parameters)
    except (TypeError, ValueError):
        params = set()

    if "window_size" in params:
        return 26, f"flash-attn {version} has the 2.6.x signature"
    if "window_size_left" in params:
        return 27, f"flash-attn {version} has the >= 2.7 signature"

    # The signature is not always visible. From 2.7 flash-attn registers this
    # function as a torch.library custom op, and the resulting `CustomOpDef`
    # reports `(*args, **kwargs)` -- the real parameter names are behind the
    # wrapper. So the probe cannot see what it was built to see in exactly the
    # versions it most needs to identify, and falls back to the version string.
    #
    # Signature first and version second, rather than the other way round,
    # because the signature is the thing that actually breaks and survives a
    # fork renumbering itself. The fallback only runs when the signature is
    # opaque, and says so.
    major_minor = version.split(".")[:2]
    try:
        numeric = tuple(int(part) for part in major_minor)
    except ValueError:
        return 0, f"flash-attn {version}: signature hidden and version "                  f"unparseable"
    if numeric < (2, 7):
        return 26, f"flash-attn {version} (by version; signature is wrapped)"
    return 27, f"flash-attn {version} (by version; signature is wrapped)"
