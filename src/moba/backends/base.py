"""The Backend protocol and registry.

Every implementation of MoBA -- ours, vendored, or baseline -- is reachable
through this one interface, so neither the correctness harness nor the
benchmark suite ever special-cases a particular implementation. Adding a new
one means writing one file, not editing the harness.

The design's one unusual choice is that `is_available` returns a *reason*
string rather than a bare bool. A backend missing from a results table must be
able to say why. A silently short table is worse than no table, because it
looks complete.
"""

from typing import Protocol, runtime_checkable

import torch

from moba.config import MoBAConfig


@runtime_checkable
class Backend(Protocol):
    """One MoBA implementation, uniformly callable.

    Attributes:
        name: Registry key, unique.
        vendored: True for third-party implementations we did not write.
            These are compared against the oracle and their divergences are
            *reported*, never asserted -- per design decision D1, a reference
            disagreeing with us is a finding to investigate, not a test
            failure. Ours is the spec; theirs is evidence.
    """

    name: str
    vendored: bool = False

    def is_available(self, cfg: MoBAConfig) -> tuple[bool, str]:
        """Reports whether this backend can run `cfg` on this machine.

        Implementations must never raise. A missing import, an unsupported GPU
        architecture, or an unsupported dtype is a return value here, not an
        exception: callers use this to decide whether to skip work, and a
        throwing predicate pushes every call site toward `except: pass`.

        Args:
            cfg: The configuration to check.

        Returns:
            A tuple of (available, reason). `reason` is empty when available.
        """
        ...

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cfg: MoBAConfig,
    ) -> torch.Tensor:
        """Computes MoBA attention.

        Args:
            q: Queries shaped [B, H, S, D].
            k: Keys shaped [B, H, S, D].
            v: Values shaped [B, H, S, D].
            cfg: Configuration.

        Returns:
            Output shaped [B, H, S, D], same dtype as `q`.
        """
        ...


_REGISTRY: dict[str, Backend] = {}


def register(backend: Backend) -> Backend:
    """Adds a backend to the global registry.

    Args:
        backend: The backend instance to register.

    Returns:
        The same backend, so this can be used inline.

    Raises:
        ValueError: If a backend with the same name is already registered.
            Overwriting silently would make benchmark results depend on module
            import order, which is not something anyone should have to debug.
    """
    if backend.name in _REGISTRY:
        raise ValueError(f"Backend {backend.name!r} is already registered")
    _REGISTRY[backend.name] = backend
    return backend


def all_backends() -> list[Backend]:
    """Returns every registered backend, available or not."""
    return list(_REGISTRY.values())


def get_backend(name: str) -> Backend:
    """Looks up a backend by name.

    Args:
        name: Registered backend name.

    Returns:
        The backend.

    Raises:
        KeyError: If no backend is registered under `name`. The message lists
            the registered names, since the usual cause is a typo or a module
            that was never imported.
    """
    if name not in _REGISTRY:
        raise KeyError(f"No backend named {name!r}; have {sorted(_REGISTRY)}")
    return _REGISTRY[name]


def available_backends(cfg: MoBAConfig) -> list[Backend]:
    """Returns the backends that can run `cfg` on this machine.

    Args:
        cfg: The configuration to check.

    Returns:
        Backends whose `is_available` reports True.
    """
    return [b for b in _REGISTRY.values() if b.is_available(cfg)[0]]


def skipped_backends(cfg: MoBAConfig) -> list[tuple[str, str]]:
    """Returns (name, reason) for each backend that cannot run `cfg`.

    Args:
        cfg: The configuration to check.

    Returns:
        A list of (name, reason) pairs, one per unavailable backend.
    """
    out = []
    for backend in _REGISTRY.values():
        ok, reason = backend.is_available(cfg)
        if not ok:
            out.append((backend.name, reason))
    return out
