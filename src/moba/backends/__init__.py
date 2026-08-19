"""Backend registry. Importing this package registers all built-in backends.

Registration happens as an import side effect, which is why the seemingly
unused imports below are load-bearing. The alternative -- an explicit
`REGISTRY = [NaiveTorchBackend(), ...]` list -- would centralize knowledge of
every backend in this file, so adding one would mean editing shared code. The
side-effect pattern keeps the design property that a new backend is exactly
one new file.
"""

from moba.backends.base import (
    Backend,
    all_backends,
    available_backends,
    get_backend,
    register,
    skipped_backends,
)

# Imported for side effect: each module registers its backend on import.
from moba.backends import naive_torch as _naive_torch  # noqa: F401
from moba.backends import sdpa_dense as _sdpa_dense  # noqa: F401
from moba.backends import triton_ours as _triton_ours  # noqa: F401

# Vendored references. These register unconditionally and report their own
# availability, so a missing checkout or an unbuilt extension shows up as a
# skip-with-reason rather than an ImportError at package import time.
from moba.backends import flash_moba as _flash_moba  # noqa: F401
from moba.backends import moonshot as _moonshot  # noqa: F401
from moba.backends import triton_moba as _triton_moba  # noqa: F401

__all__ = [
    "Backend",
    "all_backends",
    "available_backends",
    "get_backend",
    "register",
    "skipped_backends",
]
