"""MoBA (Mixture of Block Attention) kernels and reference implementations.

The pipeline, in the order the modules run:

    kernels.pooling   mean-pool each key block into one representative key
    kernels.gating    score queries against those representatives, keep top-k
    kernels.routing   invert that choice into the index the kernel walks
    kernels.moba_fwd  attend, one program per (KV block, batch, kv head)
    kernels.moba_bwd  the backward of the same

`moba_attention` runs all of it and is differentiable. `kernels.tuning` holds
the host-side launch policy the last three share; `naive` is the PyTorch
oracle every kernel is checked against.
"""

from moba.config import MoBAConfig
from moba.kernels.autograd import moba_attention

__all__ = ["MoBAConfig", "moba_attention"]
__version__ = "0.1.0"
