"""Autograd wiring for the MoBA kernels.

The forward and backward kernels are written and tested as plain functions, so
this module holds only what connects them to PyTorch: which tensors survive
into the backward, and which inputs receive gradient.

WHAT IS SAVED, AND WHY EACH IS UNAVOIDABLE.

* `q`, `k`, `v` -- the backward recomputes the softmax rather than storing it,
  which is the central FlashAttention trade. Recomputation needs the inputs.
* `out` -- only for `delta = rowsum(dO * O)`. Cheap and already materialized.
* `lse` -- each query's global normalizer, in log2 units. It cannot be
  reconstructed in the backward: a program there sees one key sub-tile, and the
  normalizer is a reduction over every block the query selected. The forward's
  combine is the one place those maxima meet.
* `part_m` -- the forward's per-slot maxima, kept purely as a written-marker so
  the `dQ` partial buffer can be left uninitialized. A factor of `head_dim`
  smaller than the buffer it guards.
* `index` -- the CSC routing. Not a tensor, so it rides on the context object.
  Rebuilding it in the backward would be wasteful and, worse, could route
  differently if anything about the inputs changed.

WHAT RECEIVES NO GRADIENT: the gate. MoBA's routing is a hard top-k, and
Moonshot's reference treats it as non-differentiable -- `gate_top_k_idx` is used
only as an index and no gate score is saved for backward. So `q` receives
gradient through the attention scores and not through its role in choosing
blocks, and the backward reuses the forward's selection as a fixed sparsity
pattern.
"""

import torch

from moba.config import MoBAConfig
from moba.kernels.index import CSCIndex
from moba.kernels.index_fused import build_csc_fused
from moba.kernels.moba_bwd import moba_backward
from moba.kernels.moba_fwd import (
    _validate,
    attend_partials,
    combine_partials,
)


class _MoBAAttention(torch.autograd.Function):
    """Differentiable MoBA attention over the Triton kernels."""

    @staticmethod
    def forward(ctx, q, k, v, cfg, index, causal, cu_seqlens, max_seqlen):
        """Runs the three forward stages and saves the backward's inputs."""
        if index is not None:
            cu_seqlens, max_seqlen = index.cu_seqlens, index.max_seqlen
        _validate(q, k, v, cfg, cu_seqlens, max_seqlen)
        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
        if index is None:
            index = build_csc_fused(q, k, cfg, cu_seqlens=cu_seqlens,
                                    max_seqlen=max_seqlen)

        part_m, part_l, part_acc = attend_partials(
            q, k, v, cfg, index, causal=causal
        )
        out, lse = combine_partials(
            part_m, part_l, part_acc, q.dtype, return_lse=True
        )

        # `part_l` and `part_acc` are deliberately not saved. They are the
        # largest buffers in the design -- 64 MB at S=8192 -- and the backward
        # recomputes what it needs from q, k and v instead. `part_m` survives
        # only because it is the written-marker, and it is `head_dim` times
        # smaller.
        ctx.save_for_backward(q, k, v, out, lse, part_m)
        ctx.cfg, ctx.index, ctx.causal = cfg, index, causal
        return out

    @staticmethod
    def backward(ctx, dout):
        """Returns gradients for q, k and v; none for the routing."""
        q, k, v, out, lse, part_m = ctx.saved_tensors
        dq, dk, dv = moba_backward(
            dout, q, k, v, out, lse, part_m, ctx.cfg, ctx.index,
            causal=ctx.causal,
        )
        # cfg, index, causal and the packing metadata are not differentiable.
        return dq, dk, dv, None, None, None, None, None


def moba_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cfg: MoBAConfig,
    index: CSCIndex | None = None,
    causal: bool = True,
    cu_seqlens: torch.Tensor | None = None,
    max_seqlen: int | None = None,
) -> torch.Tensor:
    """Computes MoBA attention with gradient support.

    Args:
        q: Queries shaped [B, S, H, D].
        k: Keys shaped [B, S, H_kv, D], where `H_kv` divides `H`. Fewer
            key/value heads than query heads is grouped-query attention; the
            kernel shares a K/V tile by owning it for the whole group, never by
            materializing copies.
        v: Values, same shape as `k`.
        cfg: Configuration.
        index: Precomputed CSC index; built with the fused device-side gate
            when omitted. Passing one holds routing fixed while inputs vary.
        causal: Whether to apply causal masking within the query's own block.
        cu_seqlens: Cumulative token counts, [num_sequences + 1] int32. When
            given, the inputs hold one packed batch shaped [1, T, H, D] and each
            sequence attends -- and receives gradient -- independently. Ignored
            when `index` is supplied, which carries its own.
        max_seqlen: Longest sequence, required with `cu_seqlens`. Deriving it
            would cost a host synchronization, so it is an argument, as in
            FlashAttention's `max_seqlen_q`.

    Returns:
        Attention output shaped [B, S, H, D], same dtype as `q`. `k` and `v`
        receive gradients with their own `H_kv` heads, each already summed over
        the query heads that shared it.
    """
    return _MoBAAttention.apply(q, k, v, cfg, index, causal, cu_seqlens,
                                max_seqlen)
