"""A minimal FlashAttention-2 forward kernel, written to be read.

This is the smallest thing that is genuinely FA-2: tiled, never materializing
the [S, S] score matrix, using online softmax. Everything optional has been
removed -- no autotuning, no backward pass, no fp8, no variable-length
sequences, no GQA. Those belong in a production kernel and would only obscure
the mechanism here.

Its role in this project is twofold, and neither is "the base class the MoBA
kernel extends" -- the MoBA kernel is KV-block-major, so its grid, loop bounds
and output ownership all differ.

1. **Reference for the inner loop.** Load a K/V tile, score it, fold it into a
   running softmax, normalize once at the end. That sequence is unchanged in
   any FlashAttention-family kernel, MoBA's included. Only *which* tiles get
   visited and *which* queries are grouped change.
2. **A dense baseline we own.** The benchmark's dense roofline is torch SDPA,
   which is a fused CUDA kernel; this is the Triton-side comparison, so a
   MoBA/dense gap can be attributed to the algorithm rather than to the
   language.

   (An earlier draft listed this kernel as a candidate for a separate
   "local pass" over each query's own block. Decision D9 in the kernel spec
   merged that path into the unified gathered kernel, so no such pass exists.)

It also stands as the project's only independent check that Triton produces
correct attention on this machine -- that `tl.dot`, the fp32 accumulators and
the online softmax behave as expected on sm_86. When the MoBA kernel later
disagrees with the oracle, that whole class of suspect is already eliminated.

Layout assumption: q, k, v are contiguous [B, S, H, D]. Requiring contiguity
lets us index with a single flat offset instead of carrying four strides per
tensor, which removes a lot of noise from the pointer arithmetic.
"""

import torch
import triton
import triton.language as tl

from implementations.base import register
from moba.config import MoBAConfig
from moba.device import TENSOR_CORE_TILE


# -------------------------------------------------------------------------
# Device code
# -------------------------------------------------------------------------


@triton.jit
def fa2_fwd_kernel(
    Q, K, V, Out, Lse,
    S,
    sm_scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    D: tl.constexpr,
    H: tl.constexpr,
    CAUSAL: tl.constexpr,
    WRITE_LSE: tl.constexpr,
):
    """One program instance computes one tile of BLOCK_M output rows.

    The grid is 2-D: (query tiles, batch*heads). Every program is independent
    -- no communication, no atomics, no shared state. That independence is
    what makes attention easy to parallelize, and it is exactly what the MoBA
    inversion will give up.
    """
    pid_m = tl.program_id(0)      # which query tile
    pid_bh = tl.program_id(1)     # which (batch, head) pair

    # Token-major [B, S, H, D], so a head no longer owns a contiguous slab --
    # its rows are H*D apart, interleaved with the other heads. Offsetting the
    # base pointer once still lets the rest of the kernel pretend it is a
    # single-head problem; only the row stride changes.
    base = (pid_bh // H) * S * H * D + (pid_bh % H) * D

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)   # query rows, absolute
    offs_d = tl.arange(0, D)                           # feature columns

    # --- Load this program's queries, once, and keep them resident. ---------
    # Q stays in registers for the whole kernel. This is FA-2's central
    # bookkeeping choice: the query tile is loaded from HBM exactly once, and
    # every K/V tile streams past it.
    q_ptrs = Q + base + offs_m[:, None] * (H * D) + offs_d[None, :]
    q = tl.load(q_ptrs, mask=offs_m[:, None] < S, other=0.0)

    # --- Online softmax state, one entry per query row. ---------------------
    # m_i: running max of the logits seen so far. Needed for numerical
    #      stability -- exp() of a large positive number overflows.
    # l_i: running sum of exp(logit - m_i), the softmax denominator so far.
    # acc: running weighted sum of V, un-normalized.
    #
    # All three are float32 regardless of input dtype. The inputs may be
    # bf16, but accumulating thousands of terms in bf16 (8 mantissa bits)
    # loses precision badly; the tensor cores accumulate in fp32 anyway.
    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, D], dtype=tl.float32)

    # With causal masking, key positions beyond this tile's last query are all
    # masked, so we simply never load those tiles. This is where causal
    # attention gets its ~2x saving -- not from masking, but from not looking.
    hi = (pid_m + 1) * BLOCK_M if CAUSAL else S

    for start_n in range(0, hi, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)

        # --- Load one K tile into SRAM. ------------------------------------
        k = tl.load(
            K + base + offs_n[:, None] * (H * D) + offs_d[None, :],
            mask=offs_n[:, None] < S,
            other=0.0,
        )

        # --- Scores for ALL BLOCK_M queries against ALL BLOCK_N keys. ------
        # One K tile load feeds BLOCK_M queries. That amortization is the
        # entire performance argument of FlashAttention, and the assumption
        # MoBA's per-query routing breaks.
        qk = tl.dot(q, tl.trans(k)) * sm_scale

        # Mask before the max, so masked entries cannot win it.
        if CAUSAL:
            qk = tl.where(offs_m[:, None] >= offs_n[None, :], qk, float("-inf"))
        qk = tl.where(offs_n[None, :] < S, qk, float("-inf"))

        # --- The online softmax update. -----------------------------------
        # A new tile may contain a larger logit than anything seen so far. If
        # it does, every previously accumulated term was scaled by the wrong
        # exponent, so l_i and acc must be corrected by exp(m_old - m_new)
        # before the new tile is folded in. That correction factor is `alpha`.
        m_new = tl.maximum(m_i, tl.max(qk, axis=1))
        alpha = tl.exp(m_i - m_new)

        p = tl.exp(qk - m_new[:, None])

        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]

        v = tl.load(
            V + base + offs_n[:, None] * (H * D) + offs_d[None, :],
            mask=offs_n[:, None] < S,
            other=0.0,
        )
        acc += tl.dot(p.to(v.dtype), v)

        m_i = m_new

    # --- Normalize once, at the end. ---------------------------------------
    # The division by the softmax denominator is deferred out of the loop
    # entirely. Doing it per tile would be arithmetically equivalent and
    # strictly more work -- this is one of FA-2's "fewer non-matmul FLOPs"
    # improvements over FA-1.
    acc = acc / l_i[:, None]

    tl.store(
        Out + base + offs_m[:, None] * (H * D) + offs_d[None, :],
        acc.to(Out.dtype.element_ty),
        mask=offs_m[:, None] < S,
    )

    # The backward needs each row's log-sum-exp to rebuild the softmax without
    # storing it. `m_i + log(l_i)` is exactly that: p = exp(qk - lse) recovers
    # the normalized probabilities in one step, with no second max pass and no
    # [S, S] matrix ever existing. Written head-major [B, H, S] because it is
    # our own buffer -- see `moba.gqa` on the two layout conventions.
    if WRITE_LSE:
        tl.store(Lse + pid_bh * S + offs_m, m_i + tl.log(l_i),
                 mask=offs_m < S)


@triton.jit
def fa2_bwd_preprocess_kernel(O, DO, Delta, S, D: tl.constexpr,
                              H: tl.constexpr, BLOCK_M: tl.constexpr):
    """Computes `delta_i = rowsum(dO_i * O_i)`, one value per query.

    This is the term the softmax Jacobian contributes. For a row,
    `dS = P * (dP - delta)`, and `delta` is the same for every column of that
    row -- so it is computed once here rather than re-derived inside the main
    backward loop for every key tile.
    """
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    base = (pid_bh // H) * S * H * D + (pid_bh % H) * D
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, D)
    live = offs_m[:, None] < S

    o = tl.load(O + base + offs_m[:, None] * (H * D) + offs_d[None, :],
                mask=live, other=0.0).to(tl.float32)
    do = tl.load(DO + base + offs_m[:, None] * (H * D) + offs_d[None, :],
                 mask=live, other=0.0).to(tl.float32)
    tl.store(Delta + pid_bh * S + offs_m, tl.sum(o * do, axis=1),
             mask=offs_m < S)


@triton.jit
def fa2_bwd_kernel(
    Q, K, V, DO, DQ, DK, DV, Lse, Delta,
    S,
    sm_scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    D: tl.constexpr,
    H: tl.constexpr,
    CAUSAL: tl.constexpr,
):
    """One program owns one K/V tile and streams every query that sees it.

    The loop order is inverted relative to the forward, and for a reason worth
    naming: dK and dV accumulate over *queries*, so giving a program a K/V tile
    lets it hold those accumulators in registers and write them once. The
    forward makes the opposite choice for the same reason applied to `acc`.

    dQ does not fit that pattern -- it accumulates over *keys*, which are split
    across programs -- so it is accumulated with `tl.atomic_add` into a float32
    buffer. That is the simple option, and it is what makes this a *minimal*
    FA-2: the alternative is a second pass over the transpose, which is faster
    and is what `moba_bwd` does with its dQ partials and `moba_dq_reduce`.

    Nothing here is recomputed that the forward could have saved except the
    scores themselves, which is the whole trade: `S x S` never exists, so it is
    rebuilt tile by tile from Q, K and the stored log-sum-exp.
    """
    pid_n = tl.program_id(0)
    pid_bh = tl.program_id(1)

    base = (pid_bh // H) * S * H * D + (pid_bh % H) * D
    row = pid_bh * S                      # Lse/Delta are head-major [B, H, S]

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, D)
    kv_live = offs_n[:, None] < S

    k = tl.load(K + base + offs_n[:, None] * (H * D) + offs_d[None, :],
                mask=kv_live, other=0.0)
    v = tl.load(V + base + offs_n[:, None] * (H * D) + offs_d[None, :],
                mask=kv_live, other=0.0)

    dk = tl.zeros([BLOCK_N, D], dtype=tl.float32)
    dv = tl.zeros([BLOCK_N, D], dtype=tl.float32)

    # Causal: a key at position n is only seen by queries at m >= n, so start
    # at the query tile containing this key tile. Same saving as the forward's
    # `hi`, from the other end.
    lo = ((pid_n * BLOCK_N) // BLOCK_M) * BLOCK_M if CAUSAL else 0

    for start_m in range(lo, S, BLOCK_M):
        offs_m = start_m + tl.arange(0, BLOCK_M)
        q_live = offs_m[:, None] < S

        q = tl.load(Q + base + offs_m[:, None] * (H * D) + offs_d[None, :],
                    mask=q_live, other=0.0)
        do = tl.load(DO + base + offs_m[:, None] * (H * D) + offs_d[None, :],
                     mask=q_live, other=0.0)
        lse = tl.load(Lse + row + offs_m, mask=offs_m < S, other=0.0)
        delta = tl.load(Delta + row + offs_m, mask=offs_m < S, other=0.0)

        qk = tl.dot(q, tl.trans(k)) * sm_scale
        if CAUSAL:
            qk = tl.where(offs_m[:, None] >= offs_n[None, :], qk,
                          float("-inf"))
        qk = tl.where(offs_n[None, :] < S, qk, float("-inf"))

        # The stored lse already carries the normalizer, so this is the true
        # softmax probability -- no second reduction needed.
        p = tl.exp(qk - lse[:, None])

        dv += tl.dot(tl.trans(p).to(do.dtype), do)
        dp = tl.dot(do, tl.trans(v))
        ds = p * (dp - delta[:, None]) * sm_scale
        dk += tl.dot(tl.trans(ds).to(q.dtype), q)

        tl.atomic_add(DQ + base + offs_m[:, None] * (H * D) + offs_d[None, :],
                      tl.dot(ds.to(k.dtype), k), mask=q_live)

    tl.store(DK + base + offs_n[:, None] * (H * D) + offs_d[None, :],
             dk.to(DK.dtype.element_ty), mask=kv_live)
    tl.store(DV + base + offs_n[:, None] * (H * D) + offs_d[None, :],
             dv.to(DV.dtype.element_ty), mask=kv_live)


# -------------------------------------------------------------------------
# Launcher
# -------------------------------------------------------------------------


def fa2_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = True,
    block_m: int = 64,
    block_n: int = 64,
    return_lse: bool = False,
):
    """Runs the minimal FA-2 forward kernel.

    Args:
        q: Queries shaped [B, S, H, D], contiguous.
        k: Keys shaped [B, S, H, D], contiguous.
        v: Values shaped [B, S, H, D], contiguous.
        causal: Whether to apply causal masking.
        block_m: Query rows per program instance.
        block_n: Key rows per inner-loop step.

    Returns:
        Attention output shaped [B, S, H, D], same dtype as `q`.

    Raises:
        ValueError: If shapes disagree or the head dimension is unsupported.
    """
    if not (q.shape == k.shape == v.shape):
        raise ValueError(f"shape mismatch: {q.shape}, {k.shape}, {v.shape}")

    b, s, h, d = q.shape
    # Dense attention reads Q contiguously, so it is not subject to the
    # cache-line floor that `check_gathered_head_dim` imposes on the gathered
    # MoBA kernel. The only requirement here is that tl.dot can fill its
    # operands.
    if d & (d - 1) or d < TENSOR_CORE_TILE:
        raise ValueError(
            f"head_dim {d} unsupported; must be a power of two >= "
            f"{TENSOR_CORE_TILE}"
        )

    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    out = torch.empty_like(q)
    # Head-major [B, H, S]: ours, so it follows the buffer convention rather
    # than the token-major one the API boundary uses.
    lse = (torch.empty((b, h, s), device=q.device, dtype=torch.float32)
           if return_lse else out)

    # 2-D grid: one program per (query tile, batch-head).
    grid = (triton.cdiv(s, block_m), b * h)

    fa2_fwd_kernel[grid](
        q, k, v, out, lse,
        s,
        d ** -0.5,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        D=d,
        H=h,
        CAUSAL=causal,
        WRITE_LSE=return_lse,
    )
    return (out, lse) if return_lse else out


def fa2_backward(q, k, v, out, dout, lse, causal=True, block_m=64, block_n=64):
    """Gradients of the minimal FA-2 forward.

    Args:
        q, k, v: The forward's inputs, [B, S, H, D].
        out: The forward's output.
        dout: Incoming gradient, same shape as `out`.
        lse: Per-row log-sum-exp from the forward, [B, H, S] float32.
        causal: Must match the forward.
        block_m: Query rows per inner-loop step.
        block_n: Key rows per program.

    Returns:
        A tuple of (dq, dk, dv), each shaped like its input.
    """
    b, s, h, d = q.shape
    dout = dout.contiguous()

    # float32 because dQ is accumulated with atomics across key tiles, and
    # atomic_add on bf16 is both unsupported and would lose the accumulation
    # this exists to get right. Cast back once at the end.
    dq32 = torch.zeros((b, s, h, d), device=q.device, dtype=torch.float32)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)
    delta = torch.empty((b, h, s), device=q.device, dtype=torch.float32)

    fa2_bwd_preprocess_kernel[(triton.cdiv(s, block_m), b * h)](
        out, dout, delta, s, D=d, H=h, BLOCK_M=block_m)

    # The backward keeps more tiles live than the forward -- q, do, k, v, plus
    # float32 dk/dv accumulators -- so 64x64 overflows shared memory at
    # head_dim 128 (measured: 123,904 bytes against this card's 101,376).
    #
    # Rather than predict the compiler's allocation, ask it and step down. That
    # is the lesson `moba.kernels.tuning.PIPELINE_STAGES` records the hard way:
    # a conservative estimate silently costs speed, and shared-memory overflow
    # is raised at compile time, before any argument is touched, so a rejected
    # attempt has nothing to undo.
    for tile in [t for t in (block_n, 64, 32, 16) if t <= block_n]:
        try:
            fa2_bwd_kernel[(triton.cdiv(s, tile), b * h)](
                q, k, v, dout, dq32, dk, dv, lse, delta,
                s,
                d ** -0.5,
                BLOCK_M=tile,
                BLOCK_N=tile,
                D=d,
                H=h,
                CAUSAL=causal,
            )
            break
        except triton.runtime.errors.OutOfResources:
            dq32.zero_()          # a rejected launch ran nothing, but be exact
    else:
        raise ValueError(
            f"head_dim {d} does not fit shared memory even at the smallest "
            f"backward tile"
        )
    return dq32.to(q.dtype), dk, dv


class Fa2Function(torch.autograd.Function):
    """Differentiable minimal FA-2."""

    @staticmethod
    def forward(ctx, q, k, v, causal):
        out, lse = fa2_forward(q, k, v, causal=causal, return_lse=True)
        ctx.save_for_backward(q, k, v, out, lse)
        ctx.causal = causal
        return out

    @staticmethod
    def backward(ctx, dout):
        q, k, v, out, lse = ctx.saved_tensors
        dq, dk, dv = fa2_backward(q, k, v, out, dout, lse, causal=ctx.causal)
        return dq, dk, dv, None


def fa2_attention(q, k, v, causal=True):
    """Minimal FA-2 with gradient support.

    Args:
        q, k, v: [B, S, H, D], equal shapes, same dtype.
        causal: Whether to mask the future.

    Returns:
        Attention output shaped [B, S, H, D].
    """
    return Fa2Function.apply(q, k, v, causal)


# -------------------------------------------------------------------------
# Backend adapter
# -------------------------------------------------------------------------


# Dtypes with a tensor-core `tl.dot` path on Ampere. Same list as `ours.py`,
# kept local rather than shared: it describes what this kernel can feed to
# `tl.dot`, which is a property of the kernel, not of the registry.
TENSOR_CORE_DTYPES = (torch.bfloat16, torch.float16)


class Fa2MinimalBackend:
    """Our from-scratch FlashAttention-2 -- the second dense baseline.

    Registered so the suite reports it beside `sdpa_dense`. That pairing is
    the point: torch's SDPA is a hand-tuned production kernel and this is a
    teaching rebuild, so the gap between them is a measurement of our Triton
    on a problem with no sparsity in it at all. It is the floor under every
    MoBA number, and a MoBA speedup means less if the dense case is already
    far behind.

    Attends to everything, so `cfg`'s gate parameters are ignored by design,
    exactly as `sdpa_dense` ignores them.

    Has a backward: `forward_grad` routes through `fa2_attention`, so this
    appears on the backward plot too. That line is the point of it -- it is the
    only dense *Triton* backward in the comparison, so it separates "is our
    backward design slow" from "is our Triton slow" the way the forward line
    already does.

    No grouped-query path: `fa2_forward` requires q, k and v to have the same
    head count and raises otherwise. `is_available` cannot check that -- it
    sees a config and a GPU, never a shape -- so the check stays where the
    shape exists. Neither the suite nor the correctness tiers build unequal
    head counts for the dense baselines, so this has no consumer today.
    """

    name = "fa2_minimal"
    vendored = False
    computes_moba = False

    def is_available(self, cfg: MoBAConfig) -> tuple[bool, str]:
        if not torch.cuda.is_available():
            return False, "no CUDA device"
        if cfg.dtype not in TENSOR_CORE_DTYPES:
            return False, f"{cfg.dtype} has no tensor-core tl.dot path here"
        return True, ""

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cfg: MoBAConfig,
    ) -> torch.Tensor:
        """Computes dense causal attention, ignoring cfg's gate parameters."""
        return fa2_forward(q, k, v, causal=True)

    def forward_grad(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cfg: MoBAConfig,
    ) -> torch.Tensor:
        """Same computation as `forward`, but differentiable."""
        return fa2_attention(q, k, v, causal=True)


register(Fa2MinimalBackend())
