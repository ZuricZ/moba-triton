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

Layout assumption: q, k, v are contiguous [B, H, S, D]. Requiring contiguity
lets us index with a single flat offset instead of carrying four strides per
tensor, which removes a lot of noise from the pointer arithmetic.
"""

import torch
import triton
import triton.language as tl

from moba.device import TENSOR_CORE_TILE


@triton.jit
def _fa2_fwd(
    Q, K, V, Out,
    S,
    sm_scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    D: tl.constexpr,
    CAUSAL: tl.constexpr,
):
    """One program instance computes one tile of BLOCK_M output rows.

    The grid is 2-D: (query tiles, batch*heads). Every program is independent
    -- no communication, no atomics, no shared state. That independence is
    what makes attention easy to parallelize, and it is exactly what the MoBA
    inversion will give up.
    """
    pid_m = tl.program_id(0)      # which query tile
    pid_bh = tl.program_id(1)     # which (batch, head) pair

    # Each (batch, head) owns a contiguous [S, D] slab. Offsetting the base
    # pointer once means the rest of the kernel can pretend it is working on a
    # single-head problem.
    base = pid_bh * S * D

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)   # query rows, absolute
    offs_d = tl.arange(0, D)                           # feature columns

    # --- Load this program's queries, once, and keep them resident. ---------
    # Q stays in registers for the whole kernel. This is FA-2's central
    # bookkeeping choice: the query tile is loaded from HBM exactly once, and
    # every K/V tile streams past it.
    q_ptrs = Q + base + offs_m[:, None] * D + offs_d[None, :]
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
            K + base + offs_n[:, None] * D + offs_d[None, :],
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
            V + base + offs_n[:, None] * D + offs_d[None, :],
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
        Out + base + offs_m[:, None] * D + offs_d[None, :],
        acc.to(Out.dtype.element_ty),
        mask=offs_m[:, None] < S,
    )


def fa2_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = True,
    block_m: int = 64,
    block_n: int = 64,
) -> torch.Tensor:
    """Runs the minimal FA-2 forward kernel.

    Args:
        q: Queries shaped [B, H, S, D], contiguous.
        k: Keys shaped [B, H, S, D], contiguous.
        v: Values shaped [B, H, S, D], contiguous.
        causal: Whether to apply causal masking.
        block_m: Query rows per program instance.
        block_n: Key rows per inner-loop step.

    Returns:
        Attention output shaped [B, H, S, D], same dtype as `q`.

    Raises:
        ValueError: If shapes disagree or the head dimension is unsupported.
    """
    if not (q.shape == k.shape == v.shape):
        raise ValueError(f"shape mismatch: {q.shape}, {k.shape}, {v.shape}")

    b, h, s, d = q.shape
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

    # 2-D grid: one program per (query tile, batch-head).
    grid = (triton.cdiv(s, block_m), b * h)

    _fa2_fwd[grid](
        q, k, v, out,
        s,
        d ** -0.5,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        D=d,
        CAUSAL=causal,
    )
    return out
