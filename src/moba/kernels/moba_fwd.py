"""KV-block-major MoBA forward.

One program owns one KV block, loads it into SRAM once, and streams every query
that selected it past that single load. This is decision D6: the loop order is
inverted relative to FlashAttention-2 because MoBA's per-query routing destroys
the amortization a query-major tile depends on. Measured query-major
utilization at the reference configuration (block_size 128, top_k 8) is 0.121
once num_blocks reaches 128 -- 88% of tensor-core work discarded, which at that
block size means S=16384 and up (`docs/divergence-measurements.md`).

Three things about this kernel differ from a dense FA-2 inner loop, all of them
simplifications:

1. **The online-softmax rescale is relocated, not removed.** In FA-2 one
   program folds many KV tiles into one query, so the `exp(m_old - m_new)`
   correction is loop-carried inside the kernel. Here a program sees each of
   its queries against exactly *one* block, so there is no sequence of tiles
   to fold and no loop-carried rescale in this kernel.

   The query's full row still spans several blocks, computed by different
   programs, and those partials still require the identical correction --
   `exp(m_local - m_global)` -- before they can be summed. That is
   `combine_partials`, and it is mandatory (D10). Same algebra, applied across
   programs instead of across loop iterations.
2. **No branch on selection.** Every query the program holds selected this
   block, so "does this query need this block?" is uniform across the warp.
3. **Causality is one predicated select** that is a no-op for any block wholly
   in the past, and compiles away entirely when CAUSAL is False (D9).

What it gives up is output ownership: a query's contributions come from several
programs and must be merged afterwards (D10).
"""

import torch
import triton
import triton.language as tl
from triton.runtime.errors import OutOfResources

from moba.config import MoBAConfig
from moba.device import (
    TENSOR_CORE_TILE,
    WARP_SIZE,
    check_gathered_head_dim,
    round_up_to_tensor_core_tile,
    shared_memory_budget,
)
from moba.heads import head_group_size
from moba.kernels.index import CSCIndex
from moba.packing import validate_cu_seqlens
from moba.kernels.index_fused import build_csc_fused


@triton.jit
def _moba_attn_fwd(
    Q, K, V, RowIdx, HeadIdx, SlotIdx, ColOffsets, ColNnz, Order, CuSeqlens,
    PartM, PartL, PartAcc,
    S, T, TOKENS, sm_scale, num_blocks,
    BLOCK_SIZE: tl.constexpr, GROUP: tl.constexpr, D: tl.constexpr,
    K_SLOTS: tl.constexpr, H_Q: tl.constexpr, H_KV: tl.constexpr,
    CAUSAL: tl.constexpr, VARLEN: tl.constexpr,
):
    """One program per (KV block, batch, key/value head).

    Writes un-normalized partials; nothing here is divided by a softmax
    denominator, because this program does not know the other programs'
    running maxima.

    The grid follows what a program *owns*, which is a block of keys -- so it
    is sized by key/value heads, and grouped-query attention shrinks it. The
    queries streamed through it come from every query head sharing this
    key/value head, so the query side is addressed by an explicit
    (head, position) pair carried in the column rather than by the program id.

    Packed mode changes only where a block's keys start and where the query
    tensors are based. Blocks are cut per sequence, so the range is computed
    from `cu_seqlens` using the batch the grid already carries. Row indices are
    packed token indices, which is why the causal comparison below needs no
    change: within a sequence, packed order *is* position order, and a query is
    never offered another sequence's block. That last part is guaranteed by the
    gate, not here -- see the varlen plan's D-VL-7 and
    `test_no_selected_block_crosses_a_sequence_boundary`.
    """
    pid_col = tl.program_id(0)
    pid_bh = tl.program_id(1)

    # Keys are what got shared, so `pid_bh` is a kv head and `kv_base` below is
    # unchanged in form. Queries are wider: a batch holds H_Q of them, indexed
    # by the head each entry names. At H_Q == H_KV this reduces to `pid_bh`
    # exactly, which is why the multi-head path is untouched.
    pid_b = pid_bh // H_KV
    if VARLEN:
        # The query tensors carry a pinned batch of 1, so there is no batch
        # stride to add; the sequence enters only through the key range below.
        q_base = 0
        q_part_base = 0
        part_span = TOKENS
    else:
        # Unchanged by the move: the batch stride is H_Q*S*D head-major and
        # S*H_Q*D token-major, which is the same number.
        q_base = pid_b * H_Q * S * D
        q_part_base = pid_b * H_Q * S * K_SLOTS
        part_span = S

    # Longest-first dispatch (D8). Only the block id changes -- results are
    # written to query addresses, which never depended on the program id, so
    # there is nothing to undo afterwards.
    blk = tl.load(Order + pid_bh * num_blocks + pid_col)
    n = tl.load(ColNnz + pid_bh * num_blocks + blk)
    if n == 0:
        return

    lo = tl.load(ColOffsets + pid_bh * (num_blocks + 1) + blk)

    offs_d = tl.arange(0, D)

    if VARLEN:
        bos = tl.load(CuSeqlens + pid_b)
        eos = tl.load(CuSeqlens + pid_b + 1)
        lo_k = bos + blk * BLOCK_SIZE
        hi_k = tl.minimum(lo_k + BLOCK_SIZE, eos)
        kv_base = (pid_bh % H_KV) * D
    else:
        lo_k = blk * BLOCK_SIZE
        hi_k = S
        kv_base = pid_b * S * H_KV * D + (pid_bh % H_KV) * D

    key_pos = lo_k + tl.arange(0, BLOCK_SIZE)
    key_ok = key_pos < hi_k

    # The single HBM trip this whole design exists to amortize.
    k = tl.load(K + kv_base + key_pos[:, None] * (H_KV * D) + offs_d[None, :],
                mask=key_ok[:, None], other=0.0)
    v = tl.load(V + kv_base + key_pos[:, None] * (H_KV * D) + offs_d[None, :],
                mask=key_ok[:, None], other=0.0)

    for start in range(0, n, GROUP):
        offs = start + tl.arange(0, GROUP)
        live = offs < n

        rows = tl.load(RowIdx + pid_bh * T + lo + offs, mask=live, other=0)
        heads = tl.load(HeadIdx + pid_bh * T + lo + offs, mask=live, other=0)
        slots = tl.load(SlotIdx + pid_bh * T + lo + offs, mask=live, other=0)

        # One extra [GROUP] int32 gather -- 128 bytes against a 16 KiB K/V tile
        # -- and one multiply-add. That is the entire inner-loop cost of
        # grouped-query support: the tile shapes, register count and shared
        # memory are untouched, because a longer column means more trips around
        # this loop, not a wider tile. See
        # `docs/learning/03-gqa-and-the-column-merge.md`.
        #
        # `rows` deliberately stays a raw position, so the causal comparison
        # below needs no change and no runtime integer modulo appears here.
        #
        # Two different things used to be one expression. `part_row` says
        # *which query this is*, addressing our own head-major partial buffers;
        # the load below says *where that query lives*, in the caller's
        # token-major tensor. Head-major let them coincide. Re-fusing them
        # would look like a head permutation in the output.
        part_row = heads * part_span + rows

        # Gathered rows (D7). One row is D*itemsize bytes -- exactly one
        # 128-byte cache line at D=64 bf16 -- so this costs the same
        # transaction count as a contiguous read. Measured at 1.03x.
        #
        # Token-major: tokens are H_Q*D apart and heads D apart. Neither stride
        # contains the sequence length, which is why this one expression serves
        # the packed and padded cases alike.
        q = tl.load(Q + q_base + rows[:, None] * (H_Q * D)
                    + heads[:, None] * D + offs_d[None, :],
                    mask=live[:, None], other=0.0)

        # Masks are *added* into a zeroed accumulator before the matmul, not
        # selected over its result afterwards. Both forms compute the same
        # thing -- adding -inf and selecting -inf are the same when the other
        # operand is finite -- but this one does not depend on the dot, so the
        # compiler can issue the mask work while the MMA is still in flight.
        # Borrowed from FSA's `forward_kernel_opt`, which builds `qk` this way.
        # Measured 1.5-2% off this kernel, winning all six paired runs at
        # S=4096 and S=8192.
        #
        # Hoisting the masks behind scalar guards instead -- skipping the tail
        # mask on non-tail blocks, and the causal mask when no query in the
        # group can reach this block -- was also measured. Alone it is worth
        # 0.7%; combined with the additive form it is 22% *slower*, because the
        # guard's reduction over `rows` reintroduces a dependency on the loaded
        # indices that the additive form exists to avoid.
        qk = tl.zeros((GROUP, BLOCK_SIZE), dtype=tl.float32)
        qk += tl.where(key_ok[None, :], 0.0, float("-inf"))
        if CAUSAL:
            # A no-op for every block wholly in the past; trims only the
            # query's own block (D9). Predicated, not a branch, so it does not
            # reintroduce warp divergence.
            qk += tl.where(key_pos[None, :] <= rows[:, None], 0.0,
                           float("-inf"))
        qk += tl.dot(q, tl.trans(k)) * sm_scale

        # This is the online-softmax state, computed against this block only.
        # `m` and `l` are LOCAL -- they know nothing of the other blocks this
        # query selected. They are written out precisely so `combine_partials`
        # can rescale them by exp(m_local - m_global) and sum them. Dividing
        # acc by l here would discard the information that merge needs.
        #
        # Every row has at least one live entry: a query only reaches a block
        # it selected, and it can only select its own block if that block
        # starts at or before it. So `m` is never -inf and exp() is safe.
        # Base 2, not base e. The SFU has a native 2^x instruction and none
        # for e^x, so `tl.exp` compiles to `exp2(x * 1.4427)` -- one extra
        # multiply on every element of a [GROUP, BLOCK_SIZE] tile. `sm_scale`
        # already carries that constant (see `moba_forward`), so the multiply
        # is folded into work the kernel was doing anyway and disappears.
        # Measured 7% off this kernel at S=4096 and S=8192.
        #
        # `p`, `l` and `acc` are unchanged in value: exp2(log2(e) * x) is
        # exp(x) exactly. Only `m` changes units, and `combine_partials`
        # consumes it with the matching `exp2`.
        m = tl.max(qk, axis=1)
        p = tl.exp2(qk - m[:, None])
        l = tl.sum(p, axis=1)
        acc = tl.dot(p.to(v.dtype), v)

        # Fixed-slot destination (D10). Task 1 guarantees each (query, slot)
        # pair is written by exactly one program, so these scattered stores
        # never collide.
        dst = part_row * K_SLOTS + slots
        tl.store(PartM + q_part_base + dst, m, mask=live)
        tl.store(PartL + q_part_base + dst, l, mask=live)
        tl.store(
            PartAcc + q_part_base * D + dst[:, None] * D + offs_d[None, :],
            acc,
            mask=live[:, None],
        )


# log2(e). The softmax runs in base 2 because the SFU has a native 2^x
# instruction and none for e^x, so `tl.exp` would compile to `exp2(x * 1.4427)`
# -- one extra multiply on every element of the score tile. Folding the
# constant into the scale that already multiplies those scores makes it free.
LOG2_E = 1.4426950408889634


# Absolute ceiling on the inner-loop chunk height, before the shared-memory
# budget is applied. Beyond this, register pressure bites even when shared
# memory allows more: measured 255 registers with 8 spills at D=64,
# BLOCK_SIZE=128, GROUP=128 (`docs/learning/02`).
MAX_GROUP = 128

# Pipelining depth Triton allocates for the gathered q tile.
#
# This is the *optimistic* depth, and it is deliberately not an upper bound.
# The depth is not a constant: measured on this card it is 2 at head_dim 128
# and 4 at head_dim 64, so any single number is either wrong or wasteful.
#
# The history is worth keeping, because the same mistake has now been made
# twice in this codebase. This was 3, "so the estimate is an upper bound and
# never predicts a launch that then fails". The launch it wrongly predicted
# would fail was GROUP=64 at head_dim 128, which Triton compiles to 98,816
# bytes against a 101,376 budget and which runs 12% faster than the GROUP=32
# the over-estimate forced. `check_gate_fits` did the same thing to the gate.
# A conservative guess about the compiler is still a guess, and both guesses
# cost speed silently.
#
# So the estimate no longer decides anything on its own. It proposes the
# largest chunk height that *might* fit, and `attend_partials` asks the
# compiler, stepping down if the launch is rejected. Shared-memory overflow is
# raised at compile time, before any argument is touched, so a rejected attempt
# has no effect to undo.
PIPELINE_STAGES = 2


def estimate_shared_memory(
    block_size: int, group: int, head_dim: int, itemsize: int
) -> int:
    """Estimates shared memory for one attention program, in bytes.

    The K and V tiles load once outside the inner loop, so they are
    single-buffered at `block_size x head_dim` each. The gathered q tile loads
    every iteration and Triton pipelines it, costing `PIPELINE_STAGES` copies
    of `group x head_dim`.

    This is a *lower* bound on what Triton allocates, not an upper one, so it
    proposes rather than decides; see `PIPELINE_STAGES` and
    `attend_partials`.

    Args:
        block_size: KV block size.
        group: Inner-loop chunk height.
        head_dim: Head dimension.
        itemsize: Bytes per element of the input dtype.

    Returns:
        Estimated bytes of shared memory.
    """
    return (2 * block_size + PIPELINE_STAGES * group) * head_dim * itemsize


def choose_group(
    cfg: MoBAConfig, head_dim: int, itemsize: int, device=None
) -> int:
    """Picks the largest inner-loop chunk height that fits in shared memory.

    `group` is how many queries are processed per iteration -- NOT how many
    selected this block. The loop iterates as many times as needed either way,
    and K/V load once *outside* it, so a larger group buys no extra
    amortization; it only trims loop overhead. What it costs is shared memory,
    because the gathered q tile is pipelined.

    This is why an earlier `GROUP = top_k * block_size` was wrong: it sized a
    tile from a consumer count. Uncapped it asked for 112 KiB against a 99 KiB
    limit, and shared-memory overflow fails the launch outright rather than
    degrading.

    Public rather than private because `bench/load_balance.py` must simulate
    the tile height the kernel actually uses. It previously hardcoded 32 and
    silently went stale.

    Args:
        cfg: Configuration supplying `block_size` and `top_k`.
        head_dim: Head dimension.
        itemsize: Bytes per element of the input dtype.
        device: CUDA device, for the shared-memory budget.

    Returns:
        The largest multiple of `TENSOR_CORE_TILE` that may fit. Whether it
        actually compiles is settled by `attend_partials`, not here.

    Raises:
        ValueError: If no chunk height down to `TENSOR_CORE_TILE` fits, which
            means the K and V tiles alone exceed shared memory.
    """
    return group_candidates(cfg, head_dim, itemsize, device)[0]


def group_candidates(
    cfg: MoBAConfig, head_dim: int, itemsize: int, device=None
) -> list[int]:
    """Returns chunk heights to try, largest first.

    `estimate_shared_memory` is a lower bound on what Triton allocates, so the
    first entry may not actually compile. `attend_partials` walks this list and
    lets the compiler reject what does not fit, which is the only authority on
    the question -- see `PIPELINE_STAGES` for what modelling it cost.

    Args:
        cfg: Configuration supplying `block_size` and `top_k`.
        head_dim: Head dimension.
        itemsize: Bytes per element of the input dtype.
        device: CUDA device, for the shared-memory budget.

    Returns:
        Multiples of `TENSOR_CORE_TILE` in descending order, non-empty.

    Raises:
        ValueError: If not even `TENSOR_CORE_TILE` can fit, which means the K
            and V tiles alone exceed shared memory.
    """
    budget = shared_memory_budget(device)

    # A power of two, not merely a multiple of TENSOR_CORE_TILE. `tl.arange`
    # requires a power-of-two range, and the kernel calls `tl.arange(0, GROUP)`.
    # Rounding to a multiple of 16 was enough for every configuration tested
    # until now only by accident: block_size 32 and above with top_k 4 and above
    # gives a product of at least 128, which MAX_GROUP caps to a power of two.
    # top_k 3 at block_size 16 gives 48, and the launch fails to compile.
    group = min(MAX_GROUP,
                max(TENSOR_CORE_TILE,
                    triton.next_power_of_2(cfg.top_k * cfg.block_size)))

    out = []
    while group >= TENSOR_CORE_TILE:
        if estimate_shared_memory(
            cfg.block_size, group, head_dim, itemsize
        ) <= budget:
            out.append(group)
        group //= 2

    if not out:
        needed = estimate_shared_memory(
            cfg.block_size, TENSOR_CORE_TILE, head_dim, itemsize
        )
        raise ValueError(
            f"block_size {cfg.block_size} with head_dim {head_dim} needs "
            f"{needed} bytes of shared memory at the minimum chunk height, "
            f"exceeding this GPU's {budget}. Reduce block_size."
        )
    return out


# Queries per combine program.
#
# This pass runs at the memory bus's speed and nothing else. At head_dim 128
# with 12 slots and S=8192 it moves ~221 MB in 1313 us, which is 168 GB/s --
# against 163 GB/s measured for a bare `torch.zeros` on the same card. There is
# no arrangement of tiles or warps that beats the wire the data travels on, and
# the measurement says so: every tile from 1 to 64 lands within 1% of the same
# time (1309-1328 us). Only moving *fewer* bytes helps, which is what
# `partials_dtype` does.
#
# So the tile size is a robustness choice, not a speed one, and small wins.
# A 64-query tile holds a 32 KB accumulator entirely in registers and sits on
# the 255-register ceiling, kept off it only by computing the right warp count;
# an 8-query tile uses 168 and cannot spill. This mirrors FSA's `reduce_kernel`,
# which runs one query per program for the same reason. The tile shape we
# started with was inherited from FlashAttention-2, where a wide query tile is
# forced by `tl.dot` needing a matrix to fill a tensor core. There is no
# `tl.dot` here, so nothing forced it.
#
# Eight rather than one: equally fast, still 4096 programs at S=8192 -- ample
# for 20 SMs -- and it avoids the degenerate single-element vector shapes that
# Triton has handled inconsistently across versions.
COMBINE_BLOCK_M = 8

# Accumulator elements one thread may hold before the compiler spills. Measured
# on sm_86: at 32 elements per thread the kernel uses at most 255 registers with
# zero spills across every (head_dim, slot count) combination tested; at 64 it
# spills up to 220 values and runs 58% slower (2073 us against 1311 us at
# head_dim 128, 12 slots, S=4096). Spilling is silent -- unlike shared-memory
# overflow it does not fail the launch -- so this bound is enforced rather than
# discovered.
#
# Retained as a safety net even though the default tile no longer comes close
# to the limit: head_dim 64 at tile 16 still spilled 4 values on one warp, so
# the rule catches cases the default does not cover.
COMBINE_ELEMENTS_PER_THREAD = 32

# Triton's default is 4. Four is the floor because fewer warps leaves too
# little to hide memory latency with. Eight is the ceiling because warps stop
# helping past it: at block_m 128, head_dim 128 raising 8 to 16 halved
# registers per thread but still spilled 106 values and ran slower (1998 us
# against 1726). Warps divide the accumulator; they do not divide the
# per-iteration working set of the unrolled slot loop. Past one doubling of
# headroom the tile itself has to shrink.
MIN_COMBINE_WARPS = 4
MAX_COMBINE_WARPS = 8

# The largest accumulator one program may hold, in elements.
MAX_COMBINE_TILE_ELEMENTS = (
    WARP_SIZE * COMBINE_ELEMENTS_PER_THREAD * MAX_COMBINE_WARPS
)


def choose_combine_tile(head_dim: int, block_m: int | None = None) -> tuple[int, int]:
    """Picks a combine tile and warp count that do not spill registers.

    More warps do not change how much data a program holds -- they change how
    many threads hold it, so registers per thread fall while memory traffic is
    untouched. That trade is free here only because this pass is memory-bound;
    in the attention kernel the warp count also re-tiles `tl.dot`, so it is not
    an independent knob there.

    Args:
        head_dim: Head dimension.
        block_m: Queries per program. Derived from the register budget when
            omitted; validated against it when given.

    Returns:
        A tuple of (block_m, num_warps).

    Raises:
        ValueError: If an explicit `block_m` cannot avoid spilling, or if no
            tile down to `TENSOR_CORE_TILE` fits the budget.
    """
    budget = MAX_COMBINE_TILE_ELEMENTS
    if block_m is not None:
        if block_m * head_dim > budget:
            raise ValueError(
                f"block_m {block_m} at head_dim {head_dim} needs "
                f"{block_m * head_dim} accumulator elements, above the "
                f"{budget} that fit without spilling registers"
            )
    else:
        block_m = COMBINE_BLOCK_M
        while block_m > TENSOR_CORE_TILE and block_m * head_dim > budget:
            block_m //= 2
        if block_m * head_dim > budget:
            raise ValueError(
                f"head_dim {head_dim} cannot be combined without spilling "
                f"even at block_m {block_m}"
            )

    per_thread = WARP_SIZE * COMBINE_ELEMENTS_PER_THREAD
    warps = (block_m * head_dim + per_thread - 1) // per_thread
    return block_m, max(MIN_COMBINE_WARPS, min(MAX_COMBINE_WARPS, warps))


@triton.jit
def _moba_combine(
    PartM, PartL, PartAcc, Out, Lse,
    S,
    K_SLOTS: tl.constexpr, D: tl.constexpr, BLOCK_M: tl.constexpr,
    H_Q: tl.constexpr, WRITE_LSE: tl.constexpr,
):
    """Merges each query's partial softmaxes into its final output.

    This is FlashAttention-2's `alpha` correction applied *across programs*
    rather than across loop iterations. Every partial was exponentiated against
    its own local maximum, so the partials are on incompatible scales and
    cannot simply be summed -- that omission is the failure that produces an
    800x error with no crash and entirely plausible-looking output.

    Two passes over the slots, not one. The online single-pass form carries a
    running maximum and rescales the accumulator whenever it grows, costing two
    D-wide operations per slot; two passes cost one, and re-read only a
    BLOCK_M-length vector of maxima. FA-2 cannot make that trade because its
    maxima do not exist until the scores are computed and a second pass would
    mean repeating the QK matmul. Here the maxima are already in HBM.
    """
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    live = offs_m < S
    offs_d = tl.arange(0, D)

    m_base = PartM + pid_bh * S * K_SLOTS + offs_m * K_SLOTS
    l_base = PartL + pid_bh * S * K_SLOTS + offs_m * K_SLOTS
    a_base = (PartAcc + pid_bh * S * K_SLOTS * D
              + offs_m[:, None] * K_SLOTS * D + offs_d[None, :])

    # K_SLOTS is a constexpr, so `static_range` unrolls both loops at compile
    # time. It is used rather than loading a [BLOCK_M, K_SLOTS] tile because
    # `tl.arange` requires a power-of-two length, while `max_slots` is whatever
    # the index build produced -- 3, 5 and 12 are all ordinary values.
    m_g = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    for slot in tl.static_range(K_SLOTS):
        m_g = tl.maximum(
            m_g, tl.load(m_base + slot, mask=live, other=float("-inf"))
        )

    l = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, D], dtype=tl.float32)
    for slot in tl.static_range(K_SLOTS):
        m_i = tl.load(m_base + slot, mask=live, other=float("-inf"))
        l_i = tl.load(l_base + slot, mask=live, other=0.0)
        a_i = tl.load(a_base + slot * D, mask=live[:, None], other=0.0)

        # A slot no program wrote still holds its initial m = -inf, so its
        # weight is exp(-inf - m_g) = 0. That kills its contribution
        # arithmetically -- except against uninitialized memory, where 0 * NaN
        # is NaN. Selecting on the marker closes that hole, which is what lets
        # the caller skip zeroing the accumulator entirely: `tl.where` picks a
        # branch rather than evaluating both, so a NaN in the discarded operand
        # cannot leak. That saves writing the largest buffer in the design
        # twice -- 128 MB at S=8192, ~790 us of pure bandwidth.
        #
        # `part_m` is still filled, because it is the marker. It is a factor of
        # D smaller (0.5 MB against 128 MB), so filling it is not worth
        # avoiding.
        written = m_i > float("-inf")
        w = tl.exp2(m_i - m_g)
        l += tl.where(written, l_i * w, 0.0)
        acc += tl.where(written[:, None], a_i * w[:, None], 0.0)

    # Not guarded against l == 0. Every query selects at least its own block,
    # so a zero denominator means the index is wrong, and a NaN propagating to
    # the output is the loud failure that wants to happen. Substituting a safe
    # value here would hide an index bug behind a plausible number.
    out = acc / l[:, None]

    # The partials above are ours and head-major; `Out` crosses the API
    # boundary and is token-major, so `pid_bh` has to split into a batch and a
    # head. `Lse` below is ours and does not.
    tl.store(Out + (pid_bh // H_Q) * S * H_Q * D + (pid_bh % H_Q) * D
             + offs_m[:, None] * (H_Q * D) + offs_d[None, :],
             out.to(Out.dtype.element_ty), mask=live[:, None])

    if WRITE_LSE:
        # The backward pass recomputes P without storing it, and to do that it
        # needs each query's *global* normalizer -- the same `softmax_lse` that
        # Moonshot saves and hands to flash-attn's backward. It exists here and
        # nowhere else: the attention kernel knows only per-block maxima, and
        # this is the one place they are reduced across a query's blocks.
        #
        # Kept in log2 units, matching `m_g`, so the backward's `exp2(qk - lse)`
        # needs no conversion. Recovering natural units is a multiply by
        # ln 2 that no consumer currently wants.
        tl.store(Lse + pid_bh * S + offs_m, m_g + tl.log2(l), mask=live)


def combine_partials(
    part_m: torch.Tensor,
    part_l: torch.Tensor,
    part_acc: torch.Tensor,
    out_dtype: torch.dtype,
    block_m: int | None = None,
    return_lse: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Merges per-slot partial softmaxes into the final attention output.

    Args:
        part_m: Per-slot maxima in log2 units, [B, H, S, K] fp32, as
            `attend_partials` writes them.
        part_l: Per-slot denominators, [B, H, S, K] fp32.
        part_acc: Per-slot un-normalized outputs, [B, H, S, K, D] fp32.
        out_dtype: Dtype of the returned tensor.
        block_m: Queries per program. Derived from the register budget when
            omitted, which is what callers should do.
        return_lse: Also return each query's global log-sum-exp, in log2 units.
            The backward pass needs it to recompute the softmax without having
            stored it, and this is the only place a query's per-block maxima
            are reduced together.

    Returns:
        Attention output shaped [B, S, H, D], or a tuple of it and the
        log-sum-exp shaped [B, H, S] when `return_lse` is set.

    Raises:
        ValueError: If the three buffers disagree on shape, or if `block_m`
            would spill registers.
    """
    b, h, s, slots = part_m.shape
    d = part_acc.shape[-1]
    if part_l.shape != part_m.shape:
        raise ValueError(
            f"shape mismatch: part_l {tuple(part_l.shape)} against part_m "
            f"{tuple(part_m.shape)}"
        )
    if part_acc.shape != (b, h, s, slots, d):
        raise ValueError(
            f"shape mismatch: part_acc {tuple(part_acc.shape)} against "
            f"{(b, h, s, slots, d)}"
        )

    part_m, part_l, part_acc = (
        t.contiguous() for t in (part_m, part_l, part_acc)
    )
    out = torch.empty((b, s, h, d), device=part_m.device, dtype=out_dtype)
    lse = (torch.empty((b, h, s), device=part_m.device, dtype=torch.float32)
           if return_lse else out)
    block_m, warps = choose_combine_tile(d, block_m)

    _moba_combine[(triton.cdiv(s, block_m), b * h)](
        part_m, part_l, part_acc, out, lse, s,
        K_SLOTS=slots, D=d, BLOCK_M=block_m, H_Q=h, WRITE_LSE=return_lse,
        num_warps=warps,
    )
    return (out, lse) if return_lse else out


def _validate(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
              cfg: MoBAConfig,
              cu_seqlens: torch.Tensor | None = None,
              max_seqlen: int | None = None) -> None:
    """Checks shapes and configuration before any launch.

    Args:
        q: Queries shaped [B, S, H, D].
        k: Keys shaped [B, S, H_kv, D], where `H_kv` divides `H`. Fewer
            key/value heads than query heads is grouped-query attention.
        v: Values, same shape as `k`.
        cfg: Configuration.

    Raises:
        ValueError: If shapes disagree, if the head counts do not divide, or
            if head_dim or block_size is unsupported.
    """
    if k.shape != v.shape:
        raise ValueError(f"key/value shape mismatch: {k.shape}, {v.shape}")
    # Token-major: batch, sequence and head_dim must agree; only the head
    # count may differ, and only by a whole group.
    if (q.shape[0] != k.shape[0] or q.shape[1] != k.shape[1]
            or q.shape[3] != k.shape[3]):
        raise ValueError(f"shape mismatch: {q.shape}, {k.shape}, {v.shape}")

    # Grouped-query attention is a shape property, not a configuration one:
    # the group size is H_q // H_kv and there is nowhere else it could be
    # stated. Raises if the counts do not divide.
    head_group_size(q.shape[2], k.shape[2])

    if cu_seqlens is not None:
        if q.shape[0] != 1:
            raise ValueError(
                f"packed input must have batch 1, got {q.shape[0]}; every "
                f"sequence lives on the token axis and cu_seqlens says where"
            )
        if max_seqlen is None:
            raise ValueError("max_seqlen is required when cu_seqlens is given")
        validate_cu_seqlens(cu_seqlens, total_tokens=q.shape[1],
                            max_seqlen=max_seqlen)
    ok, reason = check_gathered_head_dim(q.shape[-1], q.element_size())
    if not ok:
        raise ValueError(reason)
    if cfg.block_size < TENSOR_CORE_TILE:
        raise ValueError(
            f"block_size {cfg.block_size} below the tl.dot minimum of "
            f"{TENSOR_CORE_TILE}"
        )


def empty_partials(
    shape: tuple[int, int, int, int],
    slots: int,
    device,
    partials_dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Allocates partial buffers without initializing them.

    Separate from `attend_partials` so a caller can pay the allocation once
    and reuse it. `attend_partials` resets whatever it is given, so uninitialized
    memory here is safe.

    The argument is the *query* shape, token-major, because every caller has a
    query to hand. The buffers themselves are head-major: they are ours, no
    consumer outside this module reads them, and `part_row` in the attention
    kernel addresses them as `heads * S + position`. So this function is where
    the two conventions meet, and it transposes the axes rather than the data.

    Args:
        shape: The query shape, (batch, seq_len, heads, head_dim).
        slots: Partial slots per query.
        device: CUDA device.
        partials_dtype: Storage dtype for the accumulator. Callers that want
            the default should pass `q.dtype`; this function has no query to
            read it from, so it does not guess.

    Returns:
        A tuple of (part_m, part_l, part_acc), uninitialized and head-major.
    """
    b, s, h, d = shape
    return (
        torch.empty((b, h, s, slots), device=device, dtype=torch.float32),
        torch.empty((b, h, s, slots), device=device, dtype=torch.float32),
        torch.empty((b, h, s, slots, d), device=device, dtype=partials_dtype),
    )


# Chunk heights the compiler has already accepted, keyed by everything
# that can change the allocation. Populated by `attend_partials` on the
# first successful launch of each configuration.
_GROUP_CACHE: dict[tuple, int] = {}


def attend_partials(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cfg: MoBAConfig,
    index: CSCIndex,
    causal: bool = True,
    partials_dtype: torch.dtype | None = None,
    partials: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Runs the attention stage, returning un-merged per-slot partials.

    Separate from `moba_forward` so the three stages -- index, attention,
    combine -- can be timed independently. Composing them in `moba_forward`
    rather than inlining costs nothing at runtime and makes the per-stage
    benchmark measure the same code path production uses.

    `part_acc` is the largest single term in the design's memory traffic and,
    since the gate stopped dominating, the largest single term in the whole
    forward pass. Profiled at S=8192: 63.4 MB of the attention kernel's 104 MB,
    against 31.7 MB for the gathered queries and 8.4 MB for all the K and V
    loads the KV-major inversion exists to amortize. The inversion worked well
    enough that its own bookkeeping became the bottleneck.

    So the default is the compute dtype, not float32. Storing partials at the
    same width as the inputs halves that 63.4 MB and is worth ~22% end to end.
    What it costs is judged by `moba.tolerance`, not by a threshold picked to
    pass: the policy asks whether our error stays within 2x of what the same
    algorithm in the same dtype already costs, so narrowing storage to the
    input's own precision has to earn its place against that bound.

    The maxima and denominators stay float32 regardless. They are a factor of
    D smaller, so narrowing them saves nothing measurable while costing the
    precision the whole merge pivots on.

    Args:
        q: Queries shaped [B, S, H, D], contiguous.
        k: Keys shaped [B, S, H_kv, D], contiguous, where `H_kv` divides `H`.
        v: Values, same shape as `k`.
        cfg: Configuration.
        index: Precomputed CSC index.
        causal: Whether to apply causal masking within the query's own block.
        partials_dtype: Storage dtype for `part_acc`. Defaults to `q.dtype`;
            pass `torch.float32` to keep the wider intermediate. Ignored when
            `partials` is supplied.
        partials: Buffers to reuse, from a previous call or `empty_partials`.
            Fresh allocation is not free -- on Windows WDDM a `cudaMalloc`
            goes through the display driver and costs milliseconds -- so a
            caller running the kernel repeatedly should allocate once and pass
            the buffers back in. They are reset here regardless, because a slot
            no program writes must read as -inf.

    Returns:
        A tuple of (part_m, part_l, part_acc). `part_m` is in log2
        units; `combine_partials` expects that and no other consumer
        exists.
    """
    b, s, h_q, d = q.shape
    h_kv = k.shape[2]
    slots = index.max_slots
    varlen = index.cu_seqlens is not None
    n_seq = index.col_nnz.shape[0]

    if partials_dtype is None:
        partials_dtype = q.dtype
    if partials is None:
        partials = empty_partials(q.shape, slots, q.device, partials_dtype)
    part_m, part_l, part_acc = partials

    # Only the marker is initialized. A slot no program writes keeps m = -inf,
    # which is how `combine_partials` knows to skip it; `part_l` and `part_acc`
    # are never read for such a slot, so their contents do not matter.
    #
    # This is worth the coupling. `part_acc` is the largest buffer in the
    # design -- 128 MB at S=8192, head_dim 128 -- and zeroing it means writing
    # it once before the kernel writes it again, at ~790 us of pure memory
    # bandwidth. `part_m` is a factor of D smaller, so filling it costs
    # nothing measurable.
    part_m.fill_(float("-inf"))

    # Largest chunk height first, letting the compiler reject what will not
    # fit. `estimate_shared_memory` is a lower bound, so the head of this list
    # is a proposal; Triton raises OutOfResources at compile time, before any
    # argument is read, so a rejected attempt leaves nothing to undo. The
    # winner is cached, making this a one-time cost per configuration.
    #
    # The head group is in the key even though it changes no allocation: a
    # stale entry would be very hard to diagnose, and the key is free.
    key = (cfg.block_size, d, q.element_size(), slots, causal, h_q // h_kv,
           varlen, torch.cuda.get_device_capability(q.device))
    cached = _GROUP_CACHE.get(key)
    candidates = ([cached] if cached is not None
                  else group_candidates(cfg, d, q.element_size(), q.device))

    # One program per (KV block, batch, kv head). Grouped-query attention
    # shrinks this grid by the group size and lengthens each program's column
    # by the same factor -- identical total work, `group` times fewer K and V
    # loads, which is the whole gain.
    for group in candidates:
        try:
            _moba_attn_fwd[(index.num_blocks, n_seq * h_kv)](
                q, k, v,
                index.row_indices, index.head_indices, index.slot_of_pair,
                index.col_offsets, index.col_nnz, index.order,
                index.cu_seqlens if varlen else q,
                part_m, part_l, part_acc,
                s, index.row_indices.shape[-1], s,
                d ** -0.5 * LOG2_E,
                index.num_blocks,
                BLOCK_SIZE=cfg.block_size, GROUP=group, D=d,
                K_SLOTS=slots, H_Q=h_q, H_KV=h_kv, CAUSAL=causal,
                VARLEN=varlen,
            )
        except OutOfResources:
            continue
        _GROUP_CACHE[key] = group
        return part_m, part_l, part_acc

    raise ValueError(
        f"no chunk height in {candidates} compiles for block_size "
        f"{cfg.block_size} at head_dim {d}; every one exceeded this GPU's "
        f"shared memory"
    )


def moba_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cfg: MoBAConfig,
    index: CSCIndex | None = None,
    causal: bool = True,
    partials_dtype: torch.dtype | None = None,
    cu_seqlens: torch.Tensor | None = None,
    max_seqlen: int | None = None,
) -> torch.Tensor:
    """Computes MoBA attention with the KV-block-major kernel.

    Args:
        q: Queries shaped [B, S, H, D], contiguous.
        k: Keys shaped [B, S, H_kv, D], contiguous, where `H_kv` divides `H`.
        v: Values, same shape as `k`.
        cfg: Configuration.
        index: Precomputed CSC index; built from `q` and `k` when omitted
            with the fused device-side gate. Passing one explicitly holds
            routing fixed while inputs vary, which is what the invariant tests
            need.
        causal: Whether to apply causal masking within the query's own block.
        partials_dtype: Storage dtype for the per-slot accumulators. Defaults
            to `q.dtype`, which halves the largest memory term in the pass;
            see `attend_partials`.
        cu_seqlens: Cumulative token counts, [num_sequences + 1] int32. When
            given, the inputs hold one packed batch shaped [1, T, H, D] and each
            sequence attends independently. Ignored when `index` is supplied,
            which carries its own.
        max_seqlen: Longest sequence, required with `cu_seqlens`. Deriving it
            would cost a host synchronization, so it is an argument -- the same
            reason FlashAttention takes `max_seqlen_q`.

    Returns:
        Attention output shaped [B, S, H, D], same dtype as `q`.

    Raises:
        ValueError: If shapes disagree, or head_dim or block_size is
            unsupported.
    """
    if index is not None:
        cu_seqlens, max_seqlen = index.cu_seqlens, index.max_seqlen
    _validate(q, k, v, cfg, cu_seqlens, max_seqlen)
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    if index is None:
        index = build_csc_fused(q, k, cfg, cu_seqlens=cu_seqlens,
                                max_seqlen=max_seqlen)

    part_m, part_l, part_acc = attend_partials(
        q, k, v, cfg, index, causal=causal, partials_dtype=partials_dtype
    )
    return combine_partials(part_m, part_l, part_acc, q.dtype)
