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

import os

import torch
import triton
import triton.language as tl
import triton.testing

from moba.config import MoBAConfig
from moba.device import (
    TENSOR_CORE_TILE,
    check_gathered_head_dim,
    shared_memory_budget,
)
from moba.gqa import head_group_size
from moba.kernels.routing import CSCIndex, build_csc_fused
from moba.kernels.tuning import (
    LOG2_E,
    choose_combine_tile,
    estimate_shared_memory,
    launch_policy,
)
from moba.varlen import validate_cu_seqlens


# -------------------------------------------------------------------------
# Device code
# -------------------------------------------------------------------------


# Chunk heights offered to the autotuner, largest first. Not a claim that any
# of them fits: `estimate_shared_memory` is a lower bound on what the compiler
# allocates, so the pruner below drops the obviously impossible and Triton's
# own OutOfResources handling scores whatever is left but still fails as
# infinite time. Powers of two because the kernel calls `tl.arange(0, GROUP)`.
AUTOTUNE_GROUPS = (128, 64, 32, 16)

# `num_warps` is NOT tuned, and that is a measured decision rather than an
# oversight. Measured on sm_86 at S=8192, block_size 128, top_k 8, D=64,
# attend pass only:
#
#     GROUP 128, warps 2 -> 3.146 ms      GROUP 64, warps 2 -> 0.961 ms
#     GROUP 128, warps 4 -> 0.873 ms      GROUP 64, warps 4 -> 0.707 ms
#     GROUP 128, warps 8 -> 0.787 ms      GROUP 64, warps 8 -> 1.422 ms
#
# The axes interact hard -- the best warp count at one height is the worst at
# the other, across a 4.5x spread -- but Triton's default of 4 is already
# optimal at the height that wins, so tuning it as well bought 0.7%. What it
# cost was every configuration compiling three kernels instead of one, and the
# kernel test file went from 30 s to 500 s: at twelve distinct shapes, that is
# 144 compilations, and compilation rather than benchmarking is what dominates.
#
# So the tuned axis is the one that pays. Widening to `num_warps=w for w in
# (2, 4, 8)` is a one-line change in the configs below, and is worth making on
# hardware where the default is not already the answer -- an A100 has five
# times the SMs and no reason to agree with this card. Do it on evidence.
AUTOTUNE_WARPS = (4,)


def _prune_by_shared_memory(configs, named_args, **kwargs):
    """Drops chunk heights that cannot fit, before anything is compiled.

    Triton would score an over-large config as infinite time anyway, so this is
    an optimisation rather than a correctness measure -- but compiling to find
    out costs seconds per configuration, and the estimate is free.

    Deliberately keeps anything the estimate merely *doubts*: the estimate is a
    lower bound, so a config it allows may still be rejected by the compiler,
    and that is the compiler's call to make. See `PIPELINE_STAGES` for what
    modelling this too confidently cost.

    Args:
        configs: Every `triton.Config` offered.
        named_args: Positional kernel arguments, unused here.
        **kwargs: The constexprs, which is where `BLOCK_SIZE` and `D` arrive.

    Returns:
        The surviving configs, never empty -- if the estimate rejects every
        one, the smallest is kept so the compiler produces the error rather
        than the autotuner.
    """
    del named_args
    block_size, head_dim = kwargs["BLOCK_SIZE"], kwargs["D"]
    itemsize = 2 if kwargs.get("ITEMSIZE") is None else kwargs["ITEMSIZE"]
    budget = shared_memory_budget()
    kept = [c for c in configs
            if estimate_shared_memory(block_size, c.kwargs["GROUP"], head_dim,
                                      itemsize) <= budget]
    if kept:
        return kept
    return [min(configs, key=lambda c: c.kwargs["GROUP"])]


def _best_of_three(fn, quantiles=None, **kwargs):
    """Benchmarks `fn` three times and keeps the fastest run.

    Handed to `triton.autotune` as its `do_bench`, replacing the single
    measurement per config it takes by default.

    WHY ONE MEASUREMENT IS NOT ENOUGH HERE. The autotuner walks its configs in
    order, and a slow one leaves the GPU hot for whatever follows: on this
    laptop `GROUP=128, num_warps=2` runs at 4 ms a rep, so it holds the card at
    full load for about a second immediately before the `GROUP=64` rows are
    timed. Those then measured 1.04 ms against 0.71 timed on a settled card --
    enough to change which config wins, and to lose 5% end to end.

    Timing error is one-sided: nothing makes a kernel finish faster than it
    can, so the minimum of several runs is the least contaminated sample rather
    than an optimistic pick. The same argument `bench/compare.py` records.

    This is timing hygiene, not a benchmark harness -- `src/moba` may not
    import from `bench/`, and the thermal protocol that belongs there stays
    there.

    Args:
        fn: Callable to time.
        quantiles: Passed through to `triton.testing.do_bench`.
        **kwargs: Ditto.

    Returns:
        Whatever `do_bench` returns, from the fastest of three runs.
    """
    runs = [triton.testing.do_bench(fn, quantiles=quantiles, **kwargs)
            for _ in range(3)]
    if quantiles is None:
        return min(runs)
    return min(runs, key=lambda r: r[0])


# Table by default, autotuner on MOBA_AUTOTUNE=1. The autotuner below is still
# the thing that decides these numbers -- it just does so in
# `bench/tune_table.py` on a machine someone is watching, rather than on the
# first launch of whatever shape a caller happens to bring. See
# `moba/kernels/tuned_configs.py` for why a serving path must never tune, and
# why pruning the autotuner to a single config would not have achieved it.
@launch_policy("fwd", triton.autotune(
    configs=[triton.Config({"GROUP": group}, num_warps=warps)
             for group in AUTOTUNE_GROUPS for warps in AUTOTUNE_WARPS],
    # Every constexpr that changes the *shape of the work* is here. The
    # autotuner's cache is keyed on these argument values plus the dtypes of
    # tensor arguments and nothing else, so a name left out means two
    # configurations share one tuned result -- which is sometimes exactly what
    # is wanted.
    #
    # `CAUSAL` and `VARLEN` are deliberately absent. They change which branches
    # the kernel compiles, not how much shared memory a tile needs or how much
    # work a program does, so they cannot change which height is fastest by
    # enough to matter. Including them cost twice over: it doubled the number
    # of shapes to tune, and -- because chunk height is observable at ~1e-4 --
    # it let the packed and unpacked paths settle on different heights, which
    # breaks the bitwise equality `test_packed_forward_under_gqa` asserts
    # between a packed batch and the same sequences run one at a time. That
    # equality is the check that packing composes with head sharing rather than
    # merely coexisting, and it is worth more than tuning the two apart.
    key=["BLOCK_SIZE", "D", "K_SLOTS", "H_Q", "H_KV"],
    prune_configs_by={"early_config_prune": _prune_by_shared_memory},
    # No `restore_value`. The obvious candidate is `PartM`, which the launcher
    # fills with -inf before the kernel runs and which the autotuner then runs
    # against many times off that single fill -- but the kernel only ever
    # `tl.store`s to it and never loads it, so repeated launches are idempotent
    # and there is nothing to restore.
    #
    # Declaring it anyway was not free. `restore_value` installs a pre-hook
    # that clones the tensor and a post-hook that copies it back, both *inside*
    # the timed region, and the timings came out wrong in a way that changed
    # the winner: every GROUP=64 config measured about 2.6x slower than the
    # same config timed directly, while the GROUP=128 rows agreed within 5%.
    # A benchmark harness has to measure the kernel and not itself.
    # Persisting the winner to disk is off by default, and that is the
    # opposite of what it looks like it should be.
    #
    # It is genuinely attractive: tuning is paid once per machine instead of
    # once per process, and a second test run of one file went from 59 s to
    # 7.5 s. It is also the one property the hand-rolled search this replaces
    # could not offer.
    #
    # But a cached choice is only as good as the conditions of the single run
    # that made it, and those are not controllable. Measured on this card, same
    # code, same machine:
    #
    #     tuned while the test suite held the GPU -> GROUP=128, 1.579 ms
    #     tuned on a quiet box                    -> GROUP=64,  1.384 ms
    #
    # The first is the whole 14% given back, written to disk, silently, for
    # every later run on that machine -- and the first thing that ever tunes is
    # usually the test suite. Re-tuning per process is merely slow; a cached
    # wrong answer is permanent and invisible, which is the worse failure for a
    # project whose output is measurements.
    #
    # Set MOBA_TUNE_CACHE=1 where conditions are stable -- a datacenter card
    # with `nvidia-smi -lgc` locked clocks, which is what `bench/suite.py`
    # arranges on hardware that allows it. Clear a bad cache by removing
    # ~/.triton/cache or setting TRITON_CACHE_DIR.
    cache_results=os.environ.get("MOBA_TUNE_CACHE", "0") == "1",
    do_bench=_best_of_three,
))
@triton.jit
def moba_fwd_kernel(
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
        # int64 from here down. The partial buffers are the largest
        # objects this design allocates -- [B, H, S, K_SLOTS, D] against
        # the output's [B, S, H, D] -- so they are the first thing to
        # cross 2^31 *elements*, which is where Triton's default int32
        # offsets wrap. Measured on an H100: B=4 H=32 S=65536 D=128 is
        # 8.59e9 elements and raises an illegal memory access, while the
        # same product reached three other ways (B=2 S=65536, B=4
        # S=32768, B=8 S=16384) fails identically and 2^31 exactly
        # passes -- so the boundary is the product, not any one axis.
        # Widening the base is enough: every term added to an int64 is
        # promoted. FlashMoBA avoids this by declaring `using index_t =
        # int64_t` once in flash.h; Triton has no such declaration, so
        # the cast is written where the chain starts. `routing.py`'s
        # PACK64 guard is the same problem solved in the same way.
        q_base = pid_b.to(tl.int64) * H_Q * S * D
        q_part_base = pid_b.to(tl.int64) * H_Q * S * K_SLOTS
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


@triton.jit
def moba_combine_kernel(
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
    # int64 for the reason moba_fwd_kernel records: PartAcc crosses 2^31
    # elements before anything else in the pipeline does.
    a_base = (PartAcc + pid_bh.to(tl.int64) * S * K_SLOTS * D
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


# -------------------------------------------------------------------------
# Launchers, in the order the forward runs them
# -------------------------------------------------------------------------


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
    What it costs is judged by `tests/tolerance.py`, not by a threshold picked to
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

    # GROUP and num_warps are chosen by `triton.autotune` on the decorator,
    # not here: the launcher passes neither. Everything that changes the shape
    # of the work is in the autotuner's `key`, so a new configuration tunes
    # once and every later call with the same shape reuses the winner --
    # persisted to disk, so it is decided once per machine rather than once
    # per process.
    moba_fwd_kernel[(index.num_blocks, n_seq * h_kv)](
        q, k, v,
        index.row_indices, index.head_indices, index.slot_of_pair,
        index.col_offsets, index.col_nnz, index.order,
        index.cu_seqlens if varlen else q,
        part_m, part_l, part_acc,
        s, index.row_indices.shape[-1], s,
        d ** -0.5 * LOG2_E,
        index.num_blocks,
        BLOCK_SIZE=cfg.block_size, D=d,
        K_SLOTS=slots, H_Q=h_q, H_KV=h_kv, CAUSAL=causal,
        VARLEN=varlen,
    )
    return part_m, part_l, part_acc


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

    moba_combine_kernel[(triton.cdiv(s, block_m), b * h)](
        part_m, part_l, part_acc, out, lse, s,
        K_SLOTS=slots, D=d, BLOCK_M=block_m, H_Q=h, WRITE_LSE=return_lse,
        num_warps=warps,
    )
    return (out, lse) if return_lse else out


# -------------------------------------------------------------------------
# Public entry point
# -------------------------------------------------------------------------


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
