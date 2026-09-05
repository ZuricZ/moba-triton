"""Stage 2 of MoBA: score blocks against each query and keep the top-k.

Consumes the block representatives `pooling.py` produces, and hands its choice
to `routing.py`, which turns it into the structure the attention kernel walks.

Fused: the score matrix is never materialized. The PyTorch gate computes an
`[B, H, S, num_blocks]` score matrix, writes it to HBM, reads it back to mask
it, writes it again, sorts it, and writes a dense boolean mask. Profiling put
~40% of the gate's GPU time in `aten::copy_` and device-to-device memcpy --
moving intermediates, not computing anything.

This kernel never lets that matrix exist. It streams the block representatives
past a resident query tile in chunks, keeps a running top-k on chip, and emits
only the answer: for each query, the block ids it selected, ascending.

This is the same trick as FlashAttention itself. FA never writes the `[S, S]`
score matrix because it only ever needs a running summary of it -- the running
max and sum. Here we only ever need a running top-k. In both cases the large
intermediate is scaffolding, not the answer, and scaffolding you can keep on
chip is scaffolding you never pay to move.

Pattern borrowed from FlashMoBA's "Flash TopK" (its stage 2, in CUDA, which
reports 22x over the unfused gate) and from fla-org's NSA
`parallel_nsa_kernel_topk`, which does the same in Triton.

WHAT IS DIFFERENT FROM NSA. NSA merges each streamed chunk into a *sorted*
running list with a bitonic network. Sortedness is not needed here: the only
thing the merge ever asks of the running list is its minimum -- the bar a new
score must clear to get in. Finding a minimum over `top_k` entries is one
reduction; keeping them sorted costs `log^2` compare-exchange stages per chunk.
NSA pays that because its candidate list is long. MoBA's is `top_k`, typically
8, where the sort would be pure overhead -- and Triton exposes no warp shuffle,
so a bitonic network here would have to be built out of reshapes.

WHY STREAMING AT ALL, given that an earlier version held every representative
at once and compiled without spilling even at 512 blocks. Because the cost of
the one-shot version was never spills; it was that its working tile was
`[BLOCK_M, num_blocks]`, so the tile grew with the sequence, the warp count had
to grow with the tile to avoid spilling, and occupancy fell as the warp count
rose. Measured at head_dim 128: num_blocks 128 ran 4 warps at 154 registers per
thread, fitting 3 programs per SM; num_blocks 256 ran 8 warps at 155, fitting
one. Per-thread register use barely moved -- the *program* got wider, and an
SM's 65536 registers are a budget shared by everything resident on it.
Streaming pins the tile width to `GATE_BLOCK_N` and the warp count with it, so
occupancy no longer depends on sequence length.

The loop that streaming introduces also admits an optimization the one-shot
form cannot express at all: its trip count is a runtime value, so a query tile
stops at the last block any of its queries can causally see instead of scoring
every block and masking the future away afterwards. See `gate_kernel`.
"""

import torch
import triton
import triton.language as tl

from moba.config import MoBAConfig
from moba.device import TENSOR_CORE_TILE
from moba.gqa import head_group_size
from moba.varlen import validate_cu_seqlens

# Queries per gate program. 32 rather than the 16 the one-shot gate used:
# halving the program count halves how many times the representatives are
# re-read, and the one-shot gate could not afford the wider tile because its
# tile was already `num_blocks` wide. Measured 1.3-1.9x over BLOCK_M=16 at every
# sequence length swept.
GATE_BLOCK_M = 32

# Representatives scored per streaming step, and the width of the only working
# tile in the kernel -- the whole point of the streaming form is that this does
# not grow with the sequence. Must be at least `TENSOR_CORE_TILE`, since the
# chunk is an operand of `tl.dot`.
#
# This is a real trade-off rather than a free parameter. Useful work per chunk
# is `TAKE * BLOCK_N`; the merge bookkeeping is `TAKE * 2 * K_POW2` regardless
# of chunk width. So overhead is `2 * K_POW2 / BLOCK_N` -- wide chunks amortize
# it -- while the causal horizon below is only tracked to `BLOCK_N` granularity,
# so wide chunks overshoot it. Swept at 16/32/64 over seq_len 4K-64K: 32 wins
# everywhere except 64K, where it ties 64.
GATE_BLOCK_N = 32

# Fixed, because the tile shape is now fixed. The one-shot gate had to derive
# this from `num_blocks`, and that derivation was what collapsed occupancy.
GATE_WARPS = 4


# -------------------------------------------------------------------------
# Device code
# -------------------------------------------------------------------------


@triton.jit
def gate_kernel(
    Q, Kbar, BlockIdx, CuSeqlens,
    S, T, sm_scale,
    BLOCK_SIZE: tl.constexpr, BLOCK_M: tl.constexpr, D: tl.constexpr,
    BLOCK_N: tl.constexpr, N_BLOCKS: tl.constexpr, K_POW2: tl.constexpr,
    CAPACITY: tl.constexpr, TAKE: tl.constexpr, TOP_K: tl.constexpr,
    ALWAYS_CURRENT: tl.constexpr,
    H_Q: tl.constexpr, Q_PER_KV: tl.constexpr, VARLEN: tl.constexpr,
):
    """One program per (query tile, batch, query head). Emits selected block ids.

    Writes `BlockIdx[b, h, s, j] = block id of the query's j-th selection`,
    ascending in `j`, padded with -1. Nothing else is written: no scores, no
    dense mask.

    Per *query* head, not per key/value head, and that survives grouped-query
    attention unchanged. Query heads sharing a key/value head rank the same
    representatives, but they rank them with different queries, so they choose
    different blocks -- the ranking is independent even though the thing ranked
    is shared. Only `Kbar`'s base pointer is remapped.

    The merging of a group's selections into one list happens downstream, in
    the CSC transpose, which is the first stage that deals in blocks rather
    than queries. Nothing here gets longer under grouped-query attention.

    Packed mode adds a sequence and changes nothing else. Because the grid is
    per sequence, a tile belongs to exactly one of them, so `bos`, `eos` and the
    candidate count are *scalars* -- and that is what keeps the streaming top-k
    below a zero-line diff. A tile spanning two sequences would give its lanes
    different candidate ranges, and a Triton program has one loop bound for all
    its lanes.

    The emitted ids are sequence-local in both modes, so nothing is added to
    them on the way out. What the packed path must never do is offer a query a
    block belonging to another sequence: the attention kernel's causal mask
    compares packed token indices and is correct only because that cannot
    happen. See the varlen plan's D-VL-7 and
    `test_no_selected_block_crosses_a_sequence_boundary`.
    """
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)

    # `pid_bh` enumerates B * H_Q. Queries, the output, and this program's whole
    # job are per query head; only the block representatives are shared, so only
    # their base pointer is remapped. At Q_PER_KV == 1 this is `pid_bh` exactly,
    # which is why the multi-head path is untouched.
    kv_bh = (pid_bh // H_Q) * (H_Q // Q_PER_KV) + (pid_bh % H_Q) // Q_PER_KV

    offs_d = tl.arange(0, D)
    slots = tl.arange(0, K_POW2)

    b = pid_bh // H_Q
    hq = pid_bh % H_Q

    if VARLEN:
        # The batch comes from the grid, so every value here is a scalar.
        # `offs_m` is the *packed* token index, used to address Q and the
        # output; `local` is the position within the sequence, which is what
        # decides the query's own block and the causal horizon. Confusing the
        # two is invisible in the first sequence and wrong in every other one.
        bos = tl.load(CuSeqlens + b)
        eos = tl.load(CuSeqlens + b + 1)
        seqlen = eos - bos
        if pid_m * BLOCK_M >= seqlen:
            return              # an over-launched tile: this sequence is short
        local = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_m = bos + local
        live = local < seqlen
        # This sequence's own block count. Distinct from N_BLOCKS, which stays
        # the *stride* of Kbar and the sentinel for "no block" -- conflating
        # them would let a query rank against another sequence's
        # representatives.
        n_cand = tl.cdiv(seqlen, BLOCK_SIZE)
        q_base = hq * D                     # Q is [1, T, H_q, D]
        out_base = hq * T * TOP_K           # BlockIdx is ours: [1, H_q, T, K]
    else:
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        local = offs_m
        live = offs_m < S
        n_cand = N_BLOCKS
        q_base = b * S * H_Q * D + hq * D    # Q is [B, S, H_q, D]
        out_base = pid_bh * S * TOP_K        # BlockIdx is ours: [B, H_q, S, K]

    # Token-major: tokens are H_Q*D apart. `q_base` is the only part that
    # differs between packed and padded, and it is zero when packed.
    q = tl.load(Q + q_base + offs_m[:, None] * (H_Q * D) + offs_d[None, :],
                mask=live[:, None], other=0.0)

    own_block = local // BLOCK_SIZE

    # --- The running top-k, held in registers for the whole kernel. ----------
    # `best_val` holds the k best scores seen so far and `best_idx` where each
    # came from. Empty slots sit at -inf, which makes them the natural first
    # things to be replaced and removes the need for a separate "is the list
    # full yet" flag.
    best_val = tl.full([BLOCK_M, K_POW2], float("-inf"), tl.float32)
    best_idx = tl.full([BLOCK_M, K_POW2], N_BLOCKS, tl.int32)

    # `K_POW2` is rounded up to a power of two because `tl.arange` demands it,
    # so at top_k 12 there are 16 slots for 12 entries. The surplus has to be
    # retired or it becomes extra capacity: a -inf slot reads as "empty, fill
    # me", and across two chunks the list would admit 16 blocks where 12 were
    # allowed. +inf retires them -- never the minimum, so never an eviction
    # victim, and their sentinel block index makes them emit as padding.
    #
    # This needs at least two chunks to go wrong, so it is invisible whenever
    # num_blocks <= BLOCK_N.
    best_val = tl.where(slots[None, :] >= CAPACITY, float("inf"), best_val)

    if ALWAYS_CURRENT:
        # Seeded rather than appended at the end. An entry at +inf can never be
        # beaten, so it is guaranteed to survive to the emit step, and the merge
        # loop below needs no special case for it. `K_POW2` is sized to leave
        # room for this seed on top of the `TAKE` past blocks.
        seed = slots[None, :] == 0
        best_val = tl.where(seed, float("inf"), best_val)
        best_idx = tl.where(seed, own_block[:, None].to(tl.int32), best_idx)

    # --- Stop at the causal horizon of this tile. ----------------------------
    # A block is a candidate only if it lies wholly in a query's past, so the
    # highest block any query here can see is `max(own_block) - 1`. The one-shot
    # gate scored every block and masked the future away afterwards, so a query
    # tile early in the sequence did exactly as much work as one at the end.
    # Averaged over the sequence this halves the gate's arithmetic, the matmul
    # included, and it is expressible only because the trip count is a runtime
    # value rather than a tile width.
    horizon = tl.max(tl.where(live, own_block, 0))
    n_chunks = tl.cdiv(tl.minimum(horizon, n_cand), BLOCK_N)

    for c in range(0, n_chunks):
        cols = c * BLOCK_N + tl.arange(0, BLOCK_N)
        in_range = cols < n_cand

        kbar = tl.load(
            Kbar + kv_bh * N_BLOCKS * D + cols[:, None] * D + offs_d[None, :],
            mask=in_range[:, None], other=0.0,
        )

        # `tl.dot` accumulates in float32. The PyTorch gate multiplies bf16 by
        # bf16, and cuBLAS also accumulates in float32 but rounds its *output*
        # back to bf16. Rounding here reproduces that exactly, which matters
        # because the routing test compares selections for equality and a
        # near-tie decided differently is a mismatch, not an approximation.
        s = tl.dot(q, tl.trans(kbar)) * sm_scale
        s = s.to(kbar.dtype).to(tl.float32)

        # The query's own block is excluded here even when ALWAYS_CURRENT is
        # set: it needs an internal causal mask at attention time, so it is
        # never ranked against the others.
        is_past = (cols[None, :] < own_block[:, None]) & in_range[None, :]
        s = tl.where(is_past, s, float("-inf"))

        # --- Merge this chunk into the running list. -------------------------
        # TAKE rounds, each offering the chunk's best remaining candidate to the
        # list. The list's minimum is the entry bar; a candidate that clears it
        # overwrites that minimum's slot.
        for _ in tl.static_range(TAKE):
            bv = tl.max(s, axis=1)

            # Lowest block index wins a tie. This is `tie_break_low_index`, a
            # determinism requirement rather than an accuracy one: an unstable
            # tie-break selects different blocks on different runs, which is
            # nearly indistinguishable from a race when debugging.
            bi = tl.min(tl.where(s == bv[:, None], cols[None, :], N_BLOCKS),
                        axis=1)

            # The eviction victim: the weakest entry in the list. When several
            # entries tie for weakest, the one from the *highest* block index
            # goes, so the lowest-indexed of the tied blocks survives -- the
            # same `tie_break_low_index` rule that governs admission.
            #
            # Getting this wrong is subtle and was the one bug this rewrite
            # introduced. Choosing the victim by slot position instead evicts
            # whichever tied entry happened to be written to a lower slot,
            # which is arbitrary; measured at block_size 32 it disagreed with
            # the oracle on 764 of 8.4M decisions, every one an exact tie. The
            # one-shot gate could not hit this at all, because it only ever set
            # bits in a `chosen` mask and never took one back out. Eviction is
            # a concept streaming introduces, so the tie rule has to reach it.
            floor = tl.min(best_val, axis=1)
            weakest = best_val == floor[:, None]
            victim = tl.max(tl.where(weakest, best_idx, -1), axis=1)
            # Empty slots all carry the same sentinel index, so `victim` can
            # match several of them; narrowing to the lowest such slot keeps
            # exactly one write target.
            floor_slot = tl.min(
                tl.where(weakest & (best_idx == victim[:, None]),
                         slots[None, :], K_POW2),
                axis=1,
            )

            # Strictly greater, not >=. Chunks are visited in ascending block
            # order, so a score that merely ties the current k-th best loses to
            # the lower-indexed block already holding that value -- the same
            # tie-break rule as above, obtained here for free from the visit
            # order. Strictness also makes a round that finds nothing admit
            # nothing, which is what a query with fewer past blocks than the
            # budget needs: top_k is a ceiling, not a quota.
            enter = bv > floor
            target = (slots[None, :] == floor_slot[:, None]) & enter[:, None]
            best_val = tl.where(target, bv[:, None], best_val)
            best_idx = tl.where(target, bi[:, None].to(tl.int32), best_idx)

            # Retire the admitted candidate so the next round finds the one
            # below it. If nothing was admitted, nothing else in this chunk can
            # be either -- every remaining score is below the one that just
            # failed -- so the leftover rounds are harmless no-ops.
            s = tl.where((cols[None, :] == bi[:, None]) & enter[:, None],
                         float("-inf"), s)

    # --- Emit ascending. -----------------------------------------------------
    # Slot j is the rank of a block among the query's selections ordered by
    # ascending block index -- the same rule the PyTorch index used, so the
    # combine pass's fixed-slot addressing is unchanged. Repeatedly taking the
    # lowest remaining selection produces exactly that order. Unlike the
    # one-shot gate this scans K_POW2 entries rather than num_blocks, so the
    # emit step no longer grows with the sequence either.
    remaining = tl.where(best_val > float("-inf"), best_idx, N_BLOCKS)
    for j in tl.static_range(TOP_K):
        nxt = tl.min(remaining, axis=1)
        valid = nxt < N_BLOCKS
        tl.store(
            BlockIdx + out_base + offs_m * TOP_K + j,
            tl.where(valid, nxt, -1).to(tl.int32),
            mask=live,
        )
        # Safe to clear every slot equal to `nxt`: a block can be admitted only
        # once, and the seeded current block is never a past candidate, so no
        # index ever appears twice.
        remaining = tl.where(remaining == nxt[:, None], N_BLOCKS, remaining)


# -------------------------------------------------------------------------
# Launcher
# -------------------------------------------------------------------------


def gate_select(
    q: torch.Tensor,
    kbar: torch.Tensor,
    cfg: MoBAConfig,
    cu_seqlens: torch.Tensor | None = None,
    max_seqlen: int | None = None,
) -> torch.Tensor:
    """Selects blocks for every query, returning ids rather than a dense mask.

    Args:
        q: Queries shaped [B, S, H, D], contiguous.
        kbar: Block representatives shaped [B, H_kv, num_blocks, D],
            contiguous, where `H_kv` divides `H`. Fewer representative heads
            than query heads is grouped-query attention: the group shares what
            it ranks, not how it ranks.
        cfg: Configuration supplying `block_size`, `top_k` and the selection
            conventions.
        cu_seqlens: Cumulative token counts, [num_sequences + 1] int32. When
            given, `q` holds one packed batch and `B` must be 1.
        max_seqlen: Longest sequence, required with `cu_seqlens`. It shapes the
            launch grid; deriving it would cost a host synchronization.

    Returns:
        Selected block ids shaped [B, H, S, top_k], int32, ascending within
        each query and padded with -1. One row per *query* head regardless of
        how many representative heads there were. Ids are **sequence-local** in
        both modes: under packing, `j` means "block j of my own sequence".

    Raises:
        ValueError: If the shapes disagree, if the head counts do not divide,
            if packed input has a batch axis above 1, or if `max_seqlen` is
            missing.
    """
    b, s, h, d = q.shape
    block_m = max(TENSOR_CORE_TILE, GATE_BLOCK_M)
    varlen = cu_seqlens is not None

    # The packed checks run before the shape guard because they decide what the
    # guard should expect: `q` carries a pinned batch of 1 while `kbar` carries
    # one row per *sequence*, so comparing the two batch axes directly is only
    # right in padded mode.
    if varlen:
        if b != 1:
            raise ValueError(
                f"packed input must have batch 1, got {b}; every sequence "
                f"lives on the token axis and cu_seqlens says where"
            )
        if max_seqlen is None:
            raise ValueError("max_seqlen is required when cu_seqlens is given")
        n_seq = validate_cu_seqlens(cu_seqlens, total_tokens=s,
                                    max_seqlen=max_seqlen)
        expected_rows = n_seq
    else:
        expected_rows = b

    if kbar.shape[0] != expected_rows or kbar.shape[-1] != d:
        raise ValueError(
            f"representative shape {tuple(kbar.shape)} does not match queries "
            f"{tuple(q.shape)}"
        )
    q_per_kv = head_group_size(h, kbar.shape[1])
    n = kbar.shape[2]

    q, kbar = q.contiguous(), kbar.contiguous()

    if varlen:
        grid = (triton.cdiv(max_seqlen, block_m), n_seq * h)
        # A query can never see more than its own sequence's blocks, so the
        # budget is capped by the longest sequence rather than the packed total.
        take = cfg.past_blocks_selected(cfg.num_blocks(max_seqlen))
    else:
        grid = (triton.cdiv(s, block_m), b * h)
        take = cfg.past_blocks_selected(n)

    # Room for `take` past blocks plus the seeded current one. Rounded to a
    # power of two because `tl.arange` requires it.
    seed = 1 if cfg.always_select_current else 0
    capacity = take + seed
    k_pow2 = triton.next_power_of_2(max(1, capacity))

    out = torch.empty((b, h, s, cfg.top_k), device=q.device, dtype=torch.int32)

    gate_kernel[grid](
        q, kbar, out, cu_seqlens if varlen else q,
        s, s,
        # The gate score is deliberately unscaled: softmax's 1/sqrt(D) is a
        # positive monotone transform and cannot reorder a ranking, so applying
        # it would be arithmetic that changes no decision.
        1.0,
        BLOCK_SIZE=cfg.block_size, BLOCK_M=block_m, D=d,
        BLOCK_N=max(TENSOR_CORE_TILE, GATE_BLOCK_N),
        N_BLOCKS=n, K_POW2=k_pow2, CAPACITY=capacity,
        TAKE=take, TOP_K=cfg.top_k,
        ALWAYS_CURRENT=cfg.always_select_current,
        H_Q=h, Q_PER_KV=q_per_kv, VARLEN=varlen,
        num_warps=GATE_WARPS,
    )
    return out
