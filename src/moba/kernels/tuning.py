"""Host-side tuning policy for the MoBA kernels.

Nothing here runs on the GPU. These are the numbers that decide how a launch is
shaped -- chunk heights, tile sizes, warp counts -- and the shared-memory
arithmetic behind them. Both the forward and the backward read from here.

The recurring theme is that these are *proposals*, not decisions. Triton is the
only authority on what fits: `estimate_shared_memory` is a lower bound on what
the compiler will actually allocate, so `attend_partials` walks the candidate
list and lets a launch be rejected rather than trusting the estimate. Twice now
a conservative guess about the compiler has silently cost speed; see
`PIPELINE_STAGES`.
"""

import triton

from moba.config import MoBAConfig
from moba.device import TENSOR_CORE_TILE, WARP_SIZE, shared_memory_budget
from moba.kernels import tuned_configs


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


# -------------------------------------------------------------------------
# Attention chunk height
# -------------------------------------------------------------------------


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
    """Returns the largest inner-loop chunk height that fits in shared memory.

    NOT NECESSARILY THE ONE THE KERNEL USES. This is an upper bound from the
    shared-memory budget; `attend_partials` times the candidates that compile
    and keeps the fastest, which on sm_86 is consistently 64 rather than this
    function's 128. Largest-that-fits was the policy until 2026-09-07 and cost
    up to 15% -- a tile that large spills registers, and an SM hosting one
    resident block has nothing to hide memory latency behind. See `_AUTOTUNE`
    in `moba_fwd`.

    Still worth computing: it bounds the search, and `bench/load_balance.py`
    needs the tile height to simulate the kernel's job sizes.

    `group` is how many queries are processed per iteration -- NOT how many
    selected this block. The loop iterates as many times as needed either way,
    and K/V load once *outside* it, so a larger group buys no extra
    amortization; it only trims loop overhead. What it costs is shared memory,
    because the gathered q tile is pipelined, and registers, which is what
    actually decides.

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


# -------------------------------------------------------------------------
# Combine tile and warp count
# -------------------------------------------------------------------------


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



# ---------------------------------------------------------------------------
# Launch policy: table by default, autotuner on request
# ---------------------------------------------------------------------------


def clamp_group_to_shared_memory(group, block_size, head_dim, itemsize,
                                 device=None) -> int:
    """Halves `group` until the shared-memory estimate admits it.

    A tuned table entry is measured on one card and read on whatever card is
    present. Architectures differ in shared memory per block -- 99 KiB on the
    sm_86 laptop against 164 KiB on the A100 -- so an entry that fits where it
    was measured can be too large where it is used. Clamping keeps that a
    slowdown rather than a launch failure.

    Halving rather than stepping by `TENSOR_CORE_TILE` because `GROUP` feeds
    `tl.arange(0, GROUP)`, which requires a power of two.

    Args:
        group: The chunk height the table proposed.
        block_size: MoBA block size.
        head_dim: Head dimension.
        itemsize: Bytes per element of the input dtype.
        device: CUDA device, for the shared-memory budget.

    Returns:
        The largest power of two at or below `group` that the estimate admits,
        never below `TENSOR_CORE_TILE`. The estimate is a lower bound, so the
        compiler may still refuse -- and that remains its call.
    """
    budget = shared_memory_budget(device)
    while group > TENSOR_CORE_TILE:
        if estimate_shared_memory(block_size, group, head_dim,
                                  itemsize) <= budget:
            return group
        group //= 2
    return TENSOR_CORE_TILE


def launch_policy(pass_name: str, autotuner, key_tile: int = 0):
    """Returns the decorator that decides a kernel's launch configuration.

    Two policies, chosen at import time by `MOBA_AUTOTUNE`:

    * **Default -- `triton.heuristics` reading the baked table.** No
      benchmarking, ever. The heuristic runs per launch and costs a cached dict
      lookup on a tuple of ints.
    * **`MOBA_AUTOTUNE=1` -- the autotuner.** How the table is produced and how
      a new card is measured. Never the default, because tuning on the first
      launch at a new shape puts a benchmark on whatever request arrives first.

    Keeping both behind one call means the kernels declare *that* they have a
    launch policy without either restating it. The autotuner is passed in
    already constructed, because its configs and key are specific to the kernel
    and belong beside it.

    Args:
        pass_name: "fwd" or "bwd", the table's pass key.
        autotuner: A configured `triton.autotune(...)` decorator, used only
            when tuning is requested.
        key_tile: The backward's `KEY_TILE`; 0 for the forward, which has no
            such axis.

    Returns:
        A decorator to apply to the `@triton.jit` kernel.
    """
    if tuned_configs.autotune_requested():
        return autotuner

    return triton.heuristics({
        "GROUP": lambda args: resolve_launch_config(
            pass_name, args, key_tile)["GROUP"],
        "num_warps": lambda args: resolve_launch_config(
            pass_name, args, key_tile)["num_warps"],
        "num_stages": lambda args: resolve_launch_config(
            pass_name, args, key_tile)["num_stages"],
    })


def resolve_launch_config(pass_name: str, args: dict,
                          key_tile: int = 0) -> dict:
    """Returns the clamped table entry for one launch.

    Module-level rather than a closure inside `launch_policy` so that tests can
    ask what the table policy would choose without launching anything, and so
    there is exactly one place that turns kernel arguments into a config.

    Args:
        pass_name: "fwd" or "bwd".
        args: The kernel's arguments, as `triton.heuristics` supplies them.
        key_tile: The backward's `KEY_TILE`; 0 for the forward.

    Returns:
        A dict of GROUP, num_warps and num_stages, with GROUP already reduced
        to what this card's shared memory admits.
    """
    chosen = tuned_configs.lookup(
        pass_name,
        block_size=args["BLOCK_SIZE"],
        head_dim=args["D"],
        top_k=args["K_SLOTS"],
        h_q=args["H_Q"],
        h_kv=args["H_KV"],
        seq_len=args["S"],
        key_tile=key_tile,
    )
    chosen["GROUP"] = clamp_group_to_shared_memory(
        chosen["GROUP"], args["BLOCK_SIZE"], args["D"],
        # The kernel does not carry its own itemsize, and every dtype it
        # accepts is 2 bytes (`device.supports_dtype`). Stating it rather than
        # plumbing an argument that can only hold one value.
        itemsize=2,
    )
    return chosen


def group_in_force(kernel, pass_name: str, args: dict,
                   key_tile: int = 0) -> int:
    """Returns the chunk height the *current* policy chose, whichever it is.

    Exists so a test can assert a property of the height that ran without
    knowing which policy picked it. The selection policy has now changed three
    times -- hand-rolled search, then `triton.autotune`, then this table -- and
    each change broke tests that asserted a policy-independent property through
    a policy-specific attribute. `best_config` is an autotuner field and is
    simply absent under `triton.heuristics`.

    Args:
        kernel: The decorated Triton kernel, already launched at least once if
            the autotuner is in force.
        pass_name: "fwd" or "bwd".
        args: The kernel arguments for the launch in question.
        key_tile: The backward's `KEY_TILE`; 0 for the forward.

    Returns:
        The `GROUP` that ran.
    """
    best = getattr(kernel, "best_config", None)
    if best is not None:
        return best.kwargs["GROUP"]
    return resolve_launch_config(pass_name, args, key_tile)["GROUP"]
