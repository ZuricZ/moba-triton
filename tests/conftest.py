"""Shared fixtures and the correctness shape sweep.

Memory budget. The oracle materializes a [query_chunk, S] logit matrix per
(batch, head) rather than the full [S, S], so peak memory is roughly
B * H * query_chunk * S * 18 bytes in fp64 (logits + probs at 8 bytes each,
plus two boolean masks). At QUERY_CHUNK=256 that puts B2/H8/S8192 at ~1.4 GB,
comfortably inside this machine's 4 GB.

Without chunking, S=4096 alone consumed 2.5 GB and S=8192 was unreachable.
That is why long-context shapes are in the sweep at all: MoBA exists for long
context, and small-block effects that matter at S=8192 can be invisible at
S=1024.

Long shapes are marked `slow` and deselected with -m 'not slow'. A suite
nobody runs because it takes minutes is a suite that stops catching things.
"""

import math

import pytest
import torch

# Query rows processed at once by the oracle. 256 keeps peak memory low while
# staying large enough that cuBLAS picks the same GEMM kernel it would for the
# full matrix -- see test_query_chunking_is_bitwise_identical_at_realistic_sizes.
QUERY_CHUNK = 256

# (batch, heads, seq_len, head_dim) -- fast tier, runs by default.
#
# head_dim 128 is primary: it is what both reference implementations use
# (FlashMoBA benchmarks and MoonshotAI's llama example), and it is the size at
# which a query row spans two 128-byte cache lines rather than one. 64 is kept
# in the sweep as the secondary size, not dropped.
# Sequence lengths start at 1024 rather than 128. Below that, no realistic
# gate is sparse: at seq_len 512 with block_size 128 there are four blocks, so
# top_k 8 selects all of them. See `gate_is_sparse`.
SHAPE_SWEEP = [
    (1, 4, 1024, 128),
    (2, 4, 1024, 128),
    (1, 4, 2048, 128),
    (1, 4, 2048, 64),
]

# Long context. Slower, and the regime the project actually targets. seq_len
# 8192 exists so block_size 128 with top_k 12 is covered somewhere at all: it
# needs 48 blocks to clear the sparsity floor, hence seq_len >= 6144.
LONG_SHAPE_SWEEP = [
    (1, 4, 4096, 128),
    (1, 4, 4096, 64),
    (1, 2, 8192, 128),
]

# (block_size, top_k).
#
# Chosen to match what the reference implementations actually run, checked
# against the vendored source rather than assumed:
#
#   FlashMoBA benchmarks   block 128, top_k 8      (tests: block 64, top_k 4)
#   MoonshotAI llama       chunk 4096, top_k 12
#   MoonshotAI speed test  chunk 512,  top_k 3
#
# Earlier sweeps used block_size 16 and top_k 2. Neither appears anywhere in
# either reference: 16 is below every block size they use, and top_k 2 is
# below every budget they use. block_size is now a multiple of 32, which also
# matches the 32-lane warp width -- a 16-wide reduction leaves half of every
# warp idle.
#
# block_size 256 is absent at head_dim 128: its K and V tiles alone exceed
# this GPU's 99 KiB of shared memory. See `_choose_group`.
GATE_SWEEP = [
    (32, 4),
    (32, 8),
    (64, 4),
    (64, 8),
    (128, 8),
    (128, 12),
]


def make_qkv(shape, dtype=torch.float64, seed=0, device="cuda"):
    """Builds deterministic q, k, v tensors.

    Args:
        shape: Tuple of (batch, heads, seq_len, head_dim).
        dtype: Tensor dtype.
        seed: RNG seed, so a failure reproduces exactly.
        device: Torch device string.

    Returns:
        A tuple (q, k, v), each shaped `shape`.
    """
    g = torch.Generator(device=device).manual_seed(seed)
    return tuple(
        torch.randn(shape, generator=g, device=device, dtype=dtype) for _ in range(3)
    )


def case_id(shape, gate):
    """Builds a readable pytest id, e.g. b1h4s512d64-blk16-top2."""
    b, h, s, d = shape
    bk, tk = gate
    return f"b{b}h{h}s{s}d{d}-blk{bk}-top{tk}"


# A gate is only meaningfully sparse when there are several times more blocks
# than the budget selects. Below this ratio MoBA degenerates to dense
# attention, which Tier 1 already covers exactly and which teaches a kernel
# sweep nothing.
#
# This matters more than it looks because realistic top_k is 8-12, and
# `num_blocks = seq_len / block_size`. So block_size 128 with top_k 8 needs
# seq_len >= 4096 merely to be 4x sparse. Realistic gate settings *require*
# long sequences; the old sweep hid that by using top_k=2, which stays sparse
# even with very few blocks.
MIN_BLOCKS_PER_SELECTION = 4


def gate_is_sparse(shape, gate):
    """Reports whether a (shape, gate) pair is sparse enough to be worth running.

    Args:
        shape: Tuple of (batch, heads, seq_len, head_dim).
        gate: Tuple of (block_size, top_k).

    Returns:
        True if the configuration selects a small enough fraction of blocks.
    """
    seq_len = shape[2]
    block_size, top_k = gate
    num_blocks = math.ceil(seq_len / block_size)
    return num_blocks >= MIN_BLOCKS_PER_SELECTION * top_k


def sweep_cases(include_long=True):
    """Yields pytest params over shapes x gates, marking long shapes slow.

    Degenerate pairs -- those failing `gate_is_sparse` -- are not emitted at
    all, rather than emitted and skipped. A sweep that reports 11 skips for
    configurations it should never have generated makes real skips harder to
    notice.

    Args:
        include_long: Whether to emit the long-context shapes.

    Returns:
        A list of pytest.param entries.
    """
    cases = []
    for shape in SHAPE_SWEEP:
        for gate in GATE_SWEEP:
            if gate_is_sparse(shape, gate):
                cases.append(pytest.param(shape, gate, id=case_id(shape, gate)))
    if include_long:
        for shape in LONG_SHAPE_SWEEP:
            for gate in GATE_SWEEP:
                if gate_is_sparse(shape, gate):
                    cases.append(
                        pytest.param(
                            shape, gate, id=case_id(shape, gate),
                            marks=pytest.mark.slow,
                        )
                    )
    return cases


@pytest.fixture(scope="session")
def device():
    if not torch.cuda.is_available():
        pytest.skip("No CUDA device available")
    return "cuda"


@pytest.fixture(autouse=True)
def _free_memory():
    """Releases cached blocks between tests so a 4 GB card survives the sweep."""
    yield
    torch.cuda.empty_cache()
