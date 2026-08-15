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

import pytest
import torch

# Query rows processed at once by the oracle. 256 keeps peak memory low while
# staying large enough that cuBLAS picks the same GEMM kernel it would for the
# full matrix -- see test_query_chunking_is_bitwise_identical_at_realistic_sizes.
QUERY_CHUNK = 256

# (batch, heads, seq_len, head_dim) -- fast tier, runs by default.
SHAPE_SWEEP = [
    (1, 4, 128, 64),
    (1, 4, 512, 64),
    (2, 4, 512, 64),
    (1, 8, 1024, 64),
]

# Long context. Slower, and the regime the project actually targets.
LONG_SHAPE_SWEEP = [
    (1, 8, 2048, 64),
    (1, 8, 4096, 64),
    (1, 4, 8192, 64),
]

# (block_size, top_k). Small blocks are deliberately over-represented: they
# are the regime this project exists to fix, and the regime where a naive
# kernel loop order degrades.
GATE_SWEEP = [
    (16, 2),
    (16, 4),
    (32, 2),
    (32, 8),
    (64, 2),
    (128, 2),
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


def sweep_cases(include_long=True):
    """Yields pytest params over shapes x gates, marking long shapes slow.

    Args:
        include_long: Whether to emit the long-context shapes.

    Returns:
        A list of pytest.param entries.
    """
    cases = []
    for shape in SHAPE_SWEEP:
        for gate in GATE_SWEEP:
            cases.append(pytest.param(shape, gate, id=case_id(shape, gate)))
    if include_long:
        for shape in LONG_SHAPE_SWEEP:
            for gate in GATE_SWEEP:
                cases.append(
                    pytest.param(
                        shape, gate, id=case_id(shape, gate), marks=pytest.mark.slow
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
