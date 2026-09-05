"""Tier 1: exact identities. No tolerance argument is permitted here.

Every assertion in this file holds exactly, in exact arithmetic. We run them
in float64 so that floating-point noise is far below the 1e-10 thresholds
used. A failure here is therefore unambiguously a logic bug, never a
numerical artifact -- which is what makes this the strongest tier in the
project.
"""

import pytest
import torch
import torch.nn.functional as F

from moba.config import MoBAConfig
from moba.naive import (
    pool_blocks_naive,
    moba_attention_naive,
    gate_select_naive,
)

DEVICE = "cuda"


def make_qkv(b, s, h, d, dtype=torch.float64, seed=0):
    """Deterministic q, k, v so a failure reproduces exactly.

    Token-major, so the argument order is (batch, seq_len, heads, head_dim) --
    the same order the tensor has, which is the point of the layout.
    """
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    shape = (b, s, h, d)
    return tuple(
        torch.randn(shape, generator=g, device=DEVICE, dtype=dtype) for _ in range(3)
    )


# --------------------------------------------------------------------------
# Step 2 of the algorithm: block representatives
# --------------------------------------------------------------------------


@pytest.mark.parametrize("seq_len,block_size", [(128, 32), (129, 32), (64, 64)])
def test_pool_blocks_naive_are_block_means(seq_len, block_size):
    _, k, _ = make_qkv(1, seq_len, 2, 16)
    cfg = MoBAConfig(block_size=block_size, top_k=1)
    kbar = pool_blocks_naive(k, cfg)

    assert kbar.shape == (1, 2, cfg.num_blocks(seq_len), 16)
    # kbar stays head-major; k is token-major, so its blocks live on axis 1.
    torch.testing.assert_close(kbar[:, :, 0],
                               k[:, :block_size].mean(dim=1))


def test_short_final_block_is_not_diluted_by_padding():
    """seq_len=129 with block_size=32 leaves a final block of one key.

    Implementations that pad to a block multiple and take a plain mean would
    divide that key by 32 instead of 1. This is the classic off-by-padding
    bug, and it is invisible whenever seq_len happens to divide evenly.
    """
    _, k, _ = make_qkv(1, 129, 1, 16)
    cfg = MoBAConfig(block_size=32, top_k=1)
    kbar = pool_blocks_naive(k, cfg)

    assert cfg.num_blocks(129) == 5
    torch.testing.assert_close(kbar[:, :, -1], k[:, 128:].mean(dim=1))


# --------------------------------------------------------------------------
# Steps 4-6: causal candidacy, forced current block, top-k
# --------------------------------------------------------------------------


def test_current_block_always_selected():
    q, k, _ = make_qkv(1, 128, 1, 16)
    cfg = MoBAConfig(block_size=32, top_k=1)
    sel = gate_select_naive(q, pool_blocks_naive(k, cfg), cfg)

    pos = torch.arange(128, device=DEVICE)
    assert sel[0, 0, pos, pos // 32].all()


def test_future_blocks_never_selected():
    q, k, _ = make_qkv(1, 128, 1, 16)
    cfg = MoBAConfig(block_size=32, top_k=8)
    sel = gate_select_naive(q, pool_blocks_naive(k, cfg), cfg)

    pos = torch.arange(128, device=DEVICE)
    blocks = torch.arange(cfg.num_blocks(128), device=DEVICE)
    is_future = blocks.view(1, -1) > (pos // 32).view(-1, 1)
    assert not sel[0, 0][is_future].any()


def test_selection_count_respects_top_k():
    """Under the default convention top_k counts the query's own block."""
    q, k, _ = make_qkv(1, 256, 1, 16)
    cfg = MoBAConfig(block_size=32, top_k=2)
    sel = gate_select_naive(q, pool_blocks_naive(k, cfg), cfg)

    # Query 255 sits in block 7; candidates are blocks 0..6. top_k=2 buys the
    # own block plus one past block.
    assert sel[0, 0, 255].sum().item() == 2


def test_top_k_convention_flag_changes_the_budget():
    """Both readings of top_k are reachable, and they differ by exactly one.

    Default (True) matches MoonshotAI's reference, verified bitwise. Setting
    it False restores the reading where the own block is free -- kept because
    it is the more natural definition of "how many blocks of context do I get
    to look at", and because a future reference may use it.
    """
    q, k, _ = make_qkv(1, 256, 1, 16)

    inclusive = MoBAConfig(block_size=32, top_k=3, top_k_includes_current=True)
    exclusive = MoBAConfig(block_size=32, top_k=3, top_k_includes_current=False)

    n_inc = gate_select_naive(q, pool_blocks_naive(k, inclusive), inclusive)
    n_exc = gate_select_naive(q, pool_blocks_naive(k, exclusive), exclusive)

    assert n_inc[0, 0, 255].sum().item() == 3
    assert n_exc[0, 0, 255].sum().item() == 4


def test_top_k_of_one_selects_only_the_current_block():
    """Edge case of the inclusive convention: the budget is entirely spent."""
    q, k, _ = make_qkv(1, 128, 1, 16)
    cfg = MoBAConfig(block_size=32, top_k=1)
    sel = gate_select_naive(q, pool_blocks_naive(k, cfg), cfg)

    pos = torch.arange(128, device=DEVICE)
    assert (sel.sum(-1) == 1).all()
    assert sel[0, 0, pos, pos // 32].all()


def test_early_queries_select_all_available_candidates():
    """A query in block 1 has only one past block; top_k=4 cannot invent more."""
    q, k, _ = make_qkv(1, 128, 1, 16)
    cfg = MoBAConfig(block_size=32, top_k=4)
    sel = gate_select_naive(q, pool_blocks_naive(k, cfg), cfg)

    assert sel[0, 0, 40].sum().item() == 2  # block 0 + own block 1
    assert sel[0, 0, 0].sum().item() == 1  # block 0 only, it is its own


def test_ties_resolve_to_lower_block_index():
    """Identical keys everywhere make every gate score equal."""
    b, h, s, d = 1, 1, 128, 16
    g = torch.Generator(device=DEVICE).manual_seed(0)
    q = torch.randn(b, s, h, d, generator=g, device=DEVICE, dtype=torch.float64)
    k = torch.ones(b, s, h, d, device=DEVICE, dtype=torch.float64)
    # top_k=2 under the inclusive convention buys exactly one past block, so
    # the tie-break decides which one.
    cfg = MoBAConfig(block_size=32, top_k=2)

    sel = gate_select_naive(q, pool_blocks_naive(k, cfg), cfg)

    # Query 127 is in block 3; candidates {0,1,2} are perfectly tied.
    assert bool(sel[0, 0, 127, 0]) is True
    assert bool(sel[0, 0, 127, 1]) is False
    assert bool(sel[0, 0, 127, 2]) is False


# --------------------------------------------------------------------------
# Step 7: full attention. The keystone identities.
# --------------------------------------------------------------------------


def test_reduces_to_dense_causal_attention_when_top_k_covers_all_blocks():
    """THE keystone test.

    When top_k is large enough that every past block is selectable, the gate
    becomes a no-op and MoBA must equal ordinary dense causal attention. This
    validates pooling, masking, scaling, softmax and the output projection all
    at once, against an independent implementation (torch SDPA), with no
    tolerance argument available to hide behind.
    """
    b, s, h, d = 1, 128, 2, 32
    q, k, v = make_qkv(b, s, h, d)
    cfg = MoBAConfig(block_size=16, top_k=s // 16)

    got = moba_attention_naive(q, k, v, cfg)
    # torch SDPA is head-major; we are not. Transposing here rather than
    # comparing in SDPA's layout keeps the assertion about our output.
    want = F.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True
    ).transpose(1, 2)

    torch.testing.assert_close(got, want, rtol=1e-10, atol=1e-10)


def test_single_block_equals_plain_causal_attention():
    """With one block total, there is nothing to select; only causality acts."""
    b, s, h, d = 1, 32, 1, 16
    q, k, v = make_qkv(b, s, h, d)
    cfg = MoBAConfig(block_size=32, top_k=1)

    got = moba_attention_naive(q, k, v, cfg)
    # torch SDPA is head-major; we are not. Transposing here rather than
    # comparing in SDPA's layout keeps the assertion about our output.
    want = F.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True
    ).transpose(1, 2)

    torch.testing.assert_close(got, want, rtol=1e-10, atol=1e-10)


def test_first_query_attends_only_to_itself():
    """Position 0 sees exactly one key, so softmax is degenerate and out == v[0]."""
    b, s, h, d = 1, 64, 1, 16
    q, k, v = make_qkv(b, s, h, d)
    cfg = MoBAConfig(block_size=16, top_k=2)

    got = moba_attention_naive(q, k, v, cfg)
    torch.testing.assert_close(got[0, 0, 0], v[0, 0, 0], rtol=1e-10, atol=1e-10)


def test_output_rows_are_convex_combinations_of_values():
    """Softmax weights are non-negative and sum to 1, so every output row must
    lie within the elementwise min/max of the values it could attend to.

    This catches sign errors and mask leaks that a shape check would miss.
    """
    b, s, h, d = 1, 96, 1, 8
    q, k, v = make_qkv(b, s, h, d)
    cfg = MoBAConfig(block_size=16, top_k=2)

    out = moba_attention_naive(q, k, v, cfg)
    # Token-major: the key axis the output mixes over is 1, not 2.
    lo = v.amin(dim=1, keepdim=True)
    hi = v.amax(dim=1, keepdim=True)
    assert (out >= lo - 1e-12).all() and (out <= hi + 1e-12).all()


@pytest.mark.parametrize("chunk", [64, 128, 1000])
def test_query_chunking_is_bitwise_identical_at_realistic_sizes(chunk):
    """Chunking over queries is a memory optimization, not an algorithm change.

    Softmax is per-row, so a block of queries evaluated against all keys has
    every column it needs. Nothing whatsoever is accumulated across chunks.

    At chunk sizes we would actually use, cuBLAS picks the same GEMM kernel it
    would for the full matrix, so the result is bitwise identical. `1000`
    exceeds S and checks the clamp.
    """
    q, k, v = make_qkv(1, 128, 2, 32)
    cfg = MoBAConfig(block_size=16, top_k=2)

    unchunked = moba_attention_naive(q, k, v, cfg)
    chunked = moba_attention_naive(q, k, v, cfg, query_chunk=chunk)

    assert torch.equal(chunked, unchunked), (
        f"chunk={chunk} changed the result; max diff "
        f"{(chunked - unchunked).abs().max().item():.3e}"
    )


@pytest.mark.parametrize("chunk", [1, 7, 16])
def test_tiny_query_chunks_agree_to_floating_point_noise(chunk):
    """Pathologically small chunks differ in the last bits, and that is cuBLAS.

    With M=1 a GEMM becomes a matrix-vector product and cuBLAS selects a
    different kernel, with a different accumulation order. Floating-point
    addition is not associative, so reassociating a dot product moves the last
    bit or two. Verified independently: a bare einsum with no softmax and no
    masking reproduces this at chunk=1 and 7.

    The chunking logic itself is exact -- nothing crosses a chunk boundary --
    so the residual is bounded by fp64 rounding, ~1e-14, which is twelve
    orders of magnitude below the bf16 tolerance budget of ~1e-2.
    """
    q, k, v = make_qkv(1, 128, 2, 32)
    cfg = MoBAConfig(block_size=16, top_k=2)

    unchunked = moba_attention_naive(q, k, v, cfg)
    chunked = moba_attention_naive(q, k, v, cfg, query_chunk=chunk)

    diff = (chunked - unchunked).abs().max().item()
    assert diff < 1e-12, f"chunk={chunk} diff {diff:.3e} exceeds fp64 noise"


def test_query_chunking_identical_in_low_precision():
    """Also holds in bf16, where any drift would be far easier to hide."""
    q, k, v = make_qkv(1, 256, 2, 64)
    cfg = MoBAConfig(block_size=32, top_k=2, dtype=torch.bfloat16)
    qb, kb, vb = (t.to(torch.bfloat16) for t in (q, k, v))

    assert torch.equal(
        moba_attention_naive(qb, kb, vb, cfg, query_chunk=64),
        moba_attention_naive(qb, kb, vb, cfg),
    )


def test_no_nan_when_only_one_key_is_visible():
    """Guards the all-masked-row failure mode.

    If the current block were ever excluded, early queries would have a fully
    masked row, softmax would produce NaN, and it would silently poison
    everything downstream.
    """
    q, k, v = make_qkv(1, 64, 1, 16)
    cfg = MoBAConfig(block_size=8, top_k=1)
    out = moba_attention_naive(q, k, v, cfg)
    assert torch.isfinite(out).all()
