"""Reconciling a query head count with a smaller key/value head count.

Grouped-query attention is defined entirely by shapes here: the group size is
`H_q // H_kv` and there is no configuration field for it. A config field would
be a second source of truth for something the tensors already state, and the
two could disagree. See `docs/superpowers/plans/2026-08-23-gqa-support.md`,
decision D-GQA-1.
"""

import pytest
import torch

from moba.config import MoBAConfig
from moba.heads import expand_kv, head_group_size
from moba.ref.naive import moba_attention_naive, moba_selection


def test_equal_head_counts_are_a_group_of_one():
    """Plain multi-head attention must remain the G=1 special case."""
    assert head_group_size(8, 8) == 1


def test_group_size_is_the_ratio():
    assert head_group_size(32, 8) == 4
    assert head_group_size(8, 1) == 8


def test_indivisible_head_counts_are_rejected():
    """A non-integer group has no meaning; fail loudly rather than truncate."""
    with pytest.raises(ValueError, match="divisible"):
        head_group_size(8, 3)


def test_more_kv_heads_than_query_heads_is_rejected():
    with pytest.raises(ValueError, match="divisible"):
        head_group_size(4, 8)


def test_nonpositive_head_counts_are_rejected():
    with pytest.raises(ValueError, match="positive"):
        head_group_size(8, 0)


def test_expand_kv_repeats_each_head_to_cover_its_group():
    """Head g of the expansion must be head g//G of the original, exactly."""
    # Token-major [B, S, H, D], which is `expand_kv`'s default axis.
    t = torch.arange(2 * 3 * 2 * 4, dtype=torch.float32).view(2, 3, 2, 4)
    wide = expand_kv(t, 6)
    assert wide.shape == (2, 3, 6, 4)
    for qh in range(6):
        assert torch.equal(wide[:, :, qh], t[:, :, qh // 3])

    # dim=1 for the buffers that stayed head-major -- block representatives,
    # selection masks, partials. Same mapping, different axis.
    h = torch.arange(2 * 2 * 3 * 4, dtype=torch.float32).view(2, 2, 3, 4)
    wide_h = expand_kv(h, 6, dim=1)
    assert wide_h.shape == (2, 6, 3, 4)
    for qh in range(6):
        assert torch.equal(wide_h[:, qh], h[:, qh // 3])


def test_expand_kv_is_the_identity_at_equal_head_counts():
    """The common path must copy nothing, not copy cheaply."""
    t = torch.randn(1, 8, 4, 16)
    assert expand_kv(t, 4) is t


def test_oracle_under_gqa_equals_the_oracle_on_expanded_keys(device):
    """The definition of GQA: identical to MHA over repeated K/V heads.

    Checked bitwise, not approximately. `expand_kv` produces the very same
    numbers the GQA path must consume, so any difference is a routing or
    indexing bug rather than an arithmetic one, and rounding cannot explain it.
    """
    cfg = MoBAConfig(block_size=64, top_k=4, dtype=torch.float64)
    b, h_q, h_kv, s, d = 1, 8, 2, 512, 64
    gen = torch.Generator(device=device).manual_seed(0)
    q = torch.randn(b, s, h_q, d, generator=gen, device=device,
                    dtype=torch.float64)
    k = torch.randn(b, s, h_kv, d, generator=gen, device=device,
                    dtype=torch.float64)
    v = torch.randn(b, s, h_kv, d, generator=gen, device=device,
                    dtype=torch.float64)

    narrow = moba_attention_naive(q, k, v, cfg)
    wide = moba_attention_naive(q, expand_kv(k, h_q), expand_kv(v, h_q), cfg)
    assert torch.equal(narrow, wide)

    sel_narrow = moba_selection(q, k, cfg)
    sel_wide = moba_selection(q, expand_kv(k, h_q), cfg)
    assert sel_narrow.shape == (b, h_q, s, cfg.num_blocks(s))
    assert torch.equal(sel_narrow, sel_wide)
