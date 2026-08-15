"""Tests for benchmark bookkeeping.

A benchmark's numbers are worthless if the denominators are wrong, and
denominators are exactly the kind of thing that goes wrong silently. The
sharpest test here checks the analytic sparsity formula against what the
oracle actually selects -- two independent derivations of the same quantity.
"""

import math

import pytest
import torch

from bench.report import to_csv, to_markdown
from bench.suite import (
    BenchRow,
    effective_tflops,
    estimate_memory_mb,
    sparsity_ratio,
)
from moba.config import MoBAConfig
from moba.ref.naive import moba_selection

from conftest import make_qkv


# --------------------------------------------------------------------------
# Sparsity: the benchmark's central denominator
# --------------------------------------------------------------------------


def test_sparsity_ratio_is_one_when_every_block_is_selectable():
    cfg = MoBAConfig(block_size=16, top_k=1000)
    assert sparsity_ratio(256, cfg) == pytest.approx(1.0)


def test_sparsity_ratio_shrinks_with_smaller_top_k():
    small = MoBAConfig(block_size=16, top_k=2)
    large = MoBAConfig(block_size=16, top_k=8)
    assert sparsity_ratio(1024, small) < sparsity_ratio(1024, large)


def test_sparsity_ratio_shrinks_with_smaller_blocks():
    """The project's premise: at fixed top_k, smaller blocks attend less."""
    previous = None
    for block_size in (128, 64, 32, 16):
        r = sparsity_ratio(2048, MoBAConfig(block_size=block_size, top_k=4))
        if previous is not None:
            assert r < previous
        previous = r


@pytest.mark.parametrize("top_k", [1, 2, 4, 16])
def test_sparsity_ratio_within_unit_interval(top_k):
    r = sparsity_ratio(1024, MoBAConfig(block_size=32, top_k=top_k))
    assert 0.0 < r <= 1.0


@pytest.mark.parametrize(
    "seq_len,block_size,top_k",
    [(256, 32, 2), (512, 16, 4), (1024, 64, 2), (129, 32, 3)],
)
def test_analytic_sparsity_matches_the_oracle(seq_len, block_size, top_k, device):
    """Cross-checks the formula against real selection masks.

    The formula counts keys the gate *should* attend; the oracle produces the
    mask it *does*. They are derived independently, so agreement is real
    evidence. A mismatch means the benchmark is normalizing by a sparsity the
    kernel never had -- which would corrupt every derived number.
    """
    cfg = MoBAConfig(block_size=block_size, top_k=top_k)
    q, k, _ = make_qkv((1, 2, seq_len, 32), dtype=torch.float64, device=device)
    sel = moba_selection(q, k, cfg)

    pos = torch.arange(seq_len, device=sel.device)
    block_of_token = pos // block_size
    visible = sel[..., block_of_token]
    causal = pos.view(-1, 1) >= pos.view(1, -1)
    attended = (visible & causal).sum().item()
    causal_total = seq_len * (seq_len + 1) / 2 * sel.shape[0] * sel.shape[1]

    assert attended / causal_total == pytest.approx(
        sparsity_ratio(seq_len, cfg), rel=1e-9
    )


# --------------------------------------------------------------------------
# Memory and throughput bookkeeping
# --------------------------------------------------------------------------


def test_memory_estimate_scales_linearly_in_seq_len_when_chunked():
    """With a fixed query chunk, peak is O(chunk * S), not O(S^2)."""
    a = estimate_memory_mb(1, 4, 512, 64, torch.bfloat16, query_chunk=256)
    b = estimate_memory_mb(1, 4, 1024, 64, torch.bfloat16, query_chunk=256)
    assert b / a == pytest.approx(2.0, rel=0.15)


def test_memory_estimate_is_quadratic_without_chunking():
    a = estimate_memory_mb(1, 4, 512, 64, torch.bfloat16, query_chunk=None)
    b = estimate_memory_mb(1, 4, 1024, 64, torch.bfloat16, query_chunk=None)
    assert b / a == pytest.approx(4.0, rel=0.15)


def test_effective_tflops_is_positive_and_scales_with_speed():
    cfg = MoBAConfig(block_size=32, top_k=2)
    slow = effective_tflops(1, 4, 1024, 64, cfg, ms=2.0)
    fast = effective_tflops(1, 4, 1024, 64, cfg, ms=1.0)
    assert slow > 0
    assert fast == pytest.approx(2 * slow)


def test_effective_tflops_counts_only_attended_work():
    """A sparser config does less work, so at equal latency it is fewer FLOPs."""
    dense = effective_tflops(1, 4, 1024, 64, MoBAConfig(block_size=32, top_k=32), 1.0)
    sparse = effective_tflops(1, 4, 1024, 64, MoBAConfig(block_size=32, top_k=2), 1.0)
    assert sparse < dense


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------


def _row(**overrides):
    base = dict(
        backend="naive_torch", batch=1, heads=4, seq_len=512, head_dim=64,
        block_size=32, top_k=2, dtype="bfloat16", latency_ms=1.5, tflops=2.0,
        peak_memory_mb=100.0, sparsity=0.25, speedup_vs_dense=1.1,
        speedup_ceiling=4.0, attainment=0.275, gpu_name="RTX 3050 Ti",
        capability="sm_86", total_memory_mb=4095, torch_version="2.13.0+cu126",
        triton_version="3.7.1", cuda_version="12.6", status="ok", note="",
    )
    base.update(overrides)
    return BenchRow(**base)


def test_markdown_report_includes_skips_and_device():
    rows = [
        _row(),
        _row(
            backend="flash_moba", latency_ms=math.nan, tflops=math.nan,
            peak_memory_mb=math.nan, speedup_vs_dense=math.nan,
            attainment=math.nan, status="skipped",
            note="extension not built; setup.py targets archs 80;90;100;120",
        ),
    ]
    md = to_markdown(rows)
    assert "naive_torch" in md
    assert "flash_moba" in md
    assert "not built" in md, "skip reasons must survive into the report"
    assert "sm_86" in md


def test_csv_roundtrip_preserves_every_field(tmp_path):
    import csv
    from dataclasses import fields

    path = tmp_path / "r.csv"
    to_csv([_row()], path)
    with path.open(encoding="utf-8") as fh:
        got = list(csv.DictReader(fh))
    assert len(got) == 1
    assert set(got[0]) == {f.name for f in fields(BenchRow)}
    assert got[0]["backend"] == "naive_torch"


def test_empty_report_does_not_crash():
    assert to_markdown([])
