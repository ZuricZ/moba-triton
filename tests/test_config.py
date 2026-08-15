"""Tests for MoBAConfig and device probing."""

import pytest
import torch

from moba.config import MoBAConfig
from moba.device import DeviceInfo, supports_dtype


def test_num_blocks_rounds_up():
    cfg = MoBAConfig(block_size=64, top_k=2)
    assert cfg.num_blocks(128) == 2
    assert cfg.num_blocks(129) == 3
    assert cfg.num_blocks(1) == 1


def test_config_is_frozen():
    """Configs are hashable identifiers for benchmark rows; mutation would
    silently invalidate results already recorded under them."""
    cfg = MoBAConfig(block_size=64, top_k=2)
    with pytest.raises(Exception):
        cfg.block_size = 32


def test_config_defaults_match_spec():
    cfg = MoBAConfig(block_size=32, top_k=4)
    assert cfg.dtype is torch.bfloat16
    assert cfg.always_select_current is True
    assert cfg.tie_break_low_index is True
    assert cfg.pool_masked_only is False


def test_config_rejects_nonpositive():
    with pytest.raises(ValueError):
        MoBAConfig(block_size=0, top_k=2)
    with pytest.raises(ValueError):
        MoBAConfig(block_size=32, top_k=0)


def test_supports_dtype_bf16_needs_ampere():
    ok, reason = supports_dtype(torch.bfloat16, (7, 5))
    assert ok is False
    assert "sm_80" in reason

    ok, reason = supports_dtype(torch.bfloat16, (8, 6))
    assert ok is True
    assert reason == ""


def test_supports_dtype_rejects_unknown_dtype():
    ok, reason = supports_dtype(torch.int8, (8, 6))
    assert ok is False
    assert reason


def test_device_probe_populates_row():
    info = DeviceInfo.probe(0)
    row = info.as_row()
    assert row["gpu_name"]
    assert row["capability"].startswith("sm_")
    assert int(row["total_memory_mb"]) > 0


def test_device_row_keys_are_stable():
    """BenchRow splats these keys directly; renaming one breaks the suite."""
    row = DeviceInfo.probe(0).as_row()
    assert set(row) == {
        "gpu_name",
        "capability",
        "total_memory_mb",
        "torch_version",
        "triton_version",
        "cuda_version",
    }
