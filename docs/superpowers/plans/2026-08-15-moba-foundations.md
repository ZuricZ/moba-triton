# MoBA Foundations Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the test harness, benchmark suite, and reference-reading apparatus needed before writing a MoBA Triton kernel.

**Architecture:** A spec-first PyTorch oracle defines MoBA's semantics executably. Every implementation — ours and vendored — conforms to one `Backend` protocol with self-declared availability, so the same correctness and benchmark suites run locally with a subset and on rented hardware with the full set.

**Tech Stack:** Python 3.10+, PyTorch (bf16, sm_80+), Triton, pytest, `triton.testing.do_bench`.

**Spec:** `docs/superpowers/specs/2026-08-15-moba-foundations-design.md`

## Global Constraints

- `src/` layout, package name `moba`. Import as `from moba...`.
- Primary dtype `torch.bfloat16`; secondary `torch.float16`. Minimum compute capability **8.0**.
- Environment managed by **uv** (installed at `~/.local/bin/uv` in WSL). Never conda: Miniconda is installed on *Windows* and its `win-64` binaries cannot run under WSL's Linux kernel.
- Python **3.13**. torch **must** come from the **cu126** index — default wheels are `+cu130` and fail CUDA init against this machine's 12.7 driver.
- All GPU commands run in WSL2: `wsl -e bash -lc 'cd "/mnt/c/D/Side Projects/GPU Kernel" && <cmd>'`
- Inside WSL, invoke Python as `uv run <cmd>`, never bare `python`.
- Local VRAM budget: 4 GB. Never write a test config exceeding it.
- `references/` is gitignored, read-only, never imported as a package.
- `TOLERANCE_FACTOR = 2.0`, `TOLERANCE_FLOOR = 1e-6`.
- Commit messages: no Claude/Anthropic co-author lines.

---

### Task 1: Environment and project skeleton

**Files:**
- Create: `pyproject.toml`
- Create: `src/moba/__init__.py`
- Create: `scripts/setup_env.sh`
- Create: `tests/test_smoke.py`
- Modify: `.gitignore` (add `references/`, `results/`)

**Interfaces:**
- Consumes: nothing.
- Produces: an importable `moba` package with `moba.__version__: str`; a uv-managed `.venv/` inside WSL on Python 3.13 with `torch` (cu126) and `triton` importable and CUDA available; a committed `uv.lock`.

- [ ] **Step 1: Write `.gitattributes`**

CRLF endings break shell scripts under WSL with `bad interpreter: ...^M`. Fix this before any `.sh` file is committed.

```
* text=auto
*.sh text eol=lf
*.py text eol=lf
```

- [ ] **Step 2: Verify uv is present**

Run: `wsl -e bash -lc 'export PATH="$HOME/.local/bin:$PATH"; uv --version'`
Expected: `uv 0.12.x`. If missing: `curl -LsSf https://astral.sh/uv/install.sh | sh`.

- [ ] **Step 3: Write `pyproject.toml`**

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "moba"
version = "0.1.0"
description = "MoBA (Mixture of Block Attention) Triton kernel — learning project"
requires-python = ">=3.13"
dependencies = ["torch", "triton", "numpy"]

[dependency-groups]
dev = ["pytest"]

[tool.setuptools.packages.find]
where = ["src"]

# torch must come from the cu126 index. Default PyPI wheels are +cu130 and
# fail CUDA init on this machine's 12.7 driver: CUDA guarantees minor-version
# compatibility (any 12.x runtime on any 12.0+ driver) but not major.
[tool.uv.sources]
torch = { index = "pytorch-cu126" }

[[tool.uv.index]]
name = "pytorch-cu126"
url = "https://download.pytorch.org/whl/cu126"
explicit = true

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["tests"]
addopts = "-v --strict-markers"
markers = [
    "slow: benchmarks or large shapes, deselect with -m 'not slow'",
]
```

- [ ] **Step 4: Write `src/moba/__init__.py`**

```python
"""MoBA (Mixture of Block Attention) kernels and reference implementations."""

__version__ = "0.1.0"
```

- [ ] **Step 5: Write the failing smoke test**

```python
# tests/test_smoke.py
"""Smoke tests: the package imports and the GPU meets the project's floor."""

import pytest
import torch

import moba


def test_package_imports():
    assert moba.__version__ == "0.1.0"


def test_cuda_available():
    assert torch.cuda.is_available(), "No CUDA device visible from WSL"


def test_gpu_meets_compute_capability_floor():
    major, minor = torch.cuda.get_device_capability(0)
    assert (major, minor) >= (8, 0), (
        f"Project requires sm_80+ (Ampere); found sm_{major}{minor}"
    )


def test_bfloat16_supported():
    x = torch.ones(8, 8, device="cuda", dtype=torch.bfloat16)
    assert (x @ x).dtype is torch.bfloat16
```

- [ ] **Step 6: Run tests to verify they fail**

Run: `wsl -e bash -lc 'cd "/mnt/c/D/Side Projects/GPU Kernel" && uv run pytest tests/test_smoke.py -v'`
Expected: FAIL with `ModuleNotFoundError: No module named 'moba'` (or a collection error).

- [ ] **Step 7: Sync the environment**

Run: `wsl -e bash -lc 'cd "/mnt/c/D/Side Projects/GPU Kernel" && uv sync'`
Expected: creates `.venv/` on Python 3.13, installs `torch==2.13.0+cu126`, `triton==3.7.1`, numpy, pytest, and the local `moba` package in editable mode. Writes `uv.lock`.

- [ ] **Step 8: Run tests to verify they pass**

Run: `wsl -e bash -lc 'cd "/mnt/c/D/Side Projects/GPU Kernel" && uv run pytest tests/test_smoke.py -v'`
Expected: 4 passed. If `cuda_available` fails with "driver too old", the cu126 index pin in `pyproject.toml` did not take effect — check `uv run python -c "import torch;print(torch.__version__)"` shows `+cu126`, not `+cu130`.

- [ ] **Step 9: Add build artifacts to `.gitignore`**

Append these lines to `.gitignore`:

```
references/
results/
.venv/
build/
dist/
```

- [ ] **Step 10: Commit**

```bash
git add pyproject.toml uv.lock .gitattributes src/moba/__init__.py tests/test_smoke.py .gitignore
git commit -m "feat: project skeleton, uv env on py3.13 + cu126, GPU capability smoke tests"
```

---

### Task 2: MoBAConfig and device probing

**Files:**
- Create: `src/moba/config.py`
- Create: `src/moba/device.py`
- Create: `tests/test_config.py`

**Interfaces:**
- Consumes: `moba` package from Task 1.
- Produces:
  - `MoBAConfig` frozen dataclass with fields `block_size: int`, `top_k: int`, `dtype: torch.dtype = torch.bfloat16`, `always_select_current: bool = True`, `tie_break_low_index: bool = True`, `pool_masked_only: bool = False`; method `num_blocks(seq_len: int) -> int`.
  - `DeviceInfo` frozen dataclass with fields `name: str`, `capability: tuple[int, int]`, `total_memory_mb: int`, `torch_version: str`, `triton_version: str`, `driver: str`; classmethod `DeviceInfo.probe(index: int = 0) -> DeviceInfo`; method `as_row() -> dict[str, str | int]`.
  - `supports_dtype(dtype: torch.dtype, capability: tuple[int, int]) -> tuple[bool, str]`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_config.py
"""Tests for MoBAConfig and device probing."""

import math

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
    assert "8.0" in reason or "sm_80" in reason

    ok, reason = supports_dtype(torch.bfloat16, (8, 6))
    assert ok is True
    assert reason == ""


def test_device_probe_populates_row():
    info = DeviceInfo.probe(0)
    row = info.as_row()
    assert row["gpu_name"]
    assert row["capability"].startswith("sm_")
    assert int(row["total_memory_mb"]) > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `wsl -e bash -lc 'cd "/mnt/c/D/Side Projects/GPU Kernel" && .venv/bin/python -m pytest tests/test_config.py -v'`
Expected: FAIL with `ModuleNotFoundError: No module named 'moba.config'`.

- [ ] **Step 3: Write `src/moba/config.py`**

```python
"""Configuration for MoBA attention.

The semantics flags in `MoBAConfig` exist so that a disagreement with a
reference implementation can be diagnosed by flipping a single field rather
than by reading someone else's kernel. See the design spec, section
"MoBA semantics".
"""

import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class MoBAConfig:
    """Parameters defining one MoBA attention configuration.

    Attributes:
        block_size: Number of key positions per block.
        top_k: Number of past blocks selected per query, excluding the
            query's own block.
        dtype: Compute dtype for the attention itself.
        always_select_current: If True, the block containing the query is
            always attended (causally masked) and does not consume a top_k
            slot.
        tie_break_low_index: If True, equal gate scores resolve toward the
            lower block index.
        pool_masked_only: If True, a block's representative key averages only
            causally-visible positions. If False, it averages the whole block.
    """

    block_size: int
    top_k: int
    dtype: torch.dtype = torch.bfloat16
    always_select_current: bool = True
    tie_break_low_index: bool = True
    pool_masked_only: bool = False

    def __post_init__(self) -> None:
        if self.block_size <= 0:
            raise ValueError(f"block_size must be positive, got {self.block_size}")
        if self.top_k <= 0:
            raise ValueError(f"top_k must be positive, got {self.top_k}")

    def num_blocks(self, seq_len: int) -> int:
        """Returns the number of blocks covering `seq_len` positions."""
        return math.ceil(seq_len / self.block_size)
```

- [ ] **Step 4: Write `src/moba/device.py`**

```python
"""GPU capability probing and result metadata.

Every benchmark row carries a DeviceInfo so that results gathered on the
local laptop and on rented hardware merge into one comparable table.
"""

from dataclasses import dataclass

import torch

MIN_CAPABILITY = (8, 0)


def supports_dtype(
    dtype: torch.dtype, capability: tuple[int, int]
) -> tuple[bool, str]:
    """Reports whether a compute dtype is usable on a given GPU architecture.

    Args:
        dtype: The candidate compute dtype.
        capability: CUDA compute capability as (major, minor).

    Returns:
        A tuple of (supported, reason). `reason` is empty when supported.
    """
    if dtype is torch.bfloat16 and capability < (8, 0):
        return False, f"bfloat16 requires sm_80+, found sm_{capability[0]}{capability[1]}"
    if dtype is torch.float16 and capability < (7, 0):
        return False, f"float16 tensor cores require sm_70+, found sm_{capability[0]}{capability[1]}"
    if dtype not in (torch.bfloat16, torch.float16, torch.float32, torch.float64):
        return False, f"unsupported dtype {dtype}"
    return True, ""


@dataclass(frozen=True)
class DeviceInfo:
    """Identifying metadata for the GPU a result was measured on."""

    name: str
    capability: tuple[int, int]
    total_memory_mb: int
    torch_version: str
    triton_version: str
    driver: str

    @classmethod
    def probe(cls, index: int = 0) -> "DeviceInfo":
        """Reads device metadata from the running CUDA context.

        Args:
            index: CUDA device index.

        Returns:
            A populated DeviceInfo.

        Raises:
            RuntimeError: If no CUDA device is available.
        """
        if not torch.cuda.is_available():
            raise RuntimeError("No CUDA device available")
        props = torch.cuda.get_device_properties(index)
        try:
            import triton

            triton_version = triton.__version__
        except ImportError:
            triton_version = "unavailable"
        driver = getattr(torch.version, "cuda", None) or "unknown"
        return cls(
            name=props.name,
            capability=(props.major, props.minor),
            total_memory_mb=props.total_memory // (1024 * 1024),
            torch_version=torch.__version__,
            triton_version=triton_version,
            driver=driver,
        )

    def as_row(self) -> dict[str, str | int]:
        """Returns this device's metadata as flat columns for a results table."""
        return {
            "gpu_name": self.name,
            "capability": f"sm_{self.capability[0]}{self.capability[1]}",
            "total_memory_mb": self.total_memory_mb,
            "torch_version": self.torch_version,
            "triton_version": self.triton_version,
            "cuda_version": self.driver,
        }
```

- [ ] **Step 5: Run test to verify it passes**

Run: `wsl -e bash -lc 'cd "/mnt/c/D/Side Projects/GPU Kernel" && .venv/bin/python -m pytest tests/test_config.py -v'`
Expected: 6 passed.

- [ ] **Step 6: Commit**

```bash
git add src/moba/config.py src/moba/device.py tests/test_config.py
git commit -m "feat: MoBAConfig with explicit semantics flags, device probing"
```

---

### Task 3: The spec-first oracle and Tier 1 exact identities

This is the most important task in the plan. The oracle is the executable definition of MoBA; everything downstream is measured against it.

**Files:**
- Create: `src/moba/ref/__init__.py`
- Create: `src/moba/ref/naive.py`
- Create: `tests/test_semantics.py`

**Interfaces:**
- Consumes: `MoBAConfig` from Task 2.
- Produces:
  - `block_representatives(k: Tensor, cfg: MoBAConfig) -> Tensor` returning `[B, H, num_blocks, D]`.
  - `select_blocks(q: Tensor, kbar: Tensor, cfg: MoBAConfig) -> Tensor` returning a bool mask `[B, H, S, num_blocks]`.
  - `moba_attention_naive(q: Tensor, k: Tensor, v: Tensor, cfg: MoBAConfig) -> Tensor` returning `[B, H, S, D]`.
  - All accept `q, k, v` shaped `[B, H, S, D]`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_semantics.py
"""Tier 1: exact identities. No tolerance argument is permitted here.

These tests validate the oracle against properties that hold exactly, so a
failure is unambiguously a bug rather than a numerical artifact.
"""

import pytest
import torch
import torch.nn.functional as F

from moba.config import MoBAConfig
from moba.ref.naive import (
    block_representatives,
    moba_attention_naive,
    select_blocks,
)

DEVICE = "cuda"


def make_qkv(b, h, s, d, dtype=torch.float64, seed=0):
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    shape = (b, h, s, d)
    q = torch.randn(shape, generator=g, device=DEVICE, dtype=dtype)
    k = torch.randn(shape, generator=g, device=DEVICE, dtype=dtype)
    v = torch.randn(shape, generator=g, device=DEVICE, dtype=dtype)
    return q, k, v


@pytest.mark.parametrize("seq_len,block_size", [(128, 32), (129, 32), (64, 64)])
def test_block_representatives_are_block_means(seq_len, block_size):
    _, k, _ = make_qkv(1, 2, seq_len, 16)
    cfg = MoBAConfig(block_size=block_size, top_k=1)
    kbar = block_representatives(k, cfg)

    assert kbar.shape == (1, 2, cfg.num_blocks(seq_len), 16)
    # First block is a plain mean of its keys.
    expected_first = k[:, :, :block_size].mean(dim=2)
    torch.testing.assert_close(kbar[:, :, 0], expected_first)
    # Final (possibly short) block averages only the positions that exist.
    tail = k[:, :, (cfg.num_blocks(seq_len) - 1) * block_size :]
    torch.testing.assert_close(kbar[:, :, -1], tail.mean(dim=2))


def test_current_block_always_selected():
    q, k, _ = make_qkv(1, 1, 128, 16)
    cfg = MoBAConfig(block_size=32, top_k=1)
    kbar = block_representatives(k, cfg)
    sel = select_blocks(q, kbar, cfg)

    pos = torch.arange(128, device=DEVICE)
    own_block = pos // 32
    assert sel[0, 0, pos, own_block].all()


def test_future_blocks_never_selected():
    q, k, _ = make_qkv(1, 1, 128, 16)
    cfg = MoBAConfig(block_size=32, top_k=8)
    kbar = block_representatives(k, cfg)
    sel = select_blocks(q, kbar, cfg)

    pos = torch.arange(128, device=DEVICE)
    own_block = pos // 32
    blocks = torch.arange(cfg.num_blocks(128), device=DEVICE)
    is_future = blocks.view(1, -1) > own_block.view(-1, 1)
    assert not sel[0, 0][is_future].any()


def test_selection_count_respects_top_k():
    q, k, _ = make_qkv(1, 1, 256, 16)
    cfg = MoBAConfig(block_size=32, top_k=2)
    kbar = block_representatives(k, cfg)
    sel = select_blocks(q, kbar, cfg)

    # A query in the last block has 7 past candidates; it may select at most
    # top_k of them, plus its own block.
    last_query = 255
    assert sel[0, 0, last_query].sum().item() == 3


def test_reduces_to_dense_causal_attention_when_top_k_covers_all_blocks():
    """The keystone identity: with every block selectable, MoBA is dense."""
    b, h, s, d = 1, 2, 128, 32
    q, k, v = make_qkv(b, h, s, d)
    cfg = MoBAConfig(block_size=16, top_k=s // 16)  # top_k >= num_blocks

    got = moba_attention_naive(q, k, v, cfg)
    want = F.scaled_dot_product_attention(q, k, v, is_causal=True)

    torch.testing.assert_close(got, want, rtol=1e-10, atol=1e-10)


def test_single_block_equals_plain_causal_attention():
    b, h, s, d = 1, 1, 32, 16
    q, k, v = make_qkv(b, h, s, d)
    cfg = MoBAConfig(block_size=32, top_k=1)  # one block total

    got = moba_attention_naive(q, k, v, cfg)
    want = F.scaled_dot_product_attention(q, k, v, is_causal=True)

    torch.testing.assert_close(got, want, rtol=1e-10, atol=1e-10)


def test_ties_resolve_to_lower_block_index():
    """Identical keys across blocks make every gate score equal."""
    b, h, s, d = 1, 1, 128, 16
    q = torch.randn(b, h, s, d, device=DEVICE, dtype=torch.float64)
    # Every key identical => every block representative identical => all ties.
    k = torch.ones(b, h, s, d, device=DEVICE, dtype=torch.float64)
    cfg = MoBAConfig(block_size=32, top_k=1)

    kbar = block_representatives(k, cfg)
    sel = select_blocks(q, kbar, cfg)

    # Query in the last block (own block 3) has candidates {0,1,2}, all tied.
    # Lower index wins, so block 0 is chosen.
    assert sel[0, 0, 127, 0].item() is True
    assert sel[0, 0, 127, 1].item() is False
    assert sel[0, 0, 127, 2].item() is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `wsl -e bash -lc 'cd "/mnt/c/D/Side Projects/GPU Kernel" && .venv/bin/python -m pytest tests/test_semantics.py -v'`
Expected: FAIL with `ModuleNotFoundError: No module named 'moba.ref'`.

- [ ] **Step 3: Write `src/moba/ref/__init__.py`**

```python
"""Reference implementations of MoBA attention."""
```

- [ ] **Step 4: Write `src/moba/ref/naive.py`**

```python
"""Naive PyTorch MoBA — the project's oracle.

This is deliberately slow and deliberately obvious. It materializes the full
[S, S] attention matrix, which is exactly what the real kernel exists to
avoid, and that is the point: it is the executable definition of what the
kernel must compute, not a competitor to it.

Semantics implemented here are specified in
`docs/superpowers/specs/2026-08-15-moba-foundations-design.md`, section
"MoBA semantics".
"""

import torch
import torch.nn.functional as F

from moba.config import MoBAConfig


def block_representatives(k: torch.Tensor, cfg: MoBAConfig) -> torch.Tensor:
    """Computes the mean-pooled key representing each block.

    A short final block averages only the positions that exist, so the
    representative is not diluted by padding.

    Args:
        k: Keys shaped [B, H, S, D].
        cfg: Configuration supplying `block_size`.

    Returns:
        Block representatives shaped [B, H, num_blocks, D].
    """
    b, h, s, d = k.shape
    bk = cfg.block_size
    num_blocks = cfg.num_blocks(s)
    pad = num_blocks * bk - s

    k_padded = F.pad(k, (0, 0, 0, pad))
    sums = k_padded.view(b, h, num_blocks, bk, d).sum(dim=3)

    counts = torch.full((num_blocks,), bk, device=k.device, dtype=k.dtype)
    counts[-1] = bk - pad
    return sums / counts.view(1, 1, num_blocks, 1)


def select_blocks(
    q: torch.Tensor, kbar: torch.Tensor, cfg: MoBAConfig
) -> torch.Tensor:
    """Selects which blocks each query attends to.

    A block is a candidate only if it lies wholly in the query's past. The
    query's own block is always selected and does not consume a top_k slot.

    Args:
        q: Queries shaped [B, H, S, D].
        kbar: Block representatives shaped [B, H, num_blocks, D].
        cfg: Configuration supplying `block_size`, `top_k`, and semantics flags.

    Returns:
        Boolean selection mask shaped [B, H, S, num_blocks].
    """
    b, h, s, _ = q.shape
    num_blocks = kbar.shape[2]
    device = q.device

    # Gate scores. No softmax scaling: a positive monotone transform cannot
    # change a top-k ranking, so applying it here would be wasted work.
    scores = torch.einsum("bhsd,bhnd->bhsn", q, kbar)

    pos = torch.arange(s, device=device)
    own_block = pos // cfg.block_size                      # [S]
    blocks = torch.arange(num_blocks, device=device)       # [num_blocks]
    is_past = blocks.view(1, -1) < own_block.view(-1, 1)   # [S, num_blocks]
    is_current = blocks.view(1, -1) == own_block.view(-1, 1)

    masked = scores.masked_fill(~is_past.view(1, 1, s, num_blocks), float("-inf"))

    # A stable descending sort makes equal scores resolve toward the lower
    # block index, which is the tie-break rule the spec fixes. torch.topk
    # gives no such guarantee.
    if cfg.tie_break_low_index:
        order = torch.argsort(masked, dim=-1, descending=True, stable=True)
    else:
        order = torch.argsort(masked, dim=-1, descending=True)

    take = min(cfg.top_k, num_blocks)
    chosen = order[..., :take]

    selected = torch.zeros_like(masked, dtype=torch.bool)
    selected.scatter_(-1, chosen, True)

    # Positions that were never candidates can still land in the first `take`
    # slots when a query has fewer than top_k past blocks; drop them.
    selected &= is_past.view(1, 1, s, num_blocks)

    if cfg.always_select_current:
        selected |= is_current.view(1, 1, s, num_blocks)
    return selected


def moba_attention_naive(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, cfg: MoBAConfig
) -> torch.Tensor:
    """Computes MoBA attention the slow, obvious way.

    Args:
        q: Queries shaped [B, H, S, D].
        k: Keys shaped [B, H, S, D].
        v: Values shaped [B, H, S, D].
        cfg: Configuration.

    Returns:
        Attention output shaped [B, H, S, D], same dtype as `q`.
    """
    b, h, s, d = q.shape
    scale = d ** -0.5

    kbar = block_representatives(k, cfg)
    selected = select_blocks(q, kbar, cfg)          # [B, H, S, num_blocks]

    # Expand block-level selection to token-level visibility.
    pos = torch.arange(s, device=q.device)
    block_of_token = pos // cfg.block_size          # [S]
    visible = selected[..., block_of_token]         # [B, H, S, S]

    # Causal masking. Only the current block needs it — every other selected
    # block lies wholly in the past — but applying it globally is equivalent
    # and removes a special case.
    causal = pos.view(-1, 1) >= pos.view(1, -1)     # [S, S]
    mask = visible & causal.view(1, 1, s, s)

    logits = torch.einsum("bhid,bhjd->bhij", q, k) * scale
    logits = logits.masked_fill(~mask, float("-inf"))
    probs = torch.softmax(logits, dim=-1)
    return torch.einsum("bhij,bhjd->bhid", probs, v)
```

- [ ] **Step 5: Run test to verify it passes**

Run: `wsl -e bash -lc 'cd "/mnt/c/D/Side Projects/GPU Kernel" && .venv/bin/python -m pytest tests/test_semantics.py -v'`
Expected: 10 passed (3 parametrized + 7).

- [ ] **Step 6: Commit**

```bash
git add src/moba/ref/ tests/test_semantics.py
git commit -m "feat: spec-first naive MoBA oracle with exact-identity tests"
```

---

### Task 4: Backend protocol, registry, and the two always-available backends

**Files:**
- Create: `src/moba/backends/__init__.py`
- Create: `src/moba/backends/base.py`
- Create: `src/moba/backends/naive_torch.py`
- Create: `src/moba/backends/sdpa_dense.py`
- Create: `tests/test_registry.py`

**Interfaces:**
- Consumes: `MoBAConfig` (Task 2), `moba_attention_naive` (Task 3), `supports_dtype` (Task 2).
- Produces:
  - `Backend` Protocol with `name: str`, `is_available(cfg) -> tuple[bool, str]`, `forward(q, k, v, cfg) -> Tensor`.
  - `register(backend: Backend) -> Backend` decorator-compatible registrar.
  - `all_backends() -> list[Backend]`.
  - `available_backends(cfg) -> list[Backend]`.
  - `skipped_backends(cfg) -> list[tuple[str, str]]` returning `(name, reason)`.
  - `get_backend(name: str) -> Backend`.
  - Registered names: `"naive_torch"`, `"sdpa_dense"`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_registry.py
"""Tests for the backend protocol and registry."""

import pytest
import torch

from moba.backends import (
    available_backends,
    all_backends,
    get_backend,
    skipped_backends,
)
from moba.config import MoBAConfig


def test_baseline_backends_are_registered():
    names = {b.name for b in all_backends()}
    assert {"naive_torch", "sdpa_dense"} <= names


def test_naive_torch_is_always_available():
    cfg = MoBAConfig(block_size=32, top_k=2)
    ok, reason = get_backend("naive_torch").is_available(cfg)
    assert ok, reason


def test_available_and_skipped_partition_the_registry():
    cfg = MoBAConfig(block_size=32, top_k=2)
    n_available = len(available_backends(cfg))
    n_skipped = len(skipped_backends(cfg))
    assert n_available + n_skipped == len(all_backends())


def test_skipped_backends_always_carry_a_reason():
    cfg = MoBAConfig(block_size=32, top_k=2)
    for name, reason in skipped_backends(cfg):
        assert reason, f"{name} was skipped with an empty reason"


def test_is_available_never_raises_on_absurd_config():
    cfg = MoBAConfig(block_size=1, top_k=999999, dtype=torch.float64)
    for backend in all_backends():
        ok, reason = backend.is_available(cfg)
        assert isinstance(ok, bool)
        assert isinstance(reason, str)


def test_unknown_backend_raises():
    with pytest.raises(KeyError):
        get_backend("does_not_exist")


def test_naive_and_sdpa_agree_when_top_k_covers_all_blocks():
    b, h, s, d = 1, 2, 64, 32
    g = torch.Generator(device="cuda").manual_seed(0)
    q, k, v = (
        torch.randn(b, h, s, d, generator=g, device="cuda", dtype=torch.float64)
        for _ in range(3)
    )
    cfg = MoBAConfig(block_size=16, top_k=s // 16, dtype=torch.float64)

    got = get_backend("naive_torch").forward(q, k, v, cfg)
    want = get_backend("sdpa_dense").forward(q, k, v, cfg)
    torch.testing.assert_close(got, want, rtol=1e-10, atol=1e-10)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `wsl -e bash -lc 'cd "/mnt/c/D/Side Projects/GPU Kernel" && .venv/bin/python -m pytest tests/test_registry.py -v'`
Expected: FAIL with `ModuleNotFoundError: No module named 'moba.backends'`.

- [ ] **Step 3: Write `src/moba/backends/base.py`**

```python
"""The Backend protocol and registry.

Every implementation of MoBA — ours, vendored, or baseline — is reachable
through this one interface, so the correctness harness and the benchmark
suite never special-case a particular implementation.

`is_available` returns a reason string rather than a bare bool. A backend
missing from a results table must say why it is missing; a silently short
table is worse than no table, because it looks complete.
"""

from typing import Protocol, runtime_checkable

import torch

from moba.config import MoBAConfig


@runtime_checkable
class Backend(Protocol):
    """One MoBA implementation, uniformly callable."""

    name: str

    def is_available(self, cfg: MoBAConfig) -> tuple[bool, str]:
        """Reports whether this backend can run `cfg` on this machine.

        Implementations must never raise: a missing import, an unsupported
        GPU architecture, or an unsupported dtype is a return value, not an
        exception.

        Args:
            cfg: The configuration to check.

        Returns:
            A tuple of (available, reason). `reason` is empty when available.
        """
        ...

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cfg: MoBAConfig,
    ) -> torch.Tensor:
        """Computes MoBA attention.

        Args:
            q: Queries shaped [B, H, S, D].
            k: Keys shaped [B, H, S, D].
            v: Values shaped [B, H, S, D].
            cfg: Configuration.

        Returns:
            Output shaped [B, H, S, D].
        """
        ...


_REGISTRY: dict[str, Backend] = {}


def register(backend: Backend) -> Backend:
    """Adds a backend to the global registry.

    Args:
        backend: The backend instance to register.

    Returns:
        The same backend, so this can be used inline.

    Raises:
        ValueError: If a backend with the same name is already registered.
    """
    if backend.name in _REGISTRY:
        raise ValueError(f"Backend {backend.name!r} is already registered")
    _REGISTRY[backend.name] = backend
    return backend


def all_backends() -> list[Backend]:
    """Returns every registered backend, available or not."""
    return list(_REGISTRY.values())


def get_backend(name: str) -> Backend:
    """Looks up a backend by name.

    Args:
        name: Registered backend name.

    Returns:
        The backend.

    Raises:
        KeyError: If no backend is registered under `name`.
    """
    if name not in _REGISTRY:
        raise KeyError(f"No backend named {name!r}; have {sorted(_REGISTRY)}")
    return _REGISTRY[name]


def available_backends(cfg: MoBAConfig) -> list[Backend]:
    """Returns the backends that can run `cfg` on this machine."""
    return [b for b in _REGISTRY.values() if b.is_available(cfg)[0]]


def skipped_backends(cfg: MoBAConfig) -> list[tuple[str, str]]:
    """Returns (name, reason) for each backend that cannot run `cfg`."""
    out = []
    for backend in _REGISTRY.values():
        ok, reason = backend.is_available(cfg)
        if not ok:
            out.append((backend.name, reason))
    return out
```

- [ ] **Step 4: Write `src/moba/backends/naive_torch.py`**

```python
"""The oracle, exposed as a backend."""

import torch

from moba.backends.base import register
from moba.config import MoBAConfig
from moba.ref.naive import moba_attention_naive


class NaiveTorchBackend:
    """Pure-PyTorch MoBA. Always available; the correctness ground truth."""

    name = "naive_torch"

    def is_available(self, cfg: MoBAConfig) -> tuple[bool, str]:
        return True, ""

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cfg: MoBAConfig,
    ) -> torch.Tensor:
        return moba_attention_naive(q, k, v, cfg)


register(NaiveTorchBackend())
```

- [ ] **Step 5: Write `src/moba/backends/sdpa_dense.py`**

```python
"""Dense causal attention baseline.

This is not a MoBA implementation. It is the roofline: the cost of attending
to everything. Every MoBA speedup is measured against it, and the ratio
between the speedup achieved and the speedup that sparsity implies is the
number this project cares about.
"""

import torch
import torch.nn.functional as F

from moba.backends.base import register
from moba.config import MoBAConfig


class SdpaDenseBackend:
    """torch.nn.functional.scaled_dot_product_attention, causal, dense."""

    name = "sdpa_dense"

    def is_available(self, cfg: MoBAConfig) -> tuple[bool, str]:
        if not torch.cuda.is_available():
            return False, "no CUDA device"
        return True, ""

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cfg: MoBAConfig,
    ) -> torch.Tensor:
        return F.scaled_dot_product_attention(q, k, v, is_causal=True)


register(SdpaDenseBackend())
```

- [ ] **Step 6: Write `src/moba/backends/__init__.py`**

```python
"""Backend registry. Importing this module registers all built-in backends."""

from moba.backends.base import (
    Backend,
    all_backends,
    available_backends,
    get_backend,
    register,
    skipped_backends,
)

# Import for side effect: each module registers its backend on import.
from moba.backends import naive_torch as _naive_torch  # noqa: F401
from moba.backends import sdpa_dense as _sdpa_dense  # noqa: F401

__all__ = [
    "Backend",
    "all_backends",
    "available_backends",
    "get_backend",
    "register",
    "skipped_backends",
]
```

- [ ] **Step 7: Run test to verify it passes**

Run: `wsl -e bash -lc 'cd "/mnt/c/D/Side Projects/GPU Kernel" && .venv/bin/python -m pytest tests/test_registry.py -v'`
Expected: 7 passed.

- [ ] **Step 8: Commit**

```bash
git add src/moba/backends/ tests/test_registry.py
git commit -m "feat: backend protocol, registry, naive and dense baselines"
```

---

### Task 5: Tolerance policy

**Files:**
- Create: `src/moba/tolerance.py`
- Create: `tests/test_tolerance.py`

**Interfaces:**
- Consumes: `MoBAConfig` (Task 2), `moba_attention_naive` (Task 3).
- Produces:
  - `TOLERANCE_FACTOR: float = 2.0`, `TOLERANCE_FLOOR: float = 1e-6`.
  - `reference_error(q, k, v, cfg) -> tuple[torch.Tensor, float]` returning `(truth_fp64, ref_error)` where `truth_fp64` is the fp64 oracle output and `ref_error` is the max absolute deviation of the same oracle run in `cfg.dtype`.
  - `assert_within_tolerance(actual, truth, ref_error, name) -> None` raising `AssertionError` on failure.
  - `max_abs_error(a: torch.Tensor, b: torch.Tensor) -> float`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_tolerance.py
"""Tests for the relative tolerance policy.

The policy's claim is that it accepts a correct low-precision implementation
and rejects a subtly wrong one. Both halves are tested.
"""

import pytest
import torch

from moba.config import MoBAConfig
from moba.ref.naive import moba_attention_naive
from moba.tolerance import (
    TOLERANCE_FACTOR,
    assert_within_tolerance,
    max_abs_error,
    reference_error,
)

DEVICE = "cuda"


def make_qkv(b, h, s, d, dtype=torch.float64, seed=0):
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    shape = (b, h, s, d)
    return tuple(
        torch.randn(shape, generator=g, device=DEVICE, dtype=dtype) for _ in range(3)
    )


def test_max_abs_error_is_dtype_agnostic():
    a = torch.zeros(4, device=DEVICE, dtype=torch.bfloat16)
    b = torch.full((4,), 0.5, device=DEVICE, dtype=torch.float64)
    assert max_abs_error(a, b) == pytest.approx(0.5)


def test_reference_error_is_positive_for_bf16():
    q, k, v = make_qkv(1, 2, 128, 32)
    cfg = MoBAConfig(block_size=32, top_k=2, dtype=torch.bfloat16)
    truth, ref_err = reference_error(q, k, v, cfg)

    assert truth.dtype is torch.float64
    assert ref_err > 0, "bf16 must lose precision relative to fp64"
    assert ref_err < 1.0, "error this large means the oracle is wrong, not imprecise"


def test_correct_bf16_implementation_passes():
    q, k, v = make_qkv(1, 2, 128, 32)
    cfg = MoBAConfig(block_size=32, top_k=2, dtype=torch.bfloat16)
    truth, ref_err = reference_error(q, k, v, cfg)

    # A second, independently-cast bf16 run is a "correct" implementation.
    actual = moba_attention_naive(
        q.to(torch.bfloat16), k.to(torch.bfloat16), v.to(torch.bfloat16), cfg
    )
    assert_within_tolerance(actual, truth, ref_err, name="bf16_oracle")


def test_perturbed_implementation_is_rejected():
    """The harness must actually catch a wrong answer."""
    q, k, v = make_qkv(1, 2, 128, 32)
    cfg = MoBAConfig(block_size=32, top_k=2, dtype=torch.bfloat16)
    truth, ref_err = reference_error(q, k, v, cfg)

    actual = moba_attention_naive(
        q.to(torch.bfloat16), k.to(torch.bfloat16), v.to(torch.bfloat16), cfg
    )
    # Perturb by well beyond the allowed band.
    poisoned = actual + (TOLERANCE_FACTOR * ref_err * 100 + 1.0)

    with pytest.raises(AssertionError, match="exceeds tolerance"):
        assert_within_tolerance(poisoned, truth, ref_err, name="poisoned")


def test_wrong_top_k_is_rejected():
    """A semantic bug, not a numeric one, must also be caught."""
    q, k, v = make_qkv(1, 2, 256, 32)
    cfg = MoBAConfig(block_size=32, top_k=2, dtype=torch.bfloat16)
    truth, ref_err = reference_error(q, k, v, cfg)

    wrong_cfg = MoBAConfig(block_size=32, top_k=4, dtype=torch.bfloat16)
    actual = moba_attention_naive(
        q.to(torch.bfloat16), k.to(torch.bfloat16), v.to(torch.bfloat16), wrong_cfg
    )
    with pytest.raises(AssertionError, match="exceeds tolerance"):
        assert_within_tolerance(actual, truth, ref_err, name="wrong_top_k")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `wsl -e bash -lc 'cd "/mnt/c/D/Side Projects/GPU Kernel" && .venv/bin/python -m pytest tests/test_tolerance.py -v'`
Expected: FAIL with `ModuleNotFoundError: No module named 'moba.tolerance'`.

- [ ] **Step 3: Write `src/moba/tolerance.py`**

```python
"""Relative tolerance policy for correctness comparisons.

An absolute tolerance is a magic number: it is tuned until tests pass and
then means nothing. This module instead asks whether a backend's error is
comparable to the error any correct implementation would incur at the same
precision, following the methodology of the FlashAttention paper.

Procedure:
  1. Compute the oracle in fp64. Call it `truth`.
  2. Compute the oracle in the target dtype. Its deviation from `truth` is
     `ref_error` — the price of the format itself.
  3. A backend passes if its own deviation from `truth` is within
     TOLERANCE_FACTOR * ref_error.
"""

import torch

from moba.config import MoBAConfig
from moba.ref.naive import moba_attention_naive

TOLERANCE_FACTOR = 2.0
TOLERANCE_FLOOR = 1e-6


def max_abs_error(a: torch.Tensor, b: torch.Tensor) -> float:
    """Returns the maximum absolute elementwise difference.

    Both tensors are promoted to float64 first, so comparing a bf16 result
    against an fp64 truth does not silently round the truth down.

    Args:
        a: First tensor.
        b: Second tensor, same shape as `a`.

    Returns:
        The maximum absolute difference as a Python float.
    """
    return (a.to(torch.float64) - b.to(torch.float64)).abs().max().item()


def reference_error(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cfg: MoBAConfig,
) -> tuple[torch.Tensor, float]:
    """Establishes ground truth and the precision cost of `cfg.dtype`.

    Args:
        q: Queries shaped [B, H, S, D], float64.
        k: Keys shaped [B, H, S, D], float64.
        v: Values shaped [B, H, S, D], float64.
        cfg: Configuration; `cfg.dtype` is the precision being characterized.

    Returns:
        A tuple of (truth, ref_error) where `truth` is the fp64 oracle output
        and `ref_error` is the max absolute deviation of the same oracle
        evaluated in `cfg.dtype`.
    """
    truth = moba_attention_naive(q, k, v, cfg)
    low = moba_attention_naive(
        q.to(cfg.dtype), k.to(cfg.dtype), v.to(cfg.dtype), cfg
    )
    return truth, max_abs_error(low, truth)


def assert_within_tolerance(
    actual: torch.Tensor,
    truth: torch.Tensor,
    ref_error: float,
    name: str,
) -> None:
    """Asserts that `actual` is no less accurate than a same-dtype oracle.

    Args:
        actual: The backend's output.
        truth: The fp64 oracle output.
        ref_error: Deviation of the same-dtype oracle from `truth`.
        name: Backend name, used in the failure message.

    Raises:
        AssertionError: If the deviation exceeds the allowed band.
    """
    budget = max(TOLERANCE_FACTOR * ref_error, TOLERANCE_FLOOR)
    err = max_abs_error(actual, truth)
    if err > budget:
        raise AssertionError(
            f"{name}: max abs error {err:.3e} exceeds tolerance {budget:.3e} "
            f"(reference error {ref_error:.3e} x factor {TOLERANCE_FACTOR})"
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `wsl -e bash -lc 'cd "/mnt/c/D/Side Projects/GPU Kernel" && .venv/bin/python -m pytest tests/test_tolerance.py -v'`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add src/moba/tolerance.py tests/test_tolerance.py
git commit -m "feat: relative tolerance policy with poisoned-implementation test"
```

---

### Task 6: Tier 2 — every available backend against the oracle

**Files:**
- Create: `tests/conftest.py`
- Create: `tests/test_backends.py`

**Interfaces:**
- Consumes: registry (Task 4), tolerance (Task 5), `MoBAConfig` (Task 2).
- Produces:
  - pytest fixtures `device`, `sweep_config`.
  - `tests/conftest.py::SHAPE_SWEEP: list[tuple[int, int, int, int]]` as `(batch, heads, seq_len, head_dim)`.
  - `tests/conftest.py::make_qkv(shape, dtype, seed, device) -> tuple[Tensor, Tensor, Tensor]`.

- [ ] **Step 1: Write `tests/conftest.py`**

```python
"""Shared fixtures and the local shape sweep.

Shapes are sized for a 4 GB card. The oracle materializes an [S, S] attention
matrix per (batch, head), so fp64 memory is roughly
B * H * S^2 * 8 bytes — at B=2, H=8, S=2048 that is 512 MB for one tensor,
and several exist at once. Do not enlarge these without checking that number.
"""

import pytest
import torch

# (batch, heads, seq_len, head_dim)
SHAPE_SWEEP = [
    (1, 4, 128, 64),
    (1, 4, 512, 64),
    (2, 4, 512, 64),
    (1, 8, 1024, 64),
]

# (block_size, top_k). Small blocks are over-represented on purpose: they are
# the regime this project exists to fix.
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
        seed: RNG seed, for reproducible failures.
        device: Torch device string.

    Returns:
        A tuple (q, k, v), each shaped `shape`.
    """
    g = torch.Generator(device=device).manual_seed(seed)
    return tuple(
        torch.randn(shape, generator=g, device=device, dtype=dtype) for _ in range(3)
    )


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
```

- [ ] **Step 2: Write the failing test**

```python
# tests/test_backends.py
"""Tier 2: every available backend measured against the oracle."""

import itertools

import pytest
import torch

from moba.backends import available_backends, skipped_backends
from moba.config import MoBAConfig
from moba.tolerance import assert_within_tolerance, reference_error

from conftest import GATE_SWEEP, SHAPE_SWEEP, make_qkv

CASES = list(itertools.product(SHAPE_SWEEP, GATE_SWEEP))


def _ids(case):
    (b, h, s, d), (bk, tk) = case
    return f"b{b}h{h}s{s}d{d}-blk{bk}-top{tk}"


@pytest.mark.parametrize("case", CASES, ids=[_ids(c) for c in CASES])
def test_backend_matches_oracle(case, device):
    (b, h, s, d), (block_size, top_k) = case
    cfg = MoBAConfig(block_size=block_size, top_k=top_k, dtype=torch.bfloat16)

    if top_k >= cfg.num_blocks(s):
        pytest.skip("degenerates to dense; covered exactly by Tier 1")

    q, k, v = make_qkv((b, h, s, d), dtype=torch.float64, device=device)
    truth, ref_err = reference_error(q, k, v, cfg)

    ql, kl, vl = (t.to(cfg.dtype) for t in (q, k, v))

    checked = 0
    for backend in available_backends(cfg):
        if backend.name == "sdpa_dense":
            continue  # dense baseline is not a MoBA implementation
        out = backend.forward(ql, kl, vl, cfg)
        assert out.shape == (b, h, s, d)
        assert_within_tolerance(out, truth, ref_err, name=backend.name)
        checked += 1

    assert checked >= 1, "no MoBA backend was available to check"


def test_skip_reasons_are_reported(device, capsys):
    """Skipped backends must be visible, not silently absent."""
    cfg = MoBAConfig(block_size=32, top_k=2, dtype=torch.bfloat16)
    for name, reason in skipped_backends(cfg):
        print(f"skipped: {name}: {reason}")
    # Always passes; this test exists to surface the reasons in -s output.
```

- [ ] **Step 3: Run test to verify it fails**

Run: `wsl -e bash -lc 'cd "/mnt/c/D/Side Projects/GPU Kernel" && .venv/bin/python -m pytest tests/test_backends.py -v'`
Expected: FAIL with `ModuleNotFoundError: No module named 'conftest'` if `conftest.py` was not created, otherwise collection succeeds and tests pass. If the import of `conftest` fails, add to `pyproject.toml` under `[tool.pytest.ini_options]`: `pythonpath = ["tests"]`.

- [ ] **Step 4: Add the pytest path setting**

In `pyproject.toml`, under `[tool.pytest.ini_options]`, add:

```toml
pythonpath = ["tests"]
```

- [ ] **Step 5: Run test to verify it passes**

Run: `wsl -e bash -lc 'cd "/mnt/c/D/Side Projects/GPU Kernel" && .venv/bin/python -m pytest tests/test_backends.py -v -s'`
Expected: all parametrized cases pass or skip with a stated reason. Watch for CUDA OOM on the `(1, 8, 1024, 64)` shapes; if it occurs, drop that row from `SHAPE_SWEEP` and note it in the commit message.

- [ ] **Step 6: Commit**

```bash
git add tests/conftest.py tests/test_backends.py pyproject.toml
git commit -m "test: tier 2 backend-vs-oracle sweep with reported skip reasons"
```

---

### Task 7: Vendored references and their availability-gated backends

**Files:**
- Create: `scripts/fetch_references.sh`
- Create: `src/moba/backends/moonshot.py`
- Create: `src/moba/backends/triton_moba.py`
- Create: `src/moba/backends/flash_moba.py`
- Modify: `src/moba/backends/__init__.py`
- Create: `tests/test_optional_backends.py`

**Interfaces:**
- Consumes: `register` and `Backend` (Task 4), `supports_dtype` (Task 2).
- Produces: registered backends named `"moonshot"`, `"triton_moba"`, `"flash_moba"`. Each returns `(False, reason)` from `is_available` when its package is absent, and never raises at import time.

- [ ] **Step 1: Write `scripts/fetch_references.sh`**

```bash
#!/usr/bin/env bash
# Clones the reference implementations into references/ (gitignored, read-only).
set -euo pipefail

mkdir -p references
cd references

clone() {
  local url=$1 dir=$2
  if [ -d "$dir" ]; then
    echo "== $dir already present, skipping"
  else
    echo "== cloning $dir"
    git clone --depth 1 "$url" "$dir"
  fi
}

clone https://github.com/mit-han-lab/flash-moba.git            flash-moba
clone https://github.com/ReyJerry/Triton-MoBA.git              Triton-MoBA
clone https://github.com/MoonshotAI/MoBA.git                   MoBA
clone https://github.com/fla-org/native-sparse-attention.git   native-sparse-attention
clone https://github.com/Relaxed-System-Lab/Flash-Sparse-Attention.git Flash-Sparse-Attention

echo
echo "Done. references/ is gitignored and read-only — never edit these."
```

- [ ] **Step 2: Run the fetch script**

Run: `wsl -e bash -lc 'cd "/mnt/c/D/Side Projects/GPU Kernel" && bash scripts/fetch_references.sh'`
Expected: five directories under `references/`. Verify with `ls references/`.

- [ ] **Step 3: Write the failing test**

```python
# tests/test_optional_backends.py
"""Optional backends must degrade gracefully, never explode."""

import pytest
import torch

from moba.backends import all_backends, get_backend
from moba.config import MoBAConfig

OPTIONAL = ["moonshot", "triton_moba", "flash_moba"]


@pytest.mark.parametrize("name", OPTIONAL)
def test_optional_backend_is_registered(name):
    assert name in {b.name for b in all_backends()}


@pytest.mark.parametrize("name", OPTIONAL)
def test_optional_backend_reports_availability_without_raising(name):
    cfg = MoBAConfig(block_size=32, top_k=2, dtype=torch.bfloat16)
    ok, reason = get_backend(name).is_available(cfg)
    assert isinstance(ok, bool)
    if not ok:
        assert reason, f"{name} unavailable with no reason given"


@pytest.mark.parametrize("name", OPTIONAL)
def test_unavailable_backend_forward_raises_clearly(name):
    cfg = MoBAConfig(block_size=32, top_k=2, dtype=torch.bfloat16)
    backend = get_backend(name)
    ok, _ = backend.is_available(cfg)
    if ok:
        pytest.skip(f"{name} is available; nothing to check here")
    q = k = v = torch.zeros(1, 1, 32, 16, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match="not available"):
        backend.forward(q, k, v, cfg)
```

- [ ] **Step 4: Run test to verify it fails**

Run: `wsl -e bash -lc 'cd "/mnt/c/D/Side Projects/GPU Kernel" && .venv/bin/python -m pytest tests/test_optional_backends.py -v'`
Expected: FAIL — `KeyError: "No backend named 'moonshot'"`.

- [ ] **Step 5: Write `src/moba/backends/moonshot.py`**

```python
"""MoonshotAI/MoBA as a backend.

Depends on flash-attn==2.6.3, which frequently fails to build against recent
CUDA toolkits. That is expected and handled: the backend reports itself
unavailable rather than breaking the suite.
"""

import importlib.util

import torch

from moba.backends.base import register
from moba.config import MoBAConfig


class MoonshotBackend:
    """The original Moonshot AI MoBA implementation."""

    name = "moonshot"

    def is_available(self, cfg: MoBAConfig) -> tuple[bool, str]:
        if not torch.cuda.is_available():
            return False, "no CUDA device"
        if importlib.util.find_spec("moba_orig") is None:
            return False, "package 'moba_orig' not installed (see references/MoBA)"
        if importlib.util.find_spec("flash_attn") is None:
            return False, "flash-attn not installed (requires flash-attn==2.6.3)"
        return True, ""

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cfg: MoBAConfig,
    ) -> torch.Tensor:
        ok, reason = self.is_available(cfg)
        if not ok:
            raise RuntimeError(f"moonshot backend not available: {reason}")
        from moba_orig import moba_attn_varlen  # type: ignore

        raise NotImplementedError(
            "Adapter shape-mapping is written in Task 8 after reading "
            "references/MoBA to confirm its expected layout."
        )


register(MoonshotBackend())
```

- [ ] **Step 6: Write `src/moba/backends/triton_moba.py`**

```python
"""ReyJerry/Triton-MoBA as a backend."""

import importlib.util

import torch

from moba.backends.base import register
from moba.config import MoBAConfig


class TritonMoBABackend:
    """Community Triton port of MoBA — closest in spirit to our kernel."""

    name = "triton_moba"

    def is_available(self, cfg: MoBAConfig) -> tuple[bool, str]:
        if not torch.cuda.is_available():
            return False, "no CUDA device"
        if importlib.util.find_spec("triton_moba") is None:
            return False, "package 'triton_moba' not installed (see references/Triton-MoBA)"
        return True, ""

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cfg: MoBAConfig,
    ) -> torch.Tensor:
        ok, reason = self.is_available(cfg)
        if not ok:
            raise RuntimeError(f"triton_moba backend not available: {reason}")
        raise NotImplementedError(
            "Adapter shape-mapping is written in Task 8 after reading "
            "references/Triton-MoBA to confirm its expected layout."
        )


register(TritonMoBABackend())
```

- [ ] **Step 7: Write `src/moba/backends/flash_moba.py`**

```python
"""mit-han-lab/flash-moba as a backend — the performance target.

Its setup.py defaults to CUDA archs 80;90;100;120. sm_86 is absent, and even
when added the kernel's shared-memory tiling may exceed sm_86's 99 KB limit
and fail at launch rather than at compile time. Both cases surface here as an
unavailability reason.
"""

import importlib.util

import torch

from moba.backends.base import register
from moba.config import MoBAConfig
from moba.device import supports_dtype


class FlashMoBABackend:
    """FlashMoBA (CUDA), from MIT Han Lab."""

    name = "flash_moba"

    def is_available(self, cfg: MoBAConfig) -> tuple[bool, str]:
        if not torch.cuda.is_available():
            return False, "no CUDA device"
        capability = torch.cuda.get_device_capability(0)
        ok, reason = supports_dtype(cfg.dtype, capability)
        if not ok:
            return False, reason
        if importlib.util.find_spec("flash_moba") is None:
            return False, "package 'flash_moba' not built (see references/flash-moba)"
        return True, ""

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cfg: MoBAConfig,
    ) -> torch.Tensor:
        ok, reason = self.is_available(cfg)
        if not ok:
            raise RuntimeError(f"flash_moba backend not available: {reason}")
        raise NotImplementedError(
            "Adapter shape-mapping is written in Task 8 after reading "
            "references/flash-moba to confirm its expected layout."
        )


register(FlashMoBABackend())
```

- [ ] **Step 8: Register the optional backends**

In `src/moba/backends/__init__.py`, add these imports below the existing side-effect imports:

```python
from moba.backends import flash_moba as _flash_moba  # noqa: F401
from moba.backends import moonshot as _moonshot  # noqa: F401
from moba.backends import triton_moba as _triton_moba  # noqa: F401
```

- [ ] **Step 9: Run tests to verify they pass**

Run: `wsl -e bash -lc 'cd "/mnt/c/D/Side Projects/GPU Kernel" && .venv/bin/python -m pytest tests/test_optional_backends.py -v'`
Expected: 9 passed (3 backends x 3 tests), with the third test skipping for any backend that happens to be installed.

- [ ] **Step 10: Commit**

```bash
git add scripts/fetch_references.sh src/moba/backends/ tests/test_optional_backends.py
git commit -m "feat: availability-gated adapters for the three reference implementations"
```

---

### Task 8: Semantics cross-check and the Tier 3 disagreement matrix

This is the task the design spec forward-references. It closes the loop between our chosen semantics and what the references actually do.

**Files:**
- Create: `docs/semantics-crosscheck.md`
- Create: `tests/test_disagreement.py`
- Modify: `src/moba/backends/moonshot.py`, `src/moba/backends/triton_moba.py`, `src/moba/backends/flash_moba.py` (replace each `NotImplementedError` with a real adapter, or record why it stays unavailable)
- Modify: `docs/superpowers/specs/2026-08-15-moba-foundations-design.md` if a divergence changes our definition

**Interfaces:**
- Consumes: registry (Tasks 4, 7), tolerance (Task 5).
- Produces: `disagreement_matrix(q, k, v, cfg) -> dict[tuple[str, str], float]` in `tests/test_disagreement.py`.

- [ ] **Step 1: Read the reference gate implementations**

Read and take notes on:
- `references/MoBA/moba/` — the gate: how block representatives are pooled, how top-k is taken, how the current block is handled.
- `references/Triton-MoBA/triton_moba/` — same questions.
- `references/flash-moba/flash_moba/` and `references/flash-moba/csrc/` — same questions.

For each of the nine numbered semantics in the design spec, record what each reference does. Write findings into `docs/semantics-crosscheck.md` with a table whose columns are `ours | moonshot | triton_moba | flash_moba` and whose rows are the nine semantics.

- [ ] **Step 2: Resolve every divergence explicitly**

For each row where a reference differs from us, write one of:
- "Ours is correct per the paper; the reference deviates" — cite the paper section.
- "The reference is correct; update our spec" — then edit the design spec and the oracle, and rerun Task 3's tests.
- "Both are defensible; expose as a `MoBAConfig` flag" — then add the flag.

No row may be left unresolved. This document is what makes a future benchmark disagreement diagnosable.

- [ ] **Step 3: Write the adapters or record why not**

For each optional backend that imports successfully, replace the `NotImplementedError` in `forward` with a real shape-mapping adapter. If a backend cannot be built on this machine, leave `forward` raising and add a line to `docs/semantics-crosscheck.md` stating which machine it must be revisited on.

- [ ] **Step 4: Write the disagreement matrix test**

```python
# tests/test_disagreement.py
"""Tier 3: pairwise comparison across available backends.

This tier reports; it does not assert. A disagreement between two backends is
a finding to investigate against docs/semantics-crosscheck.md, not a failure.
"""

import itertools

import torch

from moba.backends import available_backends, skipped_backends
from moba.config import MoBAConfig
from moba.tolerance import max_abs_error

from conftest import make_qkv


def disagreement_matrix(q, k, v, cfg):
    """Computes max absolute difference between every pair of backends.

    Args:
        q: Queries shaped [B, H, S, D].
        k: Keys shaped [B, H, S, D].
        v: Values shaped [B, H, S, D].
        cfg: Configuration.

    Returns:
        A dict mapping (name_a, name_b) to their max absolute difference.
    """
    outputs = {}
    for backend in available_backends(cfg):
        if backend.name == "sdpa_dense":
            continue
        try:
            outputs[backend.name] = backend.forward(q, k, v, cfg)
        except (RuntimeError, NotImplementedError) as exc:
            print(f"  {backend.name}: excluded — {exc}")

    matrix = {}
    for a, b in itertools.combinations(sorted(outputs), 2):
        matrix[(a, b)] = max_abs_error(outputs[a], outputs[b])
    return matrix


def test_report_disagreements(device, capsys):
    cfg = MoBAConfig(block_size=32, top_k=2, dtype=torch.bfloat16)
    q, k, v = make_qkv((1, 4, 512, 64), dtype=cfg.dtype, device=device)

    with capsys.disabled():
        print(f"\n=== disagreement matrix ({cfg.dtype}, blk={cfg.block_size}, "
              f"top_k={cfg.top_k}) ===")
        for name, reason in skipped_backends(cfg):
            print(f"  skipped: {name}: {reason}")
        matrix = disagreement_matrix(q, k, v, cfg)
        if not matrix:
            print("  only one MoBA backend available; no pairs to compare")
        for (a, b), err in sorted(matrix.items()):
            print(f"  {a:>14} vs {b:<14} max abs diff {err:.3e}")
```

- [ ] **Step 5: Run the matrix and record the output**

Run: `wsl -e bash -lc 'cd "/mnt/c/D/Side Projects/GPU Kernel" && .venv/bin/python -m pytest tests/test_disagreement.py -v -s'`
Expected: PASS, printing either a populated matrix or an explicit "only one MoBA backend available". Paste the output into `docs/semantics-crosscheck.md` under a "Measured disagreements" heading, with the date and GPU.

- [ ] **Step 6: Commit**

```bash
git add docs/semantics-crosscheck.md tests/test_disagreement.py src/moba/backends/
git commit -m "docs: semantics cross-check against references; tier 3 disagreement matrix"
```

---

### Task 9: Benchmark suite and reporting

**Files:**
- Create: `bench/__init__.py`
- Create: `bench/suite.py`
- Create: `bench/report.py`
- Create: `tests/test_bench.py`

**Interfaces:**
- Consumes: registry (Tasks 4, 7), `MoBAConfig` (Task 2), `DeviceInfo` (Task 2).
- Produces:
  - `bench/suite.py::estimate_memory_mb(batch, heads, seq_len, head_dim, dtype) -> float`
  - `bench/suite.py::sparsity_ratio(seq_len, cfg) -> float`
  - `bench/suite.py::effective_tflops(batch, heads, seq_len, head_dim, cfg, ms) -> float`
  - `bench/suite.py::BenchRow` dataclass and `run_suite(shapes, gates, dtype) -> list[BenchRow]`
  - `bench/report.py::to_csv(rows, path) -> None` and `to_markdown(rows) -> str`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_bench.py
"""Tests for benchmark bookkeeping. The numbers must mean something."""

import math

import pytest
import torch

from bench.report import to_markdown
from bench.suite import (
    BenchRow,
    effective_tflops,
    estimate_memory_mb,
    sparsity_ratio,
)
from moba.config import MoBAConfig


def test_sparsity_ratio_is_one_when_all_blocks_selected():
    cfg = MoBAConfig(block_size=16, top_k=1000)
    assert sparsity_ratio(256, cfg) == pytest.approx(1.0)


def test_sparsity_ratio_shrinks_with_smaller_top_k():
    cfg_small = MoBAConfig(block_size=16, top_k=2)
    cfg_large = MoBAConfig(block_size=16, top_k=8)
    assert sparsity_ratio(1024, cfg_small) < sparsity_ratio(1024, cfg_large)


def test_sparsity_ratio_within_unit_interval():
    for top_k in (1, 2, 4, 8, 64):
        cfg = MoBAConfig(block_size=32, top_k=top_k)
        r = sparsity_ratio(1024, cfg)
        assert 0.0 < r <= 1.0


def test_memory_estimate_scales_quadratically_with_seq_len():
    a = estimate_memory_mb(1, 4, 512, 64, torch.bfloat16)
    b = estimate_memory_mb(1, 4, 1024, 64, torch.bfloat16)
    assert b / a == pytest.approx(4.0, rel=0.1)


def test_effective_tflops_is_positive():
    cfg = MoBAConfig(block_size=32, top_k=2)
    tflops = effective_tflops(1, 4, 1024, 64, cfg, ms=1.0)
    assert tflops > 0


def test_markdown_report_includes_skips_and_device():
    rows = [
        BenchRow(
            backend="naive_torch", batch=1, heads=4, seq_len=512, head_dim=64,
            block_size=32, top_k=2, dtype="bfloat16", latency_ms=1.5,
            tflops=2.0, peak_memory_mb=100.0, sparsity=0.25,
            speedup_vs_dense=1.1, speedup_ceiling=4.0, attainment=0.275,
            gpu_name="RTX 3050", capability="sm_86", total_memory_mb=4096,
            torch_version="2.4.0", triton_version="3.0.0", cuda_version="12.4",
            status="ok", note="",
        ),
        BenchRow(
            backend="flash_moba", batch=1, heads=4, seq_len=512, head_dim=64,
            block_size=32, top_k=2, dtype="bfloat16", latency_ms=float("nan"),
            tflops=float("nan"), peak_memory_mb=float("nan"), sparsity=0.25,
            speedup_vs_dense=float("nan"), speedup_ceiling=4.0,
            attainment=float("nan"), gpu_name="RTX 3050", capability="sm_86",
            total_memory_mb=4096, torch_version="2.4.0", triton_version="3.0.0",
            cuda_version="12.4", status="skipped",
            note="package 'flash_moba' not built",
        ),
    ]
    md = to_markdown(rows)
    assert "naive_torch" in md
    assert "flash_moba" in md
    assert "not built" in md, "skip reasons must appear in the report"
    assert "sm_86" in md
```

- [ ] **Step 2: Run test to verify it fails**

Run: `wsl -e bash -lc 'cd "/mnt/c/D/Side Projects/GPU Kernel" && .venv/bin/python -m pytest tests/test_bench.py -v'`
Expected: FAIL with `ModuleNotFoundError: No module named 'bench'`.

- [ ] **Step 3: Write `bench/__init__.py`**

```python
"""Benchmark suite for MoBA backends."""
```

- [ ] **Step 4: Write `bench/suite.py`**

```python
"""Benchmark driver.

The headline number is not latency. It is `attainment`: the speedup actually
achieved over dense attention, divided by the speedup that the sparsity
pattern implies is available. A kernel reaching 2x where sparsity offers 8x
has an attainment of 0.25, and closing that gap is the point of this project.
"""

import math
from dataclasses import asdict, dataclass, fields

import torch
import triton.testing

from moba.backends import available_backends, skipped_backends
from moba.config import MoBAConfig
from moba.device import DeviceInfo


@dataclass(frozen=True)
class BenchRow:
    """One measurement, carrying enough metadata to merge across machines."""

    backend: str
    batch: int
    heads: int
    seq_len: int
    head_dim: int
    block_size: int
    top_k: int
    dtype: str
    latency_ms: float
    tflops: float
    peak_memory_mb: float
    sparsity: float
    speedup_vs_dense: float
    speedup_ceiling: float
    attainment: float
    gpu_name: str
    capability: str
    total_memory_mb: int
    torch_version: str
    triton_version: str
    cuda_version: str
    status: str
    note: str


def sparsity_ratio(seq_len: int, cfg: MoBAConfig) -> float:
    """Fraction of causally-legal keys that MoBA actually attends to.

    Args:
        seq_len: Sequence length.
        cfg: Configuration supplying `block_size` and `top_k`.

    Returns:
        A value in (0, 1]; 1.0 means no sparsity (dense causal attention).
    """
    bk = cfg.block_size
    attended = 0
    causal = 0
    for i in range(seq_len):
        own_block = i // bk
        causal += i + 1
        # Own block, causally masked.
        attended += i - own_block * bk + 1
        # Up to top_k whole past blocks.
        n_past = min(cfg.top_k, own_block)
        attended += n_past * bk
    return attended / causal


def effective_tflops(
    batch: int,
    heads: int,
    seq_len: int,
    head_dim: int,
    cfg: MoBAConfig,
    ms: float,
) -> float:
    """FLOP/s counting only the attention arithmetic actually performed.

    Counts 2 FLOPs per multiply-add, for both QK^T and PV, over the attended
    key positions only.

    Args:
        batch: Batch size.
        heads: Number of heads.
        seq_len: Sequence length.
        head_dim: Head dimension.
        cfg: Configuration.
        ms: Measured latency in milliseconds.

    Returns:
        Effective TFLOP/s.
    """
    causal_pairs = seq_len * (seq_len + 1) / 2
    attended_pairs = causal_pairs * sparsity_ratio(seq_len, cfg)
    flops = 2 * 2 * batch * heads * attended_pairs * head_dim
    return flops / (ms * 1e-3) / 1e12


def estimate_memory_mb(
    batch: int, heads: int, seq_len: int, head_dim: int, dtype: torch.dtype
) -> float:
    """Estimates peak memory for the naive oracle's [S, S] score matrix.

    The oracle materializes the full attention matrix, which dominates its
    footprint. This is the number that decides whether a config fits in 4 GB.

    Args:
        batch: Batch size.
        heads: Number of heads.
        seq_len: Sequence length.
        head_dim: Head dimension.
        dtype: Compute dtype.

    Returns:
        Estimated megabytes.
    """
    itemsize = torch.empty(0, dtype=dtype).element_size()
    scores = batch * heads * seq_len * seq_len * itemsize
    qkv = 3 * batch * heads * seq_len * head_dim * itemsize
    # Softmax and masking need a few live copies of the score matrix.
    return (scores * 3 + qkv) / (1024 ** 2)


def run_suite(
    shapes: list[tuple[int, int, int, int]],
    gates: list[tuple[int, int]],
    dtype: torch.dtype = torch.bfloat16,
    memory_budget_mb: float = 3000.0,
) -> list[BenchRow]:
    """Benchmarks every available backend over the config sweep.

    Args:
        shapes: List of (batch, heads, seq_len, head_dim).
        gates: List of (block_size, top_k).
        dtype: Compute dtype.
        memory_budget_mb: Configs estimated above this are skipped.

    Returns:
        One BenchRow per (config, backend), including skipped ones.
    """
    device_row = DeviceInfo.probe(0).as_row()
    rows: list[BenchRow] = []

    for batch, heads, seq_len, head_dim in shapes:
        for block_size, top_k in gates:
            cfg = MoBAConfig(block_size=block_size, top_k=top_k, dtype=dtype)
            sparsity = sparsity_ratio(seq_len, cfg)
            ceiling = 1.0 / sparsity

            def blank(name: str, status: str, note: str) -> BenchRow:
                return BenchRow(
                    backend=name, batch=batch, heads=heads, seq_len=seq_len,
                    head_dim=head_dim, block_size=block_size, top_k=top_k,
                    dtype=str(dtype).replace("torch.", ""),
                    latency_ms=math.nan, tflops=math.nan,
                    peak_memory_mb=math.nan, sparsity=sparsity,
                    speedup_vs_dense=math.nan, speedup_ceiling=ceiling,
                    attainment=math.nan, status=status, note=note,
                    **device_row,
                )

            for name, reason in skipped_backends(cfg):
                rows.append(blank(name, "skipped", reason))

            estimate = estimate_memory_mb(batch, heads, seq_len, head_dim, dtype)
            if estimate > memory_budget_mb:
                for backend in available_backends(cfg):
                    rows.append(
                        blank(
                            backend.name, "skipped",
                            f"estimated {estimate:.0f} MB exceeds "
                            f"{memory_budget_mb:.0f} MB budget",
                        )
                    )
                continue

            g = torch.Generator(device="cuda").manual_seed(0)
            shape = (batch, heads, seq_len, head_dim)
            q, k, v = (
                torch.randn(shape, generator=g, device="cuda", dtype=dtype)
                for _ in range(3)
            )

            dense_ms = None
            measured: dict[str, float] = {}
            for backend in available_backends(cfg):
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                try:
                    ms = triton.testing.do_bench(
                        lambda b=backend: b.forward(q, k, v, cfg)
                    )
                except (RuntimeError, NotImplementedError) as exc:
                    rows.append(blank(backend.name, "error", str(exc)[:160]))
                    continue
                peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
                measured[backend.name] = ms
                if backend.name == "sdpa_dense":
                    dense_ms = ms
                rows.append(
                    BenchRow(
                        backend=backend.name, batch=batch, heads=heads,
                        seq_len=seq_len, head_dim=head_dim,
                        block_size=block_size, top_k=top_k,
                        dtype=str(dtype).replace("torch.", ""),
                        latency_ms=ms,
                        tflops=effective_tflops(
                            batch, heads, seq_len, head_dim, cfg, ms
                        ),
                        peak_memory_mb=peak_mb, sparsity=sparsity,
                        speedup_vs_dense=math.nan, speedup_ceiling=ceiling,
                        attainment=math.nan, status="ok", note="",
                        **device_row,
                    )
                )

            # Fill in speedups now that the dense baseline is known.
            if dense_ms is not None:
                for i, row in enumerate(rows):
                    if (
                        row.status == "ok"
                        and row.seq_len == seq_len
                        and row.block_size == block_size
                        and row.top_k == top_k
                        and row.batch == batch
                        and row.heads == heads
                        and math.isnan(row.speedup_vs_dense)
                    ):
                        speedup = dense_ms / row.latency_ms
                        rows[i] = BenchRow(
                            **{
                                **asdict(row),
                                "speedup_vs_dense": speedup,
                                "attainment": speedup / ceiling,
                            }
                        )
    return rows
```

- [ ] **Step 5: Write `bench/report.py`**

```python
"""Renders benchmark rows to CSV and markdown."""

import csv
import math
from dataclasses import asdict, fields
from pathlib import Path

from bench.suite import BenchRow

_COLUMNS = [
    "backend", "seq_len", "block_size", "top_k", "dtype",
    "latency_ms", "tflops", "sparsity", "speedup_vs_dense",
    "speedup_ceiling", "attainment", "capability", "status", "note",
]


def to_csv(rows: list[BenchRow], path: str | Path) -> None:
    """Writes every field of every row to CSV.

    Args:
        rows: Benchmark rows.
        path: Destination file; parent directories are created.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    names = [f.name for f in fields(BenchRow)]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=names)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def _fmt(value) -> str:
    if isinstance(value, float):
        if math.isnan(value):
            return "-"
        return f"{value:.3f}"
    return str(value)


def to_markdown(rows: list[BenchRow]) -> str:
    """Renders a human-readable results table.

    Skipped and errored rows are included with their reason, so the table
    never looks complete when it is not.

    Args:
        rows: Benchmark rows.

    Returns:
        A markdown string.
    """
    if not rows:
        return "_No results._"

    device = rows[0]
    header = (
        f"**Device:** {device.gpu_name} ({device.capability}), "
        f"{device.total_memory_mb} MB | torch {device.torch_version}, "
        f"triton {device.triton_version}, CUDA {device.cuda_version}\n\n"
    )
    lines = [header]
    lines.append("| " + " | ".join(_COLUMNS) + " |")
    lines.append("|" + "|".join(["---"] * len(_COLUMNS)) + "|")
    for row in rows:
        data = asdict(row)
        lines.append("| " + " | ".join(_fmt(data[c]) for c in _COLUMNS) + " |")
    return "\n".join(lines)
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `wsl -e bash -lc 'cd "/mnt/c/D/Side Projects/GPU Kernel" && .venv/bin/python -m pytest tests/test_bench.py -v'`
Expected: 6 passed. If `ModuleNotFoundError: bench`, add `pythonpath = ["tests", "."]` to `[tool.pytest.ini_options]`.

- [ ] **Step 7: Run the real suite once**

Run:
```bash
wsl -e bash -lc 'cd "/mnt/c/D/Side Projects/GPU Kernel" && .venv/bin/python -c "
from bench.suite import run_suite
from bench.report import to_csv, to_markdown
rows = run_suite([(1,4,512,64),(1,4,1024,64)], [(16,2),(32,2),(64,2)])
to_csv(rows, \"results/local_rtx3050.csv\")
print(to_markdown(rows))
"'
```
Expected: a populated table, with `flash_moba`, `moonshot`, and `triton_moba` present as `skipped` rows carrying reasons. Note the `attainment` column for `naive_torch` — it will be far below 1.0, which is expected and is the baseline the kernel must beat.

- [ ] **Step 8: Commit**

```bash
git add bench/ tests/test_bench.py results/
git commit -m "feat: benchmark suite reporting attainment vs sparsity ceiling"
```

---

### Task 10: FlashMoBA digest

**Files:**
- Create: `docs/flashmoba-digest.md`
- Modify: `CLAUDE.md` (update the Current status line)

**Interfaces:**
- Consumes: `references/flash-moba/` (Task 7), `docs/semantics-crosscheck.md` (Task 8).
- Produces: a prose document. No code.

- [ ] **Step 1: Map the CUDA source**

Run: `wsl -e bash -lc 'cd "/mnt/c/D/Side Projects/GPU Kernel" && find references/flash-moba/csrc -name "*.cu" -o -name "*.cuh" -o -name "*.h" | head -40 && echo "---" && wc -l $(find references/flash-moba/csrc -name "*.cuh" | head -10)'`
Expected: a file listing. Record which files contain the gate (block selection) and which contain the attention inner loop.

- [ ] **Step 2: Read the kernel's loop structure**

Read the main kernel file(s) identified in Step 1. For each nesting level of the loop, record: what is iterated, what lives in registers, what lives in shared memory, and how many threads cooperate. Note the tile sizes and where they are declared as compile-time constants.

- [ ] **Step 3: Read the FSA paper's problem statement**

Read `references/Flash-Sparse-Attention/` README and the FSA paper (arXiv 2508.18224), specifically its account of why NSA's kernel loses efficiency at small group sizes. Record the mechanism it identifies and the loop-order change it proposes.

- [ ] **Step 4: Write `docs/flashmoba-digest.md`**

Write the document with these sections, every claim about the code citing the file it came from:

1. **What FlashMoBA computes** — the gate and the attention, in our spec's vocabulary, noting any divergence already recorded in `docs/semantics-crosscheck.md`.
2. **The loop order** — the nesting, level by level, with what is resident where at each level.
3. **Why small blocks break it** — the specific mechanism. Cover at minimum: how tile size relates to block size, what happens to occupancy and to the fraction of each tile that does useful work when `block_size` is small, and where the wasted work goes.
4. **What FSA does about the analogous problem** — the loop-order change and why it helps.
5. **What transfers to our Triton kernel** — a short list of design constraints this implies, each phrased so it can become a requirement in the kernel spec.

Every factual claim about FlashMoBA cites a path like `references/flash-moba/csrc/<file>:<line>`. Anything you could not determine from the source is written as an open question, not guessed.

- [ ] **Step 5: Update the status line in CLAUDE.md**

Replace the status line with:

```
> Status: foundations complete — oracle, harness, benchmarks, digest. Kernel spec next.
```

- [ ] **Step 6: Commit**

```bash
git add docs/flashmoba-digest.md CLAUDE.md
git commit -m "docs: plain-English digest of FlashMoBA's CUDA loop order"
```

---

## Self-review notes

**Spec coverage.** D1 → Task 3. D2 → Tasks 1, 2, 7. D3 → Task 3 (oracle stays autograd-compatible; no `.detach()` or in-place ops on inputs). D4 → Tasks 4, 7. D5 → Task 5. MoBA semantics items 1–9 → Task 3, verified in Task 8. Shape sweep → Task 6. Benchmark suite → Task 9. Digest → Task 10. Success criteria 1–5 → Tasks 6, 9, 5, 10, 4 respectively.

**Known follow-ups, deliberately deferred.** The optional-backend adapters in Task 7 raise `NotImplementedError` in `forward` by design — their shape mapping cannot be written correctly without first reading the reference source, which is Task 8's job. This is a sequencing decision, not a placeholder: Task 7's deliverable is the availability gating, and its tests verify exactly that.

**Type consistency.** `MoBAConfig` fields, `Backend` method signatures, `BenchRow` field names, and the `reference_error` / `assert_within_tolerance` pair are used identically across Tasks 2–9. `DeviceInfo.as_row()` keys (`gpu_name`, `capability`, `total_memory_mb`, `torch_version`, `triton_version`, `cuda_version`) match `BenchRow`'s corresponding fields exactly, which is what allows the `**device_row` splat in `run_suite`.
