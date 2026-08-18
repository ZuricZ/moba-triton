# MoBA Triton Kernel Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A working KV-block-major MoBA forward kernel in Triton that matches the oracle exactly on routing and within tolerance on arithmetic, registered as a benchmark backend.

**Architecture:** Three stages. A PyTorch gate builds a CSC index mapping each KV block to the queries that selected it. One Triton kernel owns one KV block per program, gathers its queries, and writes un-normalized `(m, l, acc)` partials into fixed per-query slots. A second Triton kernel merges each query's slots with a log-sum-exp reduction.

**Tech Stack:** Python 3.13, torch 2.13.0+cu126, triton 3.7.1, pytest. All GPU work in WSL2.

**Spec:** `docs/superpowers/specs/2026-08-18-moba-kernel-design.md` (decisions D6–D11). The foundations spec `2026-08-15-moba-foundations-design.md` (D1–D5, semantics, tolerance policy, backend protocol) remains binding.

## Global Constraints

- Package layout `src/`, package name `moba`. Google-style docstrings.
- Primary dtype `torch.bfloat16`; secondary `torch.float16`. Partials always fp32.
- Minimum compute capability 8.0. Local budget 4 GB VRAM.
- `D * sizeof(dtype) >= 128 bytes`. Supported `head_dim`: 64, 128.
- `GROUP = max(16, round_up_16(top_k * block_size))`. `block_size >= 16`.
- `tl.dot` requires every matrix dimension `>= 16`.
- Routing is compared **exactly**; arithmetic under the D5 relative tolerance.
- Commands run via:
  ```bash
  wsl -e bash -lc 'export PATH="$HOME/.local/bin:$PATH"; \
    export UV_PROJECT_ENVIRONMENT="$HOME/.venvs/moba"; \
    cd "/mnt/c/D/Side Projects/GPU Kernel" && uv run <command>'
  ```
- `git` runs from **PowerShell**, not WSL (identity is Windows-side).
- Never edit anything under `references/`.

---

## File Structure

| File | Responsibility |
|---|---|
| `src/moba/kernels/index.py` | `CSCIndex` dataclass and `build_csc`. Pure PyTorch. The only place routing is turned into kernel-shaped arrays. |
| `src/moba/kernels/moba_fwd.py` | The two Triton kernels and the `moba_forward` entry point that allocates partials and launches them. |
| `src/moba/backends/triton_ours.py` | `Backend` adapter registering our kernel as `moba_triton`. |
| `tests/test_index.py` | CSC construction against the oracle mask, plus structural invariants. |
| `tests/test_moba_kernel.py` | Kernel versus oracle across the shape × gate sweep. |
| `tests/test_kernel_invariants.py` | Property tests needing no reference. |
| `bench/stages.py` | Per-stage timing so overheads are attributable. |

`index.py` is separate from `moba_fwd.py` because it is the piece D11 schedules for replacement by a Triton gate later; keeping the boundary clean means that swap touches one file.

---

## Task 1: CSC index construction

**Files:**
- Create: `src/moba/kernels/index.py`
- Test: `tests/test_index.py`

**Interfaces:**
- Consumes: `moba.config.MoBAConfig`, `moba.ref.naive.moba_selection`
- Produces:
  ```python
  @dataclass(frozen=True)
  class CSCIndex:
      col_nnz: torch.Tensor       # [B, H, N]     int32
      col_offsets: torch.Tensor   # [B, H, N + 1] int32
      row_indices: torch.Tensor   # [B, H, T]     int32
      slot_of_pair: torch.Tensor  # [B, H, T]     int32
      order: torch.Tensor         # [B, H, N]     int32
      num_blocks: int
      max_slots: int              # K == top_k

  def build_csc(q: torch.Tensor, k: torch.Tensor, cfg: MoBAConfig) -> CSCIndex
  def csc_to_mask(index: CSCIndex, seq_len: int) -> torch.Tensor
  ```

`build_csc` calls `moba_selection` rather than reimplementing the gate. Routing is therefore identical to the oracle by construction in v1; the round-trip test below checks the *conversion*, which is the part that can be wrong. When a Triton gate replaces this (D11 deferred), the same test becomes a genuine routing check.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_index.py
import pytest
import torch

from conftest import GATE_SWEEP, SHAPE_SWEEP, case_id, make_qkv
from moba.config import MoBAConfig
from moba.kernels.index import build_csc, csc_to_mask
from moba.ref.naive import moba_selection


@pytest.mark.parametrize(
    "shape,gate",
    [(s, g) for s in SHAPE_SWEEP for g in GATE_SWEEP],
    ids=[case_id(s, g) for s in SHAPE_SWEEP for g in GATE_SWEEP],
)
def test_csc_round_trips_to_the_oracle_mask(device, shape, gate):
    """Expanding the CSC back to a dense mask must reproduce it exactly."""
    block_size, top_k = gate
    cfg = MoBAConfig(block_size=block_size, top_k=top_k, dtype=torch.bfloat16)
    q, k, _ = make_qkv(shape, dtype=torch.bfloat16)

    expected = moba_selection(q, k, cfg)
    index = build_csc(q, k, cfg)
    actual = csc_to_mask(index, shape[2])

    assert torch.equal(actual, expected), (
        "CSC does not round-trip; the index and the routing disagree"
    )
```

- [ ] **Step 2: Run it and confirm it fails**

```bash
wsl -e bash -lc 'export PATH="$HOME/.local/bin:$PATH"; export UV_PROJECT_ENVIRONMENT="$HOME/.venvs/moba"; cd "/mnt/c/D/Side Projects/GPU Kernel" && uv run pytest tests/test_index.py -q'
```

Expected: `ModuleNotFoundError: No module named 'moba.kernels.index'`.

- [ ] **Step 3: Implement `index.py`**

```python
"""Turns per-query routing into the KV-block-major arrays the kernel needs.

The gate answers "for each query, which blocks?". The kernel asks the
transpose: "for each block, which queries?". This module is that transpose,
stored as CSC -- columns are KV blocks, entries are query indices.

Kept separate from the kernels because D11 schedules it for replacement by a
Triton gate; the boundary means that swap touches one file.
"""

import math
from dataclasses import dataclass

import torch

from moba.config import MoBAConfig
from moba.ref.naive import moba_selection


@dataclass(frozen=True)
class CSCIndex:
    """KV-block-major view of the routing decision."""

    col_nnz: torch.Tensor
    col_offsets: torch.Tensor
    row_indices: torch.Tensor
    slot_of_pair: torch.Tensor
    order: torch.Tensor
    num_blocks: int
    max_slots: int


def build_csc(q: torch.Tensor, k: torch.Tensor, cfg: MoBAConfig) -> CSCIndex:
    """Builds the CSC index for one batch of queries and keys.

    Args:
        q: Queries shaped [B, H, S, D].
        k: Keys shaped [B, H, S, D].
        cfg: Configuration supplying `block_size` and `top_k`.

    Returns:
        A populated CSCIndex. `row_indices` and `slot_of_pair` are allocated at
        the static upper bound `S * top_k`; entries beyond a column's `col_nnz`
        are undefined and must not be read.
    """
    mask = moba_selection(q, k, cfg)                    # [B, H, S, N] bool
    b, h, s, n = mask.shape
    device = mask.device
    slots = cfg.top_k
    total = s * slots

    # Slot number = rank of this block among the query's selections, ordered by
    # ascending block index. Any fixed rule works; a fixed rule is what makes
    # the combine pass deterministic.
    slot_grid = (mask.cumsum(dim=-1) - 1).to(torch.int32)

    # Block-major flattening: [B, H, N, S]. Reading this in order visits every
    # entry of column 0, then column 1, and so on -- exactly CSC ordering, with
    # query indices ascending inside each column for free.
    bm = mask.permute(0, 1, 3, 2).reshape(b * h, n * s)

    flat = bm.reshape(-1).nonzero(as_tuple=False).squeeze(1)
    bh_of = torch.div(flat, n * s, rounding_mode="floor")
    within = flat - bh_of * (n * s)
    n_of = torch.div(within, s, rounding_mode="floor")
    s_of = within - n_of * s

    counts = bm.sum(dim=1)                              # [B*H]
    starts = torch.zeros_like(counts)
    starts[1:] = counts.cumsum(0)[:-1]
    pos = torch.arange(flat.numel(), device=device) - starts[bh_of]

    row_indices = torch.zeros(b * h, total, dtype=torch.int32, device=device)
    slot_of_pair = torch.zeros(b * h, total, dtype=torch.int32, device=device)
    row_indices[bh_of, pos] = s_of.to(torch.int32)
    slot_of_pair[bh_of, pos] = slot_grid.reshape(b * h, s, n)[bh_of, s_of, n_of]

    col_nnz = mask.sum(dim=2).to(torch.int32)           # [B, H, N]
    col_offsets = torch.zeros(b, h, n + 1, dtype=torch.int32, device=device)
    col_offsets[..., 1:] = col_nnz.cumsum(dim=-1)

    # Longest-first dispatch (D8). Stable so the order is reproducible.
    order = torch.argsort(
        col_nnz.to(torch.int64), dim=-1, descending=True, stable=True
    ).to(torch.int32)

    return CSCIndex(
        col_nnz=col_nnz,
        col_offsets=col_offsets,
        row_indices=row_indices.view(b, h, total),
        slot_of_pair=slot_of_pair.view(b, h, total),
        order=order,
        num_blocks=n,
        max_slots=slots,
    )


def csc_to_mask(index: CSCIndex, seq_len: int) -> torch.Tensor:
    """Expands a CSC index back to a dense boolean selection mask.

    Test support only -- this is the inverse the round-trip test needs, and it
    is deliberately slow and obvious.

    Args:
        index: The index to expand.
        seq_len: Sequence length.

    Returns:
        Boolean mask shaped [B, H, seq_len, num_blocks].
    """
    b, h, n = index.col_nnz.shape
    out = torch.zeros(
        b, h, seq_len, n, dtype=torch.bool, device=index.col_nnz.device
    )
    for bi in range(b):
        for hi in range(h):
            for blk in range(n):
                lo = int(index.col_offsets[bi, hi, blk])
                cnt = int(index.col_nnz[bi, hi, blk])
                rows = index.row_indices[bi, hi, lo:lo + cnt].long()
                out[bi, hi, rows, blk] = True
    return out
```

- [ ] **Step 4: Run the test and confirm it passes**

```bash
wsl -e bash -lc 'export PATH="$HOME/.local/bin:$PATH"; export UV_PROJECT_ENVIRONMENT="$HOME/.venvs/moba"; cd "/mnt/c/D/Side Projects/GPU Kernel" && uv run pytest tests/test_index.py -q'
```

- [ ] **Step 5: Add the structural invariant tests**

```python
def test_col_offsets_is_a_valid_exclusive_prefix_sum(device):
    cfg = MoBAConfig(block_size=32, top_k=4, dtype=torch.bfloat16)
    q, k, _ = make_qkv((1, 4, 512, 64), dtype=torch.bfloat16)
    ix = build_csc(q, k, cfg)

    assert torch.equal(ix.col_offsets[..., 0],
                       torch.zeros_like(ix.col_offsets[..., 0]))
    assert torch.equal(ix.col_offsets[..., 1:] - ix.col_offsets[..., :-1],
                       ix.col_nnz)
    assert torch.equal(ix.col_offsets[..., -1], ix.col_nnz.sum(-1))


def test_row_indices_ascend_within_every_column(device):
    """Ascending rows preserve what DRAM locality the gather can get."""
    cfg = MoBAConfig(block_size=32, top_k=4, dtype=torch.bfloat16)
    q, k, _ = make_qkv((1, 2, 512, 64), dtype=torch.bfloat16)
    ix = build_csc(q, k, cfg)

    for bi in range(ix.col_nnz.shape[0]):
        for hi in range(ix.col_nnz.shape[1]):
            for blk in range(ix.num_blocks):
                lo = int(ix.col_offsets[bi, hi, blk])
                cnt = int(ix.col_nnz[bi, hi, blk])
                rows = ix.row_indices[bi, hi, lo:lo + cnt]
                assert torch.all(rows[1:] > rows[:-1]), (
                    f"column {blk} is not strictly ascending"
                )


def test_slots_are_dense_and_unique_per_query(device):
    """Each query's slots must be exactly range(k_i), or the combine misreads."""
    cfg = MoBAConfig(block_size=32, top_k=4, dtype=torch.bfloat16)
    q, k, _ = make_qkv((1, 2, 512, 64), dtype=torch.bfloat16)
    ix = build_csc(q, k, cfg)
    mask = moba_selection(q, k, cfg)

    b, h, n = ix.col_nnz.shape
    for bi in range(b):
        for hi in range(h):
            per_query: dict[int, list[int]] = {}
            for blk in range(n):
                lo = int(ix.col_offsets[bi, hi, blk])
                cnt = int(ix.col_nnz[bi, hi, blk])
                for r, sl in zip(
                    ix.row_indices[bi, hi, lo:lo + cnt].tolist(),
                    ix.slot_of_pair[bi, hi, lo:lo + cnt].tolist(),
                ):
                    per_query.setdefault(r, []).append(sl)
            for r, sl in per_query.items():
                expected = int(mask[bi, hi, r].sum())
                assert sorted(sl) == list(range(expected)), (
                    f"query {r} has slots {sorted(sl)}, expected "
                    f"{list(range(expected))}"
                )


def test_order_is_longest_first(device):
    cfg = MoBAConfig(block_size=16, top_k=2, dtype=torch.bfloat16)
    q, k, _ = make_qkv((1, 4, 512, 64), dtype=torch.bfloat16)
    ix = build_csc(q, k, cfg)

    sorted_counts = torch.gather(ix.col_nnz, -1, ix.order.long())
    assert torch.all(sorted_counts[..., 1:] <= sorted_counts[..., :-1])
```

- [ ] **Step 6: Run and commit**

```bash
wsl -e bash -lc 'export PATH="$HOME/.local/bin:$PATH"; export UV_PROJECT_ENVIRONMENT="$HOME/.venvs/moba"; cd "/mnt/c/D/Side Projects/GPU Kernel" && uv run pytest tests/test_index.py -q'
```

```powershell
git add src/moba/kernels/index.py tests/test_index.py
git commit -m "feat: CSC index mapping KV blocks to their consumer queries"
```

---

## Task 2: The attention kernel

**Files:**
- Create: `src/moba/kernels/moba_fwd.py`
- Test: `tests/test_moba_kernel.py`

**Interfaces:**
- Consumes: `CSCIndex` from Task 1.
- Produces:
  ```python
  def moba_forward(
      q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
      cfg: MoBAConfig, index: CSCIndex | None = None,
  ) -> torch.Tensor
  ```
  Partial buffers, allocated inside `moba_forward`:
  - `part_m`   `[B, H, S, K]`     fp32, initialised to `-inf`
  - `part_l`   `[B, H, S, K]`     fp32, zeros
  - `part_acc` `[B, H, S, K, D]`  fp32, zeros

Task 3 consumes those three buffers. This task ends with them correct but not yet merged, so its test compares a **single-slot** configuration where no merging is required.

- [ ] **Step 1: Write the failing test**

Choosing `top_k=1` makes every query select exactly its own block, so slot 0 is the only slot and `acc / l` is the whole answer. That isolates the attention kernel from the combine pass.

```python
# tests/test_moba_kernel.py
import pytest
import torch

from conftest import make_qkv
from moba.config import MoBAConfig
from moba.kernels.moba_fwd import moba_forward
from moba.ref.naive import moba_attention_naive
from moba.tolerance import assert_within_tolerance, reference_error


def test_single_slot_matches_the_oracle(device):
    """top_k=1 means one block per query, so no cross-slot merging occurs."""
    cfg = MoBAConfig(block_size=32, top_k=1, dtype=torch.bfloat16)
    q, k, v = make_qkv((1, 4, 512, 64), dtype=torch.bfloat16)

    actual = moba_forward(q, k, v, cfg)
    expected = moba_attention_naive(q, k, v, cfg)

    assert_within_tolerance(
        actual, expected, reference_error(q, k, v, cfg), "single-slot kernel"
    )
```

- [ ] **Step 2: Run it and confirm it fails**

Expected: `ModuleNotFoundError: No module named 'moba.kernels.moba_fwd'`.

- [ ] **Step 3: Write the kernel**

```python
"""KV-block-major MoBA forward.

One program owns one KV block, loads it into SRAM once, and streams every
query that selected it past that single load. This is decision D6: the loop
order is inverted relative to FlashAttention-2 because MoBA's per-query
routing destroys the amortization a query-major tile depends on.

Because a program sees each of its queries against exactly one block, there is
no online-softmax loop here -- the max and sum are computed in one shot. All
the running-max machinery lives in the combine pass instead.
"""

import torch
import triton
import triton.language as tl

from moba.config import MoBAConfig
from moba.kernels.index import CSCIndex, build_csc


@triton.jit
def _moba_attn_fwd(
    Q, K, V, RowIdx, SlotIdx, ColOffsets, ColNnz, Order,
    PartM, PartL, PartAcc,
    S, T, sm_scale, num_blocks,
    BLOCK_SIZE: tl.constexpr, GROUP: tl.constexpr, D: tl.constexpr,
    K_SLOTS: tl.constexpr, CAUSAL: tl.constexpr,
):
    """One program per (KV block, batch-head)."""
    pid_col = tl.program_id(0)
    pid_bh = tl.program_id(1)

    # Longest-first dispatch (D8). Only the block id changes; results are
    # written to query addresses, so nothing needs undoing afterwards.
    blk = tl.load(Order + pid_bh * num_blocks + pid_col)
    n = tl.load(ColNnz + pid_bh * num_blocks + blk)
    if n == 0:
        return

    lo = tl.load(ColOffsets + pid_bh * (num_blocks + 1) + blk)

    offs_d = tl.arange(0, D)
    key_pos = blk * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    key_ok = key_pos < S

    kv_base = pid_bh * S * D
    k = tl.load(K + kv_base + key_pos[:, None] * D + offs_d[None, :],
                mask=key_ok[:, None], other=0.0)
    v = tl.load(V + kv_base + key_pos[:, None] * D + offs_d[None, :],
                mask=key_ok[:, None], other=0.0)

    for start in range(0, n, GROUP):
        offs = start + tl.arange(0, GROUP)
        live = offs < n

        rows = tl.load(RowIdx + pid_bh * T + lo + offs, mask=live, other=0)
        slots = tl.load(SlotIdx + pid_bh * T + lo + offs, mask=live, other=0)

        # Gathered rows (D7). Each row is one cache line at D=64 bf16, so this
        # costs the same transaction count as a contiguous read.
        q = tl.load(Q + kv_base + rows[:, None] * D + offs_d[None, :],
                    mask=live[:, None], other=0.0)

        qk = tl.dot(q, tl.trans(k)) * sm_scale
        qk = tl.where(key_ok[None, :], qk, float("-inf"))
        if CAUSAL:
            # A no-op for every block wholly in the past; trims only the
            # query's own block (D9).
            qk = tl.where(key_pos[None, :] <= rows[:, None], qk, float("-inf"))

        m = tl.max(qk, axis=1)
        p = tl.exp(qk - m[:, None])
        l = tl.sum(p, axis=1)
        acc = tl.dot(p.to(v.dtype), v)

        dst = rows * K_SLOTS + slots
        tl.store(PartM + pid_bh * S * K_SLOTS + dst, m, mask=live)
        tl.store(PartL + pid_bh * S * K_SLOTS + dst, l, mask=live)
        tl.store(
            PartAcc + pid_bh * S * K_SLOTS * D + dst[:, None] * D + offs_d[None, :],
            acc,
            mask=live[:, None],
        )


def _round_up_16(x: int) -> int:
    return max(16, ((x + 15) // 16) * 16)


def moba_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cfg: MoBAConfig,
    index: CSCIndex | None = None,
) -> torch.Tensor:
    """Computes MoBA attention with the KV-block-major kernel.

    Args:
        q: Queries shaped [B, H, S, D], contiguous.
        k: Keys shaped [B, H, S, D], contiguous.
        v: Values shaped [B, H, S, D], contiguous.
        cfg: Configuration.
        index: Precomputed CSC index; built from `q` and `k` when omitted.

    Returns:
        Attention output shaped [B, H, S, D], same dtype as `q`.

    Raises:
        ValueError: If shapes disagree, or head_dim or block_size is
            unsupported.
    """
    if not (q.shape == k.shape == v.shape):
        raise ValueError(f"shape mismatch: {q.shape}, {k.shape}, {v.shape}")
    b, h, s, d = q.shape
    if d not in (64, 128):
        raise ValueError(f"head_dim {d} unsupported; D*itemsize must be >= 128 B")
    if cfg.block_size < 16:
        raise ValueError(f"block_size {cfg.block_size} below the tl.dot minimum")

    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    if index is None:
        index = build_csc(q, k, cfg)

    slots = index.max_slots
    group = _round_up_16(cfg.top_k * cfg.block_size)

    part_m = torch.full((b, h, s, slots), float("-inf"),
                        device=q.device, dtype=torch.float32)
    part_l = torch.zeros((b, h, s, slots), device=q.device, dtype=torch.float32)
    part_acc = torch.zeros((b, h, s, slots, d),
                           device=q.device, dtype=torch.float32)

    _moba_attn_fwd[(index.num_blocks, b * h)](
        q, k, v,
        index.row_indices, index.slot_of_pair,
        index.col_offsets, index.col_nnz, index.order,
        part_m, part_l, part_acc,
        s, index.row_indices.shape[-1], d ** -0.5, index.num_blocks,
        BLOCK_SIZE=cfg.block_size, GROUP=group, D=d,
        K_SLOTS=slots, CAUSAL=True,
    )

    from moba.kernels.moba_fwd import combine_partials  # defined in Task 3
    return combine_partials(part_m, part_l, part_acc, q.dtype)
```

- [ ] **Step 4: Add a temporary shim so this task is independently testable**

Task 3 replaces it. Append to `moba_fwd.py`:

```python
def combine_partials(part_m, part_l, part_acc, out_dtype):
    """Reference merge in PyTorch. Replaced by a Triton kernel in Task 3."""
    m_g = part_m.max(dim=-1, keepdim=True).values
    w = torch.exp(part_m - m_g)
    l = (part_l * w).sum(dim=-1)
    acc = (part_acc * w.unsqueeze(-1)).sum(dim=-2)
    return (acc / l.unsqueeze(-1)).to(out_dtype)
```

Move the local import to the top of the module once this exists.

- [ ] **Step 5: Run the test and confirm it passes**

```bash
wsl -e bash -lc 'export PATH="$HOME/.local/bin:$PATH"; export UV_PROJECT_ENVIRONMENT="$HOME/.venvs/moba"; cd "/mnt/c/D/Side Projects/GPU Kernel" && uv run pytest tests/test_moba_kernel.py -q'
```

If it fails, check in this order: `n_spills` on the compiled kernel; whether `GROUP` and `BLOCK_SIZE` are both `>= 16`; whether `part_m` was initialised to `-inf` rather than zeros.

- [ ] **Step 6: Add the full sweep test**

```python
from conftest import sweep_cases


@pytest.mark.parametrize("shape,gate", sweep_cases(include_long=False))
def test_kernel_matches_the_oracle(device, shape, gate):
    block_size, top_k = gate
    cfg = MoBAConfig(block_size=block_size, top_k=top_k, dtype=torch.bfloat16)
    q, k, v = make_qkv(shape, dtype=torch.bfloat16)

    actual = moba_forward(q, k, v, cfg)
    expected = moba_attention_naive(q, k, v, cfg, query_chunk=256)

    assert_within_tolerance(
        actual, expected, reference_error(q, k, v, cfg, query_chunk=256),
        f"blk{block_size}-top{top_k}",
    )
```

- [ ] **Step 7: Run and commit**

```powershell
git add src/moba/kernels/moba_fwd.py tests/test_moba_kernel.py
git commit -m "feat: KV-block-major MoBA attention kernel"
```

---

## Task 3: The combine kernel

**Files:**
- Modify: `src/moba/kernels/moba_fwd.py`
- Test: `tests/test_moba_kernel.py`

**Interfaces:**
- Consumes: `part_m`, `part_l`, `part_acc` from Task 2.
- Produces: `combine_partials(part_m, part_l, part_acc, out_dtype) -> torch.Tensor`, same signature as the Task 2 shim, so nothing else changes.

- [ ] **Step 1: Write the failing test**

```python
def test_triton_combine_matches_the_pytorch_reference(device):
    """The Triton merge must agree with the obvious PyTorch one."""
    torch.manual_seed(0)
    b, h, s, kk, d = 1, 2, 256, 4, 64
    part_m = torch.randn(b, h, s, kk, device=device) * 5
    part_l = torch.rand(b, h, s, kk, device=device) + 0.1
    part_acc = torch.randn(b, h, s, kk, d, device=device)
    part_m[:, :, :, 3] = float("-inf")      # an empty slot must contribute nothing
    part_l[:, :, :, 3] = 0.0
    part_acc[:, :, :, 3] = 0.0

    m_g = part_m.max(dim=-1, keepdim=True).values
    w = torch.exp(part_m - m_g)
    expected = ((part_acc * w.unsqueeze(-1)).sum(-2)
                / (part_l * w).sum(-1).unsqueeze(-1))

    actual = combine_partials(part_m, part_l, part_acc, torch.float32)

    err = (actual - expected).abs().max().item()
    assert err < 1e-5, f"combine mismatch {err:.3e}"
```

- [ ] **Step 2: Run it and confirm it fails against the shim**

The shim is the same computation, so this passes trivially until the shim is replaced. Replace first, then run — this test exists to protect the replacement.

- [ ] **Step 3: Write the Triton combine kernel**

```python
@triton.jit
def _moba_combine(
    PartM, PartL, PartAcc, Out,
    S,
    K_SLOTS: tl.constexpr, D: tl.constexpr, BLOCK_M: tl.constexpr,
):
    """Merges a query's partial softmaxes into its final output.

    This is FA-2's `alpha` correction applied *across programs* rather than
    across loop iterations. Each partial was normalized against its own local
    max, so they cannot simply be added -- that is the failure measured at 800x
    error in learning note 02.
    """
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    live = offs_m < S
    offs_d = tl.arange(0, D)
    base = pid_bh * S * K_SLOTS

    m_g = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    for slot in tl.static_range(K_SLOTS):
        m_i = tl.load(PartM + base + offs_m * K_SLOTS + slot,
                      mask=live, other=float("-inf"))
        m_g = tl.maximum(m_g, m_i)

    l = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, D], dtype=tl.float32)
    for slot in tl.static_range(K_SLOTS):
        m_i = tl.load(PartM + base + offs_m * K_SLOTS + slot,
                      mask=live, other=float("-inf"))
        l_i = tl.load(PartL + base + offs_m * K_SLOTS + slot,
                      mask=live, other=0.0)
        a_i = tl.load(
            PartAcc + pid_bh * S * K_SLOTS * D
            + (offs_m[:, None] * K_SLOTS + slot) * D + offs_d[None, :],
            mask=live[:, None], other=0.0,
        )
        # An empty slot carries m = -inf, so its weight is exp(-inf) = 0 and it
        # drops out without a branch.
        w = tl.exp(m_i - m_g)
        l += l_i * w
        acc += a_i * w[:, None]

    out = acc / l[:, None]
    tl.store(Out + pid_bh * S * D + offs_m[:, None] * D + offs_d[None, :],
             out.to(Out.dtype.element_ty), mask=live[:, None])


def combine_partials(
    part_m: torch.Tensor,
    part_l: torch.Tensor,
    part_acc: torch.Tensor,
    out_dtype: torch.dtype,
    block_m: int = 64,
) -> torch.Tensor:
    """Merges per-slot partial softmaxes into the final attention output.

    Args:
        part_m: Per-slot running maxima, [B, H, S, K] fp32.
        part_l: Per-slot softmax denominators, [B, H, S, K] fp32.
        part_acc: Per-slot un-normalized outputs, [B, H, S, K, D] fp32.
        out_dtype: Dtype of the returned tensor.
        block_m: Queries per program.

    Returns:
        Attention output shaped [B, H, S, D].
    """
    b, h, s, kk = part_m.shape
    d = part_acc.shape[-1]
    out = torch.empty((b, h, s, d), device=part_m.device, dtype=out_dtype)
    _moba_combine[(triton.cdiv(s, block_m), b * h)](
        part_m, part_l, part_acc, out, s,
        K_SLOTS=kk, D=d, BLOCK_M=block_m,
    )
    return out
```

Delete the PyTorch shim and the local import in `moba_forward`.

- [ ] **Step 4: Run both kernel tests**

```bash
wsl -e bash -lc 'export PATH="$HOME/.local/bin:$PATH"; export UV_PROJECT_ENVIRONMENT="$HOME/.venvs/moba"; cd "/mnt/c/D/Side Projects/GPU Kernel" && uv run pytest tests/test_moba_kernel.py -q'
```

- [ ] **Step 5: Commit**

```powershell
git add src/moba/kernels/moba_fwd.py tests/test_moba_kernel.py
git commit -m "feat: Triton log-sum-exp combine over partial softmaxes"
```

---

## Task 4: Register as a backend

**Files:**
- Create: `src/moba/backends/triton_ours.py`
- Modify: `src/moba/backends/__init__.py`
- Test: `tests/test_registry.py`

**Interfaces:**
- Consumes: `moba_forward`, the `Backend` protocol from `moba.backends.base`.
- Produces: a backend named `moba_triton`, `vendored = False`, exposing `selection()` so the Tier-2 exact routing check applies to it.

- [ ] **Step 1: Write the failing test**

```python
def test_our_triton_backend_is_registered_and_available():
    from moba.backends import get_backend
    from moba.config import MoBAConfig

    backend = get_backend("moba_triton")
    ok, reason = backend.is_available(
        MoBAConfig(block_size=32, top_k=2, dtype=torch.bfloat16)
    )
    assert ok, reason
    assert backend.vendored is False
```

- [ ] **Step 2: Run it and confirm it fails** with an unknown-backend error.

- [ ] **Step 3: Implement the adapter**

```python
"""Our Triton MoBA kernel, behind the common Backend interface."""

import torch

from moba.backends.base import register
from moba.config import MoBAConfig
from moba.device import MIN_CAPABILITY, supports_dtype
from moba.kernels.index import build_csc
from moba.kernels.moba_fwd import moba_forward


class TritonMoBABackend:
    """Adapter for `moba.kernels.moba_fwd`."""

    name = "moba_triton"
    vendored = False

    def is_available(self, cfg: MoBAConfig) -> tuple[bool, str]:
        """Reports whether this backend can run the given configuration."""
        if not torch.cuda.is_available():
            return False, "no CUDA device"
        cap = torch.cuda.get_device_capability(0)
        if cap < MIN_CAPABILITY:
            return False, f"needs sm_{MIN_CAPABILITY[0]}{MIN_CAPABILITY[1]}, got {cap}"
        ok, reason = supports_dtype(cfg.dtype, cap)
        if not ok:
            return False, reason
        if cfg.block_size < 16:
            return False, f"block_size {cfg.block_size} below the tl.dot minimum"
        return True, ""

    def forward(self, q, k, v, cfg: MoBAConfig) -> torch.Tensor:
        """Runs the kernel."""
        return moba_forward(q, k, v, cfg)

    def selection(self, q, k, cfg: MoBAConfig) -> torch.Tensor:
        """Exposes routing so the exact-match check applies to this backend."""
        from moba.kernels.index import csc_to_mask
        return csc_to_mask(build_csc(q, k, cfg), q.shape[2])


register(TritonMoBABackend())
```

Add `from moba.backends import triton_ours  # noqa: F401` to `src/moba/backends/__init__.py` alongside the existing registration imports.

- [ ] **Step 4: Run the full suite**

```bash
wsl -e bash -lc 'export PATH="$HOME/.local/bin:$PATH"; export UV_PROJECT_ENVIRONMENT="$HOME/.venvs/moba"; cd "/mnt/c/D/Side Projects/GPU Kernel" && uv run pytest -q -m "not slow"'
```

Tier-2 backend tests now exercise our kernel automatically, since it is non-vendored.

- [ ] **Step 5: Commit**

```powershell
git add src/moba/backends/triton_ours.py src/moba/backends/__init__.py tests/test_registry.py
git commit -m "feat: register the Triton MoBA kernel as a backend"
```

---

## Task 5: Invariant tests

**Files:**
- Create: `tests/test_kernel_invariants.py`

These catch bugs no reference can, because the oracle could share the same mistake. Each asserts a property the algorithm must have.

- [ ] **Step 1: Write all four tests**

```python
"""Properties the kernel must satisfy regardless of any reference.

Comparing against an oracle only catches bugs the oracle does not share. These
test the algorithm's own invariants, and the first is a direct test of the
gather indices -- the part of the inverted design most exposed to error.
"""

import torch

from conftest import make_qkv
from moba.config import MoBAConfig
from moba.kernels.index import build_csc
from moba.kernels.moba_fwd import moba_forward
from moba.ref.naive import moba_selection

CFG = MoBAConfig(block_size=32, top_k=2, dtype=torch.bfloat16)
SHAPE = (1, 2, 512, 64)


def test_unselected_blocks_do_not_affect_output(device):
    """Perturbing a block a query did not select must change nothing for it."""
    q, k, v = make_qkv(SHAPE, dtype=torch.bfloat16)
    mask = moba_selection(q, k, CFG)                 # [B, H, S, N]
    index = build_csc(q, k, CFG)

    before = moba_forward(q, k, v, CFG, index=index)

    # Pick a block, then a query that did not select it.
    blk = 4
    not_selected = (~mask[0, 0, :, blk]).nonzero().squeeze(1)
    assert not_selected.numel() > 0, "test shape gives no unselected query"
    row = int(not_selected[-1])

    k2, v2 = k.clone(), v.clone()
    lo = blk * CFG.block_size
    k2[0, 0, lo:lo + CFG.block_size] += 10.0
    v2[0, 0, lo:lo + CFG.block_size] += 10.0

    # Hold routing fixed: the perturbation would otherwise change the gate too,
    # and we are testing the attention path, not the router.
    after = moba_forward(q, k2, v2, CFG, index=index)

    assert torch.equal(before[0, 0, row], after[0, 0, row]), (
        f"query {row} changed when only unselected block {blk} was perturbed "
        "-- the gather is reading rows it should not"
    )


def test_causality(device):
    """Perturbing keys after position i must leave rows <= i untouched."""
    q, k, v = make_qkv(SHAPE, dtype=torch.bfloat16)
    index = build_csc(q, k, CFG)
    before = moba_forward(q, k, v, CFG, index=index)

    cut = 256
    k2, v2 = k.clone(), v.clone()
    k2[:, :, cut:] += 10.0
    v2[:, :, cut:] += 10.0
    after = moba_forward(q, k2, v2, CFG, index=index)

    assert torch.equal(before[:, :, :cut], after[:, :, :cut]), (
        "future keys leaked into earlier output rows"
    )
    assert not torch.equal(before[:, :, cut:], after[:, :, cut:]), (
        "nothing changed at all -- the perturbation never reached the kernel"
    )


def test_dispatch_order_does_not_change_results(device):
    """Reordering which program runs when must not alter the answer.

    Doubles as a race detector: if two programs ever write the same slot, the
    winner depends on timing, and timing depends on dispatch order.
    """
    q, k, v = make_qkv(SHAPE, dtype=torch.bfloat16)
    index = build_csc(q, k, CFG)
    sorted_out = moba_forward(q, k, v, CFG, index=index)

    identity = torch.arange(
        index.num_blocks, device=q.device, dtype=torch.int32
    ).expand_as(index.order).contiguous()
    unsorted = type(index)(
        col_nnz=index.col_nnz, col_offsets=index.col_offsets,
        row_indices=index.row_indices, slot_of_pair=index.slot_of_pair,
        order=identity, num_blocks=index.num_blocks, max_slots=index.max_slots,
    )
    unsorted_out = moba_forward(q, k, v, CFG, index=unsorted)

    assert torch.equal(sorted_out, unsorted_out), (
        "dispatch order changed the result -- likely a write race"
    )


def test_determinism(device):
    """Identical inputs must give bitwise identical outputs across runs."""
    q, k, v = make_qkv(SHAPE, dtype=torch.bfloat16)
    index = build_csc(q, k, CFG)
    first = moba_forward(q, k, v, CFG, index=index)
    for _ in range(3):
        assert torch.equal(first, moba_forward(q, k, v, CFG, index=index))
```

- [ ] **Step 2: Run them**

```bash
wsl -e bash -lc 'export PATH="$HOME/.local/bin:$PATH"; export UV_PROJECT_ENVIRONMENT="$HOME/.venvs/moba"; cd "/mnt/c/D/Side Projects/GPU Kernel" && uv run pytest tests/test_kernel_invariants.py -q'
```

A failure in `test_unselected_blocks_do_not_affect_output` means the gather is reading wrong rows. A failure in `test_dispatch_order_does_not_change_results` means two programs are writing the same slot — check `slot_of_pair` uniqueness from Task 1.

- [ ] **Step 3: Commit**

```powershell
git add tests/test_kernel_invariants.py
git commit -m "test: kernel invariants requiring no reference implementation"
```

---

## Task 6: Per-stage benchmarking

**Files:**
- Create: `bench/stages.py`
- Modify: `docs/baseline-results.md`

The spec's first risk is that the crossover is unmeasured and the inversion's overheads are unpriced. This makes each stage attributable.

- [ ] **Step 1: Write the stage timer**

```python
"""Times each stage of the MoBA kernel separately.

The design's central risk is that utilization 0.028 proves query-major cannot
win, without proving the inverted kernel does -- because index construction,
gathered reads and the combine pass are unpriced. This prices them.

D11 keeps the gate in PyTorch on the argument that it should be measured before
being optimized. This is that measurement.
"""

import torch
import triton.testing

from moba.config import MoBAConfig
from moba.kernels.index import build_csc
from moba.kernels.moba_fwd import combine_partials, moba_forward


def time_stages(shape, gate, dtype=torch.bfloat16, seed=0) -> dict:
    """Times index build, attention and combine for one configuration.

    Args:
        shape: (batch, heads, seq_len, head_dim).
        gate: (block_size, top_k).
        dtype: Compute dtype.
        seed: RNG seed.

    Returns:
        A dict of per-stage milliseconds plus the end-to-end total.
    """
    block_size, top_k = gate
    cfg = MoBAConfig(block_size=block_size, top_k=top_k, dtype=dtype)
    g = torch.Generator(device="cuda").manual_seed(seed)
    q, k, v = (
        torch.randn(shape, generator=g, device="cuda", dtype=dtype)
        for _ in range(3)
    )

    index = build_csc(q, k, cfg)
    t_index = triton.testing.do_bench(lambda: build_csc(q, k, cfg))
    t_total = triton.testing.do_bench(lambda: moba_forward(q, k, v, cfg))
    t_attn_plus_combine = triton.testing.do_bench(
        lambda: moba_forward(q, k, v, cfg, index=index)
    )

    return {
        "seq_len": shape[2],
        "block_size": block_size,
        "top_k": top_k,
        "index_ms": t_index,
        "attn_combine_ms": t_attn_plus_combine,
        "total_ms": t_total,
        "index_fraction": t_index / t_total,
    }


if __name__ == "__main__":
    rows = [
        time_stages((1, 8, s, 64), (bk, 2))
        for s in (1024, 2048, 4096)
        for bk in (16, 32, 128)
    ]
    print("| S | blk | index ms | attn+combine ms | total ms | index share |")
    print("|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        print(f"| {r['seq_len']} | {r['block_size']} | {r['index_ms']:.3f} | "
              f"{r['attn_combine_ms']:.3f} | {r['total_ms']:.3f} | "
              f"{r['index_fraction']:.1%} |")
```

- [ ] **Step 2: Run it**

```bash
wsl -e bash -lc 'export PATH="$HOME/.local/bin:$PATH"; export UV_PROJECT_ENVIRONMENT="$HOME/.venvs/moba"; cd "/mnt/c/D/Side Projects/GPU Kernel" && uv run python -m bench.stages'
```

- [ ] **Step 3: Run the full benchmark suite with our backend included**

```bash
wsl -e bash -lc 'export PATH="$HOME/.local/bin:$PATH"; export UV_PROJECT_ENVIRONMENT="$HOME/.venvs/moba"; cd "/mnt/c/D/Side Projects/GPU Kernel" && uv run python -c "
from bench.suite import run_suite
from bench.report import to_csv, to_markdown
rows = run_suite(shapes=[(1,8,1024,64),(1,8,2048,64)],
                 gates=[(16,2),(32,2),(64,2),(128,2)])
to_csv(rows, \"results/kernel_v1.csv\")
print(to_markdown(rows))
"'
```

- [ ] **Step 4: Record results**

Add a "Kernel v1" section to `docs/baseline-results.md` with the attainment table and the per-stage breakdown. State plainly whether the spec's success criteria were met:

- Threshold: faster than dense SDPA at S >= 2048, `block_size=16`
- Target: attainment >= 0.10 at S=4096
- If `index_fraction` exceeds ~20%, the D11 deferral trigger has fired — record that, do not silently absorb it.

- [ ] **Step 5: Commit**

```powershell
git add bench/stages.py docs/baseline-results.md results/kernel_v1.csv
git commit -m "feat: per-stage kernel benchmarking and v1 results"
```

---

## Self-review notes

**Spec coverage.** D6 → Task 2 (grid over KV blocks). D7 → Task 2 (gathered `tl.load`). D8 → Task 1 (`order`) and Task 2 (indirection). D9 → Task 2 (`CAUSAL` predicated select, `K = top_k` slots). D10 → Tasks 2 and 3 (fixed slots, separate combine). D11 → Task 1 (PyTorch gate) and Task 6 (its measurement). All four spec invariants → Task 5. Index unit tests → Task 1.

**Not covered here, by design.** The query-major control kernel, the Triton gate, chunking, and the backward pass are all in the spec's Deferred table with triggers. Task 6 produces the measurement that fires the D11 trigger if it applies.

**Known sharp edges.**
- Task 2's `for start in range(0, n, GROUP)` has a runtime bound. If Triton rejects it, restructure as a `while` loop over a counter.
- `tl.static_range` in Task 3 requires `K_SLOTS` to be a compile-time constant; it is, since it comes from `cfg.top_k`.
- `part_acc` at `[B,H,S,K,D]` fp32 is 25 MB at S=4096, K=4, H=8. Watch the 4 GB budget when the sweep grows.
