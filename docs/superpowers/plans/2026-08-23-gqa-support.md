# Grouped-Query Attention Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let the MoBA kernels accept fewer key/value heads than query heads
(`H_kv < H_q`, `H_q % H_kv == 0`), keeping multi-head attention as the `H_kv ==
H_q` special case.

**Architecture:** One rule governs the whole change. **Tensors indexed by a
query keep `H_q`; tensors indexed by a key switch to `H_kv`.** A CSC column is
keyed by `(batch, kv_head, block)` rather than `(batch, head, block)`, so the
column no longer determines which query head an entry came from — every entry
therefore carries its query head alongside its position, in a new
`head_indices` array parallel to `row_indices`.

**Tech Stack:** Triton 3.7.1, PyTorch 2.13.0+cu126, Python 3.13, pytest.

**Spec:** No separate design doc; the decisions were settled in conversation and
are recorded under "Design decisions" below. Background on the loop order these
decisions build on is in
`docs/superpowers/specs/2026-08-18-moba-kernel-design.md`, decisions D6 (KV-block-major),
D8 (longest-first dispatch) and D10 (fixed-slot partials).

---

## Global Constraints

- **Google Style** for all code and docstrings (Args/Returns/Raises, 4-space
  indent). Project-wide, per `CLAUDE.md`.
- **Never edit `references/`.** Vendored, read-only, gitignored.
- **No `Co-Authored-By:` lines** referencing Claude or Anthropic in commits.
- **Run everything through WSL**, never native Windows:
  ```bash
  wsl -e bash -lc 'export PATH="$HOME/.local/bin:$PATH"; \
    export UV_PROJECT_ENVIRONMENT="$HOME/.venvs/moba"; \
    cd "/mnt/c/D/Side Projects/GPU Kernel" && uv run <command>'
  ```
- **`git` runs from PowerShell**, not WSL — git identity is set on Windows only.
- **4 GiB VRAM.** Keep test shapes small: batch 1, `seq_len` ≤ 2048 for
  correctness work. GQA multiplies column lengths, so a shape that fit under MHA
  can OOM here.
- **Correctness policy is `moba.tolerance`**, never an absolute threshold: pass
  if our error is within `TOLERANCE_FACTOR` (2.0) of the same-dtype oracle's own
  deviation from an fp64 truth.
- **Measured hardware:** RTX 3050 Ti Laptop, sm_86, 20 SMs, 101376 B shared
  memory per block.

---

## Design decisions

Each of these was chosen against a named alternative. Implementers should not
silently re-decide them.

**D-GQA-1 — Group size is derived from shapes, not configured.**
`group = H_q // H_kv`, validated by `H_q % H_kv == 0`. *Rejected:* a
`MoBAConfig` field. A config field could contradict the tensors it is used with,
producing a second source of truth for a fact the shapes already state
unambiguously. `MoBAConfig` exists for genuinely ambiguous *semantics* (tie
breaks, own-block handling), not for restating shapes.

**D-GQA-2 — Each query head runs its own independent top-k.**
*Rejected:* one shared selection per group. Shared selection would give perfect
K/V amortization and a `group`× smaller index, but it changes the algorithm —
neither the MoBA paper nor FlashMoBA specifies group-shared routing, and a
kernel that quietly attends to different blocks than the reference is not the
kernel we set out to write. Independent routing keeps `moba_selection` the
executable definition it has been throughout.

**D-GQA-3 — Query head travels in a separate `head_indices` array.**
*Rejected:* packing head and position into one integer as `head * S + pos` and
recovering the position with `%`. Packing saves one `[GROUP]` int32 gather per
iteration (128 B against a 16 KB K/V tile — noise) but costs a runtime integer
modulo *inside the inner loop*, on hardware with no integer divider, and forces
an edit to the causal comparison `key_pos <= rows`, which is currently correct
and should stay untouched. The separate array leaves the causal mask a zero-line
diff.

**D-GQA-4 — Longer lists, not wider tiles.**
Under GQA a column's length `n` grows by roughly `group` because `group` columns
merged into one. The kernel already streams an arbitrary-length list through a
fixed `GROUP`-sized tile (`for start in range(0, n, GROUP)`), so this is more
loop iterations at **identical** register and shared-memory footprint. No tuning
constant changes. *Rejected:* fusing the group's heads into one wider tile,
which would multiply register pressure by `group` and collapse occupancy for no
gain — the loop already amortizes the K/V load, which is the entire point.

**D-GQA-5 — `_moba_combine` and `_moba_dq_reduce` are not modified.**
Both launch over `b * h` and address everything by `(query, slot)`. Under GQA
their `h` simply is `H_q` and every tensor they touch is query-indexed already.
Verified by reading `moba_fwd.py:410` and `moba_bwd.py:244`. If an implementer
finds themselves editing either kernel, something upstream is wrong — stop and
re-check the index layout rather than patching here.

### Index layout, before and after

| Tensor | Shape (MHA) | Shape (GQA) |
|---|---|---|
| `q`, `out`, `dq`, `lse`, `delta` | `[B, H, S, D]` | `[B, H_q, S, D]` |
| `k`, `v`, `dk`, `dv` | `[B, H, S, D]` | `[B, H_kv, S, D]` |
| `kbar` | `[B, H, N, D]` | `[B, H_kv, N, D]` |
| `part_m`, `part_l` | `[B, H, S, K]` | `[B, H_q, S, K]` |
| `part_acc` | `[B, H, S, K, D]` | `[B, H_q, S, K, D]` |
| `block_idx` (gate output) | `[B, H, S, top_k]` | `[B, H_q, S, top_k]` |
| `col_nnz`, `col_offsets`, `order` | `[B, H, N(+1)]` | `[B, H_kv, N(+1)]` |
| `row_indices`, `slot_of_pair` | `[B, H, S*top_k]` | `[B, H_kv, G*S*top_k]` |
| **`head_indices`** (new) | — | `[B, H_kv, G*S*top_k]` |

`G = H_q // H_kv`. Total `(query, block)` pairs is unchanged at
`B * H_q * S * top_k`; they are redistributed over `G`× fewer columns.

### Address arithmetic in the KV-major kernels

`pid_bh` currently enumerates `B * H` and bases both the query side and the key
side. It splits:

```python
pid_bh = tl.program_id(1)          # now enumerates B * H_KV
pid_b = pid_bh // H_KV             # batch
# key side, unchanged in form:
kv_base = pid_bh * S * D
# query side, new:
q_row = heads * S + rows           # heads and rows both [GROUP] int32
q_base = pid_b * H_Q * S * D
part_base = pid_b * H_Q * S * K_SLOTS
```

The causal mask still compares `key_pos <= rows` — `rows` remains a raw
position, which is why D-GQA-3 keeps the head separate.

---

## File Structure

**Create:**
- `src/moba/heads.py` — group-size validation and a test-only KV expander. One
  responsibility: reconciling a query head count with a KV head count.
- `tests/test_heads.py` — unit tests for the above.
- `tests/test_gqa.py` — end-to-end GQA correctness, forward and backward.
- `docs/gqa-measurements.md` — measured arithmetic-intensity and speedup effect.

**Modify:**
- `src/moba/ref/naive.py` — oracle accepts `H_kv < H_q`.
- `src/moba/kernels/moba_fwd.py` — `_validate`, `_moba_attn_fwd`,
  `attend_partials`, `empty_partials`.
- `src/moba/kernels/gate.py` — `_moba_gate_fwd` (one pointer line), `gate_select`.
- `src/moba/kernels/index.py` — `CSCIndex` gains `head_indices`; `build_csc` and
  `csc_to_mask` become GQA-aware.
- `src/moba/kernels/index_fused.py` — three passes gain a head axis.
- `src/moba/kernels/moba_bwd.py` — `_moba_bwd`, `moba_backward`.
- `tests/conftest.py` — `make_qkv` grows an optional KV-head count.
- `bench/suite.py` — a GQA row.

**Untouched by design:** `_moba_combine`, `_moba_dq_reduce`, `moba/config.py`,
`moba/tolerance.py`, `moba/device.py`.

---

## Task 1: Head-group arithmetic and the oracle

Pure PyTorch, no kernels. Establishes the definition everything else is checked
against, and gets the "GQA is MHA when `G == 1`" property nailed down before any
Triton is touched.

**Files:**
- Create: `src/moba/heads.py`
- Create: `tests/test_heads.py`
- Modify: `src/moba/ref/naive.py`
- Test: `tests/test_heads.py`

**Interfaces:**
- Produces:
  - `moba.heads.head_group_size(num_query_heads: int, num_kv_heads: int) -> int`
  - `moba.heads.expand_kv(t: torch.Tensor, num_query_heads: int) -> torch.Tensor`
  - `moba.ref.naive.block_representatives`, `select_blocks`, `moba_selection`,
    `moba_attention_naive` all accept `k`/`v` with `H_kv <= H_q`.
  - `moba_selection(q, k, cfg)` returns `[B, H_q, S, num_blocks]`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_heads.py`:

```python
"""Reconciling a query head count with a smaller key/value head count.

Grouped-query attention is defined entirely by shapes here: the group size is
`H_q // H_kv` and there is no configuration field for it. A config field would
be a second source of truth for something the tensors already state, and the
two could disagree. See `docs/superpowers/plans/2026-08-23-gqa-support.md`,
decision D-GQA-1.
"""

import pytest
import torch

from moba.heads import expand_kv, head_group_size


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


def test_expand_kv_repeats_each_head_to_cover_its_group():
    """Head g of the expansion must be head g//G of the original, exactly."""
    t = torch.arange(2 * 2 * 3 * 4, dtype=torch.float32).view(2, 2, 3, 4)
    wide = expand_kv(t, 6)
    assert wide.shape == (2, 6, 3, 4)
    for qh in range(6):
        assert torch.equal(wide[:, qh], t[:, qh // 3])


def test_expand_kv_is_the_identity_at_equal_head_counts():
    t = torch.randn(1, 4, 8, 16)
    assert torch.equal(expand_kv(t, 4), t)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run:
```bash
wsl -e bash -lc 'export PATH="$HOME/.local/bin:$PATH"; \
  export UV_PROJECT_ENVIRONMENT="$HOME/.venvs/moba"; \
  cd "/mnt/c/D/Side Projects/GPU Kernel" && uv run pytest tests/test_heads.py -v'
```
Expected: FAIL — `ModuleNotFoundError: No module named 'moba.heads'`.

- [ ] **Step 3: Create `src/moba/heads.py`**

```python
"""Reconciling a query head count with a smaller key/value head count.

Grouped-query attention lets several query heads share one key/value head. The
group size is not configured anywhere: it is `H_q // H_kv`, read off the
tensors. A `MoBAConfig` field would be a second source of truth for a fact the
shapes already state, and the two could disagree; `MoBAConfig` exists for
genuinely ambiguous *semantics*, not for restating shapes.

`expand_kv` is oracle and test support only. Materializing the repeated heads
is precisely the memory traffic GQA exists to avoid, so no kernel path may call
it -- the kernels share the K/V tile by owning it, not by copying it.
"""

import torch


def head_group_size(num_query_heads: int, num_kv_heads: int) -> int:
    """Returns how many query heads share each key/value head.

    Args:
        num_query_heads: Number of query heads.
        num_kv_heads: Number of key/value heads.

    Returns:
        The group size, 1 for plain multi-head attention.

    Raises:
        ValueError: If the query head count is not a positive multiple of the
            key/value head count.
    """
    if num_kv_heads <= 0 or num_query_heads <= 0:
        raise ValueError(
            f"head counts must be positive, got q={num_query_heads} "
            f"kv={num_kv_heads}"
        )
    if num_query_heads % num_kv_heads != 0:
        raise ValueError(
            f"query heads {num_query_heads} must be divisible by key/value "
            f"heads {num_kv_heads}; a fractional group has no meaning"
        )
    return num_query_heads // num_kv_heads


def expand_kv(t: torch.Tensor, num_query_heads: int) -> torch.Tensor:
    """Repeats each key/value head across the query heads that share it.

    Test and oracle support only; see the module docstring.

    Args:
        t: Tensor shaped [B, H_kv, S, D].
        num_query_heads: Target head count.

    Returns:
        Tensor shaped [B, num_query_heads, S, D]. Returns `t` unchanged when
        the head counts already match, so the common case copies nothing.

    Raises:
        ValueError: If the head counts are incompatible.
    """
    group = head_group_size(num_query_heads, t.shape[1])
    if group == 1:
        return t
    return t.repeat_interleave(group, dim=1)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_heads.py -v` (with the WSL prefix).
Expected: 6 passed.

- [ ] **Step 5: Write the failing oracle test**

Append to `tests/test_heads.py`:

```python
def test_oracle_under_gqa_equals_the_oracle_on_expanded_keys(device):
    """The definition of GQA: identical to MHA over repeated K/V heads.

    Checked bitwise, not approximately. `expand_kv` produces the very same
    numbers the GQA path must consume, so any difference is a routing or
    indexing bug rather than an arithmetic one, and rounding cannot explain it.
    """
    from moba.config import MoBAConfig
    from moba.ref.naive import moba_attention_naive, moba_selection

    cfg = MoBAConfig(block_size=64, top_k=4, dtype=torch.float64)
    b, h_q, h_kv, s, d = 1, 8, 2, 512, 64
    torch.manual_seed(0)
    q = torch.randn(b, h_q, s, d, dtype=torch.float64, device=device)
    k = torch.randn(b, h_kv, s, d, dtype=torch.float64, device=device)
    v = torch.randn(b, h_kv, s, d, dtype=torch.float64, device=device)

    narrow = moba_attention_naive(q, k, v, cfg)
    wide = moba_attention_naive(q, expand_kv(k, h_q), expand_kv(v, h_q), cfg)
    assert torch.equal(narrow, wide)

    sel_narrow = moba_selection(q, k, cfg)
    sel_wide = moba_selection(q, expand_kv(k, h_q), cfg)
    assert sel_narrow.shape == (b, h_q, s, cfg.num_blocks(s))
    assert torch.equal(sel_narrow, sel_wide)
```

- [ ] **Step 6: Run it to verify it fails**

Run: `uv run pytest tests/test_heads.py::test_oracle_under_gqa_equals_the_oracle_on_expanded_keys -v`
Expected: FAIL — a shape error from `torch.einsum` in `select_blocks`, since
`q` has 8 heads and `kbar` has 2.

- [ ] **Step 7: Make the oracle GQA-aware**

In `src/moba/ref/naive.py`, add the import:

```python
from moba.heads import expand_kv
```

In `select_blocks`, immediately after `num_blocks = kbar.shape[2]`, insert:

```python
    # Grouped-query attention. Expanding here rather than broadcasting in the
    # einsum keeps this function one line per line of the algorithm, which is
    # the oracle's whole job -- the kernels get to be clever, this does not.
    kbar = expand_kv(kbar, h)
```

In `moba_attention_naive`, immediately after `b, h, s, d = q.shape`, insert:

```python
    # See `select_blocks`: the oracle materializes the shared heads. This is
    # exactly the traffic the kernel avoids, and doing it here is what makes
    # the kernel's avoidance checkable.
    k, v = expand_kv(k, h), expand_kv(v, h)
```

Update the three docstrings to say `k: Keys shaped [B, H_kv, S, D], where
`H_kv` divides `H`.`

- [ ] **Step 8: Run the whole oracle-facing suite**

Run: `uv run pytest tests/test_heads.py tests/test_semantics.py tests/test_smoke.py -v`
Expected: all pass. The pre-existing tests use `H_kv == H_q`, where `expand_kv`
returns its input unchanged, so none of them may change behaviour.

- [ ] **Step 9: Commit**

```bash
git add src/moba/heads.py tests/test_heads.py src/moba/ref/naive.py docs/superpowers/plans/2026-08-23-gqa-support.md
git commit -m "feat: GQA in the oracle, with the group size read off the shapes"
```

---

## Task 2: Gate under GQA

The gate scores `H_q` query heads against `H_kv` sets of block representatives.
One pointer line in the kernel; the rest is plumbing.

**Files:**
- Modify: `src/moba/kernels/gate.py:82` (`_moba_gate_fwd`), `:153` (the `Kbar`
  load), `:251` (`gate_select`)
- Test: `tests/test_gate.py`

**Interfaces:**
- Consumes: `moba.heads.head_group_size` (Task 1).
- Produces: `gate_select(q, kbar, cfg)` accepts `q` `[B, H_q, S, D]` and `kbar`
  `[B, H_kv, N, D]`, returning `[B, H_q, S, top_k]` int32.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_gate.py`:

```python
def test_gate_scores_each_query_head_against_its_shared_representatives(device):
    """A query head must rank against its own KV head's blocks, and no other.

    The failure this guards against is silent and plausible: with the wrong
    divisor the gate ranks head 5 against head 5's representatives instead of
    head 1's, which produces a perfectly well-formed index that attends to the
    wrong keys. Comparing against the expanded oracle catches it.
    """
    from moba.heads import expand_kv
    from moba.kernels.gate import block_representatives_triton, gate_select

    cfg = MoBAConfig(block_size=64, top_k=4, dtype=torch.bfloat16)
    b, h_q, h_kv, s, d = 1, 8, 2, 512, 64
    torch.manual_seed(0)
    q = torch.randn(b, h_q, s, d, dtype=torch.bfloat16, device=device)
    k = torch.randn(b, h_kv, s, d, dtype=torch.bfloat16, device=device)

    kbar = block_representatives_triton(k, cfg)
    assert kbar.shape == (b, h_kv, cfg.num_blocks(s), d)

    narrow = gate_select(q, kbar, cfg)
    wide = gate_select(q, expand_kv(kbar, h_q), cfg)
    assert narrow.shape == (b, h_q, s, cfg.top_k)
    assert torch.equal(narrow, wide)


def test_gate_rejects_indivisible_head_counts(device):
    from moba.kernels.gate import gate_select

    cfg = MoBAConfig(block_size=64, top_k=4, dtype=torch.bfloat16)
    q = torch.randn(1, 8, 256, 64, dtype=torch.bfloat16, device=device)
    kbar = torch.randn(1, 3, 4, 64, dtype=torch.bfloat16, device=device)
    with pytest.raises(ValueError, match="divisible"):
        gate_select(q, kbar, cfg)
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/test_gate.py -k "shared_representatives or indivisible" -v`
Expected: FAIL — `gate_select` raises "representative shape ... does not match
queries", from its own shape guard.

- [ ] **Step 3: Add the head mapping to the kernel**

In `src/moba/kernels/gate.py`, extend the `_moba_gate_fwd` signature with two
constexprs after `N_BLOCKS`:

```python
    N_BLOCKS: tl.constexpr, H_Q: tl.constexpr, Q_PER_KV: tl.constexpr,
```

After `pid_bh = tl.program_id(1)` (line 97), insert:

```python
    # `pid_bh` enumerates B * H_Q -- the gate is per *query* head, because each
    # query head runs its own independent top-k (D-GQA-2). Only the block
    # representatives are shared, so only their base pointer is remapped.
    kv_bh = (pid_bh // H_Q) * (H_Q // Q_PER_KV) + (pid_bh % H_Q) // Q_PER_KV
```

Change the `Kbar` load at line 153 from `pid_bh * N_BLOCKS * D` to:

```python
            Kbar + kv_bh * N_BLOCKS * D + cols[:, None] * D + offs_d[None, :],
```

Leave the `Q` load (line 104) and the `BlockIdx` store (line 241) exactly as
they are — both are query-indexed and `pid_bh` is still the right base.

- [ ] **Step 4: Update the wrapper**

In `gate_select`, replace the shape guard:

```python
    b, h, s, d = q.shape
    if kbar.shape[0] != b or kbar.shape[-1] != d:
        raise ValueError(
            f"representative shape {tuple(kbar.shape)} does not match queries "
            f"{tuple(q.shape)}"
        )
    q_per_kv = head_group_size(h, kbar.shape[1])
    n = kbar.shape[2]
```

Add the import at the top of the module:

```python
from moba.heads import head_group_size
```

Pass the two new constexprs in the launch, alongside `N_BLOCKS=n`:

```python
        N_BLOCKS=n, H_Q=h, Q_PER_KV=q_per_kv,
```

- [ ] **Step 5: Run the full gate suite**

Run: `uv run pytest tests/test_gate.py -v`
Expected: all pass, including the 34 pre-existing tests — every one of them uses
`H_kv == H_q`, where `Q_PER_KV == 1` and `kv_bh == pid_bh`.

- [ ] **Step 6: Commit**

```bash
git add src/moba/kernels/gate.py tests/test_gate.py
git commit -m "feat: gate scores query heads against shared representatives"
```

---

## Task 3: CSC index keyed by KV head

The structural heart of the change. Columns merge; entries gain a head.

**Files:**
- Modify: `src/moba/kernels/index.py` (`CSCIndex`, `build_csc`, `csc_to_mask`)
- Modify: `src/moba/kernels/index_fused.py` (all three passes,
  `build_csc_from_indices`, `build_csc_fused`)
- Test: `tests/test_index.py`, `tests/test_index_fused.py`

**Interfaces:**
- Consumes: `gate_select` returning `[B, H_q, S, top_k]` (Task 2);
  `head_group_size` (Task 1).
- Produces:
  - `CSCIndex` gains fields `head_indices: torch.Tensor` (`[B, H_kv, T]` int32,
    aligned with `row_indices`) and `num_query_heads: int`.
  - `build_csc(q, k, cfg)` and `build_csc_fused(q, k, cfg)` accept `H_kv < H_q`.
  - `csc_to_mask(index, seq_len)` returns `[B, H_q, seq_len, num_blocks]`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_index_fused.py`:

```python
def test_columns_merge_across_the_query_heads_that_share_a_kv_head(device):
    """Four query heads sharing one KV head produce one column, not four.

    This is the whole mechanism: the total number of (query, block) pairs is
    unchanged, but they are redistributed over `group` times fewer columns.
    Each column is one program's job and loads K and V once, so `group` times
    fewer columns means `group` times fewer K/V loads for identical arithmetic.
    """
    from moba.config import MoBAConfig
    from moba.kernels.index_fused import build_csc_fused

    cfg = MoBAConfig(block_size=64, top_k=4, dtype=torch.bfloat16)
    b, h_q, h_kv, s, d = 1, 8, 2, 512, 64
    torch.manual_seed(0)
    q = torch.randn(b, h_q, s, d, dtype=torch.bfloat16, device=device)
    k = torch.randn(b, h_kv, s, d, dtype=torch.bfloat16, device=device)

    index = build_csc_fused(q, k, cfg)
    assert index.col_nnz.shape == (b, h_kv, cfg.num_blocks(s))
    assert index.head_indices.shape == index.row_indices.shape
    assert index.num_query_heads == h_q

    # Pair count is conserved: same queries, same top_k, fewer containers.
    from moba.kernels.gate import block_representatives_triton, gate_select
    kbar = block_representatives_triton(k, cfg)
    expected = int((gate_select(q, kbar, cfg) >= 0).sum())
    assert int(index.col_nnz.sum()) == expected


def test_index_round_trips_to_the_gate_mask_under_gqa(device):
    """Expanding the index back out must reproduce the gate's own decision.

    A wrong transpose does not crash -- it attends to the wrong keys and
    returns plausible numbers. The round trip is what makes it visible.
    """
    from moba.config import MoBAConfig
    from moba.kernels.gate import block_representatives_triton, gate_select
    from moba.kernels.index import csc_to_mask
    from moba.kernels.index_fused import build_csc_fused

    cfg = MoBAConfig(block_size=64, top_k=4, dtype=torch.bfloat16)
    b, h_q, h_kv, s, d = 1, 8, 2, 512, 64
    torch.manual_seed(0)
    q = torch.randn(b, h_q, s, d, dtype=torch.bfloat16, device=device)
    k = torch.randn(b, h_kv, s, d, dtype=torch.bfloat16, device=device)

    kbar = block_representatives_triton(k, cfg)
    block_idx = gate_select(q, kbar, cfg)

    n = cfg.num_blocks(s)
    expected = torch.zeros(b, h_q, s, n, dtype=torch.bool, device=device)
    for j in range(cfg.top_k):
        sel = block_idx[..., j]
        ok = sel >= 0
        expected.scatter_(-1, sel.clamp(min=0).long().unsqueeze(-1),
                          ok.unsqueeze(-1))

    got = csc_to_mask(build_csc_fused(q, k, cfg), s)
    assert got.shape == (b, h_q, s, n)
    assert torch.equal(got, expected)
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/test_index_fused.py -k gqa -v` and
`uv run pytest tests/test_index_fused.py -k merge -v`
Expected: FAIL — `AttributeError: 'CSCIndex' object has no attribute
'head_indices'`.

- [ ] **Step 3: Extend `CSCIndex`**

In `src/moba/kernels/index.py`, add two fields to the dataclass and document
them:

```python
    head_indices: torch.Tensor
    num_query_heads: int
```

Add to the class docstring's Attributes section:

```
        head_indices: Which query head each entry came from, [B, H_kv, T]
            int32, aligned with `row_indices`. Required because a column is
            keyed by (batch, kv_head, block), which under grouped-query
            attention no longer determines the query head: entries from every
            query head sharing that kv head land in the same column. Held
            separately rather than packed into `row_indices` so the kernels'
            causal comparison `key_pos <= rows` stays a raw position compare
            with no integer modulo in the inner loop.
        num_query_heads: H_q. Equal to `col_nnz.shape[1]` for multi-head
            attention; a multiple of it under grouped-query attention.
```

- [ ] **Step 4: Make `build_csc` GQA-aware**

In `build_csc`, after `mask = moba_selection(q, k, cfg)`:

```python
    b, h_q, s, n = mask.shape
    h_kv = k.shape[1]
    group = head_group_size(h_q, h_kv)
```

Reshape the mask so query heads within a group become extra *rows* of the same
column — which is exactly the stacking the design describes:

```python
    # Stack the group's query heads onto the row axis. Reading
    # [B, H_kv, N, group * S] in memory order visits all of column 0 for every
    # head in the group, then column 1, and so on -- which is CSC ordering for
    # the merged column at no extra cost, exactly as the multi-head path got
    # ascending query order for free.
    stacked = mask.view(b, h_kv, group, s, n)          # [B,H_kv,G,S,N]
    bm = stacked.permute(0, 1, 4, 2, 3).reshape(b * h_kv, n * group * s)
```

Then `within` decodes to `(n_of, head_in_group, s_of)` instead of `(n_of,
s_of)`:

```python
    n_of = torch.div(within, group * s, rounding_mode="floor")
    rem = within - n_of * (group * s)
    g_of = torch.div(rem, s, rounding_mode="floor")
    s_of = rem - g_of * s
```

Size the buffers at `total = group * s * cfg.top_k`. Each entry records the
**absolute** query head, not its index within the group, so that the kernels can
address `Q` directly without knowing which group they are in:

```python
    kv_of = bh_of % h_kv
    head_indices = torch.zeros(b * h_kv, total, dtype=torch.int32,
                               device=device)
    head_indices[bh_of, pos] = (kv_of * group + g_of).to(torch.int32)
```

`col_nnz` becomes `stacked.sum(dim=(2, 3))` giving `[B, H_kv, N]`. `slot_grid`
is indexed with the decoded `(g_of, s_of, n_of)`.

Return the index with `head_indices=head_indices.view(b, h_kv, total)` and
`num_query_heads=h_q`.

- [ ] **Step 5: Make `csc_to_mask` GQA-aware**

```python
    b, h_kv, n = index.col_nnz.shape
    h_q = index.num_query_heads
    out = torch.zeros(b, h_q, seq_len, n, dtype=torch.bool,
                      device=index.col_nnz.device)
    for bi in range(b):
        for hi in range(h_kv):
            for blk in range(n):
                lo = int(index.col_offsets[bi, hi, blk])
                cnt = int(index.col_nnz[bi, hi, blk])
                rows = index.row_indices[bi, hi, lo:lo + cnt].long()
                heads = index.head_indices[bi, hi, lo:lo + cnt].long()
                out[bi, heads, rows, blk] = True
```

- [ ] **Step 6: Run the PyTorch index tests**

Run: `uv run pytest tests/test_index.py -v`
Expected: all pass.

- [ ] **Step 7: Commit the PyTorch builder**

```bash
git add src/moba/kernels/index.py tests/test_index.py
git commit -m "feat: CSC columns keyed by kv head, entries carry a query head"
```

- [ ] **Step 8: Make the fused builder GQA-aware**

In `src/moba/kernels/index_fused.py`, all three kernels take `H_Q` and
`Q_PER_KV` constexprs and map `pid_bh` (which enumerates `B * H_Q`) to a
column base.

`_count_columns` — add after `pid_bh = tl.program_id(1)`:

```python
    # `pid_bh` enumerates B * H_Q, because BlockIdx is per query head. Columns
    # are keyed by kv head, so several programs now target the same counter.
    # `atomic_add` already made that safe -- the contention simply rises by the
    # group size, and these are int32 adds on a `num_blocks`-sized array that
    # stays resident in L2.
    h_kv: tl.constexpr = H_Q // Q_PER_KV
    kv_bh = (pid_bh // H_Q) * h_kv + (pid_bh % H_Q) // Q_PER_KV
```

then replace `ColNnz + pid_bh * N_BLOCKS` with `ColNnz + kv_bh * N_BLOCKS`.

`_offsets_and_order` — unchanged in body; it is launched over `B * H_kv` now,
so only the grid changes at the call site.

`_scatter_pairs` — same `kv_bh` derivation, plus it writes the head. Replace the
store block with:

```python
        within = tl.atomic_add(Cursor + kv_bh * N_BLOCKS + safe, 1, mask=ok)
        start = tl.load(ColOffsets + kv_bh * (N_BLOCKS + 1) + safe, mask=ok,
                        other=0)
        dst = start + within

        tl.store(RowIdx + kv_bh * T + dst, offs.to(tl.int32), mask=ok)
        tl.store(HeadIdx + kv_bh * T + dst, (pid_bh % H_Q).to(tl.int32),
                 mask=ok)
        tl.store(SlotIdx + kv_bh * T + dst, j, mask=ok)
```

adding `HeadIdx` to the parameter list after `RowIdx`.

- [ ] **Step 9: Update `build_csc_from_indices`**

```python
def build_csc_from_indices(
    block_idx: torch.Tensor, num_blocks: int, cfg: MoBAConfig,
    num_kv_heads: int | None = None,
) -> CSCIndex:
```

Body changes:

```python
    b, h_q, s, top_k = block_idx.shape
    h_kv = h_q if num_kv_heads is None else num_kv_heads
    group = head_group_size(h_q, h_kv)
    device = block_idx.device
    total = group * s * cfg.top_k
```

Allocate `col_nnz`, `col_offsets`, `order`, `cursor` at `(b, h_kv, ...)` and
`row_indices`, `head_indices`, `slot_of_pair` at `(b, h_kv, total)`. Grids:

```python
    grid = (triton.cdiv(s, INDEX_BLOCK_M), b * h_q)
    ...
    _offsets_and_order[(b * h_kv,)](...)
```

Every launch gains `H_Q=h_q, Q_PER_KV=group`, except `_offsets_and_order`.
Return with `head_indices=head_indices` and `num_query_heads=h_q`.

- [ ] **Step 10: Update `build_csc_fused`**

```python
    kbar = block_representatives_triton(k, cfg)
    block_idx = gate_select(q, kbar, cfg)
    return build_csc_from_indices(block_idx, kbar.shape[2], cfg,
                                  num_kv_heads=k.shape[1])
```

- [ ] **Step 11: Run the fused index suite**

Run: `uv run pytest tests/test_index.py tests/test_index_fused.py -v`
Expected: all pass, new and old.

- [ ] **Step 12: Commit**

```bash
git add src/moba/kernels/index_fused.py tests/test_index_fused.py
git commit -m "feat: fused index builder merges columns across a head group"
```

---

## Task 4: Forward attention kernel

**Files:**
- Modify: `src/moba/kernels/moba_fwd.py:52` (`_moba_attn_fwd`), `:559`
  (`_validate`), `:585` (`empty_partials`), `:622` (`attend_partials`)
- Create: `tests/test_gqa.py`
- Test: `tests/test_gqa.py`, `tests/test_moba_kernel.py`

**Interfaces:**
- Consumes: `CSCIndex.head_indices`, `CSCIndex.num_query_heads` (Task 3).
- Produces: `moba_forward(q, k, v, cfg, index=None, causal=True,
  partials_dtype=None)` accepts `k`/`v` with `H_kv < H_q`, returning
  `[B, H_q, S, D]`.
- **Not produced, deliberately:** any change to `combine_partials` or
  `_moba_combine` (D-GQA-5).

- [ ] **Step 1: Write the failing test**

Create `tests/test_gqa.py`:

```python
"""Grouped-query attention end to end, forward and backward.

Every test here is anchored the same way: the GQA kernel must agree with the
*same code path* run on expanded key/value heads. That comparison isolates the
head-sharing logic from everything else -- arithmetic, routing and tuning are
identical on both sides, so a disagreement can only be an indexing bug.

See `docs/superpowers/plans/2026-08-23-gqa-support.md`.
"""

import pytest
import torch

from conftest import make_qkv
from moba.config import MoBAConfig
from moba.heads import expand_kv
from moba.kernels.index import csc_to_mask
from moba.kernels.index_fused import build_csc_fused
from moba.kernels.moba_fwd import moba_forward
from moba.ref.naive import moba_attention_naive
from moba.tolerance import TOLERANCE_FACTOR, TOLERANCE_FLOOR, max_abs_error

# (batch, query heads, kv heads, seq_len, head_dim). Small: 4 GiB of VRAM, and
# GQA lengthens every column, so a shape that fit under MHA may not here.
GQA_SHAPES = [
    (1, 4, 1, 512, 64),     # group 4, the multi-query extreme
    (1, 8, 2, 512, 64),     # group 4
    (1, 4, 4, 512, 64),     # group 1 -- must reproduce plain MHA exactly
    (1, 4, 2, 1024, 128),   # group 2, head_dim 128
]
GQA_GATES = [(64, 4), (128, 8)]


def _shapes(shape, dtype, device):
    b, h_q, h_kv, s, d = shape
    torch.manual_seed(0)
    gen = torch.Generator(device=device).manual_seed(0)
    q = torch.randn(b, h_q, s, d, generator=gen, device=device, dtype=dtype)
    k = torch.randn(b, h_kv, s, d, generator=gen, device=device, dtype=dtype)
    v = torch.randn(b, h_kv, s, d, generator=gen, device=device, dtype=dtype)
    return q, k, v


@pytest.mark.parametrize("shape", GQA_SHAPES)
@pytest.mark.parametrize("gate", GQA_GATES)
def test_forward_matches_the_oracle_under_gqa(device, shape, gate):
    """Routing held fixed, so this measures arithmetic and not gate flips."""
    block_size, top_k = gate
    cfg = MoBAConfig(block_size=block_size, top_k=top_k, dtype=torch.bfloat16)
    b, h_q, h_kv, s, d = shape

    q64, k64, v64 = _shapes(shape, torch.float64, device)
    qb, kb, vb = (t.to(torch.bfloat16) for t in (q64, k64, v64))

    index = build_csc_fused(qb, kb, cfg)
    selected = csc_to_mask(index, s)
    assert selected.shape == (b, h_q, s, cfg.num_blocks(s))

    truth = moba_attention_naive(q64, k64, v64, cfg, selected=selected)
    ref = moba_attention_naive(qb, kb, vb, cfg, selected=selected)
    ours = moba_forward(qb, kb, vb, cfg, index=index)

    cap = max(TOLERANCE_FACTOR * max_abs_error(ref, truth), TOLERANCE_FLOOR)
    err = max_abs_error(ours, truth)
    assert err <= cap, f"forward error {err:.3e} exceeds {cap:.3e}"


@pytest.mark.parametrize("gate", GQA_GATES)
def test_gqa_forward_equals_the_kernel_on_expanded_heads(device, gate):
    """The sharing must be free of arithmetic consequences.

    Expanding K/V and running the ordinary multi-head path gives the identical
    mathematical problem. Results differ only by summation order within a
    column, whose entries an atomic cursor permutes -- hence allclose rather
    than bitwise. `test_index_fused` pins the routing itself exactly.
    """
    block_size, top_k = gate
    cfg = MoBAConfig(block_size=block_size, top_k=top_k, dtype=torch.bfloat16)
    q, k, v = _shapes((1, 8, 2, 512, 64), torch.bfloat16, device)

    narrow = moba_forward(q, k, v, cfg)
    wide = moba_forward(q, expand_kv(k, 8), expand_kv(v, 8), cfg)
    assert torch.allclose(narrow, wide, rtol=2e-2, atol=1e-3), (
        f"max {max_abs_error(narrow, wide):.3e}"
    )


def test_indivisible_head_counts_are_rejected_at_the_boundary(device):
    cfg = MoBAConfig(block_size=64, top_k=4, dtype=torch.bfloat16)
    q = torch.randn(1, 8, 256, 64, dtype=torch.bfloat16, device=device)
    k = torch.randn(1, 3, 256, 64, dtype=torch.bfloat16, device=device)
    with pytest.raises(ValueError, match="divisible"):
        moba_forward(q, k, k, cfg)
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/test_gqa.py -v`
Expected: FAIL — `_validate` raises "shape mismatch" for every GQA shape.

- [ ] **Step 3: Loosen `_validate`**

Replace the shape check in `moba_fwd.py:559`:

```python
    if k.shape != v.shape:
        raise ValueError(f"key/value shape mismatch: {k.shape}, {v.shape}")
    if q.shape[0] != k.shape[0] or q.shape[2:] != k.shape[2:]:
        raise ValueError(f"shape mismatch: {q.shape}, {k.shape}, {v.shape}")
    # Raises if the head counts do not divide. Grouped-query attention is a
    # shape property, not a configuration one (D-GQA-1).
    head_group_size(q.shape[1], k.shape[1])
```

with `from moba.heads import head_group_size` added to the imports.

- [ ] **Step 4: Make the kernel head-aware**

Extend `_moba_attn_fwd`'s signature: add `HeadIdx` after `RowIdx`, and
`H_Q: tl.constexpr, H_KV: tl.constexpr` after `K_SLOTS`.

After `pid_bh = tl.program_id(1)`, insert:

```python
    # `pid_bh` enumerates B * H_KV: a program owns one *key* block, and keys
    # are what got shared. The query side is wider, so it is addressed by an
    # explicit (head, position) pair carried in the column (D-GQA-3).
    pid_b = pid_bh // H_KV
    q_bh_base = pid_b * H_Q * S * D
    q_part_base = pid_b * H_Q * S * K_SLOTS
```

Inside the loop, after loading `rows` and `slots`, add:

```python
        heads = tl.load(HeadIdx + pid_bh * T + lo + offs, mask=live, other=0)
        # One [GROUP] int32 gather -- 128 bytes against a 16 KiB K/V tile.
        # This is the whole cost of grouped-query support in the inner loop:
        # the tile shapes, register count and shared-memory footprint are
        # untouched. GQA makes the *list* longer, not the tile wider
        # (D-GQA-4).
        q_row = heads * S + rows
```

Change the `q` load to use `q_bh_base + q_row[:, None] * D`, and the three
partial stores to use `dst = q_row * K_SLOTS + slots` with bases
`PartM + q_part_base + dst`, `PartL + q_part_base + dst`, and
`PartAcc + q_part_base * D + dst[:, None] * D + offs_d[None, :]`.

**Leave the causal mask exactly as it is** — `key_pos[None, :] <= rows[:, None]`
still compares a key position to a query position, because `rows` is still a raw
position.

- [ ] **Step 5: Update `empty_partials` and `attend_partials`**

`empty_partials` already takes an explicit `shape`; callers pass `q.shape`, so
it needs no change — confirm this rather than assuming, and if a caller passes
`k.shape` anywhere, fix the caller.

In `attend_partials`, derive both head counts and fix the grid:

```python
    b, h_q, s, d = q.shape
    h_kv = k.shape[1]
```

Grid becomes `(index.num_blocks, b * h_kv)`. Add to the launch kwargs:

```python
                K_SLOTS=slots, H_Q=h_q, H_KV=h_kv, CAUSAL=causal,
```

and pass `index.head_indices` positionally after `index.row_indices`. Extend the
`_GROUP_CACHE` key with `h_q // h_kv` — a different group size changes nothing
about shared memory, but the key is cheap and a stale entry would be very hard
to diagnose.

- [ ] **Step 6: Run the GQA and regression suites**

Run: `uv run pytest tests/test_gqa.py tests/test_moba_kernel.py tests/test_kernel_invariants.py -v`
Expected: all pass. Every pre-existing test has `H_kv == H_q`, where
`q_row == rows` and `pid_b * H_Q == pid_bh`, so the arithmetic is unchanged.

- [ ] **Step 7: Commit**

```bash
git add src/moba/kernels/moba_fwd.py tests/test_gqa.py
git commit -m "feat: forward attention under grouped-query attention"
```

---

## Task 5: Backward kernel

**Files:**
- Modify: `src/moba/kernels/moba_bwd.py:125` (`_moba_bwd`), `:337`
  (`moba_backward`)
- Modify: `src/moba/kernels/autograd.py`
- Test: `tests/test_gqa.py`, `tests/test_backward.py`

**Interfaces:**
- Consumes: everything from Task 4.
- Produces: `moba_backward(...)` returns `dq` `[B, H_q, S, D]` and `dk`, `dv`
  `[B, H_kv, S, D]`; `moba_attention` differentiates end to end under GQA.
- **Not produced, deliberately:** any change to `_moba_dq_reduce`,
  `compute_delta` or `reduce_dq` (D-GQA-5) — all three are query-indexed and
  launch over `b * h_q`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_gqa.py`:

```python
def _grads(fn, q, k, v, g):
    q, k, v = (t.detach().clone().requires_grad_(True) for t in (q, k, v))
    out = fn(q, k, v)
    (out * g.to(out.dtype)).sum().backward()
    return q.grad, k.grad, v.grad


@pytest.mark.parametrize("gate", GQA_GATES)
def test_gradients_under_gqa_are_within_the_precision_budget(device, gate):
    """dk and dv must accumulate every sharing head's contribution.

    The failure mode to fear is a dk that is correct in shape and plausible in
    magnitude but sums only one query head's gradient -- an off-by-`group`
    that no shape check catches.
    """
    from moba.kernels.autograd import moba_attention

    block_size, top_k = gate
    cfg = MoBAConfig(block_size=block_size, top_k=top_k, dtype=torch.bfloat16)
    shape = (1, 8, 2, 512, 64)
    b, h_q, h_kv, s, d = shape

    q64, k64, v64 = _shapes(shape, torch.float64, device)
    qb, kb, vb = (t.to(torch.bfloat16) for t in (q64, k64, v64))
    gen = torch.Generator(device=device).manual_seed(7)
    g = torch.randn(b, h_q, s, d, generator=gen, device=device,
                    dtype=torch.float64) * 0.1

    index = build_csc_fused(qb, kb, cfg)
    selected = csc_to_mask(index, s)
    oracle = lambda a, x, y: moba_attention_naive(a, x, y, cfg,
                                                  selected=selected)
    kernel = lambda a, x, y: moba_attention(a, x, y, cfg, index=index)

    tq, tk, tv = _grads(oracle, q64, k64, v64, g)
    rq, rk, rv = _grads(oracle, qb, kb, vb, g)
    dq, dk, dv = _grads(kernel, qb, kb, vb, g)

    assert dk.shape == (b, h_kv, s, d)
    assert dv.shape == (b, h_kv, s, d)
    assert dq.shape == (b, h_q, s, d)

    for name, ours, ref, truth in [("dq", dq, rq, tq), ("dk", dk, rk, tk),
                                   ("dv", dv, rv, tv)]:
        cap = max(TOLERANCE_FACTOR * max_abs_error(ref, truth),
                  TOLERANCE_FLOOR)
        err = max_abs_error(ours, truth)
        assert err <= cap, f"{name} error {err:.3e} exceeds {cap:.3e}"
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/test_gqa.py -k gradients -v`
Expected: FAIL — a shape or illegal-memory-access error from `_moba_bwd`, which
still bases `Q`, `DO`, `Lse` and `Delta` on `pid_bh`.

- [ ] **Step 3: Make `_moba_bwd` head-aware**

Mirror Task 4 exactly. Add `HeadIdx` after `RowIdx` in the signature and
`H_Q: tl.constexpr, H_KV: tl.constexpr` after `DQ_SLOTS`. After the `pid_bh`
line, insert:

```python
    # As in the forward: keys are shared, queries are not. dK and dV stay owned
    # outright by this program -- their head axis is H_KV and the group's
    # contributions accumulate in the same registers, which is why grouped-query
    # attention needs no atomics here either.
    pid_b = pid_bh // H_KV
    q_base = pid_b * H_Q * S * D
    q_lin = pid_b * H_Q * S
```

Inside the loop, load `heads` alongside `rows` and `slots`, form
`q_row = heads * S + rows`, then:

- `q` and `do` load from `q_base + q_row[:, None] * D + offs_d[None, :]`
- `lse` and `delta` load from `Lse + q_lin + q_row` and `Delta + q_lin + q_row`
- the `dq_part` store uses
  `dst = q_row * DQ_SLOTS + slots * (BLOCK_SIZE // KEY_TILE) + sub` with base
  `DQPart + pid_b * H_Q * S * DQ_SLOTS * D`
- `dk` and `dv` stores keep `pid_bh` — they are key-indexed and already right

Leave the causal mask and `p = tl.where(live[:, None], p, 0.0)` untouched.

- [ ] **Step 4: Update `moba_backward`**

```python
    b, h_q, s, d = q.shape
    h_kv = k.shape[1]
    ...
    dq_part = torch.empty((b, h_q, s, slots * tiles, d), device=q.device,
                          dtype=q.dtype)
```

Grid becomes `(index.num_blocks * tiles, b * h_kv)`; add
`H_Q=h_q, H_KV=h_kv` to the launch kwargs and `index.head_indices`
positionally after `index.row_indices`. Extend the `_BWD_GROUP_CACHE` key with
`h_q // h_kv`, matching the forward.

`compute_delta`, `reduce_dq` and `_moba_dq_reduce` are unchanged: all three see
`[B, H_q, S, ...]` and launch over `b * h_q`, which is already correct.

- [ ] **Step 5: Update the autograd docstring**

In `src/moba/kernels/autograd.py`, extend `moba_attention`'s Args:

```
        k: Keys shaped [B, H_kv, S, D], where H_kv divides H. Fewer key/value
            heads than query heads is grouped-query attention; the kernel
            shares the K/V tile by owning it, never by materializing copies.
        v: Values, same shape as `k`.
```

No code change is needed — `_MoBAAttention.forward` already forwards shapes
through and `backward` returns gradients shaped like their inputs.

- [ ] **Step 6: Run the backward suites**

Run: `uv run pytest tests/test_gqa.py tests/test_backward.py -v`
Expected: all pass.

- [ ] **Step 7: Run the entire suite**

Run: `uv run pytest -q`
Expected: 343 pre-existing tests still pass, plus the new ones. **Any
pre-existing failure means a multi-head path was changed; do not proceed.**

- [ ] **Step 8: Commit**

```bash
git add src/moba/kernels/moba_bwd.py src/moba/kernels/autograd.py tests/test_gqa.py
git commit -m "feat: backward under grouped-query attention"
```

---

## Task 6: Benchmark, measure, document

The point of the change is arithmetic intensity. That claim is unverified until
measured, and this project's convention is that every non-obvious constant or
claim has a measurement document behind it.

**Files:**
- Modify: `tests/conftest.py` (`make_qkv` gains `num_kv_heads`)
- Modify: `bench/suite.py`
- Create: `docs/gqa-measurements.md`
- Modify: `CLAUDE.md` (status line)

**Interfaces:**
- Consumes: everything above.
- Produces: `make_qkv(shape, dtype, seed, device, num_kv_heads=None)`; a GQA row
  in the benchmark suite.

- [ ] **Step 1: Extend `make_qkv`**

In `tests/conftest.py`, add a keyword-only `num_kv_heads: int | None = None`.
When set, `k` and `v` are generated with that head count instead of `shape[1]`.
Default `None` keeps every existing call byte-identical — verify by running the
suite, not by inspection.

- [ ] **Step 2: Add a GQA benchmark row**

In `bench/suite.py`, add configurations at `H_q=8, H_kv=2` alongside the
existing `H=8` rows, at `seq_len` 2048 and 4096 (4 GiB ceiling). Report both
wall time and the derived FLOP/byte, since intensity is the quantity the design
argument is about.

- [ ] **Step 3: Measure**

Run the suite and record, for `H_q=8` with `H_kv` in `{8, 4, 2, 1}`:
- forward and backward wall time
- bytes moved by the attention kernel (K/V loads should fall by the group size;
  gathered Q and `part_acc` should not move at all)
- measured FLOP/byte against the card's 78 FLOP/byte ridge point
- speedup over dense SDPA with the same head configuration

- [ ] **Step 4: Write `docs/gqa-measurements.md`**

Structure it like `docs/backward-measurements.md`. It must state plainly:
- the predicted effect (K/V traffic ÷ group; arithmetic unchanged; registers and
  shared memory unchanged) and whether measurement bore it out
- **the honest ceiling.** Arithmetic intensity was 111.5 FLOP/byte, and `part_acc`
  was already the largest memory term at 63.4 MB of the attention kernel's 104 MB
  against 8.4 MB for all K/V loads. GQA divides the 8.4 MB, not the 63.4 MB. So
  the end-to-end gain on *this card* will be small, and the plan predicts that
  in advance rather than explaining it afterwards. The gain is largest where
  K/V traffic dominates — long context on hardware with a high ridge point,
  which is the rented-GPU run.
- whether any tuned constant needed re-deriving (the prediction, per D-GQA-4, is
  that none did)

- [ ] **Step 5: Update the status line**

One line in `CLAUDE.md`, kept current.

- [ ] **Step 6: Commit**

```bash
git add tests/conftest.py bench/suite.py docs/gqa-measurements.md CLAUDE.md
git commit -m "bench: measure grouped-query attention's effect on intensity"
```

---

## Risks

**A wrong head mapping does not crash.** `kv_bh` off by a factor produces a
well-formed index that attends to the wrong keys and returns plausible numbers —
the same failure class the CSC transpose has. This is why every task compares
against `expand_kv`-widened tensors rather than only checking shapes and
magnitudes.

**Column length grows by the group size.** `T` is sized `group * S * top_k`, and
load imbalance across columns (D8's concern) may change character now that a
column merges several heads' independent choices. If the benchmark shows worse
balance than multi-head at the same total work, that is a real finding for
`docs/load-balance-measurements.md`, not a bug to patch.

**Memory.** `dq_part` is `[B, H_q, S, slots*tiles, D]` and unchanged by GQA, but
GQA models have large `H_q`. The 4 GiB ceiling binds sooner than head count
alone suggests; keep test shapes at the sizes given.
