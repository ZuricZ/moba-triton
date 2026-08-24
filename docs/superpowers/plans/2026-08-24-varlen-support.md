# Variable-Length (Packed Sequence) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let the MoBA kernels attend over a batch of sequences of *different*
lengths packed end to end, with the existing padded rectangular path unchanged
and byte-identical.

**Architecture:** **Pack the data, pad the index.** The big tensors — `q`, `k`,
`v`, the partials — hold only real tokens, shaped `[1, H, T, D]`. The small
metadata — block counts, column offsets, dispatch order — keeps a real batch
axis, padded to `max_seqlen`'s worth of blocks per sequence. Every kernel
therefore learns its sequence from its own `program_id`, and a block's key range
is *computed* from `cu_seqlens` rather than looked up. Block ids stay
sequence-local, so the gate's streaming top-k, the CSC transpose, and every
tuning constant are untouched.

**Tech Stack:** Triton 3.7.1, PyTorch 2.13.0+cu126, Python 3.13, pytest.

**Spec:** No separate design doc; decisions were settled in conversation and are
recorded under "Design decisions" below. Background:
`docs/superpowers/specs/2026-08-18-moba-kernel-design.md` (D6 KV-block-major,
D8 longest-first dispatch, D10 fixed-slot partials),
`docs/superpowers/plans/2026-08-23-gqa-support.md` (the head-axis split this
builds on), `docs/learning/04-packed-sequences-and-where-the-work-goes.md` (the
plain-language version) and `docs/deferred-decisions.md` DD-1 (why head-major,
and when to reconsider).

> **This plan supersedes an earlier draft of the same name**, which carried a
> host-built `SeqLayout` table, a tile→sequence map, and globally numbered
> blocks. All three were removed after reading the references' public
> interfaces: FlashMoBA keeps its MoBA index at
> `(batch_size, num_heads, max_lg_col_num)` while packing `q` to
> `(total_q, nheads, headdim)` — packed data, padded index — and FlashAttention
> requires the caller to pass `max_seqlen`, which is enough to shape a
> per-sequence grid. The result is materially smaller. Git history holds the
> earlier version.

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
- **4 GiB VRAM.** Keep test totals small: `total_tokens` ≤ 2048, a handful of
  sequences.
- **Correctness policy is `moba.tolerance`**, never an absolute threshold: pass
  if our error is within `TOLERANCE_FACTOR` (2.0) of the same-dtype oracle's own
  deviation from an fp64 truth.
- **Measured hardware:** RTX 3050 Ti Laptop, sm_86, 20 SMs, 101376 B shared
  memory per block.
- **The padded path must not change.** 379 tests and every published benchmark
  number depend on it. A diff that alters padded behaviour is a bug in this
  work, not an improvement.
- **No new host synchronizations.** `index_fused.py` exists to remove them
  (~230 us each on this box). Nothing here may read a scalar off a device
  tensor; every host-side number comes from `.numel()`, `.shape`, or a caller
  argument.

---

## The interface

```python
moba_attention(q, k, v, cfg, *, cu_seqlens=None, max_seqlen=None, causal=True)
```

* `cu_seqlens=None` — the padded path, exactly as today, `[B, H, S, D]`.
* `cu_seqlens` given — the packed path. `q, k, v` are `[1, H, T, D]`;
  `cu_seqlens` is `[B+1]` int32 **on the device**; `max_seqlen` is a plain
  Python `int` and is **required**.

`max_seqlen` is an argument rather than a derived value for the reason
FlashAttention makes it one: deriving it means reading a scalar off
`cu_seqlens`, which is a host synchronization. `flash_moba_attn_varlen_func`
takes `max_seqlen_q` and `max_seqlen_k` as ints for exactly this. It also earns
its keep — see D-VL-4.

Everything else the kernels need is host-known without touching the device:
`B = cu_seqlens.numel() - 1`, `T = q.shape[2]`,
`NB = cdiv(max_seqlen, block_size)`.

---

## Design decisions

**D-VL-1 — Packed tensors keep four dimensions and pin `B = 1`.**
`q` is `[1, H_q, T, D]`, matching fla-org's NSA, which asserts this exactly
(`ops/naive.py:84`) and names the ordering `head_first=True` (`:44`).
*Rejected:* FlashAttention's `[T, H, D]`. Both are legitimate — NSA supports
both from one kernel, because the layout is invisible where the compute happens
— and the criterion is **consistency with our own API beats consistency with
someone else's**: 379 tests, every benchmark number and the `sdpa_dense`
baseline speak `[B, H, S, D]`. Recorded as provisional with a revisit condition
in `docs/deferred-decisions.md`, DD-1.

**D-VL-2 — One entry point, `cu_seqlens=None` meaning padded.**
FlashMoBA does it the single-entry way: `Varlen` is a C++ template parameter
and `cu_seqlens_q == nullptr` selects padded (`csrc/.../block_info.h:19-25`).
*Rejected:* a separate `moba_attention_varlen`; and making packed the only
internal form with padded callers packing first, which costs a copy of Q, K and
V — ~25 MB at S=8192, ≈300 us against a 2280 us forward, about 13%.

**D-VL-3 — Pack the data, pad the index.**
`q`, `k`, `v`, `part_*`, `lse`, `delta`, `dq_part` hold only real tokens. The
CSC metadata keeps a real batch axis: `col_nnz` is `[B, H_kv, NB]` where
`NB = cdiv(max_seqlen, block_size)`. Taken directly from FlashMoBA, whose varlen
entry point takes `moba_col_offsets: (batch_size, num_heads, max_lg_col_num)`
alongside a packed `q`.

The asymmetry is the point: the data is `O(T·H·D)` and worth packing; the
metadata is `O(B·H·NB)` and is not. Padding the small thing buys three things —
a batch axis the grid can carry, sequence-local block ids, and an
`_offsets_and_order` whose reduction stays bounded by *one sequence's* block
count rather than the whole batch's.

**D-VL-4 — Every kernel learns its sequence from its `program_id`.**
Grids gain the batch as a real dimension: the gate launches
`(cdiv(max_seqlen, BLOCK_M), B * H_q)`, attention launches `(NB, B * H_kv)`. A
program decodes `b` by division and reads `bos = cu_seqlens[b]`,
`eos = cu_seqlens[b+1]` — two scalar loads, once per program.

*Rejected:* a host-built tile→sequence map (NSA's `token_indices`,
`ops/parallel.py:64`). The map makes the grid exact; the per-sequence grid
over-launches for short sequences and lets those programs return immediately. On
a card measured as **grid-starved** — `docs/gqa-measurements.md` found a small
grid costing up to 2x — extra programs that return immediately are not the thing
to economise on, and the map is an array, a kernel argument and a construction
step we then do not need. This is what `max_seqlen` buys.

**D-VL-5 — Blocks are cut per sequence, and their ranges are computed, not stored.**
Cutting on the packed buffer would put two documents' tokens in one block, and a
query selecting it would attend across the boundary — well-formed, plausible,
silent. Each sequence gets `ceil(len/block_size)` blocks, the last possibly
short. A program owning sequence-local block `blk` computes:

```python
lo = bos + blk * BLOCK_SIZE
hi = tl.minimum(lo + BLOCK_SIZE, eos)
```

*Rejected:* `block_start` / `block_end` lookup arrays, which the earlier draft
used. With `b` known from the grid, the range is two arithmetic ops on values
already loaded — cheaper than a gather, and it removes the arrays entirely.

**D-VL-6 — Block ids stay sequence-local everywhere.**
The gate emits `0 .. n_seq_blocks-1`; columns are keyed by
`(b, kv_head, local_block)`. *Rejected:* global ids across the packed batch.
Local ids keep the gate's streaming top-k, its tie-break and its horizon
computation a **zero-line diff**, because they were always local; they keep the
CSC transpose kernels' bodies unmodified; and they keep `_offsets_and_order`'s
reduction the size it is today (see Risks).

**D-VL-7 — The causal mask is not modified, and the gate is what makes that safe.**
`qk += tl.where(key_pos <= rows, 0.0, -inf)` compares packed token indices.
Within a sequence, packed order *is* position order, so it is correct —
**provided a query is never offered a block from another sequence**. That
invariant is enforced entirely by the gate. This is the one place where an
untouched line's correctness is guaranteed by a distant one; comment it on both
sides and test it directly
(`test_no_selected_block_crosses_a_sequence_boundary`).

**D-VL-8 — `_moba_combine`, `_moba_dq_reduce` and `_moba_bwd_preprocess` are not modified.**
Verified by reading the code, not assumed: all three address memory as
`pid_bh * S * ...` over a grid of `B * H_q` (`moba_fwd.py:465`,
`moba_bwd.py:298`, `moba_bwd.py:116`). With `B = 1` and `S = T` those
expressions are already correct, and none has any notion of a sequence. As under
GQA: an implementer editing these has an upstream problem.

### Exactly which expressions branch

`VARLEN` is a `tl.constexpr`, so Triton compiles two specializations and there
is no runtime branch. Every line not listed here is shared.

| File:line | padded | packed |
|---|---|---|
| `moba_fwd.py:81` `q_base` | `pid_b * H_Q * S * D` | `0` |
| `moba_fwd.py:82` `q_part_base` | `pid_b * H_Q * S * K_SLOTS` | `0` |
| `moba_fwd.py:98` `kv_base` | `pid_bh * S * D` | `(pid_bh % H_KV) * T * D` |
| `moba_fwd.py` `key_pos`, `key_ok` | `blk*BLOCK_SIZE + arange`, `< S` | `bos + blk*BLOCK_SIZE + arange`, `< min(lo+BLOCK_SIZE, eos)` |
| `moba_bwd.py:159,160` `q_base`, `q_lin` | `pid_b * H_Q * S * (D or 1)` | `0` |
| `moba_bwd.py:173` `kv_base` | `pid_bh * S * D` | `(pid_bh % H_KV) * T * D` |
| `moba_bwd.py:256` `DQPart` base | `pid_b * H_Q * S * DQ_SLOTS * D` | `0` |
| `moba_bwd.py:262` `kv_out` | `pid_bh * S * D + …` | `(pid_bh % H_KV) * T * D + …` |
| `moba_bwd.py` `key_pos`, `key_ok` | as above, plus `sub * KEY_TILE` | as above, plus `sub * KEY_TILE` |
| `gate.py:122` Q load base | `pid_bh * S * D` | `(pid_bh % H_Q) * T * D` |
| `gate.py` `offs_m`, `live`, candidate count | `pid_m*BLOCK_M + arange`, `< S`, `N_BLOCKS` | `bos + local`, `< eos`, `cdiv(eos-bos, BLOCK_SIZE)` |
| `gate.py:352` K load base | `pid_bh * S * D` | `(pid_bh % H_KV) * T * D` |
| `gate.py` rep rows, occupancy | `blk*BLOCK_SIZE`, `< S` | `bos + blk*BLOCK_SIZE`, `< eos` |

Thirteen sites. Nothing else moves: not the matmuls, not the online softmax, not
the additive masks, not the streaming top-k, not the CSC transpose, not one
tuning constant.

### Shapes

| Tensor | padded | packed |
|---|---|---|
| `q`, `out`, `dq` | `[B, H_q, S, D]` | `[1, H_q, T, D]` |
| `k`, `v`, `dk`, `dv` | `[B, H_kv, S, D]` | `[1, H_kv, T, D]` |
| `lse`, `delta` | `[B, H_q, S]` | `[1, H_q, T]` |
| `part_m`, `part_l` | `[B, H_q, S, K]` | `[1, H_q, T, K]` |
| `part_acc` | `[B, H_q, S, K, D]` | `[1, H_q, T, K, D]` |
| `dq_part` | `[B, H_q, S, K·tiles, D]` | `[1, H_q, T, K·tiles, D]` |
| `kbar` | `[B, H_kv, N, D]` | `[B, H_kv, NB, D]` |
| `block_idx` (gate out) | `[B, H_q, S, top_k]` | `[1, H_q, T, top_k]` |
| `col_nnz`, `order` | `[B, H_kv, N]` | `[B, H_kv, NB]` |
| `col_offsets` | `[B, H_kv, N+1]` | `[B, H_kv, NB+1]` |
| `row_indices`, `head_indices`, `slot_of_pair` | `[B, H_kv, G·S·top_k]` | `[B, H_kv, G·max_seqlen·top_k]` |

`NB = cdiv(max_seqlen, block_size)`, `G = H_q // H_kv`. `kbar` and the CSC
arrays keep `B` — they are the padded index of D-VL-3.

---

## File Structure

**Create:**
- `src/moba/packing.py` — caller-facing converters between a padded batch plus
  mask and the packed form, in our head-major layout, plus shared validation of
  `cu_seqlens`.
- `tests/test_packing.py`
- `tests/test_varlen.py` — end-to-end packed correctness, forward and backward.
- `docs/varlen-measurements.md`

**Modify:**
- `src/moba/ref/naive.py` — oracle accepts `cu_seqlens`.
- `src/moba/kernels/gate.py` — `_block_representatives_fwd`, `_moba_gate_fwd`,
  both wrappers.
- `src/moba/kernels/index_fused.py` — the three transpose kernels learn `b` from
  the grid.
- `src/moba/kernels/index.py` — `CSCIndex` carries `cu_seqlens`/`max_seqlen`;
  `csc_to_mask` packed form.
- `src/moba/kernels/moba_fwd.py` — `_validate`, `_moba_attn_fwd`,
  `attend_partials`, `moba_forward`.
- `src/moba/kernels/moba_bwd.py` — `_moba_bwd`, `moba_backward`.
- `src/moba/kernels/autograd.py` — plumb `cu_seqlens`, `max_seqlen`.
- `src/moba/backends/sdpa_dense.py` — nested-tensor packed baseline.
- `bench/suite.py`

**Untouched by design:** `_moba_combine`, `_moba_dq_reduce`,
`_moba_bwd_preprocess`, `moba/config.py`, `moba/tolerance.py`,
`moba/device.py`, `moba/heads.py`.

---

## Task 1: Packing helpers and the oracle

Pure PyTorch, no kernels. Establishes the definition everything is checked
against.

**Files:**
- Create: `src/moba/packing.py`, `tests/test_packing.py`
- Modify: `src/moba/ref/naive.py`
- Test: `tests/test_packing.py`

**Interfaces:**
- Produces:
  - `validate_cu_seqlens(cu_seqlens, total_tokens, max_seqlen) -> int`
    returning the sequence count; raises on anything malformed.
  - `pack(x, attention_mask) -> (packed, cu_seqlens, max_seqlen)`, `x` is
    `[B, H, S, D]`, `packed` is `[1, H, T, D]`.
  - `unpack(packed, cu_seqlens, batch, seq_len) -> [B, H, S, D]`.
  - `from_nested(nt) -> (packed, cu_seqlens, max_seqlen)`.
  - `moba_attention_naive(..., cu_seqlens=None)`,
    `moba_selection(..., cu_seqlens=None)`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_packing.py`:

```python
"""Converting between a padded batch and the packed form, in our layout.

FlashAttention ships `unpad_input`/`pad_input` for this
(`references/flash-moba/flash_moba/bert_padding.py:98`), but they cannot be
lifted verbatim: they gather *rows*, which works because FA's token axis is
outermost. Ours is head-major, so one token's data is `H` separate runs of `D`
rather than a contiguous block, and packing is a strided gather. Same idea,
different index arithmetic. See `docs/deferred-decisions.md`, DD-1.
"""

import pytest
import torch

from moba.packing import from_nested, pack, unpack, validate_cu_seqlens


def test_pack_keeps_only_real_tokens(device):
    """Lengths 5, 3, 4 in a batch padded to 5 -- countable by hand."""
    x = torch.arange(3 * 2 * 5 * 4, dtype=torch.float32,
                     device=device).view(3, 2, 5, 4)
    mask = torch.tensor([[1, 1, 1, 1, 1],
                         [1, 1, 1, 0, 0],
                         [1, 1, 1, 1, 0]], device=device)

    packed, cu, max_seqlen = pack(x, mask)
    assert packed.shape == (1, 2, 12, 4)
    assert cu.tolist() == [0, 5, 8, 12]
    assert max_seqlen == 5

    for i, (lo, n) in enumerate([(0, 5), (5, 3), (8, 4)]):
        assert torch.equal(packed[0, :, lo:lo + n], x[i, :, :n]), f"seq {i}"


def test_unpack_inverts_pack_on_the_real_tokens(device):
    x = torch.randn(3, 2, 5, 4, device=device)
    mask = torch.tensor([[1, 1, 1, 1, 1],
                         [1, 1, 1, 0, 0],
                         [1, 1, 1, 1, 0]], device=device)
    packed, cu, _ = pack(x, mask)
    back = unpack(packed, cu, batch=3, seq_len=5)

    m = mask.bool()[:, None, :, None].expand_as(x)
    assert torch.equal(back[m], x[m]), "a real token changed"
    assert not bool(back[~m].any()), "padding must come back zeroed"


def test_from_nested_unwraps_a_jagged_tensor(device):
    """A jagged nested tensor is already packed plus cu_seqlens.

    `nt.values()` is the packed data and `nt.offsets()` *is* cu_seqlens; the
    only work is the transpose into our head-major layout.
    """
    nt = torch.nested.nested_tensor(
        [torch.randn(5, 2, 4), torch.randn(3, 2, 4), torch.randn(4, 2, 4)],
        layout=torch.jagged, device=device, dtype=torch.float32)
    packed, cu, max_seqlen = from_nested(nt)
    assert packed.shape == (1, 2, 12, 4)
    assert cu.tolist() == [0, 5, 8, 12]
    assert max_seqlen == 5


def test_validate_returns_the_sequence_count(device):
    cu = torch.tensor([0, 5, 8, 12], dtype=torch.int32, device=device)
    assert validate_cu_seqlens(cu, total_tokens=12, max_seqlen=5) == 3


@pytest.mark.parametrize("cu,total,mx,match", [
    ([0, 5, 8, 12], 11, 5, "total"),
    ([0, 5, 8, 12], 12, 4, "max_seqlen"),
    ([5, 8, 12], 12, 5, "start at 0"),
    ([0, 8, 5], 5, 8, "increasing"),
    ([0, 0, 5], 5, 5, "increasing"),
])
def test_malformed_metadata_is_rejected(device, cu, total, mx, match):
    t = torch.tensor(cu, dtype=torch.int32, device=device)
    with pytest.raises(ValueError, match=match):
        validate_cu_seqlens(t, total_tokens=total, max_seqlen=mx)


def test_validation_does_not_synchronize_with_the_host(device):
    """Checks run on device; reading a scalar back costs ~230 us.

    `index_fused` exists to remove synchronizations of this kind. Validation
    that reintroduced one on every forward call would be a silent ~10% tax that
    looked like a kernel regression.
    """
    cu = torch.tensor([0, 5, 8, 12], dtype=torch.int32, device=device)
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        validate_cu_seqlens(cu, total_tokens=12, max_seqlen=5)
    finally:
        torch.cuda.set_sync_debug_mode("default")
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/test_packing.py -q` (WSL prefix).
Expected: `ModuleNotFoundError: No module named 'moba.packing'`.

- [ ] **Step 3: Write `src/moba/packing.py`**

Constraints for the implementer:

* `validate_cu_seqlens` must not read a scalar off the device. `numel()` and
  `shape` are host metadata and free; value checks stay as device tensors
  (`torch._assert_async`) or compare against the host arguments. The last-entry
  check compares against `total_tokens`, which the caller supplied — never
  `int(cu[-1])`.
* `pack` **may** synchronize; it runs once at data-prep time, exactly as FA's
  `unpad_input` does with `.item()` on `max_seqlen_in_batch`. Say so in the
  docstring so the asymmetry is deliberate rather than accidental.
* `from_nested` transposes `nt.values()` `[T, H, D]` into `[1, H, T, D]` and
  must `.contiguous()`, which is a copy — document it and point at DD-1.

```python
def validate_cu_seqlens(
    cu_seqlens: torch.Tensor, total_tokens: int, max_seqlen: int
) -> int:
    """Checks packed metadata and returns the sequence count.

    Deliberately reads nothing off the device. `cu_seqlens` lives in GPU memory
    and pulling a scalar from it costs a host synchronization -- ~230 us on this
    box, measured in `index_fused`. Every host-side number here comes from
    `numel()` or from an argument the caller already had.

    Args:
        cu_seqlens: Cumulative token counts, [num_sequences + 1] int32, on the
            same device as the data.
        total_tokens: `q.shape[2]`. The last entry of `cu_seqlens` must equal
            this, and the caller knows it without a transfer.
        max_seqlen: Longest sequence in the batch, as a plain int. Required for
            the reason FlashAttention requires it: deriving it would mean
            reading `cu_seqlens.max()` back to the host.

    Returns:
        Number of sequences.

    Raises:
        ValueError: If the shape, dtype, start, monotonicity, total or
            max_seqlen is wrong.
    """
```

- [ ] **Step 4: Run to verify they pass**

Expected: 10 passed.

- [ ] **Step 5: Write the failing oracle test**

Append to `tests/test_packing.py`:

```python
def test_packed_oracle_equals_running_each_sequence_alone(device):
    """The definition of varlen, checked bitwise.

    Slicing the packed buffer gives the same inputs in the same order, so
    anything but equality is a boundary bug rather than rounding.
    """
    from moba.config import MoBAConfig
    from moba.ref.naive import moba_attention_naive

    cfg = MoBAConfig(block_size=4, top_k=2, dtype=torch.float64)
    cu = torch.tensor([0, 20, 32, 48], dtype=torch.int32, device=device)
    gen = torch.Generator(device=device).manual_seed(0)
    mk = lambda: torch.randn(1, 2, 48, 16, generator=gen, device=device,
                             dtype=torch.float64)
    q, k, v = mk(), mk(), mk()

    packed = moba_attention_naive(q, k, v, cfg, cu_seqlens=cu)
    lo = 0
    for n in [20, 12, 16]:
        piece = moba_attention_naive(q[:, :, lo:lo + n], k[:, :, lo:lo + n],
                                     v[:, :, lo:lo + n], cfg)
        assert torch.equal(packed[:, :, lo:lo + n], piece), f"at {lo}"
        lo += n
```

- [ ] **Step 6: Run to verify it fails, then add `cu_seqlens` to the oracle**

`moba_attention_naive` and `moba_selection` gain
`cu_seqlens: torch.Tensor | None = None`. When set, require `B == 1` and loop
over sequences in Python, slicing and delegating to the existing per-sequence
code:

```python
    # Deliberately a Python loop. This is the oracle: one line per line of the
    # algorithm, and "a packed batch means each sequence attends alone" is the
    # definition being encoded. fla-org's NSA does the same in its reference
    # (`ops/naive.py:111`), and FSA's *production* path still does
    # (`FSA_topk_sparse_attention.py:583`, with a TODO to fuse it) -- fusing is
    # the kernel's job, not the definition's.
```

`moba_selection` returns `[1, H, T, NB]` with **sequence-local** block ids,
`NB = cdiv(max length, block_size)`. This is the convention `csc_to_mask` must
reproduce in Task 4; write it down here, once.

- [ ] **Step 7: Run the oracle-facing suite**

Run: `uv run pytest tests/test_packing.py tests/test_semantics.py tests/test_smoke.py -q`
Expected: all pass; existing tests take the unchanged branch.

- [ ] **Step 8: Commit**

```bash
git add src/moba/packing.py tests/test_packing.py src/moba/ref/naive.py docs/superpowers/plans/2026-08-24-varlen-support.md
git commit -m "feat: packing helpers and a packed-aware oracle"
```

---

## Task 2: Block representatives over a packed batch

Smallest kernel change; isolates "blocks are cut per sequence" before anything
depends on it.

**Files:**
- Modify: `src/moba/kernels/gate.py` — `_block_representatives_fwd`,
  `block_representatives_triton`
- Test: `tests/test_gate.py`

**Interfaces:**
- Consumes: `validate_cu_seqlens` (Task 1).
- Produces: `block_representatives_triton(k, cfg, cu_seqlens=None,
  max_seqlen=None)` returning `[B, H_kv, NB, D]` when packed.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_gate.py`:

```python
def test_block_representatives_are_pooled_per_sequence(device):
    """A block averages its own sequence's keys and stops at the boundary.

    The bug this catches is a block straddling two sequences: its mean would
    mix documents, every downstream stage would accept it, and the numbers stay
    finite. Compared against per-sequence pooling of the same keys.
    """
    cfg = MoBAConfig(block_size=4, top_k=2, dtype=torch.bfloat16)
    cu = torch.tensor([0, 20, 32, 48], dtype=torch.int32, device=device)
    gen = torch.Generator(device=device).manual_seed(0)
    k = torch.randn(1, 2, 48, 16, generator=gen, device=device,
                    dtype=torch.bfloat16)

    got = block_representatives_triton(k, cfg, cu_seqlens=cu, max_seqlen=20)
    assert got.shape == (3, 2, 5, 16)      # B=3, H_kv=2, NB=cdiv(20,4)=5

    lo = 0
    for i, n in enumerate([20, 12, 16]):
        want = block_representatives_triton(k[:, :, lo:lo + n], cfg)
        nb = want.shape[2]
        assert torch.equal(got[i:i + 1, :, :nb], want), f"sequence {i}"
        lo += n
```

- [ ] **Step 2: Run to verify it fails**

Expected: `TypeError` on the `cu_seqlens` keyword.

- [ ] **Step 3: Patch the kernel**

Add `CuSeqlens` pointer, `T`, `H_KV`, `N_BLOCKS` (runtime), `VARLEN:
tl.constexpr`. Replace the row computation:

```python
    if VARLEN:
        # `b` comes from the grid (D-VL-4), so a block's extent is two
        # arithmetic ops on values already loaded -- no lookup array (D-VL-5).
        b = pid_bh // H_KV
        h = pid_bh % H_KV
        bos = tl.load(CuSeqlens + b)
        eos = tl.load(CuSeqlens + b + 1)
        if blk * BLOCK_SIZE >= eos - bos:
            return              # a padded index slot; this block does not exist
        lo = bos + blk * BLOCK_SIZE
        hi = tl.minimum(lo + BLOCK_SIZE, eos)
        kbase = h * T * D                       # [1, H_kv, T, D]
    else:
        lo = blk * BLOCK_SIZE
        hi = tl.minimum(lo + BLOCK_SIZE, S)
        kbase = pid_bh * S * D

    rows = lo + tl.arange(0, BLOCK_POW2)
    live = (rows < hi) & (tl.arange(0, BLOCK_POW2) < BLOCK_SIZE)
    occupancy = hi - lo
```

`N_BLOCKS` becomes a runtime argument (`NB` packed,
`cdiv(S, BLOCK_SIZE)` padded) so the store's row stride is one expression in
both modes.

- [ ] **Step 4: Update the wrapper**

```python
def block_representatives_triton(
    k: torch.Tensor, cfg: MoBAConfig,
    cu_seqlens: torch.Tensor | None = None, max_seqlen: int | None = None,
) -> torch.Tensor:
```

Padded: grid and shapes exactly as today. Packed: require `k.shape[0] == 1`,
call `validate_cu_seqlens`, set `nb = cdiv(max_seqlen, cfg.block_size)`, grid
`(nb, b * h_kv)`, output `[B, H_kv, nb, D]`.

- [ ] **Step 5: Run**

Run: `uv run pytest tests/test_gate.py -q` — 39 passed.

- [ ] **Step 6: Commit**

```bash
git add src/moba/kernels/gate.py tests/test_gate.py
git commit -m "feat: block representatives pooled per sequence"
```

---

## Task 3: The gate over a packed batch

The correctness linchpin; D-VL-7 delegates the causal mask's safety here.

**Files:**
- Modify: `src/moba/kernels/gate.py` — `_moba_gate_fwd`, `gate_select`
- Create: `tests/test_varlen.py`
- Test: `tests/test_varlen.py`

**Interfaces:**
- Produces: `gate_select(q, kbar, cfg, cu_seqlens=None, max_seqlen=None)`
  returning `[1, H_q, T, top_k]` int32 of **sequence-local** block ids,
  -1 padded.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_varlen.py`:

```python
"""Packed (variable-length) batches, end to end.

Anchored throughout against the *same code path* run on each sequence alone.
Arithmetic, routing and tuning are identical on both sides, so a disagreement
can only be a boundary or indexing bug.

The failure to fear does not crash. A block straddling a boundary, or a gate
offering a query another document's block, produces finite, plausible numbers
of exactly the right shape.

See `docs/superpowers/plans/2026-08-24-varlen-support.md` and
`docs/learning/04-packed-sequences-and-where-the-work-goes.md`.
"""

import pytest
import torch

from moba.config import MoBAConfig
from moba.kernels.gate import block_representatives_triton, gate_select

LENS = [20, 12, 16, 5]
CU = [0, 20, 32, 48, 53]
MAXLEN = 20
TOTAL = 53


def _packed(device, dtype, heads=2, d=16, seed=0):
    gen = torch.Generator(device=device).manual_seed(seed)
    mk = lambda: torch.randn(1, heads, TOTAL, d, generator=gen,
                             device=device, dtype=dtype)
    return mk(), mk(), mk()


def _cu(device):
    return torch.tensor(CU, dtype=torch.int32, device=device)


def test_no_selected_block_crosses_a_sequence_boundary(device):
    """The invariant the untouched causal mask depends on (D-VL-7).

    Block ids are sequence-local, so "crossing" appears as an id at or beyond
    that sequence's own block count -- a block that does not exist for the
    query but would address a real, padded index slot.
    """
    cfg = MoBAConfig(block_size=4, top_k=3, dtype=torch.bfloat16)
    cu = _cu(device)
    q, k, _ = _packed(device, torch.bfloat16)

    kbar = block_representatives_triton(k, cfg, cu_seqlens=cu,
                                        max_seqlen=MAXLEN)
    idx = gate_select(q, kbar, cfg, cu_seqlens=cu, max_seqlen=MAXLEN)
    assert idx.shape == (1, 2, TOTAL, cfg.top_k)

    lo = 0
    for n in LENS:
        n_blocks = -(-n // cfg.block_size)
        chosen = idx[0, :, lo:lo + n]
        real = chosen[chosen >= 0]
        assert bool((real < n_blocks).all()), (
            f"sequence at {lo} (length {n}, {n_blocks} blocks) selected id "
            f"{int(real.max())}"
        )
        lo += n


def test_gate_matches_running_each_sequence_alone(device):
    """Selections must be identical -- routing is a ranking, not an estimate."""
    cfg = MoBAConfig(block_size=4, top_k=3, dtype=torch.bfloat16)
    cu = _cu(device)
    q, k, _ = _packed(device, torch.bfloat16)

    packed = gate_select(
        q, block_representatives_triton(k, cfg, cu_seqlens=cu,
                                        max_seqlen=MAXLEN),
        cfg, cu_seqlens=cu, max_seqlen=MAXLEN)

    lo = 0
    for n in LENS:
        piece = gate_select(
            q[:, :, lo:lo + n],
            block_representatives_triton(k[:, :, lo:lo + n], cfg), cfg)
        # Both are sequence-local, so no offset is needed (D-VL-6).
        assert torch.equal(packed[:, :, lo:lo + n], piece), f"at {lo}"
        lo += n


def test_short_sequences_get_programs_that_return_immediately(device):
    """The per-sequence grid over-launches; those programs must be harmless.

    Sequence 3 holds 5 tokens against a max of 20, so most of its query tiles
    have nothing to do. They must write nothing and leave no id in range.
    """
    cfg = MoBAConfig(block_size=4, top_k=3, dtype=torch.bfloat16)
    cu = _cu(device)
    q, k, _ = _packed(device, torch.bfloat16, seed=2)
    idx = gate_select(
        q, block_representatives_triton(k, cfg, cu_seqlens=cu,
                                        max_seqlen=MAXLEN),
        cfg, cu_seqlens=cu, max_seqlen=MAXLEN)
    nb = -(-MAXLEN // cfg.block_size)
    assert bool(((idx >= -1) & (idx < nb)).all())
```

- [ ] **Step 2: Run to verify they fail**

Expected: `TypeError` on the `cu_seqlens` keyword.

- [ ] **Step 3: Patch the kernel**

Add `CuSeqlens`, `T`, `H_KV`, `VARLEN`. Replace the tile setup:

```python
    if VARLEN:
        # `b` from the grid (D-VL-4). Every value here is a *scalar*, which is
        # what keeps the streaming top-k below a zero-line diff: a tile
        # spanning two sequences would make the candidate range per-lane, and a
        # Triton program has one loop bound for all its lanes. The per-sequence
        # grid makes straddling impossible by construction.
        b = pid_bh // H_Q
        hq = pid_bh % H_Q
        bos = tl.load(CuSeqlens + b)
        eos = tl.load(CuSeqlens + b + 1)
        seqlen = eos - bos
        if pid_m * BLOCK_M >= seqlen:
            return
        local = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_m = bos + local          # packed token index: Q and BlockIdx
        live = local < seqlen
        n_cand = tl.cdiv(seqlen, BLOCK_SIZE)
        qbase = hq * T * D
        kbar_base = (b * H_KV + hq // Q_PER_KV) * N_BLOCKS * D
    else:
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        local = offs_m
        live = offs_m < S
        n_cand = N_BLOCKS
        qbase = pid_bh * S * D
        kbar_base = kv_bh * N_BLOCKS * D

    own_block = local // BLOCK_SIZE    # sequence-local in both modes
```

Then substitute in the existing body: `N_BLOCKS` becomes `n_cand` wherever it
bounds the **candidate range** (`in_range`, the horizon's `tl.minimum`) and
stays `N_BLOCKS` where it is the **stride of `Kbar`** and the sentinel value in
the tie-break. **Conflating those is the likeliest bug in this task.** The merge
loop — tie-break, eviction, +inf retirement, seeding — is unchanged, and emitted
ids are already sequence-local so nothing is added to them.

- [ ] **Step 4: Update `gate_select`**

Padded: unchanged. Packed: require `q.shape[0] == 1`, call
`validate_cu_seqlens`, grid `(cdiv(max_seqlen, block_m), b * h_q)`, output
`[1, H_q, T, top_k]`.

- [ ] **Step 5: Run**

Run: `uv run pytest tests/test_gate.py tests/test_varlen.py -q`

- [ ] **Step 6: Commit**

```bash
git add src/moba/kernels/gate.py tests/test_varlen.py
git commit -m "feat: gate ranks within a sequence over a packed batch"
```

---

## Task 4: The CSC index over a packed batch

**Files:**
- Modify: `src/moba/kernels/index_fused.py`, `src/moba/kernels/index.py`
- Test: `tests/test_varlen.py`

**Interfaces:**
- Produces: `build_csc_fused(q, k, cfg, cu_seqlens=None, max_seqlen=None)`;
  `CSCIndex` gains `cu_seqlens: torch.Tensor | None` and
  `max_seqlen: int | None`; `csc_to_mask` returns `[1, H_q, T, NB]` when packed.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_varlen.py`:

```python
def test_columns_keep_a_batch_axis_and_hold_only_their_sequence(device):
    """Pack the data, pad the index (D-VL-3).

    Block ids are sequence-local, so a column is `(b, kv_head, local_block)`
    and the CSC arrays keep a real batch axis -- exactly FlashMoBA's
    `moba_col_offsets: (batch_size, num_heads, max_lg_col_num)` alongside a
    packed `q`.
    """
    from moba.kernels.index_fused import build_csc_fused

    cfg = MoBAConfig(block_size=4, top_k=3, dtype=torch.bfloat16)
    cu = _cu(device)
    q, k, _ = _packed(device, torch.bfloat16)

    index = build_csc_fused(q, k, cfg, cu_seqlens=cu, max_seqlen=MAXLEN)
    assert index.col_nnz.shape == (4, 2, 5)     # B, H_kv, NB
    assert index.num_blocks == 5

    for b, (lo, n) in enumerate(zip(CU[:-1], LENS)):
        for kv in range(2):
            for blk in range(index.num_blocks):
                cnt = int(index.col_nnz[b, kv, blk])
                if cnt == 0:
                    continue
                off = int(index.col_offsets[b, kv, blk])
                rows = index.row_indices[b, kv, off:off + cnt]
                assert bool(((rows >= lo) & (rows < lo + n)).all()), (
                    f"column ({b}, {kv}, {blk}) holds a foreign query")


def test_index_round_trips_to_the_gate_under_packing(device):
    from moba.kernels.index import csc_to_mask
    from moba.kernels.index_fused import build_csc_fused

    cfg = MoBAConfig(block_size=4, top_k=3, dtype=torch.bfloat16)
    cu = _cu(device)
    q, k, _ = _packed(device, torch.bfloat16)
    index = build_csc_fused(q, k, cfg, cu_seqlens=cu, max_seqlen=MAXLEN)
    got = csc_to_mask(index, TOTAL)
    assert got.shape == (1, 2, TOTAL, 5)
    assert int(got.sum()) == int(index.col_nnz.sum())
```

- [ ] **Step 2: Run to verify it fails**

- [ ] **Step 3: Give the transpose kernels a batch from the grid**

`_count_columns` and `_scatter_pairs` gain `CuSeqlens`, `T`, `H_KV`, `VARLEN`.
In packed mode they launch `(cdiv(max_seqlen, BLOCK_M), B * H_q)`; each derives
`b` and `hq`, reads `bos`/`eos`, early-returns past the sequence, and addresses
`BlockIdx` at the packed token `bos + local`. The column base becomes
`(b * H_KV + hq // Q_PER_KV) * N_BLOCKS`, and `RowIdx`/`HeadIdx`/`SlotIdx` are
based at `(b * H_KV + kv) * T_col`.

`_offsets_and_order` launches `(B * H_kv,)` and its **body is unchanged**. Note
that `N_BLOCKS` here is `NB` — one sequence's block count, not the batch's.
That is what keeps its `static_range(N_BLOCKS)` reduction the size it is in the
padded path; global block numbering would have multiplied it by the batch
(D-VL-6, and Risks).

`row_indices` stores the **packed token index**, so the attention kernel can
address `Q` directly and the causal comparison stays a raw position compare.

- [ ] **Step 4: `csc_to_mask` packed form and `CSCIndex` fields**

Returns `[1, H_q, T, NB]` with sequence-local ids — the convention Task 1
established for `moba_selection`. `CSCIndex` carries `cu_seqlens` and
`max_seqlen` so the backward needs no extra argument.

- [ ] **Step 5: Run**

Run: `uv run pytest tests/test_index.py tests/test_index_fused.py tests/test_varlen.py -q`

- [ ] **Step 6: Commit**

```bash
git add src/moba/kernels/index.py src/moba/kernels/index_fused.py tests/test_varlen.py
git commit -m "feat: CSC index over a packed batch, padded per sequence"
```

---

## Task 5: Forward attention over a packed batch

**Files:**
- Modify: `src/moba/kernels/moba_fwd.py`
- Test: `tests/test_varlen.py`

**Interfaces:**
- Produces: `moba_forward(q, k, v, cfg, index=None, causal=True,
  partials_dtype=None, cu_seqlens=None, max_seqlen=None)`.
- **Not produced, deliberately:** any change to `_moba_combine` (D-VL-8).

- [ ] **Step 1: Write the failing tests**

```python
@pytest.mark.parametrize("gate", [(4, 3), (8, 4)])
def test_forward_matches_running_each_sequence_alone(device, gate):
    """Bitwise per sequence. Slicing gives identical inputs in identical order,
    and the kernel addresses output per (query, slot) rather than by position,
    so anything but equality is a boundary bug."""
    from moba.kernels.moba_fwd import moba_forward

    block_size, top_k = gate
    cfg = MoBAConfig(block_size=block_size, top_k=top_k, dtype=torch.bfloat16)
    cu = _cu(device)
    q, k, v = _packed(device, torch.bfloat16)

    packed = moba_forward(q, k, v, cfg, cu_seqlens=cu, max_seqlen=MAXLEN)
    lo = 0
    for n in LENS:
        piece = moba_forward(q[:, :, lo:lo + n], k[:, :, lo:lo + n],
                             v[:, :, lo:lo + n], cfg)
        assert torch.equal(packed[:, :, lo:lo + n], piece), f"at {lo}"
        lo += n


def test_packed_forward_matches_the_oracle(device):
    """Routing held fixed, so this measures arithmetic and not gate flips."""
    from moba.kernels.index import csc_to_mask
    from moba.kernels.index_fused import build_csc_fused
    from moba.kernels.moba_fwd import moba_forward
    from moba.ref.naive import moba_attention_naive
    from moba.tolerance import (
        TOLERANCE_FACTOR, TOLERANCE_FLOOR, max_abs_error)

    cfg = MoBAConfig(block_size=4, top_k=3, dtype=torch.bfloat16)
    cu = _cu(device)
    q64, k64, v64 = _packed(device, torch.float64)
    qb, kb, vb = (t.to(torch.bfloat16) for t in (q64, k64, v64))

    index = build_csc_fused(qb, kb, cfg, cu_seqlens=cu, max_seqlen=MAXLEN)
    selected = csc_to_mask(index, TOTAL)

    truth = moba_attention_naive(q64, k64, v64, cfg, selected=selected,
                                 cu_seqlens=cu)
    ref = moba_attention_naive(qb, kb, vb, cfg, selected=selected,
                               cu_seqlens=cu)
    ours = moba_forward(qb, kb, vb, cfg, index=index)

    cap = max(TOLERANCE_FACTOR * max_abs_error(ref, truth), TOLERANCE_FLOOR)
    assert max_abs_error(ours, truth) <= cap


def test_packed_input_with_batch_above_one_is_rejected(device):
    from moba.kernels.moba_fwd import moba_forward
    cfg = MoBAConfig(block_size=4, top_k=2, dtype=torch.bfloat16)
    cu = torch.tensor([0, 8, 16], dtype=torch.int32, device=device)
    q = torch.randn(2, 2, 16, 16, dtype=torch.bfloat16, device=device)
    with pytest.raises(ValueError, match="batch 1"):
        moba_forward(q, q, q, cfg, cu_seqlens=cu, max_seqlen=8)


def test_max_seqlen_is_required_when_packing(device):
    """Not derivable without a host synchronization, so it is an argument."""
    from moba.kernels.moba_fwd import moba_forward
    cfg = MoBAConfig(block_size=4, top_k=2, dtype=torch.bfloat16)
    cu = torch.tensor([0, 8, 16], dtype=torch.int32, device=device)
    q = torch.randn(1, 2, 16, 16, dtype=torch.bfloat16, device=device)
    with pytest.raises(ValueError, match="max_seqlen"):
        moba_forward(q, q, q, cfg, cu_seqlens=cu)
```

- [ ] **Step 2: Run to verify they fail**

- [ ] **Step 3: Patch `_moba_attn_fwd`**

Four branched sites, per the table above. Add `CuSeqlens`, `T`, `VARLEN`:

```python
    if VARLEN:
        b = pid_bh // H_KV
        bos = tl.load(CuSeqlens + b)
        eos = tl.load(CuSeqlens + b + 1)
        q_base = 0                      # the tensor's batch axis is 1
        q_part_base = 0
        kv_base = (pid_bh % H_KV) * T * D
        lo_k = bos + blk * BLOCK_SIZE
        hi_k = tl.minimum(lo_k + BLOCK_SIZE, eos)
        q_span = T
    else:
        pid_b = pid_bh // H_KV
        q_base = pid_b * H_Q * S * D
        q_part_base = pid_b * H_Q * S * K_SLOTS
        kv_base = pid_bh * S * D
        lo_k = blk * BLOCK_SIZE
        hi_k = S
        q_span = S

    key_pos = lo_k + tl.arange(0, BLOCK_SIZE)
    key_ok = key_pos < hi_k
```

and `q_row = heads * q_span + rows`. **The causal mask is not touched** —
comment it with a pointer to D-VL-7 and to
`test_no_selected_block_crosses_a_sequence_boundary`.

- [ ] **Step 4: Validation and plumbing**

`_validate` gains `cu_seqlens`/`max_seqlen` and calls `validate_cu_seqlens`;
missing `max_seqlen` with a given `cu_seqlens` raises. `attend_partials` and
`moba_forward` accept both and take them from `index` when an index is passed.
Extend `_GROUP_CACHE`'s key with `bool(varlen)`.

- [ ] **Step 5: Run the full suite**

Run: `uv run pytest -q`
Expected: 379 existing pass unchanged plus the new ones. **A pre-existing
failure means the padded path moved — stop.**

- [ ] **Step 6: Commit**

```bash
git add src/moba/kernels/moba_fwd.py tests/test_varlen.py
git commit -m "feat: forward attention over a packed batch"
```

---

## Task 6: Backward over a packed batch

Mirrors Task 5; five branched sites instead of four.

**Files:**
- Modify: `src/moba/kernels/moba_bwd.py`, `src/moba/kernels/autograd.py`
- Test: `tests/test_varlen.py`

**Interfaces:**
- Produces: `moba_attention(q, k, v, cfg, index=None, causal=True,
  cu_seqlens=None, max_seqlen=None)`, differentiable over a packed batch.
- **Not produced, deliberately:** changes to `_moba_dq_reduce`,
  `_moba_bwd_preprocess`, `compute_delta`, `reduce_dq` (D-VL-8).

- [ ] **Step 1: Write the failing test**

```python
def _grads(fn, q, k, v, g):
    q, k, v = (t.detach().clone().requires_grad_(True) for t in (q, k, v))
    (fn(q, k, v) * g).sum().backward()
    return q.grad, k.grad, v.grad


def test_gradients_match_running_each_sequence_alone(device):
    """Gradient must not leak across a boundary.

    dK and dV are reductions down a column. If a column ever held a query from
    another sequence, that query's gradient would be added to this document's
    keys -- finite, plausible, and wrong.
    """
    from moba.kernels.autograd import moba_attention

    cfg = MoBAConfig(block_size=4, top_k=3, dtype=torch.bfloat16)
    cu = _cu(device)
    q, k, v = _packed(device, torch.bfloat16, seed=4)
    g = torch.randn_like(q) * 0.1

    pq, pk, pv = _grads(
        lambda a, x, y: moba_attention(a, x, y, cfg, cu_seqlens=cu,
                                       max_seqlen=MAXLEN), q, k, v, g)
    lo = 0
    for n in LENS:
        sq, sk, sv = _grads(
            lambda a, x, y: moba_attention(a, x, y, cfg),
            q[:, :, lo:lo + n], k[:, :, lo:lo + n], v[:, :, lo:lo + n],
            g[:, :, lo:lo + n])
        for name, a, b in [("dq", pq, sq), ("dk", pk, sk), ("dv", pv, sv)]:
            assert torch.allclose(a[:, :, lo:lo + n].float(), b.float(),
                                  rtol=2e-2, atol=1e-3), f"{name} at {lo}"
        lo += n
```

- [ ] **Step 2: Run to verify it fails**

- [ ] **Step 3: Patch `_moba_bwd`**

Same treatment: `q_base`, `q_lin`, `kv_base`, the `DQPart` base and `kv_out`,
plus the key range keeping its sub-tile offset:

```python
    key_pos = lo_k + sub * KEY_TILE + tl.arange(0, KEY_TILE)
    key_ok = key_pos < hi_k
```

where `hi_k` is `tl.minimum(lo_k + BLOCK_SIZE, eos)` packed and `S` padded.

- [ ] **Step 4: Plumb**

`moba_backward` takes `cu_seqlens`/`max_seqlen` off `index`; `autograd.py`
plumbs them into `forward`. Extend `_BWD_GROUP_CACHE`'s key with
`bool(varlen)`.

- [ ] **Step 5: Run the full suite**

- [ ] **Step 6: Commit**

```bash
git add src/moba/kernels/moba_bwd.py src/moba/kernels/autograd.py tests/test_varlen.py
git commit -m "feat: backward over a packed batch"
```

---

## Task 7: Baseline, measurement, documentation

**Files:**
- Modify: `src/moba/backends/sdpa_dense.py`, `bench/suite.py`, `CLAUDE.md`
- Create: `bench/varlen_sweep.py`, `docs/varlen-measurements.md`

- [ ] **Step 1: Give the dense baseline a packed path**

Verified on this box: `torch.nested.nested_tensor(..., layout=torch.jagged)`
passed to `scaled_dot_product_attention` runs on the `FLASH_ATTENTION` backend
while `MATH` *refuses* with "no viable backend" — proof it dispatches to the
fused varlen kernel rather than silently padding.

This matters for honesty: today's benchmark compares us against a baseline that
is *also* wasting work on padding. A packed comparison is the fair one.

- [ ] **Step 2: Benchmark, with allocation excluded**

Reuse the lesson from `docs/gqa-measurements.md`: preallocate with
`empty_partials` and pass `partials=`, and take medians. Timing `moba_forward`
directly measures the WDDM allocator, not the kernel — that mistake produced
incoherent numbers once already.

At `H_q=8, H_kv=2, block_size=128, top_k=8`, on a ragged batch (lengths sampled
256–2048), measure: tokens saved, `part_acc` bytes saved, **grid size packed vs
padded**, and time against nested-tensor dense SDPA.

- [ ] **Step 3: Write `docs/varlen-measurements.md`, prediction first**

State before the numbers: padding waste is `1 - mean(len)/max(len)`, so the gain
is a property of the *data*, not the kernel, and is ~0% on a uniform batch. Then
measure. Also predict that the grid *grows* — a ragged batch yields more real
blocks than a padded one yields useful ones — which per
`docs/gqa-measurements.md` may be worth more on 20 SMs than the bytes saved.

- [ ] **Step 4: Update the status line, then commit**

```bash
git add src/moba/backends/sdpa_dense.py bench/varlen_sweep.py bench/suite.py docs/varlen-measurements.md CLAUDE.md
git commit -m "bench: packed batches against a packed dense baseline"
```

---

## Risks

**A boundary crossing does not crash.** It produces finite, plausible numbers of
the right shape and the model quietly attends across documents. Every task
compares against per-sequence slices rather than checking shapes, and Task 3
asserts the boundary property directly.

**`N_BLOCKS` means two things in the gate** — the stride of `Kbar`, and the
candidate range (now `n_cand`). Conflating them is the likeliest bug in Task 3
and would appear as queries ranking against another sequence's representatives.

**Over-launched programs must be inert.** The per-sequence grid launches
`cdiv(max_seqlen, ·)` programs for every sequence including short ones. Each
must return before writing anything.
`test_short_sequences_get_programs_that_return_immediately` checks the gate
side; the attention kernel's existing `if n == 0: return` covers the block side.

**Padded-index memory.** `row_indices` is `[B, H_kv, G·max_seqlen·top_k]`; its
padding waste is the same raggedness ratio as the data's, but on metadata. At
`B=32, H_kv=2, G=4, max_seqlen=2048, top_k=8` that is ~16 MB. Acceptable, but
measure it in Task 7 rather than assume; the exact-sized alternative is a scan,
which is separate work.

**`_offsets_and_order` scaling is unchanged, deliberately.** It runs
`static_range(N_BLOCKS)` full-vector reductions in a single program. Because
block ids stay sequence-local (D-VL-6), `N_BLOCKS` is `NB` — one sequence's
block count — exactly as in the padded path. Global numbering would have
multiplied it by the batch. If a future change reintroduces global ids, this
becomes O(N²) over the whole batch.
