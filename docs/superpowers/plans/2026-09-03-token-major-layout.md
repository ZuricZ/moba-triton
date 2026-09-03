# Token-Major Layout Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move every kernel from head-major `[B, H, S, D]` to token-major
`[B, S, H, D]`, matching FlashAttention-2, NSA and FlashMoBA, with no
compatibility flag.

**Architecture:** The layout lives entirely in address arithmetic derived from
`constexpr` values — roughly eighteen expressions across five kernels. Nothing
becomes a runtime stride, so `constexpr` folding is preserved and there is no
performance cost to generalization, because nothing is generalized. Tests and
benchmarks are flipped to token-major *first*, kept green by a transposing shim
at the public boundary; the shim is then removed one kernel group at a time and
deleted in Task 5.

**Tech Stack:** Triton 3.7.1, PyTorch 2.13.0+cu126, Python 3.13, pytest.

**Spec:** `docs/deferred-decisions.md` DD-1, which this plan resolves and
whose conclusion it reverses. The reversal rests on two measurements taken
2026-09-03, both recorded in Task 6.

---

## Global Constraints

- **Branch:** `token-major-layout`, already created off `master` at `92ba645`.
- **No `head_first` flag.** FA-2 has none; DD-1's suggestion of one predates
  the measurement showing the layouts cost the same, which was the only reason
  to keep both callable. YAGNI. Adding one later is ~15 lines at two entry
  points.
- **Padded layout:** `[B, S, H, D]`, contiguous. **Packed layout:**
  `[1, T, H, D]`, contiguous, batch axis pinned to 1 — the direct translation of
  D-VL-1, and what NSA does (`assert q.shape[0] == 1`).
- **Our own buffers do not move.** `part_m`, `part_l`, `part_acc`, `dq_part`,
  `lse`, `delta` and every `CSCIndex` field keep their current layout. They are
  ours, nothing outside reads them, and moving them is churn with no caller.
- Google style, Google docstrings. Never add Claude as commit co-author.
- `references/` is read-only. Never edit.
- Run everything through the WSL prefix in `CLAUDE.md`. `uv run pytest`.
- The suite is 429 passed / 7 skipped at the start of this plan. No task may
  reduce that count.

---

## The one idea this plan turns on

Head-major and token-major differ in two strides:

| | `stride_head` | `stride_seq` | `stride_batch` |
|---|---|---|---|
| head-major `[B, H, S, D]` | `S*D` | `D` | `H*S*D` |
| token-major `[B, S, H, D]` | `D` | `H*D` | `S*H*D` |

Two consequences drive every task below.

**1. `stride_batch` is unchanged.** `H*S*D` and `S*H*D` are the same number, so
`q_base = pid_b * H_Q * S * D` survives verbatim.

**2. No token-major stride contains `S`.** Head-major's `stride_head = S*D`
does, which is the entire reason the kernels carry a `q_span` branch:

```python
# moba_fwd.py:94-100 -- today
if VARLEN:
    q_base = 0
    q_span = TOKENS          # packed: heads are TOKENS apart
else:
    q_base = pid_b * H_Q * S * D
    q_span = S               # padded: heads are S apart
...
q_row = heads * q_span + rows
q = tl.load(Q + q_base + q_row[:, None] * D + ...)
```

In token-major, heads are `D` apart and tokens are `H*D` apart **whether or not
the batch is packed**. The address becomes one expression:

```python
q_base = 0 if VARLEN else pid_b * H_Q * S * D
addr   = q_base + rows[:, None] * (H_Q * D) + heads[:, None] * D + offs_d[None, :]
```

> **`q_span` does not get a new value. It stops existing.**

**3. The fused index must split.** `q_row = heads * q_span + rows` is used for
three different things today, and head-major let them coincide:

| use | what it means | moves? |
|---|---|---|
| `Q + q_base + q_row*D` | *where the query lives in memory* | **yes** |
| `Lse + q_lin + q_row` | *which query this is* (our `[B,H,S]` buffer) | no |
| `dst = q_row * K_SLOTS + slots` | *which partial slot* (ours) | no |

So `q_row` stays exactly as it is, for our own buffers, and the *address*
becomes a separate expression. Conflating them again is the likeliest silent
bug in this plan.

---

## File Structure

**Kernels — address arithmetic only, no signature changes except deletions:**

| file | sites | what changes |
|---|---|---|
| `src/moba/kernels/moba_fwd.py` | 6 | `kv_base`, key stride, Q gather, `Out` store; `q_span` deleted |
| `src/moba/kernels/moba_bwd.py` | 8 | same, plus `DO`, `DQ`, `DK`/`DV` stores |
| `src/moba/kernels/gate.py` | 4 | `q_base`, `kbase`, and their row strides |
| `src/moba/kernels/fa2_minimal.py` | 1 | `base = pid_bh * S * D` |
| `src/moba/kernels/index_fused.py` | 0 | reads no raw Q/K; index fields are ours |

**Python surface:**

| file | what changes |
|---|---|
| `src/moba/packing.py` | `pack` becomes a row gather (DD-1's point 2); shapes to `[1, T, H, D]` |
| `src/moba/ref/naive.py` | the oracle, token-major |
| `src/moba/backends/sdpa_dense.py` | transpose for SDPA, which is head-major |
| `src/moba/backends/_refs.py` | `to_varlen`/`from_varlen` become near no-ops |
| `src/moba/kernels/autograd.py` | docstrings; the transposing shim lives here in Tasks 1-4 |

**Tests and benchmarks:** `tests/conftest.py` (`make_qkv`, `SHAPE_SWEEP` ids),
then the per-file counts surveyed on this branch — `test_varlen.py` 15,
`test_gqa.py` 11, `test_gate.py` 11, `test_packing.py` 9, `test_tolerance.py` 8,
`test_moba_kernel.py` 7, `test_semantics.py` 6, and the rest in single digits.
Most funnel through helpers (`make_qkv`, `_packed`, `_grads`), so the true site
count is far below the 87 raw `randn(` matches.

---

## Task 1: Flip the surface to token-major, kept green by a shim

The user's ordering: tests define token-major first, implementation follows.
The shim exists so this task ends green rather than opening a 400-test red
window. It is scaffolding and is deleted in Task 5.

**Files:**
- Modify: `tests/conftest.py`, `src/moba/ref/naive.py`, `src/moba/packing.py`,
  `src/moba/kernels/autograd.py`, `src/moba/kernels/moba_fwd.py`
  (`moba_forward` only, not the kernel)
- Test: every file in `tests/`

**Plan correction, made during execution.** `ref/naive.py` and `packing.py`
were originally scheduled for Task 5. That was wrong: tests call the oracle
*directly* with token-major tensors from this task onward, so it cannot lag
behind them. Both are pure PyTorch with no kernel risk, and the oracle is the
spec -- it should define token-major first, not last. Only the *kernels* are
shimmed. `pack` becoming a row gather (DD-1's point 2) moves here with it.

**Interfaces:**
- Consumes: nothing.
- Produces: `moba_attention(q, k, v, cfg, ...)` and `moba_forward(...)` taking
  `[B, S, H, D]` and returning `[B, S, H, D]`. `conftest.make_qkv(shape)` where
  `shape` is `(batch, seq_len, heads, head_dim)`.

- [x] **Step 1: Rewrite `make_qkv` and the sweep constants**

`SHAPE_SWEEP` entries are `(batch, heads, seq_len, head_dim)` today. Reorder to
`(batch, seq_len, heads, head_dim)` and fix `case_id` to match, so ids stay
readable and comparable to old runs:

```python
# (batch, seq_len, heads, head_dim) -- token-major, matching FA-2.
SHAPE_SWEEP = [
    (1, 1024, 4, 128),
    (2, 1024, 4, 128),
    (1, 2048, 4, 128),
    (1, 2048, 4, 64),
]

LONG_SHAPE_SWEEP = [
    (1, 4096, 4, 128),
    (1, 4096, 4, 64),
    (1, 8192, 2, 128),
]


def case_id(shape, gate):
    """Builds a readable pytest id, e.g. b1s1024h4d128-blk128-top8."""
    b, s, h, d = shape
    bk, tk = gate
    return f"b{b}s{s}h{h}d{d}-blk{bk}-top{tk}"
```

`gate_is_sparse` reads `shape[2]` for the sequence length. It must become
`shape[1]`. This is the kind of positional read that will not fail loudly —
it would silently start reading the head count and declare everything sparse.
Grep for `shape[2]` across `tests/` and `bench/` before moving on.

- [x] **Step 2: Add the temporary shim, clearly marked**

In `src/moba/kernels/autograd.py`, above `moba_attention`:

```python
def _shim_to_head_major(*tensors):
    """TEMPORARY. Deleted in Task 5 of the token-major plan.

    Lets the public API speak token-major before the kernels do, so the suite
    stays green through the migration instead of going red for four tasks. Each
    call is a real transpose and copy; every benchmark run while this exists is
    measuring the shim as well as the kernel.
    """
    return tuple(t.transpose(1, 2).contiguous() for t in tensors)


def _shim_from_head_major(t):
    """TEMPORARY. See `_shim_to_head_major`."""
    return t.transpose(1, 2).contiguous()
```

Apply it at the two public entry points only — `moba_attention` and
`moba_forward` — leaving every kernel untouched.

- [x] **Step 3: Flip the tests, file by file, running after each**

Work in ascending order of construction count so the cheap files shake out
helper problems first: `test_smoke`, `test_kernel_invariants`, `test_bench`,
`test_index_fused`, `test_index`, `test_registry`, `test_backward`,
`test_optional_backends`, `test_fa2_minimal`, `test_heads`, `test_semantics`,
`test_moba_kernel`, `test_tolerance`, `test_packing`, `test_gate`, `test_gqa`,
`test_varlen`.

Two mechanical rules cover nearly all of it:

```python
# tensor construction
torch.randn(1, H, S, D, ...)   ->  torch.randn(1, S, H, D, ...)

# any axis-2 sequence slice
t[:, :, lo:lo + n]             ->  t[:, lo:lo + n]
```

The second is the dangerous one. `t[:, :, lo:lo+n]` on a token-major tensor
slices *heads*, is legal, returns the right rank, and is wrong. Grep
`\[:, :, ` across `tests/` and inspect every hit; there is no shape error to
catch this.

- [x] **Step 4: Run the whole suite**

Run: `uv run pytest -q`
Expected: `429 passed, 7 skipped`. Any other number means a slice went wrong,
not that the shim is imperfect — the shim is exact by construction.

- [x] **Step 5: Flip the benchmarks**

`bench/gqa_sweep.py`, `bench/gqa_grid_probe.py`, `bench/stages.py`,
`bench/divergence.py`, `bench/varlen_sweep.py`, `bench/suite.py`,
`bench/load_balance.py`, `bench/gather_cost.py`. Same two rules. Do not record
any number produced in this state — the shim is in the timed region.

- [x] **Step 6: Commit**

```bash
git add tests/ bench/ src/moba/kernels/autograd.py src/moba/kernels/moba_fwd.py
git commit -m "test: token-major surface, kernels still head-major behind a shim"
```

---

## Task 2: Gate and index, native token-major

The gate is the cleanest separation point in the pipeline: it reads raw Q and K
and emits a `CSCIndex`, which is entirely ours and layout-free. So it can go
native while everything downstream still runs through the shim.

**Files:**
- Modify: `src/moba/kernels/gate.py` (4 address sites),
  `src/moba/kernels/index_fused.py` (call sites only),
  `src/moba/kernels/autograd.py` (shim moves below the index build)
- Test: `tests/test_gate.py`, `tests/test_index_fused.py`, `tests/test_index.py`

**Interfaces:**
- Consumes: token-major `q`, `k` from Task 1's public surface.
- Produces: `build_csc_fused(q, k, cfg, cu_seqlens=None, max_seqlen=None)`
  taking `[B, S, H, D]`, returning an unchanged `CSCIndex`.

- [x] **Step 1: Convert `_moba_gate_fwd`'s query addressing**

`gate.py:156-166` today:

```python
    if VARLEN:
        q_base = hq * T * D
        out_base = hq * T * TOP_K
    else:
        q_base = pid_bh * S * D
        out_base = pid_bh * S * TOP_K
    q = tl.load(Q + q_base + offs_m[:, None] * D + offs_d[None, :], ...)
```

Becomes:

```python
    # Token-major: heads are D apart and tokens are H_Q*D apart, neither of
    # which depends on the sequence length -- so packed and padded share one
    # expression and only the batch term differs.
    if VARLEN:
        q_base = hq * D
        out_base = hq * T * TOP_K            # BlockIdx is ours; unchanged
    else:
        q_base = pid_b * S * H_Q * D + hq * D
        out_base = pid_bh * S * TOP_K        # ours; unchanged
    q = tl.load(Q + q_base + offs_m[:, None] * (H_Q * D) + offs_d[None, :], ...)
```

`BlockIdx` is our output, so `out_base` and the store at `gate.py:303` do not
move. Note `pid_b` must be available in the non-varlen branch; today only the
varlen branch derives it.

- [x] **Step 2: Convert `block_representatives_triton`'s key addressing**

`gate.py:448-457`:

```python
        kbase = h * T * D                       # K is [1, H_kv, T, D]
        ...
        kbase = pid_bh * S * D
    k = tl.load(K + kbase + rows[:, None] * D + offs_d[None, :], ...)
```

Becomes:

```python
        kbase = h * D                           # K is [1, T, H_kv, D]
        ...
        kbase = pid_b * S * H_KV * D + h * D
    k = tl.load(K + kbase + rows[:, None] * (H_KV * D) + offs_d[None, :], ...)
```

`Kbar`, the pooled output, is ours and stays as it is.

- [x] **Step 3: Move the shim below the index build**

In `moba_forward`, build the index from the token-major tensors, *then* shim:

```python
    if index is None:
        index = build_csc_fused(q, k, cfg, cu_seqlens, max_seqlen)
    q, k, v = _shim_to_head_major(q, k, v)   # TEMPORARY, Tasks 3-4 remove this
```

- [x] **Step 4: Run gate and index tests, then the suite**

Run: `uv run pytest tests/test_gate.py tests/test_index.py tests/test_index_fused.py -q`
Then: `uv run pytest -q`
Expected: `429 passed, 7 skipped`.

The gate tests compare selection against a PyTorch oracle, so a wrong stride
shows up as *different blocks selected*, not a crash. If only the long-sequence
cases fail, suspect `H_Q * D` versus `H_KV * D` — they coincide when the head
counts are equal, which most fast-tier shapes have.

- [x] **Step 5: Commit**

```bash
git add src/moba/kernels/gate.py src/moba/kernels/index_fused.py src/moba/kernels/autograd.py
git commit -m "feat: token-major gate and block representatives"
```

---

## Task 3: Forward attention and combine, native token-major

**Files:**
- Modify: `src/moba/kernels/moba_fwd.py` (6 sites), `src/moba/kernels/autograd.py`
- Test: `tests/test_moba_kernel.py`, `tests/test_semantics.py`,
  `tests/test_gqa.py`, `tests/test_tolerance.py`

**Interfaces:**
- Consumes: token-major `q`, `k`, `v`; the `CSCIndex` from Task 2.
- Produces: `attend_partials` and `combine_partials` unchanged in signature;
  `Out` written token-major.

- [x] **Step 1: Delete `q_span` and convert the Q gather**

`moba_fwd.py:94-100, 151-156` today:

```python
    if VARLEN:
        q_base = 0
        q_part_base = 0
        q_span = TOKENS
    else:
        q_base = pid_b * H_Q * S * D
        q_part_base = pid_b * H_Q * S * K_SLOTS
        q_span = S
    ...
        q_row = heads * q_span + rows
        q = tl.load(Q + q_base + q_row[:, None] * D + offs_d[None, :], ...)
```

Becomes:

```python
    # `q_span` is gone. It existed only because head-major's stride_head was
    # S*D, which contains the sequence length; token-major's is D, which does
    # not, so packed and padded index queries identically.
    #
    # `q_row` stays -- but it now means *which query*, not *where it lives*.
    # It addresses our own partial buffers, which did not move.
    if VARLEN:
        q_base = 0
        q_part_base = 0
        q_span = TOKENS          # partials only, still head-major
    else:
        q_base = pid_b * H_Q * S * D      # unchanged: H*S*D == S*H*D
        q_part_base = pid_b * H_Q * S * K_SLOTS
        q_span = S
    ...
        q_row = heads * q_span + rows                    # ours: partial slots
        q_addr = (q_base + rows[:, None] * (H_Q * D)
                  + heads[:, None] * D + offs_d[None, :])
        q = tl.load(Q + q_addr, ...)
```

Keep the name `q_span` for the partials, and add a comment saying it no longer
describes Q. Renaming it to `part_span` in the same commit is fine and probably
clearer; do one or the other, not neither.

- [x] **Step 2: Convert `kv_base` and the key stride**

`moba_fwd.py:119-131`:

```python
        kv_base = (pid_bh % H_KV) * TOKENS * D
    else:
        kv_base = pid_bh * S * D
    k = tl.load(K + kv_base + key_pos[:, None] * D + offs_d[None, :], ...)
```

Becomes:

```python
        kv_base = (pid_bh % H_KV) * D
    else:
        kv_base = pid_b * S * H_KV * D + (pid_bh % H_KV) * D
    k = tl.load(K + kv_base + key_pos[:, None] * (H_KV * D) + offs_d[None, :], ...)
```

Same for `V` on the following line. Note the non-varlen branch changes from
`pid_bh * S * D` to a `pid_b`/`pid_bh % H_KV` split — the packed path already
does this split, so the two branches converge.

- [x] **Step 3: Convert the `Out` store in `_moba_combine`**

`moba_fwd.py:546`:

```python
    tl.store(Out + pid_bh * S * D + offs_m[:, None] * D + offs_d[None, :], ...)
```

Becomes:

```python
    tl.store(Out + (pid_bh // H_Q) * S * H_Q * D + (pid_bh % H_Q) * D
             + offs_m[:, None] * (H_Q * D) + offs_d[None, :], ...)
```

`PartM`, `PartL`, `PartAcc` and the `Lse` store in the same kernel are ours and
do not move. Check that `H_Q` is in this kernel's signature; add it as a
`constexpr` if not.

- [x] **Step 4: Allocate `Out` token-major**

Wherever `moba_fwd` allocates the output, it must now be `[B, S, H_Q, D]`.
Grep `torch.empty` and `empty_like` in the file; `empty_like(q)` is already
correct and needs no change, which is the outcome to prefer.

- [x] **Step 5: Shim the backward only**

`moba_attention`'s forward is now native. Its backward is not, so save the
head-major copies for `ctx` and mark it:

```python
    # TEMPORARY: Task 4 makes the backward native and deletes this.
```

- [x] **Step 6: Run forward tests, then the suite**

Run: `uv run pytest tests/test_moba_kernel.py tests/test_semantics.py tests/test_gqa.py -q`
Then: `uv run pytest -q`
Expected: `429 passed, 7 skipped`.

`test_semantics.py` compares against the fp64 oracle, so a stride error is a
numeric mismatch rather than a crash. If GQA cases fail but multi-head cases
pass, the cause is `H_Q * D` used where `H_KV * D` belongs — they are equal
exactly when the group size is 1.

- [x] **Step 7: Commit**

```bash
git add src/moba/kernels/moba_fwd.py src/moba/kernels/autograd.py
git commit -m "feat: token-major forward attention and combine"
```

---

## Task 4: Backward, native token-major

Eight sites, all the same two transformations as Task 3 plus the `DK`/`DV`
stores. Per `docs/learning/05-the-backward-pass.md`, `dK` and `dV` are written
by the program that owns those keys, so their store shares `kv_base` with the
load — convert both together or neither.

**Files:**
- Modify: `src/moba/kernels/moba_bwd.py` (8 sites),
  `src/moba/kernels/autograd.py` (last shim call removed)
- Test: `tests/test_backward.py`, `tests/test_gqa.py`,
  `tests/test_kernel_invariants.py`

**Interfaces:**
- Consumes: token-major `q`, `k`, `v`, `dout`; `lse` and `delta` unchanged.
- Produces: `dq`, `dk`, `dv` token-major.

- [x] **Step 1: Convert `_moba_bwd_preprocess`**

`moba_bwd.py:116-119` reads `DO` and `Out`, both user tensors:

```python
    base = pid_bh * S * D + offs_m[:, None] * D + offs_d[None, :]
```

Becomes:

```python
    base = ((pid_bh // H_Q) * S * H_Q * D + (pid_bh % H_Q) * D
            + offs_m[:, None] * (H_Q * D) + offs_d[None, :])
```

`Delta`, its output, is ours and stays `[B, H, S]`.

- [x] **Step 2: Convert the main kernel's Q, DO, K and V loads**

`moba_bwd.py:167-231`. Identical to Task 3 Steps 1-2: delete `q_span` from the
address, split `kv_base`, multiply row strides by `H_Q * D` or `H_KV * D`.
`DO` uses the same address as `Q`. `Lse` and `Delta` use `q_lin + q_row` and do
not move.

- [x] **Step 3: Convert the `DK`/`DV` and `DQ` stores**

`moba_bwd.py:290-292` already routes through `kv_base`, so it follows Step 2
for free — this is the site that was missed once before and corrupted
neighbouring allocations without faulting. Verify it by reading, not by
assuming the tests would catch it.

`moba_bwd.py:339`, the `DQ` store in the reduce pass:

```python
    tl.store(DQ + pid_bh * S * D + offs_m[:, None] * D + offs_d[None, :], ...)
```

takes the same `(pid_bh // H_Q, pid_bh % H_Q)` split as Step 1. `dq_part`, its
input, is ours and does not move.

- [x] **Step 4: Delete the last shim call**

`autograd.py` should now hold no `_shim_*` calls. The helpers themselves are
removed in Task 5.

- [x] **Step 5: Run backward tests, then the suite**

Run: `uv run pytest tests/test_backward.py tests/test_kernel_invariants.py -q`
Then: `uv run pytest -q`
Expected: `429 passed, 7 skipped`.

If `dq` fails while `dk`/`dv` pass, suspect the `DQ` store rather than the
gather — and read `docs/learning/05-the-backward-pass.md` on why the reverse
pattern (a `dk` bug surfacing as a `dq` failure) has happened here before.

- [x] **Step 6: Commit**

```bash
git add src/moba/kernels/moba_bwd.py src/moba/kernels/autograd.py
git commit -m "feat: token-major backward"
```

---

## Task 5: Packing, the reference path, and deleting the shim

This is where DD-1's second argument finally pays: `pack` stops being a strided
gather and becomes FlashAttention's row gather.

**Files:**
- Modify: `src/moba/packing.py`, `src/moba/ref/naive.py`,
  `src/moba/backends/sdpa_dense.py`, `src/moba/backends/_refs.py`,
  `src/moba/kernels/fa2_minimal.py`, `src/moba/kernels/autograd.py`
- Test: `tests/test_packing.py`, `tests/test_varlen.py`,
  `tests/test_fa2_minimal.py`, `tests/test_optional_backends.py`

**Interfaces:**
- Consumes: everything from Tasks 2-4.
- Produces: `pack(x, attention_mask) -> ([1, T, H, D], cu_seqlens, max_seqlen)`;
  `unpack(packed, cu_seqlens, batch, seq_len) -> [B, S, H, D]`;
  `from_nested(nt) -> ([1, T, H, D], cu_seqlens, max_seqlen)`.

- [x] **Step 1: Make `pack` a row gather**

`packing.py:154-157` today:

```python
    b_idx, s_idx = mask.nonzero(as_tuple=True)
    packed = x[b_idx, :, s_idx]                       # [T, H, D]
    packed = packed.permute(1, 0, 2).unsqueeze(0).contiguous()
```

Becomes:

```python
    # One token is now one contiguous H*D run, so this is a row gather --
    # the same operation FlashAttention's `unpad_input` performs
    # (`references/flash-moba/flash_moba/bert_padding.py:98`). Under
    # head-major it was H short transactions per token; see DD-1.
    b_idx, s_idx = mask.nonzero(as_tuple=True)
    packed = x[b_idx, s_idx].unsqueeze(0)             # [1, T, H, D]
```

No permute and no `.contiguous()`: advanced indexing on the leading two axes
already yields a contiguous `[T, H, D]`.

`from_nested` simplifies further — `nt.values()` *is* `[T, H, D]`, so the
transpose disappears entirely and the function becomes `unsqueeze(0)` plus a
dtype cast on the offsets. Delete the DD-1 cost note in its docstring; it no
longer applies.

`unpack` inverts the same way: `out[b_idx, s_idx] = packed[0]`.

- [x] **Step 2: Update the module docstring**

`packing.py`'s "WHY NOT FLASHATTENTION'S HELPERS" section argues from
head-major and is now false. Replace it with a note saying the layouts agree
and the helpers *could* be used, with the reason we still ship our own (they
carry an autograd `Function` and a `bert_padding` dependency we do not want).

- [x] **Step 3: Convert `ref/naive.py` and `fa2_minimal.py`**

The oracle is pure PyTorch; the conversion is in its einsum/permute
expressions. `fa2_minimal.py:66` is one address site, `base = pid_bh * S * D`,
taking the same split as Task 4 Step 1. `fa2_minimal.py:191`'s `.contiguous()`
calls stay — its inputs are token-major and already contiguous, so they become
no-ops.

- [x] **Step 4: Simplify `_refs.py`**

`to_varlen` is `x.permute(0, 2, 1, 3).reshape(b*s, h, d).contiguous()`. Under
token-major it is `x.reshape(b*s, h, d)` — no permute, no copy. `from_varlen`
becomes `x.view(batch, seq_len, h, d)`. Both keep their names and docstrings;
only the bodies and the shape lines in the docstrings change.

- [x] **Step 5: Give `sdpa_dense` its transposes**

Torch SDPA is head-major, so the baseline now converts rather than the kernel.
`forward` becomes a transpose in and out; `forward_packed`'s `jag` helper gets
*simpler*, since `nested_tensor_from_jagged` wants exactly `[T, H, D]`:

```python
        def jag(t: torch.Tensor) -> torch.Tensor:
            # [1, T, H, D] -> [T, H, D] -> nested [B, j, H, D] -> [B, H, j, D].
            return nested_tensor_from_jagged(
                t[0], offsets=offsets).transpose(1, 2)
```

The `.contiguous()` disappears. Record in the docstring that the transposes
moved from us to the baseline, and that they are inside the timed region — so
`docs/varlen-measurements.md`'s dense column is no longer comparable to its
Task 7 values and is re-measured in Task 6.

- [x] **Step 6: Delete `_shim_to_head_major` and `_shim_from_head_major`**

Then prove they are gone:

```bash
grep -rn "_shim_" src/ tests/ bench/     # expected: no output
grep -rn "transpose(1, 2)" src/moba/kernels/    # expected: no output
```

- [x] **Step 7: Run the full suite**

Run: `uv run pytest -q`
Expected: `429 passed, 7 skipped`.

- [x] **Step 8: Commit**

```bash
git add src/moba/packing.py src/moba/ref/naive.py src/moba/backends/ src/moba/kernels/
git commit -m "feat: token-major packing, oracle and baselines; shim removed"
```

---

## Task 6: Measure, document, resolve DD-1

**Files:**
- Create: `bench/layout_probe.py`, `docs/layout-measurements.md`
- Modify: `docs/deferred-decisions.md`, `docs/varlen-measurements.md`,
  `docs/gqa-measurements.md`, `CLAUDE.md`
- Test: `tests/test_bench.py` if it enumerates `bench/` modules

- [x] **Step 1: Promote the layout probe out of scratch**

The probe run on 2026-09-03 is the measurement this whole plan rests on, so it
belongs in `bench/` where it can be re-run on rented hardware. It loads K and V
tiles in both layouts and times only that, holding bytes, block size and
occupancy identical. Its result on this card:

```
  B   H      S    D  BLK | head-maj  token-maj   ratio
  1   8   8192   64  128 |   0.1434     0.1393  0.971x
  1   8   8192  128  128 |   0.2427     0.2396  0.987x
  1   2   8192   64  128 |   0.0420     0.0420  1.000x
  4   8   2048   64  128 |   0.1270     0.1280  1.008x
  1   8   8192   64   64 |   0.1167     0.1178  1.009x
  1   8   8192   64  256 |   0.1382     0.1362  0.985x
```

- [x] **Step 2: Re-run every benchmark and record before/after**

`bench/stages.py`, `bench/gqa_sweep.py`, `bench/varlen_sweep.py`. The forward
was 2.18x over dense SDPA at S=8192 and the backward 2.58x on `master`; varlen
was 0.99x uniform and 8.49x at 82% waste. Any movement beyond noise in these is
a finding, not a rounding difference — the probe predicts none.

- [x] **Step 3: Write `docs/layout-measurements.md`, prediction first**

State before the numbers: the change should be *neutral*, because nothing was
generalized — the addressing is still `constexpr`-folded, the same bytes are
read, and the probe says the transaction shape does not matter on this card.
The one place a real improvement is expected is `pack`, which went from `H`
short transactions per token to one long one.

Also state the prediction that is *not* testable here: token-major removes a
layout conversion at the FlashMoBA benchmark boundary, which is DD-1's third
revisit trigger and needs rented sm_80/90.

- [x] **Step 4: Rewrite DD-1 as resolved**

`docs/deferred-decisions.md` must record the reversal honestly, including that
its own reasoning was wrong in two specific places:

1. *"it is what fla-org's NSA calls `head_first=True`"* — true but misleading.
   NSA's kernels are token-major; `head_first` is a `rearrange` shim
   (`ops/parallel.py:1381`) and NSA **refuses** it for varlen
   (`ops/naive.py:86`). The citation supported the opposite of what it was used
   for.
2. *"The kernel is indifferent"* — asserted, and under-argued for our
   block-major loop order where a program owns contiguous keys. Now measured
   at 1.00x ± 3%, so the conclusion held for a reason DD-1 did not give.

Add the argument DD-1 could not have made, because it predates varlen: no
token-major stride contains `S`, so packed and padded query addressing become
one expression and the `q_span` branch disappears.

- [x] **Step 5: Update the status line and the docs that cite layout**

`CLAUDE.md`'s status line, plus any sentence in `docs/varlen-measurements.md`
or `docs/gqa-measurements.md` that describes shapes as `[B, H, S, D]`.

- [x] **Step 6: Commit and open the PR**

```bash
git add bench/layout_probe.py docs/ CLAUDE.md
git commit -m "docs: resolve DD-1 in favour of token-major, with measurements"
git push -u origin token-major-layout
```

---

## Risks

**A wrong stride does not crash.** Every expression here produces an in-bounds
address for *some* element, so the failure is a plausible number from the wrong
token or the wrong head. The suite's fp64 oracle comparisons and per-sequence
slice comparisons are what catch it; shape assertions are not.

**`H_Q * D` and `H_KV * D` coincide when the group size is 1**, which is most
of the fast tier. A GQA-only failure means a query stride was used for keys or
the reverse. `tests/test_gqa.py` is the tripwire and must be run in Tasks 2-4,
not saved for the end.

**`t[:, :, lo:lo+n]` is legal and wrong** on a token-major tensor — it slices
heads. There is no shape error. Grep for it in Task 1 Step 3 and inspect every
occurrence by hand.

**The fused `q_row` means two things.** It is both "which query" (our buffers)
and "where the query lives" (the input tensor). Head-major let one expression
serve both; token-major does not. Re-fusing them is the likeliest silent bug in
this plan, and it would look like a head-permutation in the output.

**`conftest.gate_is_sparse` reads `shape[2]`** for the sequence length. After
the reorder that position holds the head count, and the function would silently
declare every configuration sparse — emitting cases the sweep was designed to
exclude rather than failing.

**Benchmarks taken during Tasks 1-4 are meaningless.** The shim is a real copy
inside the public entry point. Only Task 6's numbers count.
