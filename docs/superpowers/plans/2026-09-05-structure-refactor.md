# Structure Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task.
> Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `src/moba` legible to someone who knows MoBA and Triton but not
this repository, by adopting the naming and grouping conventions the reference
kernels already use.

**Architecture:** Three moves, none of which touch behaviour. (1) Encode the
*tier* in every function name -- `_kernel` suffix for `@triton.jit` device code,
bare names for host launchers -- as fla-org/NSA and Triton-MoBA do. (2) Name
modules after the *stages of the MoBA algorithm* rather than after
implementation nouns. (3) Order and section the surviving large files so device
code, launchers and public API are visually separated.

**Tech Stack:** Triton 3.7.1, torch 2.13.0+cu126, Python 3.13, pytest.

**Spec:** No separate spec. The design was settled in conversation on
2026-09-05; the argument and the evidence are in "Why these conventions" below,
which quotes the vendored references directly.

## Global Constraints

- **No behaviour changes.** Every task is a rename, a move, or a comment. If a
  test goes red, a rename was applied wrongly -- do not "fix" the test.
- **435 passed, 1 skipped** is the count at every checkpoint. Compare the
  **skip count too**: this project has twice shipped tests that silently
  stopped asserting, and a rising skip count is the only cheap signal.
- Renames fail *closed* -- Python raises `ImportError`/`NameError` loudly. This
  is the opposite of the token-major layout migration, where a wrong axis
  produced a well-formed tensor. Trust the suite more here than there.
- Google style for all code and docstrings.
- `references/` is vendored and read-only. Never edit it.
- Never add Claude/Anthropic as a commit co-author.
- Run everything through WSL:
  `wsl -e bash -lc 'export PATH="$HOME/.local/bin:$PATH"; export UV_PROJECT_ENVIRONMENT="$HOME/.venvs/moba"; cd "/mnt/c/D/Side Projects/GPU Kernel" && uv run <cmd>'`
- Commit from PowerShell, not WSL -- git identity is set on Windows only.
- **Comments and docstrings are explicitly out of scope.** The prose pass is a
  separate later job. Do not shorten, expand or rewrite prose here beyond what a
  rename mechanically requires. Keeping the two apart is what makes the diff
  reviewable.

---

## Why these conventions

Not invented here. Every vendored reference converges on the same three rules.

**Tier in the name.** fla-org/NSA, `native_sparse_attention/ops/parallel.py`,
1,434 lines and twenty functions:

```
parallel_nsa_fwd_kernel          @triton.jit   device code
parallel_nsa_bwd_kernel_dq       @triton.jit
parallel_nsa_fwd                 host launcher
parallel_nsa_bwd                 host launcher
ParallelNSAFunction(torch.autograd.Function)    <- the only class
parallel_nsa                     public entry point
```

Triton-MoBA and FlashMoBA do the same. Across all three repositories there is
exactly one class per file family and it is always `torch.autograd.Function`.
There is no kernel object, no builder, no strategy. This is not stylistic
preference: a `@triton.jit` function is a source-level template the compiler
specializes per `constexpr`. It cannot be a method, cannot close over `self`,
and takes every argument flat. The device code *must* be free functions, and the
host code stays flat to sit beside it.

**Our current names lose the distinction.** Eleven `_`-prefixed names are
`@triton.jit` device code; nine are ordinary module-private helpers. The same
marker means "compiler template, different language rules" and "small internal
function". Meanwhile `_moba_attn_fwd` does *not* carry `_kernel`, which is the
marker the field actually uses.

**Large files are fine; mixed files are not.** NSA is 1,434 lines in one module,
Triton-MoBA 1,005. Size is not the defect. What those files do that ours do not
is keep one strict order -- every `@triton.jit` first, then every launcher, then
the autograd class, then the entry point -- so the reader always knows which
tier they are in. `moba_fwd.py` interleaves a kernel, tuning heuristics, a
second kernel, launchers, validation and the public API.

**Module names should name the algorithm.** MoBA has three stages, and a reader
who knows the paper knows them: pool keys into per-block representatives, score
queries against those representatives and take top-k, attend to the winners.
Today none of that is visible. `gate.py` holds stages 1 and 2 with stage 1
*below* stage 2, so the file runs backwards. `index.py` and `index_fused.py` are
two builders of one structure, named for the structure's data layout (CSC)
rather than its job.

---

## File Structure

Target layout. Line counts are estimates and carry the current prose with them;
the prose pass is separate.

```
src/moba/
  __init__.py        3 ->  ~25   exports moba_attention, MoBAConfig  (new)
  config.py         99          unchanged
  device.py        207          unchanged
  gqa.py            84          was heads.py
  varlen.py        236          was packing.py
  kernels/
    pooling.py       ~135  stage 1: mean-pool K -> block representatives
                           (from gate.py:415-549)
    gating.py        ~420  stage 2: score q vs representatives, top-k
                           (from gate.py:1-414)
    routing.py       ~565  stage 3 prep: selection -> CSC the kernel walks
                           (index.py + index_fused.py merged)
    tuning.py        ~180  GROUP / warp / shared-memory heuristics
                           (from moba_fwd.py, shared with moba_bwd.py)
    moba_fwd.py      ~730  attention + combine kernels and their launchers
    moba_bwd.py      490   unchanged but re-sectioned
    fa2_minimal.py   209   unchanged but re-sectioned
    autograd.py      123   unchanged
  backends/
    layout.py        125   was _refs.py
    ...              unchanged
  ref/naive.py       329   unchanged
tests/
  tolerance.py       170   moved out of src/moba
```

Two deliberate non-moves:

- **`moba_fwd.py` stays one file** after `tuning.py` leaves. Splitting the
  combine kernel out would separate it from the attention kernel whose output it
  consumes, and the references keep exactly this pairing together.
- **`routing.py` merges two files rather than splitting one.** The torch builder
  and the Triton builder produce the same `CSCIndex`; they are one stage with
  two implementations, which is a section boundary, not a file boundary.

---

## Task 0: Land the call-site fixes

Already complete on branch `fix/token-major-call-sites`, verified at 435 passed
/ 1 skipped. This plan assumes that branch is merged first. Do not start Task 1
until `git log --oneline -1` shows those fixes on the working branch.

---

## Task 1: Suffix the device code

Eleven `@triton.jit` functions gain `_kernel` and lose the leading underscore.
Nothing else changes.

**Files:**
- Modify: `src/moba/kernels/gate.py`, `moba_fwd.py`, `moba_bwd.py`,
  `index_fused.py`, `fa2_minimal.py`
- Modify (call sites): `tests/test_moba_kernel.py`,
  `tests/test_kernel_invariants.py`, `tests/test_index_fused.py`

**Interfaces:**
- Produces: the renamed device symbols below. Nothing outside `src/moba` and the
  three test modules named above may reference them.

| old | new |
|---|---|
| `_moba_attn_fwd` | `moba_fwd_kernel` |
| `_moba_combine` | `moba_combine_kernel` |
| `_moba_bwd` | `moba_bwd_kernel` |
| `_moba_bwd_preprocess` | `moba_bwd_preprocess_kernel` |
| `_moba_dq_reduce` | `moba_dq_reduce_kernel` |
| `_moba_gate_fwd` | `gate_kernel` |
| `_block_representatives_fwd` | `pool_blocks_kernel` |
| `_count_columns` | `count_columns_kernel` |
| `_offsets_and_order` | `offsets_and_order_kernel` |
| `_scatter_pairs` | `scatter_pairs_kernel` |
| `_fa2_fwd` | `fa2_fwd_kernel` |

- [x] **Step 1: Confirm the starting count**

```bash
uv run pytest -q -rs 2>&1 | tail -3
```

Expected: `435 passed, 1 skipped`.

- [x] **Step 2: Apply the renames, asserting every occurrence count**

Use exact-count substitution, never a loose regex. A pattern that silently
matches zero times is how the token-major migration left files unpatched.

```python
import pathlib, sys

RENAMES = {
    "_moba_attn_fwd": "moba_fwd_kernel",
    "_moba_combine": "moba_combine_kernel",
    "_moba_bwd_preprocess": "moba_bwd_preprocess_kernel",
    "_moba_dq_reduce": "moba_dq_reduce_kernel",
    "_moba_bwd": "moba_bwd_kernel",          # after _moba_bwd_preprocess
    "_moba_gate_fwd": "gate_kernel",
    "_block_representatives_fwd": "pool_blocks_kernel",
    "_count_columns": "count_columns_kernel",
    "_offsets_and_order": "offsets_and_order_kernel",
    "_scatter_pairs": "scatter_pairs_kernel",
    "_fa2_fwd": "fa2_fwd_kernel",
}

paths = (list(pathlib.Path("src").rglob("*.py"))
         + list(pathlib.Path("tests").rglob("*.py"))
         + list(pathlib.Path("bench").rglob("*.py")))

total = {k: 0 for k in RENAMES}
for p in paths:
    s = p.read_text(encoding="utf-8")
    out = s
    for old, new in RENAMES.items():
        total[old] += out.count(old)
        out = out.replace(old, new)
    if out != s:
        p.write_text(out, encoding="utf-8")
        print("rewrote", p)

for old, n in total.items():
    print(f"{n:3d}  {old}")
    if n == 0:
        sys.exit(f"{old} matched nothing -- check the name")
```

Note the ordering: `_moba_bwd_preprocess` must be substituted before
`_moba_bwd`, or the shorter key corrupts the longer name.

- [x] **Step 3: Verify no old name survives**

```bash
grep -rn '_moba_attn_fwd\|_moba_combine\|_moba_bwd\b\|_moba_gate_fwd\|_block_representatives_fwd\|_fa2_fwd' src tests bench --include=*.py
```

Expected: no output.

- [x] **Step 4: Run the full suite**

```bash
uv run pytest -q -rs 2>&1 | tail -3
```

Expected: `435 passed, 1 skipped` -- both numbers.

- [x] **Step 5: Commit**

```bash
git add -A
git commit -m "refactor: suffix Triton device code with _kernel"
```

---

## Task 2: Make the launcher tier consistent

Two launchers and one class do not follow the tier convention.

**Files:**
- Modify: `src/moba/kernels/gate.py`, `src/moba/kernels/autograd.py`,
  `src/moba/kernels/index_fused.py`
- Modify (call sites): `tests/test_gate.py` (7), `tests/test_varlen.py` (2)

**Interfaces:**
- Produces: `pool_blocks(k, cfg, ...)` -- same signature as
  `block_representatives_triton`, only the name changes.
- Produces: `MoBAAttentionFunction` -- same class as `_MoBAAttention`.

| old | new | why |
|---|---|---|
| `block_representatives_triton` | `pool_blocks` | The `_triton` suffix announced device code one tier too high; every launcher in this package launches Triton. |
| `_MoBAAttention` | `MoBAAttentionFunction` | The references name this tier `ParallelNSAFunction`, `FlashMobaAttnVarlenFunc`, `MixedAttention`. It is the one class in the package; hiding it behind `_` hides the tier. |
| `gate_select` | `gate_select` | unchanged -- already correct |

### Plan correction, made during execution

The plan claimed `_triton` "announced device code one tier too high". That was
wrong. `ref/naive.py` already defines `block_representatives`, so the suffix was
doing real disambiguation between the oracle and the kernel -- dropping it
naively would have collided in `tests/test_gate.py`, which imports both by bare
name.

Looking at all three oracle/kernel pairs showed the actual defect: three pairs,
three different rules.

| oracle (`ref/naive.py`) | kernel (`kernels/`) | rule |
|---|---|---|
| `block_representatives` | `block_representatives_triton` | kernel marked |
| `select_blocks` | `gate_select` | unrelated names |
| `moba_attention_naive` | `moba_attention` | oracle marked |

The third is right, and it is the pair that matters most. **The oracle carries
the marker; the kernel gets the bare name** -- the kernel is the product, the
reference is the thing qualified against it. So this task also renames the two
oracle functions, which the plan had listed as untouched:

| old | new |
|---|---|
| `ref.naive.block_representatives` | `pool_blocks_naive` |
| `ref.naive.select_blocks` | `gate_select_naive` |
| `kernels.gate.block_representatives_triton` | `pool_blocks` |

Fixing one pair and leaving two would have left the codebase with two rules
instead of three, which is not obviously better than three.

**Which underscores stay, and why.** The `_` prefix is not being removed from
the codebase; it is being narrowed back to its one Python meaning. After Tasks 1
and 2 the only `_`-prefixed names left are genuine module-private helpers with
no consumer outside their own file:

```
_validate          moba_fwd.py     argument checking for moba_forward
_packed_naive      ref/naive.py    the packed branch of the oracle
_require_packed    varlen.py       shape guard
_blank_row         bench/suite.py  row builder for a measurement that failed
_load              backends/       lazy import of a vendored module
_GROUP_CACHE       moba_fwd.py     memo table
_MODULE_ALIAS      backends/       vendored-module constants
_RELATIVE_PATH     backends/
```

Every one of those is host-side Python that a reader can step through. None of
them is device code. That is the distinction the prefix should carry and
currently does not.

- [x] **Step 1: Apply, with the same count-asserting substitution as Task 1**

```python
RENAMES = {
    "block_representatives_triton": "pool_blocks",
    "_MoBAAttention": "MoBAAttentionFunction",
}
```

- [x] **Step 2: Verify no old name survives**

```bash
grep -rn 'block_representatives_triton\|_MoBAAttention' src tests bench --include=*.py
```

Expected: no output.

- [x] **Step 3: Run the full suite**

Expected: `435 passed, 1 skipped`.

- [x] **Step 4: Commit**

```bash
git add -A
git commit -m "refactor: name the launcher and autograd tiers consistently"
```

---

## Task 3: Split gate.py into pooling.py and gating.py

`gate.py` holds two unrelated kernels for two different pipeline stages, with
stage 1 *below* stage 2, so the file reads backwards relative to the algorithm.

**Files:**
- Create: `src/moba/kernels/pooling.py` (from `gate.py:415-549`)
- Modify: `src/moba/kernels/gate.py` -> renamed to `gating.py`, keeping lines
  1-414
- Modify (importers): `src/moba/kernels/index_fused.py`, `tests/test_gate.py`,
  `tests/test_varlen.py`

**Interfaces:**
- Produces: `moba.kernels.pooling.pool_blocks_kernel`,
  `moba.kernels.pooling.pool_blocks`
- Produces: `moba.kernels.gating.gate_kernel`,
  `moba.kernels.gating.gate_select`, and the `GATE_BLOCK_M` / `GATE_BLOCK_N` /
  `GATE_WARPS` constants, which stay with the gate kernel.
- Consumes: `pooling` imports nothing from `gating`; `gating` imports nothing
  from `pooling`. They are independent stages joined only by the caller. Verify
  this before splitting -- if either half references the other, stop and report.

- [x] **Step 1: Confirm the two halves are independent**

```bash
sed -n '1,414p' src/moba/kernels/gate.py | grep -n 'pool_blocks\|_representatives'
sed -n '415,549p' src/moba/kernels/gate.py | grep -n 'gate_select\|gate_kernel'
```

Expected: no output from either. If there is output, the seam is not where this
plan claims -- stop and report rather than improvising a split.

- [x] **Step 2: Split the file**

```bash
git mv src/moba/kernels/gate.py src/moba/kernels/gating.py
```

Then move the pooling half into a new `src/moba/kernels/pooling.py`, carrying
its imports. Give it a module docstring naming its stage:

```python
"""Stage 1 of MoBA: pool each key block into one representative key.

The gate ranks blocks by scoring a query against these representatives rather
than against every key, which is where MoBA's arithmetic saving begins. See
`gating.py` for the ranking that consumes them.
"""
```

Delete lines 415-549 from `gating.py` and give it the matching docstring opener
naming it stage 2.

- [x] **Step 3: Update the importers**

```bash
grep -rn 'kernels.gate\b\|from .gate import\|from ..kernels.gate' src tests bench --include=*.py
```

Repoint each hit to `gating` or `pooling` by which symbol it uses.

- [x] **Step 4: Run the full suite**

Expected: `435 passed, 1 skipped`.

- [x] **Step 5: Commit**

```bash
git add -A
git commit -m "refactor: split gate.py into pooling.py and gating.py"
```

---

## Task 4: Merge the two index builders into routing.py

`index.py` (the `CSCIndex` dataclass plus a torch builder) and `index_fused.py`
(a Triton builder of the same structure) are one stage with two implementations.
Both are named for the data layout rather than the job.

**Files:**
- Create: `src/moba/kernels/routing.py` from `index.py` + `index_fused.py`
- Delete: `src/moba/kernels/index.py`, `src/moba/kernels/index_fused.py`
- Modify (importers): 13 files import `index`, 12 import `index_fused`;
  enumerate with the grep in Step 1 rather than trusting these counts.

**Interfaces:**
- Produces: `moba.kernels.routing` exporting `CSCIndex`, `build_csc`,
  `csc_to_mask`, `build_csc_from_indices`, `build_csc_fused`, and the
  `INDEX_BLOCK_M` constant plus the three `*_kernel` names from Task 1.
- Every symbol keeps its current name and signature. This task moves code
  between files and changes nothing else.

- [x] **Step 1: Enumerate every importer before moving anything**

```bash
grep -rn 'kernels.index\b\|kernels.index_fused\|from .index import\|from .index_fused import' \
  src tests bench --include=*.py
```

- [x] **Step 2: Build routing.py in stage order**

`git mv src/moba/kernels/index.py src/moba/kernels/routing.py`, then append
`index_fused.py`'s contents, reconciling the imports. Order the result by tier,
not by which file it came from:

```
module docstring: stage 3 prep, selection -> the structure the kernel walks
imports
CSCIndex                      the structure
--- Device code ---
count_columns_kernel
offsets_and_order_kernel
scatter_pairs_kernel
--- Launchers ---
build_csc_from_indices        Triton builder
build_csc_fused               Triton builder, gate included
build_csc                     torch builder, the reference
csc_to_mask                   the reverse, for tests
```

- [x] **Step 3: Repoint every importer from the Step 1 list**

- [x] **Step 4: Verify nothing references the old modules**

```bash
grep -rn 'kernels\.index\b\|kernels\.index_fused\|index_fused' src tests bench --include=*.py
```

Expected: no output.

- [x] **Step 5: Run the full suite**

Expected: `435 passed, 1 skipped`.

- [x] **Step 6: Commit**

```bash
git add -A
git commit -m "refactor: merge index.py and index_fused.py into routing.py"
```

---

## Task 5: Extract tuning.py

`moba_fwd.py` holds the shared-memory and warp-count heuristics for both the
forward and the backward. They are host-side policy, not kernel code, and
`moba_bwd.py` already reaches into `moba_fwd` for them.

**Files:**
- Create: `src/moba/kernels/tuning.py`
- Modify: `src/moba/kernels/moba_fwd.py` (remove the extracted block),
  `src/moba/kernels/moba_bwd.py` (repoint imports)
- Modify (importers): `bench/load_balance.py` imports `choose_group`;
  `tests/test_moba_kernel.py` imports `group_candidates`. Enumerate first.

**Interfaces:**
- Produces: `moba.kernels.tuning` exporting `MAX_GROUP`, `PIPELINE_STAGES`,
  `LOG2_E`, `COMBINE_BLOCK_M`, `COMBINE_ELEMENTS_PER_THREAD`,
  `MIN_COMBINE_WARPS`, `MAX_COMBINE_WARPS`, `MAX_COMBINE_TILE_ELEMENTS`,
  `estimate_shared_memory`, `choose_group`, `group_candidates`,
  `choose_combine_tile`.
- `LOG2_E` moves here and is imported by `moba_fwd.py`. It is currently declared
  *after* the kernel that motivates it, which is why the base-2 softmax argument
  appears twice in that file.
- The `BWD_*` constants stay in `moba_bwd.py`: they are that kernel's tile shape,
  not shared policy. Only move a constant with more than one consumer.

- [x] **Step 1: Enumerate importers**

```bash
grep -rn 'choose_group\|group_candidates\|estimate_shared_memory\|choose_combine_tile\|MAX_GROUP\|PIPELINE_STAGES\|LOG2_E\|COMBINE_' \
  src tests bench --include=*.py
```

- [x] **Step 2: Move the block to tuning.py, carrying its comments verbatim**

The comments on `PIPELINE_STAGES` and `choose_group` are long and are staying
long for now -- the prose pass is a separate job. Move them unchanged.

- [x] **Step 3: Repoint imports and delete the originals from moba_fwd.py**

- [x] **Step 4: Run the full suite**

Expected: `435 passed, 1 skipped`.

- [x] **Step 5: Confirm the tuning still picks the same chunk height**

This is the one task where a mistake could change behaviour rather than fail
loudly: if a constant is dropped or duplicated, `choose_group` returns a
different height and the kernel still runs, just differently.

```bash
uv run pytest tests/test_moba_kernel.py -q -k 'chunk_height or shared' 2>&1 | tail -3
```

Expected: pass. That test asserts the height used fits and the next one up does
not, so a changed constant fails it.

- [x] **Step 6: Commit**

```bash
git add -A
git commit -m "refactor: extract tuning.py from moba_fwd.py"
```

---

## Task 6: Rename the utility modules

**Files:**
- `src/moba/heads.py` -> `src/moba/gqa.py`
- `src/moba/packing.py` -> `src/moba/varlen.py`
- `src/moba/backends/_refs.py` -> `src/moba/backends/layout.py`
- `src/moba/tolerance.py` -> `tests/tolerance.py`

**Interfaces:**
- Every symbol keeps its name. Only module paths change.
- `tolerance.py` is imported by **tests only** -- verified: the two `src/` hits
  are prose mentions inside docstrings, and `bench/` does not import it.
  Re-verify in Step 1 before moving; if `src/` or `bench/` gained a real import
  since this plan was written, leave it in `src/moba` and report.

| old | why it reads badly | new |
|---|---|---|
| `heads.py` | "heads" is every attention file's subject; this one is only about the GQA divisor | `gqa.py` |
| `packing.py` | names the operation, not the feature; the feature is called varlen everywhere in the field | `varlen.py` |
| `_refs.py` | private-marked, and "refs" collides with `references/` and with `ref/naive.py`, which are three different things | `layout.py` |
| `tolerance.py` | test policy shipped inside the package | `tests/tolerance.py` |

- [x] **Step 1: Re-verify tolerance.py has no non-test importer**

```bash
grep -rn 'import tolerance\|from moba.tolerance\|from .tolerance' src bench --include=*.py
```

Expected: no output. Any hit means stop and report.

- [x] **Step 2: Move the files**

```bash
git mv src/moba/heads.py src/moba/gqa.py
git mv src/moba/packing.py src/moba/varlen.py
git mv src/moba/backends/_refs.py src/moba/backends/layout.py
git mv src/moba/tolerance.py tests/tolerance.py
```

- [x] **Step 3: Repoint every importer**

```bash
grep -rn 'moba\.heads\|moba\.packing\|_refs\|moba\.tolerance\|from \.heads\|from \.packing\|from \._refs\|from \.tolerance' \
  src tests bench --include=*.py
```

Repoint each. Tests import tolerance as a sibling module now, so
`from moba.tolerance import ...` becomes `from tolerance import ...` -- confirm
`tests/` is on the path via the pytest config in `pyproject.toml` before relying
on that; if it is not, add `tests` to `pythonpath` there rather than using a
relative import.

- [x] **Step 4: Run the full suite**

Expected: `435 passed, 1 skipped`.

- [x] **Step 5: Commit**

```bash
git add -A
git commit -m "refactor: name the utility modules for what they are"
```

---

## Task 7: Give the package a front door

`src/moba/__init__.py` exports a version string and nothing else;
`kernels/__init__.py` is empty. The only way to call this library is
`from moba.kernels.autograd import moba_attention` -- four levels ending in a
module named after a PyTorch mechanism. NSA exports `parallel_nsa` from its
package root.

**Files:**
- Modify: `src/moba/__init__.py`

**Interfaces:**
- Produces: `from moba import moba_attention, MoBAConfig` as the supported way
  in. Deep paths keep working; nothing is removed.

- [x] **Step 1: Write the exports**

```python
"""MoBA (Mixture of Block Attention) kernels and reference implementations.

The pipeline, in the order the modules run:

    kernels.pooling   mean-pool each key block into one representative key
    kernels.gating    score queries against representatives, take top-k
    kernels.routing   turn that choice into the index the kernel walks
    kernels.moba_fwd  attend, one program per (KV block, batch, kv head)
    kernels.moba_bwd  the backward of the same

`moba_attention` runs all of it and is differentiable.
"""

from moba.config import MoBAConfig
from moba.kernels.autograd import moba_attention

__all__ = ["MoBAConfig", "moba_attention"]
__version__ = "0.1.0"
```

- [x] **Step 2: Verify the front door works and costs nothing at import**

```bash
uv run python -c "import moba; print(moba.__all__); print(moba.moba_attention)"
```

Expected: the two names, no Triton compilation triggered.

- [x] **Step 3: Run the full suite**

Expected: `435 passed, 1 skipped`.

- [x] **Step 4: Commit**

```bash
git add -A
git commit -m "refactor: export moba_attention and MoBAConfig from the package root"
```

---

## Task 8: Order and section the surviving large files

The last structural defect: within a file, tiers are interleaved. Fix by
ordering, and mark the boundaries so the reader can see them.

**Files:**
- Modify: `src/moba/kernels/moba_fwd.py`, `moba_bwd.py`, `gating.py`,
  `routing.py`, `fa2_minimal.py`

**Interfaces:**
- Nothing changes but the order of definitions and the addition of banner
  comments. No signature, name or body is touched.

- [x] **Step 1: Adopt one banner form and one order**

```python
# ---------------------------------------------------------------------------
# Device code
# ---------------------------------------------------------------------------
```

Order within every module: module docstring, imports, module constants, **device
code**, **launchers**, **autograd**, **public entry point**. This is NSA's
order; it is what lets a reader open any file at any line and know which tier
they are in.

- [x] **Step 2: Reorder each file, moving whole definitions only**

Do not edit inside any function. If a definition must move above something it
references at module scope, that is a real dependency -- leave the order and
note it rather than reordering the constant.

- [x] **Step 3: Verify the diff is pure motion**

```bash
git diff | grep -E '^[+-]' | grep -vE '^[+-]{3}' | \
  grep -vE '^[+-]# -+$|^[+-]# (Device code|Launchers|Autograd|Public)' | \
  sed 's/^[+-]//' | sort | uniq -c | awk '$1 % 2 == 1'
```

Expected: no output. Every non-banner line that left appears again, so the counts
pair up. An odd count is a line that was changed, not moved.

- [x] **Step 4: Run the full suite**

Expected: `435 passed, 1 skipped`.

- [x] **Step 5: Commit**

```bash
git add -A
git commit -m "refactor: one tier order and section banners across the kernels"
```

---

## Task 9: Confirm nothing moved but the names

**Files:** none modified.

- [x] **Step 1: Full suite from a cold cache**

```bash
find . -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null
uv run pytest -q -rs 2>&1 | tail -5
```

Expected: `435 passed, 1 skipped`, and the one skip is
`test_optional_backends.py:42: moonshot is available; nothing to check here`.

- [x] **Step 2: Same-session A/B on the stage benchmark**

Renaming a `@triton.jit` function changes its cache key, so the first launch
recompiles. That is a one-off; steady-state timing must not move. Run master and
the branch back to back in one sitting -- a cross-session comparison is not an
A/B, per `docs/layout-measurements.md`.

```bash
uv run python bench/stages.py
```

Expected: within the +/-3% this machine drifts by. Anything larger is a real
finding -- investigate before merging, do not average it away.

- [x] **Step 3: Update the CLAUDE.md status line**

One line, current. Name the new module layout so a fresh session finds it.

- [x] **Step 4: Merge with --no-ff**

```bash
git checkout master
git merge --no-ff structure-refactor
```

`--no-ff` deliberately: this touches most files in the package and should stay
one revertible unit, as the token-major merge did.

---

## Outcome

All tasks executed inline on branch `structure-refactor`, 2026-09-05.
435 passed / 1 skipped at every checkpoint, skip count included.

Same-session A/B on `bench/stages.py`, master against the branch:

| S | dtype | master | branch |
|---|---|---|---|
| 1024 | f32 | 337 us | 338 us |
| 1024 | bf16 | 261 us | 258 us |
| 2048 | f32 | 707 us | 708 us |
| 2048 | bf16 | 532 us | 536 us |
| 4096 | f32 | 1451 us | 1446 us |
| 4096 | bf16 | 1079 us | 1075 us |
| 8192 | f32 | 2980 us | 2958 us |
| 8192 | bf16 | 2198 us | 2178 us |

-1.1% to +0.8%, straddling zero. A rename cost nothing, which is what a rename
should cost.

Three things the plan got wrong, all corrected during execution and recorded
where they happened:

1. The `_triton` suffix was disambiguating the oracle from the kernel, not
   mislabelling a tier. Fixed by giving all three oracle/kernel pairs one rule
   instead of one pair a new one. See the correction block in Task 2.
2. Task 8 budgeted for reordering every module into tier order. Checked first:
   all six were **already** in tier order. The order was right and nothing said
   so, which is what the banners fix. Only `moba_fwd.py`'s host tier genuinely
   moved -- it ran reduce, validate, allocate, attend.
3. `moba_fwd.py` also carried a dead import (`round_up_to_tensor_core_tile`),
   found while splitting out `tuning.py`.

Two test modules keep stale names: `test_heads.py` covers `gqa.py` and
`test_packing.py` covers `varlen.py`. Renaming them would collide with the
existing feature-level `test_gqa.py` and `test_varlen.py`, so they were left
rather than given an awkward scheme. Worth revisiting if those files are
touched for another reason.

---

## Out of scope, on purpose

- **The prose pass.** `src/` is 50% comments and docstrings against 4-5% in the
  references, and much of it duplicates `docs/`. That is the next job and it is
  deliberately not this one: mixing a rename diff with a prose diff makes both
  unreviewable.
- **Any behaviour, tuning constant, or kernel body.**
- **`bench/` structure.** It has the same naming looseness but no consumers
  outside itself.
- **`docs/`.** The 16 existing documents keep their current shape. Only the
  status line changes.
