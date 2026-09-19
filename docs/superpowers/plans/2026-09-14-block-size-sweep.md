# Block-Size Sweep Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task.
> Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure how this kernel behaves as the MoBA block shrinks to 64, 32 and
16, which is the project's stated contribution and the one regime no benchmark in
`bench/` currently covers.

**Architecture:** A new `bench/block_sweep.py` builds two *panels* of
`(block_size, top_k)` gate configurations and hands each to the existing
`bench.suite.run_suite`, which already sweeps shapes x gates, measures every
available backend, and records every refusal with its reason. The module adds
only what `run_suite` has no opinion about: panel construction, a scratch-memory
comparison between the two kernel designs, and report writing. No existing
module changes.

**Tech Stack:** Triton 3.7.1, torch 2.13.0+cu126, Python 3.13, pytest.

**Spec:** No separate spec. The goal statement is `CLAUDE.md`'s ("a clean, tested,
benchmarked Triton port that handles small blocks well"). The rationale for the
two panels and for the scratch-memory column is `docs/traffic-derivation.md` §3
and §5; the reason small blocks should favour this design is
`docs/divergence-measurements.md` Finding 1.

## Global Constraints

- **Google style** for all code and docstrings (Args/Returns/Raises, 4-space
  indent).
- **`src/moba` must not be touched.** This is a benchmark-only change. If a task
  seems to need a kernel change, stop and report it instead.
- **`references/` is vendored and read-only.** Never edit it.
- **Never add Claude/Anthropic as a commit co-author.**
- **Run everything through WSL**, always with this exact prefix:
  `wsl -e bash -lc 'export PATH="$HOME/.local/bin:$PATH"; export UV_PROJECT_ENVIRONMENT="$HOME/.venvs/moba"; cd "/mnt/c/D/Side Projects/GPU Kernel" && uv run <cmd>'`
- **Run benchmarks as modules**: `uv run python -m bench.block_sweep`, never
  `python bench/block_sweep.py`. A script gets only its own directory on the
  path and cannot find top-level `implementations/`.
- **Commit from PowerShell, not WSL** — git identity is set on Windows only.
- **Baseline test count before starting:** run the suite once and record it. Every
  later checkpoint must match that number plus the tests this plan adds, and the
  **skip count must not rise** — this project has twice shipped tests that
  silently stopped asserting.
- The laptop is a 4 GB RTX 3050 Ti. Tasks 1-4 run there. Task 5 needs the rented
  A100 and is the only task that costs money.

---

## Background an implementer needs

**What MoBA's two gate numbers mean.** `block_size` is how many consecutive keys
form one block. `top_k` is how many blocks each query attends. So a query attends
`top_k * block_size` keys, and the sequence is divided into
`num_blocks = S / block_size` blocks.

**Why smaller blocks are the interesting case.** `docs/divergence-measurements.md`
Finding 1 measures that a query-major kernel's utilization is governed by
`num_blocks`: 0.432 at 32 blocks, 0.121 at 128. Shrinking `block_size` at fixed
`S` is exactly how `num_blocks` grows, so the penalty this kernel's loop order
avoids gets worse precisely as blocks shrink. That is the claim this sweep tests.

> **Outcome, 2026-09-20: the claim is false as stated, and the sweep is what
> showed it.** `num_blocks` does not predict the result on its own -- the knob
> that produced it does. Reached by lengthening the sequence, eight times the
> blocks makes this kernel faster; reached by shrinking the block at a fixed
> key budget, the same eight times makes it 29.5x slower. The sweep's two
> panels isolate why: they differ only in whether `top_k` rises, and the
> penalty is 29.52x when it does against 2.05x when it does not. This kernel's
> scratch is O(S * top_k) because each selected block is handled by a separate
> program that must write a partial softmax for a second pass to merge, so its
> cost is counted per (query, block) pair rather than per key. Long context
> adds pairs and arithmetic together; shrinking the block adds pairs alone.
> Full result and tables in `docs/block-size-results.md`. Nothing below needed
> changing to reach this -- the plan's experiment was right, its stated
> expectation was not.

**Why FlashMoBA will be absent from the two smallest rows.** It requires its block
dimension to be a multiple of 64 (`flash_moba_api.cpp:465`,
`TORCH_CHECK(n_lg_block_dim % 64 == 0)`), and `implementations/flash_moba.py:52-56`
declares that as an availability reason. It cannot run 32 or 16. **This is the
result, not a hole in the table** — `run_suite` emits those as rows with
`status="skipped"` and the reason string, and the report must keep them.

**Our own floor is 16.** `implementations/ours.py:74` rejects
`block_size < TENSOR_CORE_TILE` (16), because Ampere's MMA atom is 16x8x16 and a
narrower tile cannot fill a tensor core. So 16 is the smallest meaningful row, and
if the kernel loses there, that is a genuine finding about the contribution's
limit — report it, do not tune around it.

**Correctness is already covered.** `tests/` uses `block_size=32` in 55 places and
16 in 12. This sweep measures speed only; it must not re-verify numerics.

**Known risks, to report rather than fix:**
- At `(block_size=16, top_k=64)` the partials buffer has 64 slots, and
  `moba_combine_kernel` unrolls `tl.static_range(K_SLOTS)`. Compile time or
  shared memory may blow up. `run_suite` catches `RuntimeError` per backend and
  records `status="error"`, so a failure becomes a row. Let it.
- `run_suite`'s memory gate uses the *oracle's* O(S^2) estimate and skips **every**
  backend for a config that exceeds it (`bench/suite.py:468-479`). On the laptop
  this bites above S=8192. Keep sequence lengths short there.

---

## File Structure

| file | responsibility |
|---|---|
| `bench/block_sweep.py` | **create.** Panel construction, scratch-memory comparison, entry point. |
| `tests/test_bench.py` | **modify.** Append tests for the panel and scratch-memory functions (pure logic; the timing path is not unit-testable). |
| `results/block-sweep-budget.csv`, `results/block-sweep-fixed-top-k.csv` | **create** by running. |
| `docs/block-size-results.md` | **create** in Task 5, from the A100 run. |

---

### Task 1: The two panels

**Files:**
- Create: `bench/block_sweep.py`
- Test: `tests/test_bench.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `BLOCK_SIZES: tuple[int, ...]`, `BUDGET: int`, `FIXED_TOP_K: int`,
  `budget_panel(budget: int = BUDGET, block_sizes: tuple[int, ...] = BLOCK_SIZES) -> list[tuple[int, int]]`,
  `fixed_top_k_panel(top_k: int = FIXED_TOP_K, block_sizes: tuple[int, ...] = BLOCK_SIZES) -> list[tuple[int, int]]`.
  Both return `(block_size, top_k)` pairs in the order `run_suite` takes as `gates`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_bench.py`, and add
`from bench.block_sweep import budget_panel, fixed_top_k_panel` to its imports:

```python
# --------------------------------------------------------------------------
# Block-size sweep panels
# --------------------------------------------------------------------------


def test_budget_panel_holds_attended_keys_constant():
    """Every row attends top_k * block_size = 1024 keys, so only granularity
    changes. This is what makes the panel a comparison rather than a sweep of
    four unrelated workloads."""
    assert budget_panel() == [(128, 8), (64, 16), (32, 32), (16, 64)]
    assert all(bk * tk == 1024 for bk, tk in budget_panel())


def test_budget_panel_rejects_an_indivisible_budget():
    with pytest.raises(ValueError):
        budget_panel(budget=1000, block_sizes=(128,))


def test_fixed_top_k_panel_keeps_top_k_and_varies_sparsity():
    panel = fixed_top_k_panel()
    assert panel == [(128, 8), (64, 8), (32, 8), (16, 8)]
    cfgs = [MoBAConfig(block_size=bk, top_k=tk) for bk, tk in panel]
    ratios = [sparsity_ratio(4096, cfg) for cfg in cfgs]
    assert ratios == sorted(ratios, reverse=True)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `wsl -e bash -lc 'export PATH="$HOME/.local/bin:$PATH"; export UV_PROJECT_ENVIRONMENT="$HOME/.venvs/moba"; cd "/mnt/c/D/Side Projects/GPU Kernel" && uv run pytest tests/test_bench.py -k panel -v'`

Expected: FAIL, `ModuleNotFoundError: No module named 'bench.block_sweep'`.

- [ ] **Step 3: Write the module docstring and the panels**

Create `bench/block_sweep.py`:

```python
"""How the kernel behaves as the MoBA block shrinks -- the project's claim.

Every other benchmark in this directory runs `block_size=128`, which is
FlashMoBA's published configuration and the largest block this project cares
about. The stated contribution is the opposite end: a Triton port that still
works when blocks are small. Nothing measured that until this module.

TWO PANELS, because "smaller blocks" is two different experiments.

* `budget_panel` holds `top_k * block_size` at 1024, so every row attends the
  same number of keys and only granularity changes. This is the comparison
  FlashMoBA's own theory argues for -- finer blocks at equal budget should
  retrieve better -- and it is the headline.
* `fixed_top_k_panel` holds `top_k` at 8, so sparsity rises as blocks shrink
  and each row does strictly less work. Those rows get faster for reasons that
  have nothing to do with the kernel, which is why they are reported second.

WHAT THE MISSING ROWS ARE FOR. FlashMoBA requires its block dimension to be a
multiple of 64 (`flash_moba_api.cpp:465`), so it cannot run the two smallest
rows at all, and `implementations/flash_moba.py` reports that as an
availability reason. `run_suite` turns it into a row carrying the reason rather
than a gap. Those rows are the evidence for where the contribution lies; a
report that drops them is worse than no report, because it looks complete.
"""

import torch

from bench.report import to_csv, to_markdown
from bench.suite import BYTES_PER_MIB, default_seq_lens, run_suite

# Largest first, matching every other sweep in this directory. 16 is the floor:
# Ampere's MMA atom is 16x8x16, so a narrower tile cannot fill a tensor core,
# and `implementations/ours.py` rejects anything below it.
BLOCK_SIZES = (128, 64, 32, 16)

# Keys each query attends in the budget panel. 1024 because that is
# `block_size=128, top_k=8`, the configuration every existing result in this
# repository uses, so the first row of the panel is comparable with them.
BUDGET = 1024

FIXED_TOP_K = 8


def budget_panel(budget: int = BUDGET,
                 block_sizes: tuple[int, ...] = BLOCK_SIZES
                 ) -> list[tuple[int, int]]:
    """Gate configs holding `top_k * block_size` constant.

    Args:
        budget: Keys each query attends, `top_k * block_size`.
        block_sizes: Block sizes to cover, largest first.

    Returns:
        List of `(block_size, top_k)` pairs, in `run_suite`'s `gates` order.

    Raises:
        ValueError: If `budget` is not divisible by one of `block_sizes`, which
            would silently change the attended budget between rows and make the
            panel incomparable.
    """
    pairs = []
    for block_size in block_sizes:
        if budget % block_size:
            raise ValueError(
                f"budget {budget} is not a multiple of block_size "
                f"{block_size}; the panel would not hold the budget constant"
            )
        pairs.append((block_size, budget // block_size))
    return pairs


def fixed_top_k_panel(top_k: int = FIXED_TOP_K,
                      block_sizes: tuple[int, ...] = BLOCK_SIZES
                      ) -> list[tuple[int, int]]:
    """Gate configs holding `top_k` constant, so sparsity rises as blocks shrink.

    Args:
        top_k: Blocks each query attends, the same in every row.
        block_sizes: Block sizes to cover, largest first.

    Returns:
        List of `(block_size, top_k)` pairs, in `run_suite`'s `gates` order.
    """
    return [(block_size, top_k) for block_size in block_sizes]
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `wsl -e bash -lc 'export PATH="$HOME/.local/bin:$PATH"; export UV_PROJECT_ENVIRONMENT="$HOME/.venvs/moba"; cd "/mnt/c/D/Side Projects/GPU Kernel" && uv run pytest tests/test_bench.py -k panel -v'`

Expected: 3 passed.

- [ ] **Step 5: Commit** (from PowerShell)

```bash
git add bench/block_sweep.py tests/test_bench.py
git commit -m "bench: two panels for a block-size sweep"
```

---

### Task 2: Scratch memory, both designs

Why this belongs in the same table as the speed numbers: in the budget panel,
`top_k` rises as `block_size` falls, and our scratch buffer is proportional to
`top_k`. The design wins on bytes moved and pays in bytes held
(`docs/traffic-derivation.md` §3). A sweep that reports only latency hides the
cost that shows up exactly in the regime being claimed.

**Files:**
- Modify: `bench/block_sweep.py`
- Test: `tests/test_bench.py`

**Interfaces:**
- Consumes: `BYTES_PER_MIB` from `bench.suite`.
- Produces:
  `partials_mb(batch: int, heads: int, seq_len: int, head_dim: int, top_k: int, dtype: torch.dtype = torch.bfloat16) -> float`
  and `running_output_mb(batch: int, heads: int, seq_len: int, head_dim: int) -> float`,
  both returning MiB.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_bench.py`, extending the `bench.block_sweep` import with
`partials_mb, running_output_mb`:

```python
def test_partials_scale_linearly_with_top_k():
    """Our scratch is one slot per (query, selected block), so doubling the
    blocks a query selects doubles it exactly."""
    shape = (1, 4, 8192, 64)
    assert partials_mb(*shape, top_k=16) == pytest.approx(
        2 * partials_mb(*shape, top_k=8))


def test_running_output_does_not_depend_on_top_k():
    """A query-group design keeps one output row per query however many blocks
    it selects, which is the trade docs/traffic-derivation.md section 3 prices."""
    shape = (1, 4, 8192, 64)
    assert running_output_mb(*shape) == pytest.approx(
        running_output_mb(*shape))


def test_our_scratch_dominates_at_flashmobas_published_shape():
    """At their benchmark's longest row -- batch 2, 16 heads, 512K tokens,
    head_dim 128 -- our design needs several times their scratch. This is the
    reason their kernel is built the way it is, and it belongs in a test so it
    cannot quietly stop being true."""
    shape = (2, 16, 524288, 128)
    ours = partials_mb(*shape, top_k=8)
    theirs = running_output_mb(*shape)
    assert ours > 4 * theirs
    assert ours / 1024 == pytest.approx(33.0, abs=0.5)     # GiB
    assert theirs / 1024 == pytest.approx(8.0, abs=0.5)    # GiB
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `wsl -e bash -lc 'export PATH="$HOME/.local/bin:$PATH"; export UV_PROJECT_ENVIRONMENT="$HOME/.venvs/moba"; cd "/mnt/c/D/Side Projects/GPU Kernel" && uv run pytest tests/test_bench.py -k "partials or running_output or scratch" -v'`

Expected: FAIL with `ImportError: cannot import name 'partials_mb'`.

- [ ] **Step 3: Implement both functions**

Append to `bench/block_sweep.py`:

```python
def partials_mb(batch: int, heads: int, seq_len: int, head_dim: int,
                top_k: int, dtype: torch.dtype = torch.bfloat16) -> float:
    """Scratch HBM this kernel's design needs, in MiB.

    One slot per (query, selected block): `part_acc` is `[B, H, S, top_k, D]`
    at the compute dtype, and `part_m` and `part_l` are `[B, H, S, top_k]`
    float32 each (`moba.kernels.moba_fwd.empty_partials`).

    Args:
        batch: Batch size.
        heads: Number of query heads.
        seq_len: Sequence length.
        head_dim: Head dimension.
        top_k: Blocks each query selects.
        dtype: Storage dtype of `part_acc`.

    Returns:
        Scratch requirement in MiB.
    """
    itemsize = torch.tensor([], dtype=dtype).element_size()
    slots = batch * heads * seq_len * top_k
    return (slots * head_dim * itemsize + 2 * slots * 4) / BYTES_PER_MIB


def running_output_mb(batch: int, heads: int, seq_len: int,
                      head_dim: int) -> float:
    """Scratch a query-group design needs, in MiB.

    FlashMoBA keeps one float32 output row per query and updates it in place
    (`o_tmp`, `[N, H, D]` float32, `flash_moba_api.cpp:505`), so its scratch does
    not grow with `top_k`. Reported beside `partials_mb` because that difference
    is the trade this project makes, not a detail.

    Args:
        batch: Batch size.
        heads: Number of query heads.
        seq_len: Sequence length.
        head_dim: Head dimension.

    Returns:
        Scratch requirement in MiB.
    """
    return batch * seq_len * heads * head_dim * 4 / BYTES_PER_MIB
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `wsl -e bash -lc 'export PATH="$HOME/.local/bin:$PATH"; export UV_PROJECT_ENVIRONMENT="$HOME/.venvs/moba"; cd "/mnt/c/D/Side Projects/GPU Kernel" && uv run pytest tests/test_bench.py -k "partials or running_output or scratch" -v'`

Expected: 3 passed. If the GiB assertions fail, do **not** loosen them — the
formula is wrong; check `empty_partials` in `src/moba/kernels/moba_fwd.py:503`.

- [ ] **Step 5: Commit** (from PowerShell)

```bash
git add bench/block_sweep.py tests/test_bench.py
git commit -m "bench: price the scratch both MoBA designs need"
```

---

### Task 3: The entry point

**Files:**
- Modify: `bench/block_sweep.py`

**Interfaces:**
- Consumes: `budget_panel`, `fixed_top_k_panel`, `partials_mb`,
  `running_output_mb` from Tasks 1-2; `run_suite`, `default_seq_lens` from
  `bench.suite`; `to_csv`, `to_markdown` from `bench.report`.
- Produces: `scratch_table(shapes, gates) -> str`, `main() -> None`, and the
  files `results/block-sweep-budget.csv` and
  `results/block-sweep-fixed-top-k.csv`.

- [ ] **Step 1: Write the runner**

Append to `bench/block_sweep.py`:

```python
# One batch, four heads, head_dim 64 -- the shape every other result in this
# repository uses, so the block_size=128 row lands beside them without a
# translation note.
BATCH, HEADS, HEAD_DIM = 1, 4, 64

PANELS = (("budget", budget_panel), ("fixed-top-k", fixed_top_k_panel))


def scratch_table(shapes: list[tuple[int, int, int, int]],
                  gates: list[tuple[int, int]]) -> str:
    """Renders scratch requirements for both designs over the panel.

    Args:
        shapes: List of `(batch, seq_len, heads, head_dim)`.
        gates: List of `(block_size, top_k)`.

    Returns:
        A markdown table, one row per (shape, gate).
    """
    lines = ["", "Scratch memory (MiB)", "",
             "| S | block | top_k | ours | query-group design | ratio |",
             "|---:|---:|---:|---:|---:|---:|"]
    for batch, seq_len, heads, head_dim in shapes:
        theirs = running_output_mb(batch, heads, seq_len, head_dim)
        for block_size, top_k in gates:
            ours = partials_mb(batch, heads, seq_len, head_dim, top_k)
            lines.append(
                f"| {seq_len} | {block_size} | {top_k} | {ours:.1f} | "
                f"{theirs:.1f} | {ours / theirs:.1f}x |"
            )
    return "\n".join(lines)


def main() -> None:
    """Runs both panels and writes a CSV and a markdown table for each."""
    from pathlib import Path

    seq_lens = default_seq_lens(longest=32768)
    shapes = [(BATCH, s, HEADS, HEAD_DIM) for s in seq_lens]
    Path("results").mkdir(exist_ok=True)
    print(f"sequence lengths: {seq_lens}")

    for name, build in PANELS:
        gates = build()
        print(f"\n{'=' * 72}\npanel: {name}   gates: {gates}\n{'=' * 72}")
        rows = run_suite(shapes, gates)
        to_csv(rows, f"results/block-sweep-{name}.csv")
        print(to_markdown(rows))
        print(scratch_table(shapes, gates))


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke-run it on the shortest sequences only**

Run, with the sweep pinned short so this finishes in a couple of minutes:

```
wsl -e bash -lc 'export PATH="$HOME/.local/bin:$PATH"; export UV_PROJECT_ENVIRONMENT="$HOME/.venvs/moba"; cd "/mnt/c/D/Side Projects/GPU Kernel" && uv run python -c "
from bench import block_sweep as bs
rows = bs.run_suite([(1, 1024, 4, 64)], bs.budget_panel())
from bench.report import to_markdown
print(to_markdown(rows))
"'
```

Expected: a table with one row per (gate, backend). Confirm three things, and
report rather than fix if any is wrong:
1. `ours` produces a latency at all four block sizes, including 16.
2. `flash_moba` appears at 128 and 64, and at 32 and 16 appears with
   `status=skipped` and a reason mentioning "multiple of 64".
3. No exception escaped; a backend that failed shows as an `error` row.

- [ ] **Step 3: Commit** (from PowerShell)

```bash
git add bench/block_sweep.py
git commit -m "bench: run the block-size panels and report scratch beside them"
```

---

### Task 4: Laptop run, and what it is allowed to conclude

**Files:**
- Create: `results/block-sweep-budget.csv`, `results/block-sweep-fixed-top-k.csv`

- [ ] **Step 1: Run the full sweep on the laptop**

```
wsl -e bash -lc 'export PATH="$HOME/.local/bin:$PATH"; export UV_PROJECT_ENVIRONMENT="$HOME/.venvs/moba"; cd "/mnt/c/D/Side Projects/GPU Kernel" && uv run python -m bench.block_sweep 2>&1 | tee results/block-sweep-laptop.log'
```

- [ ] **Step 2: Sanity-check the output before trusting any of it**

- `(16, 64)` has 64 partial slots. If it errors on compile time or shared
  memory, that row must say so — check it is an `error` row with a message, not
  a silent gap.
- The laptop is 20 SMs and 4 GB. Absolute latencies here are **not** results;
  `CLAUDE.md` says this box is for correctness, not for benchmark numbers. The
  only conclusions this task may draw are: every config runs, refusals are
  recorded correctly, and the CSV is well-formed.

- [ ] **Step 3: Commit** (from PowerShell)

```bash
git add results/block-sweep-budget.csv results/block-sweep-fixed-top-k.csv results/block-sweep-laptop.log
git commit -m "bench: laptop block-size sweep, shakedown only"
```

---

### Task 5: A100 run and write-up

This is the only task that costs money. Resume the JarvisLabs instance first;
pausing it afterwards stops GPU billing and leaves only storage
(~$0.10/GB-month), so pause rather than delete.

**Files:**
- Create: `docs/block-size-results.md`
- Modify: `CLAUDE.md` (the one-line status)

- [ ] **Step 1: Run both panels on the A100**

```
wsl -e bash -lc '... && uv run python -m bench.block_sweep 2>&1 | tee results/a100-block-sweep.log'
```

Run it on the pass with flash-attn >= 2.7 installed, so `triton_moba` is live;
`moonshot_efficient` needs 2.6.x and will be absent. Note which pass in the doc
— `docs/a100-results.md` "The two passes" explains why the two cannot coexist.

- [ ] **Step 2: Write `docs/block-size-results.md`**

Follow `docs/a100-results.md`'s structure: date, device, config, source command,
raw log path, then the tables, then caveats. It must contain, in this order:

1. The budget panel table, all backends, with the skipped rows and their reasons
   kept.
2. The fixed-top_k panel table, second, labelled as the weaker comparison
   because sparsity changes between rows.
3. The scratch-memory table, and one sentence noting that our requirement grows
   as blocks shrink in the budget panel while a query-group design's does not.
4. A statement of where FlashMoBA stops (block 32) and why, citing
   `flash_moba_api.cpp:465`.
5. A caveat section carrying forward the two open items from
   `docs/traffic-derivation.md` §10: we have not run FlashMoBA's own benchmark
   shape, and our measured FlashMoBA-vs-FA-2 ratio (1.22x) is below their
   published 2.0x, so any claim against them stays shape-qualified.

- [ ] **Step 3: Update the status line in `CLAUDE.md`**

Replace the "Open:" clause so it names what this sweep settled and what it did
not. Keep it to one line, as that section requires.

- [ ] **Step 4: Commit** (from PowerShell)

```bash
git add docs/block-size-results.md results/a100-block-sweep.log CLAUDE.md
git commit -m "docs: the kernel at block sizes 128 down to 16, on an A100"
```

- [ ] **Step 5: Pause the A100 instance.**

---

## Self-review notes

- **Spec coverage:** the goal names "benchmarked" and "small blocks"; Tasks 1-3
  build the benchmark, Task 5 produces the numbers. "Clean" and "tested" are
  already satisfied and are explicitly out of scope here.
- **Not covered by design:** correctness at small blocks (already in `tests/`),
  any kernel change (forbidden by the constraints), and the FlashMoBA-vs-FA-2
  discrepancy, which is carried forward as a caveat rather than resolved.
- **Type consistency:** `budget_panel` and `fixed_top_k_panel` both return
  `list[tuple[int, int]]` ordered `(block_size, top_k)`, matching `run_suite`'s
  `gates` parameter; `partials_mb` and `running_output_mb` both return MiB and
  are used only by `scratch_table`.
