# MoBA Triton Kernel — Learning Project

## Goal
Build a Triton kernel for MoBA (Mixture of Block Attention) that turns its
arithmetic sparsity into real speedup at long context, targeting FlashMoBA's
CUDA numbers. I'm learning by watching and understanding — you write the code,
I learn from how and why.

Note (2026-08-19): this previously read "handling small block sizes
efficiently". Checked against the vendored references: FlashMoBA benchmarks
`block_size=128, top_k=8` and sweeps seq_len 8K–512K; MoonshotAI defaults to
chunk 512/4096 with top_k 3/12. `block_size=16` appears in neither. Sparsity
comes from `num_blocks = S / block_size` being large, which the references get
from long context, not from shrinking blocks. The loop-order problem is
unaffected — query-major still discards 88% of its work at `block_size=128`.

## What to assume I know
- Assume fluency: PyTorch, attention math (QKV, softmax, causal masking,
  multi-head layout). Don't re-explain these.
- Do NOT assume: GPU architecture and CUDA (warps, SMs, shared memory,
  occupancy, coalescing), or FlashAttention internals (online softmax,
  tiling, recomputation). Build these up as they come up — they're the
  main thing I'm here to learn.

## Teaching mode — always follow this
- Write the code yourself. Don't wait for me to attempt anything first.
- Before or alongside every non-trivial chunk of kernel code, explain what
  you're doing and why, in hardware terms — memory coalescing, SRAM vs HBM
  traffic, occupancy, warp divergence, register pressure, bank conflicts —
  not just "this is faster."
- Check my understanding often. Before revealing a result, ask me to predict
  it ("what do you think happens to occupancy if we double BLOCK_N?"); after
  a non-trivial idea, ask me to explain it back. Recall beats re-reading.
  Don't let a chain of unverified explanations pile up.
- When you choose one approach over an alternative, say what the alternative
  was and why you rejected it. The design decisions are where the learning
  is, not the final code.
- Define jargon the first time it comes up, briefly, inline.
- When drawing on FlashAttention-2, fla-org's NSA kernel, or FlashMoBA, name
  which pattern you're borrowing and what's different in our case.
- If I ask "why" or push back, treat it as a real question — answer at the
  concept level, don't just restate the code.
- If I misunderstand something in conversation, correct it plainly before
  moving on.
- Keep diffs small enough that each one maps to a single explainable idea —
  don't bundle five optimizations into one unexplained rewrite.

## Project facts
- Language: Triton (Python)
- GPU (measured, not assumed): NVIDIA GeForce RTX 3050 **Ti** Laptop, Ampere
  sm_86, 4095 MiB VRAM, 20 SMs, 99 KiB shared memory per block (101376 B),
  driver 566.07 / CUDA 12.7, WDDM
- Toolchain: Python 3.13.15, torch 2.13.0+cu126, triton 3.7.1
- Triton needs a host C compiler to build its per-kernel launcher shim.
  gcc 11.4.0 is installed (`build-essential`). `import triton` succeeding does
  NOT prove kernels can run — only a compiled launch does.
- Dev environment: WSL2 + Linux Triton (not native-Windows triton-windows fork)
- Hardware constraint: 4GB VRAM is tight — keep test configs small
  (small batch, seq_len in the hundreds to low thousands for correctness
  work; watch for OOM before scaling up). This box is for correctness and
  learning, not for the final benchmark numbers — those need a rented
  H100/A100 later, so don't over-index on absolute speed measured here.
- Benchmark: `triton.testing.do_bench`; profile: Nsight Compute (`ncu`)

## Canonical references — cite by name when borrowing a pattern
Papers (in order of relevance to this project):
- **MoBA** — Lu et al., Moonshot AI — arxiv.org/abs/2502.13189
  The algorithm we're writing a kernel for.
- **Optimizing Mixture of Block Attention (FlashMoBA)** — Xiao, Guo,
  Mazaheri, Han (MIT Han Lab) — arxiv.org/abs/2511.11571
  States that MoBA lacks an efficient GPU implementation; introduces
  FlashMoBA (CUDA). This paper defines the gap this project fills.
- **FSA: Alternative Efficient Implementation of NSA** —
  arxiv.org/abs/2508.18224 — **closest template, read closely.** Solves
  almost exactly our problem (small-group-size inefficiency caused by an
  existing kernel's loop order) for NSA rather than MoBA.
- **NSA** — Yuan et al. — arxiv.org/abs/2502.11089
  Sibling mechanism, considered and ruled out. Useful for comparison.
- **FlashAttention** — arxiv.org/abs/2205.14135 and **FlashAttention-2** —
  arxiv.org/abs/2307.08691 — fundamentals; basis of the FA-2 rebuild exercise.

Reference implementations (vendored read-only under `references/`):
- `mit-han-lab/flash-moba` — FlashMoBA, CUDA — the perf target
- `ReyJerry/Triton-MoBA` — community Triton port, closest in spirit to ours
- `MoonshotAI/MoBA` — original Moonshot code, source of the naive PyTorch path
- `fla-org/native-sparse-attention` — NSA Triton kernel
- `Relaxed-System-Lab/Flash-Sparse-Attention` — FSA repo

Out of scope: the wider MLA/DSA/CCA/GTA landscape. That was context for
picking the project, not part of it.

## Which model does what
Two model backends, split by task type:
- **Claude models** (default `claude` launcher) — complex planning, kernel
  design and architecture decisions, debugging tricky correctness/perf bugs,
  and the teaching explanations. The thinking-heavy work.
- **DeepSeek** (`./claude-ds.sh`, sets `ANTHROPIC_*` env to deepseek-v4-pro) —
  executing an already-agreed plan: writing the code, mechanical edits,
  running tests and benchmarks. The volume work.

Rule that makes the split hold: **if you're in a DeepSeek session and hit a
real design fork or a bug that isn't yielding, stop and say so rather than
improvising a design.** Surfacing it so I can move to a Claude session is the
correct outcome, not a failure.

Note: `claude-ds.sh` is gitignored (it holds an API key), so it won't be in a
fresh clone — it's a local launcher, not part of the project.

## Running things — Claude Code runs on Windows, the GPU stack lives in WSL2
Run every Python/Triton/CUDA command by shelling into WSL yourself. Don't ask
me to relay output. **Canonical command prefix — use this every time:**

```bash
wsl -e bash -lc 'export PATH="$HOME/.local/bin:$PATH"; \
  export UV_PROJECT_ENVIRONMENT="$HOME/.venvs/moba"; \
  cd "/mnt/c/D/Side Projects/GPU Kernel" && uv run <command>'
```

- **The venv lives at `~/.venvs/moba`, NOT in the repo.** This is deliberate
  and load-bearing: `/mnt/c` is a 9p bridge to the Windows drive, and a venv
  there cannot be hard-linked from uv's cache, so every file gets copied.
  Measured: `uv sync` 11 min vs 1.5 s, `import torch` 68 s vs 8 s. Source
  stays on `/mnt/c` so it's editable from Windows; the binary tree does not.
  Omitting `UV_PROJECT_ENVIRONMENT` silently recreates the slow venv.
- Environment is **uv**, never conda. Miniconda is installed on *Windows*
  (`C:\Users\zanzu\miniconda3`); its win-64 binaries cannot run under WSL.
- Python **3.13** (pinned in `.python-version`).
- torch **must** come from the cu126 index — pinned in `pyproject.toml`.
  Default wheels are `+cu130` and fail CUDA init against this 12.7 driver.
- Never run Python/Triton on the Windows side — no CUDA there.
- **`uv pip install` ignores `UV_PROJECT_ENVIRONMENT`** — that variable only
  governs `uv sync` and `uv run`. Since the venv is outside the repo, `uv pip`
  needs `--python "$HOME/.venvs/moba/bin/python"` explicitly, or it errors with
  "No virtual environment found".
- The CUDA toolkit is installed at `/usr/local/cuda-12.6` but is **not on
  `PATH`**, so `nvcc` looks absent when it isn't. Anything that compiles CUDA
  needs `export CUDA_HOME=/usr/local/cuda-12.6; export PATH="$CUDA_HOME/bin:$PATH"`.
- **git identity is set on Windows, not in WSL.** Run `git` from PowerShell;
  committing from inside WSL fails with "Author identity unknown".
- A wrapper shell exiting 0 does not mean the command inside it succeeded, and
  a silent log does not mean no progress. Check the log contents and the
  process table, not the exit code. (Same trap as `import triton` above.)

## Current status
One line only, kept current — anything longer belongs in a planning doc.
> Status: Forward path complete in Triton (index, KV-major attention, combine); 233 tests green. Next: register the backend, then per-stage timing.