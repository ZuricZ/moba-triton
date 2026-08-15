# Session digest — 2026-08-15: foundations planning

Compacted record of the first working session. This is a decision log, not a
transcript: it keeps *why* things are the way they are, so a cold session does
not re-derive or re-litigate them.

## Where the project stands

Empty repo turned into: git initialised, CLAUDE.md rewritten as a guidelines
document, a design spec and a 10-task implementation plan committed, and a
verified WSL toolchain. No project code written yet. Task 1 of the plan is
next.

## Verified environment facts

Measured on this machine, not assumed:

| Fact | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 3050 **Ti** Laptop |
| VRAM | 4095 MiB |
| SMs | 20 |
| Compute capability | sm_86 (Ampere) |
| Shared memory / block (opt-in) | 101376 B (99 KiB) |
| Shared memory / SM | 100 KiB |
| WSL kernel | 6.18.33.2-microsoft-standard-WSL2 |
| Driver CUDA version | 12.7 |
| Working stack | Python 3.13.15, torch 2.13.0+cu126, triton 3.7.1 |

Note the GPU is a 3050 **Ti**, not a plain 3050 as originally recorded.

## Decisions and why

**Spec-first oracle (D1).** We write our own naive PyTorch MoBA and treat it as
ground truth. Reference implementations are third parties whose disagreements
are findings, not failures. Reason: you cannot debug a kernel against an oracle
you do not understand, and it decouples us from a `flash-attn==2.6.3` build
that may never work locally.

**bf16 primary, sm_80+ only (D2).** Target is modern LLM hardware. The local
3050 Ti is sm_86 Ampere — same architecture family as an A100, differing in
capacity (4 GB vs 80 GB, 20 vs 108 SMs, 99 vs 164 KiB shared memory) not
capability. It has bf16 tensor cores and `cp.async`.

**Kaggle rejected.** Kaggle offers P100 (sm_60) or T4 x2 (sm_75), ~30 GPU-h per
week. Both are *older* architectures than the local laptop: T4 has no bf16 and
only 64 KiB shared memory per SM; P100 has no tensor cores at all. Kaggle would
have been the sole reason to constrain the design to fp16, and it is not worth
it. Final benchmark push goes to a rented A100/H100 instead; vendor undecided.

**Forward-only, designed for backward (D3).** Oracle stays autograd-friendly so
gradient tests can be added as a fourth tier without redesign.

**Pluggable backends with availability gating (D4).** One `Backend` protocol;
`is_available()` returns a *reason string*, so a skipped backend appears in
reports as an explicit row rather than vanishing. A silently short benchmark
table is worse than no table.

**Relative tolerance policy (D5).** Tolerance is defined against a same-dtype
oracle's error versus fp64, times a factor of 2.0 — the FlashAttention paper's
methodology. Absolute tolerances are magic numbers tuned until tests pass.

**No FA-2 rebuild milestone.** FA-2 machinery (online softmax, tiling,
recomputation) is explained inline as it appears in the MoBA kernel rather than
built first as a separate exercise.

**uv, not conda, in WSL.** See "Toolchain" below.

## Toolchain: how we landed on uv + Python 3.13 + cu126

Three findings, each empirical:

1. **Miniconda is installed on Windows** (`C:\Users\zanzu\miniconda3`, conda
   24.3.0, 11 envs) — *not* in WSL. A Windows conda env cannot serve this
   project: WSL runs a Linux kernel and loads ELF binaries, while conda envs
   contain platform-specific builds (`win-64` vs `linux-64`). Reaching across
   via WSL interop would launch a *Windows* Python process, requiring the
   `triton-windows` fork this project rules out. Only an env *specification*
   can be shared across the boundary, never the binaries.
2. **uv beats both conda and venv here.** It manages the interpreter itself, so
   no second Python distribution is needed, and `uv.lock` pins the exact
   resolved graph — which matters because this project has a rented-H100 phase
   where reproducing the env must be one command on a machine billed by the
   hour. Conda's advantages are non-Python binaries (MKL, CUDA toolkits) that
   this project gets from the driver and the torch wheel instead.
3. **Default torch wheels are CUDA 13 and do not run here.** `torch 2.13.0+cu130`
   installs cleanly, then fails at CUDA init with "driver too old (found version
   12070)". CUDA guarantees *minor* version compatibility — any 12.x runtime
   works on any driver supporting 12.0+ — but crossing the 12→13 *major*
   boundary breaks it. **Pin the cu126 index.** This is independent of Python
   version; 3.13 and 3.14 both hit it.

Python 3.14.7 was verified to resolve torch and triton successfully. 3.13.15
was chosen for maturity; 3.15.0rc1 is a release candidate and excluded.

## Reference landscape

Five repos, all verified to exist, vendored read-only under `references/`
(gitignored). Canonical list with papers lives in CLAUDE.md.

`mit-han-lab/flash-moba` is the performance target. Its `setup.py` defaults to
CUDA archs `80;90;100;120` — **sm_86 is absent**, and even when added its
shared-memory tiling may exceed sm_86's 99 KiB and fail at *launch* rather than
compile. The digest depends on reading that source, not running it.

Both Python references pin `flash-attn==2.6.3`, a 30-60 minute source build that
predates CUDA 12.7. Neither is on the project's critical path by design (D1/D4).

`Relaxed-System-Lab/Flash-Sparse-Attention` (FSA, arXiv 2508.18224) is the
closest template: it solves small-group-size loop-order inefficiency for NSA,
which is structurally our problem for MoBA.

## Corrections made during the session

- FlashMoBA initially appeared not to exist — GitHub repo search for
  "FlashMoBA" surfaces only unrelated flashmob projects, and arXiv queries came
  back empty. The repo is named `flash-moba`. A failed search is weak evidence
  of absence.
- Miniconda was installed into WSL before checking Windows for it. This was
  wrong and has been fully removed (`~/miniconda3`, `~/.conda`, `~/.condarc`).
  `.bashrc` was never modified.

## Known traps

- **CRLF line endings.** Git converts LF to CRLF in the working copy. Markdown
  is unaffected, but a `.sh` script with CRLF fails in WSL with
  `bad interpreter: /usr/bin/env bash^M`. Needs `.gitattributes` with
  `*.sh text eol=lf`.
- **`claude-ds.sh` holds a plaintext API key.** Gitignored before the first
  commit, so it never entered history. Do not un-ignore it.
- **`python3`, never `python`** in WSL. With uv, prefer `uv run`.
- **Import shadowing.** `references/` will contain differently-named but similar
  MoBA packages. The `src/` layout exists partly to prevent accidental pickup.

## Open questions

- Rented GPU vendor for the final benchmark push (RunPod / Lambda / Vast) —
  undecided, does not affect the design.
- Whether `flash_moba` can be built for sm_86 at all. Task 8 records the answer.
- Whether any reference's MoBA semantics diverge from our spec's nine numbered
  rules. Task 8 resolves each divergence explicitly.

## Next step

Task 1 of `docs/superpowers/plans/2026-08-15-moba-foundations.md`, executed
inline (not via subagents) so each decision is explained as it is made.
