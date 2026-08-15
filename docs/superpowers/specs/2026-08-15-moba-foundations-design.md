# MoBA Kernel Project — Foundations Design

**Date:** 2026-08-15
**Status:** Approved
**Scope:** Project foundations only. The MoBA Triton kernel itself is a
separate, later spec.

## Goal

Build the infrastructure that makes a MoBA Triton kernel *developable*: an
executable definition of what MoBA means, a harness that can tell whether an
implementation is correct, a benchmark suite that can tell whether it is fast
and how far it is from the achievable ceiling, and a written understanding of
why the existing CUDA implementation is shaped the way it is.

Nothing here implements MoBA attention on the GPU. The deliverable is the
apparatus you need before that is worth attempting.

## Non-goals

- The MoBA Triton kernel (separate spec).
- A FlashAttention-2 rebuild. FA-2 mechanics are taught inline when they
  appear; they are not a milestone.
- Backward pass. Forward-only, but designed so backward can be added without
  redesign (see Decision D3).
- Support for pre-Ampere hardware (see Decision D2).
- The wider MLA/DSA/CCA/GTA landscape.

## Decisions

**D1 — Spec-first oracle.** We write our own naive PyTorch MoBA and treat it
as ground truth. Vendored reference implementations are third parties whose
disagreements are *findings to investigate*, not test failures.

*Rationale:* you cannot debug a kernel against an oracle you do not
understand. It also decouples the project from a `flash-attn==2.6.3` build
that may never succeed locally, and forces every ambiguous semantic choice to
be written down rather than silently inherited.

**D2 — bf16 primary, sm_80+ (Ampere and newer) only.** fp16 supported as a
secondary dtype. Kaggle (T4 sm_75 / P100 sm_60) is explicitly out of scope.

*Rationale:* the target is modern LLM hardware. The local RTX 3050 is sm_86
Ampere — the same architecture family as the A100, with bf16 tensor cores and
`cp.async`. It differs from an A100 in capacity (4 GB vs 80 GB VRAM, 20 vs 108
SMs, 99 KB vs 164 KB shared memory per SM), not capability. Kaggle's T4 is
Turing: no bf16, no `cp.async`, and *less* shared memory than the laptop. It
was the only thing pushing the design toward fp16, and it is not worth the
constraint.

**D3 — Forward-only, designed for backward.** The oracle stays
autograd-friendly and the harness's comparison layer is dtype- and
tensor-generic, so gradient comparisons can be added as a fourth test tier
without restructuring.

**D4 — Pluggable backends with availability gating.** Every implementation
conforms to one `Backend` protocol and declares its own availability. The same
suite runs locally with a subset and on rented hardware with the full set,
unchanged.

**D5 — Relative tolerance policy.** Correctness tolerance is defined relative
to a same-dtype reference's error against an fp64 computation, never as an
absolute constant. See "Tolerance policy".

## Global constraints

- Python package layout: `src/` layout, package name `moba`.
- Primary dtype: `torch.bfloat16`. Secondary: `torch.float16`.
- Minimum GPU: compute capability 8.0.
- Local test budget: must fit in 4 GB VRAM.
- All GPU commands run inside WSL2 via
  `wsl -e bash -lc 'cd "/mnt/c/D/Side Projects/GPU Kernel" && <cmd>'`.
- `python3`, never `python`.
- Benchmarking via `triton.testing.do_bench`.
- `references/` is vendored, read-only, and gitignored.

## MoBA semantics (the executable definition)

These are *our* definitions. They are what the oracle implements and what the
kernel must match. Task 8 cross-checks them against the paper and the
reference implementations and records any divergence; divergences change this
document, they do not silently change the code.

Given queries `Q [B, H, S, D]`, keys/values `K, V [B, H, S, D]`, block size
`Bk`, and selection count `top_k`:

1. **Blocking.** `K` is partitioned along the sequence axis into
   `num_blocks = ceil(S / Bk)` contiguous blocks. The final block may be
   short.
2. **Block representative.** Each block's representative key is the mean of
   its keys along the sequence axis, producing `Kbar [B, H, num_blocks, D]`.
3. **Gate score.** For query at position `i`, the score for block `b` is
   `dot(Q[i], Kbar[b])`. No scaling is applied at the gate — scaling affects
   only the attention softmax, and a monotone transform cannot change a
   top-k ranking.
4. **Causal candidacy.** Block `b` is a candidate for query `i` only if the
   block lies strictly in the past — that is, every position in `b` is `< i`'s
   own block start. Blocks containing future positions are excluded.
5. **Current-block selection.** The block containing query `i` is *always*
   selected, and does not consume a `top_k` slot. Attention within it is
   causally masked.
6. **Top-k over candidates.** The `top_k` highest-scoring candidate blocks are
   selected. If fewer than `top_k` candidates exist (early queries), all
   candidates are selected.
7. **Tie-breaking.** Ties in gate score resolve toward the **lower block
   index**, deterministically.
8. **Partial-block pooling.** For a block that is short (the final block), the
   mean is over the positions that exist. Causal masking within the current
   block does *not* affect its representative — the representative is computed
   over the whole block. Governed by `MoBAConfig.pool_masked_only`, default
   `False`.
9. **Attention.** Standard scaled dot-product attention over the union of
   selected blocks' keys/values, with softmax scale `1/sqrt(D)`, causal
   masking applied within the current block only (all other selected blocks
   are wholly in the past and need no mask).

Each of items 5, 7, and 8 is exposed as an explicit `MoBAConfig` field so a
disagreement with a reference implementation can be diagnosed by flipping one
flag.

## Architecture

### File structure

```
pyproject.toml
src/moba/
  config.py            MoBAConfig dataclass: shapes, block_size, top_k,
                       semantics flags, dtype
  ref/naive.py         The oracle. Pure PyTorch, one op per line of the
                       math above. Autograd-friendly. No flash-attn.
  kernels/             Our Triton kernels (empty in this spec)
  backends/
    base.py            Backend protocol, BackendResult, registry
    naive_torch.py     Oracle wrapped as a backend
    sdpa_dense.py      torch SDPA dense causal — the roofline baseline
    moonshot.py        MoonshotAI/MoBA, availability-gated
    triton_moba.py     ReyJerry/Triton-MoBA, availability-gated
    flash_moba.py      mit-han-lab/flash-moba, availability-gated
  device.py            Device capability probing + metadata record
tests/
  conftest.py          Device fixtures, shape sweep parametrization
  test_semantics.py    Tier 1: exact identities, no tolerance
  test_tolerance.py    Tolerance policy machinery
  test_backends.py     Tier 2: each available backend vs oracle
  test_disagreement.py Tier 3: pairwise matrix, reported not asserted
bench/
  suite.py             do_bench sweep over configs x available backends
  report.py            Results -> CSV + markdown, with device metadata
docs/
  flashmoba-digest.md  Plain-English digest of flash-moba's csrc/
scripts/
  setup_env.sh         WSL venv + torch/triton install
  fetch_references.sh  Clone the five reference repos into references/
references/            Vendored clones (gitignored)
```

### Backend protocol

```python
class Backend(Protocol):
    name: str

    def is_available(self, cfg: MoBAConfig) -> tuple[bool, str]:
        """Return (available, reason). Never raises."""

    def forward(self, q, k, v, cfg: MoBAConfig) -> torch.Tensor:
        """Return output [B, H, S, D]."""
```

`is_available` returns a *reason string* so skipped backends appear in
reports as `flash_moba: skipped - requires sm_80+, found sm_75` rather than
vanishing. A silently short benchmark table is worse than no table.

### Tolerance policy

For a given config and dtype:

1. Compute the oracle in **fp64** — call it `truth`.
2. Compute the oracle in the **target dtype** (bf16/fp16) — call it
   `ref_error = max_abs(oracle_dtype - truth)`.
3. A backend passes if
   `max_abs(backend_out - truth) <= max(TOLERANCE_FACTOR * ref_error, floor)`.

`TOLERANCE_FACTOR = 2.0`, following the FlashAttention paper's methodology.
`floor = 1e-6` guards the degenerate case where `ref_error` is ~0 on tiny
inputs.

The question is never "is the error small" but "is our error comparable to
what any correct implementation in this dtype would produce". Absolute
constants do not survive changes in shape, dtype, or hardware; this does.

### Shape sweep (local, 4 GB budget)

| Parameter | Values |
|---|---|
| batch | 1, 2 |
| heads | 4, 8 |
| seq_len | 128, 512, 1024, 2048 |
| head_dim | 64 |
| block_size | 16, 32, 64, 128 |
| top_k | 2, 4, 8 |

Small block sizes are deliberately over-represented: they are the regime this
project exists to fix. Configs where `top_k >= num_blocks` are pruned from
Tier 2 (they degenerate to dense and are covered exactly by Tier 1).

### Benchmark suite

`bench/suite.py` sweeps `(seq_len, block_size, top_k) x available_backends`
through `triton.testing.do_bench`, recording per row:

- latency (ms, median), effective TFLOP/s, peak memory (MB)
- **sparsity ratio** — selected keys / total causal keys
- **attained vs theoretical speedup** — measured speedup over the dense SDPA
  baseline, divided by `1 / sparsity_ratio`

That last column is the project's thesis in a single number. A kernel
achieving 2x where sparsity implies 8x is exactly the gap this work targets,
and the report should show it rather than make the reader derive it.

Device metadata (GPU name, compute capability, VRAM, driver, torch and Triton
versions) is recorded on every row so local and rented-GPU result sets merge
into one comparable table.

Configs that would OOM get a pre-flight estimate and a skip-with-reason,
consistent with the `is_available` philosophy.

### FlashMoBA digest

`docs/flashmoba-digest.md` is written after cloning `mit-han-lab/flash-moba`
and reading `csrc/` directly. It must explain, in plain English and grounded
in named source files:

- the loop order the CUDA kernel uses, and what is held in registers vs
  shared memory at each level
- why that loop order degrades at small block sizes
- what FSA (arXiv 2508.18224) does about the analogous problem in NSA, and
  which parts transfer to MoBA

It is prose grounded in source actually read. Claims about the code cite the
file they come from.

## Testing strategy

| Tier | File | Asserts | Tolerance |
|---|---|---|---|
| 1 | `test_semantics.py` | Yes | None — exact identities |
| 2 | `test_backends.py` | Yes | Relative policy |
| 3 | `test_disagreement.py` | No — reports | Relative policy |

Tier 1's keystone: with `top_k >= num_blocks`, MoBA must reduce to dense
causal attention, matching `F.scaled_dot_product_attention(is_causal=True)`.
This validates all non-gate machinery with no tolerance argument, and any
failure is unambiguously a bug.

## Risks

- **`flash-attn==2.6.3` may not build** against current torch/CUDA 12.7. Both
  Python references depend on it. Mitigated by D1 and D4: the project does not
  block on it, and those backends simply report unavailable.
- **FlashMoBA may not build for sm_86.** Its `setup.py` defaults to archs
  `80;90;100;120`. Even with `86` added, its shared-memory tiling may exceed
  sm_86's 99 KB and fail at launch. Mitigated by D4; the digest depends on
  *reading* the source, not running it.
- **The oracle could be wrong.** Mitigated by Tier 1's exact identities and by
  the Task 8 cross-check against paper and references.

## Success criteria

1. `pytest` runs green locally with at least the oracle, SDPA baseline, and
   dense-reduction identity tests passing.
2. `bench/suite.py` produces a populated results table on the local GPU,
   with explicit skip reasons for every unavailable backend.
3. The tolerance policy is exercised by a test that deliberately perturbs an
   implementation and confirms the harness catches it.
4. `docs/flashmoba-digest.md` explains the small-block loop-order problem with
   citations to specific files in `references/flash-moba/csrc/`.
5. Adding a new backend requires touching exactly one new file.
