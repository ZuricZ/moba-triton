# MoBA Triton Kernel

A Triton implementation of **MoBA** (Mixture of Block Attention) built to stay
efficient at *small* block sizes, where existing implementations collapse.

This is a learning project. The documentation is written to be read, not just
referenced — see [Documentation](#documentation).

---

## The problem, in one table

MoBA routes each query to a handful of KV blocks, so shrinking `block_size`
raises sparsity sharply. Measured on an RTX 3050 Ti (sm_86), S=1024, `top_k=2`:

| block_size | sparsity | speedup available | reference latency | attainment |
|---:|---:|---:|---:|---:|
| 128 | 0.344 | 2.90× | 5.95 ms | 0.008 |
| 64 | 0.180 | 5.54× | 6.75 ms | 0.003 |
| 32 | 0.093 | 10.79× | 10.01 ms | 0.001 |
| **16** | **0.047** | **21.13×** | **15.35 ms** | **0.000** |

As the available speedup rises 7×, the implementation gets **2.6× slower**.
Dense `torch` SDPA runs the same shape in 0.13 ms.

**Attainment** — measured speedup ÷ available speedup — is the project's
headline metric. Raw speedup is gameable by choosing a sparser config;
attainment is not. Details in [`docs/baseline-results.md`](docs/baseline-results.md).

## Why it happens, and the architectural decision

MoBA's selection is **per query**, not per query block. Two queries sitting
side by side routinely choose different KV blocks.

FlashAttention's entire performance argument is that one KV tile load is
amortized across a whole query tile. MoBA breaks that precondition. A
query-major kernel must load the *union* of blocks its tile selected, then mask
away what each individual query did not want:

| seq_len | block_size | BLOCK_M=16 | BLOCK_M=64 | BLOCK_M=128 |
|---:|---:|---:|---:|---:|
| 1024 | 128 | 0.429 | 0.417 | 0.417 |
| 4096 | **16** | **0.134** | **0.043** | **0.028** |

At S=4096 with `block_size=16` and a conventional tile height of 128, a
query-major kernel **discards 97.2% of its tensor-core work** — and it gets
worse as context grows, which is the regime MoBA exists to serve.

**The decision: invert the loop to KV-block-major.** Instead of a program
owning a query tile and looping over blocks, a program owns one KV block and
gathers exactly the queries that selected it. Every key loaded is then used by
every query held — utilization is 100% by construction rather than by tuning.
The group size has a closed form, `top_k × block_size`, which is **independent
of sequence length** (measured 31.9 against a predicted 32).

The cost is that a query now receives partial results from several programs,
which must be combined with an online-softmax merge in a separate pass.

Derivations: [learning note 01](docs/learning/01-loop-order-and-memory-traffic.md)
· measurements: [`docs/divergence-measurements.md`](docs/divergence-measurements.md)
· prior art: [`docs/flashmoba-digest.md`](docs/flashmoba-digest.md)

---

## Status

Foundations and groundwork complete; the MoBA kernel itself is not yet written.

- Spec-first PyTorch oracle defining the semantics, cross-checked bitwise
  against MoonshotAI's reference
- Pluggable backend registry with availability gating (unavailable backends are
  reported with reasons, never silently omitted)
- Correctness harness separating smooth arithmetic error from discrete routing
  flips
- Benchmark suite reporting attainment against the sparsity ceiling
- Minimal FlashAttention-2 forward kernel in Triton, verified against SDPA
- **Next:** MoBA kernel spec — open questions are gather placement and
  load imbalance (measured 6.2× hot-column skew)

Tests: 157 passed, 7 skipped, 18 slow deselected.

## Layout

```
src/moba/
  config.py          MoBAConfig — semantics flags, block arithmetic
  device.py          GPU capability probing
  tolerance.py       correctness policy (exact routing, relative arithmetic)
  ref/naive.py       the oracle: executable definition of MoBA
  kernels/           Triton kernels
  backends/          uniform interface over every implementation
bench/
  suite.py           benchmark driver; attainment is the headline column
  divergence.py      routing divergence, measured from oracle selection masks
docs/                see below
tests/
```

## Running

The GPU stack lives in WSL2; the venv lives outside the repo (see
`CLAUDE.md` for why). Canonical prefix:

```bash
wsl -e bash -lc 'export PATH="$HOME/.local/bin:$PATH"; \
  export UV_PROJECT_ENVIRONMENT="$HOME/.venvs/moba"; \
  cd "/mnt/c/D/Side Projects/GPU Kernel" && uv run <command>'
```

```bash
uv run pytest -m "not slow"        # fast suite
uv run python -m bench.divergence  # routing divergence measurements
```

## Documentation

Written to be read in order:

| Document | What it covers |
|---|---|
| [learning/01 — loop order and memory traffic](docs/learning/01-loop-order-and-memory-traffic.md) | Why small blocks break FlashAttention, from first principles. No GPU knowledge assumed. |
| [learning/02 — anatomy of a FlashAttention kernel](docs/learning/02-anatomy-of-a-flash-attention-kernel.md) | Reading a real Triton kernel: grids, tiling, occupancy, warps, online softmax — then breaking it on purpose. |
| [flashmoba-digest](docs/flashmoba-digest.md) | What FlashMoBA's CUDA actually does, cited line by line. |
| [divergence-measurements](docs/divergence-measurements.md) | Routing divergence measured from real selection masks. |
| [baseline-results](docs/baseline-results.md) | The problem, quantified. |
| [semantics-crosscheck](docs/semantics-crosscheck.md) | Rule-by-rule agreement with MoonshotAI's reference. |

## References

- **MoBA** — Lu et al., Moonshot AI — [arXiv:2502.13189](https://arxiv.org/abs/2502.13189)
- **FlashMoBA** — Xiao, Guo, Mazaheri, Han (MIT Han Lab) — [arXiv:2511.11571](https://arxiv.org/abs/2511.11571)
- **FSA** — [arXiv:2508.18224](https://arxiv.org/abs/2508.18224) — closest structural template
- **FlashAttention** — [arXiv:2205.14135](https://arxiv.org/abs/2205.14135) · **FlashAttention-2** — [arXiv:2307.08691](https://arxiv.org/abs/2307.08691)

Reference implementations are vendored read-only under `references/`
(gitignored) and are never modified.
