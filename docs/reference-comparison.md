# Against the reference implementations — measurements

First head-to-head between our kernel and the vendored references.
RTX 3050 Ti Laptop (sm_86, 20 SMs, 4 GB), bf16, `block_size=128, top_k=8`
(FlashMoBA's published configuration), shape `[1, S, 4, 64]`,
`uv run python -m bench.suite`-equivalent driver, median of `do_bench`.

**Read the ratios, not the absolute times.** 20 SMs and 4 GB. Both sides of
every ratio ran on this card in the same process.

---

## Before the numbers: a correction to every previous "vs dense"

`bench/suite.py` measures the dense baseline **first for every config**, so
speedups can be filled in as rows are produced. Nothing warmed the device
first, so the dense row absorbed the GPU's clock ramp and every backend
measured after it looked correspondingly fast.

Five interleaved passes at S=4096, same process, median of 100 each:

| pass | sdpa_dense | fa2_minimal | ours |
|---|---|---|---|
| 0 | **1.182** | **1.165** | **0.821** |
| 1 | 0.743 | 0.754 | 0.748 |
| 2 | 0.741 | 0.760 | 0.775 |
| 3 | 0.767 | 0.802 | 0.765 |
| 4 | 0.756 | 0.757 | 0.750 |
| spread | 1.59x | 1.55x | 1.10x |

Pass 0 is inflated for all three. `ours` shows a smaller spread only because
it never ran first, so it never paid the ramp.

**Effect: the denominator was ~1.57x too large at S=4096, turning a true 0.96x
into a reported 1.53x.** `docs/baseline-results.md` and `results/kernel_v1.csv`
carry this bias and should be read as optimistic until re-run.

`do_bench(warmup=25)` does not prevent it. That warms the *callable* --
compiling it, allocating its buffers, filling caches. The clock ramp belongs to
the *device* and is paid once per process, by whichever measurement goes first.
Two different things sharing a name.

Fixed by `bench.suite.warm_up_device`, called once before the first
measurement. All numbers below are post-fix.

This is the third machine artifact this project has read as a kernel result,
after the WDDM allocator (`docs/gqa-measurements.md`) and clock drift
(`docs/layout-measurements.md`). All three flattered us, which is the direction
nobody thinks to check.

---

## The numbers

| S | blocks | ours | sdpa_dense | vs dense | fa2_minimal | naive_torch | moonshot | vs moonshot |
|---|---|---|---|---|---|---|---|---|
| 1024 | 8 | 0.381 ms | 0.083 ms | 0.22x | 0.078 ms | 2.581 ms | 5.848 ms | 15x |
| 2048 | 16 | 0.500 ms | 0.228 ms | 0.46x | 0.215 ms | 5.209 ms | 9.853 ms | 20x |
| 4096 | 32 | 0.773 ms | 0.739 ms | 0.96x | 0.759 ms | 17.543 ms | 46.922 ms | 61x |
| 8192 | 64 | 1.535 ms | 2.766 ms | **1.80x** | 2.974 ms | 62.547 ms | 114.406 ms | 75x |

`moonshot` is MoonshotAI's own MoBA -- the algorithm's reference
implementation, in PyTorch. `naive_torch` is our oracle. `fa2_minimal` is our
from-scratch FlashAttention-2, a second dense baseline.

### Against the reference MoBA: 15x to 75x

Large, and widening with sequence length. Both are correct implementations of
the same algorithm; the difference is entirely that theirs materializes what
ours keeps on chip. Treat the exact multiples as approximate -- `moonshot`'s
own time moved 31.9 / 33.8 / 46.9 ms across three runs at S=4096, being naive
PyTorch with no fused kernel. The order of magnitude is the result.

### Against dense: crossover between S=4096 and S=8192

    0.22x  ->  0.46x  ->  0.96x  ->  1.80x
     S=1K       S=2K       S=4K       S=8K

Monotonic, and the mechanism is not subtle. At `block_size=128`:

* **S=1024 gives 8 blocks and `top_k=8`.** The gate selects *every* block.
  There is no sparsity, and we pay routing, partial storage and a combine pass
  to compute a dense result. A sparse kernel *should* lose here, and 0.22x is
  the honest cost of the machinery when the machinery buys nothing.
* **S=8192 gives 64 blocks**, of which a query attends 8. Now the arithmetic
  saving exists and the kernel collects it.

> Benchmarking MoBA at short context measures the overhead and none of the
> benefit. It is not a fair fight in either direction.

This is the prediction in `CLAUDE.md` -- sparsity comes from `num_blocks`
growing with context, not from shrinking blocks -- confirmed on real hardware
for the first time.

### The dense baselines agree with each other

`fa2_minimal` tracks `sdpa_dense` within 7% at every length (1.06x, 1.06x,
0.97x, 0.93x). Our from-scratch Triton FA-2 is level with torch's production
kernel, which is the floor under everything above: the Triton is not what
limits the MoBA numbers.

---

## What is missing, and why

| | status | blocker |
|---|---|---|
| `flash_moba` | **not measured** | CUDA extension unbuilt |
| `triton_moba` | **not measured** | needs `flash_attn` |

Neither blocker is about this GPU.

**flash-moba** ships `setup.py` with `FLASH_MOBA_CUDA_ARCHS` defaulting to
`"80;90;100;120"`, and there is no `86` branch -- so the env var cannot add one.
The way in is that an unlisted arch produces an *empty* flag list, and PyTorch's
`CUDAExtension` fills in from `TORCH_CUDA_ARCH_LIST` only when the caller
supplies nothing. Verified:

    caller supplies nothing  ->  ['-gencode=arch=compute_86,code=sm_86']
    caller supplies gencode  ->  []

So `FLASH_MOBA_CUDA_ARCHS=86 TORCH_CUDA_ARCH_LIST=8.6` builds for this card
without editing a vendored file. Its `setup_requires` also needs
`packaging`/`psutil`/`ninja` present, because setuptools fetches them with
`python -m pip` and this venv has no pip.

Even built, `docs/flashmoba-digest.md` predicts a launch failure: their tiling
assumes an A100's 164 KiB of shared memory and this card has 99 KiB.

**Triton-MoBA** itself constrains nothing. It is pure Python -- no `.so`, no
`.cu`, no `setup.py` -- and its `requirements.txt` asks for `torch>=2.1.0`,
which our 2.13 satisfies. The blocker is one level down: it imports
`flash_attn`, which *is* a compiled extension.

`flash_attn`'s prebuilt wheels are ABI-pinned per torch minor, because the
extension links `libtorch`. Upstream ships no wheels on PyPI at all (76
releases, every one an sdist); the wheels live on GitHub Releases and cover
torch **1.12 through 2.10**. We run 2.13, three minors past anything they
built -- so no wheel exists for us, and none will.

**The source has no such limit.** `flash_attn`'s `setup.py` compiles against
whatever torch is installed, so building it here against 2.13 is the direct
route and keeps one environment. Two things make it unattractive rather than
impossible: it is a 1-3 hour compile, and flash-attn 2.6.3 (their pin) has
never been tested against a torch this new, so a C++ API change is possible.

Nothing is gained by downgrading our own torch instead: 2.13 carries Triton
3.7.1, and Triton is the compiler for our kernel. Changing it changes the thing
being measured, and every number in `docs/` was taken under 3.7.1.

If the compile is attempted here, note what crashed WSL twice on 2026-09-05:
`MAX_JOBS=6` against a 15 GB VM (nvcc on CUTLASS wants ~4 GB per process), and
building under `/mnt/c`, the 9p bridge that `CLAUDE.md` already warns about for
venvs. Use `MAX_JOBS=2` and build under `~/`.

**Both resolve on the rented card**, where `sm_80`/`sm_90` are in flash-moba's
default arch list and torch can be pinned to a version flash-attn ships wheels
for. See `docs/layout-measurements.md` for why absolute numbers here are not
the deliverable anyway.

---

## What this does not measure

* **Anything past S=8192.** At S=16384 the 3000 MB budget guard skips every
  backend on a 4 GB card. FlashMoBA's own sweep runs to 512K.
* **The two reference kernels**, per above.
* **The backward pass.** Forward only.
* **Grouped-query attention.** `H_q == H_kv == 4` throughout.
* **Any absolute number.** 20 SMs, 4 GB.

## See also

* `docs/baseline-results.md` -- earlier numbers, carrying the warm-up bias.
* `docs/flashmoba-digest.md` -- why their tiling assumes A100 shared memory.
* `docs/semantics-crosscheck.md` -- correctness against the references, still
  open for Triton-MoBA.
