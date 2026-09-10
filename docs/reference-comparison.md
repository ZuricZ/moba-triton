# Against the reference implementations — measurements

> **Superseded for the headline numbers.** Everything below was measured on
> an RTX 3050 Ti laptop -- 20 SMs, 4 GB, 99 KiB of shared memory. The A100
> head-to-head is in [`a100-results.md`](a100-results.md), and it is the one to
> quote. This file is kept because the *methodology* it works out -- the
> thermal protocol, the four measurement artifacts, the FLOP conventions -- is
> what makes those numbers trustworthy, and because the laptop is still where
> correctness work happens.

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

## The thermal protocol, and why the numbers needed one

Four sweeps on 2026-09-05 produced four different answers for the same kernel.
`ours` at S=8192 read 1.535, 5.243, 7.271 and 1.596 ms across them. The kernel
never changed; the measuring apparatus was wrong in three different ways.

### Warming up is not monotonic

`warm_up_device` was added because a cold GPU sits at a low clock and a short
measurement can finish before it ramps. True, but only half the picture. With
the card heat-soaked at 84-86 C after an evening of compiling, `ours` at
S=8192, three runs at each warm-up length:

| warm-up | run 1 | run 2 | run 3 | spread | temp |
|---|---|---|---|---|---|
| 2 s | 1.583 | 1.612 | 1.610 | 1.02x | 84 |
| 5 s | 1.754 | 1.743 | 6.997 | 4.01x | 84 |
| 10 s | 5.414 | 5.858 | 5.614 | 1.08x | 85 |
| 20 s | 10.847 | 14.810 | 11.495 | 1.37x | 86 |
| 40 s | 6.536 | 5.556 | 5.663 | 1.18x | 84 |

**More warm-up is worse.** On a saturated card the warm-up is not raising the
clock, it is spending the last of the power budget before timing starts. Both
extremes are bad -- 90 s of idle with no warm-up measured 11.70 ms, a 20 s
warm-up measured 16.67 ms, and the true figure is ~1.60 ms.

Cooling first restores it, and cheaply:

    after 20 s warm-up (cooked)      16.669 ms   86 C
    cooled 26 s to 76 C, warm 2 s     1.617 ms
    immediately again                 1.585 ms
    another cooldown cycle            1.620 ms

Hence `cool_down_device()` then `warm_up_device()` before **every**
measurement, not once per run. One short measurement does not re-cook the
card; a provider running hundreds of milliseconds a rep does, which is how
`moonshot_naive` at 690 ms poisoned every backend measured after it.

### Tight error bars around a wrong number

Read the spread column again: 1.02x, 1.08x, 1.18x. In any fixed thermal state
this card is *repeatable*. Three runs agreeing within 1.37x around a figure
that is 10x wrong is what makes this hard to catch, and it is the reason a
confidence interval is not evidence of a good measurement here.

### The minimum, not the median

Cooling reduces contamination without eliminating it. Rechecking `ours` three
times per size, cooling before each:

    S=2048   0.504  0.506  1.687
    S=4096   0.785  0.797  0.782
    S=8192   1.616  1.617  3.182

One reading in three is still bad, and **every bad reading is slow**. Nothing
makes a kernel finish faster than its true time, so the error is one-sided and
the minimum is the least-contaminated sample rather than an optimistic pick.
`bench/compare.py` therefore takes the best of `BEST_OF = 3`. A median would
have recorded 1.687 ms at S=2048 had the outlier landed differently.

Cost of the whole protocol: 9 minutes for 48 measurements.

### Four artifacts, and the direction they pull

This is the fourth measurement artifact this project has caught, after the
clock ramp (below), the WDDM allocator (`docs/gqa-measurements.md`) and clock
drift (`docs/layout-measurements.md`). The first three all flattered us. This
one penalised us, which is the direction a habit of suspecting good news does
not catch.

Two things were ruled out before landing on thermal, each by measurement:

* **Not the allocator.** `empty_cache()` returned `reserved` to the identical
  cold-state 334 MB with zero `num_alloc_retries`, and the slowdown persisted.
* **Not something a backend leaks.** A pure matmul burn touching no MoBA code
  reproduced it exactly.

`nvidia-smi` was no help in either direction: it reported no throttle flags and
a byte-identical 1717 MHz / 84 C at three different moments, which under WSL2
is a stale reading rather than a stable card.

---

## The numbers

Re-measured 2026-09-06 under the thermal protocol below, with `flash_moba`
included for the first time. All six backends monotonic in S.

**Forward:**

| S | blocks | ours | flash_moba | sdpa_dense | fa2_minimal | naive_torch | moonshot_naive |
|---|---|---|---|---|---|---|---|
| 1024 | 8 | 0.280 ms | 1.392 ms | 0.084 ms | 0.077 ms | 2.160 ms | 3.784 ms |
| 2048 | 16 | 0.439 ms | 1.726 ms | 0.230 ms | 0.223 ms | 4.833 ms | 10.067 ms |
| 4096 | 32 | 0.693 ms | 2.503 ms | 0.763 ms | 0.786 ms | 16.493 ms | 31.972 ms |
| 8192 | 64 | **1.377 ms** | 4.377 ms | 2.814 ms | 3.057 ms | 62.712 ms | 321.059 ms |

**Backward** (`forward().backward(dO)`, the full step, as NSA reports it):

| S | ours | flash_moba | sdpa_dense | fa2_minimal | naive_torch | moonshot_naive |
|---|---|---|---|---|---|---|
| 1024 | 0.536 ms | 2.666 ms | 0.305 ms | 0.629 ms | 3.864 ms | 11.188 ms |
| 2048 | 1.010 ms | 2.636 ms | 0.808 ms | 2.034 ms | 9.229 ms | 32.846 ms |
| 4096 | 1.944 ms | 3.725 ms | 2.674 ms | 7.481 ms | 32.461 ms | 116.055 ms |
| 8192 | **4.044 ms** | 6.856 ms | 9.911 ms | 29.005 ms | 511.572 ms | 2898.887 ms |

**Ratios at S=8192**, the only length here with real sparsity:

| | forward | backward |
|---|---|---|
| vs `sdpa_dense` | **2.04x** | **2.45x** |
| vs `flash_moba` | 3.18x | 1.70x |
| vs `moonshot_naive` | 233x | 717x |

Re-measured 2026-09-07 after both kernels moved to `triton.autotune`: the
forward gained 14% and the backward 19% by tuning chunk height and warp count
instead of taking the largest tile that fits shared memory. The crossover
against dense moved down a step -- `ours` now wins at S=4096 (0.693 against
0.763 ms) where it previously tied.

`moonshot_naive` is MoonshotAI's readable reference, in PyTorch. `naive_torch`
is our oracle. `fa2_minimal` is our from-scratch FlashAttention-2, a second
dense baseline.

**The Moonshot column is their slow path, and the tables above say so only
because the name now does.** They ship two implementations: `moba_naive.py`
materialises the score matrix and exists to be read; `moba_efficient.py` fuses
onto FlashAttention and is what their library runs. Until 2026-09-07 this
backend was registered as plain `moonshot`, which claimed the naive path *was*
MoonshotAI's MoBA and turned a comparison against reference code into a
headline 240x. Renamed to `moonshot_naive`, with `moonshot_efficient` now
registered beside it and reporting itself unavailable until flash-attn 2.6.3
exists.

That single dependency gates two backends -- `moonshot_efficient` and
`triton_moba` -- which is the argument for building it on the rented card even
though it is an hour of compile time.

### Two independent checks that this run is sound

**Against prior records.** `sdpa_dense` and `fa2_minimal` reproduce values
taken on a different day by a different driver, to within 3%:

    sdpa_dense   0.083 / 0.231 / 0.776 / 2.830   vs   0.083 / 0.228 / 0.739 / 2.766
    fa2_minimal  0.077 / 0.221 / 0.786 / 3.034   vs   0.078 / 0.215 / 0.759 / 2.974

**Against arithmetic, which needs no prior record.** MoBA attends
`top_k * block_size` = 1024 keys per query *whatever S is*, so total work grows
linearly, and S=8192 must cost about twice S=4096. Measured: 0.693 -> 1.377 ms,
a factor of 1.99. A contaminated run of 2026-09-05 reported 1.754 ms at S=4096
against 1.639 ms at S=8192 -- more work taking less time -- and that violation
is what exposed it. Worth keeping as a habit: dense attention is quadratic in
S, MoBA is linear, and either constraint catches a bad cell on a machine with
no history at all.

### On beating flash_moba here

3.18x on the forward, on 20 SMs and 99 KiB of shared memory. Their
`decide_lg_block_m` selects a 512-query tile at S=8192, sized for the 164 KiB
an A100 provides. **This says our kernel suits this card, not that it is faster
than theirs.** The comparison that would mean something needs the rented
sm_80/90, where their tile choice is the one they designed for.

### Against the reference MoBA: 15x to 75x

Large, and widening with sequence length. Both are correct implementations of
the same algorithm; the difference is entirely that theirs materializes what
ours keeps on chip. Treat the exact multiples as approximate -- `moonshot_naive`'s
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
| `flash_moba` | **built, launches** (2026-09-05) | none -- see below |
| `triton_moba` | **not measured** | needs `flash_attn` |

Neither blocker was about this GPU, and the `flash_moba` one is now cleared.

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

`docs/flashmoba-digest.md` further predicted a launch failure even once built,
on the grounds that their tiling assumes an A100's 164 KiB of shared memory
against this card's 99 KiB. **That prediction was wrong**, and the section
below records why.

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

## flash_moba, built and running (2026-09-05)

### It launches, and the shared-memory prediction was wrong

Built with `FLASH_MOBA_CUDA_ARCHS=86 TORCH_CUDA_ARCH_LIST=8.6`, under `~/`
rather than `/mnt/c`, `MAX_JOBS` sized to RAM. 85 objects, zero errors, a
74 MB `flash_moba_cuda.cpython-313-x86_64-linux-gnu.so`.

It then launched at every sequence length, finite output throughout:

| S | lg_block_m | result |
|---|---|---|
| 1024 | 128 | OK |
| 2048 | 128 | OK |
| 4096 | 256 | OK |
| 8192 | 512 | OK |

The digest expected a shared-memory failure because their tiling is sized for
an A100. The reason it does not happen is `decide_lg_block_m`: the query tile
is chosen **at call time from predicted sparsity**, not fixed at compile time.

    budget  = top_k * block_size
    density = (2*budget*S - budget^2) / S^2      # causal
    sparsity <= 0.5 -> 128;  <= 0.7 -> 256;  <= 0.8 -> 512
    <= 0.9 -> 768;  else 1024

A closed form on the host, five branches, no autotuning and no occupancy
probe. A runtime policy fits whatever SRAM the card has; the compile-time
constant the digest assumed would not have. This is the same class of decision
as our `tuning.PIPELINE_STAGES`, where a conservative guess silently cost 12%.

### How it is installed, and why not with pip

The extension is on the venv's path via a `.pth` file, not a pip install:

    ~/.venvs/moba/lib/python3.13/site-packages/flash-moba-local.pth
        -> /home/ubuntu/fmbuild/site
    ~/fmbuild/site/
        flash_moba                                    -> the package
        flash_moba_cuda.cpython-313-...-linux-gnu.so  -> the built extension

Both `uv pip install .` and `uv pip install -e .` were tried first and both
began recompiling all 85 objects: each builds in its own `/tmp/tmp*.build-temp`
rather than the tree's existing `build/`, so the hour already spent buys
nothing. Verified by reading `-o` out of the running nvcc command lines.

The link directory holds exactly those two entries on purpose. Putting
`~/fmbuild/flash-moba` itself on `sys.path` would also expose its `tests/` and
`benchmarks/` directories, and `tests` would then shadow our own test package
-- the same collision `implementations/layout.py` documents for the name
`moba`.

This is a local convenience, not part of the repo: a fresh clone has neither
the build tree nor the `.pth`, and `flash_moba` correctly reports itself
unavailable there.

### The adapter was calling the wrong function

`implementations/flash_moba.py` called `flash_moba_attn_varlen_func` with
`(q, k, v, cu, cu, s, s, block_size, top_k, causal=True)`. That function is the
attention kernel **alone**: after the four varlen arguments it wants the routing
as three tensors -- `moba_col_offsets`, `moba_col_nnz`, `moba_row_indices` --
and does no gating at all. We passed two ints into three tensor slots.

The function whose signature we had actually written against is
`flash_moba_varlen_func`, one word shorter: their full pipeline, gate then
attend. Fixed by calling that.

The bug survived because the extension had never compiled, so the call had
never once executed. An adapter written from reading source is a hypothesis,
not code.

**Their split is our split.** Those three tensors are exactly the CSC triple
`src/moba/kernels/routing.py` produces, field for field, and their gate
(`flash_topk_varlen_func`) returns them as a separate call. Both projects cut
the pipeline at the same seam, which means a future comparison can isolate
gate from attention on both sides rather than reporting one end-to-end number.
Two layout differences to handle if that is done: their `col_offsets` is int64
with `num_blocks` entries and indexes a globally-flat `row_indices`, ours is
int32 with `num_blocks + 1` and is per-(batch, head); and ours carries
`head_indices` for grouped-query attention, which their arrays have no slot
for, so the conversion is exact only when `H_kv == H_q`.

### Divergence from the oracle: ours and theirs are level

Relative to `naive_torch` at `block_size=128, top_k=8`, shape `[1, S, 4, 64]`,
bf16, max absolute error over the output divided by the output's scale:

| S | blocks | ours vs oracle | flash_moba vs oracle | per-block counts equal |
|---|---|---|---|---|
| 1024 | 8 | 0.0038 | 0.0028 | 100% |
| 2048 | 16 | 0.036 | 0.040 | 39% |
| 4096 | 32 | 0.035 | 0.045 | 26% |
| 8192 | 64 | 0.043 | 0.049 | 26% |

**Neither implementation is closer to the oracle than the other.** The error is
not arithmetic: at S=1024 there are 8 blocks and `top_k=8`, so every block is
selected, the two gates *cannot* disagree, and both land at the bf16 floor. From
S=2048 the gates make different -- and equally valid -- block choices, and the
output difference that follows is a routing flip, which is the distinction
`tests/tolerance.py` exists to draw.

Read the last column as a divergence indicator only. It compares each block's
query *count*, so equal counts do not prove equal selections; unequal counts do
prove unequal ones.

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
