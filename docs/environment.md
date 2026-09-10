# The environment, and where each version is pinned

Every number in `docs/` depends on a stack, and "which versions?" needs an
answer that is not someone's shell history. This is the index. Nothing is
duplicated here — each row points at the file that actually decides.

## The pins

| what | version | decided by |
|---|---|---|
| Python | `3.13` | [`.python-version`](../.python-version) |
| torch | `2.13.0+cu126` | [`uv.lock`](../uv.lock) |
| triton | `3.7.1` | [`uv.lock`](../uv.lock) |
| numpy | `2.5.2` | [`uv.lock`](../uv.lock) |
| matplotlib | `3.11.1` | [`uv.lock`](../uv.lock), dev group |
| einops | `0.8.2` | [`uv.lock`](../uv.lock), dev group |
| pandas | (locked) | [`uv.lock`](../uv.lock), dev group |
| CUDA runtime (wheel) | `12.6.77` | [`uv.lock`](../uv.lock) |
| wheel index | `download.pytorch.org/whl/cu126` | [`pyproject.toml`](../pyproject.toml) |
| flash-attn | `v2.6.3` | [`scripts/bootstrap_remote.sh`](../scripts/bootstrap_remote.sh) |
| the five reference repos | exact commits | [`scripts/fetch_references.sh`](../scripts/fetch_references.sh) |

`matplotlib` and `einops` are in the **dev group**, not the runtime
dependencies: nothing under `src/` imports them, so the package stays
installable without them, but `uv sync` installs the group by default so the
harness and the vendored references get what they need. `matplotlib` is for
`bench/plots.py`; `einops` is imported by MoonshotAI's `moba_efficient.py` and
by Triton-MoBA. Both were missing on the first A100 run, which built two CUDA
extensions successfully and then could not use either backend.

**Two of the three cannot be found by auditing our imports**, which is worth
saying because the audit comes back clean and the fresh box still fails.
Nothing in this repo imports `einops` or `pandas`. They are needed by code we
*call*: `triton.testing.perf_report(print_data=True)` imports pandas
internally, and the vendored references import einops. Declare what the call
paths need, not what the files import.

`pyproject.toml` lists `dependencies = ["torch", "triton", "numpy"]` with **no
version bounds**. That is deliberate and not a gap: `uv.lock` holds the exact
resolution and `uv sync` honours it, so the lockfile is the pin and the
manifest only states the shape. Bounds in both places drift apart.

## Why each one is what it is

**torch from the cu126 index, not PyPI.** The default wheel is
`2.13.0+cu130`, which installs cleanly and then fails at CUDA init with
"driver too old". CUDA guarantees *minor* version compatibility — any 12.x
runtime runs on a driver supporting 12.0+ — but crossing the 12 → 13 *major*
boundary breaks that. The dev laptop's driver is CUDA 12.7, so the runtime
must stay on the 12.x line.

**triton 3.7.1 is not incidental — it is the compiler under test.** Every
kernel in `src/moba` is compiled by it, so changing it changes the thing being
measured. This is the reason we build flash-attn from source rather than
downgrading torch to reach a version flash-attn ships wheels for.

**flash-attn pinned to 2.6.3, not latest.** Triton-MoBA calls the *private*
`_flash_attn_varlen_forward` and unpacks eight return values. That signature
changed in later releases, so "newer" is not "compatible". It gates two
backends — `triton_moba` and `moonshot_efficient` — so one build brings up
both.

**The reference repos are pinned to commits.** Until 2026-09-07 they were
cloned `--depth 1` from HEAD, so a run today and a run next month could compare
against different FlashMoBA. For a project whose entire output is a comparison
table, that is a reproducibility bug rather than a convenience. `PIN=off`
tracks HEAD when you want to check whether upstream moved.

## The commits every published number was measured against

| repo | commit | dated |
|---|---|---|
| mit-han-lab/flash-moba | `39d9ac043b271d046a2181a9991e99a26b67bca1` | 2025-11-19 |
| ReyJerry/Triton-MoBA | `7c44c895d087521702c549fa02b5e5e9b5d383e3` | 2025-12-18 |
| MoonshotAI/MoBA | `b5d58363311d3ca946f1ec444182727c15e338b5` | 2025-04-03 |
| fla-org/native-sparse-attention | `bd67af59b90afa34b25f61d2922e612d10dba3bd` | 2025-03-19 |
| Relaxed-System-Lab/Flash-Sparse-Attention | `7ff144fd7ff485dc4220d439f31cc1708b64fef3` | 2025-09-29 |

Moving a pin means re-running the comparison and saying in the commit message
which numbers changed. The table and the code move together or the table is
fiction.

## What is *not* pinned, and cannot be

**The GPU driver and its CUDA version.** Supplied by the machine. The dev box
is 566.07 / CUDA 12.7 under WDDM; a rented box will differ, and
`bootstrap_remote.sh` reports what it found rather than assuming.

**The compute capability.** Detected at runtime.
`src/moba/device.py` reads `shared_memory_per_block_optin` and the SM count, so
`tuning.py` picks tiles for whatever card it lands on — sm_86's 99 KiB here,
an A100's 164 KiB there. This is why the kernel needed no changes to be ready
for the rented run.

**`MAX_JOBS`.** Derived from RAM, not stored. `MAX_JOBS=4` exhausted 15 GB and
4 GB of swap in ninety seconds on flash-attn's backward units, which want
8–10 GB each. `bootstrap_remote.sh` uses `RAM/8`.

## Reproducing the environment

```bash
uv sync                                    # torch, triton, numpy from uv.lock
./scripts/fetch_references.sh --recursive  # references at their pinned commits
./scripts/bootstrap_remote.sh              # both, plus builds and the sweep
```

The venv lives in the repo on a normal Linux box. On the WSL dev machine it is
deliberately outside it — `/mnt/c` is a 9p bridge to a Windows drive where uv
cannot hard-link from its cache, making `uv sync` take 11 minutes instead of
1.5 seconds. See `CLAUDE.md`.
