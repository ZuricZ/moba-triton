#!/usr/bin/env bash
#
# Brings a freshly rented Linux GPU box from bare clone to a running
# comparison. Written for jarvislabs.ai / Lambda / RunPod style images: Ubuntu,
# a CUDA driver already installed, root in a container.
#
# WHY THIS EXISTS. Setting this project up the first time cost most of a night,
# and almost none of it was interesting: the venv location, the cu126 torch
# index, MAX_JOBS sized to RAM, building outside a 9p mount, flash-moba's
# setup_requires needing packaging/psutil/ninja because the venv has no pip.
# On a rented box that rediscovery is billed by the hour.
#
# USAGE
#     ./scripts/bootstrap_remote.sh              # everything except flash-attn
#     ./scripts/bootstrap_remote.sh --all        # also build flash-attn (slow)
#     ./scripts/bootstrap_remote.sh --bench      # just re-run the comparison
#
# flash-attn's two incompatible generations both have a MoBA reference that
# needs them, so the version is a parameter and only one is live at a time:
#
#     ./scripts/bootstrap_remote.sh --all --fa=2.8.3   # build the other one
#     ./scripts/bootstrap_remote.sh --use-fa=2.8.3     # switch to it (instant)
#     ./scripts/bootstrap_remote.sh --use-fa=2.6.3     # and back
#
# Every step is idempotent; re-running skips what is already done.

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_ROOT="${BUILD_ROOT:-$HOME/build}"
WITH_FLASH_ATTN=0
BENCH_ONLY=0
SWITCH_ONLY=0
# Two flash-attn versions are needed and they are mutually exclusive; see the
# table in step 4. This names which one the next command acts on.
FA_VERSION="${FA_VERSION:-2.6.3}"

for arg in "$@"; do
  case "$arg" in
    --all)   WITH_FLASH_ATTN=1 ;;
    --bench) BENCH_ONLY=1 ;;
    --fa=*)  FA_VERSION="${arg#--fa=}" ;;
    --use-fa=*) FA_VERSION="${arg#--use-fa=}"; SWITCH_ONLY=1 ;;
    *) echo "unknown argument: $arg"; exit 2 ;;
  esac
done

say() { printf '\n\033[1m== %s\033[0m\n' "$*"; }

# The venv's site-packages, resolved rather than globbed at the point of
# use. See the .pth writes below for why that distinction cost a build.
site_packages() {
  local dir
  dir="$(echo "$REPO"/.venv/lib/python3.*/site-packages)"
  [ -d "$dir" ] || { echo "no site-packages under $REPO/.venv" >&2; exit 1; }
  printf '%s' "$dir"
}

# Point the venv at one of the built flash-attn trees.
#
# flash-attn is never installed into the venv -- it is built in place and
# reached through a .pth file, which is nothing more than "add this directory
# to sys.path". So swapping versions is a 40-byte write, not a reinstall, and
# torch and triton stay bit-identical across the two runs. That matters:
# triton is the compiler under test, so a second venv resolving it differently
# would make the `ours` column non-comparable between them.
#
# Both trees build a module called flash_attn_2_cuda, so exactly one farm may
# be on sys.path at a time. One .pth, rewritten, enforces that by construction.
use_flash_attn() {
  local version="$1" link="$BUILD_ROOT/site-fa-$1"
  [ -d "$link" ] || { echo "flash-attn $version not built: no $link" >&2
                      echo "  build it with: $0 --all --fa=$version" >&2; exit 1; }
  echo "$link" > "$(site_packages)/flash-attn-local.pth"
  echo "flash_attn -> $link"
}

# --use-fa= does one thing and exits, so it costs a second rather than a
# machine probe. It is the command you run between the two sweeps.
if [ "$SWITCH_ONLY" -eq 1 ]; then
  say "flash-attn $FA_VERSION"
  use_flash_attn "$FA_VERSION"
  exit 0
fi

# ---------------------------------------------------------------------------
# 0. What are we on?
# ---------------------------------------------------------------------------

say "machine"
nvidia-smi --query-gpu=name,memory.total,compute_cap,driver_version \
           --format=csv,noheader || { echo "no nvidia-smi -- wrong box"; exit 1; }
echo "CPU cores : $(nproc)"
echo "RAM       : $(free -g | awk '/^Mem:/ {print $2}') GB"

# nvcc is frequently installed but off PATH, which makes it look absent.
for candidate in /usr/local/cuda /usr/local/cuda-12.*; do
  if [ -x "$candidate/bin/nvcc" ]; then
    export CUDA_HOME="$candidate"
    export PATH="$CUDA_HOME/bin:$PATH"
    break
  fi
done
echo "CUDA_HOME : ${CUDA_HOME:-<not found>}"
[ -n "${CUDA_HOME:-}" ] || echo "  (no nvcc: flash-moba and flash-attn cannot build)"

# Build for exactly this card. flash-moba ships archs 80;90;100;120, so sm_80
# and sm_90 need no override -- but sm_86 (a 30-series laptop) does, and
# setting it unconditionally costs nothing and avoids compiling five
# architectures we will never run.
CC="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '.')"
export TORCH_CUDA_ARCH_LIST="${CC:0:1}.${CC:1}"
export FLASH_MOBA_CUDA_ARCHS="$CC"
echo "building for: sm_$CC (TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST)"

# nvcc on CUTLASS templates wants ~4 GB per process, and flash-attn's backward
# units want 8-10 GB. MAX_JOBS=4 on a 15 GB box exhausted RAM and swap in 90
# seconds. Size it off memory, not cores.
#
# But ask the cgroup, not `free`. Inside a container `free` reports the
# *host's* memory: this rented box reports 503 GB and is actually limited to
# 112 GiB, a 4.5x overestimate. That put MAX_JOBS at the core count, and the
# OOM killer stopped the flash-attn 2.8.3 build at unit 18 of 73. The 2.6.3
# build had survived the same wrong number, so the bug only surfaced when the
# work per process went up.
mem_limit_gb() {
  local bytes
  for f in /sys/fs/cgroup/memory.max /sys/fs/cgroup/memory/memory.limit_in_bytes; do
    if [ -r "$f" ]; then
      bytes="$(cat "$f")"
      # "max" means unlimited, and cgroup v1 spells that as a huge sentinel.
      if [ "$bytes" != "max" ] && [ "$bytes" -lt 1000000000000 ] 2>/dev/null; then
        echo $(( bytes / 1024 / 1024 / 1024 )); return
      fi
    fi
  done
  free -g | awk '/^Mem:/ {print $2}'
}
RAM_GB="$(mem_limit_gb)"

# 12 GB per job, not 8. flash-attn from 2.7 hardcodes its gencode flags and
# ignores TORCH_CUDA_ARCH_LIST, so every translation unit is compiled for
# sm_80 *and* sm_90 whatever we ask for -- roughly twice the peak memory of a
# single-arch build. NVCC_THREADS trims what is left; their setup.py reads it
# and defaults to 4.
export NVCC_THREADS="${NVCC_THREADS:-2}"
export MAX_JOBS="${MAX_JOBS:-$(( RAM_GB / 12 > 0 ? RAM_GB / 12 : 1 ))}"
[ "$MAX_JOBS" -gt "$(nproc)" ] && export MAX_JOBS="$(nproc)"
echo "MAX_JOBS  : $MAX_JOBS (from ${RAM_GB} GB usable, not core count)"
echo "NVCC_THREADS: $NVCC_THREADS"

# ---------------------------------------------------------------------------
# 1. Python environment
# ---------------------------------------------------------------------------

if [ "$BENCH_ONLY" -eq 0 ]; then
  say "uv + venv"
  if ! command -v uv > /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
  fi
  export PATH="$HOME/.local/bin:$PATH"
  cd "$REPO"

  # No UV_PROJECT_ENVIRONMENT here on purpose. That variable exists in this
  # project to keep the venv off WSL's 9p bridge to a Windows drive, where
  # uv cannot hard-link from its cache and every file gets copied (uv sync:
  # 11 min vs 1.5 s). A rented Linux box has a real filesystem and the
  # in-repo default is correct.
  uv sync
  uv run python - <<'PYEOF'
import torch
print("torch     :", torch.__version__)
print("triton    :", __import__("triton").__version__)
print("cuda ok   :", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
print("capability:", torch.cuda.get_device_capability(0))
props = torch.cuda.get_device_properties(0)
print("SMs       :", props.multi_processor_count)
print("shared/blk:", getattr(props, "shared_memory_per_block_optin", "?"), "B opt-in")
PYEOF
fi

# ---------------------------------------------------------------------------
# 2. Vendored references
# ---------------------------------------------------------------------------

if [ "$BENCH_ONLY" -eq 0 ]; then
  say "references"
  # --recursive matters: flash-moba vendors CUTLASS as a submodule and cannot
  # build without it. The default fetch skips submodules because reading the
  # source does not need 151 MB of headers.
  bash "$REPO/scripts/fetch_references.sh" --recursive
fi

# ---------------------------------------------------------------------------
# 3. flash-moba (the performance target)
# ---------------------------------------------------------------------------

if [ "$BENCH_ONLY" -eq 0 ] && [ -n "${CUDA_HOME:-}" ]; then
  say "flash-moba"
  mkdir -p "$BUILD_ROOT"
  if [ ! -d "$BUILD_ROOT/flash-moba" ]; then
    cp -r "$REPO/references/flash-moba" "$BUILD_ROOT/flash-moba"
  fi
  # setuptools resolves setup_requires by shelling to `python -m pip`, which
  # a uv-managed venv does not have. Install them first or the build dies
  # before compiling anything.
  uv pip install --python "$REPO/.venv/bin/python" packaging psutil ninja

  cd "$BUILD_ROOT/flash-moba"
  if ! ls flash_moba_cuda*.so > /dev/null 2>&1; then
    time "$REPO/.venv/bin/python" setup.py build_ext --inplace
  else
    echo "already built"
  fi

  # Link only the package and its .so. Putting the build tree itself on
  # sys.path would also expose its tests/ and benchmarks/, and `tests` would
  # then shadow ours depending on path order.
  LINK="$BUILD_ROOT/site-fm"
  mkdir -p "$LINK"
  ln -sfn "$BUILD_ROOT/flash-moba/flash_moba" "$LINK/flash_moba"
  for so in "$BUILD_ROOT"/flash-moba/flash_moba_cuda*.so; do
    [ -e "$so" ] && ln -sfn "$so" "$LINK/$(basename "$so")"
  done
  # Expand the directory, then append the filename. Globbing the whole
  # path cannot work: a glob only matches paths that already exist, and
  # the .pth is the file being created -- so the pattern matched nothing,
  # bash used it literally, and set -e killed the run after a ten-minute
  # build had already succeeded.
  echo "$LINK" > "$(site_packages)/flash-moba-local.pth"
  echo "linked flash_moba"
fi

# ---------------------------------------------------------------------------
# 4. flash-attn, only for triton_moba (optional, slow)
# ---------------------------------------------------------------------------

if [ "$WITH_FLASH_ATTN" -eq 1 ] && [ -n "${CUDA_HOME:-}" ]; then
  say "flash-attn $FA_VERSION"
  # The version is a parameter because the two MoBA references disagree about
  # flash-attn's API and no single build satisfies both:
  #
  #   2.6.x   window_size=(l, r) tuple   fwd returns 8   bwd returns 4
  #   >= 2.7  window_size_left/right     fwd returns 4   bwd returns 1
  #
  #   moonshot_efficient  unpacks 8  ->  needs 2.6.3
  #   triton_moba         unpacks 4  ->  needs >= 2.7
  #
  # FA3 and FA4 are not options on this hardware. FA3 is sm_90a Hopper only;
  # FA4 is CuTeDSL under flash_attn/cute/ with kernels for sm_90/100/120 and a
  # runtime capability check that has no branch for 8. Both leave sm_80 out.
  SRC="$BUILD_ROOT/flash-attn-$FA_VERSION"
  # Adopt the unversioned tree an older revision of this script built. It was
  # always 2.6.3, and re-cloning to rebuild it would cost half an hour.
  if [ "$FA_VERSION" = "2.6.3" ] && [ ! -d "$SRC" ] && [ -d "$BUILD_ROOT/flash-attn" ]; then
    mv "$BUILD_ROOT/flash-attn" "$SRC"
  fi
  [ -d "$SRC" ] || git clone --depth 1 --branch "v$FA_VERSION" \
      https://github.com/Dao-AILab/flash-attention.git "$SRC"
  cd "$SRC"
  export FLASH_ATTENTION_FORCE_BUILD=TRUE
  # 94 .cu files. Untested against torch 2.13 -- if their C++ API drifted this
  # fails early rather than after hours, which is the good outcome.
  if ! ls flash_attn_2_cuda*.so > /dev/null 2>&1; then
    time "$REPO/.venv/bin/python" setup.py build_ext --inplace
  else
    echo "already built"
  fi

  # One farm per version, so a rebuilt tree never half-shadows the other.
  LINK="$BUILD_ROOT/site-fa-$FA_VERSION"
  mkdir -p "$LINK"
  ln -sfn "$SRC/flash_attn" "$LINK/flash_attn"
  for so in "$SRC"/flash_attn_2_cuda*.so; do
    [ -e "$so" ] && ln -sfn "$so" "$LINK/$(basename "$so")"
  done
  use_flash_attn "$FA_VERSION"
fi

# ---------------------------------------------------------------------------
# 5. Correctness, then the comparison
# ---------------------------------------------------------------------------

cd "$REPO"

if [ "$BENCH_ONLY" -eq 0 ]; then
  say "tests"
  # Correctness before speed. A fast wrong kernel is not a result, and the
  # suite is where a new card's different tile choices would surface.
  uv run python -m pytest -q -m "not slow"
fi

say "which backends are live"
uv run python - <<'PYEOF'
from implementations import all_backends
import torch
from moba.config import MoBAConfig
cfg = MoBAConfig(block_size=128, top_k=8, dtype=torch.bfloat16)
for b in all_backends():
    ok, why = b.is_available(cfg)
    print(f"  {b.name:<14} {'yes' if ok else 'no ':<4} {why}")
PYEOF

say "comparison"
# Sequence lengths and the memory budget size themselves off the card; clocks
# are pinned if the container allows it, which makes the thermal protocol a
# no-op. Override the sweep with MOBA_SEQ_LENS=8192,16384,32768.
uv run python -m bench.compare

say "done -- results/ holds the CSV and plots"
echo "copy them off before the box is destroyed:"
echo "  scp -r <user>@<host>:$REPO/results ."
