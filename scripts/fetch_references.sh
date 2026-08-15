#!/usr/bin/env bash
# Clones the reference implementations into references/ (gitignored, read-only).
#
# Submodules are deliberately NOT fetched. flash-moba vendors CUTLASS, which is
# large and which we do not need in order to read its kernel source. Pass
# --recursive if a build is ever attempted.
set -euo pipefail

RECURSE=""
if [ "${1:-}" = "--recursive" ]; then
  RECURSE="--recurse-submodules"
fi

mkdir -p references
cd references

clone() {
  local url=$1 dir=$2
  if [ -d "$dir" ]; then
    echo "== $dir already present, skipping"
  else
    echo "== cloning $dir"
    git clone --depth 1 $RECURSE "$url" "$dir"
  fi
}

clone https://github.com/mit-han-lab/flash-moba.git                    flash-moba
clone https://github.com/ReyJerry/Triton-MoBA.git                      Triton-MoBA
clone https://github.com/MoonshotAI/MoBA.git                           MoBA
clone https://github.com/fla-org/native-sparse-attention.git           native-sparse-attention
clone https://github.com/Relaxed-System-Lab/Flash-Sparse-Attention.git Flash-Sparse-Attention

echo
echo "Done. references/ is gitignored and read-only -- never edit these."
