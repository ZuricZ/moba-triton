#!/usr/bin/env bash
# Clones the reference implementations into references/ (gitignored, read-only).
#
# EVERY REPO IS PINNED TO A COMMIT. Until 2026-09-07 this cloned --depth 1 from
# whatever HEAD happened to be, which meant a benchmark run today and one next
# month could compare against different FlashMoBA. For a project whose entire
# output is a comparison table, "which version did you measure?" has to have an
# answer, and it belongs in version control rather than in someone's shell
# history. These are the commits every number in docs/ was taken against.
#
# To move to a newer upstream: update the pin, re-run the comparison, and say
# in the commit message which numbers changed. Never bump a pin silently -- the
# point of the pin is that the table and the code move together.
#
# Submodules are NOT fetched by default. flash-moba vendors CUTLASS, which is
# 151 MB and unnecessary for reading its kernel source. Pass --recursive when
# building the extension.
#
#     ./scripts/fetch_references.sh              # read the source
#     ./scripts/fetch_references.sh --recursive  # build flash-moba
#     PIN=off ./scripts/fetch_references.sh      # track upstream HEAD instead
set -euo pipefail

RECURSE=""
if [ "${1:-}" = "--recursive" ]; then
  RECURSE="--recurse-submodules"
fi
PIN="${PIN:-on}"

mkdir -p references
cd references

# repo -> commit measured. Dates are the commit's, not ours.
clone() {
  local url=$1 dir=$2 commit=$3
  if [ -d "$dir" ]; then
    echo "== $dir already present, skipping"
    return
  fi
  if [ "$PIN" = "off" ]; then
    echo "== cloning $dir at HEAD (PIN=off -- results will not be comparable)"
    git clone --depth 1 $RECURSE "$url" "$dir"
    return
  fi
  # A shallow clone cannot check out an arbitrary commit, so fetch just that
  # one object rather than the full history: same bandwidth as --depth 1, and
  # it lands on the commit we mean.
  echo "== cloning $dir at ${commit:0:12}"
  git init --quiet "$dir"
  git -C "$dir" remote add origin "$url"
  git -C "$dir" fetch --quiet --depth 1 origin "$commit"
  git -C "$dir" checkout --quiet FETCH_HEAD
  if [ -n "$RECURSE" ]; then
    git -C "$dir" submodule update --quiet --init --recursive --depth 1
  fi
}

#      url                                                          dir                      commit                                     measured
clone https://github.com/mit-han-lab/flash-moba.git                 flash-moba               39d9ac043b271d046a2181a9991e99a26b67bca1  # 2025-11-19
clone https://github.com/ReyJerry/Triton-MoBA.git                   Triton-MoBA              7c44c895d087521702c549fa02b5e5e9b5d383e3  # 2025-12-18
clone https://github.com/MoonshotAI/MoBA.git                        MoBA                     b5d58363311d3ca946f1ec444182727c15e338b5  # 2025-04-03
clone https://github.com/fla-org/native-sparse-attention.git        native-sparse-attention  bd67af59b90afa34b25f61d2922e612d10dba3bd  # 2025-03-19
clone https://github.com/Relaxed-System-Lab/Flash-Sparse-Attention.git Flash-Sparse-Attention 7ff144fd7ff485dc4220d439f31cc1708b64fef3  # 2025-09-29

echo
echo "Done. references/ is gitignored and read-only -- never edit these."
echo "Pinned commits are in this script; PIN=off tracks HEAD instead."
