"""Is the sparse-backend comparison fair, or is the shape doing the work?

Beating FlashMoBA's CUDA kernel -- and Triton-MoBA and MoonshotAI's fused path
by even more -- deserves scrutiny before it is written down. Two ways the
result could be an artefact rather than a finding:

1. A backend is not computing the same thing. If it silently produces garbage
   on this configuration, the timing is meaningless.
2. The shape is unfair. The sweep uses B=1, H=4, D=64 -- a small workload.
   FlashMoBA's own benchmarks use realistic model shapes, and its
   `decide_lg_block_m` picks query tiles of 512 to 1024. At H=4 on 108 SMs
   that may leave most of the GPU idle for reasons that have nothing to do with
   the algorithm.

The two flash-attn-dependent backends are mutually exclusive: `triton_moba`
needs flash-attn >= 2.7 and `moonshot_efficient` needs 2.6.x. This script is
therefore run once per flash-attn build; the backend that needs the other
build prints FAILED and the two runs are composed.
"""

import torch
import triton.testing

from bench.suite import cool_down_device, warm_up_device
from implementations import get_backend
from moba.config import MoBAConfig
from moba.naive import moba_attention_naive

CFG = MoBAConfig(block_size=128, top_k=8, dtype=torch.bfloat16)

# The sparse competitors, measured against `ours` and the oracle. `sdpa_dense`
# and `fa2_minimal` are dense baselines rather than MoBA implementations, so a
# fair-vs-us question does not apply to them and they stay out of this file.
SPARSE_BACKENDS = ("flash_moba", "triton_moba", "moonshot_efficient")


def make(batch, seq_len, heads, head_dim):
    g = torch.Generator(device="cuda").manual_seed(0)
    return [torch.randn(batch, seq_len, heads, head_dim, generator=g,
                        device="cuda", dtype=CFG.dtype) for _ in range(3)]


def time_backend(name, q, k, v):
    backend = get_backend(name)
    fn = lambda: backend.forward(q, k, v, CFG)          # noqa: E731
    fn()
    cool_down_device()
    warm_up_device()
    return min(triton.testing.do_bench(fn, quantiles=[0.5, 0.2, 0.8])[0]
               for _ in range(3))


print("=" * 72)
print("1. CORRECTNESS -- does each sparse backend agree with the oracle?")
print("=" * 72)
for seq_len in (1024, 4096):
    q, k, v = make(1, seq_len, 4, 64)
    ref = moba_attention_naive(q.float(), k.float(), v.float(), CFG)
    scale = ref.abs().max().item()
    row = [f"  S={seq_len:<6}"]
    for name in ("ours",) + SPARSE_BACKENDS:
        try:
            out = get_backend(name).forward(q, k, v, CFG).float()
            err = (out - ref).abs().max().item() / scale
            row.append(f"{name} {err:.4f}")
        except Exception as exc:                        # noqa: BLE001
            row.append(f"{name} FAILED {type(exc).__name__}")
    print("   ".join(row))
print("  (relative to the PyTorch oracle; routing flips dominate, so equal")
print("   error means equally valid block choices, not identical output)")

print()
print("=" * 72)
print("2. SHAPE -- does the gap survive a realistic model shape?")
print("=" * 72)
shapes = [
    (1, 8192, 4, 64, "sweep shape"),
    (1, 8192, 32, 128, "GPT-3 13B-ish"),
    (4, 8192, 32, 128, "same, batch 4"),
    (1, 16384, 32, 128, "long, wide"),
]
# Competitor columns are the ratio to `ours` (x slower), not raw ms: the
# question this section asks is whether the *gap* survives, and a ratio is
# what answers it. '-' means the backend is unavailable in this flash-attn
# build and will be filled in by the other pass.
print(f"  {'shape':<24} {'ours':>9} "
      + " ".join(f"{name:>20}" for name in SPARSE_BACKENDS))
print("  " + "-" * 72)
print("  (competitor columns are the ratio to ours, i.e. x slower)")
for batch, seq_len, heads, head_dim, label in shapes:
    try:
        q, k, v = make(batch, seq_len, heads, head_dim)
    except torch.OutOfMemoryError:
        print(f"  {label:<24} OOM building inputs")
        continue
    results = {}
    for name in ("ours",) + SPARSE_BACKENDS:
        try:
            results[name] = time_backend(name, q, k, v)
        except Exception as exc:                        # noqa: BLE001
            results[name] = None
            print(f"  {label:<24} {name}: {type(exc).__name__}")
    del q, k, v
    torch.cuda.empty_cache()
    if not results.get("ours"):
        continue
    desc = f"B{batch} S{seq_len} H{heads} D{head_dim}"
    line = f"  {desc:<24} {results['ours']:>9.3f}"
    for name in SPARSE_BACKENDS:
        val = results.get(name)
        cell = f"{val / results['ours']:.2f}x" if val is not None else "-"
        line += f" {cell:>20}"
    line += f"  {label}"
    print(line)
