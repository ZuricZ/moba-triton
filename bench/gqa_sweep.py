"""Measures what grouped-query attention buys, holding query work fixed.

H_q is pinned at 8 throughout, so the arithmetic -- the number of query-key
pairs scored -- is identical in every row. Only the number of distinct K/V
heads changes, which is exactly the memory term. That isolates the intensity
question from any change in the problem being solved.

The `expanded` column is the cleanest A/B available: the same shared K/V
materialized back out to 8 heads and run through the ordinary multi-head path.
Identical mathematics, identical routing, only the layout differs.

PARTIAL BUFFERS ARE PREALLOCATED AND REUSED. A first attempt timed
`moba_forward` directly and produced incoherent numbers -- the expanded runs
came out faster than the identical-math narrow ones. `moba_forward` allocates
`part_acc` on every call, 16-64 MB here, and `attend_partials`' own docstring
records that on Windows WDDM a `cudaMalloc` goes through the display driver and
costs milliseconds. That was being measured instead of the kernel.
"""

import torch
import triton

from moba.config import MoBAConfig
from moba.heads import expand_kv
from moba.kernels.routing import build_csc_fused
from moba.kernels.moba_fwd import (
    attend_partials,
    combine_partials,
    empty_partials,
)

H_Q = 8
D = 64
CFG = MoBAConfig(block_size=128, top_k=8, dtype=torch.bfloat16)


def make(s, h_kv, seed=0):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    q = torch.randn(1, s, H_Q, D, generator=gen, device="cuda",
                    dtype=torch.bfloat16)
    k = torch.randn(1, s, h_kv, D, generator=gen, device="cuda",
                    dtype=torch.bfloat16)
    v = torch.randn(1, s, h_kv, D, generator=gen, device="cuda",
                    dtype=torch.bfloat16)
    return q, k, v


def bench(fn):
    """Median over 200 reps. Median, not mean: WDDM produces occasional
    multi-millisecond stalls that a mean would absorb into the signal."""
    return triton.testing.do_bench(fn, warmup=50, rep=200,
                                   return_mode="median")


def measure(q, k, v, index):
    """Times the attention stage and the full forward, allocation excluded."""
    parts = empty_partials(q.shape, index.max_slots, q.device, q.dtype)

    def attend():
        attend_partials(q, k, v, CFG, index, partials=parts)

    def full():
        attend_partials(q, k, v, CFG, index, partials=parts)
        combine_partials(parts[0], parts[1], parts[2], q.dtype)

    t_attend, t_full = bench(attend), bench(full)
    del parts
    return t_attend, t_full


print(f"cfg: block_size={CFG.block_size} top_k={CFG.top_k} "
      f"H_q={H_Q} head_dim={D} bf16   (times in ms, median of 200)\n")

for s in (2048, 4096, 8192):
    print(f"--- seq_len {s} " + "-" * 66)
    print(f"{'H_kv':>5} {'group':>6} | {'attn':>8} {'attn exp':>9} "
          f"{'gain':>7} | {'fwd':>8} {'fwd exp':>9} {'gain':>7} | "
          f"{'K/V MB':>7} {'F/B':>7}")
    for h_kv in (8, 4, 2, 1):
        try:
            q, k, v = make(s, h_kv)
            index = build_csc_fused(q, k, CFG)
            a_n, f_n = measure(q, k, v, index)

            kw, vw = expand_kv(k, H_Q).contiguous(), expand_kv(v, H_Q).contiguous()
            iw = build_csc_fused(q, kw, CFG)
            a_w, f_w = measure(q, kw, vw, iw)

            live = int((index.col_nnz > 0).sum())
            kvb = live * CFG.block_size * D * 2 * 2
            pairs = int(index.col_nnz.sum()) * CFG.block_size
            flops = 4.0 * pairs * D
            total = kvb + int(index.col_nnz.sum()) * D * 2 \
                + H_Q * s * index.max_slots * D * 2

            print(f"{h_kv:>5} {H_Q // h_kv:>6} | {a_n:>8.3f} {a_w:>9.3f} "
                  f"{a_w / a_n:>6.3f}x | {f_n:>8.3f} {f_w:>9.3f} "
                  f"{f_w / f_n:>6.3f}x | {kvb / 2**20:>7.2f} "
                  f"{flops / total:>7.1f}")
            del q, k, v, kw, vw, index, iw
            torch.cuda.empty_cache()
        except torch.OutOfMemoryError:
            print(f"{h_kv:>5} {H_Q // h_kv:>6} |   OOM")
            torch.cuda.empty_cache()
    print()
