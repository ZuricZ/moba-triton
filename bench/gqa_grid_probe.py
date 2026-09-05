"""Is the GQA penalty grid starvation? Add batch and see if it vanishes.

One program per (block, batch, kv head). Grouped-query attention divides the
kv-head axis, so the grid shrinks by the group size. On 20 SMs a small grid
starves. Batch multiplies the grid back without touching anything else, so if
the hypothesis holds the penalty must disappear as batch grows.
"""
import torch, triton
from moba.config import MoBAConfig
from moba.gqa import expand_kv
from moba.kernels.moba_fwd import attend_partials, empty_partials
from moba.kernels.routing import build_csc_fused

D, CFG = 64, MoBAConfig(block_size=128, top_k=8, dtype=torch.bfloat16)
H_Q = 8

def run(b, s, h_kv):
    g = torch.Generator(device="cuda").manual_seed(0)
    q = torch.randn(b, s, H_Q, D, generator=g, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(b, s, h_kv, D, generator=g, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(b, s, h_kv, D, generator=g, device="cuda", dtype=torch.bfloat16)
    def t(kk, vv):
        i = build_csc_fused(q, kk, CFG)
        p = empty_partials(q.shape, i.max_slots, q.device, q.dtype)
        ms = triton.testing.do_bench(
            lambda: attend_partials(q, kk, vv, CFG, i, partials=p),
            warmup=50, rep=200, return_mode="median")
        return ms, i.num_blocks * b * kk.shape[2]   # token-major head axis
    narrow, gn = t(k, v)
    wide, gw = t(expand_kv(k, H_Q).contiguous(), expand_kv(v, H_Q).contiguous())
    print(f"  batch={b:>2}  grid {gn:>4} vs {gw:>4}   "
          f"narrow {narrow:>7.3f}  expanded {wide:>7.3f}   gain {wide/narrow:>6.3f}x")
    del q, k, v; torch.cuda.empty_cache()

print("S=2048, H_kv=1 (group 8) -- 20 SMs on this card")
for b in (1, 2, 4, 8, 16):
    try: run(b, 2048, 1)
    except torch.OutOfMemoryError: print(f"  batch={b:>2}  OOM"); torch.cuda.empty_cache()
