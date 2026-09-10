"""Which half of FlashMoBA are we beating -- the gate, or the attention?

Their pipeline splits exactly like ours: flash_topk_varlen_func produces the
CSC routing, flash_moba_attn_varlen_func consumes it. Ours splits into
build_csc_fused and attend+combine. Timing the halves separately turns "3x
faster" into a statement about which stage.
"""
import torch, triton.testing
from bench.suite import cool_down_device, warm_up_device
from moba.config import MoBAConfig
from moba.kernels.routing import build_csc_fused
from moba.kernels.moba_fwd import attend_partials, combine_partials
from flash_moba import flash_topk_varlen_func, flash_moba_attn_varlen_func
from flash_moba.flash_moba_interface import decide_lg_block_m

CFG = MoBAConfig(block_size=128, top_k=8, dtype=torch.bfloat16)

def bench(fn):
    fn(); cool_down_device(); warm_up_device()
    return min(triton.testing.do_bench(fn, quantiles=[0.5,0.2,0.8])[0] for _ in range(3))

print(f"  {'shape':<22} {'stage':<22} {'ours':>8} {'theirs':>8} {'ratio':>7}")
print("  " + "-"*72)
for B, S, H, D in ((1, 8192, 4, 64), (1, 8192, 32, 128)):
    g = torch.Generator(device="cuda").manual_seed(0)
    q, k, v = (torch.randn(B, S, H, D, generator=g, device="cuda", dtype=CFG.dtype) for _ in range(3))
    qf, kf, vf = (t.reshape(B*S, H, D) for t in (q, k, v))
    cu = torch.arange(0, (B+1)*S, S, dtype=torch.int32, device="cuda")

    our_gate = bench(lambda: build_csc_fused(q, k, CFG))
    their_gate = bench(lambda: flash_topk_varlen_func(qf, kf, cu, cu, S, S, CFG.top_k, CFG.block_size, True))

    index = build_csc_fused(q, k, CFG)
    off, nnz, rows = flash_topk_varlen_func(qf, kf, cu, cu, S, S, CFG.top_k, CFG.block_size, True)
    lg_m = decide_lg_block_m(CFG.top_k, CFG.block_size, S, True)

    our_attn = bench(lambda: combine_partials(*attend_partials(q, k, v, CFG, index), q.dtype))
    their_attn = bench(lambda: flash_moba_attn_varlen_func(qf, kf, vf, cu, cu, S, S, off, nnz, rows, lg_m, CFG.block_size, 0.0, causal=True))

    shape = f"B{B} S{S} H{H} D{D}"
    print(f"  {shape:<22} {'gate + routing':<22} {our_gate:>8.3f} {their_gate:>8.3f} {their_gate/our_gate:>6.2f}x")
    print(f"  {'':<22} {'attention + combine':<22} {our_attn:>8.3f} {their_attn:>8.3f} {their_attn/our_attn:>6.2f}x")
    print(f"  {'':<22} {f'their tile lg_block_m={lg_m}':<22}")
    del q, k, v, qf, kf, vf, index, off, nnz, rows
    torch.cuda.empty_cache()
