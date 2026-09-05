"""Measures what packing a ragged batch buys, against a baseline that also packs.

THE PREDICTION, STATED BEFORE THE NUMBERS.

Padding waste is `1 - mean(len)/max(len)`. That is a property of the *data*,
not of the kernel: a batch whose sequences are all the same length has nothing
to save, and no amount of kernel work will find any. So the honest question is
not "how much faster is varlen" but "how much of the raggedness does varlen
actually convert into time".

The second prediction is a correction to one the plan made. The plan expected
the launch grid to *grow*, on the theory that a ragged batch yields more real
blocks than a padded one yields useful ones -- which, per
`docs/gqa-measurements.md`, would matter on 20 SMs. Reading the launch shows
otherwise. `attend_partials` launches `(index.num_blocks, n_seq * h_kv)`, and
`num_blocks` is `cdiv(max_seqlen, block_size)` packed and `cdiv(S, block_size)`
padded -- the same number, because we pack the data and *pad the index*. The
grid is identical. What changes is that a short sequence's high-numbered
columns are empty, and their programs return immediately.

> Packing does not shrink the launch. It empties programs.

So the gain should be almost entirely the memory term, and `part_acc` is the
term to watch: it is the largest buffer in the design, and it is sized by token
count, so it shrinks by exactly the raggedness ratio.

BASELINE. Compared against dense SDPA *packed* -- a jagged nested tensor, which
torch runs on the fused variable-length FlashAttention kernel. Comparing against
a padded baseline would credit us with padding the baseline wasted.

The baseline's K and V are expanded to `H_Q` first. Nested SDPA has no
`enable_gqa`: its preprocessing raises "This path is currently not implemented
for jagged layout NT" when the head counts differ. Expanding is what a dense
grouped-query baseline does regardless, and the expansion is outside the timed
region -- so the baseline is charged for the extra *loads*, which it genuinely
performs, but not for a copy our path never makes.

PARTIAL BUFFERS ARE PREALLOCATED. Same lesson as `bench/gqa_sweep.py`: timing
`moba_forward` directly measures the WDDM allocator, not the kernel.
"""

import torch
import triton

from moba.backends.base import get_backend
from moba.config import MoBAConfig
from moba.heads import expand_kv
from moba.kernels.routing import build_csc_fused
from moba.kernels.moba_fwd import (
    attend_partials,
    combine_partials,
    empty_partials,
)
from moba.packing import pack

H_Q, H_KV, D = 8, 2, 64
CFG = MoBAConfig(block_size=128, top_k=8, dtype=torch.bfloat16)
DTYPE = torch.bfloat16
MAXLEN = 2048

# Three shapes of raggedness, at a fixed longest sequence so the padded
# rectangle is the same in every row and only the real token count moves.
def lengths(kind, b, seed=0):
    if kind == "uniform":
        return [MAXLEN] * b
    if kind == "skewed":                    # one long document, rest short
        return [MAXLEN] + [256] * (b - 1)
    gen = torch.Generator().manual_seed(seed)
    # Multiples of 128 keep the padded and packed block counts comparable;
    # raggedness, not a ragged final block, is what this measures.
    steps = torch.randint(2, MAXLEN // 128 + 1, (b,), generator=gen)
    steps[0] = MAXLEN // 128                # pin the longest so max is fixed
    return (steps * 128).tolist()


def padded_batch(lens, seed=0):
    """[B, H, MAXLEN, D] plus the mask that says which tokens are real."""
    gen = torch.Generator(device="cuda").manual_seed(seed)
    b = len(lens)

    def mk(h):
        return torch.randn(b, MAXLEN, h, D, generator=gen, device="cuda",
                           dtype=DTYPE)

    mask = torch.zeros(b, MAXLEN, dtype=torch.int32, device="cuda")
    for i, n in enumerate(lens):
        mask[i, :n] = 1
    return mk(H_Q), mk(H_KV), mk(H_KV), mask


def bench(fn):
    """Median over 100 reps -- WDDM stalls that a mean would absorb."""
    return triton.testing.do_bench(fn, warmup=25, rep=100,
                                   return_mode="median")


def moba(q, k, v, index):
    """Times attention + combine with allocation excluded."""
    parts = empty_partials(q.shape, index.max_slots, q.device, q.dtype)

    def full():
        attend_partials(q, k, v, CFG, index, partials=parts)
        combine_partials(parts[0], parts[1], parts[2], q.dtype)

    t = bench(full)
    del parts
    torch.cuda.empty_cache()
    return t


dense = get_backend("sdpa_dense")

print(f"cfg: block_size={CFG.block_size} top_k={CFG.top_k} H_q={H_Q} "
      f"H_kv={H_KV} head_dim={D} bf16  max_seqlen={MAXLEN}")
print("times in ms, median of 100. 'waste' is 1 - mean(len)/max(len).\n")

hdr = (f"{'kind':>8} {'B':>3} {'waste':>6} | {'tok pad':>8} {'tok pk':>8} "
       f"{'acc pad':>8} {'acc pk':>8} | {'grid':>6} {'live pad':>8} "
       f"{'live pk':>8} {'pair pd':>8} {'pair pk':>8} | "
       f"{'moba pad':>9} {'moba pk':>8} {'gain':>7} | "
       f"{'dense pk':>9} {'vs dense':>8}")

for b in (4, 8, 16):
    print("-" * len(hdr))
    print(hdr)
    for kind in ("uniform", "mixed", "skewed"):
        lens = lengths(kind, b, seed=b)
        waste = 1.0 - sum(lens) / (len(lens) * MAXLEN)
        try:
            q, k, v, mask = padded_batch(lens, seed=b)
            pq, cu, mx = pack(q, mask)
            pk_, _, _ = pack(k, mask)
            pv, _, _ = pack(v, mask)

            i_pad = build_csc_fused(q, k, CFG)
            i_pk = build_csc_fused(pq, pk_, CFG, cu_seqlens=cu, max_seqlen=mx)

            grid = i_pad.num_blocks * b * H_KV
            assert grid == i_pk.num_blocks * b * H_KV, "grid must be identical"
            live_pad = int((i_pad.col_nnz > 0).sum())
            live_pk = int((i_pk.col_nnz > 0).sum())
            # Query-key pairs actually scored. Not proportional to tokens: a
            # sequence with fewer than `top_k` blocks cannot select `top_k` of
            # them, so short sequences are cheaper per token as well as fewer.
            pair_pad = int(i_pad.col_nnz.sum()) * CFG.block_size
            pair_pk = int(i_pk.col_nnz.sum()) * CFG.block_size

            def acc_mb(t, idx):
                return (t.shape[0] * H_Q * t.shape[1] * idx.max_slots * D
                        * t.element_size() / 2**20)

            t_pad, t_pk = moba(q, k, v, i_pad), moba(pq, pk_, pv, i_pk)
            ek = expand_kv(pk_, H_Q).contiguous()
            ev = expand_kv(pv, H_Q).contiguous()
            t_dense = bench(
                lambda: dense.forward_packed(pq, ek, ev, CFG, cu))
            del ek, ev

            print(f"{kind:>8} {b:>3} {waste:>5.0%} | "
                  f"{b * MAXLEN:>8} {pq.shape[1]:>8} "
                  f"{acc_mb(q, i_pad):>7.1f}M {acc_mb(pq, i_pk):>7.1f}M | "
                  f"{grid:>6} {live_pad:>8} {live_pk:>8} "
                  f"{pair_pad / 1e6:>7.1f}M {pair_pk / 1e6:>7.1f}M | "
                  f"{t_pad:>9.3f} {t_pk:>8.3f} {t_pad / t_pk:>6.3f}x | "
                  f"{t_dense:>9.3f} {t_dense / t_pk:>7.3f}x")
            del q, k, v, mask, pq, pk_, pv, i_pad, i_pk
            torch.cuda.empty_cache()
        except torch.OutOfMemoryError:
            print(f"{kind:>8} {b:>3} {waste:>5.0%} |   OOM")
            torch.cuda.empty_cache()
    print()
