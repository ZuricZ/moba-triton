"""Why does the backward get *faster* from S=1024 to S=4096?

THE OBSERVATION. In the A100 sweep `ours` forward rises monotonically -- 0.214,
0.220, 0.252 ms at S = 1024, 2048, 4096 -- while the full step falls: 0.655,
0.608, 0.536. The step includes the forward, so the backward-only portion is
falling faster than the 18% the total shows. Work over that range *doubles*
each step, because MoBA attends min(top_k, num_blocks) * block_size keys per
query and that saturates at 1024 keys from S=1024 onward.

WHAT HAS ALREADY BEEN RULED OUT, BY READING RATHER THAN MEASURING.

* **Not per-shape retuning.** `@triton.autotune`'s key is
  ["BLOCK_SIZE", "KEY_TILE", "D", "K_SLOTS", "H_Q", "H_KV"], and every one of
  those is constant across the sweep -- K_SLOTS is `cfg.top_k`, KEY_TILE is the
  module constant BWD_KEY_TILE. So one config is chosen at the first shape and
  used for all seven. This probe prints it to confirm that at runtime, and the
  confirmation is worth having on its own: a config that won a race at S=1024
  is being applied at S=65536.
* **Not atomic contention.** The backward stores dK and dV exactly once per
  (block, head) -- no atomics anywhere in `moba_bwd_kernel`.

WHAT THIS MEASURES. The step split into stages, plus the shape of the routing
that feeds them. A stage whose cost falls while its input grows is the answer;
if instead every stage is flat and only their sum moves, the effect is fixed
overhead and the dip is the card being idle rather than anything about MoBA.

`nnz` is the number of (query, block) pairs -- the real work unit, since a
program-chunk covers GROUP of them. Per-nnz cost is the number to read: it
says how much the kernel gets for each pair, independent of how many there are.
"""

import torch
import triton.testing

from bench.suite import cool_down_device, warm_up_device
from moba.config import MoBAConfig
from moba.kernels.moba_bwd import compute_delta, moba_backward
from moba.kernels.moba_fwd import attend_partials, combine_partials
from moba.kernels.routing import build_csc_fused

CFG = MoBAConfig(block_size=128, top_k=8, dtype=torch.bfloat16)
BATCH, HEADS, HEAD_DIM = 1, 4, 64
SEQ_LENS = (1024, 2048, 4096, 8192, 16384)


def time_ms(fn) -> float:
    """Best of three medians, each after settling the card.

    The same protocol as `bench/compare.py`: error here is one-sided, since
    nothing finishes faster than its true time, so the minimum is the
    maximum-likelihood estimate rather than a cherry-pick.
    """
    fn()
    cool_down_device()
    warm_up_device()
    return min(triton.testing.do_bench(fn, quantiles=[0.5, 0.2, 0.8])[0]
               for _ in range(3))


def winning_config(kernel) -> str:
    """Returns the config `triton.autotune` settled on, or '?' if none yet."""
    cache = getattr(kernel, "cache", None)
    if not cache:
        return "?"
    # One entry unless the key varied, which is the thing being checked.
    configs = {str(v) for v in list(cache.values())[0].values()} if isinstance(
        list(cache.values())[0], dict) else {str(v) for v in cache.values()}
    return " | ".join(sorted(configs))


def main() -> None:
    from moba.kernels import moba_bwd as bwd_module

    print(f"block_size={CFG.block_size} top_k={CFG.top_k} "
          f"[{BATCH}, S, {HEADS}, {HEAD_DIM}] bf16\n")
    header = (f"{'S':>7} {'blocks':>7} {'nnz':>9} {'maxcol':>7} "
              f"{'fwd':>8} {'delta':>8} {'bwd':>8} {'total':>8} "
              f"{'ns/nnz':>8}")
    print(header)
    print("-" * len(header))

    for seq_len in SEQ_LENS:
        g = torch.Generator(device="cuda").manual_seed(0)
        q, k, v = (torch.randn(BATCH, seq_len, HEADS, HEAD_DIM, generator=g,
                               device="cuda", dtype=CFG.dtype)
                   for _ in range(3))
        dout = torch.randn_like(q)

        index = build_csc_fused(q, k, CFG)
        nnz = int(index.col_nnz.sum())
        maxcol = int(index.col_nnz.max())

        part_acc, part_m, part_l = attend_partials(q, k, v, CFG, index)
        out, lse = combine_partials(part_acc, part_m, part_l, q.dtype,
                                    return_lse=True)

        fwd = time_ms(lambda: attend_partials(q, k, v, CFG, index))
        delta = time_ms(lambda: compute_delta(out, dout))
        step = time_ms(lambda: moba_backward(dout, q, k, v, out, lse, part_m,
                                             CFG, index))
        total = fwd + delta + step
        print(f"{seq_len:>7} {index.num_blocks:>7} {nnz:>9} {maxcol:>7} "
              f"{fwd:>8.3f} {delta:>8.3f} {step:>8.3f} {total:>8.3f} "
              f"{step * 1e6 / nnz:>8.1f}")

        del q, k, v, dout, part_acc, part_m, part_l, out, lse, index
        torch.cuda.empty_cache()

    print(f"\nbwd config(s) chosen: {winning_config(bwd_module.moba_bwd_kernel)}")
    print("one entry means one config served the whole sweep, as the key "
          "predicts")


if __name__ == "__main__":
    main()
