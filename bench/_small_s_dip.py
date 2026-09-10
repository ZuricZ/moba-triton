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

IS IT EVEN REAL? Ask this before explaining it. Thermal and contention error is
one-sided -- nothing finishes faster than its true time -- so noise inflates a
measurement and cannot deflate one. To manufacture a *falling* curve, noise
would have to inflate S=1024 most, S=2048 less and S=4096 least.

That is not a strange requirement. It is precisely the shape of a startup
artifact. S=1024 is the first cell of the sweep: it is where `@triton.autotune`
races its configs, where the caching allocator is cold, where the CUDA context
is still warming. All of those decay with *position in the sweep*, and in an
ascending sweep position and S are perfectly correlated, so the original data
cannot separate them.

So every shape is measured twice, ascending and descending. If the dip follows
S it is real; if it follows position it is an artifact of sweeping upward, and
the stage split below would have been an explanation of a measurement bug.
Running ascending only -- which is what this probe did first -- would have
reproduced the artifact rather than detected it.

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


def measure(seq_len: int) -> dict:
    """Times one shape's stages and reports the routing that fed them."""
    g = torch.Generator(device="cuda").manual_seed(0)
    q, k, v = (torch.randn(BATCH, seq_len, HEADS, HEAD_DIM, generator=g,
                           device="cuda", dtype=CFG.dtype)
               for _ in range(3))
    dout = torch.randn_like(q)

    index = build_csc_fused(q, k, CFG)
    part_acc, part_m, part_l = attend_partials(q, k, v, CFG, index)
    out, lse = combine_partials(part_acc, part_m, part_l, q.dtype,
                                return_lse=True)

    row = {
        "seq_len": seq_len,
        "blocks": index.num_blocks,
        "nnz": int(index.col_nnz.sum()),
        "maxcol": int(index.col_nnz.max()),
        "route": time_ms(lambda: build_csc_fused(q, k, CFG)),
        "fwd": time_ms(lambda: attend_partials(q, k, v, CFG, index)),
        "delta": time_ms(lambda: compute_delta(out, dout)),
        "bwd": time_ms(lambda: moba_backward(dout, q, k, v, out, lse, part_m,
                                             CFG, index)),
    }
    del q, k, v, dout, part_acc, part_m, part_l, out, lse, index
    torch.cuda.empty_cache()
    return row


def report(rows: list[dict], label: str) -> None:
    """Prints one pass of the sweep, ordered by shape rather than by run."""
    print(f"\n{label}")
    header = (f"{'S':>7} {'blocks':>7} {'nnz':>9} {'maxcol':>7} "
              f"{'route':>8} {'fwd':>8} {'delta':>8} {'bwd':>8} {'sum':>8} "
              f"{'ns/nnz':>8}")
    print(header)
    print("-" * len(header))
    for r in sorted(rows, key=lambda r: r["seq_len"]):
        total = r["route"] + r["fwd"] + r["delta"] + r["bwd"]
        print(f"{r['seq_len']:>7} {r['blocks']:>7} {r['nnz']:>9} "
              f"{r['maxcol']:>7} {r['route']:>8.3f} {r['fwd']:>8.3f} "
              f"{r['delta']:>8.3f} {r['bwd']:>8.3f} {total:>8.3f} "
              f"{r['bwd'] * 1e6 / r['nnz']:>8.1f}")


def main() -> None:
    from moba.kernels import moba_bwd as bwd_module

    print(f"block_size={CFG.block_size} top_k={CFG.top_k} "
          f"[{BATCH}, S, {HEADS}, {HEAD_DIM}] bf16")

    # Fire every kernel once at the largest shape before anything is timed, so
    # autotuning, JIT compilation and the caching allocator's first growth are
    # all behind us. Otherwise they land on whichever shape happens to be
    # measured first and masquerade as a property of that shape.
    print("priming (autotune, JIT, allocator) at the largest shape...")
    measure(max(SEQ_LENS))

    ascending = [measure(s) for s in sorted(SEQ_LENS)]
    descending = [measure(s) for s in sorted(SEQ_LENS, reverse=True)]

    report(ascending, "ascending -- S rises with sweep position")
    report(descending, "descending -- S falls as sweep position rises")

    print("\nbwd, both orders, to separate S from position:")
    print(f"  {'S':>7} {'up':>9} {'down':>9} {'spread':>8}")
    by_len = {r["seq_len"]: r for r in descending}
    for r in ascending:
        up, down = r["bwd"], by_len[r["seq_len"]]["bwd"]
        print(f"  {r['seq_len']:>7} {up:>9.3f} {down:>9.3f} "
              f"{abs(up - down) / min(up, down):>7.1%}")
    print("  A dip present in both orders follows S and is real. A dip that")
    print("  flips to the other end is an artefact of sweep position.")

    print(f"\nbwd config(s) chosen: {winning_config(bwd_module.moba_bwd_kernel)}")
    print("one entry means one config served every shape, as the key predicts")


if __name__ == "__main__":
    main()
