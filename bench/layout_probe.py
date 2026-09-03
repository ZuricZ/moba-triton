"""Does token-major K/V addressing cost anything at the load?

This is the measurement that decided DD-1. It isolates the one thing that
differs between the two layouts for our block-major loop order: a program owns
`BLOCK_SIZE` consecutive keys for one head, and has to walk them.

    head-major   h*S*D + t*D    -> one contiguous BLOCK_SIZE*D run
    token-major  t*H*D + h*D    -> BLOCK_SIZE runs of D, each H*D apart

Same bytes, same cache lines -- at D=64 bf16 one key's row is exactly 128 B,
one full line, so neither layout over-fetches. What differs is whether
consecutive keys are adjacent. Everything else is held identical: block size,
grid, warp count, dtype, total bytes.

WHY IT COMES OUT NEUTRAL. The structural argument for head-major is that it
issues fewer, longer transactions. That advantage does not survive contact with
the grid. Every head's program is launched and resident together, so the reads
token-major "skips" are being issued at that same instant by the sibling program
for the next head. In aggregate the memory system sees the same contiguous range
either way; only the identity of the issuing program changed.

The argument would break if the heads were not concurrent, if D were small
enough that a row did not fill a cache line (`check_gathered_head_dim` rejects
that), or in a latency-bound regime with too few requests in flight.

MEASURED, RTX 3050 Ti Laptop (sm_86, 20 SMs), 2026-09-03:

      B   H      S    D  BLK | head-maj  token-maj   ratio  GB/s hm  GB/s tm
      1   8   8192   64  128 |   0.1434     0.1393  0.971x    109.0    112.2
      1   8   8192  128  128 |   0.2427     0.2396  0.987x    128.8    130.4
      1   2   8192   64  128 |   0.0420     0.0420  1.000x     93.0     93.0
      4   8   2048   64  128 |   0.1270     0.1280  1.008x    123.1    122.1
      1   8   8192   64   64 |   0.1167     0.1178  1.009x    133.8    132.7
      1   8   8192   64  256 |   0.1382     0.1362  0.985x    113.0    114.7

Spread 0.971x-1.009x with no systematic direction: token-major wins three and
loses three. Re-run this on rented sm_80/90 before trusting the conclusion
there -- a card with more SMs and a much larger L2 may not behave the same way,
and it is the only claim in `docs/layout-measurements.md` that is hardware
specific.

This is a synthetic probe, not the kernel. It answers "does the stride cost
anything", which is a narrower question than "did the change cost anything".
The latter is `bench/stages.py` before and after.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _load_tiles(K, V, Out, S, D: tl.constexpr, H: tl.constexpr,
                BLOCK: tl.constexpr, TOKEN_MAJOR: tl.constexpr):
    """Loads one K and one V tile and reduces them, doing nothing else.

    The reduction exists only so the loads cannot be optimized away; it is a
    single pass over data already in registers and is identical in both arms.
    """
    blk, bh = tl.program_id(0), tl.program_id(1)
    b, h = bh // H, bh % H
    t = blk * BLOCK + tl.arange(0, BLOCK)
    d = tl.arange(0, D)
    if TOKEN_MAJOR:
        base = b * S * H * D + t[:, None] * (H * D) + h * D + d[None, :]
    else:
        base = b * H * S * D + h * S * D + t[:, None] * D + d[None, :]
    m = t[:, None] < S
    acc = (tl.load(K + base, mask=m, other=0.0).to(tl.float32)
           + tl.load(V + base, mask=m, other=0.0).to(tl.float32))
    tl.store(
        Out + bh * BLOCK * D + tl.arange(0, BLOCK)[:, None] * D + d[None, :],
        tl.sum(acc, 0)[None, :] + tl.zeros([BLOCK, D], tl.float32),
    )


def run(b, h, s, d, block):
    """Times both layouts for one configuration.

    Returns:
        A tuple of (head_major_ms, token_major_ms), medians.
    """
    n = b * h * s * d
    k = torch.randn(n, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(n, device="cuda", dtype=torch.bfloat16)
    out = torch.empty(b * h * block * d, device="cuda", dtype=torch.float32)
    grid = (triton.cdiv(s, block), b * h)

    times = []
    for token_major in (False, True):
        def once():
            _load_tiles[grid](k, v, out, s, D=d, H=h, BLOCK=block,
                              TOKEN_MAJOR=token_major, num_warps=4)

        times.append(triton.testing.do_bench(once, warmup=50, rep=200,
                                             return_mode="median"))
    return times[0], times[1]


CASES = [
    (1, 8, 8192, 64, 128),      # the shape the forward actually runs
    (1, 8, 8192, 128, 128),     # head_dim 128: two cache lines per row
    (1, 2, 8192, 64, 128),      # few heads: the least interleaving available
    (4, 8, 2048, 64, 128),      # batch instead of length
    (1, 8, 8192, 64, 64),       # narrower key tile
    (1, 8, 8192, 64, 256),      # wider key tile
]

if __name__ == "__main__":
    print("K/V tile loads only. Bytes identical; transaction shape differs.\n")
    print(f"{'B':>3} {'H':>3} {'S':>6} {'D':>4} {'BLK':>4} | {'head-maj':>9} "
          f"{'token-maj':>10} {'ratio':>7} {'GB/s hm':>8} {'GB/s tm':>8}")
    for b, h, s, d, block in CASES:
        hm, tm = run(b, h, s, d, block)
        gb = b * h * s * d * 2 * 2 / 2**30      # K and V, two bytes each
        print(f"{b:>3} {h:>3} {s:>6} {d:>4} {block:>4} | {hm:>9.4f} "
              f"{tm:>10.4f} {tm / hm:>6.3f}x {gb / (hm / 1e3):>8.1f} "
              f"{gb / (tm / 1e3):>8.1f}")
        torch.cuda.empty_cache()
