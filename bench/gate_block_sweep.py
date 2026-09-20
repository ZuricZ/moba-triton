"""Sweeps the gate's streaming-chunk width to re-tune it at long context.

WHY. `GATE_BLOCK_N` is fixed at 32 by a sweep over seq_len 4K-64K that ended
where 64 began to tie (`src/moba/kernels/gating.py`). The gate's merge
overhead is `2 * K_POW2 / BLOCK_N` per chunk, so wider chunks amortise the
top-k bookkeeping; the price is causal-horizon overshoot, which matters only
while a tile's horizon is a small multiple of the chunk. At 512K the horizon
is thousands of blocks, so the price vanishes and wider chunks should win.
That hypothesis was never measured because the card that reached 512K has been
deleted.

WHAT IT MEASURES. `gate_select` latency, nothing else, over chunk widths
{32, 64, 128} at sequence lengths up to 524288. It does not include pooling or
the CSC transpose; the question is purely whether a wider chunk makes the
scoring+selection kernel faster.

RUN IT (needs a card that reaches the lengths):
    uv run python -m bench.gate_block_sweep

Reads no files; writes nothing.
"""

import torch
import triton.testing

from bench.suite import cool_down_device, warm_up_device
from moba.config import MoBAConfig
from moba.kernels.gating import gate_select

BATCH, HEADS, HEAD_DIM = 1, 4, 64
BLOCK_SIZE, TOP_K = 128, 8
CHUNK_WIDTHS = (32, 64, 128)
SEQ_LENS = (8192, 32768, 131072, 262144, 524288)


def make(seq_len: int, cfg: MoBAConfig) -> tuple[torch.Tensor, torch.Tensor]:
    """Builds q and a representative tensor of the right shapes.

    The gate's cost depends on shapes, not values, so the tensors are random.
    `kbar` is [B, H_kv, num_blocks, D] -- the pooled representatives -- because
    the gate scores against them and this benchmark deliberately excludes the
    pooling step.
    """
    g = torch.Generator(device="cuda").manual_seed(0)
    n = cfg.num_blocks(seq_len)
    q = torch.randn(BATCH, seq_len, HEADS, HEAD_DIM, generator=g,
                    device="cuda", dtype=cfg.dtype)
    kbar = torch.randn(BATCH, HEADS, n, HEAD_DIM, generator=g,
                       device="cuda", dtype=cfg.dtype)
    return q, kbar


def main() -> None:
    cfg = MoBAConfig(block_size=BLOCK_SIZE, top_k=TOP_K,
                     dtype=torch.bfloat16)
    print(f"gate block_n sweep  B={BATCH} H={HEADS} D={HEAD_DIM} "
          f"block={BLOCK_SIZE} top_k={TOP_K}")
    print(f"{'S':>8} | " + " ".join(f"block_n={w:>3}" for w in CHUNK_WIDTHS))
    print(f"{'':>8} | " + " ".join(f"{'':>12}" for _ in CHUNK_WIDTHS))

    for seq_len in SEQ_LENS:
        try:
            q, kbar = make(seq_len, cfg)
        except torch.OutOfMemoryError:
            print(f"{seq_len:>8} | OOM building inputs")
            continue
        cells = []
        for block_n in CHUNK_WIDTHS:
            def step():
                return gate_select(q, kbar, cfg, block_n=block_n)
            cool_down_device()
            warm_up_device()
            ms = triton.testing.do_bench(step, quantiles=[0.5])
            ms = ms if isinstance(ms, float) else ms[0]
            cells.append(f"{ms:>12.3f}")
        print(f"{seq_len:>8} | " + " ".join(cells))
        del q, kbar
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
