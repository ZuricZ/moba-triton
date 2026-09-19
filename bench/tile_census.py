"""Count the tensor-core tiles each design issues, for the same MoBA selection.

WHY THIS EXISTS. `docs/a100-results.md` reports `ours` at ~2.7x FlashMoBA's
CUDA kernel across a 128x range of working-set sizes. A citation of their source
is not a proof of why, and a bandwidth story is contradicted by the flatness of
that ratio -- bandwidth effects have a cliff where the working set leaves L2,
and this one does not move. So the explanation has to be a *work* multiplier,
and a work multiplier is countable without a GPU.

WHAT IS COUNTED. Both kernels issue query tiles 64 rows tall. Measured, not
assumed, on both sides:

* FlashMoBA picks `Flash_moba_fwd_kernel_traits<Headdim, 64, ...>` for head_dim
  64 and for head_dim 128 on sm_80 (`flash_fwd_launch_template.h:90,147`), so
  its `kBlockM` is 64.
* Ours runs `GROUP=64` -- what the autotuner chose on both cards and what
  `moba.kernels.tuned_configs.DEFAULT` bakes in.

Same tile height, same selection, same output. The only question is how full
each tile is, and that is decided by *where each design may look for queries to
fill it*:

* FlashMoBA's worker owns a window of `lg_block_m` queries and loops over key
  blocks, so a tile can only be filled from that window
  (`flash_fwd_kernel.h:544-554`).
* Ours owns one key block and loops over queries, so a tile is filled from the
  whole sequence (`moba_fwd.py:312`).

A padded row still costs a tensor-core issue, so the ratio of tile counts is a
*lower* bound on the arithmetic gap -- lower, because it charges FlashMoBA
nothing for the K/V re-reads its extra tiles also force.

WHY RANDOM INPUTS ARE THE RIGHT INPUTS. The selection comes from this project's
own gate on `randn` q and k, which is exactly what produced every timing under
`results/`. A trained model's gate would cluster its choices; this counts the
same distribution the measured speedups were measured on, so the two are
comparable. It is not a claim about trained models.

USAGE
    uv run python -m bench.tile_census
    uv run python -m bench.tile_census --seq-len 16384 --heads 32 --head-dim 128
"""

import argparse
import math

import torch

from moba.config import MoBAConfig
from moba.naive import moba_selection

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Query-tile height. Both designs use 64; the module docstring records where
# each of those numbers was read from.
TILE_ROWS = 64


# ---------------------------------------------------------------------------
# FlashMoBA's window policy, reproduced
# ---------------------------------------------------------------------------


def decide_lg_block_m(top_k: int, chunk_size: int, seq_len: int,
                      causal: bool = True, sm8x: bool = False) -> int:
    """Returns FlashMoBA's query-window size for a shape.

    A transcription of `flash_moba/flash_moba_interface.py:20-46`, kept here so
    the census runs without importing their package, which needs the built CUDA
    extension. The branch values are theirs verbatim.

    Args:
        top_k: Blocks each query selects.
        chunk_size: MoBA block size, their `n_lg_block_dim`.
        seq_len: Sequence length.
        causal: Whether attention is causal.
        sm8x: Whether the card is sm_86/sm_89, which caps the window at 512.

    Returns:
        The window size in queries, their `m_lg_block_dim`.
    """
    budget = top_k * chunk_size
    if causal:
        density = (2 * (budget * seq_len) - budget ** 2) / (seq_len ** 2)
    else:
        density = budget / seq_len
    sparsity = 1 - density

    if sparsity <= 0.5:
        lg_block_m = 128
    elif sparsity <= 0.7:
        lg_block_m = 256
    elif sparsity <= 0.8:
        lg_block_m = 512
    elif sparsity <= 0.9:
        lg_block_m = 768
    else:
        lg_block_m = 1024
    return min(lg_block_m, 512) if sm8x else lg_block_m


# ---------------------------------------------------------------------------
# Counting
# ---------------------------------------------------------------------------


def count_ours(mask: torch.Tensor,
               tile_rows: int = TILE_ROWS) -> tuple[int, int]:
    """Counts tiles when a tile may be filled from the whole sequence.

    Args:
        mask: Selection, [B, H, S, num_blocks], True where a query picked a
            block.
        tile_rows: Query rows per tile.

    Returns:
        A tuple of (tiles, rows) -- tiles issued, and real non-padding rows.
    """
    # Per (batch, head, block): how many queries anywhere in the sequence chose
    # it. That column is what one program walks, `tile_rows` at a time.
    per_block = mask.sum(dim=2)                       # [B, H, num_blocks]
    tiles = torch.ceil(per_block / tile_rows).sum().item()
    return int(tiles), int(per_block.sum().item())


def count_flash_moba(mask: torch.Tensor, window: int,
                     tile_rows: int = TILE_ROWS) -> tuple[int, int]:
    """Counts tiles when a tile may only be filled from one query window.

    Args:
        mask: Selection, [B, H, S, num_blocks].
        window: Queries per worker, their `m_lg_block_dim`.
        tile_rows: Query rows per tile.

    Returns:
        A tuple of (tiles, rows), as `count_ours`.
    """
    batch, heads, seq_len, num_blocks = mask.shape
    # Pad the query axis up to a whole number of windows so the reshape is
    # exact. Padding rows are False, so they contribute no tiles and no rows.
    windows = math.ceil(seq_len / window)
    pad = windows * window - seq_len
    if pad:
        mask = torch.nn.functional.pad(mask, (0, 0, 0, pad))

    # [B, H, windows, window, num_blocks] -> count per (window, block).
    per_pair = mask.reshape(batch, heads, windows, window,
                            num_blocks).sum(dim=3)
    tiles = torch.ceil(per_pair / tile_rows).sum().item()
    return int(tiles), int(per_pair.sum().item())


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seq-len", type=int, default=8192)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--device", default="cpu",
                        help="the census is integer counting, so cpu is fine "
                             "and needs no GPU rental")
    args = parser.parse_args()

    cfg = MoBAConfig(block_size=args.block_size, top_k=args.top_k,
                     dtype=torch.float32)
    generator = torch.Generator(device=args.device).manual_seed(0)
    shape = (1, args.seq_len, args.heads, args.head_dim)
    q = torch.randn(shape, generator=generator, device=args.device)
    k = torch.randn(shape, generator=generator, device=args.device)

    mask = moba_selection(q, k, cfg)
    window = decide_lg_block_m(args.top_k, args.block_size, args.seq_len)

    ours_tiles, ours_rows = count_ours(mask)
    theirs_tiles, theirs_rows = count_flash_moba(mask, window)

    slots_ours = ours_tiles * TILE_ROWS
    slots_theirs = theirs_tiles * TILE_ROWS

    print(f"S={args.seq_len} H={args.heads} D={args.head_dim} "
          f"block={args.block_size} top_k={args.top_k} causal")
    print(f"tile height {TILE_ROWS} rows on both sides; "
          f"FlashMoBA window {window} queries\n")

    # The whole argument rests on both designs computing the same thing, so it
    # is asserted rather than trusted.
    assert ours_rows == theirs_rows, "the designs must cover the same work"

    print(f"  (query, block) pairs to compute   {ours_rows:>12,}"
          f"   identical for both")
    print(f"  {'':34}{'tiles':>12} {'row slots':>12} {'filled':>8}")
    print("  " + "-" * 70)
    print(f"  ours   (fill from whole sequence) {ours_tiles:>12,} "
          f"{slots_ours:>12,} {ours_rows / slots_ours:>7.1%}")
    print(f"  theirs (fill from one window)     {theirs_tiles:>12,} "
          f"{slots_theirs:>12,} {theirs_rows / slots_theirs:>7.1%}")
    print()
    print(f"  FlashMoBA issues {theirs_tiles / ours_tiles:.2f}x the tiles "
          f"for identical output.")


if __name__ == "__main__":
    main()
