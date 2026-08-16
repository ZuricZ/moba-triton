"""Measures how much routing diverges across a query tile.

This quantifies the central claim in `docs/learning/01-loop-order-and-memory-traffic.md`:
that a query-major kernel must load the *union* of blocks its tile selected,
and that at small `block_size` the union dwarfs what any single query needs.

The distinction that matters, and the reason this is not simply "average union
size": a query-major kernel loads the union into SRAM and then scores **every
query in the tile against every key in the union**, masking afterwards. Its
work is therefore `BLOCK_M * union_keys`, not `union_keys`. Utilization is the
useful fraction of that.

The output is a ceiling, not a prediction. It says what speedup a *perfect*
query-major kernel could reach -- one with zero launch overhead, perfect
occupancy, and free masking. A real one does worse. If the ceiling is already
low, no amount of tuning rescues the loop order, which is precisely the
question the inverted kernel exists to answer.
"""

from dataclasses import dataclass

import torch

from moba.config import MoBAConfig
from moba.ref.naive import moba_selection


@dataclass(frozen=True)
class DivergenceRow:
    """Routing divergence for one (config, tile size) pair."""

    seq_len: int
    block_size: int
    top_k: int
    tile_size: int
    num_blocks: int
    mean_blocks_per_query: float
    mean_union_per_tile: float
    divergence_ratio: float
    utilization: float
    true_ceiling: float
    query_major_ceiling: float
    recoverable: float


def _block_occupancy(seq_len: int, cfg: MoBAConfig, device) -> torch.Tensor:
    """Number of real keys in each block; the last one may be short.

    Args:
        seq_len: Sequence length.
        cfg: Configuration supplying `block_size`.
        device: Device for the result.

    Returns:
        Float tensor shaped [num_blocks].
    """
    num_blocks = cfg.num_blocks(seq_len)
    occ = torch.full((num_blocks,), float(cfg.block_size), device=device)
    occ[-1] = seq_len - (num_blocks - 1) * cfg.block_size
    return occ


def measure_divergence(
    selected: torch.Tensor, cfg: MoBAConfig, tile_size: int
) -> DivergenceRow:
    """Computes divergence statistics from a selection mask.

    Args:
        selected: Boolean selection mask shaped [B, H, S, num_blocks], as
            produced by `moba.ref.naive.moba_selection`.
        cfg: Configuration the mask was produced with.
        tile_size: Query tile size (`BLOCK_M`) a kernel would use. Must divide
            the sequence length.

    Returns:
        A populated DivergenceRow.

    Raises:
        ValueError: If `tile_size` does not divide the sequence length.
    """
    b, h, s, num_blocks = selected.shape
    if s % tile_size:
        raise ValueError(f"tile_size {tile_size} must divide seq_len {s}")

    device = selected.device
    occ = _block_occupancy(s, cfg, device)                    # [N]
    pos = torch.arange(s, device=device)
    own_block = pos // cfg.block_size                         # [S]
    blocks = torch.arange(num_blocks, device=device)
    is_past = blocks.view(1, -1) < own_block.view(-1, 1)      # [S, N]

    # --- Useful work: keys each query actually attends to. ------------------
    # Its own block contributes the causally visible prefix; every selected
    # past block contributes all of its keys.
    n_past = (selected & is_past.view(1, 1, s, num_blocks)).sum(-1).float()
    prefix = (pos - own_block * cfg.block_size + 1).float()   # [S]
    useful_per_query = prefix.view(1, 1, s) + n_past * cfg.block_size

    # --- Query-major work: every query scored against the whole union. ------
    tiles = s // tile_size
    per_tile = selected.view(b, h, tiles, tile_size, num_blocks)
    union = per_tile.any(dim=3)                               # [B, H, T, N]
    union_keys = (union.float() * occ).sum(-1)                # [B, H, T]
    query_major_per_tile = union_keys * tile_size

    # --- Dense causal work, the common denominator. -------------------------
    dense_per_query = (pos + 1).float()

    useful = useful_per_query.sum().item()
    query_major = query_major_per_tile.sum().item()
    dense = dense_per_query.sum().item() * b * h

    blocks_per_query = selected.sum(-1).float().mean().item()
    union_per_tile = union.sum(-1).float().mean().item()

    return DivergenceRow(
        seq_len=s,
        block_size=cfg.block_size,
        top_k=cfg.top_k,
        tile_size=tile_size,
        num_blocks=num_blocks,
        mean_blocks_per_query=blocks_per_query,
        mean_union_per_tile=union_per_tile,
        divergence_ratio=union_per_tile / blocks_per_query,
        utilization=useful / query_major,
        true_ceiling=dense / useful,
        query_major_ceiling=dense / query_major,
        recoverable=(dense / useful) / (dense / query_major),
    )


def sweep(
    shape: tuple[int, int, int, int],
    gates: list[tuple[int, int]],
    tile_sizes: list[int],
    dtype: torch.dtype = torch.bfloat16,
    seed: int = 0,
) -> list[DivergenceRow]:
    """Measures divergence across a gate and tile-size sweep.

    Selection is recomputed once per gate and reused across tile sizes, since
    tiling is a property of the kernel, not of the routing.

    Args:
        shape: (batch, heads, seq_len, head_dim).
        gates: List of (block_size, top_k).
        tile_sizes: Query tile sizes to evaluate.
        dtype: Compute dtype for the gate.
        seed: RNG seed, so runs are comparable.

    Returns:
        One DivergenceRow per (gate, tile_size).
    """
    rows: list[DivergenceRow] = []
    for block_size, top_k in gates:
        cfg = MoBAConfig(block_size=block_size, top_k=top_k, dtype=dtype)
        g = torch.Generator(device="cuda").manual_seed(seed)
        q, k = (
            torch.randn(shape, generator=g, device="cuda", dtype=dtype)
            for _ in range(2)
        )
        selected = moba_selection(q, k, cfg)
        for tile_size in tile_sizes:
            rows.append(measure_divergence(selected, cfg, tile_size))
        del q, k, selected
    return rows


def to_markdown(rows: list[DivergenceRow]) -> str:
    """Renders divergence rows as a markdown table.

    Args:
        rows: Measured rows.

    Returns:
        A markdown string.
    """
    header = (
        "| block_size | tile | blocks/query | union/tile | divergence | "
        "utilization | true ceiling | query-major ceiling | recoverable |"
    )
    sep = "|---:" * 9 + "|"
    lines = [header, sep]
    for r in rows:
        lines.append(
            f"| {r.block_size} | {r.tile_size} | {r.mean_blocks_per_query:.2f} | "
            f"{r.mean_union_per_tile:.2f} | {r.divergence_ratio:.2f}x | "
            f"{r.utilization:.3f} | {r.true_ceiling:.2f}x | "
            f"{r.query_major_ceiling:.2f}x | {r.recoverable:.2f}x |"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    rows = sweep(
        shape=(1, 8, 1024, 64),
        gates=[(128, 2), (64, 2), (32, 2), (16, 2)],
        tile_sizes=[16, 32, 64, 128],
    )
    print(to_markdown(rows))
