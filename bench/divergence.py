"""Measures routing divergence -- the quantity that decides the loop order.

Two dual statistics, computed from real oracle selection masks rather than
estimated:

*Query-major utilization.* A query tile must load the union of the blocks its
queries selected, then mask away what each individual query did not want. This
reports useful block-query pairs divided by computed ones. It is the fraction
of a query-major kernel's tensor-core work that is not thrown away.

*Consumers per KV block.* In the inverted kernel, a loaded block serves every
query that selected it. This reports how many that is. It is the amortization
factor of the KV-block-major design, and it sizes the logical query group.

Neither number depends on any kernel existing -- both are properties of the
selection pattern alone, so they can be measured now and used to choose tile
shapes before a line of Triton is written.

CAVEAT ON INPUT DATA: with random Gaussian q/k the gate scores are unstructured,
so queries select near-independently. That is the *adversarial* end of the
range. A trained model's neighbouring queries have correlated representations
and would agree more, raising utilization. Read these as a lower bound on
query-major utilization, not as the expected value in production.
"""

import torch

from moba.config import MoBAConfig
from moba.ref.naive import moba_selection


def divergence_stats(
    selected: torch.Tensor, block_m: int
) -> dict[str, float]:
    """Computes query-major utilization and inverted-kernel amortization.

    Args:
        selected: Boolean selection mask shaped [B, H, S, num_blocks].
        block_m: Query tile height a query-major kernel would use.

    Returns:
        A dict with `utilization`, `mean_union`, `mean_per_query`,
        `consumers_per_block` and `max_consumers`.
    """
    b, h, s, n = selected.shape
    tiles = selected.view(b, h, s // block_m, block_m, n)

    # A tile must load every block any of its queries wanted.
    union = tiles.any(dim=3)                       # [B, H, T, num_blocks]

    computed = union.sum(dim=-1).to(torch.float64) * block_m
    useful = tiles.sum(dim=(-1, -2)).to(torch.float64)

    # Inverted view: how many queries select each block.
    consumers = selected.sum(dim=2).to(torch.float64)   # [B, H, num_blocks]
    live = consumers[consumers > 0]

    return {
        "utilization": (useful.sum() / computed.sum()).item(),
        "mean_union": union.sum(dim=-1).to(torch.float64).mean().item(),
        "mean_per_query": selected.sum(dim=-1).to(torch.float64).mean().item(),
        "consumers_per_block": live.mean().item(),
        "max_consumers": live.max().item(),
    }


def run(
    seq_lens: tuple[int, ...] = (1024, 2048, 4096),
    block_sizes: tuple[int, ...] = (32, 64, 128),
    top_ks: tuple[int, ...] = (4, 8),
    block_ms: tuple[int, ...] = (16, 32, 64, 128),
    heads: int = 8,
    head_dim: int = 128,
    dtype: torch.dtype = torch.bfloat16,
    seed: int = 0,
) -> list[dict]:
    """Sweeps configurations and returns one record per (config, block_m).

    Args:
        seq_lens: Sequence lengths to measure.
        block_sizes: MoBA block sizes.
        top_ks: Gate budgets, inclusive convention.
        block_ms: Query tile heights to evaluate for the query-major case.
        heads: Number of attention heads.
        head_dim: Head dimension.
        dtype: Input dtype.
        seed: RNG seed.

    Returns:
        A list of flat dicts suitable for tabulation.
    """
    rows = []
    for s in seq_lens:
        g = torch.Generator(device="cuda").manual_seed(seed)
        q = torch.randn(1, heads, s, head_dim, generator=g,
                        device="cuda", dtype=dtype)
        k = torch.randn(1, heads, s, head_dim, generator=g,
                        device="cuda", dtype=dtype)
        for bk in block_sizes:
            for tk in top_ks:
                cfg = MoBAConfig(block_size=bk, top_k=tk, dtype=dtype)
                sel = moba_selection(q, k, cfg)
                for bm in block_ms:
                    if bm > s:
                        continue
                    stats = divergence_stats(sel, bm)
                    rows.append(
                        {"seq_len": s, "block_size": bk, "top_k": tk,
                         "block_m": bm, "num_blocks": sel.shape[-1], **stats}
                    )
                del sel
        del q, k
    return rows


def to_markdown(rows: list[dict], top_k: int, block_m: int) -> str:
    """Renders one slice of the sweep as a markdown table.

    Args:
        rows: Records from `run`.
        top_k: Gate budget to filter to.
        block_m: Query tile height to filter to.

    Returns:
        A markdown table string.
    """
    sel = [r for r in rows if r["top_k"] == top_k and r["block_m"] == block_m]
    head = ("| seq_len | block_size | num_blocks | mean union | per query | "
            "**utilization** | consumers/block | max |")
    sep = "|" + "---|" * 8
    body = [
        f"| {r['seq_len']} | {r['block_size']} | {r['num_blocks']} | "
        f"{r['mean_union']:.1f} | {r['mean_per_query']:.2f} | "
        f"**{r['utilization']:.3f}** | {r['consumers_per_block']:.1f} | "
        f"{r['max_consumers']:.0f} |"
        for r in sel
    ]
    return "\n".join([head, sep, *body])


if __name__ == "__main__":
    import json
    import sys

    rows = run()
    if "--json" in sys.argv:
        print(json.dumps(rows))
    else:
        for tk in (4, 8):
            for bm in (16, 64, 128):
                print(f"\n### top_k={tk}, BLOCK_M={bm}\n")
                print(to_markdown(rows, tk, bm))
