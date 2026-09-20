"""How the kernel behaves as the MoBA block shrinks.

WHAT THIS IS A TEST OF. Not "small blocks" as a goal in themselves -- CLAUDE.md
retired that framing on 2026-08-19, having checked that `block_size=16` appears
in neither reference and that the references get their sparsity from long
context instead. This is a test of the same loop-order claim by a second route.

`docs/divergence-measurements.md` Finding 1 measures query-major utilization as
a function of `num_blocks`: 0.432 at 32 blocks, 0.121 at 128. There are two ways
to grow `num_blocks = S / block_size` -- raise S, or shrink the block. The A100
results took the first. This takes the second, which is reachable on a 4 GB
laptop and independent of sequence length. If the advantage tracks `num_blocks`
whichever knob produced it, the mechanism is what we say it is rather than
something merely correlated with long context.

TWO PANELS, because "smaller blocks" is two different experiments.

* `budget_panel` holds `top_k * block_size` at 1024, so every row attends the
  same number of keys and only granularity changes. This is the comparison
  FlashMoBA's own theory argues for, and it is the headline.
* `fixed_top_k_panel` holds `top_k` at 8, so sparsity rises as blocks shrink and
  each row does strictly less work. Those rows get faster for reasons that have
  nothing to do with the kernel, which is why they are reported second.

WHAT THE MISSING ROWS ARE FOR. FlashMoBA requires its block dimension to be a
multiple of 64 (`flash_moba_api.cpp:465`), so it cannot run the two smallest
rows at all, and `implementations/flash_moba.py` reports that as an availability
reason. `run_suite` turns it into a row carrying the reason rather than a gap.
Those rows are evidence about where the contribution lies; a report that drops
them is worse than no report, because it looks complete.

OUR OWN FLOOR IS 16. `implementations/ours.py` rejects `block_size` below
`TENSOR_CORE_TILE`, because Ampere's MMA atom is 16x8x16 and a narrower tile
cannot fill a tensor core. If the kernel loses at 16, that is a finding about
the limit of the design, to be reported rather than tuned around.
"""

import torch

from bench.report import to_csv, to_markdown
from bench.suite import BYTES_PER_MIB, default_seq_lens, run_suite

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Largest first, matching every other sweep here. 16 is the floor, for the
# tensor-core reason in the module docstring.
BLOCK_SIZES = (128, 64, 32, 16)

# Keys each query attends in the budget panel. 1024 because that is
# `block_size=128, top_k=8`, the configuration every existing result in this
# repository uses, so the first row of the panel is comparable with them.
BUDGET = 1024

FIXED_TOP_K = 8

# One batch, four heads, head_dim 64 -- the shape every other result here uses,
# so the block_size=128 row lands beside them without a translation note.
BATCH, HEADS, HEAD_DIM = 1, 4, 64


# ---------------------------------------------------------------------------
# Panels
# ---------------------------------------------------------------------------


def budget_panel(budget: int = BUDGET,
                 block_sizes: tuple[int, ...] = BLOCK_SIZES
                 ) -> list[tuple[int, int]]:
    """Gate configs holding `top_k * block_size` constant.

    Args:
        budget: Keys each query attends, `top_k * block_size`.
        block_sizes: Block sizes to cover, largest first.

    Returns:
        List of `(block_size, top_k)` pairs, in `run_suite`'s `gates` order.

    Raises:
        ValueError: If `budget` is not divisible by one of `block_sizes`, which
            would silently change the attended budget between rows and make the
            panel incomparable.
    """
    pairs = []
    for block_size in block_sizes:
        if budget % block_size:
            raise ValueError(
                f"budget {budget} is not a multiple of block_size "
                f"{block_size}; the panel would not hold the budget constant"
            )
        pairs.append((block_size, budget // block_size))
    return pairs


def fixed_top_k_panel(top_k: int = FIXED_TOP_K,
                      block_sizes: tuple[int, ...] = BLOCK_SIZES
                      ) -> list[tuple[int, int]]:
    """Gate configs holding `top_k` constant, so sparsity rises as blocks shrink.

    Args:
        top_k: Blocks each query attends, the same in every row.
        block_sizes: Block sizes to cover, largest first.

    Returns:
        List of `(block_size, top_k)` pairs, in `run_suite`'s `gates` order.
    """
    return [(block_size, top_k) for block_size in block_sizes]


PANELS = (("budget", budget_panel), ("fixed-top-k", fixed_top_k_panel))


# ---------------------------------------------------------------------------
# Scratch memory, both designs
# ---------------------------------------------------------------------------


def partials_mb(batch: int, heads: int, seq_len: int, head_dim: int,
                top_k: int, dtype: torch.dtype = torch.bfloat16) -> float:
    """Scratch HBM this kernel's design needs, in MiB.

    One slot per (query, selected block): `part_acc` is `[B, H, S, top_k, D]` at
    the compute dtype, and `part_m` and `part_l` are `[B, H, S, top_k]` float32
    each (`moba.kernels.moba_fwd.empty_partials`).

    This belongs beside the latencies rather than in a footnote. In the budget
    panel `top_k` rises as `block_size` falls, so the scratch grows in exactly
    the regime the panel is claiming an advantage in: the design wins on bytes
    moved and pays in bytes held (`docs/traffic-derivation.md` section 3).

    Args:
        batch: Batch size.
        heads: Number of query heads.
        seq_len: Sequence length.
        head_dim: Head dimension.
        top_k: Blocks each query selects.
        dtype: Storage dtype of `part_acc`.

    Returns:
        Scratch requirement in MiB.
    """
    itemsize = torch.tensor([], dtype=dtype).element_size()
    slots = batch * heads * seq_len * top_k
    return (slots * head_dim * itemsize + 2 * slots * 4) / BYTES_PER_MIB


def running_output_mb(batch: int, heads: int, seq_len: int,
                      head_dim: int) -> float:
    """Scratch a query-group design needs, in MiB.

    FlashMoBA keeps one float32 output row per query and updates it in place
    (`o_tmp`, `[N, H, D]` float32, `flash_moba_api.cpp:505`), so its scratch does
    not grow with `top_k` -- which is why this function has no `top_k` argument
    at all. Reported beside `partials_mb` because that difference is the trade
    this project makes, not a detail.

    Args:
        batch: Batch size.
        heads: Number of query heads.
        seq_len: Sequence length.
        head_dim: Head dimension.

    Returns:
        Scratch requirement in MiB.
    """
    return batch * seq_len * heads * head_dim * 4 / BYTES_PER_MIB


def scratch_table(shapes: list[tuple[int, int, int, int]],
                  gates: list[tuple[int, int]]) -> str:
    """Renders scratch requirements for both designs over the panel.

    Args:
        shapes: List of `(batch, seq_len, heads, head_dim)`.
        gates: List of `(block_size, top_k)`.

    Returns:
        A markdown table, one row per (shape, gate).
    """
    lines = ["", "Scratch memory (MiB)", "",
             "| S | block | top_k | ours | query-group design | ratio |",
             "|---:|---:|---:|---:|---:|---:|"]
    for batch, seq_len, heads, head_dim in shapes:
        theirs = running_output_mb(batch, heads, seq_len, head_dim)
        for block_size, top_k in gates:
            ours = partials_mb(batch, heads, seq_len, head_dim, top_k)
            lines.append(
                f"| {seq_len} | {block_size} | {top_k} | {ours:.1f} | "
                f"{theirs:.1f} | {ours / theirs:.1f}x |"
            )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Runs both panels and writes a CSV and a markdown table for each."""
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--longest", type=int, default=32768,
                        help="cap on sequence length; the suite's memory gate "
                             "uses the oracle's O(S^2) estimate and skips every "
                             "backend above it, so keep this short on a laptop")
    parser.add_argument("--seq-lens", default=None,
                        help="explicit comma-separated lengths, overriding "
                             "--longest. `--seq-lens 65536` is the long-context "
                             "run: its two smallest gates died with a bare "
                             "`exit 1` before 838ca69, which was the gate's "
                             "compile ceiling at 2048 and 4096 columns rather "
                             "than a runtime failure")
    parser.add_argument("--tag", default="",
                        help="suffix for the output filenames, so a focused "
                             "run does not overwrite the full sweep's CSVs")
    parser.add_argument("--memory-budget", type=float, default=None,
                        help="override the skip threshold in MB. The gate "
                             "estimates the *oracle's* O(S^2) need, which at "
                             "S=65536 is 98 GB -- so every backend is skipped, "
                             "including the sparse ones that never allocate "
                             "it. Raising it lets each backend try and fail on "
                             "its own terms, which run_suite already reports "
                             "per row.")
    args = parser.parse_args()

    seq_lens = ([int(s) for s in args.seq_lens.split(",")] if args.seq_lens
                else default_seq_lens(longest=args.longest))
    shapes = [(BATCH, s, HEADS, HEAD_DIM) for s in seq_lens]
    Path("results").mkdir(exist_ok=True)
    print(f"sequence lengths: {seq_lens}")

    for name, build in PANELS:
        gates = build()
        print(f"\n{'=' * 72}\npanel: {name}   gates: {gates}\n{'=' * 72}")
        rows = run_suite(shapes, gates,
                         memory_budget_mb=args.memory_budget)
        to_csv(rows, f"results/block-sweep-{name}{args.tag}.csv")
        print(to_markdown(rows))
        print(scratch_table(shapes, gates))


if __name__ == "__main__":
    main()
