"""Draws the block-size sweep as the controlled experiment it is.

WHY A FIGURE AT ALL. `docs/block-size-results.md` argues that the penalty for
shrinking the MoBA block belongs to `top_k` and not to `block_size`, and the
evidence is that two panels sweeping `block_size` over identical values
disagree by 29.52x against 2.05x. That is a comparison between two lines, and
a table asks the reader to hold both columns in their head to see it. One
frame with both panels on the same axes does not.

WHAT IS DRAWN. Latency against `block_size`, descending left to right so the
x axis reads as "blocks get smaller", with the two panels overlaid:

* **budget** holds `top_k * block_size = 1024`, so `top_k` rises 8 -> 64 as
  the block shrinks and every row attends the same 1024 keys.
* **fixed top_k** holds `top_k = 8`, so a smaller block means *fewer* keys
  attended, 1024 down to 128.

Both are normalised to their own `block_size=128` row, because the claim is
about the shape of each curve rather than the absolute times, and the two
panels do different amounts of arithmetic so their absolute values are not
comparable. `sdpa_dense` is drawn flat as the thing to beat -- nothing about
it depends on the gate, so it is the honest reference for "is sparse
attention worth it at this configuration".

RUN IT
    uv run python -m bench.block_sweep_plot
    uv run python -m bench.block_sweep_plot --tag=-65k

No GPU needed: this reads the CSVs the sweep already wrote.
"""

import argparse
import csv
import os

from bench import paths, plots

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PANELS = (
    ("budget", "budget held at 1024 keys  (top_k rises 8 - 64)", "-"),
    ("fixed-top-k", "top_k held at 8  (keys fall 1024 - 128)", (0, (5, 2))),
)

# The backend whose curve carries the argument, and the flat reference.
SUBJECT = "ours"
REFERENCE = "sdpa_dense"


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------


def read_panel(path: str, backend: str, seq_len: int | None
               ) -> tuple[list[int], list[float], int]:
    """Returns (block_sizes, latencies, seq_len) for one backend in one panel.

    Args:
        path: CSV written by `bench.block_sweep`.
        backend: Backend name to extract.
        seq_len: Sequence length to keep, or None for the longest present.

    Returns:
        Block sizes descending, their latencies, and the sequence length used.
        Rows the backend could not run are dropped rather than plotted as
        zero, which would draw a line through a measurement that does not
        exist.
    """
    rows = [r for r in csv.DictReader(open(path, encoding="utf-8"))
            if r["backend"] == backend and r["status"] == "ok"]
    if not rows:
        return [], [], 0
    if seq_len is None:
        seq_len = max(int(r["seq_len"]) for r in rows)
    rows = [r for r in rows if int(r["seq_len"]) == seq_len]
    rows.sort(key=lambda r: -int(r["block_size"]))
    return ([int(r["block_size"]) for r in rows],
            [float(r["latency_ms"]) for r in rows],
            seq_len)


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------


def draw(tag: str, seq_len: int | None, out: str,
         card: str | None = None) -> str:
    """Renders both panels into one figure.

    Args:
        tag: Filename suffix the sweep was run with, "" for the full sweep.
        seq_len: Sequence length to draw, or None for the longest present.
        out: Path to write.

    Returns:
        The path written.

    Raises:
        SystemExit: If neither panel's CSV exists, since an empty figure is
            worse than a message saying which file is missing.
    """
    fig, ax = plots.figure(width=9.0, height=5.6)
    drawn, used_seq = False, 0

    for panel, label, dash in PANELS:
        path = os.path.join(paths.card_dir(card, "data"),
                            f"block-sweep-{panel}{tag}.csv")
        if not os.path.exists(path):
            print(f"  missing, skipping: {path}")
            continue
        blocks, times, s = read_panel(path, SUBJECT, seq_len)
        if not blocks:
            print(f"  no `{SUBJECT}` rows in {path}")
            continue
        used_seq = s
        base = times[0]
        ax.plot([str(b) for b in blocks], [t / base for t in times],
                color=plots.color(SUBJECT), linewidth=2.4, linestyle=dash,
                marker="o", markersize=5, label=label, zorder=3)
        # The end of each curve is the number the document quotes.
        ax.annotate(f"{times[-1] / base:.1f}x",
                    xy=(len(blocks) - 1, times[-1] / base),
                    xytext=(-4, 8), textcoords="offset points",
                    color=plots.color(SUBJECT), fontsize=10, weight="bold")
        drawn = True

        # The reference, from the same panel and the same run.
        ref_blocks, ref_times, _ = read_panel(path, REFERENCE, s)
        if ref_blocks and panel == "budget":
            ax.plot([str(b) for b in ref_blocks],
                    [t / base for t in ref_times],
                    color=plots.color(REFERENCE), linewidth=1.5,
                    linestyle=(0, (1, 2)), zorder=2,
                    label=f"{REFERENCE}, same rows")

    if not drawn:
        raise SystemExit(
            "no panel CSVs found in "
            f"{paths.card_dir(card, 'data')}. Cards with data on disk: "
            f"{', '.join(paths.known_cards()) or 'none'}. Run "
            "`bench.block_sweep`, or pass --card.")

    ax.set_yscale("log")
    # `titles` owns the axis labels -- it sets both unconditionally, so
    # calling `set_xlabel` before it silently loses them.
    plots.titles(
        ax,
        "Shrinking the block costs top_k, not block_size",
        f"`ours` at S={used_seq:,}, each panel normalised to its own "
        f"block_size=128 row. Identical block sweep in both; they differ\n"
        f"only in whether top_k rises. Lower is better; above the dotted "
        f"line, dense attention is faster than this kernel.",
        xlabel="MoBA block size  (blocks get smaller to the right)",
        ylabel="latency relative to block_size=128",
    )
    plots.legend(ax, loc="upper left")
    return plots.save(fig, out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default="",
                        help="filename suffix the sweep used. Pass it joined "
                             "by '=' -- `--tag=-65k` -- because a value "
                             "starting with '-' is read as another flag "
                             "otherwise, which argparse reports as the "
                             "confusing 'expected one argument'.")
    parser.add_argument("--seq-len", type=int, default=None,
                        help="which length to draw; default the longest")
    parser.add_argument("--out", default=None)
    paths.add_card_argument(parser)
    args = parser.parse_args()

    out = args.out or os.path.join(
        paths.card_dir(args.card, "figures", create=True),
        f"block-sweep{args.tag or ''}.png")
    print(draw(args.tag, args.seq_len, out, args.card))


if __name__ == "__main__":
    main()
