"""Speedup against sequence length: the claim this project actually makes.

WHY A SECOND FIGURE OF THE SAME DATA. `bench/sweep_plot.py` draws latency, and
latency is what was measured, so that is the figure that has to exist. But the
claim is about how each margin *moves* with context, and the two baselines
move differently. Against dense attention the ratio climbs without limit,
because a fixed `top_k * block_size` budget is being compared against work
that is quadratic in S. Against FlashMoBA it does not: both kernels are linear
in S, so the ratio settles near 4x and what separates them is a constant factor
in bytes moved per selected block.

That difference is the figure's reason to exist, and it is exactly what a
log-log latency plot hides. There, both facts are differences between slopes,
and a reader has to divide at each end to recover them. Here one line rises and
the other flattens, which is the whole result in one look.

The title says both halves for that reason. "The margin grows with context" was
the first draft and it is false of the orange line, which is flat; a title the
figure's own data contradicts costs more credibility than it buys.

The cost of dividing is that the baselines lose their absolute scale: a ratio
cannot say whether 3.8x is 3.8x of 150 ms or of 150 us. That is exactly why
this does not replace the latency sweep -- it sits beside it.

TWO PANELS, NOT FOUR LINES ON ONE AXES. Four lines needs two visual channels:
one for which baseline, one for which pass. The project's palette already
spends colour on backend identity and dash pattern on backend role, so a
fourth line would have to invent a third encoding and then explain it. Split
by pass instead and each panel carries two lines in the house style with
nothing new to learn. The panels share a y scale, so "backward is the wider
margin" is still read off the drawing rather than off the tick labels.

THE PARITY LINE IS DRAWN IN THE SUBJECT'S OWN COLOUR. On a ratio plot `ours`
is the denominator: it is not a line, it is the horizontal at 1.0 that
everything else is measured against. Colouring that line with the `ours` slot
says so without a legend entry for a series that has no curve -- the mistake
`bench/sweep_plot.py`'s docstring records `triton.testing` making.
"""

import os

from bench import paths, plots
from bench.sweep_plot import HARNESS_LIMIT, read_sweep

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SUBJECT = "ours"

# The baselines worth dividing by, in the order they are drawn. `fa2_minimal`
# is deliberately absent: it is our own dense reference, there to check that
# `sdpa_dense` is not doing something strange, and a ratio against both dense
# paths draws the same claim twice.
BASELINES = ("flash_moba", "sdpa_dense")

PANELS = (("fwd", "forward"), ("bwd", "forward + backward"))


# ---------------------------------------------------------------------------
# Ratios
# ---------------------------------------------------------------------------


def ratios(seq_lens, series, baseline: str):
    """Returns (sequence lengths, baseline / ours) where both have a number.

    A point is dropped rather than interpolated when either side is missing:
    the baselines here run the full sweep on both cards, so a gap would mean
    something has changed about the data and the honest response is a shorter
    line, not a bridged one.

    Args:
        seq_lens: The sweep's sequence lengths, ascending.
        series: {backend: latencies}, as `bench.sweep_plot.read_sweep` returns.
        baseline: The backend to divide by ours.

    Returns:
        A tuple of (xs, ys). Both empty if either backend is absent.
    """
    mine, theirs = series.get(SUBJECT), series.get(baseline)
    if not mine or not theirs:
        return [], []
    pairs = [(s, t / m) for s, m, t in zip(seq_lens, mine, theirs)
             if m and t]
    return [s for s, _ in pairs], [r for _, r in pairs]


def headline(card: str | None) -> str:
    """Returns a sentence stating the four ratios at the longest sequence run.

    Every number is read back out of the CSVs, not typed: the point of this
    figure is a set of ratios, and a subtitle quoting one that the lines
    contradict is worse than no subtitle.

    Args:
        card: Directory name such as "sm90-h100"; the local card when None.

    Returns:
        A sentence, or "" if the data does not support one.
    """
    rows = []
    for mode, pass_name in PANELS:
        seq_lens, series = read_sweep(card, mode)
        if not seq_lens:
            continue
        ends = []
        for baseline in BASELINES:
            xs, ys = ratios(seq_lens, series, baseline)
            if xs:
                ends.append((baseline, ys[-1], xs[-1]))
        if ends:
            rows.append((pass_name, ends))
    if not rows:
        return ""

    # Both passes normally reach the same length, and repeating "at S=524,288"
    # for each costs a third of the line. Hoist it when it is shared; keep it
    # per pass when it is not, because a shared-looking sentence over two
    # different lengths is the sort of caption this whole exercise is about.
    lengths = {at for _, ends in rows for *_, at in ends}
    if len(lengths) > 1:
        said = [f"{name} at S={ends[0][2]:,} is "
                + " and ".join(f"{r:.1f}x {b}" for b, r, _ in ends)
                for name, ends in rows]
        return "Ours: " + "; ".join(said) + "."

    # Baselines named once, in the order the lines are drawn; the second pass
    # quotes bare numbers against the same two.
    said = [f"{name} " + " and ".join(
        f"{r:.1f}x {b}" if first else f"{r:.1f}x" for b, r, _ in ends)
        for first, (name, ends) in zip([True, False] * len(rows), rows)]
    return f"Ours at S={lengths.pop():,}: " + "; ".join(said) + "."


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------


def draw(card: str | None, path: str) -> str | None:
    """Draws both panels and writes them to `path`.

    Args:
        card: Directory name such as "sm90-h100"; the local card when None.
        path: Where the PNG goes.

    Returns:
        The path written, or None if the card has no sweep to divide.
    """
    fig, axes = plots.panels(len(PANELS), width=10.4, height=4.8)
    drawn_any = False

    for ax, (mode, pass_name) in zip(axes, PANELS):
        seq_lens, series = read_sweep(card, mode)
        if not seq_lens:
            ax.set_visible(False)
            continue

        # Parity first, so it sits under the data and behind the labels.
        ax.axhline(1.0, color=plots.color(SUBJECT), linewidth=1.2,
                   linestyle=(0, (5, 3)), zorder=2)

        tips_x, tips_y, tips_label, along = [], [], [], []
        for baseline in BASELINES:
            xs, ys = ratios(seq_lens, series, baseline)
            if not xs:
                continue
            drawn_any = True
            width, dashes = plots.line_style(baseline)
            ax.plot(xs, ys, color=plots.color(baseline), linewidth=width,
                    linestyle=dashes, solid_capstyle="round", zorder=4)
            along += list(zip(xs, ys))
            tips_x.append(xs[-1])
            tips_y.append(ys[-1])
            tips_label.append(baseline)

        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        # Every measured length gets a tick, but only every other one gets a
        # label: ten "512K"-sized strings do not fit across half a figure, and
        # dropping the tick as well would cost the grid line a reader counts
        # doublings against.
        ax.set_xticks(seq_lens)
        ax.set_xticklabels([f"{x // 1024}K" if (len(seq_lens) - 1 - i) % 2 == 0
                            else "" for i, x in enumerate(seq_lens)])
        ax.minorticks_off()
        ax.set_xlim(seq_lens[0] * 0.92, seq_lens[-1] * 2.9)

        # Same shading and the same reason as the latency sweep: below this the
        # step is measuring Python as much as CUDA, and a ratio built from two
        # contaminated latencies is contaminated twice over.
        if seq_lens[0] < HARNESS_LIMIT:
            ax.axvspan(seq_lens[0] * 0.92, HARNESS_LIMIT,
                       color=plots.INK_FAINT, alpha=0.10, zorder=0,
                       linewidth=0)
            # Said once, on the left panel: a ratio of two latencies that are
            # each part Python is a ratio of two contaminated numbers, and a
            # reader is owed that before reading 0.11x off the bottom left.
            if ax is axes[0]:
                ax.text(0.02, 0.89, "shaded: 11-47% of each step here\n"
                                    "is harness, on both sides of the ratio",
                        transform=ax.transAxes, color=plots.INK_SOFT,
                        fontsize=8.5, va="top", ha="left")

        # Inside the axes rather than as an axes title: the figure's own title
        # and subtitle are anchored to the left panel, and a `set_title` there
        # would be overwritten by them. The top-left corner is the one part of
        # a ratio panel guaranteed empty -- the ratios are smallest where the
        # sequence is shortest, which is the bottom-left.
        ax.text(0.02, 0.97, pass_name, transform=ax.transAxes,
                color=plots.INK_SOFT, fontsize=10, va="top", ha="left")
        # Sitting on the line rather than beside it, so the line it names runs
        # under it and comes out the other side -- but lifted to `bottom` so
        # the rule does not strike the word through.
        ax.text(seq_lens[-1] * 2.85, 1.0, "parity", color=plots.color(SUBJECT),
                fontsize=9, va="bottom", ha="right")
        plots.label_points(ax, tips_x, tips_y, tips_label, dx=8, dy=-3,
                           avoid=along)

    if not drawn_any:
        return None

    for ax in axes:
        ax.set_xlabel("sequence length", color=plots.INK_SOFT, fontsize=10)
    facts = paths.device(card)
    where = f" on an {facts['short']}" if facts.get("short") else ""
    plots.titles(
        axes[0],
        "Against dense the margin grows; against FlashMoBA it settles",
        f"How many times faster ours is than each baseline{where}. Above the "
        f"{SUBJECT}-coloured line at 1.0 we win, below it we lose; both axes "
        f"are log.\n{headline(card)}",
        "sequence length", "baseline latency / ours")
    return plots.save(fig, path)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(card: str | None = None) -> None:
    out = draw(card, os.path.join(paths.card_dir(card, "figures", True),
                                  "speedup.png"))
    if out is None:
        print(f"  no sweep for {card or paths.local_card()}; "
              "run `python -m bench.compare` first")
    else:
        print(f"  wrote {out}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    paths.add_card_argument(parser)
    main(card=parser.parse_args().card)
