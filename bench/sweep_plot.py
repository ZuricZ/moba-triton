"""Renders the latency sweep as a figure meant to be read, not just produced.

WHY THIS EXISTS. `triton.testing.perf_report(save_path=...)` writes a plot for
free, and free is what it looks like. Three specific failures, none cosmetic:

* **The legend covered the data.** Eight entries in a boxed legend parked over
  the top-left quadrant, which on a log-log latency sweep is where the fast
  implementations live. The subject of the figure was behind its own key.
* **Every label ended in "(latency (ms))"**, repeating the y-axis label eight
  times and pushing the legend wider still.
* **Colour tracked list position.** `bench/plots.py` exists because of this;
  see its docstring. Two runs with different backend sets were not comparable.

It also drew `triton_moba` as an entry with no line, because that backend could
not run (`docs/environment.md`). A legend entry for an invisible series is
worse than no entry: it reads as a line hidden under another one.

WHAT THIS DRAWS INSTEAD. Lines direct-labelled at their own right-hand end, so
identity is carried by position and text rather than by a key the reader has to
cross-reference. The point where a series stops is marked, because two of them
stop by running out of memory and that is a result -- the old figure showed it
as a line that simply ended.
"""

import csv
import os

import numpy as np

from bench import plots

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

from bench import paths

# Below this sequence length the measured step is substantially harness rather
# than kernel: bench/_small_s_dip put the gap between the autograd step and the
# sum of its kernels at 47% (S=1024), 21% (2048), 11% (4096) and ~0 from 8192.
HARNESS_LIMIT = 8192
SUFFIX = " (latency (ms))"

# What the figure is of, per pass. Everything card-specific and every number
# is derived in `caption` below; these are the two facts that are properties of
# the experiment rather than of the run.
TITLES = {"fwd": "Forward pass", "bwd": "Forward and backward"}
SHAPES = {
    "fwd": "MoBA block_size=128, top_k=8, bf16, [1, S, 4, 64]",
    "bwd": "The full step: forward().backward(dO), same shape",
}

# A series that stops before the sweep does is marked, because a line that
# simply ends reads as a crop rather than as a wall -- and the wall is a
# feasibility result, not a missing measurement.
#
# The marker deliberately does not name a cause. This CSV carries latencies and
# nothing else, so the reason a series stopped is not in the data being plotted;
# inferring "out of memory" from "the line ended" would be right today and
# silently wrong the first time a backend stops for any other reason. Where the
# cause is known it goes in the caption, which is written by someone who
# checked.
STOP_NOTE = "stops here"

# ...and this is that someone, per card. Hand-written on purpose and keyed by
# card, because the *cause* cannot come from the data: a blank cell is a blank
# cell whether the backend died or was never asked. Each entry below was read
# off that card's own session log -- "no number produced: OutOfMemoryError" for
# all four early-stopping series on both cards. A card with no entry gets no
# clause, which is the honest default: an absent sentence is fine, a guessed
# one is not.
STOP_CAUSE = {
    "sm80-a100": "out of memory, per logs/sweep.log",
    "sm90-h100": "out of memory, per logs/compare.log",
}


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------


def read(path: str) -> tuple[list[float], dict[str, list[float | None]]]:
    """Reads a perf_report CSV into sequence lengths and per-backend latencies.

    Args:
        path: The CSV `triton.testing.perf_report` wrote.

    Returns:
        A tuple of (sequence lengths, {backend: latencies}), where a latency is
        None where that backend produced no number. Backends that produced no
        number anywhere are dropped rather than drawn as an empty line.
    """
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    seq_lens = [int(float(row["seq_len"])) for row in rows]
    series: dict[str, list[float | None]] = {}
    for column in rows[0]:
        if column == "seq_len":
            continue
        name = column[: -len(SUFFIX)] if column.endswith(SUFFIX) else column
        values = [float(row[column]) if row[column] else None for row in rows]
        if any(value is not None for value in values):
            series[name] = values
    return seq_lens, series


def read_sweep(card: str | None, mode: str):
    """Reads one pass, extended by the long-context re-run where there is one.

    The H100's sweep is two files. `moba-<mode>.csv` runs 1K to 128K with every
    backend in it; `moba-<mode>-long.csv` runs 128K to 512K with only the four
    that survive that far. Drawing the first alone stops the figure exactly
    where the project's claim becomes interesting, so they are joined.

    THE OVERLAP AT S=131072 RESOLVES TO THE LONG RUN. They are separate
    sessions, and the two numbers differ by well under a percent (3.942 against
    3.934 ms for `ours` forward), so this is not about accuracy. It is about
    the tail being one curve: 131072, 262144 and 524288 were measured back to
    back under the same clocks, and taking the first of those three from a
    different session would put the only session boundary in the middle of the
    steepest, most-quoted part of the line.

    A sequence length cannot appear twice however the files overlap, because
    the rows are keyed by it rather than concatenated.

    Args:
        card: Directory name such as "sm90-h100"; the local card when None.
        mode: "fwd" or "bwd".

    Returns:
        A tuple of (sequence lengths ascending, {backend: latencies}), a
        latency being None where that backend produced no number at that
        length. (None, None) when the main CSV is missing.
    """
    data = paths.card_dir(card, "data")
    main = os.path.join(data, f"moba-{mode}.csv")
    if not os.path.exists(main):
        return None, None

    rows: dict[int, dict[str, float | None]] = {}
    names: list[str] = []
    for source in (main, os.path.join(data, f"moba-{mode}-long.csv")):
        if not os.path.exists(source):
            continue
        seq_lens, series = read(source)
        for name in series:
            if name not in names:
                names.append(name)
        for i, seq_len in enumerate(seq_lens):
            # The later file wins the overlap, and wins the *whole row* rather
            # than cell by cell -- see the docstring. Filling a gap in the long
            # run from the main one would put two sessions inside a single
            # x position, which is the one thing a merged curve must not do.
            rows[seq_len] = {name: values[i] for name, values in series.items()}

    seq_lens = sorted(rows)
    merged = {name: [rows[s].get(name) for s in seq_lens] for name in names}
    return seq_lens, {name: values for name, values in merged.items()
                      if any(value is not None for value in values)}


def crossover(seq_lens, ours, baseline) -> float | None:
    """Returns the sequence length where `ours` overtakes `baseline`.

    The crossover is the project's headline claim -- a sparse kernel is slower
    than a dense one until the context is long enough for sparsity to pay for
    its routing -- so the figure marks it rather than leaving the reader to
    find where two lines swap order.

    It is **interpolated**, not snapped to a measured point. The sweep samples
    powers of two, so the crossing almost never lands on one: the forward
    ratio is 1.189 at S=8192 and 0.586 at S=16384, and returning 16384 -- the
    first point where we are already ahead -- overstates how long dense holds
    on by 69%. Reporting a grid point as the crossing quietly reports the
    sweep's resolution as a property of the kernel.

    Interpolation is linear in log-log space, which is the right model rather
    than a convenience: both curves are near power laws over this range (dense
    is quadratic in S, ours is linear), so their ratio is near-linear in log S
    and a straight line between two brackets is a good local fit. Linear
    interpolation on the raw values would bias the answer toward the larger
    bracket, since the x-grid doubles at every step.

    Returns:
        The interpolated sequence length at which the two are equal, or None
        if `ours` never overtakes within the sweep. If `ours` is already ahead
        at the first point, that point is returned -- the true crossing is
        below the sweep and cannot be located from this data.
    """
    x = np.asarray(seq_lens, dtype=float)
    mine = np.array([np.nan if v is None else v for v in ours], dtype=float)
    theirs = np.array([np.nan if v is None else v for v in baseline],
                      dtype=float)

    keep = np.isfinite(mine) & np.isfinite(theirs)
    if not keep.any():
        return None
    x, ratio = x[keep], np.log(mine[keep] / theirs[keep])

    if ratio[0] < 0:
        return float(x[0])

    # The first bracket where the log-ratio changes sign. Located explicitly
    # rather than with `np.interp`, which needs a monotone axis and would
    # return a plausible wrong answer without complaining if the ratio ever
    # doubled back -- and this project has already found one curve that does
    # (the backward's harness-driven dip).
    crossings = np.flatnonzero((ratio[:-1] >= 0.0) & (ratio[1:] < 0.0))
    if crossings.size == 0:
        return None

    i = int(crossings[0])
    frac = ratio[i] / (ratio[i] - ratio[i + 1])
    return float(np.exp(np.log(x[i]) + frac * np.log(x[i + 1] / x[i])))


# ---------------------------------------------------------------------------
# Captions
# ---------------------------------------------------------------------------


def _article(word: str) -> str:
    """Returns "a" or "an" for a word read aloud letter by letter first.

    "an A100", "an H100", "an RTX 3050" -- H, L, M, N, R, S, F and X are
    consonants whose *letter names* open on a vowel, which is the rule that
    governs an initialism. Two lines of English so that a caption naming a card
    this project has never run on still reads correctly.
    """
    return "an" if word[:1].upper() in "AEIOUFHLMNRSX" else "a"


def _last(seq_lens, values) -> tuple[int, float] | None:
    """Returns the (sequence length, latency) of a series' last real point."""
    real = [(s, v) for s, v in zip(seq_lens, values) if v is not None]
    return real[-1] if real else None


def caption(mode: str, card: str | None, seq_lens, series) -> tuple[str, str]:
    """Builds the title and subtitle from the card and the data being drawn.

    WHY THIS IS A FUNCTION AND NOT A CONSTANT. It used to be a dict of frozen
    strings, and every one of them described an A100-40GB. Drawn for the H100
    they made three false claims at once -- the wrong card, a 40 GB wall on an
    80 GB board, and "ours runs S=65536 in 2.7 ms" over a figure whose own line
    passes through 1.49 ms at that point. A caption is part of the figure, so
    it is derived from the same two sources the lines are: the card, from the
    `device.txt` its session recorded, and every number, from the CSV.

    The one thing not derived is *why* a series stops -- see `STOP_CAUSE`. Any
    clause whose ingredients are missing is dropped rather than approximated.

    Args:
        mode: "fwd" or "bwd".
        card: Directory name such as "sm90-h100"; the local card when None.
        seq_lens: The sweep's sequence lengths, ascending.
        series: {backend: latencies}, as `read_sweep` returns them.

    Returns:
        A tuple of (title, subtitle). The subtitle is one line per clause --
        stacked rather than run together because the derived clauses grow with
        the sweep, and at S=524,288 a single run-on line overflows the figure.
    """
    facts = paths.device(card)
    where = (f" on {_article(facts['short'])} {facts['short']}"
             if facts.get("short") else "")
    first = f"{SHAPES[mode]}{where}. Lower is faster; both axes are log."

    clauses = []
    stopped = [name for name, values in series.items()
               if (tip := _last(seq_lens, values)) and tip[0] < seq_lens[-1]]
    if stopped and card in STOP_CAUSE:
        listed = (" and ".join(stopped) if len(stopped) < 3
                  else ", ".join(stopped[:-1]) + " and " + stopped[-1])
        clauses.append(f"{listed} stop early: {STOP_CAUSE[card]}.")

    mine = _last(seq_lens, series["ours"]) if "ours" in series else None
    if mine:
        at, ms = mine
        headline = f"ours runs S={at:,} in {ms:.2f} ms"
        dense = dict(zip(seq_lens, series.get("sdpa_dense", []))).get(at)
        if dense:
            headline += f", {dense / ms:.1f}x sdpa_dense"
        clauses.append(headline + ".")

    return TITLES[mode], "\n".join([first] + clauses)


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

# How finely a drawn line is sampled before being handed to `label_points` as
# something to steer around. The sweep measures one point per doubling, and a
# doubling is ~110 px wide here -- far enough apart that a label can pass
# cleanly between two vertices and still sit on the segment joining them, which
# is how `naive_torch (stops here)` ended up printed across `fa2_minimal`.
# Six samples per segment puts an obstacle every ~18 px, narrower than a
# reserved mark, so no gap is left for a label to slip into.
LINE_SAMPLES = 6


def _along(xs, ys) -> list[tuple[float, float]]:
    """Samples a drawn line, in data coordinates, between its vertices.

    Interpolation is geometric on both axes, matching the log-log scales the
    figure is drawn on: on those scales a segment is a straight line, so evenly
    spaced samples in log space are evenly spaced on screen.
    """
    points = list(zip(xs, ys))
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        for step in range(1, LINE_SAMPLES):
            frac = step / LINE_SAMPLES
            points.append((x0 * (x1 / x0) ** frac, y0 * (y1 / y0) ** frac))
    return points


def draw(seq_lens, series, mode: str, path: str, card: str | None = None) -> str:
    """Draws one pass of the sweep and writes it to `path`."""
    fig, ax = plots.figure(width=9.6, height=6.0)

    tips_x, tips_y, tips_label, drawn = [], [], [], []
    for name, values in series.items():
        xs = [x for x, y in zip(seq_lens, values) if y is not None]
        ys = [y for y in values if y is not None]
        if not xs:
            continue
        drawn += _along(xs, ys)
        width, dashes = plots.line_style(name)
        ax.plot(xs, ys, color=plots.color(name), linewidth=width,
                linestyle=dashes, solid_capstyle="round", zorder=4)
        # Mark the last point a series managed, so the reader sees a wall
        # rather than a crop. Why it stopped is not said here -- see
        # `STOP_NOTE`; the CSV does not know.
        if xs[-1] < seq_lens[-1]:
            ax.scatter([xs[-1]], [ys[-1]], marker="X", s=46,
                       color=plots.color(name), zorder=5,
                       edgecolor=plots.SURFACE, linewidth=1.0)
            tips_label.append(f"{name} ({STOP_NOTE})")
        else:
            tips_label.append(name)
        tips_x.append(xs[-1])
        tips_y.append(ys[-1])

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(seq_lens)
    ax.set_xticklabels([f"{x // 1024}K" if x >= 1024 else str(x)
                        for x in seq_lens])
    ax.minorticks_off()
    # Room on the right for the direct labels, which sit outside the data.
    ax.set_xlim(seq_lens[0] * 0.92, seq_lens[-1] * 2.6)

    # Below this the measurement is not measuring the kernel. bench/_small_s_dip
    # timed the autograd step against the sum of the kernels it launches: the
    # gap is 47% of the total at S=1024, 21% at 2048, 11% at 4096 and gone by
    # 8192. That overhead is why the backward curve *falls* from S=1024 to
    # S=4096 while the work doubles -- a rising kernel cost outrun by a
    # faster-falling harness cost. Shading it is the honest fix: the dip is
    # real data and hiding it would be worse, but a reader is entitled to know
    # that region reports Python as much as CUDA.
    if seq_lens[0] < HARNESS_LIMIT:
        ax.axvspan(seq_lens[0] * 0.92, HARNESS_LIMIT, color=plots.INK_FAINT,
                   alpha=0.10, zorder=0, linewidth=0)
        ax.text(seq_lens[0] * 0.98, ax.get_ylim()[1] * 0.55,
                "harness overhead\nis 11-47% here", color=plots.INK_SOFT,
                fontsize=8.5, va="top", ha="left")

    if "ours" in series and "sdpa_dense" in series:
        at = crossover(seq_lens, series["ours"], series["sdpa_dense"])
        if at is not None:
            ax.axvline(at, color=plots.INK_FAINT, linewidth=0.9,
                       linestyle=(0, (3, 3)), zorder=1)
            ax.text(at * 1.06, ax.get_ylim()[0] * 1.5,
                    f"we pass dense attention here\n"
                    f"S ~ {round(at / 100) * 100:,} (interpolated)",
                    color=plots.INK_SOFT, fontsize=9, va="bottom")

    # Every line is an obstacle, not just every tip. The labels at the right
    # margin sit outside the data and only ever collide with each other, but a
    # `(stops here)` label is anchored in the middle of the figure and is wide
    # enough to cross two other series on its way out -- which is what
    # `moonshot_naive (stops here)` did to `naive_torch`'s X, and
    # `naive_torch (stops here)` to `fa2_minimal`'s line.
    plots.label_points(ax, tips_x, tips_y, tips_label, dx=8, dy=-3,
                       avoid=drawn)

    title, subtitle = caption(mode, card, seq_lens, series)
    plots.titles(ax, title, subtitle, "sequence length", "latency (ms)")
    return plots.save(fig, path)


def main(card: str | None = None) -> None:
    for mode in ("fwd", "bwd"):
        seq_lens, series = read_sweep(card, mode)
        if not seq_lens:
            print(f"  no moba-{mode}.csv for {card or paths.local_card()}; "
                  "run `python -m bench.compare` first")
            continue
        out = draw(seq_lens, series, mode,
                   os.path.join(paths.card_dir(card, "figures", True),
                                f"moba-{mode}.png"),
                   card=card)
        print(f"  wrote {out}  ({len(series)} backends, "
              f"S={seq_lens[0]}..{seq_lens[-1]})")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    paths.add_card_argument(parser)
    main(card=parser.parse_args().card)
