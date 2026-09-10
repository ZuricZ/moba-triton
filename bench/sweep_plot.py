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

RESULTS = "results"

# Below this sequence length the measured step is substantially harness rather
# than kernel: bench/_small_s_dip put the gap between the autograd step and the
# sum of its kernels at 47% (S=1024), 21% (2048), 11% (4096) and ~0 from 8192.
HARNESS_LIMIT = 8192
SUFFIX = " (latency (ms))"

# What the reader should take away, per pass. The subtitle carries it so the
# figure survives being pasted into a slide without its prose.
CAPTIONS = {
    "fwd": (
        "Forward pass",
        "MoBA block_size=128, top_k=8, bf16, [1, S, 4, 64] on an A100-40GB. "
        "Lower is faster; both axes are log.\n"
        "The two paths that stop early exhaust 40 GB at S=32768; ours runs "
        "S=65536 in 2.7 ms."),
    "bwd": (
        "Forward and backward",
        "The full step: forward().backward(dO). Same shape and card. "
        "Lower is faster; both axes are log.\n"
        "The two paths that stop early exhaust 40 GB at S=32768."),
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
# Drawing
# ---------------------------------------------------------------------------


def draw(seq_lens, series, mode: str, path: str) -> str:
    """Draws one pass of the sweep and writes it to `path`."""
    fig, ax = plots.figure(width=9.6, height=6.0)

    tips_x, tips_y, tips_label = [], [], []
    for name, values in series.items():
        xs = [x for x, y in zip(seq_lens, values) if y is not None]
        ys = [y for y in values if y is not None]
        if not xs:
            continue
        width, dashes = plots.line_style(name)
        ax.plot(xs, ys, color=plots.color(name), linewidth=width,
                linestyle=dashes, solid_capstyle="round", zorder=4)
        # A series that stops before the sweep does ran out of memory. Mark the
        # last point it managed so the reader sees a wall, not a crop.
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

    plots.label_points(ax, tips_x, tips_y, tips_label, log=True, dx=8, dy=-3)

    title, subtitle = CAPTIONS[mode]
    plots.titles(ax, title, subtitle, "sequence length", "latency (ms)")
    return plots.save(fig, path)


def main() -> None:
    for mode in ("fwd", "bwd"):
        source = os.path.join(RESULTS, f"moba-{mode}.csv")
        if not os.path.exists(source):
            print(f"  no {source}; run `python -m bench.compare` first")
            continue
        seq_lens, series = read(source)
        out = draw(seq_lens, series, mode,
                   os.path.join(RESULTS, f"moba-{mode}.png"))
        print(f"  wrote {out}  ({len(series)} backends)")


if __name__ == "__main__":
    main()
