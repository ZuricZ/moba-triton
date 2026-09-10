"""Shared plot styling, so every figure in this project reads as one system.

Before this module each plot inherited matplotlib's defaults by way of
`triton.testing.perf_report`, which cycles colours in whatever order the
providers happen to be listed. Two problems with that, both of which make a
figure harder to read rather than merely plainer:

* **Colour tracked position, not identity.** Drop one backend from a sweep and
  every remaining line changes colour, so two figures of the same experiment
  could not be compared side by side.
* **Every line looked equally important.** Six identical lines read as a
  six-way race. Ours is the subject, two are dense reference lines, one is an
  oracle we only ever want to beat by a lot -- that structure should be visible
  before the legend is read.

The palette is the validated categorical set from the project's data-viz
reference, checked with its validator: all six inside the L 0.43-0.77 band,
chroma >= 0.1, worst adjacent CVD separation dE 9.1 (protan), worst normal
vision dE 19.6. Three slots fall below 3:1 contrast against the surface, which
is why `label_points` exists and why every plot here carries a legend -- colour
is never the only channel carrying identity.
"""

import matplotlib
matplotlib.use("Agg")                       # no display on a headless box
import matplotlib.pyplot as plt             # noqa: E402
import numpy as np                          # noqa: E402

# ---------------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------------

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SOFT = "#52514e"
INK_FAINT = "#8b8a85"

# Fixed slot order. A backend keeps its colour whether or not its neighbours
# are in the figure, which is the property that makes two runs comparable.
SERIES = {
    "ours": "#2a78d6",          # blue    -- the subject
    "flash_moba": "#eb6834",    # orange  -- the CUDA target
    "triton_moba": "#1baf7a",   # aqua    -- the other Triton MoBA
    "moonshot_efficient": "#eda100",   # yellow  -- their fused path
    "moonshot_naive": "#4a3aa7",       # violet  -- their readable path
    "sdpa_dense": "#e87ba4",    # magenta -- dense baseline
    "fa2_minimal": "#008300",   # green   -- our own dense baseline
    "naive_torch": "#8c5a3c",   # brown   -- our oracle
}
FALLBACK = INK_SOFT

# What kind of thing each line is. Encoded as dash pattern so the structure
# survives greyscale printing and colour-vision deficiency alike.
ROLE = {
    "ours": (2.4, "-"),
    "flash_moba": (1.8, (0, (5, 2))),
    "triton_moba": (1.8, (0, (5, 2))),
    "moonshot_efficient": (1.8, (0, (5, 2))),
    "moonshot_naive": (1.4, (0, (6, 2, 1, 2))),
    "sdpa_dense": (1.5, (0, (1, 2))),
    "fa2_minimal": (1.5, (0, (1, 2))),
    "naive_torch": (1.4, (0, (6, 2, 1, 2))),
}


def color(name: str) -> str:
    """Returns the fixed colour for a backend, by identity not position."""
    return SERIES.get(name.replace("_bwd", ""), FALLBACK)


def line_style(name: str) -> tuple[float, object]:
    """Returns (linewidth, dash pattern) for a backend."""
    return ROLE.get(name.replace("_bwd", ""), (1.5, "-"))


# ---------------------------------------------------------------------------
# Figure furniture
# ---------------------------------------------------------------------------


def figure(width: float = 9.0, height: float = 5.6):
    """Returns a styled (fig, ax): recessive axes, no chartjunk.

    Top and right spines removed and the survivors muted, because the data
    should be the highest-contrast thing in the frame. Grid sits behind the
    marks at low alpha -- present enough to read a value off, quiet enough not
    to compete with a 2.4pt line.
    """
    fig, ax = plt.subplots(figsize=(width, height), dpi=160)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_FAINT)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=INK_SOFT, labelsize=9, length=3, width=0.8)
    ax.grid(True, which="major", color=INK_FAINT, alpha=0.18, linewidth=0.6)
    ax.set_axisbelow(True)
    return fig, ax


def titles(ax, title: str, subtitle: str = "",
           xlabel: str = "", ylabel: str = "") -> None:
    """Sets a title, an explanatory subtitle, and axis labels.

    The subtitle carries the reading instruction -- units, configuration, what
    "better" means. A reader who arrives at a figure cold should not have to
    find the prose to know which direction is good.
    """
    # The subtitle is anchored to the top of the axes and grows *upward*, so
    # the title has to be pushed up by its full height or a second line of
    # subtitle prints through it. Pad is in points; 12.5 is the line advance at
    # 9.5pt.
    lines = subtitle.count("\n") + 1 if subtitle else 0
    ax.set_title(title, color=INK, fontsize=13, fontweight="bold",
                 loc="left", pad=10 + 12.5 * lines)
    if subtitle:
        ax.text(0.0, 1.015, subtitle, transform=ax.transAxes,
                color=INK_SOFT, fontsize=9.5, va="bottom", ha="left",
                linespacing=1.35)
    ax.set_xlabel(xlabel, color=INK_SOFT, fontsize=10)
    ax.set_ylabel(ylabel, color=INK_SOFT, fontsize=10)


def slope_degrees(ax, slope: float = 1.0) -> float:
    """Returns the on-screen angle of a line of the given slope, in degrees.

    Text that labels a line has to be rotated to match it, and the naive way --
    hardcoding 45 degrees for a slope-1 line -- is wrong on every plot that is
    not square. The on-screen angle depends on the figure's aspect ratio and,
    on a log axis, on how many decades each axis spans. A roofline that spans
    five decades of x and four of y draws its slope-1 roof at nothing like 45
    degrees, and the label then visibly diverges from the line it names.

    So ask the axes instead: transform two points that differ by `slope` in
    data space into display pixels and take the angle between them. This
    survives a resize or a limit change, because it is read off the same
    transform that draws the line.

    Must be called after the scales and limits are set; before that the
    transform is still the default and the answer is meaningless.

    Args:
        ax: The axes, fully configured.
        slope: Rise over run in *scaled* units -- 1.0 is a decade of y per
            decade of x on a log-log plot.

    Returns:
        Degrees counter-clockwise from horizontal.
    """
    # Two real data points on a line of this slope, transformed by the same
    # transform that draws the data. An earlier version composed transLimits
    # with transData, which double-applies the axis scaling and returned ~85
    # degrees for a line drawn at ~35 -- the label came out vertical.
    x0, x1 = ax.get_xlim()
    y0, _ = ax.get_ylim()
    if ax.get_xscale() == "log":
        # One decade along x; `slope` decades along y.
        pair = [(x0 * 2.0, y0 * 2.0), (x0 * 20.0, y0 * 2.0 * 10.0 ** slope)]
    else:
        span = (x1 - x0) * 0.1
        pair = [(x0 + span, y0), (x0 + 2 * span, y0 + slope * span)]

    (px0, py0), (px1, py1) = ax.transData.transform(pair)
    return float(np.degrees(np.arctan2(py1 - py0, px1 - px0)))


def legend(ax, **kwargs):
    """A legend with no box, since the frame adds nothing."""
    leg = ax.legend(frameon=False, fontsize=9.5, labelcolor=INK_SOFT, **kwargs)
    return leg


def label_points(ax, xs, ys, labels, colors=None, dx=6, dy=6,
                 log=False, tol=0.08) -> None:
    """Direct-labels marks, staggering any that would collide.

    Required, not decorative: three palette slots sit below 3:1 contrast
    against the surface, so identity has to be carried by something other than
    colour. Text wears ink tokens rather than the series colour -- the mark
    beside it carries identity, and coloured text at 9pt is the thing that
    fails a contrast check.

    Two implementations with the same modelled traffic and similar throughput
    land on the same point, and their labels then print over each other. The
    stagger below is why the first render of the roofline was wrong: colliding
    labels are the most common way a technically correct figure becomes
    unreadable.

    Args:
        log: Compare positions in log space, for log-scaled axes.
        tol: Points closer than this (in decades when `log`) are one cluster.
    """
    # Comparison coordinates, distinct from the data coordinates the annotation
    # is drawn at: on a log axis two points are "close" when their logarithms
    # are, not their values.
    key_x = np.asarray(xs, dtype=float)
    key_y = np.asarray(ys, dtype=float)
    if log:
        key_x = np.log10(np.maximum(key_x, 1e-12))
        key_y = np.log10(np.maximum(key_y, 1e-12))

    # Which pairs collide, as one n x n mask rather than a rescan of every
    # already-placed label for every new one. Broadcasting states the rule
    # once -- two labels collide when they are within `tol` on both axes --
    # where the nested scan spread it over a loop, an accumulator and a
    # running max.
    collides = ((np.abs(key_x[:, None] - key_x[None, :]) < tol)
                & (np.abs(key_y[:, None] - key_y[None, :]) < tol))

    # Tallest first, so a cluster stacks downward from its topmost member and
    # the result does not depend on the caller's input order.
    order = np.argsort(-np.asarray(ys, dtype=float))
    # A label's offset is how many colliding labels were placed before it.
    # Reindexing the mask into placement order turns "placed before" into
    # "strictly below the diagonal", so the whole stagger is one tril + sum.
    earlier = np.tril(collides[np.ix_(order, order)], k=-1)
    steps = np.empty(len(order), dtype=int)
    steps[order] = earlier.sum(axis=1)

    for i, step in enumerate(steps):
        ax.annotate(labels[i], (xs[i], ys[i]), textcoords="offset points",
                    xytext=(dx, dy - 13 * int(step)), fontsize=9,
                    color=INK_SOFT)


def save(fig, path: str) -> str:
    """Writes the figure and returns the path."""
    fig.savefig(path, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return path
