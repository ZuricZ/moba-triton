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
    ax.set_title(title, color=INK, fontsize=13, fontweight="bold",
                 loc="left", pad=18 if subtitle else 10)
    if subtitle:
        ax.text(0.0, 1.02, subtitle, transform=ax.transAxes,
                color=INK_SOFT, fontsize=9.5, va="bottom", ha="left")
    ax.set_xlabel(xlabel, color=INK_SOFT, fontsize=10)
    ax.set_ylabel(ylabel, color=INK_SOFT, fontsize=10)


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
    import math

    def key(x, y):
        return (math.log10(max(x, 1e-12)), math.log10(max(y, 1e-12))) if log             else (x, y)

    order = sorted(range(len(labels)), key=lambda i: -ys[i])
    placed: list[tuple[float, float, int]] = []
    for i in order:
        kx, ky = key(xs[i], ys[i])
        step = 0
        for px, py, pstep in placed:
            if abs(px - kx) < tol and abs(py - ky) < tol:
                step = max(step, pstep + 1)
        placed.append((kx, ky, step))
        ax.annotate(labels[i], (xs[i], ys[i]), textcoords="offset points",
                    xytext=(dx, dy - 13 * step), fontsize=9, color=INK_SOFT)


def save(fig, path: str) -> str:
    """Writes the figure and returns the path."""
    fig.savefig(path, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return path
