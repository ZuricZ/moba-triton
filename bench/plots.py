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
from matplotlib.transforms import Bbox      # noqa: E402

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


def style_axes(ax):
    """Applies the house style to one axes: recessive frame, no chartjunk.

    Top and right spines removed and the survivors muted, because the data
    should be the highest-contrast thing in the frame. Grid sits behind the
    marks at low alpha -- present enough to read a value off, quiet enough not
    to compete with a 2.4pt line.

    Factored out of `figure` so a multi-panel figure gets the identical
    treatment rather than a second, drifting copy of it.
    """
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_FAINT)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=INK_SOFT, labelsize=9, length=3, width=0.8)
    ax.grid(True, which="major", color=INK_FAINT, alpha=0.18, linewidth=0.6)
    ax.set_axisbelow(True)
    return ax


def figure(width: float = 9.0, height: float = 5.6):
    """Returns a styled (fig, ax)."""
    fig, ax = plt.subplots(figsize=(width, height), dpi=160)
    fig.patch.set_facecolor(SURFACE)
    return fig, style_axes(ax)


def panels(count: int = 2, width: float = 9.6, height: float = 5.0,
           sharey: bool = True):
    """Returns a styled (fig, axes) row of panels sharing one y scale.

    A row rather than a grid, because the only reason this project splits a
    figure is to hold one variable constant across panels (forward against
    backward, say) and the eye compares left-to-right more reliably than it
    compares across two axes.

    Args:
        count: How many panels.
        width: Figure width in inches, for the whole row.
        height: Figure height in inches.
        sharey: Whether the panels share a y scale. True is almost always
            right: two panels with independent y axes invite a comparison the
            drawing does not support.

    Returns:
        A tuple of (figure, list of axes).
    """
    fig, axes = plt.subplots(1, count, figsize=(width, height), dpi=160,
                             sharey=sharey)
    fig.patch.set_facecolor(SURFACE)
    axes = list(np.atleast_1d(axes))
    for ax in axes:
        style_axes(ax)
    return fig, axes


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


def legend(ax, mask: bool = False, **kwargs):
    """A legend with no box, since the frame adds nothing.

    Args:
        mask: Whether to fill the legend's patch with the surface colour. Still
            frameless to look at -- no edge, no shadow -- but opaque, so a
            reference line that happens to run under it is interrupted rather
            than printed through its labels. The roofline's ridge is vertical
            and lands in the legend on every card measured so far.
    """
    if mask:
        kwargs.update(frameon=True, facecolor=SURFACE, edgecolor="none",
                      framealpha=1.0)
    else:
        kwargs.setdefault("frameon", False)
    return ax.legend(fontsize=9.5, labelcolor=INK_SOFT, **kwargs)


def _renderer(fig):
    """Returns a renderer that text extents can be measured against."""
    try:
        return fig.canvas.get_renderer()
    except AttributeError:            # a backend that only builds one on draw
        fig.canvas.draw()
        return fig.canvas.get_renderer()


# A backdrop in the surface colour, so a label reads even where a leader line
# or a grid rule passes behind it. Invisible on its own -- same colour as the
# page, no edge -- and it only ever masks something a label already sits on
# top of, which staggering alone cannot always prevent once leaders are in the
# figure.
_HALO = dict(boxstyle="square,pad=0.16", facecolor=SURFACE, edgecolor="none")


def _overlap_area(box, other, shift: float) -> float:
    """Returns the pixel area `box`, moved up by `shift`, shares with `other`."""
    wide = min(box.x1, other.x1) - max(box.x0, other.x0)
    tall = min(box.y1 + shift, other.y1) - max(box.y0 + shift, other.y0)
    return wide * tall if wide > 0 and tall > 0 else 0.0


def label_points(ax, xs, ys, labels, dx=6, dy=6, avoid=(), pad=2.0,
                 avoid_radius=6.0) -> None:
    """Direct-labels marks, staggering any whose *rendered text* would collide.

    Required, not decorative: three palette slots sit below 3:1 contrast
    against the surface, so identity has to be carried by something other than
    colour. Text wears ink tokens rather than the series colour -- the mark
    beside it carries identity, and coloured text at 9pt is the thing that
    fails a contrast check.

    WHY THIS IS MEASURED AND NOT ESTIMATED. The first version compared the
    *anchor points*: two labels were a cluster when their data coordinates were
    within 0.08 decades on both axes. That tests the wrong object. A label is a
    string up to ~28 characters wide, so `moonshot_naive (stops here)` at
    S=16384 and `naive_torch (stops here)` at S=32768 -- 0.30 decades apart,
    comfortably outside the tolerance -- printed straight through each other on
    the H100 backward sweep. Anchors do not overlap; boxes do. So the boxes are
    what gets tested, measured off the renderer that will draw them, in display
    pixels.

    The placement is greedy from the top down and takes the nearest free slot,
    preferring downward at equal distance. That keeps the behaviour the
    stagger was written for -- genuinely co-located marks (the roofline's
    overlapping implementations) still separate, topmost staying put and the
    next one dropping a line -- while a label never travels further from its
    own mark than it must.

    Args:
        ax: The axes, with its scales and limits already final -- the boxes are
            measured through `transData` and are meaningless before that.
        xs, ys, labels: Anchor coordinates and the string for each.
        dx: Horizontal offset from the anchor, in points. Never adjusted; a
            label that moved sideways would stop reading as belonging to its
            own mark.
        dy: Vertical offset from the anchor, in points, before staggering.
        avoid: Extra (x, y) data points the text must also clear -- the
            roofline's non-labelled sequence lengths, where the old code let
            `naive_torch` print over the `flash_moba` marker, or a sampling
            along each drawn line, which is how a label is kept off a curve it
            does not name. A label never avoids its own anchor.
        pad: Breathing room around each text box, in pixels.
        avoid_radius: Half-size of the box reserved for a point in `avoid`, in
            points. Roughly the largest marker this project draws.
    """
    fig = ax.figure
    # Text the caller has already placed -- region names, a ridge annotation,
    # the note over a shaded band. It was put where it was deliberately, so it
    # is an obstacle rather than a peer: direct labels move around it and it
    # never moves. Collected before ours exist so the list cannot include them.
    standing = [text for text in ax.texts if text.get_text().strip()]

    anns = [ax.annotate(label, (x, y), textcoords="offset points",
                        xytext=(dx, dy), fontsize=9, color=INK_SOFT,
                        annotation_clip=False, zorder=6, bbox=_HALO)
            for x, y, label in zip(xs, ys, labels)]
    if not anns:
        return

    renderer = _renderer(fig)
    boxes = [a.get_window_extent(renderer).padded(pad) for a in anns]
    fixed = [text.get_window_extent(renderer).padded(pad) for text in standing]

    # Marks to steer around, as display-space boxes. Keyed by their data
    # coordinates so a label can exempt its own anchor: the text sits up and to
    # the right of the mark it names, and colliding with it is the intended
    # arrangement rather than a fault.
    px_per_pt = fig.dpi / 72.0
    reach = avoid_radius * px_per_pt
    blockers = []
    for bx, by in avoid:
        cx, cy = ax.transData.transform((bx, by))
        blockers.append(((bx, by), Bbox.from_extents(cx - reach, cy - reach,
                                                     cx + reach, cy + reach)))

    # One step is a full line of text, so a single displacement always clears
    # the label it collided with rather than half-clearing it.
    step = max(box.height for box in boxes)
    # Tallest first, so the result does not depend on the caller's input order.
    order = np.argsort(-np.asarray(ys, dtype=float))
    # Down before up, and by increasing distance: the nearest free slot wins,
    # so a label never travels further from its mark than it has to.
    slots = [0.0]
    for k in range(1, len(order) + 2):
        slots += [-k, k]

    placed = list(fixed)
    for i in order:
        obstacles = placed + [box for anchor, box in blockers
                              if anchor != (xs[i], ys[i])]
        box = boxes[i]
        # A free slot ends the search. Where none is free -- a label whose
        # anchor sits inside a thicket of other lines -- the least-obstructed
        # slot is taken rather than the last one tried, which would fling the
        # text to the far end of the search range for no gain.
        best, best_cost = 0.0, None
        for k in slots:
            cost = sum(_overlap_area(box, other, k * step)
                       for other in obstacles)
            if cost == 0:
                best = k
                break
            if best_cost is None or cost < best_cost:
                best, best_cost = k, cost
        moved = Bbox.from_extents(box.x0, box.y0 + best * step,
                                  box.x1, box.y1 + best * step)
        placed.append(moved)
        offset = (dx, dy + best * step / px_per_pt)
        if not best:
            anns[i].set_position(offset)
            continue
        # A displaced label has lost the one thing that said which mark it
        # names: adjacency. Give it a hairline back to its own point. Cheaper
        # than a legend entry and it cannot disagree with the drawing, which a
        # legend can. Re-made rather than mutated because an annotation's arrow
        # is built at construction.
        anns[i].remove()
        ax.annotate(labels[i], (xs[i], ys[i]), textcoords="offset points",
                    xytext=offset, fontsize=9, color=INK_SOFT,
                    annotation_clip=False, zorder=6, bbox=_HALO,
                    arrowprops=dict(arrowstyle="-", color=INK_FAINT,
                                    linewidth=0.7, shrinkA=3, shrinkB=5))


def save(fig, path: str) -> str:
    """Writes the figure and returns the path."""
    fig.savefig(path, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return path
