"""Layout and styling shared by the A/B learning figures.

Every figure in this package draws **one animal per figure**, so the pieces they share
are the frame around the data: the loader guard, the figure and axis setup, the session
ticks, the reference lines and the save call. The colours are the categorical slots the package
draws in, in a fixed order, so a series keeps its colour from one figure to the next.

Package-internal: the plotters import from here, nothing outside does.
"""
from __future__ import annotations

import math

import matplotlib.pyplot as plt

from hypnose_behavior.io.save import tie
from hypnose_behavior.modelling.ab_learning.data import load_ab_data

__all__ = [
    "BAND_ALPHA",
    "BOUNDARY",
    "CATEGORICAL",
    "MAX_SESSION_TICKS",
    "REFERENCE",
    "SECOND",
    "SERIES",
    "SESSION_SPAN",
    "THIRD",
    "ZERO",
    "annotation_size",
    "figure",
    "in_order",
    "index_bounds",
    "line_with_band",
    "require_data",
    "save_scope",
    "session_ticks",
    "style_axis",
    "subject_colour",
    "text_size",
]

# Categorical slots: slot 1 blue, slot 2 orange, slot 3 aqua. `SERIES` is slot 1, the
# colour of a figure that draws a single series; `SECOND` is slot 2, `THIRD` slot 3.
SERIES = "#2a78d6"
SECOND = "#eb6834"
THIRD = "#1baf7a"
REFERENCE = "#8a8984"
BAND_ALPHA = 0.18
MAX_SESSION_TICKS = 6

# A session boundary drawn across a continuous axis, and the zero line it stands on.
BOUNDARY = dict(color=REFERENCE, linestyle=":", linewidth=1.2, zorder=0)
ZERO = dict(color="#5c5b57", linewidth=0.8, zorder=0)

# All eight categorical slots in their validated order, for a figure with one line per
# animal (`subject_colour`).
CATEGORICAL = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300",
               "#4a3aa7", "#e34948")

# In a figure laid out along sessions, a session occupies this much of its slot on the x
# axis; the rest is the boundary that follows it, so a within-session segment and the
# step across the night after it never overlap.
SESSION_SPAN = 0.72

_SMALL = 0.55
_MIN_ANNOTATION = 8.0


def text_size() -> float:
    """Titles and legends follow the active style's tick-label size."""
    return plt.rcParams["xtick.labelsize"] if isinstance(
        plt.rcParams["xtick.labelsize"], (int, float)) else 10


def annotation_size() -> float:
    """Point size for text subordinate to the axis labels: headlines, legends, panel titles.

    A fraction of the tick size so it follows the active style, with a floor, since a
    headline can carry the animal's result and a style with small ticks would shrink it
    out of reading size.
    """
    return max(text_size() * _SMALL, _MIN_ANNOTATION)


def require_data(subjids, dates, selectors, data):
    """The already-loaded frames, or `load_ab_data` over the selection."""
    if data is None:
        data = load_ab_data(subjids, dates, **selectors)
    if data["sessions"].empty:
        raise ValueError("No odour-discrimination session matched the selection; check the "
                         "subjects, the selectors and that the data location is reachable.")
    return data


def figure(n_rows: int = 1, *, width: float = 6.4, height: float = 4.2):
    """A one-column figure of ``n_rows`` stacked axes, and those axes."""
    fig, axes = plt.subplots(n_rows, 1, figsize=(width, height * n_rows), squeeze=False,
                             constrained_layout=True)
    return fig, axes[:, 0]


def style_axis(ax, *, ylabel=None, ylim=(0, 1.02)):
    """The package's axis look: no top or bottom spine, a horizontal grid behind the data.

    ``ylim=None`` leaves the limits to matplotlib, for an axis whose range is not known
    in advance (a log-odds axis, say, rather than a proportion).
    """
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#e5e4df", linewidth=0.6)
    ax.set_axisbelow(True)
    if ylabel:
        ax.set_ylabel(ylabel)


def session_ticks(ax, frame):
    """Ticks at the frame's sessions, labelled by session number, thinned to fit."""
    sessions = frame.drop_duplicates("session_idx").sort_values("session_idx")
    sessions = sessions.iloc[::math.ceil(len(sessions) / MAX_SESSION_TICKS)]
    ax.set_xticks(sessions["session_idx"])
    ax.set_xticklabels(sessions["ses"].astype(str))
    ax.set_xlabel("Session")


def subject_colour(subjid) -> str:
    """An animal's colour: its `CATEGORICAL` slot by subject number, so it keeps the
    colour in any subset of the cohort."""
    return CATEGORICAL[int(subjid) % len(CATEGORICAL)]


def index_bounds(rows):
    """Each session's ``start`` / ``end`` on a choice-count axis: the choices before its
    first and up to its last. ``rows`` has ``ses``, ``session_idx`` and ``k``."""
    return (rows.groupby(["ses", "session_idx"], as_index=False)
            .agg(start=("k", "min"), end=("k", "max"))
            .assign(end=lambda b: b["end"] + 1)
            .sort_values("session_idx"))


def line_with_band(ax, frame, x, y, lo, hi, color, label=None):
    """A line of markers with its interval as a filled band; rows without ``y`` dropped.

    The band is tied to the line's ``label``, so ``show`` hides the two together.
    """
    frame = frame[frame[y].notna()].sort_values(x)
    if frame.empty:
        return
    band = ax.fill_between(frame[x], frame[lo], frame[hi], color=color, alpha=BAND_ALPHA,
                           linewidth=0)
    if label:
        tie(band, label)
    ax.plot(frame[x], frame[y], color=color, linewidth=2, marker="o", markersize=5,
            markeredgecolor="white", markeredgewidth=1, label=label)


def in_order(ax, order):
    """``ax``'s legend handles and labels, those named in ``order`` first in that order.

    Legend order is the numbering ``show`` uses, so a series whose presence varies by
    animal goes last in ``order``, where it cannot shift the others.
    """
    handles, labels = ax.get_legend_handles_labels()
    rank = {label: i for i, label in enumerate(order)}
    pairs = sorted(zip(handles, labels), key=lambda pair: rank.get(pair[1], len(rank)))
    return [h for h, _ in pairs], [lab for _, lab in pairs]


def save_scope(sessions, subjid=None) -> dict:
    """The ``subjids`` / ``dates`` a figure covers, as `io.save.save_figure` keywords.

    ``sessions`` is the session table of the loaded data; ``subjid`` narrows it to the
    one animal the figure is of.

    This returns the arguments rather than saving, because provenance capture stops at
    the first frame outside `io.save`: the ``save_figure`` call has to stand in the
    plotter's own module or the record names a helper instead of the plotter
    (``DECISIONS.md`` section 9).
    """
    if subjid is not None:
        sessions = sessions[sessions["subjid"] == subjid]
    return {"subjids": sorted(sessions["subjid"].unique().tolist()),
            "dates": sorted(sessions["date"].unique().tolist())}
