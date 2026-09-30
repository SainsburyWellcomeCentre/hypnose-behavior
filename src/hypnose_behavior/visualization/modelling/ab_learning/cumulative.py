"""The engagement, knowledge and reward figures (``modelling.ab_learning.cumulative``).

Each ``plot_*`` takes the same subject/session selection as `load_ab_data` -- or the
already-loaded frames as ``data=``. ``save=True`` files a figure with
`io.save.save_figure`.

- `plot_cumulative_panels` -- every running count on one task-time axis, one figure per
  animal (``{subjid: Figure}``).
- `plot_excess_by_index`   -- excess correct over the mode's own row index, one figure per
  animal (``{subjid: Figure}``).
- `plot_cohort_accuracy`   -- rolling accuracy over task time, every animal in one Figure.
- `plot_cohort_accuracy_normalized` -- the same on each animal's first-to-last-reward
  span, rescaled to 0-1 so the animals line up.
"""
from __future__ import annotations

import numpy as np
from matplotlib.lines import Line2D

from hypnose_behavior.io.save import save_figure
from hypnose_behavior.modelling.ab_learning.cumulative import (
    WINDOW,
    cumulative_curves,
    excess_by_index,
    rolling_accuracy,
)
from hypnose_behavior.modelling.ab_learning.data import MODES, reward_bounds, session_bounds
from hypnose_behavior.visualization.modelling.ab_learning._common import (
    BOUNDARY,
    REFERENCE,
    SECOND,
    SERIES,
    THIRD,
    annotation_size,
    figure,
    index_bounds,
    require_data,
    save_scope,
    session_axis,
    style_axis,
    subject_colour,
    text_size,
    title,
)

__all__ = ["ACCURACY_MODES", "CURVE_STYLES", "NORMALIZED_WINDOW", "plot_cohort_accuracy",
           "plot_cohort_accuracy_normalized", "plot_cumulative_panels", "plot_excess_by_index"]

# One look per running count, the same in every figure. Rewards sit behind the other two:
# they are their product, so they are context rather than the comparison.
CURVE_STYLES = {
    "initiations": dict(color=SERIES, linewidth=2.2, zorder=3, label="initiations"),
    "attempts": dict(color=SERIES, linewidth=1.6, linestyle=(0, (4, 2)), zorder=2,
                     label="choice attempts"),
    "excess": dict(color=SECOND, linewidth=2.2, zorder=4, label="excess correct"),
    "rewards": dict(color=THIRD, linewidth=1.6, alpha=0.45, zorder=1, label="rewards"),
}

# `plot_cohort_accuracy` modes, and each choice mode's line: completed trials solid,
# choice attempts dashed.
ACCURACY_MODES = MODES + ("both",)
_MODE_LINES = {"completed": ("-", "trials"), "attempts": ((0, (4, 2)), "choice attempts")}

# Choices per window on the normalized clock: long enough to show the trend, not the
# session-to-session wobble.
NORMALIZED_WINDOW = 150

_WIDTH = 10.0
_HEIGHT = 5.0


def _save(fig, name, data, save, subjid):
    if not save:
        return
    save_figure(fig, name, **save_scope(data["sessions"], subjid))


def plot_cumulative_panels(subjids=None, dates=None, *, data=None, mode="completed",
                           save=False, **selectors):
    """Initiations, excess correct and rewards over task time, one figure per animal.

        plot_cumulative_panels(data=ab)
        plot_cumulative_panels(data=ab, mode="attempts")

    Every count is **divided by its own largest absolute value**, so each ends near 1 and
    the lines compare by shape; the legend gives the unscaled final counts. Task time
    counts only the hours an odour-discrimination run was recording, so nights and the
    gaps between runs take no width. Dotted verticals are session boundaries.

    - initiations: trials initiated -- engagement. In ``"attempts"`` mode a dashed line
      adds the failed attempts that ended in a port visit.
    - excess correct: ``cumsum(y - 0.5)`` over the mode's choices; flat is chance. Its
      slope per hour is choice rate times (accuracy - 0.5), so a rise it shows and the
      initiations do not is a change in accuracy, and a bend both share is engagement.
      Once scaled, it runs steeper than the initiations roughly where accuracy is above
      the animal's own average and flatter where it is below (roughly: initiations also
      count the trials that ended without a scored choice).
    - rewards: the product of the two, drawn faded behind them.
    """
    data = require_data(subjids, dates, selectors, data)
    curves = cumulative_curves(data, mode)
    bounds = session_bounds(data)

    figures = {}
    for subjid in sorted(curves["subjid"].unique()):
        animal = curves[curves["subjid"] == subjid]
        spans = bounds[bounds["subjid"] == subjid].sort_values("session_idx")
        fig, (ax,) = figure(width=_WIDTH, height=_HEIGHT)
        for name, style in CURVE_STYLES.items():
            part = animal[animal["series"] == name]
            if part.empty:
                continue
            scale = float(part["value"].abs().max()) or 1.0
            label = f"{style['label']} ({part['value'].iloc[-1]:g})"
            # From zero at the clock's start, so hours before the first event read as a
            # flat count rather than as missing data.
            ax.plot(np.r_[0.0, part["time"]], np.r_[0.0, part["value"] / scale],
                    drawstyle="steps-post", **{**style, "label": label})
        for start in spans["start"].to_numpy()[1:]:
            ax.axvline(start, **BOUNDARY)
        ax.axhline(0, color=REFERENCE, linestyle="--", linewidth=1, zorder=0)
        style_axis(ax, ylabel="scaled count", ylim=None)
        ax.set_xlim(0, float(np.nanmax(spans["end"])) if len(spans) else None)
        ax.set_xlabel("task time (h)")
        session_axis(ax, spans)
        ax.legend(frameon=False, fontsize=annotation_size(), loc="upper left")
        title(fig, f"initiations, excess correct, rewards | {mode}", subjid)
        _save(fig, f"ab_learning_cumulative_panels_sub-{subjid:03d}", data, save, subjid)
        figures[int(subjid)] = fig
    return figures


def plot_excess_by_index(subjids=None, dates=None, *, data=None, mode="completed",
                         save=False, **selectors):
    """Excess correct over the mode's own row index, one figure per animal.

        plot_excess_by_index(data=ab)
        plot_excess_by_index(data=ab, mode="attempts")

    ``cumsum(y - 0.5)`` against the number of choices so far: completed trials, or every
    choice attempt. Each choice is one step whatever time it took, so the slope is
    accuracy minus 0.5 and engagement is off the axis: flat is chance, a slope of 0.5
    is all correct, and a bend is a change in accuracy. Unscaled, in choices. Dotted
    verticals are session boundaries.
    """
    data = require_data(subjids, dates, selectors, data)
    excess = excess_by_index(data, mode)
    unit = "trials" if mode == "completed" else "choice attempts"

    figures = {}
    for subjid in sorted(excess["subjid"].unique()):
        animal = excess[excess["subjid"] == subjid]
        spans = index_bounds(animal)
        fig, (ax,) = figure(width=_WIDTH, height=_HEIGHT)
        ax.plot(np.r_[0, animal["k"] + 1], np.r_[0.0, animal["excess"]],
                drawstyle="steps-post", **{**CURVE_STYLES["excess"], "label": None})
        for start in spans["start"].to_numpy()[1:]:
            ax.axvline(start, **BOUNDARY)
        ax.axhline(0, color=REFERENCE, linestyle="--", linewidth=1, zorder=0)
        style_axis(ax, ylabel="excess correct", ylim=None)
        ax.set_xlim(0, len(animal))
        ax.set_xlabel(unit)
        session_axis(ax, spans)
        title(fig, f"excess correct by {unit[:-1]} | {mode}", subjid)
        _save(fig, f"ab_learning_excess_by_index_{mode}_sub-{subjid:03d}", data, save,
              subjid)
        figures[int(subjid)] = fig
    return figures


def _accuracy_figure(data, mode, window, clock, xlabel, xlim, what):
    """The cohort accuracy figure, on the x axis ``clock(subjid, hours)`` maps task hours to.

    Session boundaries, drawn for a single animal, go through ``clock`` too; ``xlim`` of
    None runs from 0 to the end of the last session. Returns the figure and the animals.
    """
    if mode not in ACCURACY_MODES:
        raise ValueError(f"mode must be one of {ACCURACY_MODES}, got {mode!r}")
    modes = MODES if mode == "both" else (mode,)
    frames = {m: rolling_accuracy(data, m, window) for m in modes}
    subjects = sorted(set().union(*(frame["subjid"].unique() for frame in frames.values())))
    bounds = session_bounds(data)
    bounds = bounds[bounds["subjid"].isin(subjects)]

    fig, (ax,) = figure(width=_WIDTH, height=_HEIGHT)
    for subjid in subjects:
        for m in modes:
            rows = frames[m][frames[m]["subjid"] == subjid]
            ax.plot(clock(subjid, rows["hours"].to_numpy()), rows["accuracy"],
                    color=subject_colour(subjid), linewidth=1.8, linestyle=_MODE_LINES[m][0],
                    label=f"sub-{subjid:03d}" if m == modes[0] else None)
    ax.axhline(0.5, color=REFERENCE, linestyle="--", linewidth=0.8, zorder=0)
    ax.set_ylim(0, 1)
    ax.set_xlim(*(xlim or (0, float(bounds["end"].max()))))
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_ylabel("accuracy")
    ax.set_xlabel(xlabel)

    handles, labels = ax.get_legend_handles_labels()
    if len(modes) > 1:
        for m in modes:
            handles.append(Line2D([], [], color=REFERENCE, linewidth=1.8,
                                  linestyle=_MODE_LINES[m][0]))
            labels.append(_MODE_LINES[m][1])
    ax.legend(handles, labels, frameon=False, fontsize=annotation_size(), loc="lower right",
              ncols=2 if len(handles) > 4 else 1)

    what = f"{what} | {mode}, {window}-choice window"
    if len(subjects) == 1:
        subjid = subjects[0]
        spans = bounds.sort_values("session_idx")
        spans = spans.assign(start=clock(subjid, spans["start"].to_numpy()),
                             end=clock(subjid, spans["end"].to_numpy()))
        for start in spans["start"].to_numpy()[1:]:
            ax.axvline(start, **BOUNDARY)
        session_axis(ax, spans)
        title(fig, what, subjid)
    else:
        fig.suptitle(f"cohort | {what}", x=0.01, ha="left", fontsize=text_size())
    return fig, subjects


def _save_name(stem, mode, subjects):
    """``{stem}_{mode}``, with the animal when the figure holds one."""
    single = f"_sub-{subjects[0]:03d}" if len(subjects) == 1 else ""
    return f"ab_learning_{stem}_{mode}{single}"


def plot_cohort_accuracy(subjids=None, dates=None, *, data=None, mode="completed",
                         window=WINDOW, save=False, **selectors):
    """Accuracy over task time, every animal of the selection in one figure.

        plot_cohort_accuracy(data=ab)
        plot_cohort_accuracy(data=ab, mode="both", window=30)

    Each line is the share correct over ``window`` consecutive choices
    (`cumulative.rolling_accuracy`), drawn at each choice's task time, in the animal's own
    colour. ``mode`` is ``"completed"``, ``"attempts"`` or ``"both"`` (completed solid,
    attempts dashed). Session boundaries are drawn when the selection is one animal.
    Returns the Figure.
    """
    data = require_data(subjids, dates, selectors, data)
    fig, subjects = _accuracy_figure(data, mode, window, lambda subjid, hours: hours,
                                     "task time (h)", None, "accuracy over task time")
    if save:
        save_figure(fig, _save_name("cohort_accuracy", mode, subjects),
                    **save_scope(data["sessions"]))
    return fig


def plot_cohort_accuracy_normalized(subjids=None, dates=None, *, data=None,
                                    mode="completed", window=NORMALIZED_WINDOW, save=False,
                                    **selectors):
    """Accuracy over each animal's own span of rewarded task time, all animals aligned.

        plot_cohort_accuracy_normalized(data=ab)
        plot_cohort_accuracy_normalized(data=ab, mode="both", window=200)

    As `plot_cohort_accuracy`, with task time rescaled per animal: 0 at its first reward,
    1 at its last (`data.reward_bounds`), so every animal spans the same width. Choices
    before the first reward fall left of 0 and are not shown; the window still counts
    them. Returns the Figure.
    """
    data = require_data(subjids, dates, selectors, data)
    rewards = reward_bounds(data).set_index("subjid")

    def clock(subjid, hours):
        first, last = rewards.loc[subjid, ["first", "last"]]
        return (hours - first) / (last - first)

    fig, subjects = _accuracy_figure(data, mode, window, clock,
                                     "time, first to last reward", (0, 1),
                                     "accuracy over normalized task time")
    if save:
        save_figure(fig, _save_name("cohort_accuracy_normalized", mode, subjects),
                    **save_scope(data["sessions"]))
    return fig
