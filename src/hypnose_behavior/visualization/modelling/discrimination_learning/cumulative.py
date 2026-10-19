"""The engagement, knowledge and reward figures
(``modelling.discrimination_learning.cumulative``).

Each ``plot_*`` takes the same subject/session selection as `load_discrimination_data` --
or the already-loaded frames as ``data=``. ``save=True`` files a figure with
`io.save.save_figure`.

- `plot_cumulative_panels` -- every running count on one task-time axis, one figure per
  animal (``{subjid: Figure}``).
- `plot_excess_by_index`   -- excess correct over the mode's own row index, one figure per
  animal (``{subjid: Figure}``).
- `plot_cohort_accuracy`   -- rolling accuracy over task time, every animal in one Figure.
- `plot_cohort_accuracy_normalized` -- the same on each animal's first-to-last-reward
  span, rescaled to 0-1 so the animals line up.

Each takes ``separate_odors``: every odour drawn on its own in its reward port's colour,
and ``show_combined`` adds both together in grey.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from hypnose_behavior.io.save import (
    finish_figure,
    legend_figure,
    save_figure,
    show_suffix,
)
from hypnose_behavior.modelling.discrimination_learning.cumulative import (
    WINDOW,
    cumulative_curves,
    excess_by_index,
    excess_by_odor,
    rolling_accuracy,
)
from hypnose_behavior.modelling.discrimination_learning.data import (
    MODES, reward_bounds, session_bounds,
)
from hypnose_behavior.visualization.modelling.discrimination_learning._common import (
    BAND_ALPHA,
    BOUNDARY,
    COMBINED,
    PORT_COLOURS,
    REFERENCE,
    SECOND,
    SERIES,
    THIRD,
    ZERO,
    annotation_size,
    figure,
    index_bounds,
    odor_parts,
    odors_tag,
    require_data,
    save_scope,
    style_axis,
    subject_colour,
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
# The same counts drawn per odour: the colour is the odour's, so the counts part by line.
_ODOR_CURVE_STYLES = {
    "initiations": dict(linewidth=1.6, linestyle=(0, (4, 2)), zorder=3),
    "attempts": dict(linewidth=1.4, linestyle=(0, (1, 1.5)), zorder=2),
    "excess": dict(linewidth=2.4, zorder=4),
    "rewards": dict(linewidth=1.4, alpha=0.4, zorder=1),
}

# `plot_cohort_accuracy` modes, and each choice mode's line: completed trials solid,
# choice attempts dashed.
ACCURACY_MODES = MODES + ("both",)
_MODE_LINES = {"completed": ("-", "trials"), "attempts": ((0, (4, 2)), "choice attempts")}

# Choices per window on the normalized clock: long enough to show the trend, not the
# session-to-session wobble.
NORMALIZED_WINDOW = 150

# With the cohort mean on top: each animal a faint grey line, the mean a thick dark one
# over its SEM band, taken at this many points across the axis.
_INDIVIDUAL = dict(color=REFERENCE, linewidth=1.2, alpha=0.45)
_MEAN = dict(color="#222222", linewidth=3.0, zorder=5)
_MEAN_POINTS = 201

_WIDTH = 10.0
_HEIGHT = 5.0


def _save(fig, name, data, save, subjid):
    if not save:
        return
    save_figure(fig, name, **save_scope(data["sessions"], subjid))


def plot_cumulative_panels(subjids=None, dates=None, *, data=None, mode="completed",
                           separate_odors=False, show_combined=False, legend=None, show=None,
                           save=False, **selectors):
    """Initiations, excess correct and rewards over task time, one figure per animal.

        plot_cumulative_panels(data=ab)
        plot_cumulative_panels(data=ab, mode="attempts")
        plot_cumulative_panels(data=ab, separate_odors=True, show_combined=True)

    Every count is **divided by its own largest absolute value**, so each ends near 1 and
    the lines compare by shape; the unscaled final counts are printed as a table, one row
    per animal. Task time counts only the hours an odour-discrimination run was
    recording, so nights and the gaps between runs take no width. Dotted verticals are
    session boundaries.

    ``separate_odors`` draws every count once per odour, counting that odour's trials and
    choices only, in its reward port's colour: excess correct thick, initiations dashed,
    rewards faded (choice attempts dotted). ``show_combined`` adds both odours together
    in grey, the pooled figure's lines. The table gets a row per odour.

    ``legend``: True draws it in each figure, False in one legend-only figure (shown,
    never saved), None as `use_style` says -- apart under the presentation style.

    ``show`` draws only some lines, to build a slide up step by step: legend numbers
    (initiations 1, excess correct 2, rewards 3; in ``"attempts"`` mode choice attempts is
    2) or labels, in any order, e.g. ``[1]``, ``[1, 2]``, ``[3, 1]``. None draws them all.
    Per odour the numbers run through the first odour's lines, then the next's: ``[2, 5]``
    is excess correct of both. The axes stay as with every line, and each selection saves
    under its own name.

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
    parts = [(label, colour, cumulative_curves(part, mode))
             for label, colour, part in odor_parts(data, separate_odors, show_combined)]
    bounds = session_bounds(data)

    figures, totals, entries = {}, [], []
    for subjid in sorted(set().union(*(curves["subjid"].unique() for *_, curves in parts))):
        spans = bounds[bounds["subjid"] == subjid].sort_values("session_idx")
        fig, (ax,) = figure(width=_WIDTH, height=_HEIGHT)
        for label, colour, curves in parts:
            animal = curves[curves["subjid"] == subjid]
            final = {"subjid": int(subjid), **({} if label is None else {"odor": label})}
            for name, style in CURVE_STYLES.items():
                part = animal[animal["series"] == name]
                if part.empty:
                    continue
                if label is not None:
                    style = {**_ODOR_CURVE_STYLES[name], "color": colour,
                             "label": f"{label} {style['label']}"}
                scale = float(part["value"].abs().max()) or 1.0
                final[CURVE_STYLES[name]["label"]] = part["value"].iloc[-1]
                # From zero at the clock's start, so hours before the first event read as
                # a flat count rather than as missing data.
                ax.plot(np.r_[0.0, part["time"]], np.r_[0.0, part["value"] / scale],
                        drawstyle="steps-post", **style)
            totals.append(final)
        for start in spans["start"].to_numpy()[1:]:
            ax.axvline(start, **BOUNDARY)
        ax.axhline(0, **ZERO)
        style_axis(ax, ylabel="Normalized count", ylim=None)
        ax.set_xlim(0, float(np.nanmax(spans["end"])) if len(spans) else None)
        ax.set_xlabel("Task time (h)")
        ax.legend(frameon=False, fontsize=annotation_size(), loc="upper left",
                  ncols=len(parts))
        entries += finish_figure(fig, legend, show)
        _save(fig, f"discrimination_learning_cumulative_panels"
                   f"{odors_tag(separate_odors, show_combined)}{show_suffix(show)}"
                   f"_sub-{subjid:03d}", data, save, subjid)
        figures[int(subjid)] = fig
    legend_figure(entries, fontsize=annotation_size())
    print(f"Final counts ({mode}):")
    print(pd.DataFrame(totals).to_string(index=False, float_format=lambda v: f"{v:g}"))
    return figures


def plot_excess_by_index(subjids=None, dates=None, *, data=None, mode="completed",
                         separate_odors=False, show_combined=False, legend=None, show=None,
                         save=False, **selectors):
    """Excess correct over the mode's own row index, one figure per animal.

        plot_excess_by_index(data=ab)
        plot_excess_by_index(data=ab, mode="attempts")
        plot_excess_by_index(data=ab, separate_odors=True, show_combined=True)

    ``cumsum(y - 0.5)`` against the number of choices so far: completed trials, or every
    choice attempt. Each choice is one step whatever time it took, so the slope is
    accuracy minus 0.5 and engagement is off the axis: flat is chance, a slope of 0.5
    is all correct, and a bend is a change in accuracy. Unscaled, in choices. Dotted
    verticals are session boundaries.

    ``separate_odors`` sums each odour over its own choices on the same axis, flat across
    the other's (`cumulative.excess_by_odor`), in its port colour: a bend is a change in
    that odour, and the slope is halved by the other odour's share of the choices.
    ``show_combined`` adds the pooled line in grey, which is the sum of the two. ``legend``
    and ``show`` follow the plotter convention; ``show`` numbers the odours, then both.
    """
    data = require_data(subjids, dates, selectors, data)
    if show_combined and not separate_odors:
        raise ValueError("show_combined draws both odours beside each one alone: pass "
                         "separate_odors=True")
    excess = excess_by_index(data, mode)
    by_odor = excess_by_odor(data, mode) if separate_odors else None
    unit = "trials" if mode == "completed" else "choice attempts"

    figures, entries = {}, []
    for subjid in sorted(excess["subjid"].unique()):
        animal = excess[excess["subjid"] == subjid]
        spans = index_bounds(animal)
        fig, (ax,) = figure(width=_WIDTH, height=_HEIGHT)
        if by_odor is None:
            ax.plot(np.r_[0, animal["k"] + 1], np.r_[0.0, animal["excess"]],
                    drawstyle="steps-post", **{**CURVE_STYLES["excess"], "label": None})
        else:
            for (port, name), rows in by_odor[by_odor["subjid"] == subjid].groupby(
                    ["port", "odor"]):
                ax.plot(np.r_[0, rows["k"] + 1], np.r_[0.0, rows["excess"]],
                        drawstyle="steps-post", color=PORT_COLOURS[port], linewidth=2.4,
                        zorder=4, label=name)
            if show_combined:
                names = by_odor.drop_duplicates("port").sort_values("port")["odor"]
                ax.plot(np.r_[0, animal["k"] + 1], np.r_[0.0, animal["excess"]],
                        drawstyle="steps-post", color=COMBINED, linewidth=1.6, zorder=3,
                        label="+".join(names))
            ax.legend(frameon=False, fontsize=annotation_size(), loc="upper left")
        for start in spans["start"].to_numpy()[1:]:
            ax.axvline(start, **BOUNDARY)
        ax.axhline(0, **ZERO)
        style_axis(ax, ylabel="Excess correct", ylim=None)
        ax.set_xlim(0, len(animal))
        ax.set_xlabel(unit.capitalize())
        entries += finish_figure(fig, legend, show)
        _save(fig, f"discrimination_learning_excess_by_index"
                   f"{odors_tag(separate_odors, show_combined)}{show_suffix(show)}_{mode}"
                   f"_sub-{subjid:03d}", data, save, subjid)
        figures[int(subjid)] = fig
    legend_figure(entries, fontsize=annotation_size())
    return figures


def _mean_curve(frame, subjects, clock, grid):
    """Mean and SEM over animals of their rolling accuracy, interpolated onto ``grid``.

    An animal counts at a grid point only inside its own data; the SEM uses the animals
    counted there.
    """
    curves = []
    for subjid in subjects:
        rows = frame[(frame["subjid"] == subjid)].dropna(subset=["hours", "accuracy"])
        x = clock(subjid, rows["hours"].to_numpy())
        curves.append(np.interp(grid, x, rows["accuracy"].to_numpy(), left=np.nan,
                                right=np.nan))
    stack = np.vstack(curves)
    n = np.isfinite(stack).sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.nanmean(stack, axis=0)
        sem = np.nanstd(stack, axis=0, ddof=1) / np.sqrt(n)
    return mean, sem


def _accuracy_figure(data, mode, window, clock, xlabel, xlim, show_mean=False,
                     separate_odors=False, show_combined=False):
    """The cohort accuracy figure, on the x axis ``clock(subjid, hours)`` maps task hours to.

    Session boundaries, drawn for a single animal, go through ``clock`` too; ``xlim`` of
    None runs from 0 to the end of the last session. ``show_mean`` greys the animals and
    draws their mean with its SEM. Per odour, the odour's colour replaces the animal's
    (and the grey). Returns the figure and the animals.
    """
    if mode not in ACCURACY_MODES:
        raise ValueError(f"mode must be one of {ACCURACY_MODES}, got {mode!r}")
    modes = MODES if mode == "both" else (mode,)
    parts = [(label, colour, {m: rolling_accuracy(part, m, window) for m in modes})
             for label, colour, part in odor_parts(data, separate_odors, show_combined)]
    subjects = sorted(set().union(*(frame["subjid"].unique() for *_, frames in parts
                                    for frame in frames.values())))
    bounds = session_bounds(data)
    bounds = bounds[bounds["subjid"].isin(subjects)]

    fig, (ax,) = figure(width=_WIDTH, height=_HEIGHT)
    for label, colour, frames in parts:
        for index, subjid in enumerate(subjects):
            for m in modes:
                rows = frames[m][frames[m]["subjid"] == subjid]
                first = index == 0 and m == modes[0]
                if label is not None:
                    look = dict(color=colour, linewidth=1.2 if show_mean else 1.8,
                                alpha=0.45 if show_mean else 1.0,
                                label=(f"{label}, animals" if show_mean else label)
                                if first else None)
                elif show_mean:
                    look = {**_INDIVIDUAL, "label": "animals" if first else None}
                else:
                    look = dict(color=subject_colour(subjid), linewidth=1.8,
                                label=f"sub-{subjid:03d}" if m == modes[0] else None)
                ax.plot(clock(subjid, rows["hours"].to_numpy()), rows["accuracy"],
                        linestyle=_MODE_LINES[m][0], **look)
    ax.axhline(0.5, **ZERO)
    ax.set_ylim(0, 1)
    ax.set_xlim(*(xlim or (0, float(bounds["end"].max()))))
    if show_mean:
        grid = np.linspace(*ax.get_xlim(), _MEAN_POINTS)
        for label, colour, frames in parts:
            for m in modes:
                mean, sem = _mean_curve(frames[m], subjects, clock, grid)
                ax.fill_between(grid, mean - sem, mean + sem, color=colour or REFERENCE,
                                alpha=0.3 if label is None else BAND_ALPHA * 2, linewidth=0,
                                zorder=4)
                text = f"mean ± SEM, n = {len(subjects)}"
                ax.plot(grid, mean, linestyle=_MODE_LINES[m][0],
                        **{**_MEAN, **({} if label is None else {"color": colour})},
                        label=(text if label is None else f"{label} {text}")
                        if m == modes[0] else None)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_ylabel("Accuracy")
    ax.set_xlabel(xlabel)

    handles, labels = ax.get_legend_handles_labels()
    if len(modes) > 1:
        for m in modes:
            handles.append(Line2D([], [], color=REFERENCE, linewidth=1.8,
                                  linestyle=_MODE_LINES[m][0]))
            labels.append(_MODE_LINES[m][1])
    ax.legend(handles, labels, frameon=False, fontsize=annotation_size(), loc="lower right",
              ncols=2 if len(handles) > 4 else 1)

    if len(subjects) == 1:
        subjid = subjects[0]
        spans = bounds.sort_values("session_idx")
        spans = spans.assign(start=clock(subjid, spans["start"].to_numpy()),
                             end=clock(subjid, spans["end"].to_numpy()))
        for start in spans["start"].to_numpy()[1:]:
            ax.axvline(start, **BOUNDARY)
    return fig, subjects


def _save_name(stem, mode, subjects):
    """``{stem}_{mode}``, with the animal when the figure holds one."""
    single = f"_sub-{subjects[0]:03d}" if len(subjects) == 1 else ""
    return f"discrimination_learning_{stem}_{mode}{single}"


def plot_cohort_accuracy(subjids=None, dates=None, *, data=None, mode="completed",
                         window=WINDOW, separate_odors=False, show_combined=False, save=False,
                         **selectors):
    """Accuracy over task time, every animal of the selection in one figure.

        plot_cohort_accuracy(data=ab)
        plot_cohort_accuracy(data=ab, mode="both", window=30)
        plot_cohort_accuracy({62: ab_cohort[62]}, separate_odors=True, show_combined=True)

    Each line is the share correct over ``window`` consecutive choices
    (`cumulative.rolling_accuracy`), drawn at each choice's task time, in the animal's own
    colour. ``mode`` is ``"completed"``, ``"attempts"`` or ``"both"`` (completed solid,
    attempts dashed). Session boundaries are drawn when the selection is one animal.

    ``separate_odors`` draws each odour's accuracy, the window counting that odour's
    choices, in its port colour; the animals are then told apart only by selecting one.
    ``show_combined`` adds the pooled accuracy in grey. Returns the Figure.
    """
    data = require_data(subjids, dates, selectors, data)
    fig, subjects = _accuracy_figure(data, mode, window, lambda subjid, hours: hours,
                                     "Task time (h)", None, separate_odors=separate_odors,
                                     show_combined=show_combined)
    if save:
        stem = "cohort_accuracy" + odors_tag(separate_odors, show_combined)
        save_figure(fig, _save_name(stem, mode, subjects), **save_scope(data["sessions"]))
    return fig


def plot_cohort_accuracy_normalized(subjids=None, dates=None, *, data=None,
                                    mode="completed", window=NORMALIZED_WINDOW,
                                    show_mean=False, separate_odors=False,
                                    show_combined=False, save=False, **selectors):
    """Accuracy over each animal's own span of rewarded task time, all animals aligned.

        plot_cohort_accuracy_normalized(data=ab)
        plot_cohort_accuracy_normalized(data=ab, mode="both", window=200)
        plot_cohort_accuracy_normalized(data=ab, show_mean=True)
        plot_cohort_accuracy_normalized(data=ab, show_mean=True, separate_odors=True)

    As `plot_cohort_accuracy`, with task time rescaled per animal: 0 at its first reward,
    1 at its last (`data.reward_bounds`), so every animal spans the same width. Choices
    before the first reward fall left of 0 and are not shown; the window still counts
    them. ``show_mean`` draws the animals as faint grey lines and their mean in black
    over its SEM band, each animal interpolated onto a common grid of normalized time.

    ``separate_odors`` / ``show_combined`` as in `plot_cohort_accuracy`; with ``show_mean``
    each odour's animals are faint and its mean thick in its colour. The time axis stays
    the animal's first-to-last reward over both odours. Returns the Figure.
    """
    data = require_data(subjids, dates, selectors, data)
    rewards = reward_bounds(data).set_index("subjid")

    def clock(subjid, hours):
        first, last = rewards.loc[subjid, ["first", "last"]]
        return (hours - first) / (last - first)

    fig, subjects = _accuracy_figure(data, mode, window, clock,
                                     "Normalized time in arena", (0, 1), show_mean,
                                     separate_odors=separate_odors,
                                     show_combined=show_combined)
    if save:
        stem = ("cohort_accuracy_normalized" + ("_mean" if show_mean else "")
                + odors_tag(separate_odors, show_combined))
        save_figure(fig, _save_name(stem, mode, subjects), **save_scope(data["sessions"]))
    return fig
