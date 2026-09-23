"""The engagement, knowledge and reward figure (``modelling.ab_learning.cumulative``).

`plot_cumulative_panels` takes the same subject/session selection as `load_ab_data` -- or
the already-loaded frames as ``data=`` -- and draws **one figure per animal**, every
running count on one task-time axis. Returns ``{subjid: Figure}``; ``save=True`` files
each figure under its own animal with `io.save.save_figure`.
"""
from __future__ import annotations

import math

import numpy as np

from hypnose_behavior.io.save import save_figure
from hypnose_behavior.modelling.ab_learning.cumulative import cumulative_curves
from hypnose_behavior.modelling.ab_learning.data import session_bounds
from hypnose_behavior.visualization.modelling.ab_learning._common import (
    MAX_SESSION_TICKS,
    REFERENCE,
    SECOND,
    SERIES,
    annotation_size,
    figure,
    require_data,
    save_scope,
    style_axis,
    title,
)

__all__ = ["CURVE_STYLES", "plot_cumulative_panels"]

# One look per running count, the same in every figure. Rewards sit behind the other two:
# they are their product, so they are context rather than the comparison.
CURVE_STYLES = {
    "initiations": dict(color=SERIES, linewidth=2.2, zorder=3, label="initiations"),
    "attempts": dict(color=SERIES, linewidth=1.6, linestyle=(0, (4, 2)), zorder=2,
                     label="choice attempts"),
    "excess": dict(color=SECOND, linewidth=2.2, zorder=4, label="excess correct"),
    "rewards": dict(color="#1baf7a", linewidth=1.6, alpha=0.45, zorder=1, label="rewards"),
}
_BOUNDARY = dict(color=REFERENCE, linestyle=":", linewidth=1.2, zorder=0)

_WIDTH = 10.0
_HEIGHT = 5.0


def _save(fig, name, data, save, subjid):
    if not save:
        return
    save_figure(fig, name, **save_scope(data["sessions"], subjid))


def _session_axis(ax, bounds):
    """Session numbers along the top, at each session's middle, thinned to fit."""
    bounds = bounds.iloc[::math.ceil(len(bounds) / MAX_SESSION_TICKS)]
    top = ax.secondary_xaxis("top")
    top.set_xticks((bounds["start"] + bounds["end"]).to_numpy() / 2)
    top.set_xticklabels(bounds["ses"].astype(str))
    top.tick_params(length=0, labelsize=annotation_size())
    top.set_xlabel("session", fontsize=annotation_size())


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
            ax.axvline(start, **_BOUNDARY)
        ax.axhline(0, color=REFERENCE, linestyle="--", linewidth=1, zorder=0)
        style_axis(ax, ylabel="scaled count", ylim=None)
        ax.set_xlim(0, float(np.nanmax(spans["end"])) if len(spans) else None)
        ax.set_xlabel("task time (h)")
        _session_axis(ax, spans)
        ax.legend(frameon=False, fontsize=annotation_size(), loc="upper left")
        title(fig, f"initiations, excess correct, rewards | {mode}", subjid)
        _save(fig, f"ab_learning_cumulative_panels_sub-{subjid:03d}", data, save, subjid)
        figures[int(subjid)] = fig
    return figures
