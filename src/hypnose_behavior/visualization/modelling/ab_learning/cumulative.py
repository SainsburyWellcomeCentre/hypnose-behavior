"""The engagement, knowledge and reward figure (``modelling.ab_learning.cumulative``).

`plot_cumulative_panels` takes the same subject/session selection as `load_ab_data` -- or
the already-loaded frames as ``data=`` -- and draws **one figure per animal**, three
panels on one task-time axis. Returns ``{subjid: Figure}``; ``save=True`` files each
figure under its own animal with `io.save.save_figure`.
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

__all__ = ["plot_cumulative_panels"]

# Panel, series drawn in it (colour, label), and the y label.
_PANELS = (
    ((("attempts", SECOND, "choice attempts"), ("initiations", SERIES, "initiations")),
     "initiations"),
    ((("excess", SERIES, None),), "excess correct"),
    ((("rewards", SERIES, None),), "rewards"),
)
_BOUNDARY = dict(color=REFERENCE, linestyle=":", linewidth=1.2)

_WIDTH = 10.0
_PANEL_HEIGHT = 2.9


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

    Task time counts only the hours an odour-discrimination run was recording, so the
    nights and the gaps between runs take no width. Dotted verticals are session
    boundaries; the session numbers run along the top.

    - A: trials initiated. In ``"attempts"`` mode a second line adds the failed attempts
      that ended in a port visit, so the gap between the two is the failed choices.
    - B: excess correct over the mode's choices, ``cumsum(y - 0.5)``. Flat is chance;
      the slope per hour is choice rate times (accuracy - 0.5), so a bend here that A
      shares is engagement, and one A lacks is accuracy.
    - C: rewards, the product of the two.
    """
    data = require_data(subjids, dates, selectors, data)
    curves = cumulative_curves(data, mode)
    bounds = session_bounds(data)

    figures = {}
    for subjid in sorted(curves["subjid"].unique()):
        animal = curves[curves["subjid"] == subjid]
        spans = bounds[bounds["subjid"] == subjid].sort_values("session_idx")
        fig, axes = figure(n_rows=3, width=_WIDTH, height=_PANEL_HEIGHT)
        for ax, (series, ylabel) in zip(axes, _PANELS):
            for name, color, label in series:
                part = animal[animal["series"] == name]
                if part.empty:
                    continue
                # From zero at the clock's start, so hours before the first event read as
                # a flat count rather than as missing data.
                ax.plot(np.r_[0.0, part["time"]], np.r_[0.0, part["value"]], color=color,
                        linewidth=2, drawstyle="steps-post", label=label)
            for start in spans["start"].to_numpy()[1:]:
                ax.axvline(start, **_BOUNDARY)
            style_axis(ax, ylabel=ylabel, ylim=None)
            if ax is not axes[0]:
                ax.sharex(axes[0])
            if ax is not axes[-1]:
                ax.tick_params(labelbottom=False)
        axes[1].axhline(0, color=REFERENCE, linestyle="--", linewidth=1)
        axes[0].set_xlim(0, float(np.nanmax(spans["end"])) if len(spans) else None)
        _session_axis(axes[0], spans)
        axes[-1].set_xlabel("task time (h)")
        if mode == "attempts":
            axes[0].legend(frameon=False, fontsize=annotation_size(), loc="upper left")
        title(fig, f"initiations, excess correct, rewards | {mode}", subjid)
        _save(fig, f"ab_learning_cumulative_panels_sub-{subjid:03d}", data, save, subjid)
        figures[int(subjid)] = fig
    return figures
