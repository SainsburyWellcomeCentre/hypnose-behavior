"""Figures for the session regression (``modelling.ab_learning.log_regression``).

`plot_gain_decomposition` takes the same subject/session selection as `load_ab_data` --
or the already-loaded frames as ``data=``, or the fitted models as ``fits=`` -- and draws
**one figure per animal**. Returns ``{subjid: Figure}``; ``save=True`` files each figure
under its own animal with `io.save.save_figure`.

Both panels are on the log-odds scale, where the decomposition is additive; the right
axis of the top panel carries the matching accuracies.
"""
from __future__ import annotations

import numpy as np
from scipy.special import logit

from hypnose_behavior.io.save import save_figure
from hypnose_behavior.modelling.ab_learning.log_regression import (
    MIN_TRIALS,
    fit_session_models,
    gain_decomposition,
    overnight_share,
    session_levels,
)
from hypnose_behavior.visualization.modelling.ab_learning._common import (
    BAND_ALPHA,
    REFERENCE,
    SERIES,
    figure,
    require_data,
    save_scope,
    session_ticks,
    style_axis,
    text_size,
    title,
)

__all__ = ["plot_gain_decomposition"]

# Slot 1 for the gain inside a session, slot 2 for the gain across the night after it.
_WITHIN = SERIES
_OVERNIGHT = "#eb6834"
_COMPONENT_STYLE = {"within": (_WITHIN, "within session"),
                    "overnight": (_OVERNIGHT, "overnight")}

# A session occupies this much of its slot on the x axis; the rest is the boundary that
# follows it, so a within-session segment and an overnight step never overlap.
_SESSION_SPAN = 0.72

# Accuracies labelled on the right of the top panel.
_PROBABILITY_TICKS = (0.1, 0.25, 0.5, 0.75, 0.9, 0.97)


def _save(fig, name, sessions, save, subjid=None):
    if not save:
        return
    save_figure(fig, name, **save_scope(sessions, subjid))


def _probability_axis(ax):
    """Label the right of a log-odds axis with the accuracies it corresponds to."""
    right = ax.twinx()
    right.set_ylim(ax.get_ylim())
    ticks = [p for p in _PROBABILITY_TICKS if ax.get_ylim()[0] < logit(p) < ax.get_ylim()[1]]
    right.set_yticks(logit(ticks))
    right.set_yticklabels([f"{p:g}" for p in ticks])
    right.spines[["top", "left"]].set_visible(False)
    right.set_ylabel("accuracy")
    right.tick_params(length=2)


def _plot_levels(ax, levels):
    """The fitted log-odds of M_c: a segment per session, a step across each boundary."""
    start_x = levels["session_idx"].to_numpy()
    end_x = start_x + _SESSION_SPAN
    start, end = levels["logit_start"].to_numpy(), levels["logit_end"].to_numpy()

    for x, lo, hi in ((start_x, levels["logit_start_lo"], levels["logit_start_hi"]),
                      (end_x, levels["logit_end_lo"], levels["logit_end_hi"])):
        ax.vlines(x, lo, hi, color=_WITHIN, linewidth=3.5, alpha=BAND_ALPHA)
    for x0, x1, y0, y1 in zip(start_x, end_x, start, end):
        ax.plot([x0, x1], [y0, y1], color=_WITHIN, linewidth=2, solid_capstyle="round")
    for x0, x1, y0, y1 in zip(end_x[:-1], start_x[1:], end[:-1], start[1:]):
        ax.plot([x0, x1], [y0, y1], color=_OVERNIGHT, linewidth=1.6, linestyle=(0, (2, 1.5)))
    ax.plot(np.r_[start_x, end_x], np.r_[start, end], linestyle="none", marker="o",
            markersize=4, color=_WITHIN, markeredgecolor="white", markeredgewidth=0.8)
    ax.axhline(0, color=REFERENCE, linestyle="--", linewidth=1)


def _plot_gains(ax, gains):
    """W_s and O_s with their 95% intervals, each above the session it belongs to."""
    for component, (color, label) in _COMPONENT_STYLE.items():
        part = gains[gains["component"] == component]
        if part.empty:
            continue
        offset = 0.0 if component == "within" else _SESSION_SPAN
        x = part["session_idx"].to_numpy() + offset
        err = np.vstack([part["value"] - part["lo"], part["hi"] - part["value"]])
        ax.errorbar(x, part["value"], yerr=err, fmt="o", color=color, markersize=6,
                    markeredgecolor="white", markeredgewidth=1, elinewidth=2, capsize=0,
                    label=label)
    ax.axhline(0, color=REFERENCE, linestyle="--", linewidth=1)


def _headline(share) -> str:
    """The animal's one-line summary of where its total change happened."""
    if share["mixed_signs"]:
        direction = "gained within sessions, given back overnight" if \
            share["within_total"] > 0 else "gained overnight, given back within sessions"
        return (f"total {share['total']:+.2f} logit ({direction}); "
                f"{share['share_of_movement']:.0%} of all movement overnight")
    return (f"total {share['total']:+.2f} logit, "
            f"{share['overnight_share']:.0%} of it overnight "
            f"[{share['overnight_share_lo']:.0%}, {share['overnight_share_hi']:.0%}]")


def plot_gain_decomposition(subjids=None, dates=None, *, data=None, fits=None,
                            mode="completed", min_trials=MIN_TRIALS, save=False,
                            **selectors):
    """Where each animal's accuracy changed: inside sessions, or between them.

        plot_gain_decomposition(data=ab)
        plot_gain_decomposition(fits=fit_session_models(ab, mode="attempts"))

    Top: M_c's fitted log-odds over training. A solid segment runs from a session's
    first choice to its last, a dashed step crosses to the next session's first, and the
    shaded bars are the 95% intervals on the two fitted edges. The dashed horizontal is
    chance.

    Bottom: the same change read as gains -- W_s over each session, O_s across each
    boundary that follows one, with 95% intervals. The line above them is the animal's
    total change and the share of it that happened between sessions.

    A session too short to fit (under ``min_trials``) leaves a gap on the x axis, and
    the overnight step drawn across that gap spans more than one night.
    """
    if fits is None:
        data = require_data(subjids, dates, selectors, data)
        fits = fit_session_models(data, mode=mode, min_trials=min_trials)
    levels, gains = session_levels(fits), gain_decomposition(fits)
    shares = overnight_share(fits).set_index("subjid")

    figures = {}
    for subjid in sorted(fits):
        animal = levels[levels["subjid"] == subjid].sort_values("session_idx")
        fig, (top, bottom) = figure(n_rows=2, height=3.0)
        top.sharex(bottom)
        _plot_levels(top, animal)
        _plot_gains(bottom, gains[gains["subjid"] == subjid])

        style_axis(top, ylabel="fitted log-odds correct", ylim=None)
        style_axis(bottom, ylabel="gain (log-odds)", ylim=None)
        top.tick_params(labelbottom=False)
        _probability_axis(top)
        session_ticks(bottom, animal)
        bottom.set_xlim(animal["session_idx"].min() - 0.4,
                        animal["session_idx"].max() + _SESSION_SPAN + 0.4)
        bottom.set_title(_headline(shares.loc[subjid]), fontsize=text_size() * 0.75,
                         loc="left", color=REFERENCE)
        fig.legend(*bottom.get_legend_handles_labels(), loc="outside lower center", ncols=2,
                   frameon=False, fontsize=text_size() * 0.7)
        title(fig, f"within-session and overnight gain | {fits[subjid]['mode']}", subjid)
        _save(fig, f"ab_learning_gain_decomposition_sub-{subjid:03d}",
              fits[subjid]["sessions"], save, subjid)
        figures[subjid] = fig
    return figures
