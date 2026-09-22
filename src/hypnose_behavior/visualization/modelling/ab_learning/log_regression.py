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
    model_comparison,
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
# Each group of gains: which rows, the colour, whether the marker is hollow, the label.
# Hollow marks a gain outside the paired within/overnight total -- the last session's
# W_s, which has no boundary after it, and a boundary holding a dropped session.
_GAIN_GROUPS = (
    ("within", True, _WITHIN, False, "within session"),
    ("within", False, _WITHIN, True, "within, unpaired"),
    ("overnight", True, _OVERNIGHT, False, "overnight"),
    ("across_gap", True, _OVERNIGHT, True, "spans a dropped session"),
)

# A session occupies this much of its slot on the x axis; the rest is the boundary that
# follows it, so a within-session segment and an overnight step never overlap.
_SESSION_SPAN = 0.72

# Accuracies labelled on the right of the top panel.
_PROBABILITY_TICKS = (0.1, 0.25, 0.5, 0.75, 0.9, 0.97)

# Two stacked panels of per-session detail need the width; a presentation style puts 24pt
# on the axis labels, which a 6.4in figure cannot hold beside this many sessions.
_WIDTH = 10.0
_PANEL_HEIGHT = 3.4

# Annotations subordinate to the axis labels: the headline and the legend. Taken as a
# fraction of the tick size so they follow the active style, with a floor, since the
# headline carries the animal's result and a style with small ticks would shrink it out
# of reading size.
_SMALL = 0.55
_MIN_ANNOTATION = 8.0


def _annotation_size() -> float:
    """Point size for the headline and the legend."""
    return max(text_size() * _SMALL, _MIN_ANNOTATION)


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
    right.set_ylabel("accuracy", fontsize=text_size())
    right.tick_params(length=2, labelsize=text_size() * 0.8)


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
            markersize=5.5, color=_WITHIN, markeredgecolor="white", markeredgewidth=1)
    ax.axhline(0, color=REFERENCE, linestyle="--", linewidth=1)


def _plot_running_totals(ax, gains):
    """The running sum of each component, drawn through the points it sums.

    Sums only the rows the totals are built from, so the within line stops one session
    short of the last W_s, which has no boundary to pair with, and neither line takes up
    a boundary that spans a dropped session. Where one exists the two lines therefore end
    short of the total by that boundary's gain.
    """
    for component, offset, color in (("within", 0.0, _WITHIN),
                                     ("overnight", _SESSION_SPAN, _OVERNIGHT)):
        part = gains[(gains["component"] == component) & gains["in_total"]]
        part = part.sort_values("session_idx")
        if part.empty:
            continue
        ax.plot(part["session_idx"].to_numpy() + offset, part["value"].cumsum(),
                color=color, linewidth=1.3, alpha=0.8, zorder=1,
                label=f"{component}, running total")


def _plot_gains(ax, gains):
    """W_s and O_s with their 95% intervals, each above the session it belongs to.

    A gain outside the paired total -- the last session's W_s, and a boundary with a
    dropped session inside it -- is drawn hollow.
    """
    for component, paired, color, hollow, label in _GAIN_GROUPS:
        part = gains[(gains["component"] == component) & (gains["in_total"] == paired)]
        if part.empty:
            continue
        x = part["session_idx"].to_numpy() + (0.0 if component == "within" else _SESSION_SPAN)
        err = np.vstack([part["value"] - part["lo"], part["hi"] - part["value"]])
        ax.errorbar(x, part["value"], yerr=err, fmt="o", color=color, markersize=8,
                    markerfacecolor="white" if hollow else color,
                    markeredgecolor=color if hollow else "white", markeredgewidth=1.6,
                    elinewidth=2.5, capsize=0, label=label)
    ax.axhline(0, color=REFERENCE, linestyle="--", linewidth=1)


def _headline(share, test) -> str:
    """The animal's result in two lines: the decomposition, then what qualifies it.

    The two sums in log-odds always; the share of the total only when both point the same
    way, since a ratio of components with opposite signs is not a percentage. Plain text
    throughout -- mathtext would fall back to a font family with no bold cut under a
    style that asks for bold, and warn once per fallback font.
    """
    line = (f"within {share['within_total']:+.2f}, "
            f"overnight {share['overnight_total']:+.2f}, "
            f"total {share['total']:+.2f} logit")
    if not share["mixed_signs"]:
        line += (f" ({share['overnight_share']:.0%} overnight, "
                 f"{share['overnight_share_lo']:.0%} to {share['overnight_share_hi']:.0%})")
    qualifiers = [f"M_a vs M_c p = {test['p']:.3g}",
                  f"last session {share['final_within']:+.2f}, unpaired"]
    if share["gap_total"]:
        qualifiers.append(f"{share['gap_total']:+.2f} across a dropped session")
    return f"{line}\n{'  |  '.join(qualifiers)}"


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
    boundary that follows one, with 95% intervals, and a thin running total through each
    component, which says whether its sum built up steadily or came from a session or
    two. The text above the panels is the animal's total change, the share of it that
    happened between sessions, and the M_a against M_c test.

    A session too short to fit (under ``min_trials``) leaves a gap on the x axis, and
    the overnight step drawn across that gap spans more than one night.
    """
    if fits is None:
        data = require_data(subjids, dates, selectors, data)
        fits = fit_session_models(data, mode=mode, min_trials=min_trials)
    levels, gains = session_levels(fits), gain_decomposition(fits)
    shares = overnight_share(fits).set_index("subjid")
    tests = model_comparison(fits)
    tests = tests[tests["comparison"] == "M_a vs M_c"].set_index("subjid")

    figures = {}
    for subjid in sorted(fits):
        animal = levels[levels["subjid"] == subjid].sort_values("session_idx")
        fig, (top, bottom) = figure(n_rows=2, width=_WIDTH, height=_PANEL_HEIGHT)
        top.sharex(bottom)
        _plot_levels(top, animal)
        _plot_running_totals(bottom, gains[gains["subjid"] == subjid])
        _plot_gains(bottom, gains[gains["subjid"] == subjid])

        style_axis(top, ylabel="log-odds", ylim=None)
        style_axis(bottom, ylabel="gain", ylim=None)
        top.tick_params(labelbottom=False)
        _probability_axis(top)
        session_ticks(bottom, animal)
        bottom.set_xlim(animal["session_idx"].min() - 0.4,
                        animal["session_idx"].max() + _SESSION_SPAN + 0.4)
        # The headline titles the top axes, where constrained_layout reserves room for it
        # under the figure title; on the lower axes it would wedge between the two panels.
        top.set_title(_headline(shares.loc[subjid], tests.loc[subjid]), loc="left",
                      fontsize=_annotation_size(), color=REFERENCE)
        fig.legend(*bottom.get_legend_handles_labels(), loc="outside lower center", ncols=3,
                   frameon=False, fontsize=_annotation_size())
        title(fig, f"within-session and overnight gain | {fits[subjid]['mode']}", subjid)
        _save(fig, f"ab_learning_gain_decomposition_sub-{subjid:03d}",
              fits[subjid]["sessions"], save, subjid)
        figures[subjid] = fig
    return figures
