"""Figures for the state-space model (``modelling.ab_learning.state_space``).

`plot_state_space` takes `state_space.fit_state_space` results and draws **one figure
per animal**. Returns ``{subjid: Figure}``; ``save=True`` files each figure under its own
animal with `io.save.save_figure`, without the headline title the displayed figure keeps.
"""
from __future__ import annotations

import numpy as np
from matplotlib.transforms import offset_copy

from hypnose_behavior.io.save import (
    finish_figure,
    legend_figure,
    save_figure,
    show_suffix,
    tie,
)
from hypnose_behavior.modelling.ab_learning.diagnostics import wilson_interval
from hypnose_behavior.modelling.ab_learning.state_space import LEARNED
from hypnose_behavior.visualization.modelling.ab_learning._common import (
    BAND_ALPHA,
    BOUNDARY,
    REFERENCE,
    SECOND,
    SERIES,
    ZERO,
    annotation_size,
    figure,
    in_order,
    index_bounds,
    save_scope,
    style_axis,
    text_size,
)

__all__ = ["plot_state_space"]

# Legend order, which numbers ``show``; the boundary-σ fit, drawn only when given, last.
_OBSERVED_LABEL = "Excess correct"
_MODEL_LABEL = "state space, 95%"
_LEARNED_LABEL = "learning trial"
_CHANGE_LABEL = "changes"
_BOUNDARY_LABEL = "with boundary σ"
_ORDER = (_OBSERVED_LABEL, _MODEL_LABEL, _LEARNED_LABEL, _CHANGE_LABEL, _BOUNDARY_LABEL)

# The observed choices keep excess correct's orange, drawn thick; the posterior median runs
# over them in dark blue with its band in slot 1, the boundary-σ fit closely dotted grey.
_OBSERVED = dict(color=SECOND, linewidth=2.8, zorder=1)
_DATA = dict(color=SECOND, markersize=6, zorder=1)
_MODEL = dict(color="#1c5cab", linewidth=1.8, zorder=4)
_BAND = dict(color=SERIES, alpha=BAND_ALPHA, linewidth=0, zorder=2)
_SESSION_MEAN = dict(color="#1c5cab", linewidth=4, alpha=0.45, zorder=3)
_BOUNDARY_FIT = dict(color="#4f4e4a", linewidth=2.2, linestyle=(0, (1, 1.6)),
                     dash_capstyle="round", zorder=5)
_LEARNED = dict(color="black", linewidth=1.3, linestyle=(0, (4, 2)), zorder=6)
_CHANGE = dict(color="black", linestyle="none", zorder=7, clip_on=False)

# Rows per accuracy bin drawn behind the curve.
_BIN = 50

# A change's ▲ / ▼: its height as a share of the tick-label size, and its gap (points)
# from the line it marks.
_MARK_SIZE = 0.7
_MARK_GAP = 3.0

_WIDTH = 10.0
_HEIGHT = 3.0


def _save(fig, name, rows, save, subjid):
    if not save:
        return
    save_figure(fig, name, titles=False, **save_scope(rows, subjid))


def _binned(rows):
    """Accuracy per `_BIN` rows, with Wilson bounds, at each bin's middle row."""
    bins = rows.groupby(rows["k"] // _BIN).agg(k=("k", "mean"), n=("y", "size"),
                                                c=("y", "sum"))
    lo, hi = wilson_interval(bins["c"], bins["n"])
    return bins.assign(p=bins["c"] / bins["n"], lo=lo, hi=hi)


def _headline(fit, boundary_fit):
    """sub, change count, learning trial and σ, and the boundary σ when given."""
    s = fit["summary"]
    learned = ("never" if s["learning_trial"] is None
               else f"{s['learning_trial']} (ses {s['learning_ses']})")
    parts = [f"sub-{fit['subjid']:03d}", f"{s['n_changes']} change(s)",
             f"learning trial {learned}",
             f"σ {s['sigma']:.3f} [{s['sigma_lo']:.3f}, {s['sigma_hi']:.3f}]"]
    lines = ["  ·  ".join(parts)]
    if boundary_fit is not None:
        b = boundary_fit["summary"]
        lines.append(f"boundary σ: {b['n_changes']} change(s)  ·  σ_b/σ {b['sigma_ratio']:.1f}"
                     f"  ·  boundary share {b['boundary_share']:.2f} "
                     f"[{b['boundary_share_lo']:.2f}, {b['boundary_share_hi']:.2f}]")
    return "\n".join(lines)


def _draw_excess(ax, fit, boundary_fit):
    """Observed excess correct and the expected excess with its 95% band."""
    curve = fit["curve"]
    count = np.r_[0, curve["k"].to_numpy() + 1]
    ax.plot(count, np.r_[0.0, np.cumsum(curve["y"].to_numpy() - 0.5)],
            drawstyle="steps-post", label=_OBSERVED_LABEL, **_OBSERVED)
    tie(ax.fill_between(count, np.r_[0.0, curve["excess_lo"]],
                        np.r_[0.0, curve["excess_hi"]], **_BAND), _MODEL_LABEL)
    ax.plot(count, np.r_[0.0, curve["excess"]], label=_MODEL_LABEL, **_MODEL)
    if boundary_fit is not None:
        ax.plot(count, np.r_[0.0, boundary_fit["curve"]["excess"]], label=_BOUNDARY_LABEL,
                **_BOUNDARY_FIT)
    ax.axhline(0, **ZERO)


def _draw_accuracy(ax, fit, boundary_fit):
    """Accuracy per `_BIN` rows, the median p_k with its 95% band, and the session means
    the changes are counted on."""
    curve, sessions = fit["curve"], fit["sessions"]
    k = curve["k"].to_numpy(dtype=float)
    bins = _binned(fit["rows"])
    spread = np.clip([bins["p"] - bins["lo"], bins["hi"] - bins["p"]], 0, None)
    ax.errorbar(bins["k"], bins["p"], yerr=spread, fmt="o", elinewidth=1, capsize=0,
                label=_OBSERVED_LABEL, **_DATA)
    tie(ax.fill_between(k, curve["p_lo"], curve["p_hi"], **_BAND), _MODEL_LABEL)
    tie(ax.plot(k, curve["p"], **_MODEL)[0], _MODEL_LABEL)
    tie(ax.hlines(sessions["p"], sessions["start"], sessions["end"], **_SESSION_MEAN),
        _MODEL_LABEL)
    if boundary_fit is not None:
        tie(ax.plot(k, boundary_fit["curve"]["p"], **_BOUNDARY_FIT)[0], _BOUNDARY_LABEL)
    ax.axhline(0.5, **ZERO)


def _draw_certainty(ax, fit, boundary_fit):
    """P(x_k > 0), against the learning criterion."""
    curve = fit["curve"]
    k = curve["k"].to_numpy(dtype=float)
    tie(ax.plot(k, curve["p_above"], **_MODEL)[0], _MODEL_LABEL)
    if boundary_fit is not None:
        tie(ax.plot(k, boundary_fit["curve"]["p_above"], **_BOUNDARY_FIT)[0],
            _BOUNDARY_LABEL)
    ax.axhline(LEARNED, color=REFERENCE, linewidth=1.0, linestyle=(0, (3, 2)), zorder=0)


def _mark_changes(excess, accuracy, fit):
    """A ▲ (rise) or ▼ (fall) where each change is half done: above the observed excess
    correct and beside the median p_k, with its size in p."""
    curve, changes = fit["curve"], fit["changes"]
    size = text_size() * _MARK_SIZE
    observed = np.cumsum(curve["y"].to_numpy() - 0.5)
    k = curve["k"].to_numpy(dtype=float)
    labelled = False
    for change in changes.itertuples():
        rise = change.delta > 0
        marker = "^" if rise else "v"
        shift = (size / 2 + _MARK_GAP) * (1 if rise else -1)
        y_excess = np.interp(change.k_mid, k + 1, observed)
        y_p = np.interp(change.k_mid, k, curve["p"])
        for ax, y in ((excess, y_excess), (accuracy, y_p)):
            at = offset_copy(ax.transData, fig=ax.figure, y=shift, units="points")
            line, = ax.plot(change.k_mid, y, transform=at, marker=marker, markersize=size,
                            **_CHANGE)
            if ax is excess and not labelled:
                line.set_label(_CHANGE_LABEL)
                labelled = True
            else:
                tie(line, _CHANGE_LABEL)
        tie(accuracy.annotate(f"{change.delta:+.2f}", (change.k_mid, y_p),
                              xytext=(size / 2 + 3, shift), textcoords="offset points",
                              va="center", fontsize=annotation_size(), zorder=7),
            _CHANGE_LABEL)
    if not labelled:
        # No change: an empty series keeps the legend's numbering.
        excess.plot([], [], marker="^", markersize=size, label=_CHANGE_LABEL, **_CHANGE)


def _mark_learning(axes, learned):
    """A dashed vertical at the learning trial, on every panel; an empty series without."""
    top = axes[0]
    if learned is None:
        top.plot([], [], label=_LEARNED_LABEL, **_LEARNED)
        return
    for ax in axes:
        line = ax.axvline(learned, **_LEARNED)
        if ax is top:
            line.set_label(_LEARNED_LABEL)
        else:
            tie(line, _LEARNED_LABEL)


def plot_state_space(fits: dict, *, boundary: dict | None = None, legend=None, show=None,
                     save=False):
    """The state-space posterior over the data, one figure per animal.

        walk = fit_state_space(ab)
        plot_state_space(walk, boundary=fit_state_space(ab, boundary=True))

    - top: observed excess correct (orange) and the expected excess,
      ``cumsum(p_k - 0.5)`` over the draws, median (dark blue) with its 95% band. Where
      orange leaves the band the walk misses the data.
    - middle: accuracy per 50 rows (Wilson 95%), the median p_k with its 95% band, and
      as thick faded bars the median session means, which the changes are counted on.
    - bottom: P(x_k > 0), with the learning criterion dashed grey.
    - the dashed black vertical is the learning trial; ▲ / ▼ mark each counted change
      where its median is half done, with its size in p; dotted greys are session
      boundaries.

    ``boundary`` (the `fit_state_space` fits with ``boundary=True``, same mode) adds that
    fit's median, closely dotted grey, and its σ_b to the headline. The headline is the
    figure's title: shown, not saved.

    ``legend`` and ``show`` follow the plotter convention (`hypnose_helpers.viz.plotter`);
    ``show`` numbers: 1 observed, 2 the walk, 3 learning trial, 4 changes, 5 the
    boundary-σ fit.
    """
    figures, entries = {}, []
    for subjid, fit in sorted(fits.items()):
        boundary_fit = None if boundary is None else boundary.get(subjid)
        rows, mode = fit["rows"], fit["mode"]
        spans = index_bounds(rows)
        fig, axes = figure(3, width=_WIDTH, height=_HEIGHT)
        excess, accuracy, certainty = axes
        for ax in (accuracy, certainty):
            ax.sharex(excess)
        _draw_excess(excess, fit, boundary_fit)
        _draw_accuracy(accuracy, fit, boundary_fit)
        _draw_certainty(certainty, fit, boundary_fit)
        _mark_changes(excess, accuracy, fit)
        _mark_learning(axes, fit["learning_trial"])
        for ax in axes:
            for start in spans["start"].to_numpy()[1:]:
                ax.axvline(start, **BOUNDARY)
        style_axis(excess, ylabel="Excess correct", ylim=None)
        style_axis(accuracy, ylabel="Accuracy", ylim=(0, 1.02))
        style_axis(certainty, ylabel="P(x > 0)", ylim=(0, 1.02))
        excess.set_xlim(0, len(rows))
        unit = "trials" if mode == "completed" else "choice attempts"
        certainty.set_xlabel(unit.capitalize())
        excess.set_title(_headline(fit, boundary_fit), loc="left",
                         fontsize=annotation_size(), color=REFERENCE)
        fig.legend(*in_order(excess, _ORDER), loc="outside lower center", ncols=5,
                   frameon=False, fontsize=annotation_size())
        entries += finish_figure(fig, legend, show)
        _save(fig, f"ab_learning_state_space{show_suffix(show)}_{mode}_sub-{subjid:03d}",
              rows, save, subjid)
        figures[int(subjid)] = fig
    legend_figure(entries, fontsize=annotation_size())
    return figures
