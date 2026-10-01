"""Figures for the raw-accuracy gains (``modelling.ab_learning.session_gain``).

Each ``plot_*`` takes the same subject/session selection as `load_ab_data` -- or the
already-loaded frames as ``data=``, or the fitted models of
`log_regression.fit_session_models` as ``fits=`` -- and draws **one figure per animal**,
over the rows the regression was fitted on. Returns ``{subjid: Figure}``; ``save=True``
files each figure under its own animal with `io.save.save_figure`.

- `plot_edge_gains`        -- first and last N choices of each session, and the gains
  between them.
- `plot_position_profile`  -- accuracy by position in the session, pooled over sessions,
  against what M_a and M_c predict for the same bins.
- `plot_session_profiles`  -- accuracy by fraction of the session, one panel per session,
  against that session's M_c line.
"""
from __future__ import annotations

import math

import matplotlib.pyplot as plt
import numpy as np

from hypnose_behavior.io.save import save_figure
from hypnose_behavior.modelling.ab_learning.log_regression import (
    MIN_TRIALS,
    fit_session_models,
    fitted_curves,
    gain_decomposition,
)
from hypnose_behavior.modelling.ab_learning.session_gain import (
    N_WINDOW,
    edge_gains,
    edge_windows,
    gain_summary,
    position_profile,
    session_profile,
)
from hypnose_behavior.visualization.modelling.ab_learning._common import (
    BAND_ALPHA,
    REFERENCE,
    SECOND,
    SERIES,
    ZERO,
    SESSION_SPAN,
    annotation_size,
    figure,
    line_with_band,
    require_data,
    save_scope,
    session_ticks,
    style_axis,
    text_size,
)

__all__ = ["plot_edge_gains", "plot_position_profile", "plot_session_profiles"]

# Slot 1 inside a session, slot 2 across the night after it.
_WITHIN = SERIES
_ACROSS = SECOND
# Each group of gains: the component, the colour, whether the marker is hollow, the label.
# Hollow marks a boundary with an unfitted session inside it.
_GAIN_GROUPS = (
    ("within", _WITHIN, False, "within session"),
    ("across", _ACROSS, False, "across sessions"),
    ("across_gap", _ACROSS, True, "spans an unfitted session"),
)

# Same frame as the regression figure: two stacked panels of per-session detail.
_WIDTH = 10.0
_PANEL_HEIGHT = 3.4

# Session grid: panels per row, and each panel's size in inches.
_GRID_COLUMNS = 5
_CELL_WIDTH = 2.9
_CELL_HEIGHT = 2.4

# A choice-number bin reached by fewer sessions than this holds too few to pool.
_MIN_SESSIONS = 3


def _unit(mode: str) -> str:
    """What one row counts: a completed trial, or a choice attempt."""
    return "trial" if mode == "completed" else "choice"


def _fits(subjids, dates, data, fits, mode, min_trials, selectors) -> dict:
    """The given fits, or `fit_session_models` over the loaded selection."""
    if fits is None:
        data = require_data(subjids, dates, selectors, data)
        fits = fit_session_models(data, mode=mode, min_trials=min_trials)
    return fits


def _save(fig, name, fit, save):
    if not save:
        return
    save_figure(fig, name, **save_scope(fit["sessions"], fit["subjid"]))


def _plot_windows(ax, windows):
    """First- and last-window accuracy per session: a segment inside, a step across.

    A session whose two windows overlap is joined by a dotted segment rather than a solid
    one, since its within gain is left undefined.
    """
    start_x = windows["session_idx"].to_numpy()
    end_x = start_x + SESSION_SPAN
    first, last = windows["first"].to_numpy(), windows["last"].to_numpy()

    for x, lo, hi in ((start_x, windows["first_lo"], windows["first_hi"]),
                      (end_x, windows["last_lo"], windows["last_hi"])):
        ax.vlines(x, lo, hi, color=_WITHIN, linewidth=3.5, alpha=BAND_ALPHA)
    for x0, x1, y0, y1, disjoint in zip(start_x, end_x, first, last, windows["disjoint"]):
        ax.plot([x0, x1], [y0, y1], color=_WITHIN, linewidth=2, solid_capstyle="round",
                linestyle="-" if disjoint else ":")
    for x0, x1, y0, y1 in zip(end_x[:-1], start_x[1:], last[:-1], first[1:]):
        ax.plot([x0, x1], [y0, y1], color=_ACROSS, linewidth=1.6, linestyle=(0, (2, 1.5)))
    ax.plot(np.r_[start_x, end_x], np.r_[first, last], linestyle="none", marker="o",
            markersize=5.5, color=_WITHIN, markeredgecolor="white", markeredgewidth=1)
    ax.axhline(0.5, **ZERO)


def _plot_gains(ax, gains):
    """Within and across gains with their 95% intervals, each above its session."""
    for component, color, hollow, label in _GAIN_GROUPS:
        part = gains[(gains["component"] == component) & gains["value"].notna()]
        if part.empty:
            continue
        x = part["session_idx"].to_numpy() + (0.0 if component == "within" else SESSION_SPAN)
        err = np.vstack([part["value"] - part["lo"], part["hi"] - part["value"]])
        ax.errorbar(x, part["value"], yerr=err, fmt="o", color=color, markersize=8,
                    markerfacecolor="white" if hollow else color,
                    markeredgecolor=color if hollow else "white", markeredgewidth=1.6,
                    elinewidth=2.5, capsize=0, label=label)
    ax.axhline(0, **ZERO)


def _plot_running_totals(ax, gains):
    """The running sum of each component, drawn through the points it sums.

    A within gain is empty where a session's two windows overlap, and a boundary spanning
    an unfitted session is left out, so the two lines need not add up to the change from
    the first window to the last.
    """
    for component, offset, color in (("within", 0.0, _WITHIN),
                                     ("across", SESSION_SPAN, _ACROSS)):
        part = gains[(gains["component"] == component) & gains["value"].notna()]
        part = part.sort_values("session_idx")
        if part.empty:
            continue
        ax.plot(part["session_idx"].to_numpy() + offset, part["value"].cumsum(),
                color=color, linewidth=1.3, alpha=0.8, zorder=1,
                label=f"{component}, running total")


def _headline(summary, n: int, unit: str) -> str:
    """The animal's mean gain per component, with its 95% interval."""
    parts = []
    for component, name in (("within", "within"), ("across", "across")):
        if component not in summary.index:
            continue
        row = summary.loc[component]
        parts.append(f"{name} {row['mean']:+.3f} ({row['lo']:+.3f} to {row['hi']:+.3f}, "
                     f"{int(row['m'])})")
    return f"mean gain, first/last {n} {unit}s: " + "  |  ".join(parts)


def plot_edge_gains(subjids=None, dates=None, *, data=None, fits=None, mode="completed",
                    min_trials=MIN_TRIALS, n=N_WINDOW, save=False, **selectors):
    """Accuracy over the first and last ``n`` choices of each session, and the gains.

        plot_edge_gains(fits=fits)
        plot_edge_gains(data=ab, mode="attempts", n=30)

    Top: each session's first-window and last-window accuracy with Wilson 95% intervals,
    joined by a solid segment inside the session (dotted where the windows overlap) and
    a dashed step to the next session's first window. The grey horizontal is chance.

    Bottom: the within gain (last minus first) at each session and the across gain (next
    first minus last) at each boundary, with Newcombe 95% intervals, and a thin running
    total through each component. The text above the panels is each component's mean
    over the animal's sessions, with the number of gains averaged.
    """
    fits = _fits(subjids, dates, data, fits, mode, min_trials, selectors)
    windows, gains = edge_windows(fits, n), edge_gains(fits, n)
    summary = gain_summary(gains).set_index(["subjid", "component"])

    figures = {}
    for subjid in sorted(fits):
        fit = fits[subjid]
        unit = _unit(fit["mode"])
        animal = windows[windows["subjid"] == subjid].sort_values("session_idx")
        fig, (top, bottom) = figure(n_rows=2, width=_WIDTH, height=_PANEL_HEIGHT)
        top.sharex(bottom)
        _plot_windows(top, animal)
        _plot_running_totals(bottom, gains[gains["subjid"] == subjid])
        _plot_gains(bottom, gains[gains["subjid"] == subjid])

        style_axis(top, ylabel="Accuracy")
        style_axis(bottom, ylabel="Gain", ylim=None)
        top.tick_params(labelbottom=False)
        session_ticks(bottom, animal)
        bottom.set_xlim(animal["session_idx"].min() - 0.4,
                        animal["session_idx"].max() + SESSION_SPAN + 0.4)
        top.set_title(_headline(summary.loc[subjid], n, unit), loc="left",
                      fontsize=annotation_size(), color=REFERENCE)
        fig.legend(*bottom.get_legend_handles_labels(), loc="outside lower center", ncols=3,
                   frameon=False, fontsize=annotation_size())
        _save(fig, f"ab_learning_edge_gains_sub-{subjid:03d}", fit, save)
        figures[subjid] = fig
    return figures


def _plot_profile(ax, profile):
    """Observed accuracy with its Wilson band, and both models' mean prediction."""
    line_with_band(ax, profile, "center", "accuracy", "accuracy_lo", "accuracy_hi", SERIES,
                   "observed")
    ax.plot(profile["center"], profile["fitted_c"], color=_ACROSS, linewidth=2,
            label="M_c fit")
    ax.plot(profile["center"], profile["fitted_a"], color=REFERENCE, linewidth=2,
            linestyle=(0, (1, 1.5)), label="M_a fit")
    ax.axhline(0.5, **ZERO)


def plot_position_profile(subjids=None, dates=None, *, data=None, fits=None,
                          mode="completed", min_trials=MIN_TRIALS, bins=10, bin_width=10,
                          min_sessions=_MIN_SESSIONS, save=False, **selectors):
    """Accuracy by position in the session, pooled over each animal's fitted sessions.

        plot_position_profile(fits=fits)
        plot_position_profile(fits=fits, bins=5, bin_width=20)

    Top: by choice number from the session start, in bins of ``bin_width``; a bin
    reached by fewer than ``min_sessions`` sessions is left out. Bottom: by fraction of
    the session, ``bins`` bins of equal count per session.

    Each panel carries the observed accuracy with its Wilson 95% band and two model
    predictions averaged over the same rows: M_a (flat within every session), which on
    the top panel is what the changing mix of sessions alone produces, and M_c (a
    straight line in log-odds per session), which is what the per-session slopes add up
    to once pooled. The grey horizontal is chance.
    """
    fits = _fits(subjids, dates, data, fits, mode, min_trials, selectors)
    by_choice = position_profile(fits, "choices", bin_width=bin_width)
    by_fraction = position_profile(fits, "fraction", bins=bins)

    figures = {}
    for subjid in sorted(fits):
        fit = fits[subjid]
        unit = _unit(fit["mode"])
        choice = by_choice[(by_choice["subjid"] == subjid)
                           & (by_choice["n_sessions"] >= min_sessions)]
        fraction = by_fraction[by_fraction["subjid"] == subjid]
        fig, (top, bottom) = figure(n_rows=2, width=_WIDTH, height=_PANEL_HEIGHT)
        _plot_profile(top, choice)
        _plot_profile(bottom, fraction)
        for ax in (top, bottom):
            style_axis(ax, ylabel="Accuracy", ylim=None)
        top.set_xlabel(f"{unit.capitalize()} in session")
        bottom.set_xlabel("Fraction of session")
        bottom.set_xlim(0, 1)
        fig.legend(*bottom.get_legend_handles_labels(), loc="outside lower center", ncols=3,
                   frameon=False, fontsize=annotation_size())
        _save(fig, f"ab_learning_position_profile_sub-{subjid:03d}", fit, save)
        figures[subjid] = fig
    return figures


def plot_session_profiles(subjids=None, dates=None, *, data=None, fits=None,
                          mode="completed", min_trials=MIN_TRIALS, bins=5, save=False,
                          **selectors):
    """Accuracy by fraction of the session, one panel per fitted session.

        plot_session_profiles(fits=fits)

    Points are the accuracy over each of ``bins`` equal-count bins with Wilson 95%
    intervals; the line is the session's M_c fit, a straight line in log-odds and so a
    gentle curve here. Each panel is titled with the session and its W_s, M_c's gain
    across the session in log-odds. A profile that rises and then falls, or climbs only in
    its first bins, is one the line describes poorly. The grey horizontal is chance.
    """
    fits = _fits(subjids, dates, data, fits, mode, min_trials, selectors)
    profiles = session_profile(fits, bins=bins)
    curves = fitted_curves(fits)
    gains = gain_decomposition(fits)
    within = gains[gains["component"] == "within"].set_index(["subjid", "session_idx"])

    figures = {}
    for subjid in sorted(fits):
        fit = fits[subjid]
        sessions = fit["sessions"][fit["sessions"]["kept"]].sort_values("session_idx")
        columns = min(_GRID_COLUMNS, len(sessions))
        rows = math.ceil(len(sessions) / columns)
        fig, axes = plt.subplots(rows, columns, figsize=(_CELL_WIDTH * columns,
                                                         _CELL_HEIGHT * rows),
                                 sharex=True, sharey=True, squeeze=False,
                                 constrained_layout=True)
        for ax, session in zip(axes.flat, sessions.itertuples()):
            key = (subjid, session.session_idx)
            curve = curves[(curves["subjid"] == subjid)
                           & (curves["session_idx"] == session.session_idx)]
            profile = profiles[(profiles["subjid"] == subjid)
                               & (profiles["session_idx"] == session.session_idx)]
            ax.plot(curve["x"], curve["p"], color=_ACROSS, linewidth=2, label="M_c fit")
            err = np.vstack([profile["accuracy"] - profile["accuracy_lo"],
                             profile["accuracy_hi"] - profile["accuracy"]])
            ax.errorbar(profile["center"], profile["accuracy"], yerr=err, fmt="o",
                        color=SERIES, markersize=6, markeredgecolor="white",
                        markeredgewidth=1, elinewidth=2, capsize=0, label="observed")
            ax.axhline(0.5, **ZERO)
            style_axis(ax)
            ax.set_xlim(0, 1)
            ax.tick_params(labelsize=annotation_size())
            ax.set_title(f"ses {session.ses}  n {session.n}  W {within.loc[key, 'value']:+.1f}",
                         loc="left", fontsize=annotation_size())
        # An empty slot in the last row leaves the panel above it at the bottom of its
        # column, where the shared x axis would otherwise have hidden its tick labels.
        for index in range(len(sessions), rows * columns):
            axes.flat[index].set_visible(False)
            axes.flat[index - columns].tick_params(labelbottom=True)
        fig.supxlabel("Fraction of session", fontsize=text_size())
        fig.supylabel("Accuracy", fontsize=text_size())
        fig.legend(*axes.flat[0].get_legend_handles_labels(), loc="outside upper right",
                   ncols=2, frameon=False, fontsize=annotation_size())
        _save(fig, f"ab_learning_session_profiles_sub-{subjid:03d}", fit, save)
        figures[subjid] = fig
    return figures
