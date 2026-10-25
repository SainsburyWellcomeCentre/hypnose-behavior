"""Figures for the raw-accuracy gains (``modelling.discrimination_learning.session_gain``).

Each ``plot_*`` takes the same subject/session selection as `load_discrimination_data` -- or the
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

All three take ``legend`` and ``show`` (plotter convention, `hypnose_helpers.viz.plotter`).

Fits per odour (``separate_odors``, or `data.per_odor` fits): `plot_edge_gains` draws one
figure per animal and odour, ``{subjid: {odour: Figure}}``; the two profiles draw every
odour on the same axes in its reward port's colour.
"""
from __future__ import annotations

import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from hypnose_behavior.io.save import (
    finish_figure,
    legend_figure,
    save_figure,
    show_suffix,
    tie,
)
from hypnose_behavior.modelling.discrimination_learning.data import is_per_odor
from hypnose_behavior.modelling.discrimination_learning.log_regression import (
    MIN_TRIALS,
    fit_session_models,
    fitted_curves,
    gain_decomposition,
)
from hypnose_behavior.modelling.discrimination_learning.session_gain import (
    N_WINDOW,
    edge_gains,
    edge_windows,
    gain_summary,
    position_profile,
    session_profile,
)
from hypnose_behavior.visualization.modelling.discrimination_learning._common import (
    BAND_ALPHA,
    PORT_COLOURS,
    REFERENCE,
    SECOND,
    SERIES,
    ZERO,
    SESSION_SPAN,
    annotation_size,
    figure,
    in_order,
    line_with_band,
    odor_figures,
    odor_tag,
    odor_title,
    odors_tag,
    require_data,
    resolve_fits,
    save_scope,
    session_ticks,
    style_axis,
    text_size,
)

__all__ = ["plot_edge_gains", "plot_position_profile", "plot_session_profiles"]

# Slot 1 inside a session, slot 2 across the night after it.
_WITHIN = SERIES
_ACROSS = SECOND
_WITHIN_LABEL = "within session"
_ACROSS_LABEL = "across sessions"
_GAP_LABEL = "spans an unfitted session"
# Each group of gains: the component, the colour, whether the marker is hollow, the label.
# Hollow marks a boundary with an unfitted session inside it.
_GAIN_GROUPS = (
    ("within", _WITHIN, False, _WITHIN_LABEL),
    ("across", _ACROSS, False, _ACROSS_LABEL),
    ("across_gap", _ACROSS, True, _GAP_LABEL),
)
# Legend orders, which number ``show``; a series present only for some animals goes last.
_EDGE_ORDER = (_WITHIN_LABEL, _ACROSS_LABEL, "within, running total", "across, running total",
               _GAP_LABEL)
_PROFILE_ORDER = ("observed", "M_c fit", "M_a fit")
# Per odour every profile series takes the odour's colour; the fits part by line.
_ODOR_FITS = (("fitted_c", "M_c fit", (0, (4, 2))), ("fitted_a", "M_a fit", (0, (1, 1.5))))
# Per odour, each bin's points sit this far apart on the fraction axis.
_ODOR_DODGE = 0.03

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


def _fits(subjids, dates, data, fits, mode, min_trials, separate_odors, selectors) -> dict:
    """The given fits, or `fit_session_models` over the loaded selection, per odour with
    ``separate_odors``."""
    return resolve_fits(fits, separate_odors,
                        lambda: require_data(subjids, dates, selectors, data),
                        lambda d: fit_session_models(d, mode=mode, min_trials=min_trials))


def _parts(fits) -> list:
    """``[(odor, fits)]``: each odour of fits per odour, or ``(None, fits)``."""
    return list(fits.items()) if is_per_odor(fits) else [(None, fits)]


def _labels(odors, order) -> list:
    """The legend order of an overlay: each odour's series in ``order``, then the next's."""
    if odors == [None]:
        return list(order)
    return [f"{odor.name} {label}" for odor in odors for label in order]


def _save(fig, name, fit, save, titles=True):
    """``titles=False`` leaves the titles out of the file; the displayed figure keeps them."""
    if not save:
        return
    save_figure(fig, name, titles=titles, **save_scope(fit["sessions"], fit["subjid"]))


def _plot_windows(ax, windows):
    """First- and last-window accuracy per session: a segment inside, a step across.

    A session whose two windows overlap is joined by a dotted segment rather than a solid
    one, since its within gain is left undefined. Segments, windows and their intervals
    are tied to the within series and the steps to the across one, so ``show`` builds
    both panels up together.
    """
    start_x = windows["session_idx"].to_numpy()
    end_x = start_x + SESSION_SPAN
    first, last = windows["first"].to_numpy(), windows["last"].to_numpy()

    for x, lo, hi in ((start_x, windows["first_lo"], windows["first_hi"]),
                      (end_x, windows["last_lo"], windows["last_hi"])):
        tie(ax.vlines(x, lo, hi, color=_WITHIN, linewidth=3.5, alpha=BAND_ALPHA),
            _WITHIN_LABEL)
    for x0, x1, y0, y1, disjoint in zip(start_x, end_x, first, last, windows["disjoint"]):
        tie(ax.plot([x0, x1], [y0, y1], color=_WITHIN, linewidth=2, solid_capstyle="round",
                    linestyle="-" if disjoint else ":")[0], _WITHIN_LABEL)
    for x0, x1, y0, y1 in zip(end_x[:-1], start_x[1:], last[:-1], first[1:]):
        tie(ax.plot([x0, x1], [y0, y1], color=_ACROSS, linewidth=1.6,
                    linestyle=(0, (2, 1.5)))[0], _ACROSS_LABEL)
    tie(ax.plot(np.r_[start_x, end_x], np.r_[first, last], linestyle="none", marker="o",
                markersize=5.5, color=_WITHIN, markeredgecolor="white",
                markeredgewidth=1)[0], _WITHIN_LABEL)
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
                    min_trials=MIN_TRIALS, n=N_WINDOW, separate_odors=False, legend=None,
                    show=None, save=False, **selectors):
    """Accuracy over the first and last ``n`` choices of each session, and the gains.

        plot_edge_gains(fits=fits)
        plot_edge_gains(data=ab, mode="attempts", n=30)
        plot_edge_gains(data=ab, separate_odors=True)

    Top: each session's first-window and last-window accuracy with Wilson 95% intervals,
    joined by a solid segment inside the session (dotted where the windows overlap) and
    a dashed step to the next session's first window. The grey horizontal is chance.

    Bottom: the within gain (last minus first) at each session and the across gain (next
    first minus last) at each boundary, with Newcombe 95% intervals, and a thin running
    total through each component. The text above the panels is each component's mean
    over the animal's sessions, with the number of gains averaged; it is shown, not saved
    (the same numbers are in `gain_summary`).

    ``show`` numbers: 1 within session, 2 across sessions, 3 and 4 the within and across
    running totals, 5 spans an unfitted session (only where one exists). Within and across
    take the top panel's segments and steps with them.

    ``separate_odors`` fits each odour alone (or takes ``fits`` per odour) and draws a
    figure per animal and odour, its windows counting that odour's choices, named top
    right in its port colour; returns ``{subjid: {odour: Figure}}``.
    """
    fits = _fits(subjids, dates, data, fits, mode, min_trials, separate_odors, selectors)
    figures, entries = odor_figures(
        fits, lambda part, odor: _edge_figures(part, odor, n, legend, show, save))
    legend_figure(entries, fontsize=annotation_size())
    return figures


def _edge_figures(fits, odor, n, legend, show, save):
    """`plot_edge_gains`' figures of one set of fits; ``odor`` names them."""
    windows, gains = edge_windows(fits, n), edge_gains(fits, n)
    summary = gain_summary(gains).set_index(["subjid", "component"])

    figures, entries = {}, []
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
        odor_title(top, odor)
        fig.legend(*in_order(bottom, _EDGE_ORDER), loc="outside lower center", ncols=3,
                   frameon=False, fontsize=annotation_size())
        entries += finish_figure(fig, legend, show)
        _save(fig, f"discrimination_learning_edge_gains{show_suffix(show)}"
                   f"{odor_tag(odor)}_sub-{subjid:03d}", fit, save, titles=False)
        figures[subjid] = fig
    return figures, entries


def _plot_profile(ax, profile, odor=None):
    """Observed accuracy with its Wilson band, and both models' mean prediction; per
    ``odor`` all three in its port colour, the fits dashed (M_c) and dotted (M_a)."""
    if odor is None:
        line_with_band(ax, profile, "center", "accuracy", "accuracy_lo", "accuracy_hi",
                       SERIES, "observed")
        ax.plot(profile["center"], profile["fitted_c"], color=_ACROSS, linewidth=2,
                label="M_c fit")
        ax.plot(profile["center"], profile["fitted_a"], color=REFERENCE, linewidth=2,
                linestyle=(0, (1, 1.5)), label="M_a fit")
    else:
        colour = PORT_COLOURS[odor.port]
        line_with_band(ax, profile, "center", "accuracy", "accuracy_lo", "accuracy_hi",
                       colour, f"{odor.name} observed")
        for column, label, linestyle in _ODOR_FITS:
            ax.plot(profile["center"], profile[column], color=colour, linewidth=2,
                    linestyle=linestyle, label=f"{odor.name} {label}")
    ax.axhline(0.5, **ZERO)


def plot_position_profile(subjids=None, dates=None, *, data=None, fits=None,
                          mode="completed", min_trials=MIN_TRIALS, bins=10, bin_width=10,
                          min_sessions=_MIN_SESSIONS, separate_odors=False, legend=None,
                          show=None, save=False, **selectors):
    """Accuracy by position in the session, pooled over each animal's fitted sessions.

        plot_position_profile(fits=fits)
        plot_position_profile(fits=fits, bins=5, bin_width=20)
        plot_position_profile(data=ab, separate_odors=True)

    Top: by choice number from the session start, in bins of ``bin_width``; a bin
    reached by fewer than ``min_sessions`` sessions is left out. Bottom: by fraction of
    the session, ``bins`` bins of equal count per session.

    Each panel carries the observed accuracy with its Wilson 95% band and two model
    predictions averaged over the same rows: M_a (flat within every session), which on
    the top panel is what the changing mix of sessions alone produces, and M_c (a
    straight line in log-odds per session), which is what the per-session slopes add up
    to once pooled. The grey horizontal is chance.

    ``show`` numbers: 1 observed (with its band), 2 M_c fit, 3 M_a fit, in both panels.

    ``separate_odors`` fits each odour alone (or takes ``fits`` per odour) and draws every
    odour on the same axes in its port colour, positions counting that odour's choices:
    observed solid, M_c dashed, M_a dotted. ``show`` then numbers the first odour's three,
    then the next's.
    """
    fits = _fits(subjids, dates, data, fits, mode, min_trials, separate_odors, selectors)
    split = is_per_odor(fits)
    parts = [(odor, part, position_profile(part, "choices", bin_width=bin_width),
              position_profile(part, "fraction", bins=bins)) for odor, part in _parts(fits)]
    order = _labels([odor for odor, *_ in parts], _PROFILE_ORDER)

    figures, entries = {}, []
    for subjid in sorted(set().union(*(part for _, part, *_ in parts))):
        fit = next(part[subjid] for _, part, *_ in parts if subjid in part)
        unit = _unit(fit["mode"])
        fig, (top, bottom) = figure(n_rows=2, width=_WIDTH, height=_PANEL_HEIGHT)
        for odor, part, by_choice, by_fraction in parts:
            if subjid not in part:
                continue
            choice = by_choice[(by_choice["subjid"] == subjid)
                               & (by_choice["n_sessions"] >= min_sessions)]
            fraction = by_fraction[by_fraction["subjid"] == subjid]
            _plot_profile(top, choice, odor)
            _plot_profile(bottom, fraction, odor)
        for ax in (top, bottom):
            style_axis(ax, ylabel="Accuracy", ylim=None)
        top.set_xlabel(f"{unit.capitalize()} in session")
        bottom.set_xlabel("Fraction of session")
        bottom.set_xlim(0, 1)
        fig.legend(*in_order(bottom, order), loc="outside lower center",
                   ncols=len(parts) if split else 3, frameon=False,
                   fontsize=annotation_size())
        entries += finish_figure(fig, legend, show)
        _save(fig, f"discrimination_learning_position_profile{odors_tag(split)}"
                   f"{show_suffix(show)}_sub-{subjid:03d}", fit, save)
        figures[subjid] = fig
    legend_figure(entries, fontsize=annotation_size())
    return figures


def _panel_title(ses, cells) -> str:
    """``ses 4  n 120  W +0.8``, one ``n`` and W per odour joined by ``/`` (``-`` where
    the odour's session was not fitted)."""
    n = "/".join("-" if cell is None else str(cell[0]) for cell in cells)
    gain = "/".join("-" if cell is None else f"{cell[1]:+.1f}" for cell in cells)
    return f"ses {ses}  n {n}  W {gain}"


def _grid_legend(axes, order):
    """The handles and labels over every panel, each label once, in ``order``."""
    found = {}
    for ax in axes:
        for handle, label in zip(*ax.get_legend_handles_labels()):
            found.setdefault(label, handle)
    labels = [label for label in order if label in found]
    return [found[label] for label in labels], labels


def plot_session_profiles(subjids=None, dates=None, *, data=None, fits=None,
                          mode="completed", min_trials=MIN_TRIALS, bins=5,
                          separate_odors=False, legend=None, show=None, save=False,
                          **selectors):
    """Accuracy by fraction of the session, one panel per fitted session.

        plot_session_profiles(fits=fits)
        plot_session_profiles(data=ab, separate_odors=True)

    Points are the accuracy over each of ``bins`` equal-count bins with Wilson 95%
    intervals; the line is the session's M_c fit, a straight line in log-odds and so a
    gentle curve here. Each panel is titled with the session and its W_s, M_c's gain
    across the session in log-odds. A profile that rises and then falls, or climbs only in
    its first bins, is one the line describes poorly. The grey horizontal is chance.

    ``show`` numbers: 1 observed, 2 M_c fit, in every panel.

    ``separate_odors`` fits each odour alone (or takes ``fits`` per odour) and draws every
    odour in each panel in its port colour, a bin's points side by side. A panel is a
    session either odour fitted; its title gives ``n`` and W per odour. ``show`` then
    numbers the first odour's two, then the next's.
    """
    fits = _fits(subjids, dates, data, fits, mode, min_trials, separate_odors, selectors)
    split = is_per_odor(fits)
    parts = []
    for odor, part in _parts(fits):
        gains = gain_decomposition(part)
        parts.append((odor, part, session_profile(part, bins=bins), fitted_curves(part),
                      gains[gains["component"] == "within"].set_index(
                          ["subjid", "session_idx"])))
    order = _labels([odor for odor, *_ in parts], _PROFILE_ORDER)
    offsets = (np.arange(len(parts)) - (len(parts) - 1) / 2) * _ODOR_DODGE

    figures, entries = {}, []
    for subjid in sorted(set().union(*(part for _, part, *_ in parts))):
        animal = [part[subjid] for _, part, *_ in parts if subjid in part]
        fit = animal[0]
        sessions = (pd.concat([f["sessions"][f["sessions"]["kept"]] for f in animal])
                    .drop_duplicates("session_idx").sort_values("session_idx"))
        columns = min(_GRID_COLUMNS, len(sessions))
        rows = math.ceil(len(sessions) / columns)
        fig, axes = plt.subplots(rows, columns, figsize=(_CELL_WIDTH * columns,
                                                         _CELL_HEIGHT * rows),
                                 sharex=True, sharey=True, squeeze=False,
                                 constrained_layout=True)
        for ax, session in zip(axes.flat, sessions.itertuples()):
            key = (subjid, session.session_idx)
            cells = []
            for offset, (odor, part, profiles, curves, within) in zip(offsets, parts):
                if key not in within.index:
                    cells.append(None)
                    continue
                name = "" if odor is None else f"{odor.name} "
                colour = None if odor is None else PORT_COLOURS[odor.port]
                curve = curves[(curves["subjid"] == subjid)
                               & (curves["session_idx"] == session.session_idx)]
                profile = profiles[(profiles["subjid"] == subjid)
                                   & (profiles["session_idx"] == session.session_idx)]
                ax.plot(curve["x"], curve["p"], color=colour or _ACROSS, linewidth=2,
                        label=f"{name}M_c fit")
                # Clipped: an all-correct bin's Wilson bound sits a rounding error below 1.
                err = np.clip([profile["accuracy"] - profile["accuracy_lo"],
                               profile["accuracy_hi"] - profile["accuracy"]], 0, None)
                ax.errorbar(profile["center"] + offset, profile["accuracy"], yerr=err,
                            fmt="o", color=colour or SERIES, markersize=6,
                            markeredgecolor="white", markeredgewidth=1, elinewidth=2,
                            capsize=0, label=f"{name}observed")
                n = part[subjid]["sessions"].set_index("session_idx").loc[
                    session.session_idx, "n"]
                cells.append((n, within.loc[key, "value"]))
            ax.axhline(0.5, **ZERO)
            style_axis(ax)
            ax.set_xlim(0, 1)
            ax.tick_params(labelsize=annotation_size())
            ax.set_title(_panel_title(session.ses, cells), loc="left",
                         fontsize=annotation_size())
        # An empty slot in the last row leaves the panel above it at the bottom of its
        # column, where the shared x axis would otherwise have hidden its tick labels.
        for index in range(len(sessions), rows * columns):
            axes.flat[index].set_visible(False)
            axes.flat[index - columns].tick_params(labelbottom=True)
        fig.supxlabel("Fraction of session", fontsize=text_size())
        fig.supylabel("Accuracy", fontsize=text_size())
        fig.legend(*_grid_legend(axes.flat, order), loc="outside upper right",
                   ncols=len(parts) if split else 2, frameon=False,
                   fontsize=annotation_size())
        entries += finish_figure(fig, legend, show)
        _save(fig, f"discrimination_learning_session_profiles{odors_tag(split)}"
                   f"{show_suffix(show)}_sub-{subjid:03d}", fit, save)
        figures[subjid] = fig
    legend_figure(entries, fontsize=annotation_size())
    return figures
