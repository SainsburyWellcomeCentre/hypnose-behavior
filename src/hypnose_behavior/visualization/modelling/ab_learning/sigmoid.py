"""Figures for the single-sigmoid fits (``modelling.ab_learning.sigmoid``).

Each ``plot_*`` takes the same subject/session selection as `load_ab_data` -- or the
already-loaded frames as ``data=``, or the fits themselves -- and draws **one figure per
animal**. Returns ``{subjid: Figure}``; ``save=True`` files each figure under its own
animal with `io.save.save_figure`.

- `plot_accuracy_sigmoid` -- the accuracy fits against excess correct and binned accuracy
  over the row index, and, as a second figure, the lnL surface over (center, width).
- `plot_sigmoid_anchors`  -- every fit on task time: the 1.1 running counts with each
  fit's expected count, and the rates and accuracy per session with the fitted curves.
"""
from __future__ import annotations

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.transforms import offset_copy

from hypnose_behavior.io.save import (
    finish_figure,
    legend_figure,
    legends_apart,
    save_figure,
    show_suffix,
    tie,
)
from hypnose_behavior.modelling.ab_learning.data import session_bounds
from hypnose_behavior.modelling.ab_learning.diagnostics import wilson_interval
from hypnose_behavior.modelling.ab_learning.sigmoid import (
    VARIANTS,
    erf_curve,
    expected_count,
    expected_excess,
    fit_accuracy,
    fit_rate,
    variant_test,
)
from hypnose_behavior.visualization.modelling.ab_learning._common import (
    BAND_ALPHA,
    BOUNDARY,
    REFERENCE,
    SECOND,
    SERIES,
    ZERO,
    annotation_size,
    figure,
    index_bounds,
    require_data,
    save_scope,
    style_axis,
    text_size,
)
from hypnose_behavior.visualization.modelling.ab_learning.cumulative import CURVE_STYLES

__all__ = ["SHOW_FITS", "plot_accuracy_sigmoid", "plot_sigmoid_anchors"]

# Rows per accuracy bin drawn behind the fits.
_BIN = 50

# lnL below the best at which the surface's colour saturates, and the 95% contour.
_FLOOR = -10.0
_CONTOUR = -1.92

# One-hue sequential ramp, surface-light to dark blue.
_SURFACE_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "ab_surface", ["#f4f8fd", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])

# The observed choices keep excess correct's colour from 1.1, drawn thick; the fits run
# over them closely dotted in a dark colour, so the data shows between the dots.
_OBSERVED_LABEL = "Excess correct"
_OBSERVED = dict(color=SECOND, linewidth=2.8, zorder=1)
_DATA = dict(color=SECOND, markersize=6, zorder=1)
_DOTTED = dict(linewidth=2.6, linestyle=(0, (1, 1.6)), dash_capstyle="round")
_FIT_STYLES = {
    "free": dict(color="#1c5cab", zorder=4, **_DOTTED),
    "chance": dict(color="#4f4e4a", zorder=3, **_DOTTED),
}
_OTHER_OPTIMA = (dict(color=SERIES, linewidth=1.4, linestyle=(0, (1, 2.5)),
                      dash_capstyle="round", alpha=0.6, zorder=2), "other optima")

# `plot_accuracy_sigmoid`'s ``show_fit``: the variant picked per animal, either one, or
# both (None).
SHOW_FITS = ("best", "free", "chance", None)

# The ∇ marking k_s: its height as a share of the tick-label size, and its gap (points)
# above the observed line.
_SWITCH_SIZE = 0.7
_SWITCH_GAP = 3.0

# `plot_sigmoid_anchors`' legend labels, and its midpoint table's columns.
_ANCHOR_LABELS = {"initiations": "Initiations", "excess": "Excess correct",
                  "rewards": "Rewards"}
# Its legend's sample lines, in font sizes: long enough to show two of the fit's dashes,
# which scale with the line width, so "Sigmoid fit" reads as dashed.
_ANCHOR_HANDLE_LENGTH = 2.5

# A fit drawn over its observed count: the count's colour, dashed.
_FITTED = dict(linewidth=2.2, linestyle=(0, (4, 2)), alpha=1.0, zorder=5)
_MIDPOINT = dict(marker="o", markersize=9, markeredgecolor="white", markeredgewidth=1.5,
                 linestyle="none", zorder=6)

_WIDTH = 10.0
_HEIGHT = 3.4


def _save(fig, name, sessions, save, subjid):
    if not save:
        return
    save_figure(fig, name, **save_scope(sessions, subjid))


def _width_text(width, lo, hi):
    """``w 168 [102, 218]`` in rows, with a step and an open end spelled out."""
    def fmt(v):
        return "inf" if np.isinf(v) else f"{v:.0f}"
    point = "< 1" if width < 1 else f"{width:.0f}"
    return f"w {point} [{fmt(lo)}, {fmt(hi)}]"


def _binned(rows):
    """Accuracy per `_BIN` rows, with Wilson bounds, at each bin's middle row."""
    bins = rows.groupby(rows["k"] // _BIN).agg(k=("k", "mean"), n=("y", "size"),
                                                c=("y", "sum"))
    lo, hi = wilson_interval(bins["c"], bins["n"])
    return bins.assign(p=bins["c"] / bins["n"], lo=lo, hi=hi)


def _fit_label(variant, fit):
    """``fit, pᵢ = 0.5``, or ``fit, pᵢ free = 0.46`` with the animal's fitted value."""
    if variant == "chance":
        return "fit, pᵢ = 0.5"
    return f"fit, pᵢ free = {fit['optima'].iloc[0]['initial']:.2f}"


def _shown_variants(fits, subjid, show_fit):
    """The variants to draw for one animal, the one that leads first."""
    present = [v for v in VARIANTS if subjid in fits.get(v, {})]
    if show_fit is None:
        return present
    if show_fit == "best":
        if len(present) < 2:
            return present
        return [variant_test({v: {subjid: fits[v][subjid]} for v in VARIANTS})
                ["picked"].iloc[0]]
    return [show_fit] if show_fit in present else []


def _curves(fits, subjid, variants):
    """``(fit, rank, style, series)`` for each curve to draw: each variant's best, and the
    free fit's other optima when it is drawn; ``series`` is the curve's legend label."""
    out = []
    for variant in variants:
        fit = fits[variant][subjid]
        out.append((fit, 0, _FIT_STYLES[variant], _fit_label(variant, fit)))
        if variant == "free":
            out += [(fit, rank, *_OTHER_OPTIMA) for rank in range(1, len(fit["optima"]))]
    return out


def _headline(fits, subjid, lead):
    """The leading fit, its next optimum, and what fixing pᵢ at 0.5 costs."""
    lines = []
    fit = fits.get(lead, {}).get(subjid)
    if fit is not None:
        best = fit["optima"].iloc[0]
        where = f"ses {best['ses']}" if best["inside"] else "outside the data"
        lines.append(f"k_s {best['center']:.0f} ({where}), "
                     + _width_text(best["width"], best["width_lo"], best["width_hi"]))
        if len(fit["optima"]) > 1:
            second = fit["optima"].iloc[1]
            lines.append(f"next optimum: k_s {second['center']:.0f}, "
                         f"ΔlnL {second['delta_lnl']:.1f}")
    free, chance = (fits.get(v, {}).get(subjid) for v in VARIANTS)
    if free is not None and chance is not None:
        lines.append("pᵢ = 0.5: ΔlnL "
                     f"{free['optima']['lnl'].iloc[0] - chance['optima']['lnl'].iloc[0]:.1f}")
    return lines


def _draw_excess(ax, rows, curves):
    """Observed excess correct over the row count, and each curve's expected excess; the
    first curve of a series carries its label, the rest are tied to it."""
    count = np.r_[0, rows["k"].to_numpy() + 1]
    observed = np.r_[0.0, np.cumsum(rows["y"].to_numpy() - 0.5)]
    ax.plot(count, observed, drawstyle="steps-post", label=_OBSERVED_LABEL, **_OBSERVED)
    labelled = set()
    for fit, rank, style, series in curves:
        line, = ax.plot(count, np.r_[0.0, expected_excess(fit, rank)], **style)
        if series in labelled:
            tie(line, series)
        else:
            line.set_label(series)
            labelled.add(series)
    ax.axhline(0, **ZERO)


def _mark_switch(ax, rows, center):
    """A solid black ∇ at ``k_s``, sitting just above the observed excess-correct line,
    the size of a tick label."""
    size = text_size() * _SWITCH_SIZE
    observed = np.cumsum(rows["y"].to_numpy() - 0.5)
    y = np.interp(center, rows["k"].to_numpy(dtype=float) + 1, observed)
    above = offset_copy(ax.transData, fig=ax.figure, y=size / 2 + _SWITCH_GAP, units="points")
    marker, = ax.plot(center, y, transform=above, marker="v", markersize=size, color="black",
                      linestyle="none", zorder=6, clip_on=False)
    return marker


def _draw_accuracy(ax, rows, curves):
    """Accuracy per `_BIN` rows and each curve's p(k), tied to their series."""
    k = rows["k"].to_numpy(dtype=float)
    bins = _binned(rows)
    spread = np.clip([bins["p"] - bins["lo"], bins["hi"] - bins["p"]], 0, None)
    ax.errorbar(bins["k"], bins["p"], yerr=spread, fmt="o", elinewidth=1, capsize=0,
                label=_OBSERVED_LABEL, **_DATA)
    for fit, rank, style, series in curves:
        optimum = fit["optima"].iloc[rank]
        line, = ax.plot(k, erf_curve(k, optimum["initial"], optimum["final"],
                                     optimum["center"], optimum["width"]), **style)
        tie(line, series)
    ax.axhline(0.5, **ZERO)


def _draw_surface(ax, fit):
    """The free fit's lnL over (center, width), relative to its best grid point."""
    surface = fit["surface"]
    centers, widths = surface["centers"], surface["widths"]
    relative = surface["lnl"] - surface["lnl"].max()
    mesh = ax.pcolormesh(centers, widths, np.maximum(relative, _FLOOR).T,
                         cmap=_SURFACE_CMAP, vmin=_FLOOR, vmax=0, shading="nearest",
                         rasterized=True)
    ax.contour(centers, widths, relative.T, levels=[_CONTOUR], colors=SECOND,
               linewidths=1.2)
    for _, optimum in fit["optima"].iterrows():
        x = np.clip(optimum["center"], centers[0], centers[-1])
        y = np.clip(optimum["width"], widths[0], widths[-1])
        ax.plot(x, y, color=SECOND, **_MIDPOINT)
        ax.annotate(str(int(optimum["rank"]) + 1), (x, y), xytext=(6, 4),
                    textcoords="offset points", fontsize=annotation_size(), zorder=7)
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(mticker.LogLocator(base=10, numticks=10))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_formatter(mticker.NullFormatter())
    ax.set_ylabel("Width")
    ax.spines[["top", "right"]].set_visible(False)
    # On top, right-aligned: the layout makes room for it and its label, and the axis keeps
    # the fit figure's width, so the two line up on a slide.
    bar = ax.figure.colorbar(mesh, ax=ax, location="top", shrink=0.25, aspect=12,
                             anchor=(1.0, 0.0), ticks=[_FLOOR, _FLOOR / 2, 0])
    bar.ax.tick_params(labelsize=annotation_size(), length=2)
    bar.set_label("ΔlnL", fontsize=annotation_size())


def plot_accuracy_sigmoid(subjids=None, dates=None, *, data=None, mode="completed",
                          fits=None, show_fit="best", surface=True, legend=None, show=None,
                          save=False, **selectors):
    """The accuracy sigmoid over the row index: a fit figure and a surface figure per
    animal.

        acc = {v: fit_accuracy(ab, "completed", v) for v in VARIANTS}
        figs = plot_accuracy_sigmoid(fits=acc)          # figs[60]["fit"], figs[60]["surface"]
        plot_accuracy_sigmoid(fits=acc, show_fit=None)  # both variants

    ``fits`` maps a `VARIANTS` entry to its `fit_accuracy` result; without it both are
    fitted from the selection in ``mode``. ``show_fit`` is ``"best"`` (each animal's pick
    of `sigmoid.variant_test`), ``"free"``, ``"chance"``, or None for both; the leading fit
    -- the one drawn, the free one when both are -- sets the shaded ``k_s +/- w``, the
    summary and the surface. Returns ``{subjid: {"fit": Figure, "surface": Figure}}``,
    without ``"surface"`` when ``surface=False``.

    Fit figure -- the observed choices thick orange, each fit closely dotted over them
    (dark blue pᵢ free, dark grey pᵢ = 0.5), the free fit's other optima faint dotted blue:

    - top: observed excess correct (every row) and each fit's expected excess,
      ``cumsum(p(k) - 0.5)``, the running sum of the bottom panel's curve; a black ∇ above
      the data marks the leading fit's k_s. Where orange leaves a fitted line, the
      one-change curve misses the data.
    - bottom: accuracy per 50 rows (Wilson 95%) and each fit's p(k).

    Surface figure: the leading fit's lnL over (center, width), relative to its best grid
    point, with the 95% contour and the optima numbered by rank. Separate dark islands are
    separate optima.

    ``legend`` and ``show`` follow the plotter convention (`hypnose_helpers.viz.plotter`);
    ``show`` numbers the series in legend order, observed first. The fit labels carry the
    animal's own pᵢ, so a legend set apart is one small figure per animal, and the fit
    summary is printed instead of drawn.
    """
    if show_fit not in SHOW_FITS:
        raise ValueError(f"show_fit must be one of {SHOW_FITS}, got {show_fit!r}")
    if fits is None:
        data = require_data(subjids, dates, selectors, data)
        fits = {variant: fit_accuracy(data, mode, variant) for variant in VARIANTS}
    reference = next(iter(fits.values()))

    figures = {}
    for subjid in sorted(reference):
        rows, mode = reference[subjid]["rows"], reference[subjid]["mode"]
        spans = index_bounds(rows)
        unit = "trials" if mode == "completed" else "choice attempts"
        variants = _shown_variants(fits, subjid, show_fit)
        lead = variants[0] if variants else None
        lead_fit = fits[lead][subjid] if lead else None

        fig, (excess, accuracy) = figure(2, width=_WIDTH, height=_HEIGHT)
        accuracy.sharex(excess)
        curves = _curves(fits, subjid, variants)
        _draw_excess(excess, rows, curves)
        _draw_accuracy(accuracy, rows, curves)
        if lead_fit is not None:
            best = lead_fit["optima"].iloc[0]
            for ax in (excess, accuracy):
                tie(ax.axvspan(best["center"] - best["width"], best["center"] + best["width"],
                               color=_FIT_STYLES[lead]["color"], alpha=BAND_ALPHA / 2,
                               linewidth=0, zorder=0),
                    _fit_label(lead, lead_fit))
            if best["inside"]:
                tie(_mark_switch(excess, rows, best["center"]), _fit_label(lead, lead_fit))
        for ax in (excess, accuracy):
            for start in spans["start"].to_numpy()[1:]:
                ax.axvline(start, **BOUNDARY)
        style_axis(excess, ylabel="Excess correct", ylim=None)
        style_axis(accuracy, ylabel="Accuracy", ylim=(0, 1.02))
        excess.set_xlim(0, len(rows))
        accuracy.set_xlabel(unit.capitalize())
        excess.legend(frameon=False, fontsize=annotation_size(), loc="lower right")

        headline = _headline(fits, subjid, lead)
        if legends_apart(legend):
            print(f"sub-{subjid:03d} | {mode}: " + " | ".join(headline))
        else:
            excess.text(0.01, 0.97, "\n".join(headline), transform=excess.transAxes,
                        fontsize=annotation_size(), va="top", ha="left")
        entries = finish_figure(fig, legend, show)
        _save(fig, f"ab_learning_accuracy_sigmoid{show_suffix(show)}_{mode}_sub-{subjid:03d}",
              rows, save, subjid)
        legend_figure(entries, fontsize=annotation_size())
        figures[int(subjid)] = {"fit": fig}

        if surface and lead_fit is not None:
            fig, (ax,) = figure(width=_WIDTH, height=_HEIGHT)
            _draw_surface(ax, lead_fit)
            for start in spans["start"].to_numpy()[1:]:
                ax.axvline(start, **BOUNDARY)
            ax.set_xlim(0, len(rows))
            ax.set_xlabel(unit.capitalize())
            _save(fig, f"ab_learning_accuracy_sigmoid_surface_{mode}_sub-{subjid:03d}", rows,
                  save, subjid)
            figures[int(subjid)]["surface"] = fig
    return figures


def _observed_and_fitted(name, fit, span):
    """``(x, observed, fitted_x, fitted, midpoint_xy)`` of one running count on task time."""
    if name == "excess":
        rows = fit["rows"]
        hours = rows["hours"].to_numpy(dtype=float)
        observed = np.cumsum(rows["y"].to_numpy() - 0.5)
        fitted = expected_excess(fit)
        best = fit["optima"].iloc[0]
        k = rows["k"].to_numpy(dtype=float)
        midpoint = (best["center_hours"], np.interp(best["center"], k, fitted))
        return (np.r_[0.0, hours], np.r_[0.0, observed], np.r_[0.0, hours],
                np.r_[0.0, fitted], midpoint)
    times = fit["times"]
    clock = np.linspace(0, span, 800)
    best = fit["optima"].iloc[0]
    midpoint = (best["center"], float(expected_count(fit, best["center"])))
    return (np.r_[0.0, times], np.r_[0.0, np.arange(1, times.size + 1)], clock,
            expected_count(fit, clock), midpoint)


def _draw_counts(ax, fits, span):
    """The scaled running counts and their fits, with the figure's one legend; returns
    each fit's midpoint in hours."""
    midpoints = {}
    for name, fit in fits.items():
        style = CURVE_STYLES[name]
        x, observed, fitted_x, fitted, (mid_x, mid_y) = _observed_and_fitted(name, fit, span)
        scale = float(np.abs(observed).max()) or 1.0
        ax.plot(x, observed / scale, drawstyle="steps-post",
                **{**style, "label": _ANCHOR_LABELS[name]})
        ax.plot(fitted_x, fitted / scale, color=style["color"], **_FITTED)
        ax.plot(mid_x, mid_y / scale, color=style["color"], **_MIDPOINT)
        midpoints[name] = mid_x
    ax.axhline(0, **ZERO)
    style_axis(ax, ylabel="Normalized count", ylim=None)
    handles, labels = ax.get_legend_handles_labels()
    handles.append(Line2D([], [], color=REFERENCE, **{k: v for k, v in _FITTED.items()
                                                       if k != "zorder"}))
    labels.append("Sigmoid fit")
    ax.legend(handles, labels, frameon=False, fontsize=annotation_size(), loc="upper left",
              handlelength=_ANCHOR_HANDLE_LENGTH)
    return midpoints


def _segments(ax, frame, column, colour):
    """One horizontal bar per session, over its task-time span."""
    ax.hlines(frame[column], frame["start"], frame["end"], color=colour, linewidth=4,
              alpha=0.35, zorder=1)


def _draw_rates(ax, data, spans, fits, subjid, span):
    """Initiations and rewards per task hour, per session, with each rate fit's curve."""
    trials = data["trials"][data["trials"]["subjid"] == subjid]
    counts = trials.groupby("session_idx").agg(
        initiations=("outcome", "size"),
        rewards=("outcome", lambda o: int((o == "rewarded").sum())))
    rates = spans.set_index("session_idx").join(counts).fillna(
        {"initiations": 0, "rewards": 0})
    hours = (rates["end"] - rates["start"]).where(lambda h: h > 0)
    clock = np.linspace(0, span, 800)
    for name in ("initiations", "rewards"):
        colour = CURVE_STYLES[name]["color"]
        _segments(ax, rates.assign(rate=rates[name] / hours), "rate", colour)
        best = fits[name]["optima"].iloc[0]
        ax.plot(clock, erf_curve(clock, best["initial"], best["final"], best["center"],
                                 best["width"]), color=colour, linewidth=2.2, zorder=3)
    style_axis(ax, ylabel="Per hour", ylim=None)
    ax.set_ylim(bottom=0)


def _draw_accuracy_on_time(ax, spans, fit):
    """Accuracy per session, and the fitted p(k) at each trial's task time."""
    rows = fit["rows"]
    colour = CURVE_STYLES["excess"]["color"]
    per_session = spans.set_index("session_idx").join(
        rows.groupby("session_idx")["y"].mean().rename("accuracy"), how="inner")
    _segments(ax, per_session, "accuracy", colour)
    best = fit["optima"].iloc[0]
    k = rows["k"].to_numpy(dtype=float)
    ax.plot(rows["hours"], erf_curve(k, best["initial"], best["final"], best["center"],
                                     best["width"]),
            color=colour, linewidth=2.2, zorder=3)
    ax.axhline(0.5, **ZERO)
    style_axis(ax, ylabel="Accuracy", ylim=(0, 1.02))


def plot_sigmoid_anchors(subjids=None, dates=None, *, data=None, accuracy=None,
                         engagement=None, rewards=None, legend=None, save=False, **selectors):
    """Every sigmoid fit on task time, as running counts and as rates, one figure per
    animal.

        plot_sigmoid_anchors(data=ab, accuracy=acc["free"], engagement=eng, rewards=rew)

    ``accuracy`` is any `fit_accuracy` result (completed, initial free when omitted);
    ``engagement`` / ``rewards`` are `fit_rate` results, fitted when omitted.

    - top: the 1.1 counts (solid; initiations, excess correct over the accuracy fit's
      rows, rewards), each divided by its own largest value, and each fit's expected
      running count scaled alike (dashed): a rate fit integrated over task time, or
      ``cumsum(p(k) - 0.5)`` at each trial's time. Dots mark the midpoints.
    - middle: initiations and rewards per task hour, per session (faded bars), and the
      rate fits.
    - bottom: accuracy per session, and the accuracy fit's p(k) at each trial's time. It
      is per trial, not per hour, so it bends wherever trials bunch up in time.
    - dashed verticals: each fit's midpoint, in its colour, on every panel.

    Rewards are initiation rate times accuracy, so their midpoint says which of the two
    drove the reward rate, not a third change.

    One legend, on the top panel; the colours carry down. ``legend`` follows the plotter
    convention, and set apart it is one figure for all animals. The midpoints are printed
    as a table, one row per animal.
    """
    data = require_data(subjids, dates, selectors, data)
    accuracy = accuracy if accuracy is not None else fit_accuracy(data)
    engagement = engagement if engagement is not None else fit_rate(data)
    rewards = rewards if rewards is not None else fit_rate(data, "rewards")
    bounds = session_bounds(data)

    figures, entries, table = {}, [], []
    for subjid in sorted(set(accuracy) & set(engagement) & set(rewards)):
        spans = bounds[bounds["subjid"] == subjid].sort_values("session_idx")
        span = float(spans["end"].max())
        fits = {"initiations": engagement[subjid], "excess": accuracy[subjid],
                "rewards": rewards[subjid]}
        fig, (counts, rates, acc) = figure(3, width=_WIDTH, height=_HEIGHT)
        for ax in (rates, acc):
            ax.sharex(counts)
        midpoints = _draw_counts(counts, fits, span)
        _draw_rates(rates, data, spans, fits, subjid, span)
        _draw_accuracy_on_time(acc, spans, fits["excess"])
        for ax in (counts, rates, acc):
            for start in spans["start"].to_numpy()[1:]:
                ax.axvline(start, **BOUNDARY)
            for name, hours in midpoints.items():
                ax.axvline(hours, color=CURVE_STYLES[name]["color"], linestyle="--",
                           linewidth=1.2, zorder=2)
        counts.set_xlim(0, span)
        acc.set_xlabel("Task time (h)")
        entries += finish_figure(fig, legend)
        _save(fig, f"ab_learning_sigmoid_anchors_sub-{subjid:03d}", data["sessions"], save,
              subjid)
        figures[int(subjid)] = fig
        table.append({"subjid": int(subjid),
                      **{_ANCHOR_LABELS[name]: hours for name, hours in midpoints.items()}})
    with plt.rc_context({"legend.handlelength": _ANCHOR_HANDLE_LENGTH}):
        legend_figure(entries, fontsize=annotation_size())
    print("Sigmoid midpoints (task time, h):")
    print(pd.DataFrame(table).to_string(index=False, float_format=lambda v: f"{v:.1f}"))
    return figures
