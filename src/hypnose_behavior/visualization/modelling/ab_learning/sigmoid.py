"""Figures for the single-sigmoid fits (``modelling.ab_learning.sigmoid``).

Each ``plot_*`` takes the same subject/session selection as `load_ab_data` -- or the
already-loaded frames as ``data=``, or the fits themselves -- and draws **one figure per
animal**. Returns ``{subjid: Figure}``; ``save=True`` files each figure under its own
animal with `io.save.save_figure`.

- `plot_accuracy_sigmoid` -- the accuracy fits against excess correct and binned accuracy
  over the row index, and the lnL surface over (center, width).
- `plot_sigmoid_anchors`  -- every fit on task time: the 1.1 running counts with each
  fit's expected count, and the rates and accuracy per session with the fitted curves.
"""
from __future__ import annotations

import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
import numpy as np
from matplotlib.lines import Line2D

from hypnose_behavior.io.save import save_figure
from hypnose_behavior.modelling.ab_learning.data import session_bounds
from hypnose_behavior.modelling.ab_learning.diagnostics import wilson_interval
from hypnose_behavior.modelling.ab_learning.sigmoid import (
    VARIANTS,
    erf_curve,
    expected_count,
    expected_excess,
    fit_accuracy,
    fit_rate,
)
from hypnose_behavior.visualization.modelling.ab_learning._common import (
    BAND_ALPHA,
    BOUNDARY,
    REFERENCE,
    SECOND,
    SERIES,
    annotation_size,
    figure,
    index_bounds,
    require_data,
    save_scope,
    session_axis,
    style_axis,
    title,
)
from hypnose_behavior.visualization.modelling.ab_learning.cumulative import CURVE_STYLES

__all__ = ["plot_accuracy_sigmoid", "plot_sigmoid_anchors"]

# Rows per accuracy bin drawn behind the fits.
_BIN = 50

# lnL below the best at which the surface's colour saturates, and the 95% contour.
_FLOOR = -10.0
_CONTOUR = -1.92

# One-hue sequential ramp, surface-light to dark blue.
_SURFACE_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "ab_surface", ["#f4f8fd", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])

# The observed choices keep excess correct's colour from 1.1; the fits take blue and grey.
_VARIANT_STYLES = {
    "free": dict(color=SERIES, linewidth=2.2, zorder=4, label="fit, initial free"),
    "chance": dict(color=REFERENCE, linewidth=2.0, linestyle=(0, (5, 2)), zorder=3,
                   label="fit, initial 0.5"),
}
_OTHER_OPTIMA = dict(color=SERIES, linewidth=1.2, linestyle=":", zorder=2)
_OBSERVED = dict(color=SECOND, linewidth=1.8, zorder=1)
_DATA = dict(color=SECOND, markersize=5, zorder=1)
_ZERO = dict(color=REFERENCE, linestyle="--", linewidth=1, zorder=0)

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


def _curves(fits, subjid):
    """``(fit, rank, style)`` for each curve to draw: every variant's best, and the free
    fit's other optima."""
    out = []
    for variant, style in _VARIANT_STYLES.items():
        fit = fits.get(variant, {}).get(subjid)
        if fit is None:
            continue
        out.append((fit, 0, style))
        if variant == "free":
            out += [(fit, rank, {**_OTHER_OPTIMA, "label": "other optima" if rank == 1
                                 else None})
                    for rank in range(1, len(fit["optima"]))]
    return out


def _headline(fits, subjid):
    """The best free fit, the next optimum, and what fixing the initial at 0.5 costs."""
    lines = []
    free, chance = (fits.get(v, {}).get(subjid) for v in VARIANTS)
    if free is not None:
        best = free["optima"].iloc[0]
        where = f"ses {best['ses']}" if best["inside"] else "outside the data"
        lines.append(f"k_s {best['center']:.0f} ({where}), "
                     + _width_text(best["width"], best["width_lo"], best["width_hi"]))
        if len(free["optima"]) > 1:
            second = free["optima"].iloc[1]
            lines.append(f"next optimum: k_s {second['center']:.0f}, "
                         f"ΔlnL {second['delta_lnl']:.1f}")
    if free is not None and chance is not None:
        lines.append("initial 0.5: ΔlnL "
                     f"{free['optima']['lnl'].iloc[0] - chance['optima']['lnl'].iloc[0]:.1f}")
    return "\n".join(lines)


def _draw_excess(ax, rows, curves):
    """Observed excess correct over the row count, and each curve's expected excess."""
    count = np.r_[0, rows["k"].to_numpy() + 1]
    observed = np.r_[0.0, np.cumsum(rows["y"].to_numpy() - 0.5)]
    ax.plot(count, observed, drawstyle="steps-post", label="observed", **_OBSERVED)
    for fit, rank, style in curves:
        ax.plot(count, np.r_[0.0, expected_excess(fit, rank)], **style)
    ax.axhline(0, **_ZERO)


def _draw_accuracy(ax, rows, curves):
    """Accuracy per `_BIN` rows and each curve's p(k)."""
    k = rows["k"].to_numpy(dtype=float)
    bins = _binned(rows)
    spread = np.clip([bins["p"] - bins["lo"], bins["hi"] - bins["p"]], 0, None)
    ax.errorbar(bins["k"], bins["p"], yerr=spread, fmt="o", elinewidth=1, capsize=0,
                label=f"per {_BIN}", **_DATA)
    for fit, rank, style in curves:
        optimum = fit["optima"].iloc[rank]
        ax.plot(k, erf_curve(k, optimum["initial"], optimum["final"], optimum["center"],
                             optimum["width"]), **{**style, "label": None})
    ax.axhline(0.5, **_ZERO)


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
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_formatter(mticker.NullFormatter())
    ax.set_ylabel("width")
    ax.spines[["top", "right"]].set_visible(False)
    bar_ax = ax.inset_axes([0.72, 1.02, 0.26, 0.05])
    bar = ax.figure.colorbar(mesh, cax=bar_ax, orientation="horizontal",
                             ticks=[_FLOOR, _FLOOR / 2, 0])
    bar.ax.tick_params(labelsize=annotation_size(), length=2)
    bar.ax.set_title("ΔlnL", fontsize=annotation_size(), loc="left")


def plot_accuracy_sigmoid(subjids=None, dates=None, *, data=None, mode="completed",
                          fits=None, save=False, **selectors):
    """The accuracy sigmoid over the row index, one figure per animal.

        acc = {v: fit_accuracy(ab, "completed", v) for v in VARIANTS}
        plot_accuracy_sigmoid(fits=acc)

    ``fits`` maps a `VARIANTS` entry to its `fit_accuracy` result; without it both are
    fitted from the selection in ``mode``. Orange is the observed choices, blue the free
    fit, dashed grey the fit with the initial accuracy at 0.5, dotted blue the free fit's
    other optima.

    - top: observed excess correct (every row) and each fit's expected excess,
      ``cumsum(p(k) - 0.5)``, the running sum of the middle panel's curve; the free fit's
      ``k_s +/- w`` shaded. Where orange leaves a fitted line, the one-change curve misses
      the data.
    - middle: accuracy per 50 rows (Wilson 95%) and each fit's p(k).
    - bottom: the free fit's lnL over (center, width), relative to its best grid point,
      with the 95% contour and the optima numbered by rank. Separate dark islands are
      separate optima.
    """
    if fits is None:
        data = require_data(subjids, dates, selectors, data)
        fits = {variant: fit_accuracy(data, mode, variant) for variant in VARIANTS}
    reference = next(iter(fits.values()))

    figures = {}
    for subjid in sorted(reference):
        rows, mode = reference[subjid]["rows"], reference[subjid]["mode"]
        spans = index_bounds(rows)
        curves = _curves(fits, subjid)
        unit = "trials" if mode == "completed" else "choice attempts"

        fig, (excess, accuracy, surface) = figure(3, width=_WIDTH, height=_HEIGHT)
        for ax in (accuracy, surface):
            ax.sharex(excess)
        _draw_excess(excess, rows, curves)
        _draw_accuracy(accuracy, rows, curves)
        free = fits.get("free", {}).get(subjid)
        if free is not None:
            best = free["optima"].iloc[0]
            for ax in (excess, accuracy):
                ax.axvspan(best["center"] - best["width"], best["center"] + best["width"],
                           color=SERIES, alpha=BAND_ALPHA / 2, linewidth=0, zorder=0)
            _draw_surface(surface, free)
        for ax in (excess, accuracy, surface):
            for start in spans["start"].to_numpy()[1:]:
                ax.axvline(start, **BOUNDARY)

        style_axis(excess, ylabel="excess correct", ylim=None)
        excess.text(0.01, 0.97, _headline(fits, subjid), transform=excess.transAxes,
                    fontsize=annotation_size(), va="top", ha="left")
        excess.legend(frameon=False, fontsize=annotation_size(), loc="lower right")
        excess.set_xlim(0, len(rows))
        session_axis(excess, spans)
        style_axis(accuracy, ylabel="accuracy", ylim=(0, 1.02))
        accuracy.legend(frameon=False, fontsize=annotation_size(), loc="lower right")
        surface.set_xlabel(unit)
        title(fig, f"accuracy sigmoid | {mode}", subjid)
        _save(fig, f"ab_learning_accuracy_sigmoid_{mode}_sub-{subjid:03d}", rows, save,
              subjid)
        figures[int(subjid)] = fig
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
    """The scaled running counts and their fits; returns each fit's midpoint in hours."""
    midpoints = {}
    for name, fit in fits.items():
        style = CURVE_STYLES[name]
        x, observed, fitted_x, fitted, (mid_x, mid_y) = _observed_and_fitted(name, fit, span)
        scale = float(np.abs(observed).max()) or 1.0
        ax.plot(x, observed / scale, drawstyle="steps-post",
                **{**style, "label": f"{style['label']}, midpoint {mid_x:.1f} h"})
        ax.plot(fitted_x, fitted / scale, color=style["color"], **_FITTED)
        ax.plot(mid_x, mid_y / scale, color=style["color"], **_MIDPOINT)
        midpoints[name] = mid_x
    ax.axhline(0, **_ZERO)
    style_axis(ax, ylabel="scaled count", ylim=None)
    handles, labels = ax.get_legend_handles_labels()
    handles.append(Line2D([], [], color=REFERENCE, **{k: v for k, v in _FITTED.items()
                                                       if k != "zorder"}))
    labels.append("sigmoid fit")
    ax.legend(handles, labels, frameon=False, fontsize=annotation_size(), loc="upper left")
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
                                 best["width"]), color=colour, linewidth=2.2, zorder=3,
                label=name)
    style_axis(ax, ylabel="per hour", ylim=None)
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, fontsize=annotation_size(), loc="upper left")


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
            color=colour, linewidth=2.2, zorder=3, label="accuracy")
    ax.axhline(0.5, **_ZERO)
    style_axis(ax, ylabel="accuracy", ylim=(0, 1.02))
    ax.legend(frameon=False, fontsize=annotation_size(), loc="lower right")


def plot_sigmoid_anchors(subjids=None, dates=None, *, data=None, accuracy=None,
                         engagement=None, rewards=None, save=False, **selectors):
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
    """
    data = require_data(subjids, dates, selectors, data)
    accuracy = accuracy if accuracy is not None else fit_accuracy(data)
    engagement = engagement if engagement is not None else fit_rate(data)
    rewards = rewards if rewards is not None else fit_rate(data, "rewards")
    bounds = session_bounds(data)

    figures = {}
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
        session_axis(counts, spans)
        acc.set_xlabel("task time (h)")
        title(fig, f"sigmoid fits on task time | accuracy {fits['excess']['mode']}, "
                   f"initial {fits['excess']['variant']}", subjid)
        _save(fig, f"ab_learning_sigmoid_anchors_sub-{subjid:03d}", data["sessions"], save,
              subjid)
        figures[int(subjid)] = fig
    return figures
