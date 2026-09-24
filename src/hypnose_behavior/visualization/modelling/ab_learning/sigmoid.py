"""Figures for the single-sigmoid fits (``modelling.ab_learning.sigmoid``).

Each ``plot_*`` takes the same subject/session selection as `load_ab_data` -- or the
already-loaded frames as ``data=``, or the fits themselves -- and draws **one figure per
animal**. Returns ``{subjid: Figure}``; ``save=True`` files each figure under its own
animal with `io.save.save_figure`.

- `plot_accuracy_sigmoid` -- the accuracy fits over the row index, and the lnL surface
  over (center, width) that shows whether they are one optimum or several.
- `plot_sigmoid_anchors`  -- engagement, reward and accuracy fits on one task-time axis.
"""
from __future__ import annotations

import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
import numpy as np

from hypnose_behavior.io.save import save_figure
from hypnose_behavior.modelling.ab_learning.data import session_bounds
from hypnose_behavior.modelling.ab_learning.diagnostics import wilson_interval
from hypnose_behavior.modelling.ab_learning.sigmoid import (
    VARIANTS,
    erf_curve,
    fit_accuracy,
    fit_rate,
)
from hypnose_behavior.visualization.modelling.ab_learning._common import (
    BAND_ALPHA,
    BOUNDARY,
    REFERENCE,
    SECOND,
    SERIES,
    THIRD,
    annotation_size,
    figure,
    index_bounds,
    require_data,
    save_scope,
    session_axis,
    style_axis,
    title,
)

__all__ = ["plot_accuracy_sigmoid", "plot_sigmoid_anchors"]

# Rows per accuracy bin drawn behind the fits.
_BIN = 50

# lnL below the best at which the surface's colour saturates, and the 95% contour.
_FLOOR = -10.0
_CONTOUR = -1.92

# One-hue sequential ramp, surface-light to dark blue.
_SURFACE_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "ab_surface", ["#f4f8fd", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])

_VARIANT_STYLES = {
    "free": dict(color=SERIES, linewidth=2.2, zorder=4, label="initial free"),
    "chance": dict(color=SECOND, linewidth=2.0, linestyle=(0, (5, 2)), zorder=3,
                   label="initial 0.5"),
}
_OTHER_OPTIMA = dict(color=SERIES, linewidth=1.2, linestyle=":", zorder=2)
_DATA = dict(color=REFERENCE, markersize=5, zorder=1)
_CHANCE = dict(color=REFERENCE, linestyle="--", linewidth=1, zorder=0)

_ANCHOR_COLOURS = {"initiations": SERIES, "rewards": THIRD, "accuracy": SECOND}

_WIDTH = 10.0
_HEIGHT = 3.8


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


def _draw_fits(ax, fits, subjid, k):
    """Each variant's best curve, and the free fit's other optima, dotted."""
    for variant, style in _VARIANT_STYLES.items():
        fit = fits.get(variant, {}).get(subjid)
        if fit is None:
            continue
        optima = fit["optima"]
        best = optima.iloc[0]
        ax.plot(k, erf_curve(k, best["initial"], best["final"], best["center"],
                             best["width"]), **style)
        if variant != "free":
            continue
        ax.axvspan(best["center"] - best["width"], best["center"] + best["width"],
                   color=style["color"], alpha=BAND_ALPHA / 2, linewidth=0, zorder=0)
        for i, (_, other) in enumerate(optima.iloc[1:].iterrows()):
            ax.plot(k, erf_curve(k, other["initial"], other["final"], other["center"],
                                 other["width"]),
                    **{**_OTHER_OPTIMA, "label": "other optima" if i == 0 else None})


def _headline(fits, subjid):
    """The best free fit, and what fixing the initial accuracy at 0.5 costs in lnL."""
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
        ax.plot(x, y, marker="o", markersize=9, color=SECOND, markeredgecolor="white",
                markeredgewidth=1.5, linestyle="none", zorder=5)
        ax.annotate(str(int(optimum["rank"]) + 1), (x, y), xytext=(6, 4),
                    textcoords="offset points", fontsize=annotation_size(), zorder=6)
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
    fitted from the selection in ``mode``.

    - top: accuracy per 50 rows (grey, Wilson 95%), each variant's best curve, the free
      fit's other optima dotted, and its ``k_s +/- w`` shaded.
    - bottom: the free fit's lnL over (center, width), relative to its best grid point,
      with the 95% contour and the optima numbered by rank. Separate islands of dark
      blue are separate optima.
    """
    if fits is None:
        data = require_data(subjids, dates, selectors, data)
        fits = {variant: fit_accuracy(data, mode, variant) for variant in VARIANTS}
    reference = next(iter(fits.values()))

    figures = {}
    for subjid in sorted(reference):
        main = reference[subjid]
        rows, mode = main["rows"], main["mode"]
        k = rows["k"].to_numpy(dtype=float)
        spans = index_bounds(rows)
        unit = "trials" if mode == "completed" else "choice attempts"

        fig, (top, bottom) = figure(2, width=_WIDTH, height=_HEIGHT)
        bottom.sharex(top)
        bins = _binned(rows)
        spread = np.clip([bins["p"] - bins["lo"], bins["hi"] - bins["p"]], 0, None)
        top.errorbar(bins["k"], bins["p"], yerr=spread, fmt="o", elinewidth=1, capsize=0,
                     label=f"per {_BIN}", **_DATA)
        _draw_fits(top, fits, subjid, k)
        top.axhline(0.5, **_CHANCE)
        for ax in (top, bottom):
            for start in spans["start"].to_numpy()[1:]:
                ax.axvline(start, **BOUNDARY)
        style_axis(top, ylabel="accuracy", ylim=(0, 1.02))
        top.text(0.01, 0.03, _headline(fits, subjid), transform=top.transAxes,
                 fontsize=annotation_size(), va="bottom", ha="left")
        top.legend(frameon=False, fontsize=annotation_size(), loc="lower right")
        top.set_xlim(0, len(rows))
        session_axis(top, spans)

        free = fits.get("free", {}).get(subjid)
        if free is not None:
            _draw_surface(bottom, free)
        bottom.set_xlabel(unit)
        title(fig, f"accuracy sigmoid | {mode}", subjid)
        _save(fig, f"ab_learning_accuracy_sigmoid_{mode}_sub-{subjid:03d}", rows, save,
              subjid)
        figures[int(subjid)] = fig
    return figures


def _session_rates(data, spans, subjid):
    """Initiations and rewards per task hour, per session."""
    trials = data["trials"][data["trials"]["subjid"] == subjid]
    counts = trials.groupby("session_idx").agg(
        initiations=("outcome", "size"),
        rewards=("outcome", lambda o: int((o == "rewarded").sum())))
    out = spans.set_index("session_idx").join(counts).fillna(
        {"initiations": 0, "rewards": 0})
    hours = (out["end"] - out["start"]).where(lambda h: h > 0)
    return out.assign(initiations=out["initiations"] / hours,
                      rewards=out["rewards"] / hours).reset_index()


def _session_accuracy(rows, spans):
    """Accuracy per session of the fitted rows, over each session's task-time span."""
    accuracy = rows.groupby("session_idx")["y"].mean().rename("accuracy")
    return spans.set_index("session_idx").join(accuracy, how="inner").reset_index()


def _segments(ax, frame, column, colour):
    """One horizontal segment per session, over its task-time span."""
    ax.hlines(frame[column], frame["start"], frame["end"], color=colour, linewidth=4,
              alpha=0.35, zorder=1)


def plot_sigmoid_anchors(subjids=None, dates=None, *, data=None, accuracy=None,
                         engagement=None, rewards=None, save=False, **selectors):
    """Where each fitted change falls on task time, one figure per animal.

        plot_sigmoid_anchors(data=ab, accuracy=acc["free"], engagement=eng, rewards=rew)

    ``accuracy`` is any `fit_accuracy` result (completed, initial free when omitted);
    ``engagement`` / ``rewards`` are `fit_rate` results, fitted when omitted.

    - top: initiations and rewards per task hour, per session (faded bars), with the
      rate fits' best curves.
    - bottom: accuracy per session, with the accuracy curve read onto task time through
      each row's timestamp.
    - dashed verticals: each fit's midpoint, in its own colour, on both panels.

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
        acc = accuracy[subjid]
        fig, (top, bottom) = figure(2, width=_WIDTH, height=_HEIGHT)
        bottom.sharex(top)
        span = float(spans["end"].max())
        clock = np.linspace(0, span, 600)

        rates = _session_rates(data, spans, subjid)
        anchors = {}
        for name, fits in (("initiations", engagement), ("rewards", rewards)):
            colour = _ANCHOR_COLOURS[name]
            best = fits[subjid]["optima"].iloc[0]
            _segments(top, rates, name, colour)
            top.plot(clock, erf_curve(clock, best["initial"], best["final"], best["center"],
                                      best["width"]), color=colour, linewidth=2.2,
                     label=name, zorder=3)
            anchors[name] = best["center"]

        rows = acc["rows"]
        _segments(bottom, _session_accuracy(rows, spans), "accuracy", SECOND)
        best = acc["optima"].iloc[0]
        k = rows["k"].to_numpy(dtype=float)
        bottom.plot(rows["hours"], erf_curve(k, best["initial"], best["final"],
                                             best["center"], best["width"]),
                    color=SECOND, linewidth=2.2, zorder=3,
                    label=f"accuracy ({acc['mode']}, initial {acc['variant']})")
        bottom.axhline(0.5, **_CHANCE)
        anchors["accuracy"] = best["center_hours"]

        for ax in (top, bottom):
            for start in spans["start"].to_numpy()[1:]:
                ax.axvline(start, **BOUNDARY)
            for name, hours in anchors.items():
                ax.axvline(hours, color=_ANCHOR_COLOURS[name], linestyle="--",
                           linewidth=1.4, zorder=2)
        style_axis(top, ylabel="per hour", ylim=None)
        top.set_ylim(bottom=0)
        style_axis(bottom, ylabel="accuracy", ylim=(0, 1.02))
        top.legend(frameon=False, fontsize=annotation_size(), loc="upper left")
        bottom.legend(frameon=False, fontsize=annotation_size(), loc="lower right")
        top.text(0.99, 0.97, "midpoints (h): " + ", ".join(
            f"{name} {hours:.1f}" for name, hours in anchors.items()),
            transform=top.transAxes, fontsize=annotation_size(), ha="right", va="top")
        top.set_xlim(0, span)
        session_axis(top, spans)
        bottom.set_xlabel("task time (h)")
        title(fig, "sigmoid midpoints on task time", subjid)
        _save(fig, f"ab_learning_sigmoid_anchors_sub-{subjid:03d}", data["sessions"], save,
              subjid)
        figures[int(subjid)] = fig
    return figures
