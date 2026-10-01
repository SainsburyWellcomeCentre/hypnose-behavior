"""Figures for the step model (``modelling.ab_learning.step``).

`plot_step_posterior` takes the step fits of `step.fit_step` and draws **one figure per
animal**. Returns ``{subjid: Figure}``; ``save=True`` files each figure under its own
animal with `io.save.save_figure`.
"""
from __future__ import annotations

import numpy as np

from hypnose_behavior.io.save import save_figure
from hypnose_behavior.modelling.ab_learning.sigmoid import erf_curve
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
    save_scope,
    style_axis,
)

__all__ = ["plot_step_posterior"]

_OBSERVED = dict(color=SECOND, linewidth=1.8, zorder=1, label="observed")
_STEP = dict(color=SERIES, linewidth=2.2, zorder=4, label="step at the peak")
_SIGMOID = dict(color=REFERENCE, linewidth=2.0, linestyle=(0, (5, 2)), zorder=3,
                label="sigmoid")
_MODE = dict(marker="o", markersize=8, color=SECOND, markeredgecolor="white",
             markeredgewidth=1.5, linestyle="none", zorder=6)

# The zoomed panel spans the posterior's central mass, padded by this share of its width.
_ZOOM_MASS = (0.001, 0.999)
_ZOOM_PAD = 0.25

# A session is named in the zoomed panel when it fills at least this share of it.
_LABEL_SHARE = 0.08

_WIDTH = 10.0
_HEIGHT = 3.4


def _save(fig, name, rows, save, subjid):
    if not save:
        return
    save_figure(fig, name, **save_scope(rows, subjid))


def _headline(summary, modes):
    """What the step says, and whether it may be read."""
    if summary["step"]:
        verdict = f"step adequate (sigmoid ΔlnL {summary['ΔlnL_vs_sigmoid']:.1f})"
    else:
        verdict = (f"gradual (sigmoid ΔlnL {summary['ΔlnL_vs_sigmoid']:.1f}): "
                   "posterior not a result")
    lines = [verdict]
    if summary["multimodal"]:
        lines.append("multimodal: " + ", ".join(
            f"{int(m.k_s)} (ses {m.ses}, {m.mass:.0%})" for m in modes.itertuples()))
    else:
        lines.append(f"k_s {summary['k_s_mean']:.0f} ± {summary['k_s_sd']:.0f} "
                     f"(ses {summary['ses']})")
    lines.append(f"95% HDI {summary['hdi_lo']}–{summary['hdi_hi']}")
    lines.append(f"vs constant: sigmoid ΔlnL {summary['ΔlnL_sigmoid_const']:.1f}, "
                 f"step {summary['ΔlnL_step_const']:.1f}")
    return "\n".join(lines)


def _draw_fit(ax, step):
    """Observed excess correct, the step at the posterior's peak, and the sigmoid."""
    rows, summary, sigmoid = step["rows"], step["summary"], step["sigmoid"]
    k = rows["k"].to_numpy(dtype=float)
    count = np.r_[0, k + 1]
    ax.plot(count, np.r_[0.0, np.cumsum(rows["y"].to_numpy() - 0.5)],
            drawstyle="steps-post", **_OBSERVED)
    p_step = np.where(k < summary["k_s"], summary["initial"], summary["final"])
    ax.plot(count, np.r_[0.0, np.cumsum(p_step - 0.5)], **_STEP)
    p_sigmoid = erf_curve(k, sigmoid["initial"], sigmoid["final"], sigmoid["center"],
                          sigmoid["width"])
    ax.plot(count, np.r_[0.0, np.cumsum(p_sigmoid - 0.5)], **_SIGMOID)
    ax.axhline(0, **ZERO)
    style_axis(ax, ylabel="Excess correct", ylim=None)
    ax.legend(frameon=False, fontsize=annotation_size(), loc="lower right")


def _draw_posterior(ax, step, zoom=False):
    """P(k_s) with its 95% HDI shaded and the modes marked."""
    tau, post, summary = step["tau"], step["posterior"], step["summary"]
    ax.fill_between(tau, post, step="mid", color=SERIES, alpha=0.35, linewidth=0)
    ax.plot(tau, post, drawstyle="steps-mid", color=SERIES, linewidth=1.2)
    ax.axvspan(summary["hdi_lo"], summary["hdi_hi"], color=SERIES, alpha=BAND_ALPHA / 2,
               linewidth=0, zorder=0)
    peaks = step["modes"]
    heights = np.interp(peaks["k_s"], tau, post)
    ax.plot(peaks["k_s"], heights, **_MODE)
    if zoom:
        for m, h in zip(peaks.itertuples(), heights):
            ax.annotate(f"{m.mass:.0%}", (m.k_s, h), xytext=(5, 3),
                        textcoords="offset points", fontsize=annotation_size(), zorder=7)
        cumulative = np.cumsum(post)
        lo, hi = (tau[np.searchsorted(cumulative, q)] for q in _ZOOM_MASS)
        pad = max((hi - lo) * _ZOOM_PAD, 5)
        ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(bottom=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_ylabel("P(k_s)")


def plot_step_posterior(steps: dict, *, save=False):
    """The step model per animal: its fit on the data and the posterior over k_s.

        plot_step_posterior(fit_step(acc_best))

    - top: observed excess correct (orange), the step at the posterior's peak (blue) and
      the sigmoid (dashed grey), as expected running sums.
    - middle: the posterior over k_s across the whole sequence, 95% HDI shaded, modes as
      dots; nothing is placed within ``EDGE`` rows of either end.
    - bottom: the same posterior zoomed onto its mass, each mode labelled with its share.

    The headline says whether the step is an adequate description at all (`step_test`);
    for a gradual change the posterior is drawn but is not a result.
    """
    figures = {}
    for subjid, step in steps.items():
        rows, mode = step["rows"], step["mode"]
        spans = index_bounds(rows)
        fig, (fit_ax, whole, zoom) = figure(3, width=_WIDTH, height=_HEIGHT)
        whole.sharex(fit_ax)
        _draw_fit(fit_ax, step)
        _draw_posterior(whole, step)
        _draw_posterior(zoom, step, zoom=True)
        for ax in (fit_ax, whole, zoom):
            for start in spans["start"].to_numpy()[1:]:
                ax.axvline(start, **BOUNDARY)
        fit_ax.text(0.01, 0.97, _headline(step["summary"], step["modes"]),
                    transform=fit_ax.transAxes, fontsize=annotation_size(), va="top",
                    ha="left")
        fit_ax.set_xlim(0, len(rows))
        unit = "trials" if mode == "completed" else "choice attempts"
        zoom.set_xlabel(unit.capitalize())
        left, right = zoom.get_xlim()
        shown = (spans[["start", "end"]].clip(left, right).diff(axis=1)["end"]
                 >= _LABEL_SHARE * (right - left))
        for s in spans[shown].itertuples():
            zoom.annotate(f"ses {s.ses}", (max(s.start, left), 1.0),
                          xycoords=("data", "axes fraction"), xytext=(3, -2),
                          textcoords="offset points", va="top",
                          fontsize=annotation_size(), color=REFERENCE)
        _save(fig, f"ab_learning_step_posterior_{mode}_sub-{subjid:03d}", rows, save,
              subjid)
        figures[int(subjid)] = fig
    return figures
