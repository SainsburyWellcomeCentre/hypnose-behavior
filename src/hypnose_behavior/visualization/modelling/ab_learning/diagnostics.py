"""Figures for the A/B learning diagnostics (``modelling.ab_learning.diagnostics``).

Each ``plot_*`` takes the same subject/session selection as `load_ab_data` -- or the
already-loaded frames as ``data=`` -- and draws **one figure per animal**, x = the animal's
odour-discrimination sessions in order, labelled by session number. Every proportion is
drawn with its 95% Wilson interval. Returns ``{subjid: Figure}``; ``save=True`` files each
figure under its own animal with `io.save.save_figure`.
"""
from __future__ import annotations

import numpy as np

from hypnose_behavior.io.save import save_figure
from hypnose_behavior.modelling.ab_learning.diagnostics import (
    PRIOR_GROUPS,
    failed_attempt_accuracy,
    initiation_rate,
    lose_shift_summary,
)
from hypnose_behavior.visualization.modelling.ab_learning._common import (
    REFERENCE,
    SERIES,
    figure,
    line_with_band,
    require_data,
    save_scope,
    session_ticks,
    style_axis,
    text_size,
    title,
)

__all__ = ["plot_failed_attempt_accuracy", "plot_initiation_rate", "plot_lose_shift"]

# Fixed categorical order: slot 1 blue, slot 2 orange, slot 3 aqua.
_SOURCE_STYLE = {
    "completed": (SERIES, "completed trials"),
    "failed": ("#eb6834", "failed attempts"),
    "zero_poke": ("#1baf7a", "failed attempts, no poke"),
}
_GROUP_LABELS = {"none": "none", "no_visit": "no visit", "wrong_port": "wrong\nport",
                 "correct_port": "correct\nport"}


def _save(fig, name, data, save, subjid=None):
    if not save:
        return
    save_figure(fig, name, **save_scope(data["sessions"], subjid))


def plot_initiation_rate(subjids=None, dates=None, *, data=None, save=False, **selectors):
    """D1: per session, the share of sampling attempts that became a trial.

    Denominator: trials + failed attempts with a poke. Zero-poke attempts are left out.
    """
    data = require_data(subjids, dates, selectors, data)
    rate = initiation_rate(data)
    figures = {}
    for subjid in sorted(rate["subjid"].unique()):
        frame = rate[rate["subjid"] == subjid]
        fig, (ax,) = figure()
        line_with_band(ax, frame, "session_idx", "rate", "rate_lo", "rate_hi", SERIES)
        style_axis(ax, ylabel="initiation rate")
        session_ticks(ax, frame)
        title(fig, "initiated trials / attempts with a poke", subjid)
        _save(fig, f"ab_learning_initiation_rate_sub-{subjid:03d}", data, save, subjid)
        figures[subjid] = fig
    return figures


def plot_failed_attempt_accuracy(subjids=None, dates=None, *, data=None, save=False,
                                 **selectors):
    """D2: port-choice accuracy after failed attempts next to completed-trial accuracy.

    Top: accuracy of completed trials, of failed attempts with a poke that led to a port
    visit, and of zero-poke failed attempts that did. The dashed line is chance.
    Bottom: the share of failed attempts followed by a port visit.
    """
    data = require_data(subjids, dates, selectors, data)
    acc = failed_attempt_accuracy(data)
    figures = {}
    for subjid in sorted(acc["subjid"].unique()):
        frame = acc[acc["subjid"] == subjid]
        fig, (top, bottom) = figure(n_rows=2, height=3.6)
        top.sharex(bottom)
        for source, (color, label) in _SOURCE_STYLE.items():
            part = frame[frame["source"] == source]
            line_with_band(top, part, "session_idx", "accuracy", "accuracy_lo",
                           "accuracy_hi", color, label)
            if source != "completed":
                bottom.plot(part["session_idx"], part["visit_rate"], color=color,
                            linewidth=2, marker="o", markersize=5, markeredgecolor="white",
                            markeredgewidth=1, label=label)
        top.axhline(0.5, color=REFERENCE, linestyle="--", linewidth=1)
        style_axis(top, ylabel="accuracy")
        style_axis(bottom, ylabel="port visit rate")
        top.tick_params(labelbottom=False)
        session_ticks(bottom, frame)
        fig.legend(*top.get_legend_handles_labels(), loc="outside lower center", ncols=2,
                   frameon=False, fontsize=text_size() * 0.7)
        title(fig, "accuracy after failed attempts", subjid)
        _save(fig, f"ab_learning_failed_attempt_accuracy_sub-{subjid:03d}", data, save,
              subjid)
        figures[subjid] = fig
    return figures


def plot_lose_shift(subjids=None, dates=None, *, data=None, save=False, **selectors):
    """D3: completed-trial accuracy by the failed attempt right before the trial.

    One figure per animal plus one pooled over them, keyed ``"pooled"``; each group is
    labelled with its trial count. Elimination would show as ``wrong port`` above
    ``none``; the dashed line is chance.
    """
    data = require_data(subjids, dates, selectors, data)
    per_animal = lose_shift_summary(data, by=("subjid",))
    panels = [(s, per_animal[per_animal["subjid"] == s])
              for s in sorted(per_animal["subjid"].unique())]
    panels.append((None, lose_shift_summary(data, by=())))
    groups = list(PRIOR_GROUPS)
    x = np.arange(len(groups))

    figures = {}
    for subjid, frame in panels:
        frame = frame.set_index("prior").reindex(groups)
        fig, (ax,) = figure(width=5.6, height=4.6)
        err = np.vstack([frame["accuracy"] - frame["accuracy_lo"],
                         frame["accuracy_hi"] - frame["accuracy"]])
        ax.errorbar(x, frame["accuracy"], yerr=err, fmt="o", color=SERIES, markersize=7,
                    markeredgecolor="white", markeredgewidth=1, elinewidth=2, capsize=0)
        ax.axhline(0.5, color=REFERENCE, linestyle="--", linewidth=1)
        style_axis(ax, ylabel="accuracy")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{_GROUP_LABELS[g]}\n{int(n)}" for g, n in zip(groups, frame["n"])])
        ax.set_xlim(-0.5, len(groups) - 0.5)
        ax.set_xlabel("attempt before (n trials)")
        title(fig, "accuracy by prior attempt", subjid)
        name = ("ab_learning_lose_shift_pooled" if subjid is None
                else f"ab_learning_lose_shift_sub-{subjid:03d}")
        _save(fig, name, data, save, subjid)
        figures["pooled" if subjid is None else subjid] = fig
    return figures
