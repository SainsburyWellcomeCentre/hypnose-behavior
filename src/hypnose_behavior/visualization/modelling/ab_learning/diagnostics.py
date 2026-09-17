"""Figures for the A/B learning diagnostics (``modelling.ab_learning.diagnostics``).

Each ``plot_*`` takes the same subject/session selection as `load_ab_data` -- or the
already-loaded frames as ``data=`` -- and draws **one figure per animal**, x = the animal's
odour-discrimination sessions in order, labelled by session number. Every proportion is
drawn with its 95% Wilson interval. Returns ``{subjid: Figure}``; ``save=True`` files each
figure under its own animal with `io.save.save_figure`.
"""
from __future__ import annotations

import math

import matplotlib.pyplot as plt
import numpy as np

from hypnose_behavior.io.save import save_figure
from hypnose_behavior.modelling.ab_learning.data import load_ab_data
from hypnose_behavior.modelling.ab_learning.diagnostics import (
    PRIOR_GROUPS,
    failed_attempt_accuracy,
    initiation_rate,
    lose_shift_summary,
)

__all__ = ["plot_failed_attempt_accuracy", "plot_initiation_rate", "plot_lose_shift"]

# Fixed categorical order: slot 1 blue, slot 2 orange, slot 3 aqua.
_SOURCE_STYLE = {
    "completed": ("#2a78d6", "completed trials"),
    "failed": ("#eb6834", "failed attempts"),
    "zero_poke": ("#1baf7a", "failed attempts, no poke"),
}
_SERIES = "#2a78d6"
_REFERENCE = "#8a8984"
_BAND_ALPHA = 0.18
_MAX_SESSION_TICKS = 6
_GROUP_LABELS = {"none": "none", "no_visit": "no visit", "wrong_port": "wrong\nport",
                 "correct_port": "correct\nport"}


def _text_size() -> float:
    """Titles and legends follow the active style's tick-label size."""
    return plt.rcParams["xtick.labelsize"] if isinstance(
        plt.rcParams["xtick.labelsize"], (int, float)) else 10


def _data(subjids, dates, selectors, data):
    if data is None:
        data = load_ab_data(subjids, dates, **selectors)
    if data["sessions"].empty:
        raise ValueError("No odour-discrimination session matched the selection; check the "
                         "subjects, the selectors and that the data location is reachable.")
    return data


def _figure(n_rows: int = 1, *, width: float = 6.4, height: float = 4.2):
    fig, axes = plt.subplots(n_rows, 1, figsize=(width, height * n_rows), squeeze=False,
                             constrained_layout=True)
    return fig, axes[:, 0]


def _style_axis(ax, *, ylabel=None, ylim=(0, 1.02)):
    ax.set_ylim(*ylim)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#e5e4df", linewidth=0.6)
    ax.set_axisbelow(True)
    if ylabel:
        ax.set_ylabel(ylabel)


def _session_ticks(ax, frame):
    sessions = frame.drop_duplicates("session_idx").sort_values("session_idx")
    sessions = sessions.iloc[::math.ceil(len(sessions) / _MAX_SESSION_TICKS)]
    ax.set_xticks(sessions["session_idx"])
    ax.set_xticklabels(sessions["ses"].astype(str))
    ax.set_xlabel("session")


def _title(fig, what: str, subjid=None):
    who = "pooled" if subjid is None else f"sub-{subjid:03d}"
    fig.suptitle(f"{who} | {what}", x=0.01, ha="left", fontsize=_text_size())


def _line_with_band(ax, frame, x, y, lo, hi, color, label=None):
    frame = frame[frame[y].notna()].sort_values(x)
    if frame.empty:
        return
    ax.fill_between(frame[x], frame[lo], frame[hi], color=color, alpha=_BAND_ALPHA,
                    linewidth=0)
    ax.plot(frame[x], frame[y], color=color, linewidth=2, marker="o", markersize=5,
            markeredgecolor="white", markeredgewidth=1, label=label)


def _save(fig, name, data, save, subjid=None):
    if not save:
        return
    sessions = data["sessions"]
    if subjid is not None:
        sessions = sessions[sessions["subjid"] == subjid]
    save_figure(fig, name, subjids=sorted(sessions["subjid"].unique().tolist()),
                dates=sorted(sessions["date"].unique().tolist()))


def plot_initiation_rate(subjids=None, dates=None, *, data=None, save=False, **selectors):
    """D1: per session, the share of sampling attempts that became a trial.

    Denominator: trials + failed attempts with a poke. Zero-poke attempts are left out.
    """
    data = _data(subjids, dates, selectors, data)
    rate = initiation_rate(data)
    figures = {}
    for subjid in sorted(rate["subjid"].unique()):
        frame = rate[rate["subjid"] == subjid]
        fig, (ax,) = _figure()
        _line_with_band(ax, frame, "session_idx", "rate", "rate_lo", "rate_hi", _SERIES)
        _style_axis(ax, ylabel="initiation rate")
        _session_ticks(ax, frame)
        _title(fig, "initiated trials / attempts with a poke", subjid)
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
    data = _data(subjids, dates, selectors, data)
    acc = failed_attempt_accuracy(data)
    figures = {}
    for subjid in sorted(acc["subjid"].unique()):
        frame = acc[acc["subjid"] == subjid]
        fig, (top, bottom) = _figure(n_rows=2, height=3.6)
        top.sharex(bottom)
        for source, (color, label) in _SOURCE_STYLE.items():
            part = frame[frame["source"] == source]
            _line_with_band(top, part, "session_idx", "accuracy", "accuracy_lo",
                            "accuracy_hi", color, label)
            if source != "completed":
                bottom.plot(part["session_idx"], part["visit_rate"], color=color,
                            linewidth=2, marker="o", markersize=5, markeredgecolor="white",
                            markeredgewidth=1, label=label)
        top.axhline(0.5, color=_REFERENCE, linestyle="--", linewidth=1)
        _style_axis(top, ylabel="accuracy")
        _style_axis(bottom, ylabel="port visit rate")
        top.tick_params(labelbottom=False)
        _session_ticks(bottom, frame)
        fig.legend(*top.get_legend_handles_labels(), loc="outside lower center", ncols=2,
                   frameon=False, fontsize=_text_size() * 0.7)
        _title(fig, "accuracy after failed attempts", subjid)
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
    data = _data(subjids, dates, selectors, data)
    per_animal = lose_shift_summary(data, by=("subjid",))
    panels = [(s, per_animal[per_animal["subjid"] == s])
              for s in sorted(per_animal["subjid"].unique())]
    panels.append((None, lose_shift_summary(data, by=())))
    groups = list(PRIOR_GROUPS)
    x = np.arange(len(groups))

    figures = {}
    for subjid, frame in panels:
        frame = frame.set_index("prior").reindex(groups)
        fig, (ax,) = _figure(width=5.6, height=4.6)
        err = np.vstack([frame["accuracy"] - frame["accuracy_lo"],
                         frame["accuracy_hi"] - frame["accuracy"]])
        ax.errorbar(x, frame["accuracy"], yerr=err, fmt="o", color=_SERIES, markersize=7,
                    markeredgecolor="white", markeredgewidth=1, elinewidth=2, capsize=0)
        ax.axhline(0.5, color=_REFERENCE, linestyle="--", linewidth=1)
        _style_axis(ax, ylabel="accuracy")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{_GROUP_LABELS[g]}\n{int(n)}" for g, n in zip(groups, frame["n"])])
        ax.set_xlim(-0.5, len(groups) - 0.5)
        ax.set_xlabel("attempt before (n trials)")
        _title(fig, "accuracy by prior attempt", subjid)
        name = ("ab_learning_lose_shift_pooled" if subjid is None
                else f"ab_learning_lose_shift_sub-{subjid:03d}")
        _save(fig, name, data, save, subjid)
        figures["pooled" if subjid is None else subjid] = fig
    return figures
