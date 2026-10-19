"""Figures for the learning diagnostics (``modelling.discrimination_learning.diagnostics``).

Each ``plot_*`` takes the same subject/session selection as `load_discrimination_data` -- or the
already-loaded frames as ``data=`` -- and draws **one figure per animal**, x = the animal's
odour-discrimination sessions in order, labelled by session number. Every proportion is
drawn with its 95% Wilson interval. Returns ``{subjid: Figure}``; ``save=True`` files each
figure under its own animal with `io.save.save_figure`.

Every plotter takes ``legend`` and ``show`` (plotter convention,
`hypnose_helpers.viz.plotter`), and ``separate_odors``: each odour drawn in its reward
port's colour, with ``show_combined`` adding both together in grey.
"""
from __future__ import annotations

import numpy as np

from hypnose_behavior.io.save import finish_figure, legend_figure, save_figure, show_suffix
from hypnose_behavior.modelling.discrimination_learning.diagnostics import (
    PRIOR_GROUPS,
    failed_attempt_accuracy,
    initiation_rate,
    lose_shift_summary,
)
from hypnose_behavior.visualization.modelling.discrimination_learning._common import (
    SECOND,
    SERIES,
    ZERO,
    annotation_size,
    figure,
    line_with_band,
    odor_parts,
    odors_tag,
    require_data,
    save_scope,
    session_ticks,
    style_axis,
    text_size,
)

__all__ = ["plot_failed_attempt_accuracy", "plot_initiation_rate", "plot_lose_shift"]

# Fixed categorical order: slot 1 blue, slot 2 orange, slot 3 aqua.
_SOURCE_STYLE = {
    "completed": (SERIES, "completed trials"),
    "failed": (SECOND, "failed attempts"),
    "zero_poke": ("#1baf7a", "failed attempts, no poke"),
}
# Per odour the colour is the odour's, so the sources part by line and marker.
_SOURCE_LINES = {
    "completed": ("-", "o"),
    "failed": ((0, (4, 2)), "s"),
    "zero_poke": ((0, (1, 1.5)), "^"),
}
_GROUP_LABELS = {"none": "none", "no_visit": "no visit", "wrong_port": "wrong\nport",
                 "correct_port": "correct\nport"}
# Per odour, each group's points sit this far apart on the x axis.
_ODOR_SPACING = 0.2


def _save(fig, name, data, save, subjid=None):
    if not save:
        return
    save_figure(fig, name, **save_scope(data["sessions"], subjid))


def plot_initiation_rate(subjids=None, dates=None, *, data=None, separate_odors=False,
                         show_combined=False, legend=None, show=None, save=False,
                         **selectors):
    """D1: per session, the share of sampling attempts that became a trial.

    Denominator: trials + failed attempts with a poke. Zero-poke attempts are left out.

    ``separate_odors`` draws each odour's rate (its trials over its attempts) in its port
    colour, with a legend; ``show_combined`` adds the pooled rate in grey. ``show``
    numbers the odours, then both.
    """
    data = require_data(subjids, dates, selectors, data)
    parts = [(label, colour, initiation_rate(part))
             for label, colour, part in odor_parts(data, separate_odors, show_combined)]
    figures, entries = {}, []
    for subjid in sorted(set().union(*(rate["subjid"].unique() for *_, rate in parts))):
        fig, (ax,) = figure()
        for label, colour, rate in parts:
            frame = rate[rate["subjid"] == subjid]
            line_with_band(ax, frame, "session_idx", "rate", "rate_lo", "rate_hi",
                           colour or SERIES, label)
        style_axis(ax, ylabel="Initiation rate")
        session_ticks(ax, frame)
        if separate_odors:
            ax.legend(frameon=False, fontsize=annotation_size(), loc="lower right")
        entries += finish_figure(fig, legend, show)
        _save(fig, f"discrimination_learning_initiation_rate"
                   f"{odors_tag(separate_odors, show_combined)}{show_suffix(show)}"
                   f"_sub-{subjid:03d}", data, save, subjid)
        figures[subjid] = fig
    legend_figure(entries, fontsize=annotation_size())
    return figures


def plot_failed_attempt_accuracy(subjids=None, dates=None, *, data=None, separate_odors=False,
                                 show_combined=False, legend=None, show=None, save=False,
                                 **selectors):
    """D2: port-choice accuracy after failed attempts next to completed-trial accuracy.

    Top: accuracy of completed trials, of failed attempts with a poke that led to a port
    visit, and of zero-poke failed attempts that did. The grey line is chance.
    Bottom: the share of failed attempts followed by a port visit.

    ``show`` numbers: 1 completed trials, 2 failed attempts, 3 failed attempts with no
    poke; each hides in both panels, with its band.

    ``separate_odors`` draws each odour in its port colour, the sources told apart by line
    and marker (completed solid circles, failed dashed squares, no poke dotted triangles);
    ``show_combined`` adds both odours together in grey. ``show`` then numbers the first
    odour's three, then the next's.
    """
    data = require_data(subjids, dates, selectors, data)
    parts = [(label, colour, failed_attempt_accuracy(part))
             for label, colour, part in odor_parts(data, separate_odors, show_combined)]
    figures, entries = {}, []
    for subjid in sorted(set().union(*(acc["subjid"].unique() for *_, acc in parts))):
        fig, (top, bottom) = figure(n_rows=2, height=3.6)
        top.sharex(bottom)
        for name, colour, acc in parts:
            frame = acc[acc["subjid"] == subjid]
            for source, (color, label) in _SOURCE_STYLE.items():
                part = frame[frame["source"] == source]
                linestyle, marker = _SOURCE_LINES[source] if name else ("-", "o")
                if name is not None:
                    color, label = colour, f"{name} {label}"
                line_with_band(top, part, "session_idx", "accuracy", "accuracy_lo",
                               "accuracy_hi", color, label, linestyle=linestyle,
                               marker=marker)
                if source != "completed":
                    bottom.plot(part["session_idx"], part["visit_rate"], color=color,
                                linewidth=2, linestyle=linestyle, marker=marker, markersize=5,
                                markeredgecolor="white", markeredgewidth=1, label=label)
        top.axhline(0.5, **ZERO)
        style_axis(top, ylabel="Accuracy")
        style_axis(bottom, ylabel="Port visit rate")
        top.tick_params(labelbottom=False)
        session_ticks(bottom, frame)
        fig.legend(*top.get_legend_handles_labels(), loc="outside lower center",
                   ncols=len(parts) if separate_odors else 2, frameon=False,
                   fontsize=text_size() * 0.7)
        entries += finish_figure(fig, legend, show)
        _save(fig, f"discrimination_learning_failed_attempt_accuracy"
                   f"{odors_tag(separate_odors, show_combined)}{show_suffix(show)}"
                   f"_sub-{subjid:03d}", data, save, subjid)
        figures[subjid] = fig
    legend_figure(entries, fontsize=text_size() * 0.7)
    return figures


def plot_lose_shift(subjids=None, dates=None, *, data=None, separate_odors=False,
                    show_combined=False, legend=None, show=None, save=False, **selectors):
    """D3: completed-trial accuracy by the failed attempt right before the trial.

    One figure per animal plus one pooled over them, keyed ``"pooled"``; each group is
    labelled with its trial count. Elimination would show as ``wrong port`` above
    ``none``; the grey line is chance.

    No legend: the x axis names the groups, so ``legend`` changes nothing. ``show``
    picks groups, numbered left to right: 1 none, 2 no visit, 3 wrong port, 4 correct port.

    ``separate_odors`` puts each odour's point beside the other's in its port colour, the
    counts under each group in the same order, with a legend; ``show_combined`` adds both
    odours together in grey. ``show`` then picks odours instead of groups.
    """
    data = require_data(subjids, dates, selectors, data)
    parts = odor_parts(data, separate_odors, show_combined)
    summaries = [(label, colour, lose_shift_summary(part, by=("subjid",)),
                  lose_shift_summary(part, by=())) for label, colour, part in parts]
    subjects = sorted(set().union(*(s["subjid"].unique() for _, _, s, _ in summaries)))
    groups = list(PRIOR_GROUPS)
    x = np.arange(len(groups))
    offsets = (np.arange(len(parts)) - (len(parts) - 1) / 2) * _ODOR_SPACING

    figures = {}
    for subjid in [*subjects, None]:
        fig, (ax,) = figure(width=5.6, height=4.6)
        counts = []
        for offset, (label, colour, per_animal, pooled) in zip(offsets, summaries):
            frame = pooled if subjid is None else per_animal[per_animal["subjid"] == subjid]
            frame = frame.set_index("prior").reindex(groups)
            counts.append(frame["n"])
            err = np.vstack([frame["accuracy"] - frame["accuracy_lo"],
                             frame["accuracy_hi"] - frame["accuracy"]])
            look = dict(fmt="o", color=colour or SERIES, markersize=7, markeredgecolor="white",
                        markeredgewidth=1, elinewidth=2, capsize=0)
            if label is not None:
                ax.errorbar(x + offset, frame["accuracy"], yerr=err, label=label, **look)
                continue
            # One errorbar per group, labelled, so ``show`` can pick groups.
            for i, group in enumerate(groups):
                ax.errorbar(x[i], frame["accuracy"].iloc[i], yerr=err[:, i:i + 1],
                            label=_GROUP_LABELS[group].replace("\n", " "), **look)
        ax.axhline(0.5, **ZERO)
        style_axis(ax, ylabel="Accuracy")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{_GROUP_LABELS[g]}\n" + " / ".join(str(int(n.iloc[i]))
                                                                 for n in counts)
                            for i, g in enumerate(groups)])
        ax.set_xlim(-0.5, len(groups) - 0.5)
        ax.set_xlabel("Attempt before (n trials)" if not separate_odors else
                      f"Attempt before (n trials: {' / '.join(p[0] for p in parts)})")
        if separate_odors:
            ax.legend(frameon=False, fontsize=annotation_size(), loc="lower right")
        finish_figure(fig, legend, show)
        tag = f"{odors_tag(separate_odors, show_combined)}{show_suffix(show)}"
        name = (f"discrimination_learning_lose_shift{tag}_pooled" if subjid is None
                else f"discrimination_learning_lose_shift{tag}_sub-{subjid:03d}")
        _save(fig, name, data, save, subjid)
        figures["pooled" if subjid is None else subjid] = fig
    return figures
