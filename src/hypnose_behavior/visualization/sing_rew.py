"""Visualizations for the single-rewarded sequence task.

In the single-reward protocol not every candidate sequence is rewarded at its
final position (see README, "Single-Reward Protocol / False Response
Information"). Completed non-rewarded ("no-go") sequences are scored via the
``false_response`` / ``fr_label`` / ``fr_window_latency_ms`` / ``fr_port`` columns,
analogous to the false-alarm columns used for aborted trials.

General plotting helpers (boxplots, rolling/daily summary lines, marker sizing,
session collection) come from the shared leaves ``visualization.prep`` and
``visualization.panels`` rather than from a sibling plotter module.
"""

from __future__ import annotations

import math
from typing import Iterable, Optional, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

from hypnose_behavior.io.save import save_figure
from hypnose_behavior.utils.helpers import session_selectors
from hypnose_behavior.io.loaders import _load_position_data
from hypnose_behavior.visualization.prep import (
    _collect_sessions,
    _count_to_marker_size,
    _load_sorted_session,
    _load_subject_trial_timeline,
    _nice_round,
    _parse_json_value,
    _summary_save_suffix,
)
from hypnose_behavior.visualization.panels import (
    _add_size_legend,
    _plot_summary_rolling,
    _plot_violins_with_stats,
)
from hypnose_behavior.visualization.primitives import mean_sem
from hypnose_behavior.metric_analysis.metrics.common import reduce_rate
from hypnose_behavior.metric_analysis.metrics.false_alarm import (
    false_response_ratio_contributions,
)
from hypnose_behavior.metric_analysis.metrics.timing import reward_delivery_latency
from hypnose_behavior.metric_analysis.sing_rew_metrics import (
    compute_sing_rew_metrics,
    compute_sing_rew_rates,
    _classify_trial,
)


def _normalize_fr_types(fr_types):
    """Coerce the ``fr_types`` argument to a set of fr_label strings (or None).

    ``None`` means "do not filter" — count every ``false_response == True`` trial
    regardless of its ``fr_label``.
    """
    if fr_types is None:
        return None
    if isinstance(fr_types, str):
        return {fr_types}
    return set(fr_types)


def _port_label(value):
    """Map a supply/reward port id (1 → A, 2 → B) to its letter, else None."""
    try:
        port_val = int(value)
    except (TypeError, ValueError):
        return None
    if port_val == 1:
        return "A"
    if port_val == 2:
        return "B"
    return None


def _subject_color_map(labels):
    """Assign a stable color per subject label using the tab10 colormap."""
    cmap = plt.get_cmap("tab10")
    return {label: cmap(i % 10) for i, label in enumerate(labels)}


def _fr_mask(frame, fr_labels):
    """Boolean mask over ``frame`` selecting false-response trials.

    A trial is a false response iff ``false_response == True`` and (when
    ``fr_labels`` is not None) its ``fr_label`` is in ``fr_labels``. Sessions
    without the single-reward columns yield an all-False mask.
    """
    if "false_response" not in frame.columns:
        return pd.Series(False, index=frame.index)
    mask = frame["false_response"] == True  # noqa: E712 (element-wise, NaN-safe)
    if fr_labels is not None and "fr_label" in frame.columns:
        mask = mask & frame["fr_label"].isin(fr_labels)
    return mask


def _trim_leading_empty(series):
    """Drop leading ``None`` (no-data) entries from a chronological session series,
    keeping middle gaps intact. Returns ([] when there is no data at all)."""
    first = next((i for i, v in enumerate(series) if v is not None), None)
    if first is None:
        return []
    return series[first:]


def _plot_fr_ratio_daily(series_by_subject, color_map, subj_order, ylabel, title):
    """Per-session FR ratio, one line per subject.

    ``series_by_subject`` maps each subject label to a chronological list of
    ``(ratio, n_completed)`` tuples (or ``None`` for a mid-run session with no
    data). Each subject's series has already had its leading empty sessions
    trimmed, so every subject starts at session 1. Mid-run gaps are left blank
    (the line breaks there). X ticks are kept sparse for long runs.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    max_len = 0
    all_counts = []
    for label in subj_order:
        series = series_by_subject.get(label)
        if not series:
            continue
        max_len = max(max_len, len(series))
        xs = list(range(1, len(series) + 1))
        ys = [np.nan if v is None else v[0] for v in series]
        color = color_map.get(label)
        # NaNs break the connecting line at mid-run gaps.
        ax.plot(xs, ys, color=color, linewidth=2, label=label)
        valid = [(x, v[0], v[1]) for x, v in zip(xs, series) if v is not None]
        if valid:
            vx = [x for x, _, _ in valid]
            vy = [y for _, y, _ in valid]
            counts = [c for _, _, c in valid]
            all_counts.extend(counts)
            ax.scatter(vx, vy, s=[_count_to_marker_size(c) for c in counts], color=color, zorder=3)

    ax.set_xlabel("Session")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(bottom=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if max_len <= 12:
        ax.set_xticks(range(1, max_len + 1))
    else:
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlim(left=0.5, right=max_len + 0.5)
    ax.legend(loc="best")
    _add_size_legend(ax, all_counts)
    fig.tight_layout()
    return fig


def FR_ratio(
    subjids: Optional[Iterable[int]] = None,
    dates: Optional[Union[Iterable[Union[int, str]], tuple]] = None,
    fr_types: Optional[Iterable[str]] = "FR_time_in",
    save: bool = False,
    moving_avg: bool = False,
    window_size: int = 10,
    step_size: int = 1,
    *,
    ses=None,
    index=None,
    date_range=None,
    ses_range=None,
    index_range=None,
):
    """Ratio of false-response trials to all completed trials.

    For every completed trial (``is_aborted == False``) a binary value is
    assigned: 1 if it is a false response (``false_response == True``, further
    filtered to ``fr_types`` matched against ``fr_label``; default
    ``"FR_time_in"`` only), else 0. The mean of this value is the
    ``#false_response / #completed`` ratio.

    All subjects are drawn on one figure, with one colored series per subject so
    matched subjects are connected across sessions. Each subject's leading empty
    sessions (no single-reward data) are trimmed so every subject starts at
    session 1 with its first data session; mid-run sessions with no data are left
    blank (the line breaks there).

    - ``moving_avg=False``: one point per subject per session (whole-session
      ratio); X axis is the ordinal session number, with sparse ticks for long
      runs.
    - ``moving_avg=True``: a rolling ratio over the last ``window_size`` completed
      trials (step ``step_size``) within each session; X axis is the continuous
      (across-session) trial id. Reuses the rolling moving-average plotter.

    ``ses`` / ``index`` / ``date_range`` / ``ses_range`` / ``index_range`` narrow the
    selection further; they intersect with ``dates`` and with each other, and ``index``
    is the subject's gap-free chronological rank (`utils.helpers.session_selectors`).
    """
    select = session_selectors(
        ses=ses, index=index, date_range=date_range,
        ses_range=ses_range, index_range=index_range,
    )
    fr_labels = _normalize_fr_types(fr_types)
    subjects = list(_collect_sessions(subjids, dates, **select))
    if not subjects:
        return []

    subj_labels = []
    all_subjids = []
    all_dates = []
    n_sessions_total = 0
    n_sessions_with_fr = 0
    # Per subject: chronological list of per-session entry-lists ([(trial_idx,
    # binary), ...]; empty list = session with no usable data).
    per_subject_sessions = {}

    for subjid, date_vals, results_dirs in subjects:
        label = f"sub-{subjid}"
        subj_labels.append(label)
        all_subjids.append(subjid)
        all_dates.extend(date_vals)
        sess_entries = []
        for results_dir in results_dirs:
            n_sessions_total += 1
            df = _load_sorted_session(results_dir)
            entries = []
            if not df.empty:
                completed = df[df.get("is_aborted") == False]
                if not completed.empty and "false_response" in completed.columns:
                    n_sessions_with_fr += 1
                    # The metric's numerator contributions, one per completed
                    # trial -- the session ratio and the rolling ratio below are
                    # both means of exactly these.
                    num, _ = false_response_ratio_contributions(
                        completed, fr_types=fr_labels)
                    entries = [
                        (int(idx), float(v))
                        for idx, v in zip(completed["_trial_idx"], num)
                    ]
            sess_entries.append(entries)
        per_subject_sessions[label] = sess_entries

    if n_sessions_with_fr == 0:
        print(
            f"[FR_ratio] No single-reward data found: none of the {n_sessions_total} "
            f"loaded session(s) have a 'false_response' column. These columns are only "
            f"written for single-reward protocol sessions (re-run trial classification "
            f"if you expected them). Nothing to plot."
        )
        return []

    color_map = _subject_color_map(subj_labels)

    # Trim each subject's leading empty sessions so everyone starts at session 1;
    # mid-run empty sessions are kept as gaps (None).
    trimmed = {}
    max_len = 0
    for label in subj_labels:
        series = [e if e else None for e in per_subject_sessions[label]]
        series = _trim_leading_empty(series)
        if series:
            trimmed[label] = series
            max_len = max(max_len, len(series))

    if moving_avg:
        # Build session_data aligned by trimmed session index (subjects as groups)
        # and reuse the rolling plotter.
        session_data = [{"n_trials": 0, "groups": {}} for _ in range(max_len)]
        for label, series in trimmed.items():
            for j, entries in enumerate(series):
                if entries:
                    session_data[j]["groups"][label] = entries
        fig = _plot_summary_rolling(
            session_data,
            color_map=color_map,
            group_order=subj_labels,
            ylabel="FR Ratio",
            title="False response ratio (rolling)",
            window_size=window_size,
            step_size=step_size,
            ylim_bottom=0,
        )
    else:
        # Per-subject series of (ratio, n_completed) for the daily plot.
        series_by_subject = {}
        for label, series in trimmed.items():
            ratios = []
            for entries in series:
                if entries:
                    # The metric's own reduction of its contributions, not a
                    # second definition of "false responses over completed".
                    vals = [v for _, v in entries]
                    _, n_completed, rate = reduce_rate(vals, np.ones(len(vals)))
                    ratios.append((float(rate), n_completed))
                else:
                    ratios.append(None)
            series_by_subject[label] = ratios
        fig = _plot_fr_ratio_daily(
            series_by_subject,
            color_map,
            subj_labels,
            "FR Ratio",
            "False response ratio (session)",
        )

    figs = []
    if fig is not None:
        figs.append(fig)
        if save:
            suffix = _summary_save_suffix(moving_avg, window_size, step_size)
            save_figure(
                fig,
                f"FR_ratio_{suffix}",
                subjids=all_subjids,
                dates=all_dates,
            )
    return figs


def _plot_latency_box_AB(data, y_label, title):
    """Grouped boxplot (mean ± SEM with jittered points) split by reward port.

    ``data`` maps each group label ("False Response", "Reward") to
    ``{"A": [...], "B": [...]}``. Port A is drawn in red (left offset), port B in
    green (right offset), mirroring the A/B styling in ``fa_analysis``.
    """
    labels = list(data.keys())
    fig, ax = plt.subplots(figsize=(8, 5))
    mean_half_width = 0.14
    cap_half_width = 0.08
    mean_lw = 2.2
    sem_lw = 1.4
    point_offset = 0.18
    jitter_half_width = 0.12
    rng = np.random.default_rng(0)

    for i, label in enumerate(labels, start=1):
        for port_label, color, offset in (
            ("A", "red", -point_offset),
            ("B", "green", point_offset),
        ):
            values = data[label].get(port_label, [])
            if not values:
                continue
            x_pos = i + offset
            xs = x_pos + rng.uniform(-jitter_half_width, jitter_half_width, size=len(values))
            ax.scatter(xs, values, s=18, color=color, alpha=0.4, zorder=2)
            mean_val, sem_val = mean_sem(values)
            sem_val = 0.0 if np.isnan(sem_val) else sem_val
            ax.hlines(
                mean_val,
                x_pos - mean_half_width,
                x_pos + mean_half_width,
                colors="black",
                linewidth=mean_lw,
                zorder=3,
            )
            ax.vlines(
                x_pos,
                mean_val - sem_val,
                mean_val + sem_val,
                colors="black",
                linewidth=sem_lw,
                zorder=3,
            )
            if sem_val > 0:
                ax.hlines(
                    mean_val - sem_val,
                    x_pos - cap_half_width,
                    x_pos + cap_half_width,
                    colors="black",
                    linewidth=sem_lw,
                    zorder=3,
                )
                ax.hlines(
                    mean_val + sem_val,
                    x_pos - cap_half_width,
                    x_pos + cap_half_width,
                    colors="black",
                    linewidth=sem_lw,
                    zorder=3,
                )

    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel(y_label)
    ax.set_xlabel("")
    ax.set_ylim(bottom=0)
    ax.set_title(title)
    ax.legend(
        handles=[
            Patch(facecolor="red", edgecolor="black", label="Port A"),
            Patch(facecolor="green", edgecolor="black", label="Port B"),
        ],
        loc="upper right",
    )
    fig.tight_layout()
    return fig


def _session_fr_latencies(completed, fr_labels):
    """False-response latencies for one session: (all_values, {"A": [...], "B": [...]}).

    ``all_values`` includes every FR latency (even when the port is unknown); the
    per-port dict only includes trials with a resolvable ``fr_port``.
    """
    if "false_response" not in completed.columns:
        return [], {"A": [], "B": []}
    fr_df = completed[_fr_mask(completed, fr_labels)]
    all_vals = []
    by_port = {"A": [], "B": []}
    for _, row in fr_df.iterrows():
        lat = row.get("fr_window_latency_ms")
        if lat is None or pd.isna(lat):
            continue
        lat = float(lat)
        all_vals.append(lat)
        port = _port_label(row.get("fr_port"))
        if port is not None:
            by_port[port].append(lat)
    return all_vals, by_port


def _session_reward_rts(completed, results_dir):
    """Reward response times for one session: (all_values, {"A": [...], "B": [...]}).

    `reward_delivery_latency` is the same quantity `pred_seq_utils.response_time`
    draws, from one definition. It is **not** `trial_data.response_time_ms`, which is
    measured from the reward-port poke.
    """
    rew_df = completed[completed.get("response_time_category") == "rewarded"]
    latencies = reward_delivery_latency(rew_df, _load_position_data(results_dir, rew_df))
    all_vals = []
    by_port = {"A": [], "B": []}
    for _, row in rew_df.iterrows():
        rt = latencies.get(row.get("global_trial_id"))
        if rt is None or pd.isna(rt):
            continue
        rt = float(rt)
        all_vals.append(rt)
        port = _port_label(row.get("first_supply_port"))
        if port is not None:
            by_port[port].append(rt)
    return all_vals, by_port


def FR_latency(
    subjids: Optional[Iterable[int]] = None,
    dates: Optional[Union[Iterable[Union[int, str]], tuple]] = None,
    fr_types: Optional[Iterable[str]] = "FR_time_in",
    save: bool = False,
    split_AB: bool = False,
    *,
    ses=None,
    index=None,
    date_range=None,
    ses_range=None,
    index_range=None,
):
    """Joint boxplot of false-response latency vs. reward response time per subject.

    One figure per subject, pooling all selected days into a single boxplot with
    two groups on the X axis:

    - "False Response": ``fr_window_latency_ms`` of false-response trials
      (``false_response == True``, optionally filtered to ``fr_types`` via
      ``fr_label``; default ``"FR_time_in"`` only).
    - "Reward": response time of rewarded trials
      (``response_time_category == "rewarded"``), computed as in
      ``pred_seq_utils`` (last odor poke-out → first supply poke).

    Leading days are trimmed up to (and including) the first day on which FR data
    is available; days after that are all pooled, including mid-run days that
    happen to have no FR data (their reward data is still included).

    With ``split_AB=True`` each group is further split by port — false responses
    by ``fr_port`` and rewards by ``first_supply_port`` — into A (red) and B
    (green), still grouped by False Response vs. Reward.

    ``ses`` / ``index`` / ``date_range`` / ``ses_range`` / ``index_range`` narrow the
    selection further; they intersect with ``dates`` and with each other, and ``index``
    is the subject's gap-free chronological rank (`utils.helpers.session_selectors`).
    """
    select = session_selectors(
        ses=ses, index=index, date_range=date_range,
        ses_range=ses_range, index_range=index_range,
    )
    fr_labels = _normalize_fr_types(fr_types)
    figs = []
    n_subjects_with_fr = 0
    for subjid, date_vals, results_dirs in _collect_sessions(subjids, dates, **select):
        # Per-session extraction in chronological order.
        sessions = []
        for results_dir in results_dirs:
            df = _load_sorted_session(results_dir)
            if df.empty:
                sessions.append(None)
                continue
            completed = df[df.get("is_aborted") == False]
            if completed.empty:
                sessions.append(None)
                continue
            fr_vals, fr_port = _session_fr_latencies(completed, fr_labels)
            rew_vals, rew_port = _session_reward_rts(completed, results_dir)
            sessions.append(
                {"fr_vals": fr_vals, "fr_port": fr_port, "rew_vals": rew_vals, "rew_port": rew_port}
            )

        # First day with FR data available; trim everything before it.
        start = next(
            (i for i, s in enumerate(sessions) if s is not None and s["fr_vals"]),
            None,
        )
        if start is None:
            continue
        n_subjects_with_fr += 1

        fr_all, rew_all = [], []
        fr_AB = {"A": [], "B": []}
        rew_AB = {"A": [], "B": []}
        for s in sessions[start:]:
            if s is None:
                continue
            fr_all += s["fr_vals"]
            rew_all += s["rew_vals"]
            for p in ("A", "B"):
                fr_AB[p] += s["fr_port"][p]
                rew_AB[p] += s["rew_port"][p]

        used_dates = list(date_vals[start:])

        if not split_AB:
            if not fr_all and not rew_all:
                continue
            groups = {"False Response": fr_all, "Reward": rew_all}
            fig, ax = plt.subplots(figsize=(7, 5))
            _plot_violins_with_stats(ax, groups, "Latency (ms)", "")
            ax.set_ylim(bottom=0)
            ax.set_title(f"Subjid {subjid} FR vs reward latency")
            fig.tight_layout()
            figs.append(fig)
            if save:
                save_figure(fig, "FR_latency", subjids=[subjid], dates=used_dates)
        else:
            data = {"False Response": fr_AB, "Reward": rew_AB}
            if not any(vals for group in data.values() for vals in group.values()):
                continue
            fig = _plot_latency_box_AB(
                data,
                "Latency (ms)",
                f"Subjid {subjid} FR vs reward latency (A/B)",
            )
            figs.append(fig)
            if save:
                save_figure(fig, "FR_latency_AB", subjids=[subjid], dates=used_dates)

    if n_subjects_with_fr == 0:
        print(
            "[FR_latency] No single-reward data found: none of the loaded session(s) "
            "have false-response data. These columns are only written for single-reward "
            "protocol sessions (re-run trial classification if you expected them). "
            "Nothing to plot."
        )
    return figs


# =========================================================================
# Single-reward metric curves over sessions
# =========================================================================
#
# A single general plotter (`_plot_sing_rew_metrics`) draws one or more
# singrew metrics as session curves, with thin wrappers below for the
# pre-determined metric combinations.
#
# Conventions (shared by all wrappers):
#   - Input is a dict {subjid: dates}, so each subject can use its own date
#     range. Sessions are numbered per subject from 1 (the subject's first
#     session that contains trial data); the X axis is "Sessions", never
#     calendar dates.
#   - Metrics are distinguished by COLOR, subjects by MARKER.
#   - NaN within a subject's learning curve is meaningful: it is rendered as a
#     discontinuity (gap, no marker, no interpolation). A genuine 0 is plotted
#     as a dot at 0. Mid-run sessions with no data at all also become gaps.
#   - Y range is always explicit; the Y label defaults to the metric name with
#     "metric_name" -> "Metric Name".

# Subjects -> marker; metrics -> color.
_SUBJECT_MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*", "h", "<", ">", "p"]
_METRIC_PALETTE = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf", "#8c564b"]

# Nicer display labels than a plain title-case of the key.
_METRIC_PRETTY = {
    "fa_rate": "FA Rate",
    "headline_sensitivity": "d'",
    "ambiguous_rate": "Ambiguous Rate",
}


def _pretty_metric(name):
    return _METRIC_PRETTY.get(name, str(name).replace("_", " ").title())


def _isnan(v):
    try:
        return math.isnan(v)
    except (TypeError, ValueError):
        return False


def _metric_value(rates, key):
    """Resolve a metric value from a `compute_sing_rew_rates` result.

    Handles top-level rate keys and raw counts (under ``counts``). Missing -> NaN.
    **Look values up; do not derive one here** -- `compute_sing_rew_rates` owns the
    arithmetic, and a second derivation is how two figures come to disagree.
    """
    if rates is None:
        return float("nan")
    if key in rates and key != "counts":
        return rates[key]
    counts = rates.get("counts", {})
    if key in counts:
        return float(counts[key])
    return float("nan")


def _singrew_session_records(data):
    """For each subject in ``data`` ({subjid: dates}), build the per-session
    rates in chronological order with leading no-data sessions trimmed.

    Returns ``{subjid: {"rates": [rates_or_None, ...], "dates": [date, ...]}}``;
    each list element corresponds to session 1, 2, ... for that subject (None =
    a session with no trial data, kept only when it falls mid-run).
    """
    records = {}
    for subjid, dates in data.items():
        sessions = []  # (date_val, df_or_None) chronological
        for _sid, date_vals, results_dirs in _collect_sessions([subjid], dates):
            pairs = sorted(zip(date_vals, results_dirs), key=lambda p: str(p[0]))
            for date_val, results_dir in pairs:
                df = _load_sorted_session(results_dir)
                sessions.append((date_val, df if (df is not None and not df.empty) else None))
        first = next((i for i, (_d, df) in enumerate(sessions) if df is not None), None)
        if first is None:
            continue
        sessions = sessions[first:]
        rates_list, date_list = [], []
        for date_val, df in sessions:
            date_list.append(date_val)
            if df is None:
                rates_list.append(None)
            else:
                rates_list.append(compute_sing_rew_rates(compute_sing_rew_metrics({"trial_data": df})))
        records[subjid] = {"rates": rates_list, "dates": date_list}
    return records


def _size_legend_handles(counts, *, title_key="n"):
    counts = [c for c in counts if c and not _isnan(c) and c > 0]
    if not counts:
        return []
    cmin, cmax = min(counts), max(counts)
    raw = [cmin] if cmin == cmax else [cmin, (cmin + cmax) / 2.0, cmax]
    vals = sorted({_nice_round(v) for v in raw if v > 0})
    handles = []
    for v in vals:
        handles.append(
            Line2D(
                [], [], marker="o", linestyle="", color="#cccccc", markeredgecolor="#444444",
                markersize=float(np.sqrt(_count_to_marker_size(v))), label=f"{title_key}={v}",
            )
        )
    return handles


def _plot_sing_rew_metrics(
    data,
    metrics,
    *,
    ylabel=None,
    ylim=(0.0, 1.0),
    size_metric=None,
    ref_line=None,
    ref_line_label=None,
    emphasis_metrics=None,
    annotate_above=None,
    annotate_below=None,
    save=False,
    save_name=None,
):
    """Plot one or more singrew metrics as per-subject session curves.

    See the section comment above for conventions (dict input, per-subject
    session numbering, color=metric / marker=subject, NaN as discontinuity).

    Parameters
    ----------
    data : dict {subjid: dates}
    metrics : list[str]
        Metric keys (rate keys, counts, or derived ``ambiguous_rate``).
    ylabel : str, optional
        Explicit Y label (use "\\n" for multi-row). Defaults to the metric
        name(s) title-cased.
    ylim : tuple
        Explicit Y range.
    size_metric : str, optional
        Count key (e.g. "n_go") that scales marker area; adds a size legend.
    ref_line : float, optional
        Horizontal reference line (e.g. 0 for d' chance, criterion neutral).
    emphasis_metrics : set[str], optional
        Metrics drawn thicker and black (others thinner, colored).
    annotate_above / annotate_below : str, optional
        Text labels for the regions above / below ``ref_line``.
    """
    records = _singrew_session_records(data)
    if not records:
        print("[sing_rew] No single-reward session data found for the requested subjids/dates.")
        return []

    emphasis = set(emphasis_metrics or [])
    subjids = list(records.keys())

    colors = {}
    palette_i = 0
    for m in metrics:
        if m in emphasis:
            colors[m] = "black"
        else:
            colors[m] = _METRIC_PALETTE[palette_i % len(_METRIC_PALETTE)]
            palette_i += 1
    markers = {s: _SUBJECT_MARKERS[i % len(_SUBJECT_MARKERS)] for i, s in enumerate(subjids)}

    fig, ax = plt.subplots(figsize=(11, 5.5))
    max_len = 0
    size_counts = []

    for subjid in subjids:
        recs = records[subjid]["rates"]
        max_len = max(max_len, len(recs))
        xs = list(range(1, len(recs) + 1))
        for m in metrics:
            # Coerce to a float ndarray so any None/missing becomes a true np.nan;
            # this guarantees matplotlib breaks the line at gaps (no interpolation)
            # rather than relying on list->array coercion.
            ys = np.array([_metric_value(r, m) for r in recs], dtype=float)
            color = colors[m]
            lw = 2.6 if m in emphasis else 1.6
            ax.plot(xs, ys, color=color, linewidth=lw, zorder=2)
            valid = [(x, y, r) for x, y, r in zip(xs, ys, recs) if not _isnan(y)]
            if not valid:
                continue
            vx = [v[0] for v in valid]
            vy = [v[1] for v in valid]
            if size_metric:
                sizes = []
                for v in valid:
                    nval = _metric_value(v[2], size_metric)
                    size_counts.append(nval)
                    sizes.append(_count_to_marker_size(0 if _isnan(nval) else nval))
            else:
                sizes = 45
            ax.scatter(
                vx, vy, s=sizes, color=color, marker=markers[subjid],
                edgecolor="black", linewidth=0.4, zorder=3,
            )

    if ref_line is not None:
        ax.axhline(ref_line, color="gray", linestyle="--", linewidth=1.0, zorder=1)
        if ref_line_label:
            ax.text(0.6, ref_line, ref_line_label, color="gray", va="bottom", ha="left", fontsize=10)
    if annotate_above is not None:
        ax.text(0.99, 0.98, annotate_above, transform=ax.transAxes, ha="right", va="top",
                color="gray", fontsize=11)
    if annotate_below is not None:
        ax.text(0.99, 0.02, annotate_below, transform=ax.transAxes, ha="right", va="bottom",
                color="gray", fontsize=11)

    if ylabel is None:
        ylabel = " / ".join(_pretty_metric(m) for m in metrics)
    ax.set_xlabel("Sessions")
    ax.set_ylabel(ylabel)
    ax.set_ylim(*ylim)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if max_len <= 12:
        ax.set_xticks(range(1, max_len + 1))
    else:
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlim(0.5, max_len + 0.5)

    # Legends: metric color (only when >1 metric), subject marker (only when >1
    # subject), and dot-size (only when size_metric). Stack so they coexist.
    legend_specs = []
    if len(metrics) > 1:
        legend_specs.append((
            "Metric", "upper left",
            [Line2D([], [], color=colors[m], lw=(2.6 if m in emphasis else 1.6),
                    label=_pretty_metric(m)) for m in metrics],
        ))
    if len(subjids) > 1:
        legend_specs.append((
            "Subject", "upper right",
            [Line2D([], [], color="gray", marker=markers[s], linestyle="",
                    markeredgecolor="black", label=f"sub-{s}") for s in subjids],
        ))
    if size_metric:
        size_handles = _size_legend_handles(size_counts, title_key=size_metric)
        if size_handles:
            legend_specs.append((size_metric, "lower right", size_handles))

    created = []
    for title, loc, handles in legend_specs:
        created.append(ax.legend(handles=handles, loc=loc, title=title, framealpha=0.9))
    for leg in created[:-1]:
        ax.add_artist(leg)

    fig.tight_layout()
    figs = [fig]
    if save:
        save_figure(
            fig,
            save_name or "sing_rew_metric",
            subjids=list(data.keys()),
            dates=[d for rec in records.values() for d in rec["dates"]],
        )
    return figs


# ---- thin wrappers: pre-determined metric combinations -------------------

def dprime(data, save: bool = False):
    """Headline sensitivity (d') over sessions; dot size = n_go, 0 = chance line."""
    return _plot_sing_rew_metrics(
        data, ["headline_sensitivity"], ylabel="d'", ylim=(-1.0, 4.0),
        size_metric="n_go", ref_line=0.0, ref_line_label="chance",
        save=save, save_name="sing_rew_dprime",
    )


def hit_fa_rate(data, show_balanced_accuracy: bool = False, save: bool = False):
    """Hit rate and FA rate per subject; optionally overlay balanced_accuracy
    (thicker black line)."""
    metrics = ["hit_rate", "fa_rate"]
    emphasis = set()
    if show_balanced_accuracy:
        metrics.append("balanced_accuracy")
        emphasis.add("balanced_accuracy")
    return _plot_sing_rew_metrics(
        data, metrics, ylabel="Hit Rate /\nFalse Alarm Rate", ylim=(0.0, 1.0),
        emphasis_metrics=emphasis, save=save, save_name="sing_rew_hit_fa_rate",
    )


def criterion(data, save: bool = False):
    """Criterion (c) over sessions; neutral line at 0, conservative/liberal regions."""
    return _plot_sing_rew_metrics(
        data, ["criterion"], ylabel="Criterion (c)", ylim=(-3.0, 3.0),
        ref_line=0.0, annotate_above="conservative (+)", annotate_below="liberal (-)",
        save=save, save_name="sing_rew_criterion",
    )


def hit_earned_reward(data, show_port_accuracy: bool = False, save: bool = False):
    """Hit rate and earned-reward rate per subject; optionally add port_accuracy."""
    metrics = ["hit_rate", "earned_reward_rate"]
    if show_port_accuracy:
        metrics.append("port_accuracy")
    return _plot_sing_rew_metrics(
        data, metrics, ylim=(0.0, 1.0),
        save=save, save_name="sing_rew_hit_earned_reward",
    )


def early_rejection_anticipatory(data, save: bool = False):
    """Early rejection index and anticipatory rate per subject."""
    return _plot_sing_rew_metrics(
        data, ["early_rejection_index", "anticipatory_rate"],
        ylabel="Early Rejection Index /\nAnticipatory Rate", ylim=(0.0, 1.0),
        save=save, save_name="sing_rew_early_rejection_anticipatory",
    )


def efficient_early_rejection(data, save: bool = False):
    """Efficient rejection rate and early rejection index per subject."""
    return _plot_sing_rew_metrics(
        data, ["efficient_rejection_rate", "early_rejection_index"],
        ylabel="Efficient Rejection Rate /\nEarly Rejection Index", ylim=(0.0, 1.0),
        save=save, save_name="sing_rew_efficient_early_rejection",
    )


def premature_omission_rates(data, save: bool = False):
    """Impulsivity, impatience, omission, and ambiguous (n_amb/n_tot) rates."""
    return _plot_sing_rew_metrics(
        data, ["impulsivity_rate", "impatience_rate", "omission_rate", "ambiguous_rate"],
        ylabel="Rate", ylim=(0.0, 1.0),
        save=save, save_name="sing_rew_premature_omission_rates",
    )


def correct_rejection_rate(data, save: bool = False):
    """Correct rejection rate (CR / n_nogo) over sessions; rises with learning."""
    return _plot_sing_rew_metrics(
        data, ["correct_rejection_rate"], ylabel="Correct Rejection Rate", ylim=(0.0, 1.0),
        save=save, save_name="sing_rew_correct_rejection_rate",
    )


# =========================================================================
# Outcome composition (stacked) over sessions
# =========================================================================
#
# Separate from the thin metric plotters above: per subject, stacked
# composition of the Go and No-Go partitions across sessions. For each
# session a single stacked column shows how the partition splits into its
# subcategories, stacked in partition order with the "best" outcome at the
# bottom (so improvement reads as the bottom band rising). Each subject gets
# four figures: Go normalized (fraction, 0-1), Go raw (counts), No-Go
# normalized, No-Go raw. Uses the normal subjids/dates iterators; no grouping
# across subjects.

# Each entry: (subcategory_key, color, legend_label), ordered bottom -> top.
# Colors: best = green, then lighter, worst = red.
_GO_COMPOSITION = [
    ("rewarded_hit", "#2ca02c", "Rewarded Hit"),
    ("port_error_hit", "#98df8a", "Port-Error Hit"),
    ("anticipatory_hit", "#1f77b4", "Anticipatory Hit"),
    ("anticipatory_port_error", "#aec7e8", "Anticipatory Port-Error"),
    ("forfeit_miss", "#ff9896", "Forfeit Miss"),
    ("omission_miss", "#d62728", "Omission Miss"),
]
_NOGO_COMPOSITION = [
    ("active_cr", "#2ca02c", "Active CR"),
    ("passive_cr", "#98df8a", "Passive CR"),
    ("completed_fa", "#ff9896", "Completed FA"),
    ("aborted_fa", "#d62728", "Aborted FA"),
]

# subcategory key -> its parent category in the compute_sing_rew_metrics tree.
_SUBCAT_CATEGORY = {
    "rewarded_hit": "hit",
    "port_error_hit": "hit",
    "anticipatory_hit": "hit",
    "anticipatory_port_error": "hit",
    "forfeit_miss": "miss",
    "omission_miss": "miss",
    "active_cr": "correct_rejection",
    "passive_cr": "correct_rejection",
    "completed_fa": "false_alarm",
    "aborted_fa": "false_alarm",
}


def _session_subcat_counts(cats, order_specs):
    """Per-session subcategory counts for the given composition order."""
    out = {}
    for key, _color, _label in order_specs:
        parent = _SUBCAT_CATEGORY[key]
        out[key] = int(cats.get(parent, {}).get("subcategories", {}).get(key, {}).get("n", 0))
    return out


def _partition_total(cats, order_specs):
    """Partition total the order's subcategories should sum to: the combined count
    of their parent categories (n_go for Go, n_nogo for No-Go)."""
    parents = {_SUBCAT_CATEGORY[key] for key, _c, _l in order_specs}
    return sum(int(cats.get(parent, {}).get("n", 0)) for parent in parents)


def _plot_outcome_composition(recs, order_specs, *, normalize, side, subjid):
    """Stacked composition of one partition across sessions.

    ``recs`` is a per-session list of ``{subcat: count}`` dicts (or None for a
    no-data session). Bars are stacked bottom->top in ``order_specs``. Sessions
    whose partition sum is 0 (or no data) are left as gaps. With ``normalize``
    each column is divided by its partition sum so it fills 0-1; otherwise raw
    counts are stacked.
    """
    n = len(recs)
    if n == 0:
        return None
    fig, ax = plt.subplots(figsize=(11, 5.5))
    any_bar = False
    for x, rec in zip(range(1, n + 1), recs):
        if rec is None:
            continue
        total = sum(rec[key] for key, _c, _l in order_specs)
        if total == 0:
            continue  # gap: no trials in this partition this session
        denom = total if normalize else 1.0
        bottom = 0.0
        for key, color, _label in order_specs:
            height = rec[key] / denom
            if height <= 0:
                continue
            ax.bar(x, height, bottom=bottom, width=0.9, color=color,
                   edgecolor="white", linewidth=0.3, zorder=2)
            bottom += height
        any_bar = True

    if not any_bar:
        plt.close(fig)
        return None

    handles = [Patch(facecolor=color, edgecolor="white", label=label)
               for _key, color, label in order_specs]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1.0),
              title=f"{side} outcome")
    ax.set_xlabel("Sessions")
    if normalize:
        ax.set_ylabel(f"{side} Composition (fraction)")
        ax.set_ylim(0.0, 1.0)
    else:
        ax.set_ylabel(f"{side} Trials (count)")
        ax.set_ylim(bottom=0.0)
    ax.set_title(f"Subjid {subjid} {side} composition ({'normalized' if normalize else 'raw'})")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if n <= 12:
        ax.set_xticks(range(1, n + 1))
    else:
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlim(0.5, n + 0.5)
    fig.tight_layout()
    return fig


def outcome_composition(
    subjids: Optional[Iterable[int]] = None,
    dates: Optional[Union[Iterable[Union[int, str]], tuple]] = None,
    save: bool = False,
    show_raw: bool = False,
    *,
    ses=None,
    index=None,
    date_range=None,
    ses_range=None,
    index_range=None,
):
    """Stacked outcome-composition plots over sessions, per subject.

    Produces the Go and No-Go normalized figures per subject; with
    ``show_raw=True`` the matching raw-count figures are also produced (and
    saved). Each figure stacks the partition's subcategories per session (best
    outcome at the bottom). Normalized columns are divided by the partition sum
    (n_go / n_nogo) so they fill 0-1; raw columns show absolute counts (so a
    2-trial session reads very differently from a 200-trial one).

    Sessions are numbered per subject from 1 (leading no-data sessions trimmed);
    mid-run sessions with no trials in the partition are left as gaps.

    As a sanity check, the plotted subcategory sum is compared against the
    partition total (n_go / n_nogo) for each session; any mismatch (a category
    not fully covered by its subcategories) is flagged in the printed output.

    ``ses`` / ``index`` / ``date_range`` / ``ses_range`` / ``index_range`` narrow the
    selection further; they intersect with ``dates`` and with each other, and ``index``
    is the subject's gap-free chronological rank (`utils.helpers.session_selectors`).
    """
    select = session_selectors(
        ses=ses, index=index, date_range=date_range,
        ses_range=ses_range, index_range=index_range,
    )
    figs = []
    flags = []
    normalize_modes = (True, False) if show_raw else (True,)
    for subjid, date_vals, results_dirs in _collect_sessions(subjids, dates, **select):
        pairs = sorted(zip(date_vals, results_dirs), key=lambda p: str(p[0]))
        loaded = []  # (date_val, df_or_None) chronological
        for date_val, results_dir in pairs:
            df = _load_sorted_session(results_dir)
            loaded.append((date_val, df if (df is not None and not df.empty) else None))
        first = next((i for i, (_d, df) in enumerate(loaded) if df is not None), None)
        if first is None:
            continue
        loaded = loaded[first:]

        side_recs = {"Go": [], "No-Go": []}
        for sess_idx, (date_val, df) in enumerate(loaded, start=1):
            if df is None:
                side_recs["Go"].append(None)
                side_recs["No-Go"].append(None)
                continue
            cats = compute_sing_rew_metrics({"trial_data": df})
            for order_specs, side in ((_GO_COMPOSITION, "Go"), (_NOGO_COMPOSITION, "No-Go")):
                rec = _session_subcat_counts(cats, order_specs)
                plotted = sum(rec.values())
                total = _partition_total(cats, order_specs)
                if plotted != total:
                    diff = plotted - total
                    word = "over" if diff > 0 else "short"
                    flags.append(
                        f"  sub-{subjid} {side} session {sess_idx} (date {date_val}): "
                        f"plotted subcategory sum {plotted} {word} vs partition total "
                        f"{total} (by {abs(diff)})"
                    )
                side_recs[side].append(rec)

        for order_specs, side, tag in (
            (_GO_COMPOSITION, "Go", "go"),
            (_NOGO_COMPOSITION, "No-Go", "nogo"),
        ):
            for normalize in normalize_modes:
                fig = _plot_outcome_composition(side_recs[side], order_specs,
                                                normalize=normalize, side=side, subjid=subjid)
                if fig is None:
                    continue
                figs.append(fig)
                if save:
                    kind = "norm" if normalize else "raw"
                    save_figure(fig, f"sing_rew_composition_{tag}_{kind}",
                                subjids=[subjid], dates=date_vals)

    if flags:
        print("[outcome_composition] subcategory sum != partition total (N_go / N_nogo):")
        for line in flags:
            print(line)
    return figs


# =========================================================================
# Cumulative Hit + Correct Rejection over time / trials
# =========================================================================
#
# Like plot_cumulative_rewards / plot_cumulative_rewards_by_trial, but the curve
# steps up by 1 on every HIT or CORRECT REJECTION trial (broad singrew category)
# and stays flat on MISS / FALSE ALARM (and premature / uncategorized). One
# figure each: continuous time axis, or continuous global_trial_id axis. Both
# start at the first trial of the first session that contains single-reward data.


def _normalize_subjids_dates(subjids, dates):
    """Normalize the (subjids, dates) inputs shared by the cumulative plotters,
    supporting a ``{subjid: date_range}`` dict passed as ``subjids``."""
    if isinstance(subjids, dict):
        dates = subjids if (dates is None or not isinstance(dates, dict)) else dates
        subjids = list(subjids.keys())
    elif isinstance(subjids, set):
        subjids = sorted(subjids)
    elif not isinstance(subjids, (list, tuple)):
        subjids = [subjids]

    def dates_for(subjid):
        if not isinstance(dates, dict):
            return dates
        if subjid in dates:
            return dates[subjid]
        try:
            if int(subjid) in dates:
                return dates[int(subjid)]
        except (TypeError, ValueError):
            pass
        return dates.get(str(subjid))

    return subjids, dates, dates_for


def _trim_timeline_to_singrew(timeline):
    """Trim a subject timeline (from ``_load_subject_trial_timeline``) to start at
    the first trial whose session contains single-reward data (``reward_determinacy``
    populated), re-zeroing the time axis and re-basing the trial axis to start at 1.
    Returns the trimmed dict, or None if there is no single-reward data."""
    combined = timeline["combined"]
    if "reward_determinacy" not in combined.columns:
        return None
    mask = combined["reward_determinacy"].notna().to_numpy()
    if not mask.any():
        return None
    start_pos = int(np.argmax(mask))
    sub = combined.iloc[start_pos:].copy()
    t0 = float(sub["time_seconds"].iloc[0])
    idx0 = int(sub["trial_index"].iloc[0])
    sub["time_seconds"] = sub["time_seconds"] - t0
    sub["trial_index"] = sub["trial_index"] - (idx0 - 1)  # first kept trial -> 1

    def _shift_gaps(gaps, off):
        return [(max(0.0, s - off), e - off) for s, e in gaps if e > off]

    def _shift_bounds(bounds, off):
        # > off + 0.5 also drops the boundary marking the first kept session's
        # own start (it would otherwise land at the left edge).
        return [b - off for b in bounds if b > off + 0.5]

    return {
        "combined": sub,
        "time_gaps": _shift_gaps(timeline["time_gaps"], t0),
        "time_boundaries": _shift_bounds(timeline["time_boundaries"], t0),
        "trial_gaps": _shift_gaps(timeline["trial_gaps"], idx0 - 1),
        "trial_boundaries": _shift_bounds(timeline["trial_boundaries"], idx0 - 1),
    }


def _plot_cumulative_hit_cr(subjids, dates, *, ses=None, index=None, date_range=None,
                            ses_range=None, index_range=None,
                            by_trial, save, verbose,
                            show_gap_shading, show_session_boundaries, figsize):
    """Core for the cumulative Hit+CR plots (time axis or trial axis)."""
    select = session_selectors(
        ses=ses, index=index, date_range=date_range,
        ses_range=ses_range, index_range=index_range,
    )
    subjids, dates, dates_for = _normalize_subjids_dates(subjids, dates)
    single_subject = len(subjids) == 1
    gap_on = show_gap_shading and single_subject
    boundary_on = show_session_boundaries and single_subject

    fig, ax = plt.subplots(figsize=figsize)
    colors = plt.cm.tab10(range(len(subjids)))

    for subj_idx, subjid in enumerate(subjids):
        subj_dates = dates_for(subjid)
        if isinstance(dates, dict) and subj_dates is None:
            print(f"Warning: No date range provided in dict for subject {subjid}, skipping")
            continue
        timeline = _load_subject_trial_timeline(subjid, subj_dates, **select)
        if timeline is None:
            print(f"Warning: No trials found for subject {subjid}")
            continue
        trimmed = _trim_timeline_to_singrew(timeline)
        if trimmed is None:
            print(f"Warning: No single-reward data found for subject {subjid}")
            continue
        combined = trimmed["combined"]

        increments = [
            1 if _classify_trial(row)[0] in ("hit", "correct_rejection") else 0
            for _, row in combined.iterrows()
        ]
        cumulative = np.cumsum(increments)

        if by_trial:
            xs = combined["trial_index"].to_numpy()
            gaps = trimmed["trial_gaps"]
            boundaries = trimmed["trial_boundaries"]
        else:
            xs = combined["time_seconds"].to_numpy()
            gaps = trimmed["time_gaps"]
            boundaries = trimmed["time_boundaries"]

        if gap_on:
            for gap_start, gap_end in gaps:
                ax.axvspan(gap_start, gap_end, alpha=0.2, color="gray", zorder=0)
        if boundary_on:
            for boundary in boundaries:
                ax.axvline(x=boundary, color="gray", linestyle="-", linewidth=0.8, alpha=0.6, zorder=3)

        ax.plot(xs, cumulative, color=colors[subj_idx], marker="o", markersize=3,
                label=f"Subject {subjid}")

    ax.set_xlabel("Trial (continuous global_trial_id)" if by_trial else "Time (s)")
    ax.set_ylabel("Cumulative Hit and Correct Rejection")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend()
    fig.tight_layout()

    if save:
        if isinstance(dates, dict):
            save_dates = []
            for v in dates.values():
                if isinstance(v, (list, tuple)):
                    save_dates.extend(v)
                elif v is not None:
                    save_dates.append(v)
        else:
            save_dates = dates
        save_name = "cumulative_hit_cr_by_trial" if by_trial else "cumulative_hit_cr"
        try:
            out_path = save_figure(fig, save_name,
                                   subjids=list(subjids) if isinstance(subjids, (list, tuple)) else [subjids],
                                   dates=save_dates)
            if verbose:
                print(f"[{save_name}] Saved figure to {out_path}")
        except Exception as exc:
            if verbose:
                print(f"[{save_name}] Failed to save figure: {exc}")

    plt.show()
    return fig, ax


def plot_cumulative_hit_cr(
    subjids,
    dates=None,
    *,
    ses=None,
    index=None,
    date_range=None,
    ses_range=None,
    index_range=None,
    save=False,
    verbose=True,
    show_gap_shading=True,
    show_session_boundaries=True,
    figsize=(12, 6.5),
):
    """Cumulative Hit + Correct Rejection vs continuous time.

    Steps up by 1 on each trial in the broad HIT or CORRECT REJECTION category
    and stays flat on MISS / FALSE ALARM (and premature / uncategorized). The X
    axis is continuous time with collapsed inter-session gaps (like
    ``plot_cumulative_rewards``); it starts at 0 at the first trial of the first
    session that contains single-reward data.

    ``subjids`` may be a ``{subjid: date_range}`` dict (pass with ``dates=None``).
    ``show_gap_shading`` / ``show_session_boundaries`` apply to a single subject
    only. Returns ``(fig, ax)``.

    ``ses`` / ``index`` / ``date_range`` / ``ses_range`` / ``index_range`` narrow the
    selection further; they intersect with ``dates`` and with each other, and ``index``
    is the subject's gap-free chronological rank (`utils.helpers.session_selectors`).
    """
    select = session_selectors(
        ses=ses, index=index, date_range=date_range,
        ses_range=ses_range, index_range=index_range,
    )
    return _plot_cumulative_hit_cr(
        subjids, dates, by_trial=False, save=save, verbose=verbose,
        show_gap_shading=show_gap_shading, show_session_boundaries=show_session_boundaries,
        figsize=figsize,
        **select,
    )


def plot_cumulative_hit_cr_by_trial(
    subjids,
    dates=None,
    *,
    ses=None,
    index=None,
    date_range=None,
    ses_range=None,
    index_range=None,
    save=False,
    verbose=True,
    show_gap_shading=True,
    show_session_boundaries=True,
    figsize=(12, 6.5),
):
    """Cumulative Hit + Correct Rejection vs continuous trial index.

    Identical to ``plot_cumulative_hit_cr`` except the X axis is the continuous
    ``global_trial_id`` (concatenated across sessions, like
    ``plot_cumulative_rewards_by_trial``): every trial advances X by one, and the
    curve steps up on Hit/CR, flat on Miss/FA. Returns ``(fig, ax)``.

    ``ses`` / ``index`` / ``date_range`` / ``ses_range`` / ``index_range`` narrow the
    selection further; they intersect with ``dates`` and with each other, and ``index``
    is the subject's gap-free chronological rank (`utils.helpers.session_selectors`).
    """
    select = session_selectors(
        ses=ses, index=index, date_range=date_range,
        ses_range=ses_range, index_range=index_range,
    )
    return _plot_cumulative_hit_cr(
        subjids, dates, by_trial=True, save=save, verbose=verbose,
        show_gap_shading=show_gap_shading, show_session_boundaries=show_session_boundaries,
        figsize=figsize,
        **select,
    )
