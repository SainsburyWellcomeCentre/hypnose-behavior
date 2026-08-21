# Defers evaluation of PEP-604 annotations (`X | None`), keeping this module
# importable on Python 3.9 for repos pinned there (hypnose-eeg-preprocessing).
from __future__ import annotations

"""Latency, response-time and inter-trial-interval figures.

``_plot_metric_over_sessions`` and its two helpers stay here: both callers
(``plot_latency_over_time``, ``plot_iti_over_time``) are in this module, so
section 3's promote-what-is-shared rule does not apply to them.

``avg_response_time`` reads the **movement** latency (b),
measured from the animal's last cue-port exit -- not the window-relative one.
"""

import pandas as pd
import matplotlib.pyplot as plt
from hypnose_behavior.metric_analysis.metrics.false_alarm import FA_avg_response_times
from hypnose_behavior.metric_analysis.metrics.timing import avg_response_time
from hypnose_behavior.io import layout
from hypnose_behavior.io.layout import (
    _iter_subject_dirs,
    session_selectors,
)
from hypnose_behavior.io.paths import (
    get_rawdata_root,
    get_derivatives_root,
    get_server_root,
)
import numpy as np
from hypnose_behavior.io.save import save_figure
from hypnose_behavior.visualization.prep import _load_subject_trial_timeline
from hypnose_behavior.visualization.primitives import (
    mean_sem,
    rolling_windows,
)
from hypnose_behavior.io.loaders import iter_sessions



def plot_response_times_completed_vs_fa(
    subjid,
    dates=None,
    figsize=(12, 8),
    y_limit=20000,
    *,
    ses=None,
    index=None,
    date_range=None,
    ses_range=None,
    index_range=None,
    save=False,
    verbose=True,
):
    """
    Scatter plot comparing average response times for completed sequences vs FA Time In abortions.
    Both metrics on the same plot sharing Y-axis for easy comparison.
    
    Parameters:
    -----------
    subjid : int
        Subject ID
    dates : tuple, list, or None
        Date or date range. If None, plots all available dates.
    figsize : tuple, optional
        Figure size (default: (12, 8))
    y_limit : float, optional
        Maximum Y-axis value to display (default: 20000). Points above this are excluded.
    save : bool, optional
        If True, save the generated figure using save_figure (default: False).
    verbose : bool, optional
        If True, print status messages (default: True).
    
    Returns:
    --------
    fig, ax : matplotlib figure and axes

    ``ses`` / ``index`` / ``date_range`` / ``ses_range`` / ``index_range`` narrow the
    selection further; they intersect with ``dates`` and with each other, and ``index``
    is the subject's gap-free chronological rank (`io.layout.session_selectors`).
    """
    select = session_selectors(
        ses=ses, index=index, date_range=date_range,
        ses_range=ses_range, index_range=index_range,
    )
    base_path = get_rawdata_root()
    server_root = get_server_root()
    derivatives_dir = get_derivatives_root()
    
    rows = []
    
    for sid, subj_dir in _iter_subject_dirs(derivatives_dir, [subjid]):
        ses_recs = iter_sessions(subj_dir, dates, **select)
        for rec in ses_recs:
            date_str = rec.date_str
            results_dir = rec.results_dir
            if not rec.analysed:
                continue

            # Both means come from the canonical metrics over trial_data. **Never add
            # a fall-back to metrics_*.json**: a saved JSON predates any later metric
            # change, so two figures could show this quantity and disagree.
            # See DECISIONS.md section 5.
            td = rec.views.get("trial_data", pd.DataFrame())

            completed_rt = avg_response_time(td).get(
                "Average Response Time (Rewarded + Unrewarded)")
            if completed_rt is not None and not np.isnan(completed_rt):
                rows.append({
                    "date": int(date_str),
                    "response_type": "Completed Sequences",
                    "response_time_ms": float(completed_rt)
                })

            fa_rt = FA_avg_response_times(td).get("FA Time In")
            if fa_rt is not None and not np.isnan(fa_rt):
                rows.append({
                    "date": int(date_str),
                    "response_type": "FA Time In Abortions",
                    "response_time_ms": float(fa_rt)
                })
    
    if not rows:
        print("No data found")
        return None, None
    
    df = pd.DataFrame(rows)
    
    # Filter by y_limit
    df_filtered = df[df["response_time_ms"] <= y_limit].copy()
    
    if df_filtered.empty:
        print(f"No data found below y_limit={y_limit}")
        return None, None
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    response_types = ["Completed Sequences", "FA Time In Abortions"]
    x_positions = [0, 1]
    colors = ['steelblue', 'coral']
    
    for x_pos, resp_type, color in zip(x_positions, response_types, colors):
        df_subset = df_filtered[df_filtered["response_type"] == resp_type].copy()
        
        if not df_subset.empty:
            values = df_subset["response_time_ms"].values

            # Scatter per session (jitter) using session means already in rows
            x_jitter = np.random.normal(x_pos, 0.08, size=len(values))
            ax.scatter(x_jitter, values, alpha=0.4, s=80, color=color, zorder=3)

            # Calculate mean and SEM across sessions
            mean_rt, sem_rt = mean_sem(values)
            sem_rt = 0.0 if np.isnan(sem_rt) else sem_rt

            # Plot mean point with SEM bars
            ax.scatter([x_pos], [mean_rt], color='darkred', s=150, zorder=5, marker='D',
                      edgecolors='black', linewidth=2, label='Mean ± SEM' if x_pos == 0 else "")
            ax.errorbar([x_pos], [mean_rt], yerr=sem_rt, fmt='none', ecolor='darkred',
                       capsize=8, capthick=2, linewidth=2.5, zorder=4)
    
    ax.set_xlim([-0.5, 1.5])
    ax.set_xticks(x_positions)
    ax.set_xticklabels(response_types)
    ax.set_ylabel('Response Time (ms)')
    ax.set_ylim([0, y_limit])
    ax.set_title(f'Average Response Times Comparison\n(Subject {str(subjid).zfill(3)})')
    ax.legend(loc='best')
    
    plt.tight_layout()

    if save:
        try:
            save_name = "response_times_completed_vs_fa"
            out_path = save_figure(fig, save_name, subjids=[subjid], dates=dates)
            if verbose:
                print(
                    f"[plot_response_times_completed_vs_fa] Saved figure to {out_path}"
                )
        except Exception as exc:
            if verbose:
                print(
                    "[plot_response_times_completed_vs_fa] Failed to save figure: "
                    f"{exc}"
                )

    return fig, ax



def _rolling_median_iqr(x, y, window_size, step_size):
    """Rolling median + 25th/75th percentiles over trial windows.

    ``x``/``y`` are equal-length arrays already ordered by x. Returns
    (mx, median, q25, q75); each window is anchored to the actual x value of its
    center trial (not an interpolated midpoint between the first and last x).
    """
    n = len(y)
    if n == 0:
        return (np.array([]),) * 4
    mx, med, q25, q75 = [], [], [], []
    # `partial=True`: a run shorter than one window still yields one window here.
    for start, end in rolling_windows(n, window_size, step=step_size, partial=True):
        seg = y[start:end]
        med.append(np.nanmedian(seg))
        q25.append(np.nanpercentile(seg, 25))
        q75.append(np.nanpercentile(seg, 75))
        center = (start + end - 1) // 2  # index of the window's center trial
        mx.append(x[center])
    return np.array(mx), np.array(med), np.array(q25), np.array(q75)



def _style_log_yaxis(ax):
    """Make a log Y axis clearly read as log: plain-number major labels at each
    decade, plus minor ticks at 2-9 within every decade (the 2x and 5x labelled
    smaller). Without this a range spanning <1 decade shows almost no ticks.

    The minor label size is resolved through matplotlib's own font machinery
    rather than ``float()``. ``ytick.labelsize`` is allowed to be a *relative*
    keyword -- and matplotlib's default is the string ``"medium"`` -- so the
    plain cast raises ``ValueError`` in any process that has not applied the repo
    style first, i.e. a bare notebook or script. Under the style the key is numeric
    and both forms give the same number.
    """
    from matplotlib.font_manager import FontProperties
    from matplotlib.ticker import LogLocator, ScalarFormatter, FuncFormatter

    ax.set_yscale("log")
    ax.yaxis.set_major_locator(LogLocator(base=10.0))
    major_fmt = ScalarFormatter()
    major_fmt.set_scientific(False)
    ax.yaxis.set_major_formatter(major_fmt)
    ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=(2, 3, 4, 5, 6, 7, 8, 9)))

    def _minor_fmt(value, _pos):
        if value <= 0:
            return ""
        lead = value / (10 ** np.floor(np.log10(value)))
        return f"{value:g}" if round(lead) in (2, 5) else ""

    ax.yaxis.set_minor_formatter(FuncFormatter(_minor_fmt))
    base_labelsize = FontProperties(size=plt.rcParams["ytick.labelsize"]).get_size_in_points()
    minor_labelsize = base_labelsize * 0.6
    ax.tick_params(axis="y", which="minor", left=True, labelleft=True, labelsize=minor_labelsize)



def _plot_metric_over_sessions(
    subjids,
    dates,
    *,
    ses=None,
    index=None,
    date_range=None,
    ses_range=None,
    index_range=None,
    value_col,
    metric_name,
    unit,
    rewarded_only,
    window_size,
    step_size,
    save,
    save_key,
    verbose,
    show_gap_shading,
    show_session_boundaries,
    show_iqr,
    figsize,
):
    """Core for latency/ITI plots: a time-axis rolling median+IQR figure and a
    trial-axis cumulative figure. See the public wrappers for the contract."""
    select = session_selectors(
        ses=ses, index=index, date_range=date_range,
        ses_range=ses_range, index_range=index_range,
    )
    if isinstance(subjids, dict):
        dates = subjids if (dates is None or not isinstance(dates, dict)) else dates
        subjids = list(subjids.keys())
    elif isinstance(subjids, set):
        subjids = sorted(subjids)
    elif not isinstance(subjids, (list, tuple)):
        subjids = [subjids]

    def _dates_for(subjid):
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

    single_subject = len(subjids) == 1
    gap_on = show_gap_shading and single_subject
    boundary_on = show_session_boundaries and single_subject

    fig_time, ax_time = plt.subplots(figsize=figsize)
    fig_trial, ax_trial = plt.subplots(figsize=figsize)
    colors = plt.cm.tab20(range(len(subjids)))

    for subj_idx, subjid in enumerate(subjids):
        subj_dates = _dates_for(subjid)
        if isinstance(dates, dict) and subj_dates is None:
            print(f"Warning: No date range provided in dict for subject {subjid}, skipping")
            continue
        timeline = _load_subject_trial_timeline(subjid, subj_dates, **select)
        if timeline is None:
            print(f"Warning: No trials found for subject {subjid}")
            continue
        combined = timeline["combined"]
        if value_col not in combined.columns:
            print(f"Warning: '{value_col}' column missing for subject {subjid}")
            continue
        color = colors[subj_idx]

        # ---- Time-axis: per-trial value, rolling median + IQR (log y) ----
        tdata = combined[combined["is_rewarded"]] if rewarded_only else combined
        tdata = tdata[["time_seconds", value_col]].copy()
        tdata = tdata[tdata[value_col].notna() & (tdata[value_col] > 0)]
        tdata = tdata.sort_values("time_seconds")
        if not tdata.empty:
            if gap_on:
                for gap_start, gap_end in timeline["time_gaps"]:
                    ax_time.axvspan(gap_start, gap_end, alpha=0.2, color="gray", zorder=0)
            if boundary_on:
                for boundary in timeline["time_boundaries"]:
                    ax_time.axvline(x=boundary, color="gray", linestyle="-", linewidth=0.8, alpha=0.6, zorder=3)
            mx, med, q25, q75 = _rolling_median_iqr(
                tdata["time_seconds"].to_numpy(),
                tdata[value_col].to_numpy(),
                window_size,
                step_size,
            )

            if len(mx):
                if show_iqr:
                    ax_time.fill_between(
                        mx, q25, q75,
                        color=color,
                        alpha=0.2,
                        zorder=2,
                    )

                ax_time.plot(
                    mx,
                    med,
                    color=color,
                    linewidth=2,
                    label=f"Subject {subjid}",
                    zorder=3,
                )
        # ---- Trial-axis: cumulative value over continuous trial index ----
        contrib = combined[value_col].to_numpy(dtype=float)
        contrib = np.nan_to_num(contrib, nan=0.0)
        # Only positive values contribute (ignore NaN and non-positive, e.g. a
        # negative ITI from overlapping trials, which would otherwise drag the
        # cumulative curve down).
        contrib = np.where(contrib > 0, contrib, 0.0)
        if rewarded_only:
            contrib = np.where(combined["is_rewarded"].to_numpy(), contrib, 0.0)
        cumulative = np.cumsum(contrib)
        xs = combined["trial_index"].to_numpy()
        if gap_on:
            for gap_start, gap_end in timeline["trial_gaps"]:
                ax_trial.axvspan(gap_start, gap_end, alpha=0.2, color="gray", zorder=0)
        if boundary_on:
            for boundary in timeline["trial_boundaries"]:
                ax_trial.axvline(x=boundary, color="gray", linestyle="-", linewidth=0.8, alpha=0.6, zorder=3)
        ax_trial.plot(xs, cumulative, color=color, linewidth=2, label=f"Subject {subjid}", zorder=3)

    rew_tag = " (rewarded only)" if rewarded_only else ""
    ax_time.set_xlabel("Time (s)")
    ax_time.set_ylabel(f"{metric_name} ({unit})")
    _style_log_yaxis(ax_time)
    ax_time.set_xlim(left=0)
    ax_time.legend()
    fig_time.tight_layout()

    ax_trial.set_xlabel("Trial (continuous global_trial_id)")
    ax_trial.set_ylabel(f"Cumulative {metric_name} ({unit})")
    ax_trial.set_xlim(left=0)
    ax_trial.legend()
    fig_trial.tight_layout()

    if save:
        rew_save = "_rewarded" if rewarded_only else ""
        if isinstance(dates, dict):
            save_dates = []
            for v in dates.values():
                if isinstance(v, (list, tuple)):
                    save_dates.extend(v)
                elif v is not None:
                    save_dates.append(v)
        else:
            save_dates = dates
        subj_list = list(subjids) if isinstance(subjids, (list, tuple)) else [subjids]
        for fig, axis_tag in ((fig_time, "timeaxis"), (fig_trial, "trialaxis")):
            try:
                out_path = save_figure(fig, f"{save_key}_{axis_tag}{rew_save}",
                                       subjids=subj_list, dates=save_dates)
                if verbose:
                    print(f"[{save_key}] Saved figure to {out_path}")
            except Exception as exc:
                if verbose:
                    print(f"[{save_key}] Failed to save figure: {exc}")

    plt.show()
    return fig_time, ax_time, fig_trial, ax_trial



def plot_latency_over_time(
    subjids,
    dates=None,
    *,
    ses=None,
    index=None,
    date_range=None,
    ses_range=None,
    index_range=None,
    rewarded_only=False,
    window_size=20,
    step_size=5,
    show_iqr=False, 
    save=False,
    verbose=True,
    show_gap_shading=True,
    show_session_boundaries=True,
    figsize=(12, 6.5),
):
    """Response-time (latency) over time, as two figures.

    1. Time axis (collapsed inter-session gaps, like ``plot_cumulative_rewards``):
       per-trial ``response_time_ms`` on a log Y, summarised by a rolling median
       over a trial window (``window_size``/``step_size``) with the 25th-75th
       percentile range shaded around it.
    2. Trial axis (continuous ``global_trial_id``, like
       ``plot_cumulative_rewards_by_trial``): cumulative response time.

    ``rewarded_only=True`` restricts to rewarded trials
    (``response_time_category == "rewarded"``); otherwise all trials with a
    response time are used. ``show_gap_shading`` / ``show_session_boundaries``
    apply to a single subject only. ``subjids`` may be a ``{subjid: date_range}``
    dict (pass with ``dates=None``).

    Returns ``(fig_time, ax_time, fig_trial, ax_trial)``.

    ``ses`` / ``index`` / ``date_range`` / ``ses_range`` / ``index_range`` narrow the
    selection further; they intersect with ``dates`` and with each other, and ``index``
    is the subject's gap-free chronological rank (`io.layout.session_selectors`).
    """
    select = session_selectors(
        ses=ses, index=index, date_range=date_range,
        ses_range=ses_range, index_range=index_range,
    )
    return _plot_metric_over_sessions(
        subjids, dates, value_col="response_time_ms", metric_name="Response Time",
        unit="ms", rewarded_only=rewarded_only, window_size=window_size, step_size=step_size,
        save=save, save_key="latency_over_time", verbose=verbose,
        show_gap_shading=show_gap_shading, show_session_boundaries=show_session_boundaries,
        figsize=figsize, show_iqr=show_iqr,
        **select,
    )



def plot_iti_over_time(
    subjids,
    dates=None,
    *,
    ses=None,
    index=None,
    date_range=None,
    ses_range=None,
    index_range=None,
    window_size=20,
    step_size=5,
    show_iqr=False,
    save=False,
    verbose=True,
    show_gap_shading=True,
    show_session_boundaries=True,
    figsize=(12, 6.5),
):
    """Inter-trial interval (ITI) over time, mirroring ``plot_latency_over_time``.

    ITI is computed within a session only: for consecutive trials (by
    ``global_trial_id``) it is ``sequence_start[i+1] - sequence_end[i]``; the last
    trial of each session has no ITI (never across sessions). Two figures: time
    axis with per-trial ITI on a log Y (rolling median + 25th-75th percentile
    band), and trial axis with cumulative ITI. No ``rewarded_only`` toggle.

    Returns ``(fig_time, ax_time, fig_trial, ax_trial)``.

    ``ses`` / ``index`` / ``date_range`` / ``ses_range`` / ``index_range`` narrow the
    selection further; they intersect with ``dates`` and with each other, and ``index``
    is the subject's gap-free chronological rank (`io.layout.session_selectors`).
    """
    select = session_selectors(
        ses=ses, index=index, date_range=date_range,
        ses_range=ses_range, index_range=index_range,
    )
    return _plot_metric_over_sessions(
        subjids, dates, value_col="iti_seconds", metric_name="ITI", unit="s",
        rewarded_only=False, window_size=window_size, step_size=step_size,
        save=save, save_key="iti_over_time", verbose=verbose,
        show_gap_shading=show_gap_shading, show_session_boundaries=show_session_boundaries,
        figsize=figsize, show_iqr=show_iqr,
        **select,
    )
