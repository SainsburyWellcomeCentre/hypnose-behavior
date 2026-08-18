# Defers evaluation of PEP-604 annotations (`X | None`), keeping this module
# importable on Python 3.9 for repos pinned there (hypnose-eeg-preprocessing).
from __future__ import annotations

"""Cumulative-reward figures.

Carved out of ``visualization_utils.py`` in restructure_2 Phase 10 (follow-up
Item 1). Source-only move -- no behaviour change.

The two differ in their x axis only: calendar time, and a continuous trial index
made contiguous across sessions.
"""

import pandas as pd
import matplotlib.pyplot as plt
from hypnose_behavior.metric_analysis.metrics.accuracy import decision_accuracy
from hypnose_behavior.utils.helpers import (
    _filter_session_dirs,
    session_selectors,
)
from hypnose_behavior.io.layout import (
    derivatives,
    normalize_subjid,
)
from hypnose_behavior.io.paths import (
    get_rawdata_root,
    get_derivatives_root,
    get_server_root,
)
import numpy as np
import json
from hypnose_behavior.io.save import save_figure
from hypnose_behavior.io.loaders import _load_trial_views



def plot_cumulative_rewards(
    subjids,
    dates,
    split_days=False,
    figsize=(12, 6.5),
    title=None,
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
    show_title=True,
    show_legend=True,
    show_da_thresh=False,
    da_thresh=80,
):
    """
    Plot cumulative rewards with inter-session gap collapsing.

    OPTIMIZATION: Skip load_session_results() for 15+ DataFrames.
    Load ONLY: completed_sequence_rewarded CSV + manifest.json
    This gives ~10-15x speedup while keeping all visual features.

    Parameters:
    -----------
    subjids : int or list
        Subject ID(s)
    dates : list, tuple, dict, or None
        Dates to include. If a dict, must map subjid → date range (each value
        is itself a list/tuple/None passed through to ``_filter_session_dirs``
        for that subject). This allows each subject to be filtered to its own
        date window — useful for comparing animals across matched training
        conditions when they are offset in calendar time. Subjids not present
        as keys are skipped with a warning.
    split_days : bool, optional
        If True, reset cumulative count per day (default: False)
    figsize : tuple, optional
        Figure size (default: (12, 6))
    title : str, optional
        Plot title
    save : bool, optional
        If True, save figure via save_figure (default: False).
    verbose : bool, optional
        If True, print save status messages (default: True).
    show_gap_shading : bool, optional
        If True, shade within-session gaps (time mouse could not perform the
        task) in grey (default: True).
    show_session_boundaries : bool, optional
        If True, draw thin grey vertical lines at session boundaries
        (default: True).
    show_da_thresh : bool, optional
        If True, draw a black finely-dashed vertical line (spanning the full y
        axis) at the start of the first session whose decision accuracy
        (rewarded / (rewarded + unrewarded)) first exceeds ``da_thresh``. One
        line per subject (default: False).
    da_thresh : float, optional
        Decision-accuracy threshold for ``show_da_thresh``, as a percentage
        (e.g. 80). Values <= 1 are treated as a fraction (default: 80).

    Returns:
    --------
    fig, ax : matplotlib figure and axes

    ``ses`` / ``index`` / ``date_range`` / ``ses_range`` / ``index_range`` narrow the
    selection further; they intersect with ``dates`` and with each other, and ``index``
    is the subject's gap-free chronological rank (`utils.helpers.session_selectors`).
    """
    select = session_selectors(
        ses=ses, index=index, date_range=date_range,
        ses_range=ses_range, index_range=index_range,
    )
    # Ensure subjids is a list
    if isinstance(subjids, dict):
        # Convenience: allow passing one dict for both ({subjid: date_range}).
        dates = subjids if not isinstance(dates, dict) or dates is None else dates
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
            int_key = int(subjid)
            if int_key in dates:
                return dates[int_key]
        except (TypeError, ValueError):
            pass
        str_key = str(subjid)
        if str_key in dates:
            return dates[str_key]
        return None

    fig, ax = plt.subplots(figsize=figsize)
    colors = plt.cm.tab20(range(len(subjids)))
    da_cross_xs = []  # x-positions of per-subject decision-accuracy crossings
    data_xmax_val = 0.0  # data extents (tracked here so axvline markers don't skew limits)
    data_ymax_val = 0.0

    for subj_idx, subjid in enumerate(subjids):
        subj_dates = _dates_for(subjid)
        if isinstance(dates, dict) and subj_dates is None:
            print(f"Warning: No date range provided in dict for subject {subjid}, skipping")
            continue
        all_rewarded = []
        session_info = []
        session_da = {}  # date_str -> decision accuracy (for show_da_thresh)

        # Find subject directory
        base_path = get_rawdata_root()
        server_root = get_server_root()
        derivatives_dir = get_derivatives_root()
        subj_str = normalize_subjid(subjid)
        subj_dir = derivatives.subject_dir(subjid, missing_ok=True)
        if subj_dir is None:
            print(f"Warning: No subject directory found for {subj_str}")
            continue

        # Use _filter_session_dirs to get session directories
        ses_dirs = _filter_session_dirs(subj_dir, subj_dates, **select)
        for ses_dir in ses_dirs:
            date_str = ses_dir.name.split("_date-")[-1]
            results_dir = ses_dir / "saved_analysis_results"
            if not results_dir.exists():
                continue
            
            views = _load_trial_views(results_dir)

            # The threshold gate is the canonical decision_accuracy, not a
            # recompute of it.
            if show_da_thresh:
                _, denom, acc = decision_accuracy(views.get("trial_data", pd.DataFrame()))
                if denom > 0:
                    session_da[date_str] = acc

            rewarded_trials = views.get("rewarded", pd.DataFrame())
            if rewarded_trials.empty:
                continue
            try:
                rewarded_trials['sequence_start'] = pd.to_datetime(rewarded_trials['sequence_start'])
            except Exception:
                pass
            rewarded_trials['date'] = date_str
            all_rewarded.append(rewarded_trials)
            
            # OPTIMIZATION 2: Load ONLY manifest.json for session timing
            manifest_path = results_dir / "summary.json"
            if manifest_path.exists():
                try:
                    with open(manifest_path, 'r', encoding='utf-8') as f:
                        manifest = json.load(f)
                    runs = manifest.get('session', {}).get('runs', [])
                    if runs:
                        session_info.append({
                            'date': date_str,
                            'runs': runs,
                        })
                except Exception:
                    pass
        
        if not all_rewarded:
            print(f"Warning: No rewarded trials found for subject {subjid}")
            continue
        
        # Combine all dates
        combined = pd.concat(all_rewarded, ignore_index=True)
        combined = combined.sort_values('sequence_start').reset_index(drop=True)
        
        # Subject-specific session boundaries and gaps
        subj_session_starts = []
        subj_gaps = []
        
        # Calculate continuous time axis with collapsed inter-session gaps for this subject
        if session_info:
            # Build time offset mapping for each session
            time_offset = 0  # Cumulative offset to add to timestamps
            session_offsets = {}  # Maps session date to its time offset
            
            for sess_idx, sess in enumerate(session_info):
                runs = sess['runs']
                session_date = sess['date']
                
                if sess_idx == 0:
                    # First session: no offset, use actual start time
                    first_run_start = pd.to_datetime(runs[0]['start_time']).tz_localize(None)
                    global_start_time = first_run_start
                    session_offsets[session_date] = 0
                else:
                    # Get end of previous session
                    prev_sess = session_info[sess_idx - 1]
                    prev_last_run = prev_sess['runs'][-1]
                    prev_end = pd.to_datetime(prev_last_run['end_time']).tz_localize(None)
                    
                    # Get start of current session
                    curr_start = pd.to_datetime(runs[0]['start_time']).tz_localize(None)
                    
                    # Calculate the gap between sessions (in seconds)
                    inter_session_gap = (curr_start - prev_end).total_seconds()
                    
                    # Add this gap to our cumulative offset (we want to subtract it from timestamps)
                    time_offset += inter_session_gap - 1  # Keep 1 second buffer
                    session_offsets[session_date] = time_offset
                    
                    # Mark session boundary (time where new session "starts" in plot)
                    boundary_seconds = (prev_end - global_start_time).total_seconds() - session_offsets[prev_sess['date']] + 1
                    subj_session_starts.append(boundary_seconds)
                
                # Calculate gaps within this session
                for run in runs:
                    if 'gap_to_next_run' in run and run['gap_to_next_run']:
                        gap_str = run['gap_to_next_run']
                        try:
                            # Parse format like "0:27:53.342496"
                            parts = gap_str.split(':')
                            if len(parts) == 3:
                                hours = int(parts[0])
                                minutes = int(parts[1])
                                seconds = float(parts[2])
                                gap_duration = hours * 3600 + minutes * 60 + seconds
                            else:
                                gap_duration = float(gap_str)
                            
                            # Gap starts at end_time of this run
                            run_end = pd.to_datetime(run['end_time']).tz_localize(None)
                            gap_start_seconds = (run_end - global_start_time).total_seconds() - session_offsets[session_date]
                            gap_end_seconds = gap_start_seconds + gap_duration
                            
                            subj_gaps.append((gap_start_seconds, gap_end_seconds))
                        except Exception:
                            pass
            
            # Apply time offsets to trial data
            combined['time_seconds'] = combined.apply(
                lambda row: (row['sequence_start'] - global_start_time).total_seconds() - session_offsets.get(row['date'], 0),
                axis=1
            )
        else:
            # Fallback if no manifest info
            global_start_time = combined['sequence_start'].iloc[0]
            combined['time_seconds'] = (combined['sequence_start'] - global_start_time).dt.total_seconds()
        
        # Add grey shading for gaps between runs (subject-specific)
        if show_gap_shading:
            for gap_start, gap_end in subj_gaps:
                ax.axvspan(gap_start, gap_end, alpha=0.2, color='gray', zorder=0)

        # Add thin grey vertical lines at session boundaries
        if show_session_boundaries:
            for boundary in subj_session_starts:
                ax.axvline(x=boundary, color='gray', linestyle='-', linewidth=0.8, alpha=0.6, zorder=3)

        # First session whose decision accuracy exceeds da_thresh, marked at that
        # session's END; drawn later as a black finely-dashed full-height vertical
        # line (after axis limits are set so it doesn't affect them). One per subject.
        if show_da_thresh and session_info:
            thresh_frac = da_thresh / 100.0 if da_thresh > 1 else float(da_thresh)
            for sess in session_info:  # chronological order
                d = sess['date']
                da = session_da.get(d)
                if da is not None and not np.isnan(da) and da > thresh_frac:
                    sess_end = pd.to_datetime(sess['runs'][-1]['end_time']).tz_localize(None)
                    cross_x = (sess_end - global_start_time).total_seconds() - session_offsets.get(d, 0)
                    da_cross_xs.append(cross_x)
                    break
        
        if split_days:
            # Reset count for each day
            combined['day_group'] = combined['date'].astype(str)
            combined['cumulative_rewards'] = combined.groupby('day_group').cumcount() + 1
            
            # Plot each day separately (no connecting lines)
            for day in combined['day_group'].unique():
                day_data = combined[combined['day_group'] == day]
                ax.plot(day_data['time_seconds'], 
                       day_data['cumulative_rewards'],
                       color=colors[subj_idx],
                       marker='o',
                       markersize=3,
                       label=f'Subject {subjid}' if day == combined['day_group'].iloc[0] else None)
        else:
            # Continuous accumulation across days
            combined['cumulative_rewards'] = range(1, len(combined) + 1)
            ax.plot(combined['time_seconds'],
                   combined['cumulative_rewards'],
                   color=colors[subj_idx],
                   marker='o',
                   markersize=3,
                   label=f'Subject {subjid}')

        if len(combined):
            data_xmax_val = max(data_xmax_val, float(combined['time_seconds'].max()))
            data_ymax_val = max(data_ymax_val, float(combined['cumulative_rewards'].max()))

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Cumulative Rewards')
    if show_title:
        ax.set_title(title if title else 'Accumulated Rewards Over Time')
    data_xmax = data_xmax_val if data_xmax_val > 0 else ax.get_xlim()[1]
    data_ymax = data_ymax_val if data_ymax_val > 0 else ax.get_ylim()[1]
    ax.set_xlim(left=0, right=data_xmax * 1.01)
    ax.set_ylim(bottom=-data_ymax * 0.01, top=data_ymax * 1.02)

    # Decision-accuracy threshold marker(s): full-height, black, finely dashed.
    for cross_x in da_cross_xs:
        ax.axvline(x=cross_x, color='black', linestyle=(0, (2, 2)), linewidth=1.2, zorder=4)

    if show_legend:
        ax.legend()
    
    plt.tight_layout()
    
    if save:
        try:
            suffix = "split_days" if split_days else "continuous"
            if isinstance(dates, dict):
                save_dates = []
                for v in dates.values():
                    if isinstance(v, (list, tuple)):
                        save_dates.extend(v)
                    elif v is not None:
                        save_dates.append(v)
            else:
                save_dates = dates
            out_path = save_figure(
                fig,
                f"cumulative_rewards_{suffix}",
                subjids=list(subjids) if isinstance(subjids, (list, tuple)) else [subjids],
                dates=save_dates,
            )
            if verbose:
                print(f"[plot_cumulative_rewards] Saved figure to {out_path}")
        except Exception as exc:
            if verbose:
                print(f"[plot_cumulative_rewards] Failed to save figure: {exc}")
    
    plt.show()

    return fig, ax



def plot_cumulative_rewards_by_trial(
    subjids,
    dates=None,
    figsize=(12, 6.5),
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
):
    """Cumulative rewards vs a continuous trial index (not calendar time).

    Like ``plot_cumulative_rewards``, but the X axis is every trial in order
    (``global_trial_id``), made continuous across sessions: session 1's last
    trial is immediately followed by session 2's first trial. The cumulative
    count increments by 1 on each rewarded trial (``response_time_category ==
    "rewarded"``) and stays flat on anything else (unrewarded, timeout_delayed,
    None/NaN, aborted). No per-day reset.

    Parameters mirror ``plot_cumulative_rewards``. ``show_gap_shading``
    (within-session inter-run gaps) and ``show_session_boundaries`` are only
    honoured for a single subject; with more than one subject they are forced
    off (the trial axis is not shared session-for-session across subjects).
    Accepts the same subjids/dates forms, including a ``{subjid: date_range}``
    dict (pass it as ``subjids`` with ``dates=None``).

    Returns
    -------
    fig, ax : matplotlib figure and axes

    ``ses`` / ``index`` / ``date_range`` / ``ses_range`` / ``index_range`` narrow the
    selection further; they intersect with ``dates`` and with each other, and ``index``
    is the subject's gap-free chronological rank (`utils.helpers.session_selectors`).
    """
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
            int_key = int(subjid)
            if int_key in dates:
                return dates[int_key]
        except (TypeError, ValueError):
            pass
        str_key = str(subjid)
        if str_key in dates:
            return dates[str_key]
        return None

    single_subject = len(subjids) == 1
    gap_on = show_gap_shading and single_subject
    boundary_on = show_session_boundaries and single_subject

    fig, ax = plt.subplots(figsize=figsize)
    colors = plt.cm.tab20(range(len(subjids)))

    for subj_idx, subjid in enumerate(subjids):
        subj_dates = _dates_for(subjid)
        if isinstance(dates, dict) and subj_dates is None:
            print(f"Warning: No date range provided in dict for subject {subjid}, skipping")
            continue

        derivatives_dir = get_derivatives_root()
        subj_str = normalize_subjid(subjid)
        subj_dir = derivatives.subject_dir(subjid, missing_ok=True)
        if subj_dir is None:
            print(f"Warning: No subject directory found for {subj_str}")
            continue

        # Sessions in chronological (date) order.
        sessions = []
        for ses_dir in _filter_session_dirs(subj_dir, subj_dates, **select):
            date_str = ses_dir.name.split("_date-")[-1]
            results_dir = ses_dir / "saved_analysis_results"
            if results_dir.exists():
                sessions.append((date_str, results_dir))
        sessions.sort(key=lambda t: t[0])

        xs = []
        rewarded_flags = []
        running_offset = 0
        session_boundaries = []
        gaps = []
        first_session_done = False

        for date_str, results_dir in sessions:
            df = _load_trial_views(results_dir).get("trial_data", pd.DataFrame())
            if df.empty or "global_trial_id" not in df.columns:
                continue
            df = df.sort_values("global_trial_id").reset_index(drop=True)
            n = len(df)

            if boundary_on and first_session_done:
                session_boundaries.append(running_offset + 0.5)

            # Within-session gaps = boundaries between consecutive runs.
            if gap_on and "run_id" in df.columns:
                run_ids = df["run_id"].tolist()
                for i in range(1, n):
                    if run_ids[i] != run_ids[i - 1]:
                        gaps.append((running_offset + i, running_offset + i + 1))

            rtc = df.get("response_time_category")
            rew = (rtc == "rewarded").tolist() if rtc is not None else [False] * n
            xs.extend(range(running_offset + 1, running_offset + n + 1))
            rewarded_flags.extend(rew)
            running_offset += n
            first_session_done = True

        if not xs:
            print(f"Warning: No trials found for subject {subjid}")
            continue

        cumulative = np.cumsum([1 if r else 0 for r in rewarded_flags])

        if gap_on:
            for gap_start, gap_end in gaps:
                ax.axvspan(gap_start, gap_end, alpha=0.2, color="gray", zorder=0)
        if boundary_on:
            for boundary in session_boundaries:
                ax.axvline(x=boundary, color="gray", linestyle="-", linewidth=0.8, alpha=0.6, zorder=3)

        ax.plot(xs, cumulative, color=colors[subj_idx], marker="o", markersize=3,
                label=f"Subject {subjid}")

    ax.set_xlabel("Trial number")
    ax.set_ylabel("Cumulative Rewards")
    # `orig=False` returns the processed ndarray. The default `orig=True` hands
    # back whatever was passed in, and `axvline` passes a two-element *list*, so
    # `.max()` raised `AttributeError` on every call that drew a session
    # boundary -- i.e. every multi-session call. Single-session calls draw no
    # boundary and worked, which is how it survived: measured 2026-08-18, no
    # `plot_regression` case reached this plotter at all until Item 1 added one.
    data_xmax = max(
        (line.get_xdata(orig=False).max() for line in ax.get_lines() if len(line.get_xdata())),
        default=ax.get_xlim()[1],
    )
    data_ymax = max(
        (line.get_ydata(orig=False).max() for line in ax.get_lines() if len(line.get_ydata())),
        default=ax.get_ylim()[1],
    )
    ax.set_xlim(left=0, right=data_xmax * 1.01)
    ax.set_ylim(bottom=-data_ymax * 0.01, top=data_ymax * 1.02)
    ax.legend()
    plt.tight_layout()

    if save:
        try:
            if isinstance(dates, dict):
                save_dates = []
                for v in dates.values():
                    if isinstance(v, (list, tuple)):
                        save_dates.extend(v)
                    elif v is not None:
                        save_dates.append(v)
            else:
                save_dates = dates
            out_path = save_figure(
                fig,
                "cumulative_rewards_by_trial",
                subjids=list(subjids) if isinstance(subjids, (list, tuple)) else [subjids],
                dates=save_dates,
            )
            if verbose:
                print(f"[plot_cumulative_rewards_by_trial] Saved figure to {out_path}")
        except Exception as exc:
            if verbose:
                print(f"[plot_cumulative_rewards_by_trial] Failed to save figure: {exc}")

    plt.show()

    return fig, ax
