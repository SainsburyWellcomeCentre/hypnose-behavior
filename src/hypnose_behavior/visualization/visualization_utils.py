# Defers evaluation of PEP-604 annotations (`X | None`), keeping this module
# importable on Python 3.9 for repos pinned there (hypnose-eeg-preprocessing).
from __future__ import annotations

import sys
import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib import cm
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from matplotlib.ticker import MaxNLocator
from collections import defaultdict
from typing import Iterable, Optional, Union, Tuple
import contextlib
import io
from hypnose_behavior.io.load_results import load_results_dir, load_session_results
from hypnose_behavior.metric_analysis.frames import (
    build_position_data,
    odor_letter,
    parse_json_column,
)
from hypnose_behavior.metric_analysis.metrics.accuracy import (
    decision_accuracy,
    global_choice_accuracy,
    rolling_reward_fraction,
)
from hypnose_behavior.metric_analysis.metrics.false_alarm import (
    FA_avg_response_times,
    fa_port_counts,
    fa_port_ratio,
    fa_port_share_a,
    fa_rate_by_odor,
    fa_rate_by_position,
)
from hypnose_behavior.metric_analysis.metrics.hidden_rule import (
    hidden_rule_mask,
    hr_abort_poke_gap,
    hr_odor_associations,
    rolling_hr_reward_fraction,
)
from hypnose_behavior.metric_analysis.metrics.sampling import (
    poke_duration_by_odor,
    poke_duration_by_position,
    poke_durations,
)
from hypnose_behavior.metric_analysis.metrics.sequence import abortion_rate_positionX
from hypnose_behavior.metric_analysis.metrics.timing import (
    avg_response_time,
    inter_trial_interval,
)
from hypnose_behavior.metric_analysis.resolvers import by_group
from hypnose_behavior.metric_analysis.run import (
    REGISTRY,
    _report_fa_abortion_stats,
    run_all_metrics,
)
from datetime import timedelta, datetime
from hypnose_behavior.trial_classification.classification_utils import load_all_streams, load_experiment
from hypnose_behavior.utils.helpers import (
    CACHE,
    _filter_session_dirs,
    _filter_sessions,
    _get_from_cache,
    _iter_subject_dirs,
    _update_cache,
    find_tracking_file,
    read_tracking_table,
)
from hypnose_behavior.io.layout import derivatives, list_sessions, normalize_subjid
from hypnose_behavior.io.paths import (
    get_data_root,
    get_rawdata_root,
    get_derivatives_root,
    get_server_root,
)
import re
import numpy as np
import json
from hypnose_behavior.io.save import save_figure
from hypnose_behavior.visualization.primitives import mean_sem, rolling_mean, sem_band
from hypnose_behavior.io.loaders import _load_table_with_trial_data, _load_trial_views, _odor_to_letter


def _clean_graph(ax, *, xlabel: Optional[str] = None, ylabel: Optional[str] = None):
    """
    Hide title, axis labels, tick labels, and legend while printing them for external editing.

    - Leaves ticks/spines in place so layout remains.
    - Prints axis labels and tick values to stdout for reference.
    """
    try:
        title = ax.get_title()
        x_lab = xlabel if xlabel is not None else ax.get_xlabel()
        y_lab = ylabel if ylabel is not None else ax.get_ylabel()
        x_ticks = [round(float(t), 3) for t in ax.get_xticks()]
        y_ticks = [round(float(t), 3) for t in ax.get_yticks()]
        if title:
            print(f"[_clean_graph] title: {title}")
        if x_lab:
            print(f"[_clean_graph] x label: {x_lab}")
        if y_lab:
            print(f"[_clean_graph] y label: {y_lab}")
        print(f"[_clean_graph] x ticks: {x_ticks}")
        print(f"[_clean_graph] y ticks: {y_ticks}")
    except Exception:
        pass

    # Clear visible annotations but keep ticks present
    ax.set_title("")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    leg = ax.get_legend()
    if leg is not None:
        try:
            leg.remove()
        except Exception:
            pass

def load_tracking_with_behavior(subjid, date):
    """
    Load combined tracking data and behavior results for a session, using cache if available.
    
    Parameters:
    -----------
    subjid : int
        Subject ID
    date : int or str
        Session date (e.g., 20251017)
    
    Returns:
    --------
    dict containing:
        - 'tracking': pd.DataFrame with tracking data (X, Y, time)
        - 'behavior': dict from load_session_results()
        - 'tracking_labeled': pd.DataFrame with added 'in_trial' column
    """
    # Try cache for full session data
    session_data = _get_from_cache(subjid, date, kind="session_data")
    if session_data is not None:
        print(f"[CACHE HIT] Session data for subjid={subjid}, date={date} loaded from cache.")
        tracking = session_data['tracking']
        behavior = session_data['behavior']
        tracking_labeled = session_data['tracking_labeled']
    else:
        print(f"[CACHE MISS] Loading session data for subjid={subjid}, date={date} from disk.")
        # Load tracking data from disk
        base_path = get_rawdata_root()
        server_root = get_server_root()
        derivatives_dir = get_derivatives_root()
        results_dir = derivatives.find_session(subjid, date=date).path / "saved_analysis_results"
        tracking_file = find_tracking_file(results_dir, "*_combined_tracking_with_timestamps")
        if tracking_file is None:
            # Fallback to SLEAP combined file
            tracking_file = find_tracking_file(results_dir, "*_combined_sleap_tracking_timestamps")
        if tracking_file is None:
            raise FileNotFoundError(
                f"No combined tracking file found. Run add_timestamps_to_tracking({subjid}, {date}) first."
            )
        tracking = read_tracking_table(tracking_file)
        # Normalize coordinate columns
        if 'X' not in tracking.columns and 'centroid_x' in tracking.columns:
            tracking['X'] = tracking['centroid_x']
        if 'Y' not in tracking.columns and 'centroid_y' in tracking.columns:
            tracking['Y'] = tracking['centroid_y']
        tracking['time'] = pd.to_datetime(tracking['time'])
        behavior = load_session_results(subjid, date)
        tracking_labeled = tracking.copy()
        tracking_labeled['in_trial'] = False
        tracking_labeled['trial_type'] = None
        trials = behavior.get('completed_sequences', pd.DataFrame())
        if not trials.empty:
            trials = trials.copy()
            trials['sequence_start'] = pd.to_datetime(trials['sequence_start'])
            trials['sequence_end'] = pd.to_datetime(trials['sequence_end'])
            for idx, trial in trials.iterrows():
                mask = (tracking_labeled['time'] >= trial['sequence_start']) & \
                       (tracking_labeled['time'] <= trial['sequence_end'])
                tracking_labeled.loc[mask, 'in_trial'] = True
                tracking_labeled.loc[mask, 'trial_type'] = trial.get('trial_type', 'trial')
        # Cache the full session data
        session_data = {
            'tracking': tracking,
            'behavior': behavior,
            'tracking_labeled': tracking_labeled
        }
        _update_cache(subjid, [date], {date: session_data}, kind="session_data")
    return {
        'tracking': tracking,
        'behavior': behavior,
        'tracking_labeled': tracking_labeled
    }

    # Utility function to print current cache keys
def print_cache_keys():
    print("[CACHE CONTENTS] Current cache keys:")
    for k in CACHE.keys():
        print(f"  {k}")

# Load metric results for visualization (NOTE: Previously in metrics_utils.py) ==============================================================================

def _extract_metric_value(metrics: dict, var_path: str):
    """
    Extract a numeric value from metrics dict given a dot-path.

    A navigator over a metrics mapping, not a reader of one: since Phase 5 the
    mapping comes from `_computed_metrics`, not from `metrics_*.json`. The
    dot-path is how a plot names a sub-entry of a metric ("avg_response_time.Rewarded"),
    so it survives the move to computing.

    Examples:
      - "decision_accuracy" -> uses the 3rd element (value) if tuple/list (num, denom, value)
      - "avg_response_time.Rewarded" -> nested dict lookup
    Returns float or np.nan if not found/unsupported.
    """
    try:
        parts = var_path.split(".")
        cur = metrics.get(parts[0], None)
        for p in parts[1:]:
            if isinstance(cur, dict):
                cur = cur.get(p, None)
            else:
                # unsupported path deeper into non-dict
                return float("nan")
        # Resolve final value
        if isinstance(cur, (int, float)) and not isinstance(cur, bool):
            return float(cur)
        if isinstance(cur, (list, tuple)) and len(cur) >= 3:
            # assume (numerator, denominator, value)
            val = cur[2]
            return float(val) if val is not None else float("nan")
        # Some dicts may hold numbers directly keyed by categories (needs explicit subkey in var_path)
        return float(cur) if isinstance(cur, (int, float)) else float("nan")
    except Exception:
        return float("nan")
    
def _load_protocol_from_summary(results_dir: Path) -> str:
    try:
        with open(results_dir / "summary.json", "r", encoding="utf-8") as f:
            summary = json.load(f)
        runs = summary.get("session", {}).get("runs", [])
        if runs and isinstance(runs, list):
            stage = runs[0].get("stage", {}) if isinstance(runs[0], dict) else {}
            name = stage.get("stage_name") or stage.get("name")
            return str(name) if name else "Unknown"
    except Exception:
        pass
    return "Unknown"

def _metric_name_for_key(key: str) -> Optional[str]:
    """Registry name for a `metrics_*.json` key. They differ in exactly one place,
    `hidden_rule_counts_by_odor`, always saved as `hidden_rule_by_odor`."""
    if key in REGISTRY:
        return key
    for name, spec in REGISTRY.items():
        if spec.key == key:
            return name
    return None


def _computed_metrics(results_dir: Path, keys: Iterable[str]) -> dict:
    """Compute the named metrics for a session, keyed and shaped exactly as
    `metrics_*.json` holds them.

    The compute-side replacement for `_ensure_metrics_json`. Plotters go through
    the registry rather than reading the saved file, which is an export and the
    record of an analysis run -- not a plotting input (`docs/DECISIONS.md`
    section 5). That deletes the staleness problem rather than managing it.

    `adapter(session(results))` is deliberately the *same expression* `run.py`
    uses to build the file, so what a plotter now computes and what would have
    been saved cannot drift apart by construction. The wrapper is used rather
    than the bare core because several cores take session configuration as
    keywords -- `hidden_rule_counts_by_odor` wants `hr_odors`/`hr_positions` --
    and knowing how to dig those out of `results` is precisely the wrapper's job.
    Its printing is suppressed: this asks for a value, not a report.
    """
    results = load_results_dir(results_dir)
    metrics = {}
    for key in keys:
        name = _metric_name_for_key(key)
        if name is None:
            continue
        spec = REGISTRY[name]
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            # `fa_abortion_stats` reports three tables rather than a value, so it
            # does not fit the wrapper -> adapter shape and `run.py` special-cases
            # it too. Calling the same builder keeps the shapes identical.
            if key == "fa_abortion_stats":
                metrics[key] = _report_fa_abortion_stats(results)
                continue
            value = spec.session(results)
        metrics[key] = spec.adapter(value) if spec.adapter else value
    return metrics


def _computed_metric(results_dir: Path, key: str):
    """One metric, in its saved-JSON shape. See `_computed_metrics`."""
    return _computed_metrics(results_dir, [key]).get(key)


# =========================================================== Metrics Plotting Functions =============================================================================

def _series_line_widths(show_mean: bool):
    """Shared per-line / mean-line widths so line-style plots (e.g.
    :func:`plot_behavior_metrics`, :func:`plot_decision_accuracy`) render with
    matching thickness.

    Returns ``(per_series_lw, mean_lw)``. When ``show_mean`` is True the
    individual lines are thinner and a thick mean line sits on top; when False
    there is no mean line and the individual lines are drawn thicker (a bit
    thicker than the with-mean group line). ``mean_lw`` is ``None`` when
    ``show_mean`` is False.
    """
    if show_mean:
        return 2.0, 4.0
    return 3.5, None


def plot_behavior_metrics(
    subjids: Optional[Iterable[int]] = None,
    dates: Optional[Union[Iterable[Union[int, str]], tuple]] = None,
    variables: Optional[Iterable[str]] = None,
    *,
    protocol_filter: Optional[str] = None,
    verbose: bool = True,
    black_white: bool = False,
    y_range: Optional[Tuple[float, float]] = None,
    plot_HR_separately: bool = False,
    mean: bool = False,
    show_title: bool = True,
    show_legend: bool = True,
    y_title: Optional[str] = None,
    lw_scale: float = 1.0,
    save: bool = False,
    return_paths: bool = False,
):
    """
    Plot selected metrics over sessions for one or more subjects.

    - X-axis: union of all available dates across selected subjects (categorical, no time gaps).
    - Y-axis: metric value.
    - One figure per variable.
    - Marker shape encodes subject; dot color encodes protocol; connecting lines are thin black.
    - Protocol filtering optional (substring match).
    - Values are computed through the metric registry, never read from
      metrics_*.json (`docs/DECISIONS.md` section 5).

    Parameters:
    - subjids: List of subject IDs to include, or None to include all subjects with matching dates.
        May also be a dict ``{subjid: date_range}`` as a shorthand — in that case
        the dict is used as ``dates`` and the subjids are its keys.
    - dates: List of specific dates (e.g., [20250101, 20250102]) or a date range
        (e.g., (20250101, 20250202)). Can also be a dict ``{subjid: date_range}``
        to give each subject its own date window (useful when subjects' training
        windows don't overlap). Subjids missing from the dict are skipped with a warning.
    - variables: List of metric names or dot-paths to plot.
    - protocol_filter: Optional substring to filter sessions by protocol.
    - verbose: If True, print progress and warnings.
    - y_range: Optional tuple (ymin, ymax); if provided, sets y-limits for each plot.
    - plot_HR_separately: If True and plotting hidden_rule_detection_rate, also plot per-HR-odor detection alongside total.
    - mean: If True, overlay a thick line showing the mean across subjects at each
        session index (per series). Individual lines are drawn thinner; line
        widths match plot_decision_accuracy via _series_line_widths.
    - show_title: If False, no plot title is rendered (useful for poster-style figures).
    - show_legend: If False, the subject / protocol / series legends are skipped.
    - y_title: If provided, overrides the default y-axis label (derived from the
        variable name); if None, the variable name is used as before.
    - lw_scale: Multiplier applied to every line width (per-series values keep
        their relative ratios). Default 1.0; use e.g. 3.0 for poster figures.
    - save: If True, save each figure as PDF via save_figure using subjids/dates to resolve the folder.
    - return_paths: If True and save=True, return (figs, paths); otherwise return figs.

    Returns:
    - List of matplotlib Figure objects, or (figs, paths) if return_paths is True.
    """
    if not variables:
        raise ValueError("Please provide `variables` (list of metric names or dot-paths).")

    # Resolve subjids + per-subject dates (mirrors plot_cumulative_rewards)
    if isinstance(subjids, dict):
        dates = subjids if not isinstance(dates, dict) or dates is None else dates
        subjids = list(subjids.keys())
    elif isinstance(dates, dict) and subjids is None:
        subjids = list(dates.keys())
    elif isinstance(subjids, set):
        subjids = sorted(subjids)
    elif subjids is not None and not isinstance(subjids, (list, tuple)):
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

    rows = []
    base_path = get_rawdata_root()
    server_root = get_server_root()
    derivatives_dir = get_derivatives_root()

    # Build (sid, subj_dir, subj_dates) list — supports per-subject windows
    if subjids is None:
        subject_iter = [(sid, sd, dates) for sid, sd in _iter_subject_dirs(derivatives_dir, None)]
    else:
        subject_iter = []
        for subjid in subjids:
            subj_dates = _dates_for(subjid)
            if isinstance(dates, dict) and subj_dates is None:
                if verbose:
                    print(f"Warning: No date range provided in dict for subject {subjid}, skipping")
                continue
            subj_str = normalize_subjid(subjid)
            subj_dir = derivatives.subject_dir(subjid, missing_ok=True)
            if subj_dir is None:
                if verbose:
                    print(f"Warning: No subject directory found for {subj_str}")
                continue
            subject_iter.append((int(subjid), subj_dir, subj_dates))

    # Gather sessions
    for sid, subj_dir, subj_dates in subject_iter:
        ses_dirs = _filter_session_dirs(subj_dir, subj_dates)
        for session_num, ses_dir in enumerate(ses_dirs, start=1):
            date_str = ses_dir.name.split("_date-")[-1]
            results_dir = ses_dir / "saved_analysis_results"
            if not results_dir.exists():
                continue
            # Protocol (for coloring)
            protocol = _load_protocol_from_summary(results_dir)
            if protocol_filter and (protocol is None or protocol_filter not in str(protocol)):
                continue
            # Load metrics (json or compute)
            # Computed through the registry rather than read from metrics_*.json
            # (`docs/DECISIONS.md` section 5). Only the roots actually requested
            # are evaluated, so a three-variable plot does not run all 25.
            wanted = {str(v).split(".")[0] for v in variables}
            if plot_HR_separately:
                wanted.add("hidden_rule_by_odor")
            try:
                metrics = _computed_metrics(results_dir, wanted)
            except Exception as e:
                if verbose:
                    print(f"[plot_behavior_metrics] Skipping sub-{sid:03d} {date_str}: {e}")
                continue
            for var in variables:
                val = _extract_metric_value(metrics, var)
                if isinstance(val, (int, float)) and not np.isnan(val):
                    rows.append({
                        "subjid": int(sid),
                        "session_num": session_num,
                        "date": int(date_str) if str(date_str).isdigit() else date_str,
                        "date_str": str(date_str),
                        "protocol": str(protocol) if protocol else "Unknown",
                        "variable": var,
                        "value": float(val),
                        "series": "Total",
                    })

                # If requested, add per-HR-odor detection rates
                if plot_HR_separately and var == "hidden_rule_detection_rate":
                    hr_block = metrics.get("hidden_rule_by_odor", {}) if isinstance(metrics, dict) else {}
                    by_odor = hr_block.get("by_odor", {}) if isinstance(hr_block, dict) else {}
                    for odor, stats in by_odor.items():
                        dr = None
                        if isinstance(stats, dict):
                            dr = stats.get("detection_rate")
                        if isinstance(dr, (int, float)) and not np.isnan(dr):
                            rows.append({
                                "subjid": int(sid),
                                "session_num": session_num,
                                "date": int(date_str) if str(date_str).isdigit() else date_str,
                                "date_str": str(date_str),
                                "protocol": str(protocol) if protocol else "Unknown",
                                "variable": var,
                                "value": float(dr),
                                "series": str(odor),
                            })

    if not rows:
        if verbose:
            print("[plot_behavior_metrics] No data found for the given filters.")
        return []

    df = pd.DataFrame(rows)
    if "series" not in df.columns:
        df["series"] = "Total"
    
    # Subject -> marker mapping
    markers_cycle = ['o', '^', 's', 'X', 'D', 'P', 'v', '>', '<', '*', 'h', 'H', '8', 'p', 'x']
    unique_subj = sorted(df["subjid"].unique())
    subj_to_marker = {sid: markers_cycle[i % len(markers_cycle)] for i, sid in enumerate(unique_subj)}

    # Protocol -> color mapping (or mono if black_white)
    prot_to_color = {}
    unique_protocols = []
    if not black_white:
        for p in df["protocol"]:
            if p not in unique_protocols and p and p != "Unknown":
                unique_protocols.append(p)
        if "Unknown" in df["protocol"].unique():
            unique_protocols.append("Unknown")
        cmap = cm.get_cmap("tab20", max(20, len(unique_protocols)))
        prot_to_color = {p: cmap(i % cmap.N) for i, p in enumerate(unique_protocols)}
        if "Unknown" in prot_to_color:
            prot_to_color["Unknown"] = (0.6, 0.6, 0.6, 1.0)

    figs = []
    saved_paths = []
    # One plot per variable
    for var in variables:
        df_var = df[df["variable"] == var]
        if df_var.empty:
            if verbose:
                print(f"[plot_behavior_metrics] No data for variable '{var}'.")
            continue

        fig, ax = plt.subplots(figsize=(12, 9))

        # Series handling: when plotting HR detection, allow per-odor series; otherwise single "Total"
        series_values = sorted(df_var["series"].unique()) if "series" in df_var.columns else ["Total"]

        # Define default palette for per-HR-odor series (C->B color, F->A color)
        odor_a_color = '#FF6B6B'  # same as odor A in plot_decision_accuracy_by_odor
        odor_b_color = '#4ECDC4'  # same as odor B in plot_decision_accuracy_by_odor

        def _map_series_color(series_label: str):
            lbl = str(series_label).lower()
            letters = [ch for ch in lbl if ch.isalpha()]
            last = letters[-1] if letters else ""
            if lbl == "total":
                return "black"
            if last == "f":  # HR odor F -> odor A color
                return odor_a_color
            if last == "c":  # HR odor C -> odor B color
                return odor_b_color
            if last == "a":
                return odor_a_color
            if last == "b":
                return odor_b_color
            return None

        hr_series_mode = (plot_HR_separately and var == "hidden_rule_detection_rate")

        # When not in HR-series mode and not black_white, color encodes SUBJECT
        # (tab20), and all markers collapse to a uniform dot.
        subject_colored_mode = (not hr_series_mode) and (not black_white)
        if subject_colored_mode:
            subj_cmap = cm.get_cmap("tab20")
            subj_to_color = {sid: subj_cmap(i % 20) for i, sid in enumerate(unique_subj)}
        else:
            subj_to_color = {}

        if hr_series_mode:
            # Always use colored series with solid lines; Total is thick black
            series_to_color = {s: (_map_series_color(s) or "black") for s in series_values}
            series_to_ls = {s: "-" for s in series_values}
            series_to_lw = {s: (3.0 if s == "Total" else 1.8) for s in series_values}
        elif black_white:
            series_to_color = {s: (0, 0, 0, 1.0) for s in series_values}
            linestyle_cycle = ["-", "--", ":", "-."]
            series_to_ls = {s: linestyle_cycle[i % len(linestyle_cycle)] for i, s in enumerate(series_values)}
            series_to_lw = {s: (2.5 if s == "Total" else 1.2) for s in series_values}
        else:
            series_cmap = cm.get_cmap("tab20", max(3, len(series_values)))
            series_to_color = {}
            for i, s in enumerate(series_values):
                mapped = _map_series_color(s)
                series_to_color[s] = mapped if mapped is not None else series_cmap(i % series_cmap.N)
            series_to_ls = {s: "-" for s in series_values}
            series_to_lw = {s: (2.5 if s == "Total" else 1.5) for s in series_values}

        # Line widths (shared with plot_decision_accuracy). Lines only, no dots.
        per_series_lw, mean_lw = _series_line_widths(mean)

        # Plot each series per subject
        for series in series_values:
            df_series = df_var[df_var.get("series", "Total") == series]
            for sid in unique_subj:
                dsub = df_series[df_series["subjid"] == sid].sort_values("session_num")
                if dsub.empty:
                    continue
                color = subj_to_color[sid] if subject_colored_mode else series_to_color.get(series, "black")
                ls = series_to_ls.get(series, "-")
                ax.plot(dsub["session_num"], dsub["value"], color=color, linestyle=ls,
                        linewidth=per_series_lw * lw_scale, alpha=0.8, zorder=1)

            # Mean across subjects at each session index (per series).
            if mean:
                grp = df_series.groupby("session_num")["value"].mean().sort_index()
                if not grp.empty:
                    ax.plot(grp.index, grp.values, color=series_to_color.get(series, "black"),
                            linestyle=series_to_ls.get(series, "-"),
                            linewidth=mean_lw * lw_scale, zorder=3)

        # X-axis: session numbers with sparse labels
        session_nums = sorted(df_var["session_num"].unique())
        n_sessions = len(session_nums)
        max_session = session_nums[-1] if session_nums else 0
        
        # Determine tick spacing (every 5-10 sessions)
        if n_sessions <= 10:
            tick_spacing = 2
        elif n_sessions <= 30:
            tick_spacing = 5
        elif n_sessions <= 80:
            tick_spacing = 10
        elif n_sessions <= 100:
            tick_spacing = 20
        else:
            tick_spacing = 50
        
        # Create x-axis ticks and labels
        x_ticks = [i for i in session_nums if i % tick_spacing == 0]
        ax.set_xticks(x_ticks)
        ax.set_xticklabels([str(i) for i in x_ticks])
        ax.set_xlim([0.5, max_session + 0.5])
        if y_range is not None and len(y_range) == 2:
            ax.set_ylim(y_range)
        y_ticks = ax.get_yticks()
        
        # Format title: split by "_" and capitalize each word
        title_formatted = " ".join(word.capitalize() for word in var.split(".")[0].split("_")) + (f" ({var.split('.')[1].capitalize()})" if '.' in var else "")

        
        # Axis labels always set; size/visibility is controlled by the active style.
        ax.set_xlabel("Day")
        ax.set_ylabel(y_title if y_title is not None else var.replace("_", " ").title())
        if show_title:
            ax.set_title(title_formatted)

        # Remove upper and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # No grid
        ax.grid(False)

        # Build legends: subjects (markers) and either protocols or series (colors)
        if subject_colored_mode:
            # Color encodes subject; uniform dot marker.
            subject_handles = [
                Line2D([0], [0],
                       marker='o',
                       color="black", linestyle="",
                       markerfacecolor=subj_to_color[sid],
                       markeredgecolor="black",
                       markersize=7,
                       label=f"sub-{sid:03d}")
                for sid in unique_subj
            ]
        else:
            subject_handles = [
                Line2D([0], [0],
                       marker=subj_to_marker[sid],
                       color="black", linestyle="",
                       markerfacecolor="white",
                       markeredgecolor="black",
                       markersize=7,
                       label=f"sub-{sid:03d}")
                for sid in unique_subj
            ]

        series_handles = []
        if len(series_values) > 1:
            for s in series_values:
                color = series_to_color.get(s, (0, 0, 0, 1))
                ls = series_to_ls.get(s, "-")
                lw = series_to_lw.get(s, 1.0)
                series_handles.append(
                    Line2D(
                        [0], [0],
                        marker='o',
                        color=color,
                        linestyle=ls,
                        linewidth=lw * lw_scale,
                        markerfacecolor='white' if black_white else color,
                        markeredgecolor="black",
                        markersize=7,
                        label=s,
                    )
                )

        protocol_handles = []
        # Skip protocol coloring when subject_colored_mode (colors encode subjects, not protocols).
        if not series_handles and not black_white and not subject_colored_mode:
            protocol_handles = [
                Line2D([0], [0],
                       marker='o',
                       color='none', linestyle="",
                       markerfacecolor=prot_to_color.get(p, (0, 0, 0, 1)),
                       markeredgecolor="black",
                       markersize=7,
                       label=p)
                for p in unique_protocols
            ]

        # Place legends
        if show_legend:
            if subject_handles:
                leg1 = ax.legend(handles=subject_handles, title="Subjects", loc="upper left", bbox_to_anchor=(1.02, 1.0))
                ax.add_artist(leg1)
            if series_handles:
                ax.legend(handles=series_handles, title="HR Series", loc="lower left", bbox_to_anchor=(1.02, 0.0))
            elif protocol_handles:
                ax.legend(handles=protocol_handles, title="Protocols", loc="lower left", bbox_to_anchor=(1.02, 0.0))

        plt.tight_layout()
        figs.append(fig)

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
                save_subjids = list(subjids) if subjids is not None else None
                save_name = f"{var}"
                out_path = save_figure(fig, save_name, subjids=save_subjids, dates=save_dates)
                saved_paths.append(out_path)
                if verbose:
                    print(f"[plot_behavior_metrics] Saved figure to {out_path}")
            except Exception as e:
                if verbose:
                    print(f"[plot_behavior_metrics] Failed to save figure for {var}: {e}")

    if return_paths and save:
        return figs, saved_paths
    return figs


# ---------------------------------------------------------------------------
# Shared odor colour scheme (used by hidden_rule_and_false_alarm and
# plot_poke_duration_by_odor so odors are coloured identically across plots).
#   A               -> bright red
#   B               -> teal/green
#   hidden-rule odor-> a lighter shade of the colour of the reward it is
#                      associated with (lighter red if it maps to reward A,
#                      lighter green if it maps to reward B)
#   any other odor  -> a distinct colour from a fixed palette (blue, yellow,
#                      orange, ...)
# The A/B association of each hidden-rule odor is not hard-coded: it is learned
# from the animal's hidden-rule sessions (conserved across sessions), so new
# odor sets work without code changes.
# ---------------------------------------------------------------------------
_ODOR_A_COLOR = "#E53935"   # bright red
_ODOR_B_COLOR = "#00796B"   # teal/green
_HR_A_COLOR = "#EF9A9A"     # lighter red  (HR odor associated with reward A)
_HR_B_COLOR = "#4DB6AC"     # lighter green (HR odor associated with reward B)
_OTHER_ODOR_COLORS = [      # distinct colours for non-A/B, non-HR odors
    "#1E88E5",  # blue
    "#FDD835",  # yellow
    "#FB8C00",  # orange
    "#8E24AA",  # purple
    "#00ACC1",  # cyan
    "#6D4C41",  # brown
]
# Colours for the pooled series that only plot_poke_duration_by_odor draws.
_POOLED_SERIES_COLORS = {
    "AB": "#AE05CF",     # magenta   (A+B pooled)
    "HR": "#FF0766",     # pink/rose (hidden-rule pair pooled)
    "OTHER": "#4D4C4B",  # dark grey (remaining odors pooled)
}


def _build_odor_colors(subj_dirs, odors_list) -> Tuple[dict, dict]:
    """Return ``({odor_letter: color}, {hr_odor_letter: 'A'|'B'})`` using the
    shared scheme: A=red, B=green, hidden-rule odor=lighter red/green by its
    learned A/B association, every other odor=a distinct palette colour."""
    assoc = hr_odor_associations(subj_dirs)
    colors: dict = {}
    other_i = 0
    for letter in odors_list:
        if letter == "A":
            colors[letter] = _ODOR_A_COLOR
        elif letter == "B":
            colors[letter] = _ODOR_B_COLOR
        elif letter in assoc:
            colors[letter] = _HR_A_COLOR if assoc[letter] == "A" else _HR_B_COLOR
        else:
            colors[letter] = _OTHER_ODOR_COLORS[other_i % len(_OTHER_ODOR_COLORS)]
            other_i += 1
    return colors, assoc


def hidden_rule_and_false_alarm(
    subjids=None,
    dates=None,
    odors=("C", "D", "E", "F", "G"),
    fa_label=None,
    figsize=(12, 9),
    title=None,
    *,
    save: bool = False,
    verbose: bool = True,
    show_title: bool = True,
    show_legend: bool = True,
    show_lines: bool = False,
    lw_scale: float = 1.0,
    marker_scale: float = 1.0,
):
    """Plot hidden-rule detection rate alongside per-odor false-alarm rate.

    ``show_lines`` additionally overlays each hidden-rule odor's own detection
    rate (from ``hidden_rule_by_odor`` in ``metrics_*.json``) as a line in that
    odor's colour, so the two contributions to the black mean line are visible.

    For each (subject, session) we produce:
    - The metric ``hidden_rule_detection_rate`` (from ``metrics_*.json``) — plotted in black.
    - One false-alarm rate per odor in ``odors``:

          fa_rate(odor) = #(aborted, fa_label-match, last_odor_name == odor)
                        --------------------------------------------------------------
                          #(aborted, fa_label-match, last_odor_name == odor)
                        + #(non-aborted, odor appears in odor_sequence)

      Each odor gets its own color (from the active style's prop_cycle).

    Across subjects, each subject is encoded by its marker shape (same convention
    as ``plot_behavior_metrics``).

    Parameters
    ----------
    subjids : int | list[int] | dict | None
        Subject id(s). May also be a dict ``{subjid: date_range}`` shorthand.
    dates : list | tuple | dict | None
        Date filter. As a dict, gives each subject its own window. Subjids
        missing from the dict are skipped with a warning.
    odors : iterable[str]
        Odor labels to include on the x-axis (case-insensitive).
    fa_label : list[str] | None
        Which aborted trials qualify as false alarms. None = all aborts except
        ``nFA`` (case-insensitive). Otherwise an explicit list, e.g.
        ``["FA_time_in", "FA_time_out"]``.
    show_title, show_legend : bool
        Toggle the title and legend (useful for poster figures).
    lw_scale, marker_scale : float
        Multipliers on line widths and marker sizes (poster scaling).
    save : bool
        Save via ``save_figure``; dict ``dates`` are flattened to a date span.

    Returns
    -------
    fig, ax
    """
    # Subject/date resolution (same pattern as plot_behavior_metrics / plot_cumulative_rewards)
    if isinstance(subjids, dict):
        dates = subjids if not isinstance(dates, dict) or dates is None else dates
        subjids = list(subjids.keys())
    elif isinstance(dates, dict) and subjids is None:
        subjids = list(dates.keys())
    elif isinstance(subjids, set):
        subjids = sorted(subjids)
    elif subjids is not None and not isinstance(subjids, (list, tuple)):
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

    derivatives_dir = get_derivatives_root()

    odors_list = [odor_letter(o) for o in odors]
    odors_set = set(odors_list)

    if subjids is None:
        subject_iter = [(sid, sd, dates) for sid, sd in _iter_subject_dirs(derivatives_dir, None)]
    else:
        subject_iter = []
        for subjid in subjids:
            subj_dates = _dates_for(subjid)
            if isinstance(dates, dict) and subj_dates is None:
                if verbose:
                    print(f"Warning: No date range provided in dict for subject {subjid}, skipping")
                continue
            subj_str = normalize_subjid(subjid)
            subj_dir = derivatives.subject_dir(subjid, missing_ok=True)
            if subj_dir is None:
                if verbose:
                    print(f"Warning: No subject directory found for {subj_str}")
                continue
            subject_iter.append((int(subjid), subj_dir, subj_dates))

    rows = []
    observed_hr_letters = set()
    for sid, subj_dir, subj_dates in subject_iter:
        ses_dirs = _filter_session_dirs(subj_dir, subj_dates)
        for session_num, ses_dir in enumerate(ses_dirs, start=1):
            date_str = ses_dir.name.split("_date-")[-1]
            results_dir = ses_dir / "saved_analysis_results"
            if not results_dir.exists():
                continue

            # Hidden rule detection rate, computed through the registry rather
            # than read from metrics_*.json (`docs/DECISIONS.md` section 5).
            try:
                metrics = _computed_metrics(
                    results_dir, ["hidden_rule_detection_rate", "hidden_rule_by_odor"])
            except Exception:
                metrics = None
            hr_val = None
            if metrics is not None:
                v = _extract_metric_value(metrics, "hidden_rule_detection_rate")
                if isinstance(v, (int, float)) and not np.isnan(v):
                    hr_val = float(v)
                hr_by_odor = metrics.get("hidden_rule_by_odor", {}) or {}
                for odor_name in hr_by_odor.get("hr_odors", []) or []:
                    letter = _odor_to_letter(odor_name)
                    if letter not in ("A", "B"):
                        observed_hr_letters.add(letter)
                for odor_name in (hr_by_odor.get("by_odor", {}) or {}).keys():
                    letter = _odor_to_letter(odor_name)
                    if letter not in ("A", "B"):
                        observed_hr_letters.add(letter)

            # Per-odor false alarm rate; odors with a zero denominator are omitted
            # by the metric rather than drawn as 0.
            td = _load_trial_views(results_dir)["trial_data"]
            rates = fa_rate_by_odor(td, fa_types=fa_label, odors=odors_list)
            for o, rate in rates.items():
                rows.append({
                    "subjid": int(sid),
                    "session_num": session_num,
                    "date_str": str(date_str),
                    "series": o,
                    "value": float(rate),
                })

            # Hidden rule rate; the value is stored in its native scale (typically 0-1).
            if hr_val is not None:
                rows.append({
                    "subjid": int(sid),
                    "session_num": session_num,
                    "date_str": str(date_str),
                    "series": "HR",
                    "value": hr_val if hr_val <= 1.0 else hr_val / 100.0,
                })

            # Per-hidden-rule-odor detection rate (show_lines): one series per HR
            # odor. These are performance lines, independent of the `odors`
            # (false-alarm) parameter — every genuine HR odor is included.
            if show_lines and metrics is not None:
                by_odor = (metrics.get("hidden_rule_by_odor", {}) or {}).get("by_odor", {}) or {}
                for odor_name, stats in by_odor.items():
                    letter = _odor_to_letter(odor_name)
                    if letter in ("A", "B"):
                        continue
                    dr = stats.get("detection_rate") if isinstance(stats, dict) else None
                    if not isinstance(dr, (int, float)) or np.isnan(dr):
                        continue
                    rows.append({
                        "subjid": int(sid),
                        "session_num": session_num,
                        "date_str": str(date_str),
                        "series": f"HRPERF_{letter}",
                        "value": float(dr) if dr <= 1.0 else float(dr) / 100.0,
                    })

    if not rows:
        if verbose:
            print("[hidden_rule_and_false_alarm] No data found for selected subjects/dates.")
        return None

    df = pd.DataFrame(rows)
    unique_subj = sorted(df["subjid"].unique())
    markers_cycle = ['o', '^', 's', 'X', 'D', 'P', 'v', '>', '<', '*', 'h', 'H', '8', 'p', 'x']
    subj_to_marker = {sid: markers_cycle[i % len(markers_cycle)] for i, sid in enumerate(unique_subj)}

    # Colors: shared odor scheme (A=red, B=green, HR odor=lighter red/green by
    # its learned reward association, other odors=distinct palette); the
    # hidden-rule detection series ("HR") is forced to black.
    series_order = list(odors_list) + ["HR"]
    # Per-HR-odor detection series (only present when show_lines added rows).
    hrperf_series = sorted(s for s in df["series"].unique() if str(s).startswith("HRPERF_"))
    series_order += hrperf_series
    # Colour over the odors AND any HR-performance odors (which may not be in
    # `odors`), so HR odors get their proper lighter-red/green regardless.
    hrperf_letters = [s.split("_", 1)[1] for s in hrperf_series]
    color_letters = list(odors_list) + [l for l in hrperf_letters if l not in odors_list]
    subj_dirs_for_colors = [t[1] for t in subject_iter]
    odor_colors, hr_assoc = _build_odor_colors(subj_dirs_for_colors, color_letters)
    series_color = dict(odor_colors)
    series_color["HR"] = "black"
    for s in hrperf_series:
        series_color[s] = odor_colors.get(s.split("_", 1)[1], "#000000")
    # Slightly thicker lines than before; the mean (HR) a bit more than the rest.
    series_lw = {s: (3.6 if s == "HR" else 2.4) for s in series_order}

    hidden_rule_letters = set(observed_hr_letters)
    hidden_rule_letters.update(hrperf_letters)
    hidden_rule_letters.update(l for l in color_letters if l in hr_assoc)
    hidden_rule_letters.discard("A")
    hidden_rule_letters.discard("B")
    hr_dash_cycle = [(0, (7, 3)), (0, (2, 2)), (0, (6, 2, 1, 2))]
    odor_linestyle = {}
    odor_alpha = {}
    for idx, letter in enumerate(sorted(hidden_rule_letters)):
        odor_linestyle[letter] = hr_dash_cycle[idx % len(hr_dash_cycle)]
        odor_alpha[letter] = 0.45
        series_color[letter] = "#000000"
        series_color[f"HRPERF_{letter}"] = "#000000"

    def _series_linestyle(series):
        if series == "HR":
            return "-"
        if str(series).startswith("HRPERF_"):
            letter = str(series).split("_", 1)[1]
            return odor_linestyle.get(letter, "--")
        return odor_linestyle.get(series, "-")

    def _series_alpha(series):
        if series == "HR":
            return 0.85
        if str(series).startswith("HRPERF_"):
            letter = str(series).split("_", 1)[1]
            return odor_alpha.get(letter, 0.45)
        return odor_alpha.get(series, 0.85)

    def _series_legend_label(series):
        if series == "HR":
            return "Hidden Rule"
        if str(series).startswith("HRPERF_"):
            letter = str(series).split("_", 1)[1]
            return f"Hidden Rule Odor {letter}"
        if series in hidden_rule_letters:
            return f"Hidden Rule Odor {series}"
        return f"Odor {series}"

    fig, ax = plt.subplots(figsize=figsize)
    ax2 = ax.twinx()

    for series in series_order:
        df_s = df[df["series"] == series]
        if df_s.empty:
            continue
        color = series_color[series]
        base_lw = series_lw[series]
        # HR mean and per-HR-odor detection lines live on the left (performance)
        # axis; the per-odor false-alarm rates on the right axis.
        target_ax = ax if (series == "HR" or str(series).startswith("HRPERF_")) else ax2
        for sid in unique_subj:
            d = df_s[df_s["subjid"] == sid].sort_values("session_num")
            if d.empty:
                continue
            target_ax.plot(
                d["session_num"], d["value"],
                color=color, linestyle=_series_linestyle(series),
                linewidth=base_lw * lw_scale,
                alpha=_series_alpha(series), zorder=1,
            )

    # X-axis tick spacing (sparse)
    session_nums = sorted(df["session_num"].unique())
    n_sessions = len(session_nums)
    if n_sessions:
        if n_sessions <= 10:
            tick_step = 2
        elif n_sessions <= 30:
            tick_step = 5
        else:
            tick_step = max(5, n_sessions // 10)
        ticks = [s for s in session_nums if (s - session_nums[0]) % tick_step == 0]
        ax.set_xticks(ticks)
        ax.set_xticklabels([str(t) for t in ticks])

    ax.set_xlabel("Day")
    ax.set_ylabel("Hidden Rule Performance")
    ax2.set_ylabel("False Alarm Rate")
    ax.set_ylim(0, 1.05)
    ax2.set_ylim(0, 1.05)

    if show_title:
        ax.set_title(title if title else "Hidden-rule detection & per-odor false-alarm rate")

    ax.spines['top'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(True)
    ax.grid(False)
    ax2.grid(False)

    if show_legend:
        series_handles = [
            Line2D([0], [0],
                color=series_color[s],
                linestyle=_series_linestyle(s),
                linewidth=series_lw[s] * lw_scale,
                label=_series_legend_label(s))
            for s in series_order if not str(s).startswith("HRPERF_")
        ]
        plotted_handle_keys = {
            s for s in series_order if not str(s).startswith("HRPERF_")
        }
        for letter in sorted(hidden_rule_letters):
            if letter in plotted_handle_keys:
                continue
            series = f"HRPERF_{letter}"
            series_handles.append(
                Line2D([0], [0],
                    color=series_color.get(series, "#000000"),
                    linestyle=_series_linestyle(series),
                    linewidth=series_lw.get(series, 2.4) * lw_scale,
                    label=_series_legend_label(series))
            )

        legend = ax.legend(
            handles=series_handles,
            title="Legend",
            loc="upper left",
            alignment="left",
        )

        legend.get_title().set_ha("left")

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
            save_subjids = list(subjids) if subjids is not None else None
            out_path = save_figure(
                fig, "hidden_rule_and_false_alarm",
                subjids=save_subjids, dates=save_dates,
            )
            if verbose:
                print(f"[hidden_rule_and_false_alarm] Saved figure to {out_path}")
        except Exception as exc:
            if verbose:
                print(f"[hidden_rule_and_false_alarm] Failed to save figure: {exc}")

    plt.show()
    return fig, ax


def plot_decision_accuracy_by_odor(
    subjid,
    dates=None,
    figsize=(12, 6),
    plot_choice_acc=False,
    plot_AB=True,
    clean_graph=False,
    *,
    save=False,
    verbose=True,
):
    """
    Plot decision accuracy by odor (A, B) and total over dates.
    Optionally include global choice accuracy as a separate line.
    Fast version using pre-computed metrics with existing helper functions.
    
    Parameters:
    -----------
    subjid : int
        Subject ID
    dates : tuple, list, or None
        Date or date range. If None, plots all available dates.
    figsize : tuple, optional
        Figure size (default: (12, 6))
    plot_choice_acc : bool, optional
        If True, also plot global choice accuracy as a dark grey line (default: False)
    plot_AB : bool, optional
        If True, plot odor-specific accuracies for A and B (default: True). If False, omit A/B lines.
    clean_graph : bool, optional
        If True, print and clear title/labels/ticks/legend for external editing.
    
    Returns:
    --------
    fig, ax : matplotlib figure and axes
    """
    rows = []
    base_path = get_rawdata_root()
    server_root = get_server_root()
    derivatives_dir = get_derivatives_root()
    
    def _normalize_odor_label(odor_raw):
        """Map assorted odor keys to canonical labels (A/B/Total/other)."""
        if isinstance(odor_raw, (int, float)) and not np.isnan(odor_raw):
            val = int(odor_raw)
            if val in (0, 1):
                return "A" if val == 0 else "B"
        if isinstance(odor_raw, str):
            raw = odor_raw.strip()
            lower = raw.lower()
            base = lower.replace("odor", "").replace("_", "").replace(" ", "")
            if base in {"a", "1", "01"}:
                return "A"
            if base in {"b", "2", "02"}:
                return "B"
            if lower in {"total", "overall"}:
                return "Total"
        return str(odor_raw)

    def _collect_odor_acc_rows(acc_block, date_int):
        """Handle both legacy flat dicts and new nested decision_accuracy_by_odor blocks."""
        collected = []

        def add_from_dict(dct):
            for odor, acc in dct.items():
                if isinstance(acc, (int, float)) and not np.isnan(acc):
                    collected.append({
                        "date": date_int,
                        "odor": _normalize_odor_label(odor),
                        "accuracy": float(acc)
                    })

        if not isinstance(acc_block, dict):
            return collected

        if "decision_accuracy_ab" in acc_block:
            add_from_dict(acc_block.get("decision_accuracy_ab", {}))
        if "decision_accuracy_total" in acc_block:
            add_from_dict(acc_block.get("decision_accuracy_total", {}))

        # If neither of the new-schema keys are present, assume legacy flat mapping
        if not collected:
            add_from_dict(acc_block)

        return collected

    for sid, subj_dir in _iter_subject_dirs(derivatives_dir, [subjid]):
        ses_dirs = _filter_session_dirs(subj_dir, dates)
        for ses_dir in ses_dirs:
            date_str = ses_dir.name.split("_date-")[-1]
            results_dir = ses_dir / "saved_analysis_results"
            if not results_dir.exists():
                continue
            
            # Computed through the registry, not read from metrics_*.json --
            # `decision_accuracy` was one of the three quantities this repo
            # obtained both ways (`docs/DECISIONS.md` section 5). `_computed_metric`
            # returns the saved key's exact shape, so `_collect_odor_acc_rows`
            # below is unchanged.
            td = _load_trial_views(results_dir).get("trial_data", pd.DataFrame())
            if td.empty:
                continue

            # Add odor-specific accuracies (supports legacy flat dict and new nested schema)
            rows.extend(_collect_odor_acc_rows(
                _computed_metric(results_dir, "decision_accuracy_by_odor") or {}, int(date_str)))

            # Add total accuracy
            acc_total = decision_accuracy(td)[2]
            if isinstance(acc_total, (int, float)) and not np.isnan(acc_total):
                rows.append({
                    "date": int(date_str),
                    "odor": "Total",
                    "accuracy": float(acc_total)
                })

            # Add global choice accuracy if requested
            if plot_choice_acc:
                gca_value = global_choice_accuracy(td)[2]
                if isinstance(gca_value, (int, float)) and not np.isnan(gca_value):
                    rows.append({
                        "date": int(date_str),
                        "odor": "Global Choice Accuracy",
                        "accuracy": float(gca_value)
                    })
    
    if not rows:
        print("No data found")
        return None, None
    
    df = pd.DataFrame(rows)
    unique_dates = sorted(df["date"].unique())
    date_to_x = {d: i for i, d in enumerate(unique_dates)}
    df["x"] = df["date"].map(date_to_x)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    colors = {'A': '#FF6B6B', 'B': '#4ECDC4', 'Total': 'black', 'Global Choice Accuracy': 'darkgreen'}
    linewidths = {'A': 1.5, 'B': 1.5, 'Total': 4, 'Global Choice Accuracy': 3.5}
    markers = {'A': 'o', 'B': 'o', 'Total': 's', 'Global Choice Accuracy': '^'}
    linestyles = {'A': '-', 'B': '-', 'Total': '-', 'Global Choice Accuracy': '--'}
    
    # Determine which odors to plot (restricted set)
    unique_odors = set(df["odor"].unique())
    odors_to_plot = []
    if plot_AB:
        for base in ["A", "B"]:
            if base in unique_odors:
                odors_to_plot.append(base)
    if "Total" in unique_odors:
        odors_to_plot.append("Total")
    if plot_choice_acc and "Global Choice Accuracy" in unique_odors:
        odors_to_plot.append("Global Choice Accuracy")
    
    for odor in odors_to_plot:
        subset = df[df["odor"] == odor]
        if subset.empty:
            continue
        ax.plot(subset["x"].values, subset["accuracy"].values, 
                label=odor,
                color=colors.get(odor, '#999999'),
                linewidth=linewidths.get(odor, 1.5),
                linestyle=linestyles.get(odor, '-'),
                marker=markers.get(odor, 'o'),
                markersize=4 if odor not in ('Total', 'Global Choice Accuracy') else 6,
                alpha=0.7 if odor not in ('Total', 'Global Choice Accuracy') else 0.8,
                zorder=10 if odor in ('Total', 'Global Choice Accuracy') else 1)
    
    ax.set_xlabel('Day')
    ax.set_ylabel('Accuracy')
    ax.set_ylim([0, 1.05])
    ax.set_xlim([-0.1, len(unique_dates) + 0.1])
    ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=1, alpha=0.3)
    ax.legend(loc='best')
        
    # Shift tick positions left by 1 while keeping labels unchanged (day 1 plotted at x=0)
    orig_xticks = ax.get_xticks()
    # Use existing tick labels if present; otherwise derive from original tick values
    existing_labels = [lbl.get_text() for lbl in ax.get_xticklabels()]
    labels = existing_labels if any(existing_labels) else [str(int(tick)) if tick.is_integer() else f"{tick:g}" for tick in orig_xticks]
    ax.set_xticks(orig_xticks - 1)
    ax.set_xticklabels(labels)
    # Re-affirm limits so the left edge stays near 0 despite shifted ticks
    ax.set_xlim([-0.1, len(unique_dates) - 1 + 0.1])

    # Remove upper and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    title = f"Subject {str(subjid).zfill(3)} - Decision Accuracy by Odor"
    if plot_choice_acc:
        title += " (with Global Choice Accuracy)"
    ax.set_title(title)
    
    if clean_graph:
        _clean_graph(ax, xlabel="Day", ylabel="Accuracy")

    plt.tight_layout()
    
    if save:
        try:
            suffix = "with_AB" if plot_AB else "total_only"
            if plot_choice_acc:
                suffix += "_choice"
            save_name = f"decision_accuracy_by_odor_{suffix}"
            out_path = save_figure(
                fig,
                save_name,
                subjids=[subjid],
                dates=dates,
            )
            if verbose:
                print(
                    f"[plot_decision_accuracy_by_odor] Saved figure to {out_path}"
                )
        except Exception as exc:
            if verbose:
                print(
                    "[plot_decision_accuracy_by_odor] Failed to save figure: "
                    f"{exc}"
                )
    
    return fig, ax

def plot_decision_accuracy_rolling_average(
    subjid,
    dates=None,
    save=False,
    window_size=20.0,
    step_size=1.0,
    include_avg=False,
    hr_only=False,
):
    """
    Plot rolling decision accuracy for one subject across one or more sessions.

    Creates two figures:
    1) Completed trials only (is_aborted == False)
    2) All trials

    Decision accuracy is computed as:
    (# trials in numerator condition) / (# trials in window)

    Numerator condition:
    - hr_only=False: response_time_category == "rewarded"
    - hr_only=True: response_time_category == "rewarded" AND hidden_rule_success == True

    Rolling windows are computed within each session only (no cross-session
    sharing). The plotted line remains continuous over global trial index.

    Parameters
    ----------
    subjid : int
        Subject ID.
    dates : tuple, list, or None
        Date range tuple, explicit list of dates, or None for all sessions.
    save : bool, optional
        If True, save both figures via save_figure().
    window_size : float
        Rolling window size in trials. Converted to int and clamped to >= 1.
    step_size : float
        Step size in trials between consecutive windows. Converted to int and
        clamped to >= 1. A larger step size reduces the number of plotted points.
    include_avg : bool, optional
        If True, fill early windows of each session using session-average padding:
        rate = (sum(available_data) + missing * session_avg) / window_size.
        If False (default), windows are plotted only when a full in-session window
        is available.
    hr_only : bool, optional
        If True, numerator counts only trials that are both rewarded and
        hidden_rule_success == True. Denominator remains unchanged.

    Returns
    -------
    (fig_completed, ax_completed, fig_all, ax_all)
        Matplotlib figures and axes for completed-only and all-trials views.
    """
    derivatives_dir = get_derivatives_root()
    window_n = max(1, int(window_size))
    step_n = max(1, int(step_size))

    # Collect per-session trial tables in chronological order.
    session_rows = []
    for sid, subj_dir in _iter_subject_dirs(derivatives_dir, [subjid]):
        ses_dirs = _filter_session_dirs(subj_dir, dates)
        for ses_dir in ses_dirs:
            date_str = ses_dir.name.split("_date-")[-1]
            results_dir = ses_dir / "saved_analysis_results"
            if not results_dir.exists():
                continue
            td = _load_table_with_trial_data(results_dir, "trial_data")
            if td.empty:
                continue

            td = td.copy()
            td["is_aborted"] = td.get("is_aborted", False).fillna(False)
            td["response_time_category"] = td.get("response_time_category", "").astype(str)

            # Prefer true trial-time ordering when available.
            if "sequence_start" in td.columns:
                td["sequence_start"] = pd.to_datetime(td["sequence_start"], errors="coerce")
                td = td.sort_values("sequence_start", na_position="last")
            elif "timestamp" in td.columns:
                td["timestamp"] = pd.to_datetime(td["timestamp"], errors="coerce")
                td = td.sort_values("timestamp", na_position="last")

            td = td.reset_index(drop=True)
            td["date"] = int(date_str) if str(date_str).isdigit() else date_str
            td["_session_uid"] = str(ses_dir.name)
            session_rows.append(td)

    if not session_rows:
        print("No data found")
        return None, None, None, None

    def _build_plot_df(session_tables, completed_only: bool) -> pd.DataFrame:
        pieces = []
        global_x_counter = 0

        for ses_df in session_tables:
            df = ses_df.copy()
            if completed_only:
                df = df[df["is_aborted"] == False].copy()
            df = df.reset_index(drop=True)
            if df.empty:
                continue

            rewarded_mask = (df["response_time_category"] == "rewarded")
            if hr_only:
                hr_mask = df.get("hidden_rule_success", False)
                if isinstance(hr_mask, pd.Series):
                    hr_mask = hr_mask.fillna(False).astype(bool)
                else:
                    hr_mask = pd.Series(False, index=df.index)
                numerator_mask = rewarded_mask & hr_mask
            else:
                numerator_mask = rewarded_mask

            df["is_rewarded"] = numerator_mask.astype(int)
            n_trials = len(df)

            # The windowing rule is `rolling_reward_fraction`, whose denominator
            # is the window rather than rewarded+unrewarded -- deliberately not
            # `over_windows(decision_accuracy, ...)`, which would draw a
            # different curve (audit finding 12).
            df["decision_accuracy"] = rolling_reward_fraction(
                df, window_n, step=step_n, include_avg=include_avg, hr_only=hr_only)

            if include_avg:
                # In include_avg mode, keep one x-unit per trial.
                x_local = np.arange(1, n_trials + 1)
            else:
                # Shift x so first valid full window of each session is at x=1.
                # Example window=30: point at trial 30 is displayed at session x=1.
                x_local = np.arange(1, n_trials + 1) - (window_n - 1)

            # Keep global trial index for debugging/reference.
            df["trial_idx"] = np.arange(1, n_trials + 1)

            # Display x-index used for plotting; session-wise compressed in standard mode.
            df["plot_x_idx"] = x_local + global_x_counter
            if include_avg:
                session_span = n_trials
            else:
                # Visual span equals number of possible full-window endpoints.
                session_span = max(1, n_trials - window_n + 1)
            global_x_counter += session_span
            pieces.append(df)

        if not pieces:
            return pd.DataFrame()
        return pd.concat(pieces, ignore_index=True)

    def _session_start_positions(plot_df: pd.DataFrame):
        if plot_df.empty or "_session_uid" not in plot_df.columns:
            return [], []
        valid = plot_df.dropna(subset=["decision_accuracy"])
        if valid.empty:
            return [], []
        starts = valid.groupby("_session_uid", sort=False)["plot_x_idx"].min().sort_values()
        return starts.values.tolist(), [str(s) for s in starts.index.tolist()]

    def _draw_plot(plot_df: pd.DataFrame, title: str):
        fig, ax = plt.subplots(figsize=(12, 6))

        if plot_df.empty:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
            return fig, ax

        # Plot each session separately so no line bridges session boundaries.
        if "_session_uid" in plot_df.columns:
            for _, ses_df in plot_df.groupby("_session_uid", sort=False):
                if ses_df.empty:
                    continue
                valid = ses_df.dropna(subset=["decision_accuracy"])
                if valid.empty:
                    continue
                ax.plot(
                    valid["plot_x_idx"].values,
                    valid["decision_accuracy"].values,
                    color="black",
                    linewidth=2.0,
                    alpha=0.9,
                )
        else:
            valid = plot_df.dropna(subset=["decision_accuracy"])
            if valid.empty:
                valid = plot_df
            ax.plot(
                valid["plot_x_idx"].values,
                valid["decision_accuracy"].values,
                color="black",
                linewidth=2.0,
                alpha=0.9,
            )

        start_x, _ = _session_start_positions(plot_df)
        if start_x:
            for i, x in enumerate(start_x):
                ax.axvline(
                    x=x,
                    color="#1f77b4",
                    linestyle=":",
                    linewidth=1.4,
                    alpha=0.9,
                    zorder=1,
                    label="Session start" if i == 0 else None,
                )

        ax.set_xlabel("Trials")
        ax.set_ylabel("Decision Accuracy")
        ax.set_ylim(0, 1.05)
        x_max = int(np.nanmax(plot_df["plot_x_idx"].values)) if not plot_df.empty else 1
        ax.set_xlim(1, max(x_max, 1))
        ax.set_title(title)
        ax.grid(False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if start_x:
            ax.legend(loc="lower right")

        plt.tight_layout()
        return fig, ax

    completed_df = _build_plot_df(session_rows, completed_only=True)
    all_df = _build_plot_df(session_rows, completed_only=False)

    mode_label = "include_avg" if include_avg else "standard"
    hr_label = "hr_only" if hr_only else "all_rewarded"

    fig_completed, ax_completed = _draw_plot(
        completed_df,
        f"Subject {str(subjid).zfill(3)} - Decision Accuracy Rolling Average (Completed Only, window={window_n}, step={step_n}, mode={mode_label}, numerator={hr_label})",
    )
    fig_all, ax_all = _draw_plot(
        all_df,
        f"Subject {str(subjid).zfill(3)} - Decision Accuracy Rolling Average (All Trials, window={window_n}, step={step_n}, mode={mode_label}, numerator={hr_label})",
    )

    if save:
        try:
            save_figure(
                fig_completed,
                f"decision_accuracy_rolling_average_completed_w{window_n}_s{step_n}_{mode_label}_{hr_label}",
                subjids=[subjid],
                dates=dates,
            )
        except Exception as exc:
            print(
                "[plot_decision_accuracy_rolling_average] Failed to save completed-only figure: "
                f"{exc}"
            )
        try:
            save_figure(
                fig_all,
                f"decision_accuracy_rolling_average_all_trials_w{window_n}_s{step_n}_{mode_label}_{hr_label}",
                subjids=[subjid],
                dates=dates,
            )
        except Exception as exc:
            print(
                "[plot_decision_accuracy_rolling_average] Failed to save all-trials figure: "
                f"{exc}"
            )

    return fig_completed, ax_completed, fig_all, ax_all

def plot_sampling_times_analysis(
    subjid,
    dates=None,
    figsize=(16, 18),
    *,
    save=False,
    verbose=True,
):
    """
    Plot sampling times (poke durations) by position and by odor for completed and aborted trials.

    Every number drawn here comes from `metric_analysis`: `poke_durations` for the
    scattered raw values, `poke_duration_by_{position,odor}` for the mean ± SD
    markers and for the per-session series in the bottom row. The two blob
    extractors this used to carry were finding 5 of the metric audit.
    """
    base_path = get_rawdata_root()
    server_root = get_server_root()
    derivatives_dir = get_derivatives_root()

    parts = []            # tidy poke durations, for the scatter panels
    pooled_positions = []  # position_data pooled over sessions, for the mean ± SD
    session_by_pos = []   # completed-trial session means, for panels 5 and 6
    session_by_odor = []

    for sid, subj_dir in _iter_subject_dirs(derivatives_dir, [subjid]):
        ses_dirs = _filter_session_dirs(subj_dir, dates)
        for session_num, ses_dir in enumerate(ses_dirs, start=1):
            date_str = ses_dir.name.split("_date-")[-1]
            results_dir = ses_dir / "saved_analysis_results"
            if not results_dir.exists():
                continue

            td = _load_trial_views(results_dir)["trial_data"]
            position_data = build_position_data(td)
            if position_data.empty:
                continue
            pooled_positions.append(position_data)
            date_val = int(date_str) if str(date_str).isdigit() else date_str

            for trial_type, aborted in (("completed", False), ("aborted", True)):
                pokes = poke_durations(position_data, aborted=aborted)
                if pokes.empty:
                    continue
                parts.append(pokes.rename(columns={"odor_name": "odor"}).assign(
                    trial_type=trial_type, session_num=session_num, date=date_val))

            by_pos = poke_duration_by_position(position_data)
            if not by_pos.empty:
                session_by_pos.append(by_pos.assign(session_num=session_num).reset_index())
            by_odor = poke_duration_by_odor(position_data)
            if not by_odor.empty:
                session_by_odor.append(by_odor.assign(session_num=session_num).reset_index())

    if not parts:
        print("No data found")
        return None, None

    df = pd.concat(parts, ignore_index=True)
    pooled = pd.concat(pooled_positions, ignore_index=True)
    comp_by_pos = poke_duration_by_position(pooled, aborted=False)
    abort_by_pos = poke_duration_by_position(pooled, aborted=True)
    comp_by_odor = poke_duration_by_odor(pooled, aborted=False)
    abort_by_odor = poke_duration_by_odor(pooled, aborted=True)

    # Create figure
    fig, axes = plt.subplots(3, 2, figsize=figsize)
    
    # ============ PLOT 1: Completed by Position ============
    ax = axes[0, 0]
    df_comp_pos = df[(df["trial_type"] == "completed") & (df["position"].notna())].copy()
    
    if not df_comp_pos.empty:
        positions = sorted(df_comp_pos["position"].unique())

        for pos in positions:
            values = df_comp_pos[df_comp_pos["position"] == pos]["poke_time_ms"].values

            # Scatter with jitter
            x_jitter = np.random.normal(pos, 0.04, size=len(values))
            ax.scatter(x_jitter, values, alpha=0.4, s=20, color='steelblue')

        means = [comp_by_pos.loc[pos, "mean"] for pos in positions]
        stds = [comp_by_pos.loc[pos, "sd"] for pos in positions]

        # Mean points with error bars (no line)
        ax.scatter(positions, means, color='darkred', s=100, zorder=5, marker='D', 
                  edgecolors='black', linewidth=1.5, label='Mean ± SD')
        ax.errorbar(positions, means, yerr=stds, fmt='none', ecolor='darkred', 
                   capsize=5, capthick=2, linewidth=2, zorder=4)
        ax.set_xticks(positions)
    
    ax.set_xlabel('Position')
    ax.set_ylabel('Poke Time (ms)')
    ax.set_title(f'Completed Trials: Sampling Time by Position\n(Subject {str(subjid).zfill(3)})')
    ax.legend(loc='best')
    
    # ============ PLOT 2: Aborted by Position ============
    ax = axes[0, 1]
    df_abort_pos = df[(df["trial_type"] == "aborted") & (df["position"].notna())].copy()
    
    if not df_abort_pos.empty:
        positions = sorted(df_abort_pos["position"].unique())

        for pos in positions:
            values = df_abort_pos[df_abort_pos["position"] == pos]["poke_time_ms"].values

            # Scatter with jitter
            x_jitter = np.random.normal(pos, 0.04, size=len(values))
            ax.scatter(x_jitter, values, alpha=0.4, s=20, color='coral')

        means = [abort_by_pos.loc[pos, "mean"] for pos in positions]
        stds = [abort_by_pos.loc[pos, "sd"] for pos in positions]

        # Mean points with error bars (no line)
        ax.scatter(positions, means, color='darkred', s=100, zorder=5, marker='D', 
                  edgecolors='black', linewidth=1.5, label='Mean ± SD')
        ax.errorbar(positions, means, yerr=stds, fmt='none', ecolor='darkred', 
                   capsize=5, capthick=2, linewidth=2, zorder=4)
        ax.set_xticks(positions)
    
    ax.set_xlabel('Position')
    ax.set_ylabel('Poke Time (ms)')
    ax.set_title(f'Aborted Trials: Sampling Time by Position\n(excl. abort position)')
    ax.legend(loc='best')
    
    # ============ PLOT 3: Completed by Odor ============
    ax = axes[1, 0]
    df_comp_odor = df[(df["trial_type"] == "completed") & (df["odor"].notna())].copy()
    
    if not df_comp_odor.empty:
        odors = sorted(df_comp_odor["odor"].unique())
        odor_to_x = {odor: i for i, odor in enumerate(odors)}

        for odor in odors:
            values = df_comp_odor[df_comp_odor["odor"] == odor]["poke_time_ms"].values
            x_pos = odor_to_x[odor]

            # Scatter with jitter
            x_jitter = np.random.normal(x_pos, 0.04, size=len(values))
            ax.scatter(x_jitter, values, alpha=0.4, s=20, color='steelblue')

        means = [comp_by_odor.loc[odor, "mean"] for odor in odors]
        stds = [comp_by_odor.loc[odor, "sd"] for odor in odors]

        # Mean points with error bars (no line)
        ax.scatter(range(len(odors)), means, color='darkred', s=100, zorder=5, marker='D', 
                  edgecolors='black', linewidth=1.5, label='Mean ± SD')
        ax.errorbar(range(len(odors)), means, yerr=stds, fmt='none', ecolor='darkred', 
                   capsize=5, capthick=2, linewidth=2, zorder=4)
        ax.set_xticks(range(len(odors)))
        ax.set_xticklabels(odors)
    
    ax.set_xlabel('Odor')
    ax.set_ylabel('Poke Time (ms)')
    ax.set_title(f'Completed Trials: Sampling Time by Odor\n(Subject {str(subjid).zfill(3)})')
    ax.legend(loc='best')
    
    # ============ PLOT 4: Aborted by Odor ============
    ax = axes[1, 1]
    df_abort_odor = df[(df["trial_type"] == "aborted") & (df["odor"].notna())].copy()
    
    if not df_abort_odor.empty:
        odors = sorted(df_abort_odor["odor"].unique())
        odor_to_x = {odor: i for i, odor in enumerate(odors)}

        for odor in odors:
            values = df_abort_odor[df_abort_odor["odor"] == odor]["poke_time_ms"].values
            x_pos = odor_to_x[odor]

            # Scatter with jitter
            x_jitter = np.random.normal(x_pos, 0.04, size=len(values))
            ax.scatter(x_jitter, values, alpha=0.4, s=20, color='coral')

        means = [abort_by_odor.loc[odor, "mean"] for odor in odors]
        stds = [abort_by_odor.loc[odor, "sd"] for odor in odors]

        # Mean points with error bars (no line)
        ax.scatter(range(len(odors)), means, color='darkred', s=100, zorder=5, marker='D', 
                  edgecolors='black', linewidth=1.5, label='Mean ± SD')
        ax.errorbar(range(len(odors)), means, yerr=stds, fmt='none', ecolor='darkred', 
                   capsize=5, capthick=2, linewidth=2, zorder=4)
        ax.set_xticks(range(len(odors)))
        ax.set_xticklabels(odors)
    
    ax.set_xlabel('Odor')
    ax.set_ylabel('Poke Time (ms)')
    ax.set_title(f'Aborted Trials: Sampling Time by Odor\n(excl. abort odor)')
    ax.legend(loc='best')
    
    # ============ PLOT 5: Average Poke Time per Position over Sessions ============
    ax = axes[2, 0]

    if session_by_pos:
        grouped = pd.concat(session_by_pos, ignore_index=True)
        positions = sorted(grouped["position"].unique())

        # Dark-to-light blue gradient for positions
        pos_palette = [
            '#0b3c68',  # dark
            '#155d8a',
            '#1f7eac',
            '#3c99c7',
            '#65b4d7',  # light
        ]

        for i, pos in enumerate(positions):
            pos_data = grouped[grouped["position"] == pos].sort_values("session_num")
            color = pos_palette[i % len(pos_palette)]
            ax.plot(pos_data["session_num"], pos_data["mean"],
                    label=f"Pos {pos}",
                    color=color,
                    linewidth=2.0,
                    marker="o",
                    markersize=5,
                    alpha=0.85)

        ax.set_xlabel("Session")
        ax.set_ylabel("Average Poke Time (ms)")
        ax.set_title(f'Average Poke Time per Position Across Sessions\n(Subject {str(subjid).zfill(3)})')
        ax.legend(loc='best')
    else:
        ax.text(0.5, 0.5, "No data", ha='center', va='center', transform=ax.transAxes)
        ax.set_axis_off()

    # ============ PLOT 6: Average Poke Time per Odor over Sessions ============
    ax = axes[2, 1]

    if session_by_odor:
        grouped = pd.concat(session_by_odor, ignore_index=True).rename(columns={"odor_name": "odor"})
        odors = sorted(grouped["odor"].unique())

        def _odor_color(odor_label: str):
            raw = str(odor_label).strip()
            lower = raw.lower()
            base = lower.replace("odor", "").replace("_", "").replace(" ", "")
            if base in {"a", "1", "01"}:
                return '#FF6B6B'
            if base in {"f"}:
                return '#E63946'  # slightly deeper red
            if base in {"b", "2", "02"}:
                return '#4ECDC4'
            if base in {"c", "3", "03"}:
                return '#1D9AB3'  # slightly deeper blue
            return '#888888'

        for odor in odors:
            odor_data = grouped[grouped["odor"] == odor].sort_values("session_num")
            ax.plot(odor_data["session_num"], odor_data["mean"],
                    label=str(odor),
                    color=_odor_color(odor),
                    linewidth=2.2,
                    marker="o",
                    markersize=5,
                    alpha=0.9)

        ax.set_xlabel("Session")
        ax.set_ylabel("Average Sampling Time (ms)")
        ax.set_title(f'Average Sampling Time per Odor Across Sessions\n(Subject {str(subjid).zfill(3)})')
        ax.legend(loc='best')
    else:
        ax.text(0.5, 0.5, "No data", ha='center', va='center', transform=ax.transAxes)
        ax.set_axis_off()

    plt.tight_layout()

    if save:
        panel_names = [
            "completed_by_position",
            "aborted_by_position",
            "completed_by_odor",
            "aborted_by_odor",
            "avg_position_over_sessions",
            "avg_odor_over_sessions",
        ]
        axes_flat = [axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1], axes[2, 0], axes[2, 1]]

        try:
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
        except Exception as exc:
            renderer = None
            if verbose:
                print(
                    "[plot_sampling_times_analysis] Unable to draw figure before saving: "
                    f"{exc}"
                )

        if renderer is not None:
            for ax_obj, name in zip(axes_flat, panel_names):
                if ax_obj is None:
                    continue
                try:
                    bbox = ax_obj.get_tightbbox(renderer)
                    if bbox is None:
                        continue
                    bbox = bbox.expanded(1.02, 1.08)
                    bbox_inches = bbox.transformed(fig.dpi_scale_trans.inverted())
                    save_name = f"sampling_times_analysis_{name}"
                    out_path = save_figure(
                        fig,
                        save_name,
                        subjids=[subjid],
                        dates=dates,
                        bbox_inches=bbox_inches,
                    )
                    if verbose:
                        print(
                            f"[plot_sampling_times_analysis] Saved subplot '{name}' to {out_path}"
                        )
                except Exception as exc:
                    if verbose:
                        print(
                            f"[plot_sampling_times_analysis] Failed to save subplot '{name}': {exc}"
                        )

    return fig, axes

def _fa_stat_count(item, key):
    """A count out of `fa_abortion_stats`, numeric or in the legacy string form.

    Phase 4b made the metric numeric (the audit's finding 3), but every
    `metrics_*.json` written before that holds `"5 (0.50)"`, and this reads
    those files rather than recomputing -- so both forms have to be understood
    until the whole tree has been re-analysed.
    """
    val = item.get(key)
    if isinstance(val, bool):
        return None
    if isinstance(val, (int, float)):
        return int(val)
    if isinstance(val, str) and val.strip():
        try:
            return int(val.split()[0])
        except (ValueError, IndexError):
            return None
    return None


def _fa_stat_rate(item, key):
    """A rate out of `fa_abortion_stats`; parses the legacy `"n/d (v)"` string.

    The parenthesised value is rounded to 2dp, so a legacy file is slightly
    coarser than a numeric one -- which is why the caller still prefers
    `"Abortion Rate Value"`, the exact number those files also carry.
    """
    val = item.get(key)
    if isinstance(val, bool):
        return None
    if isinstance(val, (int, float)):
        return float(val)
    if not isinstance(val, str):
        return None
    if "(" in val and ")" in val:
        try:
            return float(val.split("(")[-1].split(")")[0])
        except ValueError:
            pass
    if "/" in val:
        try:
            num_s, denom_s = val.split("/")[:2]
            denom = float(denom_s.split()[0].strip())
            return float(num_s.strip()) / denom if denom > 0 else None
        except (ValueError, IndexError):
            return None
    return None


def plot_abortion_and_fa_rates(
    subjid,
    dates=None,
    figsize=(18, 14),
    fa_types='FA_time_in',
    *,
    save=False,
    verbose=True,
):
    """
    Plot FA rates, abortion rates, and FA ratio by position and odor across sessions.
    
    Parameters:
    -----------
    subjid : int
        Subject ID
    dates : list, tuple, or None
        Dates to include
    figsize : tuple
        Figure size
    fa_types : str or list, optional
        Which FA types to include:
        - 'FA_Time_In' : only FA_Time_In
        - 'FA_Time_In,FA_Time_Out' : multiple specific types (comma-separated)
        - 'All' : all FA types starting with 'FA_'
        (default: 'FA_time_in')
    save : bool, optional
        If True, save each subplot as an individual PDF (default: False).
    verbose : bool, optional
        If True, print save status messages (default: True).
    """
    base_path = get_rawdata_root()
    server_root = get_server_root()
    derivatives_dir = get_derivatives_root()
    
    # DEFINE fa_filter_fn HERE - BEFORE THE LOOPS
    if isinstance(fa_types, str):
        if fa_types.lower() == 'all':
            fa_filter_fn = lambda x: x.astype(str).str.startswith('FA_', na=False)
        else:
            # Handle comma-separated list like 'FA_Time_In,FA_Time_Out'
            fa_type_list = [t.strip() for t in fa_types.split(',')]
            fa_filter_fn = lambda x: x.astype(str).isin(fa_type_list)
    elif isinstance(fa_types, list):
        fa_filter_fn = lambda x: x.astype(str).isin(fa_types)
    else:
        fa_filter_fn = lambda x: x.astype(str) == str(fa_types)
    
    rows = []
    fa_port_rows = []
    
    for sid, subj_dir in _iter_subject_dirs(derivatives_dir, [subjid]):
        ses_dirs = _filter_session_dirs(subj_dir, dates)
        for ses_dir in ses_dirs:
            date_str = ses_dir.name.split("_date-")[-1]
            results_dir = ses_dir / "saved_analysis_results"
            if not results_dir.exists():
                continue
            
            # Computed through the registry rather than read from metrics_*.json.
            # This plotter was the trap in `docs/DECISIONS.md` section 5: 4b made
            # `fa_abortion_stats` numeric, but every saved file still holds the
            # legacy `"3/10 (0.30)"` strings, so dropping the string parsing while
            # this still read those files would have made the plot draw nothing --
            # silently, because it skips what it cannot parse. Computing is what
            # discharges that, and it happens here in one step.
            try:
                metrics = _computed_metrics(results_dir, [
                    "fa_abortion_stats", "odorx_abortion_rate", "abortion_rate_positionX"])
            except Exception:
                metrics = {}

            fa_stats = metrics.get("fa_abortion_stats") or {}

            # FA rate per odor (FA Time In only)
            fa_by_odor = fa_stats.get("by_odor", [])
            if isinstance(fa_by_odor, list):
                for item in fa_by_odor:
                    if isinstance(item, dict) and "Odor" in item:
                        odor = item["Odor"]
                        total_ab = item.get("Total Abortions")
                        fa_time_in_count = _fa_stat_count(item, "FA Time In")
                        if fa_time_in_count is not None and total_ab:
                            rows.append({
                                "date": int(date_str),
                                "metric_type": "fa_rate",
                                "category": "odor",
                                "position_or_odor": str(odor),
                                "rate": fa_time_in_count / total_ab
                            })

            # FA rate per position (FA Time In only)
            fa_by_position = fa_stats.get("by_position", [])
            if isinstance(fa_by_position, list):
                for item in fa_by_position:
                    if isinstance(item, dict) and "Position" in item:
                        pos = item["Position"]
                        total_ab = item.get("Total Abortions")
                        fa_time_in_count = _fa_stat_count(item, "FA Time In")
                        if fa_time_in_count is not None and total_ab:
                            try:
                                pos_int = int(pos)
                            except (TypeError, ValueError):
                                continue
                            rows.append({
                                "date": int(date_str),
                                "metric_type": "fa_rate",
                                "category": "position",
                                "position_or_odor": pos_int,
                                "rate": fa_time_in_count / total_ab
                            })
            
            # ============ FA PORT RATIO - from trial_data aborted_fa ============
            try:
                views = _load_trial_views(results_dir)
                ab_det = views.get("aborted_fa", pd.DataFrame())
                if not ab_det.empty and "fa_label" in ab_det.columns:
                    ab_det = ab_det[fa_filter_fn(ab_det["fa_label"])]
                fa_all = ab_det.copy()

                if not fa_all.empty and {"fa_port", "last_odor_name"}.issubset(fa_all.columns):
                    for odor in sorted(fa_all["last_odor_name"].dropna().unique()):
                        fa_odor = fa_all[fa_all["last_odor_name"] == odor]
                        n_a, n_b = fa_port_counts(fa_odor)
                        n_total = n_a + n_b
                        ratio_a = fa_port_ratio(n_a, n_b)
                        fa_port_rows.append({
                            "date": int(date_str),
                            "odor": str(odor),
                            "fa_ratio_a": ratio_a
                        })
            except Exception:
                pass
            # ============ ABORTION RATES - prioritize fa_abortion_stats.by_position ============
            have_position_rates = False
            fa_by_position_full = fa_stats.get("by_position", [])
            if isinstance(fa_by_position_full, list):
                for item in fa_by_position_full:
                    if not isinstance(item, dict) or "Position" not in item:
                        continue
                    pos = item.get("Position")

                    # "Abortion Rate Value" first: legacy files carry both it and
                    # the string, and only it is exact -- the string's
                    # parenthesised value is rounded to 2dp. Numeric files (Phase
                    # 4b) have dropped it, and their "Abortion Rate" is exact.
                    rate_val = _fa_stat_rate(item, "Abortion Rate Value")
                    if rate_val is None:
                        rate_val = _fa_stat_rate(item, "Abortion Rate")
                    if rate_val is None:
                        rate_val = _fa_stat_rate(item, "FA Abortion Rate")

                    if rate_val is None:
                        continue

                    try:
                        rows.append({
                            "date": int(date_str),
                            "metric_type": "abortion_rate",
                            "category": "position",
                            "position_or_odor": int(pos),
                            "rate": float(rate_val)
                        })
                        have_position_rates = True
                    except Exception:
                        continue

            # Abortion rate per odor (fallback to legacy metrics fields if present)
            ab_odor_data = metrics.get("odorx_abortion_rate", {})
            if isinstance(ab_odor_data, dict):
                for odor, rate in ab_odor_data.items():
                    if rate is None or not isinstance(rate, (int, float)):
                        continue
                    rows.append({
                        "date": int(date_str),
                        "metric_type": "abortion_rate",
                        "category": "odor",
                        "position_or_odor": str(odor),
                        "rate": float(rate)
                    })

            # Abortion rate per position, only when `fa_abortion_stats` gave none
            # -- which is what this block's comment always claimed and the code
            # never did. It used to append a *second*, duplicate set of position
            # rows on every session; that stayed invisible only because JSON
            # stringifies dict keys, so `int("1.0")` raised and the bare `except`
            # below swallowed all of it. Computing the metric yields real float
            # keys, `int(1.0)` succeeds, and the duplicates become visible.
            ab_pos_data = metrics.get("abortion_rate_positionX", {}) if not have_position_rates else {}
            if isinstance(ab_pos_data, dict):
                for pos, rate in ab_pos_data.items():
                    if rate is None or not isinstance(rate, (int, float)):
                        continue
                    try:
                        rows.append({
                            "date": int(date_str),
                            "metric_type": "abortion_rate",
                            "category": "position",
                            "position_or_odor": int(pos),
                            "rate": float(rate)
                        })
                    except Exception:
                        continue
    
    if not rows:
        print("No data found")
        return None, None
    
    df = pd.DataFrame(rows)
    df_port = pd.DataFrame(fa_port_rows) if fa_port_rows else pd.DataFrame()
    
    # Create figure with 5 subplots (3 rows: top 2x2, bottom 1x2 centered)
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.3)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])
    ax5 = fig.add_subplot(gs[2, :])  # Spans full width
    
    axes = [ax1, ax2, ax3, ax4, ax5]
    panel_has_data = [False] * len(axes)
    
    # ============ PLOT 1: FA Rate per Position ============
    ax = ax1
    df_fa_pos = df[(df["metric_type"] == "fa_rate") & (df["category"] == "position")].copy()
    
    if not df_fa_pos.empty:
        panel_has_data[0] = True
        positions = sorted(df_fa_pos["position_or_odor"].unique())
        position_to_x = {pos: i for i, pos in enumerate(positions)}
        
        for pos in positions:
            rates = df_fa_pos[df_fa_pos["position_or_odor"] == pos]["rate"].values
            x_pos = position_to_x[pos]
            x_jitter = np.random.normal(x_pos, 0.04, size=len(rates))
            ax.scatter(x_jitter, rates, alpha=0.4, s=20, color='steelblue')
        
        _stats = [mean_sem(df_fa_pos[df_fa_pos["position_or_odor"] == pos]["rate"]) for pos in positions]
        means = [m for m, _ in _stats]
        sems = [e for _, e in _stats]
        
        ax.scatter(range(len(positions)), means, color='darkred', s=100, zorder=5, marker='D', 
                  edgecolors='black', linewidth=1.5, label='Mean ± SEM')
        ax.errorbar(range(len(positions)), means, yerr=sems, fmt='none', ecolor='darkred', 
                   capsize=5, capthick=2, linewidth=2, zorder=4)
        ax.set_xticks(range(len(positions)))
        ax.set_xticklabels(positions)
    
    ax.set_xlabel('Position')
    ax.set_ylabel('FA Rate')
    ax.set_title(f'FA Rate per Position\n(Subject {str(subjid).zfill(3)})')
    ax.set_ylim([0, 1.05])
    ax.legend(loc='best')
    
    # ============ PLOT 2: FA Rate per Odor ============
    ax = ax2
    df_fa_odor = df[(df["metric_type"] == "fa_rate") & (df["category"] == "odor")].copy()
    
    if not df_fa_odor.empty:
        panel_has_data[1] = True
        odors = sorted(df_fa_odor["position_or_odor"].unique())
        odor_to_x = {odor: i for i, odor in enumerate(odors)}
        
        for odor in odors:
            rates = df_fa_odor[df_fa_odor["position_or_odor"] == odor]["rate"].values
            x_pos = odor_to_x[odor]
            x_jitter = np.random.normal(x_pos, 0.04, size=len(rates))
            ax.scatter(x_jitter, rates, alpha=0.4, s=20, color='steelblue')
        
        _stats = [mean_sem(df_fa_odor[df_fa_odor["position_or_odor"] == odor]["rate"]) for odor in odors]
        means = [m for m, _ in _stats]
        sems = [e for _, e in _stats]
        
        ax.scatter(range(len(odors)), means, color='darkred', s=100, zorder=5, marker='D', 
                  edgecolors='black', linewidth=1.5, label='Mean ± SEM')
        ax.errorbar(range(len(odors)), means, yerr=sems, fmt='none', ecolor='darkred', 
                   capsize=5, capthick=2, linewidth=2, zorder=4)
        ax.set_xticks(range(len(odors)))
        ax.set_xticklabels(odors)
    
    ax.set_xlabel('Odor')
    ax.set_ylabel('FA Rate')
    ax.set_title(f'FA Rate per Odor\n(Subject {str(subjid).zfill(3)})')
    ax.set_ylim([0, 1.05])
    ax.legend(loc='best')
    
    # ============ PLOT 3: Abortion Rate per Position ============
    ax = ax3
    df_ab_pos = df[(df["metric_type"] == "abortion_rate") & (df["category"] == "position")].copy()
    
    if not df_ab_pos.empty:
        panel_has_data[2] = True
        positions = sorted(df_ab_pos["position_or_odor"].unique())
        
        for pos in positions:
            rates = df_ab_pos[df_ab_pos["position_or_odor"] == pos]["rate"].values
            x_jitter = np.random.normal(pos, 0.04, size=len(rates))
            ax.scatter(x_jitter, rates, alpha=0.4, s=20, color='coral')
        
        _stats = [mean_sem(df_ab_pos[df_ab_pos["position_or_odor"] == pos]["rate"]) for pos in positions]
        means = [m for m, _ in _stats]
        sems = [e for _, e in _stats]
        
        ax.scatter(positions, means, color='darkred', s=100, zorder=5, marker='D', 
                  edgecolors='black', linewidth=1.5, label='Mean ± SEM')
        ax.errorbar(positions, means, yerr=sems, fmt='none', ecolor='darkred', 
                   capsize=5, capthick=2, linewidth=2, zorder=4)
        ax.set_xticks(positions)
    
    ax.set_xlabel('Position')
    ax.set_ylabel('Abortion Rate')
    ax.set_title(f'Abortion Rate per Position\n(Subject {str(subjid).zfill(3)})')
    ax.set_ylim([0, 1.05])
    ax.legend(loc='best')
    
    # ============ PLOT 4: Abortion Rate per Odor ============
    ax = ax4
    df_ab_odor = df[(df["metric_type"] == "abortion_rate") & (df["category"] == "odor")].copy()
    
    if not df_ab_odor.empty:
        panel_has_data[3] = True
        odors = sorted(df_ab_odor["position_or_odor"].unique())
        odor_to_x = {odor: i for i, odor in enumerate(odors)}
        
        for odor in odors:
            rates = df_ab_odor[df_ab_odor["position_or_odor"] == odor]["rate"].values
            x_pos = odor_to_x[odor]
            x_jitter = np.random.normal(x_pos, 0.04, size=len(rates))
            ax.scatter(x_jitter, rates, alpha=0.4, s=20, color='coral')
        
        _stats = [mean_sem(df_ab_odor[df_ab_odor["position_or_odor"] == odor]["rate"]) for odor in odors]
        means = [m for m, _ in _stats]
        sems = [e for _, e in _stats]
        
        ax.scatter(range(len(odors)), means, color='darkred', s=100, zorder=5, marker='D', 
                  edgecolors='black', linewidth=1.5, label='Mean ± SEM')
        ax.errorbar(range(len(odors)), means, yerr=sems, fmt='none', ecolor='darkred', 
                   capsize=5, capthick=2, linewidth=2, zorder=4)
        ax.set_xticks(range(len(odors)))
        ax.set_xticklabels(odors)
    
    ax.set_xlabel('Odor')
    ax.set_ylabel('Abortion Rate')
    ax.set_title(f'Abortion Rate per Odor\n(Subject {str(subjid).zfill(3)})')
    ax.set_ylim([0, 1.05])
    ax.legend(loc='best')

    
    # ============ PLOT 5: FA Ratio (A-B) / (A+B) per Odor (full width) ============
    ax = ax5
    if not df_port.empty:
        panel_has_data[4] = True
        odors = sorted(df_port["odor"].unique())
        odor_to_x = {odor: i for i, odor in enumerate(odors)}
        
        for odor in odors:
            ratios = df_port[df_port["odor"] == odor]["fa_ratio_a"].values
            x_pos = odor_to_x[odor]
            x_jitter = np.random.normal(x_pos, 0.04, size=len(ratios))
            ax.scatter(x_jitter, ratios, alpha=0.4, s=20, color='steelblue')
        
        _stats = [mean_sem(df_port[df_port["odor"] == odor]["fa_ratio_a"]) for odor in odors]
        means = [m for m, _ in _stats]
        sems = [e for _, e in _stats]
        
        ax.scatter(range(len(odors)), means, color='darkred', s=100, zorder=5, marker='D', 
                  edgecolors='black', linewidth=1.5, label='Mean ± SEM')
        ax.errorbar(range(len(odors)), means, yerr=sems, fmt='none', ecolor='darkred', 
                   capsize=5, capthick=2, linewidth=2, zorder=4)
        ax.set_xticks(range(len(odors)))
        ax.set_xticklabels(odors)
    
    ax.set_xlabel('Odor')
    ax.set_ylabel('FA Ratio (A-B)/(A+B)')
    ax.set_title(f'FA Ratio (A-B)/(A+B) per Odor\n(Subject {str(subjid).zfill(3)})')
    ax.set_ylim([-1.1, 1.1])
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.7)
    ax.legend(loc='best')
    
    plt.tight_layout()

    saved_paths = []
    if save:
        panel_names = [
            "fa_rate_per_position",
            "fa_rate_per_odor",
            "abortion_rate_per_position",
            "abortion_rate_per_odor",
            "fa_ratio_per_odor",
        ]
        try:
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
        except Exception as exc:
            renderer = None
            if verbose:
                print(
                    "[plot_abortion_and_fa_rates] Unable to draw figure before saving: "
                    f"{exc}"
                )
        if renderer is not None:
            for ax, has_data, name in zip(axes, panel_has_data, panel_names):
                if not has_data:
                    continue
                try:
                    bbox = ax.get_tightbbox(renderer)
                    if bbox is None:
                        continue
                    bbox = bbox.expanded(1.02, 1.08)
                    bbox_inches = bbox.transformed(fig.dpi_scale_trans.inverted())
                    save_name = f"plot_abortion_and_fa_rates_{name}"
                    out_path = save_figure(
                        fig,
                        save_name,
                        subjids=[subjid],
                        dates=dates,
                        bbox_inches=bbox_inches,
                    )
                    saved_paths.append(out_path)
                    if verbose:
                        print(
                            f"[plot_abortion_and_fa_rates] Saved subplot '{name}' to {out_path}"
                        )
                except Exception as exc:
                    if verbose:
                        print(
                            f"[plot_abortion_and_fa_rates] Failed to save subplot '{name}': {exc}"
                        )

    return fig, axes

def plot_response_times_completed_vs_fa(
    subjid,
    dates=None,
    figsize=(12, 8),
    y_limit=20000,
    *,
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
    """
    base_path = get_rawdata_root()
    server_root = get_server_root()
    derivatives_dir = get_derivatives_root()
    
    rows = []
    
    for sid, subj_dir in _iter_subject_dirs(derivatives_dir, [subjid]):
        ses_dirs = _filter_session_dirs(subj_dir, dates)
        for ses_dir in ses_dirs:
            date_str = ses_dir.name.split("_date-")[-1]
            results_dir = ses_dir / "saved_analysis_results"
            if not results_dir.exists():
                continue

            # Both means come from the canonical metrics over trial_data. There
            # used to be a fall-back to metrics_*.json here, which made this one
            # of the three quantities obtainable two ways -- so two figures could
            # show it and disagree, since a saved JSON predates any later metric
            # change. `docs/DECISIONS.md` section 5 settles that on compute.
            # (The two paths did not even agree on the key: the JSON branch read
            # "Aborted FA Time In" where the metric returns "FA Time In".)
            td = _load_trial_views(results_dir).get("trial_data", pd.DataFrame())

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

def plot_fa_ratio_a_over_sessions(
    subjid,
    dates=None,
    figsize=(14, 10),
):
    """
    Plot FA Ratio A/(A+B) over sessions for each odor (OPTIMIZED).
    
    Parameters similar to original, but now loads only necessary data.
    """
    base_path = get_rawdata_root()
    server_root = get_server_root()
    derivatives_dir = get_derivatives_root()
    
    fa_data = {}  # {odor: [(session_num, ratio, n_a, n_b, n_total), ...]}
    
    for sid, subj_dir in _iter_subject_dirs(derivatives_dir, [subjid]):
        ses_dirs = _filter_session_dirs(subj_dir, dates)
        
        for session_num, ses_dir in enumerate(ses_dirs, start=1):
            date_str = ses_dir.name.split("_date-")[-1]
            results_dir = ses_dir / "saved_analysis_results"
            if not results_dir.exists():
                continue
            
            views = _load_trial_views(results_dir)
            ab_det = views["aborted_fa"]
            if not ab_det.empty:
                needed_cols = ['fa_label', 'last_odor_name', 'fa_port']
                ab_det = ab_det[[col for col in needed_cols if col in ab_det.columns]]

            if ab_det.empty:
                continue

            try:
                if 'fa_label' not in ab_det.columns:
                    continue
                fa_all = ab_det[ab_det['fa_label'].astype(str) == 'FA_time_in']
                if fa_all.empty:
                    continue
            except Exception as e:
                continue
            
            if fa_all.empty or 'fa_port' not in fa_all.columns or 'last_odor_name' not in fa_all.columns:
                continue
            
            # Calculate FA port ratio per odor
            try:
                for odor in sorted(fa_all['last_odor_name'].dropna().unique()):
                    fa_odor = fa_all[fa_all['last_odor_name'] == odor]
                    n_a, n_b = fa_port_counts(fa_odor)
                    n_total = n_a + n_b
                    ratio_a = fa_port_share_a(n_a, n_b)

                    if odor not in fa_data:
                        fa_data[odor] = []
                    fa_data[odor].append({
                        'session_num': session_num,
                        'date': int(date_str),
                        'ratio_a': ratio_a,
                        'n_a': n_a,
                        'n_b': n_b,
                        'n_total': n_total
                    })
            except Exception as e:
                continue
    
    if not fa_data:
        print("No FA data found")
        return {}
    
    # Create one figure per odor
    figs = {}
    odor_list = sorted(fa_data.keys())
    
    for odor in odor_list:
        data = fa_data[odor]
        data = sorted(data, key=lambda x: x["session_num"])
        
        x_positions = np.arange(len(data))
        session_nums = [d["session_num"] for d in data]
        ratios = [d["ratio_a"] for d in data]
        n_a_list = [d["n_a"] for d in data]
        n_total_list = [d["n_total"] for d in data]
        
        fig, ax = plt.subplots(figsize=figsize)
        
        ax.plot(x_positions, ratios, color='black', linewidth=1.0, alpha=0.6, zorder=1)
        ax.scatter(x_positions, ratios, s=40, color='black', alpha=0.8, 
                  edgecolors='black', linewidth=0.5, zorder=3)
        
        ax.axhline(y=0.5, color='#888888', linestyle='--', linewidth=1.0, alpha=0.5, zorder=0)
        
        y_text = 1.08
        for x_pos, n_a, n_total in zip(x_positions, n_a_list, n_total_list):
            ax.text(x_pos, y_text, f"{n_a}/{n_total}", 
                   ha='center', va='bottom', fontsize=9, fontweight='bold',
                   transform=ax.get_xaxis_transform())
        
        ax.set_xlim([-0.5, len(data) - 0.5])
        ax.set_xticks(x_positions)
        ax.set_xticklabels([str(sn) for sn in session_nums])
        
        ax.set_xlabel('Session Number', fontsize=12, fontweight='bold')
        ax.set_ylabel('FA Ratio A / (A+B)', fontsize=12, fontweight='bold')
        ax.set_title(f'FA Ratio Odor {odor}\n(Subject {str(subjid).zfill(3)})',
                    fontsize=13, fontweight='bold')
        ax.set_ylim([0, 1.0])
        ax.grid(False)
        
        plt.tight_layout()
        figs[odor] = fig
    
    return figs


# =========================================================== Movement & Behavior Plotting Functions ================================================================


def plot_cumulative_rewards(
    subjids,
    dates,
    split_days=False,
    figsize=(12, 6.5),
    title=None,
    *,
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
    """
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
        ses_dirs = _filter_session_dirs(subj_dir, subj_dates)
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
    """
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
        for ses_dir in _filter_session_dirs(subj_dir, subj_dates):
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
    data_xmax = max(
        (line.get_xdata().max() for line in ax.get_lines() if len(line.get_xdata())),
        default=ax.get_xlim()[1],
    )
    data_ymax = max(
        (line.get_ydata().max() for line in ax.get_lines() if len(line.get_ydata())),
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


def _coerce_tz_naive(series):
    """Return a datetime Series with any timezone dropped (subtraction-safe)."""
    s = pd.to_datetime(series, errors="coerce")
    try:
        if s.dt.tz is not None:
            s = s.dt.tz_localize(None)
    except (AttributeError, TypeError):
        pass
    return s


def _load_subject_trial_timeline(subjid, subj_dates):
    """Build a per-trial timeline for one subject across sessions.

    Returns a dict with the concatenated all-trial DataFrame (chronological,
    with ``time_seconds`` collapsing inter-session gaps as in
    ``plot_cumulative_rewards`` and a continuous ``trial_index`` as in
    ``plot_cumulative_rewards_by_trial``), plus ``iti_seconds`` (within-session
    inter-trial interval) and ``is_rewarded``. Also returns the time- and
    trial-axis gap spans and session boundaries. ``None`` if no data.
    """
    derivatives_dir = get_derivatives_root()
    subj_str = normalize_subjid(subjid)
    subj_dir = derivatives.subject_dir(subjid, missing_ok=True)
    if subj_dir is None:
        print(f"Warning: No subject directory found for {subj_str}")
        return None

    sessions = []
    for ses_dir in _filter_session_dirs(subj_dir, subj_dates):
        date_str = ses_dir.name.split("_date-")[-1]
        results_dir = ses_dir / "saved_analysis_results"
        if results_dir.exists():
            sessions.append((date_str, results_dir))
    sessions.sort(key=lambda t: t[0])

    per_session = []
    for date_str, results_dir in sessions:
        df = _load_trial_views(results_dir).get("trial_data", pd.DataFrame())
        if df.empty or "global_trial_id" not in df.columns:
            continue
        df = df.sort_values("global_trial_id").reset_index(drop=True)
        df["date"] = date_str
        for col in ("sequence_start", "sequence_end"):
            if col in df.columns:
                df[col] = _coerce_tz_naive(df[col])
        # Computed per session, never over the concatenation: shifting across a
        # session boundary would measure the gap between recordings.
        df["iti_seconds"] = inter_trial_interval(df)
        rtc = df.get("response_time_category")
        df["is_rewarded"] = (rtc == "rewarded") if rtc is not None else False
        runs = []
        summary_path = results_dir / "summary.json"
        if summary_path.exists():
            try:
                with open(summary_path, "r", encoding="utf-8") as f:
                    runs = json.load(f).get("session", {}).get("runs", [])
            except Exception:
                runs = []
        per_session.append({"date": date_str, "df": df, "runs": runs})

    if not per_session:
        return None

    # Continuous trial axis + trial-based gaps/boundaries.
    running = 0
    trial_boundaries = []
    trial_gaps = []
    for s_idx, sess in enumerate(per_session):
        df = sess["df"]
        n = len(df)
        if s_idx > 0:
            trial_boundaries.append(running + 0.5)
        if "run_id" in df.columns:
            run_ids = df["run_id"].tolist()
            for k in range(1, n):
                if run_ids[k] != run_ids[k - 1]:
                    trial_gaps.append((running + k, running + k + 1))
        df["trial_index"] = range(running + 1, running + n + 1)
        running += n

    combined = pd.concat([s["df"] for s in per_session], ignore_index=True)

    # Time axis with collapsed inter-session gaps (mirrors plot_cumulative_rewards).
    time_boundaries = []
    time_gaps = []
    session_info = [{"date": s["date"], "runs": s["runs"]} for s in per_session if s["runs"]]
    if session_info:
        time_offset = 0
        session_offsets = {}
        global_start_time = None
        for sess_idx, sess in enumerate(session_info):
            runs = sess["runs"]
            sdate = sess["date"]
            if sess_idx == 0:
                global_start_time = _coerce_tz_naive(pd.Series([runs[0]["start_time"]])).iloc[0]
                session_offsets[sdate] = 0
            else:
                prev = session_info[sess_idx - 1]
                prev_end = _coerce_tz_naive(pd.Series([prev["runs"][-1]["end_time"]])).iloc[0]
                curr_start = _coerce_tz_naive(pd.Series([runs[0]["start_time"]])).iloc[0]
                time_offset += (curr_start - prev_end).total_seconds() - 1
                session_offsets[sdate] = time_offset
                boundary = (prev_end - global_start_time).total_seconds() - session_offsets[prev["date"]] + 1
                time_boundaries.append(boundary)
            for run in runs:
                if run.get("gap_to_next_run"):
                    try:
                        parts = str(run["gap_to_next_run"]).split(":")
                        if len(parts) == 3:
                            gap_dur = int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
                        else:
                            gap_dur = float(run["gap_to_next_run"])
                        run_end = _coerce_tz_naive(pd.Series([run["end_time"]])).iloc[0]
                        gap_start = (run_end - global_start_time).total_seconds() - session_offsets[sdate]
                        time_gaps.append((gap_start, gap_start + gap_dur))
                    except Exception:
                        pass
        combined["time_seconds"] = combined.apply(
            lambda row: (row["sequence_start"] - global_start_time).total_seconds()
            - session_offsets.get(row["date"], 0),
            axis=1,
        )
    else:
        global_start_time = combined["sequence_start"].iloc[0]
        combined["time_seconds"] = (combined["sequence_start"] - global_start_time).dt.total_seconds()

    return {
        "combined": combined,
        "time_gaps": time_gaps,
        "time_boundaries": time_boundaries,
        "trial_gaps": trial_gaps,
        "trial_boundaries": trial_boundaries,
    }


def _rolling_median_iqr(x, y, window_size, step_size):
    """Rolling median + 25th/75th percentiles over trial windows.

    ``x``/``y`` are equal-length arrays already ordered by x. Returns
    (mx, median, q25, q75); each window is anchored to the actual x value of its
    center trial (not an interpolated midpoint between the first and last x).
    """
    n = len(y)
    if n == 0:
        return (np.array([]),) * 4
    win = min(max(1, int(window_size)), n)
    step = max(1, int(step_size))
    mx, med, q25, q75 = [], [], [], []
    for end in range(win, n + 1, step):
        start = end - win
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
    plain cast raised ``ValueError`` in any process that had not applied the
    repo style first, i.e. a bare notebook or script (restructure_2 Phase 5).
    Under the style the key is numeric and both forms give the same number, so
    this widens what works without moving a drawn figure.
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
        timeline = _load_subject_trial_timeline(subjid, subj_dates)
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
    """
    return _plot_metric_over_sessions(
        subjids, dates, value_col="response_time_ms", metric_name="Response Time",
        unit="ms", rewarded_only=rewarded_only, window_size=window_size, step_size=step_size,
        save=save, save_key="latency_over_time", verbose=verbose,
        show_gap_shading=show_gap_shading, show_session_boundaries=show_session_boundaries,
        figsize=figsize, show_iqr=show_iqr,
    )


def plot_iti_over_time(
    subjids,
    dates=None,
    *,
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
    """
    return _plot_metric_over_sessions(
        subjids, dates, value_col="iti_seconds", metric_name="ITI", unit="s",
        rewarded_only=False, window_size=window_size, step_size=step_size,
        save=save, save_key="iti_over_time", verbose=verbose,
        show_gap_shading=show_gap_shading, show_session_boundaries=show_session_boundaries,
        figsize=figsize, show_iqr=show_iqr
    )


def plot_choice_history(
    subjid,
    dates=None,
    figsize=(16, 8),
    title=None,
    xlim=None,
    fa_types=("FA_time_in", "FA_time_out"),
    *,
    save=False,
    verbose=True,
    show_legend=True,
    lw_scale: float = 1.0,
    marker_scale: float = 1.0,
):
    """
    Plot choice history over time for one or more sessions.
    
    - Y-axis: Choice direction (A=red up, B=blue down)
    - X-axis: Time
    - Rewarded trials: solid line with circle marker at end
    - Completed unrewarded trials: dotted line, no marker
    - Aborted trials: grey line going up
    - Hidden Rule trials: yellow line
      - HR rewarded: solid yellow line with circle marker
      - HR missed/unrewarded: dotted yellow line, no marker
    - Multiple sessions: time gaps collapsed, session boundaries marked with grey dotted lines
    
    Parameters:
    -----------
    subjid : int
        Subject ID
    dates : list, tuple, or None
        Specific dates [20250101, 20250102] or range (20250101, 20250110)
        If None, plots all available dates
    figsize : tuple, optional
        Figure size (default: (16, 8))
    title : str, optional
        Plot title. If None, uses default
    xlim : tuple, optional
        X-axis limits in plot time seconds. If None, uses the full trial range
        with automatic padding.
    fa_types : str | Iterable[str], optional
        FA labels to highlight on aborted trials. Matching is case-insensitive.
        Defaults to ("FA_time_in", "FA_time_out").
    save : bool, optional
        If True, save the generated figure via save_figure (default False).
    verbose : bool, optional
        If True, print save status messages (default True).
    
    Returns:
    --------
    fig, ax : matplotlib figure and axis objects
    """
    base_path = get_rawdata_root()
    server_root = get_server_root()
    derivatives_dir = get_derivatives_root()

    # Normalize FA filter labels
    if isinstance(fa_types, str):
        fa_set = {s.strip().lower() for s in re.split(r"[;,]", fa_types) if s.strip()}
    else:
        fa_set = {str(s).strip().lower() for s in fa_types} if fa_types is not None else set()
    
    subject_dir = derivatives.subject_dir(subjid)
    
    # Get session directories
    ses_dirs = _filter_session_dirs(subject_dir, dates)
    if not ses_dirs:
        raise FileNotFoundError(f"No sessions found for subject {subjid} with given dates")
    
    # Collect all trials across sessions
    all_trials = []
    
    for session_idx, ses_dir in enumerate(ses_dirs):
        date_str = ses_dir.name.split("_date-")[-1]
        results_dir = ses_dir / "saved_analysis_results"
        if not results_dir.exists():
            continue

        # Prefer trial_data views (new schema); fallback to legacy load_session_results tables
        views = _load_trial_views(results_dir)

        def _get_odor(row):
            for cand in ["last_odor_name", "last_odor", "odor", "odor_name"]:
                if cand in row and pd.notna(row[cand]):
                    return row[cand]
            return "Unknown"

        def _append_trials(df, trial_type, is_hr=False):
            if df.empty or "sequence_start" not in df.columns:
                return
            for _, r in df.iterrows():
                all_trials.append({
                    "sequence_start": pd.to_datetime(r["sequence_start"]),
                    "last_odor": _get_odor(r),
                    "trial_type": trial_type,
                    "is_hr": is_hr,
                    "date_str": date_str,
                    "session_idx": session_idx,
                    "abortion_time": pd.to_datetime(r.get("abortion_time"), errors="coerce") if "abortion_time" in r else pd.NaT,
                    "fa_port": r.get("fa_port") if "fa_port" in r else np.nan,
                    "fa_label": str(r.get("fa_label", "")).strip().lower(),
                })

        if not views["trial_data"].empty:
            comp = views.get("completed", pd.DataFrame())
            aborted = views.get("aborted", pd.DataFrame())
            aborted_hr = views.get("aborted_hr", pd.DataFrame())

            rewarded = comp[comp.get("response_time_category", "") == "rewarded"] if not comp.empty else pd.DataFrame()
            unrewarded = comp[comp.get("response_time_category", "") == "unrewarded"] if not comp.empty else pd.DataFrame()
            timeout = comp[comp.get("response_time_category", "") == "timeout_delayed"] if not comp.empty else pd.DataFrame()

            # Use hidden_rule_success when available; fall back to hit_hidden_rule
            hr_flag = "hidden_rule_success" if "hidden_rule_success" in comp.columns else "hit_hidden_rule"
            comp_hr = comp[comp.get(hr_flag, False) == True] if not comp.empty else pd.DataFrame()
            comp_non_hr = comp[comp.get(hr_flag, False) != True] if not comp.empty else pd.DataFrame()

            hr_rewarded = comp_hr[comp_hr.get("response_time_category", "") == "rewarded"] if not comp_hr.empty else pd.DataFrame()
            hr_unrewarded = comp_hr[comp_hr.get("response_time_category", "") == "unrewarded"] if not comp_hr.empty else pd.DataFrame()
            hr_timeout = comp_hr[comp_hr.get("response_time_category", "") == "timeout_delayed"] if not comp_hr.empty else pd.DataFrame()

            # Append non-HR trials
            _append_trials(rewarded[rewarded.index.isin(comp_non_hr.index)], "rewarded", False)
            _append_trials(unrewarded[unrewarded.index.isin(comp_non_hr.index)], "unrewarded", False)
            _append_trials(timeout[timeout.index.isin(comp_non_hr.index)], "timeout", False)
            _append_trials(aborted[aborted.get("hit_hidden_rule", False) != True], "aborted", False)

            # Append HR trials (only when HR success flag is set)
            _append_trials(hr_rewarded, "hr_rewarded", True)
            _append_trials(hr_unrewarded, "hr_unrewarded", True)
            _append_trials(hr_timeout, "hr_timeout", True)
            _append_trials(aborted_hr, "hr_aborted", True)

        else:
            try:
                results = load_session_results(subjid, date_str)
            except Exception as e:
                print(f"Warning: Could not load session {date_str}: {e}")
                continue

            # Legacy tables fallback
            comp_rew = results.get('completed_sequence_rewarded', pd.DataFrame())
            comp_unr = results.get('completed_sequence_unrewarded', pd.DataFrame())
            comp_tmo = results.get('completed_sequence_reward_timeout', pd.DataFrame())
            aborted = results.get('aborted_sequences', pd.DataFrame())
            hr_rewarded = results.get('completed_sequence_HR_rewarded', pd.DataFrame())
            hr_unrewarded = results.get('completed_sequence_HR_unrewarded', pd.DataFrame())
            hr_timeout = results.get('completed_sequence_HR_reward_timeout', pd.DataFrame())
            hr_missed = results.get('completed_sequences_HR_missed', pd.DataFrame())
            aborted_hr = results.get('aborted_sequences_HR', pd.DataFrame())

            _append_trials(comp_rew, "rewarded", False)
            _append_trials(comp_unr, "unrewarded", False)
            _append_trials(comp_tmo, "timeout", False)
            _append_trials(aborted, "aborted", False)
            _append_trials(hr_rewarded, "hr_rewarded", True)
            _append_trials(hr_unrewarded, "hr_unrewarded", True)
            _append_trials(hr_timeout, "hr_timeout", True)
            _append_trials(hr_missed, "hr_missed", True)
            _append_trials(aborted_hr, "hr_aborted", True)
    
    if not all_trials:
        print(f"No trials found for subject {subjid}")
        return None, None
    
    trials_df = pd.DataFrame(all_trials)
    trials_df = trials_df.sort_values('sequence_start').reset_index(drop=True)
    
    # Set global start time from first trial
    global_start_time = trials_df['sequence_start'].iloc[0]
    
    # Calculate time offsets for each session to collapse inter-session gaps
    session_time_offsets = {}
    session_boundaries = []
    
    time_offset = 0
    
    for session_idx in sorted(trials_df['session_idx'].unique()):
        session_data = trials_df[trials_df['session_idx'] == session_idx]
        
        if session_idx == 0:
            session_time_offsets[session_idx] = 0
        else:
            prev_session_data = trials_df[trials_df['session_idx'] == session_idx - 1]
            
            if not prev_session_data.empty and not session_data.empty:
                prev_end = prev_session_data['sequence_start'].max()
                curr_start = session_data['sequence_start'].min()
                
                gap = (curr_start - prev_end).total_seconds()
                time_offset += gap
                session_time_offsets[session_idx] = time_offset
                
                prev_time_in_plot = (prev_end - global_start_time).total_seconds() - session_time_offsets[session_idx - 1]
                session_boundaries.append((prev_time_in_plot, session_idx))
    
    # Calculate plot time for each trial
    trials_df['time_in_plot'] = trials_df.apply(
        lambda row: (row['sequence_start'] - global_start_time).total_seconds() - session_time_offsets[row['session_idx']],
        axis=1
    )
    
    # Extract odor letter (e.g., 'OdorA' -> 'A')
    def extract_odor_letter(odor_str):
        if pd.isna(odor_str):
            return 'Unknown'
        odor_str = str(odor_str)
        if odor_str.startswith('Odor'):
            return odor_str.replace('Odor', '')
        return odor_str
    
    trials_df['odor_letter'] = trials_df['last_odor'].apply(extract_odor_letter)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Define colors
    odor_colors = {
        'A': '#E53935',      # Bright red
        'B': '#00796B',      # Darker teal
        'HR': '#FFD700'      # Gold/yellow
    }
    
    odor_direction = {'A': 1, 'B': -1}  # A goes up, B goes down
    
    # Plot each trial
    for idx, trial in trials_df.iterrows():
        x = trial['time_in_plot']
        odor = trial['odor_letter']
        trial_type = trial['trial_type']
        is_hr = trial['is_hr']
        
        if odor not in odor_colors and odor != 'Unknown':
            odor = 'Unknown'
        
        # Determine color based on trial type
        if is_hr:
            color = odor_colors['HR']
        else:
            color = odor_colors.get(odor, '#999999')
        
        # Determine direction from odor
        direction = odor_direction.get(odor, 1)
        
        # Determine line style and marker based on reward status
        if trial_type == 'rewarded' or trial_type == 'hr_rewarded':
            linestyle = '-'
            linewidth = 2 * lw_scale
            alpha = 0.85
            has_marker = True
        elif trial_type in ['unrewarded', 'timeout', 'hr_unrewarded', 'hr_timeout', 'hr_missed']:
            linestyle = ':'
            linewidth = 2 * lw_scale
            alpha = 0.5
            has_marker = False
        else:
            # aborted or hr_aborted
            linestyle = '-'
            linewidth = 1.5 * lw_scale
            alpha = 0.6
            has_marker = False
        
        # Plot the trial
        if trial_type == 'aborted':
            # Regular aborted: grey line with optional FA-based port direction and marker.
            fa_label = str(trial.get('fa_label', '')).strip().lower()
            fa_port_raw = trial.get('fa_port', np.nan)
            fa_match = (fa_label in fa_set) if fa_set else False

            y_end = 0.6
            tri_marker = '^'
            if fa_match and pd.notna(fa_port_raw):
                try:
                    fa_port = int(float(fa_port_raw))
                    if fa_port == 2:
                        y_end = -0.6
                        tri_marker = 'v'
                    else:
                        y_end = 0.6
                        tri_marker = '^'
                except Exception:
                    pass

            ax.plot([x, x], [0, y_end], color='#888888', linewidth=linewidth, alpha=alpha, zorder=1)
            ax.scatter([x], [y_end], color='#888888', s=15 * marker_scale, marker=tri_marker, alpha=alpha, zorder=2)

            # Blue marker indicates this aborted trial matched the requested FA labels.
            if fa_match:
                ax.scatter([x], [y_end], color='#1f77b4', s=24 * marker_scale, marker='o',
                           edgecolors='white', linewidth=0.8 * lw_scale, zorder=3)
        
        elif trial_type == 'hr_aborted':
            # HR aborted: if matching requested FA labels, use fa_port direction and blue end-dot.
            # Otherwise keep the yellow arrowhead behavior.
            fa_label = str(trial.get('fa_label', '')).strip().lower()
            fa_port_raw = trial.get('fa_port', np.nan)
            fa_match = (fa_label in fa_set) if fa_set else False

            y_end = 0.6 * direction
            if fa_match and pd.notna(fa_port_raw):
                try:
                    fa_port = int(float(fa_port_raw))
                    y_end = 0.6 if fa_port == 1 else -0.6
                except Exception:
                    pass

            ax.plot([x, x], [0, y_end], color=color, linewidth=linewidth, alpha=alpha, zorder=1, linestyle=linestyle)

            if fa_match:
                ax.scatter([x], [y_end], color='#1f77b4', s=24 * marker_scale, marker='o',
                           edgecolors='white', linewidth=0.8 * lw_scale, zorder=3)
            else:
                tri_marker = '^' if y_end >= 0 else 'v'
                ax.scatter([x], [y_end], color=color, s=15 * marker_scale, marker=tri_marker, alpha=alpha, zorder=2)
        
        else:
            # Completed trials (regular or HR)
            y_end = 1.0 * direction
            
            # HR trials on top
            line_zorder = 2 if is_hr else 1
            marker_zorder = 4 if is_hr else 3
            
            ax.plot([x, x], [0, y_end], color=color, linewidth=linewidth, 
                   linestyle=linestyle, alpha=alpha, zorder=line_zorder)
            
            if has_marker:
                ax.scatter([x], [y_end], color=color, s=40 * marker_scale, marker='o',
                          edgecolors='black', linewidth=0.8 * lw_scale, zorder=marker_zorder)
    
    # Draw session boundaries
    for boundary_time, session_idx in session_boundaries:
        ax.axvline(x=boundary_time, color='grey', linestyle=':', linewidth=1.5 * lw_scale, alpha=0.7, zorder=0)

    # Format axes
    ax.axhline(y=0, color='black', linewidth=2 * lw_scale, alpha=0.8)
    ax.set_ylim([-1.5, 1.5])
    
    x_min = trials_df['time_in_plot'].min()
    x_max = trials_df['time_in_plot'].max()
    x_padding = (x_max - x_min) * 0.05
    if xlim is None:
        ax.set_xlim([x_min - x_padding, x_max + x_padding])
    else:
        ax.set_xlim(xlim)
    
    ax.set_yticks([-1, 1])
    ax.set_yticklabels(['B', 'A'])

    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Choice')
    
    if title is None:
        title = f"Choice History - Subject {str(subjid).zfill(3)}"
    ax.set_title(title)
    
    # Create custom legend (skip when show_legend=False, e.g. for poster figures)
    if show_legend:
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='#E53935', lw=2.5, linestyle='-', label='Odor A (regular)'),
            Line2D([0], [0], color='#00796B', lw=2.5, linestyle='-', label='Odor B (regular)'),
            Line2D([0], [0], color='#FFD700', lw=2.5, linestyle='-', label='Hidden Rule (HR)'),
            Line2D([0], [0], color='black', lw=2, linestyle='-', label='Rewarded (solid)'),
            Line2D([0], [0], color='black', lw=2, linestyle=':', label='Unrewarded/Missed (dotted)'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='black',
                   markersize=5, markeredgecolor='black', label='Rewarded marker', linestyle='none'),
        ]
        ax.legend(handles=legend_elements, loc='upper left', ncol=2)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    if save:
        try:
            save_name = "choice_history"
            out_path = save_figure(
                fig,
                save_name,
                subjids=[subjid],
                dates=dates,
            )
            if verbose:
                print(f"[plot_choice_history] Saved figure to {out_path}")
        except Exception as exc:
            if verbose:
                print(f"[plot_choice_history] Failed to save figure: {exc}")

    plt.show()
    
    return fig, ax



def plot_position_completion_rate(
    subjids,
    dates=None,
    positions=(1, 2, 3, 4),
    figsize=(8, 6.8),
    title=None,
    *,
    save=False,
    verbose=True,
    show_title=True,
    color_by_id=False,
    avg_per_animal=False,
):
    """Per-position completion rate across sessions (dot plot with mean ± SD).

    For each session:
    - `max_pos = max(num_odors)` over all trials in the session.
    - Each completed trial (`is_aborted == False`) contributes one completed-count
      to every position 1..max_pos.
    - Each aborted trial (`is_aborted == True`) contributes one completed-count
      to positions 1..last_odor_position − 1 and one aborted-count at
      last_odor_position.
    - Completion rate at position p =
          completed[p] / (completed[p] + aborted[p]) * 100

    Each session yields one rate per requested position; rates are plotted as
    blue dots horizontally jittered around each x-tick, with a black mean line
    and SD error bars.

    Parameters
    ----------
    subjids : int | list[int] | dict
        Subject id(s). May also be a dict ``{subjid: date_range}`` as a convenience
        shorthand — in that case the dict is used as ``dates`` and the subjids are
        its keys.
    dates : list | tuple | dict | None
        Specific dates [YYYYMMDD, ...] or inclusive range (start, end). If a dict,
        must map ``subjid → date_range`` (each value itself a list/tuple/None
        passed through to ``_filter_session_dirs``); this lets each subject use
        its own date window — useful when animals are offset in calendar time.
        Subjids not present as keys are skipped with a warning. ``None`` = all
        sessions for every subject.
    positions : iterable[int]
        Positions to display on the x-axis (e.g. [1, 2, 3, 4]).
    figsize : tuple
    title : str | None
    save : bool
    verbose : bool
    show_title : bool
        If False, no title is rendered (useful for poster-style figures).
    color_by_id : bool
        If True, each animal's dots are colored consistently using the same
        per-subject tab20 palette as :func:`plot_cumulative_rewards`.
    avg_per_animal : bool
        If True, no individual session dots are drawn. Instead each animal's
        session rates at a position are shown as a small violin (one violin per
        animal per position, spread horizontally within the position slot). The
        black line then shows the mean ± SEM computed across animals (each animal
        contributing its mean of session rates). Violins are colored by subject
        when ``color_by_id`` is True.

    Returns
    -------
    fig, ax
    """
    # Mirror plot_cumulative_rewards's input flexibility.
    if isinstance(subjids, dict):
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

    derivatives_dir = get_derivatives_root()
    positions = list(positions)
    rates_per_position: dict[int, list[float]] = {p: [] for p in positions}
    subj_per_position: dict[int, list] = {p: [] for p in positions}

    for subjid in subjids:
        subj_dates = _dates_for(subjid)
        if isinstance(dates, dict) and subj_dates is None:
            print(f"Warning: No date range provided in dict for subject {subjid}, skipping")
            continue

        subj_str = normalize_subjid(subjid)
        subj_dir = derivatives.subject_dir(subjid, missing_ok=True)
        if subj_dir is None:
            if verbose:
                print(f"Warning: No subject directory found for {subj_str}")
            continue

        ses_dirs = _filter_session_dirs(subj_dir, subj_dates)
        for ses_dir in ses_dirs:
            results_dir = ses_dir / "saved_analysis_results"
            if not results_dir.exists():
                continue
            views = _load_trial_views(results_dir)
            td = views["trial_data"]
            if td.empty:
                continue

            abortion = abortion_rate_positionX(td)

            for p in positions:
                if p not in abortion.index:
                    continue
                rate = 1.0 - float(abortion.loc[p])
                if np.isnan(rate):
                    continue
                rates_per_position[p].append(rate)
                subj_per_position[p].append(subjid)

    fig, ax = plt.subplots(figsize=figsize)
    rng = np.random.default_rng(0)
    x_idx_array = np.arange(len(positions))
    halfwidth = 0.25  # horizontal extent of both mean line and dot jitter

    # Per-subject color map (shared palette with plot_cumulative_rewards).
    # Sorted by ascending id so the same subject keeps its color across plots.
    subj_colors = {s: plt.cm.tab20(i % 10) for i, s in enumerate(sorted(subjids))}

    for x_idx, p in enumerate(positions):
        rates = np.array(rates_per_position[p], dtype=float)
        if rates.size == 0:
            continue
        subj_ids = subj_per_position[p]

        if avg_per_animal:
            # One violin per animal (distribution of that animal's session
            # rates), spread horizontally within the position slot.
            per_animal: dict = {}
            for s, r in zip(subj_ids, rates):
                per_animal.setdefault(s, []).append(r)
            subj_order = [s for s in subjids if s in per_animal]
            n = len(subj_order)
            offsets = np.linspace(-halfwidth, halfwidth, n) if n > 1 else np.array([0.0])
            vwidth = (2 * halfwidth / max(n, 1)) * 0.8

            for subj, off in zip(subj_order, offsets):
                vals = np.array(per_animal[subj], dtype=float)
                color = subj_colors[subj] if color_by_id else "tab:blue"
                if vals.size >= 2:
                    parts = ax.violinplot([vals], positions=[x_idx + off],
                                          widths=vwidth, showextrema=False)
                    for body in parts["bodies"]:
                        body.set_facecolor(color)
                        body.set_edgecolor(color)
                        body.set_alpha(0.2)
                else:
                    # A single session can't form a violin; mark the point.
                    ax.scatter([x_idx + off], vals, color=color, alpha=0.7,
                               s=40, edgecolors="none", zorder=2)

            # Mean ± SEM across animals (each animal = mean of its session rates).
            animal_means = np.array([np.mean(per_animal[s]) for s in subj_order], dtype=float)
            mean, err = mean_sem(animal_means)
            err = 0.0 if np.isnan(err) else err
        else:
            jitter = rng.uniform(-halfwidth, halfwidth, size=rates.size)
            xs = np.full_like(jitter, x_idx) + jitter
            if color_by_id:
                pt_colors = [subj_colors[s] for s in subj_ids]
                ax.scatter(xs, rates, c=pt_colors, alpha=0.7, s=40,
                           edgecolors="none", zorder=2)
            else:
                ax.scatter(xs, rates, color="tab:blue", alpha=0.55, s=40,
                           edgecolors="none", zorder=2)
            mean = float(rates.mean())
            err = float(rates.std(ddof=1)) if rates.size > 1 else 0.0

        ax.hlines(mean, x_idx - halfwidth, x_idx + halfwidth,
                  colors="black", linewidth=2.0, zorder=3)
        ax.errorbar(x_idx, mean, yerr=err, color="black", linewidth=1.5,
                    capsize=6, capthick=1.5, fmt="none", zorder=3)

    ax.set_xticks(x_idx_array)
    ax.set_xticklabels([str(p) for p in positions])
    ax.set_xlabel("Sequence Position")
    ax.set_ylabel("Completion Rate")
    ax.set_xlim(-0.5, len(positions) - 0.5)
    ax.set_ylim(0, 1.05)

    if color_by_id:
        present = [s for s in subjids if any(s in subj_per_position[p] for p in positions)]
        handles = [
            Line2D([0], [0], marker="o", linestyle="none", color=subj_colors[s],
                   label=f"Sub {str(s).zfill(3)}")
            for s in present
        ]
        if handles:
            ax.legend(handles=handles, title="Subject", loc="best")

    if show_title:
        ax.set_title(title if title else "Position Completion Rate by Session")

    # Extra pad so the (often long, bold) y-axis label doesn't clip in the
    # notebook display. Saved figures use bbox_inches='tight' so they're
    # already safe, but the live preview honors the figsize bbox.
    fig.tight_layout(pad=1.5)

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
                fig, "position_completion_rate",
                subjids=list(subjids), dates=save_dates,
                boxplot=True,
            )
            if verbose:
                print(f"[plot_position_completion_rate] Saved figure to {out_path}")
        except Exception as exc:
            if verbose:
                print(f"[plot_position_completion_rate] Failed to save figure: {exc}")

    plt.show()
    return fig, ax


def _positions_in_presentations(pres_json):
    """Return the list of positions present in a trial's ``presentations`` JSON.

    ``presentations`` is a list of per-odor dicts (one per position the animal
    sampled), so this is every position the trial reached.
    """
    data = parse_json_column(pres_json)
    if not isinstance(data, list):
        return []
    out = []
    for entry in data:
        if isinstance(entry, dict):
            pos = entry.get("position")
            if pos is not None:
                try:
                    out.append(int(pos))
                except (TypeError, ValueError):
                    pass
    return out


def plot_false_alarm_rate_by_position(
    subjids,
    dates=None,
    positions=(1, 2, 3, 4, 5),
    fa_label="FA_time_in",
    figsize=(8, 6.8),
    title=None,
    *,
    save=False,
    verbose=True,
    show_title=True,
    color_by_id=False,
    avg_per_animal=False,
):
    """Per-position false-alarm rate across sessions (dot plot with mean ± SD).

    For each session and each position ``p``:
    - Reach count = number of trials whose ``presentations`` include position
      ``p`` (i.e. trials that got to position ``p``, completed or aborted).
    - FA count = number of aborted trials matching ``fa_label`` whose last
      sampled position (``last_odor_position``) is ``p``.
    - FA rate at ``p`` = FA count / reach count.

    Each session yields one rate per requested position; rates are plotted as
    dots jittered around each x-tick with a black mean line and SD error bars,
    matching :func:`plot_position_completion_rate`.

    Parameters
    ----------
    subjids : int | list[int] | dict
        Subject id(s). May also be a dict ``{subjid: date_range}`` shorthand.
    dates : list | tuple | dict | None
        Dates or per-subject ``{subjid: date_range}`` dict. ``None`` = all sessions.
    positions : iterable[int]
        Positions to display on the x-axis.
    fa_label : str | list[str] | None
        Which false-alarm label(s) count toward the numerator. Default
        ``"FA_time_in"``. Accepts a single label or a list. ``None`` counts any
        false alarm (every aborted trial whose ``fa_label`` is not ``nFA``).
    figsize : tuple
    title : str | None
    save : bool
    verbose : bool
    show_title : bool
        If False, no title is rendered (useful for poster-style figures).
    color_by_id : bool
        If True, each animal's dots are colored consistently (shared tab20
        palette, assigned by ascending id).
    avg_per_animal : bool
        If True, no individual dots; each animal's session rates become a small
        violin per position and the black line shows mean ± SEM across animals.

    Returns
    -------
    fig, ax
    """
    # Mirror plot_position_completion_rate's input flexibility.
    if isinstance(subjids, dict):
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

    # Normalize fa_label into a lowercase set (or None = any non-nFA).
    if fa_label is None:
        fa_set = None
    elif isinstance(fa_label, (list, tuple, set)):
        fa_set = {str(s).strip().lower() for s in fa_label}
    else:
        fa_set = {str(fa_label).strip().lower()}

    derivatives_dir = get_derivatives_root()
    positions = list(positions)
    rates_per_position: dict[int, list[float]] = {p: [] for p in positions}
    subj_per_position: dict[int, list] = {p: [] for p in positions}

    for subjid in subjids:
        subj_dates = _dates_for(subjid)
        if isinstance(dates, dict) and subj_dates is None:
            print(f"Warning: No date range provided in dict for subject {subjid}, skipping")
            continue

        subj_str = normalize_subjid(subjid)
        subj_dir = derivatives.subject_dir(subjid, missing_ok=True)
        if subj_dir is None:
            if verbose:
                print(f"Warning: No subject directory found for {subj_str}")
            continue

        ses_dirs = _filter_session_dirs(subj_dir, subj_dates)
        for ses_dir in ses_dirs:
            results_dir = ses_dir / "saved_analysis_results"
            if not results_dir.exists():
                continue
            views = _load_trial_views(results_dir)
            td = views["trial_data"]
            if td.empty or "presentations" not in td.columns:
                continue

            rates = fa_rate_by_position(td, fa_types=fa_set)

            for p in positions:
                if p not in rates.index:
                    continue
                rates_per_position[p].append(float(rates.loc[p]))
                subj_per_position[p].append(subjid)

    fig, ax = plt.subplots(figsize=figsize)
    rng = np.random.default_rng(0)
    x_idx_array = np.arange(len(positions))
    halfwidth = 0.25  # horizontal extent of both mean line and dot jitter

    # Per-subject color map (shared palette with plot_cumulative_rewards).
    # Sorted by ascending id so the same subject keeps its color across plots.
    subj_colors = {s: plt.cm.tab20(i % 20) for i, s in enumerate(sorted(subjids))}

    for x_idx, p in enumerate(positions):
        rates = np.array(rates_per_position[p], dtype=float)
        if rates.size == 0:
            continue
        subj_ids = subj_per_position[p]

        if avg_per_animal:
            # One violin per animal (distribution of that animal's session
            # rates), spread horizontally within the position slot.
            per_animal: dict = {}
            for s, r in zip(subj_ids, rates):
                per_animal.setdefault(s, []).append(r)
            subj_order = [s for s in subjids if s in per_animal]
            n = len(subj_order)
            offsets = np.linspace(-halfwidth, halfwidth, n) if n > 1 else np.array([0.0])
            vwidth = (2 * halfwidth / max(n, 1)) * 0.8

            for subj, off in zip(subj_order, offsets):
                vals = np.array(per_animal[subj], dtype=float)
                color = subj_colors[subj] if color_by_id else "tab:blue"
                if vals.size >= 2:
                    parts = ax.violinplot([vals], positions=[x_idx + off],
                                          widths=vwidth, showextrema=False)
                    for body in parts["bodies"]:
                        body.set_facecolor(color)
                        body.set_edgecolor(color)
                        body.set_alpha(0.2)
                else:
                    # A single session can't form a violin; mark the point.
                    ax.scatter([x_idx + off], vals, color=color, alpha=0.7,
                               s=40, edgecolors="none", zorder=2)

            # Mean ± SEM across animals (each animal = mean of its session rates).
            animal_means = np.array([np.mean(per_animal[s]) for s in subj_order], dtype=float)
            mean, err = mean_sem(animal_means)
            err = 0.0 if np.isnan(err) else err
        else:
            jitter = rng.uniform(-halfwidth, halfwidth, size=rates.size)
            xs = np.full_like(jitter, x_idx) + jitter
            if color_by_id:
                pt_colors = [subj_colors[s] for s in subj_ids]
                ax.scatter(xs, rates, c=pt_colors, alpha=0.7, s=40,
                           edgecolors="none", zorder=2)
            else:
                ax.scatter(xs, rates, color="tab:blue", alpha=0.55, s=40,
                           edgecolors="none", zorder=2)
            mean = float(rates.mean())
            err = float(rates.std(ddof=1)) if rates.size > 1 else 0.0

        ax.hlines(mean, x_idx - halfwidth, x_idx + halfwidth,
                  colors="black", linewidth=2.0, zorder=3)
        ax.errorbar(x_idx, mean, yerr=err, color="black", linewidth=1.5,
                    capsize=6, capthick=1.5, fmt="none", zorder=3)

    ax.set_xticks(x_idx_array)
    ax.set_xticklabels([str(p) for p in positions])
    ax.set_xlabel("Sequence Position")
    ax.set_ylabel("False Alarm Rate")
    ax.set_xlim(-0.5, len(positions) - 0.5)
    ax.set_ylim(bottom=0, top=1.05)

    if color_by_id:
        present = [s for s in subjids if any(s in subj_per_position[p] for p in positions)]
        handles = [
            Line2D([0], [0], marker="o", linestyle="none", color=subj_colors[s],
                   label=f"Sub {str(s).zfill(3)}")
            for s in present
        ]
        if handles:
            ax.legend(handles=handles, title="Subject", loc="best")

    if show_title:
        ax.set_title(title if title else "False Alarm Rate by Position")

    fig.tight_layout(pad=1.5)

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
                fig, "false_alarm_rate_by_position",
                subjids=list(subjids), dates=save_dates,
                boxplot=True,
            )
            if verbose:
                print(f"[plot_false_alarm_rate_by_position] Saved figure to {out_path}")
        except Exception as exc:
            if verbose:
                print(f"[plot_false_alarm_rate_by_position] Failed to save figure: {exc}")

    plt.show()
    return fig, ax


def plot_poke_duration_by_position(
    subjids,
    dates=None,
    positions=None,
    figsize=(8, 6.8),
    title=None,
    *,
    save=False,
    verbose=True,
    show_title=True,
    color_by_id=False,
    avg_per_animal=False,
):
    """Per-position poke (sampling) duration across sessions, split by trial outcome.

    Produces two separate figures ("Completed" and "Aborted"). For every session
    and every position, the mean poke duration (ms) at that position is computed
    and contributes one dot to the corresponding figure:

    - Completed trials (``is_aborted == False``): poke durations are read from the
      ``position_poke_times`` column, keyed by position.
    - Aborted trials (``is_aborted == True``): poke durations are read from the
      ``presentations`` column, excluding the abort event; keyed by position.

    Dots are horizontally jittered around each x-tick, with a black mean line and
    SD error bars — mirroring :func:`plot_position_completion_rate`.

    Parameters
    ----------
    subjids : int | list[int] | dict
        Subject id(s). May also be a dict ``{subjid: date_range}`` as a convenience
        shorthand — in that case the dict is used as ``dates`` and the subjids are
        its keys.
    dates : list | tuple | dict | None
        Specific dates [YYYYMMDD, ...] or inclusive range (start, end). If a dict,
        must map ``subjid → date_range`` so each subject can use its own date
        window. Subjids not present as keys are skipped with a warning.
        ``None`` = all sessions for every subject.
    positions : iterable[int] | None
        Positions to display on the x-axis. ``None`` (default) shows every
        position present in the data (1 .. max position across any protocol).
    figsize : tuple
        Size of each individual figure.
    title : str | None
        Title base; the trial outcome ("Completed"/"Aborted") is appended per
        figure. Only rendered when ``show_title`` is True.
    save : bool
    verbose : bool
    show_title : bool
        If False, no titles are rendered (useful for poster-style figures).
    color_by_id : bool
        If True, each animal's dots are colored consistently using the same
        per-subject tab20 palette as :func:`plot_cumulative_rewards`.
    avg_per_animal : bool
        If True, no individual session dots are drawn. Instead each animal's
        session means at a position are shown as a small violin (one violin per
        animal per position, spread horizontally within the position slot). The
        black line then shows the mean ± SEM computed across animals (each animal
        contributing its mean of session means). Violins are colored by subject
        when ``color_by_id`` is True.

    Returns
    -------
    fig_completed, ax_completed, fig_aborted, ax_aborted
    """
    # Mirror plot_position_completion_rate's input flexibility.
    if isinstance(subjids, dict):
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

    derivatives_dir = get_derivatives_root()

    # One row per (subject, session, position, trial_type) = session mean poke ms.
    rows = []
    for sid, subj_dir in _iter_subject_dirs(derivatives_dir, subjids):
        subj_dates = _dates_for(sid)
        if isinstance(dates, dict) and subj_dates is None:
            print(f"Warning: No date range provided in dict for subject {sid}, skipping")
            continue

        ses_dirs = _filter_session_dirs(subj_dir, subj_dates)
        for ses_dir in ses_dirs:
            results_dir = ses_dir / "saved_analysis_results"
            if not results_dir.exists():
                continue
            position_data = build_position_data(
                _load_trial_views(results_dir)["trial_data"])

            # Collapse each session to one mean per position per trial type.
            for trial_type, aborted in (("completed", False), ("aborted", True)):
                stats = poke_duration_by_position(position_data, aborted=aborted)
                for pos, mean_ms in stats["mean"].items():
                    rows.append({
                        "subjid": sid,
                        "trial_type": trial_type,
                        "position": int(pos),
                        "mean_poke_ms": float(mean_ms),
                    })

    if not rows:
        print("No data found")
        return None, None, None, None

    if positions is not None:
        positions_list = [int(p) for p in positions]
    else:
        positions_list = sorted({r["position"] for r in rows})
    pos_to_idx = {p: i for i, p in enumerate(positions_list)}
    x_idx_array = np.arange(len(positions_list))

    # Per-subject color map (shared palette with plot_cumulative_rewards).
    # Sorted by ascending id so the same subject keeps its color across plots.
    subj_colors = {s: plt.cm.tab20(i % 20) for i, s in enumerate(sorted(subjids))}

    rng = np.random.default_rng(0)
    halfwidth = 0.25

    present = [s for s in subjids if any(r["subjid"] == s for r in rows)]

    def _draw_panel(trial_type, label, base_color):
        fig, ax = plt.subplots(figsize=figsize)
        for p in positions_list:
            pos_rows = [r for r in rows if r["trial_type"] == trial_type and r["position"] == p]
            if not pos_rows:
                continue
            x_idx = pos_to_idx[p]

            if avg_per_animal:
                # One violin per animal (distribution of that animal's session
                # means), spread horizontally within the position slot.
                per_animal = {}
                for r in pos_rows:
                    per_animal.setdefault(r["subjid"], []).append(r["mean_poke_ms"])
                subj_order = [s for s in subjids if s in per_animal]
                n = len(subj_order)
                offsets = np.linspace(-halfwidth, halfwidth, n) if n > 1 else np.array([0.0])
                vwidth = (2 * halfwidth / max(n, 1)) * 0.8

                for subj, off in zip(subj_order, offsets):
                    vals = np.array(per_animal[subj], dtype=float)
                    color = subj_colors[subj] if color_by_id else base_color
                    if vals.size >= 2:
                        parts = ax.violinplot([vals], positions=[x_idx + off],
                                              widths=vwidth, showextrema=False)
                        for body in parts["bodies"]:
                            body.set_facecolor(color)
                            body.set_edgecolor(color)
                            body.set_alpha(0.2)
                    else:
                        # A single session can't form a violin; mark the point.
                        ax.scatter([x_idx + off], vals, color=color, alpha=0.7,
                                   s=40, edgecolors="none", zorder=2)

                # Mean ± SEM across animals (each animal = mean of its session means).
                animal_means = np.array([np.mean(per_animal[s]) for s in subj_order], dtype=float)
                mean, err = mean_sem(animal_means)
                err = 0.0 if np.isnan(err) else err
            else:
                values = np.array([r["mean_poke_ms"] for r in pos_rows], dtype=float)
                jitter = rng.uniform(-halfwidth, halfwidth, size=values.size)
                xs = np.full_like(jitter, x_idx) + jitter
                if color_by_id:
                    pt_colors = [subj_colors[r["subjid"]] for r in pos_rows]
                    ax.scatter(xs, values, c=pt_colors, alpha=0.7, s=40,
                               edgecolors="none", zorder=2)
                else:
                    ax.scatter(xs, values, color=base_color, alpha=0.55, s=40,
                               edgecolors="none", zorder=2)
                mean = float(values.mean())
                err = float(values.std(ddof=1)) if values.size > 1 else 0.0

            ax.hlines(mean, x_idx - halfwidth, x_idx + halfwidth,
                      colors="black", linewidth=2.0, zorder=3)
            ax.errorbar(x_idx, mean, yerr=err, color="black", linewidth=1.5,
                        capsize=6, capthick=1.5, fmt="none", zorder=3)

        ax.set_xticks(x_idx_array)
        ax.set_xticklabels([str(p) for p in positions_list])
        ax.set_xlabel("Sequence Position")
        ax.set_ylabel("Poke Duration (ms)")
        ax.set_xlim(-0.5, len(positions_list) - 0.5)
        ax.set_ylim(bottom=0)

        if color_by_id and present:
            handles = [
                Line2D([0], [0], marker="o", linestyle="none", color=subj_colors[s],
                       label=f"Sub {str(s).zfill(3)}")
                for s in present
            ]
            ax.legend(handles=handles, title="Subject", loc="best")

        if show_title:
            base = title if title else "Poke Duration by Position"
            ax.set_title(f"{base} ({label})")

        fig.tight_layout(pad=1.5)

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
                    fig, f"poke_duration_by_position_{trial_type}",
                    subjids=list(subjids), dates=save_dates,
                    boxplot=True,
                )
                if verbose:
                    print(f"[plot_poke_duration_by_position] Saved {label} figure to {out_path}")
            except Exception as exc:
                if verbose:
                    print(f"[plot_poke_duration_by_position] Failed to save {label} figure: {exc}")

        return fig, ax

    fig_completed, ax_completed = _draw_panel("completed", "Completed", "steelblue")
    fig_aborted, ax_aborted = _draw_panel("aborted", "Aborted", "coral")

    plt.show()
    return fig_completed, ax_completed, fig_aborted, ax_aborted


def plot_decision_accuracy(
    subjids,
    dates=None,
    figsize=(10, 7),
    title=None,
    *,
    save=False,
    verbose=True,
    show_title=True,
    show_legend=True,
    color_by_id=True,
    mean=False,
    show_criterion=False,
    criterion=0.8,
):
    """Decision accuracy over training days, one series per subject.

    Each animal is drawn as a colored line: its per-session decision accuracy
    (rewarded / (rewarded + unrewarded), same as the ``decision_accuracy`` metric,
    computed here from ``trial_data``) plotted against day index. Day 1 is each
    animal's first session with an A/B decision, so animals are aligned by
    training day rather than calendar date. Markers show each subject's value on
    each day, matching the subject-series style of :func:`plot_cumulative_rewards`.

    If ``mean=True``, a thicker black line shows the mean across animals at each
    day index. At a
    given day the mean uses only the animals that have data there, so as animals
    run out of sessions the mean is averaged over fewer of them.

    Hidden-rule split: if any session in the input contains hidden-rule trials
    (``hidden_rule_success == True``), decision accuracy is computed separately
    for non-HR and HR trials. The non-HR accuracy is drawn as the usual solid
    line; the HR accuracy is drawn as a dashed line in the same per-animal color
    (present only on days that have HR trials). The group mean is likewise split
    into a solid (non-HR) and dashed (HR) black line. If no session has HR trials,
    behavior is unchanged (single solid line = overall accuracy).

    Parameters
    ----------
    subjids : int | list[int] | dict
        Subject id(s). May also be a dict ``{subjid: date_range}`` as a
        convenience shorthand — in that case the dict is used as ``dates`` and the
        subjids are its keys.
    dates : list | tuple | dict | None
        Specific dates [YYYYMMDD, ...] or inclusive range (start, end). If a dict,
        must map ``subjid → date_range`` so each subject can use its own date
        window. Subjids not present as keys are skipped with a warning.
        ``None`` = all sessions for every subject.
    figsize : tuple
    title : str | None
    save : bool
    verbose : bool
    show_title : bool
        If False, no title is rendered (useful for poster-style figures).
    show_legend : bool
        If True, show a subject legend (default: True).
    color_by_id : bool
        If True, each animal's thin line is colored using the shared per-subject
        tab20 palette (:func:`plot_cumulative_rewards`). Colors are assigned by
        ascending subject id, so the same ids keep the same colors across plots.
    mean : bool
        If True, overlay the thick group-mean line. If False (default), no mean
        line is drawn and the per-animal lines are drawn thicker. Line widths are
        shared with :func:`plot_behavior_metrics` via ``_series_line_widths``.
    show_criterion : bool
        If True, draw a dashed horizontal criterion line (default: False).
    criterion : float
        Y-value for the criterion line (default: 0.8).

    Returns
    -------
    fig, ax
    """
    # Mirror plot_behavior_metrics's input flexibility.
    if isinstance(subjids, dict):
        dates = subjids if not isinstance(dates, dict) or dates is None else dates
        subjids = list(subjids.keys())
    elif isinstance(dates, dict) and subjids is None:
        subjids = list(dates.keys())
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

    def _dates_ok(date_range):
        """Reject malformed date tokens (must be 8-digit YYYYMMDD).

        A typo like ``2025118`` (7 digits) or ``202251120`` (9 digits) would
        otherwise be treated by ``_filter_session_dirs`` as a numeric range
        endpoint and silently match every real session, so we guard here.
        """
        def _ok(tok):
            s = str(tok)
            return s.isdigit() and len(s) == 8
        if date_range is None:
            return True
        if isinstance(date_range, tuple):
            return all(t is None or _ok(t) for t in date_range)
        if isinstance(date_range, (list, set)):
            return all(_ok(t) for t in date_range)
        return _ok(date_range)

    derivatives_dir = get_derivatives_root()

    def _decision_acc_split(td):
        """``(non_hr_accuracy, hr_accuracy)`` for one session.

        VARIANT 6 of the metric audit: the HR / non-HR split is a *granularity*
        of `decision_accuracy`, not a metric of its own, so it is `by_group` over
        the canonical HR mask. A side with no trials at all is absent from the
        grouping and comes back as NaN, which is what the callers below test for.
        """
        acc = by_group(decision_accuracy, td, hidden_rule_mask(td)).reindex([False, True])
        return acc.iloc[0], acc.iloc[1]

    # Per-animal, day-aligned accuracy for non-HR ("main") and HR trials (day 1 =
    # first session with an A/B decision). HR splitting only matters if any
    # session actually has hidden-rule trials (hr_active).
    per_animal_series: dict = {}
    hr_active = False
    for subjid in subjids:
        subj_dates = _dates_for(subjid)
        if isinstance(dates, dict) and subj_dates is None:
            if verbose:
                print(f"Warning: No date range provided in dict for subject {subjid}, skipping")
            continue
        if not _dates_ok(subj_dates):
            if verbose:
                print(f"Warning: subject {subjid} has malformed date(s) {subj_dates!r} "
                      f"(expected 8-digit YYYYMMDD); skipping")
            continue

        subj_str = normalize_subjid(subjid)
        subj_dir = derivatives.subject_dir(subjid, missing_ok=True)
        if subj_dir is None:
            if verbose:
                print(f"Warning: No subject directory found for {subj_str}")
            continue

        main_vals, hr_vals = [], []
        for ses_dir in _filter_session_dirs(subj_dir, subj_dates):
            results_dir = ses_dir / "saved_analysis_results"
            if not results_dir.exists():
                continue
            td = _load_trial_views(results_dir)["trial_data"]
            if td.empty or "response_time_category" not in td.columns:
                continue
            non_hr_acc, hr_acc = _decision_acc_split(td)
            # A session counts as a "day" only if it has an A/B decision.
            if np.isnan(non_hr_acc) and np.isnan(hr_acc):
                continue
            main_vals.append(non_hr_acc)
            hr_vals.append(hr_acc)
            if not np.isnan(hr_acc):
                hr_active = True

        if main_vals:
            per_animal_series[int(subjid)] = {"main": main_vals, "hr": hr_vals}

    if not per_animal_series:
        print("No data found")
        return None, None

    fig, ax = plt.subplots(figsize=figsize)

    # Per-subject color map (shared palette with plot_cumulative_rewards).
    # Sorted by ascending id so the same subject keeps its color across plots.
    subj_colors = {s: plt.cm.tab20(i % 20) for i, s in enumerate(sorted(subjids))}

    # Line widths shared with plot_behavior_metrics.
    per_series_lw, mean_lw = _series_line_widths(mean)

    # HR (dashed) lines are nudged up ~2.5 points and get small markers so they
    # stay visible when they exactly overlap the non-HR line, and so isolated HR
    # days (a gap on either side) still show up as a point.
    from matplotlib.transforms import offset_copy
    hr_offset = offset_copy(ax.transData, fig=fig, y=2.5, units="points")

    # Per-subject dash phase so overlapping HR (dashed) lines interleave — one
    # animal's dashes fall in another's gaps — instead of hiding each other
    # (probe accuracy is often a flat 1.0 for every animal, so they coincide).
    sorted_ids = sorted(subjids)
    n_ids = max(len(sorted_ids), 1)
    dash_on = dash_off = 6
    dash_period = dash_on + dash_off
    subj_dash_phase = {s: dash_period * i / n_ids for i, s in enumerate(sorted_ids)}

    # Line per animal, aligned so day 1 = first session with data. When HR trials
    # are present, the non-HR accuracy is the solid line and HR accuracy is a
    # dashed line in the same color (NaN days leave gaps).
    max_days = max(len(v["main"]) for v in per_animal_series.values())
    for subjid, series in per_animal_series.items():
        color = subj_colors[subjid] if color_by_id else "grey"
        alpha = 0.7 if color_by_id else 0.6
        main = np.array(series["main"], dtype=float)
        x = np.arange(1, len(main) + 1)
        ax.plot(
            x, main,
            color=color,
            linewidth=per_series_lw,
            alpha=alpha,
            marker="o",
            markersize=4,
            zorder=2,
        )
        if hr_active:
            hr = np.array(series["hr"], dtype=float)
            hr_ls = (subj_dash_phase.get(subjid, 0.0), (dash_on, dash_off))
            ax.plot(x, hr, color=color, linewidth=per_series_lw, alpha=alpha,
                    linestyle=hr_ls, marker="o", markersize=4,
                    transform=hr_offset, zorder=2.5)

    # Group mean at each day index, over whichever animals have data there.
    if mean:
        def _day_mean(key):
            mx, my = [], []
            for day in range(1, max_days + 1):
                vals = [s[key][day - 1] for s in per_animal_series.values()
                        if len(s[key]) >= day and not np.isnan(s[key][day - 1])]
                if vals:
                    mx.append(day)
                    my.append(float(np.mean(vals)))
            return mx, my

        mx, my = _day_mean("main")
        ax.plot(mx, my, color="black", linewidth=mean_lw, zorder=3)
        if hr_active:
            hx, hy = _day_mean("hr")
            ax.plot(hx, hy, color="black", linewidth=mean_lw, linestyle="--",
                    marker="o", markersize=5, transform=hr_offset, zorder=3.5)

    ax.set_xlabel("Day")
    ax.set_ylabel("Decision Accuracy")
    ax.set_xlim(0.8, max_days + 0.5)
    ax.set_ylim(0, 1.05)
    if show_criterion:
        ax.axhline(
            y=criterion,
            color="gray",
            linestyle="--",
            linewidth=1.2,
            alpha=0.8,
            zorder=1,
        )
    # Only ever tick whole days (never 1.5, 2.5, ...). Local import so autoreload
    # picks it up without a kernel restart.
    from matplotlib.ticker import MaxNLocator
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    subj_leg = None
    if show_legend and color_by_id:
        handles = [
            Line2D([0], [0], color=subj_colors[s], linewidth=1.5, label=f"Sub {str(s).zfill(3)}")
            for s in sorted(per_animal_series.keys())
        ]
        if handles:
            subj_leg = ax.legend(handles=handles, title="Subject", loc="best")

    if show_legend and hr_active:
        # Solid = non-HR, dashed = HR. Keep the subject legend too, if present.
        if subj_leg is not None:
            ax.add_artist(subj_leg)
        style_handles = [
            Line2D([0], [0], color="black", linestyle="-", linewidth=2.0, label="Non-HR"),
            Line2D([0], [0], color="black", linestyle="--", linewidth=2.0,
                   marker="o", markersize=5, label="HR"),
        ]
        ax.legend(handles=style_handles, title="Trial type", loc="lower right")

    if show_title:
        ax.set_title(title if title else "Decision Accuracy over Day")

    fig.tight_layout(pad=1.5)

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
                fig, "decision_accuracy",
                subjids=list(subjids), dates=save_dates,
            )
            if verbose:
                print(f"[plot_decision_accuracy] Saved figure to {out_path}")
        except Exception as exc:
            if verbose:
                print(f"[plot_decision_accuracy] Failed to save figure: {exc}")

    plt.show()
    return fig, ax


def plot_poke_duration_by_odor(
    subjid,
    date=None,
    figsize=(10, 7),
    title=None,
    *,
    show_mean=True,
    show_lines=False,
    pool_subjids=False,
    odors=("C", "D", "E", "F", "G"),
    save=False,
    verbose=True,
    show_title=True,
):
    """Mean poke (sampling) duration per odor over training days.

    For each animal + session, every ``presentations`` entry belonging to a
    *completed* trial in ``trial_data`` is read off and grouped by odor
    (aborted trials are excluded entirely). Per animal, per session, per odor,
    the mean poke duration is plotted against day index and connected across
    days — one line per odor, colored as in :func:`hidden_rule_and_false_alarm`
    (one color per series, no point markers), following the "no dots"
    per-animal line style of :func:`plot_decision_accuracy`.

    Day 1 is each animal's first session with usable data for the requested
    odors. The x-axis then counts *consecutive sessions in the derivatives*
    (session order), not calendar dates — off-days/weekends with no session do
    not create gaps. A gap appears only when a session that exists has genuinely
    no data for that series.

    Parameters
    ----------
    subjid : int | list[int] | dict
        Subject id(s). May also be a dict ``{subjid: date_range}`` as a
        convenience shorthand — in that case the dict is used as ``date`` and
        the subjids are its keys.
    date : list | tuple | dict | None
        Specific dates [YYYYMMDD, ...] or inclusive range (start, end). If a
        dict, must map ``subjid -> date_range`` so each subject can use its own
        date window. Subjids not present as keys are skipped with a warning.
        ``None`` = all sessions for every subject.
    figsize : tuple
    title : str | None
    show_mean : bool
        If True (default):
        - When both ``"A"`` and ``"B"`` are in ``odors``, their poke durations
          are pooled into a single "A+B" line instead of two separate lines.
        - On any session flagged as a hidden-rule session (its two hidden-rule
          odors, from ``summary.json``, both present in ``odors``), those two
          odors' poke durations are pooled into a single "Hidden Rule" line for
          that session, and every *other* requested odor (excluding "A"/"B")
          is pooled into a single "Other Odors" line for that session — instead
          of each contributing to its own individual odor line. On sessions
          that aren't hidden-rule sessions, every requested odor still gets its
          own individual line.
        If False, every requested odor is always plotted as its own line.
    show_lines : bool
        Only used when ``show_mean`` is True. If True, overlays the individual
        odors that make up each mean as thin dashed low-alpha lines in their own
        colour (e.g. D/E/G under the "Other Odors" mean, the two hidden-rule
        odors under the "Hidden Rule" mean, A/B under the "A+B" mean).
    pool_subjids : bool
        If False (default), each subject gets its own line per series (odor,
        "A+B", or "Hidden Rule"), day-aligned to that subject's own day 1. If
        True, subjects are combined into a single line per series: at each day
        index, raw poke-duration samples from every subject with data that day
        are pooled together before averaging (so a subject contributing more
        trials counts for more, rather than each subject's mean counting
        equally).
    odors : iterable[str]
        Odor labels to include (case-insensitive; ``"OdorC"``-style tokens are
        also accepted). Default ``("C", "D", "E", "F", "G")``.
    save : bool
    verbose : bool
    show_title : bool
        If False, no title is rendered (useful for poster-style figures).

    Returns
    -------
    fig, ax
    """
    # Mirror plot_decision_accuracy's input flexibility.
    if isinstance(subjid, dict):
        date = subjid if not isinstance(date, dict) or date is None else date
        subjid = list(subjid.keys())
    elif isinstance(date, dict) and subjid is None:
        subjid = list(date.keys())
    elif isinstance(subjid, set):
        subjid = sorted(subjid)
    elif not isinstance(subjid, (list, tuple)):
        subjid = [subjid]

    def _dates_for(sid):
        if not isinstance(date, dict):
            return date
        if sid in date:
            return date[sid]
        try:
            int_key = int(sid)
            if int_key in date:
                return date[int_key]
        except (TypeError, ValueError):
            pass
        str_key = str(sid)
        if str_key in date:
            return date[str_key]
        return None

    odors_list = [odor_letter(o) for o in odors]
    odors_set = set(odors_list)
    ab_grouped = show_mean and "A" in odors_set and "B" in odors_set

    def _session_hr_odors(results_dir):
        """Hidden-rule odor letters for a session (from summary.json), or [] if
        the session has none / isn't a hidden-rule session."""
        summary_path = results_dir / "summary.json"
        if not summary_path.exists():
            return []
        try:
            with open(summary_path, "r", encoding="utf-8") as f:
                summary = json.load(f)
            hr_odors = summary.get("params", {}).get("hidden_rule_odors", [])
            if not hr_odors:
                runs = summary.get("session", {}).get("runs", [])
                if runs and isinstance(runs[0], dict):
                    stage = runs[0].get("stage", {}) if isinstance(runs[0].get("stage", {}), dict) else {}
                    hr_odors = stage.get("hidden_rule_odors", []) or []
            return [odor_letter(o) for o in hr_odors if o]
        except Exception:
            return []

    def _extract_odor_poke_ms(td):
        """``{odor_letter: [poke_ms, ...]}`` for the requested odors, completed trials.

        VARIANT 9 of the metric audit: this used to walk ``presentations`` with a
        ``poke_ms > 0`` filter — the fourth copy of finding 5's extractor. Both
        divergences were measured to be no-ops, so it now reads the canonical
        source. Pooling these raw samples into the A+B / Hidden Rule / Other
        series below is a display grouping, and stays here.
        """
        out: dict = {}
        pokes = poke_durations(build_position_data(td), aborted=False)
        for odor, poke_ms in zip(pokes["odor_name"], pokes["poke_time_ms"]):
            if odor is None:
                continue
            letter = odor_letter(odor)
            if letter not in odors_set:
                continue
            out.setdefault(letter, []).append(float(poke_ms))
        return out

    derivatives_dir = get_derivatives_root()

    # per_subject_days[sid][day_idx] = {series_key: [poke_ms, ...]}
    per_subject_days: dict = {}
    per_subject_days_ind: dict = {}  # same but per individual odor letter (show_lines)
    hr_active_overall = False
    observed_hr_letters = set()
    used_subj_dirs = []  # for the shared odor-colour scheme

    for sid in subjid:
        subj_date = _dates_for(sid)
        if isinstance(date, dict) and subj_date is None:
            if verbose:
                print(f"Warning: No date range provided in dict for subject {sid}, skipping")
            continue

        subj_str = normalize_subjid(sid)
        subj_dir = derivatives.subject_dir(sid, missing_ok=True)
        if subj_dir is None:
            if verbose:
                print(f"Warning: No subject directory found for {subj_str}")
            continue
        used_subj_dirs.append(subj_dir)

        # Day index = consecutive session order in the derivatives (NOT calendar
        # date), so off-days/weekends with no session don't create gaps. Day 1 is
        # the first session with data; sessions before it are skipped. After that,
        # every existing session occupies the next day slot, and a session that
        # exists but has genuinely no data for a series leaves a gap there.
        session_series = []  # per session-day: {series_key: [poke_ms, ...]}
        session_ind = []     # per session-day: {odor_letter: [poke_ms, ...]} (for show_lines)
        started = False
        for ses_dir in _filter_session_dirs(subj_dir, subj_date):
            date_str = ses_dir.name.split("_date-")[-1]
            if not (str(date_str).isdigit() and len(str(date_str)) == 8):
                continue
            results_dir = ses_dir / "saved_analysis_results"
            if not results_dir.exists():
                continue
            td = _load_trial_views(results_dir)["trial_data"]
            raw = _extract_odor_poke_ms(td)
            if not started:
                if not raw:
                    continue  # sessions before the first with data don't count
                started = True

            hr_letters = [l for l in _session_hr_odors(results_dir) if l in odors_set]
            observed_hr_letters.update(l for l in hr_letters if l not in ("A", "B"))
            session_is_hr = show_mean and len(hr_letters) >= 2
            if session_is_hr:
                hr_active_overall = True

            series: dict = {}
            for letter, vals in raw.items():
                if letter in ("A", "B"):
                    key = "AB" if ab_grouped else letter
                elif session_is_hr and letter in hr_letters:
                    key = "HR"
                elif session_is_hr:
                    # Non-A/B, non-HR-pair odor on a hidden-rule session -> pooled.
                    key = "OTHER"
                else:
                    key = letter
                series.setdefault(key, []).extend(vals)

            session_series.append(series)
            session_ind.append({l: list(v) for l, v in raw.items()})

        if not session_series:
            continue

        day_map = {i + 1: series for i, series in enumerate(session_series)}
        per_subject_days[int(sid)] = day_map
        per_subject_days_ind[int(sid)] = {i + 1: ind for i, ind in enumerate(session_ind)}

    if not per_subject_days:
        print("No data found")
        return None, None

    max_day = max(max(day_map.keys()) for day_map in per_subject_days.values())

    # Series order: requested odors in their given order (A/B collapsed to a
    # single "AB" entry when grouped), then "HR" and "OTHER" if any session used them.
    series_order = []
    for o in odors_list:
        if o in ("A", "B"):
            key = "AB" if ab_grouped else o
        else:
            key = o
        if key not in series_order:
            series_order.append(key)
    if hr_active_overall:
        series_order.append("HR")
        series_order.append("OTHER")

    fig, ax = plt.subplots(figsize=figsize)
    # Local style for the grouped view: pooled means are solid; individual odors
    # shown via show_lines are dashed in the color of their group.
    odor_colors, _ = _build_odor_colors(used_subj_dirs, odors_list)
    pooled_red = _ODOR_A_COLOR
    pooled_green = "#2E7D32"
    series_color = {
        s: (
            pooled_red if s == "AB"
            else "black" if s == "HR"
            else pooled_green if s == "OTHER"
            else odor_colors.get(s, "#000000")
        )
        for s in series_order
    }

    def _series_color(s):
        if s == "AB" or s in ("A", "B"):
            return pooled_red
        if s == "HR" or s in observed_hr_letters:
            return "black"
        if s == "OTHER":
            return pooled_green
        if s in odors_set:
            return pooled_green if observed_hr_letters else odor_colors.get(s, "#000000")
        return series_color.get(s, odor_colors.get(s, "#000000"))

    red_dash_by_letter = {
        "A": (0, (7, 3)),
        "B": (0, (2, 2)),
    }
    hr_dash_cycle = [(0, (8, 3)), (0, (3, 2)), (0, (6, 2, 1, 2))]
    other_dash_cycle = [
        (0, (6, 2)),
        (0, (2, 2)),
        (0, (5, 2, 1, 2)),
        (0, (1, 1)),
        (0, (3, 1, 1, 1, 1, 1)),
    ]
    hr_dash_by_letter = {
        letter: hr_dash_cycle[i % len(hr_dash_cycle)]
        for i, letter in enumerate(sorted(observed_hr_letters))
    }
    other_letters = [
        letter for letter in odors_list
        if letter not in observed_hr_letters and letter not in ("A", "B")
    ]
    other_dash_by_letter = {
        letter: other_dash_cycle[i % len(other_dash_cycle)]
        for i, letter in enumerate(other_letters)
    }

    def _series_linestyle(s):
        if s in ("AB", "HR", "OTHER"):
            return "-"
        if s in ("A", "B"):
            return red_dash_by_letter.get(s, "--")
        if s in observed_hr_letters:
            return hr_dash_by_letter.get(s, "--")
        if s in odors_set:
            return other_dash_by_letter.get(s, "--") if observed_hr_letters else "-"
        return "-"

    x = np.arange(1, max_day + 1)

    def _series_label(s):
        if s == "AB":
            return "Odor A+B"
        if s == "HR":
            return "Hidden Rule"
        if s == "OTHER":
            return "Other Odors"
        if s in observed_hr_letters:
            return f"Hidden Rule Odor {s}"
        return f"Odor {s}"

    plotted = set()
    if pool_subjids:
        for s_key in series_order:
            y = np.full(max_day, np.nan)
            for day in range(1, max_day + 1):
                total, count = 0.0, 0
                for day_map in per_subject_days.values():
                    vals = day_map.get(day, {}).get(s_key)
                    if vals:
                        total += sum(vals)
                        count += len(vals)
                if count > 0:
                    y[day - 1] = total / count
            if np.all(np.isnan(y)):
                continue
            plotted.add(s_key)
            ax.plot(
                x, y, color=_series_color(s_key),
                linestyle=_series_linestyle(s_key),
                linewidth=2.5, alpha=0.9, zorder=2,
            )
    else:
        for day_map in per_subject_days.values():
            for s_key in series_order:
                y = np.full(max_day, np.nan)
                for day in range(1, max_day + 1):
                    vals = day_map.get(day, {}).get(s_key)
                    if vals:
                        y[day - 1] = float(np.mean(vals))
                if np.all(np.isnan(y)):
                    continue
                plotted.add(s_key)
                ax.plot(
                    x, y, color=_series_color(s_key),
                    linestyle=_series_linestyle(s_key),
                    linewidth=2.0, alpha=0.7, zorder=2,
                )

    # Optionally overlay the individual odors that make up each mean, as thin
    # low-alpha dashed lines in their group colour (only meaningful with show_mean).
    if show_mean and show_lines:
        for letter in odors_list:
            color = _series_color(letter)
            linestyle = _series_linestyle(letter)
            if pool_subjids:
                y = np.full(max_day, np.nan)
                for day in range(1, max_day + 1):
                    total, count = 0.0, 0
                    for dm in per_subject_days_ind.values():
                        vals = dm.get(day, {}).get(letter)
                        if vals:
                            total += sum(vals)
                            count += len(vals)
                    if count > 0:
                        y[day - 1] = total / count
                if not np.all(np.isnan(y)):
                    plotted.add(letter)
                    ax.plot(
                        x, y, color=color, linestyle=linestyle,
                        linewidth=1.0, alpha=0.45, zorder=1.5,
                    )
            else:
                for dm in per_subject_days_ind.values():
                    y = np.full(max_day, np.nan)
                    for day in range(1, max_day + 1):
                        vals = dm.get(day, {}).get(letter)
                        if vals:
                            y[day - 1] = float(np.mean(vals))
                    if not np.all(np.isnan(y)):
                        plotted.add(letter)
                        ax.plot(
                            x, y, color=color, linestyle=linestyle,
                            linewidth=0.9, alpha=0.4, zorder=1.5,
                        )

    ax.set_xlabel("Day")
    ax.set_ylabel("Poke Duration (ms)")
    ax.set_xlim(0.8, max_day + 0.5)
    ax.set_ylim(bottom=0)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    # Legend: the pooled/individual series in series_order, plus any individual
    # odors split out by show_lines that series_order collapsed (A/B into "AB"),
    # each with its own scheme colour.
    legend_series = [s for s in series_order if s in plotted]
    legend_series += [l for l in odors_list if l in plotted and l not in legend_series]
    if legend_series:
        handles = [
            Line2D([0], [0],
                   color=_series_color(s),
                   linestyle=_series_linestyle(s),
                   linewidth=2.0, label=_series_label(s))
            for s in legend_series
        ]
        ax.legend(handles=handles, title="Odor", loc="best")

    if show_title:
        ax.set_title(title if title else "Poke Duration by Odor over Day")

    fig.tight_layout(pad=1.5)

    if save:
        try:
            if isinstance(date, dict):
                save_dates = []
                for v in date.values():
                    if isinstance(v, (list, tuple)):
                        save_dates.extend(v)
                    elif v is not None:
                        save_dates.append(v)
            else:
                save_dates = date
            out_path = save_figure(
                fig, "poke_duration_by_odor",
                subjids=list(subjid), dates=save_dates,
            )
            if verbose:
                print(f"[plot_poke_duration_by_odor] Saved figure to {out_path}")
        except Exception as exc:
            if verbose:
                print(f"[plot_poke_duration_by_odor] Failed to save figure: {exc}")

    plt.show()
    return fig, ax


# ================================= Debugging / Testing ================================= #


def plot_fa_ratio_by_hr_position(
    subjid,
    dates=None,
    figsize=(16, 10),
    fa_types='FA_time_in', 
    print_statistics=False,
    exclude_last_pos=False,
    last_odor_num=5,
    debug=False
):
    """
    Plot FA Ratio (A-B)/(A+B) by hidden rule odor position across sessions.
    
    For each session and each HR odor, calculates:
    1. FA on HR Odor at HR position
    2. FA at the next odor in sequence (position-independent)
    3. Total FA at or after HR position
    
    Parameters:
    -----------
    subjid : int
        Subject ID
    dates : tuple, list, or None
        Date or date range. If None, plots all available dates.
    figsize : tuple, optional
        Figure size (default: (16, 10))
    fa_types : str or list, optional
        Which FA types to include:
        - 'FA_time_in' : only FA_time_in
        - 'FA_time_in,FA_time_out' : multiple specific types (comma-separated)
        - 'All' : all FA types starting with 'FA_'
        (default: 'FA_time_in')
    print_statistics: bool, optional
        Whether to print a statistic summary table with counts for each
        FA type and position (default: False).
    exclude_last_pos: bool, optional
        If True, exclude FAs where last_odor_position == last_odor_num from all calculations.
        If False (default), include all positions.
    last_odor_num: int
        defines what position last odor is for possible exclusion of rewarded odors 
    
    Returns:
    --------
    fig, axes : matplotlib figure and axes array
    """
    base_path = get_rawdata_root()
    server_root = get_server_root()
    derivatives_dir = get_derivatives_root()
    
    # Parse FA type filter
    if isinstance(fa_types, str):
        if fa_types.lower() == 'all':
            fa_filter_fn = lambda fa_label: str(fa_label).startswith('FA_') if pd.notna(fa_label) else False
        else:
            types_list = [t.strip().lower() for t in fa_types.split(',')]
            fa_filter_fn = lambda fa_label: str(fa_label).lower() in types_list if pd.notna(fa_label) else False
    else:
        fa_filter_fn = lambda fa_label: True
    
    rows = []  # {date, session_num, odor_num, hr_odor, category, port_a, port_b, total, ratio}
    
    for sid, subj_dir in _iter_subject_dirs(derivatives_dir, [subjid]):
        ses_dirs = _filter_session_dirs(subj_dir, dates)
        
        for session_num, ses_dir in enumerate(ses_dirs, 1):
            date_str = ses_dir.name.split("_date-")[-1]
            results_dir = ses_dir / "saved_analysis_results"
            
            if not results_dir.exists():
                continue
            
            summary_path = results_dir / "summary.json"
            if not summary_path.exists():
                continue

            try:
                with open(summary_path) as f:
                    summary = json.load(f)
                # Prefer params.hidden_rule_odors, fall back to first run.stage.hidden_rule_odors if absent
                hr_odors = summary.get("params", {}).get("hidden_rule_odors", [])
                if not hr_odors:
                    runs = summary.get("session", {}).get("runs", [])
                    if runs and isinstance(runs[0], dict):
                        stage = runs[0].get("stage", {}) if isinstance(runs[0].get("stage", {}), dict) else {}
                        hr_odors = stage.get("hidden_rule_odors", []) or stage.get("hidden_rule_odors".lower(), [])
                if not hr_odors:
                    continue

                views = _load_trial_views(results_dir)
                df_hr = views.get("aborted_hr", pd.DataFrame())
                df_ab = views.get("aborted", pd.DataFrame())
                if df_hr.empty or df_ab.empty:
                    continue

                # Require FA detail columns; skip session cleanly if absent
                needed_cols = {"fa_label", "last_odor_name", "fa_port", "last_odor_position", "sequence_start"}
                if not needed_cols.issubset(df_ab.columns) or "sequence_start" not in df_hr.columns:
                    if debug:
                        missing = needed_cols - set(df_ab.columns)
                    continue

                # Match HR trials with aborted sequences
                hr_with_fa = df_hr[df_hr["sequence_start"].isin(df_ab["sequence_start"])].copy()

                # Merge to get FA details (avoid duplicate suffixes when hr_with_fa already has these cols)
                merged = hr_with_fa.copy()
                fa_cols = ["fa_label", "last_odor_name", "fa_port", "last_odor_position"]
                missing_fa_cols = [c for c in fa_cols if c not in merged.columns]
                if missing_fa_cols:
                    merged = merged.merge(
                        df_ab[["sequence_start", *missing_fa_cols]],
                        on="sequence_start",
                        how="left",
                        suffixes=("", "_fa")
                    )

                # Coalesce any suffixed duplicates that may still appear
                for col in fa_cols:
                    if col not in merged.columns:
                        if f"{col}_fa" in merged.columns:
                            merged[col] = merged[f"{col}_fa"]
                        elif f"{col}_x" in merged.columns or f"{col}_y" in merged.columns:
                            merged[col] = merged.get(f"{col}_x", merged.get(f"{col}_y"))

                # Add HR position info and odor_sequence from HR data
                hr_cols_to_merge = ["sequence_start"]
                for hr_col in ["hidden_rule_positions", "odor_sequence"]:
                    if hr_col in df_hr.columns and hr_col not in merged.columns:
                        hr_cols_to_merge.append(hr_col)
                if len(hr_cols_to_merge) > 1:
                    merged = merged.merge(
                        df_hr[hr_cols_to_merge],
                        on="sequence_start",
                        how="left",
                        suffixes=('', '_hr')
                    )

                if "fa_label" not in merged.columns:
                    if debug:
                        print(f"[DEBUG {date_str}] skipped: merged has no fa_label column; merged cols={list(merged.columns)}")
                    continue
                merged_fa = merged[
                    merged["fa_label"].notna() &
                    (merged["fa_label"] != "nFA") & 
                    (merged["fa_label"].apply(fa_filter_fn))
                ].copy()
                if merged_fa.empty and debug:
                    print(f"[DEBUG {date_str}] skipped: no FA rows after filtering (fa_types={fa_types})")
                
                # Optionally exclude FAs at the specified last_odor_position
                if exclude_last_pos:
                    merged_fa = merged_fa[merged_fa["last_odor_position"] != last_odor_num].copy()
                    if merged_fa.empty and debug:
                        print(f"[DEBUG {date_str}] skipped: all FAs at excluded position {last_odor_num}")
                
                if merged_fa.empty:
                    continue
                
                if "hidden_rule_positions" not in merged_fa.columns:
                    if debug:
                        print(f"[DEBUG {date_str}] skipped: hidden_rule_positions column missing")
                    continue
                
                # Helper functions
                def count_ports(data):
                    """`fa_port_counts` plus the total, which the rows below carry."""
                    port_a, port_b = fa_port_counts(data)
                    return port_a, port_b, port_a + port_b
                
                def get_hr_position(hr_pos_str):
                    if pd.isna(hr_pos_str):
                        return None
                    try:
                        pos_list = json.loads(str(hr_pos_str))
                        if isinstance(pos_list, list) and len(pos_list) > 0:
                            return int(pos_list[0])
                    except:
                        pass
                    return None
                
                def has_hr_odor_in_sequence(odor_seq, hr_odor):
                    if pd.isna(odor_seq):
                        return False
                    try:
                        seq_list = json.loads(str(odor_seq))
                        return hr_odor in seq_list if isinstance(seq_list, list) else False
                    except:
                        return hr_odor in str(odor_seq)
                
                # Analyze each HR odor
                for odor_num, hr_odor in enumerate(hr_odors, 1):
                    # Filter to trials where this HR odor appears in the sequence
                    if "odor_sequence" in merged_fa.columns:
                        fa_for_this_hr = merged_fa[
                            merged_fa["odor_sequence"].apply(lambda seq: has_hr_odor_in_sequence(seq, hr_odor))
                        ].copy()
                    else:
                        fa_for_this_hr = merged_fa.copy()
                    
                    if fa_for_this_hr.empty:
                        if debug:
                            print(f"[DEBUG {date_str}] HR odor {hr_odor}: no trials with odor in sequence")
                        continue
                    
                    # Extract HR position
                    fa_for_this_hr["hr_position"] = fa_for_this_hr["hidden_rule_positions"].apply(get_hr_position)
                    fa_for_this_hr = fa_for_this_hr[fa_for_this_hr["hr_position"].notna()]

                    if fa_for_this_hr.empty:
                        if debug:
                            print(f"[DEBUG {date_str}] HR odor {hr_odor}: no parsable hr_position")
                        continue
                    
                    # Category 1: FA on HR odor itself at HR position
                    fa_on_hr_odor = fa_for_this_hr[
                        (fa_for_this_hr["last_odor_name"] == hr_odor) & 
                        (fa_for_this_hr["last_odor_position"] == fa_for_this_hr["hr_position"])
                    ].copy()
                    a1, b1, t1 = count_ports(fa_on_hr_odor)
                    ratio1 = fa_port_ratio(a1, b1)
                    rows.append({
                        "date": int(date_str),
                        "session_num": session_num,
                        "odor_num": odor_num,
                        "hr_odor": hr_odor,
                        "category": f"On {hr_odor}",
                        "port_a": a1,
                        "port_b": b1,
                        "total": t1,
                        "ratio": ratio1
                    })
                    
                    # Category 2: FA at next odor after HR odor (position-based, not odor-based)
                    # Find the position of the HR odor first
                    fa_one_after = fa_for_this_hr[
                        (fa_for_this_hr["last_odor_position"] == fa_for_this_hr["hr_position"] + 1)
                    ].copy()
                    a2, b2, t2 = count_ports(fa_one_after)
                    ratio2 = fa_port_ratio(a2, b2)
                    rows.append({
                        "date": int(date_str),
                        "session_num": session_num,
                        "odor_num": odor_num,
                        "hr_odor": hr_odor,
                        "category": f"After {hr_odor}",
                        "port_a": a2,
                        "port_b": b2,
                        "total": t2,
                        "ratio": ratio2
                    })
                    
                    # Category 3: Total FA at or after HR position
                    fa_total = fa_for_this_hr[
                        (fa_for_this_hr["last_odor_position"] >= fa_for_this_hr["hr_position"])
                    ].copy()
                    a3, b3, t3 = count_ports(fa_total)
                    ratio3 = fa_port_ratio(a3, b3)
                    rows.append({
                        "date": int(date_str),
                        "session_num": session_num,
                        "odor_num": odor_num,
                        "hr_odor": hr_odor,
                        "category": f"Total {hr_odor}",
                        "port_a": a3,
                        "port_b": b3,
                        "total": t3,
                        "ratio": ratio3
                    })
                
            except Exception as e:
                print(f"Error processing date {date_str}: {e}")
                continue
    
    if not rows:
        print("No data found for FA ratio analysis by HR position")
        return None, None
    
    df = pd.DataFrame(rows)
    
    # Get unique HR odors and create subplots: 2 rows (scatter + line) per odor
    unique_odors = sorted(df["hr_odor"].unique())
    n_odors = len(unique_odors)
    
    fig, axes = plt.subplots(2, n_odors, figsize=(figsize[0], figsize[1] * 1.5))
    if n_odors == 1:
        axes = axes.reshape(2, 1)
    
    for ax_idx, hr_odor in enumerate(unique_odors):
        # ===== TOP ROW: Scatter plot by category =====
        ax_scatter = axes[0, ax_idx]
        
        df_odor = df[df["hr_odor"] == hr_odor].copy()
        
        # Define X positions for the 3 categories
        categories = [f"On {hr_odor}", f"After {hr_odor}", f"Total {hr_odor}"]
        x_positions = {cat: i for i, cat in enumerate(categories)}
        
        # Plot each session as a dot
        for cat_idx, category in enumerate(categories):
            df_cat = df_odor[df_odor["category"] == category]
            
            if not df_cat.empty:
                ratios = df_cat["ratio"].dropna()
                if not ratios.empty:
                    x_jitter = np.random.normal(cat_idx, 0.08, size=len(ratios))
                    ax_scatter.scatter(x_jitter, ratios, alpha=0.5, s=40, color='steelblue')
        
        ax_scatter.set_xticks(range(len(categories)))
        ax_scatter.set_xticklabels(categories, fontsize=10, fontweight='bold')
        ax_scatter.set_ylabel('FA Ratio (A-B)/(A+B)', fontsize=11, fontweight='bold')
        ax_scatter.set_ylim([-1.1, 1.1])
        ax_scatter.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.7)
        ax_scatter.set_title(f'HR Odor: {hr_odor} - By Category\n(Subject {str(subjid).zfill(3)})', 
                    fontsize=12, fontweight='bold')
        
        # ===== BOTTOM ROW: Line plot across sessions =====
        ax_line = axes[1, ax_idx]
        
        # Sort by date to get consecutive sessions
        df_odor_sorted = df_odor.sort_values(by="date")
        
        # Create session mapping: consecutive integers 0 to end
        unique_dates = sorted(df_odor_sorted["date"].unique())
        date_to_session = {d: i for i, d in enumerate(unique_dates)}
        df_odor_sorted["session_idx"] = df_odor_sorted["date"].map(date_to_session)
        
        # Define line properties for each category
        line_config = {
            f"On {hr_odor}": {"color": "blue", "label": f"On {hr_odor}"},
            f"After {hr_odor}": {"color": "green", "label": f"After {hr_odor}"},
            f"Total {hr_odor}": {"color": "black", "label": f"Total {hr_odor}"}
        }
        
        # Plot line for each category
        for category, config in line_config.items():
            df_cat = df_odor_sorted[df_odor_sorted["category"] == category].sort_values(by="session_idx")
            
            if not df_cat.empty and not df_cat["ratio"].isna().all():
                # Get data with values
                df_cat_valid = df_cat[df_cat["ratio"].notna()].copy()
                
                if not df_cat_valid.empty:
                    # Check if there are any gaps (missing sessions)
                    session_indices = df_cat_valid["session_idx"].values
                    all_sessions_present = len(session_indices) == (session_indices[-1] - session_indices[0] + 1)
                    
                    # Use dotted line if there are gaps in the data
                    linestyle = '-' if all_sessions_present else ':'
                    
                    ax_line.plot(df_cat_valid["session_idx"], df_cat_valid["ratio"], 
                                color=config["color"], 
                                label=config["label"],
                                linewidth=2,
                                linestyle=linestyle,
                                marker='o',
                                markersize=5,
                                alpha=0.7)
        
        ax_line.set_xlabel('Session Number', fontsize=11, fontweight='bold')
        ax_line.set_ylabel('FA Ratio (A-B)/(A+B)', fontsize=11, fontweight='bold')
        ax_line.set_ylim([-1.1, 1.1])
        ax_line.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.7)
        
        # Set x-axis to whole numbers
        max_session = int(df_odor_sorted["session_idx"].max())
        ax_line.set_xticks(range(0, max_session + 1))
        
        ax_line.legend(loc='best', fontsize=9)
        ax_line.set_title(f'HR Odor: {hr_odor} - Across Sessions\n(Subject {str(subjid).zfill(3)})', 
                    fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    if print_statistics:
        # Display summary table sorted by metric, then ratio ascending
        print("\n" + "="*100)
        print("SUMMARY TABLE (SORTED BY METRIC, THEN BY RATIO)")
        print(f"Note: Only showing dates where FA data was found on HR trials (Subject {str(subjid).zfill(3)})")
        print("="*100)
        
        # Create metric name by combining HR odor and category
        df_display = df.copy()
        df_display["metric"] = df_display["hr_odor"] + " - " + df_display["category"]
        
        # Select and order columns
        df_summary = df_display[["date", "metric", "port_a", "port_b", "total", "ratio"]].copy()
        df_summary = df_summary.sort_values(by=["metric", "ratio"], na_position='last')
        
        # Format ratio display
        df_summary["ratio"] = df_summary["ratio"].apply(
            lambda x: f"{x:+.3f}" if not pd.isna(x) else "N/A"
        )
        
        # Display table
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        pd.set_option('display.max_colwidth', None)
        print(df_summary.to_string(index=False))
        
        # Statistics per metric
        print("\n" + "="*100)
        print("STATISTICS (PER METRIC)")
        print("="*100)
        
        # Convert ratio back to numeric for stats
        df_stats = df_display[["metric", "ratio"]].copy()
        
        for metric in sorted(df_stats["metric"].unique()):
            metric_data = df_stats[df_stats["metric"] == metric]["ratio"].dropna()
            
            if len(metric_data) > 0:
                mean_val = metric_data.mean()
                min_val = metric_data.min()
                max_val = metric_data.max()
                std_val = metric_data.std()
                
                print(f"{metric}:")
                print(f"  Mean:  {mean_val:+.3f}")
                print(f"  Min:   {min_val:+.3f}")
                print(f"  Max:   {max_val:+.3f}")
                print(f"  Std:   {std_val:.3f}")
                print()
    
    return fig, axes


def plot_fa_ratio_by_abort_odor(
    subjid,
    dates=None,
    figsize=(18, 8),
    fa_types='FA_time_in'
):
    """
    Plot FA Ratio (A-B)/(A+B) by abortion odor, comparing HR and non-HR aborted sequences.
    
    For each odor where abortion occurred, compares:
    1. Aborted HR trials where abortion happens AFTER the HR odor (not on the HR)
    2. Aborted non-HR trials (no HR present in sequence)
    
    Only includes trials that match the FA type filter. FA Ratio is calculated for each category.
    
    Parameters:
    -----------
    subjid : int
        Subject ID
    dates : tuple, list, or None
        Date or date range. If None, plots all available dates.
    figsize : tuple, optional
        Figure size (default: (14, 8))
    fa_types : str or list, optional
        Which FA types to include:
        - 'FA_time_in' : only FA_time_in
        - 'FA_time_in,FA_time_out' : multiple specific types (comma-separated)
        - 'All' : all FA types starting with 'FA_'
        (default: 'FA_time_in')
    
    Returns:
    --------
    fig, axes : matplotlib figure and axes array
    """
    base_path = get_rawdata_root()
    server_root = get_server_root()
    derivatives_dir = get_derivatives_root()
    
    # Parse FA type filter
    if isinstance(fa_types, str):
        if fa_types.lower() == 'all':
            fa_filter_fn = lambda fa_label: str(fa_label).startswith('FA_') if pd.notna(fa_label) else False
        else:
            types_list = [t.strip().lower() for t in fa_types.split(',')]
            fa_filter_fn = lambda fa_label: str(fa_label).lower() in types_list if pd.notna(fa_label) else False
    else:
        fa_filter_fn = lambda fa_label: True
    
    rows = []  # {date, odor, hr_odor, category, port_a, port_b, total, ratio}
    
    # Statistics tracking
    stats = {
        'total_no_hr': 0,
        'total_no_hr_fa': 0,
        'total_hr': 0,
        'total_hr_fa': 0
    }
    
    for sid, subj_dir in _iter_subject_dirs(derivatives_dir, [subjid]):
        ses_dirs = _filter_session_dirs(subj_dir, dates)
        
        for session_num, ses_dir in enumerate(ses_dirs, 1):
            date_str = ses_dir.name.split("_date-")[-1]
            results_dir = ses_dir / "saved_analysis_results"
            
            if not results_dir.exists():
                continue
            
            summary_path = results_dir / "summary.json"
            if not summary_path.exists():
                continue

            try:
                with open(summary_path) as f:
                    summary = json.load(f)

                hr_odors = summary.get("params", {}).get("hidden_rule_odors", [])
                if not hr_odors:
                    continue

                views = _load_trial_views(results_dir)
                df_hr = views.get("aborted_hr", pd.DataFrame())
                df_ab = views.get("aborted", pd.DataFrame())
                if df_hr.empty or df_ab.empty:
                    continue
                
                # ===== PROCESS ABORTED HR TRIALS (abortion after HR) =====
                if "sequence_start" in df_hr.columns and "sequence_start" in df_ab.columns:
                    # Get HR trials with FA
                    hr_with_fa = df_hr[df_hr["sequence_start"].isin(df_ab["sequence_start"])].copy()
                    
                    if not hr_with_fa.empty:
                        # Merge with FA details while avoiding duplicate suffixes
                        merged_hr = hr_with_fa.copy()
                        fa_cols = ["fa_label", "last_odor_name", "fa_port", "last_odor_position"]
                        missing_fa_cols = [c for c in fa_cols if c not in merged_hr.columns]
                        if missing_fa_cols:
                            merged_hr = merged_hr.merge(
                                df_ab[["sequence_start", *missing_fa_cols]],
                                on="sequence_start",
                                how="left",
                                suffixes=("", "_fa")
                            )

                        # Coalesce any suffixed duplicates
                        for col in fa_cols:
                            if col not in merged_hr.columns:
                                if f"{col}_fa" in merged_hr.columns:
                                    merged_hr[col] = merged_hr[f"{col}_fa"]
                                elif f"{col}_x" in merged_hr.columns or f"{col}_y" in merged_hr.columns:
                                    merged_hr[col] = merged_hr.get(f"{col}_x", merged_hr.get(f"{col}_y"))

                        # Add HR position/odor sequence info if missing
                        hr_cols_to_merge = ["sequence_start"]
                        for hr_col in ["hidden_rule_positions", "odor_sequence"]:
                            if hr_col in df_hr.columns and hr_col not in merged_hr.columns:
                                hr_cols_to_merge.append(hr_col)
                        if len(hr_cols_to_merge) > 1:
                            merged_hr = merged_hr.merge(
                                df_hr[hr_cols_to_merge],
                                on="sequence_start",
                                how="left",
                                suffixes=('', '_hr')
                            )

                        if "fa_label" not in merged_hr.columns:
                            # Still no FA column; skip safely
                            continue

                        # Filter for actual FAs and apply FA type filter
                        merged_hr = merged_hr[
                            (merged_hr["fa_label"] != "nFA") & 
                            (merged_hr["fa_label"].apply(fa_filter_fn))
                        ].copy()
                        
                        stats['total_hr'] += len(hr_with_fa)
                        stats['total_hr_fa'] += len(merged_hr)
                        
                        if not merged_hr.empty:
                            # Filter to abortions that happen AFTER the HR (not on the HR)
                            def get_hr_position(hr_pos_str):
                                if pd.isna(hr_pos_str):
                                    return None
                                try:
                                    pos_list = json.loads(str(hr_pos_str))
                                    if isinstance(pos_list, list) and len(pos_list) > 0:
                                        return int(pos_list[0])
                                except:
                                    pass
                                return None
                            
                            merged_hr["hr_position"] = merged_hr["hidden_rule_positions"].apply(get_hr_position)
                            
                            # Keep only trials where abortion happens AFTER HR position
                            before_after_filter = len(merged_hr)
                            merged_hr = merged_hr[
                                merged_hr["last_odor_position"] > merged_hr["hr_position"]
                            ].copy()
                            stats['total_hr_fa_after_pos'] = stats.get('total_hr_fa_after_pos', 0) + len(merged_hr)
                            stats['total_hr_fa_lost_to_position'] = stats.get('total_hr_fa_lost_to_position', 0) + (before_after_filter - len(merged_hr))
                            
                            if not merged_hr.empty:
                                # Group by last odor and HR odor
                                for last_odor in merged_hr["last_odor_name"].unique():
                                    odor_data = merged_hr[merged_hr["last_odor_name"] == last_odor]
                                    
                                    for hr_odor in hr_odors:
                                        # Check if this HR odor is in the sequence for this trial
                                        odor_matches = []
                                        
                                        if "odor_sequence" in odor_data.columns:
                                            def has_hr_odor(odor_seq, target_hr):
                                                if pd.isna(odor_seq):
                                                    return False
                                                try:
                                                    seq_list = json.loads(str(odor_seq))
                                                    return target_hr in seq_list if isinstance(seq_list, list) else False
                                                except:
                                                    return target_hr in str(odor_seq)
                                            
                                            odor_matches = odor_data[
                                                odor_data["odor_sequence"].apply(lambda seq: has_hr_odor(seq, hr_odor))
                                            ]
                                        else:
                                            odor_matches = odor_data
                                        
                                        if not odor_matches.empty:
                                            port_a, port_b = fa_port_counts(odor_matches)
                                            total = port_a + port_b
                                            ratio = fa_port_ratio(port_a, port_b)
                                            
                                            rows.append({
                                                "date": int(date_str),
                                                "odor": last_odor,
                                                "category": hr_odor,
                                                "port_a": port_a,
                                                "port_b": port_b,
                                                "total": total,
                                                "ratio": ratio
                                            })
                
                # ===== PROCESS ABORTED NON-HR TRIALS =====
                # Get trials that are NOT in HR file (no HR present)
                if "sequence_start" in df_ab.columns:
                    ab_no_hr = df_ab[~df_ab["sequence_start"].isin(df_hr["sequence_start"].values)].copy()
                    
                    stats['total_no_hr'] += len(ab_no_hr)
                    
                    # Filter for actual FAs and apply FA type filter
                    ab_no_hr = ab_no_hr[
                        (ab_no_hr["fa_label"] != "nFA") & 
                        (ab_no_hr["fa_label"].apply(fa_filter_fn))
                    ].copy()
                    
                    stats['total_no_hr_fa'] += len(ab_no_hr)
                    
                    if not ab_no_hr.empty:
                        # Track how many go into breakdown
                        before_breakdown = len(ab_no_hr)
                        # Group by last odor
                        for last_odor in ab_no_hr["last_odor_name"].unique():
                            odor_data = ab_no_hr[ab_no_hr["last_odor_name"] == last_odor]
                            
                            port_a, port_b = fa_port_counts(odor_data)
                            total = port_a + port_b
                            ratio = fa_port_ratio(port_a, port_b)
                            
                            rows.append({
                                "date": int(date_str),
                                "odor": last_odor,
                                "category": "No HR",
                                "port_a": port_a,
                                "port_b": port_b,
                                "total": total,
                                "ratio": ratio
                            })
                        stats['total_no_hr_in_breakdown'] = stats.get('total_no_hr_in_breakdown', 0) + sum(
                            row['total'] for row in rows if row.get('category') == 'No HR' and row.get('date') == int(date_str)
                        )
            
            except Exception as e:
                print(f"Error processing date {date_str}: {e}")
                continue
    
    if not rows:
        print("No data found for FA ratio by abort odor")
        return None, None
    
    df = pd.DataFrame(rows)
    
    # Get unique odors and filter out rewarded odors (OdorA, OdorB)
    all_unique_odors = sorted(df["odor"].unique())
    rewarded_odors = ['OdorA', 'OdorB']
    unique_odors = [odor for odor in all_unique_odors if odor not in rewarded_odors]
    
    # Still print stats for all odors including rewarded ones
    n_odors = len(unique_odors)
    
    # Create subplots: one per odor
    fig, axes = plt.subplots(1, n_odors, figsize=(figsize[0] * 0.85, figsize[1] * 0.9) if n_odors > 2 else figsize)
    if n_odors == 1:
        axes = np.array([axes])
    else:
        axes = np.atleast_1d(axes)
    
    # Define category order
    category_order = []
    if "No HR" in df["category"].unique():
        category_order.append("No HR")
    category_order.extend(sorted([c for c in df["category"].unique() if c != "No HR"]))
    
    # Create session gradient colormap: dark blue for recent, light blue for older
    unique_dates_sorted = sorted(df["date"].unique())
    n_sessions = len(unique_dates_sorted)
    
    # Create color map: most recent = dark blue, oldest = light blue
    if n_sessions == 1:
        colors_for_dates = {unique_dates_sorted[0]: '#00008B'}  # Dark blue
    else:
        # Linear interpolation from light to dark blue
        blue_light = np.array([0.68, 0.85, 1.0])      # Light blue
        blue_dark = np.array([0.0, 0.0, 0.55])        # Dark blue
        colors_for_dates = {}
        for idx, date in enumerate(unique_dates_sorted):
            t = idx / (n_sessions - 1)  # 0 for oldest, 1 for newest
            color = blue_light * (1 - t) + blue_dark * t
            colors_for_dates[date] = color
    
    # Debug: Show how many sessions we have data from
    print(f"\nDEBUG: Data aggregated from {len(unique_dates_sorted)} sessions on dates: {sorted(unique_dates_sorted)}")
    print(f"DEBUG: Color mapping: {unique_dates_sorted} → Most recent (dark) to oldest (light)")
    print(f"DEBUG: Total rows in breakdown dataframe: {len(df)}")
    
    
    # Plot for each odor
    for ax_idx, odor in enumerate(unique_odors):
        ax = axes[ax_idx] if n_odors > 1 else axes[0]
        
        df_odor = df[df["odor"] == odor].copy()
        
        # For this specific odor, only include categories that have data
        categories_with_data = sorted([c for c in df_odor["category"].unique()])
        if not categories_with_data:
            continue
        
        x_positions = {cat: i for i, cat in enumerate(categories_with_data)}
        
        # Scatter plot with session gradient coloring
        for category in categories_with_data:
            df_cat = df_odor[df_odor["category"] == category]
            
            if not df_cat.empty:
                # Plot each date separately with its own color
                for date in unique_dates_sorted:
                    df_date = df_cat[df_cat["date"] == date]
                    if df_date.empty:
                        continue
                    
                    ratios = df_date["ratio"].dropna()
                    if not ratios.empty:
                        x_pos = x_positions[category]
                        # Add small jitter to spread out points
                        x_jitter = np.random.normal(x_pos, 0.06, size=len(ratios))
                        color = colors_for_dates[date]
                        ax.scatter(x_jitter, ratios, alpha=0.7, s=80, color=color, 
                                  edgecolors='none', label=f'{date}' if ax_idx == 0 else '')
        
        # Add black line for aggregate mean for each category that actually has data in this odor
        # Line width scales with number of categories (smaller when fewer categories)
        line_half_width = 0.15 if len(categories_with_data) > 1 else 0.08
        for category in categories_with_data:
            df_cat = df_odor[df_odor["category"] == category]
            all_ratios = df_cat["ratio"].dropna()
            if len(all_ratios) > 0:
                mean_ratio = all_ratios.mean()
                x_pos = x_positions[category]
                # Only draw line if we have data at this position
                ax.plot([x_pos - line_half_width, x_pos + line_half_width], [mean_ratio, mean_ratio], 
                       color='black', linewidth=3, alpha=0.8, zorder=10)
        
        ax.set_xticks(range(len(categories_with_data)))
        ax.set_xticklabels(categories_with_data, fontsize=10, fontweight='bold', rotation=0)
        ax.set_ylabel('FA Ratio (A-B)/(A+B)', fontsize=11, fontweight='bold')
        ax.set_ylim([-1.1, 1.1])
        ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.7)
        ax.set_title(f'{odor}', fontsize=12, fontweight='bold')
        
        # Set x-axis limits with padding
        n_cats = len(categories_with_data)
        ax.set_xlim(-0.5, n_cats - 0.5)
        ax.margins(y=0)  # Only apply margins to y-axis, not x-axis
    
    # Create a legend for the sessions (on the first subplot)
    if n_odors > 0:
        # Create custom legend entries
        legend_elements = []
        for date in reversed(unique_dates_sorted):  # Reverse so newest is first
            label = f'{date}'
            if date == unique_dates_sorted[-1]:
                label += ' (recent)'
            legend_elements.append(Line2D([0], [0], marker='o', color='w', 
                                         markerfacecolor=colors_for_dates[date], 
                                         markersize=8, label=label, alpha=0.7))
        
        fig.legend(handles=legend_elements, loc='upper right', fontsize=9, 
                  title='Sessions', title_fontsize=10, framealpha=0.95)
    
    plt.tight_layout(rect=[0, 0, 0.88, 1])  # Leave space for legend
    
    # Print statistics
    print("\n" + "="*100)
    print("FA RATIO BY ABORTION ODOR - STATISTICS")
    print("="*100)
    print(f"\nAborted Sequences WITHOUT Hidden Rule:")
    print(f"  Total aborted: {stats['total_no_hr']}")
    print(f"  Matching FA filter: {stats['total_no_hr_fa']}")
    print(f"  In breakdown by odor: {stats.get('total_no_hr_in_breakdown', 'unknown')}")
    
    # Calculate actual HR breakdown count
    hr_breakdown_count = sum(row['total'] for row in rows if row.get('category') != 'No HR')
    
    print(f"\nAborted Sequences WITH Hidden Rule (abortion AFTER HR):")
    print(f"  Total aborted: {stats['total_hr']}")
    print(f"  Matching FA filter: {stats['total_hr_fa']}")
    print(f"  After position filter (after HR): {stats.get('total_hr_fa_after_pos', 'unknown')}")
    print(f"  In breakdown table: {hr_breakdown_count}")
    print(f"\nDISCREPANCY ANALYSIS:")
    print(f"  Non-HR: FA filter count ({stats['total_no_hr_fa']}) vs breakdown count ({stats.get('total_no_hr_in_breakdown', 'unknown')})")
    print(f"  HR: FA filter count ({stats['total_hr_fa']}) vs breakdown count ({hr_breakdown_count})")
    print(f"  Missing HR trials in breakdown: {stats['total_hr_fa'] - hr_breakdown_count}")
    
    print(f"\n" + "-"*100)
    print("BREAKDOWN BY ODOR AND CATEGORY (including rewarded odors OdorA, OdorB in stats):")
    print("-"*100)
    
    # Group by odor and show per-date breakdown for ALL odors
    for odor in all_unique_odors:
        is_rewarded = odor in rewarded_odors
        odor_label = f"{odor}" + (" [REWARDED - not plotted]" if is_rewarded else "")
        print(f"\n{odor_label}:")
        df_odor = df[df["odor"] == odor]
        
        for category in category_order:
            df_cat = df_odor[df_odor["category"] == category]
            
            if not df_cat.empty:
                # Show aggregate across all dates
                # Counts pool across dates (a DISPLAY-AGG); the ratio over them is
                # still the canonical one -- `total` is `port_a + port_b` by
                # construction above, so this is `fa_port_ratio` exactly.
                port_a_total = df_cat["port_a"].sum()
                port_b_total = df_cat["port_b"].sum()
                total_trials = df_cat["total"].sum()
                ratio_agg = fa_port_ratio(port_a_total, port_b_total)
                
                ratio_str = f"{ratio_agg:+.3f}" if not pd.isna(ratio_agg) else "N/A"
                print(f"  {category:<12} - Ratio: {ratio_str}  Port A: {int(port_a_total)}, Port B: {int(port_b_total)}, Total: {int(total_trials)}")
                
                # Show per-date breakdown
                for idx, row in df_cat.iterrows():
                    date_val = int(row['date'])
                    ratio_str_date = f"{row['ratio']:+.3f}" if not pd.isna(row['ratio']) else "N/A"
                    print(f"      → {date_val}: Port A: {int(row['port_a'])}, Port B: {int(row['port_b'])}, Total: {int(row['total'])}")
            else:
                print(f"  {category:<12} - No data")
    
    print("="*100)
        
    # Show summary totals
    print("\nSUMMARY BY CATEGORY (across all odors and dates):")
    print("-"*100)
    
    total_no_hr_all = df[df["category"] == "No HR"]["total"].sum()
    total_hr_all = df[df["category"] != "No HR"]["total"].sum()
    
    print(f"No HR trials total: {int(total_no_hr_all)}")
    print(f"HR trials total: {int(total_hr_all)}")

    return fig, axes


def plot_hidden_rule_abort_poke_gap(
    subjid: int,
    dates=None,
    fa_types=None,
    figsize=(10, 6),
    ax=None,
    ax_start_end=None,
    make_second_plot: bool = True,
    return_both: bool = False,
    *,
    save: bool = False,
    verbose: bool = True,
):
    """
    For aborted trials that hit the hidden rule, compute the latency between the
    hidden-rule odor's `poke_odor_end` and the last `poke_odor_end` in that trial.

    If `fa_types` is provided (e.g., ["FA_time_in", "FA_time_out"]), trials with
    `fa_label` in that set are plotted as "FA" and the rest as "No FA". If
    `fa_types` is None, all trials are shown together.

    Parameters
    ----------
    subjid : int
        Subject ID.
    dates : iterable | tuple | None
        Specific dates, list, or inclusive range (start, end). None → all.
    fa_types : iterable | str | None
        FA labels to treat as FA. None disables FA split.
    figsize : tuple
        Figure size when creating a new axes.
    ax : matplotlib Axes or None
        Reuse an existing axes; otherwise create a new figure/axes.
    save : bool, optional
        If True, save generated figures (default False).
    verbose : bool, optional
        If True, print save status messages (default True).

    Returns
    -------
    (fig, ax, df) or (fig, ax, df, fig_start_end, ax_start_end)
        Default: first plot (HR poke_end → last poke_end) and dataframe.
        If make_second_plot=True, also draws HR poke_start → last poke_end.
        If return_both=True and second plot is created, both figures/axes are returned.

    Notes
    -----
    If `save=True`, each generated figure is written via save_figure().
    """

    derivatives_dir = get_derivatives_root()

    if fa_types is None:
        fa_set = None
    elif isinstance(fa_types, str):
        fa_set = {fa_types.lower()}
    else:
        try:
            fa_set = {str(x).lower() for x in fa_types}
        except Exception:
            fa_set = None

    rows = []

    for sid, subj_dir in _iter_subject_dirs(derivatives_dir, [subjid]):
        ses_dirs = _filter_session_dirs(subj_dir, dates)

        for ses_dir in ses_dirs:
            date_str = ses_dir.name.split("_date-")[-1]
            try:
                date_val = int(date_str)
            except Exception:
                date_val = date_str

            results_dir = ses_dir / "saved_analysis_results"
            if not results_dir.exists():
                continue

            td = _load_table_with_trial_data(results_dir, "trial_data")
            if td.empty:
                continue

            td = td.copy()
            td["is_aborted"] = td.get("is_aborted", False).fillna(False)
            td["hit_hidden_rule"] = td.get("hit_hidden_rule", False).fillna(False)

            # One session at a time: the metric keys on `global_trial_id`, which
            # repeats across sessions.
            gaps = hr_abort_poke_gap(td, build_position_data(td))
            if gaps.empty:
                continue
            trial_cols = [c for c in ("global_trial_id", "sequence_start", "fa_label")
                          if c in td.columns]
            gaps = gaps.merge(td[trial_cols], on="global_trial_id", how="left")

            for _, row in gaps.iterrows():
                if fa_set is None:
                    category = "All"
                else:
                    fa_label = row.get("fa_label")
                    is_fa = str(fa_label).lower() in fa_set if fa_label is not None else False
                    category = "FA" if is_fa else "No FA"

                rows.append({
                    "subjid": sid,
                    "date": date_val,
                    "sequence_start": row.get("sequence_start"),
                    "hidden_rule_position": row["hidden_rule_position"],
                    "delta_seconds": row["delta_seconds"],
                    "delta_start_end_seconds": row["delta_start_end_seconds"],
                    "category": category,
                })

    if not rows:
        print("No aborted hidden-rule trials found with valid poke timings.")
        return None, None, pd.DataFrame()

    df = pd.DataFrame(rows)
    df = df.sort_values(["date", "sequence_start"], na_position="last").reset_index(drop=True)
    df["order"] = range(len(df))

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    categories = ["No FA", "FA"] if fa_set is not None else ["All"]
    x_positions = {cat: idx for idx, cat in enumerate(categories)}
    point_color = "#5dade2"  # light blue

    for cat in categories:
        sub = df[df["category"] == cat]
        if sub.empty:
            continue
        x_pos = x_positions[cat]
        jitter = np.random.normal(loc=x_pos, scale=0.05, size=len(sub)) if len(sub) > 1 else [x_pos]
        ax.scatter(jitter, sub["delta_seconds"], color=point_color, alpha=0.8, s=50)

        mean_val, sem_val = mean_sem(sub["delta_seconds"])
        sem_val = 0.0 if np.isnan(sem_val) else sem_val
        ax.errorbar(x_pos, mean_val, yerr=sem_val, color="black", fmt='-', lw=2.5, capsize=6, alpha=0.9)

    ax.set_xticks(list(x_positions.values()))
    ax.set_xticklabels(list(x_positions.keys()))
    ax.set_xlabel("Category")
    ax.set_ylabel("Time Difference (s)")
    ax.set_title(f"Hidden-rule aborted trials: last poke latency (subj {subjid})")
    ax.margins(x=0.15)

    fig_start_end = None
    ax_start_end_obj = None

    if make_second_plot:
        df_start = df[df["delta_start_end_seconds"].notna()].copy()
        if not df_start.empty:
            if ax_start_end is None:
                fig_start_end, ax_start_end_obj = plt.subplots(figsize=figsize)
            else:
                ax_start_end_obj = ax_start_end
                fig_start_end = ax_start_end_obj.figure

            for cat in categories:
                sub = df_start[df_start["category"] == cat]
                if sub.empty:
                    continue
                x_pos = x_positions[cat]
                jitter = np.random.normal(loc=x_pos, scale=0.05, size=len(sub)) if len(sub) > 1 else [x_pos]
                ax_start_end_obj.scatter(jitter, sub["delta_start_end_seconds"], color=point_color, alpha=0.8, s=50)

                mean_val, sem_val = mean_sem(sub["delta_start_end_seconds"])
                sem_val = 0.0 if np.isnan(sem_val) else sem_val
                ax_start_end_obj.errorbar(x_pos, mean_val, yerr=sem_val, color="black", fmt='-', lw=2.5, capsize=6, alpha=0.9)

            ax_start_end_obj.set_xticks(list(x_positions.values()))
            ax_start_end_obj.set_xticklabels(list(x_positions.keys()))
            ax_start_end_obj.set_xlabel("Category")
            ax_start_end_obj.set_ylabel("Time Difference (s)")
            ax_start_end_obj.set_title(f"Hidden-rule aborted trials: start→last latency (subj {subjid})")
            ax_start_end_obj.margins(x=0.15)

    if save:
        figures_to_save = [
            (fig, "hidden_rule_abort_poke_gap_end_latency"),
            (fig_start_end, "hidden_rule_abort_poke_gap_start_latency"),
        ]
        for candidate_fig, suffix in figures_to_save:
            if candidate_fig is None:
                continue
            try:
                out_path = save_figure(
                    candidate_fig,
                    suffix,
                    subjids=[subjid],
                    dates=dates,
                )
                if verbose:
                    print(
                        f"[plot_hidden_rule_abort_poke_gap] Saved figure '{suffix}' to {out_path}"
                    )
            except Exception as exc:
                if verbose:
                    print(
                        f"[plot_hidden_rule_abort_poke_gap] Failed to save '{suffix}': {exc}"
                    )

    if return_both and fig_start_end is not None:
        return fig, ax, df, fig_start_end, ax_start_end_obj
    return fig, ax, df


def plot_hr_reward_fraction_over_trials(
    subjid: int,
    dates=None,
    window_size: int = 20,
    figsize=(10, 5),
    ax=None,
    *,
    save: bool = False,
    verbose: bool = True,
):
    """
    Plot moving-window % of rewarded trials that are hidden-rule rewarded.

    - Filters to rewarded trials only (is_aborted=False, response_time_category="rewarded").
    - Uses hidden_rule_success when present, else falls back to hit_hidden_rule.
    - Rolls over consecutive rewarded trials across selected sessions.

    Parameters
    ----------
    subjid : int
        Subject ID.
    dates : iterable | tuple | None
        Specific dates, list, or inclusive range; None → all.
    window_size : int
        Rolling window size for percentage (default 20).
    figsize : tuple
        Figure size when creating a new axes.
    ax : matplotlib Axes or None
        Reuse an existing axes; otherwise create a new figure/axes.
    save : bool, optional
        If True, save the generated figure (default False).
    verbose : bool, optional
        If True, print save status messages (default True).

    Returns
    -------
    (fig, ax, df)
        Matplotlib figure/axes and the dataframe with rolling percentage.
    """

    window_size = max(int(window_size), 1)
    derivatives_dir = get_derivatives_root()

    frames = []
    for sid, subj_dir in _iter_subject_dirs(derivatives_dir, [subjid]):
        ses_dirs = _filter_session_dirs(subj_dir, dates)
        for ses_dir in ses_dirs:
            date_str = ses_dir.name.split("_date-")[-1]
            try:
                date_val = int(date_str)
            except Exception:
                date_val = date_str

            results_dir = ses_dir / "saved_analysis_results"
            if not results_dir.exists():
                continue

            td = _load_table_with_trial_data(results_dir, "trial_data")
            if td.empty:
                continue

            td = td.copy()
            td["is_aborted"] = td.get("is_aborted", False).fillna(False)
            td["response_time_category"] = td.get("response_time_category", "").astype(str)
            td["sequence_start"] = pd.to_datetime(td.get("sequence_start"), errors="coerce")
            td["date"] = date_val
            frames.append(td)

    # Pool the sessions *before* rolling. Rolling per session and concatenating
    # restarts the window at every session boundary -- a different quantity, and
    # one that raises no error.
    pooled = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    hr_flags, pct = rolling_hr_reward_fraction(pooled, window_size, with_flags=True)

    if pct.empty:
        print("No rewarded trials found for requested selection.")
        return None, None, pd.DataFrame()

    df = pooled.loc[pct.index, ["date", "sequence_start"]].reset_index(drop=True)
    df["hr_rewarded"] = hr_flags.to_numpy()
    df["trial_idx"] = np.arange(1, len(df) + 1)
    df["hr_rewarded_flag"] = df["hr_rewarded"].astype(int)
    df["hr_rewarded_pct"] = pct.to_numpy()

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    ax.plot(df["trial_idx"], df["hr_rewarded_pct"], color="black", linewidth=1.6)

    ax.set_xlabel("Consecutive Rewarded Trial #")
    ax.set_ylabel("HR Rewarded (%)")
    ax.set_ylim(0, 100)
    ax.set_title(f"Hidden-rule share of rewarded trials (window={window_size}, subj {subjid})")
    ax.grid(False)

    if save:
        try:
            out_path = save_figure(
                fig,
                "hr_reward_fraction_over_trials",
                subjids=[subjid],
                dates=dates,
            )
            if verbose:
                print(
                    "[plot_hr_reward_fraction_over_trials] Saved figure to "
                    f"{out_path}"
                )
        except Exception as exc:
            if verbose:
                print(
                    "[plot_hr_reward_fraction_over_trials] Failed to save figure: "
                    f"{exc}"
                )

    return fig, ax, df
