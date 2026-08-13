# Defers evaluation of PEP-604 annotations, matching `primitives.py`.
from __future__ import annotations

"""Trajectory prep shared between the two movement plotting modules.

restructure_2 Phase 5, the survivors of the Phase 4 audit's finding 10. That
finding listed seven helpers duplicated 2-4x across `movement_analysis_utils`
and `movement_analysis/sing_rew_movement` and called for de-duplicating both
files in one pass, on the premise that "every row has a twin".

**Measured, that premise does not hold: most of them are different rules
wearing the same name, and merging them would change what is plotted.** Only
the two below are genuinely the same computation. What the others do, and the
numbers, are in `docs/DECISIONS.md` section 13 -- read that before trying the
merge again.

This is prep, not display arithmetic: it reshapes the trace *before* anything
is drawn. `primitives.py` is the other half, and takes the mean/SEM/rolling of
values that are already on the figure.
"""

import numpy as np
import pandas as pd

from pathlib import Path
from typing import Optional
import ast
import json

import matplotlib.colors as mcolors

from hypnose_behavior.io.layout import derivatives, normalize_subjid
from hypnose_behavior.io.load_results import load_session_results
from hypnose_behavior.io.loaders import _load_trial_views
from hypnose_behavior.io.paths import (
    get_derivatives_root, get_rawdata_root, get_server_root,
)
from hypnose_behavior.metric_analysis.metrics.timing import inter_trial_interval
from hypnose_behavior.utils.helpers import (
    _filter_session_dirs, _get_from_cache, _iter_subject_dirs, _update_cache,
    find_tracking_file, read_tracking_table, session_selectors,
)

__all__ = [
    "resample_trace", "smooth_xy", "_normalize_date", "_collect_sessions",
    "_load_trial_data", "_load_sorted_session", "_parse_json_value",
    "_count_to_marker_size", "_nice_round",
    "_summary_save_suffix", "_darken", "_resolve_color", "_ordered_groups",
    "_coerce_tz_naive", "_load_protocol_from_summary",
    "load_tracking_with_behavior", "_load_subject_trial_timeline",
]


def resample_trace(x, y, n_points: int = 200):
    """Resample a trajectory onto a normalised arc-length grid ``[0, 1]``.

    Returns ``(x_new, y_new)``, or None for a trace that cannot be resampled:
    fewer than two points, any non-finite coordinate, or zero total path length
    (an animal that never moved -- the arc-length parameterisation is undefined
    there, not merely degenerate).

    Resampling by arc length rather than by time is what makes traces of
    different durations averageable: every trace contributes `n_points` samples
    spaced evenly along its own path, so the mean trace follows the shared
    *route* instead of being dominated by whichever trial was slowest.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size < 2 or y.size < 2:
        return None
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        return None
    seg_len = np.hypot(np.diff(x), np.diff(y))
    cumlen = np.concatenate(([0.0], np.cumsum(seg_len)))
    total_len = cumlen[-1]
    if total_len <= 0:
        return None
    s = cumlen / total_len
    s_new = np.linspace(0.0, 1.0, num=n_points)
    return np.interp(s_new, s, x), np.interp(s_new, s, y)


def smooth_xy(tracking, window):
    """Centred rolling-mean smooth of the ``X``/``Y`` columns of `tracking`.

    Returns `tracking` unchanged for a window of None or <= 1, and never
    mutates the input. `min_periods=1` keeps the ends of the trace rather than
    trimming half a window off each, which matters because the endpoints are
    exactly what the trace plots are read for.

    A duplicated ``X``/``Y`` column name yields a DataFrame rather than a
    Series from the lookup; the first column wins. One of the three call sites
    this replaces handled that and the other two would have raised on it.
    """
    if window is None or window <= 1:
        return tracking
    df = tracking.copy()
    for col in ("X", "Y"):
        values = df[col]
        if isinstance(values, pd.DataFrame):
            values = values.iloc[:, 0]
        df[col] = pd.Series(values).rolling(
            window=window, center=True, min_periods=1
        ).mean()
    return df


def _normalize_date(value):
    if value is None:
        return None
    digits = "".join(ch for ch in str(value) if ch.isdigit())
    return int(digits) if digits.isdigit() else str(value)


def _collect_sessions(subjids, dates, *, ses=None, index=None,
                      date_range=None, ses_range=None, index_range=None):
    select = session_selectors(
        ses=ses, index=index, date_range=date_range,
        ses_range=ses_range, index_range=index_range,
    )
    derivatives_dir = get_derivatives_root()
    for subjid, subj_dir in _iter_subject_dirs(derivatives_dir, subjids):
        if subjid is None:
            continue
        date_vals = []
        results_dirs = []
        for ses_dir in _filter_session_dirs(subj_dir, dates, **select):
            date_part = ses_dir.name.split("_date-")[-1]
            date_val = _normalize_date(date_part)
            if date_val is None:
                continue
            date_vals.append(date_val)
            results_dirs.append(ses_dir / "saved_analysis_results")
        if results_dirs:
            yield subjid, date_vals, results_dirs


def _load_trial_data(results_dir: Path) -> pd.DataFrame:
    parquet_path = results_dir / "trial_data.parquet"
    if not parquet_path.exists():
        raise FileNotFoundError(f"Missing trial_data.parquet at {parquet_path}")
    return pd.read_parquet(parquet_path)


def _load_sorted_session(results_dir):
    df = _load_trial_data(results_dir)
    if df.empty:
        return df
    df = df.copy()
    if "sequence_start" in df.columns:
        df["_order_key"] = pd.to_datetime(df["sequence_start"], errors="coerce")
        df = df.sort_values("_order_key", na_position="last")
    elif "timestamp" in df.columns:
        df["_order_key"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.sort_values("_order_key", na_position="last")
    df = df.reset_index(drop=True)
    df["_trial_idx"] = np.arange(1, len(df) + 1)
    return df


def _parse_json_value(val):
    if isinstance(val, (dict, list)):
        return val
    if not isinstance(val, str):
        return None
    try:
        return json.loads(val)
    except Exception:
        try:
            return ast.literal_eval(val)
        except Exception:
            return None




def _count_to_marker_size(count, *, base_area=36.0, ref_count=10.0, min_area=10.0, max_area=300.0):
    """Scale scatter marker area linearly with event count.

    A count of ``ref_count`` produces area ``base_area`` (matches the default
    ``markersize=6`` used elsewhere). Sizing is absolute (no per-plot
    normalization) so the same count always produces the same dot across
    figures, and small differences in count produce small differences in size.
    """
    if count is None or count <= 0:
        return min_area
    size = base_area * (float(count) / float(ref_count))
    return float(np.clip(size, min_area, max_area))


def _nice_round(n):
    """Round a count to a readable integer for legend labels."""
    n = float(n)
    if n <= 0:
        return 0
    if n < 5:
        return int(round(n))
    if n < 20:
        return int(round(n / 5.0) * 5)
    if n < 100:
        return int(round(n / 10.0) * 10)
    if n < 500:
        return int(round(n / 50.0) * 50)
    if n < 2000:
        return int(round(n / 100.0) * 100)
    return int(round(n / 500.0) * 500)


def _summary_save_suffix(moving_avg, window_size, step_size):
    if moving_avg:
        return f"rolling_w{int(window_size)}_s{int(step_size)}"
    return "daily"


def _darken(color, factor=0.62):
    """Return a darker shade of a colour (for violin edges)."""
    r, g, b = mcolors.to_rgb(color)
    return (r * factor, g * factor, b * factor)


def _resolve_color(label, color_map, default="#444444"):
    return color_map.get(label, default)


def _ordered_groups(group_keys, preferred):
    """Labels in `preferred`'s canonical order first, then every other label in
    the order `group_keys` yields them.

    **`group_keys` must be ordered.** Until restructure_2 Phase 5 the three
    multi-session callers accumulated into a bare `set()`, so every label outside
    `preferred` was drawn in string-hash order -- which varies between processes,
    making those figures irreproducible run to run rather than merely oddly
    ordered. It went unnoticed because `preferred` lists only the four 3-odor
    sequences: on a 5-odor protocol *every* drawn series took the hash path.

    An unordered input is sorted rather than iterated, since a set has no order
    to preserve and sorting is the only deterministic thing left to do with one.
    That is a guard against the defect returning, not the normal path.
    """
    if isinstance(group_keys, (set, frozenset)):
        group_keys = sorted(group_keys)
    result = []
    for name in preferred:
        if name in group_keys:
            result.append(name)
    for name in group_keys:
        if name not in result:
            result.append(name)
    return result


def _coerce_tz_naive(series):
    """Return a datetime Series with any timezone dropped (subtraction-safe)."""
    s = pd.to_datetime(series, errors="coerce")
    try:
        if s.dt.tz is not None:
            s = s.dt.tz_localize(None)
    except (AttributeError, TypeError):
        pass
    return s


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


def _load_subject_trial_timeline(subjid, subj_dates, *, ses=None, index=None,
                                 date_range=None, ses_range=None, index_range=None):
    """Build a per-trial timeline for one subject across sessions.

    Returns a dict with the concatenated all-trial DataFrame (chronological,
    with ``time_seconds`` collapsing inter-session gaps as in
    ``plot_cumulative_rewards`` and a continuous ``trial_index`` as in
    ``plot_cumulative_rewards_by_trial``), plus ``iti_seconds`` (within-session
    inter-trial interval) and ``is_rewarded``. Also returns the time- and
    trial-axis gap spans and session boundaries. ``None`` if no data.

    ``ses`` / ``index`` / ``date_range`` / ``ses_range`` / ``index_range`` narrow the
    selection further; they intersect with ``dates`` and with each other, and ``index``
    is the subject's gap-free chronological rank (`utils.helpers.session_selectors`).
    """
    select = session_selectors(
        ses=ses, index=index, date_range=date_range,
        ses_range=ses_range, index_range=index_range,
    )
    derivatives_dir = get_derivatives_root()
    subj_str = normalize_subjid(subjid)
    subj_dir = derivatives.subject_dir(subjid, missing_ok=True)
    if subj_dir is None:
        print(f"Warning: No subject directory found for {subj_str}")
        return None

    sessions = []
    for ses_dir in _filter_session_dirs(subj_dir, subj_dates, **select):
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
