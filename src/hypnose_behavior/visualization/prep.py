# Defers evaluation of PEP-604 annotations, matching `primitives.py`.
from __future__ import annotations

"""The non-drawing leaf of `visualization/`: what more than one plotter module needs.

Shared non-drawing code: session collection, JSON/label parsing, colour and marker
sizing, trajectory prep, the figure-level loaders and the registry-backed metric access.

- **Every plotter module depends on this; this depends on no plotter module.** A thing
  two plotting modules share is promoted here, never reached by importing a sibling.
  Zero plotter-to-plotter imports is the invariant to preserve. See DECISIONS.md
  sections 3 and 13.
- It has 12 importers of which only 2 are the movement modules, so it does **not**
  belong under `movement/` -- that would invert the edge for the other ten.
- Prep, not display arithmetic: it reshapes data *before* anything is drawn.
  `primitives.py` is the other half, taking the mean/SEM/rolling of values already on
  the figure.
- The helpers here are the ones genuinely shared. Several similarly-named helpers in
  the plotters are **different rules wearing the same name**; do not merge them. See
  DECISIONS.md section 13.
"""

import numpy as np
import pandas as pd

from pathlib import Path
from typing import Iterable, Optional, Tuple
import ast
import json

import matplotlib.colors as mcolors

from hypnose_behavior.io import layout
from hypnose_behavior.io.layout import (
    _filter_sessions,
    _iter_subject_dirs,
    derivatives,
    normalize_subjid,
    session_selectors,
)
from hypnose_behavior.io.load_results import load_results_dir, load_session_results
from hypnose_behavior.io.loaders import _load_trial_views
from hypnose_behavior.metric_analysis.metrics.hidden_rule import hr_odor_associations
from hypnose_behavior.metric_analysis.run import REGISTRY, metric_value
from hypnose_behavior.io.paths import (
    get_derivatives_root, get_rawdata_root, get_server_root,
)
from hypnose_behavior.metric_analysis.metrics.timing import inter_trial_interval
from hypnose_behavior.utils.helpers import (
    _get_from_cache,
    _update_cache,
    find_tracking_file,
    read_tracking_table,
)

__all__ = [
    "resample_trace", "smooth_xy", "_normalize_date", "_collect_sessions",
    "_load_trial_data", "_load_sorted_session", "_parse_json_value",
    "_count_to_marker_size", "_nice_round",
    "_summary_save_suffix", "_darken", "_resolve_color", "_ordered_groups",
    "_coerce_tz_naive", "_load_protocol_from_summary",
    "load_tracking_frame", "load_tracking_with_behavior",
    "_load_subject_trial_timeline",
    # Used by two or more construct modules, hence leaves here rather than peer
    # imports. DECISIONS.md section 5 names `_computed_metrics` by its old path
    # `visualization._computed_metrics`; it is this function.
    "_extract_metric_value", "_metric_name_for_key", "_computed_metrics",
    "_computed_metric", "_series_line_widths", "_build_odor_colors",
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
        for ref in _filter_sessions(subj_dir, dates, **select):
            date_part = ref.date
            date_val = _normalize_date(date_part)
            if date_val is None:
                continue
            date_vals.append(date_val)
            results_dirs.append(layout.results_dir(ref))
        if results_dirs:
            yield subjid, date_vals, results_dirs


def _load_trial_data(results_dir: Path) -> pd.DataFrame:
    parquet_path = layout.table_path(results_dir, "trial_data.parquet")
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

    **`group_keys` must come from an ordered container.** Accumulate labels first-seen,
    never into a bare `set()`: string-hash order varies between processes, so the figure
    stops being reproducible run to run. A `preferred` list covering none of the live
    data is not a partial ordering, it is no ordering. See DECISIONS.md section 11.

    A set input is sorted rather than iterated -- a guard against the defect returning,
    not the normal path.
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
        with open(layout.table_path(results_dir, "summary.json"), "r", encoding="utf-8") as f:
            summary = json.load(f)
        runs = summary.get("session", {}).get("runs", [])
        if runs and isinstance(runs, list):
            stage = runs[0].get("stage", {}) if isinstance(runs[0], dict) else {}
            name = stage.get("stage_name") or stage.get("name")
            return str(name) if name else "Unknown"
    except Exception:
        pass
    return "Unknown"


def load_tracking_frame(results_dir):
    """The session's combined SLEAP tracking table, with `X`/`Y` normalised.

    **The one place a tracking file is located and read.** It had two: this loader and
    `movement/traces.plot_movement_trace`, which resolved its own path, and the two had
    drifted -- only this one knew about SLEAP, so the plotter could find nothing on a
    session this function reads happily. Section 13's rule, that a thing needed by two
    modules becomes a leaf rather than a second copy.

    **SLEAP is the only tracking source**, and the combined file is written by the SLEAP
    pipeline in another repo, not here. A second, older source was removed in
    2026-08-19; see `DECISIONS.md` section 35 for the measurement that made dropping it a
    no-op on every gate session.

    SLEAP writes `centroid_x` / `centroid_y`; everything downstream reads `X` / `Y`, so
    they are aliased rather than renamed -- a file that already carries `X`/`Y` is left
    exactly as it is.
    """
    tracking_file = find_tracking_file(results_dir, "*_combined_sleap_tracking_timestamps")
    if tracking_file is None:
        raise FileNotFoundError(
            f"No combined SLEAP tracking file in {results_dir}. Produce the session's "
            f"tracking first -- it is written by the SLEAP pipeline, not by this repo.")
    tracking = read_tracking_table(tracking_file)
    if 'X' not in tracking.columns and 'centroid_x' in tracking.columns:
        tracking['X'] = tracking['centroid_x']
    if 'Y' not in tracking.columns and 'centroid_y' in tracking.columns:
        tracking['Y'] = tracking['centroid_y']
    return tracking


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
        results_dir = layout.results_dir(derivatives.find_session(subjid, date=date))
        tracking = load_tracking_frame(results_dir)
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
    is the subject's gap-free chronological rank (`io.layout.session_selectors`).
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
    for ref in _filter_sessions(subj_dir, subj_dates, **select):
        date_str = ref.date
        results_dir = layout.results_dir(ref)
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
        summary_path = layout.table_path(results_dir, "summary.json")
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


# Load metric results for visualization =====================================================================================================================

def _extract_metric_value(metrics: dict, var_path: str):
    """
    Extract a numeric value from metrics dict given a dot-path.

    A navigator over a metrics mapping, not a reader of one: the mapping comes from
    `_computed_metrics`, never from `metrics_*.json`. The dot-path is how a plot names
    a sub-entry of a metric ("avg_response_time.Rewarded").

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

    Evaluates through `run.metric_value` -- the same expression `run.py` builds the file
    with -- so what a plotter computes and what would have been saved cannot drift apart.
    **Do not write that expression out again here.** See DECISIONS.md sections 5 and 34.

    **The lenient miss stays here and is not shared.** A key naming no registered metric
    is skipped rather than raised on: plotters ask for a fixed key list and draw whatever
    comes back. `Session.metrics` raises instead, because a caller naming a metric by
    hand has made a typo rather than a coverage choice.
    """
    results = load_results_dir(results_dir)
    metrics = {}
    for key in keys:
        name = _metric_name_for_key(key)
        if name is None:
            continue
        metrics[key] = metric_value(REGISTRY[name], results)
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
