# Defers evaluation of PEP-604 annotations (`X | None`), keeping this module
# importable on Python 3.9 for repos pinned there (hypnose-eeg-preprocessing).
from __future__ import annotations

"""Movement metrics derived from SLEAP tracking.

``compute_speed_analysis`` writes ``speed_analysis.parquet`` and supplies seven
metrics off SLEAP tracking:

    binned speed epoch . baseline mu/sigma and vthresh . movement-onset latency
    . movement_onset_from_valve_s . path_length_px . travel_time_s . tortuosity

- **This is the only place the speed threshold is computed.** A plotter reads
  ``speed_analysis.parquet`` or reports that the file is missing -- it must never
  recompute a threshold no value on its axes was derived from. See DECISIONS.md
  section 35.
- No gate can reach ``compute_speed_analysis``: it writes into ``results_dir``, and
  the derivatives root is read-only. Changes here need a probe against a local copy.
"""

import re
from collections import defaultdict
from typing import Union

import numpy as np
import pandas as pd

from hypnose_behavior.frames import position_entries_by_trial
from hypnose_behavior.io import layout
from hypnose_behavior.io.layout import (
    _iter_subject_dirs,
    derivatives,
    normalize_subjid,
    session_selectors,
)
from hypnose_behavior.io.loaders import _load_position_data, iter_sessions
from hypnose_behavior.io.paths import get_derivatives_root
from hypnose_behavior.io.tracking import _load_tracking_and_behavior
from hypnose_behavior.utils.helpers import _update_cache

__all__ = ["binned_speed", "compute_speed_analysis", "run_speed_analysis_batch",
           "speed_threshold", "THRESHOLD_COLUMNS"]


# --------------------------------------------------------------------------------------
# The speed-analysis defaults, in one place
# --------------------------------------------------------------------------------------

BIN_MS = 100
PRE_BUFFER_S = 1.0
MODE = "mean"
THRESHOLD_ALPHA = 10.0
THRESHOLD_BETA = 10.0
BASELINE_WINDOW_S = (-0.15, -0.05)   # seconds relative to last poke-out

# Written as constant columns on every row of `speed_analysis.parquet`, so a reader can
# tell what threshold produced that file's `speed_threshold_time` / `latency_s` instead
# of assuming today's defaults.
THRESHOLD_COLUMNS = ("baseline_mu", "baseline_sigma", "threshold_alpha",
                     "threshold_beta", "speed_threshold")


def _binned_speed(tracking_df, t_zero, t_end, pre_buffer_s, bin_s, mode):
    """Compute binned speed (mean or max) between start and end relative to t_zero.

    Returns mids (bin centers) and arr (speed per bin) or (None, None) on failure.
    """
    if pd.isna(t_zero) or pd.isna(t_end) or t_end <= t_zero:
        return None, None
    start_dt = t_zero - pd.Timedelta(seconds=pre_buffer_s)
    seg = tracking_df[(tracking_df["time"] >= start_dt) & (tracking_df["time"] <= t_end)].copy()
    if len(seg) < 2 or {"X", "Y", "time"} - set(seg.columns):
        return None, None
    t_rel = (seg["time"] - t_zero).dt.total_seconds().to_numpy()
    if not np.isfinite(t_rel).all() or np.ptp(t_rel) == 0:
        return None, None
    x = seg["X"].to_numpy()
    y = seg["Y"].to_numpy()
    vx = np.gradient(x, t_rel)
    vy = np.gradient(y, t_rel)
    speed = np.hypot(vx, vy)

    dur = t_rel.max() - t_rel.min()
    edges = np.arange(-pre_buffer_s, dur + bin_s + bin_s * 0.5, bin_s)
    if len(edges) < 2:
        return None, None

    seg_df = pd.DataFrame({"t_rel": t_rel, "speed": speed})
    seg_df["bin"] = pd.cut(seg_df["t_rel"], bins=edges, right=False, include_lowest=True)
    grouped = seg_df.groupby("bin", observed=False)["speed"]
    agg_series = grouped.max() if mode == "max" else grouped.mean()
    mids = edges[:-1] + (edges[1] - edges[0]) / 2
    arr = np.full_like(mids, np.nan, dtype=float)
    bin_to_idx = {b: i for i, b in enumerate(agg_series.index.categories)}
    for b, v in agg_series.items():
        idx = bin_to_idx.get(b)
        if idx is not None:
            arr[idx] = v
    return mids, arr


def run_speed_analysis_batch(
    subjids=None,
    dates=None,
    *,
    ses=None,
    index=None,
    date_range=None,
    ses_range=None,
    index_range=None,
    bin_ms: int = BIN_MS,
    pre_buffer_s: float = PRE_BUFFER_S,
    fa_label_filter=None,
    mode: str = MODE,
    threshold: bool = True,
    threshold_alpha: float = THRESHOLD_ALPHA,
    threshold_beta: float = THRESHOLD_BETA,
    verbose: bool = True,
):
    """Run compute_speed_analysis over all available subject/date combinations.

    Supports single or multiple subject IDs and date specs (list of dates or
    inclusive (start, end) tuple). Only sessions with existing data are passed
    to compute_speed_analysis. Returns a list of (subjid, date) processed and
    prints a summary when verbose=True.
    """

    select = session_selectors(
        ses=ses, index=index, date_range=date_range,
        ses_range=ses_range, index_range=index_range,
    )
    derivatives_dir = get_derivatives_root()
    processed: list[tuple[int, Union[int, str]]] = []

    for sid, subj_dir in _iter_subject_dirs(derivatives_dir, subjids):
        ses_recs = iter_sessions(subj_dir, dates, **select)
        if not ses_recs:
            if verbose:
                print(f"[run_speed_analysis_batch] No sessions found for sub-{sid:03d} with given dates.")
            continue

        # Extract ordered unique dates from session directories
        date_list = []
        seen_dates = set()
        for rec in ses_recs:
            date_str = rec.date_str
            try:
                date_val = int(date_str) if str(date_str).isdigit() else date_str
            except Exception:
                date_val = date_str
            if date_val in seen_dates:
                continue
            seen_dates.add(date_val)
            date_list.append(date_val)

        if not date_list:
            if verbose:
                print(f"[run_speed_analysis_batch] No matching dates after filtering for sub-{sid:03d}.")
            continue

        try:
            compute_speed_analysis(
                sid,
                dates=date_list,
                bin_ms=bin_ms,
                pre_buffer_s=pre_buffer_s,
                fa_label_filter=fa_label_filter,
                mode=mode,
                threshold=threshold,
                threshold_alpha=threshold_alpha,
                threshold_beta=threshold_beta,
            )
            processed.extend([(sid, d) for d in date_list])
        except Exception as e:
            if verbose:
                print(f"[run_speed_analysis_batch] Failed for sub-{sid:03d}: {e}")

    if verbose:
        if processed:
            print("[run_speed_analysis_batch] Completed speed analysis for:")
            by_subj: dict[int, list[Union[int, str]]] = defaultdict(list)
            for sid, d in processed:
                by_subj[sid].append(d)
            for sid in sorted(by_subj.keys()):
                dates_sorted = sorted(by_subj[sid], key=lambda x: str(x))
                dates_str = ", ".join(str(d) for d in dates_sorted)
                print(f"  sub-{sid:03d}: {dates_str}")
        else:
            print("[run_speed_analysis_batch] No sessions processed.")

    return processed


def speed_threshold(baseline_values, *, alpha: float = THRESHOLD_ALPHA,
                    beta: float = THRESHOLD_BETA, enabled: bool = True) -> dict:
    """Baseline speed statistics and the movement-onset threshold.

    ``vthresh = max(alpha * mu, mu + beta * sigma)`` over the speeds in the
    baseline window, where ``mu``/``sigma`` are the population mean and SD.

    **The one derivation of this threshold.** A plotter that re-derives it can draw a
    line no value on its axes came from, and nothing in any output would say so -- so
    the four values are written into `speed_analysis.parquet` and read back.

    ``enabled=False`` reports mu/sigma but no combined threshold, matching the
    plotters' ``threshold=False`` switch. Every value is ``None`` when the
    baseline window is empty.
    """
    values = np.asarray(baseline_values, dtype=float)
    mu = float(np.nanmean(values)) if values.size else None
    sigma = float(np.nanstd(values)) if values.size else None
    alpha_mu = mu * alpha if mu is not None else None
    mu_plus_beta_sigma = mu + beta * sigma if mu is not None and sigma is not None else None
    thr_max = None
    if enabled and mu is not None:
        candidates = [v for v in (alpha_mu, mu_plus_beta_sigma) if v is not None]
        if candidates:
            thr_max = max(candidates)
    return {"mu": mu, "sigma": sigma, "alpha": alpha, "beta": beta,
            "alpha_mu": alpha_mu, "mu_plus_beta_sigma": mu_plus_beta_sigma,
            "max_alpha_mu_mu_plus_beta_sigma": thr_max}


def compute_speed_analysis(
    subjid,
    dates=None,
    *,
    bin_ms: int = BIN_MS,
    pre_buffer_s: float = PRE_BUFFER_S,
    fa_label_filter=None,
    mode: str = MODE,
    threshold: bool = True,
    threshold_alpha: float = THRESHOLD_ALPHA,
    threshold_beta: float = THRESHOLD_BETA,
):
    """Compute cue-port speed epochs aligned to last poke-out for rewarded, unrewarded, and FA trials.

        Handles loading data, computing speeds, binning, thresholding, and movement metrics. Writes a single
        speed_analysis.parquet per session containing per-bin records with per-trial metrics repeated.
        Returns the same plotting artifacts as before for backward compatibility.

    Parameters
    ----------
    subjid : int
        Subject ID.
    dates : list | tuple | None
        List of dates or inclusive range tuple; None uses all available.
    bin_ms : int
        Epoch width in milliseconds (default 100).
    pre_buffer_s : float
        Seconds to include before last poke-out (default 0).
    fa_label_filter : str | Iterable | None
        FA labels to include (default {"fa_time_in"}); accepts comma/semicolon-separated
        string or any iterable (e.g., ["fa_time_in", "fa_time_out"]). Case-insensitive.
    mode : {"max", "mean"}
        Aggregation per epoch: max speed (current behavior) or mean speed.
    threshold : bool
        If True, compute baseline (mu, sigma) from [-0.15s, -0.05s] pooled across all trials
        in the session, and overlay baseline plus the single threshold line
        vthresh = max(alpha*mu, mu+beta*sigma).
    threshold_alpha : float
        Multiplier for mu when threshold is enabled (default THRESHOLD_ALPHA).
    threshold_beta : float
        Multiplier for sigma when threshold is enabled (default THRESHOLD_BETA).
    figsize : tuple
        Figure size for per-session plots.

    Returns
    -------
    dict with:
                - "per_session": list of dicts per session with keys date, figs (violin, traces)
                    where traces is a dict of condition -> figure, and baseline stats when threshold=True
        - "combined": dict of condition -> fig (only when multiple sessions and data present)
    """

    if mode not in {"max", "mean"}:
        raise ValueError("mode must be 'max' or 'mean'")

    # Normalize FA labels: accept comma-separated string or any iterable of labels
    if fa_label_filter is None:
        fa_labels = {"fa_time_in"}
    elif isinstance(fa_label_filter, str):
        # allow "fa_time_in,fa_time_out" or "fa_time_in; fa_time_out"
        parts = re.split(r"[;,]", fa_label_filter)
        fa_labels = {p.strip().lower() for p in parts if p.strip()}
    else:
        try:
            fa_labels = {str(s).strip().lower() for s in fa_label_filter if str(s).strip()}
        except TypeError:
            fa_labels = {str(fa_label_filter).strip().lower()}

    subj_str = normalize_subjid(subjid)
    derivatives_dir = get_derivatives_root()
    subj_dir = derivatives.subject_dir(subjid)

    ses_recs = iter_sessions(subj_dir, dates)
    if not ses_recs:
        raise FileNotFoundError(f"No sessions found for subject {subjid} with given dates")

    bin_s = bin_ms / 1000.0
    baseline_window = BASELINE_WINDOW_S
    start_target_s = -bin_ms / 2000.0  # target mid-bin time (e.g., -0.05s for 100 ms bins)

    def _safe_dt(val):
        try:
            return pd.to_datetime(val)
        except Exception:
            return pd.NaT

    def _last_poke_out(entries):
        """Scan **back by position** to the first non-null ``poke_odor_end``.

        `entries` is this trial's `position_data` rows sorted by position.

        One of three different "last poke-out" rules in this repo; do not merge them.
        See DECISIONS.md section 28.
        """
        for poke in reversed(entries or []):
            dt_val = _safe_dt(poke.get("poke_odor_end"))
            if pd.notna(dt_val):
                return dt_val

        return pd.NaT

    def _end_time(row, cond):
        if cond == "rewarded":
            return _safe_dt(row.get("first_supply_time")) or _safe_dt(row.get("sequence_end"))
        if cond == "unrewarded":
            return _safe_dt(row.get("first_reward_poke_time"))
        if cond == "fa":
            return _safe_dt(row.get("fa_time")) or _safe_dt(row.get("sequence_end"))
        return _safe_dt(row.get("sequence_end"))

    def _last_valve_start(valve_entries, poke_entries):
        """Return the last valve_start that has a matching poke_odor_start for the same position.

        Iterates positions from highest to lowest; requires both valve_start and poke_odor_start
        for that position. If none meet both criteria, returns NaT.

        The two inputs are **flag-filtered views of the same `position_data`** --
        `in_valve_times` and `in_poke_times`. The filtering matters more here than
        anywhere else in this file: the valve view is a *superset*, holding activations
        whose poke registered as ~0 ms, so an unfiltered read would offer this join
        positions the poke view never had. See DECISIONS.md section 2.
        """
        pvt_entries = valve_entries or []
        ppt_entries = poke_entries or []

        poke_by_pos = {}
        for entry in ppt_entries:
            try:
                pos_key = int(entry.get("position"))
            except Exception:
                continue
            poke_start = _safe_dt(entry.get("poke_odor_start"))
            if pd.notna(poke_start):
                poke_by_pos[pos_key] = poke_start

        for entry in reversed(pvt_entries):
            try:
                pos = int(entry.get("position"))
            except Exception:
                pos = None
            valve_ts = _safe_dt(entry.get("valve_start"))
            if pd.isna(valve_ts):
                continue
            if pos is not None and pos in poke_by_pos:
                return valve_ts

        return pd.NaT

    def _condition_label(row):
        rtc = str(row.get("response_time_category", "")).lower()
        if rtc == "rewarded" and not row.get("is_aborted", False):
            return "rewarded"
        if rtc == "unrewarded" and not row.get("is_aborted", False):
            return "unrewarded"
        fa_label = str(row.get("fa_label", "")).lower()
        if fa_label.startswith("fa_") and fa_label in fa_labels:
            return "fa"
        return None

    def _speed_by_bins(tracking_df, zero_dt, end_dt, edges):
        # legacy helper for plotting on shared edges; threshold now uses per-trial binning
        if pd.isna(zero_dt) or pd.isna(end_dt) or end_dt <= zero_dt:
            return None
        start_dt = zero_dt - pd.Timedelta(seconds=pre_buffer_s)
        seg = tracking_df[(tracking_df["time"] >= start_dt) & (tracking_df["time"] <= end_dt)].copy()
        if seg.empty or {"X", "Y", "time"} - set(seg.columns):
            return None
        if len(seg) < 2:
            return None
        t_rel = (seg["time"] - zero_dt).dt.total_seconds().to_numpy()
        if not np.isfinite(t_rel).all() or np.ptp(t_rel) == 0:
            return None
        x = seg["X"].to_numpy()
        y = seg["Y"].to_numpy()
        vx = np.gradient(x, t_rel)
        vy = np.gradient(y, t_rel)
        speed = np.hypot(vx, vy)
        seg["speed"] = speed
        seg["t_rel"] = t_rel
        seg["bin"] = pd.cut(seg["t_rel"], bins=edges, right=False, include_lowest=True)
        grouped = seg.groupby("bin", observed=False)["speed"]
        agg_series = grouped.max() if mode == "max" else grouped.mean()
        mids = edges[:-1] + (edges[1] - edges[0]) / 2
        arr = np.full_like(mids, np.nan, dtype=float)
        bin_to_idx = {b: i for i, b in enumerate(agg_series.index.categories)}
        for b, v in agg_series.items():
            idx = bin_to_idx.get(b)
            if idx is not None:
                arr[idx] = v
        return mids, arr

    def _speed_series(tracking_df, zero_dt, end_dt, pre_buffer_s_local):
        """Return per-sample relative time (s) and speed for a trial segment."""
        if pd.isna(zero_dt) or pd.isna(end_dt) or end_dt <= zero_dt:
            return None, None
        start_dt = zero_dt - pd.Timedelta(seconds=pre_buffer_s_local)
        seg = tracking_df[(tracking_df["time"] >= start_dt) & (tracking_df["time"] <= end_dt)].copy()
        if len(seg) < 2 or {"X", "Y", "time"} - set(seg.columns):
            return None, None
        t_rel = (seg["time"] - zero_dt).dt.total_seconds().to_numpy()
        if not np.isfinite(t_rel).all() or np.ptp(t_rel) == 0:
            return None, None
        x = seg["X"].to_numpy()
        y = seg["Y"].to_numpy()
        vx = np.gradient(x, t_rel)
        vy = np.gradient(y, t_rel)
        speed = np.hypot(vx, vy)
        return t_rel, speed


    def _compute_tortuosity(tracking_df, t_zero, mids_trial, t_end):
        """Compute tortuosity using per-trial bins (non-fixed coords)."""
        if mids_trial is None or len(mids_trial) == 0 or pd.isna(t_zero) or pd.isna(t_end):
            return np.nan
        # choose start bin near target (-0.05s) else first
        mids_sorted = np.asarray(mids_trial)
        start_idx = np.where(np.abs(mids_sorted - start_target_s) <= (bin_ms / 1000.0) * 0.01)[0]
        if start_idx.size == 0:
            start_idx = np.array([0])
        mid_start = float(mids_sorted[start_idx[0]])
        half = bin_s / 2.0
        # Use bin end times (start/end of tortuosity window) clamped to the actual trial span
        start_time = t_zero + pd.Timedelta(seconds=mid_start + half)
        start_time = max(start_time, t_zero)
        if pd.notna(t_end):
            start_time = min(start_time, t_end)

        mid_last = float(mids_sorted[-1])
        end_time = t_zero + pd.Timedelta(seconds=mid_last + half)
        if pd.notna(t_end):
            end_time = min(end_time, t_end)

        if pd.isna(start_time) or pd.isna(end_time) or end_time <= start_time:
            return np.nan

        seg_mask = (tracking_df["time"] >= start_time) & (tracking_df["time"] <= end_time)
        seg = tracking_df.loc[seg_mask, ["X", "Y", "time"]].copy()
        if len(seg) < 2:
            return np.nan
        seg = seg.sort_values("time")

        start_frame_idx = int(np.argmin(np.abs((seg["time"] - start_time).dt.total_seconds())))
        end_frame_idx = int(np.argmin(np.abs((seg["time"] - end_time).dt.total_seconds())))
        start_xy = seg.iloc[start_frame_idx][["X", "Y"]].to_numpy(dtype=float)
        end_xy = seg.iloc[end_frame_idx][["X", "Y"]].to_numpy(dtype=float)

        x_arr = seg["X"].to_numpy(dtype=float)
        y_arr = seg["Y"].to_numpy(dtype=float)
        path_len = float(np.sum(np.hypot(np.diff(x_arr), np.diff(y_arr))))
        straight_len = float(np.hypot(end_xy[0] - start_xy[0], end_xy[1] - start_xy[1]))
        return path_len / straight_len if straight_len > 0 else np.nan

    def _path_length(x_arr, y_arr):
        x = np.asarray(x_arr, float)
        y = np.asarray(y_arr, float)
        if x.size < 2 or y.size < 2:
            return np.nan
        dx = np.diff(x)
        dy = np.diff(y)
        return float(np.sum(np.hypot(dx, dy)))

    per_session = []
    combined_data = {"rewarded": [], "unrewarded": [], "fa": []}

    for rec in ses_recs:
        date_str = rec.date_str
        results_dir = rec.results_dir
        if not rec.analysed:
            continue

        try:
            tracking, _ = _load_tracking_and_behavior(subjid, date_str)
        except Exception as e:
            print(f"Skipping {date_str}: tracking load failed ({e})")
            continue

        tracking = tracking.copy()
        tracking["time"] = pd.to_datetime(tracking["time"], errors="coerce")
        tracking = tracking.dropna(subset=["time"]).reset_index(drop=True)

        # Resolve X/Y columns
        for cand in [("centroid_x", "centroid_y"), ("X", "Y")]:
            if cand[0] in tracking.columns and cand[1] in tracking.columns:
                tracking["X"] = tracking[cand[0]]
                tracking["Y"] = tracking[cand[1]]
                break
        tracking = tracking.dropna(subset=["X", "Y"])
        tracking = tracking.loc[:, ~tracking.columns.duplicated()]
        if tracking.empty:
            continue

        # Smoothed copy for path-length computation (centered rolling, ~5-frame default)
        smooth_window_frames = 5
        tracking_smooth = tracking.copy()
        if smooth_window_frames > 1:
            tracking_smooth["X"] = (
                pd.Series(tracking_smooth["X"]).rolling(window=smooth_window_frames, center=True, min_periods=1).mean()
            )
            tracking_smooth["Y"] = (
                pd.Series(tracking_smooth["Y"]).rolling(window=smooth_window_frames, center=True, min_periods=1).mean()
            )
        tracking_smooth = tracking_smooth.dropna(subset=["X", "Y"])

        views = rec.views
        trial_data = views.get("trial_data", pd.DataFrame()).copy()
        if trial_data.empty:
            print(f"No trial_data for {date_str}; skipping")
            continue
        for c in ["sequence_start", "sequence_end", "first_supply_time", "first_reward_poke_time", "fa_time"]:
            if c in trial_data.columns:
                trial_data[c] = pd.to_datetime(trial_data[c], errors="coerce")

        # One row per trial x position for this session. **Each view must use the
        # provenance flag matching the facts its helper reads** -- see DECISIONS.md
        # section 2.
        position_data = _load_position_data(results_dir, trial_data)
        pokes_by_trial = position_entries_by_trial(position_data, "in_poke_times")
        valves_by_trial = position_entries_by_trial(position_data, "in_valve_times")

        trials_info = []
        skipped_no_poke_end = []
        for idx_row, row in trial_data.iterrows():
            cond = _condition_label(row)
            if cond is None:
                continue
            t_zero = _last_poke_out(pokes_by_trial.get(row.get("global_trial_id")))
            if pd.isna(t_zero):
                trial_id = row.get("trial_id", idx_row) if hasattr(row, "get") else idx_row
                skipped_no_poke_end.append(trial_id)
                continue
            t_end = _end_time(row, cond)
            if pd.isna(t_end) or t_end <= t_zero:
                continue
            dur_post = (t_end - t_zero).total_seconds()
            if dur_post <= 0:
                continue
            trials_info.append((idx_row, cond, t_zero, t_end, dur_post))

        if skipped_no_poke_end:
            print(f"Warning [{date_str}]: skipped trials with no poke_odor_end in position_poke_times: {skipped_no_poke_end}")

        if not trials_info:
            print(f"No usable trials for {date_str}")
            continue

        max_post = max(dur for _, _, _, _, dur in trials_info)
        edges = np.arange(-pre_buffer_s, max_post + bin_s, bin_s)
        if len(edges) < 2:
            continue

        epoch_series = {"rewarded": [], "unrewarded": [], "fa": []}
        speeds_flat = {"rewarded": [], "unrewarded": [], "fa": []}
        epoch_records = []  # flattened per-trial, per-bin speeds for downstream use
        baseline_vals = []
        # store per-trial threshold times
        trial_data["speed_threshold_time"] = pd.NaT
        trial_data["movement_onset_from_valve_s"] = np.nan
        # cache per-trial bins for threshold computation without writing arrays into the DataFrame
        trial_bins = {}
        mids_common = None
        movement_records = []

        for idx_row, cond, t_zero, t_end, _ in trials_info:
            # per-trial binning for threshold/baseline
            mids_trial, arr_trial = _binned_speed(tracking, t_zero, t_end, pre_buffer_s, bin_s, mode)
            if mids_trial is None:
                continue
            if threshold:
                mask_base = (mids_trial >= baseline_window[0]) & (mids_trial <= baseline_window[1])
                if mask_base.any():
                    baseline_vals.extend([v for v in arr_trial[mask_base] if not np.isnan(v)])

            # legacy/global binning for plotting alignment
            res_plot = _speed_by_bins(tracking, t_zero, t_end, edges)
            if res_plot is None:
                continue
            mids_plot, arr_plot = res_plot
            if mids_common is None:
                mids_common = mids_plot
            epoch_series[cond].append(arr_plot)
            speeds_flat[cond].extend([v for v in arr_plot if not np.isnan(v)])

            # Write per-trial per-bin records using the per-trial bins (mids_trial/arr_trial) to avoid extending past t_end.
            if mids_trial is not None and arr_trial is not None:
                for mid, val in zip(mids_trial, arr_trial):
                    mid_td = pd.Timedelta(seconds=float(mid))
                    half_bin = pd.Timedelta(seconds=float(bin_s / 2))
                    bin_mid_time = t_zero + mid_td
                    bin_start_time = bin_mid_time - half_bin
                    bin_end_time = bin_mid_time + half_bin
                    # Clamp end to t_end to avoid overshooting when global edges are longer than this trial
                    if bin_end_time > t_end:
                        bin_end_time = t_end
                    epoch_records.append({
                        "trial_index": idx_row,
                        "condition": cond,
                        "bin_mid_s": float(mid),
                        "bin_start_s": float(mid - bin_s / 2),
                        "bin_end_s": float(mid + bin_s / 2),
                        "bin_mid_time": bin_mid_time,
                        "bin_start_time": bin_start_time,
                        "bin_end_time": bin_end_time,
                        "speed": float(val) if not np.isnan(val) else np.nan,
                        "date": date_str,
                        "subjid": subjid,
                        "speed_threshold_time": pd.NaT,
                        "latency_s": np.nan,
                    })

            trial_bins[idx_row] = {
                "mids": mids_trial,
                "arr": arr_trial,
                "t_zero": t_zero,
            }

            # Path length (smoothed) and travel time between t_zero and t_end
            seg_path = tracking_smooth[(tracking_smooth["time"] >= t_zero) & (tracking_smooth["time"] <= t_end)]
            path_len = _path_length(seg_path["X"], seg_path["Y"]) if len(seg_path) >= 2 else np.nan
            travel_time_s = (t_end - t_zero).total_seconds() if pd.notna(t_end) and pd.notna(t_zero) else np.nan
            tortuosity_val = _compute_tortuosity(tracking, t_zero, mids_trial, t_end)
            movement_records.append({
                "trial_index": idx_row,
                "condition": cond,
                "path_length_px": path_len,
                "travel_time_s": travel_time_s,
                "tortuosity": tortuosity_val,
                "start_time": t_zero,
                "end_time": t_end,
                "date": date_str,
                "subjid": subjid,
            })

        conds_with_data = [c for c in ["rewarded", "unrewarded", "fa"] if epoch_series[c]]
        if not conds_with_data:
            print(f"No epoch data for {date_str}")
            continue

        stats = speed_threshold(baseline_vals, alpha=threshold_alpha,
                                beta=threshold_beta, enabled=threshold)
        baseline_mean, baseline_sd = stats["mu"], stats["sigma"]
        thr_alpha_mu = stats["alpha_mu"]
        thr_mu_plus_beta_sigma = stats["mu_plus_beta_sigma"]
        thr_max = stats["max_alpha_mu_mu_plus_beta_sigma"]

        # compute and store per-trial crossing times using per-trial bins
        if "latency_s" not in trial_data.columns:
            trial_data["latency_s"] = np.nan

        if threshold and thr_max is not None:
            for idx_row, cond, t_zero, t_end, _ in trials_info:
                # Bin-gated crossing: find first bin (mean) above threshold, then refine within that bin using per-sample speed
                crossing_time = pd.NaT
                latency_val = np.nan
                movement_from_valve = np.nan
                gid = trial_data.loc[idx_row].get("global_trial_id")
                valve_start = _last_valve_start(valves_by_trial.get(gid),
                                                pokes_by_trial.get(gid))

                bins = trial_bins.get(idx_row, {})
                mids_trial = bins.get("mids")
                arr_trial = bins.get("arr")

                # Identify first bin whose mean/max (per mode) exceeds threshold, after t=0
                bin_idx = None
                if mids_trial is not None and arr_trial is not None:
                    crossing_bins = np.where((mids_trial >= 0) & (arr_trial > thr_max))[0]
                    if crossing_bins.size > 0:
                        bin_idx = crossing_bins[0]

                if bin_idx is not None:
                    bin_mid = float(mids_trial[bin_idx])
                    half = bin_s / 2.0
                    win_start = bin_mid - half
                    win_end = bin_mid + half

                    # Per-sample refinement within the bin window
                    t_rel_series, speed_series = _speed_series(tracking, t_zero, t_end, pre_buffer_s)
                    if t_rel_series is not None and speed_series is not None:
                        mask = (t_rel_series >= win_start) & (t_rel_series <= win_end) & np.isfinite(speed_series)
                        if mask.any():
                            idx_cross = np.where(speed_series[mask] > thr_max)[0]
                            if idx_cross.size > 0:
                                idx_masked = idx_cross[0]
                                idx_global = np.where(mask)[0][idx_masked]

                                if idx_masked > 0:
                                    i1 = np.where(mask)[0][idx_masked - 1]
                                    i2 = idx_global
                                    t1, t2 = t_rel_series[i1], t_rel_series[i2]
                                    s1, s2 = speed_series[i1], speed_series[i2]
                                    if s2 != s1 and np.isfinite([t1, t2, s1, s2]).all():
                                        frac = (thr_max - s1) / (s2 - s1)
                                        frac = np.clip(frac, 0.0, 1.0)
                                        t_cross = t1 + frac * (t2 - t1)
                                    else:
                                        t_cross = t_rel_series[i2]
                                else:
                                    t_cross = t_rel_series[idx_global]

                                crossing_time = t_zero + pd.Timedelta(seconds=float(t_cross))
                                latency_val = float(t_cross)
                                if pd.notna(valve_start):
                                    movement_from_valve = (crossing_time - valve_start).total_seconds()

                    # If bin mean crossed but no per-sample crossing found within window, fallback to bin midpoint
                    if pd.isna(latency_val):
                        crossing_time = t_zero + pd.Timedelta(seconds=float(bin_mid))
                        latency_val = float(bin_mid)
                        if pd.notna(valve_start):
                            movement_from_valve = (crossing_time - valve_start).total_seconds()

                if pd.notna(crossing_time):
                    trial_data.at[idx_row, "speed_threshold_time"] = crossing_time
                trial_data.at[idx_row, "latency_s"] = latency_val
                trial_data.at[idx_row, "movement_onset_from_valve_s"] = movement_from_valve

        # Fill per-bin records with per-trial metrics (repeat per bin so one file carries both)
        if "speed_threshold_time" in trial_data.columns:
            thr_map = trial_data["speed_threshold_time"].to_dict()
            lat_map = trial_data["latency_s"].to_dict() if "latency_s" in trial_data.columns else {}
            mov_map = trial_data["movement_onset_from_valve_s"].to_dict() if "movement_onset_from_valve_s" in trial_data.columns else {}
            # Movement metrics per trial
            path_map = {}
            travel_map = {}
            tort_map = {}
            for rec_mov in movement_records:
                tid = rec_mov.get("trial_index")
                path_map[tid] = rec_mov.get("path_length_px")
                travel_map[tid] = rec_mov.get("travel_time_s")
                tort_map[tid] = rec_mov.get("tortuosity")

            for rec in epoch_records:
                tid = rec["trial_index"]
                rec["speed_threshold_time"] = thr_map.get(tid, pd.NaT)
                rec["latency_s"] = lat_map.get(tid, np.nan)
                rec["movement_onset_from_valve_s"] = mov_map.get(tid, np.nan)
                rec["path_length_px"] = path_map.get(tid, np.nan)
                rec["travel_time_s"] = travel_map.get(tid, np.nan)
                rec["tortuosity"] = tort_map.get(tid, np.nan)

        # The threshold that produced the latencies above, recorded on every row so a
        # reader never has to assume today's defaults (THRESHOLD_COLUMNS).
        for rec in epoch_records:
            rec["baseline_mu"] = baseline_mean
            rec["baseline_sigma"] = baseline_sd
            rec["threshold_alpha"] = threshold_alpha
            rec["threshold_beta"] = threshold_beta
            rec["speed_threshold"] = thr_max

        analysis_path = layout.table_path(results_dir, "speed_analysis.parquet")
        try:
            analysis_path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(epoch_records).to_parquet(analysis_path, index=False)
        except Exception as e:
            print(f"Warning: failed to write {analysis_path.name}: {e}")
        else:
            _update_cache(subjid, [date_str], {date_str: pd.DataFrame(epoch_records)}, kind="speed_analysis")


        per_session.append({
            "date": date_str,
            "baseline": {
                "mu": baseline_mean,
                "sigma": baseline_sd,
                "alpha": threshold_alpha,
                "beta": threshold_beta,
                "alpha_mu": thr_alpha_mu,
                "mu_plus_beta_sigma": thr_mu_plus_beta_sigma,
                "max_alpha_mu_mu_plus_beta_sigma": thr_max,
            } if threshold else None,
            "trial_data_with_threshold": trial_data.copy(),
        })

        for cond in conds_with_data:
            stack = np.vstack(epoch_series[cond])
            session_mean = np.nanmean(stack, axis=0)
            mids_combined = mids_common if mids_common is not None else (edges[:-1] + (edges[1] - edges[0]) / 2)
            combined_data[cond].append((date_str, mids_combined, session_mean))


    return {"per_session": per_session}

# Public alias; `_binned_speed` is the name `visualization/` already imports.
binned_speed = _binned_speed
