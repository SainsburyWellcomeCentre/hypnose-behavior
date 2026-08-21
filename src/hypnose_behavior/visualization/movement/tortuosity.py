# Defers evaluation of PEP-604 annotations (`X | None`), keeping this module
# importable on Python 3.9 for repos pinned there (hypnose-eeg-preprocessing).
from __future__ import annotations

"""Tortuosity overlay figures from SLEAP tracking."""

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from hypnose_behavior.utils.helpers import _get_from_cache, _update_cache
from hypnose_behavior.io import layout
from hypnose_behavior.io.layout import (
    derivatives,
    normalize_subjid,
    session_selectors,
)
from hypnose_behavior.io.loaders import iter_sessions
from hypnose_behavior.io.tracking import _load_tracking_and_behavior
from hypnose_behavior.io.save import save_figure
import re
import numpy as np
from hypnose_behavior.io.save import MOVEMENT_FIGURES_SUBDIR



def plot_tortuosity_lines_overlay(
    subjid,
    dates=None,
    *,
    ses=None,
    index=None,
    date_range=None,
    ses_range=None,
    index_range=None,
    fa_types="FA_time_in",
    bin_ms: int = 100,
    fixed_start_xy=(575, 90),
    fixed_goal_a_xy=(208, 930),
    fixed_goal_b_xy=(973, 930),
    figsize=(8, 8),
    save: bool = False,
    verbose: bool = True,
    return_paths: bool = False,
):
    """Plot traces by condition with both data-derived tortuosity lines and fixed lines overlaid.

    Uses speed_analysis.parquet to align start/end times per trial. For each trial, draws the trajectory,
    a line from start→goal derived from tracking, and a fixed start→goal line (A/B) using provided coordinates.
    Returns a dict of figures keyed by (date, condition). When save=True, PDFs are written into
    movement_figures via save_figure(), and return_paths controls whether saved paths are returned.

    ``ses`` / ``index`` / ``date_range`` / ``ses_range`` / ``index_range`` narrow the
    selection further; they intersect with ``dates`` and with each other, and ``index``
    is the subject's gap-free chronological rank (`io.layout.session_selectors`).
    """
    select = session_selectors(
        ses=ses, index=index, date_range=date_range,
        ses_range=ses_range, index_range=index_range,
    )

    saved_paths = []

    def _slugify(value: str) -> str:
        slug = re.sub(r"[^0-9a-zA-Z]+", "_", str(value)).strip("_")
        return slug.lower() or "figure"

    def _save_fig(fig, save_name: str, date_scope):
        if not save:
            return
        try:
            out_path = save_figure(
                fig,
                save_name,
                subjids=[subjid],
                dates=date_scope,
                subdir=MOVEMENT_FIGURES_SUBDIR,
            )
            saved_paths.append(out_path)
            if verbose:
                print(f"[plot_tortuosity_lines_overlay] Saved figure to {out_path}")
        except Exception as exc:
            if verbose:
                print(f"[plot_tortuosity_lines_overlay] Failed to save figure '{save_name}': {exc}")

    # FA filter
    fa_label_display = "FA"
    if isinstance(fa_types, str):
        if fa_types.lower() == "all":
            fa_label_display = "all"
            def fa_filter_fn(lbl):
                return str(lbl).lower().startswith("fa_") if pd.notna(lbl) else False
        else:
            fa_set = {s.strip().lower() for s in re.split(r"[;,]", fa_types) if s.strip()}
            fa_label_display = ", ".join(sorted(fa_set)) if fa_set else "selected"
            def fa_filter_fn(lbl):
                return str(lbl).lower() in fa_set if pd.notna(lbl) else False
    else:
        fa_set = {str(s).strip().lower() for s in fa_types}
        fa_label_display = ", ".join(sorted(fa_set)) if fa_set else "selected"
        def fa_filter_fn(lbl):
            return str(lbl).lower() in fa_set if pd.notna(lbl) else False

    suffix_parts = [f"bin{bin_ms}"]
    if fa_label_display:
        suffix_parts.append(_slugify(fa_label_display))
    save_suffix = "_".join(filter(None, suffix_parts))

    subj_str = normalize_subjid(subjid)
    subj_dir = derivatives.subject_dir(subjid)

    ses_recs = iter_sessions(subj_dir, dates, **select)
    if not ses_recs:
        raise FileNotFoundError(f"No sessions found for subject {subjid} with given dates")

    start_target_s = -bin_ms / 2000.0
    port_colors = {1: "#FF6B6B", 2: "#4ECDC4"}
    cond_colors = {"rewarded": "#4CAF50", "unrewarded": "#F44336", "fa": "#2196F3"}
    data_line_color = "#424242"
    fixed_line_color = "#9C27B0"

    def _port_from_identity(val):
        if pd.isna(val):
            return None
        s = str(val).strip().lower()
        if s in {"a", "odora", "odor_a", "1", "porta", "port_a"}:
            return 1
        if s in {"b", "odorb", "odor_b", "2", "portb", "port_b"}:
            return 2
        return None

    def _infer_port_with_supply_identity(row):
        for col in [
            "response_port", "rewarded_port", "reward_port", "supply_port",
            "choice_port", "port", "fa_port", "first_supply_port",
            "first_reward_poke_port", "last_reward_port", "odor_port",
        ]:
            if col in row and pd.notna(row[col]):
                try:
                    return int(row[col])
                except Exception:
                    try:
                        return int(float(row[col]))
                    except Exception:
                        continue
        for col in ["first_supply_odor_identity", "last_odor_name", "last_odor", "odor_name", "odor"]:
            if col in row:
                port = _port_from_identity(row.get(col))
                if port is not None:
                    return port
        return None

    def _port_for_coloring(row, cond):
        if cond == "fa":
            preferred_cols = ["fa_port"]
        elif cond == "unrewarded":
            preferred_cols = ["first_reward_poke_port", "response_port", "choice_port"]
        else:
            preferred_cols = ["first_supply_port", "first_supply_odor_identity", "rewarded_port", "reward_port"]

        for col in preferred_cols:
            if col not in row or pd.isna(row[col]):
                continue
            port = _port_from_identity(row[col])
            if port is not None:
                return port
            try:
                return int(row[col])
            except Exception:
                try:
                    return int(float(row[col]))
                except Exception:
                    continue
        return _infer_port_with_supply_identity(row)

    def _condition_label(row):
        rtc = str(row.get("response_time_category", "")).lower()
        is_aborted = bool(row.get("is_aborted", False))
        fa_label = str(row.get("fa_label", "")).lower()
        if rtc == "rewarded" and not is_aborted:
            return "rewarded"
        if rtc == "unrewarded" and not is_aborted:
            return "unrewarded"
        if fa_label.startswith("fa_") and fa_filter_fn(fa_label):
            return "fa"
        return None

    figs = {}

    for rec in ses_recs:
        date_str = rec.date_str
        results_dir = rec.results_dir
        if not rec.analysed:
            continue

        views = rec.views
        trial_data = views.get("trial_data", pd.DataFrame()).copy()
        if trial_data.empty:
            continue
        for c in ["sequence_start", "sequence_end", "fa_time", "first_supply_time", "first_reward_poke_time"]:
            if c in trial_data.columns:
                trial_data[c] = pd.to_datetime(trial_data[c], errors="coerce")

        try:
            tracking, _ = _load_tracking_and_behavior(subjid, date_str)
        except Exception as e:
            print(f"Skipping {date_str}: tracking load failed ({e})")
            continue
        tracking = tracking.copy()
        tracking["time"] = pd.to_datetime(tracking["time"], errors="coerce")
        tracking = tracking.dropna(subset=["time"]).reset_index(drop=True)
        for cand in [("centroid_x", "centroid_y"), ("X", "Y")]:
            if cand[0] in tracking.columns and cand[1] in tracking.columns:
                tracking["X"] = tracking[cand[0]]
                tracking["Y"] = tracking[cand[1]]
                break
        tracking = tracking.dropna(subset=["X", "Y"])
        tracking = tracking.loc[:, ~tracking.columns.duplicated()]
        if tracking.empty:
            continue

        speed_df = _get_from_cache(subjid, date_str, kind="speed_analysis")
        if speed_df is None:
            path_speed = layout.table_path(results_dir, "speed_analysis.parquet")
            if not path_speed.exists():
                print(f"No speed_analysis.parquet for {date_str}; run plot_epoch_speeds_by_condition first")
                continue
            try:
                speed_df = pd.read_parquet(path_speed)
                _update_cache(subjid, [date_str], {date_str: speed_df.copy()}, kind="speed_analysis")
            except Exception as e:
                print(f"Failed to read speed_analysis for {date_str}: {e}")
                continue
        speed_df = speed_df.copy()
        for col in ["bin_mid_time", "bin_start_time", "bin_end_time"]:
            if col in speed_df.columns:
                speed_df[col] = pd.to_datetime(speed_df[col], errors="coerce")

        traces = {"rewarded": [], "unrewarded": [], "fa": []}
        data_lines = {"rewarded": [], "unrewarded": [], "fa": []}
        fixed_lines = {"rewarded": [], "unrewarded": [], "fa": []}

        for idx_row, row in trial_data.iterrows():
            cond = _condition_label(row)
            if cond is None:
                continue
            bins_df = speed_df[speed_df["trial_index"] == idx_row].sort_values("bin_mid_s")
            if bins_df.empty:
                continue
            start_bin = bins_df.loc[(bins_df["bin_mid_s"].sub(start_target_s).abs() <= (bin_ms / 1000.0) * 0.01)]
            if start_bin.empty:
                start_bin = bins_df.head(1)
            if start_bin.empty or "bin_end_time" not in start_bin.columns:
                continue
            start_time = pd.to_datetime(start_bin.iloc[0]["bin_end_time"], errors="coerce")
            end_time = pd.to_datetime(bins_df.sort_values("bin_end_time").iloc[-1]["bin_end_time"], errors="coerce") if "bin_end_time" in bins_df.columns else pd.NaT
            if pd.isna(start_time) or pd.isna(end_time) or end_time <= start_time:
                continue

            seg = tracking[(tracking["time"] >= start_time) & (tracking["time"] <= end_time)][["X", "Y", "time"]].copy()
            if len(seg) < 2:
                continue
            seg = seg.sort_values("time")
            x_arr = seg["X"].to_numpy(dtype=float)
            y_arr = seg["Y"].to_numpy(dtype=float)

            start_idx = int(np.argmin(np.abs((seg["time"] - start_time).dt.total_seconds())))
            end_idx = int(np.argmin(np.abs((seg["time"] - end_time).dt.total_seconds())))
            start_xy = seg.iloc[start_idx][["X", "Y"]].to_numpy(dtype=float)
            end_xy = seg.iloc[end_idx][["X", "Y"]].to_numpy(dtype=float)

            port = _port_for_coloring(row, cond)
            fixed_start = np.asarray(fixed_start_xy, dtype=float)
            fixed_goal = np.asarray(fixed_goal_b_xy if port == 2 else fixed_goal_a_xy, dtype=float)

            trace_color = port_colors.get(port, cond_colors[cond]) if cond in {"rewarded", "unrewarded"} else cond_colors[cond]

            traces[cond].append((x_arr, y_arr, trace_color))
            data_lines[cond].append((start_xy, end_xy))
            fixed_lines[cond].append((fixed_start, fixed_goal))

        for cond in ["rewarded", "unrewarded", "fa"]:
            if not traces[cond]:
                continue
            fig, ax = plt.subplots(figsize=figsize)
            for (x_arr, y_arr, trace_color), (sxy, gxy), (fsxy, fgxy) in zip(traces[cond], data_lines[cond], fixed_lines[cond]):
                ax.plot(x_arr, y_arr, color=trace_color)
                ax.plot([sxy[0], gxy[0]], [sxy[1], gxy[1]], color=data_line_color, linestyle="--")
                ax.plot([fsxy[0], fgxy[0]], [fsxy[1], fgxy[1]], color=fixed_line_color)
            # Always show a reference fixed B line for visual comparison
            ax.plot(
                [fixed_start_xy[0], fixed_goal_b_xy[0]],
                [fixed_start_xy[1], fixed_goal_b_xy[1]],
                color=fixed_line_color,
            )
            ax.set_title(f"{cond.capitalize()} traces with data vs fixed lines — {date_str}")
            ax.set_xlabel("X (px)")
            ax.set_ylabel("Y (px)")
            ax.set_aspect("equal", adjustable="box")
            ax.invert_yaxis()
            if cond in {"rewarded", "unrewarded"}:
                from matplotlib.lines import Line2D
                legend_handles = [
                    Line2D([0], [0], color=port_colors[1], lw=2, label="A / port 1 trace"),
                    Line2D([0], [0], color=port_colors[2], lw=2, label="B / port 2 trace"),
                    Line2D([0], [0], color=data_line_color, lw=2, linestyle="--", label="data start-end"),
                    Line2D([0], [0], color=fixed_line_color, lw=2, label="fixed start-goal"),
                ]
                ax.legend(handles=legend_handles, loc="best")
            figs[(date_str, cond)] = fig

            date_scope = [int(date_str)] if str(date_str).isdigit() else [date_str]
            save_name = f"tortuosity_overlay_{_slugify(cond)}_{_slugify(date_str)}"
            if save_suffix:
                save_name = f"{save_name}_{save_suffix}"
            _save_fig(fig, save_name, date_scope or dates)
    if save and return_paths:
        return figs, saved_paths
    return figs
