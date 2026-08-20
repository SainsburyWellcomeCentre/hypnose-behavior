# Defers evaluation of PEP-604 annotations (`X | None`), keeping this module
# importable on Python 3.9 for repos pinned there (hypnose-eeg-preprocessing).
from __future__ import annotations

"""The cross-metric overview figure.

``plot_behavior_metrics`` draws any registered metric by name, so it is the one
plotter that is about no single behavioural construct. DECISIONS section 5: it
computes through the registry and never reads ``metrics_*.json``.
"""

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from typing import (
    Iterable,
    Optional,
    Union,
    Tuple,
)
from hypnose_behavior.utils.helpers import (
    _filter_session_dirs,
    _iter_subject_dirs,
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
from hypnose_behavior.io.save import save_figure
from hypnose_behavior.visualization.prep import _load_protocol_from_summary
from hypnose_behavior.visualization.prep import (
    _computed_metrics,
    _extract_metric_value,
    _series_line_widths,
)



def plot_behavior_metrics(
    subjids: Optional[Iterable[int]] = None,
    dates: Optional[Union[Iterable[Union[int, str]], tuple]] = None,
    variables: Optional[Iterable[str]] = None,
    *,
    ses=None,
    index=None,
    date_range=None,
    ses_range=None,
    index_range=None,
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

    ``ses`` / ``index`` / ``date_range`` / ``ses_range`` / ``index_range`` narrow the
    selection further; they intersect with ``dates`` and with each other, and ``index``
    is the subject's gap-free chronological rank (`utils.helpers.session_selectors`).
    """
    select = session_selectors(
        ses=ses, index=index, date_range=date_range,
        ses_range=ses_range, index_range=index_range,
    )
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
        ses_dirs = _filter_session_dirs(subj_dir, subj_dates, **select)
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
        cmap = plt.get_cmap("tab20", max(20, len(unique_protocols)))
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
            subj_cmap = plt.get_cmap("tab20")
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
            series_cmap = plt.get_cmap("tab20", max(3, len(series_values)))
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
