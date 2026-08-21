# Defers evaluation of PEP-604 annotations (`X | None`), keeping this module
# importable on Python 3.9 for repos pinned there (hypnose-eeg-preprocessing).
from __future__ import annotations

"""Hidden-rule figures.

The A/B association of a hidden-rule odor is never hard-coded; it is learned
from the animal's own hidden-rule sessions via ``hr_odor_associations``, which
is why the colour builder these share sits in ``prep`` with it.
"""

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from hypnose_behavior.frames import odor_letter
from hypnose_behavior.metric_analysis.metrics.false_alarm import (
    fa_port_counts,
    fa_port_ratio,
    fa_rate_by_odor,
)
from hypnose_behavior.metric_analysis.metrics.hidden_rule import (
    hr_abort_poke_gap,
    rolling_hr_reward_fraction,
)
from hypnose_behavior.io import layout
from hypnose_behavior.io.layout import (
    _iter_subject_dirs,
    derivatives,
    normalize_subjid,
    session_selectors,
)
from hypnose_behavior.io.paths import (
    get_rawdata_root,
    get_derivatives_root,
    get_server_root,
)
import numpy as np
import json
from hypnose_behavior.io.save import save_figure
from hypnose_behavior.visualization.primitives import mean_sem
from hypnose_behavior.io.loaders import (
    _load_position_data,
    _load_table_with_trial_data,
    _odor_to_letter,
    iter_sessions,
)
from hypnose_behavior.visualization.prep import (
    _build_odor_colors,
    _computed_metrics,
    _extract_metric_value,
)



def hidden_rule_and_false_alarm(
    subjids=None,
    dates=None,
    odors=("C", "D", "E", "F", "G"),
    fa_label=None,
    figsize=(12, 9),
    title=None,
    *,
    ses=None,
    index=None,
    date_range=None,
    ses_range=None,
    index_range=None,
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

    ``ses`` / ``index`` / ``date_range`` / ``ses_range`` / ``index_range`` narrow the
    selection further; they intersect with ``dates`` and with each other, and ``index``
    is the subject's gap-free chronological rank (`io.layout.session_selectors`).
    """
    select = session_selectors(
        ses=ses, index=index, date_range=date_range,
        ses_range=ses_range, index_range=index_range,
    )
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
        ses_recs = iter_sessions(subj_dir, subj_dates, **select)
        for session_num, rec in enumerate(ses_recs, start=1):
            date_str = rec.date_str
            results_dir = rec.results_dir
            if not rec.analysed:
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
            td = rec.views["trial_data"]
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



def plot_fa_ratio_by_hr_position(
    subjid,
    dates=None,
    figsize=(16, 10),
    fa_types='FA_time_in', 
    print_statistics=False,
    exclude_last_pos=False,
    last_odor_num=5,
    debug=False,
    *,
    ses=None,
    index=None,
    date_range=None,
    ses_range=None,
    index_range=None,
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
        ses_recs = iter_sessions(subj_dir, dates, **select)
        
        for session_num, rec in enumerate(ses_recs, 1):
            date_str = rec.date_str
            results_dir = rec.results_dir
            
            if not rec.analysed:
                continue
            
            summary_path = layout.table_path(results_dir, "summary.json")
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

                views = rec.views
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
    ses=None,
    index=None,
    date_range=None,
    ses_range=None,
    index_range=None,
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

    ``ses`` / ``index`` / ``date_range`` / ``ses_range`` / ``index_range`` narrow the
    selection further; they intersect with ``dates`` and with each other, and ``index``
    is the subject's gap-free chronological rank (`io.layout.session_selectors`).
    """
    select = session_selectors(
        ses=ses, index=index, date_range=date_range,
        ses_range=ses_range, index_range=index_range,
    )

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
        ses_recs = iter_sessions(subj_dir, dates, **select)

        for rec in ses_recs:
            date_str = rec.date_str
            try:
                date_val = int(date_str)
            except Exception:
                date_val = date_str

            results_dir = rec.results_dir
            if not rec.analysed:
                continue

            td = _load_table_with_trial_data(results_dir, "trial_data")
            if td.empty:
                continue

            td = td.copy()
            td["is_aborted"] = td.get("is_aborted", False).fillna(False)
            td["hit_hidden_rule"] = td.get("hit_hidden_rule", False).fillna(False)

            # One session at a time: the metric keys on `global_trial_id`, which
            # repeats across sessions.
            gaps = hr_abort_poke_gap(td, _load_position_data(results_dir, td))
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
    ses=None,
    index=None,
    date_range=None,
    ses_range=None,
    index_range=None,
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

    ``ses`` / ``index`` / ``date_range`` / ``ses_range`` / ``index_range`` narrow the
    selection further; they intersect with ``dates`` and with each other, and ``index``
    is the subject's gap-free chronological rank (`io.layout.session_selectors`).
    """
    select = session_selectors(
        ses=ses, index=index, date_range=date_range,
        ses_range=ses_range, index_range=index_range,
    )

    window_size = max(int(window_size), 1)
    derivatives_dir = get_derivatives_root()

    frames = []
    for sid, subj_dir in _iter_subject_dirs(derivatives_dir, [subjid]):
        ses_recs = iter_sessions(subj_dir, dates, **select)
        for rec in ses_recs:
            date_str = rec.date_str
            try:
                date_val = int(date_str)
            except Exception:
                date_val = date_str

            results_dir = rec.results_dir
            if not rec.analysed:
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
