# Defers evaluation of PEP-604 annotations (`X | None`), keeping this module
# importable on Python 3.9 for repos pinned there (hypnose-eeg-preprocessing).
from __future__ import annotations

"""Per-condition distributions of the movement metrics, with tests.

Named ``summary_stats`` rather than ``statistics``/``stats``: the first shadows
a standard-library module, and the second would sit opposite
``metric_analysis/stats/``, which *computes* the tests this module *draws* --
the section 20 collision, in miniature.
"""

import pandas as pd
import matplotlib.pyplot as plt
from hypnose_behavior.frames import odor_letter
from hypnose_behavior.utils.helpers import _get_from_cache, _update_cache
from hypnose_behavior.io.layout import (
    _filter_session_dirs,
    derivatives,
    normalize_subjid,
    session_selectors,
)
from hypnose_behavior.io.loaders import _load_trial_views
from hypnose_behavior.visualization.panels import _clean_graph
from hypnose_behavior.visualization.primitives import mean_sem
from hypnose_behavior.metric_analysis.stats.kw_mwu import kw_mwu_by_group
from hypnose_behavior.io.save import save_figure
import re
import numpy as np
from hypnose_behavior.io.save import MOVEMENT_FIGURES_SUBDIR



def plot_movement_analysis_statistics(
    subjid,
    dates=None,
    *,
    ses=None,
    index=None,
    date_range=None,
    ses_range=None,
    index_range=None,
    fa_types="FA_time_in",
    figsize=(10, 6),
    clean_graph: bool = False,
    hidden_rule_analysis: bool = False,
    save: bool = False,
    verbose: bool = True,
    return_paths: bool = False,
):
    """Scatter movement-related metrics per condition with mean±SEM.

    Produces five figures per session when data are present (expanded category set when hidden_rule_analysis is True and only a single session is requested):
    - Movement onset latency relative to poke_out (latency_s from speed_analysis.parquet)
    - Animal's Consideration Time (Valve Onset - Movement Onset) (movement_onset_from_valve_s from speed_analysis.parquet)
    - Path length traveled per trial (path_length_px from speed_analysis.parquet)
    - Movement duration per trial (travel_time_s from speed_analysis.parquet)
    - Tortuosity per trial (tortuosity from speed_analysis.parquet)

    Returns dict with per-session figs and combined figs when multiple dates are provided. When
    save=True, each figure is written to movement_figures via save_figure(); return_paths=True
    additionally returns the list of saved file paths.

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
        if not save or fig is None:
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
                print(f"[plot_movement_analysis_statistics] Saved figure to {out_path}")
        except Exception as exc:
            if verbose:
                print(f"[plot_movement_analysis_statistics] Failed to save figure '{save_name}': {exc}")

    # FA filter
    if isinstance(fa_types, str):
        if fa_types.lower() == "all":
            def fa_filter_fn(lbl):
                return str(lbl).lower().startswith("fa_") if pd.notna(lbl) else False
        else:
            fa_set = {s.strip().lower() for s in re.split(r"[;,]", fa_types) if s.strip()}
            def fa_filter_fn(lbl):
                return str(lbl).lower() in fa_set if pd.notna(lbl) else False
    else:
        fa_set = {str(s).strip().lower() for s in fa_types}
        def fa_filter_fn(lbl):
            return str(lbl).lower() in fa_set if pd.notna(lbl) else False

    subj_str = normalize_subjid(subjid)
    subj_dir = derivatives.subject_dir(subjid)

    ses_dirs = _filter_session_dirs(subj_dir, dates, **select)
    if not ses_dirs:
        raise FileNotFoundError(f"No sessions found for subject {subjid} with given dates")


    def _has_odor(seq, odor_letter: str) -> bool:
            if seq is None:
                return False
            odor_letter = str(odor_letter).upper()
            if isinstance(seq, (list, tuple, set)):
                upper_vals = {str(x).upper() for x in seq}
                return odor_letter in upper_vals
            s = str(seq).upper()
            return odor_letter in s

    def _condition_labels_base(row):
        rtc = str(row.get("response_time_category", "")).lower()
        is_aborted = bool(row.get("is_aborted", False))
        fa_label = str(row.get("fa_label", "")).lower()
        if rtc == "rewarded" and not is_aborted:
            return ["rewarded"]
        if rtc == "unrewarded" and not is_aborted:
            return ["unrewarded"]
        if fa_label.startswith("fa_") and fa_filter_fn(fa_label):
            return ["fa"]
        return []

    def _condition_labels_hidden(row):
        labels = []
        rtc = str(row.get("response_time_category", "")).lower()
        is_aborted = bool(row.get("is_aborted", False))
        fa_label = str(row.get("fa_label", "")).lower()
        hr_success = bool(row.get("hidden_rule_success", False))
        hr_hit = bool(row.get("hit_hidden_rule", False))
        odor_seq = row.get("odor_sequence", None)
        fa_port = row.get("fa_port", None)

        if not is_aborted:
            if rtc == "rewarded":
                labels.append("Rewarded (Total)")
                labels.append("Rewarded (HR)" if hr_success else "Rewarded (no HR)")
            elif rtc == "unrewarded":
                labels.append("Unrewarded (Total)")
                labels.append("Unrewarded (HR)" if hr_success else "Unrewarded (no HR)")
        else:
            if fa_label.startswith("fa_") and fa_filter_fn(fa_label):
                labels.append("FA (Total)")
                if hr_hit:
                    labels.append("FA (HR)")
                    has_f = _has_odor(odor_seq, "F")
                    has_c = _has_odor(odor_seq, "C")
                    port = None
                    try:
                        port = int(fa_port) if fa_port is not None else None
                    except Exception:
                        port = None
                    if port is not None and (has_f or has_c):
                        if (has_f and port == 1) or (has_c and port == 2):
                            labels.append("FA (correct HR Port)")
                        elif (has_f and port == 2) or (has_c and port == 1):
                            labels.append("FA (incorrect HR Port)")
                else:
                    labels.append("FA (no HR)")

        return list(dict.fromkeys(labels))

    def _condition_labels(row):
        if hidden_rule_analysis:
            return _condition_labels_hidden(row)
        return _condition_labels_base(row)


    multi_session = len(ses_dirs) > 1

    per_session = []
    combined_rows = []  # base conditions (rewarded/unrewarded/fa)
    combined_valve_rows = []
    combined_path_rows = []
    combined_travel_rows = []
    combined_tortuosity_rows = []

    combined_rows_hr = []  # expanded HR conditions (if enabled)
    combined_valve_rows_hr = []
    combined_path_rows_hr = []
    combined_travel_rows_hr = []
    combined_tortuosity_rows_hr = []

    if hidden_rule_analysis:
        cond_groups = [
            ["Rewarded (Total)", "Rewarded (no HR)", "Rewarded (HR)"],
            ["Unrewarded (Total)", "Unrewarded (no HR)", "Unrewarded (HR)"],
            ["FA (Total)", "FA (no HR)", "FA (HR)", "FA (correct HR Port)", "FA (incorrect HR Port)"],
        ]
        cond_order = [c for group in cond_groups for c in group]
        palette = [
            "#4CAF50", "#8BC34A", "#2E7D32",  # rewarded variants
            "#F44336", "#E57373", "#B71C1C",  # unrewarded variants
            "#2196F3", "#64B5F6", "#0D47A1",  # FA variants
            "#00BCD4", "#00FFBFE8",              # FA correct/incorrect
        ]
        cond_colors = {c: palette[i % len(palette)] for i, c in enumerate(cond_order)}

        def _build_positions(groups, within=0.26, gap=0.55):
            pos = 0.0
            positions = {}
            for gi, group in enumerate(groups):
                for ci, cond in enumerate(group):
                    positions[cond] = pos
                    if ci < len(group) - 1:
                        pos += within
                if gi < len(groups) - 1:
                    pos += gap
            return positions

        cond_positions = _build_positions(cond_groups)
    else:
        cond_order = ["rewarded", "unrewarded", "fa"]
        cond_colors = {"rewarded": "#4CAF50", "unrewarded": "#F44336", "fa": "#2196F3"}
        cond_positions = {cond: idx * 0.35 for idx, cond in enumerate(cond_order)}

    cond_order_hr = cond_order if hidden_rule_analysis else []
    cond_colors_hr = cond_colors if hidden_rule_analysis else {}

    jitter_span = 0.06  # tighter jitter to match closer grouping
    cond_pos_values = list(cond_positions.values())
    _cond_xlim = (
        (min(cond_pos_values) - 0.2) if cond_pos_values else -0.2,
        (max(cond_pos_values) + 0.2) if cond_pos_values else 1.0,
    )

    def _display_label(cond: str) -> str:
        if cond in {"rewarded", "unrewarded", "fa"}:
            return cond.capitalize()
        return cond

    def _style_axis(ax, *, ylabel: str, xticklabels=None):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(2.5)
        ax.spines["bottom"].set_linewidth(2.5)
        ax.tick_params(axis="y", width=2.3, labelsize=13)
        ax.tick_params(axis="x", width=2.0, labelsize=13)
        if xticklabels is not None:
            ax.set_xticklabels(xticklabels, fontsize=13)
        ax.set_ylabel(ylabel, fontsize=16)
        if clean_graph:
            _clean_graph(ax, ylabel=ylabel)

    def _plot_by_trial_sequence(df, value_col, ylabel):
        df_seq = df.copy()
        df_seq["seq_in_condition"] = df_seq.groupby("condition").cumcount() + 1

        fig_seq, ax_seq = plt.subplots(figsize=figsize)
        for cond in cond_order:
            sub = df_seq[df_seq["condition"] == cond]
            if sub.empty:
                continue
            color = cond_colors.get(cond, "#555555")
            x_vals = sub["seq_in_condition"].astype(float).to_numpy()
            y_vals = sub[value_col].astype(float).to_numpy()
            ax_seq.scatter(x_vals, y_vals, color=color, alpha=0.7)
            if len(x_vals) >= 2:
                slope, intercept = np.polyfit(x_vals, y_vals, 1)
                x_line = np.array([x_vals.min(), x_vals.max()])
                y_line = slope * x_line + intercept
                ax_seq.plot(x_line, y_line, color=color, linewidth=2.0, alpha=0.9,
                             label=f"{cond}: y={slope:.3f}x+{intercept:.3f}")
            else:
                ax_seq.plot([], [], color=color, linewidth=0, label=f"{cond}: n={len(x_vals)}")

        ax_seq.set_xlabel("Trial # (within condition)", fontsize=14)
        _style_axis(ax_seq, ylabel=ylabel)
        ax_seq.legend()
        fig_seq.tight_layout()
        return fig_seq

    for ses_dir in ses_dirs:
        date_str = ses_dir.name.split("_date-")[-1]
        date_scope = [int(date_str)] if str(date_str).isdigit() else [date_str]
        date_slug = _slugify(date_str)
        results_dir = ses_dir / "saved_analysis_results"
        if not results_dir.exists():
            continue

        # load trial_data
        views = _load_trial_views(results_dir)
        trial_data = views.get("trial_data", pd.DataFrame()).copy()
        if trial_data.empty:
            continue
        for c in ["response_time_category", "fa_label", "is_aborted"]:
            if c in trial_data.columns:
                continue

        # load speed_analysis
        speed_df = _get_from_cache(subjid, date_str, kind="speed_analysis")
        if speed_df is None:
            path_speed = results_dir / "speed_analysis.parquet"
            if not path_speed.exists():
                print(f"No speed_analysis.parquet for {date_str}")
                continue
            speed_df = pd.read_parquet(path_speed)
            _update_cache(subjid, [date_str], {date_str: speed_df.copy()}, kind="speed_analysis")
        speed_df = speed_df.copy()

        latencies = []
        valve_latencies = []
        path_lengths = []
        travel_times = []
        tortuosities = []
        for idx_row, row in trial_data.iterrows():
            conds_base = _condition_labels_base(row)
            conds_hr = _condition_labels_hidden(row) if hidden_rule_analysis else []
            conds = conds_hr if hidden_rule_analysis else conds_base
            if not conds_base and not conds_hr:
                continue
            bins = speed_df[speed_df["trial_index"] == idx_row]
            if bins.empty:
                continue
            lat = bins["latency_s"].dropna()
            if not lat.empty:
                lat_val = float(lat.iloc[0])
                for cond in conds:
                    latencies.append({"date": date_str, "condition": cond, "latency_s": lat_val})
                for cond in conds_base:
                    combined_rows.append({"date": date_str, "condition": cond, "latency_s": lat_val})
                for cond in conds_hr:
                    combined_rows_hr.append({"date": date_str, "condition": cond, "latency_s": lat_val})

            if "movement_onset_from_valve_s" in bins.columns:
                mov = bins["movement_onset_from_valve_s"].dropna()
                if not mov.empty:
                    mov_val = float(mov.iloc[0])
                    for cond in conds:
                        valve_latencies.append({"date": date_str, "condition": cond, "movement_from_valve_s": mov_val})
                    for cond in conds_base:
                        combined_valve_rows.append({"date": date_str, "condition": cond, "movement_from_valve_s": mov_val})
                    for cond in conds_hr:
                        combined_valve_rows_hr.append({"date": date_str, "condition": cond, "movement_from_valve_s": mov_val})

            if "path_length_px" in bins.columns:
                pl = bins["path_length_px"].dropna()
                if not pl.empty:
                    pl_val = float(pl.iloc[0])
                    for cond in conds:
                        path_lengths.append({
                            "date": date_str,
                            "condition": cond,
                            "path_length_px": pl_val,
                        })
                    for cond in conds_base:
                        combined_path_rows.append({
                            "date": date_str,
                            "condition": cond,
                            "path_length_px": pl_val,
                        })
                    for cond in conds_hr:
                        combined_path_rows_hr.append({
                            "date": date_str,
                            "condition": cond,
                            "path_length_px": pl_val,
                        })
            if "travel_time_s" in bins.columns:
                tt = bins["travel_time_s"].dropna()
                if not tt.empty:
                    tt_val = float(tt.iloc[0])
                    for cond in conds:
                        travel_times.append({
                            "date": date_str,
                            "condition": cond,
                            "travel_time_s": tt_val,
                        })
                    for cond in conds_base:
                        combined_travel_rows.append({
                            "date": date_str,
                            "condition": cond,
                            "travel_time_s": tt_val,
                        })
                    for cond in conds_hr:
                        combined_travel_rows_hr.append({
                            "date": date_str,
                            "condition": cond,
                            "travel_time_s": tt_val,
                        })
            if "tortuosity" in bins.columns:
                tor = bins["tortuosity"].dropna()
                if not tor.empty:
                    tor_val = float(tor.iloc[0])
                    for cond in conds:
                        tortuosities.append({
                            "date": date_str,
                            "condition": cond,
                            "tortuosity": tor_val,
                        })
                    for cond in conds_base:
                        combined_tortuosity_rows.append({
                            "date": date_str,
                            "condition": cond,
                            "tortuosity": tor_val,
                        })
                    for cond in conds_hr:
                        combined_tortuosity_rows_hr.append({
                            "date": date_str,
                            "condition": cond,
                            "tortuosity": tor_val,
                        })

        if not any([latencies, valve_latencies, path_lengths, travel_times, tortuosities]):
            continue

        entry = {"date": date_str}

        if latencies:
            df_ses = pd.DataFrame(latencies)
            entry["data"] = df_ses

            if not multi_session:
                fig, ax = plt.subplots(figsize=figsize)
                for cond in cond_order:
                    sub = df_ses[df_ses["condition"] == cond]
                    if sub.empty or cond not in cond_positions:
                        continue
                    color = cond_colors.get(cond, "#555555")
                    y = sub["latency_s"].astype(float)
                    x0 = cond_positions[cond]
                    x_jit = x0 + (np.random.rand(len(y)) - 0.5) * jitter_span
                    ax.scatter(x_jit, y, color=color, alpha=0.6, label=f"{cond} trials")
                    mean, sem = mean_sem(y)
                    ax.errorbar(x0, mean, yerr=sem, fmt="o", color="black", capsize=4)
                ax.set_xticks([cond_positions[c] for c in cond_order])
                ax.set_xlim(_cond_xlim)
                _style_axis(ax, ylabel="Latency (s)")
                ax.set_xticklabels([_display_label(c) for c in cond_order], rotation=25, ha="right")
                fig.tight_layout()
                entry["fig"] = fig

                save_name = f"movement_stats_latency_{date_slug}"
                _save_fig(fig, save_name, date_scope)

                fig_seq = _plot_by_trial_sequence(df_ses, "latency_s", "Latency (s)")
                entry["fig_latency_by_trial"] = fig_seq
                save_name_seq = f"movement_stats_latency_sequence_{date_slug}"
                _save_fig(fig_seq, save_name_seq, date_scope)

        if valve_latencies:
            df_valve = pd.DataFrame(valve_latencies)
            entry["valve_data"] = df_valve

            if not multi_session:
                fig_v, ax_v = plt.subplots(figsize=figsize)
                for cond in cond_order:
                    sub = df_valve[df_valve["condition"] == cond]
                    if sub.empty or cond not in cond_positions:
                        continue
                    color = cond_colors.get(cond, "#555555")
                    y = sub["movement_from_valve_s"].astype(float)
                    x0 = cond_positions[cond]
                    x_jit = x0 + (np.random.rand(len(y)) - 0.5) * jitter_span
                    ax_v.scatter(x_jit, y, color=color, alpha=0.6, label=f"{cond} trials")
                    mean, sem = mean_sem(y)
                    ax_v.errorbar(x0, mean, yerr=sem, fmt="o", color="black", capsize=4)
                ax_v.set_xticks([cond_positions[c] for c in cond_order])
                ax_v.set_xlim(_cond_xlim)
                _style_axis(ax_v, ylabel="Consideration Time (s)")
                ax_v.set_xticklabels([_display_label(c) for c in cond_order], rotation=25, ha="right")
                fig_v.tight_layout()
                entry["fig_valve"] = fig_v

                save_name_valve = f"movement_stats_consideration_{date_slug}"
                _save_fig(fig_v, save_name_valve, date_scope)

                fig_valve_seq = _plot_by_trial_sequence(df_valve, "movement_from_valve_s", "Consideration Time (s)")
                entry["fig_valve_by_trial"] = fig_valve_seq
                save_name_valve_seq = f"movement_stats_consideration_sequence_{date_slug}"
                _save_fig(fig_valve_seq, save_name_valve_seq, date_scope)

        if path_lengths:
            df_path = pd.DataFrame(path_lengths)
            entry["path_data"] = df_path

            if not multi_session:
                fig_p, ax_p = plt.subplots(figsize=figsize)
                for cond in cond_order:
                    sub = df_path[df_path["condition"] == cond]
                    if sub.empty or cond not in cond_positions:
                        continue
                    color = cond_colors.get(cond, "#555555")
                    y = sub["path_length_px"].astype(float)
                    x0 = cond_positions[cond]
                    x_jit = x0 + (np.random.rand(len(y)) - 0.5) * jitter_span
                    ax_p.scatter(x_jit, y, color=color, alpha=0.6, label=f"{cond} trials")
                    mean, sem = mean_sem(y)
                    ax_p.errorbar(x0, mean, yerr=sem, fmt="o", color="black", capsize=4)
                ax_p.set_xticks([cond_positions[c] for c in cond_order])
                ax_p.set_xlim(_cond_xlim)
                _style_axis(ax_p, ylabel="Path length (px)")
                ax_p.set_xticklabels([_display_label(c) for c in cond_order], rotation=25, ha="right")
                fig_p.tight_layout()
                entry["fig_path"] = fig_p

                save_name_path = f"movement_stats_path_length_{date_slug}"
                _save_fig(fig_p, save_name_path, date_scope)

                fig_path_seq = _plot_by_trial_sequence(df_path, "path_length_px", "Path length (px)")
                entry["fig_path_by_trial"] = fig_path_seq
                save_name_path_seq = f"movement_stats_path_length_sequence_{date_slug}"
                _save_fig(fig_path_seq, save_name_path_seq, date_scope)

        if travel_times:
            df_travel = pd.DataFrame(travel_times)
            entry["travel_data"] = df_travel

            if not multi_session:
                fig_t, ax_t = plt.subplots(figsize=figsize)
                for cond in cond_order:
                    sub = df_travel[df_travel["condition"] == cond]
                    if sub.empty or cond not in cond_positions:
                        continue
                    color = cond_colors.get(cond, "#555555")
                    y = sub["travel_time_s"].astype(float)
                    x0 = cond_positions[cond]
                    x_jit = x0 + (np.random.rand(len(y)) - 0.5) * jitter_span
                    ax_t.scatter(x_jit, y, color=color, alpha=0.6, label=f"{cond} trials")
                    mean, sem = mean_sem(y)
                    ax_t.errorbar(x0, mean, yerr=sem, fmt="o", color="black", capsize=4)
                ax_t.set_xticks([cond_positions[c] for c in cond_order])
                ax_t.set_xlim(_cond_xlim)
                _style_axis(ax_t, ylabel="Duration (s)")
                ax_t.set_xticklabels([_display_label(c) for c in cond_order], rotation=25, ha="right")
                fig_t.tight_layout()
                entry["fig_travel"] = fig_t

                save_name_travel = f"movement_stats_duration_{date_slug}"
                _save_fig(fig_t, save_name_travel, date_scope)

                fig_travel_seq = _plot_by_trial_sequence(df_travel, "travel_time_s", "Duration (s)")
                entry["fig_travel_by_trial"] = fig_travel_seq
                save_name_travel_seq = f"movement_stats_duration_sequence_{date_slug}"
                _save_fig(fig_travel_seq, save_name_travel_seq, date_scope)

        if tortuosities:
            df_tort = pd.DataFrame(tortuosities)
            entry["tortuosity_data"] = df_tort

            if not multi_session:
                fig_to, ax_to = plt.subplots(figsize=figsize)
                for cond in cond_order:
                    sub = df_tort[df_tort["condition"] == cond]
                    if sub.empty or cond not in cond_positions:
                        continue
                    color = cond_colors.get(cond, "#555555")
                    y = sub["tortuosity"].astype(float)
                    x0 = cond_positions[cond]
                    x_jit = x0 + (np.random.rand(len(y)) - 0.5) * jitter_span
                    ax_to.scatter(x_jit, y, color=color, alpha=0.6, label=f"{cond} trials")
                    mean, sem = mean_sem(y)
                    ax_to.errorbar(x0, mean, yerr=sem, fmt="o", color="black", capsize=4)
                ax_to.set_xticks([cond_positions[c] for c in cond_order])
                ax_to.set_xlim(_cond_xlim)
                _style_axis(ax_to, ylabel="Tortuosity")
                ax_to.set_xticklabels([_display_label(c) for c in cond_order], rotation=25, ha="right")
                fig_to.tight_layout()
                entry["fig_tortuosity"] = fig_to
                save_name_tort = f"movement_stats_tortuosity_{date_slug}"
                _save_fig(fig_to, save_name_tort, date_scope)

                fig_tort_seq = _plot_by_trial_sequence(df_tort, "tortuosity", "Tortuosity")
                entry["fig_tortuosity_by_trial"] = fig_tort_seq
                save_name_tort_seq = f"movement_stats_tortuosity_sequence_{date_slug}"
                _save_fig(fig_tort_seq, save_name_tort_seq, date_scope)

        if not multi_session:
            per_session.append(entry)

    # Build chronological session order for combined plots (index = 0..N-1)
    raw_dates = [ses_dir.name.split("_date-")[-1] for ses_dir in ses_dirs]
    try:
        session_dates_order = sorted(set(raw_dates), key=int)
    except Exception:
        session_dates_order = sorted(set(raw_dates))
    session_index = {d: idx for idx, d in enumerate(session_dates_order)}

    def _build_session_stats(df, value_col):
        if df is None or df.empty or not session_index:
            return None
        stats = df.groupby(["condition", "date"])[value_col].agg(["mean", "sem"]).reset_index()
        stats["session_index"] = stats["date"].map(session_index)
        stats = stats.dropna(subset=["session_index"]).copy()
        return stats

    metric_frames = {
        "latency_s": combined_rows,
        "movement_from_valve_s": combined_valve_rows,
        "path_length_px": combined_path_rows,
        "travel_time_s": combined_travel_rows,
        "tortuosity": combined_tortuosity_rows,
    }

    metric_frames_hr = {
        "latency_s": combined_rows_hr,
        "movement_from_valve_s": combined_valve_rows_hr,
        "path_length_px": combined_path_rows_hr,
        "travel_time_s": combined_travel_rows_hr,
        "tortuosity": combined_tortuosity_rows_hr,
    }

    stats_by_metric = {}
    for metric, rows in metric_frames.items():
        if rows:
            stats_by_metric[metric] = _build_session_stats(pd.DataFrame(rows), metric)
        else:
            stats_by_metric[metric] = None

    stats_by_metric_hr = {}
    for metric, rows in metric_frames_hr.items():
        if rows:
            stats_by_metric_hr[metric] = _build_session_stats(pd.DataFrame(rows), metric)
        else:
            stats_by_metric_hr[metric] = None

    # Normalization factors per metric (min-max across all session means, all conditions)
    norm_factors = {}
    for metric, stats_df in stats_by_metric.items():
        if stats_df is None or stats_df.empty:
            continue
        vals = stats_df["mean"].astype(float).to_numpy()
        if vals.size == 0:
            continue
        norm_min = float(np.nanmin(vals))
        norm_max = float(np.nanmax(vals))
        norm_range = norm_max - norm_min
        norm_factors[metric] = (norm_min, norm_range)

    norm_factors_hr = {}
    for metric, stats_df in stats_by_metric_hr.items():
        if stats_df is None or stats_df.empty:
            continue
        vals = stats_df["mean"].astype(float).to_numpy()
        if vals.size == 0:
            continue
        norm_min = float(np.nanmin(vals))
        norm_max = float(np.nanmax(vals))
        norm_range = norm_max - norm_min
        norm_factors_hr[metric] = (norm_min, norm_range)

    metric_styles = {
        "latency_s": {"color": "#8BC34A", "label": "Latency (s)"},
        "movement_from_valve_s": {"color": "#FF9800", "label": "Consideration (s)"},
        "path_length_px": {"color": "#9C27B0", "label": "Path length (px)"},
        "travel_time_s": {"color": "#795548", "label": "Duration (s)"},
        "tortuosity": {"color": "#3F51B5", "label": "Tortuosity"},
    }

    def _plot_line_with_gaps(ax, x_vals, y_vals, *, color, line_width=2.0, gap_pad=0.18):
        x_arr = np.asarray(x_vals, dtype=float)
        y_arr = np.asarray(y_vals, dtype=float)
        for i in range(len(x_arr) - 1):
            x1, y1 = x_arr[i], y_arr[i]
            x2, y2 = x_arr[i + 1], y_arr[i + 1]
            if x2 <= x1:
                continue
            if (x2 - x1) > 1.0:
                pad = min(gap_pad, (x2 - x1) * 0.4)
                mid = 0.5 * (x1 + x2)
                x_left = mid - pad / 2.0
                x_right = mid + pad / 2.0
                frac_left = (x_left - x1) / (x2 - x1)
                frac_right = (x_right - x1) / (x2 - x1)
                y_left = y1 + (y2 - y1) * frac_left
                y_right = y1 + (y2 - y1) * frac_right
                ax.plot([x1, x_left], [y1, y_left], color=color, linewidth=line_width, alpha=0.9)
                ax.plot([x_right, x2], [y_right, y2], color=color, linewidth=line_width, alpha=0.9)
            else:
                ax.plot([x1, x2], [y1, y2], color=color, linewidth=line_width, alpha=0.9)

    def _plot_combined_metric(stats_df, ylabel, conds=("rewarded", "unrewarded", "fa"), colors=None):
        if stats_df is None or stats_df.empty or not session_index:
            return None
        fig, ax = plt.subplots(figsize=figsize)
        palette_map = colors or {"rewarded": "#4CAF50", "unrewarded": "#F44336", "fa": "#2196F3"}
        for cond in conds:
            color = palette_map.get(cond, "#555555")
            sub = stats_df[stats_df["condition"] == cond].sort_values("session_index")
            if sub.empty:
                continue
            x_vals = sub["session_index"].to_numpy(dtype=float)
            y_vals = sub["mean"].to_numpy(dtype=float)
            y_errs = sub["sem"].to_numpy(dtype=float)
            ax.plot(x_vals, y_vals, "o", color=color, label=f"{cond} session means", markersize=6)
            _plot_line_with_gaps(ax, x_vals, y_vals, color=color, line_width=2.0, gap_pad=0.18)
            ax.fill_between(x_vals, y_vals - y_errs, y_vals + y_errs, color=color, alpha=0.2, linewidth=0)
        ax.set_xticks(np.arange(len(session_index)))
        ax.set_xticklabels([str(i) for i in range(len(session_index))])
        ax.set_xlim(-0.5, len(session_index) - 0.5 if session_index else 0.5)
        ax.set_xlabel("Sessions")
        _style_axis(ax, ylabel=ylabel)
        ax.legend()
        fig.tight_layout()
        return fig

    def _plot_normalized_by_condition(cond, *, stats_src, norm_src):
        fig, ax = plt.subplots(figsize=figsize)
        plotted = False
        for metric, style in metric_styles.items():
            stats_df = stats_src.get(metric)
            if stats_df is None or stats_df.empty or metric not in norm_src:
                continue
            norm_min, norm_range = norm_src[metric]
            sub = stats_df[stats_df["condition"] == cond].sort_values("session_index")
            if sub.empty:
                continue
            if norm_range <= 0:
                y_vals = np.zeros(len(sub))
                y_errs = np.zeros(len(sub))
            else:
                y_vals = (sub["mean"].to_numpy(dtype=float) - norm_min) / norm_range
                y_errs = sub["sem"].to_numpy(dtype=float) / norm_range
            x_vals = sub["session_index"].to_numpy(dtype=float)
            ax.plot(x_vals, y_vals, "o", color=style["color"], label=style["label"], markersize=6)
            _plot_line_with_gaps(ax, x_vals, y_vals, color=style["color"], line_width=2.0, gap_pad=0.18)
            ax.fill_between(x_vals, y_vals - y_errs, y_vals + y_errs, color=style["color"], alpha=0.2, linewidth=0)
            plotted = True
        if not plotted:
            plt.close(fig)
            return None
        ax.set_xticks(np.arange(len(session_index)))
        ax.set_xticklabels([str(i) for i in range(len(session_index))])
        ax.set_xlim(-0.5, len(session_index) - 0.5 if session_index else 0.5)
        ax.set_xlabel("Sessions")
        _style_axis(ax, ylabel="Normalized metric (0-1)")
        ax.legend()
        fig.tight_layout()
        return fig

    if len(session_index) > 1:
        combined_fig = _plot_combined_metric(stats_by_metric.get("latency_s"), "Latency (s)")
        _save_fig(combined_fig, "movement_stats_combined_latency", dates)
        combined_valve_fig = _plot_combined_metric(stats_by_metric.get("movement_from_valve_s"), "Consideration Time (s)")
        _save_fig(combined_valve_fig, "movement_stats_combined_consideration", dates)
        combined_path_fig = _plot_combined_metric(stats_by_metric.get("path_length_px"), "Path length (px)")
        _save_fig(combined_path_fig, "movement_stats_combined_path_length", dates)
        combined_travel_fig = _plot_combined_metric(stats_by_metric.get("travel_time_s"), "Duration (s)")
        _save_fig(combined_travel_fig, "movement_stats_combined_duration", dates)
        combined_tortuosity_fig = _plot_combined_metric(stats_by_metric.get("tortuosity"), "Tortuosity")
        _save_fig(combined_tortuosity_fig, "movement_stats_combined_tortuosity", dates)

        if hidden_rule_analysis:
            combined_fig_hr = _plot_combined_metric(stats_by_metric_hr.get("latency_s"), "Latency (s)", cond_order_hr, cond_colors_hr)
            _save_fig(combined_fig_hr, "movement_stats_combined_latency_hr", dates)
            combined_valve_fig_hr = _plot_combined_metric(stats_by_metric_hr.get("movement_from_valve_s"), "Consideration Time (s)", cond_order_hr, cond_colors_hr)
            _save_fig(combined_valve_fig_hr, "movement_stats_combined_consideration_hr", dates)
            combined_path_fig_hr = _plot_combined_metric(stats_by_metric_hr.get("path_length_px"), "Path length (px)", cond_order_hr, cond_colors_hr)
            _save_fig(combined_path_fig_hr, "movement_stats_combined_path_length_hr", dates)
            combined_travel_fig_hr = _plot_combined_metric(stats_by_metric_hr.get("travel_time_s"), "Duration (s)", cond_order_hr, cond_colors_hr)
            _save_fig(combined_travel_fig_hr, "movement_stats_combined_duration_hr", dates)
            combined_tortuosity_fig_hr = _plot_combined_metric(stats_by_metric_hr.get("tortuosity"), "Tortuosity", cond_order_hr, cond_colors_hr)
            _save_fig(combined_tortuosity_fig_hr, "movement_stats_combined_tortuosity_hr", dates)
        else:
            combined_fig_hr = None
            combined_valve_fig_hr = None
            combined_path_fig_hr = None
            combined_travel_fig_hr = None
            combined_tortuosity_fig_hr = None
    else:
        combined_fig = None
        combined_valve_fig = None
        combined_path_fig = None
        combined_travel_fig = None
        combined_tortuosity_fig = None
        combined_fig_hr = None
        combined_valve_fig_hr = None
        combined_path_fig_hr = None
        combined_travel_fig_hr = None
        combined_tortuosity_fig_hr = None

    def _cond_title(c):
        if c == "rewarded":
            return "Rewarded"
        if c == "unrewarded":
            return "Unrewarded"
        if c == "fa":
            return "FA"
        return c

    combined_normalized_by_condition = {}
    combined_normalized_by_condition_hr = {}
    if len(session_index) > 1 and session_index:
        for cond in ["rewarded", "unrewarded", "fa"]:
            fig_norm = _plot_normalized_by_condition(cond, stats_src=stats_by_metric, norm_src=norm_factors)
            if fig_norm is not None:
                fig_norm.axes[0].set_title(_cond_title(cond))
                combined_normalized_by_condition[cond] = fig_norm
                save_name_norm = f"movement_stats_normalized_{_slugify(cond)}"
                _save_fig(fig_norm, save_name_norm, dates)

        if hidden_rule_analysis and cond_order_hr:
            for cond in cond_order_hr:
                fig_norm_hr = _plot_normalized_by_condition(cond, stats_src=stats_by_metric_hr, norm_src=norm_factors_hr)
                if fig_norm_hr is not None:
                    fig_norm_hr.axes[0].set_title(cond)
                    combined_normalized_by_condition_hr[cond] = fig_norm_hr
                    save_name_norm_hr = f"movement_stats_normalized_{_slugify(cond)}_hr"
                    _save_fig(fig_norm_hr, save_name_norm_hr, dates)

    # Statistical summaries across all pooled sessions/trials (by condition)
    stats_summary = {}
    stats_summary["latency_s"] = kw_mwu_by_group(pd.DataFrame(combined_rows) if combined_rows else pd.DataFrame(), "latency_s")
    stats_summary["movement_from_valve_s"] = kw_mwu_by_group(pd.DataFrame(combined_valve_rows) if combined_valve_rows else pd.DataFrame(), "movement_from_valve_s")
    stats_summary["path_length_px"] = kw_mwu_by_group(pd.DataFrame(combined_path_rows) if combined_path_rows else pd.DataFrame(), "path_length_px")
    stats_summary["travel_time_s"] = kw_mwu_by_group(pd.DataFrame(combined_travel_rows) if combined_travel_rows else pd.DataFrame(), "travel_time_s")
    stats_summary["tortuosity"] = kw_mwu_by_group(pd.DataFrame(combined_tortuosity_rows) if combined_tortuosity_rows else pd.DataFrame(), "tortuosity")

    # Print statistical summary
    print("\n" + "="*60)
    print("STATISTICAL SUMMARY (Kruskal-Wallis + Pairwise Mann-Whitney U with Holm-Bonferroni correction)")
    print("="*60)

    for variable, results in stats_summary.items():
        if results["kruskal"] is None:
            print(f"{variable}: No data")
            continue
        
        kw_p = results["kruskal"]["p"]
        print(f"\n{variable}: Kruskal-Wallis: p = {kw_p:.4f}")
        
        # Only print pairwise comparisons if KW is significant
        if kw_p < 0.05 and results["pairwise"]:
            for comparison in results["pairwise"]:
                g1 = comparison["g1"]
                g2 = comparison["g2"]
                p_corr = comparison["p_corr"]
                print(f"      {g1.capitalize()} vs {g2.capitalize()}: p = {p_corr:.4f} (corrected)")
        elif kw_p >= 0.05:
            print("      (not significant)")

    print("\n" + "="*60 + "\n")

    result = {
        "per_session": per_session,
        "combined": combined_fig,
        "combined_valve": combined_valve_fig,
        "combined_path": combined_path_fig,
        "combined_travel": combined_travel_fig,
        "combined_tortuosity": combined_tortuosity_fig,
        "combined_hr": combined_fig_hr,
        "combined_valve_hr": combined_valve_fig_hr,
        "combined_path_hr": combined_path_fig_hr,
        "combined_travel_hr": combined_travel_fig_hr,
        "combined_tortuosity_hr": combined_tortuosity_fig_hr,
        "combined_normalized_by_condition": combined_normalized_by_condition,
        "combined_normalized_by_condition_hr": combined_normalized_by_condition_hr,
        "stats": stats_summary,
    }

    if save and return_paths:
        return result, saved_paths
    return result
