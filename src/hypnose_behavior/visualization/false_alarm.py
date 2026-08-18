# Defers evaluation of PEP-604 annotations (`X | None`), keeping this module
# importable on Python 3.9 for repos pinned there (hypnose-eeg-preprocessing).
from __future__ import annotations

"""False-alarm and abortion figures.

Carved out of ``visualization_utils.py`` in restructure_2 Phase 10 (follow-up
Item 1). Source-only move -- no behaviour change.

``_fa_stat_count`` / ``_fa_stat_rate`` stay here rather than becoming leaves:
``plot_abortion_and_fa_rates`` is their only caller, and DECISIONS section 3's
rule promotes what two modules share, not what one module uses twice. Their
string branches have been unreachable since Phase 5 (DECISIONS section 5).
"""

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from hypnose_behavior.metric_analysis.metrics.false_alarm import (
    fa_port_counts,
    fa_port_ratio,
    fa_port_share_a,
    fa_rate_by_position,
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
import json
from hypnose_behavior.io.save import save_figure
from hypnose_behavior.visualization.primitives import mean_sem
from hypnose_behavior.io.loaders import (
    _load_position_data,
    _load_trial_views,
)
from hypnose_behavior.visualization.prep import _computed_metrics



def _fa_stat_count(item, key):
    """A count out of `fa_abortion_stats`.

    The legacy `"5 (0.50)"` string form is gone: Phase 4b made the metric
    numeric (the audit's finding 3), and this plotter no longer reads
    `metrics_*.json` at all, so the only shape that reaches here is the one the
    registry computes -- counts `int`, rates `float`. `DECISIONS.md` section 5.

    `bool` is excluded before the numeric test because it is a subclass of
    `int`, and `True` would otherwise read as the count 1.
    """
    val = item.get(key)
    if isinstance(val, bool) or not isinstance(val, (int, float)):
        return None
    return int(val)



def _fa_stat_rate(item, key):
    """A rate out of `fa_abortion_stats`. See `_fa_stat_count` for the shape."""
    val = item.get(key)
    if isinstance(val, bool) or not isinstance(val, (int, float)):
        return None
    return float(val)



def plot_abortion_and_fa_rates(
    subjid,
    dates=None,
    figsize=(18, 14),
    fa_types='FA_time_in',
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

    ``ses`` / ``index`` / ``date_range`` / ``ses_range`` / ``index_range`` narrow the
    selection further; they intersect with ``dates`` and with each other, and ``index``
    is the subject's gap-free chronological rank (`utils.helpers.session_selectors`).
    """
    select = session_selectors(
        ses=ses, index=index, date_range=date_range,
        ses_range=ses_range, index_range=index_range,
    )
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
        ses_dirs = _filter_session_dirs(subj_dir, dates, **select)
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

                    # The computed shape carries "Abortion Rate" on `by_position`
                    # rows; the "Abortion Rate Value" probe that used to come
                    # first went with the legacy string form (DECISIONS.md 5).
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



def plot_fa_ratio_a_over_sessions(
    subjid,
    dates=None,
    figsize=(14, 10),
    *,
    ses=None,
    index=None,
    date_range=None,
    ses_range=None,
    index_range=None,
):
    """
    Plot FA Ratio A/(A+B) over sessions for each odor (OPTIMIZED).
    
    Parameters similar to original, but now loads only necessary data.

    ``ses`` / ``index`` / ``date_range`` / ``ses_range`` / ``index_range`` narrow the
    selection further; they intersect with ``dates`` and with each other, and ``index``
    is the subject's gap-free chronological rank (`utils.helpers.session_selectors`).
    """
    select = session_selectors(
        ses=ses, index=index, date_range=date_range,
        ses_range=ses_range, index_range=index_range,
    )
    base_path = get_rawdata_root()
    server_root = get_server_root()
    derivatives_dir = get_derivatives_root()
    
    fa_data = {}  # {odor: [(session_num, ratio, n_a, n_b, n_total), ...]}
    
    for sid, subj_dir in _iter_subject_dirs(derivatives_dir, [subjid]):
        ses_dirs = _filter_session_dirs(subj_dir, dates, **select)
        
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



def plot_false_alarm_rate_by_position(
    subjids,
    dates=None,
    positions=(1, 2, 3, 4, 5),
    fa_label="FA_time_in",
    figsize=(8, 6.8),
    title=None,
    *,
    ses=None,
    index=None,
    date_range=None,
    ses_range=None,
    index_range=None,
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

    ``ses`` / ``index`` / ``date_range`` / ``ses_range`` / ``index_range`` narrow the
    selection further; they intersect with ``dates`` and with each other, and ``index``
    is the subject's gap-free chronological rank (`utils.helpers.session_selectors`).
    """
    select = session_selectors(
        ses=ses, index=index, date_range=date_range,
        ses_range=ses_range, index_range=index_range,
    )
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

        ses_dirs = _filter_session_dirs(subj_dir, subj_dates, **select)
        for ses_dir in ses_dirs:
            results_dir = ses_dir / "saved_analysis_results"
            if not results_dir.exists():
                continue
            views = _load_trial_views(results_dir)
            td = views["trial_data"]
            # Guarded on `presentations` until Phase 7b.4b, though the metric never read
            # that blob -- its denominator is `position_data`. Kept, it would have made
            # this figure silently blank once the column went.
            if td.empty:
                continue

            rates = fa_rate_by_position(td, _load_position_data(results_dir, td), fa_types=fa_set)

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



def plot_fa_ratio_by_abort_odor(
    subjid,
    dates=None,
    figsize=(18, 8),
    fa_types='FA_time_in',
    *,
    ses=None,
    index=None,
    date_range=None,
    ses_range=None,
    index_range=None,
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

    ``ses`` / ``index`` / ``date_range`` / ``ses_range`` / ``index_range`` narrow the
    selection further; they intersect with ``dates`` and with each other, and ``index``
    is the subject's gap-free chronological rank (`utils.helpers.session_selectors`).
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
    
    rows = []  # {date, odor, hr_odor, category, port_a, port_b, total, ratio}
    
    # Statistics tracking
    stats = {
        'total_no_hr': 0,
        'total_no_hr_fa': 0,
        'total_hr': 0,
        'total_hr_fa': 0
    }
    
    for sid, subj_dir in _iter_subject_dirs(derivatives_dir, [subjid]):
        ses_dirs = _filter_session_dirs(subj_dir, dates, **select)
        
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
