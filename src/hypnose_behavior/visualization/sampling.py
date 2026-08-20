# Defers evaluation of PEP-604 annotations (`X | None`), keeping this module
# importable on Python 3.9 for repos pinned there (hypnose-eeg-preprocessing).
from __future__ import annotations
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from hypnose_behavior.frames import odor_letter
from hypnose_behavior.metric_analysis.metrics.sampling import (
    poke_duration_by_odor,
    poke_duration_by_position,
    poke_durations,
)
from hypnose_behavior.io import layout
from hypnose_behavior.io.layout import (
    _filter_session_dirs,
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
    _load_trial_views,
)
from hypnose_behavior.visualization.prep import (
    _ODOR_A_COLOR,
    _build_odor_colors,
)



def plot_sampling_times_analysis(
    subjid,
    dates=None,
    figsize=(16, 18),
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
    Plot sampling times (poke durations) by position and by odor for completed and aborted trials.

    Every number drawn here comes from `metric_analysis`: `poke_durations` for the
    scattered raw values, `poke_duration_by_{position,odor}` for the mean ± SD markers
    and for the per-session series in the bottom row. **Do not extract poke durations
    here** -- that reintroduces a second definition of the same quantity.

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

    parts = []            # tidy poke durations, for the scatter panels
    pooled_positions = []  # position_data pooled over sessions, for the mean ± SD
    session_by_pos = []   # completed-trial session means, for panels 5 and 6
    session_by_odor = []

    for sid, subj_dir in _iter_subject_dirs(derivatives_dir, [subjid]):
        ses_dirs = _filter_session_dirs(subj_dir, dates, **select)
        for session_num, ses_dir in enumerate(ses_dirs, start=1):
            date_str = ses_dir.name.split("_date-")[-1]
            results_dir = layout.results_dir(ses_dir)
            if not results_dir.exists():
                continue

            td = _load_trial_views(results_dir)["trial_data"]
            position_data = _load_position_data(results_dir, td)
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



def plot_poke_duration_by_position(
    subjids,
    dates=None,
    positions=None,
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
    """Per-position poke (sampling) duration across sessions, split by trial outcome.

    Produces two separate figures ("Completed" and "Aborted"). For every session
    and every position, the mean poke duration (ms) at that position is computed
    and contributes one dot to the corresponding figure:

    - Completed trials (``is_aborted == False``): poke durations come from the
      ``in_poke_times`` rows of ``position_data``, keyed by position.
    - Aborted trials (``is_aborted == True``): from the ``in_presentations`` rows,
      excluding the abort event; keyed by position.

    (The provenance flags name which per-position source each reads; filtering on the
    one matching the facts you want is required -- ``DECISIONS.md`` section 2.)

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

    ``ses`` / ``index`` / ``date_range`` / ``ses_range`` / ``index_range`` narrow the
    selection further; they intersect with ``dates`` and with each other, and ``index``
    is the subject's gap-free chronological rank (`io.layout.session_selectors`).
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

    derivatives_dir = get_derivatives_root()

    # One row per (subject, session, position, trial_type) = session mean poke ms.
    rows = []
    for sid, subj_dir in _iter_subject_dirs(derivatives_dir, subjids):
        subj_dates = _dates_for(sid)
        if isinstance(dates, dict) and subj_dates is None:
            print(f"Warning: No date range provided in dict for subject {sid}, skipping")
            continue

        ses_dirs = _filter_session_dirs(subj_dir, subj_dates, **select)
        for ses_dir in ses_dirs:
            results_dir = layout.results_dir(ses_dir)
            if not results_dir.exists():
                continue
            position_data = _load_position_data(
                results_dir, _load_trial_views(results_dir)["trial_data"])

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



def plot_poke_duration_by_odor(
    subjid,
    date=None,
    figsize=(10, 7),
    title=None,
    *,
    ses=None,
    index=None,
    date_range=None,
    ses_range=None,
    index_range=None,
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

    ``ses`` / ``index`` / ``date_range`` / ``ses_range`` / ``index_range`` narrow the
    selection further; they intersect with ``dates`` and with each other, and ``index``
    is the subject's gap-free chronological rank (`io.layout.session_selectors`).
    """
    select = session_selectors(
        ses=ses, index=index, date_range=date_range,
        ses_range=ses_range, index_range=index_range,
    )
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
        summary_path = layout.table_path(results_dir, "summary.json")
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

    def _extract_odor_poke_ms(td, results_dir):
        """``{odor_letter: [poke_ms, ...]}`` for the requested odors, completed trials.

        Reads the canonical ``poke_durations``; **do not walk ``presentations`` with a
        ``poke_ms > 0`` filter instead**, which averages in the synthetic grace entries.
        Pooling these raw samples into the A+B / Hidden Rule / Other series below is a
        display grouping and stays here.
        """
        out: dict = {}
        pokes = poke_durations(_load_position_data(results_dir, td), aborted=False)
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
        for ses_dir in _filter_session_dirs(subj_dir, subj_date, **select):
            date_str = ses_dir.name.split("_date-")[-1]
            if not (str(date_str).isdigit() and len(str(date_str)) == 8):
                continue
            results_dir = layout.results_dir(ses_dir)
            if not results_dir.exists():
                continue
            td = _load_trial_views(results_dir)["trial_data"]
            raw = _extract_odor_poke_ms(td, results_dir)
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
