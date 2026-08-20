# Defers evaluation of PEP-604 annotations (`X | None`), keeping this module
# importable on Python 3.9 for repos pinned there (hypnose-eeg-preprocessing).
from __future__ import annotations

"""Choice-history figures."""

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from hypnose_behavior.io.load_results import load_session_results
from hypnose_behavior.io import layout
from hypnose_behavior.io.layout import _filter_sessions, derivatives, session_selectors
from hypnose_behavior.io.paths import (
    get_rawdata_root,
    get_derivatives_root,
    get_server_root,
)
import re
import numpy as np
from hypnose_behavior.io.save import save_figure
from hypnose_behavior.io.loaders import _load_trial_views



def plot_choice_history(
    subjid,
    dates=None,
    figsize=(16, 8),
    title=None,
    xlim=None,
    fa_types=("FA_time_in", "FA_time_out"),
    *,
    ses=None,
    index=None,
    date_range=None,
    ses_range=None,
    index_range=None,
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

    # Normalize FA filter labels
    if isinstance(fa_types, str):
        fa_set = {s.strip().lower() for s in re.split(r"[;,]", fa_types) if s.strip()}
    else:
        fa_set = {str(s).strip().lower() for s in fa_types} if fa_types is not None else set()
    
    subject_dir = derivatives.subject_dir(subjid)
    
    # Get session directories
    ses_refs = _filter_sessions(subject_dir, dates, **select)
    if not ses_refs:
        raise FileNotFoundError(f"No sessions found for subject {subjid} with given dates")
    
    # Collect all trials across sessions
    all_trials = []
    
    for session_idx, ref in enumerate(ses_refs):
        date_str = ref.date
        results_dir = layout.results_dir(ref)
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
