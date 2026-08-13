# Defers evaluation of PEP-604 annotations (`X | None`), keeping this module
# importable on Python 3.9 for repos pinned there (hypnose-eeg-preprocessing).
from __future__ import annotations

import sys
import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib import cm
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from collections import defaultdict
from typing import Iterable, Optional, Union, Tuple
from hypnose_behavior.io.load_results import load_session_results
from hypnose_behavior.frames import parse_json_column, odor_letter, position_entries_by_trial
from hypnose_behavior.metric_analysis.run import run_all_metrics
from datetime import timedelta, datetime
from hypnose_behavior.io.loaders import load_all_streams, load_experiment
from hypnose_behavior.io.paths import (
    get_data_root,
    get_rawdata_root,
    get_derivatives_root,
    get_server_root,
)
from hypnose_behavior.utils.helpers import (
    _filter_session_dirs,
    _filter_sessions,
    _get_from_cache,
    _iter_subject_dirs,
    _update_cache,
    session_selectors,
    find_tracking_file,
    read_tracking_table,
)
from hypnose_behavior.io.layout import derivatives, normalize_subjid
from hypnose_behavior.io.loaders import (
    _load_position_data, _load_table_with_trial_data, _load_trial_views,
)
from hypnose_behavior.visualization.prep import resample_trace, smooth_xy
from hypnose_behavior.visualization.prep import (
    load_tracking_with_behavior,
)
from hypnose_behavior.visualization.panels import (
    _clean_graph,
)
from hypnose_behavior.visualization.primitives import mean_sem
# Moved out of this file in Phase 4a: the tracking loader is io/, and
# compute_speed_analysis is a metrics module (it does no plotting at all).
from hypnose_behavior.io.tracking import _load_tracking_and_behavior
from hypnose_behavior.metric_analysis.stats.kw_mwu import kw_mwu_by_group
from hypnose_behavior.metric_analysis.movement import (
    _binned_speed,
    compute_speed_analysis,
    run_speed_analysis_batch,
    speed_threshold,
)
from hypnose_behavior.io.save import save_figure
import re
import numpy as np
import json


MOVEMENT_FIGURES_SUBDIR = "movement_figures"


def plot_movement_trace(subjid, date, smooth_window=10, linewidth=1, alpha=0.5, figsize=(10, 10), 
                       xlim=None, ylim=None, invert_y=True, title=None, save_path=None):
    """
    Plot animal movement trace from ezTrack location tracking CSV.
    
    Parameters:
    -----------
    subjid : int
        Subject ID
    date : int or str
        Session date (e.g., 20251017)
    smooth_window : int, optional
        Number of frames for moving average smoothing (default: 10)
        Set to 1 for no smoothing
    linewidth : float, optional
        Width of the trace line (default: 1)
    alpha : float, optional
        Transparency of the trace (0-1, default: 0.5)
    figsize : tuple, optional
        Figure size (width, height) (default: (10, 10))
    xlim : tuple, optional
        X-axis limits (min, max). If None, auto-scales to data
    ylim : tuple, optional
        Y-axis limits (min, max). If None, auto-scales to data
    invert_y : bool, optional
        Whether to invert Y-axis to match video coordinates (default: True)
    title : str, optional
        Plot title. If None, uses default
    save_path : str or Path, optional
        If provided, saves the plot to this path
    
    Returns:
    --------
    fig, ax : matplotlib figure and axis objects
    """
    # Build path to combined tracking CSV
    base_path = get_rawdata_root()
    server_root = get_server_root()
    derivatives_dir = get_derivatives_root()
    
    session = derivatives.find_session(subjid, date=date)
    session_dir = session.path
    date_str = session.date  # used in the figure title below

    # Find combined tracking CSV
    results_dir = session_dir / "saved_analysis_results"
    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")
    
    # Look for combined tracking file (exclude macOS metadata files)
    csv_path = find_tracking_file(results_dir, "*_combined_tracking_with_timestamps")
    if csv_path is None:
        raise FileNotFoundError(
            f"No combined tracking file found in {results_dir}\n"
            f"Run add_timestamps_to_tracking({subjid}, {date}) first to create it."
        )

    print(f"Loading tracking data from: {csv_path.name}")

    # Load the tracking data (parquet or csv, with encoding fallback)
    df = read_tracking_table(csv_path)

    # Extract X and Y coordinates
    x = df['X'].values
    y = df['Y'].values
    
    # Apply moving average smoothing
    if smooth_window > 1:
        x_smooth = pd.Series(x).rolling(window=smooth_window, center=True, min_periods=1).mean().values
        y_smooth = pd.Series(y).rolling(window=smooth_window, center=True, min_periods=1).mean().values
    else:
        x_smooth = x
        y_smooth = y
    
    # Create the plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot the trace
    ax.plot(x_smooth, y_smooth, color='black', linewidth=linewidth, alpha=alpha)
    
    # Set axis properties
    ax.set_xlabel('X Position (pixels)')
    ax.set_ylabel('Y Position (pixels)')
    ax.set_title(title if title else f'Animal Movement Trace - Subject {subjid}, {date_str}')
    
    # Set axis limits
    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)
    
    # Invert y-axis to match video coordinates (origin at top-left)
    if invert_y:
        ax.invert_yaxis()
    
    # Equal aspect ratio for proper spatial representation
    ax.set_aspect('equal', adjustable='box')
    
    # White background
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')
    
    # Add grid for reference
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Plot saved to {save_path}")
    
    plt.show()
    
    return fig, ax



def plot_movement_by_trial_state(subjid, date, smooth_window=10, linewidth=1, alpha=0.6, 
                                 figsize=(10, 10), xlim=None, ylim=None, invert_y=True,
                                 in_trial_color='blue', out_trial_color='gray',
                                 title=None, save_path=None, show=True):
    """
    Plot animal movement trace colored by trial state (in-trial vs between-trials).
    
    Parameters:
    -----------
    subjid : int
        Subject ID
    date : int or str
        Session date (e.g., 20251017)
    smooth_window : int, optional
        Number of frames for moving average smoothing (default: 10)
    linewidth : float, optional
        Width of the trace line (default: 1)
    alpha : float, optional
        Transparency of the trace (0-1, default: 0.6)
    figsize : tuple, optional
        Figure size (width, height) (default: (10, 10))
    xlim, ylim : tuple, optional
        Axis limits (min, max). If None, auto-scales
    invert_y : bool, optional
        Whether to invert Y-axis (default: True)
    in_trial_color : str, optional
        Color for in-trial segments (default: 'blue')
    out_trial_color : str, optional
        Color for between-trial segments (default: 'gray')
    title : str, optional
        Plot title
    save_path : str or Path, optional
        If provided, saves the plot
    show : bool, optional
        If True, displays the plot (default: True)
    
    Returns:
    --------
    fig, ax : matplotlib figure and axis objects
    """
    # Load data
    data = load_tracking_with_behavior(subjid, date)
    df = data['tracking_labeled']
    
    # Apply smoothing
    if smooth_window > 1:
        df['X_smooth'] = df['X'].rolling(window=smooth_window, center=True, min_periods=1).mean()
        df['Y_smooth'] = df['Y'].rolling(window=smooth_window, center=True, min_periods=1).mean()
    else:
        df['X_smooth'] = df['X']
        df['Y_smooth'] = df['Y']
    
    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot in segments based on trial state
    # This ensures continuous lines within each state
    current_state = None
    segment_x = []
    segment_y = []
    
    for idx, row in df.iterrows():
        if row['in_trial'] != current_state:
            # State changed, plot accumulated segment
            if segment_x:
                color = in_trial_color if current_state else out_trial_color
                label = 'In trial' if current_state else 'Between trials'
                # Only add label once per state
                if current_state is not None:
                    existing_labels = [t.get_label() for t in ax.lines]
                    if label in existing_labels:
                        label = None
                
                ax.plot(segment_x, segment_y, color=color, linewidth=linewidth, 
                       alpha=alpha, label=label)
            
            # Start new segment
            segment_x = [row['X_smooth']]
            segment_y = [row['Y_smooth']]
            current_state = row['in_trial']
        else:
            # Continue current segment
            segment_x.append(row['X_smooth'])
            segment_y.append(row['Y_smooth'])
    
    # Plot final segment
    if segment_x:
        color = in_trial_color if current_state else out_trial_color
        label = 'In trial' if current_state else 'Between trials'
        existing_labels = [t.get_label() for t in ax.lines]
        if label in existing_labels:
            label = None
        ax.plot(segment_x, segment_y, color=color, linewidth=linewidth, 
               alpha=alpha, label=label)
    
    # Set properties
    ax.set_xlabel('X Position (pixels)')
    ax.set_ylabel('Y Position (pixels)')
    ax.set_title(title if title else f'Movement by Trial State - Subject {subjid}, {date}')
    
    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)
    
    if invert_y:
        ax.invert_yaxis()
    
    ax.set_aspect('equal', adjustable='box')
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Plot saved to {save_path}")
    
    if show:
        plt.show()
    
    return fig, ax


def plot_movement_with_behavior(
    subjid, date,
    mode="simple",                # "simple" | "trial_state" | "last_odor" | "time_windows" | "trial_windows"
    time_windows=None,            # list of ("HH:MM:SS","HH:MM:SS")
    trial_windows=None,           # list of (start, end). negatives allowed, e.g. (-20, None) = last 20..last
    smooth_window=10, linewidth=1, alpha=0.6,
    figsize=(10, 10), xlim=None, ylim=None, invert_y=True,
    last_odor_colors=None,        # {'A':'red','B':'blue','other':'gray'}
    title=None, save_path=None, show=True
):
    """
    Minimal modes:
      - simple: baseline trace
      - trial_state: in-trial vs outside-trial
      - last_odor: within-trial colored by last odor (A vs B)
      - time_windows: plot only movement within provided clock-time windows (can be multiple)
      - trial_windows: plot only trials in provided windows; supports negatives from the end
    Also auto-creates per-condition facet plots when multiple categories/windows exist.
    """
    assert mode in {"simple", "trial_state", "by_odor", "by_odor_rew", "by_odor_outcome", "time_windows", "trial_windows", "trial_windows_rew"}
    tracking, behavior = _load_tracking_and_behavior(subjid, date)
    def _infer_last_odor_column(trials: pd.DataFrame) -> str | None:
        """
        Try to find a column that represents the last odor identity.
        Returns column name or None.
        """
        cols = set(trials.columns.str.lower())

        # Direct matches
        for cand in ["last_odor", "lastodor", "last_odor_name", "final_odor", "finalodor"]:
            for c in trials.columns:
                if c.lower() == cand:
                    return c

        # If there are odorN columns (odor1, odor2, ...), we will derive per-row later
        has_odorN = any(re.match(r"^odor\d+$", c.lower()) for c in trials.columns)
        if has_odorN:
            return None  # signal to derive from odorN columns

        # If there's an 'odors' list/JSON column
        for c in trials.columns:
            if c.lower() == "odors":
                return c

        return None
    def _last_odor_series(trials: pd.DataFrame) -> pd.Series:
        """
        Build a per-trial Series with the last odor identity.
        Handles:
        - explicit last_odor-like columns,
        - odorN columns (takes last non-null),
        - 'odors' list/JSON.
        Falls back to 'other' if not resolvable.
        """
        if trials.empty:
            return pd.Series([], dtype=object, index=trials.index)

        col = _infer_last_odor_column(trials)
        s = pd.Series(index=trials.index, dtype=object)

        if col is not None and col.lower() not in {"odors"}:
            # Direct column present
            s = trials[col].astype(object).fillna("other")
            return s

        # Try odorN columns
        odorN_cols = sorted(
            [c for c in trials.columns if re.match(r"^odor\d+$", c.lower())],
            key=lambda x: int(re.findall(r"\d+", x)[0]) if re.findall(r"\d+", x) else 0
        )
        if odorN_cols:
            def last_non_null(row):
                vals = [row[c] for c in odorN_cols if pd.notna(row[c])]
                return vals[-1] if vals else "other"
            return trials.apply(last_non_null, axis=1)

        # Try 'odors' as list/JSON
        if col is not None and col.lower() == "odors":
            def from_list(v):
                try:
                    if isinstance(v, str):
                        # maybe JSON-like
                        import json
                        v2 = json.loads(v)
                    else:
                        v2 = v
                    if isinstance(v2, (list, tuple)) and len(v2) > 0:
                        return v2[-1]
                except Exception:
                    pass
                return "other"
            return trials[col].apply(from_list)

        return s.fillna("other")
    # Smoothing
    if smooth_window > 1:
        tracking['X_smooth'] = pd.Series(tracking['X']).rolling(
            window=smooth_window, center=True, min_periods=1
        ).mean()
        tracking['Y_smooth'] = pd.Series(tracking['Y']).rolling(
            window=smooth_window, center=True, min_periods=1
        ).mean()
    else:
        tracking['X_smooth'] = tracking['X']
        tracking['Y_smooth'] = tracking['Y']

    fig, ax = plt.subplots(figsize=figsize)

    def _plot_segments_by_mask(df, mask, color, label=None, axes=None):
        # Plot continuous segments where mask is True
        target_ax = axes if axes is not None else ax
        m = mask.fillna(False).astype(bool)
        if m.sum() == 0:
            return
        seg_id = (m != m.shift(1, fill_value=False)).cumsum()
        first = True
        for _, g in df[m].groupby(seg_id[m]):
            target_ax.plot(g['X_smooth'].values, g['Y_smooth'].values,
                           color=color, linewidth=linewidth, alpha=alpha,
                           label=(label if first else None))
            first = False

    facet_plots = []  # collect per-condition masks to facet later

    if mode == "simple":
        ax.plot(tracking['X_smooth'].values, tracking['Y_smooth'].values,
                color='black', linewidth=linewidth, alpha=alpha, label='Movement trace')

    elif mode == "trial_state":
        trials = behavior.get('initiated_sequences', pd.DataFrame())
        in_trial = pd.Series(False, index=tracking.index)
        if not trials.empty:
            trials = trials.copy()
            trials['sequence_start'] = pd.to_datetime(trials['sequence_start'])
            trials['sequence_end'] = pd.to_datetime(trials['sequence_end'])
            t_time = tracking['time']
            for _, tr in trials.iterrows():
                in_trial |= ((t_time >= tr['sequence_start']) & (t_time <= tr['sequence_end']))

        colors = {'in': 'blue', 'out': 'gray'}
        _plot_segments_by_mask(tracking, in_trial, colors['in'], label='In trial')
        _plot_segments_by_mask(tracking, ~in_trial, colors['out'], label='Between trials')
        facet_plots = [
            ('In trial', in_trial, colors['in']),
            ('Between trials', ~in_trial, colors['out']),
        ]

        
    elif mode in ["by_odor", "by_odor_rew"]:
        if mode == "by_odor":
            comps = behavior.get('completed_sequences', pd.DataFrame())
        elif mode == "by_odor_rew":
            comps = behavior.get('completed_sequence_rewarded', pd.DataFrame())
        if comps.empty:
            raise ValueError("No completed_sequences found; last_odor plot requires completed trials.")
        comps = comps.copy()
        comps['sequence_start'] = pd.to_datetime(comps['sequence_start'])
        comps['sequence_end'] = pd.to_datetime(comps['sequence_end'])

        if 'last_odor' not in comps.columns:
            raise ValueError("The 'last_odor' column is missing in completed_sequences.")
        
        if last_odor_colors is None:
            last_odor_colors = {'OdorA': 'red', 'OdorB': 'blue', 'other': 'lightgray'}

        # Map each tracking frame to its odor category
        t_time = tracking['time']
        trial_odor = pd.Series('', index=tracking.index, dtype=object)
        
        for _, tr in comps.iterrows():
            mask = (t_time >= tr['sequence_start']) & (t_time <= tr['sequence_end'])
            trial_odor.loc[mask] = str(tr['last_odor'])
        
        # Filter to only frames within trials
        in_trial_mask = trial_odor != ''
        tracking_in_trial = tracking[in_trial_mask].copy()
        trial_odor_filtered = trial_odor[in_trial_mask]
        
        unique_odors = sorted(trial_odor_filtered.unique())
        
        # Plot combined view with all odors
        for odor in unique_odors:
            odor_mask = trial_odor_filtered == odor
            full_mask = pd.Series(False, index=tracking.index)
            full_mask.loc[odor_mask.index[odor_mask]] = True
            color = last_odor_colors.get(odor, last_odor_colors.get('other', 'gray'))
            _plot_segments_by_mask(tracking, full_mask, color, label=f"Odors: {odor}", axes=ax)
        
        # Create facet plots for individual odors
        facet_plots = []
        for odor in unique_odors:
            odor_mask = trial_odor_filtered == odor
            full_mask = pd.Series(False, index=tracking.index)
            full_mask.loc[odor_mask.index[odor_mask]] = True
            color = last_odor_colors.get(odor, last_odor_colors.get('other', 'gray'))
            facet_plots.append((f"{odor}", full_mask, color))

    elif mode == "by_odor_outcome":
        comps = behavior.get('completed_sequences', pd.DataFrame())
        if comps.empty:
            raise ValueError("No completed_sequences found; by_odor_outcome plot requires completed trials.")
        comps = comps.copy()
        comps['sequence_start'] = pd.to_datetime(comps['sequence_start'])
        comps['sequence_end'] = pd.to_datetime(comps['sequence_end'])

        if 'last_odor' not in comps.columns:
            raise ValueError("The 'last_odor' column is missing in completed_sequences.")

        # Try to infer the rewarded/outcome column
        rewarded_col = None
        for cand in ['rewarded', 'is_rewarded', 'outcome', 'correct', 'success']:
            if cand in comps.columns:
                rewarded_col = cand
                break
        if rewarded_col is None:
            # Fallback: if completed_sequence_rewarded exists, mark those trials as rewarded
            rewarded_trials = behavior.get('completed_sequence_rewarded', pd.DataFrame())
            if not rewarded_trials.empty and 'sequence_start' in rewarded_trials.columns:
                rewarded_starts = set(pd.to_datetime(rewarded_trials['sequence_start']))
                comps['__rewarded'] = comps['sequence_start'].isin(rewarded_starts)
                rewarded_col = '__rewarded'
            else:
                raise ValueError("No rewarded/outcome column found in completed_sequences and cannot infer from completed_sequence_rewarded.")

        if last_odor_colors is None:
            last_odor_colors = {'OdorA': 'red', 'OdorB': 'blue', 'other': 'lightgray'}

        # Map each tracking frame to its odor and outcome category
        t_time = tracking['time']
        trial_odor = pd.Series('', index=tracking.index, dtype=object)
        trial_outcome = pd.Series('', index=tracking.index, dtype=object)

        for _, tr in comps.iterrows():
            mask = (t_time >= tr['sequence_start']) & (t_time <= tr['sequence_end'])
            trial_odor.loc[mask] = str(tr['last_odor'])
            is_rewarded = bool(tr[rewarded_col])
            trial_outcome.loc[mask] = 'rewarded' if is_rewarded else 'not_rewarded'

        # Filter to only frames within trials
        in_trial_mask = trial_odor != ''
        tracking_in_trial = tracking[in_trial_mask].copy()
        trial_odor_filtered = trial_odor[in_trial_mask]
        trial_outcome_filtered = trial_outcome[in_trial_mask]

        unique_odors = sorted(trial_odor_filtered.unique())

        # Plot combined view: correct in color, incorrect/timeout in grey
        for odor in unique_odors:
            odor_mask = trial_odor_filtered == odor
            # Rewarded trials for this odor
            rewarded_mask = odor_mask & (trial_outcome_filtered == 'rewarded')
            # Not rewarded trials for this odor
            not_rewarded_mask = odor_mask & (trial_outcome_filtered == 'not_rewarded')

            full_rewarded_mask = pd.Series(False, index=tracking.index)
            full_rewarded_mask.loc[rewarded_mask.index[rewarded_mask]] = True
            full_not_rewarded_mask = pd.Series(False, index=tracking.index)
            full_not_rewarded_mask.loc[not_rewarded_mask.index[not_rewarded_mask]] = True

            color = last_odor_colors.get(odor, last_odor_colors.get('other', 'gray'))
            _plot_segments_by_mask(tracking, full_rewarded_mask, color, label=f"{odor} (correct)", axes=ax)
            _plot_segments_by_mask(tracking, full_not_rewarded_mask, 'lightgray', label=f"{odor} (incorrect/timeout)", axes=ax)

        # Create facet plots for each odor/outcome
        facet_plots = []
        for odor in unique_odors:
            odor_mask = trial_odor_filtered == odor
            rewarded_mask = odor_mask & (trial_outcome_filtered == 'rewarded')
            not_rewarded_mask = odor_mask & (trial_outcome_filtered == 'not_rewarded')

            full_rewarded_mask = pd.Series(False, index=tracking.index)
            full_rewarded_mask.loc[rewarded_mask.index[rewarded_mask]] = True
            full_not_rewarded_mask = pd.Series(False, index=tracking.index)
            full_not_rewarded_mask.loc[not_rewarded_mask.index[not_rewarded_mask]] = True

            color = last_odor_colors.get(odor, last_odor_colors.get('other', 'gray'))
            # Use a slightly darker grey for incorrect/timeout
            dark_grey = '#888888'
            facet_plots.append((
                f"{odor} by Outcome",
                [  # list of (mask, color, label)
                    (full_rewarded_mask, color, "Correct"),
                    (full_not_rewarded_mask, dark_grey, "Incorrect/Timeout")
                ]
            ))

    elif mode == "time_windows":
        if not time_windows:
            raise ValueError("time_windows must be provided for mode='time_windows'.")
        # Normalize input
        if isinstance(time_windows, tuple):
            time_windows = [time_windows]
        if isinstance(time_windows, str):
            parts = [p.strip() for p in time_windows.split(",")]
            if len(parts) != 2:
                raise ValueError("time_windows string must be 'HH:MM:SS, HH:MM:SS'")
            time_windows = [tuple(parts)]

        t = tracking['time']
        tz = t.dt.tz if hasattr(t.dt, 'tz') else None
        unique_dates = sorted(pd.to_datetime(t.dt.date).unique())
        cmap = plt.cm.Set2
        facet_plots = []
        for i, (ts, te) in enumerate(time_windows):
            color = cmap(i % 8)
            mask = pd.Series(False, index=t.index)
            for d in unique_dates:
                start_dt = pd.to_datetime(f"{pd.to_datetime(d).date()} {ts}")
                end_dt = pd.to_datetime(f"{pd.to_datetime(d).date()} {te}")
                if tz is not None:
                    if start_dt.tzinfo is None:
                        start_dt = start_dt.tz_localize(tz)
                    if end_dt.tzinfo is None:
                        end_dt = end_dt.tz_localize(tz)
                mask |= ((t >= start_dt) & (t <= end_dt))
            _plot_segments_by_mask(tracking, mask, color, label=f"Window {i+1}: {ts}-{te}")
            facet_plots.append((f"Window {i+1}: {ts}-{te}", mask, color))

    elif mode in ["trial_windows", "trial_windows_rew"]:
        if mode == "trial_windows":
            trials = behavior.get('initiated_sequences', pd.DataFrame())
        elif mode == "trial_windows_rew":
            trials = behavior.get('completed_sequence_rewarded', pd.DataFrame())
        if trials.empty:
            raise ValueError(f"{mode} requires appropriate trial data.")
        trials = trials.copy()
        trials['sequence_start'] = pd.to_datetime(trials['sequence_start'])
        trials['sequence_end'] = pd.to_datetime(trials['sequence_end'])
        # Sort by time
        trials = trials.sort_values('sequence_start').reset_index(drop=True)

        # Normalize input to list
        if isinstance(trial_windows, tuple):
            trial_windows = [trial_windows]

        n = len(trials)

        cmap = plt.cm.Dark2
        facet_plots = []
        t_time = tracking['time']
        for i, (start_idx, end_idx) in enumerate(trial_windows):
            # If end_idx is 0 or None, select to the end
            if end_idx in [0, None]:
                sel = trials.iloc[start_idx:]
            else:
                sel = trials.iloc[start_idx:end_idx]
            
            if sel.empty:
                continue
            
            mask = pd.Series(False, index=tracking.index)
            for _, tr in sel.iterrows():
                mask |= ((t_time >= tr['sequence_start']) & (t_time <= tr['sequence_end']))
            
            color = cmap(i % 8)
            # Display label using actual indices for clarity
            actual_indices = sel.index.tolist()
            label = f"Trials {actual_indices[0]}-{actual_indices[-1]}"
            _plot_segments_by_mask(tracking, mask, color, label=label)
            facet_plots.append((label, mask, color))

    # Axes/styling (overlay)
    ax.set_xlabel('X Position (pixels)')
    ax.set_ylabel('Y Position (pixels)')
    if title is None:
        title = f"Movement - Subject {subjid}, {date} ({mode})"
    ax.set_title(title)
    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)
    if invert_y:
        ax.invert_yaxis()
    ax.set_aspect('equal', adjustable='box')
    ax.legend()
    plt.tight_layout()

    # Faceted per-condition plots when multiple categories/windows exist
    if len(facet_plots) > 1:
        n = len(facet_plots)
        ncols = min(3, n)
        nrows = int(np.ceil(n / ncols))
        facet_fig, facet_axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(figsize[0]*1.2, figsize[1]*1.2))
        # Ensure facet_axes is always 2D
        if nrows == 1 and ncols == 1:
            facet_axes = np.array([[facet_axes]])
        elif nrows == 1 or ncols == 1:
            facet_axes = facet_axes.reshape(nrows, ncols)
        
        facet_axes_flat = facet_axes.flatten()
        
        for i, facet in enumerate(facet_plots):
            ax_i = facet_axes_flat[i]
            label = facet[0]
            mask_or_list = facet[1]
            # If mask_or_list is a list, it's the new format (by_odor_outcome)
            if isinstance(mask_or_list, list):
                for mask, color, sublabel in mask_or_list:
                    _plot_segments_by_mask(tracking, mask, color, label=sublabel, axes=ax_i)
            else:
                # Old format: (label, mask, color)
                mask = mask_or_list
                color = facet[2]
                _plot_segments_by_mask(tracking, mask, color, label=label, axes=ax_i)
            ax_i.set_title(label)
            ax_i.set_xlabel('X Position (px)')
            ax_i.set_ylabel('Y Position (px)')
            if xlim:
                ax_i.set_xlim(xlim)
            if ylim:
                ax_i.set_ylim(ylim)
            if invert_y:
                ax_i.invert_yaxis()
            ax_i.set_aspect('equal', adjustable='box')
        
        # Hide unused axes if any
        for j in range(len(facet_plots), len(facet_axes_flat)):
            facet_axes_flat[j].axis('off')
        
        facet_fig.suptitle(f"Per-condition views - Subject {subjid}, {date} ({mode})")
        facet_fig.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Plot saved to {save_path}")
    if show:
        plt.show()

    return fig, ax


def plot_trial_traces_by_mode(
    subjid,
    dates=None,
    mode="rewarded",
    xlim=None,
    ylim=None,
    position_units="cm",
    arena_size_cm=50.0,
    show_average=False,
    highlight_hr=False,
    color_by_index=False,
    color_by_speed=False,
    color_by_trial_id=False,
    figsize=(18, 6),
    smooth_window=5,
    fa_types="FA_time_in",
    invert_y=True,
    *,
    ses=None,
    index=None,
    date_range=None,
    ses_range=None,
    index_range=None,
    save=False,
    verbose=True,
    return_paths=False,
    show_title=True,
    show_legend=True,
):
    """
    Plot centroid traces (SLEAP) for trials filtered by mode, collapsing multiple dates into one plane.

    Parameters
    ----------
    subjid : int
        Subject ID.
    dates : list | tuple | None
        Single date, list of dates, or inclusive range tuple; all sessions are merged into one plot space.
    mode : str
        One of: rewarded, rewarded_hr, completed, all_trials, fa_by_response, fa_by_odor, hr, hr_only.
        "hr" is accepted as an alias for "hr_only".
    xlim, ylim : tuple | None
        Pixel axis limits. When position_units="cm", each limit range is mapped
        to 0-arena_size_cm on the displayed axis.
    position_units : {"cm", "px"}
        Display position coordinates in centimetres or raw pixels.
    arena_size_cm : float
        Physical size represented by the xlim/ylim ranges when displaying cm.
    show_average : bool
        If True, draw a black mean trace per category with a light-grey SEM tube.
    highlight_hr : bool
        Applies to rewarded/all_trials: recolor HR trials with HR palette; ignored elsewhere unless specified.
    color_by_index : bool
        Debug: ignore A/B colors and instead color each trace by normalized sample index (start→end) using a gradient.
    color_by_speed : bool
        If True, color each line segment by speed bins from speed_analysis.parquet (per-trial, per-bin). Segments
        with no speed data are grey. Overrides color_by_index when enabled.
    color_by_trial_id : bool
        If True (modes: rewarded, rewarded_hr, fa_by_response, fa_by_odor, hr_only), color by normalized
        trial order per reward port (A/B) using a dark→light blue gradient. Overrides color_by_index/speed.
    figsize : tuple
        Figure size.
    smooth_window : int
        Rolling window for centroid smoothing (frames).
    fa_types : str
        Comma-separated FA labels to include (e.g., "FA_time_in" or "FA_time_in,FA_time_out" or "all").
    invert_y : bool
        If True, invert Y-axis to match video coordinates.
    save : bool
        When True, persist each generated figure to the movement_figures subdirectory.
    verbose : bool
        Print save status messages when saving figures.
    return_paths : bool
        If True and save=True, also return the saved figure paths alongside the figure handles.

    ``ses`` / ``index`` / ``date_range`` / ``ses_range`` / ``index_range`` narrow the
    selection further; they intersect with ``dates`` and with each other, and ``index``
    is the subject's gap-free chronological rank (`utils.helpers.session_selectors`).
    """
    select = session_selectors(
        ses=ses, index=index, date_range=date_range,
        ses_range=ses_range, index_range=index_range,
    )

    allowed_modes = {
        "rewarded",
        "rewarded_hr",
        "completed",
        "all_trials",
        "fa_by_response",
        "fa_by_odor",
        "hr",
        "hr_only",
    }
    if mode not in allowed_modes:
        raise ValueError(f"mode must be one of {sorted(allowed_modes)}")
    if mode == "hr":
        mode = "hr_only"
    if position_units not in {"cm", "px"}:
        raise ValueError("position_units must be either 'cm' or 'px'")
    if position_units == "cm" and arena_size_cm <= 0:
        raise ValueError("arena_size_cm must be positive")

    # FA filter
    if isinstance(fa_types, str):
        fa_types_list = [t.strip().lower() for t in fa_types.split(",")]
        if fa_types.lower() == "all":
            def fa_filter_fn(lbl):
                return str(lbl).startswith("FA_") if pd.notna(lbl) else False
        else:
            def fa_filter_fn(lbl):
                return str(lbl).lower() in fa_types_list if pd.notna(lbl) else False
    elif isinstance(fa_types, (list, tuple, set)):
        fa_set = {str(t).lower() for t in fa_types}
        def fa_filter_fn(lbl):
            return str(lbl).lower() in fa_set if pd.notna(lbl) else False
    else:
        def fa_filter_fn(lbl):
            return True

    # Colors
    port_colors = {1: "#FF6B6B", 2: "#4ECDC4"}
    port_colors_hr = {1: "#E53935", 2: "#00796B"}
    port_colors_fa = {1: "#FF8E8E", 2: "#7EE9DF"}  # slightly altered
    aborted_color = "#555555"
    timeout_color = "#9E9E9E"
    unrewarded_color = "#000000"
    index_cmap = cm.get_cmap("plasma")
    index_norm = Normalize(vmin=0.0, vmax=1.0)
    trial_cmap = cm.get_cmap("Blues")
    speed_cmap = cm.get_cmap("viridis")
    speed_vals_global = []

    subj_str = normalize_subjid(subjid)
    derivatives_dir = get_derivatives_root()
    subj_dir = derivatives.subject_dir(subjid)

    ses_dirs = _filter_session_dirs(subj_dir, dates, **select)
    if not ses_dirs:
        raise FileNotFoundError(f"No sessions found for subject {subjid} with given dates")

    def _odor_letter(val):
        """Canonical odor-token normaliser, plus this figure's label for a
        missing odor. Measured over every odor value in 15 sessions, the two
        agree on all of them except NaN -- and "Unknown" vs "NAN" only ever
        reaches a label, never a branch. The relabelling stays here rather than
        moving into `metric_analysis` (audit finding 14)."""
        return "Unknown" if pd.isna(val) else odor_letter(val)

    def _infer_port_from_response(row):
        for col in [
            "response_port",
            "rewarded_port",
            "reward_port",
            "supply_port",
            "choice_port",
            "port",
            "fa_port",
        ]:
            if col in row and pd.notna(row[col]):
                try:
                    return int(row[col])
                except Exception:
                    try:
                        return int(float(row[col]))
                    except Exception:
                        continue
        return None

    def _hr_port_from_identity(val):
        if pd.isna(val):
            return None
        s = str(val).strip().upper()
        if s in {"A", "ODORA", "1"}:
            return 1
        if s in {"B", "ODORB", "2"}:
            return 2
        return None

    def _port_from_first_supply(row):
        return _hr_port_from_identity(row.get("first_supply_odor_identity"))

    def _category_from_row(row):
        # Priority: explicit first_supply_odor_identity -> inferred port -> odor letter fallback
        port = _port_from_first_supply(row)
        if port is None:
            port = _infer_port_from_response(row)
        if port in {1, 2}:
            return ("A" if port == 1 else "B"), port
        odor = _odor_letter(row.get("last_odor_name") or row.get("last_odor"))
        category = "A" if odor in {"A", "OdorA"} else "B"
        return category, port

    def _extract_segment(tracking_df, start, end):
        if pd.isna(start) or pd.isna(end):
            return None
        m = (tracking_df["time"] >= start) & (tracking_df["time"] <= end)
        if not m.any():
            return None
        seg = tracking_df.loc[m, ["time", "X", "Y"]]
        if seg.empty:
            return None
        return seg["X"].to_numpy(), seg["Y"].to_numpy(), seg["time"].to_numpy()

    def _last_poke_out_by_position(row, entries):
        """The **last entry by position**, null accepted -- not a scan back.

        `entries` is this trial's `position_data` rows sorted by position (Phase
        7b.4b; it used to parse the `position_poke_times` blob off `row`). The rule
        is deliberately unchanged: when there are entries, the last one's
        `poke_odor_end` is the answer *even when it is null*, and only a trial with
        no entries at all falls through to the row-level columns.
        """
        if entries:
            return pd.to_datetime(entries[-1].get("poke_odor_end"), errors="coerce")
        for cand in ["poke_odor_end", "last_poke_out_time", "last_poke_time"]:
            if cand in row:
                return pd.to_datetime(row.get(cand), errors="coerce")
        if "sequence_start" in row:
            return pd.to_datetime(row.get("sequence_start"), errors="coerce")
        return pd.NaT

    def _add_segment(store, axis_key, category, color, x, y, *, label=None, time=None, t_zero=None, speed_bins=None):
        store[axis_key].append({
            "category": category,
            "color": color,
            "x": x,
            "y": y,
            "time": time,
            "t_zero": t_zero,
            "speed_bins": speed_bins,
            "label": label if label is not None else category,
        })

    # Containers
    segments = defaultdict(list)
    avg_pool = defaultdict(lambda: defaultdict(list))
    hr_odors_seen = set()
    speed_analysis_cache = {}

    def _compute_trial_color_map(df, port_fn):
        """Map (port, global_trial_id) -> RGBA using a dark→light blue gradient per port."""
        per_port_ids = defaultdict(list)
        for _, r in df.iterrows():
            tid = r.get("global_trial_id")
            try:
                tid = int(tid)
            except Exception:
                tid = None
            p = port_fn(r)
            if p in {1, 2} and tid is not None:
                per_port_ids[p].append(tid)

        color_map: dict[tuple[int, int], tuple] = {}
        for p, ids in per_port_ids.items():
            ids_sorted = sorted(ids)
            n = len(ids_sorted)
            for i, tid in enumerate(ids_sorted):
                frac = i / (n - 1) if n > 1 else 0.5
                color_map[(p, tid)] = trial_cmap(frac)
        return color_map

    for ses_dir in ses_dirs:
        date_str = ses_dir.name.split("_date-")[-1]
        results_dir = ses_dir / "saved_analysis_results"
        if not results_dir.exists():
            continue

        # Load tracking/behavior
        tracking, behavior = _load_tracking_and_behavior(subjid, date_str)
        tracking = tracking.copy()
        tracking["time"] = pd.to_datetime(tracking["time"], errors="coerce")
        tracking = tracking.dropna(subset=["time"]).reset_index(drop=True)
        tracking = tracking.rename(columns={"centroid_x": "X", "centroid_y": "Y"})

        # Resolve possible duplicate X/Y columns (e.g., both X and centroid_x) to a single Series
        def _resolve_coord(df, candidates):
            for name in candidates:
                if name in df.columns:
                    col = df.loc[:, df.columns == name]
                    if isinstance(col, pd.DataFrame):
                        if col.shape[1] == 0:
                            continue
                        return col.iloc[:, 0]
                    return df[name]
            return None

        x_series = _resolve_coord(tracking, ["X", "centroid_x", "x"])
        y_series = _resolve_coord(tracking, ["Y", "centroid_y", "y"])
        if x_series is not None:
            tracking["X"] = x_series
        if y_series is not None:
            tracking["Y"] = y_series

        tracking = tracking.dropna(subset=["X", "Y"])
        # Drop duplicate columns to avoid DataFrame returns when selecting by name
        tracking = tracking.loc[:, ~tracking.columns.duplicated()]
        tracking = smooth_xy(tracking, smooth_window)

        # Load speed analysis parquet if available (per-bin speeds + threshold times)
        speed_df = _get_from_cache(subjid, date_str, kind="speed_analysis")
        if speed_df is None:
            path_speed = results_dir / "speed_analysis.parquet"
            if path_speed.exists():
                try:
                    speed_df = pd.read_parquet(path_speed)
                    _update_cache(subjid, [date_str], {date_str: speed_df.copy()}, kind="speed_analysis")
                except Exception as e:
                    print(f"Warning: could not read {path_speed.name}: {e}")
        speed_bins_map = {}
        if speed_df is not None and not speed_df.empty:
            if "trial_index" in speed_df.columns:
                speed_df = speed_df.copy()
                for col in ["speed_threshold_time", "bin_mid_time", "bin_start_time", "bin_end_time"]:
                    if col in speed_df.columns:
                        speed_df[col] = pd.to_datetime(speed_df[col], errors="coerce")
                for tidx, group in speed_df.groupby("trial_index"):
                    speed_bins_map[tidx] = group.copy()
                finite_speeds = speed_df["speed"].to_numpy()
                speed_vals_global.extend([v for v in finite_speeds if np.isfinite(v)])
        speed_analysis_cache[date_str] = speed_bins_map

        views = _load_trial_views(results_dir)
        td = views.get("trial_data", pd.DataFrame()).copy()
        if not td.empty:
            for c in ["sequence_start", "sequence_end"]:
                if c in td.columns:
                    td[c] = pd.to_datetime(td[c], errors="coerce")
        else:
            td = pd.DataFrame()
        # `in_poke_times` is the flag matching `position_poke_times`, the blob this read
        # before Phase 7b.4b (`DECISIONS.md` section 2).
        pokes_by_trial = position_entries_by_trial(
            _load_position_data(results_dir, td), "in_poke_times")

        if td.empty:
            # Fallback to behavior tables if trial_data is unavailable
            comp = behavior.get("completed_sequences", pd.DataFrame()).copy()
            if not comp.empty:
                comp["is_aborted"] = False
            aborted = behavior.get("aborted_sequences", pd.DataFrame()).copy()
            if not aborted.empty:
                aborted["is_aborted"] = True
            td = pd.concat([comp, aborted], ignore_index=True) if not comp.empty or not aborted.empty else pd.DataFrame()
            if not td.empty:
                for c in ["sequence_start", "sequence_end"]:
                    if c in td.columns:
                        td[c] = pd.to_datetime(td[c], errors="coerce")

        if td.empty:
            continue

        hr_flag = "hidden_rule_success" if "hidden_rule_success" in td.columns else ("hit_hidden_rule" if "hit_hidden_rule" in td.columns else None)
        hr_mask = td[hr_flag] == True if hr_flag else pd.Series(False, index=td.index)

        # Helper to iterate trials
        def iter_trials(df):
            for idx_row, row in df.iterrows():
                start = row.get("sequence_start")
                # For false alarms, use fa_time as end; for rewarded, prefer first_supply_time to cap at reward delivery
                fa_label = str(row.get("fa_label", "")).lower()
                fa_time = row.get("fa_time")
                resp_cat = str(row.get("response_time_category", "")).lower()
                first_supply_time = row.get("first_supply_time")
                first_reward_poke_time = row.get("first_reward_poke_time")

                if pd.notna(fa_time) and fa_label.startswith("fa_"):
                    end = fa_time
                elif resp_cat == "rewarded" and pd.notna(first_supply_time):
                    end = first_supply_time
                elif resp_cat == "unrewarded" and pd.notna(first_reward_poke_time):
                    end = first_reward_poke_time
                else:
                    end = row.get("sequence_end")
                seg = _extract_segment(tracking, start, end)
                if seg is None:
                    continue
                t_zero = _last_poke_out_by_position(
                    row, pokes_by_trial.get(row.get("global_trial_id")))
                speed_bins = speed_analysis_cache.get(date_str, {}).get(idx_row)
                yield idx_row, row, seg, t_zero, speed_bins

        # Mode-specific selection
        if mode in {"rewarded", "rewarded_hr"}:
            trials = td[(td.get("response_time_category") == "rewarded") & (td.get("is_aborted") == False)]
            include_hr = (mode == "rewarded_hr") or highlight_hr
            if hr_flag and not include_hr:
                trials = trials[~hr_mask]

            trial_color_map = {}
            if color_by_trial_id:
                def _port_trial(row):
                    p = _port_from_first_supply(row)
                    if p is None:
                        p = _infer_port_from_response(row)
                    return p
                trial_color_map = _compute_trial_color_map(trials, _port_trial)

            for idx_row, row, seg, t_zero, speed_bins in iter_trials(trials):
                port = None
                if hr_flag and bool(row.get(hr_flag, False)):
                    port = _port_from_first_supply(row) or _infer_port_from_response(row)
                category, port_fallback = _category_from_row(row)
                if port is None:
                    port = port_fallback
                color_map = port_colors_hr if (highlight_hr and hr_flag and bool(row.get(hr_flag, False))) else port_colors
                color = color_map.get(port, port_colors[1 if category == "A" else 2])
                if color_by_trial_id:
                    tid = row.get("global_trial_id")
                    try:
                        tid = int(tid)
                    except Exception:
                        tid = None
                    if port in {1, 2} and tid is not None:
                        color = trial_color_map.get((port, tid), color)
                _add_segment(segments, "combined", category, color, seg[0], seg[1], time=seg[2], t_zero=t_zero, speed_bins=speed_bins)
                _add_segment(segments, category, category, color, seg[0], seg[1], time=seg[2], t_zero=t_zero, speed_bins=speed_bins)
                resampled = resample_trace(seg[0], seg[1])
                if resampled is not None:
                    avg_pool["combined"][category].append(resampled)
                    avg_pool[category][category].append(resampled)

        elif mode == "completed":
            trials = td[td.get("is_aborted") == False]
            for idx_row, row, seg, t_zero, speed_bins in iter_trials(trials):
                category, port = _category_from_row(row)
                rtc = str(row.get("response_time_category", "")).lower()
                if rtc == "rewarded":
                    color = port_colors.get(port, port_colors[1 if category == "A" else 2])
                elif rtc == "timeout_delayed":
                    color = timeout_color
                elif rtc == "unrewarded":
                    color = unrewarded_color
                else:
                    color = timeout_color
                _add_segment(segments, "combined", category, color, seg[0], seg[1], time=seg[2], t_zero=t_zero, speed_bins=speed_bins)
                _add_segment(segments, category, category, color, seg[0], seg[1], time=seg[2], t_zero=t_zero, speed_bins=speed_bins)
                resampled = resample_trace(seg[0], seg[1])
                if resampled is not None:
                    avg_pool["combined"][category].append(resampled)
                    avg_pool[category][category].append(resampled)

        elif mode == "all_trials":
            trials = td.copy()
            for idx_row, row, seg, t_zero, speed_bins in iter_trials(trials):
                category, port = _category_from_row(row)
                if row.get("is_aborted"):
                    color = aborted_color
                    if highlight_hr and hr_flag and bool(row.get(hr_flag, False)):
                        color = "#000000"
                else:
                    color_map = port_colors_hr if (highlight_hr and hr_flag and bool(row.get(hr_flag, False))) else port_colors
                    color = color_map.get(port, port_colors[1 if category == "A" else 2])
                _add_segment(segments, "combined", category, color, seg[0], seg[1], time=seg[2], t_zero=t_zero, speed_bins=speed_bins)
                # Only include completed trials in the averages for all_trials
                if not row.get("is_aborted"):
                    resampled = resample_trace(seg[0], seg[1])
                    if resampled is not None:
                        avg_pool["combined"][category].append(resampled)

        elif mode == "fa_by_response":
            # Only aborted FA trials, filtered by fa_types; plot sequence_start -> fa_time
            fa_df = td[(td.get("is_aborted") == True) & (td.get("fa_label").notna())].copy()
            if not fa_df.empty:
                fa_df = fa_df[fa_df["fa_label"].apply(fa_filter_fn)]
                # Require fa_time for window end
                if "fa_time" in fa_df.columns:
                    fa_df["fa_time"] = pd.to_datetime(fa_df["fa_time"], errors="coerce")
                    fa_df = fa_df.dropna(subset=["fa_time"])
            if not fa_df.empty:
                label_counts = fa_df["fa_label"].value_counts().to_dict()
                print(f"[fa_by_response] session {date_str}: trials after filter={len(fa_df)}, fa_label counts={label_counts}")
            if fa_df.empty:
                continue

            trial_color_map = {}
            if color_by_trial_id:
                def _port_trial(row):
                    p = row.get("fa_port") if pd.notna(row.get("fa_port")) else _infer_port_from_response(row)
                    return p
                trial_color_map = _compute_trial_color_map(fa_df, _port_trial)

            for idx_row, row, seg, t_zero, speed_bins in iter_trials(fa_df):
                # Use FA port first, then supply/response port
                port = row.get("fa_port") if pd.notna(row.get("fa_port")) else _infer_port_from_response(row)
                if port not in {1, 2}:
                    continue
                category = "A" if port == 1 else "B"
                color = port_colors_fa.get(port, port_colors_fa[1])
                if color_by_trial_id:
                    tid = row.get("global_trial_id")
                    try:
                        tid = int(tid)
                    except Exception:
                        tid = None
                    if tid is not None:
                        color = trial_color_map.get((port, tid), color)
                _add_segment(segments, "combined", category, color, seg[0], seg[1], time=seg[2], t_zero=t_zero, speed_bins=speed_bins)
                _add_segment(segments, category, category, color, seg[0], seg[1], time=seg[2], t_zero=t_zero, speed_bins=speed_bins)
                resampled = resample_trace(seg[0], seg[1])
                if resampled is not None:
                    avg_pool["combined"][category].append(resampled)
                    avg_pool[category][category].append(resampled)

        elif mode == "fa_by_odor":
            # Only aborted FA trials; require fa_time for window end
            fa_df = td[(td.get("is_aborted") == True) & (td.get("fa_label").notna())].copy()
            if not fa_df.empty:
                fa_df = fa_df[fa_df["fa_label"].apply(fa_filter_fn)]
                if "fa_time" in fa_df.columns:
                    fa_df["fa_time"] = pd.to_datetime(fa_df["fa_time"], errors="coerce")
                    fa_df = fa_df.dropna(subset=["fa_time"])
            if fa_df.empty:
                continue

            trial_color_map = {}
            if color_by_trial_id:
                def _port_trial(row):
                    p = row.get("fa_port") if pd.notna(row.get("fa_port")) else _infer_port_from_response(row)
                    return p
                trial_color_map = _compute_trial_color_map(fa_df, _port_trial)

            for idx_row, row, seg, t_zero, speed_bins in iter_trials(fa_df):
                odor_name = row.get("last_odor_name") or row.get("last_odor")
                odor = _odor_letter(odor_name)
                if odor in {"A", "B", "OdorA", "OdorB"}:
                    continue
                port = row.get("fa_port") if pd.notna(row.get("fa_port")) else _infer_port_from_response(row)
                color = port_colors_fa.get(port, port_colors_fa[1])
                if color_by_trial_id:
                    tid = row.get("global_trial_id")
                    try:
                        tid = int(tid)
                    except Exception:
                        tid = None
                    if port in {1, 2} and tid is not None:
                        color = trial_color_map.get((port, tid), color)
                label = "FA to A" if port == 1 else ("FA to B" if port == 2 else "FA")
                _add_segment(segments, odor, label, color, seg[0], seg[1], time=seg[2], t_zero=t_zero, speed_bins=speed_bins)
                resampled = resample_trace(seg[0], seg[1])
                if resampled is not None:
                    avg_pool[odor][label].append(resampled)

        elif mode == "hr_only":
            if hr_flag is None:
                continue
            # Determine hidden-rule odors for this session (from summary.json)
            hr_odors_raw = []
            summary_path = results_dir / "summary.json"
            if summary_path.exists():
                try:
                    with open(summary_path, "r", encoding="utf-8") as f:
                        summary = json.load(f)
                    hr_odors_raw = summary.get("params", {}).get("hidden_rule_odors", []) or []
                    if not hr_odors_raw:
                        runs = summary.get("session", {}).get("runs", [])
                        if runs and isinstance(runs[0], dict):
                            stage = runs[0].get("stage", {}) if isinstance(runs[0].get("stage", {}), dict) else {}
                            hr_odors_raw = stage.get("hidden_rule_odors", []) or stage.get("hidden_rule_odors".lower(), []) or []
                except Exception:
                    hr_odors_raw = []
            hr_targets = [_odor_letter(o) for o in hr_odors_raw if o is not None]
            hr_targets = hr_targets[:2]  # only need first two hidden-rule odors

            def _parse_odor_sequence(val):
                if isinstance(val, list):
                    return val
                if pd.isna(val):
                    return []
                if isinstance(val, str):
                    try:
                        obj = json.loads(val)
                        if isinstance(obj, list):
                            return obj
                    except Exception:
                        pass
                    # fallback: split by comma/semicolon
                    return [s.strip().strip("[]'\"") for s in re.split(r"[;,]", val) if s.strip()]
                return []

            hr_trials = td[(hr_mask) & (td.get("is_aborted") == False)]
            if hr_trials.empty:
                continue

            trial_color_map = {}
            if color_by_trial_id:
                def _port_trial(row):
                    p = _hr_port_from_identity(row.get("first_supply_odor_identity"))
                    if p is None:
                        p = _infer_port_from_response(row)
                    return p
                trial_color_map = _compute_trial_color_map(hr_trials, _port_trial)
            for idx_row, row, seg, t_zero, speed_bins in iter_trials(hr_trials):
                odor_seq = _parse_odor_sequence(row.get("odor_sequence"))
                odor_match = None
                for o in odor_seq:
                    ol = _odor_letter(o)
                    if hr_targets and ol in hr_targets:
                        odor_match = ol
                        break
                if odor_match is None:
                    # fallback to last_odor if no sequence match
                    odor_match = _odor_letter(row.get("last_odor_name") or row.get("last_odor"))
                hr_odors_seen.add(odor_match)

                port = _hr_port_from_identity(row.get("first_supply_odor_identity"))
                if port is None:
                    port = _infer_port_from_response(row)
                rtc = str(row.get("response_time_category", "")).lower()

                axis_key = f"HR {odor_match}"
                label_base = f"{odor_match}"
                if rtc == "rewarded":
                    color = port_colors_hr.get(port, port_colors_hr[1])
                    if color_by_trial_id:
                        tid = row.get("global_trial_id")
                        try:
                            tid = int(tid)
                        except Exception:
                            tid = None
                        if port in {1, 2} and tid is not None:
                            color = trial_color_map.get((port, tid), color)
                    _add_segment(segments, axis_key, f"{label_base} rewarded", color, seg[0], seg[1], time=seg[2], t_zero=t_zero, speed_bins=speed_bins)
                    _add_segment(segments, "HR Summary", f"{label_base} rewarded", color, seg[0], seg[1], time=seg[2], t_zero=t_zero, speed_bins=speed_bins)
                    resampled = resample_trace(seg[0], seg[1])
                    if resampled is not None:
                        avg_pool[axis_key][f"{label_base} rewarded"].append(resampled)
                        avg_pool["HR Summary"][f"{label_base} rewarded"].append(resampled)
                elif rtc == "timeout_delayed":
                    color = timeout_color
                    _add_segment(segments, axis_key, f"{label_base} timeout", color, seg[0], seg[1], time=seg[2], t_zero=t_zero, speed_bins=speed_bins)
                    _add_segment(segments, "HR Summary", f"{label_base} timeout", color, seg[0], seg[1], time=seg[2], t_zero=t_zero, speed_bins=speed_bins)
                    resampled = resample_trace(seg[0], seg[1])
                    if resampled is not None:
                        avg_pool[axis_key][f"{label_base} timeout"].append(resampled)
                        avg_pool["HR Summary"][f"{label_base} timeout"].append(resampled)
                else:  # unrewarded / other
                    color = unrewarded_color
                    _add_segment(segments, axis_key, f"{label_base} unrewarded", color, seg[0], seg[1], time=seg[2], t_zero=t_zero, speed_bins=speed_bins)
                    _add_segment(segments, "HR Summary", f"{label_base} unrewarded", color, seg[0], seg[1], time=seg[2], t_zero=t_zero, speed_bins=speed_bins)
                    resampled = resample_trace(seg[0], seg[1])
                    if resampled is not None:
                        avg_pool[axis_key][f"{label_base} unrewarded"].append(resampled)

    if not segments:
        print("No matching trials found for the requested mode.")
        return None, None

    speed_norm = None
    if color_by_speed and speed_vals_global and not color_by_trial_id:
        vmin = np.nanmin(speed_vals_global)
        vmax = np.nanmax(speed_vals_global)
        if np.isfinite(vmin) and np.isfinite(vmax) and vmax > vmin:
            speed_norm = Normalize(vmin=vmin, vmax=vmax)
    color_by_speed_active = color_by_speed and (speed_norm is not None) and (not color_by_trial_id)
    if color_by_trial_id:
        color_by_index = False

    def _coord_limits(axis):
        vals = []
        for segs in segments.values():
            for seg in segs:
                arr = np.asarray(seg[axis], dtype=float)
                vals.extend(arr[np.isfinite(arr)])
        if not vals:
            return None
        return float(np.nanmin(vals)), float(np.nanmax(vals))

    def _normalize_limits(limits, axis):
        if limits is None:
            limits = _coord_limits(axis)
        if limits is None:
            return None
        lo, hi = limits
        return float(lo), float(hi)

    x_source_limits = _normalize_limits(xlim, "x")
    y_source_limits = _normalize_limits(ylim, "y")
    x_display_lim = x_source_limits
    y_display_lim = y_source_limits

    def _scale_axis(values, limits):
        values = np.asarray(values, dtype=float)
        if position_units == "px":
            return values
        lo, hi = limits
        span = hi - lo
        if span == 0:
            return np.full_like(values, np.nan, dtype=float)
        scaled = (values - lo) / span * arena_size_cm
        return np.clip(scaled, 0.0, arena_size_cm)

    def _x_display(values):
        return _scale_axis(values, x_source_limits)

    def _y_display(values):
        return _scale_axis(values, y_source_limits)

    if position_units == "cm":
        if x_source_limits is None or y_source_limits is None:
            raise ValueError("Cannot convert to cm without finite x/y coordinate limits")
        x_display_lim = tuple(float(v) for v in _x_display(x_source_limits))
        y_display_lim = tuple(float(v) for v in _y_display(y_source_limits))
        if verbose:
            print(
                "[plot_trial_traces_by_mode] position mapping: "
                f"x px {x_source_limits} -> cm {x_display_lim}; "
                f"y px {y_source_limits} -> cm {y_display_lim}"
            )

    def _plot_axis(ax, axis_key):
        segs = segments.get(axis_key, [])
        used_labels = set()

        def _plot_segment(seg, label=None):
            x = _x_display(seg["x"])
            y = _y_display(seg["y"])
            if color_by_speed_active:
                t_arr = np.asarray(seg.get("time"))
                bins_df = seg.get("speed_bins")
                if t_arr is None or bins_df is None or len(x) < 2:
                    ax.plot(x, y, color="#B0B0B0", label=label)
                    return
                try:
                    seg_mid_times = t_arr[:-1] + (t_arr[1:] - t_arr[:-1]) / 2
                except Exception:
                    ax.plot(x, y, color="#B0B0B0", label=label)
                    return
                seg_arr = np.stack([np.column_stack([x[:-1], y[:-1]]), np.column_stack([x[1:], y[1:]])], axis=1)
                colors = []
                bins_df = bins_df.sort_values("bin_start_s") if not bins_df.empty else bins_df
                for t_mid in seg_mid_times:
                    if bins_df is None or bins_df.empty:
                        colors.append("#B0B0B0")
                        continue
                    hit = bins_df[(bins_df["bin_start_time"] <= t_mid) & (t_mid < bins_df["bin_end_time"])]
                    if hit.empty:
                        colors.append("#B0B0B0")
                        continue
                    spd = float(hit.iloc[0]["speed"])
                    if np.isfinite(spd):
                        colors.append(speed_cmap(speed_norm(spd)))
                    else:
                        colors.append("#B0B0B0")
                lc = LineCollection(seg_arr, colors=colors)
                ax.add_collection(lc)
            elif color_by_index:
                if x.size < 2 or y.size < 2:
                    return
                points = np.array([x, y]).T.reshape(-1, 1, 2)
                if points.shape[0] < 2:
                    return
                seg_arr = np.concatenate([points[:-1], points[1:]], axis=1)
                idx_vals = np.linspace(0, 1, len(seg_arr))
                lc = LineCollection(seg_arr, cmap=index_cmap, norm=index_norm)
                lc.set_array(idx_vals)
                ax.add_collection(lc)
            else:
                ax.plot(x, y, color=seg["color"], label=label)

        for seg in segs:
            label = None
            if (not color_by_index) and (not color_by_speed_active) and (seg["label"] not in used_labels):
                label = seg["label"]
                used_labels.add(seg["label"])
            _plot_segment(seg, label)
        if show_average and axis_key in avg_pool:
            for category, traces in avg_pool[axis_key].items():
                if not traces:
                    continue
                xs = [t[0] for t in traces if t is not None]
                ys = [t[1] for t in traces if t is not None]
                if not xs or not ys:
                    continue
                xs = np.vstack([_x_display(x) for x in xs])
                ys = np.vstack([_y_display(y) for y in ys])
                mean_x = np.nanmean(xs, axis=0)
                mean_y = np.nanmean(ys, axis=0)
                sem_x = np.nanstd(xs, axis=0) / np.sqrt(xs.shape[0])
                sem_y = np.nanstd(ys, axis=0) / np.sqrt(ys.shape[0])
                sem_r = np.sqrt(np.square(sem_x) + np.square(sem_y))

                dx = np.gradient(mean_x)
                dy = np.gradient(mean_y)
                norm = np.hypot(dx, dy)
                norm[norm == 0] = 1.0
                nx = -dy / norm
                ny = dx / norm

                poly_x = np.concatenate([mean_x + nx * sem_r, (mean_x - nx * sem_r)[::-1]])
                poly_y = np.concatenate([mean_y + ny * sem_r, (mean_y - ny * sem_r)[::-1]])

                ax.fill(poly_x, poly_y, color="#DDDDDD", alpha=0.35)
                ax.plot(mean_x, mean_y, color="black", label=f"{category} mean")

        if color_by_speed_active:
            sm = cm.ScalarMappable(norm=speed_norm, cmap=speed_cmap)
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label("Speed")
        elif color_by_index:
            sm = cm.ScalarMappable(norm=index_norm, cmap=index_cmap)
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label("Normalized sample index")

        unit_label = "cm" if position_units == "cm" else "px"
        ax.set_xlabel(f"X Position ({unit_label})")
        ax.set_ylabel(f"Y Position ({unit_label})")
        if x_display_lim is not None:
            ax.set_xlim(x_display_lim)
        if y_display_lim is not None:
            ax.set_ylim(y_display_lim)
        if position_units == "cm":
            tick_values = np.linspace(0.0, arena_size_cm, 6)
            ax.set_xticks(tick_values)
            ax.set_yticks(tick_values)
        if invert_y:
            ax.invert_yaxis()
        if position_units == "cm":
            tick_values = np.linspace(0.0, arena_size_cm, 6)
            ax.set_yticks(tick_values)
            if invert_y:
                ax.set_yticklabels([f"{v:g}" for v in tick_values[::-1]])
        ax.set_aspect('equal', adjustable='box')
        if show_legend:
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                ax.legend()

    figs = []
    axes_out = []
    saved_paths = []

    def _slugify(value: str) -> str:
        slug = re.sub(r"[^0-9a-zA-Z]+", "_", str(value)).strip("_")
        return slug.lower() or "figure"

    suffix_parts = [mode]
    if show_average:
        suffix_parts.append("average")
    if highlight_hr:
        suffix_parts.append("hr_highlight")
    if color_by_index:
        suffix_parts.append("idx_color")
    if color_by_speed:
        suffix_parts.append("speed_color")
    if color_by_trial_id:
        suffix_parts.append("trialid_color")
    save_suffix = "_".join(filter(None, suffix_parts))

    def _make_fig(axis_key, title=None):
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        _plot_axis(ax, axis_key)
        if title and show_title:
            ax.set_title(title)
        plt.tight_layout()
        figs.append(fig)
        axes_out.append(ax)

        if save:
            axis_slug = _slugify(axis_key)
            save_name = f"trial_traces_{axis_slug}_{save_suffix}" if save_suffix else f"trial_traces_{axis_slug}"
            try:
                out_path = save_figure(
                    fig,
                    save_name,
                    subjids=[subjid],
                    dates=dates,
                    subdir=MOVEMENT_FIGURES_SUBDIR,
                )
                saved_paths.append(out_path)
                if verbose:
                    print(f"[plot_trial_traces_by_mode] Saved figure to {out_path}")
            except Exception as exc:
                if verbose:
                    print(f"[plot_trial_traces_by_mode] Failed to save figure '{save_name}': {exc}")

    # Layout by mode (separate figure per axis)
    if mode in {"rewarded", "rewarded_hr", "completed", "fa_by_response"}:
        for axis_key, title in zip(["combined", "A", "B"], ["Combined", "Odor A / Port 1", "Odor B / Port 2"]):
            _make_fig(axis_key, title)
    elif mode == "all_trials":
        _make_fig("combined", "All trials")
    elif mode == "fa_by_odor":
        odor_keys = [k for k in segments.keys()]
        if not odor_keys:
            print("No FA trials found for fa_by_odor")
            return None, None
        for key in odor_keys:
            _make_fig(key, f"Odor {key}")
    elif mode == "hr_only":
        axis_keys = [k for k in segments.keys() if k.startswith("HR ")]
        if "HR Summary" in segments:
            axis_keys.append("HR Summary")
        if not axis_keys:
            print("No HR trials found.")
            return None, None
        for key in axis_keys:
            _make_fig(key, key)

    result = figs if len(figs) > 1 else (figs[0], axes_out[0])
    if save and return_paths:
        return result, saved_paths
    return result


def plot_epoch_speeds_by_condition(
    subjid,
    dates=None,
    *,
    ses=None,
    index=None,
    date_range=None,
    ses_range=None,
    index_range=None,
    bin_ms: int = 100,
    fa_label_filter=None,
    mode: str = "mean",
    threshold: bool = True,
    threshold_alpha: float = 10.0,
    threshold_beta: float = 10.0,
    figsize=(8, 5),
    save: bool = False,
    verbose: bool = True,
    return_paths: bool = False,
):
    """Plot cue-port speed epochs from precomputed speed_analysis.parquet.

    Uses outputs from compute_speed_analysis (same parameters) to build per-session, per-condition
    per-trial traces with session mean overlay and optional threshold lines. Violin plots are omitted.
    Figures can optionally be saved into the movement_figures subdirectory when `save=True`.

    ``ses`` / ``index`` / ``date_range`` / ``ses_range`` / ``index_range`` narrow the
    selection further; they intersect with ``dates`` and with each other, and ``index``
    is the subject's gap-free chronological rank (`utils.helpers.session_selectors`).
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
                print(f"[plot_epoch_speeds_by_condition] Saved figure to {out_path}")
        except Exception as exc:
            if verbose:
                print(f"[plot_epoch_speeds_by_condition] Failed to save figure '{save_name}': {exc}")

    if mode not in {"max", "mean"}:
        raise ValueError("mode must be 'max' or 'mean'")

    # Normalize FA labels: accept comma-separated string or any iterable of labels (used at compute time)
    if fa_label_filter is None:
        fa_labels = {"fa_time_in"}
    elif isinstance(fa_label_filter, str):
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

    ses_dirs = _filter_session_dirs(subj_dir, dates, **select)
    if not ses_dirs:
        raise FileNotFoundError(f"No sessions found for subject {subjid} with given dates")

    bin_s = bin_ms / 1000.0
    baseline_window = (-0.15, -0.05)

    per_session = []
    combined_data = {"rewarded": [], "unrewarded": [], "fa": []}

    for ses_dir in ses_dirs:
        date_str = ses_dir.name.split("_date-")[-1]
        results_dir = ses_dir / "saved_analysis_results"
        if not results_dir.exists():
            continue

        df_speed = _get_from_cache(subjid, date_str, kind="speed_analysis")
        if df_speed is None:
            path_speed = results_dir / "speed_analysis.parquet"
            if not path_speed.exists():
                raise FileNotFoundError(f"Missing speed_analysis.parquet for {date_str}; run compute_speed_analysis first")
            df_speed = pd.read_parquet(path_speed)
            _update_cache(subjid, [date_str], {date_str: df_speed.copy()}, kind="speed_analysis")

        df_speed = df_speed.copy()
        conds_with_data = [c for c in ["rewarded", "unrewarded", "fa"] if not df_speed[df_speed["condition"] == c].empty]
        if not conds_with_data:
            continue

        # Baseline stats from stored speeds
        baseline_mask = (df_speed["bin_mid_s"] >= baseline_window[0]) & (df_speed["bin_mid_s"] <= baseline_window[1])
        baseline_vals = df_speed.loc[baseline_mask, "speed"].dropna().to_numpy()
        # One definition of the threshold, shared with compute_speed_analysis --
        # which is what produced the latencies this figure draws (finding 7).
        stats = speed_threshold(baseline_vals, alpha=threshold_alpha,
                                beta=threshold_beta, enabled=threshold)
        baseline_mean, baseline_sd = stats["mu"], stats["sigma"]
        thr_alpha_mu = stats["alpha_mu"]
        thr_mu_plus_beta_sigma = stats["mu_plus_beta_sigma"]
        thr_max = stats["max_alpha_mu_mu_plus_beta_sigma"]

        figs_by_cond = {}
        for cond in conds_with_data:
            sub = df_speed[df_speed["condition"] == cond].copy()
            if sub.empty:
                continue
            # Trial-wise traces
            trials = []
            trial_arrays = []
            mids_all = np.sort(sub["bin_mid_s"].unique())
            fig_t, ax_t = plt.subplots(figsize=figsize)

            for tid, g in sub.groupby("trial_index"):
                g = g.sort_values("bin_mid_s")
                mids = g["bin_mid_s"].to_numpy(float)
                speeds = g["speed"].to_numpy(float)
                if mids.size and speeds.size:
                    ax_t.plot(mids, speeds, color="gray", alpha=0.2)
                trials.append((tid, mids, speeds))

                arr_full = np.full_like(mids_all, np.nan, dtype=float)
                mid_to_idx = {m: i for i, m in enumerate(mids_all)}
                for m, s in zip(mids, speeds):
                    idx = mid_to_idx.get(m)
                    if idx is not None:
                        arr_full[idx] = s
                trial_arrays.append(arr_full)

            if trial_arrays:
                stack = np.vstack(trial_arrays)
                mean_speeds = np.nanmean(stack, axis=0)
                ax_t.plot(mids_all, mean_speeds, color="blue", linewidth=2, label="session mean")

            if threshold and baseline_mean is not None:
                ax_t.axhline(baseline_mean, color="red", linestyle="-", linewidth=1.5, label="baseline μ")
                if thr_max is not None:
                    ax_t.axhline(thr_max, color="#2F4F4F", linestyle="--", linewidth=1.4, label=f"max(αμ, μ+βσ), α={threshold_alpha:g}, β={threshold_beta:g}")

            ax_t.set_title(f"{cond} — sub {subjid}, {date_str} ({mode})")
            ax_t.set_xlabel("Time from last poke-out (s)")
            ax_t.set_ylabel("Speed (units/s)")
            ax_t.legend()
            fig_t.tight_layout()
            figs_by_cond[cond] = fig_t

            date_scope = [int(date_str)] if str(date_str).isdigit() else [date_str]
            save_name = f"epoch_speeds_{_slugify(cond)}_{_slugify(mode)}_{_slugify(date_str)}"
            _save_fig(fig_t, save_name, date_scope)

            if trial_arrays:
                combined_data[cond].append((date_str, mids_all, np.nanmean(np.vstack(trial_arrays), axis=0)))

        per_session.append({
            "date": date_str,
            "fig_traces": figs_by_cond,
            "baseline": {
                "mu": baseline_mean,
                "sigma": baseline_sd,
                "alpha": threshold_alpha,
                "beta": threshold_beta,
                "alpha_mu": thr_alpha_mu,
                "mu_plus_beta_sigma": thr_mu_plus_beta_sigma,
                "max_alpha_mu_mu_plus_beta_sigma": thr_max,
            } if threshold else None,
        })

    combined_figs = {}
    if len(per_session) > 1:
        colors = plt.cm.tab10.colors
        for idx, cond in enumerate(["rewarded", "unrewarded", "fa"]):
            if not combined_data[cond]:
                continue
            fig, ax = plt.subplots(figsize=figsize)
            for j, (date_str, mids, session_mean) in enumerate(combined_data[cond]):
                ax.plot(mids, session_mean, color=colors[j % len(colors)], label=date_str)
            ax.set_title(f"Session means — {cond} ({mode})")
            ax.set_xlabel("Time from last poke-out (s)")
            ax.set_ylabel("Speed (units/s)")
            ax.legend()
            fig.tight_layout()
            combined_figs[cond] = fig

            date_scope = []
            if combined_data[cond]:
                for date_str, *_ in combined_data[cond]:
                    date_scope.append(int(date_str) if str(date_str).isdigit() else date_str)
            save_name = f"epoch_speeds_combined_{_slugify(cond)}_{_slugify(mode)}"
            _save_fig(fig, save_name, date_scope or dates)

    result = {"per_session": per_session, "combined": combined_figs}
    if save and return_paths:
        return result, saved_paths
    return result

def plot_traces_with_speed_threshold(
    subjid,
    dates=None,
    *,
    ses=None,
    index=None,
    date_range=None,
    ses_range=None,
    index_range=None,
    xlim=None,
    ylim=None,
    position_units="cm",
    arena_size_cm=50.0,
    fa_types="FA_time_in",
    bin_ms: int = 100,
    pre_buffer_s: float = 0.2,
    threshold_alpha: float = 6.0,
    threshold_beta: float = 6.0,
    mode: str = "mean",
    smooth_window: int = 5,
    figsize=(10, 8),
    invert_y: bool = True,
    save: bool = False,
    verbose: bool = True,
    return_paths: bool = False,
):
    """Plot spatial traces for rewarded, unrewarded, and FA trials with a speed threshold marker.

    For the selected sessions, builds three figures (rewarded, unrewarded, fa). Traces are overlaid
    across sessions. Each trial trace gets a black dot at the first time after last poke-out when
    speed exceeds vthresh = max(alpha*mu, mu+beta*sigma), where mu/sigma come from the pooled
    baseline window [-0.15s, -0.05s] relative to last poke-out across all trials in the session.
    If a parquet with `speed_threshold_time` exists for the session (written by
    compute_speed_analysis), it is loaded (and cached) and used directly; otherwise the
    threshold is recomputed and the result is saved + cached.

    Parameters
    ----------
    subjid : int
        Subject ID.
    dates : list | tuple | None
        Dates list or inclusive range; None uses all available for the subject.
    xlim, ylim : tuple | None
        Pixel axis limits. When position_units="cm", each limit range is mapped
        to 0-arena_size_cm on the displayed axis.
    position_units : {"cm", "px"}
        Display position coordinates in centimetres or raw pixels.
    arena_size_cm : float
        Physical size represented by the xlim/ylim ranges when displaying cm.
    fa_types : str | Iterable
        FA labels to include (default "FA_time_in"). Case-insensitive; accepts comma/semicolon list.
    bin_ms : int
        Epoch/bin width in milliseconds for speed aggregation (default 100).
    pre_buffer_s : float
        Seconds to include before last poke-out when computing speed (default 0.2). Needs >=0.15s
        to populate the baseline window.
    threshold_alpha : float
        Multiplier for mu in the threshold definition (default 6.0).
    threshold_beta : float
        Multiplier for sigma in the threshold definition (default 6.0).
    mode : {"max", "mean"}
        Aggregation per bin when computing speeds.
    smooth_window : int
        Rolling window (frames) for smoothing X/Y before speed computation and plotting.
    figsize : tuple
        Figure size for each condition plot.
    invert_y : bool
        If True, invert Y-axis to match video coordinates.
    save : bool
        When True, saves each generated figure into movement_figures via save_figure().
    verbose : bool
        If True, logs save successes/failures.
    return_paths : bool
        When True and save is enabled, returns list of saved file paths alongside the figures.

    Returns
    -------
    dict with keys "rewarded", "unrewarded", "fa" mapping to matplotlib figures. When
    return_paths is True, also returns the list of saved file paths.

    ``ses`` / ``index`` / ``date_range`` / ``ses_range`` / ``index_range`` narrow the
    selection further; they intersect with ``dates`` and with each other, and ``index``
    is the subject's gap-free chronological rank (`utils.helpers.session_selectors`).
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
                print(f"[plot_traces_with_speed_threshold] Saved figure to {out_path}")
        except Exception as exc:
            if verbose:
                print(f"[plot_traces_with_speed_threshold] Failed to save figure '{save_name}': {exc}")

    # Ensure helper is available even if an old module version was loaded
    try:
        binned_speed_fn = _binned_speed
    except NameError:
        import hypnose_behavior.visualization.movement_analysis_utils as _mau
        binned_speed_fn = getattr(_mau, "_binned_speed", None)
    if binned_speed_fn is None:
        raise RuntimeError("_binned_speed helper not available; reload hypnose_behavior.visualization.movement_analysis_utils")

    # Color palette consistent with plot_trial_traces_by_mode
    port_colors = {1: "#FF6B6B", 2: "#4ECDC4"}
    port_colors_fa = {1: "#FF8E8E", 2: "#7EE9DF"}
    aborted_color = "#555555"

    # Normalize FA labels
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

    suffix_parts = [mode]
    if fa_label_display:
        suffix_parts.append(_slugify(fa_label_display))
    if smooth_window > 1:
        suffix_parts.append(f"smooth{smooth_window}")
    save_suffix = "_".join(filter(None, suffix_parts))

    if mode not in {"max", "mean"}:
        raise ValueError("mode must be 'max' or 'mean'")
    if position_units not in {"cm", "px"}:
        raise ValueError("position_units must be either 'cm' or 'px'")
    if position_units == "cm" and arena_size_cm <= 0:
        raise ValueError("arena_size_cm must be positive")
    if pre_buffer_s < 0.15:
        print("pre_buffer_s < 0.15s: baseline window [-0.15, -0.05] may be empty")

    subj_str = normalize_subjid(subjid)
    derivatives_dir = get_derivatives_root()
    subj_dir = derivatives.subject_dir(subjid)

    ses_dirs = _filter_session_dirs(subj_dir, dates, **select)
    if not ses_dirs:
        raise FileNotFoundError(f"No sessions found for subject {subjid} with given dates")

    baseline_window = (-0.15, -0.05)
    bin_s = bin_ms / 1000.0

    def _safe_dt(val):
        try:
            return pd.to_datetime(val)
        except Exception:
            return pd.NaT

    def _last_poke_out_scanning_back(entries):
        """Scan **back by position** to the first non-null `poke_odor_end`.

        A different rule from `_last_poke_out_by_position` above, which takes the
        last entry and accepts its null -- `DECISIONS.md` sections 13 and 14 are
        about not merging helpers that differ, so both survive the Phase 7b.4b move
        onto `position_data` unchanged.
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

    def _infer_port_with_odor_fallback(row):
        # Try explicit port fields first
        for col in [
            "response_port", "rewarded_port", "reward_port", "supply_port",
            "choice_port", "port", "fa_port", "last_reward_port", "odor_port",
        ]:
            if col in row and pd.notna(row[col]):
                try:
                    return int(row[col])
                except Exception:
                    try:
                        return int(float(row[col]))
                    except Exception:
                        continue
        # Try odor-number style fields
        for col in ["last_odor_num", "odor_num", "odor_index", "odor_position"]:
            if col in row and pd.notna(row[col]):
                try:
                    val = int(row[col])
                    if val == 2:
                        return 2
                    if val == 1:
                        return 1
                except Exception:
                    continue
        # Try odor labels
        odor = str(row.get("last_odor_name") or row.get("last_odor") or row.get("odor_name") or row.get("odor") or "").strip().lower()
        if odor in {"b", "odorb", "odor_b", "2", "portb", "port_b"}:
            return 2
        if odor in {"a", "odora", "odor_a", "1", "porta", "port_a"}:
            return 1
        return None

    def _port_for_coloring(row, cond):
        """Choose plotting port by explicit behavior columns first.

        - FA trials: use fa_port
        - Rewarded trials: use first_supply_port
        - Unrewarded trials: use first_reward_poke_port
        Falls back to generic inference if needed.
        """
        if cond == "fa":
            preferred_col = "fa_port"
        elif cond == "unrewarded":
            preferred_col = "first_reward_poke_port"
        else:
            preferred_col = "first_supply_port"

        if preferred_col in row and pd.notna(row[preferred_col]):
            try:
                return int(row[preferred_col])
            except Exception:
                try:
                    return int(float(row[preferred_col]))
                except Exception:
                    pass
        return _infer_port_with_odor_fallback(row)

    def _category_from_row(row):
        odor = str(row.get("last_odor_name") or row.get("last_odor") or "A")
        if odor in {"A", "OdorA", "1"}:
            return "A"
        if odor in {"B", "OdorB", "2"}:
            return "B"
        return "A"

    traces = {"rewarded": [], "unrewarded": [], "fa": []}
    markers = {"rewarded": [], "unrewarded": [], "fa": []}
    for ses_dir in ses_dirs:
        date_str = ses_dir.name.split("_date-")[-1]
        results_dir = ses_dir / "saved_analysis_results"
        if not results_dir.exists():
            continue
        skipped_no_poke_end = []
        analysis_path = results_dir / "speed_analysis.parquet"

        trial_data = None
        use_saved_thresholds = False

        cached_df = _get_from_cache(subjid, date_str, kind="speed_analysis")
        if cached_df is None and analysis_path.exists():
            try:
                cached_df = pd.read_parquet(analysis_path)
                _update_cache(subjid, [date_str], {date_str: cached_df.copy()}, kind="speed_analysis")
            except Exception as e:
                print(f"Failed to read {analysis_path.name}: {e}")

        if cached_df is not None:
            # Extract per-trial threshold times from per-bin records
            thr_series = (cached_df.dropna(subset=["speed_threshold_time"])
                                       .drop_duplicates(subset=["trial_index"])
                                       .set_index("trial_index")["speed_threshold_time"])
        else:
            thr_series = None

        views = _load_trial_views(results_dir)
        trial_data = views.get("trial_data", pd.DataFrame()).copy()
        if trial_data.empty:
            print(f"No trial_data for {date_str}; skipping")
            continue
        for c in ["sequence_start", "sequence_end", "first_supply_time", "first_reward_poke_time", "fa_time", "speed_threshold_time"]:
            if c in trial_data.columns:
                trial_data[c] = pd.to_datetime(trial_data[c], errors="coerce")
        # `in_poke_times` is the flag matching `position_poke_times`, the blob the two
        # `_last_poke_out_scanning_back` call sites read before Phase 7b.4b (section 2).
        pokes_by_trial = position_entries_by_trial(
            _load_position_data(results_dir, trial_data), "in_poke_times")

        if thr_series is not None:
            trial_data["speed_threshold_time"] = trial_data.index.map(thr_series)
            use_saved_thresholds = True

        # Load tracking per session
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
        tracking = smooth_xy(tracking, smooth_window)

        if use_saved_thresholds:
            # Strictly use stored threshold times; no recomputation
            for idx, row in trial_data.iterrows():
                rtc = str(row.get("response_time_category", "")).lower()
                is_aborted = bool(row.get("is_aborted", False))
                fa_label = str(row.get("fa_label", "")).lower()

                if rtc == "rewarded" and not is_aborted:
                    cond = "rewarded"
                elif rtc == "unrewarded" and not is_aborted:
                    cond = "unrewarded"
                elif fa_label.startswith("fa_") and fa_filter_fn(fa_label):
                    cond = "fa"
                else:
                    continue

                t_zero = _last_poke_out_scanning_back(
                    pokes_by_trial.get(row.get("global_trial_id")))
                if pd.isna(t_zero):
                    trial_id = row.get("trial_id", idx) if hasattr(row, "get") else idx
                    skipped_no_poke_end.append(trial_id)
                    continue
                t_end = _end_time(row, cond)
                if pd.isna(t_end) or t_end <= t_zero:
                    continue

                start_dt = t_zero - pd.Timedelta(seconds=pre_buffer_s)
                seg = tracking[(tracking["time"] >= start_dt) & (tracking["time"] <= t_end)].copy()
                if len(seg) < 2 or {"X", "Y", "time"} - set(seg.columns):
                    continue
                t_rel = (seg["time"] - t_zero).dt.total_seconds().to_numpy()
                if not np.isfinite(t_rel).all() or np.ptp(t_rel) == 0:
                    continue
                x = seg["X"].to_numpy()
                y = seg["Y"].to_numpy()

                marker = None
                thr_time = row.get("speed_threshold_time") if "speed_threshold_time" in trial_data.columns else pd.NaT
                if pd.notna(thr_time):
                    nearest_idx = int(np.argmin(np.abs((seg["time"] - thr_time).dt.total_seconds())))
                    marker = (x[nearest_idx], y[nearest_idx])

                port = _port_for_coloring(row, cond)
                if cond == "fa":
                    color = port_colors_fa.get(port, port_colors_fa[1])
                else:
                    color = port_colors.get(port, port_colors[1 if _category_from_row(row) == "A" else 2])

                traces[cond].append({"x": x, "y": y, "color": color, "session": date_str})
                if marker is not None:
                    markers[cond].append({"xy": marker, "color": "black", "session": date_str})
            if skipped_no_poke_end:
                print(f"Warning [{date_str}]: skipped trials with no poke_odor_end in position_poke_times: {skipped_no_poke_end}")
            # done with this session
            continue

        baseline_vals = []
        baseline_mask = None
        trial_cache = {}

        # First pass: per-trial binned speeds to build baseline and cache for later reuse
        for idx, row in trial_data.iterrows():
            rtc = str(row.get("response_time_category", "")).lower()
            is_aborted = bool(row.get("is_aborted", False))
            fa_label = str(row.get("fa_label", "")).lower()

            if rtc == "rewarded" and not is_aborted:
                cond = "rewarded"
            elif rtc == "unrewarded" and not is_aborted:
                cond = "unrewarded"
            elif fa_label.startswith("fa_") and fa_filter_fn(fa_label):
                cond = "fa"
            else:
                continue

            t_zero = _last_poke_out_scanning_back(
                pokes_by_trial.get(row.get("global_trial_id")))
            if pd.isna(t_zero):
                trial_id = row.get("trial_id", idx) if hasattr(row, "get") else idx
                skipped_no_poke_end.append(trial_id)
                continue
            t_end = _end_time(row, cond)
            if pd.isna(t_end) or t_end <= t_zero:
                continue

            mids_trial, arr_trial = binned_speed_fn(tracking, t_zero, t_end, pre_buffer_s, bin_s, mode)
            if mids_trial is None or arr_trial is None:
                continue

            if baseline_mask is None:
                baseline_mask = (mids_trial >= baseline_window[0]) & (mids_trial <= baseline_window[1])
            if baseline_mask is not None and baseline_mask.any():
                baseline_vals.extend([v for v in arr_trial[baseline_mask] if not np.isnan(v)])

            trial_cache[idx] = {
                "cond": cond,
                "t_zero": t_zero,
                "t_end": t_end,
                "mids": mids_trial,
                "arr": arr_trial,
            }

        if not baseline_vals:
            print(f"No baseline window data for {date_str}; skipping session")
            if skipped_no_poke_end:
                print(f"Warning [{date_str}]: skipped trials with no poke_odor_end in position_poke_times: {skipped_no_poke_end}")
            continue

        baseline_vals_arr = np.asarray([v for v in baseline_vals if np.isfinite(v)])
        if baseline_vals_arr.size == 0:
            print(f"Baseline values not finite for {date_str}; skipping session")
            if skipped_no_poke_end:
                print(f"Warning [{date_str}]: skipped trials with no poke_odor_end in position_poke_times: {skipped_no_poke_end}")
            continue
        stats = speed_threshold(baseline_vals_arr, alpha=threshold_alpha,
                                beta=threshold_beta)
        mu, sigma = stats["mu"], stats["sigma"]
        vthresh = stats["max_alpha_mu_mu_plus_beta_sigma"]

        if "speed_threshold_time" not in trial_data.columns:
            trial_data["speed_threshold_time"] = pd.NaT

        # Second pass: build traces and threshold markers using computed threshold
        for idx, meta in trial_cache.items():
            cond = meta["cond"]
            t_zero = meta["t_zero"]
            t_end = meta["t_end"]
            row = trial_data.loc[idx]

            thr_time = pd.NaT
            saved_thr = trial_data.at[idx, "speed_threshold_time"] if "speed_threshold_time" in trial_data.columns else pd.NaT
            if pd.notna(saved_thr):
                thr_time = saved_thr
            else:
                mids_trial = meta.get("mids")
                arr_trial = meta.get("arr")
                if mids_trial is not None and arr_trial is not None:
                    crossing_idx = np.where((mids_trial >= 0) & (arr_trial > vthresh))[0]
                    if crossing_idx.size > 0:
                        k = crossing_idx[0]
                        thr_time = t_zero + pd.Timedelta(seconds=float(mids_trial[k]))

            start_dt = t_zero - pd.Timedelta(seconds=pre_buffer_s)
            seg = tracking[(tracking["time"] >= start_dt) & (tracking["time"] <= t_end)].copy()
            if len(seg) < 2 or {"X", "Y", "time"} - set(seg.columns):
                continue
            t_rel = (seg["time"] - t_zero).dt.total_seconds().to_numpy()
            if not np.isfinite(t_rel).all() or np.ptp(t_rel) == 0:
                continue
            x = seg["X"].to_numpy()
            y = seg["Y"].to_numpy()

            marker = None
            trial_data.at[idx, "speed_threshold_time"] = thr_time
            if pd.notna(thr_time):
                nearest_idx = int(np.argmin(np.abs((seg["time"] - thr_time).dt.total_seconds())))
                marker = (x[nearest_idx], y[nearest_idx])

            port = _port_for_coloring(row, cond)
            if cond == "fa":
                color = port_colors_fa.get(port, port_colors_fa[1])
            else:
                color = port_colors.get(port, port_colors[1 if _category_from_row(row) == "A" else 2])

            traces[cond].append({"x": x, "y": y, "color": color, "session": date_str})
            if marker is not None:
                markers[cond].append({"xy": marker, "color": "black", "session": date_str})

        if skipped_no_poke_end:
            print(f"Warning [{date_str}]: skipped trials with no poke_odor_end in position_poke_times: {skipped_no_poke_end}")

    figs = {}

    def _coord_limits(axis):
        vals = []
        for cond_traces in traces.values():
            for tr in cond_traces:
                arr = np.asarray(tr[axis], dtype=float)
                vals.extend(arr[np.isfinite(arr)])
        for cond_markers in markers.values():
            for mk in cond_markers:
                xy = mk.get("xy")
                if xy is None:
                    continue
                val = xy[0] if axis == "x" else xy[1]
                if np.isfinite(val):
                    vals.append(float(val))
        if not vals:
            return None
        return float(np.nanmin(vals)), float(np.nanmax(vals))

    def _normalize_limits(limits, axis):
        if limits is None:
            limits = _coord_limits(axis)
        if limits is None:
            return None
        lo, hi = limits
        return float(lo), float(hi)

    x_source_limits = _normalize_limits(xlim, "x")
    y_source_limits = _normalize_limits(ylim, "y")
    x_display_lim = x_source_limits
    y_display_lim = y_source_limits

    def _scale_axis(values, limits):
        values = np.asarray(values, dtype=float)
        if position_units == "px":
            return values
        lo, hi = limits
        span = hi - lo
        if span == 0:
            return np.full_like(values, np.nan, dtype=float)
        scaled = (values - lo) / span * arena_size_cm
        return np.clip(scaled, 0.0, arena_size_cm)

    def _x_display(values):
        return _scale_axis(values, x_source_limits)

    def _y_display(values):
        return _scale_axis(values, y_source_limits)

    if position_units == "cm":
        if x_source_limits is None or y_source_limits is None:
            raise ValueError("Cannot convert to cm without finite x/y coordinate limits")
        x_display_lim = tuple(float(v) for v in _x_display(x_source_limits))
        y_display_lim = tuple(float(v) for v in _y_display(y_source_limits))
        if verbose:
            print(
                "[plot_traces_with_speed_threshold] position mapping: "
                f"x px {x_source_limits} -> cm {x_display_lim}; "
                f"y px {y_source_limits} -> cm {y_display_lim}"
            )

    for cond, label in [("rewarded", "Rewarded"), ("unrewarded", "Unrewarded"), ("fa", "False Alarms")]:
        if not traces[cond]:
            continue
        fig, ax = plt.subplots(figsize=figsize)
        for tr in traces[cond]:
            ax.plot(_x_display(tr["x"]), _y_display(tr["y"]), color=tr["color"])
        for mk in markers[cond]:
            ax.scatter(_x_display([mk["xy"][0]])[0], _y_display([mk["xy"][1]])[0], color="black", zorder=5)
        ax.set_title(f"{label} traces with speed-threshold crossing")
        unit_label = "cm" if position_units == "cm" else "px"
        ax.set_xlabel(f"X Position ({unit_label})")
        ax.set_ylabel(f"Y Position ({unit_label})")
        if x_display_lim is not None:
            ax.set_xlim(x_display_lim)
        if y_display_lim is not None:
            ax.set_ylim(y_display_lim)
        if position_units == "cm":
            tick_values = np.linspace(0.0, arena_size_cm, 6)
            ax.set_xticks(tick_values)
            ax.set_yticks(tick_values)
        if invert_y:
            ax.invert_yaxis()
        if position_units == "cm":
            tick_values = np.linspace(0.0, arena_size_cm, 6)
            ax.set_yticks(tick_values)
            if invert_y:
                ax.set_yticklabels([f"{v:g}" for v in tick_values[::-1]])
        ax.set_aspect('equal', adjustable='box')
        fig.tight_layout()
        figs[cond] = fig

        cond_dates_raw = sorted({tr.get("session") for tr in traces[cond] if tr.get("session")})
        date_scope = []
        for date_str in cond_dates_raw:
            date_scope.append(int(date_str) if str(date_str).isdigit() else date_str)
        save_name = f"speed_threshold_traces_{_slugify(cond)}"
        if save_suffix:
            save_name = f"{save_name}_{save_suffix}"
        _save_fig(fig, save_name, date_scope or dates)

    if save and return_paths:
        return figs, saved_paths
    return figs


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
    is the subject's gap-free chronological rank (`utils.helpers.session_selectors`).
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

    ses_dirs = _filter_session_dirs(subj_dir, dates, **select)
    if not ses_dirs:
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

    for ses_dir in ses_dirs:
        date_str = ses_dir.name.split("_date-")[-1]
        results_dir = ses_dir / "saved_analysis_results"
        if not results_dir.exists():
            continue

        views = _load_trial_views(results_dir)
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
            path_speed = results_dir / "speed_analysis.parquet"
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
    is the subject's gap-free chronological rank (`utils.helpers.session_selectors`).
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
