# Defers evaluation of PEP-604 annotations (`X | None`), keeping this module
# importable on Python 3.9 for repos pinned there (hypnose-eeg-preprocessing).
from __future__ import annotations

"""Spatial trajectory figures from SLEAP tracking.

Carved out of ``movement_analysis_utils.py`` in restructure_2 Phase 10
(follow-up Item 1). Source-only move -- no behaviour change.

DECISIONS section 13: the ``_infer_port`` / ``_last_poke_out`` /
``_extract_segment`` helpers in here and in ``sing_rew_movement`` are
**different rules wearing the same name** -- merging ``_infer_port``'s variants
changes trace colour and grouping on 63.8% of trials. They were renamed rather
than merged, and the names are the documentation. Do not merge them.

``plot_movement_trace`` reads an ezTrack ``add_timestamps_to_tracking`` CSV that
no QC coverage session has, so it is the one plotter here that no gate case can
execute.
"""

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from collections import defaultdict
from hypnose_behavior.frames import (
    odor_letter,
    position_entries_by_trial,
)
from hypnose_behavior.io.paths import (
    get_rawdata_root,
    get_derivatives_root,
    get_server_root,
)
from hypnose_behavior.utils.helpers import (
    _filter_session_dirs,
    _get_from_cache,
    _update_cache,
    session_selectors,
    find_tracking_file,
    read_tracking_table,
)
from hypnose_behavior.io.layout import (
    derivatives,
    normalize_subjid,
)
from hypnose_behavior.io.loaders import (
    _load_position_data,
    _load_trial_views,
)
from hypnose_behavior.visualization.prep import (
    resample_trace,
    smooth_xy,
)
from hypnose_behavior.visualization.prep import load_tracking_with_behavior
from hypnose_behavior.io.tracking import _load_tracking_and_behavior
from hypnose_behavior.io.save import save_figure
import re
import numpy as np
import json
from hypnose_behavior.io.save import MOVEMENT_FIGURES_SUBDIR



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
