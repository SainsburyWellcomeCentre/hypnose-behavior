# Defers evaluation of PEP-604 annotations (`X | None`), keeping this module
# importable on Python 3.9 for repos pinned there (hypnose-eeg-preprocessing).
from __future__ import annotations

"""Session loader for SLEAP tracking joined to the behavioural results.

Resolves the session directory, reads the combined tracking table and back-fills
the behaviour views.

- Lives in ``io/``, not ``visualization/``: ``metric_analysis.movement`` is a
  consumer, and ``metric_analysis`` must not import from ``visualization``.
"""

import pandas as pd

from hypnose_behavior.io import layout
from hypnose_behavior.io.layout import derivatives
from hypnose_behavior.io.loaders import _load_trial_views
from hypnose_behavior.io.paths import (
    get_derivatives_root,
    get_rawdata_root,
    get_server_root,
)
from hypnose_behavior.io.load_results import load_session_results
from hypnose_behavior.utils.helpers import (
    _get_from_cache,
    _update_cache,
    find_tracking_file,
    read_tracking_table,
)

__all__ = ["load_tracking_and_behavior"]


def _load_tracking_and_behavior(subjid, date, tracking_source='sleap'):
    """
    Load combined tracking CSV (SLEAP) and behavior results for a session.
    """
    # Try cache first
    cached = _get_from_cache(subjid, date, kind="sleap_session")
    if cached is not None:
        print(f"[CACHE HIT] SLEAP session for subjid={subjid}, date={date}")
        return cached["tracking"], cached["behavior"]

    base_path = get_rawdata_root()
    server_root = get_server_root()
    derivatives_dir = get_derivatives_root()

    session_dir = derivatives.find_session(subjid, date=date).path

    results_dir = layout.results_dir(session_dir)
    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    # Find tracking files (SLEAP only)
    csv_path = find_tracking_file(results_dir, "*_combined_sleap_tracking_timestamps")

    if csv_path is None:
        raise FileNotFoundError(
            f"No SLEAP tracking file found in {results_dir}."
        )

    source_used = 'sleap'

    tracking = read_tracking_table(csv_path)

    tracking['time'] = pd.to_datetime(tracking['time'], errors='coerce')

    # For SLEAP data: use 'centroid_x' and 'centroid_y' if available, else 'X'/'Y'
    if 'centroid_x' in tracking.columns and 'centroid_y' in tracking.columns:
        tracking['X'] = tracking['centroid_x']
        tracking['Y'] = tracking['centroid_y']
    elif 'X' not in tracking.columns:
        # Try to find any x/y columns
        x_cols = [c for c in tracking.columns if 'x' in c.lower() and 'score' not in c.lower()]
        y_cols = [c for c in tracking.columns if 'y' in c.lower() and 'score' not in c.lower()]
        if x_cols and y_cols:
            tracking['X'] = tracking[x_cols[0]]
            tracking['Y'] = tracking[y_cols[0]]

    behavior = load_session_results(subjid, date)

    # Fallback: populate key tables from trial_data when legacy tables are missing
    views = _load_trial_views(results_dir)
    td = views.get("trial_data", pd.DataFrame())
    if td is not None and not td.empty:
        # Completed sequences
        if behavior.get('completed_sequences', pd.DataFrame()).empty:
            comp = views.get("completed", pd.DataFrame()).copy()
            if not comp.empty:
                if 'last_odor' not in comp.columns and 'last_odor_name' in comp.columns:
                    comp = comp.rename(columns={'last_odor_name': 'last_odor'})
                behavior['completed_sequences'] = comp
        # Completed rewarded
        if behavior.get('completed_sequence_rewarded', pd.DataFrame()).empty:
            comp = views.get("completed", pd.DataFrame())
            if comp is not None and not comp.empty:
                rew = comp[comp.get("response_time_category", "") == "rewarded"].copy()
                if not rew.empty:
                    if 'last_odor' not in rew.columns and 'last_odor_name' in rew.columns:
                        rew = rew.rename(columns={'last_odor_name': 'last_odor'})
                    behavior['completed_sequence_rewarded'] = rew
        # Initiated sequences fallback (use completed as proxy)
        if behavior.get('initiated_sequences', pd.DataFrame()).empty:
            init_df = td.copy()
            if not init_df.empty:
                behavior['initiated_sequences'] = init_df

    
    print(f"Loaded {source_used.upper()} tracking: {len(tracking)} frames from {csv_path.name}")

    # Cache the processed session (tracking+behavior)
    session_data = {
        "tracking": tracking,
        "behavior": behavior,
    }
    _update_cache(subjid, [date], {date: session_data}, kind="sleap_session")

    return tracking, behavior

# Public alias; the underscore name is what the `visualization/` call sites import.
load_tracking_and_behavior = _load_tracking_and_behavior
