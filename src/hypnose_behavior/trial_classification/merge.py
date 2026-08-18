"""Merging per-run trial classifications into a single session-level result.

Extracted from trial_classification/classification_utils.py during the restructuring
(Phase 3). Pure move -- behaviour unchanged (to be re-verified by the regression
harness once the data mount is available).
"""
from __future__ import annotations

from collections import defaultdict
from typing import Iterable

import numpy as np
import pandas as pd

from hypnose_behavior.trial_classification.index import build_classification_index


def _concat_align(dfs: Iterable[pd.DataFrame]) -> pd.DataFrame:
    dfs = [d for d in dfs if isinstance(d, pd.DataFrame) and not d.empty]
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, axis=0, ignore_index=True, sort=False)

def _assign_global_trial_ids(classif: dict) -> dict:
    """
    Ensure unique trial_id across merged runs.
    Uses sequence_start + run_id + original trial_id for stable ordering.
    """
    comp = classif.get('completed_sequences', pd.DataFrame())
    abo = classif.get('aborted_sequences', pd.DataFrame())
    cols = ['trial_id', 'run_id', 'sequence_start']
    frames = []
    if isinstance(comp, pd.DataFrame) and not comp.empty:
        frames.append(comp[[c for c in cols if c in comp.columns]])
    if isinstance(abo, pd.DataFrame) and not abo.empty:
        frames.append(abo[[c for c in cols if c in abo.columns]])
    if not frames:
        return classif

    all_trials = _concat_align(frames).dropna(subset=['trial_id']).copy()
    # Fill missing run_id with 1 before ordering (shouldn't happen if we add run_id)
    if 'run_id' not in all_trials.columns:
        all_trials['run_id'] = 1
    all_trials = all_trials.sort_values(
        [c for c in ['sequence_start', 'run_id', 'trial_id'] if c in all_trials.columns]
    ).reset_index(drop=True)

    mapping = { (int(r), int(t)): i+1 for i, (r, t) in enumerate(zip(all_trials['run_id'], all_trials['trial_id'])) }

    def remap_df(df: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(df, pd.DataFrame) or df.empty or 'trial_id' not in df.columns:
            return df
        df = df.copy()
        if 'run_id' not in df.columns:
            df['run_id'] = 1
        df['trial_id'] = [mapping.get((int(r), int(t)), t) for r, t in zip(df['run_id'], df['trial_id'])]
        return df

    for k, v in list(classif.items()):
        if isinstance(v, pd.DataFrame) and 'trial_id' in v.columns:
            classif[k] = remap_df(v)

    return classif

def _coerce_int_like(s):
    try:
        return pd.to_numeric(s, errors='coerce').astype('Int64')
    except Exception:
        return s

def _with_run_id(df: pd.DataFrame, run_id: int) -> pd.DataFrame:
    """Ensure run_id matches the merged run index and preserve any existing value."""

    if not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame()

    out = df.copy()
    if 'run_id' in out.columns:
        # Keep the original numbering (from the raw run index) for debugging/alignment checks
        out['run_id_original'] = out['run_id']
    out['run_id'] = run_id
    return out

def merge_classifications(run_results: list[dict], verbose: bool = True) -> dict:
    """
    Robust multi-run merger that preserves per-run parameters.
    - Concatenates tables directly with run_id
    - **Keeps parameters per-run** (not collapsed to single values)
    - Stores per-run metadata (stage, parameters, timings)
    - Rebuilds 'index' with build_classification_index on merged dict
    """

    if not run_results:
        raise ValueError("merge_classifications: no run results provided")

    # Normalize inputs to classification dicts
    per_run_cls = []
    per_run_rta = []
    per_run_metadata = []  # NEW: track all per-run info
    
    for ridx, r in enumerate(run_results, start=1):
        if r is None:
            continue
        # Either top-level dict with 'classification' or directly a classification dict
        if isinstance(r, dict) and 'classification' in r:
            cls = r['classification'] or {}
            rta = r.get('response_time_analysis') or cls.get('response_time_analysis') or {}
        else:
            cls = r or {}
            rta = cls.get('response_time_analysis') or {}
        per_run_cls.append((ridx, cls))
        per_run_rta.append((ridx, rta))

        # Collect per-run parameters and metadata
        per_run_metadata.append({
            'run_id': ridx,
            'sample_offset_time_ms': cls.get('sample_offset_time_ms'),
            'minimum_sampling_time_ms': cls.get('minimum_sampling_time_ms'),
            'default_minimum_sampling_time_ms': cls.get('default_minimum_sampling_time_ms'),
            'minimum_sampling_time_ms_by_odor': cls.get('minimum_sampling_time_ms_by_odor'),
            'response_time_window_sec': cls.get('response_time_window_sec'),
            'protocol_mode': cls.get('protocol_mode'),
            'hidden_rule_location': cls.get('hidden_rule_location'),
            'hidden_rule_position': cls.get('hidden_rule_position'),
            'hidden_rule_locations': cls.get('hidden_rule_locations'),
            'hidden_rule_positions': cls.get('hidden_rule_positions'),
            'hidden_rule_odors': cls.get('hidden_rule_odors'),
        })

    # Keys we will merge
    preferred_tables = [
        'initiated_sequences',
        'non_initiated_sequences',
        'non_initiated_odor1_attempts',
        'completed_sequences',
        'completed_sequences_with_response_times',
        'completed_sequence_rewarded',
        'completed_sequence_unrewarded',
        'completed_sequence_reward_timeout',
        'completed_sequence_false_response',
        'completed_sequences_HR',
        'completed_sequence_HR_rewarded',
        'completed_sequence_HR_unrewarded',
        'completed_sequence_HR_reward_timeout',
        'completed_sequences_HR_missed',
        'completed_sequence_HR_missed_rewarded',
        'completed_sequence_HR_missed_unrewarded',
        'completed_sequence_HR_missed_reward_timeout',
        'aborted_sequences',
        'aborted_sequences_HR',
        'aborted_sequences_detailed',
        'non_initiated_attempts',
    ]

    def _normalize_trial_id(s):
        if pd.isna(s):
            return None
        try:
            return int(s)
        except (ValueError, TypeError):
            return s
    
    merged: dict = {}
    
    # Concatenate tables
    for key in preferred_tables:
        parts = []
        for ridx, cls in per_run_cls:
            df = cls.get(key)
            if isinstance(df, pd.DataFrame) and not df.empty:
                df2 = _with_run_id(df, ridx)
                if 'trial_id' in df2.columns:
                    df2['trial_id'] = df2['trial_id'].apply(_normalize_trial_id)
                parts.append(df2)
        merged[key] = pd.concat(parts, axis=0, ignore_index=True, sort=False) if parts else pd.DataFrame()

    # Synthesize RT table if missing
    if merged['completed_sequences_with_response_times'].empty and not merged['completed_sequences'].empty:
        tmp = merged['completed_sequences'].copy()
        if 'response_time_ms' not in tmp.columns:
            tmp['response_time_ms'] = np.nan
        if 'response_time_category' not in tmp.columns:
            tmp['response_time_category'] = np.nan
        merged['completed_sequences_with_response_times'] = tmp

    # Sort time-based tables by sequence_start
    for key in preferred_tables:
        df = merged.get(key)
        if isinstance(df, pd.DataFrame) and not df.empty and 'sequence_start' in df.columns:
            merged[key] = df.sort_values(['sequence_start', 'run_id'] if 'run_id' in df.columns else ['sequence_start']).reset_index(drop=True)

    # Merge response_time_analysis lists
    rta_agg = defaultdict(list)
    for _, rta in per_run_rta:
        for k in ['rewarded_response_times', 'unrewarded_response_times',
                  'timeout_delayed_response_times', 'timeout_response_delay_times',
                  'all_response_times']:
            vals = rta.get(k, [])
            if isinstance(vals, (list, tuple, np.ndarray)):
                rta_agg[k].extend(list(vals))
    merged['response_time_analysis'] = dict(rta_agg)

    # ============ NEW: Store per-run metadata instead of collapsing ============
    merged['per_run_parameters'] = per_run_metadata

    # For backward compatibility, expose the first run's parameters as defaults
    # (but downstream code should prefer per_run_parameters[run_id])
    if per_run_metadata:
        first = per_run_metadata[0]
        merged['sample_offset_time_ms'] = first['sample_offset_time_ms']
        merged['minimum_sampling_time_ms'] = first['minimum_sampling_time_ms']
        merged['default_minimum_sampling_time_ms'] = first.get('default_minimum_sampling_time_ms')
        if isinstance(first.get('minimum_sampling_time_ms_by_odor'), dict):
            merged['minimum_sampling_time_ms_by_odor'] = dict(first['minimum_sampling_time_ms_by_odor'])
        else:
            merged['minimum_sampling_time_ms_by_odor'] = {}
        merged['response_time_window_sec'] = first['response_time_window_sec']
        # The session's protocol mode, which decides `trial_data`'s column set. Taken from
        # run 1: a session is one protocol, and every run's mode is kept in
        # `per_run_parameters` regardless. Were runs ever to disagree, the merged table
        # would be the union of both modes' columns, so the manifest's mode stays a
        # subset and the loader check cannot raise a false alarm on it.
        merged['protocol_mode'] = first.get('protocol_mode')
        merged['hidden_rule_location'] = first.get('hidden_rule_location')
        merged['hidden_rule_position'] = first.get('hidden_rule_position')
        merged['hidden_rule_locations'] = list(first.get('hidden_rule_locations') or [])
        merged['hidden_rule_positions'] = list(first.get('hidden_rule_positions') or [])
    
    # Aggregate unique hidden rule odors across all runs
    hr_odors_all: list[str] = []
    hr_positions_all: list[int] = []
    hr_locations_all: list[int] = []
    for meta in per_run_metadata:
        od = meta.get('hidden_rule_odors')
        if isinstance(od, (list, tuple)):
            hr_odors_all.extend([str(x) for x in od if isinstance(x, str) and x])
        pos_list = meta.get('hidden_rule_positions')
        if isinstance(pos_list, (list, tuple)):
            hr_positions_all.extend([
                int(pos) for pos in pos_list
                if isinstance(pos, (int, np.integer))
            ])
        else:
            pos_value = meta.get('hidden_rule_position')
            if isinstance(pos_value, (int, np.integer)):
                hr_positions_all.append(int(pos_value))

        loc_list = meta.get('hidden_rule_locations')
        if isinstance(loc_list, (list, tuple)):
            hr_locations_all.extend([
                int(idx) for idx in loc_list
                if isinstance(idx, (int, np.integer))
            ])
        else:
            loc_value = meta.get('hidden_rule_location')
            if isinstance(loc_value, (int, np.integer)):
                hr_locations_all.append(int(loc_value))
    if hr_odors_all:
        seen = set()
        uniq = []
        for x in hr_odors_all:
            if x not in seen:
                seen.add(x)
                uniq.append(x)
        merged['hidden_rule_odors'] = uniq
    else:
        merged.setdefault('hidden_rule_odors', [])

    if hr_positions_all:
        seen_pos: set[int] = set()
        ordered_pos: list[int] = []
        for pos in hr_positions_all:
            if pos not in seen_pos:
                seen_pos.add(pos)
                ordered_pos.append(pos)
        merged['hidden_rule_positions'] = ordered_pos

    if hr_locations_all:
        seen_loc: set[int] = set()
        ordered_loc: list[int] = []
        for loc in hr_locations_all:
            if loc not in seen_loc:
                seen_loc.add(loc)
                ordered_loc.append(loc)
        merged['hidden_rule_locations'] = ordered_loc

    # Per-run counts
    runs_meta = []
    for ridx, cls in per_run_cls:
        def _n(k):
            df = cls.get(k)
            return int(len(df)) if isinstance(df, pd.DataFrame) else 0
        runs_meta.append({
            'run_id': ridx,
            'counts': {
                'initiated_sequences': _n('initiated_sequences'),
                'non_initiated_sequences': _n('non_initiated_sequences'),
                'non_initiated_odor1_attempts': _n('non_initiated_odor1_attempts'),
                'completed_sequences': _n('completed_sequences'),
                'completed_sequences_with_response_times': _n('completed_sequences_with_response_times'),
                'aborted_sequences': _n('aborted_sequences'),
                'aborted_sequences_detailed': _n('aborted_sequences_detailed'), 
            }
        })
    merged['runs'] = runs_meta

    try:
        merged['index'] = build_classification_index(merged)
    except Exception:
        merged['index'] = {}

    if verbose:
        # Warn if parameters differ across runs
        params_differ = False
        if per_run_metadata and len(per_run_metadata) > 1:
            first_params = per_run_metadata[0]
            for meta in per_run_metadata[1:]:
                for key in [
                    'sample_offset_time_ms',
                    'minimum_sampling_time_ms',
                    'default_minimum_sampling_time_ms',
                    'minimum_sampling_time_ms_by_odor',
                    'response_time_window_sec',
                    'hidden_rule_location',
                    'hidden_rule_position',
                    'hidden_rule_locations',
                    'hidden_rule_positions',
                ]:
                    if meta.get(key) != first_params.get(key):
                        if not params_differ:
                            print("[merge_classifications] WARNING: Parameters differ across runs:")
                            params_differ = True
                        print(f"  Run {first_params['run_id']}: {key}={first_params.get(key)}")
                        print(f"  Run {meta['run_id']}: {key}={meta.get(key)}")
        
        # Sanity check: merged counts vs per-run sums
        try:
            total_non_ini = int(len(merged.get('non_initiated_sequences', [])))
            total_pos1 = int(len(merged.get('non_initiated_odor1_attempts', [])))
            total_initiated = int(len(merged.get('initiated_sequences', [])))
            sum_non_ini = sum(r['counts']['non_initiated_sequences'] for r in merged.get('runs', []))
            sum_pos1 = sum(r['counts']['non_initiated_odor1_attempts'] for r in merged.get('runs', []))
            sum_initiated = sum(r['counts']['initiated_sequences'] for r in merged.get('runs', []))
            if (total_non_ini != sum_non_ini) or (total_pos1 != sum_pos1) or (total_initiated != sum_initiated):
                print("[merge_classifications] WARNING: count mismatch after merge")
        except Exception:
            pass
    
    return merged
