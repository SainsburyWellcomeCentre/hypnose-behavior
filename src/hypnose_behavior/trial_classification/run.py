"""Session orchestration: classify one run, then discover runs, merge, save, summarise.

Extracted from trial_classification/classification_utils.py during the restructuring
(Phase 3). Pure move -- behaviour unchanged (to be re-verified by the regression
harness once the data mount is available).
"""
from __future__ import annotations

import io
import re
import contextlib
import zoneinfo
from glob import glob
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional
from collections.abc import Mapping

import numpy as np
import pandas as pd

import hypnose_behavior.trial_classification.detect_settings as detect_settings
from hypnose_behavior.io.loaders import (
    load_experiment, load_all_streams, load_experiment_events, load_odor_mapping,
)
from hypnose_behavior.io.save_results import save_session_analysis_results
from hypnose_behavior.io.layout import rawdata
from hypnose_behavior.io.protocol_schema import ConflictingProtocolError
from hypnose_behavior.trial_classification.aborted_trials import (
    abortion_classification, classify_noninitiated_FA,
)
from hypnose_behavior.trial_classification.classify_trials import classify_trials
from hypnose_behavior.trial_classification.detect_trials import detect_trials
from hypnose_behavior.trial_classification.hidden_rule import (
    _drop_final_hidden_rule_index, _ensure_int_list, _resolve_hidden_rule_from_stage,
)
from hypnose_behavior.trial_classification.index import build_classification_index
from hypnose_behavior.trial_classification.params import (
    _get_single_reward_info, get_experiment_parameters,
)
from hypnose_behavior.trial_classification.response_times import analyze_response_times
from hypnose_behavior.trial_classification.merge import merge_classifications
from hypnose_behavior.trial_classification.summary import print_merged_session_summary
from hypnose_behavior.utils.helpers import vprint

# --------------------------------------------------------------------------------------
# One run: detect -> classify -> response times -> abortions, assembled into one dict.
# Moved here in Phase 6c: it orchestrates classify_trials.py, response_times.py,
# aborted_trials.py, index.py and params.py, so it belongs above all of them rather than
# inside any one. analyze_session_multi_run_by_id_date below drives it once per run.
# --------------------------------------------------------------------------------------

def classify_and_analyze_with_response_times(data, events, trial_counts, odor_map, stage, root, verbose=True, run_id=None):# Wrapper function to fully classify all trials. 
    """
    Orchestrates classification + valve/poke timing + response-time augmentation.

    Returns:
      {
        'classification': <dict from classify_trial_outcomes_with_pokes_and_valves2>,
        'response_time_analysis': <dict from analyze_response_times>,
        'completed_sequences_with_response_times': <DataFrame of completed trials with RT columns>
      }
    """
    sample_offset_time, minimum_sampling_time_by_odor, response_time = get_experiment_parameters(root)
    sample_offset_time_ms = sample_offset_time * 1000
    minimum_sampling_time_ms_by_odor = {
        str(odor): float(threshold) * 1000.0
        for odor, threshold in (minimum_sampling_time_by_odor or {}).items()
        if threshold is not None
    }
    if not minimum_sampling_time_ms_by_odor:
        raise ValueError("minimumSamplingTime_by_odor missing or empty in schema; cannot run classification without per-odor thresholds")
    default_minimum_sampling_time_ms = max(minimum_sampling_time_ms_by_odor.values())

    response_time_sec = response_time
    if response_time_sec is None:
        raise ValueError("Response time parameter cannot be extracted from Schema file. Check detect_settings function.")

    params = {
        'sample_offset_time_ms': sample_offset_time_ms,
        'minimum_sampling_time_ms_by_odor': dict(minimum_sampling_time_ms_by_odor),
        'default_minimum_sampling_time_ms': float(default_minimum_sampling_time_ms),
        'minimum_sampling_time_ms': float(default_minimum_sampling_time_ms),
        'response_time_window_sec': response_time_sec
    }


    # 0) Detect single-reward protocol once and share with both classifiers (schema-based).
    #    When this is the default protocol (all sequences rewarded), single_reward_info disables
    #    every new code path so behaviour/output are identical to before.
    single_reward_info = _get_single_reward_info(root)

    # 1) Run the stable classifier (valve/poke timing included)
    classification = classify_trials(
        data, events, trial_counts, odor_map, stage, root, verbose=verbose,
        single_reward_info=single_reward_info
    )

    # 2) Run the response-time summary analyzer (prints/aggregates like the notebook)
    rt_summary = analyze_response_times(
        data, trial_counts, events, odor_map, stage, root, verbose=verbose,
        single_reward_info=single_reward_info
    )

    # 3) Aborted trial details
    aborted_detailed = abortion_classification(
        data, events, classification, odor_map, root, verbose=verbose
    )
    if run_id is not None and isinstance(aborted_detailed, pd.DataFrame) and not aborted_detailed.empty:
        if 'run_id' not in aborted_detailed.columns:
            aborted_detailed = aborted_detailed.copy()
            aborted_detailed['run_id'] = run_id
    classification['aborted_sequences_detailed'] = aborted_detailed

    # 3) Build fast lookup indices for downstream use
    classification['index'] = build_classification_index(classification)

    # 4) Hidden rule position from stage name/index
    hidden_rule_indices, sequence_name = _resolve_hidden_rule_from_stage(stage)
    schema_settings = {}
    try:
        _, schema_settings = detect_settings.detect_settings(root)
    except Exception:
        schema_settings = {}

    if not hidden_rule_indices:
        inferred_indices = schema_settings.get('hiddenRuleIndicesInferred')
        if inferred_indices is None:
            inferred_indices = schema_settings.get('hiddenRuleIndexInferred')
        hidden_rule_indices = _ensure_int_list(inferred_indices)

    hidden_rule_indices = sorted({idx for idx in hidden_rule_indices if isinstance(idx, int)})
    # Final position is always rewarded -> never a hidden-rule position (single-reward untouched).
    hidden_rule_indices = _drop_final_hidden_rule_index(hidden_rule_indices, schema_settings, single_reward_info[0])
    hidden_rule_positions = [idx + 1 for idx in hidden_rule_indices]
    hidden_rule_location = hidden_rule_indices[0] if hidden_rule_indices else None
    hidden_rule_pos = hidden_rule_positions[0] if hidden_rule_positions else None

    if hidden_rule_positions:
        if len(hidden_rule_positions) > 1:
            pos_str = ", ".join(str(pos) for pos in hidden_rule_positions)
            idx_str = ", ".join(str(idx) for idx in hidden_rule_indices)
            print(f"Hidden rule locations extracted: Positions {pos_str} (indices {idx_str})")
        else:
            print(f"Hidden rule location extracted: Location{hidden_rule_location} (index {hidden_rule_location}, position {hidden_rule_pos})")
    else:
        seq_label = sequence_name or str(stage)
        print(f"No Hidden Rule Location found in sequence name: {seq_label}. Proceeding without Hidden Rule analysis.")

    # Single-reward protocol status (always printed, like the hidden-rule message above)
    if single_reward_info[0]:
        print(f"Single-reward protocol detected: {len(single_reward_info[1])} rewarded sequence(s); "
              f"non-rewarded completions classified as false_response.")
    else:
        print("Single-reward protocol: not detected (all sequences rewarded at final position; standard analysis).")

# 5) Attach params and RT summary to classification
    classification['hidden_rule_location'] = hidden_rule_location
    classification['hidden_rule_position'] = hidden_rule_pos
    classification['hidden_rule_locations'] = list(hidden_rule_indices)
    classification['hidden_rule_positions'] = list(hidden_rule_positions)
    classification.update(params)
    classification['response_time_analysis'] = rt_summary
    
# 6) Build completed_sequences_with_response_times by merging analyzer per_trial (no recomputation)
    completed_df = classification.get('completed_sequences', pd.DataFrame()).copy()
    per_trial_df = rt_summary.get('per_trial')
    if isinstance(completed_df, pd.DataFrame) and not completed_df.empty and isinstance(per_trial_df, pd.DataFrame) and not per_trial_df.empty:
        if 'trial_id' in completed_df.columns and 'trial_id' in per_trial_df.columns:
            completed_with_rt = completed_df.merge(
                per_trial_df[[c for c in ('trial_id', 'response_time_ms', 'response_time_category',
                                          'completed_window_latency_ms')
                              if c in per_trial_df.columns]],
                on='trial_id',
                how='left',
                validate='one_to_one'
            )
        else:
            completed_with_rt = completed_df.copy()
            completed_with_rt['response_time_ms'] = np.nan
            completed_with_rt['response_time_category'] = np.nan
    else:
        completed_with_rt = completed_df
        if isinstance(completed_with_rt, pd.DataFrame) and not completed_with_rt.empty:
            # ensure RT columns exist
            if 'response_time_ms' not in completed_with_rt.columns:
                completed_with_rt['response_time_ms'] = np.nan
            if 'response_time_category' not in completed_with_rt.columns:
                completed_with_rt['response_time_category'] = np.nan

    classification['completed_sequences_with_response_times'] = completed_with_rt

    # 7) Build indices after everything is attached
    classification['index'] = build_classification_index(classification)

    # 8) Return wrapper payload
    return {
        'classification': classification,
        'response_time_analysis': rt_summary,
        'completed_sequences_with_response_times': completed_with_rt,
    }


def analyze_session_multi_run_by_id_date(subject_id: str, date_str: str, *, verbose: bool = True, max_runs: int = 32, save: bool = True, print_summary: bool = True, save_csv: bool = False):
    """
    Analyze all experiment files for a subject on a given date, then merge and (optionally) save.
    Now passes full per-run stage/parameter info to save_session_analysis_results.
    """
    
    subject_id = str(subject_id)
    date_str = str(date_str)

    def _maybe_silent(callable_, *args, **kwargs):
        if verbose:
            return callable_(*args, **kwargs)
        import io, contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            return callable_(*args, **kwargs)
    
    # Discover experiment roots
    session_roots: list[Path] = []
    visited: set[Path] = set()
    le = globals().get("load_experiment")
    if callable(le):
        for i in range(max_runs):
            try:
                root_i = _maybe_silent(le, subject_id, date_str, index=i)
                if not root_i:
                    break
                p = Path(root_i).resolve()
                if p in visited:
                    vprint(verbose, f"[analyze_session_multi_run] Duplicate experiment root at index {i}: {p}. Stopping discovery.")
                    break
                visited.add(p)
                session_roots.append(p)
            except (IndexError, FileNotFoundError) as e:
                vprint(verbose, f"[analyze_session_multi_run] Stopping at index {i}: {e}")
                break
    
    if not session_roots:
        raise RuntimeError(f"No experiment runs found for subject={subject_id} date={date_str}")

    # Sort oldest -> newest
    def _parse_ts(p: Path):
        from datetime import datetime
        try:
            return datetime.strptime(p.name, "%Y-%m-%dT%H-%M-%S")
        except Exception:
            return datetime.min
    try:
        session_roots.sort(key=_parse_ts)
    except Exception:
        session_roots.sort(key=lambda p: str(p))

    per_run = []
    merge_inputs = []
    roots: list[Optional[Path]] = []
    stages = []

    def extract_run_end_time(data, events):
        """Extract the latest timestamp from data and events for a single run"""
        all_timestamps = []
        
        for key, stream in data.items():
            if isinstance(stream, pd.DataFrame) and "Time" in stream.columns:
                timestamps = pd.to_datetime(stream["Time"], errors="coerce").dropna()
                all_timestamps.extend(timestamps)
            elif isinstance(stream, pd.Series) and hasattr(stream.index, 'dtype') and pd.api.types.is_datetime64_any_dtype(stream.index):
                all_timestamps.extend(stream.index)
        
        for key, event in events.items():
            if isinstance(event, pd.DataFrame) and "Time" in event.columns:
                timestamps = pd.to_datetime(event["Time"], errors="coerce").dropna()
                all_timestamps.extend(timestamps)
        
        if all_timestamps:
            return max(all_timestamps)
        return None

    for i, root in enumerate(session_roots[:max_runs]):
        try:
            vprint(verbose, f"[analyze_session_multi_run] Loaded run index {i}: root={root}")

            # Detect stage for THIS run
            try:
                import hypnose_behavior.trial_classification.detect_stage as detect_stage_module
                stage = detect_stage_module.detect_stage(root)
            except Exception:
                stage = {'stage_name': str(root)}

            # Single-reward protocol status — always printed, alongside the stage/hidden-rule
            # info above, so it shows even in non-verbose runs. When verbose, the per-run wrapper
            # prints this instead (this guard avoids a duplicate line).
            if not verbose:
                try:
                    _sri = _get_single_reward_info(root)
                    if _sri[0]:
                        print(f"Single-reward protocol detected: {len(_sri[1])} rewarded sequence(s); "
                              f"non-rewarded completions classified as false_response.")
                    else:
                        print("Single-reward protocol: not detected (all sequences rewarded at final position; standard analysis).")
                except Exception:
                    pass

            # Run pipeline
            data = _maybe_silent(load_all_streams, root, verbose=verbose)
            events = _maybe_silent(load_experiment_events, root, verbose=verbose)
            run_end_time = extract_run_end_time(data, events)
            odor_map = _maybe_silent(load_odor_mapping, root, data=data, verbose=verbose)
            trial_counts = detect_trials(data, events, root, odor_map, verbose=verbose, stage=stage)

            out = _maybe_silent(
                classify_and_analyze_with_response_times,
                data, events, trial_counts, odor_map, stage, root, verbose=verbose, run_id=i+1
            )

            # ============ NEW: Build comprehensive run metadata ============
            if isinstance(stage, dict):
                stage['run_end_time'] = run_end_time
                stage['root'] = str(root)
                # Extract start_time from folder name
                try:
                    from datetime import datetime, timezone
                    import zoneinfo
                    m = re.search(r'\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}', root.name)
                    if m:
                        ts_str = m.group(0)
                        ts_utc = datetime.strptime(ts_str, '%Y-%m-%dT%H-%M-%S').replace(tzinfo=timezone.utc)
                        uk_tz = zoneinfo.ZoneInfo("Europe/London")
                        ts_london = ts_utc.astimezone(uk_tz)
                        stage['start_time'] = ts_london.isoformat()
                except Exception:
                    pass
            else:
                stage = {
                    'stage_name': str(stage),
                    'run_end_time': run_end_time,
                    'root': str(root)
                }

            cls = out['classification'] if isinstance(out, dict) and 'classification' in out else out
            
            # Classify non-initiated FAs
            DIP0 = data['digital_input_data'].get('DIPort0', pd.Series(dtype=bool))
            DIP1 = data['digital_input_data'].get('DIPort1', pd.Series(dtype=bool))
            DIP2 = data['digital_input_data'].get('DIPort2', pd.Series(dtype=bool))
            hr_odors = cls.get('hidden_rule_odors', [])
            non_init = cls.get('non_initiated_sequences', pd.DataFrame())
            pos_1_attempt = cls.get('non_initiated_odor1_attempts', pd.DataFrame())
            fa_input_data = pd.concat([non_init, pos_1_attempt], ignore_index=True)
            response_time = cls.get('response_time_window_sec') 
            fa_noninit_df = classify_noninitiated_FA(fa_input_data, DIP0, DIP1, DIP2, response_time, hr_odors=hr_odors)
            if isinstance(fa_noninit_df, pd.DataFrame) and not fa_noninit_df.empty and 'run_id' not in fa_noninit_df.columns:
                fa_noninit_df = fa_noninit_df.copy()
                fa_noninit_df['run_id'] = i + 1

            # Named for what it holds, not for one of its columns: this is *every*
            # non-initiated attempt -- both failed sampling attempts and failed
            # position-1 attempts, per the concat above -- annotated with an FA outcome.
            # Most of its rows are `nFA`, so the old name `non_initiated_FA` told a
            # reader the opposite of the truth.
            cls['non_initiated_attempts'] = fa_noninit_df
            out['classification'] = cls

            # Normalize outputs for merging
            if isinstance(out, dict) and 'classification' in out:
                merge_inputs.append(out)
            elif isinstance(out, dict):
                merge_inputs.append({'classification': out})
            else:
                merge_inputs.append({'classification': out or {}})

            if not isinstance(cls, dict) or not cls:
                raise RuntimeError("Empty classification output")

            per_run.append(cls)
            roots.append(root)
            stages.append(stage)

        except ConflictingProtocolError:
            # Session-level, not run-level: every run of a session shares the task schema,
            # so skipping to the next one can only re-raise. Propagate, so the message
            # reaches `batch_analyze_sessions`, which names the subject and date. The
            # generic handler below is `vprint`-gated and would swallow it outright at
            # `verbose=False` -- the batch default -- leaving only a misleading
            # "No runs analyzed" with no hint that the session is misconfigured.
            raise
        except Exception as e:
            vprint(verbose, f"[analyze_session_multi_run] Skipping run index {i} due to error: {e}")
            #import traceback #--> Will print where inidvidual run failed, useful, but cluttering. Commented out for now.
            #traceback.print_exc()
            continue

    if not per_run:
        raise RuntimeError(f"No runs analyzed for subject={subject_id} date={date_str}")

    # Merge classifications (now preserves per-run params)
    merged = merge_classifications(merge_inputs, verbose=verbose)
    merged['aborted_index'] = merged.get('index', {}).get('aborted', {})
    merged['non_initiated_attempts'] = merged.get('non_initiated_attempts', pd.DataFrame())

    save_dir = None
    save_err: Exception | None = None
    if save:
        first_root = roots[0] if roots and roots[0] is not None else None

        # Reliable single-reward flag from the schema (sequence reward info), not a name
        # match. True iff at least one candidate sequence is not rewarded at its final
        # position. Persisted to manifest/summary so downstream code can gate on it.
        try:
            is_singrew = bool(_get_single_reward_info(first_root)[0]) if first_root is not None else False
        except Exception:
            is_singrew = False

        # ============ NEW: Build session metadata with per-run stage info ============
        session_meta = {
            'multi_run': True,
            'subject_id': subject_id,
            'date': date_str,
            'is_singrew': is_singrew,
            'runs': [
                {
                    'run_id': ridx + 1,
                    'root': str(r) if r is not None else None,
                    'stage': stages[ridx] if ridx < len(stages) else None,
                    'parameters': (
                        merged['per_run_parameters'][ridx] 
                        if 'per_run_parameters' in merged and ridx < len(merged['per_run_parameters']) 
                        else {}
                    )
                }
                for ridx, r in enumerate(roots)
            ]
        }
        
        try:
            save_dir = save_session_analysis_results(merged, first_root, session_meta, data, events, verbose=verbose, save_csv=save_csv)
        except Exception as e:
            save_err = e
            vprint(verbose, f"[save] WARNING: {e}")

    if print_summary:
        print_merged_session_summary(merged, subjid=subject_id, date=date_str, save=save, out_dir=save_dir)
    
    if save:
        if save_dir:
            print(f"[save] Success: results saved to: {save_dir}")
        else:
            msg = f"[save] FAILED {save_err}" if save_err else "[save] FAILED: no output directory returned"
            print(msg)
    else:
        print(f"[save] Skipped: save=False")

    cls = merged
    return {
        "classification": cls,                      
        "merged_classification": cls,              
        "per_run_classifications": per_run,
        "roots": roots,
        "stages": stages,
        "save_dir": save_dir,

        "completed_sequences_with_response_times": cls.get("completed_sequences_with_response_times", pd.DataFrame()),
        "completed_sequences": cls.get("completed_sequences", pd.DataFrame()),
        "completed_sequence_rewarded": cls.get("completed_sequence_rewarded", pd.DataFrame()),
        "completed_sequence_unrewarded": cls.get("completed_sequence_unrewarded", pd.DataFrame()),
        "completed_sequence_reward_timeout": cls.get("completed_sequence_reward_timeout", pd.DataFrame()),
        "completed_sequence_false_response": cls.get("completed_sequence_false_response", pd.DataFrame()),
        "aborted_sequences": cls.get("aborted_sequences", pd.DataFrame()),
    }


def build_position_pokes_table(classification: dict, *, threshold_ms: float | None = None) -> pd.DataFrame:
    """
    Flatten per-position poke info from completed_sequences into a tidy DataFrame.
    Columns: run_id, trial_id, position, odor, poke_time_ms, poke_first_in, valve_open_ts, valve_close_ts.
    If threshold_ms is provided, rows with poke_time_ms >= threshold_ms are dropped.
    """

    comp = classification.get("completed_sequences", pd.DataFrame())
    if not isinstance(comp, pd.DataFrame) or comp.empty:
        return pd.DataFrame(columns=[
            "run_id","trial_id","position","odor","poke_time_ms","poke_first_in","valve_open_ts","valve_close_ts"
        ])

    def _iter_items(ppt):
        if isinstance(ppt, Mapping):
            for k, v in ppt.items():
                if not isinstance(v, Mapping):
                    try:
                        v = dict(v)
                    except Exception:
                        continue
                pos = v.get("position")
                if pos is None:
                    try:
                        pos = int(k)
                    except Exception:
                        pos = k
                yield pos, v
        elif isinstance(ppt, (list, tuple)):
            for v in ppt:
                if isinstance(v, Mapping):
                    yield v.get("position"), v

    def _norm_valves(pvt):
        out = {}
        if isinstance(pvt, Mapping):
            items = list(pvt.items())
        elif isinstance(pvt, (list, tuple)):
            items = [(v.get("position"), v) for v in pvt if isinstance(v, Mapping)]
        else:
            items = []
        for k, v in items:
            if not isinstance(v, Mapping):
                try:
                    v = dict(v)
                except Exception:
                    v = {}
            pos = v.get("position")
            if pos is None:
                try:
                    pos = int(k)
                except Exception:
                    pos = k
            out[pos] = v
        return out

    rows = []
    for _, row in comp.iterrows():
        ppt = row.get("position_poke_times")
        if not isinstance(ppt, (dict, list, tuple)):
            continue
        pvt = row.get("position_valve_times") or {}
        valve_map = _norm_valves(pvt)

        run_id = row.get("run_id")
        trial_id = row.get("trial_id")
        try:
            run_id = int(run_id) if pd.notna(run_id) else None
        except Exception:
            pass
        try:
            trial_id = int(trial_id) if pd.notna(trial_id) else trial_id
        except Exception:
            pass

        for pos, info in _iter_items(ppt):
            if not isinstance(info, Mapping):
                try:
                    info = dict(info)
                except Exception:
                    continue
            poke_ms = pd.to_numeric(info.get("poke_time_ms"), errors="coerce")
            if pd.isna(poke_ms) or poke_ms <= 0:
                continue
            if threshold_ms is not None and float(poke_ms) >= float(threshold_ms):
                continue
            try:
                pos_norm = int(pos) if pos is not None else None
            except Exception:
                pos_norm = pos
            vt = valve_map.get(pos_norm, {})
            rows.append({
                "run_id": run_id,
                "trial_id": trial_id,
                "position": pos_norm,
                "odor": info.get("odor_name") or (vt or {}).get("odor_name"),
                "poke_time_ms": float(poke_ms),
                "poke_first_in": info.get("poke_first_in"),
                "valve_open_ts": (vt or {}).get("valve_open_ts"),
                "valve_close_ts": (vt or {}).get("valve_close_ts"),
            })

    out = pd.DataFrame(rows)
    if not out.empty:
        if "valve_open_ts" in out.columns:
            out["valve_open_ts"] = pd.to_datetime(out["valve_open_ts"], errors="coerce")
        if "poke_first_in" in out.columns:
            out["poke_first_in"] = pd.to_datetime(out["poke_first_in"], errors="coerce")
        out = out.sort_values(["run_id","trial_id","position","valve_open_ts"], kind="stable", na_position="last").reset_index(drop=True)
    return out


def _parse_date_input(dates_input):
    """
    Parse date input into a list of dates to analyze.
    
    - If dates_input is a list/iterable: return as-is (specific dates)
    - If dates_input is a tuple of 2 elements: treat as (start_date, end_date) range
      and discover all dates in that range from the filesystem
    - If dates_input is None: return None (analyze all dates)
    
    Returns: list of dates (int YYYYMMDD format) or None
    """
    if dates_input is None:
        return None
    
    # If it's a tuple with exactly 2 elements, treat as a range
    if isinstance(dates_input, tuple) and len(dates_input) == 2:
        start_date = int(dates_input[0])
        end_date = int(dates_input[1])
        
        # Convert to datetime for range operations
        start_dt = pd.to_datetime(str(start_date), format='%Y%m%d')
        end_dt = pd.to_datetime(str(end_date), format='%Y%m%d')
        
        # Generate all dates in range
        date_range = pd.date_range(start=start_dt, end=end_dt, freq='D')
        dates_list = [int(dt.strftime('%Y%m%d')) for dt in date_range]
        
        return dates_list
    
    # Otherwise treat as specific dates (list, set, etc.)
    return list(dates_input)

def batch_analyze_sessions(
    subjids=None,
    dates=None,
    *,
    save=True,
    verbose=False,
    print_summary=True,
    max_runs=200,
    save_csv=False,
):
    """
    Analyze all sessions for given subject(s) and/or date(s).
    - If subjids is None: analyze all subjects found in rawdata.
    - If dates is None: analyze all dates found for each subject.
    - If both are lists: analyze all combinations.
    - Handles missing subjects/dates gracefully.
    Returns a dict: {(subjid, date): result_dict}
    """
    results = {}

    # Discover subjects
    if subjids is None:
        subjids = [sid for sid, _ in rawdata.iter_subjects()]
    else:
        subjids = [int(s) for s in subjids]

    dates_to_run_global = _parse_date_input(dates)

    for subjid in subjids:
        # A missing subject and a subject with no sessions are different problems;
        # only the first is worth warning about, as before.
        if rawdata.subject_dir(subjid, missing_ok=True) is None:
            print(f"[batch_analyze_sessions] WARNING: Subject {subjid} not found.")
            continue

        # Discover available dates for this subject
        available_dates = [int(s.date) for s in rawdata.find_sessions(subjid)]

        # Determine which dates to run for this subject
        if dates_to_run_global is None:
            # Analyze all available dates
            dates_for_subject = available_dates
        else:
            # Use only dates that exist for this subject
            dates_for_subject = [dt for dt in dates_to_run_global if dt in available_dates]
            missing = [dt for dt in dates_to_run_global if dt not in available_dates]
            if missing and verbose:
                for dt in missing:
                    print(f"[batch_analyze_sessions] WARNING: Date {dt} not found for subject {subjid}.")
        
        for date in dates_for_subject:
            try:
                print(f"\n[batch_analyze_sessions] Analyzing subject {subjid}, date {date}...")
                res = analyze_session_multi_run_by_id_date(
                    subjid, date,
                    save=save,
                    verbose=verbose,
                    print_summary=print_summary,
                    max_runs=max_runs,
                    save_csv=save_csv,
                )
                results[(subjid, date)] = res
            except Exception as e:
                print(f"[batch_analyze_sessions] WARNING: Failed to analyze subject {subjid}, date {date}: {e}")
                if verbose:
                    import traceback
                    traceback.print_exc()
                continue
    
    # Return summary of analyzed sessions
    analyzed = {}
    for (subjid, date) in results.keys():
        analyzed.setdefault(subjid, []).append(date)
    
    print("\nAnalyzed session(s) for:")
    for subjid in sorted(analyzed):
        print(f"Subject {subjid}:")
        for date in sorted(analyzed[subjid]):
            print(f"    {date}")

    return results
