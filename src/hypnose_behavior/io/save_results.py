"""Saving trial-classification results to the derivatives tree.

Extracted from trial_classification/classification_utils.py during the restructuring
(Phase 3). Pure move -- behaviour unchanged (verified by the regression harness).
"""
from __future__ import annotations

import json
import math
import zoneinfo
from pathlib import Path
from datetime import datetime, timezone, date

import numpy as np
import pandas as pd

from hypnose_behavior.io.paths import get_rawdata_root, get_derivatives_root
from hypnose_behavior.utils.helpers import vprint
# Generic serialisation moved to hypnose-helpers (restructure_2 Phase 2a); re-exported
# here because callers import these names from this module.
from hypnose_helpers.io.serialize import (  # noqa: F401
    _json_safe, _json_default, _normalize_df_for_io,
)

def _has_value(v):
    """Truthy-and-present test for a timestamp-ish cell: not NaN, not zero, not blank.

    Not the flag-column truthiness rule of ``DECISIONS.md`` section 6 -- this is applied to
    ``await_reward_time``, where the question is "is there a timestamp here at all".
    """
    try:
        return pd.notna(v) and v != 0 and v != "0" and str(v).strip().lower() not in {"", "nan"}
    except Exception:
        return False


def _derive_outcome(row):
    """Outcome of one trial re-derived from the saved supply/poke counts.

    Deliberately independent of the response-time analysis: this runs over the merged trial
    table and fills in a category for trials the response-time pass could not compute one for.
    Module-level rather than nested so `qc/outcome_agreement.py` can measure it directly.
    """
    # Single-reward protocol: a completed NON-rewarded ("no-go") sequence is neither
    # rewarded/unrewarded/timeout in the reward sense — its outcome is carried by
    # false_response / fr_label. Leave response_time_category empty for these so existing
    # metrics stay clean. (Bool-safe: pandas may store sequence_rewarded as numpy.bool_.)
    seq_rew = row.get("sequence_rewarded")
    if pd.notna(seq_rew) and not bool(seq_rew):
        return None

    supply = pd.to_numeric(row.get("total_supply_count"), errors="coerce")
    reward_pokes = pd.to_numeric(row.get("total_reward_pokes"), errors="coerce")
    await_ts = row.get("await_reward_time")

    if pd.notna(supply) and supply >= 1:
        return "rewarded"
    if _has_value(await_ts):
        if pd.notna(reward_pokes) and reward_pokes >= 1:
            return "unrewarded"
        return "timeout_delayed"
    return None


def _find_parent_named(start: Path, prefix: str) -> Path | None:
    for p in [Path(start)] + list(Path(start).parents):
        if p.name.startswith(prefix):
            return p
    return None

def _find_rawdata_root(start: Path) -> Path | None:
    for p in [Path(start)] + list(Path(start).parents):
        if p.name == "rawdata":
            return p
    return None

def resolve_derivatives_output_dir(root) -> tuple[Path, dict]:
    root = Path(root).resolve()
    rawdata_dir = get_rawdata_root()
    try: 
        rel = root.relative_to(rawdata_dir)
    except ValueError:
        rawdata_dir = get_rawdata_root()
        rel = root


    hypnose_dir = rawdata_dir.parent
    sub_dir = _find_parent_named(root, "sub-")
    ses_dir = _find_parent_named(root, "ses-")
    if sub_dir is None or ses_dir is None:
        raise ValueError(f"Could not resolve sub-/ses- from: {root}")

    out_dir = get_derivatives_root() / sub_dir.name / ses_dir.name / "saved_analysis_results"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir, {
        "hypnose_dir": str(hypnose_dir),
        "rawdata_dir": str(rawdata_dir),
        "sub_folder": sub_dir.name,
        "ses_folder": ses_dir.name,
    }

def save_session_analysis_results(classification: dict, root, session_metadata: dict | None = None, data=None, events=None, verbose: bool = True) -> Path:
    out_dir, info = resolve_derivatives_output_dir(root)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "created_at": datetime.now().isoformat(),
        "session": _json_safe(session_metadata or {}),
        "paths": info,
        "tables": {},
        "artifacts": {},
        "notes": "DataFrames saved as CSV; object columns JSON-encoded. See *.schema.json. Per-run parameters stored in session.runs[].parameters.",
    }

    saved_any = False
    saved_names: set[str] = set()

    # Build comprehensive per-trial table (includes aborted + response-time extras)
    extra_abort_cols = [
        "last_odor_position",
        "last_odor_name",
        "last_odor_valve_duration_ms",
        "last_odor_poke_time_ms",
        "last_required_min_sampling_time_ms",
        "abortion_type",
        "abortion_time",
        "fa_label",
        "fa_time",
        "fa_latency_ms",
        "fa_port",
    ]
    extra_rt_cols = [
        "response_time_ms",
        "response_time_category",
    ]

    base_trials = classification.get("initiated_sequences") if isinstance(classification, dict) else None
    if isinstance(base_trials, pd.DataFrame) and not base_trials.empty and "trial_id" in base_trials.columns:
        trial_df = base_trials.copy()

        def _merge_with_run(df_target, df_extra, cols):
            """Merge extra cols using (trial_id, run_id) when present to avoid cross-run bleed."""
            merge_keys = ["trial_id", "run_id"] if all(k in df_extra.columns and k in df_target.columns for k in ["trial_id", "run_id"]) else ["trial_id"]
            subset_cols = [k for k in merge_keys + cols if k in df_extra.columns]
            if len(subset_cols) <= len(merge_keys):
                return df_target
            dedup = df_extra[subset_cols].drop_duplicates(subset=merge_keys)
            return df_target.merge(dedup, on=merge_keys, how="left")

        # Attach aborted details (aligned by trial_id and run_id when available)
        ab_det = classification.get("aborted_sequences_detailed") if isinstance(classification, dict) else None
        if isinstance(ab_det, pd.DataFrame) and not ab_det.empty and "trial_id" in ab_det.columns:
            cols = [c for c in extra_abort_cols if c in ab_det.columns]
            if cols:
                trial_df = _merge_with_run(trial_df, ab_det, cols)

        # Attach response-time details (aligned by trial_id and run_id when available)
        comp_rt = classification.get("completed_sequences_with_response_times") if isinstance(classification, dict) else None
        if isinstance(comp_rt, pd.DataFrame) and not comp_rt.empty and "trial_id" in comp_rt.columns:
            cols = [c for c in extra_rt_cols if c in comp_rt.columns]
            if cols:
                trial_df = _merge_with_run(trial_df, comp_rt, cols)

        # Derive outcome categories from supply/poke counts (avoids response-time dependency)
        derived_outcomes = trial_df.apply(_derive_outcome, axis=1)
        trial_df["response_time_category"] = derived_outcomes.where(derived_outcomes.notna(), trial_df.get("response_time_category"))

        # Ensure all expected columns exist
        for col in extra_abort_cols + extra_rt_cols:
            if col not in trial_df.columns:
                trial_df[col] = np.nan

        # Single-reward protocol only: ensure the false-response columns appear together (they are
        # produced upstream only for single-reward sessions, so nothing is added for the default
        # protocol and legacy output is unchanged).
        fr_cols = [
            "sequence_rewarded", "reward_determinacy", "determinacy_position",
            "determined_final_odor",
            "false_response", "fr_label",
            "fr_latency_ms", "fr_time", "fr_port", "fr_odor_identity", "fr_window_end",
        ]
        if any(col in trial_df.columns for col in fr_cols):
            for col in fr_cols:
                if col not in trial_df.columns:
                    trial_df[col] = np.nan

        # Convenience flag: mark aborted trials (any abortion info present)
        trial_df["is_aborted"] = trial_df[["abortion_type", "abortion_time"]].notna().any(axis=1)

        # Build global_trial_id continuous across runs
        if "run_id" not in trial_df.columns:
            trial_df["run_id"] = 1
        sort_cols = [c for c in ["sequence_start", "run_id", "trial_id"] if c in trial_df.columns]
        mapping = {}
        if "trial_id" in trial_df.columns:
            ordered = trial_df.sort_values(sort_cols, kind="stable") if sort_cols else trial_df
            for _, r in ordered.iterrows():
                tid = r.get("trial_id")
                rid = r.get("run_id", 1)
                if pd.isna(tid):
                    continue
                try:
                    key = (int(rid) if pd.notna(rid) else 1, int(tid))
                except Exception:
                    key = (rid, tid)
                if key not in mapping:
                    mapping[key] = len(mapping)

            def _global_id(row):
                tid = row.get("trial_id")
                rid = row.get("run_id", 1)
                try:
                    return mapping.get((int(rid) if pd.notna(rid) else 1, int(tid)))
                except Exception:
                    return mapping.get((rid, tid))

            trial_df = trial_df.copy()
            gvals = trial_df.apply(_global_id, axis=1)
            if "global_trial_id" in trial_df.columns:
                trial_df["global_trial_id"] = gvals
            else:
                trial_df.insert(0, "global_trial_id", gvals)

            def _attach_global(df):
                if not isinstance(df, pd.DataFrame) or df.empty or "trial_id" not in df.columns:
                    return df
                out = df.copy()
                if "run_id" not in out.columns:
                    out["run_id"] = 1
                g = out.apply(_global_id, axis=1)
                if "global_trial_id" in out.columns:
                    out["global_trial_id"] = g
                else:
                    out.insert(0, "global_trial_id", g)
                return out

            for k, v in list(classification.items()):
                if isinstance(v, pd.DataFrame) and "trial_id" in v.columns:
                    classification[k] = _attach_global(v)

        classification["trial_data"] = trial_df
    else:
        classification["trial_data"] = pd.DataFrame()

    def _save_df(name: str, df, *, save_parquet: bool = False) -> bool:
        if not isinstance(df, pd.DataFrame) or df.empty:
            return False
        if name in saved_names:
            return True
        f_csv = out_dir / f"{name}.csv"
        f_schema = out_dir / f"{name}.schema.json"
        f_parquet = out_dir / f"{name}.parquet"
        try:
            df_norm, json_cols = _normalize_df_for_io(df)
            df_norm.to_csv(f_csv, index=False)
            with open(f_schema, "w", encoding="utf-8") as sf:
                json.dump({"jsonified_columns": json_cols}, sf, indent=2)
            manifest["tables"][name] = f_csv.name
            if save_parquet:
                try:
                    df_norm.to_parquet(f_parquet, index=False)
                    manifest.setdefault("tables_parquet", {})[name] = f_parquet.name
                except Exception as e:
                    print(f"[save] WARNING: failed writing parquet for {name}: {e}")
            saved_names.add(name)
            return True
        except Exception as e:
            vprint(verbose, f"[save] WARNING: failed writing {name}: {e}")
            return False

    # Save only the streamlined set: trial_data (csv+parquet) plus non-initiated tables
    for k, save_parquet in [
        ("trial_data", True),
        ("non_initiated_sequences", False),
        ("non_initiated_odor1_attempts", False),
        ("non_initiated_FA", False),
    ]:
        df = classification.get(k) if isinstance(classification, dict) else None
        if _save_df(k, df, save_parquet=save_parquet):
            saved_any = True

    # 3) Extract run start and end times
    runs = manifest["session"].get("runs", [])
    london_tz = zoneinfo.ZoneInfo("Europe/London")

    for run in runs:
        # Extract start time from folder path (handle both Unix and Windows paths)
        root_path = run["root"]
        # Normalize path separators and get the last component
        run_start_str = root_path.replace("\\", "/").split("/")[-1]  # Extract the timestamp part (e.g., "2025-10-17T12-57-05")
        run_start = datetime.strptime(run_start_str, "%Y-%m-%dT%H-%M-%S").replace(tzinfo=zoneinfo.ZoneInfo("UTC"))
        run_start_london = run_start.astimezone(london_tz)
        run["start_time"] = run_start_london.isoformat()

        # Use precomputed end time if available
        stage_info = run.get("stage", {})
        precomputed_end_time = stage_info.get("run_end_time") if isinstance(stage_info, dict) else None
        
        if precomputed_end_time is not None:
            # Ensure precomputed_end_time is a datetime object
            if isinstance(precomputed_end_time, str):
                precomputed_end_time = datetime.fromisoformat(precomputed_end_time)

            # Convert to London time
            if precomputed_end_time.tzinfo is None:
                run_end_london = precomputed_end_time.replace(tzinfo=london_tz)
            else:
                run_end_london = precomputed_end_time.astimezone(london_tz)
            run["end_time"] = run_end_london.isoformat()
        else:
            # Fallback: try to extract from current data/events (existing logic)
            try:
                all_timestamps = []
                
                # Only try this fallback if we have data and events for this specific run
                if data is not None and events is not None:
                    # This fallback logic would need to filter by run, but it's complex
                    # Better to ensure the precomputed end time is always available
                    pass
                
                run["end_time"] = None
                if verbose:
                    print(f"Warning: No precomputed end time for run {run.get('run_id')}")
            except Exception as e:
                print(f"Error extracting end time for run {run.get('run_id')}: {e}")
                run["end_time"] = None

    # 4) Calculate gaps between runs
    for i in range(len(runs) - 1):
        run_end = runs[i].get("end_time")
        next_run_start = runs[i + 1].get("start_time")
        if run_end and next_run_start:
            run_end_dt = datetime.fromisoformat(run_end)
            next_run_start_dt = datetime.fromisoformat(next_run_start)
            gap = next_run_start_dt - run_end_dt
            runs[i]["gap_to_next_run"] = str(gap)
        else:
            runs[i]["gap_to_next_run"] = None

    manifest["session"]["runs"] = runs
    
    # 5) Indices
    indices_dir = out_dir / "indices"
    indices_dir.mkdir(parents=True, exist_ok=True)
    idx_payloads = {
        "index": classification.get("index", {}),
        "aborted_index": classification.get("aborted_index", classification.get("index", {}).get("aborted", {})),
    }
    for name, payload in idx_payloads.items():
        with open(indices_dir / f"{name}.json", "w", encoding="utf-8") as f:
            json.dump(_json_safe(payload), f, indent=2)

    # 6) Response-time analysis artifacts
    rta = classification.get("response_time_analysis")
    if isinstance(rta, dict):
        try:
            with open(out_dir / "response_time_analysis.json", "w", encoding="utf-8") as f:
                json.dump(_json_safe(rta), f, indent=2)
        except Exception as e:
            vprint(verbose, f"[save] WARNING: failed writing response_time_analysis.json: {e}")
        per_trial = rta.get("per_trial")
        if isinstance(per_trial, pd.DataFrame) and not per_trial.empty:
            if _save_df("response_time_per_trial", per_trial):
                saved_any = True

    # 7) Manifest + summary
    try:
        with open(out_dir / "manifest.json", "w", encoding="utf-8") as f:
            json.dump(_json_safe(manifest), f, indent=2)
    except Exception as e:
        vprint(verbose, f"[save] WARNING: failed writing manifest.json: {e}")
    counts = {}
    def _n(name):
        df = classification.get(name)
        return int(len(df)) if isinstance(df, pd.DataFrame) else 0
    for k in [
        "trial_data","non_initiated_sequences","non_initiated_odor1_attempts","non_initiated_FA",
    ]:
        counts[k] = _n(k)
    # Attach per-run parameters to manifest runs
    per_run_params = classification.get('per_run_parameters', [])
    if per_run_params and 'runs' in manifest['session']:
        for run_info in manifest['session']['runs']:
            run_id = run_info.get('run_id')
            matching_params = next((p for p in per_run_params if p.get('run_id') == run_id), None)
            if matching_params:
                run_info['parameters'] = matching_params

    # Save manifest
    with open(out_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(_json_safe(manifest), f, indent=2)
        
    # Add combined non-initiated total (baseline + pos1 attempts)
    counts["non_initiated_total"] = (
        counts.get("non_initiated_sequences", 0)
        + counts.get("non_initiated_odor1_attempts", 0)
    )

    params = {
        "sample_offset_time_ms": classification.get("sample_offset_time_ms"),
        "minimum_sampling_time_ms": classification.get("minimum_sampling_time_ms"),
        "default_minimum_sampling_time_ms": classification.get("default_minimum_sampling_time_ms"),
        "minimum_sampling_time_ms_by_odor": classification.get("minimum_sampling_time_ms_by_odor"),
        "response_time_window_sec": classification.get("response_time_window_sec"),
        "hidden_rule_location": classification.get("hidden_rule_location"),
        "hidden_rule_position": classification.get("hidden_rule_position"),
        "hidden_rule_locations": classification.get("hidden_rule_locations"),
        "hidden_rule_positions": classification.get("hidden_rule_positions"),
        "hidden_rule_odors": classification.get("hidden_rule_odors"),
    }
    summary = {
        "created_at": manifest["created_at"],
        "session": manifest["session"],
        "counts": counts,
        "params": params,
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(_json_safe(summary), f, indent=2)

    vprint(verbose, f"Saved analysis to: {out_dir} ({'some tables' if saved_any else 'no tables'})")
    return out_dir
