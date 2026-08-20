"""Saving trial-classification results to the derivatives tree.

Writes `trial_data`, its `position_data` side-table and `non_initiated_attempts` into
`saved_analysis_results/`, beside a `manifest.json` recording what produced them.
`io/load_results.py` is the read side.
"""
from __future__ import annotations

import json
import math
import zoneinfo
from functools import lru_cache
from pathlib import Path
from datetime import datetime, timezone, date

import numpy as np
import pandas as pd

from hypnose_behavior.io import layout
from hypnose_behavior.io.paths import get_rawdata_root, get_derivatives_root
from hypnose_behavior.utils.helpers import vprint
from hypnose_helpers.provenance import provenance
from hypnose_behavior.io.protocol_schema import (
    ABORT_COLUMNS, ASSEMBLED_COLUMNS, POSITION_BLOB_COLUMNS, RESPONSE_TIME_COLUMNS,
    trial_data_columns,
)
from hypnose_behavior.frames import build_position_data
from hypnose_behavior.parameters import scoring_parameters
from hypnose_behavior.trial_classification.outcome import classify_completed_trial, TIMEOUT
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

    Independent of the response-time analysis: it fills in a category for the trials that
    pass could not compute one for. The rule is
    `trial_classification.outcome.classify_completed_trial`; what belongs here is reading it
    off saved columns, and this table's timeout being spelled ``timeout_delayed``.
    """
    # Bool-safe: pandas may store sequence_rewarded as numpy.bool_, and NaN means "not a
    # single-reward session" rather than "not a rewarded sequence".
    seq_rew = row.get("sequence_rewarded")
    sequence_rewarded = bool(seq_rew) if pd.notna(seq_rew) else None

    supply = pd.to_numeric(row.get("total_supply_count"), errors="coerce")
    reward_pokes = pd.to_numeric(row.get("total_reward_pokes"), errors="coerce")

    outcome = classify_completed_trial(
        supply_count=0 if pd.isna(supply) else supply,
        reward_poke_count=0 if pd.isna(reward_pokes) else reward_pokes,
        has_await_reward=_has_value(row.get("await_reward_time")),
        sequence_rewarded=sequence_rewarded,
    )
    return "timeout_delayed" if outcome == TIMEOUT else outcome


@lru_cache(maxsize=1)
def _analysis_provenance() -> dict:
    """``{"commit": ..., "version": ...}`` for the code writing these results.

    For auditing: "which sessions were produced before commit X, and should I re-run
    them?" It is **not** a cache key -- a commit stamp invalidates on every unrelated
    commit, so plotters still compute through the registry (DECISIONS.md section 5).

    **Pass both arguments explicitly; do not let `provenance()` inspect the calling
    frame.** Each omission produces a plausible-looking wrong answer rather than an
    error: an anchor resolved inside the installed `hypnose-helpers` stamps every repo
    with the helpers commit, and a `call` captured from a notebook frame resolves to
    ``__main__``, where `package_version` returns ``None``.

    **Write both keys always, even as `None`**, so a reader can tell "written before
    provenance existed" (no key) from "written by code whose commit could not be
    resolved" (``None``). Cached because it shells out to ``git`` and describes the code
    as *imported*, which cannot change while the process runs.
    """
    prov = provenance(anchor=__file__, call={"module": __name__})
    return {"commit": prov.get("commit"), "version": prov.get("version")}


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

    out_dir = layout.results_dir(get_derivatives_root() / sub_dir.name / ses_dir.name)
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir, {
        "hypnose_dir": str(hypnose_dir),
        "rawdata_dir": str(rawdata_dir),
        "sub_folder": sub_dir.name,
        "ses_folder": ses_dir.name,
    }

def save_session_analysis_results(classification: dict, root, session_metadata: dict | None = None, data=None, events=None, verbose: bool = True, save_csv: bool = False) -> Path:
    """Write a session's analysis results to the derivatives tree.

    ``save_csv`` adds a human-readable CSV of every table alongside the parquet, off by
    default. **A caller that needs the CSV must pass it explicitly rather than rely on the
    default**, so a later change to the default cannot quietly break a gate.
    """
    out_dir, info = resolve_derivatives_output_dir(root)
    out_dir.mkdir(parents=True, exist_ok=True)
    protocol_mode = classification.get("protocol_mode") if isinstance(classification, dict) else None

    manifest = {
        "created_at": datetime.now().isoformat(),
        # Manifest only.
        **_analysis_provenance(),
        "analysis_parameters": scoring_parameters(),
        # Which schema this file follows, so a reader checks it against the right field
        # set instead of guessing from the columns it happens to find. Absent on older
        # files, which is how the loader knows to fall back to the mode-independent check.
        "protocol_mode": protocol_mode,
        "session": _json_safe(session_metadata or {}),
        "paths": info,
        "tables": {},
        "artifacts": {},
        "notes": "DataFrames saved as CSV; object columns JSON-encoded. See *.schema.json. Per-run parameters stored in session.runs[].parameters.",
    }

    saved_any = False
    saved_names: set[str] = set()

    # Build comprehensive per-trial table (includes aborted + response-time extras).
    # The two column lists live in `io/protocol_schema.py` alongside the trial record, so
    # there is one declaration of what `trial_data` holds rather than a copy here that can
    # drift from the one the loader checks against.
    extra_abort_cols = list(ABORT_COLUMNS)
    extra_rt_cols = list(RESPONSE_TIME_COLUMNS)

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
        expected = [c for c in trial_data_columns(protocol_mode) if c not in ASSEMBLED_COLUMNS] \
            if protocol_mode else extra_abort_cols + extra_rt_cols
        for col in expected:
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
    classification["position_data"] = build_position_data(
        classification["trial_data"], strict=True)

    def _save_df(name: str, df) -> bool:
        """Write one table: parquet always, CSV only when asked for.

        **Parquet for every table, and `save_csv` uniform across them.** Exempting a table
        from the parquet write makes it unreadable rather than merely unreadable-by-eye,
        since CSV is off by default. See DECISIONS.md section 23.

        The `.schema.json` sidecar goes with the CSV, not the parquet: it records which
        object columns were JSON-encoded to survive flat text, which parquet does not need.
        """
        if not isinstance(df, pd.DataFrame) or df.empty:
            return False
        if name in saved_names:
            return True
        f_csv = layout.table_path(out_dir, f"{name}.csv")
        f_schema = layout.table_path(out_dir, f"{name}.schema.json")
        f_parquet = layout.table_path(out_dir, f"{name}.parquet")
        try:
            df_norm, json_cols = _normalize_df_for_io(df)
            try:
                df_norm.to_parquet(f_parquet, index=False)
                manifest.setdefault("tables_parquet", {})[name] = f_parquet.name
            except Exception as e:
                print(f"[save] WARNING: failed writing parquet for {name}: {e}")
            if save_csv:
                df_norm.to_csv(f_csv, index=False)
                with open(f_schema, "w", encoding="utf-8") as sf:
                    json.dump({"jsonified_columns": json_cols}, sf, indent=2)
                manifest["tables"][name] = f_csv.name
            saved_names.add(name)
            return True
        except Exception as e:
            vprint(verbose, f"[save] WARNING: failed writing {name}: {e}")
            return False
    for k in ("trial_data", "position_data", "non_initiated_attempts"):
        df = classification.get(k) if isinstance(classification, dict) else None
        if k == "trial_data" and isinstance(df, pd.DataFrame) and not df.empty:
            df = df.drop(columns=list(POSITION_BLOB_COLUMNS), errors="ignore")
        if _save_df(k, df):
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
    indices_dir = layout.table_path(out_dir, "indices")
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
            with open(layout.table_path(out_dir, "response_time_analysis.json"), "w", encoding="utf-8") as f:
                json.dump(_json_safe(rta), f, indent=2)
        except Exception as e:
            vprint(verbose, f"[save] WARNING: failed writing response_time_analysis.json: {e}")
        per_trial = rta.get("per_trial")
        if isinstance(per_trial, pd.DataFrame) and not per_trial.empty:
            if _save_df("response_time_per_trial", per_trial):
                saved_any = True

    # 7) Manifest + summary
    try:
        with open(layout.table_path(out_dir, "manifest.json"), "w", encoding="utf-8") as f:
            json.dump(_json_safe(manifest), f, indent=2)
    except Exception as e:
        vprint(verbose, f"[save] WARNING: failed writing manifest.json: {e}")
    counts = {}
    def _n(name):
        df = classification.get(name)
        return int(len(df)) if isinstance(df, pd.DataFrame) else 0
    # `non_initiated_sequences` is counted here although it is not saved as a table: the
    # count is a real fact about the session (how many sampling attempts failed), and it
    # is what `merge.py`'s per-run sanity check compares against.
    for k in [
        "trial_data","position_data","non_initiated_sequences","non_initiated_odor1_attempts","non_initiated_attempts",
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
    with open(layout.table_path(out_dir, "manifest.json"), "w", encoding="utf-8") as f:
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
    with open(layout.table_path(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(_json_safe(summary), f, indent=2)

    vprint(verbose, f"Saved analysis to: {out_dir} ({'some tables' if saved_any else 'no tables'})")
    return out_dir
