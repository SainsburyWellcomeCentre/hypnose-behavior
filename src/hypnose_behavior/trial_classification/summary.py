"""Printing/saving the merged session summary report.

Reads the **in-memory** `classification` dict, not the saved files, so it still sees
the per-position blobs `save_results` drops on the way to disk.
"""
from __future__ import annotations

import io
import contextlib
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

from hypnose_behavior.io import layout

def print_merged_session_summary(merged_classification: dict, subjid=None, date=None, save=False, out_dir=None) -> None:
    """
    Summary for merged multi-run results, now showing per-run parameters.
    """
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        cls = merged_classification or {}

        # ============ NEW: Show per-run parameter summary ============
        per_run_params = cls.get('per_run_parameters', [])
        if per_run_params and len(per_run_params) > 1:
            print("=" * 80)
            print("PER-RUN PARAMETERS")
            print("=" * 80)
            for meta in per_run_params:
                run_id = meta.get('run_id')
                print(f"\nRun {run_id}:")
                print(f"  Sample Offset Time: {meta.get('sample_offset_time_ms')} ms")
                default_min_ms = meta.get('default_minimum_sampling_time_ms', meta.get('minimum_sampling_time_ms'))
                print(f"  Default Minimum Sampling Time: {default_min_ms} ms")
                ms_map = meta.get('minimum_sampling_time_ms_by_odor') or {}
                if ms_map:
                    print("  Minimum Sampling Time by Odor (ms):")
                    for odor_name, threshold in sorted(ms_map.items()):
                        print(f"    - {odor_name}: {threshold}")
                print(f"  Response Time Window: {meta.get('response_time_window_sec')} s")
                hr_positions = meta.get('hidden_rule_positions') or []
                hr_locations = meta.get('hidden_rule_locations') or []
                if hr_positions:
                    print(f"  Hidden Rule Positions: {hr_positions}")
                else:
                    print(f"  Hidden Rule Position: {meta.get('hidden_rule_position')}")
                if hr_locations:
                    print(f"  Hidden Rule Indices: {hr_locations}")
                else:
                    print(f"  Hidden Rule Index: {meta.get('hidden_rule_location')}")
                print(f"  Hidden Rule Odors: {meta.get('hidden_rule_odors')}")
            print()

        # Rest of summary output (use first run's params or show per-run note)
        sample_offset_time_ms = cls.get("sample_offset_time_ms")
        minimum_sampling_time_ms = cls.get("minimum_sampling_time_ms")
        default_minimum_sampling_time_ms = cls.get("default_minimum_sampling_time_ms")
        minimum_sampling_time_ms_by_odor = cls.get("minimum_sampling_time_ms_by_odor") or {}
        response_time_window_sec = cls.get("response_time_window_sec")
        hr_pos_single = cls.get("hidden_rule_position")
        hr_positions = cls.get("hidden_rule_positions") or []
        hr_locations = cls.get("hidden_rule_locations") or []
        if not hr_positions and hr_pos_single is not None:
            hr_positions = [hr_pos_single]
        hr_positions = [int(pos) for pos in hr_positions if pos is not None]
        if not hr_locations and hr_positions:
            hr_locations = [pos - 1 for pos in hr_positions]

        
        if per_run_params and len(per_run_params) > 1:
            print(f"[Using parameters from Run 1 for summary; see per-run parameters above]\n")


        # Tables
        def get_df(key):
            df = cls.get(key)
            return df if isinstance(df, pd.DataFrame) else pd.DataFrame()

        ini = get_df("initiated_sequences")
        non_ini = get_df("non_initiated_sequences")
        non_ini_pos1 = get_df("non_initiated_odor1_attempts")

        comp_rt = get_df("completed_sequences_with_response_times")  # authoritative
        comp = get_df("completed_sequences")
        if comp.empty and not comp_rt.empty:
            comp = comp_rt  # comp_rt has everything from comp + RT cols

        comp_rew = get_df("completed_sequence_rewarded")
        comp_unr = get_df("completed_sequence_unrewarded")
        comp_tmo = get_df("completed_sequence_reward_timeout")
        comp_fr = get_df("completed_sequence_false_response")  # single-reward protocol only

        comp_hr = get_df("completed_sequences_HR")
        comp_hr_missed = get_df("completed_sequences_HR_missed")
        ab = get_df("aborted_sequences")
        ab_hr = get_df("aborted_sequences_HR")
        ab_det = get_df("aborted_sequences_detailed")

        # Helpers
        def pct(n, d):
            return (n / d * 100.0) if d else 0.0

        def fmt_ms(v):
            try:
                return f"{float(v):.1f}"
            except Exception:
                return "n/a"

        print("=" * 80, "\n")
        print("=" * 80)
        print(f"SUMMARY: TRIAL CLASSIFICATION AND POKE TIME ANALYSIS FOR SUBJECT [{subjid}] DATE [{date}]")
        print("=" * 80, "\n")
        print("=" * 80)
        if sample_offset_time_ms is not None:
            print(f"Sample offset time: {fmt_ms(sample_offset_time_ms)} ms")
        if default_minimum_sampling_time_ms is not None:
            print(f"Minimum sampling time (default): {fmt_ms(default_minimum_sampling_time_ms)} ms")
        elif minimum_sampling_time_ms is not None:
            print(f"Minimum sampling time: {fmt_ms(minimum_sampling_time_ms)} ms")
        if minimum_sampling_time_ms_by_odor:
            print("Minimum sampling times (ms) by odor:")
            for odor_name, threshold in sorted(minimum_sampling_time_ms_by_odor.items()):
                try:
                    print(f"  - {odor_name}: {fmt_ms(threshold)}")
                except Exception:
                    print(f"  - {odor_name}: {threshold}")
        if response_time_window_sec is not None:
            print(f"Response time window: {float(response_time_window_sec):.2f} s")

        # Attempts overview
        baseline_n = int(len(non_ini))
        pos1_n = int(len(non_ini_pos1))
        non_ini_total = baseline_n + pos1_n
        total_attempts = int(len(ini)) + non_ini_total
        print("\nTRIAL CLASSIFICATIONs:")
        hr_pos_display = ", ".join(str(pos) for pos in hr_positions) if hr_positions else "None"
        hr_idx_display = ", ".join(str(idx) for idx in hr_locations) if hr_locations else "None"
        print(f"Hidden Rule Locations: Positions {hr_pos_display} (indices {hr_idx_display})\n")
        hr_odors = merged_classification.get('hidden_rule_odors') or []
        print(f"Hidden Rule Odors: {', '.join(hr_odors) if hr_odors else 'None'}\n")
        print(f"Total attempts: {total_attempts}")
        print(f"-- Non-initiated sequences (total): {non_ini_total} ({pct(non_ini_total, total_attempts):.1f}%)")
        print(f"    -- Position 1 attempts within trials {pos1_n} ({pct(pos1_n, non_ini_total):.1f}%)")
        print(f"    -- Baseline non-initiated sequences {baseline_n} ({pct(baseline_n, non_ini_total):.1f}%)")
        print(f"-- Initiated sequences (\033[1mtrials\033[0m]): {int(len(ini))} ({pct(len(ini), total_attempts):.1f}%)\n")

        # Initiated breakdown
        comp_n = int(len(comp))
        ab_n = int(len(ab))
        def _count_unique_trials(df: pd.DataFrame) -> int:
            if not isinstance(df, pd.DataFrame) or df.empty:
                return 0
            subset = [col for col in ("run_id", "trial_id") if col in df.columns]
            if subset:
                return int(df.drop_duplicates(subset=subset).shape[0])
            return int(len(df))

        hr_rewarded_count = _count_unique_trials(merged_classification.get('completed_sequence_HR_rewarded', pd.DataFrame()))
        hr_missed_count = (
            _count_unique_trials(merged_classification.get('completed_sequences_HR_missed', pd.DataFrame()))
            + _count_unique_trials(merged_classification.get('aborted_sequences_HR', pd.DataFrame()))
        )
        hr_total_count = hr_rewarded_count + hr_missed_count

        print("INITIATED TRIALS BREAKDOWN:")
        print(f"-- Completed sequences: {comp_n} ({pct(comp_n, len(ini)): .1f}%)")
        print(f"-- Hidden Rule Trials (HR): {hr_total_count} ({pct(hr_total_count, len(ini)):.1f}%)")
        if hr_total_count:
            print(f"   -- Hidden Rule Trials Rewarded: {hr_rewarded_count} ({pct(hr_rewarded_count, hr_total_count):.1f}%)")
            print(f"   -- Hidden Rule Missed: {hr_missed_count} ({pct(hr_missed_count, hr_total_count):.1f}%)")
        else:
            print("   -- Hidden Rule Trials Rewarded: 0 (0.0%)")
            print("   -- Hidden Rule Missed: 0 (0.0%)")
        print(f"-- Aborted sequences: {ab_n} ({pct(ab_n, len(ini)): .1f}%)")
        # Count unique HR aborted trials (deduplicate by run_id, trial_id)
        ab_hr_count = _count_unique_trials(ab_hr)
        print(f"   -- Aborted Hidden Rule trials (HR): {int(ab_hr_count)} ({pct(ab_hr_count, ab_n):.1f}%)\n")

        print(f"REWARDED TRIALS BREAKDOWN:")
        print(f"-- Rewarded: {int(len(comp_rew))} ({pct(len(comp_rew), comp_n):.1f}%)")
        print(f"-- Unrewarded: {int(len(comp_unr))} ({pct(len(comp_unr), comp_n):.1f}%)")
        print(f"-- Reward Timeout: {int(len(comp_tmo))} ({pct(len(comp_tmo), comp_n):.1f}%)\n")

        # Single-reward protocol only: completed NON-rewarded ("no-go") sequences. The block is
        # skipped entirely for the default protocol so legacy summaries are unchanged.
        comp_fr_n = _count_unique_trials(comp_fr)
        if comp_fr_n:
            if "false_response" in comp_fr.columns:
                fr_true = int(comp_fr["false_response"].fillna(False).astype(bool).sum())
            else:
                fr_true = 0
            fr_false = comp_fr_n - fr_true
            print(f"NON-REWARDED SEQUENCE BREAKDOWN (completed no-go sequences):")
            print(f"-- Completed non-rewarded sequences: {comp_fr_n} ({pct(comp_fr_n, comp_n):.1f}%)")
            print(f"-- False Response (went to reward port): {fr_true} ({pct(fr_true, comp_fr_n):.1f}%)")
            print(f"-- Correct Withholding (nFR): {fr_false} ({pct(fr_false, comp_fr_n):.1f}%)")
            if "fr_label" in comp_fr.columns:
                fr_counts = comp_fr["fr_label"].value_counts()
                fr_in = int(fr_counts.get("FR_time_in", 0))
                fr_out = int(fr_counts.get("FR_time_out", 0))
                fr_late = int(fr_counts.get("FR_late", 0))
                rt_win = float(response_time_window_sec) if response_time_window_sec is not None else None
                rt_lbl = f"{rt_win:.0f} s" if rt_win is not None else "response window"
                rt_lbl3 = f"{rt_win * 3:.0f} s" if rt_win is not None else "3x response window"
                print(f"   -- FR Time In (within {rt_lbl}): {fr_in} ({pct(fr_in, fr_true):.1f}%)")
                print(f"   -- FR Time Out (up to {rt_lbl3}): {fr_out} ({pct(fr_out, fr_true):.1f}%)")
                print(f"   -- FR Late (after 3x, before next trial): {fr_late} ({pct(fr_late, fr_true):.1f}%)")
                if "fr_window_latency_ms" in comp_fr.columns:
                    s_fr = pd.to_numeric(comp_fr.loc[comp_fr["false_response"] == True, "fr_window_latency_ms"], errors="coerce").dropna()
                    if len(s_fr):
                        print(f"   -- FR latency: median {fmt_ms(s_fr.median())} ms, mean {fmt_ms(s_fr.mean())} ms")
            print()

        # Aggregate poke/valve time stats from completed trials (use comp, which has nested columns)
        def collect_pos_stats(df: pd.DataFrame):
            # Infer max positions from data to avoid hardcoding
            inferred_max = 0
            if not df.empty:
                all_pos_keys = []
                for _, r in df.iterrows():
                    pps = r.get("position_poke_times") or {}
                    vps = r.get("position_valve_times") or {}
                    all_pos_keys.extend([k for k in pps.keys() if isinstance(k, (int, np.integer))])
                    all_pos_keys.extend([k for k in vps.keys() if isinstance(k, (int, np.integer))])
                if all_pos_keys:
                    try:
                        inferred_max = max(int(p) for p in all_pos_keys)
                    except Exception:
                        inferred_max = 0
            pos_poke = {i: [] for i in range(1, inferred_max + 1)}
            pos_valve = {i: [] for i in range(1, inferred_max + 1)}
            odor_poke = defaultdict(list)
            odor_valve = defaultdict(list)
            if df.empty:
                return pos_poke, pos_valve, odor_poke, odor_valve
            for _, r in df.iterrows():
                pps = r.get("position_poke_times") or {}
                vps = r.get("position_valve_times") or {}
                for pos in range(1, inferred_max + 1):
                    if pos in pps:
                        v = pps[pos].get("poke_time_ms")
                        if v is not None and not (isinstance(v, float) and np.isnan(v)):
                            pos_poke[pos].append(float(v))
                        od = pps[pos].get("odor_name")
                        if od is not None:
                            odor_poke[od].append(float(v) if v is not None else np.nan)
                    if pos in vps:
                        v = vps[pos].get("valve_duration_ms")
                        if v is not None and not (isinstance(v, float) and np.isnan(v)):
                            pos_valve[pos].append(float(v))
                        od = vps[pos].get("odor_name")
                        if od is not None:
                            odor_valve[od].append(float(v) if v is not None else np.nan)
            odor_poke = {k: [x for x in vals if not (isinstance(x, float) and np.isnan(x))] for k, vals in odor_poke.items()}
            odor_valve = {k: [x for x in vals if not (isinstance(x, float) and np.isnan(x))] for k, vals in odor_valve.items()}
            return pos_poke, pos_valve, odor_poke, odor_valve

        pos_poke, pos_valve, odor_poke, odor_valve = collect_pos_stats(comp)

        def print_range_block_pos(dct):
            print("----------------------------------------")
            for pos in range(1, 6):
                vals = dct.get(pos, [])
                if not vals:
                    continue
                a = np.asarray(vals, dtype=float)
                print(f"Position {pos}: {a.min():.1f} - {a.max():.1f}ms (avg: {a.mean():.1f}ms, n={a.size})")

        def print_range_block_odor(dct):
            print("--------------------------------------------------")
            for od in sorted(dct.keys()):
                vals = dct[od]
                if not vals:
                    continue
                a = np.asarray(vals, dtype=float)
                print(f"{od}: {a.min():.1f} - {a.max():.1f}ms (avg: {a.mean():.1f}ms, n={a.size})")

        print("POKE TIME RANGES BY POSITION:")
        print_range_block_pos(pos_poke)
        print("\nVALVE TIME RANGES BY POSITION:")
        print_range_block_pos(pos_valve)
        print("\nPOKE TIME RANGES BY ODOR (ALL POSITIONS):")
        print_range_block_odor(odor_poke)
        print("\nVALVE TIME RANGES BY ODOR (ALL POSITIONS):")
        print_range_block_odor(odor_valve)

        # Non-initiated poke times
        def _choose_poke_series(df: pd.DataFrame, prefer_cols: list[str]) -> pd.Series:
            if not isinstance(df, pd.DataFrame) or df.empty:
                return pd.Series([], dtype=float)
            for c in prefer_cols:
                if c in df.columns:
                    return pd.to_numeric(df[c], errors="coerce").dropna()
            return pd.Series([], dtype=float)

        print("\nNON-INITIATED TRIALS POKE TIMES:")
        print("----------------------------------------")
        base_vals = _choose_poke_series(non_ini, ["continuous_poke_time_ms", "poke_time_ms", "poke_time", "poke_ms"])
        if base_vals.empty:
            print(f"Baseline non-initiated: n={baseline_n} (no valid poke times)")
        else:
            print(f"Baseline non-initiated: n={baseline_n} median={base_vals.median():.1f} ms range={base_vals.min():.1f}-{base_vals.max():.1f} ms")
        pos1_vals = _choose_poke_series(non_ini_pos1, ["pos1_poke_time_ms", "attempt_poke_time_ms", "poke_time_ms", "poke_time", "poke_ms"])
        if pos1_vals.empty:
            print(f"Pos1 attempts: n={pos1_n} (no valid poke times)")
        else:
            print(f"Pos1 attempts: n={pos1_n} median={pos1_vals.median():.1f} ms range={pos1_vals.min():.1f}-{pos1_vals.max():.1f} ms")

        # Response time analysis from comp_rt
        print("=" * 80)
        print("RESPONSE TIME ANALYSIS - ALL COMPLETED TRIALS")
        print("=" * 80)
        print(f"Total completed trials: {int(len(comp_rt))}\n")

        s_cat = comp_rt['response_time_category'] if 'response_time_category' in comp_rt.columns else pd.Series([], dtype='object')
        failed = int(s_cat.isna().sum()) if not comp_rt.empty else 0
        succeeded = int(len(comp_rt) - failed)
        print("RESPONSE TIME ANALYSIS RESULTS:")
        print(f"Total completed trials analyzed: {int(len(comp_rt))}")
        print(f"Failed response time calculations: {failed}")
        print(f"Successful response time calculations: {succeeded}\n")

        def rt_block(df, label):
            if df.empty or 'response_time_ms' not in df.columns:
                print(f"{label}:\n  No {label.lower()}")
                return
            s = pd.to_numeric(df['response_time_ms'], errors="coerce").dropna()
            if s.empty:
                print(f"{label}:\n  No {label.lower()}")
                return
            print(f"{label}:")
            print(f"  Count: {int(len(s))}")
            print(f"  Range: {s.min():.1f} - {s.max():.1f}ms")
            print(f"  Average: {s.mean():.1f}ms")
            print(f"  Median: {s.median():.1f}ms\n")

        def _cat(df: pd.DataFrame, cat: str) -> pd.DataFrame:
            if not isinstance(df, pd.DataFrame) or df.empty or 'response_time_category' not in df.columns:
                return pd.DataFrame()
            m = df['response_time_category'].astype('object') == cat
            return df[m]

        rew_rt = _cat(comp_rt, "rewarded")
        unr_rt = _cat(comp_rt, "unrewarded")
        tdel_rt = _cat(comp_rt, "timeout_delayed")

        rt_block(rew_rt, "REWARDED TRIALS")

        if not rew_rt.empty:
            if 'hidden_rule_success' in rew_rt.columns:
                hr_rew_rt = rew_rt[rew_rt['hidden_rule_success'].fillna(False)]
            else:
                hr_rew_df = merged_classification.get('completed_sequence_HR_rewarded', pd.DataFrame())
                if isinstance(hr_rew_df, pd.DataFrame) and not hr_rew_df.empty and 'trial_id' in hr_rew_df.columns:
                    if 'run_id' in hr_rew_df.columns and 'run_id' in rew_rt.columns:
                        hr_pairs = set(
                            (
                                int(row['trial_id']),
                                int(row['run_id'])
                            )
                            for _, row in hr_rew_df.dropna(subset=['trial_id', 'run_id']).iterrows()
                        )
                        hr_rew_rt = rew_rt[
                            rew_rt.apply(
                                lambda r: (
                                    not pd.isna(r.get('trial_id'))
                                    and not pd.isna(r.get('run_id'))
                                    and (int(r['trial_id']), int(r['run_id'])) in hr_pairs
                                ),
                                axis=1
                            )
                        ] if hr_pairs else pd.DataFrame()
                    else:
                        hr_ids = set(hr_rew_df['trial_id'].dropna().tolist())
                        hr_rew_rt = rew_rt[rew_rt['trial_id'].isin(hr_ids)] if hr_ids else pd.DataFrame()
                else:
                    hr_rew_rt = pd.DataFrame()
            if hr_rew_rt.empty:
                print("HR REWARDED TRIALS (response times): none\n")
            else:
                rt_block(hr_rew_rt, "HR REWARDED TRIALS (response times)")
        else:
            print("HR REWARDED TRIALS (response times): none\n")

        rt_block(unr_rt, "UNREWARDED TRIALS")
        if tdel_rt.empty:
            print("REWARD TIMEOUT TRIALS:\n  No reward timeout trials\n")
        else:
            rt_block(tdel_rt, "REWARD TIMEOUT TRIALS")

        s_all = pd.to_numeric(comp_rt.get("response_time_ms"), errors="coerce").dropna() if not comp_rt.empty else pd.Series([], dtype=float)
        print("ALL TRIALS WITH RESPONSE TIMES:")
        if s_all.empty:
            print("  No trials with response times")
        else:
            print(f"  Count: {int(len(s_all))}")
            print(f"  Range: {s_all.min():.1f} - {s_all.max():.1f}ms")
            print(f"  Average: {s_all.mean():.1f}ms")
            print(f"  Median: {s_all.median():.1f}ms")

        # Aborted trials summary (same logic as before, but using concatenated tables)
        if not ab_det.empty:
            print("=" * 80)
            print("ABORTED TRIALS CLASSIFICATION SUMMARY")
            print("=" * 80)
            total = int(len(ab_det))
            ini_c = int((ab_det["abortion_type"] == "initiation_abortion").sum()) if 'abortion_type' in ab_det.columns else 0
            rei_c = int((ab_det["abortion_type"] == "reinitiation_abortion").sum()) if 'abortion_type' in ab_det.columns else 0
            print(f"- Total Aborted Trials: {total}")
            print(f"  - Re-Initiation Abortions: {rei_c} ({pct(rei_c, total):.1f}%)")
            print(f"  - Initiation Abortions:    {ini_c} ({pct(ini_c, total):.1f}%)\n")

            def _norm_fa(val):
                if pd.isna(val):
                    return 'nFA'
                s = str(val).strip().lower()
                if s in ('fa_time_in', 'fa in', 'fa_in', 'in'):
                    return 'FA_time_in'
                if s in ('fa_time_out', 'fa out', 'fa_out', 'out'):
                    return 'FA_time_out'
                if s in ('fa_late', 'late'):
                    return 'FA_late'
                return 'nFA'
            if 'fa_label' in ab_det.columns:
                ab_det = ab_det.copy()
                ab_det["fa_label"] = ab_det["fa_label"].apply(_norm_fa)

            fa_in = int((ab_det["fa_label"] == "FA_time_in").sum()) if "fa_label" in ab_det.columns else 0
            fa_out = int((ab_det["fa_label"] == "FA_time_out").sum()) if "fa_label" in ab_det.columns else 0
            fa_late = int((ab_det["fa_label"] == "FA_late").sum()) if "fa_label" in ab_det.columns else 0
            fa_total = fa_in + fa_out + fa_late
            nfa = total - fa_total

            print("False Alarms:")
            print(f"  - non-FA Abortions: {nfa} ({pct(nfa, total):.1f}%)")
            print(f"  - False Alarm abortions: {fa_total} ({pct(fa_total, total):.1f}%)")
            if fa_total > 0:
                print(f"      - FA Time In - Within Response Time Window ({float(response_time_window_sec) if response_time_window_sec is not None else 'n/a'} s):  {fa_in} ({pct(fa_in, fa_total):.1f}%)")
                s_in = pd.to_numeric(ab_det.loc[ab_det['fa_label'] == 'FA_time_in', 'fa_window_latency_ms'], errors='coerce').dropna() if 'fa_window_latency_ms' in ab_det.columns else pd.Series([], dtype=float)
                if len(s_in):
                    print(f"          - Response Time: median={s_in.median():.1f} ms, avg={s_in.mean():.1f} ms, range: {s_in.min():.1f} - {s_in.max():.1f} ms")
                if response_time_window_sec is not None:
                    lower_rt = response_time_window_sec
                    upper_rt = response_time_window_sec * 3
                    print(f"      - FA Time Out - Up to 3x Response Time Window ({int(lower_rt)}-{int(upper_rt)} s):  {fa_out} ({pct(fa_out, fa_total):.1f}%)")
                else:
                    print(f"      - FA Time Out: {fa_out} ({pct(fa_out, fa_total):.1f}%)")
                s_out = pd.to_numeric(ab_det.loc[ab_det['fa_label'] == 'FA_time_out', 'fa_window_latency_ms'], errors='coerce').dropna() if 'fa_window_latency_ms' in ab_det.columns else pd.Series([], dtype=float)
                if len(s_out):
                    print(f"          - Response Time: median={s_out.median():.1f} ms, avg={s_out.mean():.1f} ms, range: {s_out.min():.1f} - {s_out.max():.1f} ms")
                print(f"      - FA Late - After 3x Response Time up to next trial: {fa_late} ({pct(fa_late, fa_total):.1f}%)")
                s_late = pd.to_numeric(ab_det.loc[ab_det['fa_label'] == 'FA_late', 'fa_window_latency_ms'], errors='coerce').dropna() if 'fa_window_latency_ms' in ab_det.columns else pd.Series([], dtype=float)
                if len(s_late):
                    print(f"          - Response Time: median={s_late.median():.1f} ms, avg={s_late.mean():.1f} ms, range: {s_late.min():.1f} - {s_late.max():.1f} ms")

            # Abortions at Hidden Rule positions: split into HR vs non-HR trials, with FA breakdown
            if hr_positions and 'last_odor_position' in ab_det.columns:
                abortions_at_hr_pos = ab_det[ab_det['last_odor_position'].isin(hr_positions)].copy()
                abortions_at_last_pos = ab_det[~ab_det['last_odor_position'].isin(hr_positions)].copy()
                
                # Resolve HR-aborted trial IDs from merged classification
                # Use (run_id, trial_id) pairs to handle multiple runs correctly
                hr_ab_df = cls.get('aborted_sequences_HR')
                if isinstance(hr_ab_df, pd.DataFrame) and not hr_ab_df.empty and 'trial_id' in hr_ab_df.columns:
                    if 'run_id' in hr_ab_df.columns:
                        # For multi-run data, use (run_id, trial_id) pairs
                        hr_pairs = set(zip(hr_ab_df['run_id'].fillna(1), hr_ab_df['trial_id']))
                        if 'run_id' in abortions_at_hr_pos.columns:
                            matched = abortions_at_hr_pos.apply(
                                lambda row: (row.get('run_id', 1), row['trial_id']) in hr_pairs, 
                                axis=1
                            )
                            in_hr_trials = abortions_at_hr_pos[matched].copy()
                            non_hr_trials = abortions_at_hr_pos[~matched].copy()
                        else:
                            # Fallback if no run_id in abortions_at_hr_pos
                            hr_aborted_ids = set(hr_ab_df['trial_id'].dropna().tolist())
                            in_hr_trials = abortions_at_hr_pos[abortions_at_hr_pos['trial_id'].isin(hr_aborted_ids)].copy()
                            non_hr_trials = abortions_at_hr_pos[~abortions_at_hr_pos['trial_id'].isin(hr_aborted_ids)].copy()
                    else:
                        # Single-run data, use trial_id alone
                        hr_aborted_ids = set(hr_ab_df['trial_id'].dropna().tolist())
                        if 'trial_id' in abortions_at_hr_pos.columns:
                            in_hr_trials = abortions_at_hr_pos[abortions_at_hr_pos['trial_id'].isin(hr_aborted_ids)].copy()
                            non_hr_trials = abortions_at_hr_pos[~abortions_at_hr_pos['trial_id'].isin(hr_aborted_ids)].copy()
                        else:
                            in_hr_trials = pd.DataFrame()
                            non_hr_trials = abortions_at_hr_pos.copy()
                else:
                    in_hr_trials = pd.DataFrame()
                    non_hr_trials = abortions_at_hr_pos.copy()

                def _print_fa_counts(df, indent="        "):
                    order = ['nFA', 'FA_time_in', 'FA_time_out', 'FA_late']
                    labels = [
                        "Non-False Alarm",
                        "FA Time In (Within Response Time)",
                        "FA Time Out (Up to 3x Response Time)",
                        "FA Late (> 3x Response Time)"
                    ]
                    if df is None or df.empty or 'fa_label' not in df.columns:
                        return
                    cnt = df['fa_label'].value_counts().reindex(order, fill_value=0)
                    total_n = int(len(df))
                    for lbl, key in zip(labels, order):
                        v = int(cnt.get(key, 0))
                        p = (v / total_n * 100.0) if total_n else 0.0
                        print(f"{indent}- {lbl}: {v} ({p:.1f}%)")

                total_at_hr = int(len(abortions_at_hr_pos))
                hr_pos_summary = ", ".join(str(pos) for pos in hr_positions) if hr_positions else "None"
                print(f"\n  Abortions at Hidden Rule Positions {hr_pos_summary}: n={total_at_hr}")
                print(f"    Of which in Hidden Rule Trials: n={int(len(in_hr_trials))}")
                _print_fa_counts(in_hr_trials)
                print(f"    Non-Hidden Rule Abortions at HR Location: n={int(len(non_hr_trials))}")
                _print_fa_counts(non_hr_trials)

            # False Alarm classification for non-initiated trials (if present)
            fa_noninit_df = merged_classification.get('non_initiated_attempts', pd.DataFrame())
            if isinstance(fa_noninit_df, pd.DataFrame) and not fa_noninit_df.empty: 
                print("\nFalse Alarm Classification for Non-Initiated Trials:")
                print(f"  Total Non-Initiated FA Trials: {int(len(fa_noninit_df))}")
                counts = fa_noninit_df['fa_label'].value_counts().reindex(['nFA','FA_time_in','FA_time_out','FA_late'], fill_value=0)
                total = int(len(fa_noninit_df))
                print(f"   - Non-False Alarm: {counts['nFA']} ({(counts['nFA']/total*100.0):.1f}%)")
                print(f"   - FA Time In (Within Response Time): {counts['FA_time_in']} ({(counts['FA_time_in']/total*100.0):.1f}%)")
                print(f"   - FA Time Out (Up to 3x Response Time): {counts['FA_time_out']} ({(counts['FA_time_out']/total*100.0):.1f}%)")
                print(f"   - FA Late (> 3x Response Time): {counts['FA_late']} ({(counts['FA_late']/total*100.0):.1f}%)")   


            # Helper for stats lines
            def _stats_line(series, label):
                s = pd.to_numeric(series, errors='coerce').dropna()
                if s.empty:
                    print(f"{label}: n=0")
                else:
                    print(f"{label}: n={len(s)} | median={s.median():.1f} ms | avg={s.mean():.1f} ms | range={s.min():.1f}-{s.max():.1f} ms")

            # Non-last Odor Pokes (exclude last_event_index per trial), only >= odor-specific thresholds
            if {'presentations', 'last_event_index'}.issubset(ab_det.columns):
                pres_df = ab_det[['trial_id', 'presentations', 'last_event_index']].explode('presentations').dropna(subset=['presentations']).copy()
                if not pres_df.empty:
                    pres = pd.concat([pres_df.drop(columns=['presentations']), pres_df['presentations'].apply(pd.Series)], axis=1)
                    pres['is_last'] = pres['index_in_trial'] == pres['last_event_index']
                    pres = pres[~pres['is_last']].copy()
                    pres['poke_time_ms'] = pd.to_numeric(pres.get('poke_time_ms'), errors='coerce')
                    pres['required_min_sampling_time_ms'] = pd.to_numeric(
                        pres.get('required_min_sampling_time_ms'), errors='coerce'
                    )
                    pres_valid = pres.dropna(subset=['required_min_sampling_time_ms']).copy()
                    pres_valid = pres_valid[
                        pres_valid['poke_time_ms'] >= pres_valid['required_min_sampling_time_ms']
                    ]

                    print("\nPoke Times for all Odors (Except aborted Odor):")
                    _stats_line(pres_valid['poke_time_ms'], "  - All Odors (except aborted)")

                    if 'position' in pres_valid.columns and not pres_valid.empty:
                        for pos, grp in pres_valid.groupby('position'):
                            _stats_line(grp['poke_time_ms'], f"  - Position {int(pos)}")

                    if 'odor_name' in pres_valid.columns and not pres_valid.empty:
                        for odor, grp in pres_valid.groupby('odor_name'):
                            _stats_line(grp['poke_time_ms'], f"  - Odor {odor}")
                else:
                    print("\nPoke Times for all Odors except aborted: n=0 (no presentations info)")
            else:
                print("\n Poke Times for all Odors except aborted: presentations not attached in aborted_sequences_detailed")

            # Last Odor Poke Times by abortion type
            if 'last_odor_poke_time_ms' in ab_det.columns and 'abortion_type' in ab_det.columns:
                print("\nAborted Odor Poke Times:")
                _stats_line(ab_det.loc[ab_det['abortion_type'] == 'reinitiation_abortion', 'last_odor_poke_time_ms'],
                            "  - Re-Initiation Abortions")
                _stats_line(ab_det.loc[ab_det['abortion_type'] == 'initiation_abortion', 'last_odor_poke_time_ms'],
                            "  - Initiation Abortions")

            # Counts by last odor
            if 'last_odor_name' in ab_det.columns and 'abortion_type' in ab_det.columns:
                print("\nCounts by last odor:")
                by_odor = (
                    ab_det
                    .groupby(['last_odor_name', 'abortion_type'])
                    .size()
                    .unstack(fill_value=0)
                    .rename(columns={'reinitiation_abortion': 'Re-initiation', 'initiation_abortion': 'Initiation'})
                )
                totals = ab_det.groupby('last_odor_name').size()
                for odor in totals.index:
                    rei_c = int(by_odor.loc[odor].get('Re-initiation', 0))
                    ini_c = int(by_odor.loc[odor].get('Initiation', 0))
                    tot = int(totals.loc[odor])
                    print(f"  - {odor}: {tot} abortions, Re-initiation {rei_c}, Initiation {ini_c}")

            # Counts by last position
            if 'last_odor_position' in ab_det.columns and 'abortion_type' in ab_det.columns:
                print("\nCounts by last position:")
                by_pos = (
                    ab_det
                    .groupby(['last_odor_position', 'abortion_type'])
                    .size()
                    .unstack(fill_value=0)
                    .rename(columns={'reinitiation_abortion': 'Re-initiation', 'initiation_abortion': 'Initiation'})
                )
                totals_pos = ab_det.groupby('last_odor_position').size()
                for pos in sorted(totals_pos.index):
                    rei_c = int(by_pos.loc[pos].get('Re-initiation', 0))
                    ini_c = int(by_pos.loc[pos].get('Initiation', 0))
                    tot = int(totals_pos.loc[pos])
                    print(f"  - Position {int(pos)}: {tot} abortions, Re-initiation {rei_c}, Initiation {ini_c}")

    print(buffer.getvalue())
    if save:
        # Determine output directory
        if out_dir is None:
            # Try to resolve from classification dict
            root = merged_classification.get("paths", {}).get("rawdata_dir", None)
            if root is None:
                root = merged_classification.get("rawdata_dir", None)
            if root is None:
                print("No output directory found for saving summary.")
                return
            out_dir = layout.results_dir(Path(root).parent / "derivatives" / f"sub-{subjid}" / f"ses-{date}")
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        fname = f"merged_summary_{subjid}_{date}.txt"
        with open(layout.write_path(out_dir, fname), "w", encoding="utf-8") as f:
            f.write(buffer.getvalue())
        print(f"Saved merged session summary to {layout.table_path(out_dir, fname)}")
    
