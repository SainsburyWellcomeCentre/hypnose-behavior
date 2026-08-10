# Defers evaluation of PEP-604 annotations (`X | None`), keeping this module
# importable on Python 3.9 for repos pinned there (hypnose-eeg-preprocessing).
from __future__ import annotations

import sys
import os
from pathlib import Path
from importlib.resources import files


SCHEMA_DIR = files("hypnose_behavior.resources.device_schemas")
BEHAVIOR_SCHEMA_PATH = SCHEMA_DIR / "behavior.yml"
OLFACTOMETER_SCHEMA_PATH = SCHEMA_DIR / "olfactometer.yml"


from hypnose_behavior.io.paths import get_rawdata_root, get_derivatives_root, get_server_root
import json
from dotmap import DotMap
import pandas as pd
import numpy as np
import math
from glob import glob
from aeon.io.reader import Reader, Csv
import aeon.io.api as api
import re
import yaml
import harp
import datetime
from datetime import timezone
import zoneinfo
import hypnose_behavior.trial_classification.detect_settings as detect_settings
import hypnose_behavior.trial_classification.detect_stage as detect_stage_module
import hypnose_behavior.trial_classification.windows as windows
from hypnose_behavior.trial_classification.outcome import classify_completed_trial, latency_label
from datetime import datetime, timezone, date
from collections import defaultdict
from bisect import bisect_left, bisect_right
from typing import Iterable, Optional
import io
import contextlib
from collections.abc import Mapping
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
import cv2 
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
from IPython import get_ipython

# The pre-odor grace window and the poke/valve window primitives moved to
# trial_classification/windows.py, which is a leaf (DECISIONS section 3).


# ============== General Utility Functions and Class Definitions =======================================
def vprint(verbose: bool, *args, **kwargs):
    if verbose:
        print(*args, **kwargs)


def _ensure_int_list(value, *, subtract_one: bool = False) -> list[int]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        candidates = value
    else:
        candidates = [value]
    out: list[int] = []
    for item in candidates:
        if item is None:
            continue
        try:
            number = int(item)
            if subtract_one:
                number -= 1
            out.append(number)
        except (TypeError, ValueError):
            continue
    return out


def _resolve_hidden_rule_from_stage(stage) -> tuple[list[int], str | None]:
    """Infer hidden-rule indices (possibly multiple) and stage name from metadata."""
    sequence_name = None
    indices: list[int] = []

    if isinstance(stage, Mapping):
        sequence_name = stage.get('stage_name') or stage.get('name')

        indices = _ensure_int_list(stage.get('hidden_rule_indices'))
        if not indices:
            indices = _ensure_int_list(stage.get('hidden_rule_index'))
        if not indices:
            indices = _ensure_int_list(stage.get('hidden_rule_positions'), subtract_one=True)
        if not indices:
            indices = _ensure_int_list(stage.get('hidden_rule_position'), subtract_one=True)

        if sequence_name is None:
            sequence_name = stage.get('name') or str(stage)
    else:
        sequence_name = str(stage)

    if sequence_name and not indices:
        match = re.search(r'location([0-9]+)', sequence_name, re.IGNORECASE)
        if match:
            digits = match.group(1)
            if digits:
                indices = [int(ch) for ch in digits if ch.isdigit()]

    indices = sorted({idx for idx in indices if isinstance(idx, int)})
    return indices, sequence_name


def _drop_final_hidden_rule_index(hidden_rule_indices, schema_settings, is_single_reward):
    """Remove the final-sequence-position index from hidden-rule candidates.

    The final position of a full sequence is always the reward position, so a
    rewarded odor appearing there is the normal reward, never a hidden rule. The
    final position index is detected in ``detect_settings`` (``finalPositionIndex``,
    == sequenceLength - 1) so it is not hardcoded here. The single-reward protocol
    is left untouched (its final position is not always rewarded).
    """
    if is_single_reward:
        return list(hidden_rule_indices)
    final_idx = schema_settings.get('finalPositionIndex')
    if final_idx is None:
        return list(hidden_rule_indices)
    return [idx for idx in hidden_rule_indices if idx != final_idx]


def _get_single_reward_info(root) -> tuple[bool, frozenset, frozenset]:
    """Determine whether a session uses the single-reward protocol and list its sequences.

    The single-reward protocol is the new task variant where NOT all candidate sequences are
    rewarded at their final position (e.g. ``singrew-task-stage1``: only ``OdorC-OdorF-OdorA``
    and ``OdorG-OdorE-OdorB`` are rewarded out of 8 candidate triples). It also covers older
    single-odor go/no-go stages (``FreeRun_StageN``). Detection is purely schema-based.

    Returns
    -------
    (is_single_reward, rewarded_sequences, all_sequences)
        is_single_reward : bool
            True iff at least one candidate sequence is NOT rewarded at its final position.
            For the default protocol (every sequence rewarded at the end) this is False and the
            false-response logic is never entered, so behaviour/output stay byte-for-byte unchanged.
        rewarded_sequences : frozenset[tuple[str, ...]]
            Concrete odor-name sequences whose final position is rewarded. A completed trial is
            "rewarded-type" iff ``tuple(its odor_sequence)`` is in this set; otherwise it is a
            non-rewarded ("no-go") sequence and gets false-response handling.
        all_sequences : frozenset[tuple[str, ...]]
            All concrete candidate sequences for the protocol (rewarded and non-rewarded).
            Used to compute reward determinacy from a presented (possibly partial) sequence:
            see ``_classify_reward_determinacy``.
    """
    try:
        _, schema_settings = detect_settings.detect_settings(Path(root))
    except Exception:
        return False, frozenset(), frozenset()
    if not schema_settings.get('isSingleRewardProtocol'):
        return False, frozenset(), frozenset()
    rewarded = schema_settings.get('rewardedSequences') or []
    rewarded_set = frozenset(tuple(seq) for seq in rewarded if seq)
    all_seqs = schema_settings.get('allSequences') or []
    all_set = frozenset(tuple(seq) for seq in all_seqs if seq)
    # If parsing produced no rewarded sequences, fall back to default behaviour (do nothing new).
    if not rewarded_set:
        return False, frozenset(), frozenset()
    return True, rewarded_set, all_set


def _classify_reward_determinacy(odor_sequence, all_sequences, rewarded_sequences):
    """Classify a presented (possibly partial) odor sequence by whether its reward outcome
    is already determined, using the protocol's full candidate set.

    Looks at every candidate in ``all_sequences`` that starts with the presented prefix
    ``odor_sequence`` (so it works for both completed sequences and partial/aborted ones):

    - all matching candidates rewarded            -> ``"rewarded"``
    - all matching candidates non-rewarded         -> ``"nonrewarded"``
    - matching candidates of both kinds            -> ``"ambiguous"``
    - no candidate starts with this prefix         -> ``"off_protocol"``

    Because it is driven entirely by the schema's candidate set, it adapts to any protocol.
    A completed full sequence is always determined (``rewarded`` / ``nonrewarded``); an
    aborted prefix may still be ``ambiguous``.

    Returns
    -------
    (label, determinacy_position, determined_final_odor)
        label : str | None
            One of the four labels above; ``None`` if the candidate set is empty.
        determinacy_position : int | None
            The 1-based position at which the outcome first became determined while reading
            the presented odors left to right (i.e. the earliest prefix length whose matching
            candidates are all the same reward type). ``None`` when never determined within the
            presented odors (``ambiguous``) or ``off_protocol``.
        determined_final_odor : str | None
            The final (last-position) odor of the matching candidates, but only when ALL of
            them share the same last odor -- i.e. the eventual reward port is already
            guaranteed by the presented prefix (e.g. "the sequence is bound to end in OdorA").
            ``None`` when the final odor is not yet pinned down or ``off_protocol``. This is
            strictly finer than ``label``: a sequence can be reward-determined yet have an
            ambiguous final odor (e.g. both candidates non-rewarded but ending A vs B).
    """
    if not all_sequences:
        return None, None, None
    prefix = tuple(odor_sequence) if odor_sequence else tuple()

    def _matches(prefix_t):
        return [s for s in all_sequences if len(s) >= len(prefix_t) and s[:len(prefix_t)] == prefix_t]

    matches = _matches(prefix)
    if not matches:
        return "off_protocol", None, None

    flags = {s in rewarded_sequences for s in matches}
    if len(flags) == 1:
        label = "rewarded" if flags.pop() else "nonrewarded"
    else:
        label = "ambiguous"

    # Earliest position at which the outcome became determined while reading the prefix.
    determinacy_position = None
    for k in range(1, len(prefix) + 1):
        k_matches = _matches(prefix[:k])
        if not k_matches:
            break
        k_flags = {s in rewarded_sequences for s in k_matches}
        if len(k_flags) == 1:
            determinacy_position = k
            break

    # Final odor only if every matching candidate ends with the same odor.
    final_odors = {s[-1] for s in matches if s}
    determined_final_odor = final_odors.pop() if len(final_odors) == 1 else None

    return label, determinacy_position, determined_final_odor


# ================= Functions for Trial Analysis and Classification ========================


def _sampling_parameters_ms(root, *, task: str):
    """Schema sampling parameters, converted to milliseconds.

    Returns ``(sample_offset_time_ms, minimum_sampling_time_ms_by_odor, default_minimum_ms,
    response_time_sec)``. ``task`` names the caller in the error message only, so the three
    call sites keep the wording they had. ``abortion_classification`` does not use this: it
    overlays thresholds carried on the classification dict before deciding whether the set is
    empty, and takes its default from there rather than from the maximum.
    """
    sample_offset_time, minimum_sampling_time_by_odor, response_time = get_experiment_parameters(root)
    sample_offset_time_ms = sample_offset_time * 1000
    minimum_sampling_time_ms_by_odor = {
        str(odor): float(threshold) * 1000.0
        for odor, threshold in (minimum_sampling_time_by_odor or {}).items()
        if threshold is not None
    }
    if not minimum_sampling_time_ms_by_odor:
        raise ValueError(
            f"minimumSamplingTime_by_odor missing or empty in schema; cannot {task} without per-odor thresholds"
        )
    default_minimum_sampling_time_ms = max(minimum_sampling_time_ms_by_odor.values())
    return (
        sample_offset_time_ms,
        minimum_sampling_time_ms_by_odor,
        default_minimum_sampling_time_ms,
        response_time,
    )


def _detect_stage_name(stage, root) -> str | None:
    """Stage name for protocol detection: the passed-in stage first, re-detection second."""
    stage_name = None
    if stage is not None:
        if isinstance(stage, Mapping):
            stage_name = stage.get('stage_name') or stage.get('name')
        else:
            stage_name = getattr(stage, 'stage_name', None) or getattr(stage, 'name', None)
            if stage_name is None:
                stage_name = str(stage)

    if not stage_name:
        try:
            stage_detected = detect_stage_module.detect_stage(root)
            stage_name = stage_detected.get('stage_name') if isinstance(stage_detected, Mapping) else None
        except Exception:
            stage_name = None
    return stage_name


def _valve_attempt_windows(valve_events, initiation_time, next_initiation_time, poke_periods):
    """One sampling attempt per valve opening in the inter-initiation window.

    Ends are capped at ``next_initiation_time``. When the session has no valve record at all,
    falls back to a single attempt spanning from the first poke to the next initiation, so
    detection still runs on the poke stream alone.
    """
    attempt_events = [
        {
            'start_time': ev['start_time'],
            'end_time': min(ev['end_time'], next_initiation_time),
            'odor_name': ev['odor_name'],
        }
        for ev in valve_events
        if ev['start_time'] >= initiation_time and ev['start_time'] < next_initiation_time
    ]

    if not attempt_events:
        attempt_events = [{
            'start_time': poke_periods[0][0],
            'end_time': next_initiation_time,
            'odor_name': None,
        }]
    return attempt_events


def _record_detected_trial(trials, initiated_sequences, *, initiation_time, start, end,
                           duration_ms, attempt_number, required_min_ms, odor_name,
                           fallback_reason=None):
    """Append the matching ``trials`` and ``initiated_sequences`` rows for one detected trial.

    The two rows carry the same facts under different names (``trial_start`` vs
    ``sequence_start``). Key insertion order becomes DataFrame column order downstream, so it
    is reproduced here rather than tidied, and ``fallback_reason`` stays last and absent unless
    the trial came from a fallback.
    """
    trial_id = len(trials)
    trial_entry = {
        'initiation_sequence_time': initiation_time,
        'trial_start': start,
        'trial_end': end,
        'continuous_poke_time_ms': duration_ms,
        'trial_id': trial_id,
        'attempt_number': attempt_number,
        'required_min_sampling_time_ms': required_min_ms,
        'odor_name': odor_name,
    }
    initiated_sequence_entry = {
        'initiation_sequence_time': initiation_time,
        'sequence_start': start,
        'sequence_end': end,
        'continuous_poke_time_ms': duration_ms,
        'trial_id': trial_id,
        'attempt_number': attempt_number,
        'timestamp': start,
        'required_min_sampling_time_ms': required_min_ms,
        'odor_name': odor_name,
    }
    if fallback_reason is not None:
        trial_entry['fallback_reason'] = fallback_reason
        initiated_sequence_entry['fallback_reason'] = fallback_reason

    trials.append(trial_entry)
    initiated_sequences.append(initiated_sequence_entry)


def _run_sampling_attempts(attempt_events, poke_periods, cue_pokes, initiation_time,
                           next_initiation_time, *, required_min_ms_for, sample_offset_time_ms,
                           verbose):
    """Walk one initiation's attempts until one reaches its odor's minimum sampling time.

    Returns ``(winner, failed_attempts, pending_failed_attempt, attempt_num)``. ``winner`` is
    ``None`` when no attempt succeeded, otherwise a dict of the facts needed to record a trial.

    Two ways to win. The plain one is reaching the threshold. The other is the *pending* rule:
    the most recent failure is promoted to a trial if the next attempt presents a **different**
    odor -- the sequence moved on, so the animal did sample, and the short measurement is an
    artefact rather than a non-initiation.
    """
    attempt_num = 0
    failed_attempts: list[dict] = []
    pending_failed_attempt: dict | None = None
    attempt_next_start = {
        idx + 1: (attempt_events[idx + 1]['start_time'] if idx + 1 < len(attempt_events) else None)
        for idx in range(len(attempt_events))
    }

    for attempt_event in attempt_events:
        attempt_num += 1
        event_start = attempt_event['start_time']
        event_end = attempt_event['end_time']
        if event_end <= event_start:
            continue

        attempt_odor = attempt_event['odor_name']
        required_minimum_ms = required_min_ms_for(attempt_odor)

        if pending_failed_attempt is not None:
            pending_odor = pending_failed_attempt.get('odor_name')
            if attempt_odor is not None and (pending_odor is None or attempt_odor != pending_odor):
                vprint(verbose, "    Fallback: subsequent distinct valve detected — counting trial despite short sampling")
                if failed_attempts and failed_attempts[-1] is pending_failed_attempt:
                    failed_attempts.pop()
                winner = {
                    'start': pending_failed_attempt.get('attempt_start', event_start),
                    'duration_ms': pending_failed_attempt.get('continuous_poke_time_ms', 0.0),
                    'attempt_number': pending_failed_attempt.get('attempt_number', 1),
                    'required_min_ms': pending_failed_attempt.get('required_min_sampling_time_ms', required_minimum_ms),
                    'odor_name': pending_failed_attempt.get('odor_name'),
                }
                return winner, failed_attempts, None, attempt_num

        if verbose:
            odor_msg = f", odor={attempt_odor}" if attempt_odor else ""
            print(f"    Attempt {attempt_num}: valve opens at {event_start} (min={required_minimum_ms:.1f}ms{odor_msg})")

        segments = windows.poke_segments_in_valve_window(
            poke_periods, cue_pokes, event_start, event_end, next_initiation_time
        )
        attempt_start = segments[0][0] if segments else event_start

        def _report(seg_idx, gap_ms, seg_duration_ms, running_total_ms):
            if not verbose:
                return
            if gap_ms is None:
                print(f"      Segment {seg_idx}: {seg_duration_ms:.1f}ms (total {running_total_ms:.1f}ms)")
            elif seg_duration_ms is None:
                print(f"      Gap {gap_ms:.1f}ms ≥ {sample_offset_time_ms}ms — sequence ends")
            else:
                print(f"      Segment {seg_idx}: gap {gap_ms:.1f}ms + {seg_duration_ms:.1f}ms (total {running_total_ms:.1f}ms)")

        continuous_time, last_seg_end, success = windows.accumulate_sampling_time(
            segments, sample_offset_time_ms, required_minimum_ms, on_segment=_report
        )

        if success:
            vprint(verbose, f"      SUCCESS: {continuous_time:.1f}ms ≥ {required_minimum_ms:.1f}ms")
            winner = {
                'start': attempt_start,
                'duration_ms': continuous_time,
                'attempt_number': attempt_num,
                'required_min_ms': required_minimum_ms,
                'odor_name': attempt_odor,
            }
            return winner, failed_attempts, None, attempt_num

        vprint(verbose, f"      FAILED: {continuous_time:.1f}ms < {required_minimum_ms:.1f}ms")
        failed_entry = {
            'initiation_sequence_time': initiation_time,
            'attempt_start': attempt_start,
            'attempt_end': last_seg_end if last_seg_end is not None else event_start,
            'continuous_poke_time_ms': continuous_time,
            'attempt_number': attempt_num,
            'timestamp': attempt_start,
            'failure_reason': 'insufficient_continuous_poke_time',
            'required_min_sampling_time_ms': required_minimum_ms,
            'odor_name': attempt_odor,
            'next_attempt_start': attempt_next_start.get(attempt_num),
        }
        failed_attempts.append(failed_entry)
        pending_failed_attempt = failed_entry

    return None, failed_attempts, pending_failed_attempt, attempt_num


def _await_reward_promotes_attempt(failed_attempts, pending_failed_attempt, await_reward_times,
                                   next_initiation_time, default_minimum_sampling_time_ms):
    """Promote a failed attempt to a trial when an AwaitReward event followed it.

    Only reached on odour-discrimination protocols, where a single short odor presentation can
    still be a real trial: the animal committed and the task emitted AwaitReward, so the
    sampling-time threshold is the wrong evidence. Prefers the pending (most recent) failure.

    Returns ``(winner, candidate)``, both ``None`` when no AwaitReward falls in the window.
    """
    candidate = None
    if pending_failed_attempt is not None:
        for fa in reversed(failed_attempts):
            if fa is pending_failed_attempt:
                candidate = fa
                break
    if candidate is None:
        candidate = failed_attempts[-1]

    attempt_start = candidate.get('attempt_start') or candidate.get('timestamp')
    if attempt_start is None:
        return None, None
    try:
        start_ts = pd.Timestamp(attempt_start)
    except Exception:
        return None, None

    window_mask = await_reward_times >= start_ts
    if next_initiation_time is not None and not pd.isna(next_initiation_time):
        window_mask &= await_reward_times <= next_initiation_time
    if await_reward_times[window_mask].empty:
        return None, None

    winner = {
        'start': start_ts,
        'duration_ms': candidate.get('continuous_poke_time_ms', 0.0),
        'attempt_number': candidate.get('attempt_number', 1),
        'required_min_ms': candidate.get('required_min_sampling_time_ms', default_minimum_sampling_time_ms),
        'odor_name': candidate.get('odor_name'),
    }
    return winner, candidate


def detect_trials(data, events, root, odor_map, verbose=True, stage=None):
    """Detect initiated trials from cue-poke and valve streams.

    One *attempt* is one valve opening between consecutive InitiationSequence events. An
    attempt initiates a trial when the animal's cue-port poke reaches the minimum sampling
    time for that attempt's odor, where pokes separated by gaps shorter than
    ``sampleOffsetTime`` count as one continuous sample. The first attempt to reach the
    threshold ends the search, and the remaining failures for that initiation are recorded as
    non-initiated sequences.

    Two fallbacks add a trial that the sampling threshold alone would reject: a following
    attempt with a *different* odor promotes the previous short attempt (the sequence moved
    on, so the sample was real), and on odour-discrimination protocols an AwaitReward event
    after the attempt does the same (the task itself decided a trial had happened).

    Returns a dict of ``trials`` / ``initiated_sequences`` / ``non_initiated_sequences``
    DataFrames.
    """
    (sample_offset_time_ms, minimum_sampling_time_ms_by_odor,
     default_minimum_sampling_time_ms, _response_time) = _sampling_parameters_ms(root, task="detect trials")

    def required_min_ms_for(odor_name):
        odor_key = str(odor_name) if odor_name is not None else None
        return minimum_sampling_time_ms_by_odor.get(odor_key, default_minimum_sampling_time_ms)

    protocol_name = (_detect_stage_name(stage, root) or "").lower()
    is_odour_discrimination = "odourdiscrimination" in protocol_name

    valve_events = windows.valve_windows_dropping_unclosed(
        (odor_map or {}).get('olfactometer_valves', {}) if odor_map is not None else {},
        (odor_map or {}).get('valve_to_odor', {}) if odor_map is not None else {},
    )

    if verbose:
        print("TRIAL DETECTION")
        print("=" * 60)
        print(f"Parameters: sample_offset_time={sample_offset_time_ms}ms")
        print("Per-odor minimum sampling times (ms):")
        for odor_name, threshold in sorted(minimum_sampling_time_ms_by_odor.items()):
            print(f"  - {odor_name}: {threshold:.1f}")

    initiation_events = events['combined_initiation_sequence_df'].copy()
    cue_pokes = data['digital_input_data']['DIPort0'].copy().astype(bool)

    await_reward_df = events.get('combined_await_reward_df') if isinstance(events, Mapping) else None
    if isinstance(await_reward_df, pd.DataFrame) and not await_reward_df.empty and 'Time' in await_reward_df.columns:
        await_reward_times = pd.to_datetime(await_reward_df['Time'], errors='coerce').dropna()
    else:
        await_reward_times = pd.Series(dtype='datetime64[ns]')

    trials = []
    initiated_sequences = []
    non_initiated_sequences = []

    for idx, initiation_row in initiation_events.iterrows():
        initiation_time = initiation_row['Time']
        if idx + 1 < len(initiation_events):
            next_initiation_time = initiation_events.iloc[idx + 1]['Time']
        else:
            next_initiation_time = cue_pokes.index[-1]

        vprint(verbose, f"\nInitiationSequence {idx}: {initiation_time}")

        period_pokes = cue_pokes[(cue_pokes.index > initiation_time) & (cue_pokes.index <= next_initiation_time)]
        if period_pokes.empty:
            vprint(verbose, "  No pokes found")
            continue

        poke_periods = windows.poke_periods(period_pokes)
        if not poke_periods:
            vprint(verbose, "  No complete poke periods found")
            continue

        vprint(verbose, f"  Found {len(poke_periods)} poke periods")

        attempt_events = _valve_attempt_windows(
            valve_events, initiation_time, next_initiation_time, poke_periods
        )

        winner, failed_attempts, pending_failed_attempt, _attempt_num = _run_sampling_attempts(
            attempt_events, poke_periods, cue_pokes, initiation_time, next_initiation_time,
            required_min_ms_for=required_min_ms_for,
            sample_offset_time_ms=sample_offset_time_ms,
            verbose=verbose,
        )

        if (
            winner is None
            and is_odour_discrimination
            and isinstance(failed_attempts, list)
            and failed_attempts
            and not await_reward_times.empty
        ):
            winner, candidate = _await_reward_promotes_attempt(
                failed_attempts, pending_failed_attempt, await_reward_times,
                next_initiation_time, default_minimum_sampling_time_ms,
            )
            if winner is not None:
                winner['fallback_reason'] = 'await_reward_event'
                vprint(verbose, "    Fallback: AwaitReward detected — counting trial despite short sampling")
                failed_attempts = [fa for fa in failed_attempts if fa is not candidate]

        if winner is not None:
            _record_detected_trial(
                trials, initiated_sequences,
                initiation_time=initiation_time,
                start=winner['start'],
                end=next_initiation_time,
                duration_ms=winner['duration_ms'],
                attempt_number=winner['attempt_number'],
                required_min_ms=winner['required_min_ms'],
                odor_name=winner['odor_name'],
                fallback_reason=winner.get('fallback_reason'),
            )

        non_initiated_sequences.extend(failed_attempts)
        if winner is None:
            vprint(verbose, "  No successful trial found for this initiation sequence")

    results = {
        'trials': pd.DataFrame(trials),
        'initiated_sequences': pd.DataFrame(initiated_sequences).sort_values('timestamp') if initiated_sequences else pd.DataFrame(),
        'non_initiated_sequences': pd.DataFrame(non_initiated_sequences).sort_values('timestamp') if non_initiated_sequences else pd.DataFrame()
    }

    vprint(verbose, "\n" + "="*50)
    vprint(verbose, "DETECTION SUMMARY:")
    vprint(verbose, f"Trials: {len(results['trials'])}")
    vprint(verbose, f"Initiated sequences: {len(results['initiated_sequences'])}")
    vprint(verbose, f"Non-initiated sequences: {len(results['non_initiated_sequences'])}")
    vprint(verbose, "="*50)

    return results

def get_experiment_parameters(root):
    """
    Extract parameters from schema, including per-odor minimum sampling times.

    Returns:
        tuple: (sampleOffsetTime, minimumSamplingTime_by_odor, responseTime)
    """
    session_settings, session_schema = detect_settings.detect_settings(root)

    def _coerce_to_float(value):
        """Best-effort conversion of nested DotMap/dict/list structures to a float."""
        if value is None:
            return None
        # Handle numpy scalars
        if isinstance(value, (np.floating, np.integer)):
            return float(value)
        # Primitive numbers
        if isinstance(value, (int, float)):
            return float(value)
        # DotMap behaves like a dict; convert to plain dict first
        if isinstance(value, DotMap):
            value = value.toDict()
        if isinstance(value, dict):
            # Prefer common scalar keys if present
            for key in ('value', 'seconds', 'Seconds', 'ms', 'milliseconds'):
                if key in value:
                    coerced = _coerce_to_float(value[key])
                    if coerced is not None:
                        return coerced
            # Fall back to any single numeric value stored in the mapping
            numeric_vals = [v for v in value.values() if isinstance(v, (int, float, np.integer, np.floating))]
            if len(numeric_vals) == 1:
                return float(numeric_vals[0])
            if len(numeric_vals) > 1:
                # Ambiguous but still try first value deterministically
                return float(numeric_vals[0])
            # Walk nested structures if needed
            for nested in value.values():
                coerced = _coerce_to_float(nested)
                if coerced is not None:
                    return coerced
            return None
        if isinstance(value, (list, tuple)):
            for item in value:
                coerced = _coerce_to_float(item)
                if coerced is not None:
                    return coerced
            return None
        return None

    # sampleOffsetTime lives per-segment in the Schema (newer sessions); detect_settings
    # resolves it and falls back to the legacy SessionSettings location for older sessions.
    sample_offset_time = _coerce_to_float(session_schema.get('sampleOffsetTime'))
    if sample_offset_time is None:
        # Last-resort direct read of SessionSettings (handles nested metadata DotMaps)
        session_meta = session_settings.iloc[0]['metadata']
        sample_offset_time = _coerce_to_float(getattr(session_meta, 'sampleOffsetTime', None))
        if sample_offset_time is None:
            nested_meta = getattr(session_meta, 'metadata', None)
            sample_offset_time = _coerce_to_float(getattr(nested_meta, 'sampleOffsetTime', None)) if nested_meta else None
    if sample_offset_time is None:
        raise ValueError("sampleOffsetTime missing or invalid in Schema sequences and SessionSettings metadata")

    # Get per-odor minimumSamplingTime dict and ensure scalar values
    raw_minimums = session_schema.get('minimumSamplingTime_by_odor', {}) or {}
    minimumSamplingTime_by_odor = {}
    for odor, threshold in raw_minimums.items():
        coerced = _coerce_to_float(threshold)
        if coerced is not None:
            minimumSamplingTime_by_odor[str(odor)] = coerced

    response_time = _coerce_to_float(session_schema.get('responseTime'))

    return sample_offset_time, minimumSamplingTime_by_odor, response_time

def _hidden_rule_indices_from_stage_or_schema(stage, root):
    """Hidden-rule indices from the stage name, falling back to the schema's inferred set.

    Returns ``(indices, sequence_name, schema_settings, schema_err)`` with the indices sorted
    and de-duplicated but the **final-position index still present**. Dropping it is left to
    the caller because the two callers do it at different moments: ``analyze_response_times``
    drops before printing its header, ``classify_trials`` after. Their headers therefore differ
    on a session whose final position is a hidden-rule candidate -- existing behaviour, not
    something to tidy here.
    """
    hidden_rule_indices, sequence_name = _resolve_hidden_rule_from_stage(stage)

    schema_settings = {}
    schema_err: Exception | None = None
    try:
        _, schema_settings = detect_settings.detect_settings(root)
    except Exception as exc:
        schema_err = exc
        schema_settings = {}

    if not hidden_rule_indices:
        inferred_indices = schema_settings.get('hiddenRuleIndicesInferred')
        if inferred_indices is None:
            inferred_indices = schema_settings.get('hiddenRuleIndexInferred')
        hidden_rule_indices = _ensure_int_list(inferred_indices)

    hidden_rule_indices = sorted({idx for idx in hidden_rule_indices if isinstance(idx, int)})
    return hidden_rule_indices, sequence_name, schema_settings, schema_err


def _hidden_rule_odor_set(hidden_rule_indices, schema_settings, schema_err, verbose):
    """The odors that count as a hidden-rule hit, or ``None`` when there is no hidden rule.

    Raises rather than degrading: a session with a hidden-rule position but no inferable odor
    identities would silently score every trial as a miss, so it is stopped instead.
    """
    if not hidden_rule_indices:
        return None
    try:
        if schema_err is not None and 'hiddenRuleOdorsInferred' not in schema_settings:
            raise ValueError(str(schema_err))
        odors = (schema_settings.get('hiddenRuleOdorsInferred') or [])
        if len(odors) < 2:
            raise ValueError("found fewer than two rewarded odors at inferred hidden rule position.")
        hr_odor_set = set(map(str, odors))
        if verbose:
            print(f"Hidden Rule Odors inferred: {sorted(hr_odor_set)}")
        return hr_odor_set
    except Exception as e:
        raise ValueError(f"Hidden Rule Odor Identities could not be inferred from Schema: {e}")


def _check_hidden_rule(odor_sequence, candidate_indices, odor_set):
    """``(enough_odors, hit, matching_indices)`` for one presented sequence.

    ``enough_odors`` is False when the sequence is too short to reach any candidate position,
    which is what separates "did not hit the hidden rule" from "never got the chance".
    """
    if not candidate_indices or odor_set is None:
        return False, False, []

    valid_indices = [idx for idx in candidate_indices if 0 <= idx < len(odor_sequence)]
    if not valid_indices:
        return False, False, []

    matching_indices = sorted({idx for idx in valid_indices if odor_sequence[idx] in odor_set})
    return True, bool(matching_indices), matching_indices


def _hidden_rule_positions(hidden_rule_indices):
    """``(positions, first_index, first_position, multiple)`` -- positions are 1-based indices."""
    positions = [idx + 1 for idx in hidden_rule_indices]
    location = hidden_rule_indices[0] if hidden_rule_indices else None
    position = positions[0] if positions else None
    return positions, location, position, len(positions) > 1


def _print_hidden_rule_header(hidden_rule_indices, hidden_rule_positions, sequence_name, stage,
                              *, label_prefix):
    """The 'Hidden rule location(s) extracted' header shared by the two classifiers.

    ``label_prefix`` differs by one space between the two call sites ("Location{n}" vs
    "Location {n}"), so it is passed rather than assumed.
    """
    seq_label = sequence_name or str(stage)
    if not hidden_rule_indices:
        print(f"No Hidden Rule Location found in sequence name: {seq_label}. Proceeding without Hidden Rule analysis.")
        return
    if len(hidden_rule_positions) > 1:
        pos_str = ", ".join(str(p) for p in hidden_rule_positions)
        idx_str = ", ".join(str(idx) for idx in hidden_rule_indices)
        print(f"Hidden rule locations extracted: Positions {pos_str} (indices {idx_str})")
    else:
        location = hidden_rule_indices[0]
        position = hidden_rule_positions[0]
        print(f"Hidden rule location extracted: {label_prefix}{location} (index {location}, position {position})")


def _next_after(sorted_series, ts):
    """First entry of a sorted series strictly after ``ts``, or ``None``."""
    if sorted_series is None or sorted_series.empty:
        return None
    later = sorted_series[sorted_series > ts]
    return later.iloc[0] if not later.empty else None


def _recording_end(initiation_starts_sorted, cue_poke_starts_sorted, supply_port1_times,
                   supply_port2_times, port1_pokes, port2_pokes, trial_end):
    """Latest timestamp any stream reaches, used to bound a reward window with no next trial.

    Falls back to ``trial_end`` when every stream is empty.
    """
    candidates = [
        initiation_starts_sorted.iloc[-1] if not initiation_starts_sorted.empty else None,
        cue_poke_starts_sorted.iloc[-1] if not cue_poke_starts_sorted.empty else None,
        supply_port1_times[-1] if supply_port1_times else None,
        supply_port2_times[-1] if supply_port2_times else None,
        port1_pokes.index.max() if not port1_pokes.empty else None,
        port2_pokes.index.max() if not port2_pokes.empty else None,
        trial_end,
    ]
    candidates = [c for c in candidates if c is not None and not pd.isna(c)]
    return max(candidates) if candidates else trial_end


def _odourdisc_reward_window_end(next_init, next_cue_after_next_init, await_time,
                                 cue_poke_starts_sorted, recording_end):
    """End of the reward window on odour-discrimination protocols.

    These sessions have no fixed response window: the animal may collect at any point before it
    re-engages, so the window runs to the later of the next initiation and the first cue poke
    after it. With no next initiation it runs to the next cue poke, or to the end of the
    recording. Never earlier than ``await_time``.

    Returns ``(reward_window_end, next_cue_poke)``.
    """
    if next_init is not None:
        candidates = [c for c in (next_init, next_cue_after_next_init) if c is not None]
        reward_window_end = max(candidates) if candidates else next_init
        next_cue_poke = next_cue_after_next_init
    else:
        next_cue_poke = _next_after(cue_poke_starts_sorted, await_time)
        reward_window_end = next_cue_poke if next_cue_poke is not None else recording_end

    if reward_window_end < await_time:
        reward_window_end = await_time
    return reward_window_end, next_cue_poke



def _assign_positions_to_valve_events(trial_valve_events, max_positions, required_min_ms_for):
    """Map a trial's valve activations onto sequence positions 1..max_positions.

    Position 1 is the **last** activation of the opening odor: the animal may sniff and leave
    several times before committing, and the trial starts at the final one. The earlier
    activations of that same valve become ``prior_presentations`` -- failed Position-1 attempts
    that are reported separately as non-initiated.

    Later positions come from the event list with *consecutive* repeats of an odor collapsed to
    their last activation. A non-consecutive re-entry of an odor is a new position, which is
    what distinguishes this from ``windows.first_occurrence_positions``.

    Returns ``(position_locations, prior_presentations)``.
    """
    position_locations: dict[int, dict] = {}
    prior_presentations: list[dict] = []

    if trial_valve_events:
        # Position 1 is keyed on the valve, not the odor name: two valves can carry the same
        # odor, and only a repeat of the same physical valve is a re-attempt.
        first_odor_valve = trial_valve_events[0]['valve_key']
        first_odor_activations = []
        for event in trial_valve_events:
            if event['valve_key'] == first_odor_valve:
                first_odor_activations.append(event)
            else:
                break
        if first_odor_activations:
            position_locations[1] = first_odor_activations[-1]
            prior_presentations = [
                {
                    'position': 1,
                    'odor_name': e['odor_name'],
                    'valve_start': e['start_time'],
                    'valve_end': e['end_time'],
                    'required_min_sampling_time_ms': required_min_ms_for(e['odor_name']),
                }
                for e in first_odor_activations[:-1]
            ]

    dedup_events = windows.collapse_consecutive_odors(trial_valve_events)
    next_pos = 2
    for event in dedup_events[1:]:
        if next_pos > max_positions:
            break
        position_locations[next_pos] = event
        next_pos += 1

    return position_locations, prior_presentations


def _position_valve_times(position_locations, max_positions, prior_presentations, required_min_ms_for):
    """Valve open window per position. Position 1 also carries its failed earlier attempts."""
    position_valve_times = {}
    for position in range(1, max_positions + 1):
        loc = position_locations.get(position)
        if loc is None:
            continue
        valve_start = loc['start_time']
        valve_end = loc['end_time']
        entry = {
            'position': position,
            'odor_name': loc['odor_name'],
            'valve_start': valve_start,
            'valve_end': valve_end,
            'valve_duration_ms': (valve_end - valve_start).total_seconds() * 1000,
            'required_min_sampling_time_ms': required_min_ms_for(loc['odor_name']),
        }
        if position == 1:
            entry['prior_presentations'] = prior_presentations
        position_valve_times[position] = entry
    return position_valve_times


def _position_poke_times(position_locations, poke_data, max_positions, sample_offset_time_ms,
                         required_min_ms_for):
    """Cue-port poke time per position, measured inside that position's valve window.

    Reports the **first merged block** of poking: intervals separated by less than
    ``sampleOffsetTime`` are one sample, and the first gap beyond it ends the measurement, so a
    later return to the port during the same odor does not inflate the sampling time.

    A position whose block is zero-length is left out entirely, which is what makes
    ``valid_positions`` downstream shorter than the valve sequence. The exception is the
    pre-odor grace window: a poke that ended within ``PRE_ODOR_GRACE_MS`` before the valve
    opened still counts, anchored at the window start.
    """
    position_poke_times = {}
    s_bool = poke_data.astype(bool)

    def _grace_entry(position, loc, odor_start, odor_end):
        grace_ms, grace_end = windows.grace_poke_ms(s_bool, odor_start, odor_end)
        if grace_ms <= 0.0:
            return None
        return {
            'position': position,
            'odor_name': loc['odor_name'],
            'poke_time_ms': grace_ms,
            'poke_odor_start': odor_start,
            'poke_odor_end': grace_end,
            'poke_first_in': odor_start,
            'required_min_sampling_time_ms': required_min_ms_for(loc['odor_name']),
        }

    for position in range(1, max_positions + 1):
        loc = position_locations.get(position)
        if loc is None:
            continue
        odor_start = loc['start_time']
        odor_end = loc['end_time']

        intervals, _first_in = windows.poke_intervals_in_window(s_bool, odor_start, odor_end)
        if not intervals:
            entry = _grace_entry(position, loc, odor_start, odor_end)
            if entry is not None:
                position_poke_times[position] = entry
            continue

        merged = windows.merge_short_gaps(intervals, sample_offset_time_ms)
        first_block_start, first_block_end = merged[0]
        consolidated_poke_time_ms = (first_block_end - first_block_start).total_seconds() * 1000.0

        if consolidated_poke_time_ms > 0:
            position_poke_times[position] = {
                'position': position,
                'odor_name': loc['odor_name'],
                'poke_time_ms': consolidated_poke_time_ms,
                # Actual poke entry/exit times rather than the valve window edges.
                'poke_odor_start': first_block_start,
                'poke_odor_end': first_block_end,
                'poke_first_in': first_block_start,
                'required_min_sampling_time_ms': required_min_ms_for(loc['odor_name']),
            }

    return position_poke_times


def _build_presentations(valid_positions, position_valve_times, position_poke_times):
    """One row per sampled position, in presentation order. Returns ``(rows, last_event_index)``."""
    num_positions = len(valid_positions)
    last_event_index = num_positions - 1 if num_positions else None

    presentations = []
    for idx_in_trial, pos in enumerate(valid_positions):
        valve_info = position_valve_times.get(pos) or {}
        poke_info = position_poke_times.get(pos) or {}
        presentations.append({
            'index_in_trial': idx_in_trial,
            'position': pos,
            'odor_name': valve_info.get('odor_name'),
            'valve_start': valve_info.get('valve_start'),
            'valve_end': valve_info.get('valve_end'),
            'valve_duration_ms': float(valve_info.get('valve_duration_ms', 0.0) or 0.0),
            'poke_time_ms': float(poke_info.get('poke_time_ms', 0.0) or 0.0),
            'poke_first_in': poke_info.get('poke_first_in'),
            'required_min_sampling_time_ms': valve_info.get('required_min_sampling_time_ms'),
            'is_last_event': last_event_index is not None and idx_in_trial == last_event_index,
        })
    return presentations, last_event_index


def _hidden_rule_success(hr_hit_positions, last_position, max_positions, has_await_reward):
    """Did the animal act on the hidden rule? Returns ``(success, position)``.

    Leaving before the full sequence only counts if AwaitReward actually fired -- otherwise the
    animal simply abandoned the trial at a position that happened to carry the hidden-rule
    odor. Reaching the final position with the hidden-rule odor there counts either way.
    """
    if not hr_hit_positions:
        return False, None

    first_hr_pos = min(hr_hit_positions)
    if last_position < max_positions:
        if has_await_reward:
            return True, first_hr_pos
        return False, None

    success = last_position in hr_hit_positions
    return success, (first_hr_pos if success else None)


def _supply_pulses_between(supply_port1_times, supply_port2_times, window_start, window_end):
    """Supply pulses in ``[window_start, window_end]`` as ``(times1, times2, tagged_sorted)``.

    The tagged list carries ``(timestamp, port_number, port_letter)`` so the first pulse
    identifies which port delivered.
    """
    supply1 = [t for t in supply_port1_times if window_start <= t <= window_end]
    supply2 = [t for t in supply_port2_times if window_start <= t <= window_end]
    tagged = [(t, 1, 'A') for t in supply1] + [(t, 2, 'B') for t in supply2]
    tagged.sort(key=lambda x: x[0])
    return supply1, supply2, tagged


def _reward_pokes_between(port1_pokes, port2_pokes, window_start, window_end):
    """Reward-port poke onsets in a window as ``(times1, times2, tagged_sorted)``."""
    pokes1 = windows.rising_edges_between(port1_pokes, window_start, window_end)
    pokes2 = windows.rising_edges_between(port2_pokes, window_start, window_end)
    tagged = [(t, 1, 'A') for t in pokes1] + [(t, 2, 'B') for t in pokes2]
    tagged.sort(key=lambda x: x[0])
    return pokes1, pokes2, tagged


def _record_supply_outcome(trial_dict, supply1, supply2, tagged_supply):
    """Write the rewarded-trial supply columns onto a trial record."""
    first_supply_time, first_supply_port, first_supply_odor = tagged_supply[0]
    trial_dict['first_supply_time'] = first_supply_time
    trial_dict['first_supply_port'] = first_supply_port
    trial_dict['first_supply_odor_identity'] = first_supply_odor
    trial_dict['supply1_count'] = len(supply1)
    trial_dict['supply2_count'] = len(supply2)
    trial_dict['total_supply_count'] = len(supply1) + len(supply2)


def _record_first_reward_poke(trial_dict, tagged_pokes):
    """Write the first reward-port poke onto a trial record, if there was one."""
    if not tagged_pokes:
        return
    (trial_dict['first_reward_poke_time'],
     trial_dict['first_reward_poke_port'],
     trial_dict['first_reward_poke_odor_identity']) = tagged_pokes[0]


def _score_odourdisc_outcome(trial_dict, *, await_time, reward_window_end, supply_port1_times,
                             supply_port2_times, port1_pokes, port2_pokes):
    """Outcome of one odour-discrimination trial. Returns ``'rewarded' | 'unrewarded' | 'timeout'``.

    Both the supply search and the poke search run over the **same** window, from AwaitReward
    to the end of the reward window -- these sessions have no fixed response deadline. The
    standard protocol instead searches supply to the trial end and pokes only to
    ``await + responseTime``, so the two rules are kept separate rather than parameterised.
    """
    supply1, supply2, tagged_supply = _supply_pulses_between(
        supply_port1_times, supply_port2_times, await_time, reward_window_end)
    pokes1, pokes2, tagged_pokes = _reward_pokes_between(
        port1_pokes, port2_pokes, await_time, reward_window_end)

    outcome = classify_completed_trial(
        supply_count=len(tagged_supply), reward_poke_count=len(tagged_pokes),
        has_await_reward=True)

    if outcome == 'rewarded':
        _record_supply_outcome(trial_dict, supply1, supply2, tagged_supply)
    elif outcome == 'unrewarded':
        _record_first_reward_poke(trial_dict, tagged_pokes)
        trial_dict['port1_pokes_count'] = len(pokes1)
        trial_dict['port2_pokes_count'] = len(pokes2)
        trial_dict['total_reward_pokes'] = len(tagged_pokes)
    return outcome


def _score_standard_outcome(trial_dict, *, await_reward_time, trial_end, supply_port1_times,
                            supply_port2_times, port1_pokes, port2_pokes, response_time_sec):
    """Outcome of one standard completed trial. Returns ``'rewarded' | 'unrewarded' | 'timeout'``.

    A supply pulse anywhere between AwaitReward and the end of the trial means the animal
    collected. Otherwise it had one response window to poke a reward port; poking the wrong one
    is ``unrewarded``, not poking at all is ``timeout``.
    """
    supply1, supply2, tagged_supply = _supply_pulses_between(
        supply_port1_times, supply_port2_times, await_reward_time, trial_end)
    if tagged_supply:
        # The reward-poke columns are deliberately not written on a rewarded trial here, unlike
        # the odour-discrimination path -- that difference in what gets recorded is why the two
        # scorers stayed separate.
        _record_supply_outcome(trial_dict, supply1, supply2, tagged_supply)
        return classify_completed_trial(
            supply_count=len(tagged_supply), reward_poke_count=0, has_await_reward=True)

    poke_window_end = await_reward_time + pd.Timedelta(seconds=response_time_sec)
    pokes1, pokes2, tagged_pokes = _reward_pokes_between(
        port1_pokes, port2_pokes, await_reward_time, poke_window_end)

    trial_dict['poke_window_end'] = poke_window_end
    trial_dict['port1_pokes_count'] = len(pokes1)
    trial_dict['port2_pokes_count'] = len(pokes2)
    trial_dict['total_reward_pokes'] = len(tagged_pokes)

    outcome = classify_completed_trial(
        supply_count=0, reward_poke_count=len(tagged_pokes), has_await_reward=True)
    if outcome == 'unrewarded':
        _record_first_reward_poke(trial_dict, tagged_pokes)
    return outcome


def _false_response_window_end(trial_end, await_reward_time, initiation_starts_sorted,
                               cue_poke_starts_sorted, port1_pokes, port2_pokes):
    """End of the window in which a poke after a completed no-go sequence counts as a response.

    Runs to the first cue poke after the next initiation -- i.e. until the animal has visibly
    re-engaged with the next trial. With no next initiation it runs to the last reward-port
    sample. Never earlier than ``await_reward_time``.
    """
    next_init_fr = None
    if not initiation_starts_sorted.empty:
        idx = initiation_starts_sorted.searchsorted(trial_end, side='right')
        if idx < len(initiation_starts_sorted):
            next_init_fr = initiation_starts_sorted.iloc[idx]

    fr_window_end = _next_after(cue_poke_starts_sorted, next_init_fr) if next_init_fr is not None else None

    if fr_window_end is None:
        candidates = [trial_end]
        if not port1_pokes.empty:
            candidates.append(port1_pokes.index.max())
        if not port2_pokes.empty:
            candidates.append(port2_pokes.index.max())
        candidates = [c for c in candidates if c is not None and not pd.isna(c)]
        fr_window_end = max(candidates) if candidates else trial_end

    return max(fr_window_end, await_reward_time)


def _score_false_response(trial_dict, *, await_reward_time, fr_window_end, port1_pokes,
                          port2_pokes, response_time_ms_window, cue_series):
    """Score a completed **non-rewarded** ("no-go") sequence, single-reward protocol only.

    There is nothing to collect, so going to a reward port anyway is a false response. Mirrors
    the false-alarm scoring on aborted trials, anchored at the completion moment rather than at
    the abort. The reward-poke columns are written too, so downstream poke-based logic sees the
    same shape as an unrewarded trial.
    """
    pokes1, pokes2, tagged_pokes = _reward_pokes_between(
        port1_pokes, port2_pokes, await_reward_time, fr_window_end)

    trial_dict['fr_window_end'] = fr_window_end
    trial_dict['port1_pokes_count'] = len(pokes1)
    trial_dict['port2_pokes_count'] = len(pokes2)
    trial_dict['total_reward_pokes'] = len(tagged_pokes)

    if not tagged_pokes:
        trial_dict['false_response'] = False
        trial_dict['fr_label'] = 'nFR'
        trial_dict['fr_time'] = pd.NaT
        trial_dict['fr_port'] = None
        trial_dict['fr_odor_identity'] = None
        trial_dict['fr_latency_ms'] = np.nan
        trial_dict['fr_movement_latency_ms'] = np.nan
        return

    fr_time, fr_port, fr_odor = tagged_pokes[0]
    fr_latency_ms = (fr_time - await_reward_time).total_seconds() * 1000.0
    trial_dict['false_response'] = True
    trial_dict['fr_time'] = fr_time
    trial_dict['fr_port'] = fr_port
    trial_dict['fr_odor_identity'] = fr_odor
    trial_dict['fr_latency_ms'] = float(fr_latency_ms)
    # (b) how fast it travelled once it finally left the cue port. fr_latency_ms above is (a),
    # time since the sequence completed, and it is what fr_label buckets. DECISIONS section 16.
    _fr_anchor = windows.last_poke_end_before(cue_series, fr_time)
    trial_dict['fr_movement_latency_ms'] = (
        float((fr_time - _fr_anchor).total_seconds() * 1000.0) if _fr_anchor is not None else np.nan)
    # Parity with unrewarded rows so downstream poke-based logic stays consistent.
    trial_dict['first_reward_poke_time'] = fr_time
    trial_dict['first_reward_poke_port'] = fr_port
    trial_dict['first_reward_poke_odor_identity'] = fr_odor
    trial_dict['fr_label'] = latency_label(fr_latency_ms, response_time_ms_window, 'FR')


def _label_non_initiated_odors(non_initiated_trials, odor_map):
    """Attach the odor each non-initiated attempt was presenting.

    Prefers a valve whose open window overlaps the attempt; failing that, the valve whose
    edges are closest in time. The fallback matters because a very short attempt can sit
    between two valve activations and still belong to one of them.
    """
    if not isinstance(non_initiated_trials, pd.DataFrame) or non_initiated_trials.empty:
        return non_initiated_trials

    odor_names = []
    for _, row in non_initiated_trials.iterrows():
        min_time_diff = float('inf')
        closest_odor = None
        attempt_start = row.get('attempt_start') or row.get('sequence_start')
        attempt_end = row.get('attempt_end') or row.get('sequence_end')
        found_odor = None
        for olf_id, valve_data in odor_map['olfactometer_valves'].items():
            if valve_data.empty:
                continue
            for i, valve_col in enumerate(valve_data.columns):
                odor_name = odor_map['valve_to_odor'].get(f"{olf_id}{i}")
                if not odor_name or odor_name.lower() == 'purge':
                    continue
                s = valve_data[valve_col]
                for st, en in zip(windows.rising_edges(s), windows.falling_edges(s)):
                    if st <= attempt_end and en >= attempt_start:
                        found_odor = odor_name
                        break
                    time_diff = min(abs((st - attempt_start).total_seconds()),
                                    abs((en - attempt_end).total_seconds()))
                    if time_diff < min_time_diff:
                        min_time_diff = time_diff
                        closest_odor = odor_name
                if found_odor:
                    break
            if found_odor:
                break
        odor_names.append(found_odor if found_odor is not None else closest_odor)

    non_initiated_trials = non_initiated_trials.copy()
    non_initiated_trials['odor_name'] = odor_names
    return non_initiated_trials


def classify_trials(data, events, trial_counts, odor_map, stage, root, verbose=True, single_reward_info=None):
    """Classify every initiated trial, with per-position valve and poke times.

    For each trial this resolves the odor sequence the animal actually sampled (a position
    counts only if it was poked), evaluates the hidden rule against it, and assigns an outcome.

    Three outcome paths, because three protocols ask different questions:

    * **odour-discrimination** -- no fixed response deadline. Supply pulses and reward pokes are
      both searched from AwaitReward to the end of the reward window.
    * **standard** -- supply anywhere up to the trial end is a reward; otherwise the animal has
      one ``responseTime`` window to poke, and poking is ``unrewarded``, not poking ``timeout``.
    * **single-reward no-go** -- a completed non-rewarded sequence has nothing to collect, so a
      reward-port poke is a *false response* and gets a latency label instead.

    A trial with no AwaitReward event never reached the end of its sequence and is aborted.

    Returns a dict of DataFrames: the completed/aborted splits, their hidden-rule and
    reward-status subsets, the non-initiated attempts, and the schema parameters used.
    """
    (sample_offset_time_ms, minimum_sampling_time_ms_by_odor,
     default_minimum_sampling_time_ms, response_time) = _sampling_parameters_ms(
        root, task="classify trials")

    def required_min_ms_for(odor_name):
        if odor_name is None:
            return default_minimum_sampling_time_ms
        return minimum_sampling_time_ms_by_odor.get(str(odor_name), default_minimum_sampling_time_ms)

    response_time_sec = response_time
    if response_time_sec is None:
        raise ValueError("Response time parameter cannot be extracted from Schema file. Check detect_settings function.")
    response_time_ms_window = float(response_time_sec) * 1000.0 if response_time_sec is not None else None

    if verbose:
        print("=" * 80)
        print("CLASSIFYING TRIAL OUTCOMES WITH HIDDEN RULE AND VALVE/POKE TIME ANALYSIS")
        print("=" * 80)
        print(f"Sample offset time: {sample_offset_time_ms} ms")
        print("Minimum sampling times (ms) by odor:")
        for odor_name, threshold in sorted(minimum_sampling_time_ms_by_odor.items()):
            print(f"  - {odor_name}: {threshold:.1f}")
        print(f"Response time window: {response_time_sec} s")

    hidden_rule_indices, sequence_name, schema_settings, schema_err = \
        _hidden_rule_indices_from_stage_or_schema(stage, root)
    protocol_name = (sequence_name or str(stage) or "").lower()
    is_odour_discrimination = "odourdiscrimination" in protocol_name

    seq_len = schema_settings.get('sequenceLength')
    max_positions = int(seq_len) if seq_len is not None else None
    if max_positions is None or max_positions < 1:
        raise ValueError("sequenceLength missing or invalid; cannot proceed without a valid sequence length")

    if verbose:
        # Printed BEFORE the final-position index is dropped below, unlike
        # analyze_response_times, which drops first. Existing behaviour; see
        # _hidden_rule_indices_from_stage_or_schema.
        pre_drop_positions, _loc, _pos, _multi = _hidden_rule_positions(hidden_rule_indices)
        _print_hidden_rule_header(hidden_rule_indices, pre_drop_positions, sequence_name, stage,
                                  label_prefix="Location")

    initiated_trials = trial_counts['initiated_sequences'].copy()
    non_initiated_trials = trial_counts['non_initiated_sequences'].copy()
    init_series_raw = initiated_trials.get('initiation_sequence_time')
    initiation_starts_sorted = pd.to_datetime(init_series_raw, errors='coerce').dropna().sort_values().reset_index(drop=True)

    await_reward_times = events['combined_await_reward_df']['Time'].tolist() if 'combined_await_reward_df' in events else []

    supply_port1_times = data['pulse_supply_1'].index.tolist() if not data['pulse_supply_1'].empty else []
    supply_port2_times = data['pulse_supply_2'].index.tolist() if not data['pulse_supply_2'].empty else []
    all_supply_port_times = sorted(supply_port1_times + supply_port2_times)

    port1_pokes = data['digital_input_data'].get('DIPort1', pd.Series(dtype=bool))
    port2_pokes = data['digital_input_data'].get('DIPort2', pd.Series(dtype=bool))
    poke_data = data['digital_input_data'].get('DIPort0', pd.Series(dtype=bool))

    poke_series_full = poke_data.astype(bool).sort_index()
    _starts = windows.rising_edges(poke_series_full)
    cue_poke_starts_sorted = pd.Series(_starts, dtype='datetime64[ns]').sort_values() if _starts else pd.Series(dtype='datetime64[ns]')
    poke_intervals = windows.cue_poke_intervals(poke_series_full)

    all_valve_activations = windows.valve_windows_closing_at_series_end(
        odor_map['olfactometer_valves'], odor_map['valve_to_odor'])

    if verbose:
        print(f"Found {len(all_valve_activations)} total valve activations (excluding Purge)")
        print(f"Analyzing {len(initiated_trials)} initiated trials...")
        print(f"Found {len(await_reward_times)} AwaitReward events")
        print(f"Found {len(all_supply_port_times)} total supply port activities")

    completed_sequences = []
    aborted_sequences = []
    aborted_sequences_hr = []
    completed_hr = []
    completed_hr_missed = []
    completed_rewarded = []
    completed_unrewarded = []
    completed_timeout = []
    completed_hr_rewarded = []
    completed_hr_unrewarded = []
    completed_hr_timeout = []
    completed_hr_missed_rewarded = []
    completed_hr_missed_unrewarded = []
    completed_hr_missed_timeout = []
    non_initiated_odor1_attempts = []
    # Single-reward protocol: completed sequences whose final position is NOT rewarded.
    # Empty (and never appended to) for the default protocol, so legacy output is unchanged.
    completed_false_response = []
    initiated_trials_list = []

    # Which list each (outcome, hidden-rule category) pair appends to on the standard path.
    outcome_buckets = {
        'rewarded': (completed_rewarded, completed_hr_rewarded, completed_hr_missed_rewarded),
        'unrewarded': (completed_unrewarded, completed_hr_unrewarded, completed_hr_missed_unrewarded),
        'timeout': (completed_timeout, completed_hr_timeout, completed_hr_missed_timeout),
    }

    if single_reward_info is None:
        single_reward_info = _get_single_reward_info(root)
    is_single_reward, rewarded_sequences, all_sequences = single_reward_info
    # The final position of a full sequence is always the reward position, so it can
    # never be a hidden-rule position -- drop it (single-reward left untouched).
    hidden_rule_indices = _drop_final_hidden_rule_index(hidden_rule_indices, schema_settings, is_single_reward)
    hidden_rule_positions, hidden_rule_location, hidden_rule_position, _multiple = \
        _hidden_rule_positions(hidden_rule_indices)

    hr_odor_set = _hidden_rule_odor_set(hidden_rule_indices, schema_settings, schema_err, verbose)

    # Aggregators for the summary prints (completed trials only)
    agg_position_poke_times = {pos: [] for pos in range(1, max_positions + 1)}
    agg_position_valve_times = {pos: [] for pos in range(1, max_positions + 1)}
    agg_odor_poke_times = defaultdict(list)
    agg_odor_valve_times = defaultdict(list)

    for _, trial in initiated_trials.iterrows():
        trial_start = trial['sequence_start']
        trial_end = trial['sequence_end']

        valve_activations = windows.valve_events_overlapping(all_valve_activations, trial_start, trial_end)
        position_locations, prior_presentations = _assign_positions_to_valve_events(
            valve_activations, max_positions, required_min_ms_for)
        position_valve_times = _position_valve_times(
            position_locations, max_positions, prior_presentations, required_min_ms_for)
        position_poke_times = _position_poke_times(
            position_locations, poke_data, max_positions, sample_offset_time_ms, required_min_ms_for)

        # A position counts only if the animal actually poked during it.
        valid_positions = [
            pos for pos in sorted(position_valve_times.keys())
            if position_poke_times.get(pos) and position_poke_times[pos].get('poke_time_ms', 0.0) > 0.0
        ]
        presentations, last_event_index = _build_presentations(
            valid_positions, position_valve_times, position_poke_times)

        pos1_info = position_valve_times.get(1, {}) or {}
        for attempt in pos1_info.get('prior_presentations', []) or []:
            a_start = attempt.get('valve_start')
            # Cap the bout at the last Pos1 valve close: the trial proper starts at the last
            # Pos1 opening, so a bout may not run into it.
            first_in, _bout_end, dur_ms = windows.bout_around_anchor(
                poke_intervals, a_start, sample_offset_time_ms, cap_end=pos1_info.get('valve_end'))
            non_initiated_odor1_attempts.append({
                'trial_id': trial['trial_id'] if 'trial_id' in trial else None,
                'attempt_start': a_start,
                'attempt_end': attempt.get('valve_end'),
                'odor_name': attempt.get('odor_name'),
                'attempt_first_poke_in': first_in,
                'attempt_poke_time_ms': dur_ms,
                'required_min_sampling_time_ms': attempt.get('required_min_sampling_time_ms', required_min_ms_for(attempt.get('odor_name'))),
            })

        trial_await_rewards = [t for t in await_reward_times if trial_start <= t <= trial_end]

        final_odor_sequence = [
            (position_valve_times[pos] or {}).get('odor_name')
            for pos in valid_positions
            if position_valve_times.get(pos) is not None
        ]

        trial_dict = trial.to_dict()
        trial_dict['odor_sequence'] = final_odor_sequence
        trial_dict['num_odors'] = len(final_odor_sequence)
        trial_dict['last_odor'] = final_odor_sequence[-1] if final_odor_sequence else None

        # Single-reward protocol: is THIS trial's full presented sequence one of the rewarded
        # ones (exact match against the schema)? Only set in single-reward mode so the default
        # protocol's output columns are untouched.
        sequence_rewarded = None
        if is_single_reward:
            sequence_rewarded = tuple(final_odor_sequence) in rewarded_sequences
            trial_dict['sequence_rewarded'] = sequence_rewarded
            reward_determinacy, determinacy_position, determined_final_odor = _classify_reward_determinacy(
                final_odor_sequence, all_sequences, rewarded_sequences
            )
            trial_dict['reward_determinacy'] = reward_determinacy
            trial_dict['determinacy_position'] = determinacy_position
            trial_dict['determined_final_odor'] = determined_final_odor

        trial_dict['hidden_rule_location'] = hidden_rule_location
        trial_dict['hidden_rule_locations'] = list(hidden_rule_indices)
        trial_dict['hidden_rule_positions'] = list(hidden_rule_positions)
        trial_dict['sequence_name'] = sequence_name
        trial_dict['position_valve_times'] = position_valve_times
        trial_dict['position_poke_times'] = position_poke_times
        trial_dict['presentations'] = presentations
        trial_dict['last_event_index'] = last_event_index
        trial_dict['minimum_sampling_time_ms_by_odor'] = dict(minimum_sampling_time_ms_by_odor)

        pos1_poke = position_poke_times.get(1)
        if pos1_poke:
            corrected_start = pos1_poke.get('poke_first_in') or pos1_poke.get('poke_odor_start')
            if corrected_start is not None:
                trial_dict['sequence_start_corrected'] = corrected_start

        enough_odors, hit_hidden_rule, hr_hit_indices = _check_hidden_rule(
            final_odor_sequence, hidden_rule_indices, hr_odor_set
        )
        hr_hit_positions = [idx + 1 for idx in hr_hit_indices]
        trial_dict['enough_odors_for_hr'] = enough_odors
        trial_dict['hit_hidden_rule'] = hit_hidden_rule
        trial_dict['hidden_rule_hit_indices'] = hr_hit_indices
        trial_dict['hidden_rule_hit_positions'] = hr_hit_positions
        hr_success, hr_success_position = _hidden_rule_success(
            hr_hit_positions, len(final_odor_sequence), max_positions, bool(trial_await_rewards))
        trial_dict['hidden_rule_success'] = hr_success
        trial_dict['hidden_rule_success_position'] = hr_success_position

        if is_odour_discrimination:
            last_valve_event = valve_activations[-1] if valve_activations else None
            trial_dict['odourdiscrimination_mode'] = True
            trial_dict['last_valve_start'] = (last_valve_event or {}).get('start_time')

            current_init_ts = pd.to_datetime(trial.get('initiation_sequence_time'), errors='coerce') \
                if trial.get('initiation_sequence_time') is not None else pd.NaT
            await_window_start = current_init_ts if not pd.isna(current_init_ts) else \
                (trial_start if trial_start is not None else trial_dict['last_valve_start'])

            if await_window_start is None or pd.isna(await_window_start):
                aborted_sequences.append(trial_dict.copy())
                initiated_trials_list.append(trial_dict)
                continue

            next_init = None
            if not initiation_starts_sorted.empty and not pd.isna(current_init_ts):
                idx = initiation_starts_sorted.searchsorted(current_init_ts, side='right')
                if idx < len(initiation_starts_sorted):
                    next_init = initiation_starts_sorted.iloc[idx]

            recording_end = _recording_end(initiation_starts_sorted, cue_poke_starts_sorted,
                                           supply_port1_times, supply_port2_times,
                                           port1_pokes, port2_pokes, trial_end)

            # AwaitReward is searched from the initiation up to the next one -- the task fires
            # it once the animal commits, which can be after the detected trial window ends.
            await_upper_bound = next_init if next_init is not None else recording_end
            await_in_window = [t for t in await_reward_times if await_window_start <= t <= await_upper_bound]
            if not await_in_window:
                trial_dict['abort_reason'] = 'no_await_reward'
                aborted_sequences.append(trial_dict.copy())
                initiated_trials_list.append(trial_dict)
                continue

            await_time = min(await_in_window)
            trial_dict['await_reward_time'] = await_time

            next_cue_after_next_init = _next_after(cue_poke_starts_sorted, next_init) if next_init is not None else None
            reward_window_end, next_cue_poke = _odourdisc_reward_window_end(
                next_init, next_cue_after_next_init, await_time, cue_poke_starts_sorted, recording_end)

            trial_dict['next_initiation_time'] = next_init
            trial_dict['next_cue_poke_start'] = next_cue_poke
            trial_dict['reward_window_end'] = reward_window_end

            if verbose:
                supply1_dbg, supply2_dbg, _ = _supply_pulses_between(
                    supply_port1_times, supply_port2_times, await_time, reward_window_end)
                print(
                    "[odourdisc] window",
                    f"init={await_window_start}",
                    f"await={await_time}",
                    f"next_init={next_init}",
                    f"next_cue={next_cue_poke}",
                    f"reward_end={reward_window_end}",
                    f"supply_counts=({len(supply1_dbg)},{len(supply2_dbg)})",
                )
                if supply_port1_times or supply_port2_times:
                    print(
                        "[odourdisc] raw supply tails",
                        f"s1_total={len(supply_port1_times)} last={supply_port1_times[-1] if supply_port1_times else None}",
                        f"s2_total={len(supply_port2_times)} last={supply_port2_times[-1] if supply_port2_times else None}",
                    )
                if supply1_dbg or supply2_dbg:
                    print("[odourdisc] supply in window", (supply1_dbg + supply2_dbg)[:5])

            outcome = _score_odourdisc_outcome(
                trial_dict, await_time=await_time, reward_window_end=reward_window_end,
                supply_port1_times=supply_port1_times, supply_port2_times=supply_port2_times,
                port1_pokes=port1_pokes, port2_pokes=port2_pokes)
            outcome_buckets[outcome][0].append(trial_dict.copy())

            completed_sequences.append(trial_dict.copy())
            initiated_trials_list.append(trial_dict)
            continue

        initiated_trials_list.append(trial_dict)
        if not trial_await_rewards:
            aborted_sequences.append(trial_dict.copy())
            if hit_hidden_rule:
                aborted_sequences_hr.append(trial_dict.copy())
            continue

        for pos, v in (position_valve_times or {}).items():
            if v and 'valve_duration_ms' in v:
                agg_position_valve_times[pos].append(v['valve_duration_ms'])
                if v.get('odor_name'):
                    agg_odor_valve_times[v['odor_name']].append(v['valve_duration_ms'])
        for pos, p in (position_poke_times or {}).items():
            if p and 'poke_time_ms' in p:
                agg_position_poke_times[pos].append(p['poke_time_ms'])
                if p.get('odor_name'):
                    agg_odor_poke_times[p['odor_name']].append(p['poke_time_ms'])

        await_reward_time = min(trial_await_rewards)
        trial_dict['await_reward_time'] = await_reward_time

        if hit_hidden_rule:
            if hr_success:
                completed_hr.append(trial_dict.copy())
                hr_category = 1
            else:
                completed_hr_missed.append(trial_dict.copy())
                hr_category = 2
        else:
            hr_category = 0

        if is_single_reward and sequence_rewarded is False:
            fr_window_end = _false_response_window_end(
                trial_end, await_reward_time, initiation_starts_sorted, cue_poke_starts_sorted,
                port1_pokes, port2_pokes)
            _score_false_response(
                trial_dict, await_reward_time=await_reward_time, fr_window_end=fr_window_end,
                port1_pokes=port1_pokes, port2_pokes=port2_pokes,
                response_time_ms_window=response_time_ms_window, cue_series=poke_series_full)
            completed_false_response.append(trial_dict.copy())
        else:
            outcome = _score_standard_outcome(
                trial_dict, await_reward_time=await_reward_time, trial_end=trial_end,
                supply_port1_times=supply_port1_times, supply_port2_times=supply_port2_times,
                port1_pokes=port1_pokes, port2_pokes=port2_pokes,
                response_time_sec=response_time_sec)
            buckets = outcome_buckets[outcome]
            buckets[0].append(trial_dict.copy())
            if hr_category:
                buckets[hr_category].append(trial_dict.copy())

        completed_sequences.append(trial_dict.copy())

    non_initiated_trials = _label_non_initiated_odors(non_initiated_trials, odor_map)
    initiated_trials = pd.DataFrame(initiated_trials_list)

    result = {
        'non_initiated_sequences': non_initiated_trials,
        'initiated_sequences': initiated_trials,
        'completed_sequences': pd.DataFrame(completed_sequences),
        'aborted_sequences': pd.DataFrame(aborted_sequences),
        'non_initiated_odor1_attempts': pd.DataFrame(non_initiated_odor1_attempts),
        'minimum_sampling_time_ms_by_odor': dict(minimum_sampling_time_ms_by_odor),
        'default_minimum_sampling_time_ms': float(default_minimum_sampling_time_ms),
        'minimum_sampling_time_ms': float(default_minimum_sampling_time_ms),

        'aborted_sequences_HR': pd.DataFrame(aborted_sequences_hr),
        'completed_sequences_HR': pd.DataFrame(completed_hr),
        'completed_sequences_HR_missed': pd.DataFrame(completed_hr_missed),

        'completed_sequence_rewarded': pd.DataFrame(completed_rewarded),
        'completed_sequence_unrewarded': pd.DataFrame(completed_unrewarded),
        'completed_sequence_reward_timeout': pd.DataFrame(completed_timeout),

        # Single-reward protocol only: completed non-rewarded ("no-go") sequences. Empty otherwise.
        'completed_sequence_false_response': pd.DataFrame(completed_false_response),

        'completed_sequence_HR_rewarded': pd.DataFrame(completed_hr_rewarded),
        'completed_sequence_HR_unrewarded': pd.DataFrame(completed_hr_unrewarded),
        'completed_sequence_HR_reward_timeout': pd.DataFrame(completed_hr_timeout),
        'completed_sequence_HR_missed_rewarded': pd.DataFrame(completed_hr_missed_rewarded),
        'completed_sequence_HR_missed_unrewarded': pd.DataFrame(completed_hr_missed_unrewarded),
        'completed_sequence_HR_missed_reward_timeout': pd.DataFrame(completed_hr_missed_timeout),
    }

    if isinstance(result['non_initiated_sequences'], pd.DataFrame) and not result['non_initiated_sequences'].empty:
        df = result['non_initiated_sequences'].copy()
        if 'continuous_poke_time_ms' in df.columns:
            df['pos1_poke_time_ms'] = pd.to_numeric(df['continuous_poke_time_ms'], errors='coerce').fillna(0.0)
        result['non_initiated_sequences'] = df

    # Plural aliases to prevent KeyErrors in downstream code
    for key in ('HR_rewarded', 'HR_unrewarded', 'HR_reward_timeout',
                'HR_missed_rewarded', 'HR_missed_unrewarded', 'HR_missed_reward_timeout'):
        result[f'completed_sequences_{key}'] = result[f'completed_sequence_{key}']

    result['hidden_rule_location'] = hidden_rule_location
    result['hidden_rule_positions'] = list(hidden_rule_positions)
    result['hidden_rule_locations'] = list(hidden_rule_indices)
    result['hidden_rule_position'] = hidden_rule_position
    result['hidden_rule_odors'] = sorted(list(hr_odor_set)) if hr_odor_set is not None else []

    if verbose:
        _print_classification_summary(
            result, initiated_trials=initiated_trials,
            hidden_rule_indices=hidden_rule_indices, hidden_rule_positions=hidden_rule_positions,
            hidden_rule_location=hidden_rule_location, hidden_rule_position=hidden_rule_position,
            max_positions=max_positions,
            agg_position_poke_times=agg_position_poke_times,
            agg_position_valve_times=agg_position_valve_times,
            agg_odor_poke_times=agg_odor_poke_times,
            agg_odor_valve_times=agg_odor_valve_times)

    return result



def _print_classification_summary(result, *, initiated_trials, hidden_rule_indices,
                                  hidden_rule_positions, hidden_rule_location, hidden_rule_position,
                                  max_positions, agg_position_poke_times, agg_position_valve_times,
                                  agg_odor_poke_times, agg_odor_valve_times):
    """The end-of-classification summary: counts, reward breakdown and poke/valve time ranges."""
    def _pct(n, d):
        try:
            d = float(d)
            return 0.0 if d == 0 else (float(n) / d * 100.0)
        except Exception:
            return 0.0

    print(f"\nTRIAL CLASSIFICATION RESULTS WITH HIDDEN RULE AND VALVE/POKE TIME ANALYSIS:")
    if hidden_rule_positions:
        if len(hidden_rule_positions) > 1:
            pos_str = ", ".join(str(pos) for pos in hidden_rule_positions)
            idx_str = ", ".join(str(idx) for idx in hidden_rule_indices)
            print(f"Hidden Rule Locations: Positions {pos_str} (indices {idx_str})\n")
        else:
            print(f"Hidden Rule Location: Position {hidden_rule_position} (index {hidden_rule_location})\n")
    else:
        print("Hidden Rule Location: None detected\n")
    print(f"Hidden Rule Odors: {', '.join(result['hidden_rule_odors']) if result['hidden_rule_odors'] else 'None'}\n")

    base_non_init_df = result.get('non_initiated_sequences', pd.DataFrame())
    pos1_attempts_df = result.get('non_initiated_odor1_attempts', pd.DataFrame())

    base_non_init_count = 0 if base_non_init_df is None or base_non_init_df.empty else len(base_non_init_df)
    pos1_attempts_count = 0 if pos1_attempts_df is None or pos1_attempts_df.empty else len(pos1_attempts_df)

    total_non_init = base_non_init_count + pos1_attempts_count
    ini_n = len(initiated_trials)
    total_attempts = ini_n + total_non_init

    print(f"Total attempts: {total_attempts}")
    print(f"-- Non-initiated sequences (total): {total_non_init} ({_pct(total_non_init, total_attempts):.1f}%)")
    if pos1_attempts_count:
        print(f"    -- Position 1 attempts within trials {pos1_attempts_count} ({_pct(pos1_attempts_count, total_non_init):.1f}%)")
        print(f"    -- Baseline non-initiated sequences {base_non_init_count} ({_pct(base_non_init_count, total_non_init):.1f}%)")
    print(f"-- Initiated sequences (trials): {ini_n} ({_pct(ini_n, total_attempts):.1f}%)\n")

    print("INITIATED TRIALS BREAKDOWN:")
    comp_n = len(result['completed_sequences'])
    print(f"-- Completed sequences: {comp_n} ({_pct(comp_n, ini_n):.1f}%)")
    print(f"   -- Hidden Rule trials (HR): {len(result['completed_sequences_HR'])} ({_pct(len(result['completed_sequences_HR']), ini_n):.1f}%)")
    print(f"   -- Hidden Rule Missed (HR_missed): {len(result['completed_sequences_HR_missed'])} ({_pct(len(result['completed_sequences_HR_missed']), ini_n):.1f}%)")
    print(f"-- Aborted sequences: {len(result['aborted_sequences'])} ({_pct(len(result['aborted_sequences']), ini_n):.1f}%)")
    print(f"   -- Aborted Hidden Rule trials (HR): {len(result['aborted_sequences_HR'])} ({_pct(len(result['aborted_sequences_HR']), ini_n):.1f}%)\n")

    print("REWARD STATUS BREAKDOWN:")
    cs = comp_n
    if cs > 0:
        print(f"-- Rewarded: {len(result['completed_sequence_rewarded'])} ({_pct(len(result['completed_sequence_rewarded']), cs):.1f}%)")
        print(f"-- Unrewarded: {len(result['completed_sequence_unrewarded'])} ({_pct(len(result['completed_sequence_unrewarded']), cs):.1f}%)")
        print(f"-- Reward timeout: {len(result['completed_sequence_reward_timeout'])} ({_pct(len(result['completed_sequence_reward_timeout']), cs):.1f}%)\n")

    print("HIDDEN RULE SPECIFIC BREAKDOWN:")
    hr_total = len(result['completed_sequences_HR'])
    if hr_total > 0:
        print(f"-- HR Rewarded: {len(result['completed_sequence_HR_rewarded'])} ({_pct(len(result['completed_sequence_HR_rewarded']), hr_total):.1f}%)")
        print(f"-- HR Unrewarded: {len(result['completed_sequence_HR_unrewarded'])} ({_pct(len(result['completed_sequence_HR_unrewarded']), hr_total):.1f}%)")
        print(f"-- HR Timeout: {len(result['completed_sequence_HR_reward_timeout'])} ({_pct(len(result['completed_sequence_HR_reward_timeout']), hr_total):.1f}%)")

    hr_missed_total = len(result['completed_sequences_HR_missed'])
    if hr_missed_total > 0:
        print(f"Completed HR Missed trials: {hr_missed_total}")
        print(f"-- HR Missed Rewarded: {len(result['completed_sequence_HR_missed_rewarded'])} ({len(result['completed_sequence_HR_missed_rewarded'])/hr_missed_total*100:.1f}%)")
        print(f"-- HR Missed Unrewarded: {len(result['completed_sequence_HR_missed_unrewarded'])} ({len(result['completed_sequence_HR_missed_unrewarded'])/hr_missed_total*100:.1f}%)")
        print(f"-- HR Missed Timeout: {len(result['completed_sequence_HR_missed_reward_timeout'])} ({len(result['completed_sequence_HR_missed_reward_timeout'])/hr_missed_total*100:.1f}%)")
    print()

    print("POKE TIME RANGES BY POSITION:")
    print("-" * 40)
    _print_time_ranges(agg_position_poke_times, range(1, max_positions + 1), lambda pos: f"Position {pos}", show_empty=True)

    print("\nVALVE TIME RANGES BY POSITION:")
    print("-" * 40)
    _print_time_ranges(agg_position_valve_times, range(1, max_positions + 1), lambda pos: f"Position {pos}", show_empty=True)

    print("\nPOKE TIME RANGES BY ODOR (ALL POSITIONS):")
    print("-" * 50)
    _print_time_ranges(agg_odor_poke_times, sorted(agg_odor_poke_times.keys()), str, show_empty=False)

    print("\nVALVE TIME RANGES BY ODOR (ALL POSITIONS):")
    print("-" * 50)
    _print_time_ranges(agg_odor_valve_times, sorted(agg_odor_valve_times.keys()), str, show_empty=False)

    print("\nNON-INITIATED TRIALS POKE TIMES:")
    print("-" * 40)
    if not result['non_initiated_sequences'].empty:
        base = result['non_initiated_sequences']
        pos1 = result['non_initiated_odor1_attempts']
        print(f"Baseline non-initiated: n={len(base)} avg={base['pos1_poke_time_ms'].mean():.1f} ms range={base['pos1_poke_time_ms'].min():.1f}-{base['pos1_poke_time_ms'].max():.1f} ms")
        if not pos1.empty:
            s = pd.to_numeric(pos1['attempt_poke_time_ms'], errors='coerce').dropna()
            print(f"Pos1 attempts: n={len(s)} avg={s.mean():.1f} ms range={s.min():.1f}-{s.max():.1f} ms")
        else:
            print("Pos1 attempts: n=0")

    total_classified = (len(result['completed_sequence_rewarded'])
                        + len(result['completed_sequence_unrewarded'])
                        + len(result['completed_sequence_reward_timeout'])
                        + len(result['aborted_sequences']))
    if total_classified == len(initiated_trials):
        print(f"\nClassification complete: all {len(initiated_trials)} trials classified")
    else:
        print(f"\nClassification mismatch: {total_classified} classified vs {len(initiated_trials)} total")


def _print_time_ranges(groups, keys, label_for, *, show_empty):
    """min/max/avg per group. ``show_empty`` prints a 'No data' line for positions never sampled."""
    for key in keys:
        times = groups[key] if key in groups else []
        if times:
            min_v = min(times); max_v = max(times); avg_v = sum(times) / len(times)
            print(f"{label_for(key)}: {min_v:.1f} - {max_v:.1f}ms (avg: {avg_v:.1f}ms, n={len(times)})")
        elif show_empty:
            print(f"{label_for(key)}: No data")

def _next_completed_trial_start(current_trial_end, all_trials):
    """Start of the earliest completed trial beginning after ``current_trial_end``."""
    next_starts = [t['sequence_start'] for t in all_trials if t['sequence_start'] > current_trial_end]
    return min(next_starts) if next_starts else None


def _rt_row(trial_id, response_time_ms, category, target=None, window_latency_ms=None):
    """One ``per_trial`` row for the response-time table.

    ``target`` is omitted on the early failures that give up before a target odor is known.
    That is deliberate: an absent key becomes NaN in the assembled DataFrame, whereas writing
    ``None`` explicitly would keep None in the object column and change the saved output.
    """
    row = {
        'trial_id': trial_id,
        'response_time_ms': response_time_ms,
        'response_time_category': category,
    }
    if target is not None:
        row['target_odor_name'] = target[0]
        row['target_required_min_sampling_time_ms'] = target[1]
    if window_latency_ms is not None:
        row['completed_window_latency_ms'] = window_latency_ms
    return row


def _print_response_time_summary(completed_count, failed_calculations, rewarded, hr_rewarded,
                                 unrewarded, timeout_delayed, timeout_delay):
    def _spread(times, indent):
        print(f"{indent}Range: {min(times):.1f} - {max(times):.1f}ms")
        print(f"{indent}Average: {sum(times) / len(times):.1f}ms")
        print(f"{indent}Median: {sorted(times)[len(times)//2]:.1f}ms")

    def _stats(times, indent="  "):
        print(f"{indent}Count: {len(times)}")
        _spread(times, indent)

    print(f"RESPONSE TIME ANALYSIS RESULTS:")
    print(f"Total completed trials analyzed: {completed_count}")
    print(f"Failed response time calculations: {failed_calculations}")
    print(f"Successful response time calculations: {len(rewarded) + len(unrewarded) + len(timeout_delayed)}")
    print()

    print(f"REWARDED TRIALS:")
    if rewarded:
        _stats(rewarded)
    else:
        print(f"  No rewarded trials with response times")

    if hr_rewarded:
        print(f"\nHR REWARDED TRIALS (response times):")
        print(f"  Count: {len(hr_rewarded)}")
        print(f"  Range: {min(hr_rewarded):.1f} - {max(hr_rewarded):.1f}ms")
        print(f"  Average: {sum(hr_rewarded)/len(hr_rewarded):.1f}ms")
    else:
        print(f"\nHR REWARDED TRIALS (response times): none")

    print(f"\nUNREWARDED TRIALS:")
    if unrewarded:
        _stats(unrewarded)
    else:
        print(f"  No unrewarded trials with response times")

    print(f"\nTIMEOUT TRIALS WITH DELAYED RESPONSES:")
    if timeout_delayed:
        print(f"  Count: {len(timeout_delayed)}")
        print(f"  Response time (poke out to delayed poke):")
        _spread(timeout_delayed, "    ")
        print(f"  Response delay time (window end to delayed poke):")
        _spread(timeout_delay, "    ")
    else:
        print(f"  No timeout trials with delayed responses")

    print(f"\nALL TRIALS WITH RESPONSE TIMES:")
    all_response_times = rewarded + unrewarded + timeout_delayed
    if all_response_times:
        _stats(all_response_times)


def analyze_response_times(data, trial_counts, events, odor_map, stage, root, verbose=True, single_reward_info=None):
    """Response time from last cue-port poke-out to first reward-port poke, per completed trial.

    A completed trial is one with an AwaitReward event inside its window. The response is timed
    from the animal's **last** exit of the cue port around the final odor -- not from
    AwaitReward -- because that is when it was free to move.

    Each trial lands in one of four buckets. ``rewarded`` and ``unrewarded`` are decided by
    whether a supply pulse followed AwaitReward; a trial with no poke inside the response
    window but one before the next trial is ``timeout_delayed``; anything whose response time
    could not be computed is counted in ``failed_calculations`` with a null category.

    Returns the four latency lists, the per-trial table, and the schema parameters used.
    """
    (sample_offset_time_ms, minimum_sampling_time_ms_by_odor,
     default_minimum_sampling_time_ms, response_time) = _sampling_parameters_ms(
        root, task="analyze response times")

    def resolve_min_sampling_time_ms(odor_name):
        if odor_name is None:
            return default_minimum_sampling_time_ms
        return minimum_sampling_time_ms_by_odor.get(str(odor_name), default_minimum_sampling_time_ms)

    response_time_sec = response_time
    if response_time_sec is None:
        raise ValueError("Response time parameter cannot be extracted from Schema file. Check detect_settings function.")

    # Single-reward protocol: keep response_time_category meaningful for rewarded-type
    # sequences only. Non-rewarded ("no-go") completions are handled as false_response in
    # classify_trials; here we simply leave their response_time_category empty so existing
    # decision/choice-accuracy metrics are not polluted. Disabled for the default protocol.
    if single_reward_info is None:
        single_reward_info = _get_single_reward_info(root)
    is_single_reward, rewarded_sequences, _all_sequences = single_reward_info

    if verbose:
        print("=" * 80)
        print("RESPONSE TIME ANALYSIS - ALL COMPLETED TRIALS")
        print("=" * 80)

    hidden_rule_indices, sequence_name, schema_settings, schema_err = \
        _hidden_rule_indices_from_stage_or_schema(stage, root)
    # Final position is always rewarded -> never a hidden-rule position (single-reward untouched).
    hidden_rule_indices = _drop_final_hidden_rule_index(hidden_rule_indices, schema_settings, is_single_reward)
    hidden_rule_positions, _location, _position, _multiple = _hidden_rule_positions(hidden_rule_indices)

    if verbose:
        print(f"Sample offset time: {sample_offset_time_ms} ms")
        print("Minimum sampling times (ms) by odor:")
        for odor_name, threshold in sorted(minimum_sampling_time_ms_by_odor.items()):
            print(f"  - {odor_name}: {threshold:.1f}")
        print(f"Response time window: {response_time_sec} s")
        _print_hidden_rule_header(hidden_rule_indices, hidden_rule_positions, sequence_name, stage,
                                  label_prefix="Location ")

    initiated_trials = trial_counts['initiated_sequences']
    await_reward_times = events['combined_await_reward_df']['Time'].tolist() if 'combined_await_reward_df' in events else []

    protocol_name = (sequence_name or str(stage) or "").lower()
    is_odour_discrimination = "odourdiscrimination" in protocol_name

    init_series_raw = initiated_trials.get('initiation_sequence_time')
    initiation_starts_sorted = pd.to_datetime(init_series_raw, errors='coerce').dropna().sort_values().reset_index(drop=True)

    poke_series_full = data['digital_input_data'].get('DIPort0', pd.Series(dtype=bool)).astype(bool)
    poke_series_full = poke_series_full.sort_index()
    cue_poke_starts_sorted = pd.Series(dtype='datetime64[ns]')
    if not poke_series_full.empty:
        starts = windows.rising_edges(poke_series_full)
        cue_poke_starts_sorted = pd.Series(starts, dtype='datetime64[ns]').sort_values().reset_index(drop=True)

    supply_port1_times = data['pulse_supply_1'].index.tolist() if not data['pulse_supply_1'].empty else []
    supply_port2_times = data['pulse_supply_2'].index.tolist() if not data['pulse_supply_2'].empty else []

    completed_trials_all = []
    for _, trial in initiated_trials.iterrows():
        trial_await_rewards = [t for t in await_reward_times
                               if trial['sequence_start'] <= t <= trial['sequence_end']]
        if trial_await_rewards:
            trial_dict = trial.to_dict()
            trial_dict['await_reward_time'] = min(trial_await_rewards)
            completed_trials_all.append(trial_dict)

    if verbose:
        print(f"Total completed trials: {len(completed_trials_all)}\n")

    poke_data = data['digital_input_data']['DIPort0'].copy() if 'DIPort0' in data['digital_input_data'] else pd.Series(dtype=bool)
    port1_pokes = data['digital_input_data']['DIPort1'] if 'DIPort1' in data['digital_input_data'] else pd.Series(dtype=bool)
    port2_pokes = data['digital_input_data']['DIPort2'] if 'DIPort2' in data['digital_input_data'] else pd.Series(dtype=bool)

    all_valve_activations = windows.valve_windows_closing_at_series_end(
        odor_map['olfactometer_valves'], odor_map['valve_to_odor'])

    hr_odor_set = _hidden_rule_odor_set(hidden_rule_indices, schema_settings, schema_err, verbose)

    rewarded_response_times = []
    unrewarded_response_times = []
    timeout_delayed_response_times = []
    timeout_response_delay_times = []
    failed_calculations = 0
    hr_rewarded_response_times = []
    per_trial_rows = []

    for trial_dict in completed_trials_all:
        trial_id = trial_dict.get('trial_id')
        trial_start = trial_dict['sequence_start']
        trial_end = trial_dict['sequence_end']
        await_reward_time = trial_dict['await_reward_time']

        trial_valve_events = windows.valve_events_overlapping(all_valve_activations, trial_start, trial_end)
        if not trial_valve_events:
            failed_calculations += 1
            per_trial_rows.append(_rt_row(trial_id, np.nan, None))
            continue

        position_locations_rt, ordered_positions_rt = windows.first_occurrence_positions(trial_valve_events)
        effective_odor_sequence = [
            position_locations_rt[pos]['odor_name']
            for pos in ordered_positions_rt
            if position_locations_rt.get(pos) is not None
        ]

        # Single-reward protocol: non-rewarded ("no-go") completions are scored as false_response
        # in classify_trials, not here. Leave their response_time_category empty so existing
        # rewarded/unrewarded/timeout-based metrics stay clean. No-op for the default protocol.
        if is_single_reward and tuple(effective_odor_sequence) not in rewarded_sequences:
            per_trial_rows.append(_rt_row(trial_id, np.nan, None))
            continue

        _, _hit_hidden_rule, hr_hit_indices = _check_hidden_rule(
            effective_odor_sequence, hidden_rule_indices, hr_odor_set
        )
        hr_hit_positions = [idx + 1 for idx in hr_hit_indices]
        hr_success = len(effective_odor_sequence) in hr_hit_positions if hr_hit_positions else False

        if not ordered_positions_rt:
            failed_calculations += 1
            per_trial_rows.append(_rt_row(trial_id, np.nan, None))
            continue

        target_valve_event = position_locations_rt.get(ordered_positions_rt[-1])
        if target_valve_event is None:
            failed_calculations += 1
            per_trial_rows.append(_rt_row(trial_id, np.nan, None, target=(None, float('nan'))))
            continue

        target_odor_name = target_valve_event.get('odor_name')
        target = (target_odor_name, resolve_min_sampling_time_ms(target_odor_name))

        # The response is timed from the last exit of the cue port around the final odor,
        # searched up to AwaitReward or one second past the valve closing, whichever is later.
        odor_start = target_valve_event['start_time']
        odor_end = target_valve_event['end_time']
        last_poke_out_time = windows.last_poke_out_before(
            poke_data, odor_start, max(await_reward_time, odor_end + pd.Timedelta(seconds=1)))

        if last_poke_out_time is None:
            # The animal had already left the cue port before this odor's valve opened, so there
            # is no exit to find inside the window -- the target here is an odor it never
            # sampled, because positions are assigned from every valve event regardless of poke.
            # Its last exit *before* the odor is the moment it was free to move, and that is
            # what the response should be timed from. Without this the trial is dropped
            # entirely: 20 of 1243 completed trials on the fixture sessions, every one of them
            # scored rewarded or unrewarded. DECISIONS section 15.
            last_poke_out_time = windows.last_poke_end_before(poke_data.astype(bool), odor_start)

        if last_poke_out_time is None:
            failed_calculations += 1
            per_trial_rows.append(_rt_row(trial_id, np.nan, None, target=target))
            continue

        if is_odour_discrimination:
            current_init_ts = pd.to_datetime(trial_dict.get('initiation_sequence_time'), errors='coerce') \
                if trial_dict.get('initiation_sequence_time') is not None else pd.NaT
            next_init = None
            if not initiation_starts_sorted.empty and not pd.isna(current_init_ts):
                idx = initiation_starts_sorted.searchsorted(current_init_ts, side='right')
                if idx < len(initiation_starts_sorted):
                    next_init = initiation_starts_sorted.iloc[idx]

            next_cue_after_next_init = _next_after(cue_poke_starts_sorted, next_init) if next_init is not None else None
            recording_end = _recording_end(initiation_starts_sorted, cue_poke_starts_sorted,
                                           supply_port1_times, supply_port2_times,
                                           port1_pokes, port2_pokes, trial_end)
            reward_window_end, _next_cue = _odourdisc_reward_window_end(
                next_init, next_cue_after_next_init, await_reward_time,
                cue_poke_starts_sorted, recording_end)
            poke_window_end = reward_window_end
            reward_window_cap = reward_window_end
        else:
            poke_window_end = await_reward_time + pd.Timedelta(seconds=response_time_sec)
            reward_window_cap = trial_end

        search_start = max(last_poke_out_time, await_reward_time)
        all_reward_pokes = (windows.rising_edges_between(port1_pokes, search_start, poke_window_end)
                            + windows.rising_edges_between(port2_pokes, search_start, poke_window_end))

        response_time_ms = None
        window_latency_ms = None
        if all_reward_pokes:
            # Timed from the animal's LAST cue-port exit before the reward poke, not from the
            # exit around the final odor. If it went back to the cue port after AwaitReward --
            # resampling the odor, or checking whether another one was coming -- that is not it
            # travelling to collect, and charging the travel time with it inflates the response.
            # The window that sets the outcome still starts at AwaitReward; only the measurement
            # anchor moves. DECISIONS section 16.
            first_reward_poke = min(all_reward_pokes)
            anchor = windows.last_poke_end_before(poke_series_full, first_reward_poke) or last_poke_out_time
            response_time_ms = (first_reward_poke - anchor).total_seconds() * 1000
            # (a) the same poke measured from where the rig starts its counter. This is what the
            # outcome window is built on, and the two answer different questions -- see the
            # naming note in DECISIONS section 16.
            window_latency_ms = float((first_reward_poke - await_reward_time).total_seconds() * 1000)

        supply1_after_await = [t for t in supply_port1_times if await_reward_time <= t <= reward_window_cap]
        supply2_after_await = [t for t in supply_port2_times if await_reward_time <= t <= reward_window_cap]

        # A poke anywhere in the response window makes an unrewarded trial an error rather than
        # a timeout, even if it landed before the animal left the cue port -- so this window
        # starts at AwaitReward, not at the poke-out that the response time is measured from.
        full_window_pokes = (windows.rising_edges_between(port1_pokes, await_reward_time, poke_window_end)
                             + windows.rising_edges_between(port2_pokes, await_reward_time, poke_window_end))

        outcome = classify_completed_trial(
            supply_count=len(supply1_after_await) + len(supply2_after_await),
            reward_poke_count=len(full_window_pokes),
            has_await_reward=True)

        if outcome in ('rewarded', 'unrewarded'):
            # This function's coverage rule: a category is emitted only when a response time
            # could also be computed. The rest are counted as failures with a null category, and
            # picked up later by save_results._derive_outcome -- DECISIONS section 14.
            if response_time_ms is None:
                failed_calculations += 1
                per_trial_rows.append(_rt_row(trial_id, np.nan, None, target=target))
                continue
            if outcome == 'rewarded':
                rewarded_response_times.append(response_time_ms)
                if hr_success:
                    hr_rewarded_response_times.append(response_time_ms)
            else:
                unrewarded_response_times.append(response_time_ms)
            per_trial_rows.append(_rt_row(trial_id, float(response_time_ms), outcome, target=target,
                                          window_latency_ms=window_latency_ms))
            continue

        # Timeout: look for a delayed response up to the next completed trial.
        next_trial_start = _next_completed_trial_start(trial_end, completed_trials_all)
        extended_search_end = next_trial_start if next_trial_start else (poke_data.index[-1] if not poke_data.empty else poke_window_end)

        delayed_reward_pokes = []
        if poke_window_end < extended_search_end:
            delayed_reward_pokes = (windows.rising_edges_between(port1_pokes, poke_window_end, extended_search_end)
                                    + windows.rising_edges_between(port2_pokes, poke_window_end, extended_search_end))

        if not delayed_reward_pokes:
            failed_calculations += 1
            per_trial_rows.append(_rt_row(trial_id, np.nan, None, target=target))
            continue

        first_delayed = min(delayed_reward_pokes)
        _delayed_anchor = windows.last_poke_end_before(poke_series_full, first_delayed) or last_poke_out_time
        response_time_ms = (first_delayed - _delayed_anchor).total_seconds() * 1000
        timeout_delayed_response_times.append(response_time_ms)
        timeout_response_delay_times.append((first_delayed - poke_window_end).total_seconds() * 1000.0)
        per_trial_rows.append(_rt_row(trial_id, float(response_time_ms), 'timeout_delayed', target=target,
                                      window_latency_ms=float((first_delayed - await_reward_time).total_seconds() * 1000)))

    if verbose:
        _print_response_time_summary(
            len(completed_trials_all), failed_calculations, rewarded_response_times,
            hr_rewarded_response_times, unrewarded_response_times,
            timeout_delayed_response_times, timeout_response_delay_times,
        )

    all_response_times = rewarded_response_times + unrewarded_response_times + timeout_delayed_response_times

    return {
        'rewarded_response_times': rewarded_response_times,
        'unrewarded_response_times': unrewarded_response_times,
        'timeout_delayed_response_times': timeout_delayed_response_times,
        'timeout_response_delay_times': timeout_response_delay_times,
        'all_response_times': all_response_times,
        'failed_calculations': failed_calculations,
        'per_trial': pd.DataFrame(per_trial_rows),
        'sample_offset_time_ms': sample_offset_time_ms,
        'minimum_sampling_time_ms_by_odor': minimum_sampling_time_ms_by_odor,
        'default_minimum_sampling_time_ms': default_minimum_sampling_time_ms,
        'minimum_sampling_time_ms': default_minimum_sampling_time_ms,
        'response_time_window_sec': response_time_sec,
    }

def _norm_fa(val):
    """Canonical false-alarm label. Anything unrecognised, including NaN, is ``'nFA'``."""
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


def _abort_positioned_events(evs_raw, max_positions):
    """Collapse consecutive odor repeats and take the first ``max_positions`` of what remains.

    Returns ``(events, positions)`` as parallel lists, positions numbered from 1.
    """
    evs = windows.collapse_consecutive_odors(evs_raw)[:max_positions]
    return evs, list(range(1, len(evs) + 1))


def _abort_presentations(evs, positions, cue_series, sample_offset_time_ms, required_min_ms_for):
    """Per-position valve and poke record for one aborted trial.

    A position with no poke is kept in ``presentations_all`` (flagged ``has_poke=False``) but
    left out of the position dicts, so the valve/poke maps only describe odors the animal
    actually sampled.

    Returns ``(presentations_all, position_valve_times, position_poke_times)``.
    """
    presentations_all: list[dict] = []
    position_valve_times: dict[int, dict] = {}
    position_poke_times: dict[int, dict] = {}

    for idx_in_trial, (e, pos) in enumerate(zip(evs, positions)):
        valve_start = e['start_time']
        valve_end = e['end_time']
        valve_dur_ms = (valve_end - valve_start).total_seconds() * 1000.0
        required_min_ms = float(required_min_ms_for(e['odor_name']))

        psum = windows.abort_window_poke_summary(cue_series, valve_start, valve_end, sample_offset_time_ms)
        has_poke = float(psum.get('poke_time_ms', 0.0)) > 0.0

        presentations_all.append({
            'index_in_trial': idx_in_trial,
            'position': int(pos),
            'odor_name': e['odor_name'],
            'valve_start': valve_start,
            'valve_end': valve_end,
            'valve_duration_ms': float(valve_dur_ms),
            'poke_time_ms': float(psum.get('poke_time_ms', 0.0)),
            'poke_first_in': psum.get('poke_first_in'),
            'required_min_sampling_time_ms': required_min_ms,
            'has_poke': has_poke,
        })

        if has_poke:
            position_valve_times[int(pos)] = {
                'position': int(pos),
                'odor_name': e['odor_name'],
                'valve_start': valve_start,
                'valve_end': valve_end,
                'valve_duration_ms': float(valve_dur_ms),
                'required_min_sampling_time_ms': required_min_ms,
            }
            psum_pos = dict(psum)
            psum_pos['odor_name'] = e['odor_name']
            psum_pos['required_min_sampling_time_ms'] = required_min_ms
            position_poke_times[int(pos)] = psum_pos

    return presentations_all, position_valve_times, position_poke_times


def _last_relevant_presentation(presentations_valid, sample_offset_time_ms):
    """Index of the last sampled odor whose valve stayed open long enough to be a real choice.

    Scanning backwards skips trailing presentations shorter than ``sampleOffsetTime``, which
    are valve switching artefacts rather than odors the animal could have aborted on. Returns
    ``None`` when no presentation qualifies.
    """
    for i in range(len(presentations_valid) - 1, -1, -1):
        if presentations_valid[i].get('valve_duration_ms', 0.0) >= sample_offset_time_ms:
            return i
    return None


def _abortion_time(cue_intervals, t_start, t_end):
    """When the animal gave up: the last cue-port poke-out inside the trial window."""
    overlapping = [(max(s, t_start), min(e, t_end)) for (s, e) in cue_intervals if e > t_start and s < t_end]
    return overlapping[-1][1] if overlapping else None


def _false_alarm(abortion_time, t_end, *, init_times, cue_rises, reward_rises, dip1_rises,
                 dip2_rises, response_time_ms, port_series, cue_series):
    """Did the animal go to a reward port after aborting? Returns ``(label, time, latency_ms, port)``.

    The window runs from the abortion to the first cue poke after the **next** initiation --
    i.e. until the animal has visibly started the next trial. With no next initiation it runs
    to the last sample in any port stream.

    Returns ``(label, time, latency_ms, port, movement_latency_ms)``. ``latency_ms`` is (a),
    the time since the animal gave up, and is what the label buckets; ``movement_latency_ms``
    is (b), measured from its last cue-port exit before the poke. DECISIONS section 16.
    """
    if abortion_time is None:
        return 'nFA', pd.NaT, np.nan, None, np.nan

    next_init = None
    if init_times:
        idx = bisect_right(init_times, t_end)
        if idx < len(init_times):
            next_init = init_times[idx]

    fa_window_end = None
    if next_init is not None and cue_rises:
        k = bisect_right(cue_rises, next_init)
        if k < len(cue_rises):
            fa_window_end = cue_rises[k]
    if fa_window_end is None:
        candidates = [s.index[-1] for s in port_series if not s.empty]
        fa_window_end = max(candidates) if candidates else abortion_time

    if not reward_rises:
        return 'nFA', pd.NaT, np.nan, None, np.nan

    lo = bisect_right(reward_rises, abortion_time)
    hi = bisect_right(reward_rises, fa_window_end)
    if lo >= hi:
        return 'nFA', pd.NaT, np.nan, None, np.nan

    fa_time = reward_rises[lo]
    fa_latency_ms = (fa_time - abortion_time).total_seconds() * 1000.0
    fa_port = 1 if fa_time in dip1_rises else (2 if fa_time in dip2_rises else None)
    anchor = windows.last_poke_end_before(cue_series, fa_time)
    movement_ms = float((fa_time - anchor).total_seconds() * 1000.0) if anchor is not None else np.nan
    return (latency_label(fa_latency_ms, response_time_ms, 'FA'), fa_time, fa_latency_ms,
            fa_port, movement_ms)


def _build_abortion_index(df: pd.DataFrame):
    """Lookup tables over the aborted-trial table: by trial, position, odor, type and FA label."""
    if df is None or df.empty:
        return {'by_trial': {}, 'by_position': {}, 'by_odor': {}, 'by_type': {}, 'by_fa_label': {}}

    df2 = df.copy().dropna(subset=['trial_id'])
    try:
        by_trial = df2.set_index('trial_id', drop=False).apply(lambda r: r.to_dict(), axis=1).to_dict()
    except Exception:
        by_trial = {row['trial_id']: row.to_dict() for _, row in df2.iterrows()}

    def group_ids(col):
        m = {}
        if col in df2.columns:
            for k, g in df2.groupby(col):
                trials = list(g.sort_values('sequence_start')['trial_id']) if 'sequence_start' in g else list(g['trial_id'])
                m[k] = trials
        return m

    return {
        'by_trial': by_trial,
        'by_position': group_ids('last_odor_position'),
        'by_odor': group_ids('last_odor_name'),
        'by_type': group_ids('abortion_type'),
        'by_fa_label': group_ids('fa_label'),
    }


def _print_abortion_summary(aborted_detailed, classification, response_time, response_time_ms):
    """The aborted-trials summary: abortion types, false alarms, and poke-time breakdowns."""
    def pct(n, d):
        return (n / d * 100.0) if d else 0.0

    def stats_line(series, label):
        s = pd.to_numeric(series, errors='coerce').dropna()
        if s.empty:
            print(f"{label}: n=0")
        else:
            print(f"{label}: n={len(s)} | avg={s.mean():.1f} ms | range={s.min():.1f}-{s.max():.1f} ms")

    def fa_latency_stats(label, indent="          "):
        s = pd.to_numeric(
            aborted_detailed.loc[aborted_detailed['fa_label'] == label, 'fa_latency_ms'],
            errors='coerce',
        ).dropna()
        if len(s):
            print(f"{indent}- Response Time: avg={s.mean():.1f} ms, range: {s.min():.1f} - {s.max():.1f} ms")

    total = int(len(aborted_detailed))
    ini = int((aborted_detailed['abortion_type'] == 'initiation_abortion').sum())
    rei = int((aborted_detailed['abortion_type'] == 'reinitiation_abortion').sum())

    print("=" * 80)
    print("ABORTED TRIALS CLASSIFICATION SUMMARY")
    print("=" * 80)

    print(f"- Total Aborted Trials: {total}")
    print(f"  - Re-Initiation Abortions: {rei} ({pct(rei, total):.1f}%)")
    print(f"  - Initiation Abortions:    {ini} ({pct(ini, total):.1f}%)")

    fa_in_count = int((aborted_detailed['fa_label'] == 'FA_time_in').sum())
    fa_out_count = int((aborted_detailed['fa_label'] == 'FA_time_out').sum())
    fa_late_count = int((aborted_detailed['fa_label'] == 'FA_late').sum())
    fa_total = fa_in_count + fa_out_count + fa_late_count
    nfa_count = total - fa_total

    print("\nFalse Alarms:")
    print(f"  - non-FA Abortions: {nfa_count}")
    print(f"  - False Alarm abortions: {fa_total} ({pct(fa_total, total):.1f}%)")
    if fa_total > 0:
        print(f"      - FA Time In (Within Response Time Window {response_time_ms}):  {fa_in_count} ({pct(fa_in_count, fa_total):.1f}%)")
        fa_latency_stats('FA_time_in')
        print(f"      - FA Time Out (Up to 3x Response Time Window {response_time}):  {fa_out_count} ({pct(fa_out_count, fa_total):.1f}%)")
        fa_latency_stats('FA_time_out')
        print(f"      - FA Late (After 3x Response Time up to next trial):{fa_late_count} ({pct(fa_late_count, fa_total):.1f}%)")
        fa_latency_stats('FA_late')

        hr_positions = classification.get('hidden_rule_positions') or []
        if not hr_positions:
            fallback_pos = classification.get('hidden_rule_position')
            if fallback_pos is not None:
                hr_positions = [fallback_pos]
        hr_positions = [int(pos) for pos in hr_positions if pos is not None]

        if hr_positions:
            abortions_at_hr_pos = aborted_detailed[aborted_detailed['last_odor_position'].isin(hr_positions)].copy()
        else:
            abortions_at_hr_pos = aborted_detailed.iloc[0:0].copy()

        # Resolve HR-aborted trial IDs from classification (robust to key naming)
        hr_ab_df = None
        for k in ('aborted_sequences_HR', 'aborted_HR_sequences', 'aborted_hidden_rule_sequences'):
            if isinstance(classification.get(k), pd.DataFrame) and not classification[k].empty and 'trial_id' in classification[k]:
                hr_ab_df = classification[k]
                break
        if hr_ab_df is not None:
            hr_aborted_ids = set(hr_ab_df['trial_id'])
        elif 'hit_hidden_rule' in abortions_at_hr_pos.columns:
            hr_aborted_ids = set(abortions_at_hr_pos.loc[abortions_at_hr_pos['hit_hidden_rule'] == True, 'trial_id'])
        else:
            hr_aborted_ids = set()

        in_hr_trials = abortions_at_hr_pos[abortions_at_hr_pos['trial_id'].isin(hr_aborted_ids)].copy()
        non_hr_trials = abortions_at_hr_pos[~abortions_at_hr_pos['trial_id'].isin(hr_aborted_ids)].copy()

        def _print_fa_counts(df, indent="    "):
            order = ['nFA', 'FA_time_in', 'FA_time_out', 'FA_late']
            cnt = df['fa_label'].value_counts().reindex(order, fill_value=0)
            n = int(len(df))
            for lbl in order:
                v = int(cnt.get(lbl, 0))
                print(f"{indent}{lbl}: {v} ({(v / n * 100.0) if n else 0.0:.1f}%)")

        hr_pos_display = ", ".join(str(pos) for pos in hr_positions) if hr_positions else "None"
        print(f"\n  Abortions at Hidden Rule Positions {hr_pos_display}: n={int(len(abortions_at_hr_pos))}")

        total_in_hr = int(len(in_hr_trials))
        print(f"    Of which in Hidden Rule Trials: n={total_in_hr}")
        if total_in_hr > 0:
            _print_fa_counts(in_hr_trials, indent="        ")

        total_non_hr = int(len(non_hr_trials))
        print(f"    Non-Hidden Rule Abortions at HR Location: n={total_non_hr}")
        if total_non_hr > 0:
            _print_fa_counts(non_hr_trials, indent="        ")

    # Non-last odor poke times (>= the odor-specific minimum), requires 'presentations'
    if 'presentations' in aborted_detailed.columns and 'last_event_index' in aborted_detailed.columns:
        pres_df = aborted_detailed[['trial_id', 'presentations', 'last_event_index']].explode('presentations')
        pres_df = pres_df.dropna(subset=['presentations']).copy()
        if not pres_df.empty:
            pres = pd.concat(
                [pres_df.drop(columns=['presentations']),
                 pres_df['presentations'].apply(pd.Series)],
                axis=1
            )
            pres['is_last'] = pres['index_in_trial'] == pres['last_event_index']
            pres = pres[~pres['is_last']].copy()

            pres['poke_time_ms'] = pd.to_numeric(pres['poke_time_ms'], errors='coerce')
            pres['required_min_sampling_time_ms'] = pd.to_numeric(
                pres.get('required_min_sampling_time_ms'), errors='coerce'
            )
            pres_valid = pres.dropna(subset=['required_min_sampling_time_ms']).copy()
            pres_valid = pres_valid[
                pres_valid['poke_time_ms'] >= pres_valid['required_min_sampling_time_ms']
            ]

            print("\nNon-last Odor Pokes:")
            stats_line(pres_valid['poke_time_ms'], "  - All non-last odors")

            if 'position' in pres_valid.columns and not pres_valid.empty:
                for pos, grp in pres_valid.groupby('position'):
                    stats_line(grp['poke_time_ms'], f"  - Position {int(pos)}")

            if 'odor_name' in pres_valid.columns and not pres_valid.empty:
                for odor, grp in pres_valid.groupby('odor_name'):
                    stats_line(grp['poke_time_ms'], f"  - Odor {odor}")
        else:
            print("\nNon-last Odor Pokes: n=0 (no presentations info)")
    else:
        print("\nNon-last odor pokes: presentations not attached; update abortion_classification to store 'presentations' and 'last_event_index'.")

    print("\nLast Odor Poke Times:")
    stats_line(
        aborted_detailed.loc[aborted_detailed['abortion_type'] == 'reinitiation_abortion', 'last_odor_poke_time_ms'],
        "  - Re-Initiation Abortions"
    )
    stats_line(
        aborted_detailed.loc[aborted_detailed['abortion_type'] == 'initiation_abortion', 'last_odor_poke_time_ms'],
        "  - Initiation Abortions"
    )

    _print_abortion_counts_by(aborted_detailed, 'last_odor_name', "\nCounts by last odor:",
                              lambda k: f"  - {k}", "  (missing last_odor_name)", sort_keys=False)
    _print_abortion_counts_by(aborted_detailed, 'last_odor_position', "\nCounts by last position:",
                              lambda k: f"  - Position {int(k)}", "  (missing last_odor_position)", sort_keys=True)


def _print_abortion_counts_by(aborted_detailed, column, header, label_for, missing_msg, *, sort_keys):
    """Abortion counts split by re-initiation vs initiation, grouped on ``column``."""
    print(header)
    if column not in aborted_detailed.columns:
        print(missing_msg)
        return
    by_group = (
        aborted_detailed
        .groupby([column, 'abortion_type'])
        .size()
        .unstack(fill_value=0)
        .rename(columns={'reinitiation_abortion': 'Re-initiation', 'initiation_abortion': 'Initiation'})
    )
    totals = aborted_detailed.groupby(column).size()
    keys = sorted(totals.index) if sort_keys else totals.index
    for key in keys:
        rei_c = int(by_group.loc[key].get('Re-initiation', 0))
        ini_c = int(by_group.loc[key].get('Initiation', 0))
        print(f"{label_for(key)}: {int(totals.loc[key])} abortions, Re-initiation {rei_c}, Initiation {ini_c}")


def abortion_classification(data, events, classification, odor_map, root, verbose=True):
    """Classify aborted trials: where the animal gave up, and whether it went to a port anyway.

    An abortion is *re-initiation* when the last odor the animal properly sampled met that
    odor's minimum sampling time -- it engaged with the odor and then chose to leave -- and
    *initiation* when it did not, meaning it never really committed. The "last odor" skips
    trailing valve openings shorter than ``sampleOffsetTime``, which are switching artefacts.

    A reward-port poke after the abortion and before the animal re-engages with the next trial
    is a false alarm, labelled by latency in units of the response window.

    Returns the detailed aborted-trial DataFrame, and attaches it plus a lookup index to
    ``classification`` in place.
    """
    schema_settings = {}
    try:
        _, schema_settings = detect_settings.detect_settings(root)
    except Exception:
        schema_settings = {}

    seq_len = schema_settings.get('sequenceLength')
    max_positions = int(seq_len) if seq_len is not None else None
    if max_positions is None or max_positions < 1:
        raise ValueError("sequenceLength missing or invalid; cannot proceed without a valid sequence length")

    DIP0 = data['digital_input_data'].get('DIPort0', pd.Series(dtype=bool)).astype(bool)  # cue port
    DIP1 = data['digital_input_data'].get('DIPort1', pd.Series(dtype=bool)).astype(bool)  # reward port 1
    DIP2 = data['digital_input_data'].get('DIPort2', pd.Series(dtype=bool)).astype(bool)  # reward port 2

    dip1_rises = windows.rising_edges(DIP1)
    dip2_rises = windows.rising_edges(DIP2)
    reward_rises = sorted(dip1_rises + dip2_rises)
    cue_rises = windows.rising_edges(DIP0)
    cue_intervals = windows.paired_intervals(DIP0)

    sample_offset_time, minimum_sampling_time_by_odor, response_time = get_experiment_parameters(root)
    sample_offset_time_ms = float(sample_offset_time) * 1000.0
    minimum_sampling_time_ms_by_odor = {
        str(odor): float(threshold) * 1000.0
        for odor, threshold in (minimum_sampling_time_by_odor or {}).items()
        if threshold is not None
    }

    # Thresholds already resolved by classify_trials win: they are the ones the trials in this
    # very classification dict were detected with.
    cls_minimums = classification.get('minimum_sampling_time_ms_by_odor') if isinstance(classification, dict) else None
    if isinstance(cls_minimums, dict):
        for odor, threshold in cls_minimums.items():
            if threshold is None:
                continue
            try:
                minimum_sampling_time_ms_by_odor[str(odor)] = float(threshold)
            except (TypeError, ValueError):
                continue

    if not minimum_sampling_time_ms_by_odor:
        raise ValueError("minimumSamplingTime_by_odor missing or empty; cannot classify aborted trials without per-odor thresholds")

    default_minimum_sampling_time_ms = classification.get('default_minimum_sampling_time_ms') if isinstance(classification, dict) else None
    if default_minimum_sampling_time_ms is None:
        default_minimum_sampling_time_ms = max(minimum_sampling_time_ms_by_odor.values())

    def required_min_ms_for(odor_name):
        if odor_name is None:
            return default_minimum_sampling_time_ms
        return minimum_sampling_time_ms_by_odor.get(str(odor_name), default_minimum_sampling_time_ms)

    response_time_ms = float(response_time) * 1000.0

    aborted_df = classification.get('aborted_sequences', pd.DataFrame())
    if not isinstance(aborted_df, pd.DataFrame) or aborted_df.empty:
        if verbose:
            print("abortion_classification: no aborted trials found.")
        return pd.DataFrame()

    all_valve_activations = windows.valve_windows_with_grid_fallback(odor_map)

    init_times = []
    ci_key = 'combined_initiation_sequence_df'
    if ci_key in events and isinstance(events[ci_key], pd.DataFrame) and not events[ci_key].empty:
        init_times = list(events[ci_key]['Time'])

    rows = []
    for _, tr in aborted_df.iterrows():
        t_start = tr.get('sequence_start') or tr.get('trial_start') or tr.get('start_time')
        t_end = tr.get('sequence_end') or tr.get('trial_end') or tr.get('end_time')
        if pd.isna(t_start) or pd.isna(t_end) or t_start is None or t_end is None:
            continue

        evs, positions = _abort_positioned_events(
            windows.valve_events_strictly_inside(all_valve_activations, t_start, t_end), max_positions)

        presentations_all, position_valve_times, position_poke_times = _abort_presentations(
            evs, positions, DIP0, sample_offset_time_ms, required_min_ms_for)

        presentations_valid = [p for p in presentations_all if p.get('has_poke')]
        last_idx = _last_relevant_presentation(presentations_valid, sample_offset_time_ms)
        for idx, pres_entry in enumerate(presentations_valid):
            pres_entry['is_last_event'] = last_idx is not None and idx == last_idx

        last_odor_name = None
        last_odor_pos = None
        last_valve_dur_ms = 0.0
        last_odor_poke_ms = 0.0
        last_required_min_ms = float('nan')
        if last_idx is not None and presentations_valid:
            last_pres = presentations_valid[last_idx]
            last_odor_name = last_pres.get('odor_name')
            last_odor_pos = last_pres.get('position')
            last_valve_dur_ms = float(last_pres.get('valve_duration_ms', 0.0) or 0.0)
            last_odor_poke_ms = float(last_pres.get('poke_time_ms', 0.0) or 0.0)
            last_required_min_ms = float(required_min_ms_for(last_odor_name))

        abortion_type = (
            'reinitiation_abortion'
            if (not np.isnan(last_required_min_ms) and last_odor_poke_ms >= last_required_min_ms)
            else 'initiation_abortion'
        )

        abortion_time = _abortion_time(cue_intervals, t_start, t_end)
        fa_label, fa_time, fa_latency_ms, fa_port, fa_movement_ms = _false_alarm(
            abortion_time, t_end, init_times=init_times, cue_rises=cue_rises,
            reward_rises=reward_rises, dip1_rises=dip1_rises, dip2_rises=dip2_rises,
            response_time_ms=response_time_ms, port_series=[DIP0, DIP1, DIP2], cue_series=DIP0)

        rows.append({
            'trial_id': tr.get('trial_id', tr.name),
            'sequence_start': t_start,
            'sequence_end': t_end,
            'odor_sequence': [p['odor_name'] for p in presentations_valid],
            'presentations': presentations_valid,
            'last_event_index': last_idx,
            'position_valve_times': position_valve_times,
            'position_poke_times': position_poke_times,
            'last_odor_position': last_odor_pos,
            'last_odor_name': last_odor_name,
            'last_odor_valve_duration_ms': float(last_valve_dur_ms),
            'last_odor_poke_time_ms': float(last_odor_poke_ms),
            'last_required_min_sampling_time_ms': float(last_required_min_ms) if not np.isnan(last_required_min_ms) else np.nan,
            'abortion_type': abortion_type,
            'abortion_time': abortion_time,
            'fa_label': fa_label,
            'fa_time': fa_time,
            'fa_latency_ms': float(fa_latency_ms) if pd.notna(fa_latency_ms) else np.nan,
            'fa_port': fa_port,
            'fa_movement_latency_ms': fa_movement_ms,
        })

    aborted_detailed = pd.DataFrame(rows)
    aborted_detailed['fa_label'] = aborted_detailed['fa_label'].apply(_norm_fa)

    if verbose and not aborted_detailed.empty:
        _print_abortion_summary(aborted_detailed, classification, response_time, response_time_ms)

    aborted_index = _build_abortion_index(aborted_detailed)

    try:
        classification['aborted_sequences_detailed'] = aborted_detailed
        classification['aborted_index'] = aborted_index
    except Exception:
        pass

    return aborted_detailed

def classify_noninitiated_FA(noninit_df, DIP0, DIP1, DIP2, response_time, hr_odors=None):
    """Classify False Alarms in non-initiated trials"""
    
    results = []
    
    # Get port rises
    dip1_rises = DIP1[DIP1 & ~DIP1.shift(1, fill_value=False)].index.tolist()
    dip2_rises = DIP2[DIP2 & ~DIP2.shift(1, fill_value=False)].index.tolist()
    reward_rises = sorted(dip1_rises + dip2_rises)
    
    cue_rises = list(DIP0[DIP0 & ~DIP0.shift(1, fill_value=False)].index)
    response_time_ms = float(response_time) * 1000.0

    for _, row in noninit_df.iterrows():
        attempt_end = row.get('attempt_end')
        if pd.isna(attempt_end):
            continue
            
        # Find next cue port poke-in after attempt_end
        next_cue_in = None
        cue_after = [t for t in cue_rises if t > attempt_end]
        if cue_after:
            next_cue_in = cue_after[0]
        else:
            next_cue_in = max(DIP0.index) if not DIP0.empty else attempt_end

        # Scan for first reward-port poke in (attempt_end, next_cue_in]
        fa_label = 'nFA'
        fa_time = pd.NaT
        fa_latency_ms = np.nan
        fa_port = None
        fa_movement_ms = np.nan

        reward_after = [t for t in reward_rises if attempt_end < t <= next_cue_in]
        if reward_after:
            fa_time = reward_after[0]
            fa_latency_ms = (fa_time - attempt_end).total_seconds() * 1000.0
            
            # Determine which port ← NEW
            if fa_time in dip1_rises:
                fa_port = 1
            elif fa_time in dip2_rises:
                fa_port = 2

            fa_label = latency_label(fa_latency_ms, response_time_ms, 'FA')

        # HR status for position 1
        is_hr = False
        if hr_odors is not None:
            odor_name = row.get('odor_name')
            is_hr = odor_name in hr_odors

        results.append({
            **row.to_dict(),
            'fa_label': fa_label,
            'fa_time': fa_time,
            'fa_latency_ms': fa_latency_ms,
            'fa_port': fa_port,
            'fa_movement_latency_ms': fa_movement_ms,
            'is_hr': is_hr
        })
        
    return pd.DataFrame(results)

def build_classification_index(classification: dict) -> dict: # Classification function for easier dictionary access later on
    """
    Build convenient lookup indices over classification outputs.
    Provides:
      - by_trial: trial_id -> full row dict (completed_with_RT preferred, else completed, else aborted_detailed)
      - categories.completed.*_ids: lists of trial_ids for major completed categories (and HR variants)
      - sets.*: quick sets of IDs for initiated, completed, aborted
      - aborted: re-exposes the aborted_index (by_position/by_odor/by_type/by_fa_label)
    """

    idx = {'by_trial': {}, 'categories': {'completed': {}}, 'sets': {}, 'aborted': {}}

    # Prefer completed_with_RT for richer rows
    comp_df = classification.get('completed_sequences_with_response_times')
    if not isinstance(comp_df, pd.DataFrame) or comp_df.empty:
        comp_df = classification.get('completed_sequences', pd.DataFrame())

    ab_det = classification.get('aborted_sequences_detailed')
    ab_df = ab_det if isinstance(ab_det, pd.DataFrame) else classification.get('aborted_sequences', pd.DataFrame())

    # by_trial: completed first (wins), then aborted to fill missing ones
    if isinstance(comp_df, pd.DataFrame) and not comp_df.empty and 'trial_id' in comp_df:
        for _, r in comp_df.iterrows():
            tid = r.get('trial_id')
            if pd.notna(tid):
                idx['by_trial'][tid] = r.to_dict()
    if isinstance(ab_df, pd.DataFrame) and not ab_df.empty and 'trial_id' in ab_df:
        for _, r in ab_df.iterrows():
            tid = r.get('trial_id')
            if pd.notna(tid) and tid not in idx['by_trial']:
                idx['by_trial'][tid] = r.to_dict()

    # Completed category ID lists
    def ids_from(name):
        df = classification.get(name, pd.DataFrame())
        return [] if not isinstance(df, pd.DataFrame) or df.empty or 'trial_id' not in df else list(df['trial_id'])

    c = idx['categories']['completed']
    c['rewarded_ids'] = ids_from('completed_sequence_rewarded')
    c['unrewarded_ids'] = ids_from('completed_sequence_unrewarded')
    c['timeout_ids'] = ids_from('completed_sequence_reward_timeout')

    # Single-reward protocol: completed non-rewarded ("no-go") sequences. Empty for default protocol.
    c['false_response_ids'] = ids_from('completed_sequence_false_response')

    c['hr_rewarded_ids'] = ids_from('completed_sequence_HR_rewarded')
    c['hr_unrewarded_ids'] = ids_from('completed_sequence_HR_unrewarded')
    c['hr_timeout_ids'] = ids_from('completed_sequence_HR_reward_timeout')

    c['hr_missed_rewarded_ids'] = ids_from('completed_sequence_HR_missed_rewarded')
    c['hr_missed_unrewarded_ids'] = ids_from('completed_sequence_HR_missed_unrewarded')
    c['hr_missed_timeout_ids'] = ids_from('completed_sequence_HR_missed_reward_timeout')

    # Sets for quick membership tests
    idx['sets']['initiated_ids'] = (
        set(classification['initiated_sequences']['trial_id']) 
        if isinstance(classification.get('initiated_sequences'), pd.DataFrame) 
        and 'trial_id' in classification['initiated_sequences'] else set()
    )
    idx['sets']['completed_ids'] = set(comp_df['trial_id']) if isinstance(comp_df, pd.DataFrame) and 'trial_id' in comp_df else set()
    idx['sets']['aborted_ids'] = (
        set(classification['aborted_sequences']['trial_id']) 
        if isinstance(classification.get('aborted_sequences'), pd.DataFrame) 
        and 'trial_id' in classification['aborted_sequences'] else set()
    )

    # Aborted sub-index (already built by abortion_classification)
    ab_index = classification.get('aborted_index')
    if isinstance(ab_index, dict):
        idx['aborted'] = ab_index
    else:
        # Minimal fallback
        idx['aborted'] = {'by_trial': {}, 'by_position': {}, 'by_odor': {}, 'by_type': {}, 'by_fa_label': {}}
        if isinstance(ab_df, pd.DataFrame) and not ab_df.empty:
            try:
                idx['aborted']['by_trial'] = ab_df.set_index('trial_id', drop=False).apply(lambda r: r.to_dict(), axis=1).to_dict()
            except Exception:
                idx['aborted']['by_trial'] = {r['trial_id']: r.to_dict() for _, r in ab_df.dropna(subset=['trial_id']).iterrows()}
            def group_ids(col):
                out = {}
                if col in ab_df.columns:
                    for k, g in ab_df.groupby(col):
                        out[k] = list(g.sort_values('sequence_start')['trial_id']) if 'sequence_start' in g else list(g['trial_id'])
                return out
            for col, key in [('last_odor_position','by_position'), ('last_odor_name','by_odor'), ('abortion_type','by_type'), ('fa_label','by_fa_label')]:
                idx['aborted'][key] = group_ids(col)

    return idx

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


# Saving, merging, the summary report, the multi-run runner and plot_valve_and_poke_events all
# live in their own modules (io/save_results, merge, summary, run, visualization/valve_poke_plots).
# They are NOT re-exported here: import them from where they are defined. The re-exports were the
# last back-compat shims from v1.0.0, and keeping them forced run.py and merge.py to import this
# module lazily to dodge the cycle they created -- and made trial_classification depend on
# visualization, which is backwards.
