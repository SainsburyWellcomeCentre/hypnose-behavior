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

PRE_ODOR_GRACE_MS = 25.0



# ============== General Utility Functions and Class Definitions =======================================
# Reader classes and loaders moved to hypnose_behavior.io.loaders during the restructuring; re-exported
# here so existing callers (the runner/plotting below, notebooks) keep importing them from
# classification_utils.
from hypnose_behavior.io.loaders import (  # noqa: F401
    SessionData, Video, TimestampedCsvReader,
    load, load_json, load_video, load_csv, concat_digi_events,
    load_experiment, exp_data, load_all_streams, load_experiment_events, load_odor_mapping,
)


def vprint(verbose: bool, *args, **kwargs):
    if verbose:
        print(*args, **kwargs)


def _last_poke_end_before(series_bool: pd.Series, ts: pd.Timestamp | None) -> pd.Timestamp | None:
    if ts is None or series_bool is None or series_bool.empty:
        return None
    before = series_bool.loc[:ts]
    if before.empty:
        return None
    falls = ~before & before.shift(1, fill_value=False)
    if not falls.any():
        return None
    return falls[falls].index[-1]


def _grace_overlap_ms(last_poke_end, window_start, window_end, grace_ms: float = PRE_ODOR_GRACE_MS) -> tuple[float, pd.Timestamp | None]:
    if last_poke_end is None or window_start is None or window_end is None:
        return 0.0, None
    if last_poke_end > window_start:
        return 0.0, None
    grace_end = last_poke_end + pd.Timedelta(milliseconds=grace_ms)
    if grace_end <= window_start:
        return 0.0, None
    overlap_end = min(window_end, grace_end)
    if overlap_end <= window_start:
        return 0.0, None
    overlap_ms = (overlap_end - window_start).total_seconds() * 1000.0
    return float(overlap_ms), overlap_end


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

def classify_trials(data, events, trial_counts, odor_map, stage, root, verbose=True, single_reward_info=None):# Classify trials and get valve/poke times. Part of wrapper function
    """
    Same classification as classify_trial_outcomes_extensive, plus:
      - position_valve_times and position_poke_times per trial
      - summary printouts for poke/valve time ranges by position and by odor
    Response-time analysis is removed.
    """
    sample_offset_time, minimum_sampling_time_by_odor, response_time = get_experiment_parameters(root)
    # Convert to milliseconds for consistency with existing logic
    sample_offset_time_ms = sample_offset_time * 1000
    minimum_sampling_time_ms_by_odor = {
        str(odor): float(threshold) * 1000.0
        for odor, threshold in (minimum_sampling_time_by_odor or {}).items()
        if threshold is not None
    }
    if not minimum_sampling_time_ms_by_odor:
        raise ValueError("minimumSamplingTime_by_odor missing or empty in schema; cannot classify trials without per-odor thresholds")
    default_minimum_sampling_time_ms = max(minimum_sampling_time_ms_by_odor.values())

    def resolve_min_sampling_time_ms(odor_name):
        if odor_name is None:
            return default_minimum_sampling_time_ms
        return minimum_sampling_time_ms_by_odor.get(str(odor_name), default_minimum_sampling_time_ms)

    response_time_sec = response_time 
    if response_time_sec is None:
        raise ValueError("Response time parameter cannot be extracted from Schema file. Check detect_settings function.")

    if verbose:
        print("=" * 80)
        print("CLASSIFYING TRIAL OUTCOMES WITH HIDDEN RULE AND VALVE/POKE TIME ANALYSIS")
        print("=" * 80)
        print(f"Sample offset time: {sample_offset_time_ms} ms")
        print("Minimum sampling times (ms) by odor:")
        for odor_name, threshold in sorted(minimum_sampling_time_ms_by_odor.items()):
            print(f"  - {odor_name}: {threshold:.1f}")
        print(f"Response time window: {response_time_sec} s")

    hidden_rule_indices, sequence_name = _resolve_hidden_rule_from_stage(stage)
    protocol_name = (sequence_name or str(stage) or "").lower()
    is_odour_discrimination = "odourdiscrimination" in protocol_name
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
    hidden_rule_positions = [idx + 1 for idx in hidden_rule_indices]
    hidden_rule_location = hidden_rule_indices[0] if hidden_rule_indices else None
    hidden_rule_position = hidden_rule_positions[0] if hidden_rule_positions else None
    multiple_hidden_rule_locations = len(hidden_rule_positions) > 1

    # Sequence length (positions) for all position-based loops
    seq_len = schema_settings.get('sequenceLength')
    max_positions = int(seq_len) if seq_len is not None else None
    if max_positions is None or max_positions < 1:
        raise ValueError("sequenceLength missing or invalid; cannot proceed without a valid sequence length")

    if verbose:
        seq_label = sequence_name or str(stage)
        if hidden_rule_indices:
            if multiple_hidden_rule_locations:
                pos_str = ", ".join(str(p) for p in hidden_rule_positions)
                idx_str = ", ".join(str(idx) for idx in hidden_rule_indices)
                print(f"Hidden rule locations extracted: Positions {pos_str} (indices {idx_str})")
            else:
                print(f"Hidden rule location extracted: Location{hidden_rule_location} (index {hidden_rule_location}, position {hidden_rule_position})")
        else:
            print(f"No Hidden Rule Location found in sequence name: {seq_label}. Proceeding without Hidden Rule analysis.")

    # Base trial data
    initiated_trials = trial_counts['initiated_sequences'].copy()
    non_initiated_trials = trial_counts['non_initiated_sequences'].copy()
    # Sorted initiation starts (canonical list). Use ONLY initiation_sequence_time.
    init_series_raw = initiated_trials.get('initiation_sequence_time')
    initiation_starts_sorted = pd.to_datetime(init_series_raw, errors='coerce').dropna().sort_values().reset_index(drop=True)

    # Ground-truth initiation events (kept for compatibility; should mirror initiation_starts_sorted)
    init_events_df = events.get('combined_initiation_sequence_df', pd.DataFrame())
    initiation_event_times_sorted = pd.to_datetime(
        init_events_df.get('Time', pd.Series(dtype='datetime64[ns]')),
        errors='coerce'
    ).dropna().sort_values().reset_index(drop=True)
    # Events
    await_reward_times = events['combined_await_reward_df']['Time'].tolist() if 'combined_await_reward_df' in events else []

    # Supply port activities
    supply_port1_times = data['pulse_supply_1'].index.tolist() if not data['pulse_supply_1'].empty else []
    supply_port2_times = data['pulse_supply_2'].index.tolist() if not data['pulse_supply_2'].empty else []
    all_supply_port_times = sorted(supply_port1_times + supply_port2_times)

    # Reward port pokes
    port1_pokes = data['digital_input_data'].get('DIPort1', pd.Series(dtype=bool))
    port2_pokes = data['digital_input_data'].get('DIPort2', pd.Series(dtype=bool))

    # Nose-poke data (Port0) for poke-time analysis during odors
    poke_data = data['digital_input_data'].get('DIPort0', pd.Series(dtype=bool))

    poke_series_full = poke_data.astype(bool)
    poke_series_full = poke_series_full.sort_index()
    _rises = poke_series_full & ~poke_series_full.shift(1, fill_value=False)
    _falls = ~poke_series_full & poke_series_full.shift(1, fill_value=False)
    _starts = list(poke_series_full.index[_rises])
    _ends = list(poke_series_full.index[_falls])
    cue_poke_starts_sorted = pd.Series(_starts, dtype='datetime64[ns]').sort_values() if _starts else pd.Series(dtype='datetime64[ns]')
    poke_intervals = []
    i = j = 0
    while i < len(_starts) and j < len(_ends):
        if _ends[j] <= _starts[i]:
            j += 1
            continue
        poke_intervals.append((_starts[i], _ends[j]))
        i += 1
        j += 1
    # If the series starts IN without a detected start edge, optionally prepend
    if poke_series_full.size and poke_series_full.iloc[0] and (not _starts or poke_series_full.index[0] < _starts[0]):
        # close it at the first fall after the beginning
        first_fall = next((t for t in _ends if t > poke_series_full.index[0]), None)
        if first_fall is not None:
            poke_intervals.insert(0, (poke_series_full.index[0], first_fall))

    # Build valve activation list
    olfactometer_valves = odor_map['olfactometer_valves']
    valve_to_odor = odor_map['valve_to_odor']

    all_valve_activations = []
    for olf_id, valve_data in olfactometer_valves.items():
        if valve_data.empty:
            continue
        for i, valve_col in enumerate(valve_data.columns):
            valve_key = f"{olf_id}{i}"
            if valve_key not in valve_to_odor:
                continue
            odor_name = valve_to_odor[valve_key]
            if odor_name.lower() == 'purge':
                continue

            valve_series = valve_data[valve_col]
            valve_activations = valve_series & ~valve_series.shift(1, fill_value=False)
            activation_times = valve_activations[valve_activations == True].index.tolist()
            valve_deactivations = ~valve_series & valve_series.shift(1, fill_value=False)
            deactivation_times = valve_deactivations[valve_deactivations == True].index.tolist()

            for activation_time in activation_times:
                next_deactivations = [t for t in deactivation_times if t > activation_time]
                deactivation_time = min(next_deactivations) if next_deactivations else valve_series.index[-1]
                all_valve_activations.append({
                    'start_time': activation_time,
                    'end_time': deactivation_time,
                    'odor_name': odor_name,
                    'valve_key': valve_key
                })

    all_valve_activations.sort(key=lambda x: x['start_time'])

    if verbose:
        print(f"Found {len(all_valve_activations)} total valve activations (excluding Purge)")
        print(f"Analyzing {len(initiated_trials)} initiated trials...")
        print(f"Found {len(await_reward_times)} AwaitReward events")
        print(f"Found {len(all_supply_port_times)} total supply port activities")

    # Result containers
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
    initiated_trials = trial_counts['initiated_sequences'].copy()
    initiated_trials_list = []

    # Single-reward (new) protocol info: (is_single_reward, rewarded sequences, all candidate sequences).
    if single_reward_info is None:
        single_reward_info = _get_single_reward_info(root)
    is_single_reward, rewarded_sequences, all_sequences = single_reward_info
    # The final position of a full sequence is always the reward position, so it can
    # never be a hidden-rule position — drop it (single-reward left untouched).
    hidden_rule_indices = _drop_final_hidden_rule_index(hidden_rule_indices, schema_settings, is_single_reward)
    hidden_rule_positions = [idx + 1 for idx in hidden_rule_indices]
    hidden_rule_location = hidden_rule_indices[0] if hidden_rule_indices else None
    hidden_rule_position = hidden_rule_positions[0] if hidden_rule_positions else None
    multiple_hidden_rule_locations = len(hidden_rule_positions) > 1
    response_time_ms_window = float(response_time_sec) * 1000.0 if response_time_sec is not None else None

    # Aggregators for summary prints (completed trials only)
    agg_position_poke_times = {pos: [] for pos in range(1, max_positions + 1)}
    agg_position_valve_times = {pos: [] for pos in range(1, max_positions + 1)}
    agg_odor_poke_times = defaultdict(list)
    agg_odor_valve_times = defaultdict(list)

    # Helpers
    def get_trial_valve_sequence(trial_start, trial_end):
        trial_valve_activations = []
        for valve_activation in all_valve_activations:
            valve_start = valve_activation['start_time']
            valve_end = valve_activation['end_time']
            if valve_start <= trial_end and valve_end >= trial_start:
                trial_valve_activations.append(valve_activation)
        trial_valve_activations.sort(key=lambda x: x['start_time'])
        odor_sequence = [activation['odor_name'] for activation in trial_valve_activations]
        return odor_sequence, trial_valve_activations


    hr_odor_set = None
    if hidden_rule_indices:
        try:
            if schema_err is not None and 'hiddenRuleOdorsInferred' not in schema_settings:
                raise ValueError(str(schema_err))
            odors = (schema_settings.get('hiddenRuleOdorsInferred') or [])
            if len(odors) < 2:
                raise ValueError("found fewer than two rewarded odors at inferred hidden rule position.")
            hr_odor_set = set(map(str, odors))
            if verbose:
                print(f"Hidden Rule Odors inferred: {sorted(hr_odor_set)}")
        except Exception as e:
            raise ValueError(f"Hidden Rule Odor Identities could not be inferred from Schema: {e}")

    def check_hidden_rule(odor_sequence, candidate_indices, odor_set):
        if not candidate_indices or odor_set is None:
            return False, False, []

        valid_indices = [idx for idx in candidate_indices if 0 <= idx < len(odor_sequence)]
        if not valid_indices:
            return False, False, []

        matching_indices = sorted({idx for idx in valid_indices if odor_sequence[idx] in odor_set})
        return True, bool(matching_indices), matching_indices
    
    def window_poke_summary(window_start, window_end):
        if window_start is None or window_end is None or window_start >= window_end:
            return {'poke_time_ms': 0.0, 'poke_first_in': None, 'poke_odor_start': window_start, 'poke_odor_end': None}
        
        s_bool = poke_series_full
        prev = s_bool.loc[:window_start]
        in_at_start = bool(prev.iloc[-1]) if len(prev) else False
        w = s_bool.loc[window_start:window_end]
        
        if w.empty and not in_at_start:
            return {'poke_time_ms': 0.0, 'poke_first_in': None, 'poke_odor_start': window_start, 'poke_odor_end': None}
        
        rises = w & ~w.shift(1, fill_value=in_at_start)
        falls = ~w & w.shift(1, fill_value=in_at_start)
        intervals = []
        cur = window_start if in_at_start else None
        first_in = window_start if in_at_start else None
        
        for ts in w.index:
            if rises.get(ts, False) and cur is None:
                cur = ts
                if first_in is None:
                    first_in = ts
            if falls.get(ts, False) and cur is not None:
                intervals.append((cur, ts))
                cur = None
        
        if cur is not None:
            intervals.append((cur, window_end))
        
        if not intervals:
            last_poke_end = _last_poke_end_before(s_bool, window_start)
            grace_ms, grace_end = _grace_overlap_ms(last_poke_end, window_start, window_end)
            if grace_ms > 0.0:
                return {
                    'poke_time_ms': grace_ms,
                    'poke_first_in': window_start,
                    'poke_odor_start': window_start,
                    'poke_odor_end': grace_end,
                }
            return {'poke_time_ms': 0.0, 'poke_first_in': None, 'poke_odor_start': window_start, 'poke_odor_end': None}
        
        # Merge across gaps <= sample_offset_time_ms
        merged = [intervals[0]]
        for s2, e2 in intervals[1:]:
            ls, le = merged[-1]
            gap_ms = (s2 - le).total_seconds() * 1000.0
            if gap_ms <= sample_offset_time_ms:
                # Extend but cap at window_end
                merged[-1] = (ls, min(max(le, e2), window_end))
            else:
                merged.append((s2, e2))
        
        # Extract first block and cap at window_end
        first_block_start, first_block_end = merged[0]
        first_block_end_capped = min(first_block_end, window_end)
        first_block_ms = max(0.0, (first_block_end_capped - first_block_start).total_seconds() * 1000.0)
        
        return {
            'poke_time_ms': float(first_block_ms),
            'poke_first_in': first_in,
            'poke_odor_start': window_start,
            'poke_odor_end': first_block_end_capped,
        }

    
    def _attempt_bout_from_poke_in(anchor_ts, cap_end=None):
        """
        Return (first_in_ts, bout_end_ts_capped, duration_ms) for the attempt whose valve starts at anchor_ts.
        - If anchor_ts falls inside an IN interval, start at that interval's start.
        - Else, use the first IN interval that starts at/after anchor_ts.
        - Merge backward across previous IN intervals while OUT gaps <= sample_offset_time_ms
        (to include pre-anchor pokes that are part of the same bout).
        - Merge forward across subsequent IN intervals while OUT gaps <= sample_offset_time_ms.
        - Cap the merged bout at cap_end if provided.
        """
        if anchor_ts is None or not poke_intervals:
            return None, None, 0.0

        starts_only = [s for s, _ in poke_intervals]

        # Find interval covering anchor or the first one after
        from bisect import bisect_left, bisect_right
        idx = bisect_right(starts_only, anchor_ts) - 1
        if 0 <= idx < len(poke_intervals) and poke_intervals[idx][0] <= anchor_ts < poke_intervals[idx][1]:
            k = idx
        else:
            k = bisect_left(starts_only, anchor_ts)
            if k >= len(poke_intervals):
                return None, None, 0.0

        # Start with the interval at k
        bout_start, bout_end = poke_intervals[k]

        # Backward merge: include prior intervals if the gap <= sample_offset_time_ms
        m = k
        while m - 1 >= 0:
            prev_start, prev_end = poke_intervals[m - 1]
            if cap_end is not None and prev_start < cap_end:
                # Don't merge intervals that start before cap_end
                break
            gap_ms = (bout_start - prev_end).total_seconds() * 1000.0
            if gap_ms <= sample_offset_time_ms:
                bout_start = prev_start
                m -= 1
            else:
                break

        # Forward merge: include next intervals if the gap <= sample_offset_time_ms (respect cap_end)
        n = k
        cur_end = bout_end
        while n + 1 < len(poke_intervals):
            next_start, next_end = poke_intervals[n + 1]
            if cap_end is not None and next_start >= cap_end:
                break
            gap_ms = (next_start - cur_end).total_seconds() * 1000.0
            if gap_ms <= sample_offset_time_ms:
                cur_end = max(cur_end, min(next_end, cap_end))
                n += 1
            else:
                break

        # Cap forward end at cap_end if provided
        bout_end_capped = cur_end
        if cap_end is not None and bout_end_capped is not None and bout_end_capped > cap_end:
            bout_end_capped = cap_end

        dur_ms = max(0.0, (bout_end_capped - bout_start).total_seconds() * 1000.0)
        return bout_start, bout_end_capped, float(dur_ms)

    def analyze_trial_valve_and_poke_times(trial_valve_events):
        position_locations = {}
        position_valve_times = {}
        position_poke_times = {}
        prior_presentations = []

        # Collapse consecutive repeats of the same odor but keep non-consecutive re-entries
        dedup_events: list[dict] = []
        for ev in trial_valve_events:
            if dedup_events and dedup_events[-1]['odor_name'] == ev['odor_name']:
                # Keep only the latest activation in a consecutive block
                dedup_events[-1] = ev
            else:
                dedup_events.append(ev)

        # Position 1: last individual activation of first odor
        if trial_valve_events:
            first_odor_valve = trial_valve_events[0]['valve_key']
            first_odor_activations = []
            for event in trial_valve_events:
                if event['valve_key'] == first_odor_valve:
                    first_odor_activations.append(event)
                else:
                    break
            if first_odor_activations:
                # Position 1 = LAST activation of first odor
                position_locations[1] = first_odor_activations[-1]
                # Earlier activations are prior_presentations (failed attempts)
                prior_presentations = [
                    {
                        'position': 1,
                        'odor_name': e['odor_name'],
                        'valve_start': e['start_time'],
                        'valve_end': e['end_time'],
                        'required_min_sampling_time_ms': resolve_min_sampling_time_ms(e['odor_name'])
                    }
                    for e in first_odor_activations[:-1]
                ]

        # Positions 2..: assign sequentially from the deduplicated (consecutive-collapsed) events
        next_pos = 2

        for event in dedup_events[1:]:
            if next_pos > max_positions:
                break
            position_locations[next_pos] = event
            next_pos += 1

        # Valve timing per position
        for position in range(1, max_positions + 1):
            if position not in position_locations:
                continue
            loc = position_locations[position]
            valve_start = loc['start_time']
            valve_end = loc['end_time']
            valve_duration_ms = (valve_end - valve_start).total_seconds() * 1000
            entry = {
                'position': position,
                'odor_name': loc['odor_name'],
                'valve_start': valve_start,
                'valve_end': valve_end,
                'valve_duration_ms': valve_duration_ms,
                'required_min_sampling_time_ms': resolve_min_sampling_time_ms(loc['odor_name'])
            }
            if position == 1:
                entry['prior_presentations'] = prior_presentations
            position_valve_times[position] = entry 

        # Poke-time analysis: use ONLY the LAST valve event per position
        poke_position_locations = dict(position_locations)

        s_bool = poke_data.astype(bool)

        # Compute consolidated poke time
        for position in range(1, max_positions + 1):
            if position not in poke_position_locations:
                continue
            loc = poke_position_locations[position]
            odor_start = loc['start_time']
            odor_end = loc['end_time']

            # State at window start
            prev_slice = s_bool.loc[:odor_start]
            state_at_start = bool(prev_slice.iloc[-1]) if len(prev_slice) else False

            # Window slice
            w = s_bool.loc[odor_start:odor_end]
            if w.empty and not state_at_start:
                last_poke_end = _last_poke_end_before(s_bool, odor_start)
                grace_ms, grace_end = _grace_overlap_ms(last_poke_end, odor_start, odor_end)
                if grace_ms > 0.0:
                    position_poke_times[position] = {
                        'position': position,
                        'odor_name': loc['odor_name'],
                        'poke_time_ms': grace_ms,
                        'poke_odor_start': odor_start,
                        'poke_odor_end': grace_end,
                        'poke_first_in': odor_start,
                        'required_min_sampling_time_ms': resolve_min_sampling_time_ms(loc['odor_name'])
                    }
                continue

            # Edges relative to start state
            rises = w & ~w.shift(1, fill_value=state_at_start)
            falls = ~w & w.shift(1, fill_value=state_at_start)

            # Build IN intervals within [odor_start, odor_end]
            intervals = []
            current_start = odor_start if state_at_start else None
            for ts in w.index:
                if rises.get(ts, False) and current_start is None:
                    current_start = ts
                if falls.get(ts, False) and current_start is not None:
                    intervals.append((current_start, ts))
                    current_start = None
            if current_start is not None:
                intervals.append((current_start, odor_end))  # clip at odor_end

            if not intervals:
                last_poke_end = _last_poke_end_before(s_bool, odor_start)
                grace_ms, grace_end = _grace_overlap_ms(last_poke_end, odor_start, odor_end)
                if grace_ms > 0.0:
                    position_poke_times[position] = {
                        'position': position,
                        'odor_name': loc['odor_name'],
                        'poke_time_ms': grace_ms,
                        'poke_odor_start': odor_start,
                        'poke_odor_end': grace_end,
                        'poke_first_in': odor_start,
                        'required_min_sampling_time_ms': resolve_min_sampling_time_ms(loc['odor_name'])
                    }
                continue

            # Merge across gaps <= sample_offset_time_ms
            merged = [intervals[0]]
            for start, end in intervals[1:]:
                ls, le = merged[-1]
                gap_ms = (start - le).total_seconds() * 1000.0
                if gap_ms <= sample_offset_time_ms:
                    merged[-1] = (ls, max(le, end))
                else:
                    merged.append((start, end))

            first_block_start, first_block_end = merged[0]
            first_block_ms = (first_block_end - first_block_start).total_seconds() * 1000.0
            consolidated_poke_time_ms = first_block_ms
            first_poke_in = first_block_start if merged else None

            if consolidated_poke_time_ms > 0:
                position_poke_times[position] = {
                    'position': position,
                    'odor_name': loc['odor_name'],
                    'poke_time_ms': consolidated_poke_time_ms,
                    # Use actual poke entry/exit times rather than valve window edges
                    'poke_odor_start': first_block_start,
                    'poke_odor_end': first_block_end,
                    'poke_first_in': first_poke_in,
                    'required_min_sampling_time_ms': resolve_min_sampling_time_ms(loc['odor_name'])
                }

        return position_valve_times, position_poke_times

    # Process trials
    for _, trial in initiated_trials.iterrows():
        trial_start = trial['sequence_start']
        trial_end = trial['sequence_end']
        odor_sequence, valve_activations = get_trial_valve_sequence(trial_start, trial_end)

        position_valve_times, position_poke_times = analyze_trial_valve_and_poke_times(valve_activations)

        valid_positions = [
            pos for pos in sorted(position_valve_times.keys())
            if position_poke_times.get(pos) and position_poke_times[pos].get('poke_time_ms', 0.0) > 0.0
        ]

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


        pos1_info = position_valve_times.get(1, {}) or {}
        last_pos1_start = pos1_info.get('valve_start')

        # Record earlier Position-1 presentations as non-initiated attempts with correct poke timing
        for attempt in pos1_info.get('prior_presentations', []) or []:
            a_start = attempt.get('valve_start')   # attempt valve start (for reference)
            # Cap at the last Pos1 valve START (trial starts at last odor 1 opening)
            last_pos1_valve_end = pos1_info.get('valve_end')  # The valve CLOSE time
            first_in, bout_end, dur_ms = _attempt_bout_from_poke_in(anchor_ts=a_start, cap_end=last_pos1_valve_end)
            non_initiated_odor1_attempts.append({
                'trial_id': trial['trial_id'] if 'trial_id' in trial else None,
                'attempt_start': a_start,
                'attempt_end': attempt.get('valve_end'),
                'odor_name': attempt.get('odor_name'),
                'attempt_first_poke_in': first_in,
                'attempt_poke_time_ms': dur_ms,
                'required_min_sampling_time_ms': attempt.get('required_min_sampling_time_ms', resolve_min_sampling_time_ms(attempt.get('odor_name'))),
            })

        # Compute corrected trial start = first poke-in within last Pos1 window (existing local window logic)
        corrected_start = None
        pos1_poke = position_poke_times.get(1)
        if pos1_poke:
            corrected_start = pos1_poke.get('poke_first_in') or pos1_poke.get('poke_odor_start')

        
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
        # protocol's output columns are untouched. Rewarded-type sequences keep the existing
        # rewarded/unrewarded/timeout handling; non-rewarded-type completions get false_response.
        sequence_rewarded = None
        if is_single_reward:
            sequence_rewarded = tuple(final_odor_sequence) in rewarded_sequences
            trial_dict['sequence_rewarded'] = sequence_rewarded
            # Reward determinacy from the presented (possibly partial) odor_sequence: is the
            # outcome already decided by what the animal has seen? Works for completed AND
            # aborted trials. determinacy_position = earliest position the outcome became
            # determined while reading the odors left to right.
            reward_determinacy, determinacy_position, determined_final_odor = _classify_reward_determinacy(
                final_odor_sequence, all_sequences, rewarded_sequences
            )
            trial_dict['reward_determinacy'] = reward_determinacy
            trial_dict['determinacy_position'] = determinacy_position
            # Guaranteed final/reward odor when already pinned by the prefix (else blank); lets
            # an early leave be scored against the port the sequence was bound to end at.
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
        if corrected_start is not None:
            trial_dict['sequence_start_corrected'] = corrected_start

        effective_odor_sequence = final_odor_sequence
        last_position = len(effective_odor_sequence)

        enough_odors, hit_hidden_rule, hr_hit_indices = check_hidden_rule(
            effective_odor_sequence, hidden_rule_indices, hr_odor_set
        )
        hr_hit_positions = [idx + 1 for idx in hr_hit_indices]
        trial_dict['enough_odors_for_hr'] = enough_odors
        trial_dict['hit_hidden_rule'] = hit_hidden_rule
        trial_dict['hidden_rule_hit_indices'] = hr_hit_indices
        trial_dict['hidden_rule_hit_positions'] = hr_hit_positions
        hr_success = False
        hr_success_position = None
        if hr_hit_positions:
            first_hr_pos = min(hr_hit_positions)
            left_before_full_sequence = last_position < max_positions
            has_await_reward = bool(trial_await_rewards)

            # Only count early-leave HR success if AwaitReward actually occurred in the trial.
            if left_before_full_sequence:
                if has_await_reward:
                    hr_success = True
                    hr_success_position = first_hr_pos
            else:
                # Final-position HR odor still counts as success regardless of AwaitReward presence.
                hr_success = last_position in hr_hit_positions
                hr_success_position = first_hr_pos if hr_success else None
        trial_dict['hidden_rule_success'] = hr_success
        trial_dict['hidden_rule_success_position'] = hr_success_position

        # Special-case logic for odourdiscrimination protocols: classify using AwaitReward and supply events only
        if is_odour_discrimination:
            last_valve_event = valve_activations[-1] if valve_activations else None
            last_valve_start = (last_valve_event or {}).get('start_time')
            trial_dict['odourdiscrimination_mode'] = True
            trial_dict['last_valve_start'] = last_valve_start

            current_init_ts = pd.to_datetime(trial.get('initiation_sequence_time'), errors='coerce') if trial.get('initiation_sequence_time') is not None else pd.NaT
            fallback_start = trial_start if trial_start is not None else last_valve_start
            await_window_start = current_init_ts if not pd.isna(current_init_ts) else fallback_start

            if await_window_start is None or pd.isna(await_window_start):
                aborted_sequences.append(trial_dict.copy())
                initiated_trials_list.append(trial_dict)
                continue

            # Pre-compute next anchors
            next_init = None
            if not initiation_starts_sorted.empty and not pd.isna(current_init_ts):
                idx = initiation_starts_sorted.searchsorted(current_init_ts, side='right')
                if idx < len(initiation_starts_sorted):
                    next_init = initiation_starts_sorted.iloc[idx]

            next_cue_poke = None
            if not cue_poke_starts_sorted.empty:
                future_cues = cue_poke_starts_sorted[cue_poke_starts_sorted > await_window_start]
                if not future_cues.empty:
                    next_cue_poke = future_cues.iloc[0]

            recording_end_candidates = [
                initiation_starts_sorted.iloc[-1] if not initiation_starts_sorted.empty else None,
                cue_poke_starts_sorted.iloc[-1] if not cue_poke_starts_sorted.empty else None,
                supply_port1_times[-1] if supply_port1_times else None,
                supply_port2_times[-1] if supply_port2_times else None,
                port1_pokes.index.max() if not port1_pokes.empty else None,
                port2_pokes.index.max() if not port2_pokes.empty else None,
                trial_end
            ]
            recording_end_candidates = [c for c in recording_end_candidates if c is not None and not pd.isna(c)]
            recording_end = max(recording_end_candidates) if recording_end_candidates else trial_end

            await_upper_bound = next_init if next_init is not None else recording_end
            await_in_window = [t for t in await_reward_times if await_window_start <= t <= await_upper_bound]
            if not await_in_window:
                trial_dict['abort_reason'] = 'no_await_reward'
                aborted_sequences.append(trial_dict.copy())
                initiated_trials_list.append(trial_dict)
                continue

            await_time = min(await_in_window)
            trial_dict['await_reward_time'] = await_time

            # Reward window: start at await_time and end at the later of
            #   (a) the next initiation time
            #   (b) the first cue-poke strictly AFTER that next initiation time
            # If no next initiation exists, fall back to the first cue after await_time, else to recording_end.
            next_cue_after_next_init = None
            if next_init is not None and not cue_poke_starts_sorted.empty:
                cues_after_next_init = cue_poke_starts_sorted[cue_poke_starts_sorted > next_init]
                if not cues_after_next_init.empty:
                    next_cue_after_next_init = cues_after_next_init.iloc[0]

            if next_init is not None:
                reward_window_end_candidates = [next_init, next_cue_after_next_init]
                reward_window_end_candidates = [c for c in reward_window_end_candidates if c is not None]
                reward_window_end = max(reward_window_end_candidates) if reward_window_end_candidates else next_init
                next_cue_poke = next_cue_after_next_init
            else:
                next_cue_poke = None
                if not cue_poke_starts_sorted.empty:
                    future_cues = cue_poke_starts_sorted[cue_poke_starts_sorted > await_time]
                    if not future_cues.empty:
                        next_cue_poke = future_cues.iloc[0]
                reward_window_end = next_cue_poke if next_cue_poke is not None else recording_end

            if reward_window_end < await_time:
                reward_window_end = await_time

            trial_dict['next_initiation_time'] = next_init
            trial_dict['next_cue_poke_start'] = next_cue_poke
            trial_dict['reward_window_end'] = reward_window_end

            supply1_after = [t for t in supply_port1_times if await_time <= t <= reward_window_end]
            supply2_after = [t for t in supply_port2_times if await_time <= t <= reward_window_end]
            all_supply_after = []
            if supply1_after:
                all_supply_after.extend([(t, 1, 'A') for t in supply1_after])
            if supply2_after:
                all_supply_after.extend([(t, 2, 'B') for t in supply2_after])
            all_supply_after.sort(key=lambda x: x[0])

            def _rises(series):
                return series & ~series.shift(1, fill_value=False)

            port1_pokes_in_window = []
            port2_pokes_in_window = []
            if not port1_pokes.empty:
                p1 = _rises(port1_pokes[await_time:reward_window_end])
                port1_pokes_in_window = p1[p1 == True].index.tolist()
            if not port2_pokes.empty:
                p2 = _rises(port2_pokes[await_time:reward_window_end])
                port2_pokes_in_window = p2[p2 == True].index.tolist()

            if verbose:
                def _head(seq, n=5):
                    return seq[:n] if isinstance(seq, list) else seq
                print(
                    "[odourdisc] window",
                    f"init={await_window_start}",
                    f"await={await_time}",
                    f"next_init={next_init}",
                    f"next_cue={next_cue_poke}",
                    f"reward_end={reward_window_end}",
                    f"supply_counts=({len(supply1_after)},{len(supply2_after)})",
                )
                if supply_port1_times or supply_port2_times:
                    print(
                        "[odourdisc] raw supply tails",
                        f"s1_total={len(supply_port1_times)} last={supply_port1_times[-1] if supply_port1_times else None}",
                        f"s2_total={len(supply_port2_times)} last={supply_port2_times[-1] if supply_port2_times else None}",
                    )
                if supply1_after or supply2_after:
                    print("[odourdisc] supply in window", _head(supply1_after + supply2_after, 5))

            if all_supply_after:
                first_supply_time, first_supply_port, first_supply_odor = all_supply_after[0]
                trial_dict['first_supply_time'] = first_supply_time
                trial_dict['first_supply_port'] = first_supply_port
                trial_dict['first_supply_odor_identity'] = first_supply_odor
                trial_dict['supply1_count'] = len(supply1_after)
                trial_dict['supply2_count'] = len(supply2_after)
                trial_dict['total_supply_count'] = len(all_supply_after)
                completed_rewarded.append(trial_dict.copy())
            elif port1_pokes_in_window or port2_pokes_in_window:
                all_reward_pokes = []
                if port1_pokes_in_window:
                    all_reward_pokes.extend([(t, 1, 'A') for t in port1_pokes_in_window])
                if port2_pokes_in_window:
                    all_reward_pokes.extend([(t, 2, 'B') for t in port2_pokes_in_window])
                all_reward_pokes.sort(key=lambda x: x[0])
                if all_reward_pokes:
                    trial_dict['first_reward_poke_time'], trial_dict['first_reward_poke_port'], trial_dict['first_reward_poke_odor_identity'] = all_reward_pokes[0]
                trial_dict['port1_pokes_count'] = len(port1_pokes_in_window)
                trial_dict['port2_pokes_count'] = len(port2_pokes_in_window)
                trial_dict['total_reward_pokes'] = len(all_reward_pokes)
                completed_unrewarded.append(trial_dict.copy())
            else:
                completed_timeout.append(trial_dict.copy())

            completed_sequences.append(trial_dict.copy())
            initiated_trials_list.append(trial_dict)
            # skip standard classification path
            continue

        initiated_trials_list.append(trial_dict)
        if trial_await_rewards:
            # Aggregate ranges for completed trials
            # Valve times
            for pos, v in (position_valve_times or {}).items():
                if v and 'valve_duration_ms' in v:
                    agg_position_valve_times[pos].append(v['valve_duration_ms'])
                    odor_name = v.get('odor_name')
                    if odor_name:
                        agg_odor_valve_times[odor_name].append(v['valve_duration_ms'])
            # Poke times
            for pos, p in (position_poke_times or {}).items():
                if p and 'poke_time_ms' in p:
                    agg_position_poke_times[pos].append(p['poke_time_ms'])
                    odor_name = p.get('odor_name')
                    if odor_name:
                        agg_odor_poke_times[odor_name].append(p['poke_time_ms'])

            await_reward_time = min(trial_await_rewards)
            trial_dict['await_reward_time'] = await_reward_time

            if hit_hidden_rule:
                if hr_success:
                    completed_hr.append(trial_dict.copy())
                    hr_category = 'completed_hr'
                else:
                    completed_hr_missed.append(trial_dict.copy())
                    hr_category = 'completed_hr_missed'
            else:
                hr_category = 'completed_normal'

            if is_single_reward and sequence_rewarded is False:
                # NON-REWARDED ("no-go") sequence completed to the end. There is no reward to
                # collect, so going to a reward port anyway is a FALSE RESPONSE. This mirrors the
                # abort-trial false-alarm logic, but for completed sequences: anchor at
                # await_reward_time (the completion moment) and bound the search by the next
                # trial's engagement, exactly like the false-alarm window.
                next_init_fr = None
                if not initiation_starts_sorted.empty:
                    _ni = initiation_starts_sorted.searchsorted(trial_end, side='right')
                    if _ni < len(initiation_starts_sorted):
                        next_init_fr = initiation_starts_sorted.iloc[_ni]
                fr_window_end = None
                if next_init_fr is not None and not cue_poke_starts_sorted.empty:
                    _cues_after = cue_poke_starts_sorted[cue_poke_starts_sorted > next_init_fr]
                    if not _cues_after.empty:
                        fr_window_end = _cues_after.iloc[0]
                if fr_window_end is None:
                    _end_cands = [trial_end]
                    if not port1_pokes.empty:
                        _end_cands.append(port1_pokes.index.max())
                    if not port2_pokes.empty:
                        _end_cands.append(port2_pokes.index.max())
                    _end_cands = [c for c in _end_cands if c is not None and not pd.isna(c)]
                    fr_window_end = max(_end_cands) if _end_cands else trial_end
                if fr_window_end < await_reward_time:
                    fr_window_end = await_reward_time

                fr_port1_pokes = []
                fr_port2_pokes = []
                if not port1_pokes.empty:
                    _w1 = port1_pokes[await_reward_time:fr_window_end]
                    _s1 = _w1 & ~_w1.shift(1, fill_value=False)
                    fr_port1_pokes = _s1[_s1 == True].index.tolist()
                if not port2_pokes.empty:
                    _w2 = port2_pokes[await_reward_time:fr_window_end]
                    _s2 = _w2 & ~_w2.shift(1, fill_value=False)
                    fr_port2_pokes = _s2[_s2 == True].index.tolist()

                fr_pokes = [(t, 1, 'A') for t in fr_port1_pokes] + [(t, 2, 'B') for t in fr_port2_pokes]
                fr_pokes.sort(key=lambda x: x[0])

                trial_dict['fr_window_end'] = fr_window_end
                trial_dict['port1_pokes_count'] = len(fr_port1_pokes)
                trial_dict['port2_pokes_count'] = len(fr_port2_pokes)
                trial_dict['total_reward_pokes'] = len(fr_pokes)

                if fr_pokes:
                    fr_time, fr_port, fr_odor = fr_pokes[0]
                    fr_latency_ms = (fr_time - await_reward_time).total_seconds() * 1000.0
                    trial_dict['false_response'] = True
                    trial_dict['fr_time'] = fr_time
                    trial_dict['fr_port'] = fr_port
                    trial_dict['fr_odor_identity'] = fr_odor
                    trial_dict['fr_latency_ms'] = float(fr_latency_ms)
                    # Parity with unrewarded rows so downstream poke-based logic stays consistent.
                    trial_dict['first_reward_poke_time'] = fr_time
                    trial_dict['first_reward_poke_port'] = fr_port
                    trial_dict['first_reward_poke_odor_identity'] = fr_odor
                    if response_time_ms_window is not None and fr_latency_ms <= response_time_ms_window:
                        trial_dict['fr_label'] = 'FR_time_in'
                    elif response_time_ms_window is not None and fr_latency_ms <= 3.0 * response_time_ms_window:
                        trial_dict['fr_label'] = 'FR_time_out'
                    else:
                        trial_dict['fr_label'] = 'FR_late'
                else:
                    trial_dict['false_response'] = False
                    trial_dict['fr_label'] = 'nFR'
                    trial_dict['fr_time'] = pd.NaT
                    trial_dict['fr_port'] = None
                    trial_dict['fr_odor_identity'] = None
                    trial_dict['fr_latency_ms'] = np.nan

                completed_false_response.append(trial_dict.copy())
            else:
                supply1_after_await = [t for t in supply_port1_times if await_reward_time <= t <= trial_end]
                supply2_after_await = [t for t in supply_port2_times if await_reward_time <= t <= trial_end]

                if supply1_after_await or supply2_after_await:
                    all_supply_times = []
                    if supply1_after_await:
                        all_supply_times.extend([(t, 1, 'A') for t in supply1_after_await])
                    if supply2_after_await:
                        all_supply_times.extend([(t, 2, 'B') for t in supply2_after_await])
                    all_supply_times.sort(key=lambda x: x[0])
                    first_supply_time, first_supply_port, first_supply_odor = all_supply_times[0]

                    trial_dict['first_supply_time'] = first_supply_time
                    trial_dict['first_supply_port'] = first_supply_port
                    trial_dict['first_supply_odor_identity'] = first_supply_odor
                    trial_dict['supply1_count'] = len(supply1_after_await)
                    trial_dict['supply2_count'] = len(supply2_after_await)
                    trial_dict['total_supply_count'] = len(supply1_after_await) + len(supply2_after_await)

                    completed_rewarded.append(trial_dict.copy())
                    if hr_category == 'completed_hr':
                        completed_hr_rewarded.append(trial_dict.copy())
                    elif hr_category == 'completed_hr_missed':
                        completed_hr_missed_rewarded.append(trial_dict.copy())
                else:
                    poke_window_end = await_reward_time + pd.Timedelta(seconds=response_time_sec)
                    port1_pokes_in_window = []
                    port2_pokes_in_window = []

                    if not port1_pokes.empty:
                        port1_window = port1_pokes[await_reward_time:poke_window_end]
                        port1_starts = port1_window & ~port1_window.shift(1, fill_value=False)
                        port1_pokes_in_window = port1_starts[port1_starts == True].index.tolist()
                    if not port2_pokes.empty:
                        port2_window = port2_pokes[await_reward_time:poke_window_end]
                        port2_starts = port2_window & ~port2_window.shift(1, fill_value=False)
                        port2_pokes_in_window = port2_starts[port2_starts == True].index.tolist()

                    all_reward_pokes = []
                    if port1_pokes_in_window:
                        all_reward_pokes.extend([(t, 1, 'A') for t in port1_pokes_in_window])
                    if port2_pokes_in_window:
                        all_reward_pokes.extend([(t, 2, 'B') for t in port2_pokes_in_window])
                    all_reward_pokes.sort(key=lambda x: x[0])

                    trial_dict['poke_window_end'] = poke_window_end
                    trial_dict['port1_pokes_count'] = len(port1_pokes_in_window)
                    trial_dict['port2_pokes_count'] = len(port2_pokes_in_window)
                    trial_dict['total_reward_pokes'] = len(all_reward_pokes)

                    if all_reward_pokes:
                        first_poke_time, first_poke_port, first_poke_odor = all_reward_pokes[0]
                        trial_dict['first_reward_poke_time'] = first_poke_time
                        trial_dict['first_reward_poke_port'] = first_poke_port
                        trial_dict['first_reward_poke_odor_identity'] = first_poke_odor
                        completed_unrewarded.append(trial_dict.copy())
                        if hr_category == 'completed_hr':
                            completed_hr_unrewarded.append(trial_dict.copy())
                        elif hr_category == 'completed_hr_missed':
                            completed_hr_missed_unrewarded.append(trial_dict.copy())
                    else:
                        completed_timeout.append(trial_dict.copy())
                        if hr_category == 'completed_hr':
                            completed_hr_timeout.append(trial_dict.copy())
                        elif hr_category == 'completed_hr_missed':
                            completed_hr_missed_timeout.append(trial_dict.copy())
            completed_sequences.append(trial_dict.copy())

        else:
            aborted_sequences.append(trial_dict.copy())
            if hit_hidden_rule:
                aborted_sequences_hr.append(trial_dict.copy())


    if isinstance(non_initiated_trials, pd.DataFrame) and not non_initiated_trials.empty:
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
                    valve_key = f"{olf_id}{i}"
                    odor_name = odor_map['valve_to_odor'].get(valve_key)
                    if not odor_name or odor_name.lower() == 'purge':
                        continue
                    s = valve_data[valve_col]
                    rises = s & ~s.shift(1, fill_value=False)
                    starts = list(s.index[rises])
                    falls = ~s & s.shift(1, fill_value=False)
                    ends = list(s.index[falls])
                    for st, en in zip(starts, ends):
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
            if found_odor is None:
                found_odor = closest_odor  # fallback to closest
            odor_names.append(found_odor)
        non_initiated_trials = non_initiated_trials.copy()
        non_initiated_trials['odor_name'] = odor_names
    initiated_trials = pd.DataFrame(initiated_trials_list)
    # Build result with both singular and plural aliases for HR categories
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
    result['completed_sequences_HR_rewarded'] = result['completed_sequence_HR_rewarded']
    result['completed_sequences_HR_unrewarded'] = result['completed_sequence_HR_unrewarded']
    result['completed_sequences_HR_reward_timeout'] = result['completed_sequence_HR_reward_timeout']
    result['completed_sequences_HR_missed_rewarded'] = result['completed_sequence_HR_missed_rewarded']
    result['completed_sequences_HR_missed_unrewarded'] = result['completed_sequence_HR_missed_unrewarded']
    result['completed_sequences_HR_missed_reward_timeout'] = result['completed_sequence_HR_missed_reward_timeout']

    result['hidden_rule_location'] = hidden_rule_location
    result['hidden_rule_positions'] = list(hidden_rule_positions)
    result['hidden_rule_locations'] = list(hidden_rule_indices)
    result['hidden_rule_position'] = hidden_rule_position
    result['hidden_rule_odors'] = sorted(list(hr_odor_set)) if hr_odor_set is not None else []

    if verbose:
        print(f"\nTRIAL CLASSIFICATION RESULTS WITH HIDDEN RULE AND VALVE/POKE TIME ANALYSIS:")
        if hidden_rule_positions:
            if multiple_hidden_rule_locations:
                pos_str = ", ".join(str(pos) for pos in hidden_rule_positions)
                idx_str = ", ".join(str(idx) for idx in hidden_rule_indices)
                print(f"Hidden Rule Locations: Positions {pos_str} (indices {idx_str})\n")
            else:
                print(f"Hidden Rule Location: Position {hidden_rule_position} (index {hidden_rule_location})\n")
        else:
            print("Hidden Rule Location: None detected\n")
        print(f"Hidden Rule Odors: {', '.join(result['hidden_rule_odors']) if result['hidden_rule_odors'] else 'None'}\n")

        # safe percent helper
        def _pct(n, d):
            try:
                d = float(d)
                return 0.0 if d == 0 else (float(n) / d * 100.0)
            except Exception:
                return 0.0

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

        # Additional requested summaries
        print("POKE TIME RANGES BY POSITION:")
        print("-" * 40)
        for pos in range(1, max_positions + 1):
            times = agg_position_poke_times[pos]
            if times:
                min_v = min(times); max_v = max(times); avg_v = sum(times) / len(times)
                print(f"Position {pos}: {min_v:.1f} - {max_v:.1f}ms (avg: {avg_v:.1f}ms, n={len(times)})")
            else:
                print(f"Position {pos}: No data")

        print("\nVALVE TIME RANGES BY POSITION:")
        print("-" * 40)
        for pos in range(1, max_positions + 1):
            times = agg_position_valve_times[pos]
            if times:
                min_v = min(times); max_v = max(times); avg_v = sum(times) / len(times)
                print(f"Position {pos}: {min_v:.1f} - {max_v:.1f}ms (avg: {avg_v:.1f}ms, n={len(times)})")
            else:
                print(f"Position {pos}: No data")

        print("\nPOKE TIME RANGES BY ODOR (ALL POSITIONS):")
        print("-" * 50)
        for odor_name in sorted(agg_odor_poke_times.keys()):
            times = agg_odor_poke_times[odor_name]
            if times:
                min_v = min(times); max_v = max(times); avg_v = sum(times) / len(times)
                print(f"{odor_name}: {min_v:.1f} - {max_v:.1f}ms (avg: {avg_v:.1f}ms, n={len(times)})")

        print("\nVALVE TIME RANGES BY ODOR (ALL POSITIONS):")
        print("-" * 50)
        for odor_name in sorted(agg_odor_valve_times.keys()):
            times = agg_odor_valve_times[odor_name]
            if times:
                min_v = min(times); max_v = max(times); avg_v = sum(times) / len(times)
                print(f"{odor_name}: {min_v:.1f} - {max_v:.1f}ms (avg: {avg_v:.1f}ms, n={len(times)})")

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

        # Verify totals
        total_classified = (len(result['completed_sequence_rewarded'])
                            + len(result['completed_sequence_unrewarded'])
                            + len(result['completed_sequence_reward_timeout'])
                            + len(result['aborted_sequences']))
        if total_classified == len(initiated_trials):
            print(f"\nClassification complete: all {len(initiated_trials)} trials classified")
        else:
            print(f"\nClassification mismatch: {total_classified} classified vs {len(initiated_trials)} total")

    return result

def analyze_response_times(data, trial_counts, events, odor_map, stage, root, verbose=True, single_reward_info=None):
    """
    Analyze response times for all completed trials (clean version).
    Behavior and prints match analyze_response_times_all_trials_fixed.

    Returns a dict with:
      - rewarded_response_times
      - unrewarded_response_times
      - timeout_delayed_response_times
      - timeout_response_delay_times
      - all_response_times
      - failed_calculations
    """

    # Get per-odor thresholds from schema
    sample_offset_time, minimum_sampling_time_by_odor, response_time = get_experiment_parameters(root)
    sample_offset_time_ms = sample_offset_time * 1000
    minimum_sampling_time_ms_by_odor = {
        str(odor): float(threshold) * 1000.0
        for odor, threshold in (minimum_sampling_time_by_odor or {}).items()
        if threshold is not None
    }
    if not minimum_sampling_time_ms_by_odor:
        raise ValueError("minimumSamplingTime_by_odor missing or empty in schema; cannot analyze response times without per-odor thresholds")
    default_minimum_sampling_time_ms = max(minimum_sampling_time_ms_by_odor.values())

    def resolve_min_sampling_time_ms(odor_name):
        """Return minimum sampling time in ms for a given odor, fallback to default if unknown"""
        if odor_name is None:
            return default_minimum_sampling_time_ms
        return minimum_sampling_time_ms_by_odor.get(str(odor_name), default_minimum_sampling_time_ms)

    response_time_sec = response_time
    if response_time_sec is None:
        raise ValueError("Response time parameter cannot be extracted from Schema file. Check detect_settings function.")

    # Single-reward (new) protocol: keep response_time_category meaningful for rewarded-type
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
    # Final position is always rewarded -> never a hidden-rule position (single-reward untouched).
    hidden_rule_indices = _drop_final_hidden_rule_index(hidden_rule_indices, schema_settings, is_single_reward)
    hidden_rule_positions = [idx + 1 for idx in hidden_rule_indices]
    hidden_rule_location = hidden_rule_indices[0] if hidden_rule_indices else None
    hidden_rule_position = hidden_rule_positions[0] if hidden_rule_positions else None
    multiple_hidden_rule_locations = len(hidden_rule_positions) > 1

    if verbose:
        print(f"Sample offset time: {sample_offset_time_ms} ms")
        print("Minimum sampling times (ms) by odor:")
        for odor_name, threshold in sorted(minimum_sampling_time_ms_by_odor.items()):
            print(f"  - {odor_name}: {threshold:.1f}")
        print(f"Response time window: {response_time_sec} s")
        seq_label = sequence_name or str(stage)
        if hidden_rule_positions:
            if multiple_hidden_rule_locations:
                pos_str = ", ".join(str(pos) for pos in hidden_rule_positions)
                idx_str = ", ".join(str(idx) for idx in hidden_rule_indices)
                print(f"Hidden rule locations extracted: Positions {pos_str} (indices {idx_str})")
            else:
                print(f"Hidden rule location extracted: Location {hidden_rule_location} (index {hidden_rule_location}, position {hidden_rule_position})")
        else:
            print(f"No Hidden Rule Location found in sequence name: {seq_label}. Proceeding without Hidden Rule analysis.")

    # Get initiated trials and events (same as main function)
    initiated_trials = trial_counts['initiated_sequences']
    await_reward_times = events['combined_await_reward_df']['Time'].tolist() if 'combined_await_reward_df' in events else []

    # Odour-discrimination flag (use same detection as classify_trials)
    protocol_name = (sequence_name or str(stage) or "").lower()
    is_odour_discrimination = "odourdiscrimination" in protocol_name

    # Canonical initiation starts and cue-poke starts (Port0 rising edges), used for odourdiscrimination reward windows
    init_series_raw = initiated_trials.get('initiation_sequence_time')
    initiation_starts_sorted = pd.to_datetime(init_series_raw, errors='coerce').dropna().sort_values().reset_index(drop=True)

    poke_series_full = data['digital_input_data'].get('DIPort0', pd.Series(dtype=bool)).astype(bool)
    poke_series_full = poke_series_full.sort_index()
    cue_poke_starts_sorted = pd.Series(dtype='datetime64[ns]')
    if not poke_series_full.empty:
        rises = poke_series_full & ~poke_series_full.shift(1, fill_value=False)
        starts = list(rises[rises == True].index)
        cue_poke_starts_sorted = pd.Series(starts, dtype='datetime64[ns]').sort_values().reset_index(drop=True)

    # Get supply port activities for reward classification
    supply_port1_times = data['pulse_supply_1'].index.tolist() if not data['pulse_supply_1'].empty else []
    supply_port2_times = data['pulse_supply_2'].index.tolist() if not data['pulse_supply_2'].empty else []

    # Filter for completed trials
    completed_trials_all = []
    for _, trial in initiated_trials.iterrows():
        trial_start = trial['sequence_start']
        trial_end = trial['sequence_end']

        trial_await_rewards = [t for t in await_reward_times if trial_start <= t <= trial_end]
        if trial_await_rewards:
            trial_dict = trial.to_dict()
            trial_dict['await_reward_time'] = min(trial_await_rewards)
            completed_trials_all.append(trial_dict)

    if verbose:
        print(f"Total completed trials: {len(completed_trials_all)}\n")

    # Get poke and port data
    poke_data = data['digital_input_data']['DIPort0'].copy() if 'DIPort0' in data['digital_input_data'] else pd.Series(dtype=bool)
    port1_pokes = data['digital_input_data']['DIPort1'] if 'DIPort1' in data['digital_input_data'] else pd.Series(dtype=bool)
    port2_pokes = data['digital_input_data']['DIPort2'] if 'DIPort2' in data['digital_input_data'] else pd.Series(dtype=bool)

    # Build valve activation list
    olfactometer_valves = odor_map['olfactometer_valves']
    valve_to_odor = odor_map['valve_to_odor']

    all_valve_activations = []
    for olf_id, valve_data in olfactometer_valves.items():
        if valve_data.empty:
            continue
        for i, valve_col in enumerate(valve_data.columns):
            valve_key = f"{olf_id}{i}"
            if valve_key in valve_to_odor:
                odor_name = valve_to_odor[valve_key]
                if odor_name.lower() == 'purge':
                    continue

                valve_series = valve_data[valve_col]
                valve_activations = valve_series & ~valve_series.shift(1, fill_value=False)
                activation_times = valve_activations[valve_activations == True].index.tolist()
                valve_deactivations = ~valve_series & valve_series.shift(1, fill_value=False)
                deactivation_times = valve_deactivations[valve_deactivations == True].index.tolist()

                for activation_time in activation_times:
                    next_deactivations = [t for t in deactivation_times if t > activation_time]
                    deactivation_time = min(next_deactivations) if next_deactivations else valve_series.index[-1]

                    all_valve_activations.append({
                        'start_time': activation_time,
                        'end_time': deactivation_time,
                        'odor_name': odor_name,
                        'valve_key': valve_key
                    })

    all_valve_activations.sort(key=lambda x: x['start_time'])

    # Helpers
    def get_trial_valve_sequence(trial_start, trial_end):
        trial_valve_activations = []
        for valve_activation in all_valve_activations:
            valve_start = valve_activation['start_time']
            valve_end = valve_activation['end_time']
            if valve_start <= trial_end and valve_end >= trial_start:
                trial_valve_activations.append(valve_activation)
        trial_valve_activations.sort(key=lambda x: x['start_time'])
        odor_sequence = [activation['odor_name'] for activation in trial_valve_activations]
        return odor_sequence, trial_valve_activations

    hr_odor_set = None
    if hidden_rule_indices:
        try:
            if schema_err is not None and 'hiddenRuleOdorsInferred' not in schema_settings:
                raise ValueError(str(schema_err))
            odors = (schema_settings.get('hiddenRuleOdorsInferred') or [])
            if len(odors) < 2:
                raise ValueError("found fewer than two rewarded odors at inferred hidden rule position.")
            hr_odor_set = set(map(str, odors))
            if verbose:
                print(f"Hidden Rule Odors inferred: {sorted(hr_odor_set)}")
        except Exception as e:
            raise ValueError(f"Hidden Rule Odor Identities could not be inferred from Schema: {e}")


    def check_hidden_rule(odor_sequence, candidate_indices, odor_set):
        if not candidate_indices or odor_set is None:
            return False, False, []

        valid_indices = [idx for idx in candidate_indices if 0 <= idx < len(odor_sequence)]
        if not valid_indices:
            return False, False, []

        matching_indices = sorted({idx for idx in valid_indices if odor_sequence[idx] in odor_set})
        return True, bool(matching_indices), matching_indices

    def find_next_trial_start(current_trial_end, all_trials):
        next_starts = [t['sequence_start'] for t in all_trials if t['sequence_start'] > current_trial_end]
        return min(next_starts) if next_starts else None

    # Analyze all completed trials
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
        target_odor_name = None
        target_required_min_ms = float('nan')

        # Get valve sequence
        odor_sequence_raw, trial_valve_events = get_trial_valve_sequence(trial_start, trial_end)
        if not trial_valve_events:
            failed_calculations += 1
            per_trial_rows.append({
                'trial_id': trial_id,
                'response_time_ms': np.nan,
                'response_time_category': None,
            })
            continue

        position_locations_rt = {}
        odor_to_pos_rt = {}
        next_pos_rt = 1
        for event in trial_valve_events:
            odor = event['odor_name']
            if odor not in odor_to_pos_rt:
                odor_to_pos_rt[odor] = next_pos_rt
                next_pos_rt += 1
            pos_idx = odor_to_pos_rt[odor]
            position_locations_rt[pos_idx] = event

        ordered_positions_rt = sorted(position_locations_rt.keys())
        effective_odor_sequence = [
            position_locations_rt[pos]['odor_name']
            for pos in ordered_positions_rt
            if position_locations_rt.get(pos) is not None
        ]

        # Single-reward protocol: non-rewarded ("no-go") completions are scored as false_response
        # in classify_trials, not here. Leave their response_time_category empty so existing
        # rewarded/unrewarded/timeout-based metrics stay clean. No-op for the default protocol.
        if is_single_reward and tuple(effective_odor_sequence) not in rewarded_sequences:
            per_trial_rows.append({
                'trial_id': trial_id,
                'response_time_ms': np.nan,
                'response_time_category': None,
            })
            continue

        _, hit_hidden_rule, hr_hit_indices = check_hidden_rule(
            effective_odor_sequence, hidden_rule_indices, hr_odor_set
        )
        hr_hit_positions = [idx + 1 for idx in hr_hit_indices]
        hr_success = len(effective_odor_sequence) in hr_hit_positions if hr_hit_positions else False

        if not ordered_positions_rt:
            failed_calculations += 1
            per_trial_rows.append({
                'trial_id': trial_id,
                'response_time_ms': np.nan,
                'response_time_category': None,
            })
            continue

        target_pos_idx = ordered_positions_rt[-1]
        target_valve_event = position_locations_rt.get(target_pos_idx)

        if target_valve_event is None:
            failed_calculations += 1
            per_trial_rows.append({
                'trial_id': trial_id,
                'response_time_ms': np.nan,
                'response_time_category': None,
                'target_odor_name': target_odor_name,
                'target_required_min_sampling_time_ms': target_required_min_ms,
            })
            continue
        target_odor_name = target_valve_event.get('odor_name')
        target_required_min_ms = resolve_min_sampling_time_ms(target_odor_name)

        # Find last poke out in extended window around target odor
        odor_start = target_valve_event['start_time']
        odor_end = target_valve_event['end_time']
        search_end = max(await_reward_time, odor_end + pd.Timedelta(seconds=1))

        extended_poke_data = poke_data.loc[odor_start:search_end]
        if extended_poke_data.empty:
            failed_calculations += 1
            per_trial_rows.append({
                'trial_id': trial_id,
                'response_time_ms': np.nan,
                'response_time_category': None,
                'target_odor_name': target_odor_name,
                'target_required_min_sampling_time_ms': target_required_min_ms,
            })
            continue

        last_poke_out_time = None
        prev_state = poke_data.loc[:odor_start].iloc[-1] if len(poke_data.loc[:odor_start]) > 0 else False
        for timestamp, current_state in extended_poke_data.items():
            if prev_state and not current_state:
                last_poke_out_time = timestamp
            prev_state = current_state

        if last_poke_out_time is None:
            failed_calculations += 1
            per_trial_rows.append({
                'trial_id': trial_id,
                'response_time_ms': np.nan,
                'response_time_category': None,
                'target_odor_name': target_odor_name,
                'target_required_min_sampling_time_ms': target_required_min_ms,
            })
            continue

        # Reward window and search for reward pokes
        if is_odour_discrimination:
            current_init_ts = pd.to_datetime(trial_dict.get('initiation_sequence_time'), errors='coerce') if trial_dict.get('initiation_sequence_time') is not None else pd.NaT
            next_init = None
            if not initiation_starts_sorted.empty and not pd.isna(current_init_ts):
                idx = initiation_starts_sorted.searchsorted(current_init_ts, side='right')
                if idx < len(initiation_starts_sorted):
                    next_init = initiation_starts_sorted.iloc[idx]

            next_cue_after_next_init = None
            if next_init is not None and not cue_poke_starts_sorted.empty:
                cues_after_next_init = cue_poke_starts_sorted[cue_poke_starts_sorted > next_init]
                if not cues_after_next_init.empty:
                    next_cue_after_next_init = cues_after_next_init.iloc[0]

            recording_end_candidates = [
                initiation_starts_sorted.iloc[-1] if not initiation_starts_sorted.empty else None,
                cue_poke_starts_sorted.iloc[-1] if not cue_poke_starts_sorted.empty else None,
                supply_port1_times[-1] if supply_port1_times else None,
                supply_port2_times[-1] if supply_port2_times else None,
                port1_pokes.index.max() if not port1_pokes.empty else None,
                port2_pokes.index.max() if not port2_pokes.empty else None,
                trial_end
            ]
            recording_end_candidates = [c for c in recording_end_candidates if c is not None and not pd.isna(c)]
            recording_end = max(recording_end_candidates) if recording_end_candidates else trial_end

            if next_init is not None:
                rw_candidates = [next_init, next_cue_after_next_init]
                rw_candidates = [c for c in rw_candidates if c is not None]
                reward_window_end = max(rw_candidates) if rw_candidates else next_init
            else:
                next_cue_after_await = None
                if not cue_poke_starts_sorted.empty:
                    future_cues = cue_poke_starts_sorted[cue_poke_starts_sorted > await_reward_time]
                    if not future_cues.empty:
                        next_cue_after_await = future_cues.iloc[0]
                reward_window_end = next_cue_after_await if next_cue_after_await is not None else recording_end

            if reward_window_end < await_reward_time:
                reward_window_end = await_reward_time

            poke_window_end = reward_window_end
            search_start = max(last_poke_out_time, await_reward_time)
            reward_window_cap = reward_window_end
        else:
            poke_window_end = await_reward_time + pd.Timedelta(seconds=response_time_sec)
            search_start = max(last_poke_out_time, await_reward_time)
            reward_window_cap = trial_end

        port1_pokes_in_window = []
        port2_pokes_in_window = []

        if not port1_pokes.empty:
            port1_window = port1_pokes[search_start:poke_window_end]
            port1_starts = port1_window & ~port1_window.shift(1, fill_value=False)
            port1_pokes_in_window = port1_starts[port1_starts == True].index.tolist()

        if not port2_pokes.empty:
            port2_window = port2_pokes[search_start:poke_window_end]
            port2_starts = port2_window & ~port2_window.shift(1, fill_value=False)
            port2_pokes_in_window = port2_starts[port2_starts == True].index.tolist()

        all_reward_pokes = port1_pokes_in_window + port2_pokes_in_window

        response_time_ms = None
        if all_reward_pokes:
            first_reward_poke = min(all_reward_pokes)
            response_time_ms = (first_reward_poke - last_poke_out_time).total_seconds() * 1000

        # Determine reward status (same as main classification)
        supply1_after_await = [t for t in supply_port1_times if await_reward_time <= t <= reward_window_cap]
        supply2_after_await = [t for t in supply_port2_times if await_reward_time <= t <= reward_window_cap]

        # NEW: authoritative per-trial categorization + row capture
        is_rewarded = bool(supply1_after_await or supply2_after_await)
        if is_rewarded:
            if response_time_ms is not None:
                rewarded_response_times.append(response_time_ms)
                # optional: HR subset if you track it
                if hr_success:
                    hr_rewarded_response_times.append(response_time_ms)
                per_trial_rows.append({
                    'trial_id': trial_id,
                    'response_time_ms': float(response_time_ms),
                    'response_time_category': 'rewarded',
                    'target_odor_name': target_odor_name,
                    'target_required_min_sampling_time_ms': target_required_min_ms,
                })
            else:
                failed_calculations += 1
                per_trial_rows.append({
                    'trial_id': trial_id,
                    'response_time_ms': np.nan,
                    'response_time_category': None,
                    'target_odor_name': target_odor_name,
                    'target_required_min_sampling_time_ms': target_required_min_ms,
                })
        else:
            # Check full window from await_reward for unrewarded vs timeout
            port1_full_window = []
            port2_full_window = []
            if not port1_pokes.empty:
                port1_window_full = port1_pokes[await_reward_time:poke_window_end]
                port1_starts_full = port1_window_full & ~port1_window_full.shift(1, fill_value=False)
                port1_full_window = port1_starts_full[port1_starts_full == True].index.tolist()
            if not port2_pokes.empty:
                port2_window_full = port2_pokes[await_reward_time:poke_window_end]
                port2_starts_full = port2_window_full & ~port2_window_full.shift(1, fill_value=False)
                port2_full_window = port2_starts_full[port2_starts_full == True].index.tolist()

            is_unrewarded = bool(port1_full_window or port2_full_window)
            if is_unrewarded:
                if response_time_ms is not None:
                    unrewarded_response_times.append(response_time_ms)
                    per_trial_rows.append({
                        'trial_id': trial_id,
                        'response_time_ms': float(response_time_ms),
                        'response_time_category': 'unrewarded',
                        'target_odor_name': target_odor_name,
                        'target_required_min_sampling_time_ms': target_required_min_ms,
                    })
                else:
                    failed_calculations += 1
                    per_trial_rows.append({
                        'trial_id': trial_id,
                        'response_time_ms': np.nan,
                        'response_time_category': None,
                        'target_odor_name': target_odor_name,
                        'target_required_min_sampling_time_ms': target_required_min_ms,
                    })
            else:
                # Timeout: look for delayed responses until next completed trial start
                next_trial_start = find_next_trial_start(trial_end, completed_trials_all)
                extended_search_end = next_trial_start if next_trial_start else (poke_data.index[-1] if not poke_data.empty else poke_window_end)

                delayed_search_start = poke_window_end

                delayed_port1_pokes = []
                delayed_port2_pokes = []
                if not port1_pokes.empty and delayed_search_start < extended_search_end:
                    w = port1_pokes[delayed_search_start:extended_search_end]
                    s = w & ~w.shift(1, fill_value=False)
                    delayed_port1_pokes = s[s == True].index.tolist()
                if not port2_pokes.empty and delayed_search_start < extended_search_end:
                    w = port2_pokes[delayed_search_start:extended_search_end]
                    s = w & ~w.shift(1, fill_value=False)
                    delayed_port2_pokes = s[s == True].index.tolist()

                delayed_reward_pokes = delayed_port1_pokes + delayed_port2_pokes
                if delayed_reward_pokes:
                    first_delayed = min(delayed_reward_pokes)
                    response_time_ms = (first_delayed - last_poke_out_time).total_seconds() * 1000
                    timeout_delayed_response_times.append(response_time_ms)
                    # also store delay beyond window if desired
                    timeout_response_delay_times.append((first_delayed - poke_window_end).total_seconds() * 1000.0)
                    per_trial_rows.append({
                        'trial_id': trial_id,
                        'response_time_ms': float(response_time_ms),
                        'response_time_category': 'timeout_delayed',
                        'target_odor_name': target_odor_name,
                        'target_required_min_sampling_time_ms': target_required_min_ms,
                    })
                else:
                    failed_calculations += 1
                    per_trial_rows.append({
                        'trial_id': trial_id,
                        'response_time_ms': np.nan,
                        'response_time_category': None,
                        'target_odor_name': target_odor_name,
                        'target_required_min_sampling_time_ms': target_required_min_ms,
                    })

    # Print results
    if verbose:
        print(f"RESPONSE TIME ANALYSIS RESULTS:")
        print(f"Total completed trials analyzed: {len(completed_trials_all)}")
        print(f"Failed response time calculations: {failed_calculations}")
        print(f"Successful response time calculations: {len(rewarded_response_times) + len(unrewarded_response_times) + len(timeout_delayed_response_times)}")
        print()

        print(f"REWARDED TRIALS:")
        if rewarded_response_times:
            print(f"  Count: {len(rewarded_response_times)}")
            print(f"  Range: {min(rewarded_response_times):.1f} - {max(rewarded_response_times):.1f}ms")
            print(f"  Average: {sum(rewarded_response_times) / len(rewarded_response_times):.1f}ms")
            print(f"  Median: {sorted(rewarded_response_times)[len(rewarded_response_times)//2]:.1f}ms")
        else:
            print(f"  No rewarded trials with response times")

        if hr_rewarded_response_times:
            print(f"\nHR REWARDED TRIALS (response times):")
            print(f"  Count: {len(hr_rewarded_response_times)}")
            print(f"  Range: {min(hr_rewarded_response_times):.1f} - {max(hr_rewarded_response_times):.1f}ms")
            print(f"  Average: {sum(hr_rewarded_response_times)/len(hr_rewarded_response_times):.1f}ms")
        else:
            print(f"\nHR REWARDED TRIALS (response times): none")

        print(f"\nUNREWARDED TRIALS:")
        if unrewarded_response_times:
            print(f"  Count: {len(unrewarded_response_times)}")
            print(f"  Range: {min(unrewarded_response_times):.1f} - {max(unrewarded_response_times):.1f}ms")
            print(f"  Average: {sum(unrewarded_response_times) / len(unrewarded_response_times):.1f}ms")
            print(f"  Median: {sorted(unrewarded_response_times)[len(unrewarded_response_times)//2]:.1f}ms")
        else:
            print(f"  No unrewarded trials with response times")

        print(f"\nTIMEOUT TRIALS WITH DELAYED RESPONSES:")
        if timeout_delayed_response_times:
            print(f"  Count: {len(timeout_delayed_response_times)}")
            print(f"  Response time (poke out to delayed poke):")
            print(f"    Range: {min(timeout_delayed_response_times):.1f} - {max(timeout_delayed_response_times):.1f}ms")
            print(f"    Average: {sum(timeout_delayed_response_times) / len(timeout_delayed_response_times):.1f}ms")
            print(f"    Median: {sorted(timeout_delayed_response_times)[len(timeout_delayed_response_times)//2]:.1f}ms")
            print(f"  Response delay time (window end to delayed poke):")
            print(f"    Range: {min(timeout_response_delay_times):.1f} - {max(timeout_response_delay_times):.1f}ms")
            print(f"    Average: {sum(timeout_response_delay_times) / len(timeout_response_delay_times):.1f}ms")
            print(f"    Median: {sorted(timeout_response_delay_times)[len(timeout_response_delay_times)//2]:.1f}ms")
        else:
            print(f"  No timeout trials with delayed responses")

        print(f"\nALL TRIALS WITH RESPONSE TIMES:")
        all_response_times = rewarded_response_times + unrewarded_response_times + timeout_delayed_response_times
        if all_response_times:
            print(f"  Count: {len(all_response_times)}")
            print(f"  Range: {min(all_response_times):.1f} - {max(all_response_times):.1f}ms")
            print(f"  Average: {sum(all_response_times) / len(all_response_times):.1f}ms")
            print(f"  Median: {sorted(all_response_times)[len(all_response_times)//2]:.1f}ms")

    all_response_times = rewarded_response_times + unrewarded_response_times + timeout_delayed_response_times

    # NEW: build per-trial DataFrame
    per_trial_df = pd.DataFrame(per_trial_rows)

    return {
        'rewarded_response_times': rewarded_response_times,
        'unrewarded_response_times': unrewarded_response_times,
        'timeout_delayed_response_times': timeout_delayed_response_times,
        'timeout_response_delay_times': timeout_response_delay_times,
        'all_response_times': all_response_times,
        'failed_calculations': failed_calculations,
        'per_trial': per_trial_df,  # NEW
        'sample_offset_time_ms': sample_offset_time_ms,
        'minimum_sampling_time_ms_by_odor': minimum_sampling_time_ms_by_odor,
        'default_minimum_sampling_time_ms': default_minimum_sampling_time_ms,
        'minimum_sampling_time_ms': default_minimum_sampling_time_ms,
        'response_time_window_sec': response_time_sec,
    }

def abortion_classification(data, events, classification, odor_map, root, verbose=True): # Classify aborted trials with response times, poke times, FA etc. Part of wrapper function
    """
    Further classify aborted trials:
      - Compute valve and poke times per odor presentation (same rules as other trials)
      - Determine last relevant odor (last valve open with duration >= sample_offset_time_ms)
      - Compute poke time for that last odor (from poke-in that covers/starts at valve_start,
        merging gaps <= sample_offset_time_ms, ends at first large gap)
      - Abortion type:
          * reinitiation_abortion if last-odor poke >= required min sampling time for that odor
          * initiation_abortion otherwise
      - Abortion time = last cue-port poke-out within the trial window
      - False alarm (FA) detection window = (abortion_time, next cue-port poke after next InitiationSequence)
          * If any reward-port poke occurs in this window -> FA
            - FA_time_in: latency <= response_time
            - FA_time_out: latency <= 3 * response_time
            - FA_late: latency > 3 * response_time
          * Else nFA

    Returns:
      pd.DataFrame with detailed aborted trials:
        ['trial_id','sequence_start','sequence_end','odor_sequence',
         'position_valve_times','position_poke_times',
         'last_odor_position','last_odor_name','last_odor_valve_duration_ms',
         'last_odor_poke_time_ms','abortion_type',
         'abortion_time','fa_label','fa_time','fa_latency_ms']
    """
    schema_settings = {}
    schema_err: Exception | None = None
    try:
        _, schema_settings = detect_settings.detect_settings(root)
    except Exception as exc:
        schema_err = exc
        schema_settings = {}
        
    seq_len = schema_settings.get('sequenceLength')
    max_positions = int(seq_len) if seq_len is not None else None
    if max_positions is None or max_positions < 1:
        raise ValueError("sequenceLength missing or invalid; cannot proceed without a valid sequence length")

    DIP0 = data['digital_input_data'].get('DIPort0', pd.Series(dtype=bool)).astype(bool)
    DIP1 = data['digital_input_data'].get('DIPort1', pd.Series(dtype=bool)).astype(bool)
    DIP2 = data['digital_input_data'].get('DIPort2', pd.Series(dtype=bool)).astype(bool)
    
    # NOW use them
    dip1_rises = DIP1[DIP1 & ~DIP1.shift(1, fill_value=False)].index.tolist()
    dip2_rises = DIP2[DIP2 & ~DIP2.shift(1, fill_value=False)].index.tolist()
    reward_rises = sorted(dip1_rises + dip2_rises)

    # Parameters
    sample_offset_time, minimum_sampling_time_by_odor, response_time = get_experiment_parameters(root)
    sample_offset_time_ms = float(sample_offset_time) * 1000.0
    minimum_sampling_time_ms_by_odor = {
        str(odor): float(threshold) * 1000.0
        for odor, threshold in (minimum_sampling_time_by_odor or {}).items()
        if threshold is not None
    }

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

    def resolve_min_sampling_time_ms(odor_name):
        if odor_name is None:
            return default_minimum_sampling_time_ms
        return minimum_sampling_time_ms_by_odor.get(str(odor_name), default_minimum_sampling_time_ms)

    minimum_sampling_time_ms = float(default_minimum_sampling_time_ms)
    response_time_ms = float(response_time) * 1000.0

    # Inputs
    aborted_df = classification.get('aborted_sequences', pd.DataFrame())
    if not isinstance(aborted_df, pd.DataFrame) or aborted_df.empty:
        if verbose:
            print("abortion_classification: no aborted trials found.")
        return pd.DataFrame()

    DIP0 = data['digital_input_data'].get('DIPort0', pd.Series(dtype=bool)).astype(bool)  # cue port
    DIP1 = data['digital_input_data'].get('DIPort1', pd.Series(dtype=bool)).astype(bool)  # reward port 1
    DIP2 = data['digital_input_data'].get('DIPort2', pd.Series(dtype=bool)).astype(bool)  # reward port 2

    # Build global poke intervals for cue port
    def build_intervals(series_bool):
        rises = series_bool & ~series_bool.shift(1, fill_value=False)
        falls = ~series_bool & series_bool.shift(1, fill_value=False)
        starts = list(series_bool.index[rises])
        ends = list(series_bool.index[falls])
        intervals = []
        i = j = 0
        while i < len(starts) and j < len(ends):
            if ends[j] <= starts[i]:
                j += 1
                continue
            intervals.append((starts[i], ends[j]))
            i += 1
            j += 1
        return intervals

    cue_intervals = build_intervals(DIP0)

    # Reward-port rising edges (for FA)
    def rising_times(series_bool):
        rises = series_bool & ~series_bool.shift(1, fill_value=False)
        return list(series_bool.index[rises])

    reward_rises = sorted(rising_times(DIP1) + rising_times(DIP2))
    cue_rises = rising_times(DIP0)

    # Helper: bout from poke-in that covers/starts after anchor, merging gaps <= sample_offset_time_ms; no cap
    def bout_from_anchor(anchor_ts):
        if anchor_ts is None or not cue_intervals:
            return None, None, 0.0
        starts_only = [s for s, _ in cue_intervals]
        # interval covering anchor?
        idx = bisect_right(starts_only, anchor_ts) - 1
        within = None
        if 0 <= idx < len(cue_intervals):
            s0, e0 = cue_intervals[idx]
            if s0 <= anchor_ts < e0:
                within = idx
        if within is not None:
            k = within
        else:
            k = bisect_left(starts_only, anchor_ts)
            if k >= len(cue_intervals):
                return None, None, 0.0
        # merge forward across short gaps
        bout_start, cur_end = cue_intervals[k]
        m = k
        while m + 1 < len(cue_intervals):
            s2, e2 = cue_intervals[m + 1]
            gap_ms = (s2 - cur_end).total_seconds() * 1000.0
            if gap_ms <= sample_offset_time_ms:
                cur_end = max(cur_end, e2)
                m += 1
            else:
                break
        dur_ms = max(0.0, (cur_end - bout_start).total_seconds() * 1000.0)
        return bout_start, cur_end, float(dur_ms)

    # Build all valve activations (exclude Purge) with odor names
    olfactometer_valves = odor_map.get('olfactometer_valves', {})
    valve_to_odor = odor_map.get('valve_to_odor', {})

    def resolve_odor_name(olf_id, idx, col=None):
        # Try explicit mapping variants
        name = valve_to_odor.get((olf_id, idx))
        if name is None and col is not None:
            name = valve_to_odor.get(col)
        if name is None:
            name = valve_to_odor.get(f"{olf_id}{idx}")
        # Fallback to grid map
        if not isinstance(name, str):
            grid = odor_map.get('odour_to_olfactometer_map') or odor_map.get('odor_to_olfactometer_map')
            if isinstance(grid, (list, tuple)) and len(grid) > olf_id:
                row = grid[olf_id]
                if isinstance(row, (list, tuple)) and 0 <= idx < len(row):
                    name = row[idx]
        return name if isinstance(name, str) else None

    all_valve_activations = []
    for olf_id, df in olfactometer_valves.items():
        if df is None or getattr(df, 'empty', True):
            continue
        for i, col in enumerate(df.columns):
            odor_name = resolve_odor_name(olf_id, i, col=col)
            if not odor_name or odor_name.lower() == 'purge':
                continue
            s = df[col].astype(bool)
            rises = s & ~s.shift(1, fill_value=False)
            falls = ~s & s.shift(1, fill_value=False)
            starts = list(s.index[rises])
            ends = list(s.index[falls])
            j = 0
            for st in starts:
                while j < len(ends) and ends[j] <= st:
                    j += 1
                if j >= len(ends):
                    break
                all_valve_activations.append({
                    'start_time': starts[starts.index(st)],  # keep ref
                    'end_time': ends[j],
                    'odor_name': odor_name,
                    'olf_id': olf_id,
                    'col_index': i,
                })
                j += 1
    all_valve_activations.sort(key=lambda x: x['start_time'])

    # Helpers to extract trial events and per-odor poke times
    def trial_valve_events(t_start, t_end):
        evs = []
        for ev in all_valve_activations:
            if ev['end_time'] <= t_start:
                continue
            if ev['start_time'] >= t_end:
                break  # because sorted
            evs.append(ev)
        return evs

    def window_poke_summary(window_start, window_end):
        if window_start is None or window_end is None or window_start >= window_end:
            return {'poke_time_ms': 0.0, 'poke_first_in': None, 'poke_odor_start': window_start}
        s_bool = DIP0  # cue-port boolean series
        s_bool = s_bool.sort_index()
        prev = s_bool.loc[:window_start]
        in_at_start = bool(prev.iloc[-1]) if len(prev) else False
        w = s_bool.loc[window_start:window_end]
        if w.empty and not in_at_start:
            return {'poke_time_ms': 0.0, 'poke_first_in': None, 'poke_odor_start': window_start}
        rises = w & ~w.shift(1, fill_value=in_at_start)
        falls = ~w & w.shift(1, fill_value=in_at_start)
        intervals = []
        cur = window_start if in_at_start else None
        first_in = window_start if in_at_start else None
        for ts in w.index:
            if rises.get(ts, False) and cur is None:
                cur = ts
                if first_in is None:
                    first_in = ts
            if falls.get(ts, False) and cur is not None:
                intervals.append((cur, ts))
                cur = None
        if cur is not None:
            intervals.append((cur, window_end))
        if not intervals:
            last_poke_end = _last_poke_end_before(s_bool, window_start)
            grace_ms, grace_end = _grace_overlap_ms(last_poke_end, window_start, window_end)
            if grace_ms > 0.0:
                return {
                    'poke_time_ms': grace_ms,
                    'poke_first_in': window_start,
                    'poke_odor_start': window_start,
                    'poke_odor_end': grace_end,
                }
            return {'poke_time_ms': 0.0, 'poke_first_in': None, 'poke_odor_start': window_start}
        merged = [intervals[0]]
        for s2, e2 in intervals[1:]:
            ls, le = merged[-1]
            gap_ms = (s2 - le).total_seconds() * 1000.0
            if gap_ms <= sample_offset_time_ms:
                merged[-1] = (ls, max(le, e2))
            else:
                merged.append((s2, e2))

        first_block_start, first_block_end = merged[0]
        first_block_ms = (first_block_end - first_block_start).total_seconds() * 1000.0
        return {
            'poke_time_ms': float(first_block_ms),
            'poke_first_in': first_in,
            'poke_odor_start': first_block_start,
            'poke_odor_end': first_block_end
        }

    # InitiationSequence times (for FA end window)
    init_times = []
    ci_key = 'combined_initiation_sequence_df'
    if ci_key in events and isinstance(events[ci_key], pd.DataFrame) and not events[ci_key].empty:
        init_times = list(events[ci_key]['Time'])

    # Process aborted trials
    rows = []
    for _, tr in aborted_df.iterrows():
        t_start = tr.get('sequence_start') or tr.get('trial_start') or tr.get('start_time')
        t_end = tr.get('sequence_end') or tr.get('trial_end') or tr.get('end_time')
        trial_id = tr.get('trial_id', tr.name)
        if pd.isna(t_start) or pd.isna(t_end) or t_start is None or t_end is None:
            continue

        # Extract valve events (collapse consecutive repeats) and odor sequence
        evs_raw = trial_valve_events(t_start, t_end)
        evs: list[dict] = []
        for ev in evs_raw:
            if evs and evs[-1]['odor_name'] == ev['odor_name']:
                evs[-1] = ev  # keep only the latest activation in a consecutive block
            else:
                evs.append(ev)

        # Assign sequential positions up to max_positions
        capped_evs: list[dict] = []
        positions: list[int] = []
        for ev in evs:
            if len(capped_evs) >= max_positions:
                break
            capped_evs.append(ev)
            positions.append(len(positions) + 1)
        evs = capped_evs

        # Collect presentations (only those with pokes are considered valid)
        presentations_all: list[dict] = []
        position_valve_times: dict[int, dict] = {}
        position_poke_times: dict[int, dict] = {}

        for idx_in_trial, (e, pos) in enumerate(zip(evs, positions)):
            valve_start = e['start_time']
            valve_end = e['end_time']
            valve_dur_ms = (valve_end - valve_start).total_seconds() * 1000.0
            required_min_ms = float(resolve_min_sampling_time_ms(e['odor_name']))

            psum = window_poke_summary(valve_start, valve_end)
            has_poke = float(psum.get('poke_time_ms', 0.0)) > 0.0

            pres_entry = {
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
            }
            presentations_all.append(pres_entry)

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

        presentations_valid = [p for p in presentations_all if p.get('has_poke')]
        odor_sequence = [p['odor_name'] for p in presentations_valid]

        # Last relevant odor: must have a poke and meet duration threshold
        last_idx = None
        for i in range(len(presentations_valid) - 1, -1, -1):
            if presentations_valid[i].get('valve_duration_ms', 0.0) >= sample_offset_time_ms:
                last_idx = i
                break

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
            last_required_min_ms = float(resolve_min_sampling_time_ms(last_odor_name))

        # Abortion type
        abortion_type = (
            'reinitiation_abortion'
            if (not np.isnan(last_required_min_ms) and last_odor_poke_ms >= last_required_min_ms)
            else 'initiation_abortion'
        )

        # Abortion time = last cue-port poke-out in [t_start, t_end]
        abortion_time = None
        # Intervals overlapping trial
        overlapping = [(max(s, t_start), min(e, t_end)) for (s, e) in cue_intervals if e > t_start and s < t_end]
        if overlapping:
            abortion_time = overlapping[-1][1]

        # FA window: from abortion_time to next cue-port poke after next InitiationSequence
        fa_label = 'nFA'
        fa_time = pd.NaT
        fa_latency_ms = np.nan
        fa_port = None 

        if abortion_time is not None:
            # Find next initiation after t_end
            next_init = None
            if init_times:
                idx = bisect_right(init_times, t_end)
                if idx < len(init_times):
                    next_init = init_times[idx]
            # Find first cue-port rising AFTER next_init
            fa_window_end = None
            if next_init is not None and cue_rises:
                k = bisect_right(cue_rises, next_init)
                if k < len(cue_rises):
                    fa_window_end = cue_rises[k]
            # If no end found, cap at session end (last timestamp we have)
            if fa_window_end is None:
                # fallback: last timestamp among known streams
                candidates = []
                for df in [DIP0, DIP1, DIP2]:
                    if not df.empty:
                        candidates.append(df.index[-1])
                fa_window_end = max(candidates) if candidates else abortion_time

            # Scan for first reward-port poke in (abortion_time, fa_window_end]
            if reward_rises:
                lo = bisect_right(reward_rises, abortion_time)
                hi = bisect_right(reward_rises, fa_window_end)
                if lo < hi:
                    fa_time = reward_rises[lo]
                    fa_latency_ms = (fa_time - abortion_time).total_seconds() * 1000.0
                    
                    # Determine which port the FA poke came from ← NEW
                    if fa_time in dip1_rises:
                        fa_port = 1
                    elif fa_time in dip2_rises:
                        fa_port = 2
                    
                    if fa_latency_ms <= response_time_ms:
                        fa_label = 'FA_time_in'
                    elif fa_latency_ms <= 3.0 * response_time_ms:
                        fa_label = 'FA_time_out'
                    else:
                        fa_label = 'FA_late'


        rows.append({
            'trial_id': trial_id,
            'sequence_start': t_start,
            'sequence_end': t_end,
            'odor_sequence': odor_sequence,
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
        })

    aborted_detailed = pd.DataFrame(rows)

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

    aborted_detailed['fa_label'] = aborted_detailed['fa_label'].apply(_norm_fa)


    if verbose and not aborted_detailed.empty:
        total = int(len(aborted_detailed))
        ini = int((aborted_detailed['abortion_type'] == 'initiation_abortion').sum())
        rei = int((aborted_detailed['abortion_type'] == 'reinitiation_abortion').sum())

        def pct(n, d):
            return (n / d * 100.0) if d else 0.0



        print("=" * 80)
        print("ABORTED TRIALS CLASSIFICATION SUMMARY")
        print("=" * 80)

        print(f"- Total Aborted Trials: {total}")
        print(f"  - Re-Initiation Abortions: {rei} ({pct(rei, total):.1f}%)")
        print(f"  - Initiation Abortions:    {ini} ({pct(ini, total):.1f}%)")

        # False Alarms summary
        fa_in_count  = int((aborted_detailed['fa_label'] == 'FA_time_in').sum())
        fa_out_count = int((aborted_detailed['fa_label'] == 'FA_time_out').sum())
        fa_late_count= int((aborted_detailed['fa_label'] == 'FA_late').sum())
        fa_total = fa_in_count + fa_out_count + fa_late_count
        nfa_count = total - fa_total

        print("\nFalse Alarms:")
        print(f"  - non-FA Abortions: {nfa_count}")
        print(f"  - False Alarm abortions: {fa_total} ({pct(fa_total, total):.1f}%)")
        if fa_total > 0:
            print(f"      - FA Time In (Within Response Time Window {response_time_ms}):  {fa_in_count} ({pct(fa_in_count, fa_total):.1f}%)")
            s_in = pd.to_numeric(
                aborted_detailed.loc[aborted_detailed['fa_label'] == 'FA_time_in', 'fa_latency_ms'],
                errors='coerce'
            ).dropna()
            if len(s_in):
                print(f"          - Response Time: avg={s_in.mean():.1f} ms, range: {s_in.min():.1f} - {s_in.max():.1f} ms")
            print(f"      - FA Time Out (Up to 3x Response Time Window {response_time}):  {fa_out_count} ({pct(fa_out_count, fa_total):.1f}%)")
            s_out = pd.to_numeric(
                aborted_detailed.loc[aborted_detailed['fa_label'] == 'FA_time_out', 'fa_latency_ms'],
                errors='coerce'
            ).dropna()
            if len(s_out):
                print(f"          - Response Time: avg={s_out.mean():.1f} ms, range: {s_out.min():.1f} - {s_out.max():.1f} ms")
            print(f"      - FA Late (After 3x Response Time up to next trial):{fa_late_count} ({pct(fa_late_count, fa_total):.1f}%)")
            s_late = pd.to_numeric(
                aborted_detailed.loc[aborted_detailed['fa_label'] == 'FA_late', 'fa_latency_ms'],
                errors='coerce'
            ).dropna()
            if len(s_late):
                print(f"          - Response Time: avg={s_late.mean():.1f} ms, range: {s_late.min():.1f} - {s_late.max():.1f} ms")

            hr_positions = classification.get('hidden_rule_positions') or []
            if not hr_positions:
                fallback_pos = classification.get('hidden_rule_position')
                if fallback_pos is not None:
                    hr_positions = [fallback_pos]
            hr_positions = [int(pos) for pos in hr_positions if pos is not None]

            # Normalize FA labels
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
            aborted_detailed['fa_label'] = aborted_detailed['fa_label'].apply(_norm_fa)

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

            # Helper to print FA breakdown
            def _print_fa_counts(df, indent="    "):
                order = ['nFA', 'FA_time_in', 'FA_time_out', 'FA_late']
                cnt = df['fa_label'].value_counts().reindex(order, fill_value=0)
                total = int(len(df))
                for lbl in order:
                    v = int(cnt.get(lbl, 0))
                    pct = (v / total * 100.0) if total else 0.0
                    print(f"{indent}{lbl}: {v} ({pct:.1f}%)")

            total_at_hr = int(len(abortions_at_hr_pos))
            hr_pos_display = ", ".join(str(pos) for pos in hr_positions) if hr_positions else "None"
            print(f"\n  Abortions at Hidden Rule Positions {hr_pos_display}: n={total_at_hr}")

            total_in_hr = int(len(in_hr_trials))
            print(f"    Of which in Hidden Rule Trials: n={total_in_hr}")
            if total_in_hr > 0:
                _print_fa_counts(in_hr_trials, indent="        ")

            total_non_hr = int(len(non_hr_trials))
            print(f"    Non-Hidden Rule Abortions at HR Location: n={total_non_hr}")
            if total_non_hr > 0:
                _print_fa_counts(non_hr_trials, indent="        ")

        # Helper for stats lines
        def stats_line(series, label):
            s = pd.to_numeric(series, errors='coerce').dropna()
            if s.empty:
                print(f"{label}: n=0")
            else:
                print(f"{label}: n={len(s)} | avg={s.mean():.1f} ms | range={s.min():.1f}-{s.max():.1f} ms")

        # Non-last odor poke times (>= odor-specific minimum sampling time), requires 'presentations'
        if 'presentations' in aborted_detailed.columns and 'last_event_index' in aborted_detailed.columns:
            pres_df = aborted_detailed[['trial_id', 'presentations', 'last_event_index']].explode('presentations')
            pres_df = pres_df.dropna(subset=['presentations']).copy()
            if not pres_df.empty:
                pres = pd.concat(
                    [pres_df.drop(columns=['presentations']),
                     pres_df['presentations'].apply(pd.Series)],
                    axis=1
                )
                # Exclude the last relevant odor per trial
                pres['is_last'] = pres['index_in_trial'] == pres['last_event_index']
                pres = pres[~pres['is_last']].copy()

                # Only count pokes meeting the odor-specific minimum sampling time
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

                # By position
                if 'position' in pres_valid.columns and not pres_valid.empty:
                    for pos, grp in pres_valid.groupby('position'):
                        stats_line(grp['poke_time_ms'], f"  - Position {int(pos)}")

                # By odor name/type
                if 'odor_name' in pres_valid.columns and not pres_valid.empty:
                    for odor, grp in pres_valid.groupby('odor_name'):
                        stats_line(grp['poke_time_ms'], f"  - Odor {odor}")
            else:
                print("\nNon-last Odor Pokes: n=0 (no presentations info)")
        else:
            print("\nNon-last odor pokes: presentations not attached; update abortion_classification to store 'presentations' and 'last_event_index'.")

        # Last-odor poke stats by abortion type
        print("\nLast Odor Poke Times:")
        stats_line(
            aborted_detailed.loc[aborted_detailed['abortion_type'] == 'reinitiation_abortion', 'last_odor_poke_time_ms'],
            "  - Re-Initiation Abortions"
        )
        stats_line(
            aborted_detailed.loc[aborted_detailed['abortion_type'] == 'initiation_abortion', 'last_odor_poke_time_ms'],
            "  - Initiation Abortions"
        )

        # Counts by last odor name
        print("\nCounts by last odor:")
        if 'last_odor_name' in aborted_detailed.columns:
            by_odor = (
                aborted_detailed
                .groupby(['last_odor_name', 'abortion_type'])
                .size()
                .unstack(fill_value=0)
                .rename(columns={'reinitiation_abortion': 'Re-initiation', 'initiation_abortion': 'Initiation'})
            )
            # Total per odor row
            totals = aborted_detailed.groupby('last_odor_name').size()
            for odor in totals.index:
                rei_c = int(by_odor.loc[odor].get('Re-initiation', 0))
                ini_c = int(by_odor.loc[odor].get('Initiation', 0))
                tot = int(totals.loc[odor])
                print(f"  - {odor}: {tot} abortions, Re-initiation {rei_c}, Initiation {ini_c}")
        else:
            print("  (missing last_odor_name)")

        # Counts by last position
        print("\nCounts by last position:")
        if 'last_odor_position' in aborted_detailed.columns:
            by_pos = (
                aborted_detailed
                .groupby(['last_odor_position', 'abortion_type'])
                .size()
                .unstack(fill_value=0)
                .rename(columns={'reinitiation_abortion': 'Re-initiation', 'initiation_abortion': 'Initiation'})
            )
            totals_pos = aborted_detailed.groupby('last_odor_position').size()
            for pos in sorted(totals_pos.index):
                rei_c = int(by_pos.loc[pos].get('Re-initiation', 0))
                ini_c = int(by_pos.loc[pos].get('Initiation', 0))
                tot = int(totals_pos.loc[pos])
                print(f"  - Position {int(pos)}: {tot} abortions, Re-initiation {rei_c}, Initiation {ini_c}")
        else:
            print("  (missing last_odor_position)")

    def build_abortion_index(df: pd.DataFrame):
        idx = {}
        if df is None or df.empty:
            return {
                'by_trial': {},
                'by_position': {},
                'by_odor': {},
                'by_type': {},
                'by_fa_label': {},
            }
        # Ensure trial_id exists as indexable key
        df2 = df.copy()
        # Some pipelines may have non-unique or NaN trial_id; drop NaN for dict keys
        df2 = df2.dropna(subset=['trial_id'])
        # by_trial -> full row as a dict for each trial_id
        try:
            by_trial = df2.set_index('trial_id', drop=False).apply(lambda r: r.to_dict(), axis=1).to_dict()
        except Exception:
            # fallback: iterate
            by_trial = {row['trial_id']: row.to_dict() for _, row in df2.iterrows()}

        # Helper to group trial IDs by a column
        def group_ids(col):
            m = {}
            if col in df2.columns:
                for k, g in df2.groupby(col):
                    # Keep order by sequence_start if present
                    trials = list(g.sort_values('sequence_start')['trial_id']) if 'sequence_start' in g else list(g['trial_id'])
                    m[k] = trials
            return m

        idx['by_trial'] = by_trial
        idx['by_position'] = group_ids('last_odor_position')
        idx['by_odor'] = group_ids('last_odor_name')
        idx['by_type'] = group_ids('abortion_type')
        idx['by_fa_label'] = group_ids('fa_label')
        return idx

    aborted_index = build_abortion_index(aborted_detailed)

    # Attach to classification dict for downstream use
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
        fa_port = None  # ← NEW
        
        reward_after = [t for t in reward_rises if attempt_end < t <= next_cue_in]
        if reward_after:
            fa_time = reward_after[0]
            fa_latency_ms = (fa_time - attempt_end).total_seconds() * 1000.0
            
            # Determine which port ← NEW
            if fa_time in dip1_rises:
                fa_port = 1
            elif fa_time in dip2_rises:
                fa_port = 2
            
            if fa_latency_ms <= response_time_ms:
                fa_label = 'FA_time_in'
            elif fa_latency_ms <= 3.0 * response_time_ms:
                fa_label = 'FA_time_out'
            else:
                fa_label = 'FA_late'

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
            'fa_port': fa_port,  # ← NEW
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
                per_trial_df[['trial_id', 'response_time_ms', 'response_time_category']],
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


# ========================== Functions for saving results ==========================
# Moved to hypnose_behavior.io.save_results during the restructuring; re-exported here so existing
# callers (the multi-run runner below, notebooks) keep importing from classification_utils.
from hypnose_behavior.io.save_results import (  # noqa: F401
    save_session_analysis_results,
    resolve_derivatives_output_dir,
    _normalize_df_for_io,
    _json_safe,
    _json_default,
    _find_rawdata_root,
    _find_parent_named,
)


# ========================== Functions for multiple session analysis ========================== 

# Merge utilities moved to hypnose_behavior.trial_classification.merge during the restructuring; re-exported
# here so the runner below and notebooks keep importing them from classification_utils.
from hypnose_behavior.trial_classification.merge import (  # noqa: F401
    merge_classifications, _concat_align, _assign_global_trial_ids, _with_run_id, _coerce_int_like,
)


# Summary report moved to hypnose_behavior.trial_classification.summary during the restructuring;
# re-exported here so the runner below and notebooks keep importing it from classification_utils.
from hypnose_behavior.trial_classification.summary import print_merged_session_summary  # noqa: F401


# Runner moved to hypnose_behavior.trial_classification.run during the restructuring; re-exported here so
# the harness, notebooks, and batch entry points keep importing it from classification_utils.
from hypnose_behavior.trial_classification.run import (  # noqa: F401
    analyze_session_multi_run_by_id_date, batch_analyze_sessions,
    build_position_pokes_table, _parse_date_input,
)


# ========================= Further functions / miscillaneous =========================
# plot_valve_and_poke_events moved to hypnose_behavior.visualization.valve_poke_plots during the
# restructuring; re-exported here so notebooks keep importing it from classification_utils.
from hypnose_behavior.visualization.valve_poke_plots import plot_valve_and_poke_events  # noqa: F401
