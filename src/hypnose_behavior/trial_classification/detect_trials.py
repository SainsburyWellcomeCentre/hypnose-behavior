"""Trial detection -- find each initiation and the sampling attempts that follow it.

``detect_trials`` walks the initiation events and resolves the valve activations after each
one into sampling attempts, including the failed-attempt bookkeeping and the AwaitReward
promotion that rescues an attempt the valve record alone would have dropped.

It runs before classification and produces the ``trial_counts`` dict that
``classify_trials`` and ``analyze_response_times`` both consume.
"""
from __future__ import annotations

from collections.abc import Mapping

import pandas as pd

import hypnose_behavior.trial_classification.detect_stage as detect_stage_module
import hypnose_behavior.trial_classification.windows as windows
from hypnose_behavior.trial_classification.params import _sampling_parameters_ms
from hypnose_behavior.utils.helpers import vprint


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
