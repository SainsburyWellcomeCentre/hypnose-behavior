"""Trial detection -- find each initiation and the sampling attempts that follow it.

``detect_trials`` walks the initiation events and resolves the valve activations after each
one into sampling attempts and the failed-attempt bookkeeping. On odour-discrimination runs the
rig's AwaitReward decides which attempt is the trial; elsewhere the sampling threshold does.

It runs before classification and produces the ``trial_counts`` dict that
``classify_trials`` and ``analyze_response_times`` both consume.
"""
from __future__ import annotations

import warnings
from collections.abc import Mapping

import pandas as pd

import hypnose_behavior.io.detect_stage as detect_stage_module
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


# More initiation windows than this losing openings to `_drop_pre_poke_openings` is not the
# start-of-run valve cleaning, which falls before or inside the first one.
MAX_PRE_POKE_WINDOWS = 2


def _drop_pre_poke_openings(valve_events, cue_pokes, initiation_times):
    """Valve openings from the run's first cue-port poke onwards.

    An odor valve cannot open for an attempt before the animal has poked the cue port at all,
    so every earlier opening is the rig's own valve cleaning at the start of the run. A run
    with no cue poke keeps none.

    Returns ``(kept_events, n_windows)``: ``n_windows`` counts the initiation windows that lost
    at least one opening.
    """
    rises = windows.rising_edges(cue_pokes)
    first_poke = rises[0] if rises else None
    dropped = [ev for ev in valve_events if first_poke is None or ev['start_time'] < first_poke]
    if not dropped:
        return valve_events, 0
    kept = [ev for ev in valve_events if first_poke is not None and ev['start_time'] >= first_poke]

    starts = pd.Series(sorted(pd.to_datetime(initiation_times)), dtype='datetime64[ns]')
    windows_hit = {int(starts.searchsorted(ev['start_time'], side='right')) - 1 for ev in dropped}
    windows_hit.discard(-1)
    return kept, len(windows_hit)


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


def _measure_attempt(attempt_event, attempt_num, poke_periods, cue_pokes, next_initiation_time,
                     *, required_minimum_ms, sample_offset_time_ms, verbose):
    """Sampling time of one attempt against its odor's minimum.

    Returns ``(attempt_start, continuous_time_ms, last_segment_end, success)``.
    ``attempt_start`` is the first poke inside the valve window, or the valve opening when
    no poke overlaps it.
    """
    event_start = attempt_event['start_time']
    event_end = attempt_event['end_time']
    attempt_odor = attempt_event['odor_name']
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
    else:
        vprint(verbose, f"      FAILED: {continuous_time:.1f}ms < {required_minimum_ms:.1f}ms")
    return attempt_start, continuous_time, last_seg_end, success


def _failed_attempt_entry(initiation_time, attempt_event, attempt_num, measured, required_minimum_ms,
                          next_attempt_start, *, failure_reason='insufficient_continuous_poke_time'):
    """One ``non_initiated_sequences`` row. ``measured`` is `_measure_attempt`'s return value.

    ``met_min_sampling`` records whether the sampling time reached the odor's minimum, so an
    attempt the rig did not initiate despite enough sampling stays identifiable.
    """
    attempt_start, continuous_time, last_seg_end, success = measured
    return {
        'initiation_sequence_time': initiation_time,
        'attempt_start': attempt_start,
        'attempt_end': last_seg_end if last_seg_end is not None else attempt_event['start_time'],
        'continuous_poke_time_ms': continuous_time,
        'attempt_number': attempt_num,
        'timestamp': attempt_start,
        'failure_reason': failure_reason,
        'required_min_sampling_time_ms': required_minimum_ms,
        'odor_name': attempt_event['odor_name'],
        'next_attempt_start': next_attempt_start,
        'met_min_sampling': bool(success),
    }


def _next_attempt_starts(attempt_events) -> dict:
    """``{attempt_number: start of the next attempt}``, numbering from 1."""
    return {
        idx + 1: (attempt_events[idx + 1]['start_time'] if idx + 1 < len(attempt_events) else None)
        for idx in range(len(attempt_events))
    }


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
    attempt_next_start = _next_attempt_starts(attempt_events)

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

        measured = _measure_attempt(
            attempt_event, attempt_num, poke_periods, cue_pokes, next_initiation_time,
            required_minimum_ms=required_minimum_ms, sample_offset_time_ms=sample_offset_time_ms,
            verbose=verbose)
        attempt_start, continuous_time, _last_seg_end, success = measured

        if success:
            winner = {
                'start': attempt_start,
                'duration_ms': continuous_time,
                'attempt_number': attempt_num,
                'required_min_ms': required_minimum_ms,
                'odor_name': attempt_odor,
            }
            return winner, failed_attempts, None, attempt_num

        failed_entry = _failed_attempt_entry(
            initiation_time, attempt_event, attempt_num, measured, required_minimum_ms,
            attempt_next_start.get(attempt_num))
        failed_attempts.append(failed_entry)
        pending_failed_attempt = failed_entry

    return None, failed_attempts, pending_failed_attempt, attempt_num


def _run_await_reward_attempts(attempt_events, poke_periods, cue_pokes, initiation_time,
                               next_initiation_time, await_reward_times, *, required_min_ms_for,
                               sample_offset_time_ms, verbose):
    """Odour discrimination: the rig's AwaitReward decides which attempt is the trial.

    The trial is the last valve opening that starts before the first AwaitReward between this
    initiation and the next. The sampling threshold only annotates it: a trial that did not
    reach it carries ``fallback_reason = 'await_reward_event'``. Every earlier opening is a
    failed attempt, including one that reached the threshold without the rig initiating
    (``met_min_sampling``, ``failure_reason = 'no_await_reward'``). Openings after the
    AwaitReward fall in the response period and are not attempts.

    Without an AwaitReward in the window, or with none preceded by an opening, there is no
    trial and every opening is a failed attempt.

    Returns ``(winner, failed_attempts)``.
    """
    attempt_next_start = _next_attempt_starts(attempt_events)
    numbered = [(num, ev) for num, ev in enumerate(attempt_events, start=1)
                if ev['end_time'] > ev['start_time']]

    in_window = await_reward_times[(await_reward_times >= initiation_time)
                                   & (await_reward_times <= next_initiation_time)]
    first_await = in_window.min() if not in_window.empty else None
    before_await = ([(num, ev) for num, ev in numbered if ev['start_time'] <= first_await]
                    if first_await is not None else [])
    winner_num = before_await[-1][0] if before_await else None
    considered = before_await if before_await else numbered
    if first_await is not None and winner_num is None:
        vprint(verbose, f"    AwaitReward at {first_await} precedes every valve opening — no trial")

    winner = None
    failed_attempts: list[dict] = []
    for num, ev in considered:
        required_minimum_ms = required_min_ms_for(ev['odor_name'])
        measured = _measure_attempt(
            ev, num, poke_periods, cue_pokes, next_initiation_time,
            required_minimum_ms=required_minimum_ms, sample_offset_time_ms=sample_offset_time_ms,
            verbose=verbose)
        attempt_start, continuous_time, _last_seg_end, success = measured
        if num == winner_num:
            vprint(verbose, f"    AwaitReward at {first_await} — attempt {num} is the trial")
            winner = {
                'start': attempt_start,
                'duration_ms': continuous_time,
                'attempt_number': num,
                'required_min_ms': required_minimum_ms,
                'odor_name': ev['odor_name'],
            }
            if not success:
                winner['fallback_reason'] = 'await_reward_event'
            continue
        failed_attempts.append(_failed_attempt_entry(
            initiation_time, ev, num, measured, required_minimum_ms, attempt_next_start.get(num),
            failure_reason='no_await_reward' if success else 'insufficient_continuous_poke_time'))
    return winner, failed_attempts


def detect_trials(data, events, root, odor_map, verbose=True, stage=None):
    """Detect initiated trials from cue-poke and valve streams.

    One *attempt* is one valve opening between consecutive InitiationSequence events, from the
    run's first cue-port poke onwards (`_drop_pre_poke_openings`). Its
    sampling time is the animal's cue-port poke inside the valve window, where pokes
    separated by gaps shorter than ``sampleOffsetTime`` count as one continuous sample.

    On odour-discrimination runs with an AwaitReward record, the rig decides: the trial is the
    last opening before the initiation's first AwaitReward, and every earlier opening is a
    non-initiated sequence (`_run_await_reward_attempts`).

    Otherwise an attempt initiates a trial when its sampling time reaches the minimum for its
    odor. The first attempt to reach it ends the search, and the failures before it are
    recorded as non-initiated sequences. A following attempt with a *different* odor promotes
    the previous short attempt to a trial (the sequence moved on, so the sample was real).

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

    n_valve_events = len(valve_events)
    valve_events, pre_poke_windows = _drop_pre_poke_openings(
        valve_events, cue_pokes, initiation_events['Time'])
    vprint(verbose, f"Valve openings before the first cue poke (not attempts): "
                    f"{n_valve_events - len(valve_events)}")
    if pre_poke_windows > MAX_PRE_POKE_WINDOWS:
        warnings.warn(
            f"{root}: valve openings before the first cue-port poke fall in {pre_poke_windows} "
            f"initiation windows (more than the start-of-run valve cleaning explains). They are "
            f"not counted as attempts; check this run's cue-port and valve records.",
            RuntimeWarning, stacklevel=2)

    await_reward_df = events.get('combined_await_reward_df') if isinstance(events, Mapping) else None
    # The loader gives a run without an ExperimentEvents folder a frame with no columns, and a
    # run whose events hold no AwaitReward an empty frame that still has `Time`. Only the first
    # lacks the rig's record; the second is a run in which the rig initiated nothing.
    await_reward_recorded = isinstance(await_reward_df, pd.DataFrame) and 'Time' in await_reward_df.columns
    if await_reward_recorded and not await_reward_df.empty:
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

        if is_odour_discrimination and await_reward_recorded:
            winner, failed_attempts = _run_await_reward_attempts(
                attempt_events, poke_periods, cue_pokes, initiation_time, next_initiation_time,
                await_reward_times,
                required_min_ms_for=required_min_ms_for,
                sample_offset_time_ms=sample_offset_time_ms,
                verbose=verbose,
            )
        else:
            winner, failed_attempts, _pending, _attempt_num = _run_sampling_attempts(
                attempt_events, poke_periods, cue_pokes, initiation_time, next_initiation_time,
                required_min_ms_for=required_min_ms_for,
                sample_offset_time_ms=sample_offset_time_ms,
                verbose=verbose,
            )

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
