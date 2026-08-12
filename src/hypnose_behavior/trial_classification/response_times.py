"""Response-time analysis for completed trials.

``analyze_response_times`` measures how long the animal took to reach the reward port and
labels each completed trial ``rewarded`` / ``unrewarded`` / ``timeout``. It emits a category
only where it can also compute a response time; the remaining trials are counted in
``failed_calculations`` and that coverage gap is deliberate (``DECISIONS.md`` section 14 --
unify the rule, do not unify the coverage).

The anchor falls back to the animal's last cue-port exit *before* the odor when the scan
inside the odor window finds no exit; that rescue is correct and still fires on 20 trials.
Do not treat it as dead because ``poke_source`` exists (``DECISIONS.md`` section 15).

Positions come from ``windows.first_occurrence_positions``, a thin entry point into
``windows.positions_by_odor`` -- the single position rule, shared with ``classify_trials``
since the two were measured and found to agree on all but one experiment-faulted trial.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import hypnose_behavior.trial_classification.windows as windows
from hypnose_behavior.trial_classification.hidden_rule import (
    _check_hidden_rule, _drop_final_hidden_rule_index,
    _hidden_rule_indices_from_stage_or_schema, _hidden_rule_odor_set,
    _hidden_rule_positions, _print_hidden_rule_header,
)
from hypnose_behavior.trial_classification.outcome import classify_completed_trial
from hypnose_behavior.trial_classification.params import (
    _get_single_reward_info, _sampling_parameters_ms,
)
from hypnose_behavior.trial_classification.windows import (
    _next_after, _odourdisc_reward_window_end, _recording_end,
)


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
