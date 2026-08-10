"""Pure poke/valve window primitives for trial classification.

A leaf, in the sense of ``DECISIONS.md`` section 3: **this module imports nothing from the
package** -- only the standard library and pandas. Everything here is a pure function of
timestamps, boolean series and plain dicts, with no session context, no ``data``/``events``
dictionaries and no I/O, so each piece is independently testable and importable from anywhere
in the package (including ``io/``) without creating a cycle. Keep it that way.

The functions here are deliberately *not* generalised across their call sites. Several look
like near-duplicates -- there are three different ways of pairing valve rise/fall edges in this
codebase, and two different poke-bout merges. They are different rules, and section 13 of
``DECISIONS.md`` is the reason each keeps its own name and its own docstring saying what it
does differently, rather than being folded into one function behind a flag.
"""
from __future__ import annotations

import pandas as pd


# --------------------------------------------------------------------------------------
# Edge detection
# --------------------------------------------------------------------------------------

def rising_edges(series_bool: pd.Series) -> list:
    """Timestamps where a boolean series goes False -> True."""
    rises = series_bool & ~series_bool.shift(1, fill_value=False)
    return list(series_bool.index[rises])


def falling_edges(series_bool: pd.Series) -> list:
    """Timestamps where a boolean series goes True -> False."""
    falls = ~series_bool & series_bool.shift(1, fill_value=False)
    return list(series_bool.index[falls])


def rising_edges_between(series, window_start, window_end) -> list:
    """Rising edges inside ``[window_start, window_end]``, resolved *within the slice*.

    Not the same as filtering ``rising_edges(series)`` to the window: the shift is taken after
    slicing, with ``fill_value=False``, so a port already IN at ``window_start`` counts as a
    rise at the first in-window sample rather than being skipped. Every reward-port count in
    ``classification_utils`` depends on that, which is why the slice happens first.

    Returns ``[]`` for an empty series, matching the ``if not port.empty`` guard at each site.
    """
    if series is None or series.empty:
        return []
    window = series[window_start:window_end]
    starts = window & ~window.shift(1, fill_value=False)
    return starts[starts == True].index.tolist()  # noqa: E712 -- preserves original semantics


# --------------------------------------------------------------------------------------
# Valve activations
# --------------------------------------------------------------------------------------

def valve_windows_dropping_unclosed(olfactometer_valves, valve_to_odor) -> list[dict]:
    """Valve open windows, ``detect_trials``' rule.

    Casts each column to bool, pairs edges with a single advancing pointer, and **drops** an
    activation that has no later deactivation. Yields dicts of
    ``start_time`` / ``end_time`` / ``odor_name``, sorted by start.

    The other two builders in this codebase differ: ``classify_trials`` closes an unterminated
    activation at the last timestamp of the series instead of dropping it, and
    ``abortion_classification`` resolves odor names through a four-way lookup.
    """
    valve_events: list[dict] = []
    for olf_id, valve_df in (olfactometer_valves or {}).items():
        if valve_df is None or getattr(valve_df, 'empty', True):
            continue
        for valve_idx, valve_col in enumerate(valve_df.columns):
            valve_key = f"{olf_id}{valve_idx}"
            odor_name = valve_to_odor.get(valve_key)
            if not odor_name or str(odor_name).lower() == 'purge':
                continue
            valve_series = valve_df[valve_col].astype(bool)
            activation_times = rising_edges(valve_series)
            deactivation_times = falling_edges(valve_series)
            j = 0
            for activation_time in activation_times:
                while j < len(deactivation_times) and deactivation_times[j] <= activation_time:
                    j += 1
                if j >= len(deactivation_times):
                    break
                valve_events.append({
                    'start_time': activation_time,
                    'end_time': deactivation_times[j],
                    'odor_name': str(odor_name),
                })
    valve_events.sort(key=lambda ev: ev['start_time'])
    return valve_events


def valve_windows_closing_at_series_end(olfactometer_valves, valve_to_odor) -> list[dict]:
    """Valve open windows, the rule shared by ``classify_trials`` and ``analyze_response_times``.

    Differs from ``valve_windows_dropping_unclosed`` in three ways: it does not cast the column
    to bool, it takes the earliest deactivation after each activation rather than advancing a
    pointer, and an activation with no later deactivation is closed at the **last timestamp of
    the series** instead of being dropped. Carries ``valve_key``, which position assignment
    needs to tell repeat activations of one valve from a genuine re-entry.
    """
    all_valve_activations: list[dict] = []
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
            activation_times = valve_activations[valve_activations == True].index.tolist()  # noqa: E712
            valve_deactivations = ~valve_series & valve_series.shift(1, fill_value=False)
            deactivation_times = valve_deactivations[valve_deactivations == True].index.tolist()  # noqa: E712

            for activation_time in activation_times:
                next_deactivations = [t for t in deactivation_times if t > activation_time]
                deactivation_time = min(next_deactivations) if next_deactivations else valve_series.index[-1]
                all_valve_activations.append({
                    'start_time': activation_time,
                    'end_time': deactivation_time,
                    'odor_name': odor_name,
                    'valve_key': valve_key,
                })
    all_valve_activations.sort(key=lambda x: x['start_time'])
    return all_valve_activations


def valve_events_overlapping(all_valve_activations: list[dict], trial_start, trial_end) -> list[dict]:
    """Activations overlapping ``[trial_start, trial_end]``, inclusive at both edges.

    An activation counts if it starts at or before ``trial_end`` and ends at or after
    ``trial_start``. ``abortion_classification`` uses strict comparisons instead, so an
    activation that closed exactly at the trial start is excluded there but included here.
    """
    events = [
        activation for activation in all_valve_activations
        if activation['start_time'] <= trial_end and activation['end_time'] >= trial_start
    ]
    events.sort(key=lambda x: x['start_time'])
    return events


def last_poke_out_before(poke_data: pd.Series, window_start, window_end):
    """Timestamp of the last cue-port poke-out in ``[window_start, window_end]``.

    Walks the samples carrying the state from before ``window_start``, so a poke that opened
    before the window and closed inside it is found. Returns ``None`` when the port never went
    from IN to OUT in the window -- the caller treats that as a failed response-time
    calculation rather than as a zero.
    """
    extended = poke_data.loc[window_start:window_end]
    if extended.empty:
        return None
    before = poke_data.loc[:window_start]
    prev_state = before.iloc[-1] if len(before) > 0 else False
    last_poke_out_time = None
    for timestamp, current_state in extended.items():
        if prev_state and not current_state:
            last_poke_out_time = timestamp
        prev_state = current_state
    return last_poke_out_time


def first_occurrence_positions(trial_valve_events: list[dict]) -> tuple[dict, list]:
    """Assign positions by **first occurrence of each odor**, ``analyze_response_times``' rule.

    Each new odor name takes the next position number; a repeat of an odor already seen keeps
    its original position and *overwrites* that position's event with the later activation.
    So a trial presenting A, B, A yields two positions, with position 1 holding the second A.

    This is not how ``classify_trials`` assigns positions -- that collapses only *consecutive*
    repeats and keeps a non-consecutive re-entry as a new position. The two therefore disagree
    on any trial where an odor re-appears after a different one, and are kept apart.

    Returns ``(position_locations, ordered_positions)``.
    """
    position_locations: dict[int, dict] = {}
    odor_to_pos: dict[str, int] = {}
    next_pos = 1
    for event in trial_valve_events:
        odor = event['odor_name']
        if odor not in odor_to_pos:
            odor_to_pos[odor] = next_pos
            next_pos += 1
        position_locations[odor_to_pos[odor]] = event
    return position_locations, sorted(position_locations.keys())


# --------------------------------------------------------------------------------------
# Cue-poke periods and the sampling-time accumulator
# --------------------------------------------------------------------------------------

def poke_periods(period_pokes: pd.Series) -> list[tuple]:
    """``(start, end)`` poke periods within an already-sliced boolean series.

    Walks the samples rather than using edge detection, so a poke still open at the end of the
    slice is closed at the slice's **last sample**. A slice that is entirely False yields an
    empty list, which is a different case from an empty slice -- ``detect_trials`` reports the
    two differently, so the slicing stays at the call site.
    """
    poke_periods: list[tuple] = []
    current_start = None
    for timestamp, state in period_pokes.items():
        if state and current_start is None:
            current_start = timestamp
        elif not state and current_start is not None:
            poke_periods.append((current_start, timestamp))
            current_start = None
    if current_start is not None:
        poke_periods.append((current_start, period_pokes.index[-1]))
    return poke_periods


def poke_segments_in_valve_window(poke_periods, cue_pokes, event_start, event_end, fallback_end):
    """Poke intervals clipped to ``[event_start, event_end]``, with the already-in repair.

    If no recorded poke period overlaps the valve window but the cue port was already IN at
    ``event_start``, synthesise one segment from ``event_start`` to the first fall after it
    (searching up to ``fallback_end``), capped at ``event_end``. Without that repair an attempt
    whose poke began before the valve opened would score zero sampling time.
    """
    overlapping_segments = []
    for poke_start, poke_end in poke_periods:
        if poke_end <= event_start:
            continue
        if poke_start >= event_end:
            break
        seg_start = max(poke_start, event_start)
        seg_end = min(poke_end, event_end)
        if seg_end > seg_start:
            overlapping_segments.append((seg_start, seg_end))

    if overlapping_segments:
        return overlapping_segments

    try:
        state_at_start = bool(cue_pokes.loc[:event_start].iloc[-1])
    except (KeyError, IndexError):
        state_at_start = False
    if not state_at_start:
        return overlapping_segments

    after_series = cue_pokes.loc[event_start:fallback_end]
    if not after_series.empty:
        after_bool = after_series.astype(bool)
        falls = (~after_bool) & after_bool.shift(1, fill_value=state_at_start)
        fall_times = list(falls[falls].index)
        if fall_times:
            inferred_end = min(fall_times[0], event_end)
        else:
            inferred_end = min(after_bool.index[-1], event_end)
    else:
        inferred_end = min(event_end, fallback_end)

    if inferred_end > event_start:
        overlapping_segments.append((event_start, inferred_end))
    return overlapping_segments


def accumulate_sampling_time(segments, sample_offset_time_ms, required_minimum_ms, on_segment=None):
    """Sum poke segments until the odor's minimum sampling time is reached, or a long gap ends it.

    Gaps shorter than ``sample_offset_time_ms`` are **counted towards** the total (the animal is
    treated as still sampling); the first gap at or beyond it terminates the attempt. Stops as
    soon as ``required_minimum_ms`` is reached rather than consuming the remaining segments.

    ``on_segment`` receives ``(seg_index, gap_ms_or_None, seg_duration_ms, running_total_ms)``
    for progress reporting. Returns ``(continuous_time_ms, last_segment_end, success)``.
    """
    continuous_time = 0.0
    last_seg_end = None
    success = False

    for seg_idx, (seg_start, seg_end) in enumerate(segments, start=1):
        seg_duration_ms = (seg_end - seg_start).total_seconds() * 1000.0
        if last_seg_end is None:
            continuous_time += seg_duration_ms
            if on_segment is not None:
                on_segment(seg_idx, None, seg_duration_ms, continuous_time)
        else:
            gap_ms = (seg_start - last_seg_end).total_seconds() * 1000.0
            if gap_ms >= sample_offset_time_ms:
                if on_segment is not None:
                    on_segment(seg_idx, gap_ms, None, continuous_time)
                break
            continuous_time += gap_ms + seg_duration_ms
            if on_segment is not None:
                on_segment(seg_idx, gap_ms, seg_duration_ms, continuous_time)
        last_seg_end = seg_end

        if continuous_time >= required_minimum_ms:
            success = True
            break

    return continuous_time, last_seg_end, success
