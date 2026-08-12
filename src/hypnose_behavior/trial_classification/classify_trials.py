"""Trial classification -- positions, presentations and the outcome of every detected trial.

``classify_trials`` is the pipeline's principal writer: it produces the
``completed_sequence_*`` frames and every per-position blob in ``trial_data``
(``position_valve_times``, ``position_poke_times``, ``presentations``), each entry carrying
its ``poke_source`` (``DECISIONS.md`` section 10).

One position rule
-----------------
Positions come from ``windows.positions_by_odor``, the single rule shared with
``analyze_response_times``: **one position per odor, and a later activation overwrites the
position that odor already holds**. A trial presenting A, B, A is therefore **2 positions**,
position 1 holding the *second* A.

This module and ``analyze_response_times`` once resolved positions differently -- the rule here
opened a *new* position for a non-consecutive re-entry, giving 3 positions for A, B, A. The two
were measured before being merged (``DECISIONS.md`` section 13's requirement): they disagree
only when an odor re-appears after a different odor, which happens on **1 of 1,731** fixture
trials and **0 of 46,112** trials across subjects 056-066.

That single trial is an experiment-side fault, not a longer sequence: the rig failed to emit an
``InitiationSequence`` between three sampling runs, so ``F,A / F,A / F,A`` was recorded as one
trial. Overwriting resolves it to the **last** run, which is the run the trial's outcome events
belong to. ``_assign_positions_to_valve_events`` returns the repeated odors and the caller
raises a ``RuntimeWarning``: silent on sound data, and meaningful precisely because of that.
"""
from __future__ import annotations

import warnings
from collections import defaultdict

import numpy as np
import pandas as pd

# The saved schema's declaration of the protocol modes. `io/protocol_schema.py` is a leaf
# (standard library only) and both package `__init__`s are docstring-only, so this is a
# one-way edge and not the cycle it looks like -- `docs/DECISIONS.md` section 3.
from hypnose_behavior.io.protocol_schema import resolve_mode
import hypnose_behavior.trial_classification.windows as windows
from hypnose_behavior.trial_classification.hidden_rule import (
    _check_hidden_rule, _drop_final_hidden_rule_index,
    _hidden_rule_indices_from_stage_or_schema, _hidden_rule_odor_set,
    _hidden_rule_positions, _hidden_rule_success, _print_hidden_rule_header,
)
from hypnose_behavior.trial_classification.outcome import classify_completed_trial, latency_label
from hypnose_behavior.trial_classification.params import (
    _get_single_reward_info, _sampling_parameters_ms,
)
from hypnose_behavior.trial_classification.windows import (
    _next_after, _odourdisc_reward_window_end, _recording_end,
)


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


def _assign_positions_to_valve_events(trial_valve_events, max_positions, required_min_ms_for):
    """Map a trial's valve activations onto sequence positions 1..max_positions.

    Position 1 is the **last** activation of the opening odor: the animal may sniff and leave
    several times before committing, and the trial starts at the final one. The earlier
    activations of that same valve become ``prior_presentations`` -- failed Position-1 attempts
    that are reported separately as non-initiated.

    Later positions come from ``windows.positions_by_odor`` -- the single position rule, shared
    with ``analyze_response_times``. Consecutive repeats are collapsed to their last activation
    first; a *non-consecutive* re-entry is a sequence **restart**, so it overwrites the position
    that odor already holds rather than opening a new one. The repeated odors are returned so
    the caller can warn.

    Returns ``(position_locations, prior_presentations, repeated_odors)``.
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
    position_locations, repeated_odors = windows.positions_by_odor(
        dedup_events[1:], seed=position_locations, max_positions=max_positions)

    return position_locations, prior_presentations, repeated_odors


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
    """Cue-port poke record per position, measured inside that position's valve window.

    Reports the **first merged block** of poking: intervals separated by less than
    ``sampleOffsetTime`` are one sample, and the first gap beyond it ends the measurement, so a
    later return to the port during the same odor does not inflate the sampling time.

    **Every position with a valve activation gets an entry**, including the ones the animal
    never poked: the odor was presented, and dropping it truncated ``odor_sequence``,
    ``num_odors`` and ``presentations`` downstream. A position with no poke is written with
    ``poke_time_ms = 0.0`` and null poke timestamps rather than omitted, and every entry
    records where its poke time came from in ``poke_source``:

    - ``"poke"``          -- a genuine poke inside the odor window;
    - ``"grace"``         -- no poke in the window, but the animal's last poke-out landed
      within ``PRE_ODOR_GRACE_MS`` of the valve opening, so the position is credited and
      anchored at the window start;
    - ``"outside_grace"`` -- no poke in the window and no grace credit: the valve opened and
      the animal was out of the port for all of it.

    ``poke_source`` is the only reliable separator. Animals genuinely poke for under 20 ms, so
    a grace entry cannot be told from a real short poke by duration, and the tell
    ``poke_first_in == poke_odor_start`` is also satisfied by a real poke already in progress
    when the valve opened. Consumers must treat an **absent** ``poke_source`` as "unknown" and
    omit the filtered variant, never as "all real pokes" -- sessions saved before this was
    written will never carry it (``DECISIONS.md`` section 2).

    Which of these positions count towards the trial's sequence is deliberately **not** decided
    here; see ``_trim_unsampled_tail``.
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
            'poke_source': 'grace',
        }

    def _unsampled_entry(position, loc):
        """The odor was presented and the animal was not at the port for any of it.

        Timestamps stay null on purpose: there is no poke to time, and a valve-window edge
        written here would be indistinguishable from a measured one downstream.
        """
        return {
            'position': position,
            'odor_name': loc['odor_name'],
            'poke_time_ms': 0.0,
            'poke_odor_start': None,
            'poke_odor_end': None,
            'poke_first_in': None,
            'required_min_sampling_time_ms': required_min_ms_for(loc['odor_name']),
            'poke_source': 'outside_grace',
        }

    for position in range(1, max_positions + 1):
        loc = position_locations.get(position)
        if loc is None:
            continue
        odor_start = loc['start_time']
        odor_end = loc['end_time']

        intervals, _first_in = windows.poke_intervals_in_window(s_bool, odor_start, odor_end)
        if not intervals:
            # Grace is attempted only when the window holds no poke at all, which is the rule
            # it has always had -- a zero-length block inside the window does not fall back
            # to it.
            position_poke_times[position] = (
                _grace_entry(position, loc, odor_start, odor_end)
                or _unsampled_entry(position, loc)
            )
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
                'poke_source': 'poke',
            }
        else:
            position_poke_times[position] = _unsampled_entry(position, loc)

    return position_poke_times


def _trim_unsampled_tail(positions, position_poke_times):
    """The leading run of positions the trial is credited with. Interior gaps stay.

    Nothing is deleted: the caller keeps every presented position in ``position_poke_times``
    and ``presentations`` and uses this only to decide how much of that counts as the
    ``odor_sequence`` -- so the returned list is always a **prefix** of ``positions``.

    Called only for a trial that never reached AwaitReward. There the last valve activation is
    the odor the rig opened as the animal was leaving: it was presented, but the sequence did
    not advance through it, and counting it would put an odor the animal never smelled at the
    end of ``odor_sequence`` and make it ``last_odor``. ``abortion_classification`` agrees --
    measured on the 9 regression sessions its ``last_odor_position``, derived by a wholly
    independent pipeline, is the last **poked** position on 74 of 74.

    An interior gap is the opposite case: the rig demonstrably opened a later valve, so the
    sequence *did* move past that position and the odor belongs in the sequence. It stays, and
    its ``poke_source`` is what keeps a 0 ms entry out of the poke-duration averages.

    The test is ``poke_source != 'poke'``, so a trailing **grace** entry is trimmed too. That
    is the ``presentations``-vs-``last_odor_position`` disagreement section 10 measured at 10
    of 1731 trials, resolved in favour of the abort pipeline.

    On a completed trial nothing is trimmed: reaching AwaitReward means the rig counted every
    position it opened, including a final one our reconstruction from DIPort0 scores as
    unpoked. ``DECISIONS.md`` section 10.
    """
    trimmed = list(positions)
    while trimmed and (position_poke_times.get(trimmed[-1]) or {}).get('poke_source') != 'poke':
        trimmed.pop()
    return trimmed


def _odourdisc_await_window(trial, *, trial_start, trial_end, valve_activations,
                            await_reward_times, initiation_starts_sorted, cue_poke_starts_sorted,
                            supply_port1_times, supply_port2_times, port1_pokes, port2_pokes):
    """Where to look for this odour-discrimination trial's AwaitReward, and what is in there.

    The task fires AwaitReward once the animal commits, which can be *after* the detected trial
    window ends, so the search runs from the initiation to the next initiation rather than to
    ``trial_end``. That makes it a wider window than the plain ``trial_start <= t <= trial_end``
    test the standard protocol uses, and the two disagree.

    Computed before the sequence is assembled because ``_trim_unsampled_tail`` needs to know
    whether the trial completed, and returned whole so the scoring branch reuses these values
    rather than recomputing them under a second definition.

    An empty ``await_in_window`` means the trial aborted.
    """
    last_valve_event = valve_activations[-1] if valve_activations else None
    last_valve_start = (last_valve_event or {}).get('start_time')

    current_init_ts = pd.to_datetime(trial.get('initiation_sequence_time'), errors='coerce') \
        if trial.get('initiation_sequence_time') is not None else pd.NaT
    await_window_start = current_init_ts if not pd.isna(current_init_ts) else \
        (trial_start if trial_start is not None else last_valve_start)

    ctx = {
        'last_valve_start': last_valve_start,
        'await_window_start': await_window_start,
        'next_init': None,
        'recording_end': None,
        'await_in_window': [],
    }
    if await_window_start is None or pd.isna(await_window_start):
        return ctx

    next_init = None
    if not initiation_starts_sorted.empty and not pd.isna(current_init_ts):
        idx = initiation_starts_sorted.searchsorted(current_init_ts, side='right')
        if idx < len(initiation_starts_sorted):
            next_init = initiation_starts_sorted.iloc[idx]

    recording_end = _recording_end(initiation_starts_sorted, cue_poke_starts_sorted,
                                   supply_port1_times, supply_port2_times,
                                   port1_pokes, port2_pokes, trial_end)

    await_upper_bound = next_init if next_init is not None else recording_end
    ctx['next_init'] = next_init
    ctx['recording_end'] = recording_end
    ctx['await_in_window'] = [t for t in await_reward_times
                              if await_window_start <= t <= await_upper_bound]
    return ctx


def _build_presentations(presented_positions, position_valve_times, position_poke_times,
                         *, sampled_count=None):
    """One row per **presented** position, in presentation order.

    Every position whose valve opened gets a row, including a trailing one the animal never
    poked: the record of what was presented stays complete, and ``poke_source`` on each row
    says whether it was sampled.

    ``last_event_index`` instead marks where the *counted* sequence ends -- on an aborted trial
    the last real poke, with the unsampled trailing rows sitting after it. ``sampled_count`` is
    the length of that leading run (see ``_trim_unsampled_tail``); it defaults to every row,
    which is the completed-trial case.

    Returns ``(rows, last_event_index)``.
    """
    if sampled_count is None:
        sampled_count = len(presented_positions)
    last_event_index = sampled_count - 1 if sampled_count else None

    presentations = []
    for idx_in_trial, pos in enumerate(presented_positions):
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
            # Carried so a consumer reading `presentations` can apply the same
            # `poke_source` filter as one reading `position_poke_times`.
            'poke_source': poke_info.get('poke_source'),
            'required_min_sampling_time_ms': valve_info.get('required_min_sampling_time_ms'),
            'is_last_event': last_event_index is not None and idx_in_trial == last_event_index,
        })
    return presentations, last_event_index


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
        trial_dict['fr_window_latency_ms'] = np.nan
        trial_dict['fr_response_time_ms'] = np.nan
        return

    fr_time, fr_port, fr_odor = tagged_pokes[0]
    fr_window_latency_ms = (fr_time - await_reward_time).total_seconds() * 1000.0
    trial_dict['false_response'] = True
    trial_dict['fr_time'] = fr_time
    trial_dict['fr_port'] = fr_port
    trial_dict['fr_odor_identity'] = fr_odor
    trial_dict['fr_window_latency_ms'] = float(fr_window_latency_ms)
    # (b) how fast it travelled once it finally left the cue port. fr_window_latency_ms above is (a),
    # time since the sequence completed, and it is what fr_label buckets. DECISIONS section 16.
    _fr_anchor = windows.last_poke_end_before(cue_series, fr_time)
    trial_dict['fr_response_time_ms'] = (
        float((fr_time - _fr_anchor).total_seconds() * 1000.0) if _fr_anchor is not None else np.nan)
    # Parity with unrewarded rows so downstream poke-based logic stays consistent.
    trial_dict['first_reward_poke_time'] = fr_time
    trial_dict['first_reward_poke_port'] = fr_port
    trial_dict['first_reward_poke_odor_identity'] = fr_odor
    trial_dict['fr_label'] = latency_label(fr_window_latency_ms, response_time_ms_window, 'FR')


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
    # Which of the three protocol modes this run follows. The two flags come from
    # independent sources and nothing in the code makes them exclusive, so the impossible
    # combination raises here rather than writing a file whose schema is undefined -- see
    # `io/protocol_schema.resolve_mode`. Called for the check alone; the mode it returns is what
    # decides the record's column set, and is threaded to the manifest in the next step.
    resolve_mode(is_odour_discrimination=is_odour_discrimination,
                 is_single_reward=is_single_reward)
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
        position_locations, prior_presentations, repeated_odors = _assign_positions_to_valve_events(
            valve_activations, max_positions, required_min_ms_for)
        if repeated_odors:
            # Silent on sound data (0 of 46,112 trials, subjects 056-066), so reaching this is
            # meaningful: the rig merged several sampling runs into one trial. Raised as a
            # warning rather than printed, so it surfaces regardless of `verbose` and
            # `print_summary`.
            warnings.warn(
                f"trial {trial.get('trial_id')}: odor(s) {sorted(set(repeated_odors))} were "
                f"presented again after a different odor, between {trial_start} and {trial_end}. "
                "Several sampling runs were recorded as one trial; the sequence has been "
                "resolved to the LAST run. Check this session's initiation events.",
                RuntimeWarning, stacklevel=2,
            )
        position_valve_times = _position_valve_times(
            position_locations, max_positions, prior_presentations, required_min_ms_for)
        position_poke_times = _position_poke_times(
            position_locations, poke_data, max_positions, sample_offset_time_ms, required_min_ms_for)

        trial_await_rewards = [t for t in await_reward_times if trial_start <= t <= trial_end]

        # Whether the rig reached AwaitReward decides how much of the valve sequence counts,
        # so it has to be known before the sequence is assembled. The two protocols search
        # different windows for it -- see `_odourdisc_await_window`.
        odourdisc_ctx = None
        if is_odour_discrimination:
            odourdisc_ctx = _odourdisc_await_window(
                trial, trial_start=trial_start, trial_end=trial_end,
                valve_activations=valve_activations, await_reward_times=await_reward_times,
                initiation_starts_sorted=initiation_starts_sorted,
                cue_poke_starts_sorted=cue_poke_starts_sorted,
                supply_port1_times=supply_port1_times, supply_port2_times=supply_port2_times,
                port1_pokes=port1_pokes, port2_pokes=port2_pokes)
            has_await_reward = bool(odourdisc_ctx['await_in_window'])
        else:
            has_await_reward = bool(trial_await_rewards)

        # Every presented position counts on a completed trial: reaching AwaitReward means the
        # rig advanced through all of them. On an aborted trial the trailing valve activation
        # is the odor the animal walked away from, so the *sequence* stops at the last real
        # poke -- but the position is still recorded in `position_poke_times` and
        # `presentations`, marked `outside_grace`, so the record of what was presented stays
        # complete and only what the trial is credited with shrinks.
        presented_positions = sorted(position_poke_times.keys())
        valid_positions = (presented_positions if has_await_reward
                           else _trim_unsampled_tail(presented_positions, position_poke_times))

        presentations, last_event_index = _build_presentations(
            presented_positions, position_valve_times, position_poke_times,
            sampled_count=len(valid_positions))

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
            trial_dict['odourdiscrimination_mode'] = True
            trial_dict['last_valve_start'] = odourdisc_ctx['last_valve_start']

            await_window_start = odourdisc_ctx['await_window_start']
            if await_window_start is None or pd.isna(await_window_start):
                aborted_sequences.append(trial_dict.copy())
                initiated_trials_list.append(trial_dict)
                continue

            next_init = odourdisc_ctx['next_init']
            recording_end = odourdisc_ctx['recording_end']

            await_in_window = odourdisc_ctx['await_in_window']
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
            # Real pokes only: a grace entry's duration is synthesised and an `outside_grace`
            # one is 0 ms, and averaging either into a measured sampling time understates it.
            if p and 'poke_time_ms' in p and p.get('poke_source', 'poke') == 'poke':
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
