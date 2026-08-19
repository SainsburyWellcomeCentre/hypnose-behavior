"""Aborted trials and false alarms.

``abortion_classification`` takes the trials ``classify_trials`` marked aborted and works out
where the sequence stopped, which odor was last sampled and whether the abort was a false
alarm. ``classify_noninitiated_FA`` does the equivalent for periods with no initiation at all.

- Bucket the false-alarm latency on the **window-relative** time, never the movement
  time: bucketing the movement latency destroys what ``FA_late`` means. See DECISIONS.md
  section 16.
"""
from __future__ import annotations

from bisect import bisect_right

import numpy as np
import pandas as pd

import hypnose_behavior.trial_classification.detect_settings as detect_settings
import hypnose_behavior.trial_classification.windows as windows
from hypnose_behavior.trial_classification.outcome import latency_label
from hypnose_behavior.trial_classification.params import get_experiment_parameters


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
    """Positions for the abort pipeline, via the shared rule ``windows.positions_by_odor``.

    Consecutive repeats collapse to their last activation, then one position per odor with a
    later activation overwriting it, capped at ``max_positions``. Identical to numbering the
    collapsed list ``1..n`` -- which is what this did before -- on every trial whose odors are
    distinct, i.e. all but the experiment-faulted ones.

    It must be the same rule ``classify_trials`` uses, not merely a similar one: this function
    produces ``last_odor_position``, and ``frames.sequence_depths`` falls back to it when a
    session carries no ``poke_source``. The two agree on 486 of 486 aborted trials, and that
    agreement is only meaningful while both number positions the same way.

    Returns ``(events, positions)`` as parallel lists, positions numbered from 1.
    """
    evs = windows.collapse_consecutive_odors(evs_raw)
    position_locations, _repeated = windows.positions_by_odor(evs, max_positions=max_positions)
    positions = sorted(position_locations)
    return [position_locations[p] for p in positions], positions


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
    fa_window_latency_ms = (fa_time - abortion_time).total_seconds() * 1000.0
    fa_port = 1 if fa_time in dip1_rises else (2 if fa_time in dip2_rises else None)
    anchor = windows.last_poke_end_before(cue_series, fa_time)
    movement_ms = float((fa_time - anchor).total_seconds() * 1000.0) if anchor is not None else np.nan
    return (latency_label(fa_window_latency_ms, response_time_ms, 'FA'), fa_time, fa_window_latency_ms,
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
            aborted_detailed.loc[aborted_detailed['fa_label'] == label, 'fa_window_latency_ms'],
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
        fa_label, fa_time, fa_window_latency_ms, fa_port, fa_movement_ms = _false_alarm(
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
            'fa_window_latency_ms': float(fa_window_latency_ms) if pd.notna(fa_window_latency_ms) else np.nan,
            'fa_port': fa_port,
            'fa_response_time_ms': fa_movement_ms,
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
        fa_window_latency_ms = np.nan
        fa_port = None
        fa_movement_ms = np.nan

        reward_after = [t for t in reward_rises if attempt_end < t <= next_cue_in]
        if reward_after:
            fa_time = reward_after[0]
            fa_window_latency_ms = (fa_time - attempt_end).total_seconds() * 1000.0
            
            # Determine which port ← NEW
            if fa_time in dip1_rises:
                fa_port = 1
            elif fa_time in dip2_rises:
                fa_port = 2

            fa_label = latency_label(fa_window_latency_ms, response_time_ms, 'FA')

        # HR status for position 1
        is_hr = False
        if hr_odors is not None:
            odor_name = row.get('odor_name')
            is_hr = odor_name in hr_odors

        results.append({
            **row.to_dict(),
            'fa_label': fa_label,
            'fa_time': fa_time,
            'fa_window_latency_ms': fa_window_latency_ms,
            'fa_port': fa_port,
            'fa_response_time_ms': fa_movement_ms,
            'is_hr': is_hr
        })
        
    return pd.DataFrame(results)
