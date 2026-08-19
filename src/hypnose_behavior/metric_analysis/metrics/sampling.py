# Defers evaluation of PEP-604 annotations (`X | None`), keeping this module
# importable on Python 3.9 for repos pinned there (hypnose-eeg-preprocessing).
from __future__ import annotations

"""How long the animal sampled: poke durations, and what it did with the valve.

``manual_vs_auto_stop_preference`` is here rather than in a ``valve.py``
(confirmed 2026-08-05): splitting valve durations at 1000 ms measures whether the
animal withdrew before the valve closed, which is a sampling decision. It is the
one metric reading ``position_valve_times``, a *superset* of the other two blobs,
so it is also the one that would gain rows if the provenance filter were dropped.

**Summation style is part of the metric.** The two pooled ``avg_sampling_time_*``
metrics accumulate ``total += x`` left to right; ``avg_sampling_time_odor_x``
calls ``np.mean`` on a per-odor list, which sums pairwise. The two disagree in
the last ULP over a few hundred values -- enough to move the metrics md5, and
invisible in any printed output. ``_sequential_mean`` exists to reproduce the
first. Do not tidy either into ``Series.mean()``, and the same goes for
``_mean_sd_by``, which resolved the one reduction the poke-duration metrics had
to pick (settled 2026-08-06).

``avg_sampling_time_aborted_sequence`` excludes the abort event itself -- the
entry whose ``index_in_trial`` equals the trial's ``last_event_index``.
``presentations`` also carries an ``is_last_event`` flag which agrees with that
rule on all 9 fixture sessions, but it is a *different* rule, and 4a reproduces
today's values rather than a rule that happens to match.
"""

import numpy as np
import pandas as pd

from hypnose_behavior.metric_analysis.metrics.common import (
    _aborted_mask,
    _position_rows,
    _trial_position_frame,
    _tz_naive,
)
from hypnose_behavior.metric_analysis.registry import (
    as_dict,
    metric,
    session_metric,
)

__all__ = [
    "avg_sampling_time_odor_x", "avg_sampling_time_odor_x_session",
    "avg_sampling_time_completed_sequence", "avg_sampling_time_completed_sequence_session",
    "avg_sampling_time_aborted_sequence", "avg_sampling_time_aborted_sequence_session",
    "manual_vs_auto_stop_preference", "manual_vs_auto_stop_preference_session",
    "poke_durations", "poke_duration_by_position", "poke_duration_by_odor",
    "trial_poke_span", "trial_poke_total",
]


def _real_pokes(rows):
    """Keep only the positions the animal actually poked, when the session says which.

    `poke_source` marks every position entry `poke` / `grace` / `outside_grace`. A duration is
    a *measurement* only on the first: a grace entry's is synthesised from
    `PRE_ODOR_GRACE_MS`, and an `outside_grace` one is 0 ms because the valve opened while the
    animal was away from the port. Averaging either into a sampling time understates it, which
    is exactly the artefact writing the unpoked positions would otherwise introduce.

    Sessions saved before the marker existed carry no `poke_source` at all. There the honest
    answer is "unknown", so the rows are returned untouched and the metric keeps the value it
    has always had, rather than being filtered on a column that is not there
    (`DECISIONS.md` section 2).
    """
    if "poke_source" not in rows.columns:
        return rows
    source = rows["poke_source"]
    if source.isna().all():
        return rows
    return rows[source == "poke"]


def _sequential_mean(values):
    """Mean by left-to-right accumulation.

    Matches the `total = 0.0; total += x` loops this replaces. `np.mean` and
    `Series.mean` sum pairwise and differ in the last ULP over a few hundred
    values, which moves the metrics fingerprint.
    """
    total = 0.0
    n = 0
    for v in values:
        total += v
        n += 1
    return total / n if n > 0 else np.nan


@metric(frame="position_data", title="Average Sampling Time per Odor (Completed)",
        adapter=as_dict)
def avg_sampling_time_odor_x(position_data):
    """Mean `poke_time_ms` per odor over completed trials, from `position_poke_times`."""
    rows = _position_rows(position_data, "in_poke_times", aborted=False)
    if rows is None or rows.empty:
        return pd.Series(dtype=float)
    rows = _real_pokes(rows)
    rows = rows[rows["odor_name"].notna() & rows["poke_time_ms"].notna()]
    if rows.empty:
        return pd.Series(dtype=float)
    # `np.mean` on each group's values rather than `Series.mean()`: this
    # replaces a per-odor Python list passed to `np.mean`, and pandas' reduction
    # kernel can land a ULP away. `rename`s keep the unnamed Series shape the
    # single-dict construction produced.
    avg_times = (rows.groupby("odor_name")["poke_time_ms"]
                 .apply(lambda s: np.mean(s.to_numpy()))
                 .rename(None).rename_axis(None).sort_index())
    return avg_times


@session_metric(avg_sampling_time_odor_x)
def avg_sampling_time_odor_x_session(results):
    avg_times = avg_sampling_time_odor_x(results.get("position_data"))
    for odor, avg_time in avg_times.items():
        print(f"{odor} Average Sampling Time: {avg_time:.2f} ms")
    return avg_times


@metric(frame="position_data", title="Average Sampling Time (Completed Sequences)")
def avg_sampling_time_completed_sequence(position_data):
    """Pooled mean `poke_time_ms` over completed trials' `position_poke_times`."""
    rows = _position_rows(position_data, "in_poke_times", aborted=False)
    if rows is None or rows.empty:
        return np.nan
    rows = _real_pokes(rows)
    return _sequential_mean(rows.loc[rows["poke_time_ms"].notna(), "poke_time_ms"])


@session_metric(avg_sampling_time_completed_sequence)
def avg_sampling_time_completed_sequence_session(results):
    if results.get("trial_data", pd.DataFrame()).empty:
        return np.nan
    avg = avg_sampling_time_completed_sequence(results.get("position_data"))
    print(f"Average Sampling Time (Completed Sequences): {avg:.2f} ms")
    return avg


@metric(frame="position_data", title="Average Sampling Time (Aborted Sequences)")
def avg_sampling_time_aborted_sequence(position_data):
    """Pooled mean `poke_time_ms` over aborted trials' `presentations`.

    Excludes the abort event itself -- the entry whose `index_in_trial` equals
    the trial's `last_event_index`. A null `last_event_index` matches nothing, so
    that trial contributes every entry, as it does today.
    """
    rows = _position_rows(position_data, "in_presentations", aborted=True)
    if rows is None or rows.empty:
        return np.nan
    rows = _real_pokes(rows)
    idx = rows["index_in_trial"]
    keep = (idx.notna() & (idx != rows["last_event_index"])
            & rows["poke_time_ms"].notna())
    return _sequential_mean(rows.loc[keep, "poke_time_ms"])


@session_metric(avg_sampling_time_aborted_sequence)
def avg_sampling_time_aborted_sequence_session(results):
    # Silent on an empty trial table and on a session with no aborted trials --
    # both bail before the print today, where a session that aborted but
    # recorded no usable presentation still prints "nan ms".
    trials = results.get("trial_data", pd.DataFrame())
    if trials.empty or not _aborted_mask(trials).any():
        return np.nan
    avg = avg_sampling_time_aborted_sequence(results.get("position_data"))
    print(f"Average Sampling Time (Aborted Sequences): {avg:.2f} ms")
    return avg


@metric(frame="position_data", title="Manual vs Auto Stop Preference")
def manual_vs_auto_stop_preference(position_data):
    """Valve durations on completed trials, split at 1000 ms.

    Reads `position_valve_times` only. That blob is a *superset* of the other
    two -- it records positions whose poke registered as ~0 ms -- so this is the
    one metric that would gain rows if the provenance filter were dropped.
    """
    rows = _position_rows(position_data, "in_valve_times", aborted=False)
    if rows is None or rows.empty:
        return {"short_valve": 0, "long_valve": 0, "ratio": np.nan}
    dur = rows.loc[rows["valve_duration_ms"].notna(), "valve_duration_ms"]
    # `if dur <= 1000 ... elif dur >= 1000`: exactly 1000 ms counts short only.
    short = int((dur <= 1000).sum())
    long = int((dur > 1000).sum())
    return {"short_valve": short, "long_valve": long,
            "ratio": short / long if long > 0 else float('nan')}


@session_metric(manual_vs_auto_stop_preference)
def manual_vs_auto_stop_preference_session(results):
    if results.get("trial_data", pd.DataFrame()).empty:
        return {"short_valve": 0, "long_valve": 0, "ratio": np.nan}
    out = manual_vs_auto_stop_preference(results.get("position_data"))
    print(f"Manual Stops: {out['short_valve']}")
    print(f"Auto Stops: {out['long_valve']}")
    print(f"Manual vs Auto Stop: {out['ratio']:.2f}")
    return out


@metric(frame="position_data")
def poke_durations(position_data, *, aborted=False):
    """Per-position poke durations for one outcome class, as a tidy frame.

    Completed trials come from `position_poke_times`; aborted trials from
    `presentations` with the abort event excluded -- the same sources, and the
    same exclusion, the canonical `avg_sampling_time_*` metrics use.

    - **Filter on `poke_source`, never on `poke_time_ms > 0`.** A bare `> 0` test still
      averages in the grace entries, whose durations are synthesised rather than
      measured. `_real_pokes` excludes both, and leaves a pre-marker session untouched.
    - **Carries `global_trial_id`**, so a poke joins back to its trial. Emitted via
      `reindex` so the column set is the same four whether or not the frame carries the
      id, rather than a function of what this session happened to have.
    """
    empty = pd.DataFrame(columns=["global_trial_id", "position", "odor_name", "poke_time_ms"])
    if aborted:
        rows = _position_rows(position_data, "in_presentations", aborted=True)
        if rows is None or rows.empty:
            return empty
        idx = rows["index_in_trial"]
        rows = rows[idx.notna() & (idx != rows["last_event_index"])]
    else:
        rows = _position_rows(position_data, "in_poke_times", aborted=False)
        if rows is None or rows.empty:
            return empty
    rows = _real_pokes(rows)
    rows = rows[rows["poke_time_ms"].notna()]
    if rows.empty:
        return empty
    return rows.reindex(
        columns=["global_trial_id", "position", "odor_name", "poke_time_ms"]
    ).reset_index(drop=True)


def _mean_sd_by(frame, key):
    """Mean, population SD and count of `poke_time_ms` per `key`.

    **`np.mean` / `np.std` on each group's array, never the pandas reductions.** Both
    give the population SD, but they sum in a different order and disagree in the last
    ULP, which moves drawn values in `plot_sampling_times_analysis`. Summation style is
    part of the metric -- do not tidy this. See DECISIONS.md section 1.
    """
    if frame.empty:
        return pd.DataFrame(columns=["mean", "sd", "n"])
    grouped = frame.dropna(subset=[key]).groupby(key, sort=True)["poke_time_ms"]
    stats = {}
    for name, values in grouped:
        arr = values.to_numpy(dtype=float)
        stats[name] = (float(np.mean(arr)), float(np.std(arr)), int(arr.size))
    out = pd.DataFrame.from_dict(stats, orient="index", columns=["mean", "sd", "n"])
    out.index.name = key
    return out


@metric(frame="position_data")
def poke_duration_by_position(position_data, *, aborted=False):
    """Mean and population SD of `poke_time_ms` per position."""
    return _mean_sd_by(poke_durations(position_data, aborted=aborted), "position")


@metric(frame="position_data")
def poke_duration_by_odor(position_data, *, aborted=False):
    """Mean and population SD of `poke_time_ms` per odor.

    In its `aborted=True` form the canonical
    `avg_sampling_time_aborted_sequence` pools every aborted trial into one
    scalar, and no per-odor version existed. With `aborted=False` it is the
    per-odor completed-trial mean, i.e. `avg_sampling_time_odor_x` with an SD
    and a count alongside.
    """
    return _mean_sd_by(poke_durations(position_data, aborted=aborted), "odor_name")


@metric(frame="position_data")
def trial_poke_span(position_data):
    """Wall-clock span of a trial's odor-sampling phase, in ms.

    `poke_odor_end` at the deepest position minus `poke_odor_start` at position 1.
    Distinct from `trial_poke_total`: the span contains the travel between ports,
    the sum does not. Trials missing either timestamp are dropped.

    Real pokes only. An `outside_grace` entry carries null timestamps, so leaving
    it in would make the deepest position's `poke_odor_end` null and drop the
    whole trial; a grace entry's `poke_odor_end` is synthetic and sits up to
    `PRE_ODOR_GRACE_MS` after the animal actually left (`DECISIONS.md` section 15).
    """
    rows = _trial_position_frame(position_data, "in_poke_times")
    if rows is None:
        return pd.Series(dtype=float)
    rows = _real_pokes(rows)
    if rows.empty:
        return pd.Series(dtype=float)
    frame = pd.DataFrame({
        "gid": rows["global_trial_id"].to_numpy(),
        # Only position 1 contributes a start, so `max` picks it out.
        "start": _tz_naive(rows["poke_odor_start"]).where(rows["position"] == 1).to_numpy(),
        "end": _tz_naive(rows["poke_odor_end"]).to_numpy(),
    })
    grouped = frame.groupby("gid", sort=True)
    span = grouped["end"].agg(lambda s: s.iloc[-1]) - grouped["start"].max()
    return span.dropna().dt.total_seconds() * 1000.0


@metric(frame="position_data")
def trial_poke_total(position_data):
    """Sum of `poke_time_ms` across a trial's positions, in ms.

    Related to `avg_sampling_time_completed_sequence` but per trial rather than a
    session mean.
    """
    rows = _position_rows(position_data, "in_poke_times")
    if rows is None or rows.empty or "global_trial_id" not in rows.columns:
        return pd.Series(dtype=float)
    usable = _real_pokes(rows)
    usable = usable[usable["poke_time_ms"].notna()]
    if usable.empty:
        return pd.Series(dtype=float)
    return usable.groupby("global_trial_id")["poke_time_ms"].sum()
