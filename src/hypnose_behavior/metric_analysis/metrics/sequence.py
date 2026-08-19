# Defers evaluation of PEP-604 annotations (`X | None`), keeping this module
# importable on Python 3.9 for repos pinned there (hypnose-eeg-preprocessing).
from __future__ import annotations

"""How far a sequence got, and where it was abandoned.

``presentation_counts_by_odor`` lives here because it is the denominator of
``odorx_abortion_rate``; ``hidden_rule.py`` imports it for the same count
restricted to the hidden-rule odors.

The per-position denominators are ``frames.reached_counts`` over
``frames.sequence_depths`` -- the package's single definition of "reached". They read
``position_data``, which is why these cores take a second frame. See DECISIONS.md
sections 10 and 18.
"""

import numpy as np
import pandas as pd

from hypnose_behavior.frames import reached_counts as _reached_counts
from hypnose_behavior.metric_analysis.metrics.common import (
    _aborted_mask,
    _initiated,
    _position_rows,
    _reduce_rate,
)
from hypnose_behavior.metric_analysis.registry import (
    as_dict,
    metric,
    session_metric,
)

__all__ = [
    "sequence_completion_rate_contributions", "sequence_completion_rate",
    "sequence_completion_rate_session",
    "presentation_counts_by_odor",
    "odorx_abortion_rate", "odorx_abortion_rate_session",
    "abortion_rate_positionX", "abortion_rate_positionX_session",
    "odor_initiation_bias", "odor_initiation_bias_session",
]


def sequence_completion_rate_contributions(trials):
    return ((~_aborted_mask(trials)).astype(int), _initiated(trials))


@metric(frame="trials", title="Sequence Completion Rate")
def sequence_completion_rate(trials):
    """completed / initiated."""
    if trials.empty:
        return 0, 0, np.nan
    return _reduce_rate(*sequence_completion_rate_contributions(trials))


@session_metric(sequence_completion_rate)
def sequence_completion_rate_session(results):
    df = results.get("trial_data", pd.DataFrame())
    if df.empty:
        print("Sequence Completion Rate: no trial_data")
        return 0, 0, np.nan
    n_completed, denom, rate = sequence_completion_rate(df)
    print(f"Sequence Completion Rate: {n_completed}/{denom} = {rate:.3f}")
    return n_completed, denom, rate


@metric(frame="position_data")
def presentation_counts_by_odor(position_data):
    """`{odor_name: n presentations}` -- the denominator of `odorx_abortion_rate`.

    Counts `in_presentations` rows, including a trailing position the animal never poked:
    the valve opened, so the odor *was* presented.

    **Do not filter this on `poke_source`.** The numerator is `last_odor_name`, the last
    odor actually sampled, so such a position contributes a presentation and no abortion.
    `presentations` answers "what did the rig deliver", `poke_source` answers "what did
    the animal sample". See DECISIONS.md section 10.
    """
    rows = _position_rows(position_data, "in_presentations")
    if rows is None or rows.empty:
        return {}
    rows = rows[rows["odor_name"].notna()]
    return {od: int(n) for od, n in rows.groupby("odor_name").size().items()}


@metric(frame="trials+position_data", title="Odor Abortion Rate", adapter=as_dict)
def odorx_abortion_rate(trials, position_data, *, with_counts=False):
    """aborts@odor / presentations@odor."""
    empty = ({}, {}, {}) if with_counts else pd.Series(dtype=float)
    # Deliberately not guarded on a `presentations` column: this reads its denominator
    # from `position_data`, and a guard on a column the function does not use turns a
    # reported metric silently empty the day that column goes.
    if trials.empty:
        return empty
    odor_col = "last_odor_name" if "last_odor_name" in trials.columns else "last_odor"
    if odor_col not in trials.columns:
        return empty

    aborted = trials[_aborted_mask(trials)]
    abortions = aborted[odor_col].dropna().value_counts().to_dict()
    presentations = presentation_counts_by_odor(position_data)

    all_odors = set(presentations.keys()).union(abortions.keys())
    rates = {}
    for od in sorted(all_odors):
        n_pres = presentations.get(od, 0)
        rates[od] = abortions.get(od, 0) / n_pres if n_pres > 0 else np.nan
    if with_counts:
        return rates, abortions, presentations
    return pd.Series(rates, dtype=float).sort_index()


@session_metric(odorx_abortion_rate)
def odorx_abortion_rate_session(results):
    parts = odorx_abortion_rate(results.get("trial_data", pd.DataFrame()),
                                results.get("position_data"), with_counts=True)
    if not isinstance(parts, tuple):
        return parts
    rates, abortions, presentations = parts
    for od in sorted(rates):
        print(f"{od}: {abortions.get(od, 0)}/{presentations.get(od, 0)} abortions, "
              f"Rate: {rates[od]:.3f}")
    return pd.Series(rates, dtype=float).sort_index()


@metric(frame="trials+position_data", title="Abortion Rate by Position", adapter=as_dict)
def abortion_rate_positionX(trials, position_data, *, with_counts=False):
    """aborts@position / trials that reached it.

    The denominator is `frames.reached_counts`, the package's single definition of
    "reached", measured from the per-position rows.
    """
    empty = ({}, {}, {}) if with_counts else pd.Series(dtype=float)
    if trials.empty:
        return empty
    position_col = "last_odor_position" if "last_odor_position" in trials.columns else "last_event_index"
    if position_col not in trials.columns:
        return empty

    aborted = trials[_aborted_mask(trials)]
    abortions = aborted[position_col].dropna().value_counts().to_dict()
    reached = _reached_counts(trials, position_data)

    rates = {}
    for pos in sorted(set(list(abortions.keys()) + list(reached.keys()))):
        n_reached = reached.get(pos, 0)
        rates[pos] = abortions.get(pos, 0) / n_reached if n_reached > 0 else np.nan
    if with_counts:
        return rates, abortions, reached
    return pd.Series(rates, dtype=float).sort_index()


@session_metric(abortion_rate_positionX)
def abortion_rate_positionX_session(results):
    parts = abortion_rate_positionX(results.get("trial_data", pd.DataFrame()),
                                    results.get("position_data"),
                                    with_counts=True)
    if not isinstance(parts, tuple):
        return parts
    rates, abortions, reached = parts
    for pos in sorted(rates):
        print(f"Position {pos}: {abortions.get(pos, 0)}/{reached.get(pos, 0)} abortions, "
              f"Rate: {rates[pos]:.3f}")
    return pd.Series(rates, dtype=float).sort_index()


@metric(frame="trials", title="Odor Initiation Bias", adapter=as_dict)
def odor_initiation_bias(trials, *, reference=None, with_counts=False):
    """Per-odor initiation-abortion share / the overall share. See `FA_odor_bias`."""
    empty = ({}, {}, {}) if with_counts else pd.Series(dtype=float)
    if trials.empty or "abortion_type" not in trials.columns:
        return empty
    odor_col = "last_odor_name" if "last_odor_name" in trials.columns else "last_odor"
    if odor_col not in trials.columns:
        return empty
    aborted = trials[_aborted_mask(trials)]
    if aborted.empty:
        return empty

    init_mask = aborted["abortion_type"] == "initiation_abortion"
    total_init = int(init_mask.sum())
    total_ab = len(aborted)
    ref = reference if reference is not None else (
        (total_init / total_ab) if total_ab > 0 and total_init > 0 else None)

    bias, n_init, n_ab = {}, {}, {}
    for od in sorted(aborted[odor_col].dropna().unique()):
        at_od = aborted[odor_col] == od
        n_init_od = int((at_od & init_mask).sum())
        n_ab_od = int(at_od.sum())
        n_init[od], n_ab[od] = n_init_od, n_ab_od
        bias[od] = (n_init_od / n_ab_od) / ref if n_ab_od > 0 and ref else np.nan
    if with_counts:
        return bias, n_init, n_ab
    return pd.Series(bias).sort_index()


@session_metric(odor_initiation_bias)
def odor_initiation_bias_session(results):
    parts = odor_initiation_bias(results.get("trial_data", pd.DataFrame()), with_counts=True)
    if not isinstance(parts, tuple):
        return parts
    bias, n_init, n_ab = parts
    for od in sorted(bias):
        print(f"{od}: {n_init[od]}/{n_ab[od]} initiation abortions, Bias: {bias[od]:.3f}")
    return pd.Series(bias).sort_index()
