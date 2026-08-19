# Defers evaluation of PEP-604 annotations (`X | None`), keeping this module
# importable on Python 3.9 for repos pinned there (hypnose-eeg-preprocessing).
from __future__ import annotations

"""Latencies: how long between one thing happening and the next.

The trial-timing family is indexed by ``global_trial_id``, so pass one session's
frames -- pooled frames repeat ids and the index alignment mis-pairs trials.

- ``reward_delivery_latency`` shares an everyday name with ``response_time_ms`` and
  is **not** it: this measures from leaving the odor port, that from the reward-port
  poke. The FA counterpart, ``fa_latency_from_pokeout``, lives in ``false_alarm.py`` --
  grouped by what it measures, not by returning a time.
- **Metrics raw; filtering is display.** The 10x-group-mean outlier rule
  ``pred_seq_utils.response_time`` and ``fa_analysis`` apply stays in
  ``visualization/``, where it can be seen and changed.
"""

import numpy as np
import pandas as pd

from hypnose_behavior.metric_analysis.metrics.common import (
    _latency_ms,
    _trial_position_frame,
    _trial_timestamp,
    _tz_naive,
)
from hypnose_behavior.metric_analysis.registry import metric, session_metric

__all__ = [
    "avg_response_time", "avg_response_time_session",
    "inter_trial_interval",
    "reward_delivery_latency", "valve_to_reward_latency",
]


@metric(frame="trials", title="Average Response Time")
def avg_response_time(trials):
    """Mean `response_time_ms` by category, plus the pooled rewarded+unrewarded."""
    if (trials.empty or "response_time_category" not in trials.columns
            or "response_time_ms" not in trials.columns):
        return {}
    vals = pd.to_numeric(trials["response_time_ms"], errors="coerce")
    out = {}
    for label, key in [("Rewarded", "rewarded"), ("Unrewarded", "unrewarded"),
                       ("Reward Timeout", "timeout_delayed")]:
        s = vals[trials["response_time_category"] == key].dropna()
        out[label] = float(s.mean()) if not s.empty else np.nan
    both = vals[trials["response_time_category"].isin(["rewarded", "unrewarded"])].dropna()
    out["Average Response Time (Rewarded + Unrewarded)"] = float(both.mean()) if not both.empty else np.nan
    return out


@session_metric(avg_response_time)
def avg_response_time_session(results):
    df = results.get("trial_data", pd.DataFrame())
    if df.empty or "response_time_category" not in df.columns or "response_time_ms" not in df.columns:
        print("No response time data available.")
        return {}
    out = avg_response_time(df)
    vals = pd.to_numeric(df["response_time_ms"], errors="coerce")
    for label, key in [("Rewarded", "rewarded"), ("Unrewarded", "unrewarded"),
                       ("Reward Timeout", "timeout_delayed")]:
        avg, n = out[label], len(vals[df["response_time_category"] == key].dropna())
        print(f"{label}: {avg:.1f} ms (n={n})" if not np.isnan(avg) else f"{label}: nan (n={n})")
    key = "Average Response Time (Rewarded + Unrewarded)"
    avg_both = out[key]
    n_both = len(vals[df["response_time_category"].isin(["rewarded", "unrewarded"])].dropna())
    print(f"{key}: {avg_both:.1f} ms (n={n_both})" if not np.isnan(avg_both)
          else f"{key}: nan (n={n_both})")
    return out


@metric(frame="trials")
def inter_trial_interval(trials):
    """Seconds from one trial ending to the next starting.

    `sequence_start.shift(-1) - sequence_end`, so the last row is NaN. Pass a
    single session's trials: shifting across a session boundary would measure the
    gap between recordings, which is not an inter-trial interval.
    """
    if (trials.empty or "sequence_start" not in trials.columns
            or "sequence_end" not in trials.columns):
        return pd.Series(np.nan, index=trials.index, dtype=float)
    start = _tz_naive(trials["sequence_start"])
    end = _tz_naive(trials["sequence_end"])
    return (start.shift(-1) - end).dt.total_seconds()


def _deepest_position_timestamp(position_data, blob, field):
    """`field` at each trial's deepest position, tz-naive, indexed by trial id."""
    rows = _trial_position_frame(position_data, blob)
    if rows is None:
        return None
    frame = pd.DataFrame({"gid": rows["global_trial_id"].to_numpy(),
                          "ts": _tz_naive(rows[field]).to_numpy()})
    return frame.groupby("gid", sort=True)["ts"].agg(lambda s: s.iloc[-1])


@metric(frame="trials+position_data")
def reward_delivery_latency(trials, position_data):
    """`first_supply_time` minus the last odor poke-out, in ms.

    **Not** `trial_data.response_time_ms`, which is measured from the reward-port poke
    rather than from leaving the odor port: an everyday name shared by two definitions.
    """
    return _latency_ms(_trial_timestamp(trials, "first_supply_time"),
                       _deepest_position_timestamp(position_data, "in_poke_times",
                                                   "poke_odor_end"))


@metric(frame="trials+position_data")
def valve_to_reward_latency(trials, position_data):
    """`first_supply_time` minus the last position's `valve_start`, in ms.

    Nothing canonical measures anything from a valve opening.
    """
    return _latency_ms(_trial_timestamp(trials, "first_supply_time"),
                       _deepest_position_timestamp(position_data, "in_valve_times",
                                                   "valve_start"))
