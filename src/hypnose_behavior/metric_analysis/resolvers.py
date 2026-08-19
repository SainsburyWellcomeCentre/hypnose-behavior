# Defers evaluation of PEP-604 annotations (`X | None`), keeping this module
# importable on Python 3.9 for repos pinned there (hypnose-eeg-preprocessing).
from __future__ import annotations

"""Granularity resolvers: apply a metric core per group, or over a moving window.

Because every metric core is `f(frame) -> value`, a granularity is a call rather
than a hand-derived reimplementation in a plotter:

    by_group(decision_accuracy, trials, "last_odor")            # per odor
    by_group(decision_accuracy, trials, "date")                 # per day
    over_windows(decision_accuracy, trials, window=30)          # rolling

- **Not a dispatcher.** No `get_metric(name, granularity=)` -- that accumulates
  kwargs for every metric it supports. A uniform core signature plus these two
  resolvers gives the same reach without it.
- A variant whose **denominator changes with granularity** cannot be expressed here.
  The rolling reward fraction divides by the window size rather than by
  rewarded+unrewarded, so it stays a separately named metric.
- **Anything collecting a metric's contributions must reduce them the way the metric
  does** -- `metrics.common.reduce_rate` is public for exactly that. A rate is not a
  per-trial quantity. See DECISIONS.md section 1.
"""

from typing import Callable, Optional

import numpy as np
import pandas as pd

__all__ = ["by_group", "over_windows"]


def _value(result):
    """Metric cores return either a scalar or an `(n, denom, value)` triple."""
    if isinstance(result, tuple) and len(result) == 3:
        return result[2]
    return result


def by_group(metric: Callable, trials: pd.DataFrame, key, *,
             values_only: bool = True, dropna: bool = True, **kwargs):
    """Apply a metric core to each group of ``trials``.

    ``key`` is anything ``DataFrame.groupby`` accepts. Returns a Series indexed
    by group. With ``values_only=False`` the cores' full return (e.g. the
    ``(n, denom, rate)`` triple) is kept, which is what a plotter wants when it
    needs to show counts alongside the rate.

    The metric is applied to a *slice of the frame*, so a rate is recomputed as
    ``num.sum() / den.sum()`` within the group -- **never averaged from per-trial
    values**. See DECISIONS.md section 1.
    """
    if trials.empty:
        return pd.Series(dtype=float)
    out = {}
    for name, sub in trials.groupby(key, dropna=dropna, observed=True):
        result = metric(sub, **kwargs)
        out[name] = _value(result) if values_only else result
    return pd.Series(out).sort_index()


def over_windows(metric: Callable, trials: pd.DataFrame, window: int, *,
                 step: int = 1, min_periods: Optional[int] = None,
                 order_by: Optional[str] = None, **kwargs) -> pd.DataFrame:
    """Apply a metric core to each trailing window of ``window`` trials.

    Returns a frame with ``end_index`` (position of the window's last trial) and
    ``value``. ``min_periods`` defaults to ``window``, i.e. no partial windows --
    matching `pred_seq_utils.performance` in both the partial-window rule *and*
    the emitted positions for any ``step`` -- and deliberately *not* the warm-up
    back-fill that `plot_decision_accuracy_rolling_average` does (that is a
    different metric, not a different granularity; see limit 1 above).

    Each window is evaluated by calling the core on that slice, so rate metrics
    reduce their numerator and denominator contributions within the window
    rather than averaging per-trial values.
    """
    if window < 1:
        raise ValueError("window must be >= 1")
    if trials.empty:
        return pd.DataFrame(columns=["end_index", "value"])

    frame = trials.sort_values(order_by) if order_by else trials
    n = len(frame)
    min_periods = window if min_periods is None else min_periods

    rows = []
    # Anchor the first window at the first position satisfying `min_periods`, then
    # step from there. Starting at 0 and discarding short windows instead would
    # align emitted points to multiples of `step`, dropping and shifting them
    # whenever `(min_periods - 1) % step != 0`: window=10, step=4 would emit
    # positions 12, 16 where the rolling this replaces emits 9, 13, 17. Identical
    # for the default step=1, which is what made it easy to miss.
    for end in range(max(min_periods, 1) - 1, n, step):
        start = max(0, end + 1 - window)
        if end + 1 - start < min_periods:
            continue
        rows.append({"end_index": end,
                     "value": _value(metric(frame.iloc[start:end + 1], **kwargs))})
    return pd.DataFrame(rows, columns=["end_index", "value"])
