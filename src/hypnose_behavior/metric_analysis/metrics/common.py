# Defers evaluation of PEP-604 annotations (`X | None`), keeping this module
# importable on Python 3.9 for repos pinned there (hypnose-eeg-preprocessing).
from __future__ import annotations

"""Primitives the metric cores share: predicates, rate reduction, frame slicing.

Nothing in here is a metric. The six modules beside it group metrics by
behavioural construct, so whatever more than one construct needs lives here
rather than in whichever module happens to define it first.

**Rate metrics expose their numerator and denominator *contributions* as
per-trial Series**, because a rate is not a per-trial quantity. Storing one value
per trial and taking a rolling mean gives ``rewarded / window_size`` -- a denominator
silently containing timeouts and aborts, which is how two rolling accuracies come to
disagree. Reducing ``num.sum() / den.sum()`` over any slice -- what ``reduce_rate``
does -- is correct at every granularity. See DECISIONS.md section 1.

**``_position_rows`` filters on a provenance flag, always.** The three JSON blobs
``frames.build_position_data`` expands do not carry the same positions:
``position_valve_times`` records valve activations whose poke registered as
~0 ms, which ``position_poke_times`` and ``presentations`` both drop. On the
fixture set that is 34 rows on sub-053, 19 on sub-057, 17 on sub-059, 7 on
sub-048, 4 on sub-040 20251124 and 1 each on sub-046 and sub-056. Reading
``position_data`` unfiltered would silently widen every sampling metric.
"""

import math

import numpy as np
import pandas as pd

__all__ = ["reduce_rate"]


def _is_truthy(val):
    """Is a flag column's cell true, however the round-trip stored it?

**The one truthiness rule for the package.** A string that parses as a
    non-zero number is as true as the number itself.

    **Use this whenever a flag column is added; do not write a second rule, and do not
    narrow it** -- widening cannot lose a row, narrowing silently can. See DECISIONS.md
    section 6.
    """
    if isinstance(val, bool):
        return val
    if isinstance(val, (int, float)):
        try:
            return not math.isnan(val) and val != 0
        except Exception:
            return val != 0
    if isinstance(val, str):
        s = val.strip().lower()
        if s in {"1", "true", "t", "yes", "y"}:
            return True
        try:
            return _is_truthy(float(s))
        except ValueError:
            return False
    return False


def _aborted_mask(trials):
    """The `df["is_aborted"] == True` mask every metric builds, once."""
    if "is_aborted" in trials.columns:
        return trials["is_aborted"] == True  # noqa: E712
    return pd.Series(False, index=trials.index)


def _flag(trials, column, value):
    """`trials[column] == value` as a boolean Series; all-False when absent."""
    if column in trials.columns:
        return trials[column] == value
    return pd.Series(False, index=trials.index)


def _truthy(trials, column):
    if column in trials.columns:
        return trials[column].apply(_is_truthy).astype(bool)
    return pd.Series(False, index=trials.index)


def reduce_rate(num, den):
    """(numerator, denominator) contributions -> (n, denom, rate).

    Public because a plotter that has already collected a metric's per-trial
    contributions -- `sing_rew.FR_ratio` does, to draw both a session ratio and a
    rolling one -- must reduce them the same way the metric does, rather than
    reaching for its own `np.mean`.
    """
    n = int(np.asarray(num, dtype=float).sum())
    d = int(np.asarray(den, dtype=float).sum())
    return n, d, (n / d if d > 0 else np.nan)


def _initiated(trials):
    """Denominator "an initiated trial": a non-null global_trial_id, else all rows."""
    if "global_trial_id" in trials.columns:
        return trials["global_trial_id"].notna().astype(int)
    return pd.Series(1, index=trials.index)


def _position_rows(position_data, blob, *, aborted=None):
    """Rows of `position_data` that came from `blob`, optionally by outcome.

    Returns None when the frame is absent or unusable, which every caller treats
    as "no positions" -- the same answer today's inline blob walk gives.
    """
    if position_data is None or len(position_data) == 0:
        return None
    if blob not in position_data.columns:
        return None
    rows = position_data[position_data[blob].astype(bool)]
    if aborted is not None:
        rows = rows[rows["is_aborted"].astype(bool) == aborted]
    return rows


def _tz_naive(series):
    """Datetime Series with any timezone dropped, so subtraction is safe."""
    s = pd.to_datetime(series, errors="coerce")
    try:
        if s.dt.tz is not None:
            s = s.dt.tz_localize(None)
    except (AttributeError, TypeError):
        pass
    return s


def _trial_position_frame(position_data, blob):
    """One blob's rows sorted by position within trial, or None if unusable."""
    rows = _position_rows(position_data, blob)
    if rows is None or rows.empty or "global_trial_id" not in rows.columns:
        return None
    return rows.sort_values(["global_trial_id", "position"], kind="stable")


def _trial_timestamp(trials, field):
    """A trial-level timestamp column, tz-naive, indexed by trial id."""
    if field not in trials.columns or "global_trial_id" not in trials.columns:
        return None
    return pd.Series(_tz_naive(trials[field]).to_numpy(),
                     index=trials["global_trial_id"].to_numpy())


def _latency_ms(later, earlier):
    """`later - earlier` in ms, dropping pairs where either side is missing.

    Vectorised `.dt.total_seconds()` on purpose. The plotters walk trials one at
    a time and go through *scalar* `Timedelta.total_seconds()`, which truncates
    to microseconds, so they silently discard the nanoseconds the blob
    timestamps carry (`...T13:49:07.507839999`). The timedeltas themselves are
    bit-identical either way; only the conversion differs. Measured on all 9
    fixture sessions the two forms agree to **0.999 ns**, on latencies of
    hundreds to thousands of ms -- so this keeps the exact value rather than
    reproducing the truncation.
    """
    if later is None or earlier is None:
        return pd.Series(dtype=float)
    return (later - earlier).dropna().dt.total_seconds() * 1000.0


# The private spelling every metric module already uses.
_reduce_rate = reduce_rate
