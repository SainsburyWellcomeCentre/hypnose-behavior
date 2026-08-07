# Defers evaluation of PEP-604 annotations, matching `primitives.py`.
from __future__ import annotations

"""Trajectory prep shared between the two movement plotting modules.

restructure_2 Phase 5, the survivors of the Phase 4 audit's finding 10. That
finding listed seven helpers duplicated 2-4x across `movement_analysis_utils`
and `movement_analysis/sing_rew_movement` and called for de-duplicating both
files in one pass, on the premise that "every row has a twin".

**Measured, that premise does not hold: most of them are different rules
wearing the same name, and merging them would change what is plotted.** Only
the two below are genuinely the same computation. What the others do, and the
numbers, are in `docs/DECISIONS.md` section 13 -- read that before trying the
merge again.

This is prep, not display arithmetic: it reshapes the trace *before* anything
is drawn. `primitives.py` is the other half, and takes the mean/SEM/rolling of
values that are already on the figure.
"""

import numpy as np
import pandas as pd

__all__ = ["resample_trace", "smooth_xy"]


def resample_trace(x, y, n_points: int = 200):
    """Resample a trajectory onto a normalised arc-length grid ``[0, 1]``.

    Returns ``(x_new, y_new)``, or None for a trace that cannot be resampled:
    fewer than two points, any non-finite coordinate, or zero total path length
    (an animal that never moved -- the arc-length parameterisation is undefined
    there, not merely degenerate).

    Resampling by arc length rather than by time is what makes traces of
    different durations averageable: every trace contributes `n_points` samples
    spaced evenly along its own path, so the mean trace follows the shared
    *route* instead of being dominated by whichever trial was slowest.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size < 2 or y.size < 2:
        return None
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        return None
    seg_len = np.hypot(np.diff(x), np.diff(y))
    cumlen = np.concatenate(([0.0], np.cumsum(seg_len)))
    total_len = cumlen[-1]
    if total_len <= 0:
        return None
    s = cumlen / total_len
    s_new = np.linspace(0.0, 1.0, num=n_points)
    return np.interp(s_new, s, x), np.interp(s_new, s, y)


def smooth_xy(tracking, window):
    """Centred rolling-mean smooth of the ``X``/``Y`` columns of `tracking`.

    Returns `tracking` unchanged for a window of None or <= 1, and never
    mutates the input. `min_periods=1` keeps the ends of the trace rather than
    trimming half a window off each, which matters because the endpoints are
    exactly what the trace plots are read for.

    A duplicated ``X``/``Y`` column name yields a DataFrame rather than a
    Series from the lookup; the first column wins. One of the three call sites
    this replaces handled that and the other two would have raised on it.
    """
    if window is None or window <= 1:
        return tracking
    df = tracking.copy()
    for col in ("X", "Y"):
        values = df[col]
        if isinstance(values, pd.DataFrame):
            values = values.iloc[:, 0]
        df[col] = pd.Series(values).rolling(
            window=window, center=True, min_periods=1
        ).mean()
    return df
