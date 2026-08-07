# Defers evaluation of PEP-604 annotations (`X | None`), keeping this module
# importable on Python 3.9 for repos pinned there (hypnose-eeg-preprocessing).
from __future__ import annotations

"""Display primitives: the arithmetic a *figure* does, not the arithmetic a metric does.

restructure_2 Phase 5. The standing rule from the Phase 4 audit is that taking the
mean +/- SEM of a metric across the subjects or sessions on a plot is a property of
the figure, not of the data -- so it belongs here and never in `metric_analysis`.
It was written longhand about 20 times across `visualization/`.

**These may roll or average *values*; they must never re-absorb a rate reduction.**
A rolling rate is `sum(numerator) / sum(denominator)` over the window, never the mean
of per-trial values -- that silently divides by the window size instead of by the
trials that actually counted. That reduction lives in
`metric_analysis.resolvers.over_windows` and stays there. See `docs/DECISIONS.md`
section 1.

On NaN, which is where the longhand versions quietly disagreed. Three idioms were in
use, and while `Series.sem()`, `Series.std(ddof=1)/sqrt(len(s))` and
`np.std(v, ddof=1)/sqrt(len(v))` are bit-identical on clean data (measured: 0
disagreements in 20,000 random samples on the pinned pandas/numpy), they diverge as
soon as a NaN appears: the first divides by the count of *finite* values, the second
by the full length, and the third propagates NaN. Only the first is right, so that is
what `mean_sem` does.
"""

import numpy as np
import pandas as pd

__all__ = ["mean_sem", "sem_band", "rolling_mean"]


def _finite(values) -> np.ndarray:
    """`values` as a 1-D float array with non-finite entries dropped."""
    arr = np.asarray(pd.Series(values, dtype="float64").to_numpy(), dtype="float64")
    return arr[np.isfinite(arr)]


def mean_sem(values, *, ddof: int = 1):
    """``(mean, sem)`` across the sessions or subjects on a plot.

    Non-finite entries are dropped first, so ``n`` is the number of values that
    actually contributed -- matching ``Series.sem()`` and *not* the longhand
    ``std(ddof=1) / sqrt(len(x))``, which divides by the full length and therefore
    understates the error whenever anything is missing.

    ``sem`` is ``nan`` for fewer than two finite values, where it is undefined.
    Callers that draw a zero-height error bar in that case should say so
    explicitly (``sem or 0.0``) rather than rely on the primitive to invent one.
    """
    arr = _finite(values)
    if arr.size == 0:
        return float("nan"), float("nan")
    mean = float(arr.mean())
    if arr.size <= ddof:
        return mean, float("nan")
    return mean, float(arr.std(ddof=ddof) / np.sqrt(arr.size))


def sem_band(ax, x, mean, sem, **kwargs):
    """Shade mean +/- sem around a line. Returns the `PolyCollection`.

    `alpha` and `linewidth` default to the repo's band styling; everything else is
    forwarded to `fill_between`, so a caller that wants a different colour or
    z-order just passes it.
    """
    x = np.asarray(x, dtype="float64")
    mean = np.asarray(mean, dtype="float64")
    sem = np.nan_to_num(np.asarray(sem, dtype="float64"), nan=0.0)
    kwargs.setdefault("alpha", 0.2)
    kwargs.setdefault("linewidth", 0)
    return ax.fill_between(x, mean - sem, mean + sem, **kwargs)


def rolling_mean(series, window: int, *, step: int = 1, min_periods=None):
    """Trailing rolling mean of `series`, as ``(end_index, value)`` pairs.

    Anchored the way `metric_analysis.resolvers.over_windows` anchors: the first
    window ends at position ``window - 1``, and `min_periods` defaults to `window`,
    so there are no partial windows at the head. Returned positions are *positional*
    indices into `series`, which is what a caller needs to look up the x value of
    each window's last element.

    **For values only.** Rolling a rate through this is the error `DECISIONS.md`
    section 1 exists to prevent -- use `over_windows` on the metric core instead.
    """
    values = pd.Series(series, dtype="float64").to_numpy()
    window = max(1, int(window))
    step = max(1, int(step))
    if values.size == 0:
        return []
    span = min(window, values.size)
    required = span if min_periods is None else max(1, int(min_periods))

    out = []
    for end in range(span, values.size + 1, step):
        chunk = values[end - span:end]
        finite = chunk[np.isfinite(chunk)]
        if finite.size < required:
            continue
        out.append((end - 1, float(finite.mean())))
    return out
