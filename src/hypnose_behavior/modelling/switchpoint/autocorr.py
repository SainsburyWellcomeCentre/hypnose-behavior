"""Residual-autocorrelation maths for the i.i.d.-Bernoulli check behind the bootstrap.

The planned parametric bootstrap null (``bootstrap.py``) redraws each animal's sequence as
independent Bernoulli trials from the fitted model. That is only valid if the *residuals* of
the fitted model are serially independent. This module provides the lag-wise autocorrelation
of those residuals and its significance band; the orchestration that forms the residuals,
prints the verdict and plots lives in ``scripts/modelling/switchpoint_analysis.py``.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

# TODO (deferred): `model_fitted_p` now returns a curve for `qlearning` too, so when the
# mechanistic null is the BIC winner this diagnostic will run on ITS residuals -- and those are
# not comparable with the descriptive models'. The Q-learner's fitted curve is the
# one-step-ahead P(SHORT), conditioned on the animal's own choice history, so it absorbs part of
# the serial dependence the ACF is trying to measure: a lag-1 near zero would then mean "the
# Q-learner already explained the autocorrelation", not "the trials are independent". The
# verdict text and the i.i.d.-bootstrap conclusion both need rewording for that case. Nothing
# changed here yet.

# Default largest lag reported by the diagnostic (clamped to n - 1 by the caller).
ACF_MAX_LAG = 50
# ~95% band multiplier, i.e. the standard +/- 1.96 / sqrt(N) autocorrelation band.
_ACF_Z = 1.96
# A significant lag-1 below this |r| is still treated as immaterial for the i.i.d. verdict.
ACF_MATERIAL_THRESHOLD = 0.1

__all__ = ["residual_acf", "acf_bounds", "ACF_MAX_LAG", "ACF_MATERIAL_THRESHOLD"]


def residual_acf(resid: np.ndarray, lags: np.ndarray,
                 session_index: Optional[np.ndarray] = None) -> tuple[np.ndarray, np.ndarray]:
    """Lag-wise autocorrelation of residuals, as a Pearson correlation per lag.

    For each lag ``k`` the value is ``sum(d_t d_{t+k}) / sqrt(sum(d_t^2) sum(d_{t+k}^2))`` over
    the included ``(t, t+k)`` pairs, with ``d`` the mean-centred residuals -- bounded in
    ``[-1, 1]`` so full and within-session versions share a scale.

    With ``session_index`` given, only pairs whose two trials fall in the same session are
    kept: the sleep gaps between sessions make a cross-session lag meaningless, and the strategy
    shift across a gap would otherwise masquerade as trial-to-trial dependence.

    Returns ``(acf, n_pairs)``; ``acf`` is NaN for any lag with no usable pair.
    """
    d = resid - resid.mean()
    n = d.size
    acf = np.full(lags.size, np.nan)
    n_pairs = np.zeros(lags.size, dtype=int)
    for i, k in enumerate(lags):
        if k >= n:
            continue
        a, b = d[:-k], d[k:]
        if session_index is not None:
            same = session_index[:-k] == session_index[k:]
            a, b = a[same], b[same]
        n_pairs[i] = a.size
        denom = float(np.sqrt(np.dot(a, a) * np.dot(b, b)))
        if a.size and denom > 0:
            acf[i] = float(np.dot(a, b) / denom)
    return acf, n_pairs


def acf_bounds(n_pairs: np.ndarray) -> np.ndarray:
    """Per-lag ``+/- 1.96 / sqrt(N)`` significance bound, NaN where a lag has no pair."""
    with np.errstate(divide="ignore"):
        return np.where(n_pairs > 0, _ACF_Z / np.sqrt(n_pairs), np.nan)
