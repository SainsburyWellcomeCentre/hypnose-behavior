"""Bernoulli switch-point model family for a binary strategy sequence.

Closed-form, O(n) profile likelihood over the candidate switch trials -- no MCMC and no
change-point package. Everything here is pure numeric (numpy + scipy): numpy arrays in,
numpy arrays and plain dicts out. No file I/O, no plotting, no pandas, no path handling.

This module holds the descriptive switch-point family and its Bernoulli primitives. The
sibling models and the machinery that scores them live alongside it:

- ``qlearning.py``   -- the mechanistic Q-learning account: the null to be rejected.
- ``compare.py``     -- ``compare_models`` (AIC/BIC over every model) and the per-trial
  fitted-P reconstruction.
- ``permutation.py`` -- ``distance_to_session_start`` and the sleep-alignment test.

Model
-----
Each solved trial ``i`` is a Bernoulli draw ``s[i] ~ Bern(p_i)``, where ``s[i] == 1`` means
the animal left via the SHORT sequence. The descriptions of ``p_i`` compared here:

- ``constant``: ``p_i = p`` -- no strategy change (1 parameter).
- ``switch``: ``p_i = p1`` for ``i < tau`` and ``p2`` for ``i >= tau`` -- one abrupt,
  directional change (3 parameters: ``tau``, ``p1``, ``p2``).
- ``logistic``: ``p_i = lo + (hi - lo) * sigmoid(slope * (i - midpoint))`` -- a graded
  change (4 parameters). ``slope`` is the abruptness: large ``slope`` approaches the step
  of the switch model, small ``slope`` is a slow drift.
- ``switch2``: three regimes ``p1 | p2 | p3`` split by an ordered pair ``tau1 < tau2``
  (5 parameters) -- a change that arrives in two stages. Constrained non-decreasing
  (``p1 <= p2 <= p3``) so it describes a staged progression toward SHORT rather than a
  transient spike (e.g. ``0 -> 1 -> 0`` over a few trials, which would otherwise let it
  beat the constant on noise).

``tau`` is the index of the FIRST trial of the post-switch regime, so it ranges over
``1 .. n-1``; a switch at ``0`` or ``n`` is just the constant model and is excluded.

Probabilities are clipped away from 0 and 1 so the log-likelihood stays finite when a
segment is all zeros or all ones.
"""
from __future__ import annotations

import warnings
from typing import Sequence

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit

# Keeps log(p) and log1p(-p) finite for degenerate (all-0 / all-1) segments.
_EPS = 1e-12

# Clamp the logistic argument before exp() so a large fitted slope cannot overflow.
_Z_MAX = 500.0

# Logistic multi-start grid (see logistic_start_points). Midpoints as fractions of the trial
# axis, and initial slopes from shallow-gradual to steep-near-step.
_LOGISTIC_MIDPOINT_QUANTILES = (0.10, 0.30, 0.50, 0.70, 0.90)
_LOGISTIC_INITIAL_SLOPES = (0.05, 0.5, 5.0)

# A logistic loglik shortfall below the switch model larger than this is flagged as an
# optimization failure (the logistic nests the step, so it cannot truly be worse).
_LOGLIK_TOL = 1e-6

# Label of the switch-point warm start in logistic_start_points -- the single source of truth,
# so callers (fit_logistic, the multi-start diagnostic) can identify it without hardcoding.
WARM_START_LABEL = "switchpoint"

__all__ = [
    "bernoulli_loglik",
    "switchpoint_loglik_profile",
    "switchpoint_posterior",
    "posterior_hdi",
    "posterior_fwhm",
    "fit_constant",
    "fit_switchpoint",
    "fit_switch2",
    "logistic_p",
    "logistic_start_points",
    "fit_logistic_multistart",
    "fit_logistic",
    "WARM_START_LABEL",
]


def _clip_p(p):
    """Clip probabilities into ``[_EPS, 1 - _EPS]`` so log-likelihoods stay finite."""
    return np.clip(p, _EPS, 1.0 - _EPS)


def _as_binary(s: Sequence[int] | np.ndarray) -> np.ndarray:
    """Coerce ``s`` to a flat int array of 0/1, raising on anything else."""
    arr = np.asarray(s).ravel()
    if arr.size and not np.isin(arr, (0, 1)).all():
        raise ValueError("s must contain only 0 and 1")
    return arr.astype(np.int64)


def bernoulli_loglik(s: Sequence[int] | np.ndarray, p) -> float:
    """Log-likelihood of ``s`` under Bernoulli success probability ``p``.

    Parameters
    ----------
    s : array_like of {0, 1}
        Binary outcome sequence.
    p : float or array_like
        Success probability. A scalar applies to every trial; an array must be the
        same length as ``s`` (used for the per-trial logistic probabilities).

    Returns
    -------
    float
    """
    s = _as_binary(s)
    p = _clip_p(np.asarray(p, dtype=float))
    return float(np.sum(s * np.log(p) + (1 - s) * np.log1p(-p)))


def switchpoint_loglik_profile(s: Sequence[int] | np.ndarray) -> np.ndarray:
    """Profile log-likelihood of a switch at each candidate trial, in O(n).

    Element ``tau`` is the log-likelihood of the switch model whose post-switch regime
    starts at trial ``tau``, with ``p1`` and ``p2`` set to their (closed-form) maximum-
    likelihood values -- the segment means. Computed from a prefix sum, so no loop over
    trials. Element ``0`` is ``-inf``: a switch at trial 0 has an empty pre-switch
    segment and is the constant model, not a switch.

    Returns
    -------
    np.ndarray
        Length ``n``. All ``-inf`` when ``n < 2``.
    """
    s = _as_binary(s)
    n = s.size
    ll = np.full(n, -np.inf)
    if n < 2:
        return ll
    prefix = np.concatenate(([0], np.cumsum(s)))  # prefix[i] = successes in s[:i]
    tau = np.arange(1, n)
    n1, k1 = tau.astype(float), prefix[tau].astype(float)
    n2, k2 = (n - tau).astype(float), (prefix[-1] - prefix[tau]).astype(float)
    p1, p2 = _clip_p(k1 / n1), _clip_p(k2 / n2)
    ll[1:] = (k1 * np.log(p1) + (n1 - k1) * np.log1p(-p1)
              + k2 * np.log(p2) + (n2 - k2) * np.log1p(-p2))
    return ll


def switchpoint_posterior(s: Sequence[int] | np.ndarray) -> np.ndarray:
    """Normalized posterior over the switch trial under a uniform prior on ``tau``.

    Exponentiates the profile log-likelihood with the max subtracted (so the largest
    term is ``exp(0) == 1`` and nothing underflows), then normalizes to sum to 1.

    Returns
    -------
    np.ndarray
        Length ``n``, non-negative, sums to 1 (all zeros when ``n < 2``).
    """
    ll = switchpoint_loglik_profile(s)
    finite = np.isfinite(ll)
    posterior = np.zeros(ll.size, dtype=float)
    if not finite.any():
        return posterior
    posterior[finite] = np.exp(ll[finite] - ll[finite].max())
    total = posterior.sum()
    return posterior / total if total > 0 else posterior


def posterior_hdi(posterior: np.ndarray, mass: float = 0.95) -> tuple[int, int]:
    """Highest-density interval of a discrete posterior, as ``(lo, hi)`` trial indices.

    Takes trials in order of descending posterior mass until ``mass`` is covered, then
    returns the smallest and largest index in that set. The interval is reported as a
    range, so it can contain trials below the density threshold when the posterior is
    multimodal -- ``posterior_fwhm`` is the sharper (secondary) width measure.

    Returns
    -------
    (int, int)
        Inclusive bounds. ``(0, n - 1)`` for an empty/degenerate posterior.
    """
    post = np.asarray(posterior, dtype=float)
    total = post.sum()
    if post.size == 0:
        return (0, 0)
    if total <= 0:
        return (0, post.size - 1)
    order = np.argsort(post)[::-1]
    covered = np.searchsorted(np.cumsum(post[order]), mass * total) + 1
    kept = order[:min(int(covered), post.size)]
    return int(kept.min()), int(kept.max())


def posterior_fwhm(posterior: np.ndarray) -> tuple[int, int]:
    """Full width at half maximum of a discrete posterior, as ``(lo, hi)`` indices.

    The first and last trial whose posterior mass is at least half the peak mass.
    Secondary to ``posterior_hdi``: narrower, and it ignores how much total mass the
    interval actually holds.
    """
    post = np.asarray(posterior, dtype=float)
    if post.size == 0:
        return (0, 0)
    peak = post.max()
    if peak <= 0:
        return (0, post.size - 1)
    above = np.flatnonzero(post >= 0.5 * peak)
    return int(above[0]), int(above[-1])


def fit_constant(s: Sequence[int] | np.ndarray) -> dict:
    """Fit the no-change model: a single Bernoulli rate for the whole sequence.

    Returns
    -------
    dict
        ``loglik``, ``p`` (the sequence mean), ``k_params`` (1).
    """
    s = _as_binary(s)
    p = float(s.mean()) if s.size else 0.5
    return {"loglik": bernoulli_loglik(s, p), "p": p, "k_params": 1}


def fit_switchpoint(s: Sequence[int] | np.ndarray) -> dict:
    """Fit the one-switch model by maximizing the profile log-likelihood over ``tau``.

    Returns
    -------
    dict
        ``tau`` (first trial of the post-switch regime), ``p1``, ``p2``, ``loglik``,
        ``posterior`` (length ``n``), ``hdi`` and ``fwhm`` as ``(lo, hi)`` index pairs,
        and ``k_params`` (3). For ``n < 2`` the fit is degenerate: ``tau`` is 0 and the
        rates are NaN.
    """
    s = _as_binary(s)
    n = s.size
    posterior = switchpoint_posterior(s)
    if n < 2:
        return {"tau": 0, "p1": float("nan"), "p2": float("nan"), "loglik": float("-inf"),
                "posterior": posterior, "hdi": (0, max(n - 1, 0)),
                "fwhm": (0, max(n - 1, 0)), "k_params": 3}
    ll = switchpoint_loglik_profile(s)
    tau = int(np.argmax(ll))  # ll[0] is -inf, so tau lands in 1 .. n-1
    return {"tau": tau, "p1": float(s[:tau].mean()), "p2": float(s[tau:].mean()),
            "loglik": float(ll[tau]), "posterior": posterior,
            "hdi": posterior_hdi(posterior), "fwhm": posterior_fwhm(posterior),
            "k_params": 3}


def _segment_loglik(k: np.ndarray | float, m: np.ndarray | float) -> np.ndarray | float:
    """Maximized Bernoulli log-likelihood of a segment with ``k`` successes in ``m`` trials.

    The ML rate of a segment is its mean ``k / m``, so the segment contributes
    ``k*log(p) + (m-k)*log(1-p)`` at ``p = k/m``. Empty segments (``m == 0``) contribute 0.
    Vectorized, so a whole scan of candidate splits is one call.
    """
    k = np.asarray(k, dtype=float)
    m = np.asarray(m, dtype=float)
    safe = np.where(m > 0, m, 1.0)
    p = _clip_p(k / safe)
    ll = k * np.log(p) + (m - k) * np.log1p(-p)
    return np.where(m > 0, ll, 0.0)


def fit_switch2(s: Sequence[int] | np.ndarray) -> dict:
    """Fit the two-switch model: three Bernoulli regimes split by an ordered pair.

    ``p1`` on ``[0, tau1)``, ``p2`` on ``[tau1, tau2)``, ``p3`` on ``[tau2, n)``, with
    ``1 <= tau1 < tau2 <= n-1``. Every segment rate is profiled out analytically (each is its
    segment mean), so the fit is an exhaustive maximization over the ordered pairs -- no
    optimizer, no local optima.

    The search is O(n^2) in candidate pairs but done in O(n) numpy calls: one loop over
    ``tau1``, and for each, a single vectorized scan over all valid ``tau2`` built from the
    same prefix sum ``switchpoint_loglik_profile`` uses.

    The regimes are gated non-decreasing (``p1 <= p2 <= p3``): pairs whose segment means are
    not monotone are excluded from the search, so a win reflects a staged progression toward
    SHORT rather than a transient spike (e.g. ``0 -> 1 -> 0`` over a few trials). This makes
    switch2 a *constrained* model: it does not nest the single switch or the constant, and
    when no monotone split exists (e.g. a strictly decreasing sequence) it is degenerate
    (``-inf``) and cannot win -- both intended.

    Returns
    -------
    dict
        ``tau1``, ``tau2`` (first trial of the 2nd and 3rd regime), ``p1``, ``p2``, ``p3``
        (with ``p1 <= p2 <= p3``), ``loglik``, ``k_params`` (5). Degenerate (``tau`` 0, NaN
        rates, ``-inf``) when ``n < 3`` (too few trials for three non-empty regimes) or when no
        monotone split exists.
    """
    s = _as_binary(s)
    n = s.size
    degenerate = {"tau1": 0, "tau2": 0, "p1": float("nan"), "p2": float("nan"),
                  "p3": float("nan"), "loglik": float("-inf"), "k_params": 5}
    if n < 3:
        return degenerate

    prefix = np.concatenate(([0], np.cumsum(s)))  # prefix[i] = successes in s[:i]
    total = float(prefix[-1])
    best = (-np.inf, 0, 0)
    for tau1 in range(1, n - 1):
        tau2 = np.arange(tau1 + 1, n)  # tau2 > tau1, and leaves a non-empty 3rd regime
        m1, m2, m3 = tau1, tau2 - tau1, n - tau2  # regime sizes (m1 scalar, m2/m3 arrays)
        k1, k2, k3 = prefix[tau1], prefix[tau2] - prefix[tau1], total - prefix[tau2]
        p1, p2, p3 = k1 / m1, k2 / m2, k3 / m3  # segment ML rates, for the loglik and the gate
        ll = _segment_loglik(k1, m1) + _segment_loglik(k2, m2) + _segment_loglik(k3, m3)
        # Monotonicity gate: keep only non-decreasing regimes p1 <= p2 <= p3, so a win reflects
        # a staged progression toward SHORT and not a transient spike (e.g. 0 -> 1 -> 0 over a
        # few trials) that would otherwise let switch2 beat the constant on noise.
        ll = np.where((p1 <= p2) & (p2 <= p3), ll, -np.inf)
        j = int(np.argmax(ll))
        if ll[j] > best[0]:
            best = (float(ll[j]), tau1, int(tau2[j]))

    loglik, tau1, tau2 = best
    if not np.isfinite(loglik):  # no monotone split exists (or n < 3): switch2 cannot win
        return degenerate
    return {"tau1": tau1, "tau2": tau2, "p1": float(s[:tau1].mean()),
            "p2": float(s[tau1:tau2].mean()), "p3": float(s[tau2:].mean()),
            "loglik": loglik, "k_params": 5}


def logistic_p(x: np.ndarray, midpoint: float, slope: float, lo: float, hi: float) -> np.ndarray:
    """Per-trial success probability of the logistic model (asymptotes ``lo`` and ``hi``).

    Exposed so a caller can draw the fitted curve on an arbitrary trial grid.
    """
    z = np.clip(slope * (x - midpoint), -_Z_MAX, _Z_MAX)
    return lo + (hi - lo) * expit(z)


def logistic_start_points(s: Sequence[int] | np.ndarray) -> list[dict]:
    """The initial conditions the logistic is fitted from -- the single source of truth.

    The likelihood surface has flat regions (any ``slope`` looks the same once the asymptotes
    collapse) and more than one basin, so a single warm start can settle in a local optimum.
    The set is therefore:

    - the switch-point warm start (``midpoint = tau``, asymptotes at ``p1`` / ``p2``), kept
      first so callers can identify it (its label is ``WARM_START_LABEL``);
    - a grid of dispersed starts: midpoints at the 10/30/50/70/90% quantiles of the trial
      axis, both asymptotes at the global SHORT rate, and initial slopes spanning
      shallow-gradual to steep-near-step.

    ``fit_logistic`` minimizes from every one of these, and the multi-start diagnostic replays
    exactly the same set -- define new starts here and nowhere else.

    Returns
    -------
    list[dict]
        One dict per start: ``label``, ``midpoint``, ``slope``, ``lo``, ``hi``, and ``theta``
        (the packed ``[midpoint, slope, logit(lo), logit(hi)]`` the optimizer takes).
    """
    s = _as_binary(s)
    n = max(s.size, 1)
    rate = float(_clip_p(s.mean())) if s.size else 0.5
    switch = fit_switchpoint(s)
    lo0 = float(_clip_p(switch["p1"])) if np.isfinite(switch["p1"]) else rate
    hi0 = float(_clip_p(switch["p2"])) if np.isfinite(switch["p2"]) else rate

    starts = [{"label": WARM_START_LABEL, "midpoint": float(switch["tau"]), "slope": 1.0,
               "lo": lo0, "hi": hi0}]
    starts += [{"label": f"q{int(q * 100):02d}/slope{slope:g}", "midpoint": float(q * (n - 1)),
                "slope": float(slope), "lo": rate, "hi": rate}
               for q in _LOGISTIC_MIDPOINT_QUANTILES for slope in _LOGISTIC_INITIAL_SLOPES]
    for start in starts:
        start["theta"] = np.array([start["midpoint"], start["slope"],
                                   logit(start["lo"]), logit(start["hi"])])
    return starts


def _fit_logistic_from(s: np.ndarray, x: np.ndarray, theta0: np.ndarray) -> dict:
    """Run one Nelder-Mead minimization of the logistic negative log-likelihood."""
    def negative_loglik(theta: np.ndarray) -> float:
        p = logistic_p(x, theta[0], theta[1], expit(theta[2]), expit(theta[3]))
        return -bernoulli_loglik(s, p)

    result = minimize(negative_loglik, theta0, method="Nelder-Mead",
                      options={"maxiter": 4000, "xatol": 1e-6, "fatol": 1e-8})
    midpoint, slope, lo_raw, hi_raw = result.x
    return {"midpoint": float(midpoint), "slope": float(slope), "lo": float(expit(lo_raw)),
            "hi": float(expit(hi_raw)), "loglik": float(-result.fun),
            "converged": bool(result.success)}


def fit_logistic_multistart(s: Sequence[int] | np.ndarray) -> list[dict]:
    """Fit the logistic from every start in ``logistic_start_points``, keeping them all.

    ``fit_logistic`` returns only the winner; this keeps the whole picture, so a caller can
    see whether the starts funnel into one optimum or split into basins.

    Returns
    -------
    list[dict]
        One dict per start, in the order of ``logistic_start_points``: the start's ``label``,
        ``initial_midpoint``, ``initial_slope``, ``initial_lo``, ``initial_hi``, plus the
        converged ``midpoint``, ``slope``, ``lo``, ``hi``, ``loglik``, ``converged``.
        Empty when ``n < 2``.
    """
    s = _as_binary(s)
    if s.size < 2:
        return []
    x = np.arange(s.size, dtype=float)
    return [{"label": start["label"], "initial_midpoint": start["midpoint"],
             "initial_slope": start["slope"], "initial_lo": start["lo"], "initial_hi": start["hi"],
             **_fit_logistic_from(s, x, start["theta"])}
            for start in logistic_start_points(s)]


def fit_logistic(s: Sequence[int] | np.ndarray) -> dict:
    """Fit the graded-change model by maximum likelihood, multi-start Nelder-Mead.

    ``p_i = lo + (hi - lo) * sigmoid(slope * (i - midpoint))``. The asymptotes are fitted
    rather than fixed at 0/1, so the model nests the switch model (as ``slope -> inf``) and
    ``slope`` is directly interpretable as abruptness. Minimized from every start in
    ``logistic_start_points``; the fit with the highest log-likelihood wins.

    Because the logistic nests the step, its optimum must be at least as good as the switch
    model's. A shortfall means every start got stuck, so it is warned about rather than
    returned silently -- it is an optimization failure, not a finding.

    Returns
    -------
    dict
        ``midpoint``, ``slope`` (abruptness; negative means SHORT was abandoned), ``lo``,
        ``hi``, ``loglik``, ``k_params`` (4), ``converged`` (True iff at least one start
        converged), and -- identifying which multi-start actually won -- ``start_label`` (the
        ``logistic_start_points`` label, e.g. ``WARM_START_LABEL`` or ``"q30/slope0.5"``),
        ``initial_midpoint`` and ``initial_slope`` (that start's initial conditions).
    """
    s = _as_binary(s)
    if s.size < 2:
        return {"midpoint": 0.0, "slope": 0.0, "lo": 0.5, "hi": 0.5,
                "loglik": float("-inf"), "k_params": 4, "converged": False,
                "start_label": None, "initial_midpoint": float("nan"),
                "initial_slope": float("nan")}
    fits = fit_logistic_multistart(s)
    best = max(fits, key=lambda fit: fit["loglik"])
    switch_loglik = fit_switchpoint(s)["loglik"]
    if best["loglik"] < switch_loglik - _LOGLIK_TOL:
        warnings.warn(
            f"logistic loglik {best['loglik']:.6f} < switch loglik {switch_loglik:.6f} "
            f"(shortfall {switch_loglik - best['loglik']:.3e}). The logistic nests the step "
            f"model, so every multi-start attempt got stuck: this is an optimization failure, "
            f"not a real result.", stacklevel=2)
    return {"midpoint": best["midpoint"], "slope": best["slope"], "lo": best["lo"],
            "hi": best["hi"], "loglik": best["loglik"], "k_params": 4,
            "converged": any(fit["converged"] for fit in fits),
            "start_label": best["label"], "initial_midpoint": best["initial_midpoint"],
            "initial_slope": best["initial_slope"]}
