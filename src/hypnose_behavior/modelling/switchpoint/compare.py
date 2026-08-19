"""Fit every model and score them with AIC / BIC, plus per-trial fitted-P reconstruction.

The scoring layer over the model families in ``switch.py`` and ``qlearning.py``: it fits them
all, penalizes their parameter counts, and names the winner. ``_model_fitted_p`` rebuilds the
per-trial P(SHORT) curve a fitted model implies -- used by the residual-autocorrelation
diagnostic (``autocorr.py``) and by the plots.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from hypnose_behavior.modelling.switchpoint.switch import (
    _as_binary,
    fit_constant,
    fit_logistic,
    fit_switch2,
    fit_switchpoint,
    logistic_p,
)
from hypnose_behavior.modelling.switchpoint.qlearning import QLEARN_DEFAULT_VARIANT, fit_qlearning

# Models in increasing flexibility, for tables, plots and the printed logliks. The nesting
# relations constant <= switch and switch <= logistic should hold along it; switch2 is a
# monotone-gated (p1 <= p2 <= p3) special case that need not nest the single switch.
# ``qlearning`` is not part of that chain at all -- it is the mechanistic null, scored on the
# same per-trial choices so its AIC / BIC are directly comparable.
MODEL_ORDER = ("constant", "switch", "logistic", "switch2", "qlearning")

__all__ = ["compare_models", "model_fitted_p", "MODEL_ORDER"]


def compare_models(s: Sequence[int] | np.ndarray, qlearning_fit: Optional[dict] = None) -> dict:
    """Fit all five models and score them with AIC and BIC (lower is better).
    
    Returns
    -------
    dict
        One entry per model name, each ``{loglik, k_params, aic, bic}``; ``best_aic`` and
        ``best_bic`` (the winning implemented model); and ``fits``, the full fit dict of each.

    Raises
    ------
    ValueError
        ``qlearning_fit`` was fitted to a sequence of a different length.
    """
    s = _as_binary(s)
    n = max(s.size, 1)
    if qlearning_fit is not None and qlearning_fit["n_trials"] != s.size:
        raise ValueError(
            f"qlearning_fit was fitted to {qlearning_fit['n_trials']} trials but s has "
            f"{s.size}; it must be a fit of this same sequence.")
    fits = {"constant": fit_constant(s), "switch": fit_switchpoint(s),
            "logistic": fit_logistic(s), "switch2": fit_switch2(s),
            "qlearning": fit_qlearning(s) if qlearning_fit is None else qlearning_fit}
    scores = {}
    for name, fit in fits.items():
        k, loglik = fit["k_params"], fit["loglik"]
        scores[name] = {"loglik": loglik, "k_params": k,
                        "aic": 2 * k - 2 * loglik, "bic": k * np.log(n) - 2 * loglik}
    eligible = [name for name, fit in fits.items() if fit.get("implemented", True)]
    scores["best_aic"] = min(eligible, key=lambda m: scores[m]["aic"])
    scores["best_bic"] = min(eligible, key=lambda m: scores[m]["bic"])
    scores["fits"] = fits
    return scores


def model_fitted_p(name: str, fit: dict, n: int) -> Optional[np.ndarray]:
    """Per-trial fitted P(SHORT) of a fitted model over the continuous ``0..n-1`` trial axis.

    Rebuilds the step / curve each model implies at every trial, matching exactly the shapes
    the model-comparison plot draws. Returns ``None`` for a model with no per-trial curve, so a
    caller can skip it.

    ``qlearning`` is the one model whose curve is not a function of the trial index: it is
    driven by the animal's own choice history, so it exists only on the trials that were
    actually fitted and is returned as stored rather than re-evaluated on ``x``. What is
    returned is its **one-step-ahead** trajectory -- conditioned on the observed choices, which
    is what the residual-ACF diagnostic and the likelihood want. It is *not* the model's
    prediction of the animal's trajectory; for that see ``qlearning_generative_band``, and note
    that residuals formed against a one-step-ahead curve are not comparable with those of the
    descriptive models (the Q-learner absorbs some of the serial structure being measured).
    ``None`` for a degenerate (all-NaN) fit.
    """
    x = np.arange(n)
    if name == "constant":
        return np.full(n, fit["p"], dtype=float)
    if name == "switch":
        return np.where(x < fit["tau"], fit["p1"], fit["p2"]).astype(float)
    if name == "switch2":
        p = np.full(n, fit["p3"], dtype=float)
        p[x < fit["tau2"]] = fit["p2"]
        p[x < fit["tau1"]] = fit["p1"]
        return p
    if name == "logistic":
        return logistic_p(x.astype(float), fit["midpoint"], fit["slope"], fit["lo"], fit["hi"])
    if name == "qlearning":
        p = np.asarray(fit.get("p_short", ()), dtype=float)
        return p if p.size == n and not np.all(np.isnan(p)) else None
    return None
