"""Q-learning account of the strategy change -- the mechanistic NULL to be rejected.

A mechanistic sibling to the descriptive switch-point family in ``switch.py``: rather than
describing the P(SHORT) curve, it derives that curve from a trial-by-trial value update. Its
role here is adversarial. Incremental reinforcement learning produces a *gradual* rise in
P(SHORT) whose steepness is set by the learning rate, so if a Q-learner fits an animal as well
as the step does, the "sudden strategy switch" reading is not supported. The three variants
below are therefore fitted at their best (multi-start, see ``fit_qlearning``) -- a strawmanned
null that loses proves nothing.

Everything here is pure numeric (numpy + scipy): numpy arrays in, numpy arrays and plain dicts
out. No file I/O, no plotting. Figures live in
``hypnose_behavior.visualization.modelling.switchpoint.plots``; the orchestration is in
``scripts/modelling/switchpoint_analysis.py``.

Model
-----
Two options, SHORT and LONG, with **fixed** rewards ``r_short = 1``, ``r_long = 0``
(``R_SHORT`` / ``R_LONG``). Only the chosen option updates::

    Q[chosen] += alpha * (r[chosen] - Q[chosen])

and the choice rule is a softmax with a perseveration (choice-history) term::

    P(SHORT at t) = 1 / (1 + exp(-(b * (Q_short - Q_long) + kappa * s_prev)))

where ``s_prev`` is ``+1`` if trial ``t-1`` was SHORT, ``-1`` if it was LONG, and ``0`` at the
first trial. ``kappa = 0`` for the non-perseveration variants. Every trial's probability is
computed from the Q values held *before* that trial's update; the update is then applied using
the route the animal actually took.

Why the rewards are constants and not parameters
------------------------------------------------
Fixing ``(r_short, r_long) = (1, 0)`` is a **choice of units, not a claim that LONG is
unrewarded** -- both routes are rewarded in the real task, and the sequence modelled here is
already conditioned on reward. The true reward advantage ``d`` of SHORT over LONG is
*unidentifiable* from choice data, because it enters the choice rule only through the product
``b * d``: the fitted inverse temperature ``b`` absorbs it.

Concretely, writing the rewards as ``(1, 1 - d)`` and mapping every value through the affine
map ``x -> d * x + (1 - d)`` (which sends ``0 -> 1 - d`` and fixes ``1``) scales every value
difference ``Q_short - Q_long`` by exactly ``d``. So::

    rewards (1, 0)      with (alpha, b,     Q0_short,               Q0_long)
    rewards (1, 1 - d)  with (alpha, b / d, d*Q0_short + (1 - d),   d*Q0_long + (1 - d))

produce the **identical** P(SHORT) trajectory and therefore the identical likelihood. Fitting
``d`` as a free parameter would add a perfectly flat direction to the likelihood surface, not
information. The reward scale is exposed as a keyword on ``qlearning_trajectory`` /
``qlearning_nll`` only so this claim can be checked numerically (see
``src/hypnose_behavior/qc/check_qlearning.py``); the fits always use the defaults.

Variants
--------
=========================  =================================  ==========================
variant                    free parameters                    ``Q0`` bounds
=========================  =================================  ==========================
``qlearn_free``            alpha, b, Q0_short, Q0_long        ``[-10, 10]``
``qlearn_constrained``     alpha, b, Q0_short, Q0_long        ``[0, 1]``
``qlearn_perseveration``   alpha, b, Q0_short, Q0_long, kappa ``[-10, 10]``, kappa
                                                              ``[-10, 10]``
=========================  =================================  ==========================

``alpha`` is bounded ``[1e-4, 1]`` and ``b`` ``[1e-3, 50]`` throughout.

``qlearn_constrained`` reads "initial values lie within the range of experienceable outcomes"
-- the animal cannot start out believing an option is worth more than any outcome it could
ever receive. That constraint has a structural consequence worth knowing before reading any
fit: **once ``Q_long`` has converged, the constrained model cannot hold P(SHORT) below 0.5**,
because ``Q_long -> 0`` while ``Q0_short >= 0`` keeps ``Q_short >= 0``. It can only describe
an animal that ends up at or above chance for SHORT.

That floor is *asymptotic*, and the escape route matters. ``Q_long`` needs ``n >> 1/alpha``
LONG choices to converge, and ``alpha`` is bounded below at ``1e-4`` -- i.e. 10,000 trials,
beyond any real dataset here. So a constrained fit to an animal that sits below chance for
SHORT will buy its sub-0.5 plateau by pinning ``alpha`` at that floor, never reaching the
asymptotic regime at all. The expected signature is therefore ``boundary_hit`` on ``alpha``,
and it costs the model the transition: an ``alpha`` that small also freezes ``Q_short``, so no
rise in P(SHORT) is possible either. Read a constrained fit with ``alpha`` at its bound as
"this variant cannot describe this animal", not as a learning rate.

The free version may initialise outside the experienceable range and has no such floor.

Two kinds of trajectory -- do not confuse them
----------------------------------------------
``qlearning_trajectory`` returns the **one-step-ahead** P(SHORT): at every trial the model is
handed the animal's true history, because both the update counts and ``s_prev`` are read off
the observed choices. That is the right quantity for the likelihood, for AIC / BIC, and for the
residual ACF -- and it is *not* a prediction of the animal's trajectory. It cannot be, because
it is conditioned on the trajectory. Plotting it alone is misleading: with a large fitted
``kappa`` the choice rule collapses to roughly ``expit(kappa * s_prev)``, a one-trial-lagged
copy of the data, so the curve will appear to track the switch perfectly no matter how badly
the value-learning part of the model is doing.

``qlearning_generative_band`` returns the **generative** trajectory: the model run forward on
its *own* choices, averaged over many simulations, with a quantile band. That is what the
fitted Q-learner actually predicts an animal would do, and it is the one to read when asking
whether the null can reproduce an abrupt switch. It gets its own figure
(``plot_qlearning_generative``) for exactly this reason, kept apart from the one-step-ahead fit
that the model-comparison figure draws, so the two are never mistaken for each other.

Closed form
-----------
Because each option's reward is a constant, the update ``Q += alpha * (r - Q)`` is a geometric
recursion with the exact solution ``Q_k = r - (r - Q0) * (1 - alpha)**k`` after ``k`` updates
*of that option*. The Q values entering trial ``t`` therefore depend only on how many SHORT
and LONG choices preceded it, so the whole one-step-ahead trajectory is computed without a
Python loop. ``simulate_qlearning`` still steps trial by trial, because there the choices are
being drawn; the two agree exactly, which is asserted in the qc check. Its loop is vectorized
across simulations rather than across trials, so drawing a 500-simulation generative band costs
one pass over the trials, not 500.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

from hypnose_behavior.modelling.switchpoint.switch import _as_binary, fit_switchpoint

# Fixed reward constants -- a choice of units, NOT fitted. See the module docstring.
R_SHORT = 1.0
R_LONG = 0.0

# Probability clip, so a saturated softmax cannot produce an infinite log-likelihood.
_P_EPS = 1e-10

# Shared parameter bounds.
_ALPHA_BOUNDS = (1e-4, 1.0)
_B_BOUNDS = (1e-3, 50.0)
_Q0_BOUNDS_FREE = (-10.0, 10.0)
_Q0_BOUNDS_EXPERIENCEABLE = (0.0, 1.0)  # "within the range of experienceable outcomes"
_KAPPA_BOUNDS = (-10.0, 10.0)

# Multi-start: alpha and Q0 trade off against each other (a small alpha with an extreme Q0
# mimics a large alpha with a moderate one), so the surface is not unimodal and a single start
# is not reliable. Starts are drawn inside the bounds from a seeded generator: log-uniformly
# for the parameters in _LOG_SCALE_PARAMS, uniformly for the rest.
N_STARTS = 32

# An estimate this close to either end of its bound range (as a fraction of the range) is
# flagged: the optimizer stopped at the edge of the parameter space, so the value is a
# constraint artefact rather than an estimate.
_BOUNDARY_RTOL = 0.01

# Proximity to a bound is measured on each parameter's natural scale. alpha and b span orders
# of magnitude (b runs 1e-3 .. 50), so a linear window would be 0.5 wide and would flag an
# ordinary b = 0.4 as an artefact; on the log scale the same 1% window is a factor of ~1.11.
# Q0 and kappa are ordinary linear quantities and are left alone.
_LOG_SCALE_PARAMS = ("alpha", "b")

# Simulations behind the generative band, and the quantiles it spans.
N_GENERATIVE_SIMS = 500
GENERATIVE_QUANTILES = (0.05, 0.95)
# Individual runs kept alongside the mean. A perseverative Q-learner steps abruptly at a
# different trial in every run, so the mean of many runs is a smooth ramp that misrepresents
# what any single run does; these are drawn next to it.
N_GENERATIVE_EXAMPLES = 8
# Minimum rise in fitted rate (p2 - p1) for a simulated run to count as having switched.
# fit_switchpoint returns a tau for every sequence, switched or not, so without this gate a run
# that never changed strategy still contributes a tau and inflates the apparent spread.
SWITCH_THRESHOLD = 0.5

# The variant that represents "qlearning" in compare_models. The other two are fitted alongside
# by fit_qlearning_variants and overlaid on the figure rather than taking a table row each.
QLEARN_DEFAULT_VARIANT = "qlearn_free"

QLEARN_VARIANTS = {
    "qlearn_free": {
        "params": ("alpha", "b", "q0_short", "q0_long"),
        "bounds": (_ALPHA_BOUNDS, _B_BOUNDS, _Q0_BOUNDS_FREE, _Q0_BOUNDS_FREE),
        "description": "Q0 free to start outside the experienceable range",
    },
    "qlearn_constrained": {
        "params": ("alpha", "b", "q0_short", "q0_long"),
        "bounds": (_ALPHA_BOUNDS, _B_BOUNDS, _Q0_BOUNDS_EXPERIENCEABLE,
                   _Q0_BOUNDS_EXPERIENCEABLE),
        "description": "Q0 within the range of experienceable outcomes [0, 1]",
    },
    "qlearn_perseveration": {
        "params": ("alpha", "b", "q0_short", "q0_long", "kappa"),
        "bounds": (_ALPHA_BOUNDS, _B_BOUNDS, _Q0_BOUNDS_FREE, _Q0_BOUNDS_FREE, _KAPPA_BOUNDS),
        "description": "Q0 free, plus a choice-history (perseveration) term kappa",
    },
}

# Fixed order for tables, plots and the printed logliks.
QLEARN_VARIANT_ORDER = ("qlearn_free", "qlearn_constrained", "qlearn_perseveration")

# Default (alpha, b) grid for the parameter-sweep figure -- 4 x 4, geometrically spaced inside
# the fitted bounds so slow-drift through near-immediate learning is covered.
QLEARN_SWEEP_ALPHAS = (0.005, 0.03, 0.15, 0.75)
QLEARN_SWEEP_BS = (0.25, 1.0, 4.0, 16.0)

__all__ = [
    "R_SHORT",
    "R_LONG",
    "N_STARTS",
    "N_GENERATIVE_SIMS",
    "N_GENERATIVE_EXAMPLES",
    "GENERATIVE_QUANTILES",
    "SWITCH_THRESHOLD",
    "QLEARN_DEFAULT_VARIANT",
    "QLEARN_VARIANTS",
    "QLEARN_VARIANT_ORDER",
    "QLEARN_SWEEP_ALPHAS",
    "QLEARN_SWEEP_BS",
    "qlearning_trajectory",
    "qlearning_nll",
    "simulate_qlearning",
    "qlearning_generative_band",
    "fit_qlearning",
    "fit_qlearning_variants",
    "qlearning_parameter_sweep",
]


def _as_one_animal(s: Sequence[int] | np.ndarray) -> np.ndarray:
    """Coerce ``s`` to one animal's flat 0/1 sequence, refusing pooled or averaged input.

    The Q-learning fits are per animal (or per session) by construction. Averaging animals
    that switched at different trials manufactures a gradual curve out of abrupt ones -- which
    is precisely the conclusion the null model is being used to test -- so an aggregate is
    refused loudly rather than fitted.
    """
    arr = np.asarray(s)
    if arr.ndim > 1:
        raise ValueError(
            f"expected ONE animal's 0/1 sequence, got an array of shape {arr.shape}. Fitting "
            f"pooled or stacked sequences is not supported: averaging animals with different "
            f"switch points manufactures a gradual curve out of abrupt ones, which is exactly "
            f"what this null model is meant to test. Fit each animal (or session) separately.")
    if arr.size and not np.isin(arr, (0, 1)).all():
        raise ValueError(
            "s must contain only 0 and 1. Group means / rolling averages cannot be fitted: "
            "the likelihood is over individual SHORT/LONG choices of ONE animal.")
    return _as_binary(arr)


def _trajectory_unchecked(s: np.ndarray, alpha: float, b: float, q0_short: float,
                          q0_long: float, kappa: float = 0.0,
                          rewards: tuple[float, float] = (R_SHORT, R_LONG)) -> dict:
    """``qlearning_trajectory`` without the input validation -- ``s`` must already be 0/1 int.

    The optimizer evaluates this tens of thousands of times per fit (n_starts x iterations x
    finite-difference steps), and re-running ``np.isin(s, (0, 1)).all()`` on every one of them
    is pure overhead: ``s`` is fixed for the whole fit and is validated once by the caller.
    """
    n = s.size
    r_short, r_long = float(rewards[0]), float(rewards[1])
    # Updates of each option BEFORE trial t: the closed form needs only these counts.
    n_short_before = np.concatenate(([0], np.cumsum(s)[:-1])) if n else np.zeros(0, dtype=int)
    n_long_before = np.arange(n) - n_short_before
    decay = 1.0 - float(alpha)
    q_short = r_short - (r_short - float(q0_short)) * decay ** n_short_before
    q_long = r_long - (r_long - float(q0_long)) * decay ** n_long_before
    # +1 after a SHORT trial, -1 after a LONG one, 0 at the first trial (no history yet).
    s_prev = np.concatenate(([0.0], 2.0 * s[:-1] - 1.0)) if n else np.zeros(0)
    p_short = np.clip(expit(float(b) * (q_short - q_long) + float(kappa) * s_prev),
                      _P_EPS, 1.0 - _P_EPS)
    return {"p_short": p_short, "q_short": q_short, "q_long": q_long, "s_prev": s_prev}


def qlearning_trajectory(s: Sequence[int] | np.ndarray, alpha: float, b: float,
                         q0_short: float, q0_long: float, kappa: float = 0.0,
                         rewards: tuple[float, float] = (R_SHORT, R_LONG)) -> dict:
    """**One-step-ahead** per-trial P(SHORT) of a Q-learner that made the choices in ``s``.

    Each trial's probability uses the Q values held *before* that trial's update; the update
    then applies to the option the animal actually took, and only to that option. Evaluated in
    closed form (see the module docstring), so it is O(n) numpy with no Python loop.

    This is conditioned on the animal's own history at every trial -- both the update counts and
    ``s_prev`` are read off ``s``. It is the right quantity for the likelihood, AIC / BIC and
    the residual ACF, and it is **not** a prediction of the animal's trajectory: a large fitted
    ``kappa`` makes it approximately a one-trial-lagged copy of ``s``, which will look like a
    perfect fit to any switch. For what the fitted model actually predicts, use
    ``qlearning_generative_band``.

    Parameters
    ----------
    s : array_like of {0, 1}
        The observed choices, 1 = SHORT. Only the counts of preceding choices and the previous
        trial's identity enter, so this is the "given the animal's actual history" trajectory,
        not a free-running simulation (that is ``simulate_qlearning``).
    alpha, b, q0_short, q0_long, kappa : float
        Learning rate, inverse temperature, initial values, perseveration weight.
    rewards : (float, float)
        ``(r_short, r_long)``. Exposed only so the identifiability claim in the module
        docstring can be checked numerically; every fit uses the default ``(1, 0)``.

    Returns
    -------
    dict
        ``p_short`` (length n, clipped into ``[1e-10, 1 - 1e-10]``), ``q_short``, ``q_long``
        (the values held *before* each trial's update), and ``s_prev`` (+1/-1/0).
    """
    return _trajectory_unchecked(_as_one_animal(s), alpha, b, q0_short, q0_long, kappa, rewards)


def _nll_unchecked(s: np.ndarray, alpha: float, b: float, q0_short: float, q0_long: float,
                   kappa: float = 0.0,
                   rewards: tuple[float, float] = (R_SHORT, R_LONG)) -> tuple[float, np.ndarray]:
    """``qlearning_nll`` without the input validation -- ``s`` must already be 0/1 int.

    This is the optimizer's inner loop; see ``_trajectory_unchecked``.
    """
    p = _trajectory_unchecked(s, alpha, b, q0_short, q0_long, kappa, rewards)["p_short"]
    if s.size == 0:
        return 0.0, p
    return -float(np.sum(np.where(s == 1, np.log(p), np.log1p(-p)))), p


def qlearning_nll(s: Sequence[int] | np.ndarray, alpha: float, b: float, q0_short: float,
                  q0_long: float, kappa: float = 0.0,
                  rewards: tuple[float, float] = (R_SHORT, R_LONG)) -> tuple[float, np.ndarray]:
    """Negative summed log-likelihood of the observed sequence, plus its P(SHORT) trajectory.

    ``-sum_t log P(observed choice at t)``, directly comparable with the ``-loglik`` of the
    descriptive models in ``switch.py`` because it scores exactly the same per-trial Bernoulli
    choices. The trajectory returned is the one-step-ahead one -- see ``qlearning_trajectory``.

    Returns
    -------
    (float, np.ndarray)
        ``(nll, p_short)``. ``nll`` is ``0.0`` for an empty sequence.
    """
    return _nll_unchecked(_as_one_animal(s), alpha, b, q0_short, q0_long, kappa, rewards)


def simulate_qlearning(n_trials: int, alpha: float, b: float, q0_short: float, q0_long: float,
                       kappa: float = 0.0, seed: Optional[int] = None,
                       n_sims: Optional[int] = None,
                       rewards: tuple[float, float] = (R_SHORT, R_LONG)) -> tuple[np.ndarray, np.ndarray]:
    """Draw SHORT/LONG sequences from the Q-learner, stepping trial by trial.

    The **generative** direction of ``qlearning_trajectory``: each trial's choice is sampled
    from the current P(SHORT), and only the sampled option is then updated -- so the values,
    and hence the next trial's probability, depend on what was actually drawn. Nothing here is
    conditioned on an observed sequence, which is what makes this (and not
    ``qlearning_trajectory``) a prediction of what an animal would do.

    The trial loop is vectorized **across simulations**, not across trials: it cannot be
    vectorized across trials, since trial ``t+1`` depends on the draw at ``t``. Drawing 500
    simulations therefore costs one pass over the trials rather than 500.

    Parameters
    ----------
    n_trials : int
        Length of each sequence to draw.
    alpha, b, q0_short, q0_long, kappa : float
        As in ``qlearning_trajectory``.
    seed : int | None
        Seed for ``np.random.default_rng``, for a reproducible draw.
    n_sims : int | None
        Draw this many independent sequences at once, returning ``(n_sims, n_trials)`` arrays.
        ``None`` (default) draws one and returns 1-D arrays -- and returns *exactly* what
        ``n_sims=1`` returns for the same seed, since one simulation consumes the same numbers
        from the generator either way.
    rewards : (float, float)
        ``(r_short, r_long)``, as on ``qlearning_trajectory`` -- present so the two agree at any
        reward scale, not only the default ``(1, 0)``. Every fit uses the default.

    Returns
    -------
    (np.ndarray, np.ndarray)
        ``(s, p_short)`` -- the drawn 0/1 choices, and the P(SHORT) each was drawn from
        (i.e. the value held *before* that trial's update).
    """
    n = int(n_trials)
    m = 1 if n_sims is None else int(n_sims)
    r_short, r_long = float(rewards[0]), float(rewards[1])
    rng = np.random.default_rng(seed)
    s = np.zeros((m, n), dtype=np.int64)
    p_short = np.zeros((m, n), dtype=float)
    q_short = np.full(m, float(q0_short))
    q_long = np.full(m, float(q0_long))
    s_prev = np.zeros(m)
    for t in range(n):
        p = np.clip(expit(b * (q_short - q_long) + kappa * s_prev), _P_EPS, 1.0 - _P_EPS)
        p_short[:, t] = p
        chose_short = rng.random(m) < p
        s[:, t] = chose_short
        # Only the chosen option updates; the other is carried forward untouched.
        q_short = np.where(chose_short, q_short + alpha * (r_short - q_short), q_short)
        q_long = np.where(chose_short, q_long, q_long + alpha * (r_long - q_long))
        s_prev = np.where(chose_short, 1.0, -1.0)
    return (s, p_short) if n_sims is not None else (s[0], p_short[0])


def qlearning_generative_band(fit: dict, n_trials: int, n_sims: int = N_GENERATIVE_SIMS,
                              seed: int = 0,
                              quantiles: tuple[float, float] = GENERATIVE_QUANTILES,
                              n_examples: int = N_GENERATIVE_EXAMPLES,
                              switch_threshold: float = SWITCH_THRESHOLD) -> dict:
    """What the fitted Q-learner actually predicts: mean P(SHORT) over its own simulated runs.

    Runs ``simulate_qlearning`` ``n_sims`` times at ``fit``'s estimates and summarizes the
    resulting P(SHORT) trajectories. Unlike ``fit["p_short"]`` this is **not** conditioned on
    the animal's choices, so it is the honest answer to "can this fitted null reproduce the
    observed switch?" -- and the one to plot as the model's trajectory.

    The spread is genuine model uncertainty, not estimation error: each simulation makes its own
    choices, so a Q-learner whose transition timing is loosely determined produces a wide band
    even at a well-identified parameter set.

    **The mean alone can mislead, and for ``qlearn_perseveration`` it does.** A strongly
    perseverative Q-learner does not drift: each run locks onto one option and flips abruptly,
    at a trial that differs from run to run. Averaging 500 such step functions gives a smooth
    ramp, which reads as "perseveration predicts gradual change" -- the opposite of what the
    model does. ``examples`` and ``switch_taus`` are returned so that the individual runs, and
    the spread of their switch trials, can be seen next to the mean.

    **A tau alone does not mean a run switched.** ``fit_switchpoint`` returns its best split for
    every sequence, including one that never changed strategy, so a scattered ``switch_taus``
    cannot by itself tell "abrupt switch at a random trial" from "no switch, tau is noise". A
    run therefore counts as switched only when its fitted rates rise by at least
    ``switch_threshold`` (``p2 - p1 >= switch_threshold``), and the spread that means anything
    is the one over ``switch_taus_switched``.

    ``frac_switched`` is itself a result, not a filtering detail, and should be reported rather
    than quietly dropped: a perseverative fit that only reaches criterion in a small minority of
    runs is failing to reproduce the animal in a *different* way from one that switches every
    time at an unpredictable trial, and the two are indistinguishable from the tau spread alone.

    Parameters
    ----------
    fit : dict
        A ``fit_qlearning`` result (needs ``alpha``, ``b``, ``q0_short``, ``q0_long``, ``kappa``).
    n_trials : int
        Length of each simulated run -- normally the animal's trial count.
    n_sims : int
        Simulations to average over.
    seed : int
        Seed for the draw, so the band is reproducible.
    quantiles : (float, float)
        Lower and upper quantile of the band.
    n_examples : int
        Individual runs to keep in ``examples``, taken as the first ``n_examples`` simulations
        (they are i.i.d., so the first few are a fair sample). Clipped to ``n_sims``.
    switch_threshold : float
        Minimum ``p2 - p1`` for a simulated run to count as having switched.

    Returns
    -------
    dict
        ``mean``, ``lo``, ``hi`` (each length ``n_trials``); ``examples``, an
        ``(n_examples, n_trials)`` array of individual P(SHORT) runs; ``n_sims`` and
        ``quantiles``.

        Per simulated sequence (each length ``n_sims``): ``switch_taus``, ``switch_p1`` and
        ``switch_p2`` from its ``fit_switchpoint``, and ``switched`` (bool,
        ``p2 - p1 >= switch_threshold``). Summarizing those: ``frac_switched``, and
        ``switch_taus_switched``, the taus of the switched runs only -- the spread to read.

        All-NaN curves, empty per-run arrays, ``frac_switched = nan`` and ``n_sims = 0`` for a
        degenerate fit, so a caller can skip drawing it.
    """
    n = max(int(n_trials), 0)
    lo_q, hi_q = float(quantiles[0]), float(quantiles[1])
    degenerate = {"mean": np.full(n, np.nan), "lo": np.full(n, np.nan),
                  "hi": np.full(n, np.nan), "examples": np.zeros((0, n)),
                  "switch_taus": np.zeros(0, dtype=int), "switch_p1": np.zeros(0),
                  "switch_p2": np.zeros(0), "switched": np.zeros(0, dtype=bool),
                  "frac_switched": float("nan"), "switch_taus_switched": np.zeros(0, dtype=int),
                  "n_sims": 0, "quantiles": (lo_q, hi_q)}
    estimates = [fit.get(name, np.nan) for name in
                 ("alpha", "b", "q0_short", "q0_long", "kappa")]
    if n == 0 or int(n_sims) < 1 or not np.all(np.isfinite(estimates)):
        return degenerate
    sims, p = simulate_qlearning(n, *estimates, seed=seed, n_sims=int(n_sims))
    lo, hi = np.quantile(p, (lo_q, hi_q), axis=0)
    # Where each individual run switched, on the same single-switch criterion the descriptive
    # model uses -- so a run's abruptness is measured the same way the animal's is. The fitted
    # rates come back too: fit_switchpoint returns a tau for every sequence, so without the
    # p2 - p1 gate a run that never switched contributes a meaningless tau to the spread.
    fits = [fit_switchpoint(sim) for sim in sims]
    taus = np.array([f["tau"] for f in fits], dtype=int)
    p1 = np.array([f["p1"] for f in fits], dtype=float)
    p2 = np.array([f["p2"] for f in fits], dtype=float)
    switched = (p2 - p1) >= float(switch_threshold)
    return {"mean": p.mean(axis=0), "lo": lo, "hi": hi,
            "examples": p[:max(int(n_examples), 0)], "switch_taus": taus,
            "switch_p1": p1, "switch_p2": p2, "switched": switched,
            "frac_switched": float(switched.mean()), "switch_taus_switched": taus[switched],
            "n_sims": int(n_sims), "quantiles": (lo_q, hi_q)}


def _unpack(variant: str, theta: Sequence[float]) -> dict:
    """Map a variant's packed parameter vector onto the full ``(alpha, b, q0, q0, kappa)`` set."""
    values = dict(zip(QLEARN_VARIANTS[variant]["params"], (float(v) for v in theta)))
    values.setdefault("kappa", 0.0)  # 0 for the non-perseveration variants
    return values


def _boundary_params(variant: str, values: dict) -> list[str]:
    """Names of the estimates sitting within ``_BOUNDARY_RTOL`` of either end of their bounds.

    Proximity is measured on each parameter's natural scale (``_LOG_SCALE_PARAMS`` are compared
    in logs). A linear window on ``b``, whose bounds span ``1e-3 .. 50``, would be 0.5 wide and
    would flag a perfectly ordinary ``b = 0.4`` as a constraint artefact; in logs the same 1%
    window is a factor of ~1.11 from either end. Non-finite estimates (a degenerate fit) are
    skipped rather than flagged.
    """
    spec = QLEARN_VARIANTS[variant]
    hit = []
    for name, (lo, hi) in zip(spec["params"], spec["bounds"]):
        value = values[name]
        if not np.isfinite(value):
            continue
        if name in _LOG_SCALE_PARAMS and min(lo, hi, value) > 0:
            value, lo, hi = np.log(value), np.log(lo), np.log(hi)
        tol = _BOUNDARY_RTOL * (hi - lo)
        if value - lo <= tol or hi - value <= tol:
            hit.append(name)
    return hit


def _degenerate_fit(variant: str, n: int) -> dict:
    """The fit dict for a sequence too short to fit (``n < 2``): NaN estimates, ``-inf`` loglik.

    ``p_short`` is all-NaN rather than zeros: there is no fitted trajectory, and zeros would
    read as "P(SHORT) = 0" and be drawn as a flat line along the SHORT row.
    """
    spec = QLEARN_VARIANTS[variant]
    k = len(spec["params"])
    return {"variant": variant, "alpha": float("nan"), "b": float("nan"),
            "q0_short": float("nan"), "q0_long": float("nan"), "kappa": float("nan"),
            "free_params": spec["params"], "nll": float("inf"), "loglik": float("-inf"),
            "aic": float("inf"), "bic": float("inf"), "n_trials": int(n), "k_params": k,
            "converged": False, "n_starts": 0, "n_starts_converged": 0, "boundary_hit": False,
            "boundary_params": [], "p_short": np.full(int(n), np.nan), "implemented": True}


def fit_qlearning(s: Sequence[int] | np.ndarray, variant: str = QLEARN_DEFAULT_VARIANT,
                  n_starts: int = N_STARTS, seed: int = 0) -> dict:
    """Fit one Q-learning variant to ONE animal's sequence by multi-start maximum likelihood.

    ``scipy.optimize.minimize`` with ``L-BFGS-B`` (the bounds are the model), started from
    ``n_starts`` points drawn inside the bounds from a seeded generator -- log-uniformly for
    the parameters in ``_LOG_SCALE_PARAMS`` (``alpha`` and ``b``, which span orders of
    magnitude), uniformly for ``Q0`` and ``kappa``; the lowest negative log-likelihood wins.
    Multi-start is not optional here: ``alpha`` and ``Q0`` trade off against each other, so the
    surface has more than one basin and a single start under-fits -- which would strawman the
    null against the descriptive models rather than test it.

    Per animal or per session, never pooled: an aggregate raises (see ``_as_one_animal``).

    Parameters
    ----------
    s : array_like of {0, 1}
        One animal's (or one session's) choices, 1 = SHORT.
    variant : str
        One of ``QLEARN_VARIANTS`` -- ``"qlearn_free"``, ``"qlearn_constrained"`` or
        ``"qlearn_perseveration"``.
    n_starts : int
        Random starting points. Must be at least 20.
    seed : int
        Seed for the starting-point draw, so a fit is reproducible.

    Returns
    -------
    dict
        ``variant``; the estimates ``alpha``, ``b``, ``q0_short``, ``q0_long``, ``kappa``
        (``kappa`` is exactly 0 and not free except in ``qlearn_perseveration``);
        ``free_params``; ``nll`` and ``loglik`` (``= -nll``); ``aic``, ``bic``; ``n_trials``,
        ``k_params``; ``converged`` (True iff at least one start converged),
        ``n_starts``, ``n_starts_converged``; ``boundary_hit`` and ``boundary_params`` (any
        estimate within 1% of a bound on its natural scale -- an edge-of-space artefact, not an
        estimate); ``p_short`` (the fitted per-trial **one-step-ahead** P(SHORT) trajectory --
        see ``qlearning_trajectory``, and use ``qlearning_generative_band`` for what the fit
        predicts); and ``implemented`` (True), so it scores in ``compare_models``.

    Raises
    ------
    ValueError
        Unknown ``variant``, fewer than 20 starts, or a pooled / non-binary sequence.
    """
    if variant not in QLEARN_VARIANTS:
        raise ValueError(f"variant must be one of {sorted(QLEARN_VARIANTS)}, got {variant!r}")
    if n_starts < 20:
        raise ValueError(f"n_starts must be >= 20 (alpha and Q0 trade off, so a small "
                         f"multi-start is unreliable), got {n_starts}")
    s = _as_one_animal(s)
    n = s.size
    if n < 2:
        return _degenerate_fit(variant, n)

    spec = QLEARN_VARIANTS[variant]
    bounds = spec["bounds"]
    k = len(spec["params"])

    def negative_loglik(theta: np.ndarray) -> float:
        # _nll_unchecked, not qlearning_nll: s was validated above and is fixed for the whole
        # fit, so re-validating it on every one of the ~10k evaluations is pure overhead.
        nll, _ = _nll_unchecked(s, **_unpack(variant, theta))
        return nll

    rng = np.random.default_rng(seed)
    lo = np.array([b[0] for b in bounds], dtype=float)
    hi = np.array([b[1] for b in bounds], dtype=float)
    starts = rng.uniform(lo, hi, size=(n_starts, k))
    for j, name in enumerate(spec["params"]):
        if name in _LOG_SCALE_PARAMS:
            starts[:, j] = np.exp(rng.uniform(np.log(lo[j]), np.log(hi[j]), n_starts))

    best, n_converged = None, 0
    for theta0 in starts:
        result = minimize(negative_loglik, theta0, method="L-BFGS-B", bounds=bounds)
        n_converged += int(bool(result.success))
        if best is None or result.fun < best.fun:
            best = result

    values = _unpack(variant, best.x)
    nll, p_short = _nll_unchecked(s, **values)
    loglik = -nll
    boundary = _boundary_params(variant, values)
    return {"variant": variant, **values, "free_params": spec["params"],
            "nll": float(nll), "loglik": float(loglik),
            "aic": float(2 * k - 2 * loglik), "bic": float(k * np.log(n) - 2 * loglik),
            "n_trials": int(n), "k_params": k, "converged": bool(n_converged > 0),
            "n_starts": int(n_starts), "n_starts_converged": int(n_converged),
            "boundary_hit": bool(boundary), "boundary_params": boundary,
            "p_short": p_short, "implemented": True}


def fit_qlearning_variants(s: Sequence[int] | np.ndarray,
                           variants: Sequence[str] = QLEARN_VARIANT_ORDER,
                           n_starts: int = N_STARTS, seed: int = 0) -> dict:
    """Fit every variant to the same sequence, keyed by variant name.

    Convenience over ``fit_qlearning`` for the overlay and sweep figures, which need all three.
    Each variant is fitted from the same ``seed``, so a rerun reproduces the same fits; the
    starting *points* are not comparable across variants, and are not meant to be.
    ``qlearn_perseveration`` draws a ``k = 5`` array rather than ``k = 4``, and
    ``qlearn_constrained``'s narrower ``Q0`` bounds map the same variates onto different
    values. Differences between the fits are therefore differences of model plus draw, which is
    fine here -- each variant is fitted at its own best over 32 starts, not raced from a shared
    initial condition.
    """
    return {variant: fit_qlearning(s, variant, n_starts=n_starts, seed=seed)
            for variant in variants}


def qlearning_parameter_sweep(s: Sequence[int] | np.ndarray, fit: dict,
                              alphas: Sequence[float] = QLEARN_SWEEP_ALPHAS,
                              bs: Sequence[float] = QLEARN_SWEEP_BS,
                              generative: bool = False,
                              n_sims: int = N_GENERATIVE_SIMS, seed: int = 0) -> list[dict]:
    """P(SHORT) trajectories over an ``(alpha, b)`` grid, holding the other estimates at their ML.

    The sweep shows what the *shape* of a Q-learning curve is controlled by: ``alpha`` sets how
    fast it rises and ``b`` how far it travels. ``Q0`` (and ``kappa``, for the perseveration
    variant) stay at ``fit``'s maximum-likelihood values, so every line differs from the ML fit
    in the two swept parameters only.

    ``generative`` additionally simulates each grid point. **Read this before drawing a sweep
    of ``qlearn_perseveration``:** ``kappa`` is held at its ML value across the whole grid, and
    the one-step-ahead trajectory of a large ``kappa`` collapses to roughly
    ``expit(kappa * s_prev)`` -- a one-trial-lagged copy of the data. So a figure meant to show
    "no grid point produces a step" would instead show all 16 lines stepping at exactly the
    right trial, for a reason that has nothing to do with ``alpha`` or ``b``. The generative
    trajectory is not conditioned on the data and cannot do that.

    Returns
    -------
    list[dict]
        One entry per grid point, in ``alpha``-major order: ``alpha``, ``b``, ``i_alpha``,
        ``i_b``, ``nll``, and ``p_short`` (the one-step-ahead trajectory, which is the one the
        ``nll`` beside it scores). With ``generative``, also ``p_generative`` (the mean over
        ``n_sims`` simulated runs at that grid point) and ``band`` (the full
        ``qlearning_generative_band`` result, for its ``examples`` and ``switch_taus``).
    """
    s = _as_one_animal(s)  # validated once here; the grid then uses the unchecked path
    out = []
    for i_alpha, alpha in enumerate(alphas):
        for i_b, b in enumerate(bs):
            nll, p_short = _nll_unchecked(s, alpha, b, fit["q0_short"], fit["q0_long"],
                                          fit["kappa"])
            point = {"alpha": float(alpha), "b": float(b), "i_alpha": i_alpha, "i_b": i_b,
                     "nll": float(nll), "p_short": p_short}
            if generative:
                # Same simulator as the ML band, at this grid point's alpha/b with Q0 and kappa
                # held at ML -- so the generative grid varies exactly what the sweep sweeps.
                band = qlearning_generative_band(
                    {**fit, "alpha": float(alpha), "b": float(b)}, s.size,
                    n_sims=n_sims, seed=seed)
                point["p_generative"] = band["mean"]
                point["band"] = band
            out.append(point)
    return out
