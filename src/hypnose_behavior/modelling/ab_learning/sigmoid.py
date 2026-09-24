"""Single-sigmoid fits: how sudden a change is and where it falls, per animal.

One curve, ``f(x) = initial + (final - initial)/2 * [1 + erf((x - center)/width)]``, fitted
three ways:

- `fit_accuracy` -- Bernoulli on each choice's correctness over the mode's row index
  ``k`` (`data.choice_sequence`); ``initial`` free, or fixed at chance.
- `fit_rate`     -- inhomogeneous Poisson on event times over task hours: initiations
  (engagement), or rewards (rate times accuracy, so descriptive only).

``width`` is the suddenness: ``center +/- width`` spans 84% of the change, 10% to 90% takes
``1.81 * width``, and a width under one row is a change faster than the data can resolve.

**The likelihood is multimodal, and the optima are the result.** lnL is maximized over
the two levels at every point of a (center, log width) grid; every peak of that surface
standing at least `PROMINENCE` above the saddle toward any higher peak is refined from its
grid point and kept. Two separated optima of similar lnL mean two changes, which one
sigmoid can only fit as a compromise, often a wide one: two steps in a row can come back
as a single slow ramp that beats either step.

``width_lo`` / ``width_hi`` are a 95% profile-likelihood interval for the width over the
optimum's own basin of the grid, at grid resolution: 0 when a step is not excluded, inf
when the widest grid width is not.

**The rate fits are overconfident.** Events come in bouts, and the rate differs from one
session to the next, so they are overdispersed against a Poisson process: rate lnL
differences and width intervals come out far too sharp. On task time a session boundary
is an instant, so a rate that differs between sessions fits as a step at a boundary.
Read the rate fits for where the change falls, not for how sudden it is.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import erf, expit, logit
from scipy.stats import chi2

from hypnose_behavior.modelling.ab_learning.data import (
    choice_sequence,
    session_bounds,
    task_time,
)

__all__ = [
    "N_CENTERS",
    "N_WIDTHS",
    "PROMINENCE",
    "RATE_EVENTS",
    "VARIANTS",
    "erf_curve",
    "expected_count",
    "expected_excess",
    "fit_accuracy",
    "fit_rate",
    "sigmoid_anchors",
    "sigmoid_optima",
]

# The (center, width) grid: centers evenly over the data, widths log-spaced from half a
# row (or 1/2000 of the task time) to twice the whole span.
N_CENTERS = 240
N_WIDTHS = 36

# lnL a peak of the grid surface must rise above the saddle toward any higher peak to
# count as an optimum of its own.
PROMINENCE = 2.0

# `fit_accuracy` variants: ``initial`` free, or fixed at chance.
VARIANTS = ("free", "chance")

# `fit_rate` events, from `load_ab_data`'s trials.
RATE_EVENTS = ("initiations", "rewards")

_CHANCE = 0.5
_EPS = 1e-6
_NEWTON_STEPS = 40
_HALVINGS = 10
_TOL = 1e-8
_INTERVAL = chi2.ppf(0.95, 1) / 2
_CHUNK = 2_000_000
_SQRT_PI = np.sqrt(np.pi)

_OPTIMA_COLUMNS = [
    "subjid", "kind", "mode", "variant", "n", "rank", "center", "width", "initial",
    "final", "lnl", "delta_lnl", "prominence", "width_lo", "width_hi", "inside",
    "center_hours", "span_lo_hours", "span_hi_hours", "ses", "k_params", "converged",
]


def erf_curve(x, initial, final, center, width):
    """The fitted curve at ``x``."""
    return initial + (final - initial) * _shape(x, center, width)


def _shape(x, center, width):
    """``(1 + erf((x - center) / width)) / 2``, broadcast over the arguments."""
    return 0.5 * (1.0 + erf((np.asarray(x, dtype=float) - center) / width))


def _shape_integral(span, center, width):
    """``_shape`` integrated over ``[0, span]``, closed form."""
    def antiderivative(z):
        return z * erf(z) + np.exp(-z ** 2) / _SQRT_PI
    return 0.5 * (span + width * (antiderivative((span - center) / width)
                                  - antiderivative(-center / width)))


# ---------------------------------------------------------------------------------------
# Levels at fixed (center, width): lnL is concave in (initial, final) for both models.
# ---------------------------------------------------------------------------------------

def _sums(g, slope, curve):
    """Gradient and Hessian of ``sum f(a + (b - a) g)`` from ``f'`` and ``-f''``."""
    h = 1 - g
    return ((slope * h).sum(axis=1), (slope * g).sum(axis=1)), (
        -(curve * h * h).sum(axis=1), -(curve * h * g).sum(axis=1),
        -(curve * g * g).sum(axis=1))


def _direction(grad, hess, a, b, lower, upper, fixed_a):
    """Newton step in (a, b), holding a level that sits on its bound and pushes out."""
    (ga, gb), (haa, hab, hbb) = grad, hess
    haa, hbb = np.minimum(haa, -1e-12), np.minimum(hbb, -1e-12)
    ridge = 1e-9 * (np.abs(haa) + np.abs(hbb))
    det = (haa - ridge) * (hbb - ridge) - hab ** 2
    da = -((hbb - ridge) * ga - hab * gb) / det
    db = -((haa - ridge) * gb - hab * ga) / det
    held_a = fixed_a | ((a <= lower) & (ga < 0)) | ((a >= upper) & (ga > 0))
    held_b = ((b <= lower) & (gb < 0)) | ((b >= upper) & (gb > 0))
    da, db = np.where(held_b, -ga / haa, da), np.where(held_b, 0.0, db)
    da, db = np.where(held_a, 0.0, da), np.where(held_a & ~held_b, -gb / hbb, db)
    return da, db


def _newton(lnl_fn, a, b, lower, upper, fixed_a):
    """Maximize lnL in (a, b) for every grid row: damped Newton in the box, rows dropping
    out once converged. ``lnl_fn(rows, a, b, derivatives)``."""
    a, b = a.astype(float), b.astype(float)
    lnl = lnl_fn(np.arange(a.size), a, b, False)
    rows = np.arange(a.size)
    for _ in range(_NEWTON_STEPS):
        if not rows.size:
            break
        ra, rb, rl = a[rows], b[rows], lnl[rows]
        da, db = _direction(*lnl_fn(rows, ra, rb, True), ra, rb, lower, upper, fixed_a)
        new_a, new_b, new_l = ra.copy(), rb.copy(), rl.copy()
        step, pending = np.ones(rows.size), np.ones(rows.size, dtype=bool)
        for _ in range(_HALVINGS):
            sub = np.flatnonzero(pending)
            ca = np.clip(ra[sub] + step[sub] * da[sub], lower, upper)
            cb = np.clip(rb[sub] + step[sub] * db[sub], lower, upper)
            trial = lnl_fn(rows[sub], ca, cb, False)
            ok = trial >= rl[sub] - 1e-12
            new_a[sub[ok]], new_b[sub[ok]], new_l[sub[ok]] = ca[ok], cb[ok], trial[ok]
            pending[sub[ok]] = False
            step[sub[~ok]] /= 2
            if not pending.any():
                break
        a[rows], b[rows], lnl[rows] = new_a, new_b, new_l
        rows = rows[new_l - rl > _TOL]
    return lnl, a, b


def _bernoulli_levels(k, y, fixed_a):
    """``levels(centers, widths) -> (lnl, initial, final)`` for the accuracy model."""
    def levels(centers, widths):
        g = _shape(k[None, :], centers[:, None], widths[:, None])

        def lnl_fn(rows, a, b, derivatives):
            gr = g[rows]
            p = np.clip(a[:, None] + (b - a)[:, None] * gr, _EPS, 1 - _EPS)
            if not derivatives:
                return np.where(y, np.log(p), np.log1p(-p)).sum(axis=1)
            slope = np.where(y, 1 / p, -1 / (1 - p))
            return _sums(gr, slope, slope ** 2)

        # Start from the two sides' accuracies, weighted by the shape.
        b = np.clip((g * y).sum(axis=1) / np.maximum(g.sum(axis=1), _EPS), 0.01, 0.99)
        a = (np.full_like(b, _CHANCE) if fixed_a else np.clip(
            ((1 - g) * y).sum(axis=1) / np.maximum((1 - g).sum(axis=1), _EPS), 0.01, 0.99))
        return _newton(lnl_fn, a, b, _EPS, 1 - _EPS, fixed_a)
    return levels


def _poisson_levels(times, span):
    """``levels(centers, widths) -> (lnl, initial, final)`` for the rate model."""
    def levels(centers, widths):
        g = _shape(times[None, :], centers[:, None], widths[:, None])
        integral = _shape_integral(span, centers, widths)

        def lnl_fn(rows, a, b, derivatives):
            gr, inside = g[rows], integral[rows]
            rate = np.maximum(a[:, None] + (b - a)[:, None] * gr, _EPS)
            if not derivatives:
                return np.log(rate).sum(axis=1) - a * (span - inside) - b * inside
            (ga, gb), hess = _sums(gr, 1 / rate, rate ** -2.0)
            return (ga - (span - inside), gb - inside), hess

        a = np.maximum((1 - g).sum(axis=1) / np.maximum(span - integral, _EPS), _EPS)
        b = np.maximum(g.sum(axis=1) / np.maximum(integral, _EPS), _EPS)
        return _newton(lnl_fn, a, b, _EPS, np.inf, False)
    return levels


# ---------------------------------------------------------------------------------------
# Grid surface, its peaks, and refinement.
# ---------------------------------------------------------------------------------------

def _grid(model):
    """lnL, initial and final at every (center, width) grid point."""
    centers, widths = model["centers"], model["widths"]
    cc, ww = (m.ravel() for m in np.meshgrid(centers, widths, indexing="ij"))
    out = np.empty((3, cc.size))
    rows = max(1, _CHUNK // max(model["x"].size, 1))
    for start in range(0, cc.size, rows):
        part = slice(start, start + rows)
        out[:, part] = model["levels"](cc[part], ww[part])
    return out.reshape(3, centers.size, widths.size)


def _surface_peaks(lnl):
    """Peaks of the grid surface with their prominence and basin.

    Cells join in order of falling lnL, each to the component of its highest
    8-neighbour; where two components meet, the lower peak's prominence is its height
    above that saddle. A peak's basin is the cells that joined it, or joined a peak too
    small to count that merged into it, before it merged into a higher one.
    """
    n_c, n_w = lnl.shape
    flat = lnl.ravel()
    label = np.full(flat.size, -1)
    parent, prominence, merged_into = {}, {}, {}

    def root(cell):
        while parent[cell] != cell:
            parent[cell] = parent[parent[cell]]
            cell = parent[cell]
        return cell

    for cell in np.argsort(flat, kind="stable")[::-1]:
        i, j = divmod(int(cell), n_w)
        near = {root(label[ni * n_w + nj])
                for ni in range(max(i - 1, 0), min(i + 2, n_c))
                for nj in range(max(j - 1, 0), min(j + 2, n_w))
                if label[ni * n_w + nj] >= 0}
        if not near:
            parent[cell] = label[cell] = cell
            continue
        top = max(near, key=lambda r: flat[r])
        label[cell] = top
        for other in near - {top}:
            prominence[other] = flat[other] - flat[cell]
            parent[other] = merged_into[other] = top
    best = int(np.argmax(flat))
    prominence[best] = np.inf
    peaks = {p for p, v in prominence.items() if v >= PROMINENCE}

    def owner(peak):
        while peak not in peaks:
            peak = merged_into[peak]
        return peak

    owners = {peak: owner(peak) for peak in np.unique(label)}
    basin = np.array([owners[peak] for peak in label]).reshape(lnl.shape)
    return [(int(p), prominence[p], basin == p) for p in peaks]


def _width_interval(lnl, widths, basin):
    """95% profile-likelihood interval of the width over one basin, at grid resolution."""
    over_width = np.where(basin, lnl, -np.inf).max(axis=0)
    inside = np.flatnonzero(over_width >= over_width.max() - _INTERVAL)
    lo = 0.0 if inside[0] == 0 else float(widths[inside[0]])
    hi = np.inf if inside[-1] == widths.size - 1 else float(widths[inside[-1]])
    return lo, hi


def _refine(model, center, width, a, b, spacing):
    """Nelder-Mead from a grid point over (levels, center, log width)."""
    to_theta, from_theta = model["to_theta"], model["from_theta"]
    theta0 = np.r_[to_theta(a, b), center, np.log(width)]
    steps = np.r_[np.full(theta0.size - 2, 0.3), spacing, 0.3]
    simplex = np.vstack([theta0] + [theta0 + np.eye(theta0.size)[i] * s
                                    for i, s in enumerate(steps)])

    def negative(theta):
        a_, b_ = from_theta(theta[:-2])
        return -model["lnl"](a_, b_, theta[-2], np.exp(theta[-1]))

    result = minimize(negative, theta0, method="Nelder-Mead",
                      options={"initial_simplex": simplex, "maxiter": 20_000,
                               "xatol": 1e-6, "fatol": 1e-9})
    a_, b_ = from_theta(result.x[:-2])
    return {"center": float(result.x[-2]), "width": float(np.exp(result.x[-1])),
            "initial": float(a_), "final": float(b_), "lnl": float(-result.fun),
            "converged": bool(result.success)}


def _optima(model):
    """The refined optima of one fit, best first, and its grid surface."""
    lnl, initial, final = _grid(model)
    centers, widths = model["centers"], model["widths"]
    spacing = float(centers[1] - centers[0])
    found = []
    for cell, prominence, basin in _surface_peaks(lnl):
        i, j = divmod(cell, widths.size)
        fit = _refine(model, centers[i], widths[j], initial[i, j], final[i, j], spacing)
        width_lo, width_hi = _width_interval(lnl, widths, basin)
        found.append({**fit, "prominence": prominence, "width_lo": width_lo,
                      "width_hi": width_hi})

    # Grid peaks that refine to the same optimum are one optimum.
    found.sort(key=lambda f: -f["lnl"])
    kept = []
    for fit in found:
        if not any(abs(fit["center"] - k["center"]) < spacing
                   and abs(np.log(fit["width"] / k["width"])) < 0.1 for k in kept):
            kept.append(fit)
    optima = pd.DataFrame(kept)
    optima["rank"] = np.arange(len(optima))
    optima["delta_lnl"] = optima["lnl"].iloc[0] - optima["lnl"]
    optima["inside"] = optima["center"].between(centers[0], centers[-1])
    surface = {"centers": centers, "widths": widths, "lnl": lnl}
    return optima, surface


def _fit(subjid, kind, mode, variant, n, model, optima, surface, **extra):
    identity = {"subjid": subjid, "kind": kind, "mode": mode, "variant": variant, "n": n}
    optima = optima.assign(**identity, k_params=model["k_params"])
    return {**identity, "optima": optima[_OPTIMA_COLUMNS], "surface": surface, **extra}


# ---------------------------------------------------------------------------------------
# The fits.
# ---------------------------------------------------------------------------------------

def _accuracy_model(k, y, variant):
    fixed_a = variant == "chance"
    yb = np.asarray(y).astype(bool)

    def lnl(a, b, center, width):
        p = np.clip(erf_curve(k, a, b, center, width), _EPS, 1 - _EPS)
        return float(np.where(yb, np.log(p), np.log1p(-p)).sum())

    if fixed_a:
        to_theta, from_theta = (lambda a, b: np.r_[logit(b)]), (
            lambda t: (_CHANCE, expit(t[0])))
    else:
        to_theta, from_theta = (lambda a, b: np.r_[logit(a), logit(b)]), (
            lambda t: (expit(t[0]), expit(t[1])))
    return {"x": k, "centers": np.linspace(0, k.size - 1, N_CENTERS),
            "widths": np.geomspace(0.5, 2 * k.size, N_WIDTHS),
            "levels": _bernoulli_levels(k, yb, fixed_a), "lnl": lnl, "to_theta": to_theta,
            "from_theta": from_theta, "k_params": 3 if fixed_a else 4}


def fit_accuracy(data: dict, mode: str = "completed", variant: str = "free") -> dict:
    """The accuracy sigmoid per animal, over the mode's row index.

        fits = fit_accuracy(ab)                          # initial accuracy free
        fits = fit_accuracy(ab, variant="chance")        # initial accuracy fixed at 0.5

    Returns ``{subjid: fit}``, each a dict of ``optima`` (every separated optimum, best
    first; columns in `sigmoid_optima`), ``surface`` (the grid's ``centers``, ``widths``
    and ``lnl``), ``rows`` (the animal's `data.choice_sequence` rows) and the fit's
    identity. Centers and widths are in rows; the ``*_hours`` columns read the center and
    ``center +/- width`` off the rows' task hours.
    """
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {VARIANTS}, got {variant!r}")
    fits = {}
    for subjid, rows in choice_sequence(data, mode).groupby("subjid"):
        rows = rows.reset_index(drop=True)
        k, hours = rows["k"].to_numpy(dtype=float), rows["hours"].to_numpy(dtype=float)
        model = _accuracy_model(k, rows["y"].to_numpy(), variant)
        optima, surface = _optima(model)
        on_clock = lambda x: np.interp(x, k, hours)
        at = np.clip(optima["center"].round(), 0, len(rows) - 1).astype(int)
        optima = optima.assign(
            center_hours=on_clock(optima["center"]),
            span_lo_hours=on_clock(optima["center"] - optima["width"]),
            span_hi_hours=on_clock(optima["center"] + optima["width"]),
            ses=rows["ses"].to_numpy()[at])
        fits[int(subjid)] = _fit(int(subjid), "accuracy", mode, variant, len(rows), model,
                                 optima, surface, rows=rows)
    return fits


def _rate_model(times, span):
    def lnl(a, b, center, width):
        rate = np.maximum(erf_curve(times, a, b, center, width), _EPS)
        integral = _shape_integral(span, center, width)
        return float(np.log(rate).sum() - a * (span - integral) - b * integral)

    return {"x": times, "centers": np.linspace(0, span, N_CENTERS),
            "widths": np.geomspace(span / 2000, 2 * span, N_WIDTHS),
            "levels": _poisson_levels(times, span), "lnl": lnl,
            "to_theta": lambda a, b: np.r_[np.log(a), np.log(b)],
            "from_theta": lambda t: (np.exp(t[0]), np.exp(t[1])), "k_params": 4}


def fit_rate(data: dict, events: str = "initiations") -> dict:
    """The rate sigmoid per animal, in events per task hour.

        engagement = fit_rate(ab)
        rewards = fit_rate(ab, "rewards")

    An inhomogeneous Poisson process over the animal's task time, from 0 to the end of
    its last run. Returns ``{subjid: fit}`` as `fit_accuracy` does, with ``times`` (event
    task hours) and ``span`` in place of ``rows``; centers and widths are in hours.
    """
    if events not in RATE_EVENTS:
        raise ValueError(f"events must be one of {RATE_EVENTS}, got {events!r}")
    trials = data["trials"]
    if events == "rewards":
        trials = trials[trials["outcome"] == "rewarded"]
    trials = trials.assign(hours=task_time(data, trials, "sequence_start").to_numpy())
    bounds = session_bounds(data)
    fits = {}
    for subjid, part in trials.groupby("subjid"):
        spans = bounds[bounds["subjid"] == subjid].sort_values("start")
        span = float(spans["end"].max())
        times = np.sort(part["hours"].dropna().to_numpy(dtype=float))
        model = _rate_model(times, span)
        optima, surface = _optima(model)
        at = np.clip(np.searchsorted(spans["start"].to_numpy(), optima["center"], "right")
                     - 1, 0, len(spans) - 1)
        optima = optima.assign(
            center_hours=optima["center"],
            span_lo_hours=(optima["center"] - optima["width"]).clip(lower=0),
            span_hi_hours=(optima["center"] + optima["width"]).clip(upper=span),
            ses=spans["ses"].to_numpy()[at])
        fits[int(subjid)] = _fit(int(subjid), events, None, None, times.size, model,
                                 optima, surface, times=times, span=span)
    return fits


# ---------------------------------------------------------------------------------------
# Fitted running counts: what a fit predicts for the curves of `cumulative`.
# ---------------------------------------------------------------------------------------

def expected_excess(fit: dict, rank: int = 0) -> np.ndarray:
    """Expected excess correct after each row of an accuracy fit, ``cumsum(p(k) - 0.5)``
    under its optimum of that ``rank``: the counterpart of the observed
    ``cumsum(y - 0.5)``."""
    optimum = fit["optima"].iloc[rank]
    k = fit["rows"]["k"].to_numpy(dtype=float)
    return np.cumsum(erf_curve(k, optimum["initial"], optimum["final"], optimum["center"],
                               optimum["width"]) - 0.5)


def expected_count(fit: dict, hours, rank: int = 0) -> np.ndarray:
    """Expected events of a rate fit by each of ``hours``: its rate integrated from 0,
    under the optimum of that ``rank``. At the best optimum it ends on the observed total."""
    optimum = fit["optima"].iloc[rank]
    hours = np.asarray(hours, dtype=float)
    return (optimum["initial"] * hours + (optimum["final"] - optimum["initial"])
            * _shape_integral(hours, optimum["center"], optimum["width"]))


# ---------------------------------------------------------------------------------------
# Tables.
# ---------------------------------------------------------------------------------------

def sigmoid_optima(*fits: dict) -> pd.DataFrame:
    """Every optimum of the fits given, one row each, best first within a fit.

    - ``center`` / ``width`` in rows (accuracy) or hours (rates); ``initial`` / ``final``
      as accuracies or events per hour.
    - ``delta_lnl`` below the fit's best optimum; ``prominence`` above the saddle toward
      a higher peak (inf for the best).
    - ``width_lo`` / ``width_hi``: the width's 95% profile interval over the basin.
    - ``inside``: the center lies within the data. An optimum outside it is part of a
      sigmoid whose bend the data never reach: a ramp across the whole sequence, not a
      change at a point.
    - ``center_hours`` / ``span_lo_hours`` / ``span_hi_hours``: the center and
      ``center +/- width`` on task time; ``ses`` the session at the center.
    """
    return pd.concat([fit["optima"] for group in fits for fit in group.values()],
                     ignore_index=True)


def sigmoid_anchors(*fits: dict) -> pd.DataFrame:
    """Each fit's best optimum on the task-time clock, one row per animal and fit.

    ``n_optima`` counts the fit's separated optima; above 1, the best is one of several.
    """
    optima = sigmoid_optima(*fits)
    keys = ["subjid", "kind", "mode", "variant"]
    counts = optima.groupby(keys, dropna=False).size().rename("n_optima").reset_index()
    best = optima[optima["rank"] == 0].merge(counts, on=keys)
    return best[keys + ["center_hours", "span_lo_hours", "span_hi_hours", "ses",
                        "initial", "final", "n_optima"]].sort_values(
        keys, na_position="first").reset_index(drop=True)
