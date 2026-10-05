"""Smith et al. (2004) state-space model: accuracy trial by trial, no curve shape assumed.

    x_0 ~ N(0, X0_SD),   x_k = x_{k-1} + σ z_k,   z_k ~ N(0, 1),   y_k ~ Bernoulli(logistic(x_k))

over the mode's row index (`data.choice_sequence`), one animal at a time, sampled with
NUTS. The walk is non-centered (``x`` is the cumulative sum of ``σ z``) to avoid the
funnel between σ and the steps.

- `fit_state_space` -- the posterior per animal: p_k with its 95% band, P(x_k > 0), the
  expected excess correct, the session means, the learning trial and the change count.
- `state_space_summary`, `state_space_changes`, `gate_table` -- the tables.

**The start is free.** ``x_0`` is estimated, as in Smith et al., so an animal that
begins above chance shows no early rise.

**The learning trial** is the first k with P(x_k > 0) > `LEARNED` from which it stays
above to the end; 0 when the data begin there, None when they never settle.

**The change count is a rule on the posterior, not a model output.** It is counted on
the session means of p_k, per draw, not on single trials: one σ serves the whole walk,
so a large change sets σ high and single trials then wander by about 0.1 on flat
stretches, and their extremes pass any test of a difference.

- The median session means are cut at their turning points, where they turn back by at
  least `MIN_CHANGE`, and in the middle of their plateaus, at least `PLATEAU_SESSIONS`
  sessions within `PLATEAU_FLAT`. Before the first turning point and after the last
  they move less than `MIN_CHANGE`; that is left out.
- A piece is a change when its ends differ by at least `MIN_CHANGE` and at least
  `CREDIBLE` of the draws agree on its direction. Rises and falls both count.
- Failing pieces go smallest first: merged across a plateau into the smaller neighbour
  if they can; a reversal between two others merges with both (a dip that is not a
  change leaves one rise, not two); a reversal at either end is left out.

The constants were set on synthetic sequences only (sessions of 90 rows), before any
animal was fitted. A constant, a step, a small ramp, two steps and a dip came out 0, 1,
1, 2 and 3 in every seed, but a steep ramp (0.5 to 0.85) split in two in 3 of 4 fits:
a flat stretch two sessions long happens by chance in a ramp. Requiring three sessions
split fewer ramps but missed the second step in every seed. The rule errs towards more,
changes, which §2.7 then tests, rather than fewer, which would let a compromise w be
read as suddenness.

**Boundary σ.** ``boundary=True`` gives the steps into each session's first row their own
``σ_b``, under the same prior as σ. ``boundary_share`` is the share of the walk's
variance at those steps. With a few boundaries the prior sets much of σ_b: on synthetic
data with no change at all the share came out near 0.45. Read it against
``boundary="shifted"``, the same fit with σ_b on the steps half a session later. §0.3
predicts a higher share at the boundaries than half a session from them.
"""
from __future__ import annotations

import logging
import time

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
from scipy.special import expit

from hypnose_behavior.modelling.ab_learning.data import choice_sequence

__all__ = [
    "BOUNDARIES",
    "CREDIBLE",
    "KEEP_DRAWS",
    "LEARNED",
    "MIN_CHANGE",
    "PLATEAU_FLAT",
    "PLATEAU_SESSIONS",
    "SIGMA_SCALE",
    "SIGMA_SCALES",
    "X0_SD",
    "count_changes",
    "fit_state_space",
    "gate_table",
    "learning_trial",
    "state_space_changes",
    "state_space_summary",
]

# Priors: σ (and σ_b) ~ HalfNormal(SIGMA_SCALE); SIGMA_SCALES for the sensitivity check.
# x_0 ~ N(0, X0_SD) in log-odds: p from about 0.05 to 0.95 within two SD.
SIGMA_SCALE = 0.5
SIGMA_SCALES = (0.1, 0.5, 2.0)
X0_SD = 1.5

# `fit_state_space`'s ``boundary``: one σ; σ_b into each session's first row; σ_b half a
# session later, the control.
BOUNDARIES = (False, True, "shifted")

# The learning trial: P(x_k > 0) above this from there to the end.
LEARNED = 0.95

# The change rule, on session means: the smallest change of interest in p (§2.7), the
# share of draws that must agree on its direction, and what counts as a plateau.
MIN_CHANGE = 0.10
CREDIBLE = 0.95
PLATEAU_SESSIONS = 2
PLATEAU_FLAT = 0.04

# Posterior draws of x kept per fit, thinned evenly; the bands use every draw.
KEEP_DRAWS = 1000

_BAND = (0.025, 0.5, 0.975)

_SUMMARY_COLUMNS = [
    "subjid", "mode", "boundary", "sigma_scale", "n", "n_changes", "n_rises", "n_falls",
    "changes", "learning_trial", "learning_ses", "p_start", "p_end", "sigma", "sigma_lo",
    "sigma_hi", "sigma_b", "sigma_b_lo", "sigma_b_hi", "sigma_ratio", "boundary_share",
    "boundary_share_lo", "boundary_share_hi", "n_boundaries", "divergences", "r_hat_max",
    "ess_min", "seconds",
]
_CHANGE_COLUMNS = [
    "subjid", "mode", "boundary", "sigma_scale", "direction", "from_ses", "to_ses",
    "k_mid", "ses_mid", "p_from", "p_to", "delta", "credibility",
]


# ---------------------------------------------------------------------------------------
# Sampling.
# ---------------------------------------------------------------------------------------

def _own_steps(session_idx, boundary):
    """Which steps (into rows 1..n-1) take σ_b: into each session's first row, or with
    ``"shifted"`` into its middle row; every session but the first."""
    starts = np.flatnonzero(np.r_[True, np.diff(session_idx) != 0])
    rows = starts[1:]
    if boundary == "shifted":
        stops = np.r_[starts[1:], session_idx.size]
        rows = ((starts + stops) // 2)[1:]
    marked = np.zeros(session_idx.size, dtype=bool)
    marked[rows] = True
    return marked[1:]


def _sample(y, own, sigma_scale, draws, tune, chains, target_accept, seed):
    """Posterior draws of the walk for one animal; ``own`` marks the steps that take σ_b,
    None for one σ."""
    with pm.Model():
        sigma = pm.HalfNormal("sigma", sigma_scale)
        step_sd = sigma
        if own is not None:
            sigma_b = pm.HalfNormal("sigma_b", sigma_scale)
            step_sd = pt.where(own, sigma_b, sigma)
        x0 = pm.Normal("x0", 0.0, X0_SD)
        z = pm.Normal("z", 0.0, 1.0, shape=y.size - 1)
        x = pm.Deterministic("x", x0 + pt.concatenate([pt.zeros(1),
                                                       pt.cumsum(step_sd * z)]))
        pm.Bernoulli("y", logit_p=x, observed=y)
        quiet = logging.getLogger("pymc")
        level = quiet.level
        quiet.setLevel(logging.WARNING)
        try:
            return pm.sample(draws, tune=tune, chains=chains, cores=1,
                             target_accept=target_accept, random_seed=seed,
                             progressbar=False, compute_convergence_checks=False)
        finally:
            quiet.setLevel(level)


def _flat(idata, name):
    """A variable's draws, chains stacked: ``(draws,)`` or ``(draws, n)``."""
    values = np.asarray(idata.posterior[name])
    return values.reshape(-1, *values.shape[2:])


def _diagnostics(idata, names):
    """Divergences, the largest R-hat and the smallest bulk ESS over ``names``."""
    r_hat = az.rhat(idata, var_names=names)
    ess = az.ess(idata, var_names=names)
    return {"divergences": int(np.asarray(idata.sample_stats["diverging"]).sum()),
            "r_hat_max": max(float(np.nanmax(np.asarray(r_hat[v]))) for v in names),
            "ess_min": min(float(np.nanmin(np.asarray(ess[v]))) for v in names)}


# ---------------------------------------------------------------------------------------
# Summaries of one posterior.
# ---------------------------------------------------------------------------------------

def learning_trial(p_above, level: float = LEARNED):
    """The first row from which ``p_above`` stays above ``level`` to the end; None if the
    last row is not above it."""
    above = np.asarray(p_above) > level
    if not above.size or not above[-1]:
        return None
    below = np.flatnonzero(~above)
    return int(below[-1] + 1) if below.size else 0


def _curve(rows, x):
    """Per row: p_k (median, 95% band), P(x_k > 0) and the expected excess correct."""
    lo, mid, hi = np.quantile(x, _BAND, axis=0)
    excess = np.cumsum(expit(x) - 0.5, axis=1)
    e_lo, e_mid, e_hi = np.quantile(excess, _BAND, axis=0)
    return rows[["k", "ses", "session_idx", "hours", "y"]].assign(
        p=expit(mid), p_lo=expit(lo), p_hi=expit(hi), p_above=(x > 0).mean(axis=0),
        excess=e_mid, excess_lo=e_lo, excess_hi=e_hi).reset_index(drop=True)


def _session_means(rows, x):
    """Per draw, the mean p_k over each session's rows ``(draws, sessions)``, and per
    session its rows and observed accuracy with the draws' median and 95% band."""
    index = rows["session_idx"].to_numpy()
    p = expit(x)
    draws = np.stack([p[:, index == s].mean(axis=1) for s in np.unique(index)], axis=1)
    lo, mid, hi = np.quantile(draws, _BAND, axis=0)
    table = rows.groupby("session_idx").agg(
        ses=("ses", "first"), n=("k", "size"), start=("k", "min"), end=("k", "max"),
        observed=("y", "mean")).reset_index()
    return draws, table.assign(end=table["end"] + 1, p=mid, p_lo=lo, p_hi=hi)


def _turning_points(m, reversal):
    """The extremes between which ``m`` moves one way, each turning back by at least
    ``reversal`` (a zigzag), first to last. What lies before the first and after the
    last moves less than ``reversal`` and is left out; both ends when ``m`` never
    moves that far."""
    points, lo, hi, direction, extreme = [], 0, 0, 0, 0
    for k in range(1, m.size):
        if direction == 0:
            lo = k if m[k] < m[lo] else lo
            hi = k if m[k] > m[hi] else hi
            if m[k] - m[lo] >= reversal:
                points, direction, extreme = [lo], 1, k
            elif m[hi] - m[k] >= reversal:
                points, direction, extreme = [hi], -1, k
        elif direction * (m[k] - m[extreme]) >= 0:
            extreme = k
        elif direction * (m[extreme] - m[k]) >= reversal:
            points, direction, extreme = points + [extreme], -direction, k
    return [0, m.size - 1] if direction == 0 else points + [extreme]


def _plateaus(m, start, stop, length, flat):
    """Middles of the stretches of ``start..stop`` at least ``length`` points long over
    which ``m`` moves less than ``flat``, found left to right."""
    middles, i = [], start
    while i <= stop - length + 1:
        j, low, high = i, m[i], m[i]
        while j < stop and max(high, m[j + 1]) - min(low, m[j + 1]) < flat:
            j += 1
            low, high = min(low, m[j]), max(high, m[j])
        if j - i + 1 >= length:
            middles.append((i + j) // 2)
            i = j + 1
        else:
            i += 1
    return middles


def count_changes(median, draws, *, min_change=MIN_CHANGE, credible=CREDIBLE,
                  plateau_length=PLATEAU_SESSIONS, plateau_flat=PLATEAU_FLAT):
    """The changes of one curve by the module's rule, as ``(start, stop, delta,
    credibility)`` in the curve's points.

    ``median`` is the curve (the median session means, say) and ``draws`` its posterior
    draws, ``(draws, points)``; ``delta`` is in ``median``, ``credibility`` the share of
    draws agreeing on its direction.
    """
    m = np.asarray(median, dtype=float)
    turns = _turning_points(m, min_change)
    cuts = {k: "turn" for k in turns}
    for a, b in zip(turns[:-1], turns[1:]):
        for k in _plateaus(m, a, b, plateau_length, plateau_flat):
            cuts.setdefault(k, "plateau")
    ends = sorted(cuts)
    kinds = [cuts[k] for k in ends]

    def judge(a, b):
        delta = m[b] - m[a]
        agree = float(np.mean(np.sign(delta) * (draws[:, b] - draws[:, a]) > 0))
        return delta, agree, abs(delta) >= min_change and agree >= credible

    while len(ends) > 1:
        pieces = [judge(a, b) for a, b in zip(ends[:-1], ends[1:])]
        failing = [i for i, piece in enumerate(pieces) if not piece[2]]
        if not failing:
            break
        if len(pieces) == 1:
            return []
        i = min(failing, key=lambda f: abs(pieces[f][0]))
        # Piece i lies between cuts i and i + 1; the outer two cuts are the extremes.
        inner = [c for c in (i, i + 1) if 0 < c < len(ends) - 1]
        plateau = [c for c in inner if kinds[c] == "plateau"]
        if plateau:
            # Across a plateau, into the smaller neighbour.
            drop = [min(plateau, key=lambda c: abs(pieces[c - 1 if c == i else c][0]))]
        elif len(inner) == 2:
            drop = inner      # a reversal that is not a change: one move with both sides
        else:
            drop = [0 if i == 0 else len(ends) - 1]    # a reversal at an edge: left out
        for c in sorted(drop, reverse=True):
            del ends[c], kinds[c]
    return [(a, b, *judge(a, b)[:2]) for a, b in zip(ends[:-1], ends[1:])]


def _changes(curve, sessions, draws, identity):
    """`count_changes` on the session means as a frame. ``k_mid`` is the row of the
    sessions it spans from which the median p_k stays past half the change."""
    m = sessions["p"].to_numpy()
    rows = []
    for a, b, delta, agree in count_changes(m, draws):
        inside = curve[curve["session_idx"].between(sessions["session_idx"].iloc[a],
                                                    sessions["session_idx"].iloc[b])]
        p = inside["p"].to_numpy()
        short = np.flatnonzero(p < m[a] + delta / 2 if delta > 0 else p > m[a] + delta / 2)
        mid = min(short[-1] + 1, len(inside) - 1) if short.size else 0
        rows.append({**identity, "direction": "rise" if delta > 0 else "fall",
                     "from_ses": sessions["ses"].iloc[a], "to_ses": sessions["ses"].iloc[b],
                     "k_mid": int(inside["k"].iloc[mid]), "ses_mid": inside["ses"].iloc[mid],
                     "p_from": m[a], "p_to": m[b], "delta": delta, "credibility": agree})
    return pd.DataFrame(rows, columns=_CHANGE_COLUMNS)


def _interval(values):
    lo, mid, hi = np.quantile(values, _BAND)
    return float(mid), float(lo), float(hi)


def _summary(identity, curve, changes, sigma, sigma_b, own, learned, diagnostics,
             seconds):
    n_b = int(own.sum())
    s, s_lo, s_hi = _interval(sigma)
    row = {**identity, "n": len(curve), "n_changes": len(changes),
           "n_rises": int((changes["direction"] == "rise").sum()),
           "n_falls": int((changes["direction"] == "fall").sum()),
           "changes": "; ".join(f"{'↑' if c.delta > 0 else '↓'} {c.p_from:.2f}→{c.p_to:.2f} "
                                f"ses {c.ses_mid}" for c in changes.itertuples()),
           "learning_trial": learned,
           "learning_ses": None if learned is None else curve["ses"].iloc[learned],
           "p_start": float(curve["p"].iloc[0]), "p_end": float(curve["p"].iloc[-1]),
           "sigma": s, "sigma_lo": s_lo, "sigma_hi": s_hi, "n_boundaries": n_b,
           **diagnostics, "seconds": seconds}
    if sigma_b is not None:
        b, b_lo, b_hi = _interval(sigma_b)
        share = n_b * sigma_b ** 2 / (n_b * sigma_b ** 2 + (own.size - n_b) * sigma ** 2)
        sh, sh_lo, sh_hi = _interval(share)
        row.update(sigma_b=b, sigma_b_lo=b_lo, sigma_b_hi=b_hi,
                   sigma_ratio=float(np.median(sigma_b / sigma)), boundary_share=sh,
                   boundary_share_lo=sh_lo, boundary_share_hi=sh_hi)
    return {c: row.get(c, np.nan) for c in _SUMMARY_COLUMNS}


# ---------------------------------------------------------------------------------------
# The fit.
# ---------------------------------------------------------------------------------------

def _fit_rows(rows, mode, boundary, sigma_scale, draws, tune, chains, target_accept, seed):
    """One animal's posterior and its summaries, from its `choice_sequence` rows."""
    subjid = int(rows["subjid"].iloc[0])
    y = rows["y"].to_numpy(dtype=int)
    own = _own_steps(rows["session_idx"].to_numpy(), boundary)
    started = time.perf_counter()
    idata = _sample(y, own if boundary else None, sigma_scale, draws, tune, chains,
                    target_accept, seed)
    seconds = time.perf_counter() - started
    names = ["sigma", "x0", "x"] + (["sigma_b"] if boundary else [])
    diagnostics = _diagnostics(idata, names)

    x = _flat(idata, "x")
    sigma = _flat(idata, "sigma")
    sigma_b = _flat(idata, "sigma_b") if boundary else None
    identity = {"subjid": subjid, "mode": mode, "boundary": boundary,
                "sigma_scale": sigma_scale}
    curve = _curve(rows, x)
    session_draws, sessions = _session_means(rows, x)
    changes = _changes(curve, sessions, session_draws, identity)
    learned = learning_trial(curve["p_above"])
    summary = _summary(identity, curve, changes, sigma, sigma_b, own, learned, diagnostics,
                       seconds)
    variant = {False: "", True: " boundary σ", "shifted": " shifted σ"}[boundary]
    print(f"[state_space] sub-{subjid:03d} {mode}{variant}, σ ~ HN({sigma_scale:g}): "
          f"{len(rows)} rows, {seconds:.0f} s, {diagnostics['divergences']} divergences, "
          f"R-hat ≤ {diagnostics['r_hat_max']:.3f}, {len(changes)} change(s)")
    keep = np.linspace(0, x.shape[0] - 1, min(KEEP_DRAWS, x.shape[0])).astype(int)
    return {**identity, "n": len(rows), "rows": rows, "curve": curve, "sessions": sessions,
            "session_draws": session_draws.astype(np.float32),
            "x": x[keep].astype(np.float32), "sigma": sigma, "sigma_b": sigma_b,
            "own_steps": own, "changes": changes, "learning_trial": learned,
            "summary": summary}


def fit_state_space(data: dict, mode: str = "completed", *, boundary=False,
                    sigma_scale: float = SIGMA_SCALE, draws: int = 1000, tune: int = 1000,
                    chains: int = 4, target_accept: float = 0.95, seed: int = 0) -> dict:
    """The state-space posterior per animal, over the mode's row index.

        walk = fit_state_space(ab)                                # the gate
        walk_b = fit_state_space(ab, boundary=True)               # σ_b at boundaries
        walk_s = fit_state_space(ab, boundary="shifted")          # its control
        prior = {s: fit_state_space(ab, sigma_scale=s) for s in SIGMA_SCALES}

    ``boundary`` is one of `BOUNDARIES`. Returns ``{subjid: fit}``, each a dict of:

    - ``curve`` -- per row: ``p`` (median), ``p_lo`` / ``p_hi`` (95%), ``p_above``
      (P(x_k > 0)), ``excess`` / ``excess_lo`` / ``excess_hi`` (the expected excess
      correct ``cumsum(p_k - 0.5)``, over the draws), with ``k``, ``ses``,
      ``session_idx``, ``hours`` and ``y``.
    - ``sessions`` -- per session: ``start`` / ``end`` rows, ``n``, ``observed``
      accuracy and the mean p_k's ``p`` / ``p_lo`` / ``p_hi``, which the changes are
      counted on; ``session_draws`` -- its draws, float32.
    - ``changes`` -- see `state_space_changes`; ``learning_trial`` -- a row, or None.
    - ``sigma`` / ``sigma_b`` -- every draw (``sigma_b`` None with one σ); ``x`` --
      `KEEP_DRAWS` draws of the walk, float32; ``own_steps`` -- the steps σ_b takes.
    - ``summary`` -- this animal's `state_space_summary` row; ``rows`` -- its
      `data.choice_sequence` rows.

    Chains run one after another (``cores=1``): 20 to 50 s per 1000 rows, longer the
    more the curve changes.
    """
    if boundary not in BOUNDARIES:
        raise ValueError(f"boundary must be one of {BOUNDARIES}, got {boundary!r}")
    fits = {}
    for subjid, rows in choice_sequence(data, mode).groupby("subjid"):
        fits[int(subjid)] = _fit_rows(rows.reset_index(drop=True), mode, boundary,
                                      sigma_scale, draws, tune, chains, target_accept, seed)
    return fits


# ---------------------------------------------------------------------------------------
# Tables.
# ---------------------------------------------------------------------------------------

def state_space_summary(*fits: dict) -> pd.DataFrame:
    """One row per animal and fit.

    - ``n_changes`` / ``n_rises`` / ``n_falls`` by the module's rule; ``changes`` lists
      them, each with the session where it is half done.
    - ``learning_trial`` / ``learning_ses``; ``p_start`` / ``p_end``: the median p at the
      first and last row.
    - ``sigma`` (median, 95% ``_lo`` / ``_hi``) per step in log-odds. With σ_b:
      ``sigma_b``, ``sigma_ratio`` (median σ_b/σ) and ``boundary_share`` (the walk's
      variance at the ``n_boundaries`` steps σ_b takes, over all of it).
    - ``divergences``, ``r_hat_max`` and ``ess_min`` over σ, σ_b, x_0 and every x_k;
      ``seconds`` of sampling.
    """
    return pd.DataFrame([fit["summary"] for group in fits for fit in group.values()],
                        columns=_SUMMARY_COLUMNS)


def state_space_changes(*fits: dict) -> pd.DataFrame:
    """Every change, one row each: ``direction``, the sessions ``from_ses`` / ``to_ses``
    its piece spans, ``k_mid`` / ``ses_mid`` where it is half done, ``p_from`` /
    ``p_to`` and ``delta`` in the median session means, and ``credibility``."""
    frames = [fit["changes"] for group in fits for fit in group.values()]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        columns=_CHANGE_COLUMNS)


def gate_table(fits: dict, steps: dict, boundary: dict | None = None,
               shifted: dict | None = None) -> pd.DataFrame:
    """The gate: per animal, the change count beside the single-change results.

        gate_table(walk, steps, walk_b, walk_s)

    ``fits`` are `fit_state_space` results and ``steps`` the `step.fit_step` results of
    the same mode. ``verdict``: one change, the 2.1–2.6 results stand; more, the animal
    goes to §2.7 and its sigmoid w is not read as suddenness; none, nothing changed
    credibly. ``boundary`` / ``shifted`` fits add their change count, ``sigma_ratio`` and
    ``boundary_share`` (``shifted_share`` for the control).
    """
    rows = []
    for subjid, fit in sorted(fits.items()):
        s = fit["summary"]
        row = {"subjid": subjid, "mode": s["mode"], "n_changes": s["n_changes"],
               "changes": s["changes"], "learning_trial": s["learning_trial"],
               "learning_ses": s["learning_ses"], "sigma": s["sigma"]}
        step = steps.get(subjid)
        if step is not None:
            sigmoid, summary = step["sigmoid"], step["summary"]
            row.update(variant=step["variant"], w=sigmoid["width"],
                       w_lo=sigmoid["width_lo"], w_hi=sigmoid["width_hi"],
                       step=summary["step"], ΔlnL_vs_sigmoid=summary["ΔlnL_vs_sigmoid"],
                       ΔlnL_sigmoid_const=summary["ΔlnL_sigmoid_const"])
        if boundary is not None and subjid in boundary:
            b = boundary[subjid]["summary"]
            row.update(n_changes_b=b["n_changes"], sigma_ratio=b["sigma_ratio"],
                       boundary_share=b["boundary_share"])
        if shifted is not None and subjid in shifted:
            row["shifted_share"] = shifted[subjid]["summary"]["boundary_share"]
        row["verdict"] = ("no change" if s["n_changes"] == 0 else
                          "one change: 2.1–2.6 stand" if s["n_changes"] == 1 else "§2.7")
        rows.append(row)
    return pd.DataFrame(rows)
