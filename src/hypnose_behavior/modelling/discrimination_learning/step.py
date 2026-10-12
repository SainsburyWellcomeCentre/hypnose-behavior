"""Step model: when the change in accuracy happened, and how precisely.

The step is the sigmoid of `sigmoid.fit_accuracy` with ``w -> 0``: accuracy ``initial``
before trial ``k_s`` and ``final`` from it on. Every function takes that module's fits of
one variant (``initial`` free or at chance; `sigmoid.pick_variant` picks per animal) and
keeps the variant.

- `step_test`    -- per animal, whether the sigmoid fits better than a step: ``ΔlnL`` of
  the best sigmoid over the best step. ``step`` is True where ``ΔlnL < STEP_CUT``, the
  one-parameter 95% cut; only there is a step an adequate description of the change.
- `fit_step`     -- the posterior over ``k_s`` under a flat prior, the levels integrated
  out under Beta(1, 1) (``initial`` held at 0.5 in a chance fit), over every ``k_s`` at
  least `EDGE` rows from either end; and lnL of the constant model.
- `step_summary` / `step_modes` -- the tables.

**Integrating out the levels widens the posterior.** Fixing them at their best values
treats both accuracies as known and understates how uncertain ``k_s`` is, which is the
number being reported.

**A multimodal posterior is not summarized.** At row resolution the posterior is ragged:
a run of errors cuts a dip into it, so peaks a few rows apart are one switch. A mode is
therefore a session's worth of peaks (each behind a dip of at least `MODE_DIP` in log
posterior), and counts when it holds at least `MODE_MASS` of the posterior. Two modes
leave open which side of a night the switch fell, or mean two changes; `step_summary`
then leaves ``k_s_mean`` / ``k_s_sd`` empty and `step_modes` lists the modes.

**Where ``step`` is False the posterior is not a result.** A gradual change still gives a
peaked posterior, near the middle of the ramp, and its spread says how well the data place
a step that did not happen.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from scipy.special import betaln
from scipy.stats import chi2

from hypnose_behavior.modelling.switchpoint.switch import (
    fit_constant,
    posterior_hdi,
    switchpoint_loglik_profile,
)

__all__ = [
    "EDGE",
    "HDI_MASS",
    "MODE_DIP",
    "MODE_MASS",
    "STEP_CUT",
    "fit_step",
    "step_modes",
    "step_summary",
    "step_test",
]

# Rows at either end no switch is placed in.
EDGE = 20

# lnL by which the sigmoid must beat the step for the change to count as gradual.
STEP_CUT = chi2.ppf(0.95, 1) / 2

# A posterior mode counts when it holds this much mass behind a dip this deep (in log
# posterior) below its peak.
MODE_MASS = 0.05
MODE_DIP = 2.0

HDI_MASS = 0.95

_CHANCE = 0.5


def _counts(y):
    """``tau`` (first row after the switch, 1..n-1) with the rows and correct on each side."""
    before = np.r_[0, np.cumsum(y)]
    tau = np.arange(1, y.size)
    n1, c1 = tau, before[tau]
    return tau, n1, c1, y.size - n1, before[-1] - c1


def _step_lnl(y, chance):
    """lnL of the best step at every ``tau`` (index), -inf at 0."""
    if not chance:
        return switchpoint_loglik_profile(y)
    tau, n1, _, n2, c2 = _counts(y)
    p2 = np.clip(c2 / n2, 1e-12, 1 - 1e-12)
    out = np.full(y.size, -np.inf)
    out[tau] = n1 * np.log(_CHANCE) + c2 * np.log(p2) + (n2 - c2) * np.log1p(-p2)
    return out


def _log_marginal(y, chance):
    """``tau`` and the log posterior (unnormalized) of a switch there, levels integrated."""
    tau, n1, c1, n2, c2 = _counts(y)
    first = n1 * np.log(_CHANCE) if chance else betaln(c1 + 1, n1 - c1 + 1)
    return tau, first + betaln(c2 + 1, n2 - c2 + 1)


def step_test(fits: dict) -> pd.DataFrame:
    """The best sigmoid against the best step, per animal.

    One row per animal: the sigmoid's ``w`` with its 95% interval, ``lnl_sigmoid``,
    ``lnl_step``, ``ΔlnL`` (sigmoid minus step) and ``step`` (``ΔlnL < STEP_CUT``: the
    data do not tell the change from a step).
    """
    rows = []
    for subjid, fit in sorted(fits.items()):
        best = fit["optima"].iloc[0]
        y = fit["rows"]["y"].to_numpy()
        step = float(_step_lnl(y, fit["variant"] == "chance").max())
        delta = max(float(best["lnl"]) - step, 0.0)
        rows.append({"subjid": subjid, "mode": fit["mode"], "variant": fit["variant"],
                     "w": best["width"], "w_lo": best["width_lo"], "w_hi": best["width_hi"],
                     "lnl_sigmoid": best["lnl"], "lnl_step": step, "ΔlnL": delta,
                     "step": delta < STEP_CUT})
    return pd.DataFrame(rows)


def _modes(tau, log_post, post, ses):
    """Posterior modes, one per session: peaks behind a dip of `MODE_DIP`, their basins'
    mass summed within the session of each peak, kept from `MODE_MASS` up.

    ``ses`` is the session of each ``tau``. Peaks within one session are the same switch
    placed a few rows apart by runs of errors; only peaks in different sessions leave
    the side of a night open.
    """
    floor = log_post.min() - 1e6
    index, _ = find_peaks(np.r_[floor, log_post, floor], prominence=MODE_DIP)
    index = np.sort(index - 1)
    # Basin i covers rows bounds[i] .. bounds[i + 1] - 1, split at the lowest point
    # between neighbouring peaks.
    bounds = [0] + [a + int(np.argmin(log_post[a:b + 1]))
                    for a, b in zip(index[:-1], index[1:])] + [tau.size]
    peaks = pd.DataFrame({
        "ses": ses[index], "k_s": tau[index], "height": log_post[index],
        "mass": [post[bounds[i]:bounds[i + 1]].sum() for i in range(index.size)],
        "from": tau[bounds[:-1]], "to": tau[np.array(bounds[1:]) - 1]})
    top = peaks.loc[peaks.groupby("ses", sort=False)["height"].idxmax()].set_index("ses")
    merged = peaks.groupby("ses", sort=False).agg(
        mass=("mass", "sum"), start=("from", "min"), stop=("to", "max"))
    modes = top[["k_s", "height"]].join(merged).reset_index().rename(
        columns={"start": "from", "stop": "to"})
    modes["Δlog_post"] = log_post.max() - modes.pop("height")
    modes = modes[modes["mass"] >= MODE_MASS].sort_values("mass", ascending=False)
    return modes.reset_index(drop=True)


def fit_step(fits: dict, edge: int = EDGE) -> dict:
    """The step model per animal, from the `sigmoid.fit_accuracy` fits of one variant.

        steps = fit_step(pick_variant(acc))

    Returns ``{subjid: step}``, each a dict of:

    - ``tau`` (candidate ``k_s``, rows ``edge .. n - edge``), ``posterior`` and
      ``log_posterior`` over them, and ``lnl`` (the best step's lnL at each).
    - ``modes`` -- see `step_modes`.
    - ``summary`` -- this animal's `step_summary` row.
    - ``rows`` and ``sigmoid`` (the fit's best optimum), for drawing against the data.
    """
    tests = step_test(fits).set_index("subjid")
    steps = {}
    for subjid, fit in sorted(fits.items()):
        rows = fit["rows"]
        y = rows["y"].to_numpy()
        chance = fit["variant"] == "chance"
        tau, log_post = _log_marginal(y, chance)
        kept = (tau >= edge) & (tau <= y.size - edge)
        tau, log_post = tau[kept], log_post[kept]
        post = np.exp(log_post - log_post.max())
        post /= post.sum()
        lnl = _step_lnl(y, chance)[tau]
        modes = _modes(tau, log_post, post, rows["ses"].to_numpy()[tau])
        multimodal = len(modes) > 1

        peak = int(tau[np.argmax(post)])
        mean = float((post * tau).sum())
        sd = float(np.sqrt(max((post * tau ** 2).sum() - mean ** 2, 0.0)))
        lo, hi = (int(tau[i]) for i in posterior_hdi(post, HDI_MASS))
        hours = rows["hours"].to_numpy(dtype=float)
        best = fit["optima"].iloc[0]
        constant = fit_constant(y)["loglik"]
        test = tests.loc[subjid]
        summary = {
            "subjid": subjid, "mode": fit["mode"], "variant": fit["variant"],
            "step": bool(test["step"]), "ΔlnL_vs_sigmoid": test["ΔlnL"],
            "multimodal": multimodal, "n_modes": len(modes),
            "k_s": peak, "k_s_mean": np.nan if multimodal else mean,
            "k_s_sd": np.nan if multimodal else sd, "hdi_lo": lo, "hdi_hi": hi,
            "ses": rows["ses"].iloc[peak], "hours": hours[peak],
            "hours_lo": hours[lo], "hours_hi": hours[hi],
            "initial": _CHANCE if chance else float(y[:peak].mean()),
            "final": float(y[peak:].mean()),
            "lnl_constant": constant,
            "ΔlnL_sigmoid_const": float(best["lnl"]) - constant,
            "ΔlnL_step_const": float(lnl.max()) - constant,
        }
        steps[subjid] = {"subjid": subjid, "mode": fit["mode"], "variant": fit["variant"],
                         "tau": tau, "posterior": post, "log_posterior": log_post,
                         "lnl": lnl, "modes": modes.assign(subjid=subjid, mode=fit["mode"]),
                         "summary": summary, "rows": rows, "sigmoid": best}
    return steps


def step_summary(steps: dict) -> pd.DataFrame:
    """One row per animal.

    - ``step``: the change is not distinguishable from a step (`step_test`);
      ``ΔlnL_vs_sigmoid`` is the sigmoid's lead over it.
    - ``k_s``: the posterior's peak; ``k_s_mean`` / ``k_s_sd`` (empty when
      ``multimodal``); ``hdi_lo`` / ``hdi_hi``: the 95% highest-density range, which
      spans the gap between modes when there are several.
    - ``ses`` / ``hours`` / ``hours_lo`` / ``hours_hi``: the peak and the range on the
      session and task-time axes; ``initial`` / ``final``: accuracy either side of the peak.
    - ``lnl_constant``, and ``ΔlnL_sigmoid_const`` / ``ΔlnL_step_const``: how far each
      model beats no change at all.
    """
    return pd.DataFrame([step["summary"] for step in steps.values()])


def step_modes(steps: dict) -> pd.DataFrame:
    """Every posterior mode, most mass first within an animal: its session ``ses``, peak
    ``k_s``, ``mass``, the stretch ``from`` / ``to`` it covers, and ``Δlog_post`` below the
    highest peak."""
    return pd.concat([step["modes"] for step in steps.values()], ignore_index=True)[
        ["subjid", "mode", "ses", "k_s", "mass", "from", "to", "Δlog_post"]]
