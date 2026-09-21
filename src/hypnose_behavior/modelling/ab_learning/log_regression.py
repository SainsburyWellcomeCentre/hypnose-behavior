"""Logistic regression of accuracy on session and on position within a session.

Three nested models per animal, fitted to the choice rows of one analysis mode (see
`data.build_choices`):

===== ============================== =========================================
M_a   ``logit p = a_s``              every gain falls between sessions
M_b   ``logit p = a_s + b * x``      one within-session slope, shared by sessions
M_c   ``logit p = a_s + b_s * x``    a within-session slope per session
===== ============================== =========================================

``x`` is a row's normalized position in its session, 0 at the first choice and 1 at the
last, so sessions of any length are comparable as total gains rather than as rates.
Session enters as a factor: each session takes a free level, which leaves the shape of
the across-session change unconstrained.

The test of interest is **M_c against M_a** (df = one per session). M_b averages a
positive slope early in training against a negative one later and can come back flat
while both are real; it stays useful as the descriptive average (`shared_slope`).

From M_c the change in log-odds splits into a within-session and an overnight part,

    W_s = b_s                        within session s
    O_s = a_{s+1} - a_s - b_s        across the boundary that follows it

which telescope to ``a_N - a_1 = sum(W) + sum(O)``, one W paired with one O at every
boundary. The last session's W has no boundary after it, so it stays out of that sum and
is reported on its own as ``final_within``: counting it would put an extra within-session
term in a total that has no matching night, which is the one asymmetry a share of the
total cannot survive.

Every reported quantity is a linear contrast of M_c's coefficients, so its standard error
comes from the full covariance matrix rather than from adding component errors -- O_s in
particular contrasts the two least-constrained points of the fit, the extrapolated end
of one session and start of the next.

A session's slope is a **straight line in x**, so a within-session profile that rises and
then falls is fitted as the single line closest to it: W_s reports that compromise slope
rather than the profile, and O_s absorbs whatever the line missed at the session's edges.
Read the two against the accuracy binned by within-session position, which is what says
whether a straight line describes the profile at all.

A session whose choices are all correct or all incorrect separates the fit perfectly and
sends its level to +/-inf, so `fit_session_models` refuses to fit one. `MIN_TRIALS` is
the default floor on session length: it keeps those sessions out, and keeps out the
short sessions whose b_s a handful of trials cannot identify.

Sessions below the floor are dropped whole, so two fitted sessions can have a dropped one
between them. The boundary contrast then covers that session's own within-session gain
and two nights rather than one, which is not an overnight gain at all; such a boundary is
kept apart as ``across_gap`` and reported beside sum(O) rather than inside it. The
telescoping identity stays exact -- ``a_N - a_1 = sum(W) + sum(O) + sum(gap)`` -- so what
a gap makes unattributable is visible as its own term instead of inflating the other two.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

from hypnose_behavior.modelling.ab_learning.data import (
    build_choices,
    within_session_position,
)

__all__ = [
    "MIN_TRIALS",
    "MODEL_FORMULAS",
    "MODES",
    "fit_session_models",
    "gain_decomposition",
    "model_comparison",
    "overnight_share",
    "regression_frame",
    "session_levels",
    "shared_slope",
]

# Which choices are rows: completed trials only, or every choice attempt (plan 0.8).
MODES = ("completed", "attempts")

MODEL_FORMULAS = {
    "M_a": "y ~ C(session_idx)",
    "M_b": "y ~ C(session_idx) + x",
    "M_c": "y ~ C(session_idx) * x",
}

# Shortest session that still gets a level and a slope of its own. Below ~20 choices a
# within-session slope is barely identified, and a session short enough to be all
# correct separates the fit outright.
MIN_TRIALS = 20

# The nested pairs scored by `model_comparison`, null first.
_COMPARISONS = (("M_a", "M_b"), ("M_a", "M_c"), ("M_b", "M_c"))

_SESSION = ["subjid", "ses", "date", "session_idx"]
_CONTRAST_COLUMNS = ["value", "se", "z", "p", "lo", "hi"]


def _rows(data: dict, mode: str) -> pd.DataFrame:
    """The choice rows of one mode, in time order within an animal."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    choices = build_choices(data)
    if mode == "completed":
        choices = choices[choices["is_completed"]]
    return choices.sort_values(["subjid", "time"])


def _counts(rows: pd.DataFrame) -> pd.DataFrame:
    """Choices (``n``), correct choices (``k``) and accuracy per session."""
    counts = (rows.groupby(_SESSION, as_index=False)["correct"]
              .agg(n="size", k=lambda s: int(s.astype(bool).sum())))
    counts["accuracy"] = counts["k"] / counts["n"]
    return counts


def _frame(rows: pd.DataFrame, min_trials: int) -> pd.DataFrame:
    """The kept rows with the regression's ``x`` and ``y``."""
    n = rows.groupby(["subjid", "session_idx"])["correct"].transform("size")
    kept = rows[n >= min_trials]
    kept = kept.assign(x=within_session_position(kept), y=kept["correct"].astype(int))
    return kept[_SESSION + ["time", "x", "y"]].reset_index(drop=True)


def regression_frame(data: dict, *, mode: str = "completed",
                     min_trials: int = MIN_TRIALS) -> pd.DataFrame:
    """The rows the models are fitted on: one per choice of a session long enough.

    ``mode`` is ``"completed"`` (completed trials) or ``"attempts"`` (every choice
    attempt); ``min_trials`` drops a session whole. Columns: the session identity,
    ``time``, ``x`` (position in the session, 0 at its first kept choice and 1 at its
    last) and ``y`` (1 when the choice was correct).

    ``x`` is recomputed over the rows kept here, so it always runs over the axis being
    fitted and never mixes the completed-trial axis with the choice-attempt one.
    """
    return _frame(_rows(data, mode), min_trials)


def _degenerate(frame: pd.DataFrame) -> pd.DataFrame:
    """Kept sessions with no incorrect, or no correct, choice."""
    counts = frame.groupby(_SESSION, as_index=False)["y"].agg(n="size", k="sum")
    return counts[(counts["k"] == 0) | (counts["k"] == counts["n"])]


def fit_session_models(data: dict, *, mode: str = "completed",
                       min_trials: int = MIN_TRIALS) -> dict:
    """Fit M_a, M_b and M_c per animal.

        fits = fit_session_models(load_ab_data(...))
        fits = fit_session_models(ab, mode="attempts")      # the robustness re-run

    Returns ``{subjid: fit}``, each fit a dict of:

    - ``models``    -- ``{"M_a", "M_b", "M_c"}`` of fitted statsmodels results.
    - ``frame``     -- the animal's rows from `regression_frame`.
    - ``sessions``  -- every session of the animal with ``n`` / ``k`` / ``accuracy`` and
      ``kept`` (it reached ``min_trials`` and entered the fit).
    - ``mode`` and ``min_trials``, so the downstream tables can carry them.

    Raises `ValueError` if a kept session holds only correct or only incorrect choices:
    its level is then unidentified and the fit would report a level of +/-20 or so with
    an enormous standard error rather than failing. Raising ``min_trials`` removes it.
    """
    rows = _rows(data, mode)
    sessions = _counts(rows)
    sessions["kept"] = sessions["n"] >= min_trials
    frame = _frame(rows, min_trials)

    separated = _degenerate(frame)
    if len(separated):
        listed = ", ".join(f"sub-{r.subjid:03d} ses-{r.ses} ({r.k}/{r.n} correct)"
                           for r in separated.itertuples())
        raise ValueError(
            f"{len(separated)} kept session(s) hold only correct or only incorrect "
            f"choices, which separates the logistic fit perfectly: {listed}. Raise "
            f"min_trials above {int(separated['n'].max())} to drop them.")

    fits = {}
    for subjid, part in frame.groupby("subjid"):
        models = {name: smf.logit(formula, data=part).fit(disp=0)
                  for name, formula in MODEL_FORMULAS.items()}
        unconverged = [name for name, m in models.items() if not m.mle_retvals["converged"]]
        if unconverged:
            warnings.warn(f"sub-{subjid:03d}: {', '.join(unconverged)} did not converge; "
                          f"its coefficients and every quantity derived from them are "
                          f"unreliable.", RuntimeWarning, stacklevel=2)
        fits[int(subjid)] = {
            "subjid": int(subjid),
            "mode": mode,
            "min_trials": min_trials,
            "models": models,
            "frame": part.reset_index(drop=True),
            "sessions": (sessions[sessions["subjid"] == subjid]
                         .sort_values("session_idx").reset_index(drop=True)),
        }
    return fits


def _kept_sessions(fit: dict) -> pd.DataFrame:
    """The animal's fitted sessions, in order."""
    sessions = fit["sessions"]
    return sessions[sessions["kept"]].reset_index(drop=True)


def _design_row(names: list, base: int, session_idx: int, x: float) -> np.ndarray:
    """M_c's design row for a choice at position ``x`` of session ``session_idx``.

    ``base`` is the session the factor holds as its reference level, whose terms are the
    model's intercept and its plain ``x`` slope.
    """
    row = np.zeros(len(names))
    row[names.index("Intercept")] = 1.0
    row[names.index("x")] = x
    if session_idx != base:
        level = f"C(session_idx)[T.{session_idx}]"
        if level not in names:
            raise ValueError(f"M_c has no design column {level!r}; its columns are {names}")
        row[names.index(level)] = 1.0
        row[names.index(f"{level}:x")] = x
    return row


def _contrasts(fit: dict):
    """``(model, rows)`` for M_c, where ``rows(session_idx, x)`` is its design row."""
    model = fit["models"]["M_c"]
    names = list(model.model.exog_names)
    sessions = _kept_sessions(fit)["session_idx"].tolist()
    base = sessions[0]
    return model, (lambda s, x: _design_row(names, base, s, x))


def _estimate(model, contrast) -> pd.DataFrame:
    """Effect, standard error, z, p and 95% interval for each row of ``contrast``."""
    test = model.t_test(np.atleast_2d(contrast))
    lo, hi = np.asarray(test.conf_int()).T
    return pd.DataFrame({
        "value": np.ravel(test.effect), "se": np.ravel(test.sd),
        "z": np.ravel(test.tvalue), "p": np.ravel(test.pvalue), "lo": lo, "hi": hi,
    })


def _ratio(model, numerator: np.ndarray, denominator: np.ndarray) -> dict:
    """A ratio of two contrasts, with a delta-method standard error and interval.

    The delta method linearizes around the estimate, so it describes the ratio only
    while the denominator is well away from zero; ``total_z`` in `overnight_share` is
    what says whether it is.
    """
    params = np.asarray(model.params, dtype=float)
    cov = np.asarray(model.cov_params(), dtype=float)
    num, den = float(numerator @ params), float(denominator @ params)
    var_num = float(numerator @ cov @ numerator)
    var_den = float(denominator @ cov @ denominator)
    covariance = float(numerator @ cov @ denominator)
    ratio = num / den
    variance = (var_num - 2 * ratio * covariance + ratio**2 * var_den) / den**2
    se = float(np.sqrt(variance)) if variance > 0 else np.nan
    half = stats.norm.ppf(0.975) * se
    return {"value": ratio, "se": se, "lo": ratio - half, "hi": ratio + half}


def _identity(fit: dict, columns=("subjid", "mode")) -> dict:
    return {c: fit[c] for c in columns}


def model_comparison(fits: dict) -> pd.DataFrame:
    """Likelihood ratio tests between the three nested models, per animal.

    One row per animal and comparison. **The hypothesis test is ``M_a vs M_c``**, with
    one degree of freedom per session: it asks whether any session has a within-session
    slope. ``M_a vs M_b`` asks only whether the average slope differs from zero, which a
    positive early slope and a negative late one can cancel.

    The hypothesis under test is the null, so a large ``p`` is not evidence for it --
    read it next to the size of the gains in `gain_decomposition`.

    Columns: ``comparison``, ``llf_null`` / ``llf_full``, ``aic_null`` / ``aic_full``,
    ``lr`` (``2 * (llf_full - llf_null)``), ``df``, ``p``, and the animal's ``n``
    (choices) and ``n_sessions``.
    """
    rows = []
    for subjid in sorted(fits):
        fit = fits[subjid]
        models = fit["models"]
        shared = {**_identity(fit), "n": len(fit["frame"]),
                  "n_sessions": int(fit["sessions"]["kept"].sum())}
        for null, full in _COMPARISONS:
            lr = 2 * (models[full].llf - models[null].llf)
            df = int(models[full].df_model - models[null].df_model)
            rows.append({
                **shared, "comparison": f"{null} vs {full}",
                "llf_null": models[null].llf, "llf_full": models[full].llf,
                "aic_null": models[null].aic, "aic_full": models[full].aic,
                "lr": lr, "df": df, "p": float(stats.chi2.sf(lr, df)),
            })
    return pd.DataFrame(rows)


def shared_slope(fits: dict) -> pd.DataFrame:
    """M_b's single within-session slope per animal, in log-odds across a whole session.

    The descriptive average: positive is a warm-up over the session, negative a decline.
    It is not the hypothesis test -- see `model_comparison`.
    """
    rows = []
    for subjid in sorted(fits):
        fit = fits[subjid]
        model = fit["models"]["M_b"]
        slope = _estimate(model, np.eye(len(model.params))[model.model.exog_names.index("x")])
        rows.append({**_identity(fit), **slope.iloc[0].to_dict()})
    return pd.DataFrame(rows)


def session_levels(fits: dict) -> pd.DataFrame:
    """M_c's fitted log-odds at each session's first and last choice.

    One row per animal and fitted session: ``logit_start`` is ``a_s`` and ``logit_end``
    is ``a_s + b_s``, each with its standard error and 95% interval (suffixes ``_se`` /
    ``_lo`` / ``_hi``). ``n``, ``k`` and ``accuracy`` are the session's observed counts,
    so a fitted edge can be read against what the session actually did.
    """
    parts = []
    for subjid in sorted(fits):
        fit = fits[subjid]
        model, row = _contrasts(fit)
        sessions = _kept_sessions(fit)
        levels = sessions[_SESSION + ["n", "k", "accuracy"]].assign(mode=fit["mode"])
        for edge, x in (("start", 0.0), ("end", 1.0)):
            estimate = _estimate(model, np.array([row(s, x) for s in sessions["session_idx"]]))
            estimate.columns = [f"logit_{edge}" if c == "value" else f"logit_{edge}_{c}"
                                for c in estimate.columns]
            levels = pd.concat([levels, estimate[[f"logit_{edge}", f"logit_{edge}_se",
                                                  f"logit_{edge}_lo", f"logit_{edge}_hi"]]],
                               axis=1)
        parts.append(levels)
    return pd.concat(parts, ignore_index=True)


def gain_decomposition(fits: dict) -> pd.DataFrame:
    """W_s and O_s: the change in log-odds within each session and across each boundary.

    One row per animal and component:

    - ``within``     -- W_s, over session ``ses``.
    - ``overnight``  -- O_s, from the end of ``ses`` to the start of ``ses_next``, the
      two recorded back to back.
    - ``across_gap`` -- the same contrast where a session between the two was dropped for
      length, so it also holds that session's own gain and a second night. Not an
      overnight gain, and kept out of sum(O).

    ``value`` carries its standard error, z, p and 95% interval, all from M_c's full
    covariance matrix. ``gap_days`` is the distance between the two dates -- a weekend
    makes an ``overnight`` boundary three days long without anything unrecorded falling
    inside it -- and ``sessions_skipped`` how many dropped sessions do.

    ``in_total`` marks the rows that make up the total change in `overnight_share`: every
    boundary, and every within row but the last session's, which has no boundary to pair
    with.
    """
    parts = []
    for subjid in sorted(fits):
        fit = fits[subjid]
        model, row = _contrasts(fit)
        sessions = _kept_sessions(fit)
        index = sessions["session_idx"].tolist()
        dates = pd.to_datetime(sessions["date"], format="%Y%m%d")

        within = sessions[_SESSION].copy()
        within["component"] = "within"
        within["in_total"] = within["session_idx"] != index[-1]
        within = pd.concat(
            [within, _estimate(model, np.array([row(s, 1.0) - row(s, 0.0) for s in index]))],
            axis=1)

        skipped = np.diff(index) - 1
        boundary = sessions.iloc[:-1][_SESSION].reset_index(drop=True)
        boundary["component"] = np.where(skipped > 0, "across_gap", "overnight")
        boundary["in_total"] = True
        boundary["session_idx_next"] = index[1:]
        boundary["ses_next"] = sessions["ses"].to_numpy()[1:]
        boundary["date_next"] = sessions["date"].to_numpy()[1:]
        boundary["gap_days"] = ((dates.to_numpy()[1:] - dates.to_numpy()[:-1])
                                / np.timedelta64(1, "D"))
        boundary["sessions_skipped"] = skipped
        boundary = pd.concat(
            [boundary,
             _estimate(model, np.array([row(b, 0.0) - row(a, 1.0)
                                        for a, b in zip(index, index[1:])]))],
            axis=1)

        parts.append(pd.concat([within, boundary], ignore_index=True).assign(
            **_identity(fit, ("mode",))))
    columns = (_SESSION + ["mode", "component", "in_total", "session_idx_next", "ses_next",
                           "date_next", "gap_days", "sessions_skipped"] + _CONTRAST_COLUMNS)
    return pd.concat(parts, ignore_index=True)[columns]


def _sum_estimate(model, contrast_rows) -> dict:
    """The estimate of a sum of contrast rows; an exact zero when there are none."""
    if len(contrast_rows) == 0:
        return {"value": 0.0, "se": 0.0, "z": np.nan, "p": np.nan, "lo": 0.0, "hi": 0.0}
    return _estimate(model, np.sum(contrast_rows, axis=0)).iloc[0].to_dict()


def overnight_share(fits: dict) -> pd.DataFrame:
    """How much of each animal's total change in log-odds happened between sessions.

    One row per animal, every figure a contrast of M_c and so carrying an exact standard
    error. ``total`` is the change from the first fitted session's start to the last
    one's start, and splits exactly:

        total = within_total + overnight_total + gap_total

    ``within_total`` is ``sum(W_s)`` over every fitted session but the last,
    ``overnight_total`` is ``sum(O_s)`` over the boundaries between sessions recorded
    back to back, and ``gap_total`` the boundaries with a dropped session inside them,
    which hold that session's own gain as well as two nights and are therefore
    attributable to neither (0 when there are none).

    ``final_within`` is the last session's W_s, which has no boundary to pair with;
    ``total_full`` is the whole change over training, ``total + final_within``.

    ``overnight_share`` is ``overnight_total / total`` with a delta-method interval.
    **It only reads as a percentage while the two components point the same way.** An
    animal that gains within sessions and gives part of it back overnight has a negative
    ``overnight_total`` against a positive ``total``, and a share below 0 or above 1;
    ``mixed_signs`` marks that row, and the two totals in log-odds are what to report
    there. ``total_z`` is ``total / total_se``: the delta-method interval widens without
    bound as the denominator approaches zero, so treat the share as undefined for an
    animal whose total change is not itself distinguishable from zero.
    """
    rows = []
    for subjid in sorted(fits):
        fit = fits[subjid]
        model, row = _contrasts(fit)
        index = _kept_sessions(fit)["session_idx"].tolist()
        boundaries = list(zip(index, index[1:]))

        within_rows = [row(s, 1.0) - row(s, 0.0) for s in index[:-1]]
        overnight_rows = [row(b, 0.0) - row(a, 1.0) for a, b in boundaries if b == a + 1]
        gap_rows = [row(b, 0.0) - row(a, 1.0) for a, b in boundaries if b != a + 1]
        total = row(index[-1], 0.0) - row(index[0], 0.0)

        w = _sum_estimate(model, within_rows)
        o = _sum_estimate(model, overnight_rows)
        g = _sum_estimate(model, gap_rows)
        t = _estimate(model, total).iloc[0]
        final = _estimate(model, row(index[-1], 1.0) - row(index[-1], 0.0)).iloc[0]
        full = _estimate(model, row(index[-1], 1.0) - row(index[0], 0.0)).iloc[0]
        share = _ratio(model, np.sum(overnight_rows, axis=0), total)

        rows.append({
            **_identity(fit), "n": len(fit["frame"]), "n_sessions": len(index),
            "within_total": w["value"], "within_total_se": w["se"],
            "overnight_total": o["value"], "overnight_total_se": o["se"],
            "gap_total": g["value"], "gap_total_se": g["se"],
            "total": t["value"], "total_se": t["se"], "total_z": t["z"],
            "final_within": final["value"], "final_within_se": final["se"],
            "total_full": full["value"], "total_full_se": full["se"],
            "overnight_share": share["value"], "overnight_share_lo": share["lo"],
            "overnight_share_hi": share["hi"],
            "mixed_signs": bool(np.sign(w["value"]) != np.sign(o["value"])),
        })
    return pd.DataFrame(rows)
