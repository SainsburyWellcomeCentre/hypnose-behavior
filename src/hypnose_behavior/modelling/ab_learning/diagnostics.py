"""Whether completed trials alone describe the A/B learning.

Four diagnostics over the frames of `data.load_ab_data`, each a pure function of them:

- `initiation_rate` (D1) -- per session, the share of sampling attempts that became a
  trial. Separates learning to hold the poke from learning the odor-port association.
- `failed_attempt_accuracy` (D2) -- per session, port choice after a failed attempt
  against completed-trial accuracy. Odor is delivered on short pokes too, so a port visit
  after a failed attempt is also a readout of the association.
- `lose_shift_trials` / `lose_shift_summary` (D3) -- completed trials split by what the
  animal did on the failed attempt right before them. Odors repeat until an attempt
  initiates, so a wrong port visit on a failed attempt followed by the correct choice on
  the repeat can be solved by elimination.
- `choice_persistence` (D4) -- whether port choices on repeated attempts of one
  initiation agree with each other, and with the trial's choice, more than their accuracy
  alone implies.

D1-D3 proportions carry their count (``k`` of ``n``) and a 95% Wilson interval; D4 is
descriptive (see `choice_persistence`).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.stats.proportion import proportion_confint

__all__ = [
    "PRIOR_GROUPS",
    "choice_persistence",
    "failed_attempt_accuracy",
    "initiation_rate",
    "lose_shift_summary",
    "lose_shift_trials",
    "wilson_interval",
]

_SESSION = ["subjid", "ses", "date", "session_idx"]

# D3 groups, by the failed attempt right before a completed trial.
PRIOR_GROUPS = {
    "none": "no failed attempt",
    "no_visit": "failed, no port visit",
    "wrong_port": "failed, wrong port visited",
    "correct_port": "failed, correct port visited",
}


def wilson_interval(k, n, alpha: float = 0.05):
    """Wilson score interval for ``k`` successes of ``n``; NaN where ``n == 0``."""
    k = np.asarray(k, dtype=float)
    n = np.asarray(n, dtype=float)
    empty = n == 0
    lo, hi = proportion_confint(k, np.where(empty, 1, n), alpha=alpha, method="wilson")
    return np.where(empty, np.nan, lo), np.where(empty, np.nan, hi)


def _proportion(frame: pd.DataFrame, k: str, n: str, prefix: str) -> pd.DataFrame:
    """Add ``{prefix}`` / ``{prefix}_lo`` / ``{prefix}_hi`` from count columns ``k`` / ``n``."""
    lo, hi = wilson_interval(frame[k], frame[n])
    with np.errstate(invalid="ignore", divide="ignore"):
        frame[prefix] = frame[k] / frame[n].where(frame[n] > 0)
    frame[f"{prefix}_lo"], frame[f"{prefix}_hi"] = lo, hi
    return frame


def _sessions(data: dict) -> pd.DataFrame:
    return data["sessions"][_SESSION].copy()


def _count(frame: pd.DataFrame, mask, name: str) -> pd.Series:
    return frame[mask].groupby(_SESSION).size().rename(name)


def initiation_rate(data: dict) -> pd.DataFrame:
    """D1: per session, trials / (trials + failed attempts with a poke).

    A zero-poke failed attempt (the poke ended before the valve opened) is left out of the
    denominator and counted in ``n_zero_poke``.

    Columns: ``n_trials``, ``n_failed`` (poke > 0), ``n_zero_poke``, ``n_attempts``
    (trials + ``n_failed``), ``rate`` / ``rate_lo`` / ``rate_hi``.
    """
    trials, attempts = data["trials"], data["attempts"]
    out = _sessions(data).set_index(_SESSION)
    out = out.join(_count(trials, slice(None), "n_trials"))
    out = out.join(_count(attempts, ~attempts["zero_poke"].astype(bool), "n_failed"))
    out = out.join(_count(attempts, attempts["zero_poke"].astype(bool), "n_zero_poke"))
    out = out.fillna(0).astype(int).reset_index()
    out["n_attempts"] = out["n_trials"] + out["n_failed"]
    return _proportion(out, "n_trials", "n_attempts", "rate")


def failed_attempt_accuracy(data: dict) -> pd.DataFrame:
    """D2: per session, port-choice accuracy on completed trials and after failed attempts.

    One row per session and ``source``:

    - ``completed``: completed trials scored rewarded / unrewarded (``k`` correct of ``n``).
    - ``failed``: failed attempts with a poke that were followed by a port visit, scored
      against the attempt's odor.
    - ``zero_poke``: the same for zero-poke failed attempts, kept apart because the animal
      had left the port when the odor started.

    ``n_attempts`` / ``n_visits`` / ``visit_rate`` say how many attempts of that kind there
    were and how many led to a port visit (NaN for ``completed``). Accuracy columns:
    ``accuracy`` / ``accuracy_lo`` / ``accuracy_hi``.
    """
    trials, attempts = data["trials"], data["attempts"]
    base = _sessions(data).set_index(_SESSION)
    rows = []

    scored = trials[trials["correct"].notna()]
    completed = base.join(_count(scored, slice(None), "n"))
    completed = completed.join(_count(scored, scored["correct"].astype(bool), "k"))
    rows.append(completed.fillna(0).astype(int).assign(source="completed"))

    zero = attempts["zero_poke"].astype(bool)
    visit = attempts["port_visit"].astype(bool)
    hit = attempts["correct_port"].eq(True)
    for source, kind in (("failed", ~zero), ("zero_poke", zero)):
        part = base.join(_count(attempts, kind, "n_attempts"))
        part = part.join(_count(attempts, kind & visit, "n"))
        part = part.join(_count(attempts, kind & visit & hit, "k"))
        part = part.fillna(0).astype(int).assign(source=source)
        rows.append(part)

    out = pd.concat(rows).reset_index()
    out["n_visits"] = out["n"].where(out["source"] != "completed")
    out.loc[out["source"] == "completed", "n_attempts"] = np.nan
    with np.errstate(invalid="ignore", divide="ignore"):
        out["visit_rate"] = out["n_visits"] / out["n_attempts"].where(out["n_attempts"] > 0)
    out = _proportion(out, "k", "n", "accuracy")
    columns = _SESSION + ["source", "k", "n", "accuracy", "accuracy_lo", "accuracy_hi",
                          "n_attempts", "n_visits", "visit_rate"]
    return out[columns].sort_values(_SESSION + ["source"]).reset_index(drop=True)


def lose_shift_trials(data: dict) -> pd.DataFrame:
    """D3: completed, scored trials with ``prior``, the failed attempt right before them.

    ``prior`` is one of `PRIOR_GROUPS`: ``none`` (initiated on the first attempt),
    ``no_visit``, ``wrong_port`` or ``correct_port``. It is read from the attempt numbered
    one below the trial's, i.e. the last failure of the same initiation, zero-poke attempts
    included: elimination needs the unrewarded visit, not a perceived odor.

    Adds ``prior_zero_poke`` (that attempt had no poke; False for ``none``).
    """
    trials = data["trials"]
    trials = trials[trials["correct"].notna()].copy()
    attempts = data["attempts"]
    attempts = attempts[~attempts["trailing"].astype(bool)]

    key = ["subjid", "ses", "run_id", "global_trial_id"]
    last = (attempts.sort_values("attempt_number")
            .groupby(key, as_index=False).tail(1)
            .assign(global_trial_id=lambda f: f["global_trial_id"].astype(int)))
    last = last.set_index(key)[["port_visit", "correct_port", "zero_poke"]]
    joined = trials.join(last, on=key)

    has_prior = joined["port_visit"].notna()
    visited = joined["port_visit"].eq(True)
    prior = np.select(
        [~has_prior, ~visited, joined["correct_port"].eq(True)],
        ["none", "no_visit", "correct_port"],
        default="wrong_port",
    )
    trials["prior"] = pd.Categorical(prior, categories=list(PRIOR_GROUPS))
    trials["prior_zero_poke"] = joined["zero_poke"].eq(True).to_numpy()
    return trials


def lose_shift_summary(data: dict, by=("subjid",)) -> pd.DataFrame:
    """D3: accuracy per ``prior`` group, with its share of the completed trials.

    ``by`` groups the rows further -- ``("subjid",)`` per animal, ``("subjid",
    "session_idx")`` per session, ``()`` pooled. Columns: ``k`` / ``n`` / ``accuracy`` (with
    Wilson bounds) and ``share`` (``n`` over all scored trials of the group).
    """
    trials = lose_shift_trials(data)
    by = list(by)
    counts = (trials.groupby(by + ["prior"], observed=False)["correct"]
              .agg(n="size", k=lambda s: int(s.astype(bool).sum()))
              .reset_index())
    totals = counts.groupby(by)["n"].transform("sum") if by else counts["n"].sum()
    counts["share"] = counts["n"] / totals
    return _proportion(counts, "k", "n", "accuracy")


_TRIAL_KEY = ["subjid", "ses", "run_id", "global_trial_id"]
_ODOR_KEY = ["subjid", "session_idx", "odor"]


def _accuracy_by_odor(frame: pd.DataFrame, correct: str) -> pd.Series:
    """Accuracy per animal, session and odor."""
    scored = frame[frame[correct].notna()]
    return scored[correct].astype(bool).groupby([scored[c] for c in _ODOR_KEY]).mean()


def _agreement_summary(pairs: pd.DataFrame, comparison: str, n_trials: pd.Series):
    """Observed agreement against the mean per-pair chance, per animal."""
    grouped = pairs.groupby("subjid")
    out = pd.DataFrame({
        "comparison": comparison,
        "n_trials": n_trials,
        "n_pairs": grouped.size(),
        "k": grouped["agree"].sum(),
        "chance": grouped["chance"].mean(),
    }).rename_axis("subjid").reset_index()
    out["agreement"] = out["k"] / out["n_pairs"]
    out["excess"] = out["agreement"] - out["chance"]
    return out


def choice_persistence(data: dict, min_choices: int = 2) -> dict:
    """D4: do repeated port choices within one initiation agree beyond their accuracy?

    A *choice attempt* is a failed attempt followed by a port visit (zero-poke attempts
    included: the visit is a choice either way). Two comparisons, each over the trials it
    needs:

    - ``attempt_to_attempt``: consecutive choice attempts (in attempt order, attempts
      without a visit skipped) choose the same port. A trial needs ``min_choices`` of
      them, since the comparison is between two attempts.
    - ``last_attempt_to_trial``: the last choice attempt and the trial's choice agree.
      One choice attempt is enough, the trial's own choice being the other half of the
      pair; completed trials with a scored choice only.

    Chance is the agreement two independent choices would reach given their accuracies,
    ``p1*p2 + (1-p1)*(1-p2)`` -- ``p**2 + (1-p)**2`` when both come from the same kind of
    choice. ``p`` is taken per animal, session and odor (choice-attempt accuracy for
    attempts, completed-trial accuracy for trials), since accuracy rises with training and
    a pooled ``p`` would understate chance. Each pair carries its own chance; ``chance``
    in the summary is their mean.

    **Descriptive only: no interval is reported.** Consecutive pairs of one trial share an
    attempt and are not independent, so a pair-level binomial interval would be
    anticonservative. ``last_attempt_to_trial`` takes one pair per trial, but its pairs
    still share an animal and a session, which a binomial interval would ignore just the
    same.

    Returns ``{"summary", "distribution"}``: ``summary`` has one row per animal and
    comparison with ``n_trials`` (the trials that comparison could use), ``n_pairs``,
    ``k`` agreeing, ``chance``, ``agreement`` and ``excess`` (agreement - chance);
    ``distribution`` counts all trials by their number of choice attempts, one row per
    animal.
    """
    trials = data["trials"]
    attempts = data["attempts"]
    choices = attempts[~attempts["trailing"].astype(bool) & attempts["port_visit"].astype(bool)]
    choices = choices.assign(global_trial_id=choices["global_trial_id"].astype(int))
    choices = choices.sort_values(_TRIAL_KEY + ["attempt_number"])

    per_trial = choices.groupby(_TRIAL_KEY).size().rename("n_choices")
    trials = trials.join(per_trial, on=_TRIAL_KEY)
    trials["n_choices"] = trials["n_choices"].fillna(0).astype(int)
    distribution = (trials.groupby(["subjid", "n_choices"]).size()
                    .unstack(fill_value=0).rename_axis(columns="choice attempts"))

    p_attempt = _accuracy_by_odor(attempts[attempts["port_visit"].astype(bool)],
                                  "correct_port").rename("p_attempt")
    p_trial = _accuracy_by_odor(trials, "correct").rename("p_trial")

    # (a) consecutive choice attempts of one trial, so the trial needs `min_choices`.
    repeated = trials[trials["n_choices"] >= min_choices]
    paired = choices.join(repeated.set_index(_TRIAL_KEY)["n_choices"], on=_TRIAL_KEY,
                          how="inner")
    previous = paired.groupby(_TRIAL_KEY)["port"].shift()
    consecutive = paired[previous.notna()].assign(
        agree=lambda f: f["port"] == previous[f.index])
    consecutive = consecutive.join(p_attempt, on=_ODOR_KEY)
    p = consecutive["p_attempt"]
    consecutive["chance"] = p**2 + (1 - p)**2

    # (b) last choice attempt against the trial's own choice, so one attempt is enough.
    last = choices.groupby(_TRIAL_KEY).tail(1).set_index(_TRIAL_KEY)["port"].rename("last_port")
    scored = trials[(trials["n_choices"] > 0) & trials["choice"].isin(("A", "B"))]
    scored = scored.join(last, on=_TRIAL_KEY)
    scored = scored.join(p_attempt, on=_ODOR_KEY).join(p_trial, on=_ODOR_KEY)
    scored["agree"] = scored["last_port"] == scored["choice"]
    pa, pt = scored["p_attempt"], scored["p_trial"]
    scored["chance"] = pa * pt + (1 - pa) * (1 - pt)

    summary = pd.concat([
        _agreement_summary(consecutive, "attempt_to_attempt",
                           repeated.groupby("subjid").size()),
        _agreement_summary(scored, "last_attempt_to_trial",
                           scored.groupby("subjid").size()),
    ], ignore_index=True)
    return {"summary": summary, "distribution": distribution}
