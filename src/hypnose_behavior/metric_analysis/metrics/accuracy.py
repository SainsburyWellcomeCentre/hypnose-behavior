# Defers evaluation of PEP-604 annotations (`X | None`), keeping this module
# importable on Python 3.9 for repos pinned there (hypnose-eeg-preprocessing).
from __future__ import annotations

"""Choice accuracy and response rate.

``rolling_reward_fraction`` is here rather than being a granularity of
``decision_accuracy`` on purpose: its denominator is the *window size*, so
timeouts -- and, unless the caller has already dropped them, aborts -- sit inside
it. See limit 1 in ``metric_analysis/resolvers.py``.
"""

import numpy as np
import pandas as pd

from hypnose_behavior.metric_analysis.metrics.common import (
    _aborted_mask,
    _flag,
    _reduce_rate,
)
from hypnose_behavior.metric_analysis.registry import metric, session_metric

__all__ = [
    "decision_accuracy_contributions", "decision_accuracy", "decision_accuracy_session",
    "global_choice_accuracy_contributions", "global_choice_accuracy",
    "global_choice_accuracy_session",
    "decision_accuracy_by_odor", "decision_accuracy_by_odor_session",
    "choice_timeout_rate_contributions", "choice_timeout_rate", "choice_timeout_rate_session",
    "response_rate_contributions", "response_rate", "response_rate_session",
    "rolling_reward_fraction",
]


def _by_odor_payload(out):
    """`decision_accuracy_by_odor`'s saved shape: `{}` when there are no rows.

    Not `as_dict`: a frame with columns but no rows would serialise as
    `{column: {}}`, which is not what this key has ever held.
    """
    return out.to_dict() if len(out) > 0 else {}


def decision_accuracy_contributions(trials):
    rtc = trials["response_time_category"]
    return ((rtc == "rewarded").astype(int),
            rtc.isin(["rewarded", "unrewarded"]).astype(int))


@metric(frame="trials", title="Decision Accuracy")
def decision_accuracy(trials):
    """rewarded / (rewarded + unrewarded)."""
    if trials.empty or "response_time_category" not in trials.columns:
        return 0, 0, np.nan
    return _reduce_rate(*decision_accuracy_contributions(trials))


@session_metric(decision_accuracy)
def decision_accuracy_session(results):
    trials = results.get("trial_data", pd.DataFrame())
    if trials.empty or "response_time_category" not in trials.columns:
        print("Decision Accuracy: no trial_data with response_time_category")
        return 0, 0, np.nan
    n_rew, denom, acc = decision_accuracy(trials)
    print(f"Decision Accuracy: {n_rew}/{denom} = {acc:.3f}")
    return n_rew, denom, acc


def global_choice_accuracy_contributions(trials):
    rtc = trials["response_time_category"]
    # Counts are summed, not or-ed: a trial flagged both ways contributes twice,
    # as it does today.
    return ((rtc == "rewarded").astype(int),
            rtc.isin(["rewarded", "unrewarded"]).astype(int)
            + _flag(trials, "fa_label", "FA_time_in").astype(int))


@metric(frame="trials", title="Global Choice Accuracy")
def global_choice_accuracy(trials):
    """rewarded / (rewarded + unrewarded + FA_time_in)."""
    if trials.empty or "response_time_category" not in trials.columns:
        return 0, 0, np.nan
    return _reduce_rate(*global_choice_accuracy_contributions(trials))


@session_metric(global_choice_accuracy)
def global_choice_accuracy_session(results):
    df = results.get("trial_data", pd.DataFrame())
    if df.empty or "response_time_category" not in df.columns:
        print("Global Choice Accuracy: no trial_data with response_time_category")
        return 0, 0, np.nan
    n_correct, n_total, accuracy = global_choice_accuracy(df)
    n_incorrect = int((df["response_time_category"] == "unrewarded").sum())
    n_fa_time_in = int(_flag(df, "fa_label", "FA_time_in").sum())
    print(f"Global Choice Accuracy: {n_correct}/{n_total} = {accuracy:.3f}")
    print(f"  - Correct choices: {n_correct}")
    print(f"  - Incorrect choices: {n_incorrect}")
    print(f"  - False alarms (FA Time In): {n_fa_time_in}")
    return n_correct, n_total, accuracy


@metric(frame="trials", title="Decision Accuracy by Odor", adapter=_by_odor_payload)
def decision_accuracy_by_odor(trials):
    """Per-odor `decision_accuracy`, plus a `_total` variant including timeouts."""
    if trials.empty or "response_time_category" not in trials.columns or "last_odor" not in trials.columns:
        return pd.DataFrame()

    def extract_odor_letter(odor_str):
        if pd.isna(odor_str):
            return np.nan
        if isinstance(odor_str, str) and odor_str.startswith("Odor"):
            return odor_str.replace("Odor", "")
        return odor_str

    df_local = trials.copy()
    df_local["odor_letter"] = df_local["last_odor"].apply(extract_odor_letter)

    rows = []
    for odor in sorted(df_local["odor_letter"].dropna().unique()):
        odor_trials = df_local[df_local["odor_letter"] == odor]
        n_rew = int((odor_trials["response_time_category"] == "rewarded").sum())
        n_unr = int((odor_trials["response_time_category"] == "unrewarded").sum())
        n_tmo = int((odor_trials["response_time_category"] == "timeout_delayed").sum())
        denom_ab = n_rew + n_unr
        denom_total = denom_ab + n_tmo
        rows.append({
            'odor': odor,
            'rewarded': n_rew,
            'unrewarded': n_unr,
            'timeout': n_tmo,
            'decision_accuracy_ab': n_rew / denom_ab if denom_ab > 0 else np.nan,
            'decision_accuracy_total': n_rew / denom_total if denom_total > 0 else np.nan,
            'denominator_ab': denom_ab,
            'denominator_total': denom_total,
        })

    return pd.DataFrame(rows).set_index('odor').sort_index()


@session_metric(decision_accuracy_by_odor)
def decision_accuracy_by_odor_session(results):
    df = results.get("trial_data", pd.DataFrame())
    if df.empty or "response_time_category" not in df.columns or "last_odor" not in df.columns:
        print("Decision Accuracy by Odor: no trial_data with response_time_category/last_odor")
        return pd.DataFrame()
    out = decision_accuracy_by_odor(df)

    def _fmt(v):
        return f"{v:.3f}" if not np.isnan(v) else "nan"

    print("Decision Accuracy by Odor:")
    for odor, r in out.iterrows():
        # int(): a row Series takes one common dtype, so these counts arrive as
        # floats and would render as "65.0 rewarded" in metrics_*.txt.
        n_rew, n_unr, n_tmo = int(r['rewarded']), int(r['unrewarded']), int(r['timeout'])
        d_ab, d_total = int(r['denominator_ab']), int(r['denominator_total'])
        print(f"  Odor {odor}: {n_rew} rewarded, {n_unr} unrewarded, {n_tmo} timeout")
        print(f"       Decision Accuracy AB: {n_rew}/{d_ab} = {_fmt(r['decision_accuracy_ab'])}, "
              f"Total: {n_rew}/{d_total} = {_fmt(r['decision_accuracy_total'])}")
    return out


def choice_timeout_rate_contributions(trials):
    completed = ~_aborted_mask(trials)
    return ((completed & _flag(trials, "response_time_category", "timeout_delayed")).astype(int),
            completed.astype(int))


@metric(frame="trials", title="Choice Timeout Rate")
def choice_timeout_rate(trials):
    """timeout_delayed / completed."""
    if trials.empty or "response_time_category" not in trials.columns:
        return 0, 0, np.nan
    return _reduce_rate(*choice_timeout_rate_contributions(trials))


@session_metric(choice_timeout_rate)
def choice_timeout_rate_session(results):
    df = results.get("trial_data", pd.DataFrame())
    if df.empty or "response_time_category" not in df.columns:
        print("Choice Timeout Rate: no trial_data/response_time_category")
        return 0, 0, np.nan
    n_tmo, denom, rate = choice_timeout_rate(df)
    print(f"Choice Timeout Rate: {n_tmo}/{denom} = {rate:.3f}")
    return n_tmo, denom, rate


def response_rate_contributions(trials):
    rtc = trials["response_time_category"]
    num = rtc.isin(["rewarded", "unrewarded"]).astype(int)
    return num, num + (rtc == "timeout_delayed").astype(int)


@metric(frame="trials", title="Response Rate")
def response_rate(trials):
    """(rewarded + unrewarded) / (rewarded + unrewarded + timeout)."""
    if trials.empty or "response_time_category" not in trials.columns:
        return 0, 0, np.nan
    return _reduce_rate(*response_rate_contributions(trials))


@session_metric(response_rate)
def response_rate_session(results):
    df = results.get("trial_data", pd.DataFrame())
    if df.empty or "response_time_category" not in df.columns:
        print("Response Rate: no trial_data/response_time_category")
        return 0, 0, np.nan
    num, denom, rate = response_rate(df)
    print(f"Response Rate: {num}/{denom} = {rate:.3f}")
    return num, denom, rate


@metric(frame="trials")
def rolling_reward_fraction(trials, window, *, step=1, include_avg=False, hr_only=False):
    """Rolling fraction of trials rewarded, divided by the **window**.

    Deliberately not `over_windows(decision_accuracy, ...)`.
    The denominator is the window size, so timeouts -- and, unless the caller has
    already dropped them, aborts -- sit inside it. The curve differs visibly from a
    rolling `decision_accuracy`, which is why this is a separately named metric rather
    than a granularity of an existing one.

    `include_avg` back-fills the warm-up, completing a not-yet-full window with
    the frame's overall rate so the series starts at the first trial instead of
    at trial `window`. `hr_only` narrows the numerator to hidden-rule rewards.

    Returns one value per row of `trials`, NaN where no window ends there.
    """
    n = len(trials)
    out = np.full(n, np.nan)
    if n == 0:
        return out

    numerator = _flag(trials, "response_time_category", "rewarded")
    if hr_only:
        hr = trials.get("hidden_rule_success")
        hr = (hr.fillna(False).astype(bool) if isinstance(hr, pd.Series)
              else pd.Series(False, index=trials.index))
        numerator = numerator & hr
    rewards = numerator.astype(int).to_numpy(dtype=float)
    overall = float(np.mean(rewards))

    if include_avg:
        for i in range(0, n, step):
            if i < window:
                avail = rewards[: i + 1]
                out[i] = (float(np.sum(avail)) + (window - len(avail)) * overall) / float(window)
            else:
                out[i] = float(np.mean(rewards[i - window + 1: i + 1]))
    else:
        for end in range(window, n + 1, step):
            out[end - 1] = float(np.mean(rewards[end - window: end]))
    return out
