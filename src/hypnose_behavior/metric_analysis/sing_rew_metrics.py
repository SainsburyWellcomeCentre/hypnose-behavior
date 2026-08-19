"""Single-reward ("singrew") outcome-category metrics.

Sorts every trial of a single-reward session into one of the signal-detection
outcomes -- Hit, Miss, False Alarm, Correct Rejection -- plus two "premature"
categories for ambiguous aborts, using the trial_data columns produced by the
single-reward trial classification (``is_aborted``, ``response_time_category``,
``reward_determinacy``, ``determined_final_odor``, ``fa_label``, ``false_response``,
``fa_port``).

For each (sub)category we record the trial count and the list of ``global_trial_id``s
so the membership can be recovered downstream.

Classification (mutually exclusive)
-----------------------------------
Hit:
  - is_aborted=False & response_time_category in {rewarded, unrewarded}
  - is_aborted=True  & reward_determinacy=rewarded    & fa_label != nFA
Miss:
  - is_aborted=False & response_time_category = timeout_delayed
  - is_aborted=True  & reward_determinacy=rewarded    & fa_label = nFA
False Alarm:
  - false_response = True
  - is_aborted=True  & reward_determinacy=nonrewarded & fa_label != nFA
Correct Rejection:
  - false_response = False
  - is_aborted=True  & reward_determinacy=nonrewarded & fa_label = nFA
Premature (ambiguous aborts):
  - premature_port_entry: is_aborted=True & reward_determinacy=ambiguous & fa_label != nFA
  - premature_abort:      is_aborted=True & reward_determinacy=ambiguous & fa_label = nFA

Subcategories
-------------
Hit:  rewarded_hit (completed, rewarded) / port_error_hit (completed, unrewarded) /
      anticipatory_hit (aborted, determined_final_odor port == fa_port) /
      anticipatory_port_error (aborted, determined_final_odor port != fa_port)
Miss: omission_miss (completed timeout_delayed) / forfeit_miss (aborted, rewarded, nFA)
False Alarm: completed_fa (false_response=True) / aborted_fa (aborted, nonrewarded, FA)
Correct Rejection: passive_cr (false_response=False) /
      active_cr (aborted, nonrewarded, nFA)

Anything matching no rule (e.g. an aborted ``off_protocol`` trial, or a session
missing the singrew columns) is collected under ``uncategorized``. A ``validation``
block flags any ``global_trial_id`` that is in no category or in more than one
category, so the partition can be audited.
"""

from __future__ import annotations

import math
from collections import defaultdict
from statistics import NormalDist

import pandas as pd


# main category -> ordered list of its subcategories
CATEGORY_SUBCATEGORIES = {
    "hit": ["rewarded_hit", "port_error_hit", "anticipatory_hit", "anticipatory_port_error"],
    "miss": ["omission_miss", "forfeit_miss"],
    "false_alarm": ["completed_fa", "aborted_fa"],
    "correct_rejection": ["passive_cr", "active_cr"],
}

# top-level categories that have no subcategories
PREMATURE_CATEGORIES = ["premature_port_entry", "premature_abort"]

# every mutually-exclusive top-level category (excludes "uncategorized")
MAIN_CATEGORIES = list(CATEGORY_SUBCATEGORIES.keys()) + PREMATURE_CATEGORIES


def _to_bool(value):
    """Coerce a possibly-NaN / numpy / string boolean to a Python bool, or None."""
    if value is None:
        return None
    if isinstance(value, str):
        s = value.strip().lower()
        if s in ("true", "1"):
            return True
        if s in ("false", "0", ""):
            return False
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return bool(value)


def _odor_to_identity(odor):
    """Map a final-odor name to its reward identity letter (OdorA -> "A"), else None."""
    if odor is None:
        return None
    try:
        if pd.isna(odor):
            return None
    except (TypeError, ValueError):
        pass
    s = str(odor).strip()
    if s.startswith("Odor"):
        s = s[4:].strip()
    return s or None


def _port_to_identity(port):
    """Map a reward-port id (1 -> "A", 2 -> "B") to its identity letter, else None."""
    try:
        p = int(port)
    except (TypeError, ValueError):
        return None
    return {1: "A", 2: "B"}.get(p)


def _fa_is_false_alarm(fa_label):
    """True if fa_label denotes an actual false alarm (i.e. present and != 'nFA')."""
    if fa_label is None:
        return False
    try:
        if pd.isna(fa_label):
            return False
    except (TypeError, ValueError):
        pass
    return str(fa_label) != "nFA"


def _fa_is_nfa(fa_label):
    """True if fa_label is exactly 'nFA' (no false alarm)."""
    return str(fa_label) == "nFA"


def _classify_trial(row):
    """Return (main_category, subcategory) for one trial row.

    ``subcategory`` is None for the premature categories and for "uncategorized".
    """
    is_aborted = _to_bool(row.get("is_aborted")) or False
    rtc = row.get("response_time_category")
    rd = row.get("reward_determinacy")
    fa_label = row.get("fa_label")
    fr = _to_bool(row.get("false_response"))
    fa_is_fa = _fa_is_false_alarm(fa_label)
    fa_is_nfa = _fa_is_nfa(fa_label)

    # --- HIT ---
    if (not is_aborted) and rtc in ("rewarded", "unrewarded"):
        return "hit", ("rewarded_hit" if rtc == "rewarded" else "port_error_hit")
    if is_aborted and rd == "rewarded" and fa_is_fa:
        expected = _odor_to_identity(row.get("determined_final_odor"))
        actual = _port_to_identity(row.get("fa_port"))
        if expected is not None and actual is not None and expected == actual:
            return "hit", "anticipatory_hit"
        return "hit", "anticipatory_port_error"

    # --- MISS ---
    if (not is_aborted) and rtc == "timeout_delayed":
        return "miss", "omission_miss"
    if is_aborted and rd == "rewarded" and fa_is_nfa:
        return "miss", "forfeit_miss"

    # --- FALSE ALARM ---
    if (not is_aborted) and rd == "nonrewarded" and fr is True:
        return "false_alarm", "completed_fa"
    if is_aborted and rd == "nonrewarded" and fa_is_fa:
        return "false_alarm", "aborted_fa"

    # --- CORRECT REJECTION ---
    if (not is_aborted) and rd == "nonrewarded" and fr is False:
        return "correct_rejection", "passive_cr"
    if is_aborted and rd == "nonrewarded" and fa_is_nfa:
        return "correct_rejection", "active_cr"

    # --- PREMATURE (ambiguous aborts) ---
    if is_aborted and rd == "ambiguous" and fa_is_fa:
        return "premature_port_entry", None
    if is_aborted and rd == "ambiguous" and fa_is_nfa:
        return "premature_abort", None

    return "uncategorized", None


def _empty_bucket():
    return {"n": 0, "global_trial_ids": []}


def _empty_structure():
    out = {}
    for cat, subs in CATEGORY_SUBCATEGORIES.items():
        out[cat] = {"n": 0, "global_trial_ids": [], "subcategories": {s: _empty_bucket() for s in subs}}
    for cat in PREMATURE_CATEGORIES:
        out[cat] = _empty_bucket()
    out["uncategorized"] = _empty_bucket()
    out["total_trials"] = 0
    out["validation"] = {
        "n_total_trials": 0,
        "n_classified": 0,
        "not_in_any_category": {"n": 0, "global_trial_ids": []},
        "in_multiple_categories": {"n": 0, "global_trial_ids": {}},
        "n_trials_missing_global_trial_id": 0,
    }
    return out


def _stage_names(results):
    """Collect all stage_name strings from the session summary/manifest runs."""
    names = []
    for key in ("summary", "manifest"):
        container = results.get(key)
        if not isinstance(container, dict):
            continue
        runs = (container.get("session", {}) or {}).get("runs", []) or []
        for run in runs:
            if isinstance(run, dict):
                stage = run.get("stage", {}) or {}
                name = stage.get("stage_name")
                if name:
                    names.append(str(name))
    return names


def is_singrew_session(results) -> bool:
    """Whether the session uses the single-reward protocol.

    Prefers the reliable ``session.is_singrew`` flag written to manifest/summary by
    the trial classifier (derived from the schema's sequence reward info). Falls back
    to a stage_name "singrew" substring match for older sessions saved before the flag
    existed.
    """
    for key in ("summary", "manifest"):
        container = results.get(key)
        if isinstance(container, dict):
            flag = (container.get("session", {}) or {}).get("is_singrew")
            if flag is not None:
                return bool(flag)
    return any("singrew" in name.lower() for name in _stage_names(results))


def compute_sing_rew_metrics(results) -> dict:
    """Sort all trials into outcome (sub)categories with per-(sub)category trial counts
    and ``global_trial_id`` lists, plus a ``validation`` block (see module docstring)."""
    out = _empty_structure()
    df = results.get("trial_data")
    if df is None or len(df) == 0:
        return out

    out["total_trials"] = int(len(df))
    has_gid = "global_trial_id" in df.columns

    # Track each gid's set of top-level categories to flag not-in-any / in-multiple.
    gid_to_categories = defaultdict(set)
    n_missing_gid = 0

    for _, row in df.iterrows():
        cat, sub = _classify_trial(row)
        gid = row.get("global_trial_id") if has_gid else None
        try:
            gid = int(gid) if gid is not None and not pd.isna(gid) else None
        except (TypeError, ValueError):
            gid = None
        if gid is None:
            n_missing_gid += 1

        if cat == "uncategorized":
            out["uncategorized"]["n"] += 1
            if gid is not None:
                out["uncategorized"]["global_trial_ids"].append(gid)
                gid_to_categories[gid].add("uncategorized")
            continue

        out[cat]["n"] += 1
        if gid is not None:
            out[cat]["global_trial_ids"].append(gid)
            gid_to_categories[gid].add(cat)
        if sub is not None:
            bucket = out[cat]["subcategories"][sub]
            bucket["n"] += 1
            if gid is not None:
                bucket["global_trial_ids"].append(gid)

    # --- Validation: flag gids in no real category, or in more than one ---
    not_in_any = sorted(out["uncategorized"]["global_trial_ids"])
    in_multiple = {
        gid: sorted(cats) for gid, cats in gid_to_categories.items() if len(cats) > 1
    }
    n_classified = sum(out[cat]["n"] for cat in MAIN_CATEGORIES)
    out["validation"] = {
        "n_total_trials": out["total_trials"],
        "n_classified": n_classified,
        "not_in_any_category": {"n": len(not_in_any), "global_trial_ids": not_in_any},
        "in_multiple_categories": {
            "n": len(in_multiple),
            "global_trial_ids": {str(k): v for k, v in sorted(in_multiple.items())},
        },
        "n_trials_missing_global_trial_id": n_missing_gid,
    }
    return out


# =========================================================================
# Signal-detection rates / metrics derived from the sorted categories
# =========================================================================
#
# Built from the category counts produced by ``compute_sing_rew_metrics``.
#
# Counts
#   n_go    (reward-determined)      = hit + miss
#   n_nogo  (nonrewarded-determined) = false_alarm + correct_rejection
#   n_amb   (ambiguous)              = premature_port_entry + premature_abort
#   n_det                            = n_go + n_nogo
#   n_tot                            = n_det + n_amb
#
# On determined trials:
#   hit_rate            h   = hit / n_go
#   fa_rate             f   = fa  / n_nogo
#   H', F' (log-linear)     = (hit+0.5)/(n_go+1), (fa+0.5)/(n_nogo+1)
#   headline_sensitivity d' = Phi^-1(H') - Phi^-1(F')
#   criterion            c  = -0.5 * (Phi^-1(H') + Phi^-1(F'))
#   balanced_accuracy       = 0.5 * (H + (1 - F))
#   earned_reward_rate      = rewarded_hit / n_go
#   port_accuracy           = rewarded_hit / (rewarded_hit + port_error_hit)
#   efficient_rejection_rate= active_cr / n_nogo
#   early_rejection_index   = active_cr / CR
#   anticipatory_rate       = (anticipatory_hit + anticipatory_port_error) / n_go
#   forfeit_rate            = forfeit_miss / n_go
#   omission_rate           = omission_miss / n_go
#   impulsivity_rate        = premature_port_entry / n_tot
#   impatience_rate         = premature_abort / n_tot
#
# Every empty denominator yields NaN (never 0). Note the log-linear correction
# keeps d'/criterion finite when hit/fa is 0 or maxed out, but it does NOT cover
# an empty go/no-go side: (hit+0.5)/(n_go+1) is a valid 0.5 even when n_go==0, so
# H'/F' (and thus d'/criterion) are guarded to NaN on n_go==0 / n_nogo==0
# separately. H' and F' are persisted alongside d'/criterion.

NAN = float("nan")


def _ratio(num, den):
    """num/den, or NaN when the denominator is 0 (never a misleading 0)."""
    return (num / den) if den else NAN


def _phi_inv(p):
    """Inverse standard-normal CDF (Phi^-1); NaN for NaN or out-of-(0,1) input."""
    if p is None:
        return NAN
    try:
        if math.isnan(p):
            return NAN
    except (TypeError, ValueError):
        return NAN
    if not (0.0 < p < 1.0):
        return NAN
    return NormalDist().inv_cdf(p)


def compute_sing_rew_rates(categories: dict) -> dict:
    """Signal-detection rates/metrics from a ``compute_sing_rew_metrics`` result.

    Returns a flat dict of the counts (under ``counts``) and every scalar metric.
    Empty denominators are NaN. See the section comment above for definitions.
    """
    def cat_n(cat):
        return int(categories.get(cat, {}).get("n", 0))

    def sub_n(cat, sub):
        return int(categories.get(cat, {}).get("subcategories", {}).get(sub, {}).get("n", 0))

    hit = cat_n("hit")
    miss = cat_n("miss")
    fa = cat_n("false_alarm")
    cr = cat_n("correct_rejection")
    ppe = cat_n("premature_port_entry")
    pa = cat_n("premature_abort")

    rewarded_hit = sub_n("hit", "rewarded_hit")
    port_error_hit = sub_n("hit", "port_error_hit")
    anticipatory_hit = sub_n("hit", "anticipatory_hit")
    anticipatory_port_error = sub_n("hit", "anticipatory_port_error")
    forfeit_miss = sub_n("miss", "forfeit_miss")
    omission_miss = sub_n("miss", "omission_miss")
    active_cr = sub_n("correct_rejection", "active_cr")

    n_go = hit + miss
    n_nogo = fa + cr
    n_amb = ppe + pa
    n_det = n_go + n_nogo
    n_tot = n_det + n_amb

    # Raw rates (NaN on empty side)
    hit_rate = _ratio(hit, n_go)
    fa_rate = _ratio(fa, n_nogo)

    # Log-linear corrected rates, guarded to NaN when the side is empty (the
    # correction alone would otherwise yield a meaningful 0.5 at n==0).
    h_prime = ((hit + 0.5) / (n_go + 1)) if n_go > 0 else NAN
    f_prime = ((fa + 0.5) / (n_nogo + 1)) if n_nogo > 0 else NAN
    z_h = _phi_inv(h_prime)
    z_f = _phi_inv(f_prime)
    headline_sensitivity = z_h - z_f          # NaN if either side empty
    criterion = -0.5 * (z_h + z_f)            # NaN if either side empty
    balanced_accuracy = 0.5 * (hit_rate + (1.0 - fa_rate))  # NaN if either rate NaN

    return {
        "counts": {
            "n_go": n_go,
            "n_nogo": n_nogo,
            "n_amb": n_amb,
            "n_det": n_det,
            "n_tot": n_tot,
            "hit": hit,
            "miss": miss,
            "false_alarm": fa,
            "correct_rejection": cr,
            "premature_port_entry": ppe,
            "premature_abort": pa,
            "rewarded_hit": rewarded_hit,
            "port_error_hit": port_error_hit,
            "anticipatory_hit": anticipatory_hit,
            "anticipatory_port_error": anticipatory_port_error,
            "forfeit_miss": forfeit_miss,
            "omission_miss": omission_miss,
            "active_cr": active_cr,
        },
        "hit_rate": hit_rate,
        "fa_rate": fa_rate,
        # Stored rather than derived plot-side: both are one line from counts already
        # in this dict, so keeping them here leaves no metric math in the plotter.
        "ambiguous_rate": _ratio(n_amb, n_tot),
        "correct_rejection_rate": _ratio(cr, n_nogo),
        "H_prime": h_prime,
        "F_prime": f_prime,
        "headline_sensitivity": headline_sensitivity,
        "criterion": criterion,
        "balanced_accuracy": balanced_accuracy,
        "earned_reward_rate": _ratio(rewarded_hit, n_go),
        "port_accuracy": _ratio(rewarded_hit, rewarded_hit + port_error_hit),
        "efficient_rejection_rate": _ratio(active_cr, n_nogo),
        "early_rejection_index": _ratio(active_cr, cr),
        "anticipatory_rate": _ratio(anticipatory_hit + anticipatory_port_error, n_go),
        "forfeit_rate": _ratio(forfeit_miss, n_go),
        "omission_rate": _ratio(omission_miss, n_go),
        "impulsivity_rate": _ratio(ppe, n_tot),
        "impatience_rate": _ratio(pa, n_tot),
    }
