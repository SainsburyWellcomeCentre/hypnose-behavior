# Defers evaluation of PEP-604 annotations (`X | None`), keeping this module
# importable on Python 3.9 for repos pinned there (hypnose-eeg-preprocessing).
from __future__ import annotations

"""The hidden rule: did the animal detect it, and did detecting it pay.

``hidden_rule_mask`` is the grouping *key* for the HR / non-HR split, not a
metric: ``by_group(decision_accuracy, trials, hidden_rule_mask(trials))`` is the
audit's checklist 7. It is deliberately **not** ``hidden_rule_performance``,
which has a different numerator *and* denominator.

``hr_odor_associations`` is session **metadata** rather than a plotting concern
(judgement call 1 of the audit): ``visualization/`` only used it to pick a
colour, but which reward an animal's hidden-rule odor pays out is an analysis
result.

**One truthiness rule**, as of Phase 4b: ``_truthy`` / ``_is_truthy``, used by
``hidden_rule_mask``, ``hidden_rule_counts_by_odor`` and
``hr_odor_associations`` alike. The last of those arrived from
``visualization/`` in 4a with its own inline test; see ``common._is_truthy``
for which way that was resolved and why.
"""

import ast
import json
from collections import defaultdict

import numpy as np
import pandas as pd

from hypnose_behavior.io.layout import list_sessions
from hypnose_behavior.io.loaders import _load_trial_views, _odor_to_letter
from hypnose_behavior.frames import parse_json_column
from hypnose_behavior.metric_analysis.metrics.common import (
    _aborted_mask,
    _flag,
    _is_truthy,
    _position_rows,
    _reduce_rate,
    _truthy,
    _tz_naive,
)
from hypnose_behavior.metric_analysis.metrics.sequence import presentation_counts_by_odor
from hypnose_behavior.metric_analysis.registry import metric, session_metric

__all__ = [
    "hidden_rule_performance_contributions", "hidden_rule_performance",
    "hidden_rule_performance_session",
    "hidden_rule_detection_rate_contributions", "hidden_rule_detection_rate",
    "hidden_rule_detection_rate_session",
    "hidden_rule_mask",
    "hidden_rule_counts_by_odor", "hidden_rule_counts_by_odor_session",
    "hr_odor_associations",
    "hr_abort_poke_gap",
    "rolling_hr_reward_fraction",
]


def hidden_rule_performance_contributions(trials):
    return (((_truthy(trials, "hidden_rule_success")
              & _flag(trials, "response_time_category", "rewarded")).astype(int)),
            _truthy(trials, "hit_hidden_rule").astype(int))


@metric(frame="trials", title="Hidden Rule Performance")
def hidden_rule_performance(trials):
    """(HR success & rewarded) / hit_hidden_rule."""
    if trials.empty:
        return 0, 0, np.nan
    return _reduce_rate(*hidden_rule_performance_contributions(trials))


@session_metric(hidden_rule_performance)
def hidden_rule_performance_session(results):
    df = results.get("trial_data", pd.DataFrame())
    if df.empty:
        print("Hidden Rule Performance: no trial_data")
        return 0, 0, np.nan
    n_hr_rewarded, denom, rate = hidden_rule_performance(df)
    print(f"Hidden Rule Performance: {n_hr_rewarded}/{denom} = {rate:.3f}")
    return n_hr_rewarded, denom, rate


def hidden_rule_detection_rate_contributions(trials):
    return ((((~_aborted_mask(trials)) & _truthy(trials, "hidden_rule_success")).astype(int)),
            _truthy(trials, "hit_hidden_rule").astype(int))


@metric(frame="trials", title="Hidden Rule Detection Rate")
def hidden_rule_detection_rate(trials):
    """(not aborted & HR success) / hit_hidden_rule."""
    if trials.empty:
        return 0, 0, np.nan
    return _reduce_rate(*hidden_rule_detection_rate_contributions(trials))


@session_metric(hidden_rule_detection_rate)
def hidden_rule_detection_rate_session(results):
    df = results.get("trial_data", pd.DataFrame())
    if df.empty:
        print("Hidden Rule Detection Rate: no trial_data")
        return 0, 0, np.nan
    n_hr_completed, denom, rate = hidden_rule_detection_rate(df)
    print(f"Hidden Rule Detection Rate: {n_hr_completed}/{denom} = {rate:.3f}")
    return n_hr_completed, denom, rate


@metric(frame="trials")
def hidden_rule_mask(trials):
    """Boolean mask of hidden-rule trials -- the grouping key for the HR split.

    `by_group(decision_accuracy, trials, hidden_rule_mask(trials))` is the
    audit's checklist 7 (decision accuracy, HR vs non-HR): a granularity of
    `decision_accuracy`, not a metric of its own. It is deliberately **not**
    `hidden_rule_performance`, which has a different numerator *and* denominator.
    """
    return _truthy(trials, "hidden_rule_success")


def _extract_hr_config(results):
    """Return (hr_odors, hr_positions) from session metadata or results dict if available."""
    # Prefer values already attached to results by classification
    hr_odors = results.get("hidden_rule_odors") or []
    if isinstance(hr_odors, str):
        hr_odors = [hr_odors]

    hr_positions = results.get("hidden_rule_positions") or []
    if isinstance(hr_positions, (int, float)):
        hr_positions = [hr_positions]

    manifest = results.get("manifest", {}) or {}
    manifest_params = manifest.get("params", {}) if isinstance(manifest, dict) else {}
    manifest_session = manifest.get("session", {}) if isinstance(manifest, dict) else {}

    # Fallback to summary params
    summary = results.get("summary", {}) or {}
    params = summary.get("params", {}) if isinstance(summary, dict) else {}
    if not hr_odors:
        hr_odors = (
            params.get("hidden_rule_odors")
            or params.get("hiddenrule_odors")
            or manifest_params.get("hidden_rule_odors")
            or manifest_params.get("hiddenrule_odors")
            or manifest_session.get("hidden_rule_odors")
            or manifest.get("hidden_rule_odors")
            or []
        )
        if isinstance(hr_odors, str):
            hr_odors = [hr_odors]
    hr_odors = [str(o) for o in hr_odors if o]

    if not hr_positions:
        hr_positions = (
            params.get("hidden_rule_positions")
            or params.get("hiddenrule_positions")
            or manifest_params.get("hidden_rule_positions")
            or manifest_params.get("hiddenrule_positions")
            or manifest_session.get("hidden_rule_positions")
            or manifest.get("hidden_rule_positions")
            or []
        )
        if isinstance(hr_positions, (int, float)):
            hr_positions = [hr_positions]

    hr_pos_clean = []
    hr_iter = hr_positions if isinstance(hr_positions, (list, tuple)) else []
    for pos in hr_iter:
        try:
            hr_pos_clean.append(int(pos))
        except Exception:
            continue
    return hr_odors, hr_pos_clean


def _infer_hr_odors_from_row(row, hr_odors, hr_positions):
    """Best-effort identification of HR odor(s) for a trial row. Returns list of candidates."""

    def _parse_seq(val):
        seq = parse_json_column(val)
        if isinstance(seq, (list, tuple)):
            return list(seq)
        if isinstance(seq, str):
            try:
                return list(ast.literal_eval(seq)) if seq.strip() else []
            except Exception:
                return [seq]
        return []

    seq_fields = ["odor_sequence", "odor_sequence_full", "odor_sequence_list"]
    seq = []
    for key in seq_fields:
        if key in row:
            seq = _parse_seq(row.get(key))
            if seq:
                break

    # Per-row hidden rule positions, if present
    hr_pos_row = _parse_seq(row.get("hidden_rule_positions")) if "hidden_rule_positions" in row else []
    hr_pos_row_int = []
    for p in hr_pos_row if isinstance(hr_pos_row, (list, tuple)) else []:
        try:
            hr_pos_row_int.append(int(p))
        except Exception:
            continue

    positions_to_use = hr_pos_row_int or hr_positions

    found = []

    # Try using positions to pick odor from sequence
    if seq and positions_to_use:
        for pos in positions_to_use:
            idx = pos - 1
            if 0 <= idx < len(seq):
                candidate = seq[idx]
                if candidate is not None:
                    found.append(candidate)

    # If we have HR odor list, look for unique match in sequence
    if not found and seq and hr_odors:
        matches = [o for o in seq if o in hr_odors]
        if matches:
            found.extend(matches)

    # Hidden-rule-specific columns
    for key in ["hidden_rule_odor", "hidden_rule_odors"]:
        if key in row:
            vals = _parse_seq(row.get(key))
            if vals:
                found.extend(vals)

    # Fallback: last odor name
    for key in ["last_odor_name", "last_odor"]:
        if key in row:
            val = row.get(key)
            if val:
                found.append(val)

    # Normalize and deduplicate while preserving order
    out = []
    seen = set()
    for od in found:
        if od is None:
            continue
        s = str(od)
        if s not in seen:
            seen.add(s)
            out.append(s)

    return out or ["Unknown"]


def _fmt_rate(val):
    return f"{val:.3f}" if isinstance(val, (int, float, np.floating)) and not np.isnan(val) else "nan"


@metric(frame="trials+position_data", key="hidden_rule_by_odor",
        title="Hidden Rule Performance/Detection by Odor")
def hidden_rule_counts_by_odor(trials, position_data, hr_odors, hr_positions):
    """
    Aggregate HR trials by odor across outcome categories to support per-odor performance/detection.
    Returns a dict with hr_odors, hr_positions, and per-odor counts plus rates.

    `hr_odors` / `hr_positions` are session *metadata*, not trial data, so the
    core takes them as arguments and `_extract_hr_config` stays in the wrapper.
    """
    df = trials
    if df.empty:
        return {"hr_odors": [], "hr_positions": [], "by_odor": {}}

    hr_set = set(hr_odors)
    counts = defaultdict(lambda: defaultdict(int))

    # Pre-seed known HR odors to ensure they appear even if zero counts
    for od in hr_odors:
        _ = counts[od]

    seen_odors = set(hr_odors)

    def _add_counts(mask: pd.Series, label: str):
        subset = df[mask] if isinstance(mask, pd.Series) else pd.DataFrame()
        if subset.empty:
            return
        for _, row in subset.iterrows():
            odors = _infer_hr_odors_from_row(row, hr_odors, hr_positions)
            for od in odors:
                if od not in hr_set:
                    continue
                seen_odors.add(od)
                counts[od][label] += 1

    aborted_mask = df["is_aborted"] == True if "is_aborted" in df.columns else pd.Series(False, index=df.index)
    success_mask = df["hidden_rule_success"].apply(_is_truthy) if "hidden_rule_success" in df.columns else pd.Series(False, index=df.index)
    hit_mask = df["hit_hidden_rule"].apply(_is_truthy) if "hit_hidden_rule" in df.columns else pd.Series(False, index=df.index)

    # Completed HR trials by outcome (only count HR successes)
    if "response_time_category" in df.columns:
        _add_counts((df["response_time_category"] == "rewarded") & success_mask, "rewarded")
        _add_counts((df["response_time_category"] == "unrewarded") & success_mask, "unrewarded")
        _add_counts((df["response_time_category"] == "timeout_delayed") & success_mask, "timeout")

    # Aborted HR trials (any aborted hit)
    _add_counts(aborted_mask & hit_mask, "aborted")

    # Missed HR trials: not aborted and not successful
    _add_counts((~aborted_mask) & (~success_mask), "missed")

    # Total presentations per odor -- the same count `odorx_abortion_rate` uses,
    # restricted to the hidden-rule odors.
    presentations = {od: n for od, n in presentation_counts_by_odor(position_data).items()
                     if od in hr_set}

    by_odor = {}
    for odor in sorted(seen_odors):
        c = counts.get(odor, {})
        rewarded = c.get("rewarded", 0)
        unrewarded = c.get("unrewarded", 0)
        timeout = c.get("timeout", 0)
        missed = c.get("missed", 0)
        aborted = c.get("aborted", 0)

        total_presentations = presentations.get(odor, 0)
        completed_no_timeout = rewarded + unrewarded
        completed_with_timeout = completed_no_timeout + timeout

        performance = rewarded / completed_no_timeout if completed_no_timeout > 0 else np.nan
        detection_rate = completed_no_timeout / total_presentations if total_presentations > 0 else np.nan

        by_odor[odor] = {
            "rewarded": int(rewarded),
            "unrewarded": int(unrewarded),
            "timeout": int(timeout),
            "missed": int(missed),
            "aborted": int(aborted),
            "total_presentations": int(total_presentations),
            "completed_total": int(completed_with_timeout),
            "completed_no_timeout": int(completed_no_timeout),
            "performance": performance,
            "performance_fraction": [int(rewarded), int(completed_no_timeout)],
            "detection_rate": detection_rate,
            "detection_fraction": [int(completed_no_timeout), int(total_presentations)],
        }

    return {
        "hr_odors": sorted(seen_odors),
        "hr_positions": hr_positions,
        "by_odor": by_odor,
    }


@session_metric(hidden_rule_counts_by_odor)
def hidden_rule_counts_by_odor_session(results):
    trials = results.get("trial_data", pd.DataFrame())
    if trials.empty:
        print("Hidden Rule Counts by Odor: no trial_data")
        return {"hr_odors": [], "hr_positions": [], "by_odor": {}}
    hr_odors, hr_positions = _extract_hr_config(results)
    out = hidden_rule_counts_by_odor(trials, results.get("position_data"),
                                     hr_odors, hr_positions)
    for odor in out["hr_odors"]:
        c = out["by_odor"][odor]
        print(
            f"Hidden Rule Odor {odor}: {c['rewarded']} Rewarded, {c['unrewarded']} Unrewarded, "
            f"{c['timeout']} Timeout, {c['total_presentations']} Total Presentations."
        )
        print(
            f"  HR Odor {odor} Performance: {c['rewarded']}/{c['completed_no_timeout']} = "
            f"{_fmt_rate(c['performance'])}, "
            f"HR Odor {odor} Detection Rate: {c['completed_no_timeout']}/{c['total_presentations']} = "
            f"{_fmt_rate(c['detection_rate'])}"
        )
    return out


def hr_odor_associations(subj_dirs) -> dict:
    """Learn which reward ('A' or 'B') each hidden-rule odor maps to.

    Scans hidden-rule sessions for the given subject directories and, for every
    hidden-rule *success* trial, reads which HR odor fired (odor_sequence at the
    success position) and the reward identity it produced
    (``first_supply_odor_identity``). Votes are accumulated per odor; since the
    association is conserved for an animal, we stop scanning a subject once all
    of its HR odors are resolved.

    Returns ``{odor_letter: 'A' | 'B'}`` (empty if no HR sessions found).

    Judgement call 1 of the metric audit: it is **session metadata**, not a
    plotting concern -- `visualization/` only used it to pick a colour, but the
    fact it establishes (which reward an animal's hidden-rule odor pays out) is
    an analysis result. Moved here verbatim.

    Its truthiness test used to be an inline ``isin(["true", "1", "1.0"])``,
    which accepted a string `_is_truthy` did not. Phase 4b reconciled the two on
    `_is_truthy`'s side (see it), so this now uses the package's one rule.
    """
    votes: dict = defaultdict(lambda: {"A": 0, "B": 0})
    for subj_dir in subj_dirs:
        if subj_dir is None:
            continue
        for session in list_sessions(subj_dir):
            results_dir = session.path / "saved_analysis_results"
            summary_path = results_dir / "summary.json"
            if not summary_path.exists():
                continue
            try:
                with open(summary_path, "r", encoding="utf-8") as f:
                    hr_raw = json.load(f).get("params", {}).get("hidden_rule_odors", []) or []
            except Exception:
                hr_raw = []
            # A/B are decision/reward odors, not genuine hidden-rule odors; some
            # probe sessions list them as hidden_rule_odors, so ignore them here.
            hr_letters = {_odor_to_letter(o) for o in hr_raw if o} - {"A", "B"}
            if not hr_letters:
                continue
            td = _load_trial_views(results_dir)["trial_data"]
            if td.empty or "hidden_rule_success" not in td.columns:
                continue
            mask = _truthy(td, "hidden_rule_success")
            for _, r in td[mask].iterrows():
                ident = r.get("first_supply_odor_identity")
                if ident not in ("A", "B"):
                    continue
                seq = parse_json_column(r.get("odor_sequence"))
                pos = r.get("hidden_rule_success_position")
                if not isinstance(seq, (list, tuple, np.ndarray)) or pos is None:
                    continue
                try:
                    if isinstance(pos, float) and np.isnan(pos):
                        continue
                    letter = _odor_to_letter(seq[int(pos) - 1])
                except Exception:
                    continue
                if letter in hr_letters:
                    votes[letter][ident] += 1
            # Association is conserved per animal; stop once all HR odors resolved.
            if all(sum(votes[l].values()) > 0 for l in hr_letters):
                break
    return {l: ("A" if v["A"] >= v["B"] else "B") for l, v in votes.items() if sum(v.values()) > 0}


def _first_hr_position(val):
    """First entry of `hidden_rule_hit_positions`, however it is stored."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    if isinstance(val, (int, float)):
        return int(val)
    if isinstance(val, (list, tuple, np.ndarray)) and len(val) > 0:
        try:
            return int(val[0])
        except Exception:
            return None
    if isinstance(val, str):
        parsed = parse_json_column(val)
        try:
            if isinstance(parsed, (list, tuple)) and parsed:
                return int(parsed[0])
            if isinstance(parsed, (int, float)):
                return int(parsed)
        except Exception:
            return None
    return None


@metric(frame="trials+position_data")
def hr_abort_poke_gap(trials, position_data):
    """Latency from the hidden-rule poke to the last poke of an aborted trial.

    Checklist 8: `last poke_odor_end - hidden-rule poke_odor_end`, on trials that
    aborted having hit the hidden rule, plus the start-to-end variant. No
    canonical metric measures any latency *between positions*.

    One row per qualifying trial; trials without a hidden-rule position or
    without usable poke timestamps are dropped rather than reported as NaN.
    """
    cols = ["global_trial_id", "hidden_rule_position",
            "delta_seconds", "delta_start_end_seconds"]
    if trials.empty or position_data is None or len(position_data) == 0:
        return pd.DataFrame(columns=cols)
    if "global_trial_id" not in position_data.columns:
        return pd.DataFrame(columns=cols)

    hr_trials = trials[_aborted_mask(trials) & _truthy(trials, "hit_hidden_rule")]
    if hr_trials.empty:
        return pd.DataFrame(columns=cols)

    poke = _position_rows(position_data, "in_poke_times")
    if poke is None or poke.empty:
        return pd.DataFrame(columns=cols)
    poke = poke.assign(_end=_tz_naive(poke["poke_odor_end"]),
                       _start=_tz_naive(poke["poke_odor_start"]))
    by_trial = {gid: sub for gid, sub in poke.groupby("global_trial_id")}

    rows = []
    for _, trial in hr_trials.iterrows():
        hr_pos = _first_hr_position(trial.get("hidden_rule_hit_positions"))
        if hr_pos is None:
            continue
        sub = by_trial.get(trial.get("global_trial_id"))
        if sub is None or sub.empty:
            continue
        ends = sub["_end"].dropna()
        if ends.empty:
            continue
        at_hr = sub[sub["position"] == hr_pos]
        hr_end = at_hr["_end"].dropna()
        if hr_end.empty:
            continue
        hr_start = at_hr["_start"].dropna()
        last_end = ends.max()
        rows.append({
            "global_trial_id": trial.get("global_trial_id"),
            "hidden_rule_position": hr_pos,
            "delta_seconds": (last_end - hr_end.iloc[-1]).total_seconds(),
            "delta_start_end_seconds": ((last_end - hr_start.iloc[-1]).total_seconds()
                                        if not hr_start.empty else np.nan),
        })
    return pd.DataFrame(rows, columns=cols)


@metric(frame="trials")
def rolling_hr_reward_fraction(trials, window, *, with_flags=False):
    """Rolling percentage of rewarded trials that were hidden-rule rewarded.

    Checklist 9. Related to `hidden_rule_performance` but not a granularity of
    it: the denominator is rewarded trials, not hidden-rule hits. Indexed by the
    rows of `trials` it kept, in `sequence_start` order.

    **Pass the pooled frame, not one session.** The window is meant to run across
    session boundaries; rolling per session and concatenating restarts it at each
    boundary, which is a different quantity and raises no error.

    `with_flags=True` also returns the per-trial hidden-rule indicator the
    percentage is rolled over, on the same index -- the plotter reports both, and
    re-deriving the flag on its side is the duplication this metric removes.
    """
    rewarded = trials[(~_aborted_mask(trials))
                      & _flag(trials, "response_time_category", "rewarded")]
    if rewarded.empty:
        empty = pd.Series(dtype=float)
        return (pd.Series(dtype=bool), empty) if with_flags else empty
    for col in ("hidden_rule_success", "hit_hidden_rule"):
        if col in rewarded.columns:
            hr = rewarded[col].fillna(False).astype(bool)
            break
    else:
        hr = pd.Series(False, index=rewarded.index)
    if "sequence_start" in rewarded.columns:
        hr = hr.loc[rewarded["sequence_start"].sort_values().index]
    pct = hr.astype(int).rolling(window, min_periods=1).mean() * 100.0
    return (hr, pct) if with_flags else pct
