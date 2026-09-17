"""Per-animal trial and failed-attempt frames for the A/B learning analysis.

Reads each session's saved ``trial_data`` and ``non_initiated_attempts`` and keeps the
**odour-discrimination runs only**, stage 2 onwards: the stage is read per run from
``summary.json``, because a session can mix stages (sub-066 ses-010: run 1 odour
discrimination, run 2 doubles) while its ``protocol_mode`` names one.

Two frames, keyed alike (``subjid`` / ``ses`` / ``date`` / ``session_idx`` / ``run_id`` /
``initiation_sequence_time``):

- ``trials``   -- one row per trial: the valve opening the rig's AwaitReward followed.
- ``attempts`` -- one row per failed initiation attempt, joined to the trial its
  initiation went on to produce.

Within an initiation the odor does not change, and every attempt carries its initiation;
a row breaking either is kept and reported with a warning.
"""
from __future__ import annotations

import json
import re
import warnings

import numpy as np
import pandas as pd

from hypnose_behavior.io import layout
from hypnose_behavior.io.layout import derivatives, subject_selections
from hypnose_behavior.io.load_results import load_non_initiated_attempts
from hypnose_behavior.io.loaders import _odor_to_letter, iter_sessions

__all__ = [
    "PORT_LETTERS",
    "PORT_VISIT_LABELS",
    "is_ab_stage",
    "load_ab_data",
]

# `odourdiscrimination-stageN`, also inside a schema path. Stage 1 presents no odor (light
# and reward-port pokes only) and is not part of the analysis.
_STAGE_RE = re.compile(r"odourdiscrimination\W*stage(\d+)", re.IGNORECASE)
_EXCLUDED_STAGES = {1}

# A failed attempt counts as a port visit when the visit fell inside the response window
# or up to its late multiple. `FA_late` is a visit long after the attempt, not a response
# to it; `nFA` is no visit.
PORT_VISIT_LABELS = ("FA_time_in", "FA_time_out")

# Reward port -> the odor it pays out for.
PORT_LETTERS = {1: "A", 2: "B"}

_KEY = ["run_id", "initiation_sequence_time"]
_IDENTITY = ["subjid", "ses", "date", "session_idx"]
_TRIAL_COLUMNS = _IDENTITY + [
    "run_id", "global_trial_id", "initiation_sequence_time", "sequence_start", "odor",
    "choice", "outcome", "correct", "is_aborted", "attempt_number", "poke_ms",
    "short_sampling", "n_failed",
]
_ATTEMPT_COLUMNS = _IDENTITY + [
    "run_id", "initiation_sequence_time", "attempt_start", "attempt_number", "odor",
    "poke_ms", "zero_poke", "met_min_sampling", "failure_reason", "fa_label", "port_visit",
    "port", "correct_port", "global_trial_id", "trailing",
]
_SESSION_COLUMNS = _IDENTITY + ["n_runs", "n_trials", "n_attempts"]


def is_ab_stage(stage_name) -> bool:
    """Whether a run's stage name is an odour-discrimination stage the analysis uses."""
    match = _STAGE_RE.search(str(stage_name or ""))
    return bool(match) and int(match.group(1)) not in _EXCLUDED_STAGES


def _letters(series: pd.Series) -> pd.Series:
    """Stored odor names as bare letters (``"OdorA"`` -> ``"A"``)."""
    return series.map(lambda v: "" if pd.isna(v) else _odor_to_letter(v))


def _as_ns(series: pd.Series) -> pd.Series:
    """Timestamps at one resolution, so the two tables join on equality."""
    return pd.to_datetime(series, errors="coerce").astype("datetime64[ns]")


def _ab_runs(results_dir) -> set:
    """The run ids whose stage (from ``summary.json``) passes `is_ab_stage`."""
    path = layout.table_path(results_dir, "summary.json")
    try:
        with open(path, encoding="utf-8") as f:
            runs = json.load(f).get("session", {}).get("runs", [])
    except (OSError, ValueError):
        return set()
    kept = set()
    for run in runs:
        stage = run.get("stage")
        name = stage.get("stage_name") if isinstance(stage, dict) else stage
        if is_ab_stage(name):
            kept.add(run.get("run_id"))
    return kept


def _trial_frame(td: pd.DataFrame) -> pd.DataFrame:
    """The analysis columns of one session's trials."""
    outcome = td["response_time_category"].astype(object).where(
        td["response_time_category"].notna(), "").astype(str)
    correct = pd.Series(np.nan, index=td.index, dtype=object)
    correct[outcome == "rewarded"] = True
    correct[outcome == "unrewarded"] = False

    # The port the animal chose: the reward it collected, or the wrong port it poked.
    choice = pd.Series("", index=td.index, dtype=object)
    for column, when in (("first_supply_odor_identity", "rewarded"),
                         ("first_reward_poke_odor_identity", "unrewarded")):
        if column in td.columns:
            picked = td[column].astype(str)
            choice = choice.mask((outcome == when) & picked.isin(PORT_LETTERS.values()), picked)

    reason = (td["fallback_reason"] if "fallback_reason" in td.columns
              else pd.Series(None, index=td.index, dtype=object))
    return pd.DataFrame({
        "run_id": td["run_id"],
        "global_trial_id": td["global_trial_id"],
        "initiation_sequence_time": _as_ns(td["initiation_sequence_time"]),
        "sequence_start": _as_ns(td["sequence_start"]),
        "odor": _letters(td["odor_name"]),
        "choice": choice,
        "outcome": outcome,
        "correct": correct,
        "is_aborted": td["is_aborted"].fillna(False).astype(bool),
        "attempt_number": pd.to_numeric(td["attempt_number"], errors="coerce"),
        "poke_ms": pd.to_numeric(td["continuous_poke_time_ms"], errors="coerce"),
        # The rig initiated although the sampling time stayed below the minimum.
        "short_sampling": (reason == "await_reward_event").to_numpy(),
    })


def _attempt_frame(ni: pd.DataFrame) -> pd.DataFrame:
    """The analysis columns of one session's failed attempts."""
    odor = _letters(ni["odor_name"])
    poke_ms = pd.to_numeric(ni["continuous_poke_time_ms"], errors="coerce")
    fa_label = ni["fa_label"].astype(str)
    port_visit = fa_label.isin(PORT_VISIT_LABELS)
    port = pd.to_numeric(ni["fa_port"], errors="coerce").map(PORT_LETTERS)
    port = port.where(port_visit).fillna("")
    correct_port = pd.Series(np.nan, index=ni.index, dtype=object)
    correct_port[port_visit] = (port == odor)[port_visit]
    met_min = (ni["met_min_sampling"] if "met_min_sampling" in ni.columns
               else pd.Series(np.nan, index=ni.index))
    return pd.DataFrame({
        "run_id": ni["run_id"],
        "initiation_sequence_time": _as_ns(ni["initiation_sequence_time"]),
        "attempt_start": _as_ns(ni["attempt_start"]),
        "attempt_number": pd.to_numeric(ni["attempt_number"], errors="coerce"),
        "odor": odor,
        "poke_ms": poke_ms,
        "zero_poke": (poke_ms == 0).to_numpy(),
        "met_min_sampling": met_min,
        "failure_reason": ni["failure_reason"],
        "fa_label": fa_label,
        "port_visit": port_visit.to_numpy(),
        "port": port,
        "correct_port": correct_port,
    })


def _join(trials: pd.DataFrame, attempts: pd.DataFrame, label: str):
    """Attach each failed attempt to its initiation's trial, and count them per trial."""
    by_initiation = trials.drop_duplicates(_KEY).set_index(_KEY)
    tid = pd.Series(np.nan, index=attempts.index, dtype=float)
    trial_odor = pd.Series(np.nan, index=attempts.index, dtype=object)
    if len(by_initiation):
        joined = attempts.join(by_initiation[["global_trial_id", "odor"]], on=_KEY,
                               rsuffix="_trial")
        tid, trial_odor = joined["global_trial_id"], joined["odor_trial"]
    attempts = attempts.assign(global_trial_id=tid, trailing=tid.isna().to_numpy())

    unkeyed = attempts["initiation_sequence_time"].isna() | attempts["attempt_number"].isna()
    if unkeyed.any():
        warnings.warn(f"{label}: {int(unkeyed.sum())} failed attempt(s) have no initiation or "
                      f"attempt number, so they cannot be joined to a trial.",
                      RuntimeWarning, stacklevel=3)
    mixed = trial_odor.notna() & (trial_odor != attempts["odor"])
    if mixed.any():
        warnings.warn(f"{label}: {int(mixed.sum())} failed attempt(s) carry a different "
                      f"odor from the trial their initiation produced.", RuntimeWarning,
                      stacklevel=3)

    counts = attempts.groupby(_KEY).size().rename("n_failed")
    trials = trials.join(counts, on=_KEY) if len(trials) else trials.assign(n_failed=0)
    trials["n_failed"] = trials["n_failed"].fillna(0).astype(int)
    return trials, attempts


def load_ab_data(subjids, dates=None, *, ses=None, index=None, date_range=None,
                 ses_range=None, index_range=None, verbose: bool = True) -> dict:
    """The trial and failed-attempt frames of the odour-discrimination runs.

        load_ab_data({63: {"ses_range": (1, 10)}, 64: {"ses_range": (3, 11)}})
        load_ab_data([63, 64, 65, 66], ses_range=(1, 14))

    ``subjids`` / ``dates`` / the selectors take every form of
    ``io.layout.subject_selections``. Runs of other stages are dropped whatever the
    selection, and a session with neither a trial nor a failed attempt left is skipped.

    Returns ``{"trials", "attempts", "sessions"}``:

    - ``trials``: ``odor`` / ``choice`` (``"A"``/``"B"``, ``""`` when no choice),
      ``outcome`` (``response_time_category``), ``correct`` (True / False, NaN unless
      rewarded or unrewarded), ``attempt_number``, ``poke_ms``, ``short_sampling`` (the rig
      initiated below the minimum sampling time) and ``n_failed`` (failed attempts in its
      initiation).
    - ``attempts``: ``odor``, ``poke_ms``, ``zero_poke``, ``met_min_sampling`` (reached the
      minimum without the rig initiating), ``failure_reason``, ``fa_label``,
      ``port_visit`` (``fa_label`` in ``PORT_VISIT_LABELS``), ``port`` (``"A"``/``"B"``
      when visited), ``correct_port`` (NaN without a visit), ``global_trial_id`` of the
      trial the initiation produced and ``trailing`` (it produced none).
    - ``sessions``: one row per kept session, with the kept ``n_runs`` and the row counts.

    ``session_idx`` is the 0-based rank of a session among the subject's kept ones.
    """
    entries = subject_selections(subjids, dates, ses=ses, index=index,
                                 date_range=date_range, ses_range=ses_range,
                                 index_range=index_range)
    trial_parts, attempt_parts, session_rows = [], [], []
    for subjid, subj_dates, select in entries:
        subj_dir = derivatives.subject_dir(subjid, missing_ok=True)
        if subj_dir is None:
            print(f"Warning: no derivatives directory for subject {subjid}, skipping")
            continue
        session_idx = 0
        for rec in iter_sessions(subj_dir, subj_dates, **select):
            if not rec.analysed:
                continue
            label = f"sub-{subjid} ses-{rec.ref.ses}"
            runs = _ab_runs(rec.results_dir)
            td = rec.views["trial_data"]
            td = td[td["run_id"].isin(runs)] if len(td.columns) else td
            ni = load_non_initiated_attempts(rec.results_dir)
            ni = ni[ni["run_id"].isin(runs)] if len(ni.columns) else ni
            if td.empty and ni.empty:
                if verbose:
                    print(f"[load_ab_data] {label}: no odour-discrimination trial or "
                          f"attempt, skipped")
                continue

            trials = (_trial_frame(td.sort_values(["run_id", "global_trial_id"])) if len(td)
                      else pd.DataFrame(columns=_TRIAL_COLUMNS[len(_IDENTITY):-1]))
            identity = {"subjid": int(subjid), "ses": rec.ref.ses,
                        "date": rec.date_str, "session_idx": session_idx}
            if len(ni):
                trials, attempts = _join(trials, _attempt_frame(ni), label)
                attempt_parts.append(attempts.assign(**identity))
            else:
                trials = trials.assign(n_failed=0)
            if len(trials):
                trial_parts.append(trials.assign(**identity))
            session_rows.append({**identity, "n_runs": len(runs), "n_trials": len(trials),
                                 "n_attempts": len(ni)})
            session_idx += 1

    trials = (pd.concat(trial_parts, ignore_index=True) if trial_parts
              else pd.DataFrame(columns=_TRIAL_COLUMNS))
    attempts = (pd.concat(attempt_parts, ignore_index=True) if attempt_parts
                else pd.DataFrame(columns=_ATTEMPT_COLUMNS))
    return {
        "trials": trials[_TRIAL_COLUMNS],
        "attempts": attempts[_ATTEMPT_COLUMNS],
        "sessions": pd.DataFrame(session_rows, columns=_SESSION_COLUMNS),
    }
