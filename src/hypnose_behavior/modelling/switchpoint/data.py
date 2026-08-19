"""Build a continuous SHORT/LONG trial sequence for an animal from its saved trial data.

The data layer of the switch-point analysis: it reads the ``trial_data`` written by trial
classification (via ``hypnose_behavior.io``), filters and concatenates the kept trials of every session
in date order, and re-indexes them 0..n-1 so the trial axis is continuous across sessions. Pure
data assembly -- no models, no plotting. The model fits consume the ``s`` array it returns; the
plots consume the whole prep dict.

A trial is kept when ``is_aborted == False`` (and, with ``rewarded_only``, when
``response_time_category == "rewarded"``), and it scores 1 (SHORT) when ``hidden_rule_success``
is truthy, else 0 (LONG). ``trial_data``'s own ``global_trial_id`` restarts at 0 each session,
so it orders trials only *within* a session.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd

from hypnose_behavior.io.loaders import _load_trial_views, _odor_to_letter
from hypnose_behavior.utils.helpers import _filter_session_dirs, _iter_subject_dirs

# Truthy spellings of hidden_rule_success: bool in parquet, str via the CSV fallback.
# Mirrors the coercion the visualization helpers use.
_HR_TRUE = ("true", "1", "1.0")

# The two reward identities a trial can resolve to; anything else is unresolved (``""``).
# Colours for these live with the plots, not here -- this module only decides identity.
AB_LETTERS = ("A", "B")
_AB_UNKNOWN = ""

__all__ = [
    "AB_LETTERS",
    "normalize_subjids_dates",
    "prepare_subject",
    "subset_by_ab",
    "subject_label",
]


def normalize_subjids_dates(subjids, dates):
    """Normalize the ``(subjids, date_ranges)`` inputs, supporting a ``{subjid: date_range}``
    dict passed as ``subjids``.

    Reimplements ``hypnose_behavior.visualization.sing_rew._normalize_subjids_dates`` (that module
    currently fails to import; see the analysis script's README).
    """
    if isinstance(subjids, dict):
        dates = subjids if (dates is None or not isinstance(dates, dict)) else dates
        subjids = list(subjids.keys())
    elif isinstance(subjids, set):
        subjids = sorted(subjids)
    elif not isinstance(subjids, (list, tuple)):
        subjids = [subjids]

    def dates_for(subjid):
        if not isinstance(dates, dict):
            return dates
        if subjid in dates:
            return dates[subjid]
        try:
            if int(subjid) in dates:
                return dates[int(subjid)]
        except (TypeError, ValueError):
            pass
        return dates.get(str(subjid))

    return subjids, dates, dates_for


def _short_mask(td: pd.DataFrame) -> pd.Series:
    """Boolean mask of SHORT-sequence trials (``hidden_rule_success`` truthy)."""
    if "hidden_rule_success" not in td.columns:
        return pd.Series(False, index=td.index)
    return td["hidden_rule_success"].astype(str).str.lower().isin(_HR_TRUE)


def _ab_label(td: pd.DataFrame) -> pd.Series:
    """Reward identity of each trial: ``"A"``, ``"B"``, or ``""`` when unresolved.

    ``first_supply_odor_identity`` is the reward the trial actually delivered, so it is
    authoritative -- but it is null whenever nothing was supplied (every unrewarded/timeout
    trial, and a few rewarded ones), and a handful of early sessions lack the column.
    ``last_odor`` then resolves LONG trials, where the animal ran to the end of the sequence
    and the final odor *is* the reward odor. It cannot resolve SHORT trials, whose last odor
    is the hidden-rule odor rather than A or B, so those stay unresolved.
    """
    if "first_supply_odor_identity" in td.columns:
        supply = td["first_supply_odor_identity"].astype(str)
        letters = supply.where(supply.isin(AB_LETTERS), _AB_UNKNOWN)
    else:
        letters = pd.Series(_AB_UNKNOWN, index=td.index, dtype=object)
    if "last_odor" in td.columns:
        fallback = td["last_odor"].map(_odor_to_letter)
        letters = letters.mask(letters == _AB_UNKNOWN, fallback.where(fallback.isin(AB_LETTERS), _AB_UNKNOWN))
    return letters


def prepare_subject(
    subjid: int,
    date_range: Optional[Union[Sequence[Union[int, str]], tuple]] = None,
    rewarded_only: bool = False,
    derivatives_dir: Optional[Path] = None,
) -> dict:
    """Build one animal's continuous SHORT/LONG trial sequence and its sleep markers.

    Concatenates the kept trials of every session in ``date_range``, in date order, and
    re-indexes them 0..n-1 so the trial axis is continuous across sessions. Shared by both
    entry points -- all filtering lives here.

    Parameters
    ----------
    subjid : int
        Subject id.
    date_range : None | tuple[start, end] | iterable of dates
        Inclusive ``YYYYMMDD`` range or explicit date list. ``None`` = all sessions.
    rewarded_only : bool
        Additionally require ``response_time_category == "rewarded"``.
    derivatives_dir : Path | None
        Derivatives root; defaults to the resolved project root.

    Returns
    -------
    dict
        ``subjid``, ``trial_ids`` (0..n-1), ``global_ids`` (position on the full trial axis;
        equal to ``trial_ids`` here, and meaningful after ``subset_by_ab``), ``s``
        (1 = SHORT, 0 = LONG), ``ab`` (reward identity per trial), ``session_ends`` (global
        trial id of the LAST kept trial of each session -- the sleep markers),
        ``session_starts`` (the trial after each sleep period; always starts at 0),
        ``session_labels`` (dates), ``session_index`` (session of each trial),
        ``session_sizes``, ``n_trials``, ``ab_split`` (None), ``subject_dir``.

    Raises
    ------
    FileNotFoundError
        No derivatives directory for ``subjid``.
    """
    subj_dirs = [d for _, d in _iter_subject_dirs(derivatives_dir, [subjid])]
    if not subj_dirs:
        raise FileNotFoundError(f"No derivatives directory for subject {subjid}")
    subj_dir = subj_dirs[0]

    segments, ab_segments, labels, sizes = [], [], [], []
    for ses_dir in _filter_session_dirs(subj_dir, date_range):
        results_dir = ses_dir / "saved_analysis_results"
        if not results_dir.exists():
            continue
        views = _load_trial_views(results_dir)
        td = views["rewarded"] if rewarded_only else views["completed"]
        if td.empty:
            continue
        if "global_trial_id" in td.columns:
            td = td.sort_values("global_trial_id")
        segments.append(_short_mask(td).to_numpy(dtype=np.int8))
        ab_segments.append(_ab_label(td).to_numpy(dtype="<U1"))
        labels.append(ses_dir.name.split("_date-")[-1])
        sizes.append(len(td))

    s = np.concatenate(segments) if segments else np.zeros(0, dtype=np.int8)
    ab = np.concatenate(ab_segments) if ab_segments else np.zeros(0, dtype="<U1")
    sizes_arr = np.asarray(sizes, dtype=int)
    session_ends = np.cumsum(sizes_arr) - 1 if sizes_arr.size else np.zeros(0, dtype=int)
    session_starts = np.concatenate(([0], session_ends[:-1] + 1)) if sizes_arr.size else np.zeros(0, dtype=int)
    return {
        "subjid": subjid,
        "trial_ids": np.arange(s.size),
        "global_ids": np.arange(s.size),
        "s": s,
        "ab": ab,
        "session_ends": session_ends,
        "session_starts": session_starts,
        "session_labels": labels,
        "session_index": np.repeat(np.arange(sizes_arr.size), sizes_arr),
        "session_sizes": sizes_arr,
        "n_trials": int(s.size),
        "ab_split": None,
        "subject_dir": subj_dir,
    }


def subset_by_ab(prep: dict, letter: str) -> dict:
    """Restrict a prepared subject to its A- or B-reward trials, re-indexing the trial axis.

    The subset gets its own contiguous ``0..m-1`` modelling axis, because the switch-point
    index must index the sequence being fitted. ``global_ids`` keeps each trial's position on
    the full, unsplit axis so a ``tau`` can be reported in both. Sessions holding no trial of
    this identity drop out, so the sleep markers stay on real trials. Trials whose reward
    identity is unresolved belong to neither subset and are dropped.
    """
    mask = prep["ab"] == letter
    sizes = np.bincount(prep["session_index"][mask], minlength=len(prep["session_labels"]))
    kept_sessions = sizes > 0
    sizes = sizes[kept_sessions]
    ends = np.cumsum(sizes) - 1 if sizes.size else np.zeros(0, dtype=int)
    starts = np.concatenate(([0], ends[:-1] + 1)) if sizes.size else np.zeros(0, dtype=int)
    return {**prep,
            "trial_ids": np.arange(int(mask.sum())),
            "global_ids": prep["trial_ids"][mask],
            "s": prep["s"][mask],
            "ab": prep["ab"][mask],
            "session_ends": ends,
            "session_starts": starts,
            "session_labels": [lab for lab, keep in zip(prep["session_labels"], kept_sessions) if keep],
            "session_index": np.repeat(np.arange(sizes.size), sizes),
            "session_sizes": sizes,
            "n_trials": int(mask.sum()),
            "ab_split": letter}


def subject_label(prep: dict) -> str:
    """``"Subject 40"``, or ``"Subject 40 | reward A"`` for an A/B split."""
    suffix = f" | reward {prep['ab_split']}" if prep.get("ab_split") else ""
    return f"Subject {prep['subjid']}{suffix}"
