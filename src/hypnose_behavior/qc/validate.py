"""Validation helpers: confirm rawdata exists before running a pipeline.

Keeps the pipeline from running on missing data and gives a clean early message.
A single missing date inside a multi-date request is *flagged but not fatal*
(gaps in a date range are expected); a missing subject, or a request where none
of the dates exist, is treated as nothing-to-run.

Only lists directories (cheap) -- it does not read data files.
"""
from __future__ import annotations

from hypnose_behavior.io.layout import normalize_subjid, rawdata


def _normalize_dates(dates):
    if dates is None:
        return None
    if isinstance(dates, (list, tuple, set)):
        return [str(d) for d in dates]
    return [str(dates)]


def validate_subject(subjid, dates=None, *, verbose: bool = True) -> dict:
    """Check that rawdata exists for ``subjid`` (and optionally specific ``dates``).

    Parameters
    ----------
    subjid : int | str
        Subject id (e.g. 53 or "053").
    dates : None | str | int | iterable
        Specific date(s) ``YYYYMMDD`` to check. ``None`` checks only that the
        subject has any sessions.
    verbose : bool
        Print warnings for missing data.

    Returns
    -------
    dict with keys: subjid, subject_dir, requested_dates, available_dates,
    found_dates, missing_dates, ok. ``ok`` is False only when there is nothing
    to run (subject missing, or none of the requested dates exist).
    """
    subj_str = normalize_subjid(subjid)
    result = {
        "subjid": str(subjid), "subject_dir": None, "requested_dates": None,
        "available_dates": [], "found_dates": [], "missing_dates": [], "ok": False,
    }

    subj_dir = rawdata.subject_dir(subjid, missing_ok=True)
    if subj_dir is None:
        if verbose:
            print(f"[validate_subject] No rawdata directory for {subj_str}; nothing to run.")
        return result
    result["subject_dir"] = str(subj_dir)

    available = sorted({s.date for s in rawdata.find_sessions(subjid)})
    result["available_dates"] = available

    requested = _normalize_dates(dates)
    if requested is None:
        result["found_dates"] = available
        result["ok"] = bool(available)
        if verbose and not available:
            print(f"[validate_subject] {subj_str}: no sessions found; nothing to run.")
        return result

    result["requested_dates"] = requested
    result["found_dates"] = [d for d in requested if d in available]
    result["missing_dates"] = [d for d in requested if d not in available]

    if verbose and result["missing_dates"]:
        print(f"[validate_subject] {subj_str}: {len(result['missing_dates'])} requested date(s) "
              f"not found (skipping; gaps are expected): {', '.join(result['missing_dates'])}")
    result["ok"] = bool(result["found_dates"])
    if verbose and not result["ok"]:
        print(f"[validate_subject] {subj_str}: none of the requested dates exist; nothing to run.")
    return result
