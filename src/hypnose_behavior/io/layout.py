"""Session and subject discovery for the behavioural rawdata and derivatives trees.

The walking itself is layout knowledge, not data knowledge, so it lives in
`hypnose_helpers.io.layout`. What stays here is the part only this repo knows: that it
has *two* trees, where each is rooted, and that its subject directories always carry
the `_id-` suffix.

    from hypnose_behavior.io.layout import derivatives, rawdata

    ses = derivatives.find_session(66, date=20260709)      # exactly one, or raise
    for ses in derivatives.find_sessions(66, ses="03-09"): ...
    x = ses.session_index                                   # gap-free plotting ordinal

- Both trees are named objects with **no default**. `rawdata` holds every recorded
  session, `derivatives` only the analysed ones, so the same subject legitimately has
  different sessions in each; a default makes picking the wrong tree a silent
  one-character mistake.
- The roots are passed as **functions**, never resolved Paths. `qc/_common` redirects
  derivatives per session by setting `HYPNOSE_DERIVATIVES_ROOT` and clearing the
  `lru_cache`; anything holding a Path captured at import keeps answering from the
  real server.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Union

from hypnose_helpers.io.layout import (  # noqa: F401  (re-exported: one import for callers)
    DuplicateSessionError,
    SessionLayout,
    SessionRef,
    filter_sessions,
    list_sessions,
    normalize_subjid,
    parse_session_dirname,
    parse_subject,
    parse_subject_dirname,
)

from hypnose_behavior.io.paths import get_derivatives_root, get_rawdata_root

# Every subject directory in this dataset is `sub-NNN_id-XXX`. Narrowing the pattern
# keeps `iter_subjects()` returning exactly what the previous `glob("sub-*_id-*")`
# calls returned, rather than picking up any other `sub-`-prefixed directory.
SUBJECT_PATTERN = "{subject}_id-*"

rawdata = SessionLayout(
    get_rawdata_root, name="rawdata", subject_pattern=SUBJECT_PATTERN
)
derivatives = SessionLayout(
    get_derivatives_root, name="derivatives", subject_pattern=SUBJECT_PATTERN
)


def layout_for(root=None) -> SessionLayout:
    """The derivatives layout, or an ad-hoc one rooted at ``root``.

    For the handful of callers that accept an explicit derivatives directory rather
    than using the configured one.
    """
    if root is None:
        return derivatives
    return SessionLayout(root, name="derivatives", subject_pattern=SUBJECT_PATTERN)


def _iter_subject_dirs(derivatives_dir: Optional[Path], subjids: Optional[Iterable[int]]):
    """Yield (subjid, subject_dir) tuples from derivatives.

    Thin wrapper over the shared layout walker. A named subject that does not exist is
    skipped rather than raised on; two directories for one subject raise.
    """
    yield from layout_for(derivatives_dir).iter_subjects(subjids)


def session_selectors(*, ses=None, index=None, date_range=None,
                      ses_range=None, index_range=None) -> dict:
    """Bundle the non-date session selectors for forwarding, unchanged.

    Every public plotter takes these five alongside `dates` and passes them straight
    to `_filter_sessions` / `_filter_session_dirs`, which hand them to the shared
    `filter_sessions`. This exists so that threading them through ~40 plotters is one
    line each rather than five, and -- more to the point -- so no plotter is tempted
    to *interpret* them. `find_sessions` already takes exactly these keywords.

    Nothing is validated or rejected here, deliberately: **the keys intersect**, so
    `ses_range=(1, 9), index_range=(3, 5)` legitimately means "of ses 1-9, the 3rd to
    5th sessions chronologically". `None` means "do not filter on this" and an empty
    list means "match nothing" -- see `_filter_sessions`.
    """
    return {"ses": ses, "index": index, "date_range": date_range,
            "ses_range": ses_range, "index_range": index_range}


def _filter_sessions(subj_dir: Path,
                     dates: Optional[Union[Iterable[Union[int, str]], tuple]] = None,
                     *, ses=None, index=None,
                     date_range=None, ses_range=None, index_range=None) -> list:
    """`SessionRef`s for a subject directory, narrowed by any of the six selectors.

    `dates` stays positional because ~36 call sites pass it that way. A 2-tuple is an
    inclusive range (either bound may be None); any other iterable is a membership
    test; None means every session -- unchanged, and now handled by `filter_sessions`
    itself, whose `_split_selector` reads a 2-tuple as a range exactly as the explicit
    branch here used to.

    **The selectors intersect; none is required.** `None` means "do not filter on
    this", and an empty list means "match nothing" -- load-bearing rather than
    incidental, because callers build a per-subject date list and pass it straight
    through, so a subject with no requested dates must yield no sessions rather than
    its whole history.

    `index` is the subject's gap-free chronological rank over its *whole* history, so
    it stays comparable across cohorts. **It selects; it does not position** -- see
    `docs/DECISIONS.md` section 8.

    Prefer this over `_filter_session_dirs` in new code: a `SessionRef` carries `ses`,
    `date` and `session_index` alongside the path, which saves the caller re-parsing the
    directory name -- the habit that produced 17 copies of this lookup.
    """
    return filter_sessions(
        list_sessions(subj_dir),
        date=dates, ses=ses, index=index,
        date_range=date_range, ses_range=ses_range, index_range=index_range,
    )


def _filter_session_dirs(subj_dir: Path,
                         dates: Optional[Union[Iterable[Union[int, str]], tuple]] = None,
                         *, ses=None, index=None,
                         date_range=None, ses_range=None, index_range=None):
    """Session directories for a subject, narrowed by any of the six selectors.

    The paths-only shim over `_filter_sessions`; see there for the semantics.
    """
    return [s.path for s in _filter_sessions(
        subj_dir, dates, ses=ses, index=index,
        date_range=date_range, ses_range=ses_range, index_range=index_range,
    )]


__all__ = [
    "rawdata", "derivatives", "layout_for", "SUBJECT_PATTERN",
    "session_selectors",
    "SessionRef", "SessionLayout", "DuplicateSessionError",
    "list_sessions", "filter_sessions", "normalize_subjid",
    "parse_subject", "parse_subject_dirname", "parse_session_dirname",
]
