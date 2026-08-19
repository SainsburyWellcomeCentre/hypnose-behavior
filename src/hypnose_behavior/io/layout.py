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


__all__ = [
    "rawdata", "derivatives", "layout_for", "SUBJECT_PATTERN",
    "SessionRef", "SessionLayout", "DuplicateSessionError",
    "list_sessions", "filter_sessions", "normalize_subjid",
    "parse_subject", "parse_subject_dirname", "parse_session_dirname",
]
