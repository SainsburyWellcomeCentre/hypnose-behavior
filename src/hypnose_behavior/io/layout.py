"""Session and subject discovery for the behavioural rawdata and derivatives trees.

The walking itself is layout knowledge, not data knowledge, so it lives in
`hypnose_helpers.io.layout`. What stays here is the part only this repo knows: that it
has *two* trees, where each is rooted, that its subject directories always carry the
`_id-` suffix, and where an analysed session keeps its outputs -- `results_dir()`,
`table_path()` and `write_path()`, the functions every reader and writer of that
directory goes through.

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

# An analysed session keeps its outputs in one directory beside the raw session. The
# name is `hypnose_behavior`'s convention, not a property of a session, which is why it
# lives here and not on `SessionRef` -- that class is shared with the other modalities.
RESULTS_DIRNAME = "saved_analysis_results"

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


def results_dir(session) -> Path:
    """The analysis-output directory of a session directory or a `SessionRef`.

    Either, because both are in circulation: selecting sessions hands back `SessionRef`s,
    while a caller that assembles a session directory itself -- out of a `sub-`/`ses-`
    pair, or out of a path it was given -- holds only the `Path`. Nothing is checked -- a
    session the pipeline has never analysed yields a path that does not exist, and whether
    that is an error or a session to skip is the caller's decision, not this function's.

    Import the module rather than the name (`layout.results_dir(ref)`): `results_dir`
    is the local variable and the parameter name at most of the call sites, so a bare
    import would be shadowed by the very assignment that calls it.
    """
    return Path(getattr(session, "path", session)) / RESULTS_DIRNAME


# Which subfolder of a results directory each output belongs to. `manifest.json` and
# `summary.json` are deliberately absent: they describe the session as a whole and every
# subfolder's readers depend on them, so they stay at the top level.
#
# Matched by exact name first, then by prefix, because three outputs carry the subject and
# date in their file name (`metrics_57_20260807.json`).
RESULTS_SUBFOLDERS = {
    "metrics_by_trial.parquet": "metric_analysis",
    "metrics_by_poke.parquet": "metric_analysis",
    "trial_data.parquet": "trial_classification_results",
    "trial_data.csv": "trial_classification_results",
    "trial_data.schema.json": "trial_classification_results",
    "position_data.parquet": "trial_classification_results",
    "speed_analysis.parquet": "movement_analysis",
}

RESULTS_SUBFOLDER_PREFIXES = (
    ("metrics_", "metric_analysis"),
    ("merged_summary_", "trial_classification_results"),
    # Covers `non_initiated_attempts`, which is written today, and the three
    # `io.loaders` still offers from older sessions -- `non_initiated_sequences`,
    # `non_initiated_odor1_attempts` and `non_initiated_FA`. A prefix rather than four
    # names because all four are the same table family and must resolve alike; a legacy
    # file that moved without a mapping entry would read back as an empty frame.
    ("non_initiated_", "trial_classification_results"),
)

# Named separately because the tracking files are *discovered* rather than looked up:
# the SLEAP repo writes them under names this package does not choose, so
# `find_tracking_file` below matches them by pattern and needs the folder to search,
# not a mapping entry.
MOVEMENT_SUBFOLDER = "movement_analysis"


def results_subfolder(name: str) -> Optional[str]:
    """The subfolder `name` belongs in, or None for a top-level file."""
    if name in RESULTS_SUBFOLDERS:
        return RESULTS_SUBFOLDERS[name]
    for prefix, folder in RESULTS_SUBFOLDER_PREFIXES:
        if name.startswith(prefix):
            return folder
    return None


def table_path(results_dir, name: str) -> Path:
    """One entry of a results directory, by file name.

    **The single place the layout inside a results directory is decided.** Outputs are
    grouped by the stage that produces them -- `metric_analysis/`,
    `trial_classification_results/`, `movement_analysis/` -- while `manifest.json` and
    `summary.json` stay at the top level, because they describe the session rather than
    any one stage and all three groups' readers depend on them.

    **Both layouts are readable, and the grouped one wins.** A session written before the
    grouping is flat, and re-analysing the tree is a separate and much more expensive
    operation, so the lookup prefers the grouped path when it exists, falls back to the
    flat path when *that* exists, and otherwise returns the grouped path -- which is what
    a writer needs. A session half-migrated by an interrupted move therefore still reads
    correctly, file by file.

    Returning the grouped path for a file that does not exist yet means **a writer must
    create the parent**; `write_path` is that function, and is what every writer calls.

    It answers *where a named file lives*, never *what the directory holds*. A caller
    that globs a results directory is discovering rather than looking up, so it does not
    come through here and has to be found on its own -- `rglob` is the spelling that
    reads a flat session and a grouped one alike.
    """
    base = Path(results_dir)
    folder = results_subfolder(name)
    if folder is None:
        return base / name
    grouped = base / folder / name
    if grouped.exists():
        return grouped
    flat = base / name
    if flat.exists():
        return flat
    return grouped


def results_dir_of(path) -> Path:
    """The results directory a file inside one belongs to, whichever layout it sits in.

    A caller that *found* a file -- by glob, rather than by asking `table_path` where it
    should be -- holds a path that is one level deeper in a grouped session than in a
    flat one, so `.parent` answers differently for the two. This walks up to the
    `saved_analysis_results` directory itself, which is the same answer for both.
    """
    p = Path(path)
    for ancestor in (p, *p.parents):
        if ancestor.name == RESULTS_DIRNAME:
            return ancestor
    raise ValueError(f"{path} is not inside a {RESULTS_DIRNAME} directory")


def find_tracking_file(results_dir, stem_glob: str) -> Optional[Path]:
    """A tracking file matching ``stem_glob`` (a filename glob WITHOUT extension),
    preferring .parquet over .csv. None if nothing matches.

    Example: ``find_tracking_file(results_dir, "*_combined_sleap_tracking_timestamps")``

    **Discovery, not lookup, which is why it is a glob and not `table_path`.** The SLEAP
    repo writes these files under names this package does not choose, so they are matched
    by pattern rather than looked up. `movement_analysis/` is searched first and the
    results directory second, so a session written either way resolves and a grouped file
    wins over a flat one of the same name.

    Lives here rather than in `utils/` because it decides where inside a results
    directory to look, and because `io/` imports `utils/` -- the reverse edge would make
    the two a directory cycle.
    """
    for root in (Path(results_dir) / MOVEMENT_SUBFOLDER, Path(results_dir)):
        for ext in ("parquet", "csv"):
            matches = [f for f in sorted(root.glob(f"{stem_glob}.{ext}"))
                       if not f.name.startswith("._")]
            if matches:
                return matches[0]
    return None


def write_path(results_dir, name: str) -> Path:
    """`table_path`, with the parent directory created. What every writer calls.

    Separate from `table_path` because a lookup that creates directories would leave
    empty `metric_analysis/` folders behind every time a reader asked for a file that is
    not there.
    """
    path = table_path(results_dir, name)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _iter_subject_dirs(derivatives_dir: Optional[Path], subjids: Optional[Iterable[int]]):
    """Yield (subjid, subject_dir) tuples from derivatives.

    Thin wrapper over the shared layout walker. A named subject that does not exist is
    skipped rather than raised on; two directories for one subject raise.
    """
    yield from layout_for(derivatives_dir).iter_subjects(subjids)


def session_selectors(*, ses=None, index=None, date_range=None,
                      ses_range=None, index_range=None) -> dict:
    """Bundle the non-date session selectors for forwarding, unchanged.

    Every public plotter takes these five alongside `dates` and passes them straight to
    `_filter_sessions`, which hands them to the shared `filter_sessions`. This exists so
    that threading them through ~40 plotters is one line each rather than five, and --
    more to the point -- so no plotter is tempted to *interpret* them. `find_sessions`
    already takes exactly these keywords.

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

    **The one way into a subject's sessions**, and it returns a `list`: several callers
    `enumerate` the result or take its `len()`. Each `SessionRef` carries `ses`, `date`
    and `session_index` alongside the path, so no caller re-parses the directory name to
    recover what the listing already read out of it.
    """
    return filter_sessions(
        list_sessions(subj_dir),
        date=dates, ses=ses, index=index,
        date_range=date_range, ses_range=ses_range, index_range=index_range,
    )


__all__ = [
    "rawdata", "derivatives", "layout_for", "SUBJECT_PATTERN",
    "RESULTS_DIRNAME", "results_dir", "table_path", "write_path",
    "RESULTS_SUBFOLDERS", "RESULTS_SUBFOLDER_PREFIXES", "results_subfolder",
    "MOVEMENT_SUBFOLDER", "find_tracking_file", "results_dir_of",
    "session_selectors",
    "SessionRef", "SessionLayout", "DuplicateSessionError",
    "list_sessions", "filter_sessions", "normalize_subjid",
    "parse_subject", "parse_subject_dirname", "parse_session_dirname",
]
