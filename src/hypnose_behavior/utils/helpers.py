from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Iterable, Optional, Union

from hypnose_behavior.io.layout import filter_sessions, layout_for, list_sessions

CACHE = OrderedDict()
CACHE_MAX_ITEMS = 40


def vprint(verbose: bool, *args, **kwargs):
    """print(...) only when verbose is True."""
    if verbose:
        print(*args, **kwargs)


def read_tracking_table(path: Union[str, Path]):
    """Read a tracking table from .parquet or .csv.

    Parquet preserves dtypes (tz-aware datetimes, nullable ints) natively; CSV keeps
    the historical utf-8/latin1 fallback.
    """
    import pandas as pd

    path = Path(path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    try:
        return pd.read_csv(path, encoding="utf-8")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin1")


def find_tracking_file(results_dir: Path, stem_glob: str) -> Optional[Path]:
    """Find a tracking file matching ``stem_glob`` (a filename glob WITHOUT extension),
    preferring .parquet over .csv. Returns None if nothing matches.

    Example: find_tracking_file(results_dir, "*_combined_sleap_tracking_timestamps")
    """
    for ext in ("parquet", "csv"):
        matches = [f for f in sorted(results_dir.glob(f"{stem_glob}.{ext}"))
                   if not f.name.startswith("._")]
        if matches:
            return matches[0]
    return None


def _update_cache(subjid, dates, data, kind):
    """Update cache entries for a subject/date set and kind."""
    global CACHE
    for date in dates:
        key = (subjid, date, kind)
        if key in CACHE:
            del CACHE[key]
        CACHE[key] = {
            "kind": kind,
            "data": data[date],
        }
    while len(CACHE) > CACHE_MAX_ITEMS:
        CACHE.popitem(last=False)


def _get_from_cache(subjid, date, kind):
    """Retrieve cached data for (subjid, date, kind)."""
    key = (subjid, date, kind)
    if key in CACHE and CACHE[key]["kind"] == kind:
        return CACHE[key]["data"]
    return None


def clear_cache():
    """Clear all cached items."""
    CACHE.clear()


def _iter_subject_dirs(derivatives_dir: Optional[Path], subjids: Optional[Iterable[int]]):
    """Yield (subjid, subject_dir) tuples from derivatives.

    Thin wrapper over the shared layout walker (restructure_2 Phase 2b) so this repo's
    ~21 call sites keep working unchanged. Named subjects that do not exist are still
    skipped rather than raised on; two directories for one subject now raise instead of
    yielding both.
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
