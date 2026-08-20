from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Optional, Union

from hypnose_behavior.parameters import CACHE_MAX_ITEMS

CACHE = OrderedDict()


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


# Prints the cache these helpers maintain. It lives here because `CACHE` is declared
# in this module, and it draws nothing.
    # Utility function to print current cache keys
def print_cache_keys():
    print("[CACHE CONTENTS] Current cache keys:")
    for k in CACHE.keys():
        print(f"  {k}")
