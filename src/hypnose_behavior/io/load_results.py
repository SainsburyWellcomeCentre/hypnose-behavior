# Defers evaluation of PEP-604 annotations (`X | None`), keeping this module
# importable on Python 3.9 for repos pinned there (hypnose-eeg-preprocessing).
from __future__ import annotations

"""Reading a session's saved analysis results.

The read side of `io/save_results.py`, which writes them: trial classification
produces `saved_analysis_results/`, this reads it back into the `results` dict
every metric wrapper consumes. Moved out of
`metric_analysis/metrics_utils.py` in restructure_2 Phase 4b, which separates
plumbing from metric definitions.

**Not** `metrics_*.json` -- that is a different file, written by
`metric_analysis.run` and read by `visualization._ensure_metrics_json`.

Deliberately its own module rather than part of `io/loaders.py`, which reads the
same directory via `_load_trial_views`: `loaders` is imported by
`trial_classification`, and folding this in would make classification depend on
`metric_analysis` for `build_position_data`.

`load_session_results` calls `metric_analysis.frames.build_position_data`. That
edge is deliberate and was checked, not assumed: `frames.py` is a leaf (standard
library and pandas only) and both package `__init__`s are docstring-only, so
`io -> metric_analysis.frames` is one-way with no cycle. See `docs/DECISIONS.md`
section 3. **Keep `frames.py` a leaf** -- the day it imports anything else in
the package this becomes a real cycle.
"""

import json

import pandas as pd

from hypnose_behavior.io.layout import derivatives
from hypnose_behavior.metric_analysis.frames import build_position_data

__all__ = ["SessionResults", "load_session_results"]

_UNBUILT = object()


class SessionResults(dict):
    """A session's `results` mapping, with `position_data` derived on first access.

    `build_position_data` is 22 of the 29 ms it costs to compute every metric for a
    session (`docs/DECISIONS.md` section 5), and the callers that never look at it
    are real ones: `io/tracking._load_tracking_and_behavior` and
    `visualization.load_tracking_with_behavior` load a session's tables purely to
    join them against SLEAP tracking, once per session across a whole cohort.
    `run_all_metrics` still materialises it, since 9 of the 25 reported metrics
    declare a `position_data` frame -- laziness helps the tracking paths, not that
    one.

    Subclasses `dict` because that is exactly what consumers expect it to be, and
    they reach for the key in three different ways: `MetricSpec.call` uses `.get`,
    `merge.pool_results_dicts` walks `.keys()` and tests `in`, and everything else
    subscripts. So the key is genuinely present, holding a placeholder, and the
    accessors resolve it -- `__missing__` would fire for none of those. `items`,
    `values` and `copy` are overridden for the same reason: to keep the placeholder
    from ever escaping as a value.
    """

    @classmethod
    def from_trials(cls, trial_data):
        """A results mapping over an already-loaded trial table.

        For callers that hold `trial_data` and want to evaluate registered
        metrics against it -- `visualization/` reads a session directory through
        `_load_trial_views`, not `load_session_results`, but still wants
        `MetricSpec.call`. `position_data` stays lazy, so a `frame="trials"`
        metric never derives one.
        """
        obj = cls(trial_data=trial_data)
        dict.__setitem__(obj, "position_data", _UNBUILT)
        return obj

    def __getitem__(self, key):
        value = dict.__getitem__(self, key)
        if value is _UNBUILT:
            value = build_position_data(dict.__getitem__(self, "trial_data"))
            dict.__setitem__(self, key, value)
        return value

    def get(self, key, default=None):
        return self[key] if key in self else default

    def items(self):
        return [(key, self[key]) for key in self]

    def values(self):
        return [self[key] for key in self]

    def copy(self):
        # Raw values, so a copy of an unbuilt mapping stays unbuilt.
        return SessionResults(dict.items(self))


def load_session_results(subjid, date):
    """
    Load saved analysis results for a given subject and date.
    Returns a dict with trial_data, non-initiated tables, and metadata.
    """
    # One resolver for the whole family (restructure_2 Phase 2b); it reports the
    # available sessions on a miss and raises rather than warning on an ambiguous
    # subject or date.
    session = derivatives.find_session(subjid, date=date)
    subject_dir = session.subject_dir
    session_dir = session.path

    results_dir = session_dir / "saved_analysis_results"
    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    # Load manifest and summary
    manifest = json.load(open(results_dir / "manifest.json"))
    summary = json.load(open(results_dir / "summary.json"))

    results = SessionResults()

    # Prefer the unified trial_data parquet; fall back to CSV if needed
    trial_parquet = results_dir / "trial_data.parquet"
    trial_csv = results_dir / "trial_data.csv"
    trial_df = pd.DataFrame()
    if trial_parquet.exists():
        try:
            trial_df = pd.read_parquet(trial_parquet)
        except Exception as e:
            print(f"Warning: failed to read {trial_parquet}: {e}")
    if trial_df.empty and trial_csv.exists():
        trial_df = pd.read_csv(trial_csv)
    results["trial_data"] = trial_df

    # Long per-position frame, derived here rather than written by the classifier,
    # so metrics never parse a JSON blob and legacy sessions need no
    # compatibility branch (D0, tier 2). Phase 7b's position_data side-table
    # turns this from a derivation into a read. Deferred until something reads
    # it -- see `SessionResults`.
    dict.__setitem__(results, "position_data", _UNBUILT)

    # The three `non_initiated_*` tables are deliberately not loaded. Phase 4a
    # step 6 dropped non-initiated trials from the metric set: they are not in
    # `trial_data`, so every metric over them needed its own frame and its own
    # shape, and integrating them properly is its own piece of work. Trial
    # classification still writes the tables; nothing in `metric_analysis` reads
    # them.

    # Attach manifest and summary
    results["manifest"] = manifest
    results["summary"] = summary
    results["results_dir"] = str(results_dir)

    return results
