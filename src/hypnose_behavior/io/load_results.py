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
`metric_analysis.run`. Nothing reads it back: since restructure_2 Phase 5 the
plotters compute through the registry, so it is an export and the record of an
analysis run, never an input (`docs/DECISIONS.md` section 5).

Deliberately its own module rather than part of `io/loaders.py`, which reads the
same directory via `_load_trial_views`. The original reason was that `loaders` is
imported by `trial_classification`, so folding this in would have made
classification depend on `metric_analysis` for `build_position_data`. **That reason
no longer applies** -- Phase 7b.4 promoted the module to `hypnose_behavior.frames`,
a leaf below both packages -- but the two stay split, because they serve different
consumers off the same directory.

`load_session_results` calls `frames.build_position_data`. `hypnose_behavior.frames`
imports nothing from the package (standard library and pandas only), so this is a
one-way edge into a leaf, and `io/` no longer imports `metric_analysis` at all. See
`docs/DECISIONS.md` section 3. **Keep `frames.py` a leaf** -- the day it imports
anything else in the package, every layer standing on it inherits that dependency.
"""

import json
import warnings
from pathlib import Path

import pandas as pd

from hypnose_behavior.io.layout import derivatives
from hypnose_behavior.io.protocol_schema import (
    mode_independent_columns, trial_data_columns,
)
from hypnose_behavior.frames import build_position_data

__all__ = ["SessionResults", "load_position_data", "load_results_dir",
           "load_session_results"]

_UNBUILT = object()


def load_position_data(results_dir, trials):
    """One row per ``trial x position`` for `trials` -- read if saved, derived if not.

    **The single place that decides where the per-position facts come from.** Phase
    7b.4a began writing `position_data.parquet`; 7b.4b moved every reader onto this
    frame and then removed the three JSON blobs from `trial_data`, so from here on the
    saved table is the source and the derivation is the compatibility path.

    Prefers the file, falls back to `build_position_data`, and the fallback is not a
    formality: sessions saved before 7b.4a have no such file, and `DECISIONS.md`
    section 2's rule is that an absent source means *unknown*, never "there were no
    positions". Those sessions still carry the blobs, so the derivation answers
    correctly for exactly the files the file is missing from.

    **Filtered back to `trials`, always.** The saved table holds the whole session,
    while callers do not all pass the whole session -- `sing_rew._session_reward_rts`
    passes the rewarded trials only. Returning the file unfiltered would silently widen
    such a caller from its subset to every trial in the session, which is a changed
    metric with no error. When the two frames cannot be keyed on `global_trial_id` the
    derivation is used instead, because it is defined by whatever frame it is handed
    and so cannot make that mistake.
    """
    trials_empty = trials is None or getattr(trials, "empty", True)
    path = Path(results_dir) / "position_data.parquet"
    if not trials_empty and path.exists() and "global_trial_id" in trials.columns:
        try:
            saved = pd.read_parquet(path)
        except Exception as e:
            warnings.warn(f"failed to read {path}: {e} -- deriving from trial_data instead.",
                          RuntimeWarning, stacklevel=2)
        else:
            if not saved.empty and "global_trial_id" in saved.columns:
                return saved[saved["global_trial_id"].isin(trials["global_trial_id"])].copy()

    derived = build_position_data(trials)
    if not trials_empty and getattr(derived, "empty", True):
        # Neither source answered: no saved table, and no blobs to expand. Warn rather
        # than hand back an empty frame, which every per-position metric would read as
        # "this session had no positions" and silently report nothing (section 27's
        # failure, one level up from the field lists).
        warnings.warn(
            f"{results_dir}: no position_data.parquet and no position blobs in "
            "trial_data, so every per-position metric and figure will be empty -- "
            "re-run trial classification for this session.",
            RuntimeWarning, stacklevel=2)
    return derived


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
            trials = dict.__getitem__(self, "trial_data")
            # `results_dir` is already a key of this mapping, so the saved table can be
            # preferred without a second attribute to keep in step through `copy()`.
            # `from_trials` sets no such key -- a caller holding only a frame has no
            # directory to read, and the derivation is the right answer for it.
            results_dir = dict.get(self, "results_dir")
            value = (load_position_data(results_dir, trials) if results_dir
                     else build_position_data(trials))
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


def _session_label(results_dir, manifest) -> str:
    paths = manifest.get("paths") if isinstance(manifest, dict) else None
    if isinstance(paths, dict) and paths.get("sub_folder") and paths.get("ses_folder"):
        return f"{paths['sub_folder']} {paths['ses_folder']}"
    return str(results_dir)


def _warn_if_stale(results_dir, trial_df, manifest) -> None:
    """Warn when a saved `trial_data` is missing columns the current schema declares.

    **Why a field set rather than 7a's commit stamp.** A git SHA says *something* changed
    between the file and now, not whether *this file* is affected -- a one-line plotter fix
    and a trial-classification restructure look identical to it. Comparing field sets answers
    the question actually asked, and costs no maintenance because the record is already the
    source of truth. The two are complementary: the stamp catches changed *values*, this
    catches a changed *schema*.

    **The case it exists for is silent today.** Every derivative saved before Phase 6's
    latency rename carries `fa_latency_ms` and friends, so `FA_avg_response_times` and
    `sing_rew`'s `FR_latency` find no column, hit their `if col not in trials.columns` guard
    and return empty -- a blank figure with no error. Measured on the archive:
    `plot_regression`'s `FR_latency` lost all 35 lines and
    `plot_response_times_completed_vs_fa` all 12. This is what makes that speak.

    **An unknown mode is checked, not skipped.** Files written before 7b carry no
    `protocol_mode`, and guessing one from the columns present would be circular. But the
    base record's fields are common to all three modes, and the merged and assembled columns
    do not depend on the mode at all, so `mode_independent_columns()` can be compared with no
    risk of a false alarm. That is not a weaker check where it counts: the three renamed
    latency columns are merged ones, so an untagged `sub-040 20251124` reports all three,
    where comparing the record's own fields alone would report only `fallback_reason`.

    Emitted through `warnings`, so it lands on stderr and cannot disturb the stdout that
    `verbose_diff.py` and `plot_regression.py` compare.
    """
    if trial_df is None or getattr(trial_df, "empty", True):
        return

    mode = manifest.get("protocol_mode") if isinstance(manifest, dict) else None
    try:
        expected = trial_data_columns(mode) if mode else mode_independent_columns()
    except ValueError:
        # An unrecognised mode is a file this code cannot reason about; say so rather than
        # fall back to a check whose result would be meaningless.
        warnings.warn(f"{_session_label(results_dir, manifest)}: manifest records an unknown "
                      f"protocol_mode {mode!r}; schema not checked.", RuntimeWarning, stacklevel=3)
        return

    missing = [c for c in expected if c not in trial_df.columns]
    label = _session_label(results_dir, manifest)

    if mode:
        if missing:
            warnings.warn(
                f"{label}: saved before {', '.join(missing)} existed -- re-run trial "
                f"classification. Metrics and figures reading those columns will be empty.",
                RuntimeWarning, stacklevel=3)
        return

    # No mode recorded: the file predates Phase 7b, so it is stale by construction -- the
    # schema stamp is only one of the things it lacks. Warn either way, and name the columns
    # when there are any, because those are the ones already breaking figures.
    detail = (f" It is missing {', '.join(missing)}, so metrics and figures reading those "
              f"columns will be empty.") if missing else ""
    warnings.warn(
        f"{label}: saved before the analysis recorded its protocol schema, so it cannot be "
        f"fully checked and may be out of date -- consider re-running trial classification."
        f"{detail}",
        RuntimeWarning, stacklevel=3)


def load_session_results(subjid, date):
    """
    Load saved analysis results for a given subject and date.
    Returns a dict with trial_data, non-initiated tables, and metadata.
    """
    # One resolver for the whole family (restructure_2 Phase 2b); it reports the
    # available sessions on a miss and raises rather than warning on an ambiguous
    # subject or date.
    session = derivatives.find_session(subjid, date=date)

    results_dir = session.path / "saved_analysis_results"
    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")
    return load_results_dir(results_dir)


def load_results_dir(results_dir):
    """The same `results` mapping, for a `saved_analysis_results` path you already hold.

    Split out of `load_session_results` in restructure_2 Phase 5. The two differ
    only in how the directory is found, and finding it is the expensive half: a
    cold `derivatives.find_session` walk costs 14.6 s against 29 ms to compute
    every metric for the session it found (`docs/DECISIONS.md` section 5). Every
    plotter already holds the directory, having walked the tree once itself, so
    routing them through the subject/date resolver would re-pay that walk per
    session purely to arrive back where they started.
    """
    results_dir = Path(results_dir)

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
    _warn_if_stale(results_dir, trial_df, manifest)

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
