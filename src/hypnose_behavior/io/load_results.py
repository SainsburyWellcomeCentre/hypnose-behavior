# Defers evaluation of PEP-604 annotations (`X | None`), keeping this module
# importable on Python 3.9 for repos pinned there (hypnose-eeg-preprocessing).
from __future__ import annotations

"""Reading a session's saved analysis results.

The read side of `io/save_results.py`: it reads `saved_analysis_results/` back
into the `results` mapping every metric wrapper consumes. Kept separate from
`io/loaders.py`, which serves a different set of consumers off the same directory.

**`metrics_*.json` is not read here and is not a plotting input.** Plotters compute
through the registry; that file is an export and the record of an analysis run.
See DECISIONS.md section 5.

**Keep `hypnose_behavior.frames` a leaf** -- the day it imports anything else in the
package, every layer standing on it inherits that dependency. See DECISIONS.md
section 3.
"""

import json
import warnings
from pathlib import Path

import pandas as pd

from hypnose_behavior.io import layout
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

    The single place that decides where the per-position facts come from: prefer
    `position_data.parquet`, fall back to `build_position_data` off the JSON blobs for
    sessions saved before that file existed.

    **Always filter the saved table back to `trials`.** Callers do not all pass the whole
    session -- `sing_rew._session_reward_rts` passes the rewarded trials only -- and an
    unfiltered read silently widens such a caller to every trial in the session, which is a
    changed metric with no error. See DECISIONS.md section 28.
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
        # "this session had no positions" and silently report nothing.
        warnings.warn(
            f"{results_dir}: no position_data.parquet and no position blobs in "
            "trial_data, so every per-position metric and figure will be empty -- "
            "re-run trial classification for this session.",
            RuntimeWarning, stacklevel=2)
    return derived


class SessionResults(dict):
    """A session's `results` mapping, with `position_data` built on first access."""

    @classmethod
    def from_trials(cls, trial_data):
        """A results mapping over an already-loaded trial table.

        For callers that hold `trial_data` and want to evaluate registered metrics
        against it. `position_data` stays lazy, so a `frame="trials"` metric never
        builds one.
        """
        obj = cls(trial_data=trial_data)
        dict.__setitem__(obj, "position_data", _UNBUILT)
        return obj

    def __getitem__(self, key):
        value = dict.__getitem__(self, key)
        if value is _UNBUILT:
            trials = dict.__getitem__(self, "trial_data")
            # `results_dir` is a key of this mapping, not a separate attribute, so it
            # survives `copy()` on its own. `from_trials` sets no such key: a caller
            # holding only a frame has no directory to read, and the derivation is the
            # right answer for it.
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

    A session saved before a column was renamed still loads, and the metrics and figures
    reading that column return empty with no error; this is what makes that speak. The
    manifest's commit stamp catches changed *values*, this catches a changed *schema*.
    See DECISIONS.md section 22.

    **A file with no recorded `protocol_mode` is checked, not skipped** -- against
    `mode_independent_columns()`, which is a strict subset of all three declarations, so
    there is no risk of a false alarm.

    **Emit through `warnings`, never `print`:** it must land on stderr, where it cannot
    disturb the stdout `verbose_diff.py` and `plot_regression.py` compare.
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

    # No mode recorded: the file predates the schema stamp, so it is stale by construction.
    # Warn either way, and name the columns when there are any, because those are the ones
    # already breaking figures.
    detail = (f" It is missing {', '.join(missing)}, so metrics and figures reading those "
              f"columns will be empty.") if missing else ""
    warnings.warn(
        f"{label}: saved before the analysis recorded its protocol schema, so it cannot be "
        f"fully checked and may be out of date -- consider re-running trial classification."
        f"{detail}",
        RuntimeWarning, stacklevel=3)


def load_session_results(subjid, date):
    """Resolve a subject and date to its session directory and load the results mapping.

    Prefer `load_results_dir` where the directory is already known: resolving is the
    expensive half. See DECISIONS.md section 5.
    """
    # The shared resolver: it reports the available sessions on a miss and raises rather
    # than warning on an ambiguous subject or date.
    session = derivatives.find_session(subjid, date=date)

    results_dir = layout.results_dir(session)
    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")
    return load_results_dir(results_dir)


def load_results_dir(results_dir):
    """The same `results` mapping, for a `saved_analysis_results` path you already hold.

    **This is the entry point for anything that has already walked the tree.** Finding
    the directory is the expensive half -- a cold `derivatives.find_session` costs 14.6 s
    against 29 ms to compute every metric for the session it found -- so routing a plotter
    through the subject/date resolver re-pays that walk to arrive where it started. See
    DECISIONS.md section 5.
    """
    results_dir = Path(results_dir)

    # Load manifest and summary
    manifest = json.load(open(layout.table_path(results_dir, "manifest.json")))
    summary = json.load(open(layout.table_path(results_dir, "summary.json")))

    results = SessionResults()

    # Prefer the unified trial_data parquet; fall back to CSV if needed
    trial_parquet = layout.table_path(results_dir, "trial_data.parquet")
    trial_csv = layout.table_path(results_dir, "trial_data.csv")
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

    # The long per-position frame, resolved by `load_position_data` on first access so
    # that metrics never parse a JSON blob -- see `SessionResults` for why it is deferred.
    dict.__setitem__(results, "position_data", _UNBUILT)

    # The `non_initiated_*` tables are deliberately not loaded: no metric is defined over
    # them, and nothing in `visualization/` or `modelling/` reads them either. Trial
    # classification still writes them, and `loaders._load_table_with_trial_data` still
    # reads them on request.

    # Attach manifest and summary
    results["manifest"] = manifest
    results["summary"] = summary
    results["results_dir"] = str(results_dir)

    return results
