# Defers evaluation of PEP-604 annotations, matching the modules it re-exports.
from __future__ import annotations

"""The curated public surface: what the other repos in the family may use.

    from hypnose_behavior.api import session
    s = session(57, 20260709)

**Hand-maintained.** Every name below was chosen; nothing
arrives here by being importable. Anything not listed -- the workers, the plotters'
private helpers, anything with a leading underscore -- is internal and moves without
notice.

- **No eager re-exports in any package `__init__.py`.** They stay docstring-only or
  empty, which is what keeps `frames.py` and `parameters.py` importable as leaves: an
  eager re-export makes `import hypnose_behavior.frames` pull matplotlib, harp, aeon and
  dotmap. `hypnose_behavior/__init__.py` forwards four names here through a lazy
  PEP 562 `__getattr__`, so nothing is imported until one is touched.
- **This module pulls no matplotlib**, which is a constraint on what may go in it. The
  plotters are named here and deliberately not imported: a repo wanting a number should
  not pay for a drawing library.
- **`load_session_results` is deliberately absent** -- it resolves *and* loads on every
  call, which is what `session(...)` exists to stop. So is
  `frames.build_position_data`, which would offer a second per-position source that
  skips `load_position_data`'s filter back to the caller's trials (section 28).
- **This surface must never grow a loader for `metrics_*.json`.** `Session.metrics`
  computes through the registry; that file is an export, never an input (section 5).
  It is stated here as well as at the accessor because this is where someone would add
  the convenience.
"""

from hypnose_behavior.accessors import (
    Session, metric_names, pooled, pooled_metrics, session, sessions,
)
from hypnose_behavior.io.layout import SessionRef, derivatives, rawdata, session_selectors
from hypnose_behavior.io.load_results import (
    SessionResults, load_position_data, load_results_dir,
)
from hypnose_behavior.io.parquet_peek import peek
from hypnose_behavior.io.protocol_schema import (
    MODES, ODOUR_DISCRIMINATION, SINGLE_REWARD, STANDARD,
    mode_independent_columns, trial_data_columns,
)
from hypnose_behavior.metric_analysis.registry import REGISTRY, MetricSpec
from hypnose_behavior.metric_analysis.run import REPORT, metric_value
from hypnose_behavior.parameters import scoring_parameters

__all__ = [
    # -- a session, resolved once -----------------------------------------------------
    "session",                  # session(subjid, date) -> Session
    "sessions",                 # every analysed session of one or more subjects
    "Session",                  # .trial_data / .position_data / .metrics / .peek
    "metric_names",             # what you may ask `Session.metrics` for

    # -- a cohort, in one frame --------------------------------------------------------
    # `subjid`/`date`/`ses` are stamped on and nothing else is rewritten, so a pooled row
    # is byte-identical to the session's own. `global_trial_id` therefore stays
    # non-unique across sessions -- key on (subjid, date, global_trial_id), section 28.
    "pooled",                   # trial_data / position_data over many sessions
    "pooled_metrics",           # one row per session, one column per metric

    # -- reading a directory you already hold ------------------------------------------
    # The cheap door, for a caller that has walked the tree itself. `load_position_data`
    # is the single seam between "where the per-position facts live" and everything that
    # reads them (section 28).
    "load_results_dir",
    "load_position_data",
    "SessionResults",

    # -- what a saved session is -------------------------------------------------------
    # The protocol mode is part of the saved schema (section 20) and `trial_data`'s
    # columns depend on it (section 21). A reader checking a file against the current
    # declaration wants these; note that they describe what *today's* pipeline writes,
    # not what any given file carries -- measured, the two differ in both directions on
    # sessions saved before the restructure.
    "trial_data_columns",
    "mode_independent_columns",
    "MODES", "STANDARD", "SINGLE_REWARD", "ODOUR_DISCRIMINATION",

    # -- the metric registry -----------------------------------------------------------
    # `metric_value(spec, results)` is the one expression a consumer evaluates a metric
    # with (section 5); `Session.metrics` is it with the session resolved for you.
    "REGISTRY", "MetricSpec", "metric_value", "REPORT",

    # -- finding and looking at sessions -----------------------------------------------
    "derivatives", "rawdata", "SessionRef", "session_selectors",
    "peek",

    # -- what the analysis was scored with (section 31) --------------------------------
    "scoring_parameters",
]
