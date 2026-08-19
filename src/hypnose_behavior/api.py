# Defers evaluation of PEP-604 annotations, matching the modules it re-exports.
from __future__ import annotations

"""The curated public surface: what the other repos in the family may use.

    from hypnose_behavior.api import session
    s = session(57, 20260709)

**Hand-maintained, and that is the whole design.** Every name below was chosen; nothing
arrives here by being importable. A downstream repo that pins to these names is pinning
to something this repo has undertaken to keep working, while everything else --
`hypnose_behavior.trial_classification.classify_trials`, the plotters' private helpers,
anything with a leading underscore -- is internal and moves without notice. Three
restructures have moved most of this code-base at least once, so "it imported last week"
is not a contract.

### Why this is a module and not `__init__.py` re-exports

Follow-up item 2, and it is settled: **no eager re-exports in any package
`__init__.py`.** They are docstring-only, or empty, deliberately -- that is what keeps
`frames.py` and `parameters.py` importable as leaves (`docs/DECISIONS.md` sections 3 and
31). Eager re-exports would make `import hypnose_behavior.frames` pull matplotlib, harp,
aeon and dotmap, and every downstream repo would pay it, including the ones pinned to
Python 3.9 that `frames.py` is kept importable for. Measured: `import hypnose_behavior`
loads **39** modules in 0.01 s, `import hypnose_behavior.frames` **614** with none of
those four, and importing this module **1,326** -- which is the right cost to pay
*explicitly* and the wrong one to pay by accident.

`hypnose_behavior/__init__.py` forwards four names here through a lazy `__getattr__`
(PEP 562), so `hypnose_behavior.session(...)` works without importing anything until it
is called. That is the form item 2 permits; it is not a re-export.

### What is deliberately NOT here

- **The plotters.** `visualization/` is 19 modules and importing any of them pulls
  matplotlib. A repo wanting a number should not pay for a drawing library, so figures
  stay at `hypnose_behavior.visualization.<module>` -- named here, not imported here.
- **`load_session_results(subjid, date)`.** It still exists and still works, but it
  resolves the session *and* loads it on every call, which is what item 7b was written
  to stop: `derivatives.find_session` costs 14.6 s on a cold mount against 29 ms to
  compute every metric (section 5). `session(...)` resolves once and `.results()` is the
  same mapping.
- **`frames.build_position_data`.** It is the *compatibility path* for sessions saved
  before `position_data.parquet` existed, reached through
  `load_position_data`, which is the one place that decides where the per-position facts
  come from and which filters back to the caller's trials (section 28). Publishing the
  builder separately would offer a second source that skips that filter.
- **The batch entry points and the CLI wiring.** `scripts/` is the interface for running
  this repo's pipeline; another repo consumes its output rather than driving it.

### And one thing this surface must never grow

A loader for `metrics_*.json`. `Session.metrics` **computes** through the registry; that
file is an export and the record of an analysis run, never an input (section 5). The
whole reason it is stated here as well as at the accessor is that this module is where
someone would add the convenience.
"""

from hypnose_behavior.accessors import Session, metric_names, session, sessions
from hypnose_behavior.io.layout import SessionRef, derivatives, rawdata
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
from hypnose_behavior.utils.helpers import session_selectors

__all__ = [
    # -- a session, resolved once (item 7b) --------------------------------------------
    "session",                  # session(subjid, date) -> Session
    "sessions",                 # every analysed session matching the six selectors
    "Session",                  # .trial_data / .position_data / .metrics / .peek
    "metric_names",             # what you may ask `Session.metrics` for

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
