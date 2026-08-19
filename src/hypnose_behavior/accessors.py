# Defers evaluation of PEP-604 annotations (`X | None`), matching the modules below.
from __future__ import annotations

"""A session handle: resolve once, then read its tables and compute its metrics.

    s = hypnose_behavior.session(57, 20260709)   # resolves ONCE
    s.trial_data(columns=None)                   # None = every column
    s.position_data()
    s.metrics(["decision_accuracy", "poke_durations"])

and the same over a cohort, one tree walk per subject:

    hs = sessions([57, 40], date_range=(20260101, 20260731))
    pooled(hs, "trial_data", columns=["response_time_ms"])   # + subjid/date/ses
    pooled_metrics(hs, ["decision_accuracy"])                # one row per session

- **`metrics()` computes through the registry.** It does not read `metrics_*.json`, and
  there is deliberately no sibling that does -- that file is an export, never an input.
  **Do not "optimise" this into a disk read**: it would save 25 ms against the 14.6 s
  mount walk the caller has already paid, and cost the guarantee that two figures
  showing one quantity cannot disagree. See DECISIONS.md section 5.
- Every metric is evaluated by `metric_analysis.run.metric_value`, the same function
  `visualization.prep._computed_metrics` calls, so a plotter and this handle cannot
  drift apart.
- **Resolve once.** `derivatives.find_session` costs 14.6 s on a cold mount against
  29 ms to compute every metric, so resolution happens in `session()` / `sessions()`
  and everything after is a file read against a path in hand.
  `Session.from_results_dir` is the cheaper door for a caller that already walked.
- **Nothing inside the package may import this module.** It sits at the root beside
  `frames.py` and `parameters.py` but is the opposite of a leaf: it imports downward
  into `io/` and `metric_analysis/` and exists to be the top of the stack. The moment
  something below reaches up for it, the leaves inherit the whole import graph.
"""

import difflib
import inspect
import warnings
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Union

import pandas as pd

from hypnose_behavior.io.layout import (
    derivatives, parse_session_dirname, parse_subject_dirname,
)
from hypnose_behavior.io.load_results import load_results_dir
from hypnose_behavior.io.parquet_peek import DEFAULT_ROWS, peek
from hypnose_behavior.io.protocol_schema import (
    mode_independent_columns, trial_data_columns,
)
from hypnose_behavior.metric_analysis.registry import REGISTRY
# `metric_value` is the section 5 expression; importing `run` is also what *populates*
# `REGISTRY`, since a metric registers itself where it is defined and `run` imports
# every definition module for exactly that side effect (section 4).
from hypnose_behavior.metric_analysis.run import REPORT, metric_value
from hypnose_behavior.utils.helpers import session_selectors

__all__ = ["Session", "session", "sessions", "metric_names", "pooled", "pooled_metrics"]

# Prepended to every pooled frame, in this order. `trial_data` and `position_data` carry
# none of them: measured across the fixtures, the only ids either frame holds are
# `global_trial_id`, `trial_id` and `run_id` (section 28).
_IDENTITY = ("subjid", "date", "ses")

# What `pooled` will read. The per-grain metric tables are deliberately absent: they are
# an export and a record of an analysis run, never an input (section 25).
_POOLABLE = ("trial_data", "position_data")
_EXPORT_ONLY = ("metrics_by_trial", "metrics_by_poke", "non_initiated_attempts")


# --------------------------------------------------------------------------------------
# Naming things that do not exist
# --------------------------------------------------------------------------------------

def _closest(name: str, candidates: Iterable[str], label: str) -> str:
    """`" Did you mean ...?"` for a name that missed, or `""` when nothing is close."""
    close = difflib.get_close_matches(str(name), [str(c) for c in candidates], n=3)
    if not close:
        return ""
    return f" Closest {label}: {', '.join(repr(c) for c in close)}."


def _as_names(value, argument: str) -> List[str]:
    """A list of names from an iterable of them, rejecting a bare string.

    A `str` is iterable, so `columns="response_time_ms"` would otherwise be read as
    eighteen one-character column names and fail with a message about `'r'`.
    """
    if isinstance(value, str):
        raise TypeError(f"{argument}= takes a sequence of names, not a single string; "
                        f"pass [{value!r}].")
    return [v for v in value]


# --------------------------------------------------------------------------------------
# Which metrics can be asked for without inventing a figure choice
# --------------------------------------------------------------------------------------

def _unsuppliable_parameters(spec) -> tuple:
    """Required parameters of `spec`'s core that nothing can supply for it.

    Only meaningful for a metric with **no session wrapper**. A wrapper's whole job is
    to dig session configuration out of `results` -- `hidden_rule_counts_by_odor`'s core
    requires `hr_odors` and `hr_positions`, and its wrapper finds them in
    `manifest`/`summary` (section 5) -- so a metric that has one is always askable, and
    reporting its core's requirements would be reporting a solved problem.

    Measured over all 43 registered metrics: this is non-empty for exactly two,
    `rolling_reward_fraction` and `rolling_hr_reward_fraction`, which take `window`
    positionally with no default. That is the same boundary `qc/regression.py` draws
    when it fingerprints 16 of the 18 unreported metrics (section 26), reached
    independently -- and it is section 5's line, that a window is a property of the
    figure being drawn and not of the session.

    A parameter with a *default* is not one of these: `fa_types=None` means
    **unfiltered** and `aborted=False` means **completed**, both of which are definite
    values a session has (section 25).
    """
    if spec.session is not None:
        return ()
    params = list(inspect.signature(spec.core).parameters.values())
    n_frames = 2 if spec.frame == "trials+position_data" else 1
    kinds = (inspect.Parameter.POSITIONAL_ONLY,
             inspect.Parameter.POSITIONAL_OR_KEYWORD,
             inspect.Parameter.KEYWORD_ONLY)
    return tuple(p.name for p in params[n_frames:]
                 if p.default is inspect.Parameter.empty and p.kind in kinds)


def _resolve_spec(name: str):
    """The `MetricSpec` for `name`, or a message saying precisely why there is none."""
    spec = REGISTRY.get(name)
    if spec is None:
        # `hidden_rule_counts_by_odor` is saved as `hidden_rule_by_odor`, and it is the
        # only metric whose registry name and `metrics_*.json` key differ. Someone
        # reading a saved file will reach for the key, so say what it maps to rather
        # than accepting it -- two spellings of one metric is how the two come apart.
        for registered, other in REGISTRY.items():
            if other.key == name:
                raise KeyError(
                    f"{name!r} is the metrics_*.json key for the metric {registered!r}; "
                    f"ask for {registered!r}.")
        raise KeyError(f"no registered metric {name!r}."
                       + _closest(name, REGISTRY, "metric")
                       + " See hypnose_behavior.api.metric_names().")

    needed = _unsuppliable_parameters(spec)
    if needed:
        raise TypeError(
            f"{name!r} needs {', '.join(repr(p) for p in needed)}, which is a property "
            f"of the figure you are drawing and not of the session, so this accessor "
            f"has no correct value to choose (docs/DECISIONS.md section 5). Call it "
            f"directly with the choice you mean:\n"
            f'    REGISTRY[{name!r}].call(s.results(), {needed[0]}=...)\n'
            f"    from hypnose_behavior.metric_analysis.registry import REGISTRY")
    return spec


def metric_names(*, reported: Optional[bool] = None) -> List[str]:
    """Every registered metric name, sorted. `reported=` narrows to `run.REPORT`.

    `reported=True` gives the 25 metrics an analysis run saves into `metrics_*.json`;
    `reported=False` gives the 18 it does not, which are registered and therefore
    computable but are not part of the saved record (section 4).

    **Two of them cannot be asked for through `Session.metrics`** and will raise:
    `rolling_reward_fraction` and `rolling_hr_reward_fraction` need a `window`. That is
    deliberate rather than an oversight -- see `_unsuppliable_parameters` -- so
    `s.metrics(metric_names())` raises by design.
    """
    names = sorted(REGISTRY)
    if reported is None:
        return names
    in_report = set(REPORT)
    return [n for n in names if (n in in_report) is bool(reported)]


# --------------------------------------------------------------------------------------
# The handle
# --------------------------------------------------------------------------------------

class Session:
    """One analysed session's saved directory, plus the metrics computed from it.

    Build one with `session(...)`, `sessions(...)` or `Session.from_results_dir(...)`
    rather than calling this directly; those are where resolution happens, and doing it
    once is the point of the class.

    Loading is lazy and cached: the constructor touches no file, and the first call that
    needs data reads the directory once. `position_data` is lazier still, because
    `SessionResults` defers it (section 5) -- a handle used only for `trial_data` never
    materialises it.
    """

    def __init__(self, results_dir, *, ref=None):
        self.results_dir = Path(results_dir)
        self.ref = ref
        self._loaded = None

        if ref is not None:
            self.subjid, self.date = ref.subjid, str(ref.date)
            self.ses, self.session_index = ref.ses, ref.session_index
        else:
            # The directory name *is* the identity, so reading it invents nothing.
            # `session_index` stays None on purpose: it is a rank within a listing and
            # is only defined against one (sections 8 and 32), so producing it would
            # mean walking the subject's tree -- the 14.6 s this class exists to avoid.
            parsed = parse_session_dirname(self.results_dir.parent.name)
            self.ses, self.date = parsed if parsed else (None, None)
            self.subjid = parse_subject_dirname(self.results_dir.parent.parent.name)
            self.session_index = None

    @classmethod
    def from_results_dir(cls, results_dir) -> "Session":
        """A handle on a `saved_analysis_results` path you already hold.

        The cheap door. Every plotter has already walked the derivatives tree, so
        routing it back through the subject/date resolver would re-pay that walk per
        session purely to arrive where it started -- the reason `load_results_dir`
        exists beside `load_session_results` (section 5).
        """
        results_dir = Path(results_dir)
        if not results_dir.exists():
            raise FileNotFoundError(f"Results directory not found: {results_dir}")
        return cls(results_dir)

    # -- the loaded session ------------------------------------------------------------

    def results(self):
        """The `results` mapping every metric wrapper consumes, loaded once and cached.

        Public because it is the escape hatch: a metric needing a figure parameter is
        refused by `metrics()` (section 5) and is reached from here instead --
        `REGISTRY["rolling_reward_fraction"].call(s.results(), window=20)` -- without
        re-resolving the session or re-reading the directory.

        `io/load_results.load_results_dir` emits the section 22 staleness warning on the
        way, once per handle: a saved file missing columns the current schema declares
        says so here rather than silently drawing an empty figure later.
        """
        if self._loaded is None:
            self._loaded = load_results_dir(self.results_dir)
        return self._loaded

    def manifest(self) -> dict:
        """`manifest.json`: the protocol mode (section 20), the provenance stamp
        (section 19) and the scoring knobs applied (section 31)."""
        return self.results().get("manifest", {}) or {}

    def summary(self) -> dict:
        """`summary.json`: the per-session counts and the *task schema* parameters this
        session was configured with -- not the analysis knobs, which are the manifest's
        `analysis_parameters` (section 31)."""
        return self.results().get("summary", {}) or {}

    def protocol_mode(self) -> Optional[str]:
        """`standard` / `single_reward` / `odour_discrimination`, or None.

        None means the file was written before the mode was recorded, not that the
        session had no protocol -- an absent marker is *unknown* (section 2). **Do not
        guess one from the columns present**; that is circular, which is why the
        loader's schema check falls back to `mode_independent_columns()`.
        """
        return self.manifest().get("protocol_mode")

    # -- the measured tables -----------------------------------------------------------

    def trial_data(self, columns: Optional[Sequence[str]] = None) -> pd.DataFrame:
        """One row per trial. `columns=None` is every column.

        Returns a copy, so a caller reshaping the frame cannot corrupt the cached one
        that this handle's metrics are computed from.

        **Check a requested column against this session's frame, never against
        `trial_data_columns(mode)`.** Saved sessions differ from the declaration in both
        directions, so validating against it refuses columns that are in the file and
        accepts ones that are not. Any name is accepted as input; the only failure is
        "this session does not have it". See DECISIONS.md section 34.

        The declaration is used only to *word* that failure: a missing column the schema
        declares means the session wants re-analysing (section 22), while one it does
        not declare is a typo.
        """
        frame = self.results()["trial_data"]
        if columns is None:
            return frame.copy()
        names = _as_names(columns, "columns")
        self._check_trial_columns(frame, names)
        return frame.loc[:, names].copy()

    def position_data(self, columns: Optional[Sequence[str]] = None) -> pd.DataFrame:
        """One row per `trial x position`, for this session's trials.

        Read from `position_data.parquet` where it exists and derived from the trial
        blobs where it does not, by `io/load_results.load_position_data` -- the single
        seam that decides where the per-position facts come from, and which **filters
        back to the trial frame** so a metric cannot silently widen (section 28). This
        goes through `SessionResults`, so it is that one function and not a second
        reader.

        There is no `position_data_columns(mode)` to check against -- the field lists in
        `frames.py` are global rather than per-mode -- so a requested column is checked
        against the frame alone.
        """
        frame = self.results()["position_data"]
        if columns is None:
            return frame.copy()
        names = _as_names(columns, "columns")
        missing = [c for c in names if c not in frame.columns]
        if missing:
            raise KeyError(
                f"{self}: position_data has no column(s) "
                f"{', '.join(repr(c) for c in missing)}."
                + _closest(missing[0], frame.columns, "column"))
        return frame.loc[:, names].copy()

    def _check_trial_columns(self, frame: pd.DataFrame, names: Sequence[str]) -> None:
        missing = [c for c in names if c not in frame.columns]
        if not missing:
            return
        mode = self.protocol_mode()
        try:
            declared = set(trial_data_columns(mode)) if mode else set(mode_independent_columns())
        except ValueError:
            # An unrecognised mode: the file cannot be reasoned about, so diagnose
            # nothing rather than diagnose it wrongly (section 22 takes the same line).
            declared = set()

        stale = [c for c in missing if c in declared]
        unknown = [c for c in missing if c not in declared]
        parts = [f"{self}: trial_data has no column(s) "
                 f"{', '.join(repr(c) for c in missing)}."]
        if stale:
            parts.append(
                f" {', '.join(repr(c) for c in stale)} " +
                ("is" if len(stale) == 1 else "are") +
                " declared by the current schema, so this session was saved before "
                "it existed -- re-run trial classification for it.")
        if unknown:
            parts.append(_closest(unknown[0], frame.columns, "column"))
        raise KeyError("".join(parts))

    # -- the metrics -------------------------------------------------------------------

    def metrics(self, names: Iterable[str]) -> dict:
        """Compute the named metrics for this session. **Each in its own shape.**

            s.metrics(["decision_accuracy", "poke_durations"])
            # {"decision_accuracy": (num, den, value),   <- the saved scalar shape
            #  "poke_durations":    <DataFrame>}          <- a per-poke table

        **The caller does not pick a grain.** The registry already knows each metric's
        frame, grain and adapter, so one call returns a scalar, a Series and a DataFrame
        side by side rather than making you choose the right one of three functions and
        be silently wrong when you choose it badly.

        Keyed by the name asked for, so the result maps 1:1 onto the request.

        **Every name is resolved before any metric is computed**, so a typo in the
        fifth name raises before four metrics have been evaluated -- and raises rather
        than being skipped, unlike `visualization.prep._computed_metrics`, whose callers
        pass a fixed key list and draw whatever comes back. Naming a metric by hand is a
        typo when it misses, not a coverage choice.
        """
        if isinstance(names, str):
            raise TypeError("metrics() takes a sequence of names; for one metric use "
                            f"s.metric({names!r}).")
        wanted = _as_names(names, "names")
        specs = [_resolve_spec(n) for n in wanted]
        results = self.results()
        return {name: metric_value(spec, results) for name, spec in zip(wanted, specs)}

    def metric(self, name: str):
        """One metric's value. See `metrics`."""
        return self.metrics([name])[name]

    # -- looking at the files ----------------------------------------------------------

    def peek(self, *, table: Optional[str] = None, column: Optional[str] = None,
             rows: int = DEFAULT_ROWS, max_columns: Optional[int] = None) -> str:
        """The `parquet_peek` report for this session's directory, as text.

        Three narrowing views: the session's tables, then one line per column of one
        table, then one column with its values (section 29). Returns the text rather
        than printing it, as the tool does, so the caller decides where it goes.

        This is why that tool's library functions are path-based: the handle already
        holds the directory, so looking at the files costs no second resolution.
        """
        return peek(self.results_dir, table=table, column=column, rows=rows,
                    max_columns=max_columns)

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        if self.subjid is None and self.date is None:
            return f"Session({self.results_dir})"
        ses = f" ses-{self.ses:03d}" if isinstance(self.ses, int) else ""
        return f"Session(sub-{self.subjid:03d} {self.date}{ses})"


# --------------------------------------------------------------------------------------
# Resolution -- the expensive step, paid once
# --------------------------------------------------------------------------------------

def session(subjid, date=None, *, ses=None, index=None) -> Session:
    """Resolve one analysed session and return a handle on it.

        s = hypnose_behavior.session(57, 20260709)

    Resolves against **derivatives**, the analysed tree, because everything the handle
    offers is read from a saved analysis. Note that `index` therefore ranks the
    subject's *analysed* sessions, which section 32 measured to be a different session
    from rawdata's Nth on 7 of 8 subjects; `ses` is the selector that means the same
    thing in both trees.

    Raises immediately if the session does not resolve or holds no
    `saved_analysis_results`, rather than deferring to the first data access -- the
    failure belongs at the call the caller made.
    """
    ref = derivatives.find_session(subjid, date=date, ses=ses, index=index)
    results_dir = ref.path / "saved_analysis_results"
    if not results_dir.exists():
        raise FileNotFoundError(
            f"{ref.path.name} has no saved_analysis_results -- it is in the "
            f"derivatives tree but has not been analysed. Run trial classification "
            f"for it.")
    return Session(results_dir, ref=ref)


def sessions(subjids, *, dates=None, ses=None, index=None, date_range=None,
             ses_range=None, index_range=None) -> List[Session]:
    """Handles for every analysed session matching the selectors, across one or more
    subjects.

        for s in sessions([57, 40], date_range=(20260101, 20260731)):
            ...

    All six selectors (section 32), forwarded through `session_selectors` unchanged so
    that nothing here interprets them. **One tree walk per subject**, which is the case
    the resolve-once design is really for: a loop calling `session()` per session pays
    `find_session` per session, and on a cold mount that dominates everything else the
    loop does.

    > **Resolving per subject is what makes `index` mean something here.** Section 32
    > had to *refuse* `--index` in `batch_process.py`, because
    > `batch_run_all_metrics_with_merge` takes one `dates` list for all subjects, so an
    > index resolved across a cohort over-selects -- `--subjids 53 58 --index-range 1 5`
    > would give subject 58 any of *its* sessions whose date fell inside 53's first five.
    > That cannot happen here: each subject is resolved on its own, so `index` is "each
    > subject's Nth **analysed** session". It is still tree-relative -- section 32
    > measured `--index N` naming a different session in rawdata and derivatives on 7 of
    > 8 subjects -- so `ses` remains the selector that chains between trees.

    A matched session with no `saved_analysis_results` is skipped **with a warning
    naming it**, never silently: a selection that quietly returns fewer sessions than it
    matched is how a cohort figure comes to be drawn from half the data.
    """
    wanted = [subjids] if isinstance(subjids, (str, int)) else list(subjids)
    selectors = session_selectors(ses=ses, index=index, date_range=date_range,
                                  ses_range=ses_range, index_range=index_range)
    out, skipped, matched = [], [], 0
    for subjid in wanted:
        for ref in derivatives.find_sessions(subjid, date=dates, **selectors):
            matched += 1
            results_dir = ref.path / "saved_analysis_results"
            if results_dir.exists():
                out.append(Session(results_dir, ref=ref))
            else:
                skipped.append(ref.path.name)
    if skipped:
        warnings.warn(
            f"{len(skipped)} of {matched} matched session(s) have no "
            f"saved_analysis_results and were skipped: {', '.join(skipped)}.",
            RuntimeWarning, stacklevel=2)
    return out


# --------------------------------------------------------------------------------------
# Pooling a cohort -- identity stamped on, nothing rewritten
# --------------------------------------------------------------------------------------

def _stamped(frame: pd.DataFrame, handle: Session) -> pd.DataFrame:
    """`frame` with `subjid` / `date` / `ses` prepended, refusing to overwrite."""
    clash = [c for c in _IDENTITY if c in frame.columns]
    if clash:
        raise ValueError(
            f"{handle}: the frame already carries {', '.join(repr(c) for c in clash)}, "
            f"which pooling would overwrite. Rename before pooling.")
    out = frame.copy()
    for name, value in reversed(list(zip(_IDENTITY, (handle.subjid, handle.date, handle.ses)))):
        out.insert(0, name, value)
    return out


def _warn_ragged(frames: Sequence[pd.DataFrame], table: str) -> None:
    """Say so when the sessions do not agree on their columns.

    A ragged `pd.concat` fills the difference with nulls and reports nothing, so a
    cohort mixing protocol modes -- whose `trial_data` genuinely carries different
    column families (section 21) -- or mixing sessions analysed either side of a schema
    change silently gains columns that are null for most of it.
    """
    sets = [set(f.columns) for f in frames]
    partial = sorted(set().union(*sets) - set.intersection(*sets))
    if partial:
        shown = ", ".join(partial[:8]) + (f", ... (+{len(partial) - 8})" if len(partial) > 8 else "")
        warnings.warn(
            f"pooled {table}: {len(partial)} column(s) are on some of the {len(frames)} "
            f"sessions and not others, so they are null for the rest: {shown}. The "
            f"sessions differ in protocol mode, or were analysed either side of a schema "
            f"change.", RuntimeWarning, stacklevel=3)


def _warn_widened(frames: Sequence[pd.DataFrame], pool: pd.DataFrame, table: str) -> None:
    """Say so when `concat` widened a column's dtype.

    **Section 21's trap, arriving at a pooled concat.** A column absent from one session,
    or all-null on it and therefore typed `object`, widens the whole pooled column: on
    two fixture sessions `sequence_rewarded` goes `bool -> object`,
    `hidden_rule_location` `int64 -> object` and `hidden_rule_success_position`
    `float64 -> object`.

    **No value moves** -- measured cell by cell on those three, 0 of 339 / 273 / 273
    differ once compared dtype-blind, with identical null counts. But an `object` column
    of `True`/`False` is not a boolean mask, so a caller who indexes with it gets a
    different answer than they would from the session's own frame, and pandas says
    nothing. Naming the columns is what turns that from silent into a choice.
    """
    # The test is `sources != {target}`, not `target not in sources`. The narrower form
    # was tried and under-reports: `hidden_rule_location` is `object` on a session with
    # no hidden rule (all-null, hence untyped -- section 21) and `int64` on one with,
    # so `object` is "an input dtype" and the pooled `object` looked unchanged -- while
    # a reader of the second session's rows plainly sees `int64 -> object`. A guard
    # narrower than the hazard reports a pass for something still surprising; measured,
    # it named 1 of the 3 columns that actually moved.
    widened = []
    for col in pool.columns:
        if col in _IDENTITY:
            continue
        sources = {str(f[col].dtype) for f in frames if col in f.columns}
        target = str(pool[col].dtype)
        if sources != {target}:
            widened.append(f"{col} ({'/'.join(sorted(sources))} -> {target})")
    if widened:
        shown = ", ".join(widened[:6]) + (f", ... (+{len(widened) - 6})" if len(widened) > 6 else "")
        warnings.warn(
            f"pooled {table}: {len(widened)} column(s) changed dtype at the concat, "
            f"because the sessions disagree on type or the column is missing from some "
            f"of them: {shown}. No value moved -- but an object column of True/False is "
            f"not a boolean mask, so cast before using one as one.",
            RuntimeWarning, stacklevel=3)


def pooled(handles: Iterable[Session], table: str = "trial_data",
           columns: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """One frame over many sessions, with `subjid` / `date` / `ses` prepended.

        hs = sessions([57, 40], date_range=(20260101, 20260731))
        pooled(hs, "trial_data", columns=["response_time_ms"])

    **Identity is stamped; no value is rewritten.** Every value is the value that
    session's own accessor returns -- measured, 0 cells of 612 differ. What *can* change
    is a **dtype**: `pd.concat` widens a column the sessions disagree on or that some of
    them lack, so `bool` can arrive as `object`. `_warn_widened` names those columns
    rather than leaving it to be discovered. In particular `global_trial_id` is left
    exactly as saved, which means it is **not unique in a pooled frame** -- measured on
    two sessions, 612 rows carry 339 distinct ids, so 273 collide. That is section 28's
    finding, that a pooled frame has no key separating two sessions' trials, and the
    remedy is the key `(subjid, date, global_trial_id)` rather than a synthetic id: a
    second identity for one trial is the thing section 13 exists to prevent, and it
    would exist only in pooled frames and never in a saved file.

    > **Do not group a rate metric off this frame by averaging per-session values.** A
    > rate is not a per-trial quantity (section 1): pool the numerator and denominator
    > contributions with `metrics.common.reduce_rate`. That is the defect two rolling
    > accuracies disagreed over for years.

    `table` is `trial_data` or `position_data`. The per-grain metric tables are refused
    on purpose: they are an export and a record, never an input (section 25).

    Raises on an empty selection rather than returning an empty frame -- a cohort call
    that silently pools zero sessions looks exactly like one that succeeded.
    """
    handles = list(handles)
    if not handles:
        raise ValueError("pooled() needs at least one session; the selection matched "
                         "none. Check the selectors, or the subject's analysed sessions "
                         "with sessions(subjid).")
    if table in _EXPORT_ONLY:
        raise ValueError(
            f"{table!r} is an export and the record of an analysis run, never an input "
            f"(docs/DECISIONS.md section 25). Compute it: pooled_metrics(handles, [...]) "
            f"or s.metrics([...]). Poolable tables: {', '.join(_POOLABLE)}.")
    if table not in _POOLABLE:
        raise ValueError(f"pooled() reads {' or '.join(_POOLABLE)}, not {table!r}.")

    frames = [_stamped(getattr(h, table)(columns=columns), h) for h in handles]
    _warn_ragged(frames, table)
    pool = pd.concat(frames, ignore_index=True)
    _warn_widened(frames, pool, table)
    return pool


def pooled_metrics(handles: Iterable[Session], names: Iterable[str]) -> pd.DataFrame:
    """One row per session, one column per metric, plus `subjid` / `date` / `ses`.

        pooled_metrics(sessions(57), ["decision_accuracy", "global_FA_rate"])

    The shape most cohort figures want: a metric over sessions. Computed through the
    registry for every session, so it is section 5 compliant for the same reason
    `Session.metrics` is -- there is no path here that reads `metrics_*.json`.

    **Each cell holds exactly what `Session.metrics` returned, unflattened.** That is
    deliberate and it is the part to read before using the frame:

    - A **rate** arrives as `(numerator, denominator, value)`, not as a number. Taking
      element 2 per session and averaging across sessions is wrong -- section 1's rule,
      that a rate is not a per-trial quantity and must be reduced from its
      contributions (`metrics.common.reduce_rate`). Nothing here flattens for you,
      because a flattening rule quietly applied is how that defect returns.
    - A **table-shaped** metric (`poke_durations`, `hr_abort_poke_gap`) arrives as a
      DataFrame in a cell. That is legal and rarely what you want; `pooled(handles,
      "position_data")` is usually the frame you were reaching for.

    Sessions are evaluated in the order given, and every name is validated against the
    registry once, before any session is loaded.
    """
    handles = list(handles)
    if not handles:
        raise ValueError("pooled_metrics() needs at least one session; the selection "
                         "matched none.")
    wanted = _as_names(names, "names")
    for name in wanted:            # fail on a typo before loading anything
        _resolve_spec(name)

    rows = []
    for handle in handles:
        row = dict(zip(_IDENTITY, (handle.subjid, handle.date, handle.ses)))
        row.update(handle.metrics(wanted))
        rows.append(row)
    return pd.DataFrame(rows, columns=list(_IDENTITY) + wanted)
