# Defers evaluation of PEP-604 annotations (`X | None`), keeping this module
# importable on Python 3.9 for repos pinned there (hypnose-eeg-preprocessing).
from __future__ import annotations

"""What a saved trial record *is* -- the single declaration of `trial_data`'s schema.

`trial_data` is written by `io/save_results.py`, read by `io/load_results.py`, and
built by `trial_classification/classify_trials.py`. Before restructure_2 Phase 7b
nothing declared its shape: `save_results` named 27 columns while the table carried
60, the other 33 existing only because some line assigned them. This module is that
declaration.

**A leaf.** It imports nothing from the package -- the standard library only. Both
`io/__init__.py` and `metric_analysis/__init__.py` are docstring-only, so
`trial_classification -> io.protocol_schema` triggers no package-level side effects and closes
no cycle. Keep it that way, for the same reason `frames.py` must stay a leaf
(`docs/DECISIONS.md` section 3): every layer here already imports it.

### The protocol mode decides the column set

A session's columns depend on which branch of `classify_trials` ran, and the three
branches write different families. Declaring one uniform record for all of them would
put ~26 all-NaN columns on the average session; declaring one per mode reproduces
what is on disk today to within 7 columns. So the mode is part of the schema, and
`resolve_mode` is where it is decided -- once, from the two flags, rather than
re-derived at each site that needs to know.
"""

from dataclasses import dataclass, fields

__all__ = [
    "ConflictingProtocolError",
    "STANDARD",
    "SINGLE_REWARD",
    "ODOUR_DISCRIMINATION",
    "MODES",
    "resolve_mode",
    "StandardTrialRecord",
    "SingleRewardTrialRecord",
    "OdourDiscriminationTrialRecord",
    "record_class_for",
    "ABORT_COLUMNS",
    "RESPONSE_TIME_COLUMNS",
    "ASSEMBLED_COLUMNS",
    "POSITION_BLOB_COLUMNS",
    "trial_data_columns",
    "mode_independent_columns",
]


# The three per-position JSON blobs: declared fields, built by `classify_trials`, and
# **never written to `trial_data`** since Phase 7b.4b. `save_results` expands them into
# `position_data.parquet` -- one row per `trial x position`, with typed columns and
# section 2's provenance flags -- and then drops them from the frame it saves.
#
# They stay *fields* because the expansion reads them off the in-memory trial frame, and
# `@dataclass(slots=True)` would refuse the assignment in `classify_trials` if they were
# removed. So this constant is the single statement of "in memory, not on disk", read by
# `columns()` (hence by the conform and the loader's schema check) and by `save_results`.
POSITION_BLOB_COLUMNS = ("position_valve_times", "position_poke_times", "presentations")


class ConflictingProtocolError(Exception):
    """A session's two protocol flags disagree about which schema it follows.

    Named rather than a bare `ValueError` so it is greppable in a batch log and
    separately catchable: `batch_analyze_sessions` wraps each session in
    `except Exception`, where a `ValueError` is indistinguishable from pandas
    complaining about an index.
    """


# The three protocol modes. A session is exactly one of them, and the value is written
# to `manifest.json` so `io/load_results.py` can check the file against the right field
# set rather than guessing from the columns it happens to find.
STANDARD = "standard"
SINGLE_REWARD = "single_reward"
ODOUR_DISCRIMINATION = "odour_discrimination"

MODES = (STANDARD, SINGLE_REWARD, ODOUR_DISCRIMINATION)


def resolve_mode(*, is_odour_discrimination: bool, is_single_reward: bool) -> str:
    """Which of `MODES` this run follows. Raises `ConflictingProtocolError` on the impossible one.

    The two flags come from **independent sources** -- `is_odour_discrimination` from
    the protocol name in the detected stage, `is_single_reward` from the schema's
    `isSingleRewardProtocol` flag (`trial_classification/params.py`). Nothing in the
    code makes them exclusive; the *experiment* does, and by construction: odour
    discrimination presents a sequence of length 1, while the single-reward protocol
    needs at least 2 positions for a sequence to be rewarded-or-not at its end. There
    is no session that is both, and there cannot be one.

    **So this raises rather than warns.** Both flags true means the session was run
    with a structurally impossible configuration, which is a problem at the rig, not
    in the analysis -- and it needs fixing before any number off that session is used.
    Continuing would write a `trial_data` whose schema is undefined: today's control
    flow reaches the odour-discrimination branch first and `continue`s past the
    false-response scoring, so the four determinacy columns would be *silently absent*
    from a file that still looked complete.

    Raising is also the safer failure in bulk. `batch_analyze_sessions` catches per
    session, prints which subject and date failed, and carries on -- so one broken
    schema names itself and skips, writing no derivative, while the rest of the batch
    completes. A warning would do the opposite: write the malformed file and bury the
    notice in thousands of lines of batch output.

    This is **not** Phase 9. Phase 9 validates data *values*; this is a contradiction
    between two schema-derived flags that leaves the output schema undecidable, and it
    exists only because Phase 7b made the schema mode-dependent.
    """
    if is_odour_discrimination and is_single_reward:
        raise ConflictingProtocolError(
            "session is flagged as BOTH odour-discrimination and single-reward, which is "
            "impossible by design: odour discrimination presents a sequence of length 1 "
            "and the single-reward protocol requires at least 2 positions. The stage's "
            "protocol name contains 'odourdiscrimination' and the schema sets "
            "'isSingleRewardProtocol'. This is a structural fault in the session as it "
            "was run -- fix the task schema or the stage name before analysing it; the "
            "saved schema is undefined while both hold."
        )
    if is_odour_discrimination:
        return ODOUR_DISCRIMINATION
    if is_single_reward:
        return SINGLE_REWARD
    return STANDARD


# --------------------------------------------------------------------------------------
# The trial record
# --------------------------------------------------------------------------------------
#
# `classify_trials` used to build a free-form dict per trial, so a column existed only
# because some line assigned it: `save_results` named 27 columns while the table carried
# 60, and `trial_dict['fr_laency_ms'] = ...` was valid Python that silently invented a
# column of NaNs. These classes are the declaration. `slots=True` makes that typo an
# AttributeError at the assignment site.
#
# Fields are declared in the order they are written, so `to_row()` yields a stable column
# order rather than one that depends on which branch a session's first trial took.
#
# **Every field defaults to None**, which is what the old dict's *absence* became once
# pandas built a frame from it: a numeric column reads back as NaN either way, an object
# column as an empty cell. The difference is that the columns now exist uniformly within
# a mode, instead of depending on whether any trial in the session happened to take the
# branch that wrote them.


@dataclass(slots=True)
class _TrialRecordBase:
    """Fields every protocol writes. Never instantiated directly -- use `record_class_for`.

    The first ten come from `detect_trials._record_detected_trial`; the rest are written
    by `classify_trials` on every path.

    **`run_id` is deliberately absent**, though it is a `trial_data` column. It is assigned
    downstream by `merge._with_run_id`, which copies any *existing* `run_id` to a new
    `run_id_original` column before overwriting it. Declaring it here would emit it on
    every row, so every merged session would silently gain an all-null `run_id_original`
    that exists on no session today. `is_aborted` and `global_trial_id` are absent for the
    same reason -- `save_results` derives them. See `ASSEMBLED_COLUMNS`.
    """

    # --- from detect_trials ---
    initiation_sequence_time: object = None
    sequence_start: object = None
    sequence_end: object = None
    continuous_poke_time_ms: float | None = None
    trial_id: int | None = None
    attempt_number: int | None = None
    timestamp: object = None
    required_min_sampling_time_ms: float | None = None
    odor_name: str | None = None
    # Written only when a trial came from the pending-attempt fallback, so it is null on
    # most sessions -- present on 2 of the 9 regression fixtures.
    fallback_reason: str | None = None

    # --- the presented sequence ---
    odor_sequence: object = None
    num_odors: int | None = None
    last_odor: str | None = None
    sequence_name: str | None = None
    minimum_sampling_time_ms_by_odor: object = None

    # --- per-position blobs: IN MEMORY ONLY since Phase 7b.4b ---
    # `classify_trials` still builds them and `save_results` still expands them into
    # `position_data.parquet` -- but they are no longer *written* to `trial_data`. See
    # `POSITION_BLOB_COLUMNS`; they are excluded from `columns()`, so the loader's schema
    # check does not demand a column nothing saves any more.
    position_valve_times: object = None
    position_poke_times: object = None
    presentations: object = None
    last_event_index: int | None = None
    sequence_start_corrected: object = None

    # --- hidden rule ---
    hidden_rule_location: int | None = None
    hidden_rule_locations: object = None
    hidden_rule_positions: object = None
    enough_odors_for_hr: bool | None = None
    hit_hidden_rule: bool | None = None
    hidden_rule_hit_indices: object = None
    hidden_rule_hit_positions: object = None
    hidden_rule_success: bool | None = None
    hidden_rule_success_position: int | None = None

    # --- outcome: reward delivery ---
    await_reward_time: object = None
    first_supply_time: object = None
    first_supply_port: object = None
    first_supply_odor_identity: str | None = None
    supply1_count: int | None = None
    supply2_count: int | None = None
    total_supply_count: int | None = None

    # --- outcome: reward-port pokes. Written by both scorers, hence common. ---
    first_reward_poke_time: object = None
    first_reward_poke_port: object = None
    first_reward_poke_odor_identity: str | None = None
    port1_pokes_count: int | None = None
    port2_pokes_count: int | None = None
    total_reward_pokes: int | None = None

    def to_row(self) -> dict:
        """This record as a mapping, for `pd.DataFrame`. Every declared field, always."""
        return {f.name: getattr(self, f.name) for f in fields(self)}

    @classmethod
    def columns(cls) -> tuple:
        """The column names this record **writes**, in declaration order.

        `POSITION_BLOB_COLUMNS` are declared fields but not written columns: since Phase
        7b.4b they exist only in memory, long enough for `save_results` to expand them
        into `position_data.parquet`. Excluding them here is what keeps the declaration
        honest in both directions -- `save_results` conforms to it, and the loader checks
        a saved file against it, so leaving them in would make every newly written file
        report itself as missing three columns.
        """
        return tuple(f.name for f in fields(cls) if f.name not in POSITION_BLOB_COLUMNS)


@dataclass(slots=True)
class StandardTrialRecord(_TrialRecordBase):
    """The default protocol: a sequence of odors, rewarded at its final position."""

    # Written by `_score_standard_outcome` only, which the odour-discrimination path never
    # reaches -- it has no fixed response deadline.
    poke_window_end: object = None


@dataclass(slots=True)
class SingleRewardTrialRecord(StandardTrialRecord):
    """Single-reward: only some candidate sequences are rewarded at their final position.

    Extends `StandardTrialRecord` rather than the base **because a single-reward session
    uses both scorers**: a rewarded sequence goes through `_score_standard_outcome` and so
    writes `poke_window_end`, while a non-rewarded one goes through `_score_false_response`.
    Measured: sub-057's fixture carries `poke_window_end`.

    The twelve fields are one group even though the first four are written before the
    false-response scoring and the other eight inside it. That reproduces `save_results`'
    existing behaviour, which added all twelve together as soon as any one was present.
    """

    # Written for every trial of a single-reward session, before scoring.
    sequence_rewarded: bool | None = None
    reward_determinacy: str | None = None
    determinacy_position: int | None = None
    determined_final_odor: str | None = None

    # Written by `_score_false_response`, i.e. only on a completed non-rewarded sequence.
    false_response: bool | None = None
    fr_label: str | None = None
    fr_time: object = None
    fr_port: object = None
    fr_odor_identity: str | None = None
    fr_window_end: object = None
    fr_window_latency_ms: float | None = None
    fr_response_time_ms: float | None = None


@dataclass(slots=True)
class OdourDiscriminationTrialRecord(_TrialRecordBase):
    """Odour discrimination: a single-position sequence, scored over an open reward window.

    Does **not** extend `StandardTrialRecord`: `poke_window_end` belongs to the standard
    scorer's fixed response deadline, which this protocol does not have, and both fixture
    sessions of this mode lack the column.
    """

    odourdiscrimination_mode: bool | None = None
    last_valve_start: object = None
    next_initiation_time: object = None
    next_cue_poke_start: object = None
    reward_window_end: object = None
    # Written when AwaitReward never lands in the window. Never fires on either fixture
    # session, so the column is new -- and that absence is exactly what a declaration is
    # meant to make visible.
    abort_reason: str | None = None


_RECORD_CLASSES = {
    STANDARD: StandardTrialRecord,
    SINGLE_REWARD: SingleRewardTrialRecord,
    ODOUR_DISCRIMINATION: OdourDiscriminationTrialRecord,
}


def record_class_for(mode: str):
    """The trial-record class for one of `MODES`."""
    try:
        return _RECORD_CLASSES[mode]
    except KeyError:
        raise ValueError(f"unknown protocol mode {mode!r}; expected one of {MODES}") from None


# --------------------------------------------------------------------------------------
# The rest of `trial_data`: columns the record does not write
# --------------------------------------------------------------------------------------
#
# `trial_data` has three producers. `classify_trials` builds the record above;
# `save_results` merges in the two frames below and derives `is_aborted` /
# `global_trial_id`; `merge` assigns `run_id`. All three are part of the saved schema, so
# a reader checking a file against "what a trial is" has to know about all three --
# `fa_window_latency_ms` and `completed_window_latency_ms` are merged columns, and they
# are precisely the ones Phase 6's rename moved.

# Merged from `aborted_sequences_detailed` by `save_results`.
ABORT_COLUMNS = (
    "last_odor_position",
    "last_odor_name",
    "last_odor_valve_duration_ms",
    "last_odor_poke_time_ms",
    "last_required_min_sampling_time_ms",
    "abortion_type",
    "abortion_time",
    "fa_label",
    "fa_time",
    "fa_window_latency_ms",
    "fa_port",
    "fa_response_time_ms",
)

# Merged from `completed_sequences_with_response_times` by `save_results`.
RESPONSE_TIME_COLUMNS = (
    "response_time_ms",
    "response_time_category",
    "completed_window_latency_ms",
)

# Assigned during assembly rather than by any trial: `run_id` by `merge._with_run_id`,
# the other two by `save_results`.
ASSEMBLED_COLUMNS = ("run_id", "is_aborted", "global_trial_id")


def mode_independent_columns() -> tuple:
    """Columns a `trial_data` carries whatever protocol wrote it.

    The base record's fields are common to all three modes by construction, and the merged
    and assembled columns do not depend on the mode at all. So this is the largest set that
    can be checked against a file whose mode is **unknown** -- every file written before
    Phase 7b -- with no risk of a false alarm from guessing wrong.

    It is not a lesser check for the case that matters. `fa_window_latency_ms`,
    `fa_response_time_ms` and `completed_window_latency_ms` are merged columns, hence
    mode-independent, and they are exactly the ones Phase 6's rename moved: measured on the
    server's `sub-040 20251124`, this reports all three, while comparing against the record's
    own fields alone reports only `fallback_reason` -- a column nothing reads.
    """
    return (_TrialRecordBase.columns()
            + ABORT_COLUMNS + RESPONSE_TIME_COLUMNS + ASSEMBLED_COLUMNS)


def trial_data_columns(mode: str) -> tuple:
    """Every column a `trial_data` written in `mode` should carry.

    The single declaration `io/load_results.py` checks a saved file against. Comparing
    field sets answers the question a commit stamp cannot: a git SHA says *something*
    changed between the file and now, not whether *this file* is affected -- a one-line
    plotter fix and a trial-classification restructure look identical to it.
    """
    return (record_class_for(mode).columns()
            + ABORT_COLUMNS + RESPONSE_TIME_COLUMNS + ASSEMBLED_COLUMNS)


# Record fields holding a timestamp. Declared because a column of *nothing but* None
# carries no type, and pandas then infers `object` -- see `_as_declared_datetime` in
# `trial_classification/classify_trials.py` for the concat that makes that visible.
#
# Measured from the reference tree's `trial_data.parquet` rather than assumed. The abort
# frame's `abortion_time` / `fa_time` are datetimes too, but they are merged in by
# `save_results` from a frame this declaration does not build, and are left alone.
DATETIME_FIELDS = frozenset({
    # base
    "initiation_sequence_time", "sequence_start", "sequence_end", "timestamp",
    "sequence_start_corrected", "await_reward_time", "first_supply_time",
    "first_reward_poke_time",
    # standard / single-reward
    "poke_window_end",
    # single-reward
    "fr_time", "fr_window_end",
    # odour discrimination
    "last_valve_start", "next_initiation_time", "next_cue_poke_start", "reward_window_end",
})
