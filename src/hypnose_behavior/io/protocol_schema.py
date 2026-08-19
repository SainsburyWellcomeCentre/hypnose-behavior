from __future__ import annotations

"""What a saved trial record *is* -- the single declaration of `trial_data`'s schema.

`trial_data` is built by `trial_classification/classify_trials.py`, written by
`io/save_results.py` and read by `io/load_results.py`; all three conform to the
declarations here. A session's columns depend on which branch of `classify_trials`
ran, so the schema is per protocol mode, and `resolve_mode` decides that mode once
rather than at each site that needs it.

**This module imports nothing from the package -- the standard library only.**
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


# The three per-position JSON blobs: declared fields that are **never written to
# `trial_data`**. `save_results` expands them into `position_data.parquet` and then drops
# them from the frame it saves.
#
# They must stay declared *fields* -- the expansion reads them off the in-memory trial
# frame, and `@dataclass(slots=True)` would refuse `classify_trials`' assignment otherwise.
# This constant is the single statement of "in memory, not on disk", read by `columns()`
# (hence by the conform and the loader's schema check) and by `save_results`.
POSITION_BLOB_COLUMNS = ("position_valve_times", "position_poke_times", "presentations")


class ConflictingProtocolError(Exception):
    """A session's two protocol flags disagree about which schema it follows.

    **Keep it a named class, not a bare `ValueError`:** `batch_analyze_sessions` wraps
    each session in `except Exception`, where a `ValueError` is indistinguishable from
    pandas complaining about an index.
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

    The two flags come from independent sources -- `is_odour_discrimination` from the
    detected stage's protocol name, `is_single_reward` from the schema's
    `isSingleRewardProtocol`. Nothing in the code makes them exclusive; the experiment
    does, by construction, so both being true is a structural fault in the session as
    it was run.

    **Raise, never warn.** Raising makes the broken session name itself, write no derivative, and let
    the batch finish. 
    """
    if is_odour_discrimination and is_single_reward:
        raise ConflictingProtocolError(
            "session is flagged as BOTH odour-discrimination and single-reward,"
            "impossible by design: odour discrimination presents a sequence of length 1 "
            "and the single-reward protocol requires at least 2 positions."
            "'This is a structural fault in the session as it "
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
# These classes declare every column `classify_trials` writes. **Keep `slots=True`**: it
# is what makes `rec.fr_laency_ms = ...` an AttributeError at the assignment site rather
# than a silently invented column of NaNs.
#
# Fields are declared in the order they are written, so `to_row()` yields a stable column
# order rather than one that depends on which branch a session's first trial took.
#
# Every field defaults to None, so a mode's columns exist uniformly across the session
# instead of depending on whether any trial took the branch that writes them.


@dataclass(slots=True)
class _TrialRecordBase:
    """Fields every protocol writes. Never instantiated directly -- use `record_class_for`.

    The first ten come from `detect_trials._record_detected_trial`; the rest are written
    by `classify_trials` on every path.

    **Do not declare `run_id`, `is_aborted` or `global_trial_id` here**, though all three
    are `trial_data` columns -- they are assigned during assembly, see `ASSEMBLED_COLUMNS`.
    Declaring `run_id` would emit it on every row, and `merge._with_run_id` copies any
    *existing* `run_id` to `run_id_original` before overwriting, so every merged session
    would silently gain an all-null column that exists on no session today.
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
    # most sessions.
    fallback_reason: str | None = None

    # --- the presented sequence ---
    odor_sequence: object = None
    num_odors: int | None = None
    last_odor: str | None = None
    sequence_name: str | None = None
    minimum_sampling_time_ms_by_odor: object = None

    # --- per-position blobs: IN MEMORY ONLY ---
    # Built by `classify_trials` and expanded into `position_data.parquet` by
    # `save_results`, never written to `trial_data`. See `POSITION_BLOB_COLUMNS`.
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

        `POSITION_BLOB_COLUMNS` are declared fields but not written columns, so they are
        excluded here. `save_results` conforms to this list and the loader checks a saved
        file against it, so including them would make every newly written file report
        itself as missing three columns.
        """
        return tuple(f.name for f in fields(cls) if f.name not in POSITION_BLOB_COLUMNS)


@dataclass(slots=True)
class StandardTrialRecord(_TrialRecordBase):
    """The default protocol: a sequence of odors, rewarded at its final position."""

    # Written by `_score_standard_outcome` only, which the odour-discrimination path
    # never reaches -- it has no fixed response deadline.
    poke_window_end: object = None


@dataclass(slots=True)
class SingleRewardTrialRecord(StandardTrialRecord):
    """Single-reward: only some candidate sequences are rewarded at their final position.

    **Extends `StandardTrialRecord`, not the base, because a single-reward session uses
    both scorers:** a rewarded sequence goes through `_score_standard_outcome` and so
    writes `poke_window_end`, while a non-rewarded one goes through `_score_false_response`.

    The twelve fields are one group even though the first four are written before the
    false-response scoring and the other eight inside it -- all twelve appear together
    on any session that writes any of them.
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

    **Does not extend `StandardTrialRecord`**, deliberately: `poke_window_end` belongs to
    the standard scorer's fixed response deadline, which this protocol has no equivalent
    of, and `slots` then makes it physically impossible to set on this record.
    """

    odourdiscrimination_mode: bool | None = None
    last_valve_start: object = None
    next_initiation_time: object = None
    next_cue_poke_start: object = None
    reward_window_end: object = None
    # Written when AwaitReward never lands in the window, so it is null on most
    # sessions.
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
# a reader checking a file against "what a trial is" must know about all three.

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

    The largest set checkable against a file whose mode is unknown, with no risk of a
    false alarm: the base record's fields are common to all three modes, and the merged
    and assembled columns do not depend on mode at all.

    **Build it from `columns()`, not from `fields(_TrialRecordBase)`** -- the latter
    includes `POSITION_BLOB_COLUMNS`, and a file written by current code would then report
    itself missing three columns it is not supposed to have. See DECISIONS.md section 22.
    """
    return (_TrialRecordBase.columns()
            + ABORT_COLUMNS + RESPONSE_TIME_COLUMNS + ASSEMBLED_COLUMNS)


def trial_data_columns(mode: str) -> tuple:
    """Every column a `trial_data` written in `mode` should carry.

    The single declaration `io/load_results.py` checks a saved file against, and the one
    `accessors` uses to word a failed column lookup.
    """
    return (record_class_for(mode).columns()
            + ABORT_COLUMNS + RESPONSE_TIME_COLUMNS + ASSEMBLED_COLUMNS)


# Record fields holding a timestamp. Declared because a column of *nothing but* None
# carries no type, so pandas infers `object` and a multi-run merge then turns the whole
# merged column `object` -- see `_as_declared_datetime` in
# `trial_classification/classify_trials.py`, and DECISIONS.md section 21.
#
# The abort frame's `abortion_time` / `fa_time` are datetimes too, but `save_results`
# merges them in from a frame this declaration does not build, so they are left alone.
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
