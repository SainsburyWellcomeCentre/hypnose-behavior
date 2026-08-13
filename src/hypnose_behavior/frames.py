# Defers evaluation of PEP-604 annotations (`X | None`), keeping this module
# importable on Python 3.9 for repos pinned there (hypnose-eeg-preprocessing).
from __future__ import annotations

"""The frames metric cores consume, and the helpers that build them.

Phase 4a (restructure_2) gives every metric a pure ``f(frame) -> value`` core.
There are two frames:

``trials``
    ``trial_data`` -- one row per trial. Unchanged.

``position_data``
    long form, one row per ``trial x position``. Derived at *load* time by
    expanding the per-trial JSON blobs ``position_poke_times`` /
    ``presentations`` / ``position_valve_times``, so metrics never parse a blob
    and carry no legacy-session branch. Pairs with Phase 7b's ``position_data``
    side-table, which will make the expansion a read rather than a derivation.

This module also owns the two "how far did the sequence get" helpers that the
Phase 4a audit's Q5 resolution mandates. See ``sequence_depth`` for the one
place where today's behaviour and the eventual definition differ, and why.
"""

import json
import re
import warnings
from typing import Optional

import pandas as pd

__all__ = [
    "parse_json_column",
    "odor_letter",
    "odor_sequence_tokens",
    "presented_positions",
    "sequence_depth",
    "sampled_positions",
    "reached_counts",
    "build_position_data",
]


def parse_json_column(val):
    """Best-effort decode of a JSON-blob cell.

    Cells arrive as ``str`` from CSV and as already-decoded objects from
    parquet, so non-strings pass straight through.
    """
    if isinstance(val, str):
        try:
            val_fixed = val.replace('""', '"')
            return json.loads(val_fixed)
        except Exception:
            return {} if val.strip().startswith("{") else []
    return val


def odor_letter(value) -> str:
    """Normalise a stored odor token to a bare upper-case letter.

    ``'OdorC'`` / ``'"OdorC"'`` / ``'["OdorC'`` / ``'odor c'`` / ``'C'`` -> ``'C'``.
    ``odor_sequence`` is sometimes stored as a JSON-encoded string such as
    ``'["OdorE", "OdorG"]'``, hence the bracket/quote stripping before the
    ``Odor`` prefix check.

    Written out four times in ``visualization/`` before Phase 4a
    (``visualization_utils`` twice, ``pred_seq_utils._canonical_odor``,
    ``sing_rew_movement._odor_letter``). It knows what the data *is*, so it
    belongs here rather than in ``hypnose_helpers``.
    """
    s = str(value).strip().strip('[]"\'').strip()
    if s.lower().startswith("odor"):
        s = s[4:].strip()
    return s.upper()


def odor_sequence_tokens(seq) -> list:
    """Split a trial's ``odor_sequence`` cell into raw odor tokens.

    The column is a JSON string on some sessions, a delimited string on others,
    and a list/ndarray when it round-trips through parquet.
    """
    if seq is None:
        return []
    if isinstance(seq, float) and pd.isna(seq):
        return []
    if isinstance(seq, str):
        if not seq.strip():
            return []
        if any(c in seq for c in ",; |"):
            return [t for t in re.split(r"[\s,;|]+", seq) if t]
        return [seq]
    try:
        return list(seq)
    except TypeError:
        return []


def _is_aborted(trial) -> bool:
    # `== True` rather than `bool(...)`, to match the `df["is_aborted"] == True`
    # mask every caller builds: a missing column yields False, and a stringy
    # "True" is not silently promoted.
    return trial.get("is_aborted") == True  # noqa: E712


def _as_int(val) -> Optional[int]:
    try:
        return int(val) if pd.notnull(val) else None
    except (TypeError, ValueError):
        return None


def _last_position(trial) -> Optional[int]:
    """Last counted **position** (1-based).

    ``last_odor_position`` is already a position. ``last_event_index`` is a
    0-based index into ``presentations``, so it needs ``+1`` to be the same
    unit -- the two were previously returned interchangeably, which was a latent
    off-by-one that never fired because ``last_odor_position`` is a column on
    every session written by this pipeline.
    """
    if "last_odor_position" in trial:
        return _as_int(trial.get("last_odor_position"))
    idx = _as_int(trial.get("last_event_index"))
    return None if idx is None else idx + 1


def _presented_max(trial) -> Optional[int]:
    """Deepest position the **rig presented** -- every position whose valve opened.

    ``position_poke_times`` is read **first** even though ``presentations`` is
    the more natural name for "what was presented". On sessions written by this
    pipeline the two carry the same position set -- verified, their maxima agree
    on 1,730 of 1,730 trials -- so the order cannot matter there. It matters for
    sessions saved earlier, where the blobs did *not* agree (section 2): reading
    ``position_poke_times`` first reproduces exactly what this function returned
    before, so a re-analysis of old derivatives cannot move.
    """
    for col in ("position_poke_times", "presentations"):
        entries = _entries_by_position(trial.get(col))
        if entries:
            return max(entries)
    return _last_position(trial)


def _sampled_max(trial) -> Optional[int]:
    """Deepest position the **animal sampled** (``poke_source == 'poke'``), or None.

    Returns None when no entry carries the marker -- which includes every
    session saved before Phase 6b, none of which has ``poke_source`` at all.
    Callers must fall back rather than read "no marker" as "all real pokes"
    (``DECISIONS.md`` sections 2 and 10).
    """
    poked = [pos for pos, entry in _entries_by_position(trial.get("presentations")).items()
             if entry.get("poke_source") == "poke"]
    if not poked:
        poked = [pos for pos, entry in _entries_by_position(trial.get("position_poke_times")).items()
                 if entry.get("poke_source") == "poke"]
    return max(poked) if poked else None


def _max_poke_time_position(trial) -> Optional[int]:
    """Deepest position with a poke entry, or None.

    All-or-nothing on the key cast, matching the walk this replaced: one
    unparseable key discards the whole blob and defers to ``_last_position``.
    """
    ppt = parse_json_column(trial.get("position_poke_times", {}))
    if isinstance(ppt, dict) and ppt:
        try:
            return max(int(k) for k in ppt.keys())
        except Exception:
            return None
    return None


def sequence_depth(trial) -> Optional[int]:
    """How far the sequence got, as a contiguous depth ``1..n``. Never filtered.

    This is the denominator of every per-position rate ("of the trials that
    reached position *p*, how many aborted there"), so it must be **monotonic**:
    a trial that reached position 5 reached positions 1-4 as well. That is why
    it returns a depth to be filled contiguously rather than a set of positions
    -- see ``sampled_positions`` for the filterable, possibly-gappy counterpart,
    which answers a different question and must not be substituted here.

    The depth is what the trial is **credited** with, and that is
    ``max(presentations)`` filtered by ``poke_source`` only where the filter is
    required:

    - **completed** -- every presented position counts. The rig advanced through
      all of them, including a final one our DIPort0 reconstruction scores as
      unpoked, so the filter must *not* be applied.
    - **aborted** -- the sequence stops at the last real poke, so the filter
      *is* applied. Falls back to ``last_odor_position``, which the abort
      pipeline derives independently and which agrees with the filtered maximum
      on **486 of 486** trials where both are defined.

    That split is ``DECISIONS.md`` section 10's rule, and it is why neither
    single-meaning form may be substituted. Measured on the 9 fixture sessions
    (1,731 trials):

    ==========================================  ==========================
    ``max(presentations)`` unfiltered            moves **84** trials
    this rule                                    the current values
    ``max(poke_source == 'poke')`` everywhere    moves **32** trials
    ==========================================  ==========================

    The first re-credits the trailing positions the abort trim removed; the last
    drops a completed trial's trailing presented position.

    It must also stay **monotonic** -- a trial that reached position 5 reached
    1-4 as well -- which is why it returns a depth to be filled contiguously
    rather than a set. See ``sampled_positions`` for the filterable,
    possibly-gappy counterpart, which answers a different question and must not
    be substituted here.

    Warns when neither the filtered maximum nor ``last_odor_position`` resolves
    an aborted trial: that is a trial with no sampled position at all, and it
    drops out of every per-position denominator, so it is worth investigating
    rather than silently counting as nothing.
    """
    if not _is_aborted(trial):
        return _presented_max(trial)

    sampled = _sampled_max(trial)
    if sampled is not None:
        return sampled
    # Sessions saved before `poke_source` existed carry no marker to filter on.
    fallback = _last_position(trial)
    if fallback is None:
        warnings.warn(
            "aborted trial has no sampled position: neither a `poke_source == 'poke'` "
            "entry nor `last_odor_position` resolves its depth, so it contributes to no "
            "per-position denominator. "
            f"trial_id={trial.get('trial_id')} run_id={trial.get('run_id')}.",
            RuntimeWarning, stacklevel=2,
        )
    return fallback


def presented_positions(trial) -> list[int]:
    """``sequence_depth`` as the contiguous position list ``[1..depth]``."""
    depth = sequence_depth(trial)
    return list(range(1, depth + 1)) if depth else []


def reached_counts(trials) -> dict[int, int]:
    """``{position: n trials that reached it}`` -- the per-position denominator.

    The single definition of "reached" for the whole package. Before Phase 4a
    this walk was written out four times: twice identically in ``metrics_utils``
    (``abortion_rate_positionX`` and ``fa_abortion_stats``) and twice more in
    ``visualization/`` under two *different* definitions.
    """
    reached: dict[int, int] = {}
    for _, trial in trials.iterrows():
        for pos in presented_positions(trial):
            reached[pos] = reached.get(pos, 0) + 1
    return reached


def sampled_positions(trial, *, only_true_pokes: bool = False) -> Optional[list[int]]:
    """Positions with a recorded poke on this trial. May be gappy; filterable.

    "Was this position sampled" -- the counterpart to ``sequence_depth``'s "how
    far did the sequence get". A gap is meaningless for the latter and perfectly
    natural here ("no sample at position 3 on this trial"), which is why they are
    two helpers rather than one with a flag.

    ``only_true_pokes=True`` keeps only entries marked ``poke_source == "poke"``,
    excluding grace-synthesised and ``outside_grace`` entries. That field will
    never exist on sessions saved before Phase 6b, so when it is **absent this
    returns None** -- callers must omit the filtered variant rather than fall
    back to the unfiltered value, which would make old and new sessions look
    comparable when they are not.
    """
    ppt = parse_json_column(trial.get("position_poke_times", {}))
    entries = ppt if isinstance(ppt, dict) else {}
    if not entries:
        pres = parse_json_column(trial.get("presentations", []))
        if isinstance(pres, list):
            entries = {
                p.get("position"): p for p in pres
                if isinstance(p, dict) and p.get("position") is not None
            }

    if only_true_pokes:
        if not entries:
            # No positions recorded at all is an *empty* answer, not an
            # unanswerable one. Only an absent `poke_source` yields None, so a
            # caller testing `is None` to decide whether to emit the filtered
            # variant does not also silently skip position-less trials.
            return []
        if not any(isinstance(v, dict) and v.get("poke_source") is not None
                   for v in entries.values()):
            return None
        entries = {k: v for k, v in entries.items()
                   if isinstance(v, dict) and v.get("poke_source") == "poke"}

    out = [_as_int(k) for k in entries.keys()]
    return sorted(p for p in out if p is not None)


# ---------------------------------------------------------------- position_data

# Fields carried by each blob. `presentations` is the only one with
# index_in_trial / is_last_event; only `position_poke_times` has the poke
# start/end timestamps; only `position_valve_times` has valve_end.
_POKE_FIELDS = ("poke_time_ms", "poke_odor_start", "poke_odor_end", "poke_first_in",
                "poke_source")
_VALVE_FIELDS = ("valve_start", "valve_end", "valve_duration_ms")
_PRES_FIELDS = ("index_in_trial", "is_last_event", "poke_time_ms", "poke_first_in",
                "poke_source", "valve_start", "valve_end", "valve_duration_ms")

# The fields taken from `presentations` and from **nowhere else**. Separate from
# `_PRES_FIELDS`, which documents everything that blob happens to carry and is
# deliberately *not* read by the builder: its other members are already sourced from
# `position_poke_times` / `position_valve_times` with `presentations` as the fallback.
#
# It exists as its own constant so the copy loop and `CARRIED_FIELDS` are driven by one
# declaration. They were not: the loop hardcoded this pair while `CARRIED_FIELDS` was
# built from `_PRES_FIELDS`, so a field added to `_PRES_FIELDS` alone would have passed
# the guard and still been dropped -- the exact silent loss the guard exists to catch.
_PRES_ONLY_FIELDS = ("index_in_trial", "is_last_event")

_ID_COLUMNS = ("trial_id", "global_trial_id", "subjid", "date", "session_num")

# Trial-level columns denormalised onto every position row, because a
# per-position metric needs them and joining back to `trials` for one scalar is
# not worth it. `is_aborted` is already carried this way.
#
# `last_event_index` is the abort event's index within the trial:
# `avg_sampling_time_aborted_sequence` excludes exactly the entry whose
# `index_in_trial` equals it. The `is_last_event` flag `presentations` carries
# agrees with that rule on all 9 fixture sessions, but it is a *different* rule,
# and 4a reproduces today's values rather than a rule that happens to match.
_TRIAL_COLUMNS = ("last_event_index",)

# Every blob field `build_position_data` copies onto a position row. The field lists
# above are a **whitelist**, not a passthrough: a key not named here is dropped
# silently -- no error, no empty column, nothing. Once Phase 7b.4b removes the blobs
# from `trial_data`, that silence is data loss, so the set is declared here and
# checked rather than left implicit.
# Built from exactly what the copy loop below reads -- **not** from `_PRES_FIELDS`,
# which is documentation. A guard whose whitelist is wider than the behaviour it guards
# reports a pass for a field that is still being dropped.
CARRIED_FIELDS = frozenset(
    _POKE_FIELDS + _VALVE_FIELDS + _PRES_ONLY_FIELDS
    + ("position", "odor_name", "required_min_sampling_time_ms")
)

# Blob fields deliberately **not** carried. A field belongs here only when the
# information is not lost -- "nothing reads it today" is not a reason, because the
# point of the check is the reader that does not exist yet.
#
# `prior_presentations` is the failed Position-1 attempts preceding a trial: a list of
# dicts written into position 1's valve entry by `classify_trials._position_valve_times`
# and read exactly once, *in memory*, by `classify_trials` itself, to build
# `non_initiated_odor1_attempts` -- which `save_results` already persists as its own
# table. It also does not fit this grain: one row per failed *attempt*, not per
# `trial x position`. Measured: 1,730 occurrences over the nine fixture sessions.
KNOWN_UNCARRIED_FIELDS = {
    "prior_presentations":
        "per-attempt, not per-position; already saved as non_initiated_odor1_attempts",
}


class UncarriedPositionFieldError(Exception):
    """A position blob carries a field `build_position_data` would silently drop.

    Named rather than a bare `ValueError` so it is greppable in a batch log and
    separately catchable -- `batch_analyze_sessions` wraps each session in
    `except Exception`, where a `ValueError` is indistinguishable from pandas
    complaining about an index (the section 20 rule, applied to a second schema check).
    """


def _check_carried(seen_fields, *, strict: bool) -> None:
    """Report blob fields that `build_position_data` does not carry.

    **Raises at write time, warns at read time**, and the split is the point:

    * `save_results` passes ``strict=True``. The data was just produced by the current
      classifier, so an unrecognised field means somebody added one and did not carry
      it across -- exactly the mistake this exists to catch, and it should stop the
      session rather than write a `position_data` quietly missing a column. Raising is
      the safe failure in bulk for the section 20 reason: the batch catches per
      session, names it, writes nothing, and carries on.
    * every read path leaves it ``False``. This function also runs over sessions saved
      years ago, whose blobs are *not* today's (section 2), and refusing to read
      historical data because it carries a field we since stopped writing would be a
      far worse failure than dropping it.
    """
    unknown = sorted(f for f in seen_fields
                     if f not in CARRIED_FIELDS and f not in KNOWN_UNCARRIED_FIELDS)
    if not unknown:
        return
    detail = (f"position blob field(s) {', '.join(unknown)} are not carried into "
              f"`position_data` and would be silently dropped. Add them to "
              f"`_POKE_FIELDS` / `_VALVE_FIELDS` / `_PRES_ONLY_FIELDS` in `frames.py` "
              f"(NOT `_PRES_FIELDS`, which the builder does not read), or to "
              f"`KNOWN_UNCARRIED_FIELDS` with a reason the information is not lost.")
    if strict:
        raise UncarriedPositionFieldError(detail)
    warnings.warn(detail, RuntimeWarning, stacklevel=3)


def _entries_by_position(val) -> dict[int, dict]:
    """Normalise a position blob (dict-of-dicts or list-of-dicts) to {position: entry}."""
    parsed = parse_json_column(val)
    out: dict[int, dict] = {}
    if isinstance(parsed, dict):
        items = parsed.items()
    elif isinstance(parsed, list):
        items = ((e.get("position") if isinstance(e, dict) else None, e) for e in parsed)
    else:
        return out
    for key, entry in items:
        if not isinstance(entry, dict):
            continue
        pos = _as_int(entry.get("position", key))
        if pos is None:
            pos = _as_int(key)
        if pos is not None:
            out[pos] = entry
    return out


def build_position_data(trials, *, strict: bool = False) -> pd.DataFrame:
    """Expand the per-trial position blobs into one row per ``trial x position``.

    ``strict=True`` raises `UncarriedPositionFieldError` when a blob carries a field
    this function would drop. `save_results` passes it; read paths do not. See
    ``_check_carried``.

    The long frame every per-position and per-odor metric groups on. Built from
    the union of ``position_poke_times``, ``presentations`` and
    ``position_valve_times``, because **the three do not carry the same
    positions** and a metric that reads one must not silently pick up rows from
    another:

    - on a *completed* trial all three now hold every position with a valve
      activation, including ones the animal never poked -- Phase 6b writes those
      rather than dropping them, marking each ``poke_source == "outside_grace"``;
    - on an *aborted* trial ``position_valve_times`` still holds one position more
      than the other two whenever the trial ended on an unpoked odor, because the
      trailing entry is trimmed back to the last real poke (``classification_utils
      ._trim_unsampled_tail``).

    So each row records **which blobs it came from** (``in_poke_times`` /
    ``in_presentations`` / ``in_valve_times``) and every metric filters on the
    provenance matching the blob it reads today. Without that,
    ``manual_vs_auto_stop_preference`` -- which counts valve durations -- would
    gain the trimmed positions and change value.

    ``poke_source`` is carried through but never **synthesised**: an absent column
    is how ``sampled_positions`` knows to omit the ``only_true_pokes`` variants,
    and how ``metrics.sampling._real_pokes`` knows to leave a pre-6b session's
    sampling averages exactly as they were.
    """
    if trials is None or len(trials) == 0:
        return pd.DataFrame()

    id_cols = [c for c in _ID_COLUMNS + _TRIAL_COLUMNS if c in trials.columns]
    rows = []
    # Accumulated across the whole frame and checked once, rather than per entry: the
    # answer is a property of the session, and a per-entry check would emit the same
    # warning thousands of times.
    seen_fields: set = set()
    for _, trial in trials.iterrows():
        poke = _entries_by_position(trial.get("position_poke_times", {}))
        pres = _entries_by_position(trial.get("presentations", []))
        valve = _entries_by_position(trial.get("position_valve_times", {}))
        aborted = _is_aborted(trial)
        for entries in (poke, pres, valve):
            for entry in entries.values():
                seen_fields.update(entry)

        for pos in sorted(set(poke) | set(pres) | set(valve)):
            p_entry, r_entry, v_entry = poke.get(pos, {}), pres.get(pos, {}), valve.get(pos, {})
            row = {c: trial.get(c) for c in id_cols}
            row.update({
                "position": pos,
                "is_aborted": aborted,
                "in_poke_times": pos in poke,
                "in_presentations": pos in pres,
                "in_valve_times": pos in valve,
                # odor_name agrees across blobs; take whichever is present.
                "odor_name": (p_entry.get("odor_name") or v_entry.get("odor_name")
                              or r_entry.get("odor_name")),
                "required_min_sampling_time_ms": (
                    p_entry.get("required_min_sampling_time_ms")
                    or v_entry.get("required_min_sampling_time_ms")
                    or r_entry.get("required_min_sampling_time_ms")),
            })
            # Poke fields: position_poke_times wins, presentations fills in.
            for f in _POKE_FIELDS:
                row[f] = p_entry.get(f, r_entry.get(f))
            # Valve fields: position_valve_times wins, presentations fills in.
            for f in _VALVE_FIELDS:
                row[f] = v_entry.get(f, r_entry.get(f))
            # Only `presentations` carries these, so there is no fallback source.
            for f in _PRES_ONLY_FIELDS:
                row[f] = r_entry.get(f)
            rows.append(row)

    _check_carried(seen_fields, strict=strict)
    return pd.DataFrame(rows)
