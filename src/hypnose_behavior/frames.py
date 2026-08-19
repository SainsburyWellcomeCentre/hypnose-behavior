# Defers evaluation of PEP-604 annotations (`X | None`), keeping this module
# importable on Python 3.9 for repos pinned there (hypnose-eeg-preprocessing).
from __future__ import annotations

"""The frames metric cores consume, and the helpers that build them.

Every metric core is a pure ``f(frame) -> value`` over one of two frames:

``trials``
    ``trial_data`` -- one row per trial.

``position_data``
    long form, one row per ``trial x position``, so metrics never parse a JSON blob
    and carry no legacy-session branch.

- **This module imports nothing from the package** -- standard library and pandas
  only. Every layer stands on it, so the day it imports back, they all inherit that
  dependency. See DECISIONS.md section 3.
- It owns the single definition of "how far did the sequence get"
  (``sequence_depths``) and of "reached" (``reached_counts``). Both are stated at the
  **frame** grain: a per-trial signature over a long frame invites being handed the
  wrong trial's rows.
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
    "sequence_depths",
    "reached_counts",
    "position_entries_by_trial",
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

    **The one normaliser for an odor token.** It knows what the data *is*, so it
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


# The one column that identifies a trial row in both frames. `trial_data` carries no
# `subjid` / `date` column -- measured on all nine fixtures, the only ids the two frames
# share are `trial_id` and `global_trial_id` -- and `trial_id` restarts per run, so on a
# multi-run session it is not a key at all. `global_trial_id` is unique within a session
# (verified on all nine, including the 3-run `sub-040 20251124`).
_TRIAL_KEY = "global_trial_id"

# The provenance flags the depth rule reads, and the marker it filters on.
_DEPTH_COLUMNS = ("position", _TRIAL_KEY, "in_poke_times", "in_presentations")


def position_entries_by_trial(position_data, flag: str = "in_poke_times") -> dict:
    """``{global_trial_id: [entry, ...]}``, each list sorted by position.

    The replacement, for callers outside ``metric_analysis``, for parsing a trial
    row's JSON blob and sorting its entries by hand. Each entry is a plain dict,
    so a caller keeps whatever reduction it already applied -- and they do *not*
    share one:

    - ``sing_rew_movement._last_poke_out`` takes the **maximum** ``poke_odor_end``
      across entries, deliberately not trusting position order;
    - ``metric_analysis.movement._last_poke_out`` scans **back by position** to the
      first non-null one;
    - ``movement_analysis_utils._last_poke_out_by_position`` takes the **last
      entry** and accepts its null.

    Those are three different rules and must not be merged, so this yields the entries
    and reduces nothing. See DECISIONS.md sections 13 and 28.

    ``flag`` is the provenance column naming which per-position source to read:
    ``in_poke_times``, ``in_valve_times`` or ``in_presentations``. **It is not optional
    in meaning** -- an unfiltered read picks up the ~0 ms valve positions the other two
    sources drop, which changes a metric's value. See DECISIONS.md section 2.

    Returns ``{}`` for an absent or unusable frame, which every caller treats as
    "no positions".
    """
    if position_data is None or len(position_data) == 0:
        return {}
    if flag not in position_data.columns or _TRIAL_KEY not in position_data.columns:
        return {}
    rows = position_data[position_data[flag].astype(bool)]
    if rows.empty:
        return {}
    rows = rows.sort_values([_TRIAL_KEY, "position"], kind="stable")
    out: dict = {}
    for entry in rows.to_dict("records"):
        out.setdefault(entry.get(_TRIAL_KEY), []).append(entry)
    return out


def _aborted_flags(trials):
    """`is_aborted == True` per row -- the vectorised form of the old `_is_aborted`.

    `== True` rather than `.astype(bool)`, so a stringy "True" is not silently
    promoted and a missing column yields all-False, exactly as before.
    """
    if "is_aborted" in trials.columns:
        return trials["is_aborted"] == True  # noqa: E712
    return pd.Series(False, index=trials.index)


def _fallback_depths(trials):
    """``last_odor_position`` as a depth, per trial row.

    ``last_odor_position`` is already a position. ``last_event_index`` is a
    0-based index into ``presentations``, so it needs ``+1`` to be the same
    unit. **Do not return the two interchangeably** -- that is an off-by-one that
    stays latent while ``last_odor_position`` is present on every session.
    """
    if "last_odor_position" in trials.columns:
        vals = pd.to_numeric(trials["last_odor_position"], errors="coerce")
    elif "last_event_index" in trials.columns:
        vals = pd.to_numeric(trials["last_event_index"], errors="coerce") + 1
    else:
        return pd.Series(pd.NA, index=trials.index, dtype="Int64")
    return vals.astype("Float64").astype("Int64")


def _deepest_by_trial(position_data, mask):
    """``{global_trial_id: max position}`` over the rows `mask` selects."""
    rows = position_data[mask]
    if rows.empty:
        return pd.Series(dtype="Int64")
    positions = pd.to_numeric(rows["position"], errors="coerce")
    return positions.groupby(rows[_TRIAL_KEY]).max().astype("Float64").astype("Int64")


def _per_trial(by_id, trials):
    """A `{trial_key: value}` mapping projected onto `trials`' own rows.

    A `.map` rather than a merge, deliberately: it yields exactly one value per
    trial row, so the number of trials counted never depends on the join. That
    matters because `merge.pool_results_dicts` concatenates sessions and
    `global_trial_id` repeats across them -- with a groupby, colliding ids would
    silently *merge trials*, changing a denominator. `sequence_depths` warns on
    that collision rather than pretending it resolved it.
    """
    if by_id.empty or _TRIAL_KEY not in trials.columns:
        return pd.Series(pd.NA, index=trials.index, dtype="Int64")
    return trials[_TRIAL_KEY].map(by_id).astype("Float64").astype("Int64")


def sequence_depths(trials, position_data):
    """How far each sequence got, as a contiguous depth ``1..n``. Never filtered.

    Returns an ``Int64`` Series aligned to ``trials.index``. This is the
    denominator of every per-position rate ("of the trials that reached position
    *p*, how many aborted there"), so it must be **monotonic**: a trial that
    reached position 5 reached positions 1-4 as well. That is why it is a depth
    to be filled contiguously rather than a set of positions -- a gap is
    meaningless for *reached* and perfectly natural for *sampled*, which is a
    different question (``metrics.sampling._real_pokes``) and must not be
    substituted here.

    The depth is what the trial is **credited** with, filtered by ``poke_source``
    only where the filter is required:

    - **completed** -- every presented position counts. The rig advanced through
      all of them, including a final one our DIPort0 reconstruction scores as
      unpoked, so the filter must *not* be applied.
    - **aborted** -- the sequence stops at the last real poke, so the filter
      *is* applied. Falls back to ``last_odor_position``, which the abort
      pipeline derives independently and which agrees with the filtered maximum
      on **486 of 486** trials where both are defined.

    **Neither single-meaning form may be substituted for the split.** Unfiltered
    ``max(presentations)`` everywhere moves 84 trials; ``max(poke_source == 'poke')``
    everywhere moves 32. See DECISIONS.md sections 10 and 18.

    The two precedences below are **opposite**, and that is deliberate:

    - presented: ``in_poke_times`` first, ``in_presentations`` second. On
      sessions written by this pipeline the two carry the same position set --
      their maxima agree on 1,730 of 1,730 trials -- so the order cannot matter
      there. It matters for sessions saved earlier, where the blobs did *not*
      agree, and this order reproduces exactly what the blob form returned.
    - sampled: ``in_presentations`` first, ``in_poke_times`` second -- **the
      opposite order**, and the opposite of the merge inside
      ``build_position_data``, where ``position_poke_times`` wins. That
      reconstruction is exact only because section 24's inventory measured the
      blobs never to disagree on a shared field (``differs: 0`` on all 4,791
      rows). It is a measurement, not a property of this code, so it was
      re-measured directly against the blob form before the switch landed:
      1,731 of 1,731 trials equal, on both the written ``position_data.parquet``
      and the load-time derivation.

    Warns when neither the filtered maximum nor ``last_odor_position`` resolves
    an aborted trial: that is a trial with no sampled position at all, and it
    drops out of every per-position denominator, so it is worth investigating
    rather than silently counting as nothing.
    """
    if trials is None or len(trials) == 0:
        return pd.Series(dtype="Int64")

    aborted = _aborted_flags(trials)
    fallback = _fallback_depths(trials)

    usable = (position_data is not None and len(position_data) > 0
              and all(c in position_data.columns for c in _DEPTH_COLUMNS))
    if not usable:
        # No position rows resolves every depth to `last_odor_position`, which is
        # what the blob form returned when both blobs were empty. Section 2: an
        # absent source is unknown, never "all of them".
        presented = sampled = pd.Series(pd.NA, index=trials.index, dtype="Int64")
    else:
        if _TRIAL_KEY in trials.columns and not trials[_TRIAL_KEY].is_unique:
            warnings.warn(
                f"{_TRIAL_KEY} repeats in this trial frame, so `position_data` rows "
                "cannot be attributed to a single trial -- depths for the colliding "
                "trials are the maximum across them. This is a pooled frame "
                "(`merge.pool_results_dicts` concatenates sessions); compute "
                "per-position metrics one session at a time.",
                RuntimeWarning, stacklevel=2,
            )
        in_poke = position_data["in_poke_times"].astype(bool)
        in_pres = position_data["in_presentations"].astype(bool)
        real_poke = (position_data["poke_source"] == "poke"
                     if "poke_source" in position_data.columns
                     else pd.Series(False, index=position_data.index))

        presented = _per_trial(_deepest_by_trial(position_data, in_poke), trials)
        presented = presented.fillna(_per_trial(_deepest_by_trial(position_data, in_pres),
                                                trials))
        sampled = _per_trial(_deepest_by_trial(position_data, in_pres & real_poke), trials)
        sampled = sampled.fillna(_per_trial(_deepest_by_trial(position_data,
                                                              in_poke & real_poke), trials))

    unresolved = aborted & sampled.isna() & fallback.isna()
    for _, trial in trials[unresolved].iterrows():
        warnings.warn(
            "aborted trial has no sampled position: neither a `poke_source == 'poke'` "
            "entry nor `last_odor_position` resolves its depth, so it contributes to no "
            "per-position denominator. "
            f"trial_id={trial.get('trial_id')} run_id={trial.get('run_id')}.",
            RuntimeWarning, stacklevel=2,
        )

    return sampled.fillna(fallback).where(aborted, presented.fillna(fallback))


def reached_counts(trials, position_data) -> dict[int, int]:
    """``{position: n trials that reached it}`` -- the per-position denominator.

    **The single definition of "reached" for the whole package.** Do not count a
    trial's presented positions instead; that is a different denominator.

    Counts **per trial row**, so a depth of *n* fills positions 1..n. A null or zero
    depth contributes nothing.
    """
    reached: dict[int, int] = {}
    for depth in sequence_depths(trials, position_data).dropna():
        for pos in range(1, int(depth) + 1):
            reached[pos] = reached.get(pos, 0) + 1
    return reached


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
# `index_in_trial` equals it. **`is_last_event` is a different rule**, even though it
# agrees on every fixture session -- do not substitute one for the other.
_TRIAL_COLUMNS = ("last_event_index",)

# Every blob field `build_position_data` copies onto a position row. The field lists
# above are a **whitelist**, not a passthrough: a key not named here is dropped
# silently -- no error, no empty column, nothing. Since the blobs are not written to
# `trial_data`, that silence is data loss, so the set is declared here and checked.
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
# `non_initiated_odor1_attempts` -- which `save_results` persists, **as rows of
# `non_initiated_attempts`**. It also does not fit this grain: one row per failed
# *attempt*, not per `trial x position`. Measured: 1,730 occurrences over the nine
# fixture sessions.
#
# **The information surviving in `non_initiated_attempts` is what licenses the entry.**
# If that ever stops being true, the entry must come off this list.
KNOWN_UNCARRIED_FIELDS = {
    "prior_presentations":
        "per-attempt, not per-position; already saved as rows of non_initiated_attempts",
}


class UncarriedPositionFieldError(Exception):
    """A position blob carries a field `build_position_data` would silently drop.

    **Keep it a named class, not a bare `ValueError`:** `batch_analyze_sessions` wraps
    each session in `except Exception`, where a `ValueError` is indistinguishable from
    pandas complaining about an index.
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

    - on a *completed* trial all three hold every position with a valve activation,
      including ones the animal never poked, each marked
      ``poke_source == "outside_grace"``;
    - on an *aborted* trial ``position_valve_times`` holds one position more than the
      other two whenever the trial ended on an unpoked odor, because the trailing entry
      is trimmed back to the last real poke
      (``trial_classification.classify_trials._trim_unsampled_tail``).

    So each row records **which blobs it came from** (``in_poke_times`` /
    ``in_presentations`` / ``in_valve_times``) and every metric filters on the
    provenance matching the blob it reads today. Without that,
    ``manual_vs_auto_stop_preference`` -- which counts valve durations -- would
    gain the trimmed positions and change value.

    ``poke_source`` is carried through but **never synthesised**: an absent column is
    how ``sampled_positions`` knows to omit the ``only_true_pokes`` variants, and how
    ``metrics.sampling._real_pokes`` knows to leave an older session's sampling averages
    exactly as they were. Absent means *unknown*, never "all real pokes".
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
