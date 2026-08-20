"""Hidden-rule resolution, shared by ``classify_trials`` and ``analyze_response_times``.

Both classifiers resolve the hidden-rule positions identically -- from the stage name when it
encodes them, otherwise from the schema's inferred indices -- then check each trial's sequence
against them and print the same header.

That shared logic lives here so neither classifier imports from the other. It is the same
shape as ``windows.py`` and ``outcome.py``: the shared rule becomes a leaf, and every caller
depends on the leaf rather than on a peer (``DECISIONS.md`` sections 3 and 13).
"""
from __future__ import annotations

import re
from collections.abc import Mapping

import hypnose_behavior.io.detect_settings as detect_settings


def _ensure_int_list(value, *, subtract_one: bool = False) -> list[int]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        candidates = value
    else:
        candidates = [value]
    out: list[int] = []
    for item in candidates:
        if item is None:
            continue
        try:
            number = int(item)
            if subtract_one:
                number -= 1
            out.append(number)
        except (TypeError, ValueError):
            continue
    return out


def _resolve_hidden_rule_from_stage(stage) -> tuple[list[int], str | None]:
    """Infer hidden-rule indices (possibly multiple) and stage name from metadata."""
    sequence_name = None
    indices: list[int] = []

    if isinstance(stage, Mapping):
        sequence_name = stage.get('stage_name') or stage.get('name')

        indices = _ensure_int_list(stage.get('hidden_rule_indices'))
        if not indices:
            indices = _ensure_int_list(stage.get('hidden_rule_index'))
        if not indices:
            indices = _ensure_int_list(stage.get('hidden_rule_positions'), subtract_one=True)
        if not indices:
            indices = _ensure_int_list(stage.get('hidden_rule_position'), subtract_one=True)

        if sequence_name is None:
            sequence_name = stage.get('name') or str(stage)
    else:
        sequence_name = str(stage)

    if sequence_name and not indices:
        match = re.search(r'location([0-9]+)', sequence_name, re.IGNORECASE)
        if match:
            digits = match.group(1)
            if digits:
                indices = [int(ch) for ch in digits if ch.isdigit()]

    indices = sorted({idx for idx in indices if isinstance(idx, int)})
    return indices, sequence_name


def _drop_final_hidden_rule_index(hidden_rule_indices, schema_settings, is_single_reward):
    """Remove the final-sequence-position index from hidden-rule candidates.

    The final position of a full sequence is always the reward position, so a
    rewarded odor appearing there is the normal reward, never a hidden rule. The
    final position index is detected in ``detect_settings`` (``finalPositionIndex``,
    == sequenceLength - 1) so it is not hardcoded here. The single-reward protocol
    is left untouched (its final position is not always rewarded).
    """
    if is_single_reward:
        return list(hidden_rule_indices)
    final_idx = schema_settings.get('finalPositionIndex')
    if final_idx is None:
        return list(hidden_rule_indices)
    return [idx for idx in hidden_rule_indices if idx != final_idx]


def _hidden_rule_indices_from_stage_or_schema(stage, root):
    """Hidden-rule indices from the stage name, falling back to the schema's inferred set.

    Returns ``(indices, sequence_name, schema_settings, schema_err)`` with the indices sorted
    and de-duplicated but the **final-position index still present**. Dropping it is left to
    the caller because the two callers do it at different moments: ``analyze_response_times``
    drops before printing its header, ``classify_trials`` after. Their headers therefore differ
    on a session whose final position is a hidden-rule candidate -- existing behaviour, not
    something to tidy here.
    """
    hidden_rule_indices, sequence_name = _resolve_hidden_rule_from_stage(stage)

    schema_settings = {}
    schema_err: Exception | None = None
    try:
        _, schema_settings = detect_settings.detect_settings(root)
    except Exception as exc:
        schema_err = exc
        schema_settings = {}

    if not hidden_rule_indices:
        inferred_indices = schema_settings.get('hiddenRuleIndicesInferred')
        if inferred_indices is None:
            inferred_indices = schema_settings.get('hiddenRuleIndexInferred')
        hidden_rule_indices = _ensure_int_list(inferred_indices)

    hidden_rule_indices = sorted({idx for idx in hidden_rule_indices if isinstance(idx, int)})
    return hidden_rule_indices, sequence_name, schema_settings, schema_err


def _hidden_rule_odor_set(hidden_rule_indices, schema_settings, schema_err, verbose):
    """The odors that count as a hidden-rule hit, or ``None`` when there is no hidden rule.

    Raises rather than degrading: a session with a hidden-rule position but no inferable odor
    identities would silently score every trial as a miss, so it is stopped instead.
    """
    if not hidden_rule_indices:
        return None
    try:
        if schema_err is not None and 'hiddenRuleOdorsInferred' not in schema_settings:
            raise ValueError(str(schema_err))
        odors = (schema_settings.get('hiddenRuleOdorsInferred') or [])
        if len(odors) < 2:
            raise ValueError("found fewer than two rewarded odors at inferred hidden rule position.")
        hr_odor_set = set(map(str, odors))
        if verbose:
            print(f"Hidden Rule Odors inferred: {sorted(hr_odor_set)}")
        return hr_odor_set
    except Exception as e:
        raise ValueError(f"Hidden Rule Odor Identities could not be inferred from Schema: {e}")


def _check_hidden_rule(odor_sequence, candidate_indices, odor_set):
    """``(enough_odors, hit, matching_indices)`` for one presented sequence.

    ``enough_odors`` is False when the sequence is too short to reach any candidate position,
    which is what separates "did not hit the hidden rule" from "never got the chance".
    """
    if not candidate_indices or odor_set is None:
        return False, False, []

    valid_indices = [idx for idx in candidate_indices if 0 <= idx < len(odor_sequence)]
    if not valid_indices:
        return False, False, []

    matching_indices = sorted({idx for idx in valid_indices if odor_sequence[idx] in odor_set})
    return True, bool(matching_indices), matching_indices


def _hidden_rule_positions(hidden_rule_indices):
    """``(positions, first_index, first_position, multiple)`` -- positions are 1-based indices."""
    positions = [idx + 1 for idx in hidden_rule_indices]
    location = hidden_rule_indices[0] if hidden_rule_indices else None
    position = positions[0] if positions else None
    return positions, location, position, len(positions) > 1


def _print_hidden_rule_header(hidden_rule_indices, hidden_rule_positions, sequence_name, stage,
                              *, label_prefix):
    """The 'Hidden rule location(s) extracted' header shared by the two classifiers.

    ``label_prefix`` differs by one space between the two call sites ("Location{n}" vs
    "Location {n}"), so it is passed rather than assumed.
    """
    seq_label = sequence_name or str(stage)
    if not hidden_rule_indices:
        print(f"No Hidden Rule Location found in sequence name: {seq_label}. Proceeding without Hidden Rule analysis.")
        return
    if len(hidden_rule_positions) > 1:
        pos_str = ", ".join(str(p) for p in hidden_rule_positions)
        idx_str = ", ".join(str(idx) for idx in hidden_rule_indices)
        print(f"Hidden rule locations extracted: Positions {pos_str} (indices {idx_str})")
    else:
        location = hidden_rule_indices[0]
        position = hidden_rule_positions[0]
        print(f"Hidden rule location extracted: {label_prefix}{location} (index {location}, position {position})")


def _hidden_rule_success(hr_hit_positions, last_position, max_positions, has_await_reward):
    """Did the animal act on the hidden rule? Returns ``(success, position)``.

    Leaving before the full sequence only counts if AwaitReward actually fired -- otherwise the
    animal simply abandoned the trial at a position that happened to carry the hidden-rule
    odor. Reaching the final position with the hidden-rule odor there counts either way.
    """
    if not hr_hit_positions:
        return False, None

    first_hr_pos = min(hr_hit_positions)
    if last_position < max_positions:
        if has_await_reward:
            return True, first_hr_pos
        return False, None

    success = last_position in hr_hit_positions
    return success, (first_hr_pos if success else None)
