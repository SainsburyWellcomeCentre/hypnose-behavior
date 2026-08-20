"""Schema-derived experiment parameters.

The functions that read a session's schema and turn it into the numbers the classifiers run
on: the sampling offset, the per-odor minimum sampling times, the response window, and the
single-reward protocol detection.

They live apart from the classifiers because all three of ``detect_trials``,
``classify_trials`` and ``analyze_response_times`` need the same parameters, and none of them
should own the parsing.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from dotmap import DotMap

import hypnose_behavior.io.detect_settings as detect_settings


def get_experiment_parameters(root):
    """
    Extract parameters from schema, including per-odor minimum sampling times.

    Returns:
        tuple: (sampleOffsetTime, minimumSamplingTime_by_odor, responseTime)
    """
    session_settings, session_schema = detect_settings.detect_settings(root)

    def _coerce_to_float(value):
        """Best-effort conversion of nested DotMap/dict/list structures to a float."""
        if value is None:
            return None
        # Handle numpy scalars
        if isinstance(value, (np.floating, np.integer)):
            return float(value)
        # Primitive numbers
        if isinstance(value, (int, float)):
            return float(value)
        # DotMap behaves like a dict; convert to plain dict first
        if isinstance(value, DotMap):
            value = value.toDict()
        if isinstance(value, dict):
            # Prefer common scalar keys if present
            for key in ('value', 'seconds', 'Seconds', 'ms', 'milliseconds'):
                if key in value:
                    coerced = _coerce_to_float(value[key])
                    if coerced is not None:
                        return coerced
            # Fall back to any single numeric value stored in the mapping
            numeric_vals = [v for v in value.values() if isinstance(v, (int, float, np.integer, np.floating))]
            if len(numeric_vals) == 1:
                return float(numeric_vals[0])
            if len(numeric_vals) > 1:
                # Ambiguous but still try first value deterministically
                return float(numeric_vals[0])
            # Walk nested structures if needed
            for nested in value.values():
                coerced = _coerce_to_float(nested)
                if coerced is not None:
                    return coerced
            return None
        if isinstance(value, (list, tuple)):
            for item in value:
                coerced = _coerce_to_float(item)
                if coerced is not None:
                    return coerced
            return None
        return None

    # sampleOffsetTime lives per-segment in the Schema (newer sessions); detect_settings
    # resolves it and falls back to the legacy SessionSettings location for older sessions.
    sample_offset_time = _coerce_to_float(session_schema.get('sampleOffsetTime'))
    if sample_offset_time is None:
        # Last-resort direct read of SessionSettings (handles nested metadata DotMaps)
        session_meta = session_settings.iloc[0]['metadata']
        sample_offset_time = _coerce_to_float(getattr(session_meta, 'sampleOffsetTime', None))
        if sample_offset_time is None:
            nested_meta = getattr(session_meta, 'metadata', None)
            sample_offset_time = _coerce_to_float(getattr(nested_meta, 'sampleOffsetTime', None)) if nested_meta else None
    if sample_offset_time is None:
        raise ValueError("sampleOffsetTime missing or invalid in Schema sequences and SessionSettings metadata")

    # Get per-odor minimumSamplingTime dict and ensure scalar values
    raw_minimums = session_schema.get('minimumSamplingTime_by_odor', {}) or {}
    minimumSamplingTime_by_odor = {}
    for odor, threshold in raw_minimums.items():
        coerced = _coerce_to_float(threshold)
        if coerced is not None:
            minimumSamplingTime_by_odor[str(odor)] = coerced

    response_time = _coerce_to_float(session_schema.get('responseTime'))

    return sample_offset_time, minimumSamplingTime_by_odor, response_time


def _sampling_parameters_ms(root, *, task: str):
    """Schema sampling parameters, converted to milliseconds.

    Returns ``(sample_offset_time_ms, minimum_sampling_time_ms_by_odor, default_minimum_ms,
    response_time_sec)``. ``task`` names the caller in the error message only, so the three
    call sites keep the wording they had. ``abortion_classification`` does not use this: it
    overlays thresholds carried on the classification dict before deciding whether the set is
    empty, and takes its default from there rather than from the maximum.
    """
    sample_offset_time, minimum_sampling_time_by_odor, response_time = get_experiment_parameters(root)
    sample_offset_time_ms = sample_offset_time * 1000
    minimum_sampling_time_ms_by_odor = {
        str(odor): float(threshold) * 1000.0
        for odor, threshold in (minimum_sampling_time_by_odor or {}).items()
        if threshold is not None
    }
    if not minimum_sampling_time_ms_by_odor:
        raise ValueError(
            f"minimumSamplingTime_by_odor missing or empty in schema; cannot {task} without per-odor thresholds"
        )
    default_minimum_sampling_time_ms = max(minimum_sampling_time_ms_by_odor.values())
    return (
        sample_offset_time_ms,
        minimum_sampling_time_ms_by_odor,
        default_minimum_sampling_time_ms,
        response_time,
    )


def _get_single_reward_info(root) -> tuple[bool, frozenset, frozenset]:
    """Determine whether a session uses the single-reward protocol and list its sequences.

    The single-reward protocol is the new task variant where NOT all candidate sequences are
    rewarded at their final position (e.g. ``singrew-task-stage1``: only ``OdorC-OdorF-OdorA``
    and ``OdorG-OdorE-OdorB`` are rewarded out of 8 candidate triples). It also covers older
    single-odor go/no-go stages (``FreeRun_StageN``). Detection is purely schema-based.

    Returns
    -------
    (is_single_reward, rewarded_sequences, all_sequences)
        is_single_reward : bool
            True iff at least one candidate sequence is NOT rewarded at its final position.
            For the default protocol (every sequence rewarded at the end) this is False and the
            false-response logic is never entered, so behaviour/output stay byte-for-byte unchanged.
        rewarded_sequences : frozenset[tuple[str, ...]]
            Concrete odor-name sequences whose final position is rewarded. A completed trial is
            "rewarded-type" iff ``tuple(its odor_sequence)`` is in this set; otherwise it is a
            non-rewarded ("no-go") sequence and gets false-response handling.
        all_sequences : frozenset[tuple[str, ...]]
            All concrete candidate sequences for the protocol (rewarded and non-rewarded).
            Used to compute reward determinacy from a presented (possibly partial) sequence:
            see ``_classify_reward_determinacy``.
    """
    try:
        _, schema_settings = detect_settings.detect_settings(Path(root))
    except Exception:
        return False, frozenset(), frozenset()
    if not schema_settings.get('isSingleRewardProtocol'):
        return False, frozenset(), frozenset()
    rewarded = schema_settings.get('rewardedSequences') or []
    rewarded_set = frozenset(tuple(seq) for seq in rewarded if seq)
    all_seqs = schema_settings.get('allSequences') or []
    all_set = frozenset(tuple(seq) for seq in all_seqs if seq)
    # If parsing produced no rewarded sequences, fall back to default behaviour (do nothing new).
    if not rewarded_set:
        return False, frozenset(), frozenset()
    return True, rewarded_set, all_set
