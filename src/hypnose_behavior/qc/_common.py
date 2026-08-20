"""Shared core for the restructuring regression harness.

The restructuring of this code-base is done as a series of *pure moves* (split /
rename / relocate, no logic changes). To prove each step preserves behaviour we
fingerprint the two pipeline outputs that must stay byte-for-byte identical:

  1. per-session ``trial_data`` (trial classification)
  2. the metrics dict (metric analysis)

Design choices that make the fingerprint trustworthy:

* Reads the real, read-only ``rawdata``; redirects ALL derivatives I/O to a
  throwaway temp dir via ``HYPNOSE_DERIVATIVES_ROOT`` so nothing touches the
  server and runs never collide.
* Fingerprints the *canonical CSV* of ``trial_data`` (sorted columns, reset
  index) -- NOT parquet bytes, whose pyarrow/version/compression metadata is
  non-deterministic -- and never the manifest/summary files (wall-clock
  timestamps live there).
* Fingerprints metrics from the *returned dict* (no file, no timestamps).
* Imports every pipeline entry point in ONE place (below). When modules move
  during the restructuring, only these import lines change -- the md5s must not.
"""
from __future__ import annotations

import os
import io
import json
import hashlib
import tempfile
import contextlib
from pathlib import Path

import pandas as pd

# --- single import surface -------------------------------------------------
# Update ONLY these lines as modules move during the restructuring. The md5
# fingerprints they produce must remain identical at every step.
import hypnose_behavior.io.paths as _paths
from hypnose_behavior.io import layout
from hypnose_behavior.trial_classification.run import analyze_session_multi_run_by_id_date
from hypnose_behavior.io.load_results import load_session_results
from hypnose_behavior.metric_analysis.run import run_all_metrics
from hypnose_behavior.metric_analysis.registry import REGISTRY
# ---------------------------------------------------------------------------


# Registered metrics that `run.REPORT` does not save, and which are therefore absent
# from the metrics dict this harness fingerprints. Before Phase 7b.6 all 18 had
# **zero** coverage: a change to any of them was invisible to every gate.
#
# **Sixteen of the eighteen, and the boundary is principled rather than arbitrary.**
# `rolling_reward_fraction` and `rolling_hr_reward_fraction` take `window`
# *positionally, with no default*, so fingerprinting them would mean inventing a
# figure choice -- exactly the session-vs-figure line `DECISIONS.md` section 5 draws.
# The rest are callable from their frames alone, or have defaults that mean something
# definite: `fa_types=None` / `fr_types=None` are *unfiltered*, `aborted=False` is
# *completed*. Only that default variant is covered; a change reachable only with a
# non-default argument still is not.
#
# **Every one was checked to be populated before being fingerprinted** (section 26),
# because `--generate` blesses whatever a metric returns and a hash of an empty
# result is a canonised bug. Measured over the nine sessions: 14 populated on all 9,
# `fa_latency_from_pokeout` on 7 (it needs false alarms) and `hr_abort_poke_gap` on 3
# (it needs a hidden rule). None was empty everywhere.
UNREPORTED_METRICS = (
    "fa_latency_from_pokeout",
    "fa_port_counts",
    "fa_rate_by_odor",
    "fa_rate_by_position",
    "false_response_ratio",
    "hidden_rule_mask",
    "hr_abort_poke_gap",
    "inter_trial_interval",
    "poke_duration_by_odor",
    "poke_duration_by_position",
    "poke_durations",
    "presentation_counts_by_odor",
    "reward_delivery_latency",
    "trial_poke_span",
    "trial_poke_total",
    "valve_to_reward_latency",
)

# The tables written beside `trial_data`, fingerprinted from the **written file** so the
# save path added in 7b.4a / 7b.5 is covered rather than just the in-memory frame.
#
# The three `non_initiated_*` tables were added for Item 5, and the reason is section 26's
# lesson in a new place: item 5 *deletes* one of them and *renames* another, and not one of
# the six fingerprints this harness carried could see either. `verify_scripts` compares
# `trial_data` and `metrics` alone, so it could not either. A change to what the pipeline
# writes would have gone GREEN by nothing looking at the files it changed.
#
# A table absent on a session fingerprints as the md5 of `"ABSENT"` (below), which is what
# makes them safe to list even though they are sparse: `non_initiated_odor1_attempts` exists
# on 2 of the 9 coverage sessions and sub-048 carries none of the three, because they are not
# written when empty. `"ABSENT"` records that deliberately, so a session that silently *stops*
# writing one is a RED rather than a shorter fingerprint -- the section 2 rule.
#
# Populated before blessed (section 26): measured across the nine sessions before generating,
# `non_initiated_sequences` and `non_initiated_FA` are populated on 8, `non_initiated_odor1_attempts`
# on 2. None is empty everywhere, so no fixture canonises an empty result.
#
# The last three are **deletion guards, not tables**: Item 5 stopped writing
# `non_initiated_sequences` and `non_initiated_odor1_attempts` (each contained in
# `non_initiated_attempts` by construction) and renamed `non_initiated_FA` to
# `non_initiated_attempts`. All three are kept here fingerprinting as `"ABSENT"` so that
# resurrecting any of those files is a RED. Dropping them from this list right after
# using it to gate the deletion would leave the deletion asserted by nothing -- which is
# the failure this whole item is about.
SIDE_TABLES = (
    "position_data",
    "metrics_by_trial",
    "metrics_by_poke",
    "non_initiated_attempts",
    "non_initiated_sequences",
    "non_initiated_odor1_attempts",
    "non_initiated_FA",
)
# ---------------------------------------------------------------------------


def _md5(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def _canonical_trial_data_df(csv_path: Path) -> "pd.DataFrame":
    """Read trial_data with columns sorted and index reset (order-independent)."""
    df = pd.read_csv(csv_path)
    return df.reindex(sorted(df.columns), axis=1).reset_index(drop=True)


def _canonical_trial_data(csv_path: Path) -> str:
    """Column-order- and index-independent CSV serialization of trial_data."""
    return _canonical_trial_data_df(csv_path).to_csv(index=False)


def _canonical_metrics(metrics: dict) -> str:
    """Deterministic, timestamp-free serialization of the metric values."""
    return json.dumps(metrics, sort_keys=True, default=str)


def _trial_data_fingerprint(csv_path: Path) -> tuple[str, dict]:
    """Return (overall md5, {column_name: md5 of that column's values})."""
    df = _canonical_trial_data_df(csv_path)
    overall = _md5(df.to_csv(index=False))
    per_col = {str(c): _md5(df[c].to_csv(index=False, header=False)) for c in df.columns}
    return overall, per_col


def _metrics_fingerprint(metrics: dict) -> tuple[str, dict]:
    """Return (overall md5, {top_level_metric_key: md5 of its value})."""
    overall = _md5(_canonical_metrics(metrics))
    per_key = {str(k): _md5(json.dumps(v, sort_keys=True, default=str)) for k, v in metrics.items()}
    return overall, per_key


def _canonical_table_df(parquet_path: Path) -> "pd.DataFrame":
    """A saved table in canonical form: columns sorted, index reset.

    Reads the **parquet**, because that is the only copy `save_csv=False` writes, but
    fingerprints its *CSV rendering* rather than the file bytes -- pyarrow version and
    compression metadata are not deterministic, which is why `trial_data` has always
    been fingerprinted through the canonical CSV rather than the parquet.
    """
    df = pd.read_parquet(parquet_path)
    return df.reindex(sorted(df.columns), axis=1).reset_index(drop=True)


def _table_fingerprint(parquet_path: Path) -> tuple[str, dict]:
    """Return (overall md5, {column_name: md5}) for a saved side-table."""
    df = _canonical_table_df(parquet_path)
    overall = _md5(df.to_csv(index=False))
    per_col = {str(c): _md5(df[c].to_csv(index=False, header=False)) for c in df.columns}
    return overall, per_col


def _canonical_metric_value(value) -> str:
    """Deterministic text for any registered metric's return value.

    The unreported metrics return four different shapes -- `DataFrame`, `Series`,
    `dict` and `tuple` (`fa_port_counts`, `false_response_ratio`) -- so this dispatches
    rather than assuming. Frames go through `to_csv`, which preserves row order and is
    stable across pandas' repr changes; everything else through `json.dumps` with
    `sort_keys` and `default=str`, which renders a tuple as a list, deterministically.
    """
    if isinstance(value, pd.DataFrame):
        return value.to_csv(index=True)
    if isinstance(value, pd.Series):
        return value.to_csv(index=True, header=False)
    return json.dumps(value, sort_keys=True, default=str)


def _unreported_metrics_fingerprint(results) -> tuple[str, dict]:
    """Fingerprint the registered metrics `run.REPORT` does not save.

    Each is called at its **default** arguments via `MetricSpec.call`, i.e. the same
    `f(frame)` expression the registry defines. A metric that raises is recorded as
    `ERROR: ...` rather than skipped, so losing the ability to compute one is a RED
    instead of a silently shorter fingerprint.
    """
    per_key = {}
    for name in UNREPORTED_METRICS:
        spec = REGISTRY.get(name)
        if spec is None:
            per_key[name] = _md5("ABSENT FROM REGISTRY")
            continue
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                value = spec.call(results)
            per_key[name] = _md5(_canonical_metric_value(value))
        except Exception as e:
            per_key[name] = _md5(f"ERROR: {type(e).__name__}: {e}")
    overall = _md5(json.dumps(per_key, sort_keys=True))
    return overall, per_key


def diff_report(label: str, fixture_parts: dict, current_parts: dict, indent: str = "      ") -> list[str]:
    """Human-readable added/removed/changed lines between two {name: md5} maps."""
    fset, cset = set(fixture_parts), set(current_parts)
    added = sorted(cset - fset)
    removed = sorted(fset - cset)
    changed = sorted(k for k in (fset & cset) if fixture_parts[k] != current_parts[k])
    lines = []
    if added:
        lines.append(f"{indent}+ added {label}: {', '.join(added)}")
    if removed:
        lines.append(f"{indent}- removed {label}: {', '.join(removed)}")
    if changed:
        lines.append(f"{indent}~ changed {label}: {', '.join(changed)}")
    if not lines:
        lines.append(f"{indent}(overall md5 differs but every {label} md5 matches "
                     f"-- likely row order / dtype / a column not captured here)")
    return lines


def _redirect_derivatives(tmp: Path) -> None:
    """Point all derivatives I/O at `tmp` and clear cached path lookups."""
    os.environ["HYPNOSE_DERIVATIVES_ROOT"] = str(tmp)
    for name in ("get_derivatives_root", "get_server_root", "get_rawdata_root"):
        fn = getattr(_paths, name, None)
        if fn is not None and hasattr(fn, "cache_clear"):
            fn.cache_clear()


def fingerprint_session(subjid, date) -> dict:
    """Run classification + metrics for one session in an isolated temp derivatives
    dir and return its fingerprint:

        {'trial_data':         md5, 'trial_data_columns':         {col: md5},
         'metrics':            md5, 'metrics_keys':               {key: md5},
         'unreported_metrics': md5, 'unreported_metrics_keys':    {name: md5},
         <name>:              md5, f'{name}_columns':            {col: md5}
                                     ... for every name in SIDE_TABLES}

    The overall md5s are the pass/fail signal; the per-column / per-key md5s let a
    mismatch report exactly *what* changed. Raises on any failure so a broken
    session is never silently fingerprinted.

    **Every side table is fingerprinted from the written file** (Phase 7b.6), so the
    save path is covered and not merely the in-memory frame. Before this, `trial_data`
    and the reported metrics were the whole gate: `position_data.parquet` and the two
    metric tables were written by code no gate read back, and 18 registered metrics
    were computed by code no gate ran. Both GREENs that landed them were *additivity*,
    not coverage.

    A missing side-table is recorded as the md5 of `"ABSENT"` rather than skipped, so a
    session that silently stops writing one is a RED and not a shorter fingerprint --
    the section 2 rule that an absent marker must not read as agreement.
    """
    subjid = str(subjid)
    date = str(date)
    with tempfile.TemporaryDirectory(prefix="hyp_regress_") as tmp_str:
        tmp = Path(tmp_str)
        _redirect_derivatives(tmp)

        sink = io.StringIO()
        with contextlib.redirect_stdout(sink):
            # `save_csv=True` explicitly, never by default: this harness fingerprints the
            # *canonical CSV* of trial_data, so relying on `save_csv`'s default would let a
            # later change to it break the gate silently (Phase 7b.3).
            analyze_session_multi_run_by_id_date(
                subjid, date, verbose=False, save=True, print_summary=False, save_csv=True
            )

        matches = list(tmp.glob(f"**/ses-*_date-{date}/{layout.RESULTS_DIRNAME}/trial_data.csv"))
        if not matches:
            raise FileNotFoundError(
                f"trial_data.csv not found for subj={subjid} date={date} under {tmp}"
            )
        trial_data_md5, trial_data_columns = _trial_data_fingerprint(matches[0])
        results_dir = matches[0].parent

        with contextlib.redirect_stdout(sink):
            results = load_session_results(subjid, date)
            # `save_tables=True` explicitly, never by default -- the section 23 rule. The
            # value differs from the other two flags because these tables are *what is
            # being fingerprinted*, so they have to exist; the rule was "pass it
            # explicitly", not "pass False".
            metrics = run_all_metrics(results, save_txt=False, save_json=False, save_tables=True)
        metrics_md5, metrics_keys = _metrics_fingerprint(metrics)

        fingerprint = {
            "trial_data": trial_data_md5,
            "trial_data_columns": trial_data_columns,
            "metrics": metrics_md5,
            "metrics_keys": metrics_keys,
        }

        for name in SIDE_TABLES:
            path = layout.table_path(results_dir, f"{name}.parquet")
            if path.exists():
                table_md5, table_columns = _table_fingerprint(path)
            else:
                table_md5, table_columns = _md5("ABSENT"), {}
            fingerprint[name] = table_md5
            fingerprint[f"{name}_columns"] = table_columns

        with contextlib.redirect_stdout(sink):
            unreported_md5, unreported_keys = _unreported_metrics_fingerprint(results)
        fingerprint["unreported_metrics"] = unreported_md5
        fingerprint["unreported_metrics_keys"] = unreported_keys

    return fingerprint


def env_fingerprint() -> dict:
    """Versions that the md5s depend on; recorded with the fixtures."""
    import sys
    import numpy as np
    try:
        import pyarrow
        pyarrow_version = pyarrow.__version__
    except ImportError:
        pyarrow_version = None
    return {
        "python": sys.version.split()[0],
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "pyarrow": pyarrow_version,
    }
