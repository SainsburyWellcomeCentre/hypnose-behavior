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
from hypnose_behavior.trial_classification.run import analyze_session_multi_run_by_id_date
from hypnose_behavior.io.load_results import load_session_results
from hypnose_behavior.metric_analysis.run import run_all_metrics
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

        {'trial_data': md5, 'trial_data_columns': {col: md5},
         'metrics': md5,    'metrics_keys': {key: md5}}

    The overall md5s are the pass/fail signal; the per-column / per-key md5s let a
    mismatch report exactly *what* changed. Raises on any failure so a broken
    session is never silently fingerprinted.
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

        matches = list(tmp.glob(f"**/ses-*_date-{date}/saved_analysis_results/trial_data.csv"))
        if not matches:
            raise FileNotFoundError(
                f"trial_data.csv not found for subj={subjid} date={date} under {tmp}"
            )
        trial_data_md5, trial_data_columns = _trial_data_fingerprint(matches[0])

        with contextlib.redirect_stdout(sink):
            results = load_session_results(subjid, date)
            metrics = run_all_metrics(results, save_txt=False, save_json=False)
        metrics_md5, metrics_keys = _metrics_fingerprint(metrics)

    return {
        "trial_data": trial_data_md5,
        "trial_data_columns": trial_data_columns,
        "metrics": metrics_md5,
        "metrics_keys": metrics_keys,
    }


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
