# Defers evaluation of PEP-604 annotations (`X | None`), keeping this module
# importable on Python 3.9 for repos pinned there (hypnose-eeg-preprocessing).
from __future__ import annotations

"""Pooling several sessions' results dicts into one.

Mirrors ``trial_classification/merge.py``.

- Pooling concatenates every DataFrame key, so a metric run on the pooled dict is
  computed over the **raw trials** of every session -- never averaged from per-session
  values. That is the same reduction rule ``resolvers.by_group`` follows, and what lets
  ``run_all_metrics`` take a pooled dict unchanged.
- A pooled frame has **no key separating two sessions' trials**: ``global_trial_id``
  collides across sessions. Per-position metrics must be computed one session at a time.
  See DECISIONS.md section 28.
"""

import pandas as pd

__all__ = ["pool_results_dicts"]


def pool_results_dicts(results_dicts):
    """
    Given a list of results dicts (from load_session_results), pool all DataFrames by key.
    Returns a single results dict with concatenated DataFrames and merged manifest/summary.
    """
    pooled = {}
    # Pool DataFrames
    all_keys = set()
    for r in results_dicts:
        all_keys.update(r.keys())
    for key in all_keys:
        dfs = [r[key] for r in results_dicts if key in r and isinstance(r[key], pd.DataFrame)]
        if dfs:
            pooled[key] = pd.concat(dfs, ignore_index=True)
        else:
            pooled[key] = results_dicts[0].get(key, None)
    # Merge manifest/summary for merged info

    def get_subjid(r):
        sess = r.get("manifest", {}).get("session", {})
        return str(sess.get("subject_id") or sess.get("subjid") or "")

    def get_date(r):
        sess = r.get("manifest", {}).get("session", {})
        return str(sess.get("date") or sess.get("session_date") or "")

    subjids = sorted({get_subjid(r) for r in results_dicts if get_subjid(r)})
    dates = sorted({get_date(r) for r in results_dicts if get_date(r)})

    protocol = None
    for r in results_dicts:
        runs = r.get("summary", {}).get("session", {}).get("runs", [])
        if runs and "stage" in runs[0]:
            protocol = runs[0]["stage"].get("stage_name", None)
            if protocol:
                break
    pooled["manifest"] = {
        "merged_subjects": subjids,
        "merged_dates": dates,
        "protocol": protocol
    }
    pooled["summary"] = {
        "merged_subjects": subjids,
        "merged_dates": dates,
        "protocol": protocol
    }
    return pooled
