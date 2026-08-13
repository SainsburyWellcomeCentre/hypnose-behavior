#!/usr/bin/env python
"""Is `position_data` a lossless projection of the three per-trial blobs?

Phase 7b.4b drops `position_valve_times`, `position_poke_times` and `presentations`
from `trial_data`, leaving `position_data.parquet` as the only record of what was
measured per position. This asserts the precondition for that drop.

**The precondition is NOT "the two are identical".** That can never be true, and a
gate built on it would be wrong rather than merely strict. `build_position_data` is a
*projection*: it unions the three blobs into one row per ``trial x position``, then
copies a **fixed list of named fields**, with `position_poke_times` winning over
`presentations` for poke fields and `position_valve_times` winning for valve fields.
Anything outside that list is dropped, and where two blobs disagree on a shared field
the loser's value is not represented anywhere.

So the assertion is the one that actually licenses the drop:

    every field of every blob entry is recoverable from the matching
    `position_data` row, with an equal value -- except a short, named
    allow-list of fields deliberately not carried.

What it reports, per ``(blob, key)`` pair, over every trial and position:

  equal    the value is carried and matches
  differs  the value is carried but does NOT match  -- always a failure
  absent   the key is not carried at all            -- a failure unless allow-listed

**Why the allow-list must stay short and explicit.** The field list is a *whitelist*,
so a new field added to a blob by `classify_trials` disappears from `position_data`
silently -- no error, no empty column, nothing. Before the drop that is harmless;
after it, it is data loss with no signal. This gate turns that into a RED, and the
only way to accept a dropped field is to name it below and say why.

Recomputes from rawdata into a throwaway temp dir, exactly as `regression.py` does, so
it never reads the archive and never writes to the server.

Usage
-----
  PY=~/miniconda3/envs/hypnose-analysis-test/bin/python
  QC=~/repos/harris_lab/hypnose/hypnose-behavior-analysis/src/hypnose_behavior/qc

  $PY -u $QC/position_data_lossless.py                 # all sessions in sessions.yml
  $PY -u $QC/position_data_lossless.py 053:20260520    # just one

Exit code 0 == GREEN (lossless for everything not allow-listed); 1 == RED.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
import tempfile
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from hypnose_behavior.qc import _common
from hypnose_behavior.qc.regression import _load_sessions, _select
from hypnose_behavior.frames import (
    build_position_data, _entries_by_position, KNOWN_UNCARRIED_FIELDS,
)
from hypnose_behavior.trial_classification.run import analyze_session_multi_run_by_id_date

GREEN = "\033[32m"
RED = "\033[31m"
YELLOW = "\033[33m"
RESET = "\033[0m"

# The blob columns, and the `position_data` flag marking a row as having come from each.
BLOBS = {
    "position_poke_times": "in_poke_times",
    "position_valve_times": "in_valve_times",
    "presentations": "in_presentations",
}

# Fields deliberately not carried into `position_data`, keyed by field name.
#
# **Imported, not redeclared.** `frames.KNOWN_UNCARRIED_FIELDS` is the same list
# `build_position_data` checks against at write time, so this gate and the runtime
# guard cannot drift into disagreeing about what is allowed to be dropped. Keeping a
# second copy here is precisely the duplication that would let one be updated and the
# other not.
ALLOWED_UNCARRIED = KNOWN_UNCARRIED_FIELDS


def _values_equal(blob_value, row_value) -> bool:
    """Equality tolerant of null==null and of Timestamp vs its ISO rendering.

    `position_data` may hold a real `datetime64` where the blob held a `Timestamp`, or
    an ISO string where the frame came back through a JSON-encoded column, so a bare
    `==` would report a difference that is only a rendering. Values that are genuinely
    different still compare unequal.
    """
    b_null = blob_value is None or blob_value is pd.NaT or (
        isinstance(blob_value, float) and np.isnan(blob_value))
    r_null = row_value is None or row_value is pd.NaT or (
        isinstance(row_value, float) and np.isnan(row_value))
    if b_null and r_null:
        return True
    if b_null or r_null:
        return False
    if isinstance(blob_value, pd.Timestamp) or isinstance(row_value, pd.Timestamp):
        try:
            return pd.to_datetime(blob_value) == pd.to_datetime(row_value)
        except Exception:
            return False
    if isinstance(blob_value, float) or isinstance(row_value, float):
        try:
            return float(blob_value) == float(row_value)
        except (TypeError, ValueError):
            return False
    if isinstance(blob_value, (list, dict)) or isinstance(row_value, (list, dict)):
        dump = lambda v: json.dumps(v, default=str, sort_keys=True)  # noqa: E731
        return dump(blob_value) == dump(row_value)
    return blob_value == row_value or str(blob_value) == str(row_value)


def _check_session(subjid, date) -> dict:
    """Walk one session's blobs against its `position_data`. Returns per-(blob,key) counts."""
    stats = defaultdict(lambda: {"n": 0, "equal": 0, "differs": 0, "absent": 0,
                                 "example": None})
    with tempfile.TemporaryDirectory(prefix="hyp_lossless_") as tmp_str:
        _common._redirect_derivatives(Path(tmp_str))
        sink = io.StringIO()
        with contextlib.redirect_stdout(sink), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = analyze_session_multi_run_by_id_date(
                subjid, date, verbose=False, save=True, print_summary=False, save_csv=False)

        trials = out["classification"]["trial_data"]
        # Compared in memory, before serialisation, so a rendering difference in the
        # saved file cannot be mistaken for a dropped field. That the *written* table
        # matches this frame is `regression.py`'s `position_data` fingerprint, not this
        # gate's job.
        position_data = build_position_data(trials)

        rows = {}
        for _, r in position_data.iterrows():
            rows[(r.get("global_trial_id"), r.get("position"))] = r

        for _, trial in trials.iterrows():
            gid = trial.get("global_trial_id")
            for blob in BLOBS:
                for pos, entry in _entries_by_position(trial.get(blob)).items():
                    row = rows.get((gid, pos))
                    for key, value in (entry or {}).items():
                        st = stats[(blob, key)]
                        st["n"] += 1
                        if row is None or key not in row.index:
                            st["absent"] += 1
                            if st["example"] is None:
                                st["example"] = (f"trial {gid} pos {pos}: "
                                                 f"{'no position_data row' if row is None else repr(value)[:50]}")
                        elif _values_equal(value, row[key]):
                            st["equal"] += 1
                        else:
                            st["differs"] += 1
                            if st["example"] is None:
                                st["example"] = (f"trial {gid} pos {pos}: "
                                                 f"blob={value!r} position_data={row[key]!r}")
    return stats


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sessions", nargs="*", default=[],
                    help="limit to these subjid:date keys (default: all in sessions.yml)")
    args = ap.parse_args(argv)

    sessions = _select(_load_sessions(), set(args.sessions))
    print(f"Checking position_data against the blobs for {len(sessions)} session(s)...\n")

    total = defaultdict(lambda: {"n": 0, "equal": 0, "differs": 0, "absent": 0,
                                 "example": None})
    failed_sessions = []
    for s in sessions:
        subjid, date, label = s["subjid"], s["date"], s.get("label", "")
        try:
            stats = _check_session(subjid, date)
        except Exception as e:
            print(f"  [ERROR] sub-{subjid} {date} ({label}): {e!r}")
            failed_sessions.append((subjid, date))
            continue
        bad = sum(v["differs"] for v in stats.values()) + sum(
            v["absent"] for k, v in stats.items() if k[1] not in ALLOWED_UNCARRIED)
        mark = f"{GREEN}[green]{RESET}" if bad == 0 else f"{RED}[RED]  {RESET}"
        print(f"  {mark} sub-{subjid} {date} ({label}): "
              f"{len(stats)} (blob,key) pairs, {sum(v['n'] for v in stats.values())} occurrences")
        for k, v in stats.items():
            t = total[k]
            t["n"] += v["n"]; t["equal"] += v["equal"]
            t["differs"] += v["differs"]; t["absent"] += v["absent"]
            if t["example"] is None:
                t["example"] = v["example"]

    print()
    width = max((len(f"{b}.{k}") for b, k in total), default=20)
    print(f"{'(blob, key)':<{width}} {'n':>8} {'equal':>8} {'differs':>8} {'absent':>8}")
    carried_ok, allowed, violations = [], [], []
    for (blob, key), v in sorted(total.items()):
        name = f"{blob}.{key}"
        note = ""
        if v["differs"]:
            violations.append((name, v)); note = f"  {RED}<<< VALUE MISMATCH{RESET}"
        elif v["absent"]:
            if key in ALLOWED_UNCARRIED:
                allowed.append((name, v)); note = f"  {YELLOW}<<< allow-listed{RESET}"
            else:
                violations.append((name, v)); note = f"  {RED}<<< NOT CARRIED{RESET}"
        else:
            carried_ok.append(name)
        print(f"{name:<{width}} {v['n']:>8} {v['equal']:>8} {v['differs']:>8} {v['absent']:>8}{note}")

    print()
    print(f"  {len(carried_ok)} (blob, key) pair(s) carried losslessly")
    for name, v in allowed:
        key = name.split(".", 1)[1]
        print(f"  {YELLOW}allow-listed{RESET}: {name} ({v['absent']} occurrences) "
              f"-- {ALLOWED_UNCARRIED[key]}")
    if violations:
        print(f"\n{RED}RED{RESET}: {len(violations)} (blob, key) pair(s) not accounted for:")
        for name, v in violations:
            kind = "values differ" if v["differs"] else "never carried into position_data"
            print(f"    {name}: {kind} ({v['differs'] or v['absent']} of {v['n']})")
            if v["example"]:
                print(f"        e.g. {v['example']}")
        print("\n  Either carry the field in `frames.build_position_data`, or add it to "
              "ALLOWED_UNCARRIED with a reason the information is not lost.")
    if failed_sessions:
        print(f"\n{RED}RED{RESET}: {len(failed_sessions)} session(s) failed to check.")
    if violations or failed_sessions:
        return 1
    print(f"\n{GREEN}POSITION_DATA LOSSLESS{RESET}: every blob field is carried with an "
          f"equal value, except {len(allowed)} allow-listed. Safe to drop the blob columns.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
