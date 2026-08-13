#!/usr/bin/env python
"""Golden-master regression: confirm the pipeline output is unchanged.

Fingerprints six things for each coverage session in ``sessions.yml`` and compares
them against stored baselines in ``fixtures/``:

  ``trial_data``          the canonical CSV
  ``metrics``             the reported metrics dict (``run.REPORT``, 25 entries)
  ``position_data``       the per-position side-table, as written
  ``metrics_by_trial``    the per-trial metric table, as written
  ``metrics_by_poke``     the per-poke metric table, as written
  ``unreported_metrics``  the 16 registered metrics ``REPORT`` does not save

The last four were added in Phase 7b.6 and closed a real gap: the three tables were
written by code no gate read back, and 18 registered metrics were computed by code no
gate ran, so the GREENs that landed them measured *additivity* rather than coverage.

On a mismatch it reports *what* changed (added / removed / changed columns, and
added / removed / changed metric keys), so an intended change is easy to confirm as
"everything identical except X".

Usage
-----
  # write/refresh baselines from the CURRENT code -- run on a known-good baseline,
  # BEFORE making changes (optionally limit to specific sessions):
  python src/hypnose_behavior/qc/regression.py --generate
  python src/hypnose_behavior/qc/regression.py --generate 060:20260601

  # check current code against the baselines (optionally limit to sessions):
  python src/hypnose_behavior/qc/regression.py
  python src/hypnose_behavior/qc/regression.py 053:20260528

Sessions are given as ``subjid:date`` keys. Compare and generate only act on the
sessions listed in ``sessions.yml`` (intersected with any keys you pass);
fixtures for sessions removed from sessions.yml are simply ignored (not deleted).

Exit code 0 == GREEN (all match); 1 == RED (mismatch / failure). Run in the
pinned conda env recorded in fixtures/env.json. Reads real rawdata; writes only
to a throwaway temp dir (never the server).
"""
from __future__ import annotations

import sys
import json
import argparse
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
FIXTURES = HERE / "fixtures"
REPO = HERE.parents[2]

# Make both the harness module and the src package importable without install.
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "src"))

import _common  # noqa: E402


def _load_sessions() -> list[dict]:
    with open(HERE / "sessions.yml") as fh:
        data = yaml.safe_load(fh)
    return data["sessions"]


def _fixture_path(subjid, date) -> Path:
    return FIXTURES / f"sub-{str(subjid).zfill(3)}_date-{date}.json"


def _select(sessions: list[dict], targets: set[str]) -> list[dict]:
    """Filter sessions by 'subjid:date' keys (zero-padded subjid). Empty = all."""
    if not targets:
        return sessions
    return [s for s in sessions
            if f"{str(s['subjid']).zfill(3)}:{s['date']}" in targets
            or f"{s['subjid']}:{s['date']}" in targets]


def generate(targets: set[str]) -> int:
    sessions = _select(_load_sessions(), targets)
    FIXTURES.mkdir(parents=True, exist_ok=True)
    print(f"Generating fixtures for {len(sessions)} session(s)"
          f"{' (filtered)' if targets else ''}...\n")
    failures = 0
    for s in sessions:
        subjid, date, label = s["subjid"], s["date"], s.get("label", "")
        try:
            fp = _common.fingerprint_session(subjid, date)
        except Exception as e:
            print(f"  [FAIL] sub-{subjid} {date} ({label}): {e}")
            failures += 1
            continue
        payload = {"subjid": str(subjid), "date": str(date), "label": label, **fp}
        _fixture_path(subjid, date).write_text(json.dumps(payload, indent=2))
        print(f"  [OK]   sub-{subjid} {date} ({label}): "
              f"trial_data={fp['trial_data'][:8]} metrics={fp['metrics'][:8]}")
    (FIXTURES / "env.json").write_text(json.dumps(_common.env_fingerprint(), indent=2))
    print(f"\nWrote fixtures to {FIXTURES} (env.json recorded). "
          f"Existing fixtures not regenerated are left untouched.")
    if failures:
        print(f"{failures} session(s) failed to fingerprint -- fix before relying on fixtures.")
        return 1
    return 0


def compare(targets: set[str]) -> int:
    sessions = _select(_load_sessions(), targets)
    print(f"Checking {len(sessions)} session(s) against fixtures...\n")
    red = 0
    for s in sessions:
        subjid, date, label = s["subjid"], s["date"], s.get("label", "")
        fpath = _fixture_path(subjid, date)
        if not fpath.exists():
            print(f"  [MISSING FIXTURE] sub-{subjid} {date} ({label}) -- run --generate first")
            red += 1
            continue
        expected = json.loads(fpath.read_text())
        try:
            got = _common.fingerprint_session(subjid, date)
        except Exception as e:
            print(f"  [ERROR] sub-{subjid} {date} ({label}): {e}")
            red += 1
            continue
        # overall md5 is the pass/fail signal; on mismatch, report what changed
        for key, parts_key, label_word in (
            ("trial_data", "trial_data_columns", "column"),
            ("metrics", "metrics_keys", "metric"),
            # Added in Phase 7b.6. Before it, the three written tables and the 18
            # unreported metrics had no coverage at all.
            ("position_data", "position_data_columns", "column"),
            ("metrics_by_trial", "metrics_by_trial_columns", "column"),
            ("metrics_by_poke", "metrics_by_poke_columns", "column"),
            ("unreported_metrics", "unreported_metrics_keys", "metric"),
        ):
            # A fixture written before this key existed carries no baseline for it.
            # Say so rather than raising KeyError, and count it -- "no baseline" is not
            # a pass.
            if key not in expected:
                print(f"  [NO BASELINE] sub-{subjid} {date} ({label}) {key}: "
                      f"fixture predates this check -- run --generate")
                red += 1
                continue
            if got[key] != expected[key]:
                print(f"  [RED]  sub-{subjid} {date} ({label}) {key}: "
                      f"expected {expected[key][:8]} got {got[key][:8]}")
                if parts_key in expected and parts_key in got:
                    for line in _common.diff_report(label_word, expected[parts_key], got[parts_key]):
                        print(line)
                else:
                    print(f"        (regenerate fixtures to get per-{label_word} diffs)")
                red += 1
            else:
                print(f"  [green] sub-{subjid} {date} ({label}) {key} ok ({got[key][:8]})")
    print()
    if red:
        print(f"REGRESSION RED: {red} mismatch(es). Output changed -- see the +/-/~ lines above.")
        return 1
    print("REGRESSION GREEN: all outputs byte-identical to baseline.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--generate", action="store_true",
                    help="write fixtures from current code (run on the baseline, before changes)")
    ap.add_argument("targets", nargs="*",
                    help="optional 'subjid:date' keys to limit to (e.g. 053:20260528); default: all in sessions.yml")
    args = ap.parse_args()
    targets = set(args.targets)
    return generate(targets) if args.generate else compare(targets)


if __name__ == "__main__":
    raise SystemExit(main())
