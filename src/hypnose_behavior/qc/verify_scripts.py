#!/usr/bin/env python
"""Verify the terminal entry-point scripts reproduce byte-identical results.

regression.py checks the pipeline *functions*. This checks the actual CLI scripts
in scripts/ -- the production entry points -- by invoking them via subprocess into
a throwaway derivatives dir and md5-comparing trial_data + metrics against the same
fixtures. This exercises the CLI argument wiring and the batch_* loops (where the
int-vs-str dates bug hid), which the function-level regression does not touch.

  run_trial_classification.py  -> trial_data md5 must match fixture
  run_metrics_analysis.py      -> runs on those derivatives; re-derived metrics md5 must match
  batch_process.py             -> both, on one session (verifies the chained composition)

Usage:
  python src/hypnose_behavior/qc/verify_scripts.py            # all fixture sessions
  python src/hypnose_behavior/qc/verify_scripts.py 053:20260520 053:20260429   # subset

Exit 0 = GREEN; 1 = mismatch / script failure. Run in the pinned conda env.
"""
from __future__ import annotations

import os
import io
import sys
import contextlib
import subprocess
import tempfile
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
SCRIPTS = REPO / "scripts"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "src"))

import _common  # canonicalization + md5 helpers (single import surface)
import json

PY = sys.executable


def _load_fixtures(filter_keys):
    sessions = yaml.safe_load((HERE / "sessions.yml").read_text())["sessions"]
    out = []
    for s in sessions:
        key = f"{str(s['subjid']).zfill(3)}:{s['date']}"
        if filter_keys and key not in filter_keys:
            continue
        fpath = HERE / "fixtures" / f"sub-{str(s['subjid']).zfill(3)}_date-{s['date']}.json"
        if not fpath.exists():
            print(f"  [skip] no fixture for {key}")
            continue
        out.append((str(s["subjid"]), str(s["date"]), s.get("label", ""), json.loads(fpath.read_text())))
    return out


def _run_cli(script: str, subjid: str, date: str, deriv: Path, extra=()) -> subprocess.CompletedProcess:
    env = {**os.environ, "HYPNOSE_DERIVATIVES_ROOT": str(deriv)}
    cmd = [PY, str(SCRIPTS / script), "--subjids", str(subjid), "--dates", str(date), *extra]
    return subprocess.run(cmd, env=env, capture_output=True, text=True)


def _trial_data_md5(deriv: Path, date: str):
    m = list(deriv.glob(f"**/ses-*_date-{date}/saved_analysis_results/trial_data.csv"))
    return _common._md5(_common._canonical_trial_data(m[0])) if m else None


def _metrics_md5_from_derivatives(subjid: str, date: str, deriv: Path):
    """Re-derive the metrics dict from the script-produced derivatives and md5 it."""
    _common._redirect_derivatives(deriv)  # set env + clear cached path lookups
    from hypnose_behavior.io.load_results import load_session_results
    from hypnose_behavior.metric_analysis.run import run_all_metrics
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        results = load_session_results(str(subjid).zfill(3), str(date))
        metrics = run_all_metrics(results, save_txt=False, save_json=False, save_tables=False)
    return _common._md5(_common._canonical_metrics(metrics))


def main() -> int:
    fixtures = _load_fixtures(set(sys.argv[1:]))
    if not fixtures:
        print("No fixture sessions to verify.")
        return 1

    red = 0
    batch_done = False
    print(f"Verifying scripts against {len(fixtures)} fixture session(s)...\n")
    for subjid, date, label, fx in fixtures:
        tag = f"sub-{subjid} {date} ({label})"

        with tempfile.TemporaryDirectory(prefix="hyp_vscripts_") as tmp:
            deriv = Path(tmp)
            # 1) run_trial_classification.py -> trial_data md5
            r = _run_cli("run_trial_classification.py", subjid, date, deriv, ["--no-summary", "--save-csv"])
            if r.returncode != 0:
                print(f"  [ERROR] run_trial_classification {tag}: exit {r.returncode}\n{r.stderr[-500:]}"); red += 1; continue
            td = _trial_data_md5(deriv, date)
            if td == fx["trial_data"]:
                print(f"  [green] run_trial_classification {tag} trial_data ok ({td[:8]})")
            else:
                print(f"  [RED]   run_trial_classification {tag} trial_data: exp {fx['trial_data'][:8]} got {str(td)[:8]}"); red += 1

            # 2) run_metrics_analysis.py -> runs on those derivatives; re-derived metrics md5
            r = _run_cli("run_metrics_analysis.py", subjid, date, deriv, ["--quiet"])
            if r.returncode != 0:
                print(f"  [ERROR] run_metrics_analysis {tag}: exit {r.returncode}\n{r.stderr[-500:]}"); red += 1; continue
            mm = _metrics_md5_from_derivatives(subjid, date, deriv)
            if mm == fx["metrics"]:
                print(f"  [green] run_metrics_analysis     {tag} metrics ok ({mm[:8]})")
            else:
                print(f"  [RED]   run_metrics_analysis     {tag} metrics: exp {fx['metrics'][:8]} got {str(mm)[:8]}"); red += 1

        # 3) batch_process.py on the FIRST session only (verifies the chained composition)
        if not batch_done:
            batch_done = True
            with tempfile.TemporaryDirectory(prefix="hyp_vbatch_") as tmp:
                deriv = Path(tmp)
                r = _run_cli("batch_process.py", subjid, date, deriv, ["--save-csv"])
                if r.returncode != 0:
                    print(f"  [ERROR] batch_process {tag}: exit {r.returncode}\n{r.stderr[-500:]}"); red += 1
                else:
                    td = _trial_data_md5(deriv, date)
                    mm = _metrics_md5_from_derivatives(subjid, date, deriv)
                    ok = (td == fx["trial_data"]) and (mm == fx["metrics"])
                    print(f"  [{'green' if ok else 'RED'}]{'  ' if ok else '   '}batch_process            {tag} trial_data+metrics {'ok' if ok else 'MISMATCH'}")
                    red += 0 if ok else 1

    print()
    if red:
        print(f"SCRIPT VERIFY RED: {red} mismatch/failure(s).")
        return 1
    print("SCRIPT VERIFY GREEN: scripts reproduce byte-identical trial_data + metrics.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
