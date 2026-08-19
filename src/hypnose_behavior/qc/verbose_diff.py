"""Old-vs-new diff of what `trial_classification` *prints*.

`regression.py` fingerprints `trial_data` and the metrics dict. It never looks at stdout, so
the several hundred lines of verbose summary that `classify_trials`, `analyze_response_times`
and `abortion_classification` emit are invisible to it -- and those summaries are how the lab
reads a session at the bench. Splitting a print block is exactly the kind of change that keeps
`trial_data` byte-identical while quietly dropping a line or changing an indent.

Written for Phase 6a, and deliberately the same shape as `plot_regression.py`: a **two-tree
diff**, not a golden master. It runs the same sessions with `verbose=True` against a git
revision and against the working tree, and diffs the captured stdout line by line.

    python src/hypnose_behavior/qc/verbose_diff.py                  # working tree vs HEAD
    python src/hypnose_behavior/qc/verbose_diff.py --rev main       # ... vs another revision
    python src/hypnose_behavior/qc/verbose_diff.py 061:20260729     # limit to some sessions

Timestamps and paths vary between runs of the same code, so lines matching `VOLATILE` are
normalised before comparison. Everything else is compared exactly.
"""
from __future__ import annotations

import argparse
import contextlib
import difflib
import io
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

_HERE = Path(__file__).resolve()
_SRC = _HERE.parents[2]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from hypnose_behavior.qc import _common  # noqa: E402
from hypnose_behavior.io import paths as _paths  # noqa: E402

REPO = _SRC.parent

# Absolute paths and the temp derivatives dir differ between the two runs by construction.
VOLATILE = [
    (re.compile(r"/(?:private/)?(?:var|tmp)/[^\s'\"]+"), "<TMP>"),
    (re.compile(r"/Users/[^\s'\"]+"), "<PATH>"),
    (re.compile(r"/Volumes/[^\s'\"]+"), "<PATH>"),
]


def _normalise(text: str) -> list[str]:
    lines = []
    for line in text.splitlines():
        for pattern, replacement in VOLATILE:
            line = pattern.sub(replacement, line)
        lines.append(line.rstrip())
    return lines


def capture(subjid, date) -> str:
    """Run one session's classification with verbose=True and return everything it printed."""
    from hypnose_behavior.trial_classification.run import analyze_session_multi_run_by_id_date

    with tempfile.TemporaryDirectory(prefix="hyp_verbose_") as tmp_str:
        _common._redirect_derivatives(Path(tmp_str))
        sink = io.StringIO()
        with contextlib.redirect_stdout(sink):
            analyze_session_multi_run_by_id_date(
                str(subjid), str(date), verbose=True, save=True, print_summary=True
            )
        return sink.getvalue()


def _capture_all(sessions) -> dict:
    out = {}
    for s in sessions:
        subjid, date = str(s["subjid"]), str(s["date"])
        key = f"{subjid}:{date}"
        try:
            out[key] = capture(subjid, date)
        except Exception as e:  # a session that cannot run is a result, not a crash
            out[key] = f"<<VERBOSE_DIFF_ERROR>> {type(e).__name__}: {e}"
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sessions", nargs="*", help="limit to these 'subjid:date' keys")
    ap.add_argument("--rev", default="HEAD", help="git revision to compare against (default HEAD)")
    ap.add_argument("--context", type=int, default=2, help="diff context lines")
    ap.add_argument("--_capture-to", help=argparse.SUPPRESS)
    args = ap.parse_args()

    sessions = _load_sessions()
    if args.sessions:
        targets = set(args.sessions)
        sessions = [s for s in sessions
                    if f"{str(s['subjid']).zfill(3)}:{s['date']}" in targets
                    or f"{s['subjid']}:{s['date']}" in targets]

    # Child mode: we are running inside the temporary worktree; just capture and write out.
    if args._capture_to:
        Path(args._capture_to).write_text(json.dumps(_capture_all(sessions)))
        return 0

    print(f"Capturing verbose output for {len(sessions)} session(s) from the working tree...")
    current = _capture_all(sessions)

    with tempfile.TemporaryDirectory(prefix="hyp_verbose_tree_") as tree_str:
        tree = Path(tree_str) / "wt"
        print(f"Checking out {args.rev} into a temporary worktree...")
        subprocess.run(["git", "-C", str(REPO), "worktree", "add", "--detach", str(tree), args.rev],
                       check=True, capture_output=True)
        try:
            out_json = Path(tree_str) / "old.json"
            # This script is usually newer than the revision under test, so run *this* copy of
            # it against the old tree's package rather than whatever it shipped with.
            child = tree / "src/hypnose_behavior/qc/verbose_diff.py"
            child.write_text(_HERE.read_text())
            cmd = [sys.executable, "-u", str(child),
                   "--_capture-to", str(out_json), *args.sessions]
            env = dict(os.environ)
            env["PYTHONPATH"] = str(tree / "src")
            # The data-location profile lives in a git-ignored config, so a fresh worktree
            # cannot resolve the raw data roots. Hand the child the roots this tree resolved.
            env.setdefault("HYPNOSE_RAWDATA_ROOT", str(_paths.get_rawdata_root()))
            env.setdefault("HYPNOSE_SERVER_ROOT", str(_paths.get_server_root()))
            print(f"Capturing verbose output from {args.rev}...")
            subprocess.run(cmd, check=True, env=env)
            old = json.loads(out_json.read_text())
        finally:
            subprocess.run(["git", "-C", str(REPO), "worktree", "remove", "--force", str(tree)],
                           check=False, capture_output=True)

    failures = 0
    for key in current:
        a = _normalise(old.get(key, "<<MISSING>>"))
        b = _normalise(current[key])
        if a == b:
            print(f"  [green] {key}: stdout identical ({len(b)} lines)")
            continue
        failures += 1
        print(f"  [RED]   {key}: stdout differs")
        for line in difflib.unified_diff(a, b, fromfile=args.rev, tofile="working tree",
                                         lineterm="", n=args.context):
            print(f"      {line}")

    if failures:
        print(f"\nVERBOSE DIFF RED: {failures} session(s) print differently.")
        return 1
    print("\nVERBOSE DIFF GREEN: printed output identical for every session.")
    return 0


def _load_sessions():
    import yaml
    data = yaml.safe_load((_HERE.parent / "sessions.yml").read_text())
    return data["sessions"] if isinstance(data, dict) else data


if __name__ == "__main__":
    raise SystemExit(main())
