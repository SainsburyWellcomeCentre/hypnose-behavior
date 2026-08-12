#!/usr/bin/env python
"""Run trial classification for given subject(s) and date(s).

Thin CLI wrapper over hypnose_behavior.trial_classification.run.batch_analyze_sessions;
contains no analysis logic.

Examples
--------
  python scripts/run_trial_classification.py --subjids 53 --dates 20260528
  python scripts/run_trial_classification.py --subjids 53 58 --date-range 20260501 20260531
  python scripts/run_trial_classification.py                      # all subjects, all dates
"""
import sys
import argparse
from pathlib import Path

# Make the package importable when running straight from the repo (no install needed).
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hypnose_behavior.trial_classification.run import batch_analyze_sessions
from hypnose_behavior.qc.validate import validate_subject


def _resolve_dates(args):
    if args.date_range:
        return (args.date_range[0], args.date_range[1])  # (start, end) -> range
    if args.dates:
        return list(args.dates)
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subjids", nargs="*", type=int, default=None, help="subject id(s); default: all")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--dates", nargs="*", type=int, default=None, help="specific date(s) YYYYMMDD")
    g.add_argument("--date-range", nargs=2, type=int, metavar=("START", "END"), help="inclusive YYYYMMDD range")
    ap.add_argument("--no-save", action="store_true", help="do not write derivatives")
    ap.add_argument("--save-csv", action="store_true",
                    help="also write a human-readable CSV of every table (parquet is always written)")
    ap.add_argument("--no-summary", action="store_true", help="suppress merged summary")
    ap.add_argument("--verbose", action="store_true", help="verbose per-run logging")
    args = ap.parse_args()

    dates = _resolve_dates(args)

    # Pre-flight validation: drop subjects/dates with no data (clear message, no crash later).
    subjids = args.subjids
    if subjids:
        check_dates = list(args.dates) if args.dates else None
        subjids = [s for s in subjids if validate_subject(s, check_dates)["ok"]]
        if not subjids:
            print("Nothing to run after validation.")
            return 1

    batch_analyze_sessions(
        subjids=subjids,
        dates=dates,
        save=not args.no_save,
        save_csv=args.save_csv,
        print_summary=not args.no_summary,
        verbose=args.verbose,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
