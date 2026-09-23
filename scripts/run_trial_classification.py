#!/usr/bin/env python
"""Run trial classification for given subject(s) and date(s).

Thin CLI wrapper over hypnose_behavior.trial_classification.run.batch_analyze_sessions;
contains no analysis logic.

Examples
--------
  python scripts/run_trial_classification.py --subjids 53 --dates 20260528
  python scripts/run_trial_classification.py --subjids 53 58 --date-range 20260501 20260531
  python scripts/run_trial_classification.py --subjids 53 --ses 20
  python scripts/run_trial_classification.py --subjids 53 --index-range 1 9
  python scripts/run_trial_classification.py                      # all subjects, all dates

The six selectors intersect; none is required. `--index` here is the rank within
**rawdata**, which is not the rank within derivatives -- see `docs/DECISIONS.md`
section 32.
"""
import sys
import argparse
from pathlib import Path

# Make the package importable when running straight from the repo (no install needed).
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hypnose_helpers.cli.selector_args import add_selector_args
from hypnose_behavior.trial_classification.run import batch_analyze_sessions
from hypnose_behavior.qc.validate import validate_subject


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_selector_args(ap, index=True)
    ap.add_argument("--no-save", action="store_true", help="do not write derivatives")
    ap.add_argument("--save-csv", action="store_true",
                    help="also write a human-readable CSV of every table (parquet is always written)")
    ap.add_argument("--no-summary", action="store_true", help="suppress merged summary")
    ap.add_argument("--verbose", action="store_true", help="verbose per-run logging")
    args = ap.parse_args()

    # Pre-flight validation: drop subjects/dates with no data (clear message, no crash later).
    subjids = args.subjids
    if subjids:
        subjids = [s for s in subjids if validate_subject(s, args.dates)["ok"]]
        if not subjids:
            print("Nothing to run after validation.")
            return 1

    batch_analyze_sessions(
        subjids=subjids,
        dates=args.dates,
        date_range=args.date_range,
        ses=args.ses,
        index=args.index,
        ses_range=args.ses_range,
        index_range=args.index_range,
        save=not args.no_save,
        save_csv=args.save_csv,
        print_summary=not args.no_summary,
        verbose=args.verbose,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
