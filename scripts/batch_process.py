#!/usr/bin/env python
"""Run trial classification AND metric analysis for given subject(s)/date(s).

Thin CLI wrapper: classifies first (writes derivatives), then runs metrics over
those results. Contains no analysis logic.

Examples
--------
  python scripts/batch_process.py --subjids 53 --dates 20260528
  python scripts/batch_process.py --subjids 53 58 --date-range 20260501 20260531 --protocol singrew
  python scripts/batch_process.py                      # all subjects, all dates
"""
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hypnose_behavior.trial_classification.run import batch_analyze_sessions
from hypnose_behavior.metric_analysis.run import batch_run_all_metrics_with_merge
from hypnose_behavior.qc.validate import validate_subject


def _resolve_dates(args):
    if args.date_range:
        return (args.date_range[0], args.date_range[1])
    if args.dates:
        return list(args.dates)
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subjids", nargs="*", type=int, default=None, help="subject id(s); default: all")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--dates", nargs="*", type=int, default=None, help="specific date(s) YYYYMMDD")
    g.add_argument("--date-range", nargs=2, type=int, metavar=("START", "END"), help="inclusive YYYYMMDD range")
    ap.add_argument("--protocol", default=None, help="metrics: only sessions whose stage name contains this string")
    ap.add_argument("--no-save", action="store_true", help="do not write derivatives / metrics")
    ap.add_argument("--save-csv", action="store_true",
                    help="also write a human-readable CSV of every table (parquet is always written)")
    ap.add_argument("--verbose", action="store_true", help="verbose per-run logging (classification)")
    args = ap.parse_args()

    dates = _resolve_dates(args)

    subjids = args.subjids
    if subjids:
        check_dates = list(args.dates) if args.dates else None
        subjids = [s for s in subjids if validate_subject(s, check_dates)["ok"]]
        if not subjids:
            print("Nothing to run after validation.")
            return 1

    print("=== Trial classification ===")
    batch_analyze_sessions(
        subjids=subjids, dates=dates,
        save=not args.no_save, save_csv=args.save_csv, print_summary=True, verbose=args.verbose,
    )
    print("\n=== Metric analysis ===")
    batch_run_all_metrics_with_merge(
        subjids=subjids, dates=dates, protocol=args.protocol,
        save_txt=not args.no_save, save_json=not args.no_save, verbose=not args.verbose,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
