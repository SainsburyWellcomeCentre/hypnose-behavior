#!/usr/bin/env python
"""Run trial classification AND metric analysis for given subject(s)/date(s).

Thin CLI wrapper: classifies first (writes derivatives), then runs metrics over
those results. Contains no analysis logic.

Examples
--------
  python scripts/batch_process.py --subjids 53 --dates 20260528
  python scripts/batch_process.py --subjids 53 58 --date-range 20260501 20260531 --protocol singrew
  python scripts/batch_process.py --subjids 53 --ses 20
  python scripts/batch_process.py                      # all subjects, all dates

`--ses` / `--dates` / `--date-range` / `--ses-range` name a session by an intrinsic label
and are stable across both trees, so they pass through to both halves. `--index` /
`--index-range` are refused: this script resolves rawdata for classification and
derivatives for metrics, and an index is a rank *within* a tree. See
`docs/DECISIONS.md` section 32.
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
    ap.add_argument("--ses", nargs="*", default=None,
                    help="session number(s) as written in the ses-NNN directory (40, 040 or ses-040)")
    ap.add_argument("--ses-range", nargs=2, metavar=("START", "END"), default=None,
                    help="inclusive ses range")
    # Accepted by the parser only so the refusal below can name them; see the module
    # docstring. Silently ignoring them, or omitting them so argparse says "unrecognized
    # arguments", would both be worse than saying why.
    ap.add_argument("--index", nargs="*", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--index-range", nargs=2, metavar=("START", "END"), default=None,
                    help=argparse.SUPPRESS)
    ap.add_argument("--protocol", default=None, help="metrics: only sessions whose stage name contains this string")
    ap.add_argument("--no-save", action="store_true", help="do not write derivatives / metrics")
    ap.add_argument("--save-csv", action="store_true",
                    help="also write a human-readable CSV of every table (parquet is always written)")
    ap.add_argument("--verbose", action="store_true", help="verbose per-run logging (classification)")
    args = ap.parse_args()

    # DECISIONS.md sections 8 and 32: an index is a rank within one tree, and this script
    # resolves rawdata for classification and derivatives for metrics.
    if args.index or args.index_range:
        print("batch_process.py does not accept --index / --index-range: indices do not "
              "resolve cleanly between rawdata and derivatives.\n"
              "Use --ses / --dates, or run the two scripts separately.")
        return 2

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
        subjids=subjids, dates=dates, ses=args.ses, ses_range=args.ses_range,
        save=not args.no_save, save_csv=args.save_csv, print_summary=True, verbose=args.verbose,
    )
    print("\n=== Metric analysis ===")
    batch_run_all_metrics_with_merge(
        subjids=subjids, dates=dates, ses=args.ses, ses_range=args.ses_range,
        protocol=args.protocol,
        save_txt=not args.no_save, save_json=not args.no_save, verbose=not args.verbose,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
