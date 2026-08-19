#!/usr/bin/env python
"""Run behavioural metric analysis for given subject(s) and date(s).

Thin CLI wrapper over hypnose_behavior.metric_analysis.run.batch_run_all_metrics_with_merge;
contains no analysis logic. Metrics read saved trial-classification results from the
derivatives tree, so run trial classification first.

Examples
--------
  python scripts/run_metrics_analysis.py --subjids 53 --dates 20260528
  python scripts/run_metrics_analysis.py --subjids 53 58 --date-range 20260501 20260531 --protocol singrew
  python scripts/run_metrics_analysis.py --subjids 53 --ses 20
  python scripts/run_metrics_analysis.py --subjids 53 --index-range 1 9
  python scripts/run_metrics_analysis.py                      # all subjects, all dates

The six selectors intersect; none is required. `--index` here is the rank within
**derivatives** -- i.e. among the *analysed* sessions, which is not the rank within
rawdata that `run_trial_classification.py` uses. See `docs/DECISIONS.md` section 32.
"""
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

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
    ap.add_argument("--index", nargs="*", default=None,
                    help="session index/indices: gap-free chronological rank among ANALYSED sessions")
    ap.add_argument("--ses-range", nargs=2, metavar=("START", "END"), default=None,
                    help="inclusive ses range")
    ap.add_argument("--index-range", nargs=2, metavar=("START", "END"), default=None,
                    help="inclusive session-index range (derivatives)")
    ap.add_argument("--protocol", default=None, help="only sessions whose stage name contains this string")
    ap.add_argument("--no-save", action="store_true", help="do not write metrics txt/json")
    ap.add_argument("--quiet", action="store_true", help="suppress per-session logging")
    args = ap.parse_args()

    dates = _resolve_dates(args)

    subjids = args.subjids
    if subjids:
        check_dates = list(args.dates) if args.dates else None
        subjids = [s for s in subjids if validate_subject(s, check_dates)["ok"]]
        if not subjids:
            print("Nothing to run after validation.")
            return 1

    batch_run_all_metrics_with_merge(
        subjids=subjids,
        dates=dates,
        ses=args.ses,
        index=args.index,
        ses_range=args.ses_range,
        index_range=args.index_range,
        protocol=args.protocol,
        save_txt=not args.no_save,
        save_json=not args.no_save,
        verbose=not args.quiet,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
