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

from hypnose_helpers.cli.selector_args import add_selector_args
from hypnose_behavior.metric_analysis.run import batch_run_all_metrics_with_merge
from hypnose_behavior.qc.validate import validate_subject


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_selector_args(ap, index=True)
    ap.add_argument("--protocol", default=None, help="only sessions whose stage name contains this string")
    ap.add_argument("--no-save", action="store_true", help="do not write metrics txt/json")
    ap.add_argument("--quiet", action="store_true", help="suppress per-session logging")
    args = ap.parse_args()

    subjids = args.subjids
    if subjids:
        subjids = [s for s in subjids if validate_subject(s, args.dates)["ok"]]
        if not subjids:
            print("Nothing to run after validation.")
            return 1

    batch_run_all_metrics_with_merge(
        subjids=subjids,
        dates=args.dates,
        date_range=args.date_range,
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
