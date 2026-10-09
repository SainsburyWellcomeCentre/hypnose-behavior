"""Run trial classification AND metric analysis for given subject(s)/session(s).

Thin CLI wrapper: classifies first (writes derivatives), then runs metrics over
those results. Contains no analysis logic.

Examples
--------
  hypnose-batch-process -s 53 -d 20260528
  hypnose-batch-process --subjids 53 58 --date-range 20260501 20260531 --protocol singrew
  hypnose-batch-process -sub 053,058 -date-range 20260501-20260531
  hypnose-batch-process -s 53 -ses 20
  hypnose-batch-process -s 63 64 --ses-range 1-10
  hypnose-batch-process --sub-range 60-66 -d 20260921      # subjects without that date are skipped
  hypnose-batch-process                                  # all subjects, all dates

Subjects: -s / --sub / --subs / --subj / --subject(s) / --subjid(s), or --sub-range. Dates: -d / --date(s),
--date-range. Sessions: --ses / --session(s), --ses-range / --session-range. Every long
flag also works with one dash; the selectors intersect. See
`hypnose_helpers.cli.selector_args`.

`--index` / `--index-range` are refused: this resolves rawdata for classification and
derivatives for metrics, and an index is a rank *within* a tree. See
`docs/DECISIONS.md` section 32.
"""
import argparse

from hypnose_helpers.cli.selector_args import add_selector_args

from hypnose_behavior.trial_classification.run import batch_analyze_sessions
from hypnose_behavior.metric_analysis.run import batch_run_all_metrics_with_merge
from hypnose_behavior.qc.validate import validate_subject


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="hypnose-batch-process", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_selector_args(ap)
    # Accepted by the parser only so the refusal below can name them; see the module
    # docstring. Silently ignoring them, or omitting them so argparse says "unrecognized
    # arguments", would both be worse than saying why.
    ap.add_argument("--index", "-index", nargs="*", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--index-range", "-index-range", nargs="*", default=None,
                    help=argparse.SUPPRESS)
    ap.add_argument("--protocol", default=None, help="metrics: only sessions whose stage name contains this string")
    ap.add_argument("--no-save", action="store_true", help="do not write derivatives / metrics")
    ap.add_argument("--save-csv", action="store_true",
                    help="also write a human-readable CSV of every table (parquet is always written)")
    ap.add_argument("--verbose", action="store_true", help="verbose per-run logging (classification)")
    args = ap.parse_args(argv)

    # DECISIONS.md sections 8 and 32: an index is a rank within one tree, and this script
    # resolves rawdata for classification and derivatives for metrics.
    if args.index is not None or args.index_range is not None:
        print("batch_process does not accept --index / --index-range: indices do not "
              "resolve cleanly between rawdata and derivatives.\n"
              "Use --ses / --dates, or run the two scripts separately.")
        return 2

    subjids = args.subjids
    if subjids:
        subjids = [s for s in subjids if validate_subject(s, args.dates)["ok"]]
        if not subjids:
            print("Nothing to run after validation.")
            return 1

    selectors = dict(dates=args.dates, date_range=args.date_range,
                     ses=args.ses, ses_range=args.ses_range)

    print("=== Trial classification ===")
    batch_analyze_sessions(
        subjids=subjids, **selectors,
        save=not args.no_save, save_csv=args.save_csv, print_summary=True, verbose=args.verbose,
    )
    print("\n=== Metric analysis ===")
    batch_run_all_metrics_with_merge(
        subjids=subjids, **selectors, protocol=args.protocol,
        save_txt=not args.no_save, save_json=not args.no_save, verbose=not args.verbose,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
