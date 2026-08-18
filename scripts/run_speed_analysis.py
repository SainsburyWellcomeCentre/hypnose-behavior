#!/usr/bin/env python
"""Compute movement speed analysis for given subject(s) and session(s).

Thin CLI wrapper over
hypnose_behavior.metric_analysis.movement.speed_analysis.run_speed_analysis_batch;
contains no analysis logic. Writes one `speed_analysis.parquet` per session, beside
that session's `trial_data.parquet`, from SLEAP tracking plus the saved trial
classification -- so run trial classification first.

This is the only place the speed threshold is computed. The movement plotters
(`visualization/movement/speed.py`) **read** the file it writes and report the
session when it is absent; they no longer recompute it, because two derivations of
one quantity is how two figures come to disagree (`docs/DECISIONS.md` section 14).

Examples
--------
  python scripts/run_speed_analysis.py --subjids 57 --dates 20260717
  python scripts/run_speed_analysis.py --subjids 57 58 59 --date-range 20260701 20260731
  python scripts/run_speed_analysis.py --subjids 57 --ses 20
  python scripts/run_speed_analysis.py --subjids 57 --index-range 1 9
  python scripts/run_speed_analysis.py --fa-labels FA_time_in FA_time_out
  python scripts/run_speed_analysis.py                      # all subjects, all sessions

The six selectors intersect; none is required. **`--index` is the rank among
*analysed* sessions**, because this resolves the derivatives tree -- section 32
measured that to be a different session from rawdata's Nth on 7 of 8 subjects, so
do not carry an index across from `run_trial_classification.py`. `--ses` is
tree-stable and is the one to use when chaining.

A session that already has `speed_analysis.parquet` is recomputed and overwritten;
there is no skip-if-present flag, because the parameters below change what the file
holds and silently keeping a file computed with different ones is the failure this
whole path exists to avoid.
"""
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hypnose_behavior.metric_analysis.movement.speed_analysis import (
    run_speed_analysis_batch,
)


def _resolve_dates(args):
    if args.date_range:
        return (args.date_range[0], args.date_range[1])
    if args.dates:
        return list(args.dates)
    return None


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subjids", nargs="*", type=int, default=None,
                    help="subject id(s); default: all")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--dates", nargs="*", type=int, default=None,
                   help="specific date(s) YYYYMMDD")
    g.add_argument("--date-range", nargs=2, type=int, metavar=("START", "END"),
                   help="inclusive YYYYMMDD range")
    ap.add_argument("--ses", nargs="*", default=None,
                    help="session number(s) as written in the ses-NNN directory "
                         "(40, 040 or ses-040)")
    ap.add_argument("--index", nargs="*", default=None,
                    help="session index/indices: gap-free chronological rank among "
                         "ANALYSED sessions")
    ap.add_argument("--ses-range", nargs=2, metavar=("START", "END"), default=None,
                    help="inclusive ses range")
    ap.add_argument("--index-range", nargs=2, metavar=("START", "END"), default=None,
                    help="inclusive session-index range (derivatives)")
    ap.add_argument("--bin-ms", type=int, default=100,
                    help="speed bin width in ms (default 100)")
    ap.add_argument("--pre-buffer-s", type=float, default=1.0,
                    help="seconds before last poke-out to include (default 1.0)")
    ap.add_argument("--fa-labels", nargs="*", default=None,
                    help="FA labels to include, e.g. FA_time_in FA_time_out "
                         "(default: FA_time_in)")
    ap.add_argument("--mode", choices=("mean", "max"), default="mean",
                    help="per-bin aggregation (default mean)")
    ap.add_argument("--no-threshold", action="store_true",
                    help="report baseline mu/sigma but compute no combined threshold")
    ap.add_argument("--threshold-alpha", type=float, default=10.0,
                    help="multiplier for mu in vthresh = max(alpha*mu, mu+beta*sigma)")
    ap.add_argument("--threshold-beta", type=float, default=10.0,
                    help="multiplier for sigma in vthresh")
    ap.add_argument("--quiet", action="store_true", help="suppress per-session logging")
    args = ap.parse_args()

    processed = run_speed_analysis_batch(
        subjids=args.subjids,
        dates=_resolve_dates(args),
        ses=args.ses,
        index=args.index,
        ses_range=args.ses_range,
        index_range=args.index_range,
        bin_ms=args.bin_ms,
        pre_buffer_s=args.pre_buffer_s,
        fa_label_filter=args.fa_labels,
        mode=args.mode,
        threshold=not args.no_threshold,
        threshold_alpha=args.threshold_alpha,
        threshold_beta=args.threshold_beta,
        verbose=not args.quiet,
    )
    # Exit non-zero when a selection matched nothing: a batch that silently
    # processes zero sessions looks identical to one that succeeded.
    if not processed:
        print("No sessions processed.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
