#!/usr/bin/env python
"""Peek at a session's saved parquet tables: what is in this file.

Thin CLI wrapper over hypnose_behavior.io.parquet_peek.peek; contains no analysis
logic and writes nothing. Takes `--subjids` / `--dates` like the other entry points
and resolves them against the *derivatives* tree, so it reads what trial
classification and metric analysis already wrote.

Three views, narrowing:

  # every table in the session: rows, columns, size on disk
  python scripts/parquet_peek.py --subjids 57 --dates 20260709

  # one table: ONE LINE PER COLUMN -- dtype, non-null count, distinct count, values
  python scripts/parquet_peek.py --subjids 57 --dates 20260709 --table trial_data

  # one column, with its actual values
  python scripts/parquet_peek.py --subjids 57 --dates 20260709 \
      --table trial_data --column response_time_ms
  python scripts/parquet_peek.py --subjids 57 --dates 20260709 \
      --table trial_data --column odor_sequence --rows 50

It never prints the frame: `trial_data` is 58-73 columns wide and hundreds of rows
long, so a row-shaped view is unreadable in a terminal. One line per column is.
"""
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hypnose_helpers.cli.selector_args import add_selector_args
from hypnose_behavior.io import layout
from hypnose_behavior.io.layout import derivatives
from hypnose_behavior.io.parquet_peek import peek, DEFAULT_ROWS


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_selector_args(ap, subjects_required=True,
                      subjects_help="subject id(s) -- required, so a peek never walks the whole tree")
    ap.add_argument("--table", default=None,
                    help="table name, e.g. trial_data; omit for an inventory of all of them")
    ap.add_argument("--column", default=None, help="show this column alone, with its values")
    ap.add_argument("--rows", type=int, default=DEFAULT_ROWS,
                    help=f"values to list for --column (default {DEFAULT_ROWS}; 0 = statistics only)")
    ap.add_argument("--max-columns", type=int, default=None,
                    help="show only the first N columns (default: all of them)")
    args = ap.parse_args()

    selector = {k: v for k, v in dict(date=args.dates, date_range=args.date_range,
                                      ses=args.ses, ses_range=args.ses_range).items()
                if v is not None}

    sessions = []
    for subjid in args.subjids:
        # `missing_ok` so one subject with nothing in derivatives does not abort the
        # rest: this is a reader, and reporting what it did find beats raising.
        found = derivatives.find_sessions(subjid, missing_ok=True, **selector)
        if not found:
            print(f"no derivatives sessions for subject {subjid}"
                  f"{' matching ' + str(selector) if selector else ''}", file=sys.stderr)
        sessions.extend(found)

    if not sessions:
        return 1

    status = 0
    for i, ses in enumerate(sessions):
        results_dir = layout.results_dir(ses)
        if len(sessions) > 1:
            print(f"{'' if i == 0 else chr(10)}=== {ses.subject} {ses.path.name} ===")
        if not results_dir.exists():
            print(f"error: no saved_analysis_results in {ses.path} -- "
                  f"run trial classification first", file=sys.stderr)
            status = 1
            continue
        try:
            print(peek(results_dir, table=args.table, column=args.column,
                       rows=args.rows, max_columns=args.max_columns))
        except (FileNotFoundError, ValueError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            status = 1
        except KeyError as exc:
            # KeyError renders its message with quotes around it; args[0] is the sentence.
            print(f"error: {exc.args[0]}", file=sys.stderr)
            status = 1
    return status


if __name__ == "__main__":
    raise SystemExit(main())
