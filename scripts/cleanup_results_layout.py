#!/usr/bin/env python
"""Delete the two outputs nothing reads: `response_time_analysis.json` and `indices/`.

Step 2 of 2, and the only step that removes anything. Run
`migrate_results_layout.py` first, confirm the tree reads correctly, then run this.

**Why these two, and nothing else.** Both were written by code no reader ever called:

- `response_time_analysis.json` -- five lists of response times. Every value in it is a
  non-null `response_time_ms` of `trial_data`, which is where the analysis reads them
  from; the JSON had no reader in this repo or its siblings.
- `indices/index.json` + `indices/aborted_index.json` -- lookup tables over frames that
  are themselves saved. `index.json` alone is larger than every other file in a session
  combined, and nothing ever read either one back.

Defaults to a dry run. Pass `--delete` to actually remove; there is no undo, so the
default is deliberately the harmless one.

Examples
--------
  python scripts/cleanup_results_layout.py --subjids 57 --dates 20260807
  python scripts/cleanup_results_layout.py --subjids 57 --delete
  python scripts/cleanup_results_layout.py --delete            # every analysed session
"""
import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hypnose_behavior.io import layout
from hypnose_behavior.io.loaders import iter_sessions

# Exactly what this script may remove, relative to a results directory. Both are
# write-only outputs -- see the module docstring for how that was established.
OBSOLETE_FILES = ("response_time_analysis.json",)
OBSOLETE_DIRS = ("indices",)


def _ignore_vanished(func, path, exc):
    """`shutil.rmtree` handler: tolerate an entry that disappeared mid-walk.

    Deleting a file on the SMB mount takes its `._` AppleDouble shadow with it, so a
    directory listing taken before the walk names entries that are already gone by the
    time `rmtree` unlinks them. Anything other than "not found" is a real failure and is
    re-raised.
    """
    if not isinstance(exc, FileNotFoundError):
        raise exc


def _entry_bytes(path: Path) -> int:
    """Bytes held by a file or directory. Entries that vanish mid-walk count as zero."""
    try:
        if path.is_file():
            return path.stat().st_size
        total = 0
        for p in path.rglob("*"):
            try:
                if p.is_file():
                    total += p.stat().st_size
            except OSError:
                continue
        return total
    except OSError:
        return 0


def clean_session(results_dir: Path, delete: bool = False):
    """Remove the obsolete outputs of one session. Returns (names, bytes_freed)."""
    removed, freed = [], 0
    if not results_dir.is_dir():
        return removed, freed

    targets = [results_dir / n for n in OBSOLETE_FILES]
    targets += [results_dir / n for n in OBSOLETE_DIRS]
    # The AppleDouble shadow an SMB mount leaves beside each one; it is part of the same
    # artefact and is meaningless once its partner is gone.
    targets += [results_dir / f"._{n}" for n in OBSOLETE_FILES + OBSOLETE_DIRS]

    for target in targets:
        if not target.exists():
            continue
        freed += _entry_bytes(target)
        removed.append(target.name)
        if delete:
            if target.is_dir():
                shutil.rmtree(target, onexc=_ignore_vanished)
                # `onexc` swallows the vanished shadows, but it also swallows the failure
                # to remove the directory itself if one of them was the last entry the
                # walk expected. Take it out explicitly when it survived.
                if target.exists():
                    try:
                        target.rmdir()
                    except OSError as e:
                        removed.pop()
                        print(f"    WARNING: {target} not removed: {e}")
                        continue
            else:
                try:
                    target.unlink()
                except FileNotFoundError:
                    pass
    return removed, freed


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subjids", nargs="*", type=int, default=None,
                    help="subject id(s); default: all")
    ap.add_argument("--dates", nargs="*", type=int, default=None,
                    help="specific date(s) YYYYMMDD; default: all")
    ap.add_argument("--delete", action="store_true",
                    help="actually delete; without it this is a dry run")
    args = ap.parse_args()

    subjects = layout.derivatives.iter_subjects(args.subjids)
    if not subjects:
        print("No subjects found.")
        return 1

    if not args.delete:
        print("DRY RUN -- nothing will be deleted. Re-run with --delete to remove.\n")

    dates = [str(d) for d in args.dates] if args.dates else None
    n_sessions = n_entries = total = 0
    for _subjid, subj_dir in subjects:
        for rec in iter_sessions(subj_dir, dates):
            removed, freed = clean_session(rec.results_dir, args.delete)
            if not removed:
                continue
            n_sessions += 1
            n_entries += len(removed)
            total += freed
            verb = "removed" if args.delete else "would remove"
            print(f"{subj_dir.name}/{rec.date_str}: {verb} {', '.join(removed)} "
                  f"({freed / 1e6:.1f} MB)")

    print(f"\n{n_sessions} session(s), {n_entries} entr(ies), "
          f"{total / 1e6:.1f} MB {'freed' if args.delete else 'recoverable'}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
