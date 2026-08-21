#!/usr/bin/env python
"""Move a session's saved outputs into the grouped layout. Moves only; deletes nothing.

Step 1 of 2. `cleanup_results_layout.py` is step 2 and is the only one that removes
anything, so a tree can be migrated, read, and checked before a single byte is deleted.

What moves is a **whitelist**: the file names this package writes, plus `*.slp`. Anything
else in a results directory -- notes, exports, files another repo put there -- is left
exactly where it is. A file already in its destination is counted as done, and a
destination that is occupied by a *different* file is reported and skipped rather than
overwritten.

    saved_analysis_results/
      manifest.json  summary.json            <- stay at the top level
      metric_analysis/                       <- metrics_by_*.parquet, metrics_<subj>_<date>.*
      trial_classification_results/          <- trial_data.*, position_data, non_initiated,
                                                merged_summary_<subj>_<date>.txt
      movement_analysis/                     <- speed_analysis.parquet, *.slp, tracking files

Examples
--------
  python scripts/migrate_results_layout.py --subjids 57 --dates 20260807 --dry-run
  python scripts/migrate_results_layout.py --subjids 57
  python scripts/migrate_results_layout.py                      # every analysed session

Prints one line per session and a final count. `--dry-run` reports what it would do and
touches nothing.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hypnose_behavior.io import layout
from hypnose_behavior.io.loaders import iter_sessions

# Files that legitimately stay at the top level: they describe the session as a whole,
# not the output of any one stage, and every subfolder's readers depend on them.
TOP_LEVEL = frozenset({"manifest.json", "summary.json"})

# Patterns whose matches move into a subfolder. Glob entries carry a `*`; everything
# else is an exact file name. Deliberately explicit rather than "every parquet": a
# results directory is shared with other repos, and a mover that guesses is a mover that
# moves someone else's file.
MOVE_RULES = (
    ("metrics_by_trial.parquet", "metric_analysis"),
    ("metrics_by_poke.parquet", "metric_analysis"),
    ("metrics_*", "metric_analysis"),
    ("trial_data.parquet", "trial_classification_results"),
    ("trial_data.csv", "trial_classification_results"),
    ("trial_data.schema.json", "trial_classification_results"),
    ("position_data.parquet", "trial_classification_results"),
    # `non_initiated_attempts` is written today; `non_initiated_sequences`,
    # `non_initiated_odor1_attempts` and `non_initiated_FA` are older tables that
    # `io.loaders` still reads, so they move with the family rather than being stranded.
    ("non_initiated_*", "trial_classification_results"),
    ("merged_summary_*", "trial_classification_results"),
    ("speed_analysis.parquet", "movement_analysis"),
    # The three forms the SLEAP repo writes into a session, per `sleap_utils.py`:
    # the raw predictions, the per-video tracking tables
    # (`sleap_tracking_video{N}_{tag}.parquet|csv`), and the combined timestamps file
    # (`sub-NNN_ses-DATE_combined_sleap_tracking_timestamps.parquet|csv`), which carries
    # a subject/session prefix and so does not start with `sleap_`.
    ("*.slp", "movement_analysis"),
    ("sleap_*", "movement_analysis"),
    ("*sleap_tracking*", "movement_analysis"),
)


def _destination(name: str):
    """The subfolder `name` moves into, or None if this mover does not claim it."""
    for pattern, folder in MOVE_RULES:
        if pattern.endswith("*") or pattern.startswith("*"):
            if Path(name).match(pattern):
                return folder
        elif name == pattern:
            return folder
    return None


def migrate_session(results_dir: Path, dry_run: bool = False):
    """Move one session's files. Returns (moved, skipped, unclaimed) name lists.

    An already-migrated session reports nothing moved: only files sitting *directly* in
    the results directory are considered, and there are none left to claim.

    `unclaimed` is every other file left at the top level. Reporting it is the point of a
    whitelist -- it is how you check that nothing this script does not understand was
    moved, and it surfaces files worth a decision rather than silently relocating them.
    """
    moved, skipped, unclaimed = [], [], []
    if not results_dir.is_dir():
        return moved, skipped, unclaimed

    for entry in sorted(results_dir.iterdir()):
        # Only files directly in the results directory. An existing subfolder is either
        # already-migrated output or something this script does not own.
        if not entry.is_file():
            continue
        # AppleDouble shadows an SMB mount leaves beside every real file. Moving one
        # without its partner is worse than leaving both.
        if entry.name.startswith("._"):
            continue
        folder = _destination(entry.name)
        if folder is None:
            # `manifest.json` and `summary.json` belong at the top level; anything else
            # here is a file this script does not claim, and is worth seeing.
            if entry.name not in TOP_LEVEL:
                unclaimed.append(entry.name)
            continue
        target = results_dir / folder / entry.name
        if target.exists():
            # A flat file and a grouped file of the same name: a session re-analysed
            # under the new layout before being migrated. Which one is current is not
            # this script's call, so it reports both and moves neither.
            skipped.append(f"{entry.name} (destination already exists)")
            continue
        if not dry_run:
            target.parent.mkdir(parents=True, exist_ok=True)
            entry.rename(target)
        moved.append(f"{folder}/{entry.name}")
    return moved, skipped, unclaimed


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subjids", nargs="*", type=int, default=None,
                    help="subject id(s); default: all")
    ap.add_argument("--dates", nargs="*", type=int, default=None,
                    help="specific date(s) YYYYMMDD; default: all")
    ap.add_argument("--dry-run", action="store_true",
                    help="report what would move; change nothing")
    args = ap.parse_args()

    subjects = layout.derivatives.iter_subjects(args.subjids)
    if not subjects:
        print("No subjects found.")
        return 1

    dates = [str(d) for d in args.dates] if args.dates else None
    n_sessions = n_moved = n_skipped = 0
    all_unclaimed: dict[str, int] = {}
    for _subjid, subj_dir in subjects:
        for rec in iter_sessions(subj_dir, dates):
            moved, skipped, unclaimed = migrate_session(rec.results_dir, args.dry_run)
            for name in unclaimed:
                all_unclaimed[name] = all_unclaimed.get(name, 0) + 1
            if not (moved or skipped):
                continue
            n_sessions += 1
            n_moved += len(moved)
            n_skipped += len(skipped)
            verb = "would move" if args.dry_run else "moved"
            bits = [f"{verb} {len(moved)}"]
            if skipped:
                bits.append(f"{len(skipped)} SKIPPED")
            print(f"{subj_dir.name}/{rec.date_str}: " + ", ".join(bits))
            for s in skipped:
                print(f"    SKIP {s}")

    print(f"\n{n_sessions} session(s) touched, {n_moved} file(s) "
          f"{'to move' if args.dry_run else 'moved'}, {n_skipped} skipped.")
    if n_skipped:
        print("Skipped files were left in place; nothing was overwritten or deleted.")
    if all_unclaimed:
        print(f"\nLeft at the top level -- not claimed by any rule ({len(all_unclaimed)} "
              f"distinct name(s)):")
        for name, count in sorted(all_unclaimed.items(), key=lambda kv: (-kv[1], kv[0])):
            print(f"    {count:5d} session(s)  {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
