#!/usr/bin/env python
"""Delete the outputs nothing reads: `response_time_analysis.json`, `indices/`, and the
pooled-metrics files.

Step 2 of 2, and the only step that removes anything. Run
`migrate_results_layout.py` first, confirm the tree reads correctly, then run this.

**Why these two, and nothing else.** Both were written by code no reader ever called:

- `response_time_analysis.json` -- five lists of response times. Every value in it is a
  non-null `response_time_ms` of `trial_data`, which is where the analysis reads them
  from; the JSON had no reader in this repo or its siblings.
- `indices/index.json` + `indices/aborted_index.json` -- lookup tables over frames that
  are themselves saved. `index.json` alone is larger than every other file in a session
  combined, and nothing ever read either one back.
- the pooled metrics -- `<subject>/merged_results/merged_<subjid>_*` and
  `merged/{merged,protocol_merged}/merged_subjids_*`, `.json` and `.txt`. Keyed by the
  selection that produced them, so every distinct set of subjects or dates left another
  pair, and keyed only by `MMDD`, so the same day-range in two years collided. Pooled
  metrics are now reported to the console instead.

Those directories are swept whole, so **every file in them that the prefix does not claim
is listed and left alone** -- that report is how anything not written by this pipeline
gets noticed rather than deleted. Pooled files are selected by `--subjids` only, since
they span dates; the cross-subject `merged/` tree is swept only on a whole-tree run.

Defaults to a dry run. Pass `--delete` to actually remove; there is no undo, so the
default is deliberately the harmless one.

The report is grouped by entry name -- how many sessions hold each one, what it costs,
and which files a directory holds -- so a sweep over the whole tree is a handful of lines
to check rather than one per session. `--verbose` prints the per-session lines as well; a
`--delete` run prints them regardless, as a record of what it removed.

Examples
--------
  python scripts/cleanup_results_layout.py --subjids 57 --dates 20260807
  python scripts/cleanup_results_layout.py                     # summary for every session
  python scripts/cleanup_results_layout.py --subjids 57 --delete
  python scripts/cleanup_results_layout.py --delete            # every analysed session
"""
import argparse
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import NamedTuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hypnose_behavior.io import layout
from hypnose_behavior.io.loaders import iter_sessions

# Exactly what this script may remove, relative to a results directory. Both are
# write-only outputs -- see the module docstring for how that was established.
OBSOLETE_FILES = ("response_time_analysis.json",)
OBSOLETE_DIRS = ("indices",)

# The pooled-metrics files, which sit outside any session. Each entry is
# (directory relative to its root, filename prefix). Only `.json` and `.txt` matching the
# prefix are removed; every other file in those directories is reported untouched, which
# is how something not written by this pipeline gets noticed instead of deleted.
MERGED_SUFFIXES = (".json", ".txt")
MERGED_SUBJ_DIR = "merged_results"          # under a subject directory
MERGED_ROOT_DIR = "merged"                  # under the derivatives root
MERGED_ROOT_SUBDIRS = ("merged", "protocol_merged")
MERGED_ROOT_PREFIX = "merged_subjids_"


def _ignore_vanished(func, path, exc):
    """`shutil.rmtree` handler: tolerate an entry that disappeared mid-walk.

    Deleting a file on the SMB mount takes its `._` AppleDouble shadow with it, so a
    directory listing taken before the walk names entries that are already gone by the
    time `rmtree` unlinks them. Anything other than "not found" is a real failure and is
    re-raised.
    """
    if not isinstance(exc, FileNotFoundError):
        raise exc


class Removed(NamedTuple):
    """One entry a session gives up. `name` carries a trailing `/` for a directory."""

    name: str
    nbytes: int
    inner: tuple[str, ...]


def _entry_stats(path: Path) -> tuple[int, tuple[str, ...]]:
    """Bytes held by a file or directory, and the names of the files inside it.

    The byte total counts everything on disk, AppleDouble shadows included, because that
    is what removing the entry frees. The name list leaves them out: it exists to show
    what a directory holds, and a `._index.json` beside every real file says nothing about
    that. A plain file has no inner names, and entries that vanish mid-walk count as zero.
    """
    try:
        if path.is_file():
            return path.stat().st_size, ()
        total, names = 0, []
        for p in path.rglob("*"):
            try:
                if not p.is_file():
                    continue
                total += p.stat().st_size
                if not p.name.startswith("._"):
                    names.append(p.name)
            except OSError:
                continue
        return total, tuple(sorted(names))
    except OSError:
        return 0, ()


def clean_session(results_dir: Path, delete: bool = False) -> tuple[list[Removed], int]:
    """Remove the obsolete outputs of one session. Returns (entries, bytes_freed).

    An entry is recorded only once it is actually gone, so a directory that resisted
    removal is warned about and left out of both the list and the byte total rather than
    reported as freed space that is still occupied.
    """
    removed: list[Removed] = []
    freed = 0
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
        is_dir = target.is_dir()
        nbytes, inner = _entry_stats(target)
        if delete:
            if is_dir:
                shutil.rmtree(target, onexc=_ignore_vanished)
                # `onexc` swallows the vanished shadows, but it also swallows the failure
                # to remove the directory itself if one of them was the last entry the
                # walk expected. Take it out explicitly when it survived.
                if target.exists():
                    try:
                        target.rmdir()
                    except OSError as e:
                        print(f"    WARNING: {target} not removed: {e}")
                        continue
            else:
                try:
                    target.unlink()
                except FileNotFoundError:
                    pass
        freed += nbytes
        removed.append(Removed(f"{target.name}/" if is_dir else target.name, nbytes, inner))
    return removed, freed


def clean_merged_dir(directory: Path, prefix: str, delete: bool = False):
    """Remove the pooled-metrics files in one directory. Returns (removed, other, freed).

    `other` is every file the prefix does not claim. Reporting it is the point: these
    directories are swept whole, so anything in them that this pipeline did not write has
    to be seen rather than assumed away.
    """
    removed: list[Removed] = []
    other: list[str] = []
    freed = 0
    if not directory.is_dir():
        return removed, other, freed

    for entry in sorted(directory.iterdir()):
        if entry.name.startswith("._"):
            continue
        if not entry.is_file():
            other.append(entry.name + "/")
            continue
        if not (entry.name.startswith(prefix) and entry.suffix in MERGED_SUFFIXES):
            other.append(entry.name)
            continue
        try:
            nbytes = entry.stat().st_size
        except OSError:
            nbytes = 0
        if delete:
            shadow = entry.parent / f"._{entry.name}"
            try:
                entry.unlink()
            except FileNotFoundError:
                pass
            if shadow.exists():
                try:
                    shadow.unlink()
                except OSError:
                    pass
        freed += nbytes
        removed.append(Removed(entry.name, nbytes, ()))
    return removed, other, freed


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subjids", nargs="*", type=int, default=None,
                    help="subject id(s); default: all")
    ap.add_argument("--dates", nargs="*", type=int, default=None,
                    help="specific date(s) YYYYMMDD; default: all")
    ap.add_argument("--delete", action="store_true",
                    help="actually delete; without it this is a dry run")
    ap.add_argument("--verbose", action="store_true",
                    help="one line per session as well as the grouped summary "
                         "(implied by --delete)")
    args = ap.parse_args()

    subjects = layout.derivatives.iter_subjects(args.subjids)
    if not subjects:
        print("No subjects found.")
        return 1

    if not args.delete:
        print("DRY RUN -- nothing will be deleted. Re-run with --delete to remove.")

    dates = [str(d) for d in args.dates] if args.dates else None
    per_session = args.delete or args.verbose
    verb = "removed" if args.delete else "would remove"
    # Counted per entry name rather than per session: over a whole-tree sweep the
    # question is "which artefacts is this taking, and does that list hold surprises",
    # and 240 near-identical lines answer it worse than one row per name does.
    n_sessions = n_entries = total = 0
    sessions_with: Counter = Counter()   # name -> sessions holding it
    bytes_by: Counter = Counter()        # name -> bytes across those sessions
    inner_by: dict[str, Counter] = {}    # name -> inner file name -> times seen
    for _subjid, subj_dir in subjects:
        for rec in iter_sessions(subj_dir, dates):
            removed, freed = clean_session(rec.results_dir, args.delete)
            if not removed:
                continue
            n_sessions += 1
            n_entries += len(removed)
            total += freed
            for entry in removed:
                sessions_with[entry.name] += 1
                bytes_by[entry.name] += entry.nbytes
                if entry.inner:
                    inner_by.setdefault(entry.name, Counter()).update(entry.inner)
            if per_session:
                print(f"{subj_dir.name}/{rec.date_str}: {verb} "
                      f"{', '.join(e.name for e in removed)} ({freed / 1e6:.1f} MB)")

    # --- Pooled-metrics files, which live outside any session ---
    # Governed by `--subjids` only: they span dates, so `--dates` cannot describe one.
    # The tree-level directory is cross-subject, so it is only swept on a whole-tree run.
    merged_removed = merged_bytes = 0
    unclaimed: dict[str, list[str]] = {}
    merged_dirs = [(subj_dir / MERGED_SUBJ_DIR, f"merged_{subj_dir.name.split('_')[0].replace('sub-', '')}_")
                   for _subjid, subj_dir in subjects]
    if args.subjids is None:
        merged_dirs += [(layout.derivatives.root / MERGED_ROOT_DIR / sub, MERGED_ROOT_PREFIX)
                        for sub in MERGED_ROOT_SUBDIRS]
    else:
        print(f"\n(skipping {MERGED_ROOT_DIR}/ -- it is cross-subject; run without "
              f"--subjids to sweep it)")

    for directory, prefix in merged_dirs:
        removed, other, freed = clean_merged_dir(directory, prefix, args.delete)
        if other:
            unclaimed[str(directory)] = other
        if not removed:
            continue
        merged_removed += len(removed)
        merged_bytes += freed
        total += freed
        if per_session:
            print(f"{directory.parent.name}/{directory.name}: {verb} {len(removed)} "
                  f"pooled file(s) ({freed / 1e6:.1f} MB)")

    if sessions_with:
        print(f"\n{verb.capitalize()}, by entry:")
        width = max(len(name) for name in sessions_with)
        for name, count in sorted(sessions_with.items(), key=lambda kv: (-kv[1], kv[0])):
            line = f"  {count:5d} x  {name:<{width}}  {bytes_by[name] / 1e6:9.1f} MB"
            inner = inner_by.get(name)
            if inner:
                # Every distinct name seen inside, so an unexpected file in a directory
                # this deletes whole shows up here rather than only in the byte total.
                shown = sorted(inner)
                more = f", +{len(shown) - 6} more" if len(shown) > 6 else ""
                line += f"   {sum(inner.values())} file(s): {', '.join(shown[:6])}{more}"
            print(line)

    if merged_removed:
        print(f"\n{verb.capitalize()}, pooled metrics: {merged_removed} file(s) across "
              f"{len([d for d, _ in merged_dirs if d.is_dir()])} director(ies), "
              f"{merged_bytes / 1e6:.1f} MB")

    if unclaimed:
        print("\nNOT touched -- files in those directories that no rule claims:")
        for directory, names in sorted(unclaimed.items()):
            print(f"  {directory}")
            for name in names[:12]:
                print(f"      {name}")
            if len(names) > 12:
                print(f"      ... +{len(names) - 12} more")
    elif merged_dirs:
        print("\nEvery file in the pooled-metrics directories matched the expected "
              "pattern; nothing unexpected was found.")

    print(f"\n{n_sessions} session(s), {n_entries} entr(ies), "
          f"{merged_removed} pooled file(s), "
          f"{total / 1e6:.1f} MB {'freed' if args.delete else 'recoverable'}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
