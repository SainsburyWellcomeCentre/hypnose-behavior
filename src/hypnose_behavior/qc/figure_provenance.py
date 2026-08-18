#!/usr/bin/env python
"""Does a saved figure's provenance record still name the plotter that drew it?

``DECISIONS.md`` section 9: provenance capture walks the stack and stops at the
**first** non-skipped frame, so a wrapper between the plotter and
``hypnose_helpers.viz.save.save_figure`` silently becomes the recorded author.
Section 9 names two things that reintroduce that hazard -- Phase 5's plotting
primitives, and "the proposed Phase 10 ``visualization_utils.py`` split".

Nothing watched it. ``plot_regression.py`` runs every case with ``save=False``
precisely so it writes nothing, so ``save_figure`` is never called there and the
record is never built; ``regression.py`` never sees a figure at all. The metadata
embedded in every saved PDF was therefore produced by code no gate executed.

What this asserts, per saved figure
-----------------------------------
1. ``chain`` names the case's own plotter. Section 9's rule is *read ``chain``
   before ``function``*, because ``function`` is only ever "the nearest frame we
   did not skip" and is routinely a local ``_save_fig`` closure.
2. ``module`` is a ``hypnose_behavior`` plotting module -- not helpers, not this
   gate, not ``io.save``. This is the wrapper hazard stated directly.

What it deliberately does NOT assert: ``module``, ``file``, ``lineno`` and
``path`` are *derived from where the code lives*, and moving code is the whole
point of Phase 10. Gating them would only restate what ``git diff`` says. They
are printed, so a move's effect on the saved metadata is legible rather than
silent. ``commit`` / ``created_at`` / ``version`` are runtime facts, not code.

Nothing is written. ``save_figure`` is replaced by a recorder for the duration,
so neither the real figure directory nor ``resolve_figure_dir``'s ``mkdir`` is
ever reached -- which matters because the real one resolves under
``/Volumes/harris``, and that tree is strictly read-only.

Usage
-----
  PY=~/miniconda3/envs/hypnose-analysis-test/bin/python
  QC=~/repos/harris_lab/hypnose/hypnose-behavior-analysis/src/hypnose_behavior/qc

  $PY -u $QC/figure_provenance.py
  $PY -u $QC/figure_provenance.py --only plot_decision_accuracy
  $PY -u $QC/figure_provenance.py --break-skip-modules   # section 17 probe
  $PY -u $QC/figure_provenance.py --break-chain          # section 17 probe

Exit code 0 == every saved figure names its plotter; 1 == at least one does not.
"""
from __future__ import annotations

import argparse
import contextlib
import importlib
import io
import sys
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(HERE))

GREEN = "\033[32m"
RED = "\033[31m"
YELLOW = "\033[33m"
DIM = "\033[2m"
RESET = "\033[0m"

# The case list and the module search order have ONE declaration, in the gate
# that already owns them. A second copy here is section 27's trap: the copy that
# decides what is checked drifting from the copy that decides what is run.
import plot_regression as PR  # noqa: E402

# Modules whose figures this gate is about. A record naming anything else is the
# wrapper hazard, whatever it names.
_PLOTTING_PREFIX = "hypnose_behavior.visualization"


def _install_recorder(records, *, break_skip_modules=False):
    """Replace `save_figure` everywhere it is bound, with a recorder.

    The plotter modules do ``from hypnose_behavior.io.save import save_figure``,
    which binds by value at import, so patching ``io.save`` alone reaches none of
    them. Every module in ``MODULES`` that carries the name is patched too.

    The recorder must sit at the *same stack depth* as the real ``save_figure``
    or the frame walk answers a different question than the one in production.
    It does: plotter -> recorder -> provenance(), against plotter ->
    save_figure -> provenance(). Skipping this module reproduces `io/save.py`'s
    own ``skip_modules=(__name__,)``; ``--break-skip-modules`` omits it, which is
    exactly the defect section 9 exists for.
    """
    from hypnose_helpers.provenance import provenance as _provenance

    skip = () if break_skip_modules else ("hypnose_behavior.io.save", __name__)

    def recorder(fig, save_name, *, subjids, dates=None, subdir=None,
                 fig_dir=None, provenance=None, **kwargs):
        rec = provenance if provenance is not None else _provenance(skip_modules=skip)
        records.append({"save_name": save_name, "record": rec})
        return Path("/dev/null") / f"{save_name}.pdf"

    import hypnose_behavior.io.save as io_save
    io_save.save_figure = recorder
    # Second layer: anything that still reaches the real save_figure must not be
    # able to mkdir under the read-only derivatives root.
    io_save._FIGURE_DIR_RESOLVER = lambda subjids, dates=None: Path("/dev/null")

    patched = ["hypnose_behavior.io.save"]
    for mod_name in PR.MODULES:
        try:
            mod = importlib.import_module(mod_name)
        except Exception:
            continue
        if getattr(mod, "save_figure", None) is not None:
            mod.save_figure = recorder
            patched.append(mod_name)
    return patched


def _resolve(name):
    name = name.split("#", 1)[0]
    for mod_name in PR.MODULES:
        try:
            mod = importlib.import_module(mod_name)
        except Exception:
            continue
        fn = getattr(mod, name, None)
        if fn is not None:
            return fn
    return None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="*", default=[], help="limit to these plotter names")
    ap.add_argument("--break-skip-modules", action="store_true",
                    help="section 17 probe: omit this module from skip_modules, "
                         "reproducing the section 9 wrapper defect")
    ap.add_argument("--break-chain", action="store_true",
                    help="section 17 probe: shorten MAX_CHAIN to 1, so a plotter "
                         "that saves from a nested closure loses its own name")
    args = ap.parse_args(argv)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from hypnose_helpers.viz.styles import use_style
    use_style("nature")

    if args.break_chain:
        import hypnose_helpers.provenance as prov
        prov.MAX_CHAIN = 1

    records: list[dict] = []
    patched = _install_recorder(records, break_skip_modules=args.break_skip_modules)
    print(f"save_figure recorder installed in {len(patched)} module(s)")
    if args.break_skip_modules or args.break_chain:
        print(f"{YELLOW}PROBE MODE -- expected to FAIL{RESET}")
    print()

    saves_by_case: dict[str, list[dict]] = {}
    errors: dict[str, str] = {}
    for name, case_args, kwargs in PR.CASES:
        if args.only and name not in args.only:
            continue
        fn = _resolve(name)
        if fn is None:
            errors[name] = "function not found in this tree"
            continue
        records.clear()
        # save=True on purpose: this gate exists to exercise the save path that
        # plot_regression deliberately avoids.
        for attempt in ({**kwargs, "save": True}, kwargs):
            buf = io.StringIO()
            try:
                np.random.seed(0)
                with contextlib.redirect_stdout(buf):
                    fn(*case_args, **attempt)
                break
            except TypeError as e:
                if "save" in str(e) and attempt is not kwargs:
                    plt.close("all")
                    continue
                errors[name] = traceback.format_exc().strip().splitlines()[-1]
                break
            except Exception:
                errors[name] = traceback.format_exc().strip().splitlines()[-1]
                break
        saves_by_case[name] = list(records)
        plt.close("all")

    red = 0
    raised = 0
    no_save = []
    total_saves = 0
    for name in sorted(saves_by_case):
        saves = saves_by_case[name]
        if name in errors:
            # Counted apart from `red`. A case that raises is *untestable* here,
            # not a wrong provenance record -- section 7's "both raise" read as a
            # green is the failure this separation avoids in the other direction.
            # Both still fail the gate: a case that silently stops running is how
            # coverage evaporates (section 22).
            print(f"  {RED}[ERROR]{RESET} {name}: {errors[name]}")
            raised += 1
            continue
        if not saves:
            no_save.append(name)
            continue
        plotter = name.split("#", 1)[0]
        bad = []
        for s in saves:
            rec = s["record"]
            chain = rec.get("chain", [])
            module = rec.get("module", "")
            if plotter not in chain:
                bad.append((s, f"chain does not name {plotter}"))
            elif not module.startswith(_PLOTTING_PREFIX):
                bad.append((s, f"module {module!r} is not a plotting module"))
        total_saves += len(saves)
        first = saves[0]["record"]
        status = f"{RED}[RED]  {RESET}" if bad else f"{GREEN}[green]{RESET}"
        print(f"  {status} {name}  ({len(saves)} save{'s' if len(saves) != 1 else ''})")
        print(f"          module={first.get('module')}  file={first.get('file')}"
              f"  function={first.get('function')}")
        print(f"          chain={first.get('chain')}")
        for s, why in bad:
            red += 1
            print(f"          {RED}{why}{RESET}  save_name={s['save_name']!r} "
                  f"record={ {k: v for k, v in s['record'].items()
                              if k in ('function', 'module', 'file', 'chain')} }")

    print()
    if no_save:
        print(f"{DIM}{len(no_save)} case(s) saved no figure through save_figure "
              f"(they take save_path=, return a table, or do not save):{RESET}")
        for n in no_save:
            print(f"    {n}")
        print()

    checked = len(saves_by_case) - len(no_save) - raised
    print(f"  {checked} case(s) saved {total_saves} figure(s)   |   "
          f"{red} wrong provenance, {raised} case(s) raised")
    if red or raised:
        parts = []
        if red:
            parts.append(f"{red} saved figure(s) do not name their plotter")
        if raised:
            parts.append(f"{raised} case(s) raised and were not checked")
        print(f"{RED}FIGURE PROVENANCE RED: {'; '.join(parts)}.{RESET}")
        return 1
    print(f"{GREEN}FIGURE PROVENANCE GREEN: {total_saves} saved figure(s) across "
          f"{checked} cases each name the plotter that drew it.{RESET}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
