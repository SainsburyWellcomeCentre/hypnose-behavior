#!/usr/bin/env python
"""Static import / global-name checker.

Disassembles every function (including nested ones) defined in a module and
verifies that every global name it references (LOAD_GLOBAL / STORE_GLOBAL /
DELETE_GLOBAL) resolves to either the module namespace or builtins. This catches
the class of bug where a function uses a name that was never imported into its
module -- which a plain ``import`` test misses because it only fails at call time
(a NameError deep inside a pipeline). Especially useful during the restructuring,
where functions are moved between modules and an import can easily be left behind.

It does NOT check that the names refer to the *right* thing or that behaviour is
unchanged -- that's what the value regression (regression.py) is for.

Usage
-----
  # one or more dotted module names:
  python src/hypnose_behavior/qc/check_imports.py hypnose_behavior.trial_classification.run

  # or file paths (anything under src/):
  python src/hypnose_behavior/qc/check_imports.py src/hypnose_behavior/trial_classification/run.py

  # default (no args): check every .py module under src/hypnose_behavior:
  python src/hypnose_behavior/qc/check_imports.py

Exit code 0 = all globals resolve; 1 = missing name(s) found / import failed.
"""
from __future__ import annotations

import sys
import dis
import types
import builtins
import argparse
import importlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
SRC = REPO / "src"
sys.path.insert(0, str(SRC))

_BUILTINS = set(dir(builtins))


def _to_module_name(target: str) -> str:
    """Accept a dotted module name or a path under src/ and return the dotted name."""
    if target.endswith(".py") or "/" in target or "\\" in target:
        p = Path(target).resolve()
        try:
            rel = p.relative_to(SRC)
        except ValueError:
            raise SystemExit(f"Path not under {SRC}: {target}")
        parts = list(rel.with_suffix("").parts)
        if parts[-1] == "__init__":
            parts = parts[:-1]
        return ".".join(parts)
    return target


def _referenced_globals(code: types.CodeType, acc: set[str]) -> set[str]:
    for ins in dis.get_instructions(code):
        if ins.opname in ("LOAD_GLOBAL", "STORE_GLOBAL", "DELETE_GLOBAL"):
            acc.add(ins.argval)
    for const in code.co_consts:
        if isinstance(const, types.CodeType):
            _referenced_globals(const, acc)
    return acc


def check_module(modname: str) -> list[tuple[str, str]]:
    """Return a list of (function_name, unresolved_global) for one module."""
    mod = importlib.import_module(modname)
    namespace = set(vars(mod)) | _BUILTINS
    mod_file = getattr(mod, "__file__", None)
    missing: list[tuple[str, str]] = []
    for name, obj in vars(mod).items():
        # only functions actually defined in THIS file (skip re-exports/imports)
        if isinstance(obj, types.FunctionType) and getattr(obj.__code__, "co_filename", None) == mod_file:
            for ref in sorted(_referenced_globals(obj.__code__, set())):
                if ref not in namespace:
                    missing.append((name, ref))
    return missing


def _default_targets() -> list[str]:
    pkg = SRC / "hypnose_behavior"
    # The qc runnable tools (this checker, the regression/verify scripts and their
    # _common helper) are executed, not imported as library modules, so skip them
    # in the default scan. validate.py is library code and is still checked.
    skip = {"regression", "verify_scripts", "check_imports", "_common"}
    return [
        ".".join(p.relative_to(SRC).with_suffix("").parts)
        for p in sorted(pkg.rglob("*.py"))
        if p.name != "__init__.py" and not (p.parent.name == "qc" and p.stem in skip)
    ]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("targets", nargs="*", help="dotted module names or .py paths under src/ (default: all of hypnose_behavior)")
    args = ap.parse_args()

    targets = [_to_module_name(t) for t in args.targets] or _default_targets()
    any_bad = False
    for modname in targets:
        short = modname.split(".")[-1]
        try:
            missing = check_module(modname)
        except Exception as e:
            print(f"[IMPORT FAIL] {modname}: {e}")
            any_bad = True
            continue
        if missing:
            any_bad = True
            print(f"[MISSING] {short} ({modname}):")
            for fn, ref in missing:
                print(f"    {fn}() -> {ref}")
        else:
            print(f"[OK] {short}")
    print("\nRESULT:", "FAIL" if any_bad else "PASS")
    return 1 if any_bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
