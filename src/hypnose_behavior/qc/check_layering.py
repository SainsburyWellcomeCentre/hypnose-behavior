#!/usr/bin/env python
"""Static layering checker.

Parses every module under ``src/hypnose_behavior`` with ``ast``, builds the
module -> module import graph, and asserts three things about it:

1. **The module graph is a DAG.** No file-to-file import cycle, at any depth.
   This is the invariant worth protecting: it is what lets a reader follow an
   import chain to its end.
2. **Every cycle at directory granularity is a declared decision.** Directories
   are tiers, and a loop between tiers fails unless one of its tier-to-tier
   edges is listed in ``DECLARED`` below, carrying the reason it is accepted.
3. **Every declared decision still holds.** An accepted upward edge is safe
   because the module it reaches is a leaf, so each entry names what that module
   may import. The day it reaches further, the entry's reason has stopped being
   true, and a leaf rule that has quietly stopped being true is worse than no
   leaf rule at all (DECISIONS.md section 31).

Nothing is hidden. Every cycle is printed with every edge that forms it, and a
declared edge is *marked*, never removed -- an accepted cycle stays visible in
the report, which is the difference between a decision and a suppression. A
``DECLARED`` entry that no longer describes the tree fails the gate too, so the
allow-list cannot rot into a lie about the code.

Tiers
-----
A module inside a subpackage belongs to its top-level directory, so
``metric_analysis/metrics/accuracy.py`` is ``metric_analysis/``. A module at the
package root is its own tier, because the root is not one tier: ``frames.py``
and ``parameters.py`` are leaves below everything (DECISIONS.md sections 3, 31),
while ``api.py`` and ``accessors.py`` sit above everything.

Usage
-----
  python src/hypnose_behavior/qc/check_layering.py
  python src/hypnose_behavior/qc/check_layering.py --tiers   # + the fan-out table

Exit code 0 = the module graph is a DAG and every directory cycle is declared;
1 = an undeclared cycle, or a ``DECLARED`` entry that does not match the tree.

It reads source and imports nothing from the package, so it needs no mount, no
pandas and no installed environment -- ``qc/`` stays a sink with fan-in 0.
"""
from __future__ import annotations

import ast
import textwrap
import argparse
from pathlib import Path
from typing import Iterator, NamedTuple

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
SRC = REPO / "src"
PKG = SRC / "hypnose_behavior"
ROOT = PKG.name

# Elementary cycles to enumerate before giving up. A graph with more than this
# has a layering problem that no per-cycle report can usefully list.
CYCLE_CAP = 200


class Declared(NamedTuple):
    """One accepted edge: which module imports which, why that is allowed, and the
    leaf rule the reason rests on.

    ``imported_may_import`` is the complete set of package modules ``imported`` may
    reach. It is what makes the edge safe, so it is required: an exception whose
    justification is unchecked is prose, and prose drifts.
    """

    importer: str
    imported: str
    reason: str
    imported_may_import: frozenset[str]


# Each entry is a decision that has been taken and written down. The cycle it
# covers is still reported in full, marked with this reason. An entry whose
# import no longer exists, that covers no cycle, or whose leaf rule has stopped
# holding, fails the gate.
DECLARED: tuple[Declared, ...] = (
    Declared(
        importer="hypnose_behavior.io.save_results",
        imported="hypnose_behavior.trial_classification.outcome",
        reason="outcome.py is an 84-line trial-classification rule with three sibling "
               "callers in that package, so it stays there rather than being promoted "
               "to the root tier. It imports nothing from the package except the "
               "root-level leaves, which is what lets io/save_results.py reach it "
               "without pulling trial_classification down with it.",
        imported_may_import=frozenset({
            "hypnose_behavior.parameters",
            "hypnose_behavior.frames",
        }),
    ),
)


class Edge(NamedTuple):
    """One import statement resolving to another module in the package."""

    importer: str
    imported: str
    lineno: int


def _module_name(path: Path) -> str:
    parts = list(path.relative_to(SRC).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _tier(path: Path) -> str:
    rel = path.relative_to(PKG)
    return rel.name if rel.parent == Path(".") else rel.parts[0] + "/"


def discover() -> dict[str, Path]:
    """Every module in the package, dotted name -> file."""
    return {_module_name(p): p for p in sorted(PKG.rglob("*.py"))}


def _resolve(dotted: str, modules: dict[str, Path]) -> str | None:
    """The longest prefix of a dotted name that is a module in the package."""
    parts = dotted.split(".")
    while parts:
        candidate = ".".join(parts)
        if candidate in modules:
            return candidate
        parts.pop()
    return None


def _targets(node: ast.AST, mod: str, path: Path, modules: dict[str, Path]) -> Iterator[str]:
    """The package modules one import statement reaches."""
    if isinstance(node, ast.Import):
        for alias in node.names:
            if alias.name == ROOT or alias.name.startswith(ROOT + "."):
                target = _resolve(alias.name, modules)
                if target:
                    yield target
        return

    if not isinstance(node, ast.ImportFrom):
        return

    if node.level:  # relative import: rebuild the absolute prefix
        parts = mod.split(".")
        if path.name != "__init__.py":
            parts = parts[:-1]
        drop = node.level - 1
        parts = parts[: len(parts) - drop] if drop < len(parts) else []
        prefix = ".".join([*parts, node.module] if node.module else parts)
    else:
        prefix = node.module or ""

    if not (prefix == ROOT or prefix.startswith(ROOT + ".")):
        return
    for alias in node.names:
        # "from pkg.mod import name" reaches pkg.mod; "from pkg import mod"
        # reaches pkg.mod -- both are the longest prefix that is a module.
        target = _resolve(f"{prefix}.{alias.name}", modules) or _resolve(prefix, modules)
        if target:
            yield target


def collect_edges(modules: dict[str, Path]) -> list[Edge]:
    """One edge per import statement per module it reaches."""
    edges: set[Edge] = set()
    for mod, path in modules.items():
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (SyntaxError, UnicodeDecodeError) as exc:
            # A tree that cannot be read cannot be judged; say so instead of
            # reporting a graph with a module silently missing from it.
            raise SystemExit(f"[PARSE FAIL] {path.relative_to(PKG)}: {exc}")
        for node in ast.walk(tree):
            for target in _targets(node, mod, path, modules):
                edges.add(Edge(mod, target, node.lineno))
    return sorted(edges)


def sccs(nodes, adj: dict[str, set[str]]) -> list[list[str]]:
    """Tarjan's strongly connected components."""
    index: dict[str, int] = {}
    low: dict[str, int] = {}
    stack: list[str] = []
    on_stack: set[str] = set()
    out: list[list[str]] = []
    counter = 0

    def visit(v: str) -> None:
        nonlocal counter
        index[v] = low[v] = counter
        counter += 1
        stack.append(v)
        on_stack.add(v)
        for w in sorted(adj.get(v, ())):
            if w not in index:
                visit(w)
                low[v] = min(low[v], low[w])
            elif w in on_stack:
                low[v] = min(low[v], index[w])
        if low[v] == index[v]:
            component = []
            while True:
                w = stack.pop()
                on_stack.discard(w)
                component.append(w)
                if w == v:
                    break
            out.append(sorted(component))

    for n in sorted(nodes):
        if n not in index:
            visit(n)
    return out


def elementary_cycles(nodes, adj: dict[str, set[str]], cap: int = CYCLE_CAP) -> list[list[str]]:
    """Every simple cycle, each rooted at its lowest-ordered node so it is found once."""
    order = {n: i for i, n in enumerate(sorted(nodes))}
    found: list[list[str]] = []

    def walk(start: str, path: list[str], on_path: set[str]) -> None:
        for nxt in sorted(adj.get(path[-1], ())):
            if len(found) >= cap:
                return
            if order[nxt] < order[start]:
                continue
            if nxt == start:
                found.append(list(path))
            elif nxt not in on_path:
                path.append(nxt)
                on_path.add(nxt)
                walk(start, path, on_path)
                path.pop()
                on_path.discard(nxt)

    for start in sorted(nodes):
        if len(found) >= cap:
            break
        walk(start, [start], {start})
    found.sort(key=lambda cycle: (len(cycle), cycle))
    return found


def _short(mod: str) -> str:
    return mod[len(ROOT) + 1:] if mod.startswith(ROOT + ".") else mod


def _site(edge: Edge, modules: dict[str, Path]) -> str:
    return f"{modules[edge.importer].relative_to(PKG)}:{edge.lineno}"


def _reason_lines(reason: str, indent: int) -> str:
    return textwrap.fill(
        f"reason: {reason}",
        width=96,
        initial_indent=" " * indent,
        subsequent_indent=" " * (indent + 8),
    )


def _print_tiers(modules, edges, tier) -> None:
    files: dict[str, list[str]] = {}
    for mod in modules:
        files.setdefault(tier[mod], []).append(mod)
    out: dict[str, set[str]] = {}
    into: dict[str, set[str]] = {}
    for edge in edges:
        a, b = tier[edge.importer], tier[edge.imported]
        if a != b:
            out.setdefault(a, set()).add(b)
            into.setdefault(b, set()).add(a)
    print("\n[TIERS]")
    for name in sorted(files, key=lambda t: (-len(files[t]), t)):
        print(f"  {name:24s} files={len(files[name]):3d}  fan-out={len(out.get(name, ())):2d}"
              f"  imported-by={len(into.get(name, ())):2d}")
        if out.get(name):
            print(f"      imports  {', '.join(sorted(out[name]))}")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--tiers", action="store_true", help="also print the per-tier fan-out table")
    args = ap.parse_args()

    modules = discover()
    edges = collect_edges(modules)
    tier = {mod: _tier(path) for mod, path in modules.items()}
    tiers = sorted(set(tier.values()))
    failures: list[str] = []

    print(f"=== {ROOT} layering ===")
    print(f"{len(modules)} modules, {len(edges)} internal import edges, {len(tiers)} tiers")

    # 1. the module graph must be a DAG
    self_imports = [e for e in edges if e.importer == e.imported]
    module_adj: dict[str, set[str]] = {}
    for edge in edges:
        if edge.importer != edge.imported:
            module_adj.setdefault(edge.importer, set()).add(edge.imported)
    module_cycles = [c for c in sccs(modules, module_adj) if len(c) > 1]

    if module_cycles or self_imports:
        failures.append("module cycle")
        print(f"\n[MODULES] FAIL -- {len(module_cycles) + len(self_imports)} cycle(s)")
        for edge in self_imports:
            print(f"    {_site(edge, modules)} imports itself")
        for component in module_cycles:
            print(f"    {len(component)} modules: {', '.join(_short(m) for m in component)}")
    else:
        print("\n[MODULES] no cycles -- the file graph is a DAG")

    # 2. every directory cycle must be a declared decision
    pairs: dict[tuple[str, str], list[Edge]] = {}
    for edge in edges:
        a, b = tier[edge.importer], tier[edge.imported]
        if a != b:
            pairs.setdefault((a, b), []).append(edge)
    tier_adj: dict[str, set[str]] = {}
    for a, b in pairs:
        tier_adj.setdefault(a, set()).add(b)

    declared = {(d.importer, d.imported): d for d in DECLARED}

    def legs(cycle: list[str]) -> list[tuple[str, str]]:
        return [(cycle[i], cycle[(i + 1) % len(cycle)]) for i in range(len(cycle))]

    def fully_declared(pair: tuple[str, str]) -> bool:
        return all((e.importer, e.imported) in declared for e in pairs[pair])

    cycles = elementary_cycles(tiers, tier_adj)
    print(f"\n[DIRECTORIES] {len(cycles)} cycle(s) over {len(pairs)} tier-to-tier edges")
    if len(cycles) >= CYCLE_CAP:
        print(f"    (stopped at the {CYCLE_CAP}-cycle cap; the listing below is partial)")

    cyclic_pairs: set[tuple[str, str]] = set()
    for number, cycle in enumerate(cycles, start=1):
        cycle_legs = legs(cycle)
        cyclic_pairs.update(cycle_legs)
        total = sum(len(pairs[p]) for p in cycle_legs)
        route = " -> ".join([*cycle, cycle[0]])
        print(f"\n[CYCLE {number}] {route}   ({total} edges)")
        for pair in cycle_legs:
            count = len(pairs[pair])
            print(f"    {pair[0]} -> {pair[1]}   ({count} edge{'' if count == 1 else 's'})")
            for edge in sorted(pairs[pair], key=lambda e: (e.importer, e.lineno)):
                entry = declared.get((edge.importer, edge.imported))
                tag = "   [DECLARED]" if entry else ""
                print(f"        {_site(edge, modules):<44} -> {_short(edge.imported)}{tag}")
                if entry:
                    print(_reason_lines(entry.reason, 12))
        broken_by = [p for p in cycle_legs if fully_declared(p)]
        if broken_by:
            print(f"    OK -- declared: {broken_by[0][0]} -> {broken_by[0][1]}")
        else:
            failures.append(f"cycle {number}")
            # Every leg of a cycle is listed above; the narrowest one is where
            # cutting is cheapest, and where a freshly added import shows up.
            thin = min(cycle_legs, key=lambda p: (len(pairs[p]), p))
            n_thin = len(pairs[thin])
            print(f"    FAIL -- no declared edge closes this cycle; narrowest leg "
                  f"{thin[0]} -> {thin[1]} ({n_thin} edge{'' if n_thin == 1 else 's'})")

    # 3. the allow-list must still describe the tree, and each leaf rule must hold
    plural = "entry" if len(DECLARED) == 1 else "entries"
    print(f"\n[DECLARED] {len(DECLARED)} {plural}")
    present = {(e.importer, e.imported) for e in edges}
    for entry in DECLARED:
        label = f"{_short(entry.importer)} -> {_short(entry.imported)}"
        if not entry.reason.strip():
            failures.append(f"declared {label}")
            print(f"    [NO REASON] {label} -- an exception without a reason is a suppression")
            continue
        if not entry.imported_may_import:
            failures.append(f"declared {label}")
            print(f"    [NO LEAF RULE] {label} -- name what {_short(entry.imported)} may "
                  f"import, or the reason is unchecked prose")
            continue
        if (entry.importer, entry.imported) not in present:
            failures.append(f"declared {label}")
            print(f"    [STALE] {label} -- no such import in the tree")
            continue
        if (tier[entry.importer], tier[entry.imported]) not in cyclic_pairs:
            failures.append(f"declared {label}")
            print(f"    [UNUSED] {label} -- the import exists but closes no cycle")
            continue

        print(f"    {label}")
        print(_reason_lines(entry.reason, 8))
        reached = sorted({e for e in edges if e.importer == entry.imported},
                         key=lambda e: (e.imported, e.lineno))
        broken = [e for e in reached if e.imported not in entry.imported_may_import]
        allowed = ", ".join(sorted(_short(m) for m in entry.imported_may_import))
        print(f"        leaf rule: {_short(entry.imported)} may import {allowed}")
        if broken:
            failures.append(f"leaf rule {label}")
            for edge in broken:
                print(f"        [LEAF BROKEN] {_site(edge, modules)} -> {_short(edge.imported)}")
        else:
            names = ", ".join(_short(e.imported) for e in reached) or "nothing"
            print(f"        holds -- it imports {names}")

    if args.tiers:
        _print_tiers(modules, edges, tier)

    print("\nRESULT:", "FAIL" if failures else "PASS")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
