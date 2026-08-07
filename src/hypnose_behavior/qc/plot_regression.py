#!/usr/bin/env python
"""Old-vs-new regression for what the plotters DRAW.

``regression.py`` fingerprints ``trial_data`` + the metrics dict and never sees a
figure, so every change inside ``visualization/`` is unguarded by it. This closes
that gap: it runs each plotter under Agg against a reference git revision *and*
the working tree, then compares every Line2D's xy data, every collection's
offsets, every patch's geometry, the axis decoration, and stdout.

Deliberately **not** a golden master. Figures are expected to change as the
plotters evolve, so a stored fixture would be stale within a phase; the useful
question is always "did *this* change move a curve", which is a two-tree diff.

Usage
-----
  PY=~/miniconda3/envs/hypnose-analysis-test/bin/python
  QC=~/repos/harris_lab/hypnose/hypnose-behavior-analysis/src/hypnose_behavior/qc

  $PY -u $QC/plot_regression.py                 # working tree vs HEAD
  $PY -u $QC/plot_regression.py --ref f72d201   # ... vs any revision
  $PY -u $QC/plot_regression.py --only plot_decision_accuracy

Exit code 0 == GREEN (nothing drawn differently); 1 == RED. Reads the real
derivatives tree; writes only to a temp dir.

Two things it has already caught, both invisible to ``regression.py``:
a `pd.concat` over a variable deleted by a refactor (every session silently
skipped behind a bare ``except``), and confirmation that Q5's denominator change
moved exactly two plots and nothing else.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]

# Modules searched for each case's function, in order. A function is looked up by
# name across all of them, so **moving** one between modules -- which Phase 4a does
# repeatedly -- stays invisible to the diff, while any change to what it draws does
# not. Without this a pure move reads as "function not found in this tree".
MODULES = [
    "hypnose_behavior.visualization.visualization_utils",
    "hypnose_behavior.visualization.pred_seq_utils",
    "hypnose_behavior.visualization.sing_rew",
    "hypnose_behavior.visualization.movement_analysis_utils",
    "hypnose_behavior.visualization.movement_analysis.sing_rew_movement",
    # Phase 5's display primitives. No case resolves here today, but a plotter
    # moved onto them later must stay resolvable -- an unlisted module reads as
    # "function not found", which is untestable rather than green.
    "hypnose_behavior.visualization.primitives",
    "hypnose_behavior.visualization.prep",
    # Phase 4b split `metrics_utils` into the modules below. It is still named
    # here so the gate can resolve a case against a *pre-split* revision --
    # `_resolve` swallows the ImportError, so naming a module that no longer
    # exists (or does not exist yet) costs nothing, and without it every moved
    # function reads as "not found in this tree", i.e. a RED for a pure move.
    "hypnose_behavior.metric_analysis.metrics_utils",
    "hypnose_behavior.metric_analysis.metrics.accuracy",
    "hypnose_behavior.metric_analysis.metrics.false_alarm",
    "hypnose_behavior.metric_analysis.metrics.sequence",
    "hypnose_behavior.metric_analysis.metrics.hidden_rule",
    "hypnose_behavior.metric_analysis.metrics.sampling",
    "hypnose_behavior.metric_analysis.metrics.timing",
    "hypnose_behavior.metric_analysis.metrics.common",
    "hypnose_behavior.metric_analysis.movement",
    "hypnose_behavior.metric_analysis.sing_rew_metrics",
    "hypnose_behavior.metric_analysis.run",
]

# Subjects/dates from the QC coverage set: hidden-rule multi-run (040 20251124),
# hidden-rule single run (040 20251229), and 048 20260306 for a second animal.
CASES = [
    ("plot_decision_accuracy_rolling_average", [40], {"dates": [20251124, 20251229]}),
    ("plot_cumulative_rewards", [[40]], {"dates": [20251124, 20251229]}),
    ("plot_response_times_completed_vs_fa", [40], {"dates": [20251124, 20251229]}),
    ("plot_fa_ratio_a_over_sessions", [40], {"dates": [20251124, 20251229]}),
    ("get_fa_ratio_a_stats", [40], {"dates": [20251124, 20251229]}),
    ("plot_abortion_and_fa_rates", [40], {"dates": [20251124, 20251229]}),
    ("plot_fa_ratio_by_hr_position", [40], {"dates": [20251124, 20251229]}),
    ("plot_fa_ratio_by_abort_odor", [40], {"dates": [20251124, 20251229]}),
    ("plot_position_completion_rate", [[40, 48]], {"dates": [20251124, 20251229, 20260306]}),
    ("plot_false_alarm_rate_by_position", [[40, 48]], {"dates": [20251124, 20251229, 20260306]}),
    ("hidden_rule_and_false_alarm", [[40]], {"dates": [20251124, 20251229]}),
    ("plot_hr_reward_fraction_over_trials", [40], {"dates": [20251124, 20251229]}),
    ("plot_hidden_rule_abort_poke_gap", [40], {"dates": [20251124, 20251229]}),
    ("plot_sampling_times_analysis", [40], {"dates": [20251124, 20251229]}),
    ("plot_poke_duration_by_position", [40], {"dates": [20251124, 20251229]}),
    ("plot_poke_duration_by_odor", [[40]], {"date": [20251124, 20251229]}),
    ("plot_decision_accuracy", [[40, 48]], {"dates": [20251124, 20251229, 20260306]}),
    ("plot_behavior_metrics", [[40]], {
        "dates": [20251124, 20251229],
        "variables": ["decision_accuracy", "global_FA_rate", "sequence_completion_rate"]}),
    ("plot_decision_accuracy_by_odor", [40], {"dates": [20251124, 20251229]}),
    # The two consumers of `_load_subject_trial_timeline` -- the only thing that
    # sees checklist 7 (`inter_trial_interval`).
    ("plot_iti_over_time", [[40]], {"dates": [20251124, 20251229]}),
    ("plot_latency_over_time", [[40]], {"dates": [20251124, 20251229]}),
    # pred_seq_utils (checklist 10-15). sub-053 20260520 is the seqLen-2 session,
    # so it exercises the `_sequence_len_ok` skip alongside sub-040's 3+ sequences.
    ("trial_poke_duration", [[40]], {"dates": [20251124, 20251229]}),
    ("response_time", [[40]], {"dates": [20251124, 20251229]}),
    ("fa_analysis", [[40]], {"dates": [20251124, 20251229]}),
    ("valve_to_reward", [[40]], {"dates": [20251124, 20251229]}),
    ("cummulative_poke_time", [[40]], {"dates": [20251124, 20251229]}),
    ("performance", [[40]], {"dates": [20251124, 20251229]}),
    # The rolling branch is a different code path (`_plot_performance_rolling`,
    # i.e. `over_windows`), and nothing else reaches it.
    ("performance#rolling", [[40]], {"dates": [20251124, 20251229],
                                     "moving_avg": True, "window_size": 10}),
    # sing_rew (checklist 16-17). sub-057 20260709 is the one single-reward
    # session in the coverage set; on any other subject these draw nothing.
    ("FR_ratio", [[57]], {"dates": [20260709]}),
    ("FR_latency", [[57]], {"dates": [20260709]}),
    # movement_analysis_utils (checklist 18-19). sub-040 20251124 is the one
    # coverage session with SLEAP tracking *and* a speed_analysis.parquet.
    ("plot_epoch_speeds_by_condition", [40], {"dates": [20251124]}),
    ("plot_traces_with_speed_threshold", [40], {"dates": [20251124]}),
    # Added in Phase 5 with the shared trajectory prep. These three are the
    # consumers of `prep.resample_trace` / `prep.smooth_xy` and of the three
    # divergent `_infer_port` variants, and until now none of them was covered
    # -- so that de-duplication would have been verified by nothing.
    ("plot_trial_traces_by_mode", [40], {"dates": [20251124], "mode": "all_trials",
                                         "show_average": True, "save": False}),
    ("plot_tortuosity_lines_overlay", [40], {"dates": [20251124], "save": False,
                                             "verbose": False}),
    ("plot_category_traces", [40], {"dates": [20251124], "show_average": True,
                                    "save": False}),
]

# Runs inside the child process, against whichever tree is on sys.path.
_CHILD = r'''
import contextlib, importlib, io, json, sys, traceback
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# The notebooks apply the repo style before plotting, and some plotters assume it:
# `_style_log_yaxis` reads `rcParams["ytick.labelsize"]` as a float, which is the
# string "medium" under matplotlib's defaults. Without this the latency/ITI cases
# raise identically in both trees, i.e. are silently ungated rather than green.
from hypnose_helpers.viz.styles import use_style
use_style("nature")

CASES = json.loads(sys.argv[2])
ONLY = json.loads(sys.argv[3])
MODULES = json.loads(sys.argv[4])


def _resolve(name):
    """First module in MODULES that defines `name`, or None.

    A case may be labelled `name#variant` so one function can be exercised with
    two argument sets (e.g. `performance` daily vs rolling) without the two
    results overwriting each other in the report.
    """
    name = name.split("#", 1)[0]
    for mod_name in MODULES:
        try:
            mod = importlib.import_module(mod_name)
        except Exception:
            continue
        fn = getattr(mod, name, None)
        if fn is not None:
            return fn
    return None


def canon(o):
    if isinstance(o, dict):
        return {str(k): canon(v) for k, v in sorted(o.items(), key=lambda kv: str(kv[0]))}
    if isinstance(o, (list, tuple)):
        return [canon(v) for v in o]
    if isinstance(o, np.ndarray):
        return canon(o.tolist())
    if isinstance(o, (np.floating, float)):
        return repr(float(o))
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, (np.bool_, bool)):
        return bool(o)
    if isinstance(o, (int, str)) or o is None:
        return o
    return repr(o)


def axes_sig(fig):
    out = []
    for ax in fig.get_axes():
        out.append({
            "title": ax.get_title(), "xlabel": ax.get_xlabel(), "ylabel": ax.get_ylabel(),
            "xlim": canon(ax.get_xlim()), "ylim": canon(ax.get_ylim()),
            "xticklabels": [t.get_text() for t in ax.get_xticklabels()],
            "yticklabels": [t.get_text() for t in ax.get_yticklabels()],
            "legend": ([t.get_text() for t in ax.get_legend().get_texts()]
                       if ax.get_legend() else []),
            "lines": [canon(np.asarray(l.get_xydata(), dtype=float)) for l in ax.get_lines()],
            "collections": [canon(np.asarray(c.get_offsets(), dtype=float))
                            for c in ax.collections if hasattr(c, "get_offsets")],
            "patches": [canon([p.get_x(), p.get_y(), p.get_width(), p.get_height()])
                        for p in ax.patches if hasattr(p, "get_x")],
        })
    return out


def sig(res):
    if res is None:
        return None
    if isinstance(res, plt.Figure):
        return {"__fig__": axes_sig(res)}
    if isinstance(res, pd.DataFrame):
        return {"__frame__": canon(res.to_dict(orient="records"))}
    if isinstance(res, (list, tuple)):
        return [sig(r) for r in res]
    if hasattr(res, "figure") and hasattr(res, "get_xlabel"):
        return {"__axes__": None}
    return canon(res)


out = {}
for name, args, kwargs in CASES:
    if ONLY and name not in ONLY:
        continue
    fn = _resolve(name)
    if fn is None:
        out[name] = {"error": "function not found in this tree"}
        continue
    for attempt_kwargs in ({"save": False, **kwargs}, kwargs):
        buf = io.StringIO()
        try:
            # Several plotters jitter with the global RNG and never seed it.
            np.random.seed(0)
            with contextlib.redirect_stdout(buf):
                res = fn(*args, **attempt_kwargs)
            out[name] = {"result": sig(res), "stdout": buf.getvalue()}
            break
        except TypeError as e:
            if "save" in str(e) and attempt_kwargs is not kwargs:
                plt.close("all")
                continue
            out[name] = {"error": traceback.format_exc().strip().splitlines()[-1],
                         "stdout": buf.getvalue()}
            break
        except Exception:
            out[name] = {"error": traceback.format_exc().strip().splitlines()[-1],
                         "stdout": buf.getvalue()}
            break
    plt.close("all")

with open(sys.argv[1], "w") as fh:
    json.dump(out, fh, indent=1, sort_keys=True)
'''


def _flatten(obj, prefix=""):
    out = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.update(_flatten(v, f"{prefix}.{k}" if prefix else str(k)))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            out.update(_flatten(v, f"{prefix}[{i}]"))
    else:
        out[prefix] = obj
    return out


def _run(tree: Path, out_path: Path, only: list[str]) -> None:
    script = out_path.parent / "_plot_child.py"
    script.write_text(_CHILD)
    # PYTHONHASHSEED: several plotters order groups by iterating a `set` of
    # strings (`pred_seq_utils._ordered_groups`, fed from `all_groups = set()`),
    # so the drawn order varies between processes. That is a real reproducibility
    # defect in those figures -- but it is pre-existing, and left unpinned it
    # makes the two trees differ by 340 values with no source change at all.
    env = {**__import__("os").environ, "PYTHONPATH": str(tree / "src"),
           "PYTHONHASHSEED": "0"}
    subprocess.run([sys.executable, "-u", str(script), str(out_path),
                    json.dumps(CASES), json.dumps(only), json.dumps(MODULES)],
                   env=env, check=True)


def _materialise(ref: str, dest: Path) -> Path:
    """Export `ref` from the repo, plus the git-ignored local data-location profile."""
    dest.mkdir(parents=True, exist_ok=True)
    archive = subprocess.run(["git", "-C", str(REPO), "archive", ref],
                             check=True, capture_output=True).stdout
    subprocess.run(["tar", "-x", "-C", str(dest)], input=archive, check=True)
    local = REPO / "configs" / "data_locations.local.yml"
    if local.exists():
        # Without it the exported tree resolves its own (absent) config and
        # silently falls through to a wrong derivatives root -- the 2a __file__ trap.
        shutil.copy(local, dest / "configs" / local.name)
    return dest


def _magnitudes(old_flat, new_flat, changed) -> str:
    """How *big* the changed values are, not just how many.

    A repoint onto a canonical metric routinely moves drawn values by a ULP or by
    the nanosecond that vectorised `.dt.total_seconds()` keeps and the scalar path
    truncates. Both are RED, and both look identical to a moved curve in a list of
    ten diffs. This line separates them: `max |dy|` and `max rel` say at a glance
    whether anything left floating-point noise.
    """
    abs_d, rel_d, non_numeric = 0.0, 0.0, 0
    for k in changed:
        try:
            a, b = float(old_flat[k]), float(new_flat[k])
        except (TypeError, ValueError):
            non_numeric += 1
            continue
        d = abs(b - a)
        abs_d = max(abs_d, d)
        scale = max(abs(a), abs(b))
        if scale > 0:
            rel_d = max(rel_d, d / scale)
    if not changed:
        return ""
    numeric = len(changed) - non_numeric
    parts = [f"{numeric} numeric: max |dy| = {abs_d:.3g}, max rel = {rel_d:.3g}"]
    if non_numeric:
        parts.append(f"{non_numeric} non-numeric (labels, limits, text)")
    return " | ".join(parts)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ref", default="HEAD", help="git revision to compare against (default HEAD)")
    ap.add_argument("--only", nargs="*", default=[], help="limit to these plotter names")
    args = ap.parse_args()

    with tempfile.TemporaryDirectory(prefix="hyp_plotreg_") as tmp_str:
        tmp = Path(tmp_str)
        print(f"Materialising {args.ref} ...")
        ref_tree = _materialise(args.ref, tmp / "ref")
        old_path, new_path = tmp / "old.json", tmp / "new.json"
        print(f"Running plotters against {args.ref} ...")
        _run(ref_tree, old_path, args.only)
        print("Running plotters against the working tree ...")
        _run(REPO, new_path, args.only)
        old = json.loads(old_path.read_text())
        new = json.loads(new_path.read_text())

    red = 0
    print()
    for name in sorted(set(old) | set(new)):
        if name not in old or name not in new:
            print(f"  [MISSING] {name}")
            red += 1
            continue
        o, n = old[name], new[name]
        if ("error" in o) != ("error" in n) or o.get("error") != n.get("error"):
            print(f"  [RED]  {name} raise state changed")
            print(f"           ref: {o.get('error', '(ran ok)')}")
            print(f"           new: {n.get('error', '(ran ok)')}")
            red += 1
            continue
        if "error" in o:
            print(f"  [both raise, unchanged] {name}: {o['error'][:60]}")
            continue

        fo, fn_ = _flatten(o["result"]), _flatten(n["result"])
        added = sorted(set(fn_) - set(fo))
        removed = sorted(set(fo) - set(fn_))
        changed = sorted(k for k in set(fo) & set(fn_) if fo[k] != fn_[k])
        stdout_differs = o["stdout"] != n["stdout"]
        if added or removed or changed or stdout_differs:
            red += 1
            print(f"  [RED]  {name}: "
                  f"{len(added)} added, {len(removed)} removed, {len(changed)} changed"
                  f"{', stdout differs' if stdout_differs else ''}")
            for k in (removed[:10] + added[:10]):
                src = fo if k in fo else fn_
                print(f"           {'-' if k in fo else '+'} {k} = {src[k]!r}")
            for k in changed[:10]:
                print(f"           ~ {k}: {fo[k]!r} -> {fn_[k]!r}")
            if len(added) + len(removed) + len(changed) > 30:
                print(f"           ... {len(added) + len(removed) + len(changed)} entries total")
            summary = _magnitudes(fo, fn_, changed)
            if summary:
                print(f"           {summary}")
        else:
            print(f"  [green] {name}")

    print()
    if red:
        print(f"PLOT REGRESSION RED: {red} plotter(s) draw something different.")
        return 1
    print(f"PLOT REGRESSION GREEN: {len(old)} plotters draw identically to {args.ref}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
