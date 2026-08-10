# restructure_2 — consolidation & reuse plan (target: v2.0.0)

Hand-off plan for the next round of work on `hypnose-behavior-analysis`, on branch
**`hypnose-restructure`**.

Goal: make the code tidier, faster and reusable across the growing repo family — **without
accidentally changing analysis output**.

**Three live documents, and nothing else:**

| document | what it is |
|---|---|
| `docs/restructure_2_plan.md` (this) | the spine: remaining phases and the operating rules |
| `docs/DECISIONS.md` | settled rules and standing traps — **read at the start of every phase** |

`docs/archive/` holds closed working documents. **Nothing live points into it**, and no phase
needs to read it.

---

## 0. Context a fresh session needs

**What v1.0.0 delivered** (done — do NOT redo): package `hypnose` under `src/hypnose/` with
`io/`, `trial_classification/`, `metric_analysis/`, `visualization/`, `utils/`, `qc/`;
terminal entry points in `scripts/`; no back-compat shims, all imports canonical.

**The repo family this now lives in:**

| repo | package | role |
|---|---|---|
| **hypnose-behavior-analysis** | `hypnose_behavior` | behavioural analysis (this repo) |
| hypnose-somnotate | `hypnose_somnotate` | EEG sleep scoring (done, v1.0.0) |
| hypnose-eeg-analysis | `hypnose_eeg` | EEG analysis (coming) |
| neuropixel analysis | `hypnose_ephys` | ephys (planned) |
| **hypnose-helpers** | `hypnose_helpers` | shared, modality-agnostic utilities (exists, local only) |

**Current scale** — measured 2026-08-07, after Phase 5. The total *rose*: Phase 5 added
documentation and three shared leaves, and it was never a de-bloat (see Phase 10).

| area | lines |
|---|---|
| `visualization/` **total** | **16,015** |
| ├ `visualization_utils.py` | 6,777 |
| ├ `movement_analysis_utils.py` | 3,606 |
| ├ `pred_seq_utils.py` | 1,658 |
| ├ `sing_rew.py` | 1,364 |
| ├ `modelling/switchpoint/plots.py` | 648 |
| ├ `valve_poke_plots.py` | 635 |
| ├ `prep.py` / `panels.py` / `primitives.py` (the shared leaves) | 501 / 264 / 125 |
| └ `movement_analysis/sing_rew_movement.py` | 428 |
| `trial_classification/classification_utils.py` | 3,703 |
| `metric_analysis/` | `run` / `merge` / `summary` / `registry` / `frames` / `resolvers` / `movement` / `stats/` + 7 modules under `metrics/`, none over 710 lines |

`metric_analysis/metrics/__init__.py` is the map of where every metric lives, in one screen.

**Out of scope (explicit):** do NOT change protocol detection — the
`"odourdiscrimination" in name` string matching stays as-is.

---

## 1. The QC safety net — use it after every change

`src/hypnose_behavior/qc/`, see `qc/README.md`:

- **`regression.py`** — golden-master. Fingerprints `trial_data` + metrics dict for the 9
  coverage sessions in `sessions.yml`, md5-compares to `fixtures/`. On mismatch it reports
  **added/removed/changed columns and metric keys**, so an intended change is easy to confirm.
  `--generate` writes baselines; optional `subjid:date` args limit scope. ~15 min.
- **`plot_regression.py`** — old-vs-new diff of what the **plotters draw**. `regression.py`
  never sees a figure, so everything in `visualization/` is invisible to it. Runs 35 plotter
  cases under Agg against a git revision and the working tree and diffs every line's xy data,
  collection offsets, patch geometry, axis decoration and stdout. Deliberately a two-tree diff,
  not a golden master: figures are meant to change, and the question is always whether *this*
  change moved a curve. ~5 min. **Phases 5 and 10 depend on it** — see `DECISIONS.md` §7 for
  the three things it works around.
- **`verify_scripts.py`** — regression through the actual CLI scripts (covers arg wiring).
- **`check_imports.py`** — static check for referenced-but-not-imported globals. Seconds.
- **`validate.py`** — `validate_subject()`, used by the scripts.

**Operating rules**

- Run everything with `~/miniconda3/envs/hypnose-analysis-test/bin/python`. Fixtures are only
  valid in the env recorded in `fixtures/env.json` — `py3.12.13 / pandas 3.0.1 / numpy 1.26.4`,
  plus pyarrow, which `env_fingerprint` now records because a pyarrow upgrade silently switched
  `load_session_results` from the CSV fallback to parquet and moved a ULP.

- **Invoke the QC tools by absolute path, with no `cd`** (learned the hard way, 2026-08-04):

  ```bash
  PY=~/miniconda3/envs/hypnose-analysis-test/bin/python
  QC=~/repos/harris_lab/hypnose/hypnose-behavior-analysis/src/hypnose_behavior/qc

  $PY -u $QC/regression.py            # optional: subjid:date ... to limit scope
  $PY -u $QC/plot_regression.py       # optional: --ref <rev>, --only <name>
  $PY -u $QC/check_imports.py
  $PY -u $QC/verify_scripts.py
  ```

  Two reasons, both of which have already cost a session:

  - The `cd <repo> && $PY src/...` form gets stopped at the agent permission layer, and
    the tool reports it as a rejected call — indistinguishable from a hung or slow run.
    None of the tools needs a working directory: each derives `HERE`/`REPO` from `__file__`
    and puts `src/` on `sys.path` itself.
  - `-u` matters when the output is redirected (backgrounded runs, tee to a log). Without
    it Python block-buffers stdout, so a 5-minute run shows nothing at all until it exits
    and there is no way to tell progress from a stall.

- **Gate on reachability, not on how large the change looks.** Run `regression.py` for any edit
  inside a function `run_all_metrics` reaches, or anything touching `load_session_results`. Skip
  it for code the metrics pipeline never calls — the fingerprint cannot see it and a 15-minute
  run proves nothing; verify those by byte-identity against `git show HEAD:...` plus
  `check_imports`.

- **Check printed output, not just values.** `metrics_*.txt` is written from the wrappers'
  stdout and is **not** in the fingerprint, so a print-only drift is invisible to
  `regression.py`. The old-vs-new parity harness used through Phase 4 — export the pre-change
  tree with `git archive`, compare both the return value and captured stdout across all 9
  sessions — runs in ~7 s and has caught exactly that. **Copy the repo's git-ignored
  `configs/data_locations.local.yml` into the exported tree**, or it resolves its own absent
  config and silently reads a wrong derivatives root.

- **Byte-identical philosophy.** Pure refactors and moves must keep regression GREEN.
  Intended output changes (schema, vectorisation numerics) get fixtures **regenerated
  deliberately in the same commit**, with the column/metric diff confirming only the
  intended fields changed.

- Commit per logical step; keep the tree GREEN between commits where possible.

---

## 2. How to work through this plan

### One phase per chat

**Do not attempt this in a single long session.** Start a fresh chat for each phase,
using this document as the handoff. Section 0 exists precisely so a cold session can
pick up without prior context.

Long sessions degrade in exactly the way this work cannot tolerate: dropped items when
summarising, repeated identical tool mistakes, and mislabelled results where the
narration disagrees with the output. On work whose entire premise is *do not silently
change the output*, that failure mode matters more than the convenience of continuity.

Commit at every phase boundary so each new chat starts from a green, known state.

### Progress

Update this table at the end of each phase, in the same commit as the work.

| phase | status | commit | what it delivered / what still constrains later phases |
|---|---|---|---|
| 0 — re-baseline QC fixtures | **done** 2026-08-03 | `481110b` | 9 sessions baselined. Supersedes Phase 3 — do NOT re-run `--generate` |
| 0.1 package name | **done** 2026-08-03 | `9aad717` | `hypnose_behavior`, dist `hypnose-behavior-analysis` |
| 0.2 helpers boundary | **done** 2026-08-03 | `9793cbc`..`b840ba1` | Decided by the "knows the data vs knows the layout" test |
| 0.3 collapse loaders/readers | **done** 2026-08-03 | `5d9c14a` | `readers.py` is the single definition site for the 8 primitives; `loaders.py` re-exports. Kept as two files — deleting `readers.py` makes `loaders → detect_settings → loaders` a cycle |
| 1 rename | **done** 2026-08-03 | `9aad717` | `hypnose` → `hypnose_behavior`. **Outstanding:** the folder move invalidated the editable-install `.pth`; `hypnose`, `hypnose-analysis`, `hypnose-somnotate`, `sleap`, `sleap-2` still need `pip install -e .` (user action) |
| 2a helpers extraction | **done** 2026-08-03 | `9793cbc`..`b840ba1` (helpers `1333955`..`d11d0dc`, somnotate `9e3c155`..`7de20a5`) | `hypnose-helpers` created (local only, no remote). Behaviour repo −1226 lines. rcParams no longer mutated at import. **Trap for any later "move whole": check for `__file__`-derived state first** — `io/paths.py` could not move because everything derived from `Path(__file__).parents[3]`, and relocating it silently repointed the config with no error |
| 2b canonical session discovery | **done** 2026-08-04 | `312854d`..`2fbd5f5` (helpers `8dadcf8`, somnotate `80afbda`) | `hypnose_helpers.io.layout` owns the walking; each repo binds its own roots. All 17 lookups and ~30 subject globs repointed. Ambiguity raises `DuplicateSessionError`. **`session_index` selects, never positions — `DECISIONS.md` §8** |
| 2c figure provenance | **done** 2026-08-04 | `7d117b8` (helpers `059c652`, somnotate `892e56c`) | Every `save_figure` PDF carries commit/version/caller/params; `read_figure_metadata` reads it back without pypdf. Phase 7a can call `provenance()` directly. **Wrapper hazard — `DECISIONS.md` §9** |
| 3 re-baseline QC | **done** — superseded by 0 | `481110b` | Nothing further to do |
| 4a strip metrics from visualization | **done** 2026-08-06 | audit `58387ce`..`6aac5de`; moves `9672f1e`..`aa0355f` | **`visualization/` fetches and plots, and computes no metrics.** 32 sites of metric math resolved: 24 metrics added to `metric_analysis`, 8 exact dedups, 9 variants collapsed onto `by_group`/`over_windows`. D0 complete for every tier. `compute_speed_analysis` (711 lines, 7 movement metrics) moved wholesale to `metric_analysis/movement.py`. Two deliberate output changes landed (non-initiated trials left the metric set; `ambiguous_rate`/`correct_rejection_rate` added). `qc/plot_regression.py` written and grown to 32 cases. **Deviation that outranks the audit: `_load_tracking_and_behavior` → `io/tracking.py`, not `visualization/io/`** |
| 4b modularise metric_analysis | **done** 2026-08-07 | `604355f`..`cbc7059` | **`metrics_utils.py` (2,639 lines) is gone.** Plumbing to `io/load_results.py` + `run`/`merge`/`summary`; definitions to 7 modules under `metrics/`. Registry: 43 registered / 25 reported. Every carve verified by an ast pass requiring all 112 pre-4b function bodies byte-identical. One intended output change (`fa_abortion_stats` numeric). **Registry contract, `frames.py`-as-leaf and the legacy-reader trap — `DECISIONS.md` §3-5** |
| 5 visualization primitives | **done** 2026-08-07 | `e532ab7`..`2ce3dcc` | **Both 4a defects fixed.** **No plotter reads `metrics_*.json`** — trap discharged, and it exposed a duplicate-rows bug the JSON path was hiding. `position_data` lazy; `load_results_dir` split out. Three shared leaves: `primitives.py` (`mean_sem` 18 sites, `sem_band`, `rolling_windows`), `prep.py`, `panels.py`. **All 44 session-selecting functions take the six selectors.** **Zero plotter-to-plotter imports.** Two briefs were measurably wrong and were not followed: finding 10's "duplicates" disagree on up to **63.8%** of trials, and section 2's helpers were never duplicated — `DECISIONS.md` §13. `style_axis` dropped as unearned (§12). **Gate grew 32 → 35 cases**; "no plot function over ~100 lines" is explicitly left to Phase 10 |
| 6a split the 4 long functions + unify the outcome rule | **done** 2026-08-10 | `97a01ad`..`HEAD` | **The four monoliths are gone**: 3,005 → 977 lines (`classify_trials` 1226→419, `abortion_classification` 736→162, `analyze_response_times` 580→263, `detect_trials` 462→133); the file 3,703 → 3,023. **`regression.py` GREEN on every commit, including the unification** — nothing in this phase moved a value. Two new leaves: `windows.py` (poke/valve primitives) and `outcome.py` (`classify_completed_trial` + `latency_label`), both importing nothing from the package, so `io/save_results.py` reaches the shared rule without a cycle (§3). **3 outcome sites → 1, 3 latency-bucket copies → 1.** The three valve-edge builders and the two poke summaries were measured and **kept apart** under names saying how they differ. 106 lines of dead nested defs dropped. **Measured before merging (§13): 1,731 trials, the rule never conflicts — `DECISIONS.md` §14.** New guard `qc/verbose_diff.py` (stdout, which `regression.py` never sees) and `qc/outcome_agreement.py` |
| 6b `poke_source` + the 0 ms positions | not started | | `DECISIONS.md` §10. **Intended output change**, fixtures regenerated in the same commit. 6a is done, so this is unblocked. Two acceptance tests waiting for it: re-run `qc/outcome_agreement.py` and the one remaining conflict should be **gone** (§14), and the response-time fallback of §15 should stop firing once positions carry `poke_source` |
| 7a manifest provenance | not started | | ~40% done in advance by 2c |
| 7b schema & formats | not started | | intended output change |
| 8 profile, then vectorise | not started | | |
| 9 validation | not started | | |
| 10 modularise `visualization_utils.py` | **proposed**, after the restructure | | not scheduled |
| ∥ time-base audit | not started, deferred | | parallelisable |

### Model and reasoning effort

Use **Opus 5 throughout** — the failure modes here are subtle correctness, not
throughput. Vary the effort by phase:

| phase | effort | why |
|---|---|---|
| 5 | high | judgement about what a figure should look like, on top of moves that must not change one |
| **6 classification dedup** | **max** | riskiest item in the plan — 3 divergent implementations of one rule, ~1000-line function |
| 7-9 | high | schema is deliberate-change territory; profiling is evidence-led |
| 10 | standard–high | pure moves, gated exactly by `plot_regression.py` |
| any unexpected RED | **max** | always — diagnose before touching anything |

Standard effort on Phase 6 is how a subtle behaviour change passes regression on 8 sessions
and breaks on the 9th.

### Handoff prompt for each new chat

Adapt the phase name and paste:

```
I'm continuing a planned restructure of hypnose-behavior-analysis
(/Users/joschua/repos/harris_lab/hypnose/hypnose-behavior-analysis), on branch
`hypnose-restructure`. Use Opus 5 at <EFFORT> effort.

FIRST, read these two, in this order:
1. `docs/DECISIONS.md` in full — the settled rules and standing traps. Every one of them
   prevents a silently wrong number or constrains a choice you would otherwise make freely.
2. `docs/restructure_2_plan.md` — section 1 (QC safety net + operating rules), section 2
   (Progress table — do not redo completed phases), and the whole <PHASE> section.

Do NOT read anything in `docs/archive/`. It is closed working material, its line numbers are
stale, and several of its proposed destinations were overruled during implementation.

This chat covers exactly one phase: <PHASE>. Do not start any other phase.

Hard constraints:
1. The ceph mount `/Volumes/harris` is STRICTLY READ-ONLY. Never write, move, rename,
   chmod or delete anything under it. If you think you need to write there, stop and ask me.
2. Do not explore ceph — no browsing subject folders, no inventorying sessions, no
   `find` over the mount. To learn about a session, call the pipeline's own loaders for
   specific subject/date pairs. If every session fails with FileNotFoundError or "No
   experiment runs found", that is a dropped or weak mount, not a code regression — CHECK
   THE MOUNT BEFORE DIAGNOSING. It also goes slow after a heavy gate run; an 85 s
   `load_session_results` is cache eviction, not your bug.
3. Run everything with `~/miniconda3/envs/hypnose-analysis-test/bin/python`. Do not
   install, upgrade or remove packages in any conda env — fixtures are only valid in
   the env recorded in `qc/fixtures/env.json`. If something is missing, tell me.
4. Invoke the QC tools by ABSOLUTE path with `-u` and no `cd` — see "Operating rules"
   in section 1. The `cd <repo> && python src/...` form is blocked at the permission
   layer and surfaces as a rejected call that looks like a hang.
5. `~/repos/harris_lab` is itself a commitless git repo. Always use `git -C /full/path`.
6. `hypnose_helpers` must import nothing from the other two repos.

Workflow:
- Tell me what you are about to do and what gate result you expect, before doing it.
- After the change, run the gate that can see it and SHOW ME THE FULL OUTPUT — do not
  truncate it.
- GREEN → commit. Unexpected RED → stop and diagnose, do not regenerate fixtures to
  make it pass.
- An intended output change gets fixtures regenerated in the same commit, with the
  +/-/~ diff confirming only the intended fields moved — and ask me first.
- Commit messages: subject + 2-3 short sentences MAX, then gate results, then
  `Co-Authored-By: Claude Opus 5` (no email). Rationale goes in the chat or the plan.
- At the end: update the Progress table in the plan, and commit that with the work. Add
  anything newly settled to `docs/DECISIONS.md`. If you run out of room first, say so
  plainly and leave an accurate status.

Ask me rather than guessing if a decision is not settled in the plan.
```

### Context strategy for large phases

If a phase's inputs exceed one context, work file by file and **write the working notes into
the repo as you go**, then have the implementing chat read those notes instead of the source.
That is how Phase 4 was done. When the phase closes, extract what is still load-bearing into
`DECISIONS.md` and archive the rest — a 1,500-line working document is a tax on every
subsequent chat.

---

## Phases 0-4 — done

Summarised in the Progress table above, with commit ranges. What they decided that still
binds later work is in `docs/DECISIONS.md`, with the measurements it rests on; the narrative
is in `git log`.

- **Phase 0** — package name, helpers boundary, `loaders`/`readers` collapse, QC re-baseline.
- **Phase 1** — the rename.
- **Phase 2** — `hypnose-helpers` extracted (2a), canonical session discovery (2b), embedded
  figure provenance (2c).
- **Phase 3** — superseded by Phase 0.
- **Phase 4** — metrics single source of truth: 4a stripped all metric math out of
  `visualization/`, 4b split `metrics_utils.py` into `metric_analysis/` proper with a registry.

**Still open from Phase 1, needing the user:** five conda envs still need `pip install -e .`
after the folder rename (`hypnose`, `hypnose-analysis`, `hypnose-somnotate`, `sleap`,
`sleap-2`). Only `hypnose-analysis-test` was repointed. `hypnose-helpers` must also be
`pip install -e`'d into any env that runs either repo.

---

## Phase 5 — Visualization: primitives, then thin plotters

*Closed. The work list is archived at `docs/archive/phase5_brief.md`; everything still
load-bearing from it is in `DECISIONS.md` §12-13 and in Phase 10 below. Do not read the
archived brief — two of its sections were measurably wrong about the code.*

`visualization/` is 15,124 lines across 7 files. 4a took the metric math out; what is left is
data prep, axis construction and styling.

**Measured primitive usage across `visualization/`:**

```
.plot(  64     .scatter( 69     .errorbar( 20     rolling( 20
.legend( 53    .set_xlabel( 55
.boxplot( 0    .hist( 0    .barh( 0    .bar( 1
```

One thing follows. The largest real repetition is **axis decoration**: 53 legends and 55 axis
labels — not plotting.

**Target shape:**

```python
# primitives (thin, no metric knowledge)
line(ax, df, x, y, **style)
scatter(ax, df, x, y, **style)
boxplot(ax, df, by, value, **style)
rolling_mean(series, window)          # + SEM/CI band helper
style_axis(ax, xlabel=…, ylabel=…, legend=…, title=…)

# per-metric plotters stay thin and explicit
plot_accuracy(ses, kind="line")   ->  registry call + primitives
```

These thin helpers live in extra `visualization/` helper modules, so they are shared across
files and can be re-imported.

**Deliberately avoid** a single `plot_metric(kind, ses)` dispatcher — it accumulates kwargs
for every plot type it supports and becomes a god-function. Thin primitives plus one small
function per metric give the same ergonomics without that.

### What Phase 5 delivered  *(closed 2026-08-07, `2ce3dcc`)*

All five open items are done. The gate went from 32 to **35 cases** on the way, because three of
the plotters this phase touched were covered by none.

1. **Session selectors threaded** (`ff3aaea`, `78c1ff7`, `543271b`). `_filter_sessions` /
   `_filter_session_dirs` take all six keywords and forward them to the shared `filter_sessions`;
   `session_selectors()` bundles the five non-date ones so each plotter forwards in one line.
   **44 functions across 5 modules; nothing interprets them.** `session_index` stayed a selector
   — no retrofit, no plot x-axis (`DECISIONS.md` §8).
2. **The rolling call sites** (`87b6680`) — but not onto `rolling_mean`, which fits none of them.
   They share the *windowing*, so that is the primitive: `rolling_windows`. `_rolling_pts` still
   calls `over_windows`, and switchpoint keeps its `np.convolve`. `DECISIONS.md` §12.
3. **`style_axis` dropped, deliberately** — 54 collapsible runs, ~76 lines, and no correctness
   payoff. The despining it would have carried already lives in the style. `DECISIONS.md` §12.
4. **Finding 10 and the shared prep module** (`463f0c4`, `2ce3dcc`) — **both briefs were wrong
   about what was there.** Finding 10's helpers are mostly different rules sharing a name
   (measured: `_infer_port` variants disagree on **63.8%** of trials), so two were shared and the
   rest renamed. Section 2's helpers were not duplicated at all; the real defect was 15 names
   imported *between sibling plotter modules*, now fixed by two leaves (`prep.py`, `panels.py`).
   **Zero plotter-to-plotter imports remain.** `DECISIONS.md` §13.
5. **The legacy `_fa_stat_*` string branches deleted** (`9dc66e1`).

**One stated completion criterion is deliberately not met:** "no plot function over ~100 lines".
`visualization_utils` still averages ~280 lines per plotter. That was never a Phase 5 job — it is
the Phase 10 split, and nothing here was going to move it.

Two things worth carrying forward:

- **`compute_if_missing` was removed** from `plot_behavior_metrics` and
  `hidden_rule_and_false_alarm`. It could not mean anything once metrics are always computed, and
  a parameter that does nothing is worse than no parameter — but it *is* a public signature
  change, so a notebook passing it now raises `TypeError`.
- **The gate's jitter sensitivity.** Several plotters jitter from an unseeded global RNG, so a
  change in the *number* of drawn points shifts every later draw and surfaces as dozens of
  "changed" values rather than a clean "added" list. Read a RED's `added`/`removed` counts first.

### Two defects 4a found and deliberately did not fix — **both now fixed**

Fixed in `7eddf84` (`_style_log_yaxis`, no drawn change) and `372d262` (`_ordered_groups`, an
intended reordering verified to be a pure permutation). `DECISIONS.md` sections 11 and 7 carry
what outlived them. The original statements follow for context.

Both surfaced while wiring `qc/plot_regression.py` up to the remaining plotters. Neither is
metric math, so neither belonged in 4a; both are exactly Phase 5's subject matter, and **both
are currently worked *around* in the gate rather than fixed in the source.**

1. **`_style_log_yaxis` crashes under default rcParams.** It does
   `float(plt.rcParams.get("ytick.labelsize", 12))`, and matplotlib's default for that key is
   the **string `"medium"`** — so it raises `ValueError: could not convert string to float`
   unless the repo style has been applied first. `plot_iti_over_time` and
   `plot_latency_over_time` are therefore broken in any process that has not called
   `use_style(...)`, which includes a plain notebook or script. The gate now calls
   `use_style("nature")` in its child so the two are testable at all; before that they raised
   identically in both trees and read as *ungated*, not as green. **Fix:** resolve the tick size
   through matplotlib's own font-size machinery rather than `float()`, or make the style a
   precondition the plotters assert. Good canary for the wider issue — how many other plotters
   silently assume the style?

2. **`pred_seq_utils._ordered_groups` iterates a `set`.** It is fed `all_groups = set()`
   (three sites), so any group label outside the hard-coded `preferred` list is drawn in
   **string-hash order**, which varies between processes. Those figures are not reproducible run
   to run: measured, two runs of the *identical* tree disagreed on 340 drawn values. The gate
   pins `PYTHONHASHSEED=0` to make its diff meaningful; that hides the defect, it does not fix
   it. **Fix:** order the residual labels deterministically (sorted, or first-seen). This **is
   an output change** — it reorders series and legends — so it wants its own commit and a look
   at the affected figures.

### Load vs compute — plotters stop reading `metrics_*.json`  ✅ **done**

Delivered in `c56107f`, `d35638e`, `c3e21d6`. `_ensure_metrics_json` is gone and **no plotter
reads the file**; `_computed_metrics` evaluates `adapter(session(results))`, the same expression
`run.py` uses to write it. `position_data` is lazy and `load_results_dir` splits the directory
read from the expensive session lookup.

**How a plotter computes a metric, why the bare core is not enough, and what converting the trap
exposed are now in `DECISIONS.md` §5** — read that, not this. Item 3 (where the nine per-trial
tables live) remains Phase 7b's.

### Thread the session selectors through the plotters *(decided 2026-08-04)*

Public plotting functions currently accept **`subjids` and `dates` only**. Phase 2b built
three interchangeable selectors in `hypnose_helpers.io.layout`; the plotters expose none of
them. Widening that interface is a Phase 5 job, because `dates` reaches ~36
`_filter_session_dirs` call sites and changing it earlier would touch files this phase
rewrites anyway.

**Accept all three, and pass them straight through:**

```python
def plot_accuracy(subjids, *, dates=None, ses=None, index=None,
                  date_range=None, ses_range=None, index_range=None, ...):
    for subjid in subjids:
        for session in derivatives.find_sessions(subjid, dates=…, ses=…, index=…):
            ...
```

`find_sessions` already takes exactly these six keywords, so a plotter should forward them
rather than reinterpret them. `utils/helpers._filter_sessions` is the `SessionRef`-returning
form to build on; `_filter_session_dirs` (paths only) is the legacy shim.

**Semantics — none is required, and they combine.** Verified against the implementation:

| given | result |
|---|---|
| all of `dates` / `ses` / `index` are `None` | **every session for the subject** — unchanged from today |
| exactly one | filter on that key |
| two or more | **intersection** — a session must satisfy *all* of them |
| `[]` (empty list) rather than `None` | **matches nothing** |

So they are *not* mutually exclusive alternatives, and a plotter must not treat them as
"pick one". `find_sessions(66, ses_range=(1, 9), index_range=(3, 5))` legitimately means
"of ses 1-9, the 3rd to 5th sessions chronologically". No validation should reject a
combination.

The `None` vs `[]` distinction is load-bearing rather than incidental: callers build a
per-subject date list and pass it straight through, so a subject with no requested dates
must yield no sessions rather than its whole history. There is a regression test for it.

**Why all three rather than picking one:**

| key | question it answers | when it is the right one |
|---|---|---|
| `date` | "what happened on 7 July" | a specific session you can name |
| `ses` | "session 40 of this animal" | the number in the lab book; stable, quotable |
| `index` | "its first nine sessions" | **comparing subjects across cohorts** |

`ses` and `index` are *not* interchangeable, and the difference is silent. Measured on a
three-subject tree: `ses="01-09"` returns **9, 3 and 0** sessions for a contiguous subject,
one with gaps, and one whose numbering carried over from an earlier protocol — while
`index_range=(1, 9)` returns **9 each**, spanning cohorts months apart. A subject numbered
from `ses-038` yields *nothing* for `ses 1-9` and does not error.

Most current subjects are contiguous, so `ses` usually behaves like `index`. That is exactly
why the distinction has to be explicit: it works until the one animal where it does not.

**`index` selects; it does not position** — see `DECISIONS.md` §8. Do not make it a plot x-axis.

**Risk:** low–med (plot-only changes don't move the `regression.py` fingerprint — which is
precisely why `plot_regression.py` is the gate that matters here).

**Delivered 2026-08-07** (`ff3aaea`, `78c1ff7`, `543271b`): every session-selecting function —
44 across the five plotter modules — accepts `dates`/`ses`/`index` and their range forms and
forwards them unchanged. `session_selectors()` in `utils/helpers.py` bundles the five non-date
keywords. Verified against sub-040's 48 sessions: the keys intersect, `[]` matches nothing, and
the legacy 2-tuple `dates` form is unchanged.

*Of the original "Done" criteria, all are met except "no plot function over ~100 lines", which
is Phase 10's split and was never reachable from here.*

---

### Naming debt to settle later — the (a)/(b) latency pair *(6c, 2026-08-10)*

Every reward-port latency now exists twice: **(a)** measured from where the response window
starts, which is what the label buckets, and **(b)** measured from the animal's last cue-port
exit before the poke, which is how fast it actually moved. The base names are **not** consistent
about which is which:

| family | (a) window-relative | (b) movement |
|---|---|---|
| completed | `completed_window_latency_ms` | **`response_time_ms`** |
| aborted (FA) | **`fa_latency_ms`** | `fa_movement_latency_ms` |
| no-go (FR) | **`fr_latency_ms`** | `fr_movement_latency_ms` |

`response_time_ms` is (b) while `fa_latency_ms` / `fr_latency_ms` are (a). That was a deliberate
trade: making `response_time_ms` mean (a) would have been consistent but repointed ~1063 values
and moved `avg_response_time` by about a second, where keeping it as (b) moved 39. **Unify the
naming in a later phase**, as one intended output change with its own fixture regeneration —
and note that the metric `avg_response_time` reads (b) today.

## Phase 6 — Trial classification  *(highest risk; split into 6a and 6b)*

**Split into two chats 2026-08-07**, because they are different *kinds* of change and mixing
them makes a RED undiagnosable: **6a must keep `regression.py` GREEN throughout**, while 6b
deliberately alters `trial_data` and regenerates fixtures. One chat doing both cannot tell a
broken refactor from the intended schema change.

### The shape of the problem, measured *(2026-08-07)*

`classification_utils.py` is 3,703 lines / 16 top-level functions, but **four of them are 84%
of the file's function body**. The other twelve are 13–147 lines and need nothing.

| lines | function | nested defs |
|---|---|---|
| 1,226 | `classify_trials` | 9 |
| 736 | `abortion_classification` | 14 |
| 580 | `analyze_response_times` | 4 |
| 462 | `detect_trials` | 1 |

---

### Phase 6a — split the four long functions, and unify the outcome rule

**Two jobs, deliberately in this order.**

1. **Split.** Take the four functions above and break each into small single-purpose functions
   — detect the cue poke, resolve the odor sequence, compute poke/valve windows, classify the
   outcome — rather than several at once. Pure refactor: **`regression.py` GREEN is a hard
   invariant on every commit.**
2. **Then unify.** rewarded/unrewarded/timeout is derived **three times independently** —
   `classify_trials` (the `completed_sequence_*` frames), `analyze_response_times` (the
   `response_time_category` column) and `save_results._derive_outcome` (re-derived from
   supply/poke counts). The FR/FA latency-bucket logic is duplicated too. Extract a **pure**
   `classify_completed_trial(record) -> outcome` taking a small per-trial record
   (await_reward_time, supply pulses, port-poke windows, response window, sequence_rewarded) —
   no `data`/`events` dicts inside — and point all three at it.

> **Measure the three against each other before merging them** (`DECISIONS.md` §13). "They
> share no code and can drift" is a hypothesis about drift, not a finding. If they agree on all
> 9 fixture sessions the merge is cheap and safe; if they disagree, **which one is right is a
> scientific decision, not a refactoring one** — bring the numbers and ask.

Doing the split first means any RED during it is unambiguously the refactor. The unification is
the only step that can legitimately move a value, and it arrives alone.

`plot_choice_history` in `visualization/` re-derives the same *display* category rule from
`response_time_category` + `hidden_rule_success` + `fa_label`. It computes no rate, so 4a left
it alone — it belongs to this consolidation.

**On "shorter functions":** length is the symptom, not the goal. Extract at seams that are
**pure and independently testable**; shortness follows. Splitting purely to reduce line count
produces functions taking twelve arguments, which is worse than what you started with.

**Done:** the four functions are gone as monoliths, 3 outcome sites → 1, `regression.py` GREEN
(or one deliberate, measured, agreed change from the unification), `verify_scripts.py` GREEN.

### Phase 6b — write `poke_source` and the 0 ms positions  *(separate chat)*

`DECISIONS.md` §10 in full: the two data-writing bugs that make the position record incomplete
and ambiguous. Writing happens in `classify_trials`, which is why it follows 6a rather than
preceding it — it lands in a trial loop that has already been made legible.

**This is an intended output change.** It alters `trial_data`, so fixtures are regenerated
deliberately in the same commit with the +/−/~ diff confirming only the intended columns moved.
It unblocks `only_true_pokes` on the sampling metrics and collapses `sequence_depth`'s
aborted/completed branch to a one-liner.

**Do not attempt 6b before 6a is committed and GREEN.**

### The gate for Phase 6 is `regression.py`, not `plot_regression.py`

The inverse of Phase 5. `trial_classification/` writes `trial_data`, so **`regression.py` (~15
min) sees everything** and is the gate that matters; run `verify_scripts.py` too, since this is
where the CLI wiring lives. `plot_regression.py` is a useful extra because a changed
`response_time_category` moves what the plotters draw, but it is not the primary guard.

**Phase 5's standing warning for this phase:** before consolidating any two things that "look
like duplicates", *measure whether they actually agree* — `DECISIONS.md` §13. Phase 5 was sent
to merge seven helpers described as twins and found five of them were different rules sharing a
name, one pair disagreeing on **63.8%** of trials. Phase 6's premise is that three outcome
derivations "share no code and can drift". That is a hypothesis about *drift*; the first
deliverable is the measurement of how often the three currently disagree, on real sessions.
If they agree everywhere, the consolidation is safe and cheap. If they do not, **which one is
right is a scientific decision, not a refactoring one** — bring the numbers and ask.

---

## Phase 7 — Schema, save formats & manifest provenance

### 7a. Manifest provenance  *(quick win, ~½ day)*

Add the **git commit** and the **package version** alongside the existing `created_at` date in
`manifest.json`. **Phase 2c already did the work:** call
`hypnose_helpers.provenance.provenance()` rather than writing a second implementation — it
handles the `-dirty` suffix and the fact that the import package and the distribution differ
here (`hypnose_behavior` ← `hypnose-behavior-analysis`), which makes the naive
underscore-to-hyphen guess return `None`.

Keep these **in the manifest only** — the regression already ignores it, so they never enter
the fingerprint and never cause spurious RED.

**Risk:** low. **Progress:** ~40%. **Done:** manifest carries commit + version + date;
regression unaffected.

### 7b. Schema & save formats

`trial_data` already saves parquet + CSV. Decisions:

- Standardise on **parquet for tables, JSON for metadata**. **No pickle** for saved outputs
  (version-fragile — the somnotate work is a live example of pickle/version coupling biting).
  Keep a CSV of `trial_data` only for human-readability; if dropped, update `qc/_common.py` to
  read parquet → canonical form.
- **Typed `@dataclass TrialRecord`** for the flat trial table: replace the free-form ~60-key
  dict (with its singular/plural aliases) with explicit typed fields, validation in
  `__post_init__`, and `.to_row()` for the DataFrame.
- **Flatten the JSON-blob columns** (`position_valve_times`, `position_poke_times`,
  `presentations`) into a tidy long-format side-table `position_data` — one row per
  `trial_id × position` with odor / valve_start / valve_end / poke_time_ms. `frames.build_position_data`
  already derives exactly this shape at load time; 7b is where it becomes a written artifact,
  and where the loader can stop expanding blobs. **Carry its provenance flags across —
  `DECISIONS.md` §2.**
- **Decide where the per-trial metric tables live.** Nine registered metrics return per-trial or
  per-poke tables rather than session values — the latencies, `inter_trial_interval`,
  `trial_poke_span` / `_total`, `hr_abort_poke_gap`, and `poke_durations` (739 rows for one
  session). They are deliberately absent from `metrics_*.json`, which is a summary; they are the
  same shape as the `position_data` side-table and should ride with it if they are to be saved
  at all. See `DECISIONS.md` §5, item 3.
- **Write the 0 ms positions and add `poke_source`.** Two data-writing bugs make the position
  record incomplete and ambiguous; both surface as per-position metrics that cannot be defined
  consistently. **The full specification, the measurements, and the one-line `sequence_depth`
  change it unblocks are in `DECISIONS.md` §10.** The writing happens in `classify_trials`, so
  it lands naturally with Phase 6's trial-loop cleanup; `position_data` is where `poke_source`
  becomes a column. Alters `trial_data` ⇒ deliberate fixture regeneration.

The dataclass and the side-table are complementary, not alternatives: the dataclass governs the
flat per-trial table, the side-table replaces the per-position blobs that don't belong in it.
Queryable, type-safe, smaller/faster parquet, kills the alias hacks.

**Intended schema change → regenerate fixtures deliberately.** Phase it: add the side-table
additively, keep blobs during transition, drop blobs last. Couples tightly with Phase 6.

**Risk:** med (touches downstream readers). **Done:** no pickle outputs; `position_data`
side-table exists; blobs removed; `poke_source` written; fixtures regenerated with only the
intended diff.

---

*OPTIONAL BONUS, NOT PART OF THE CORE CHANGES: Phases 8 and 9*

## Phase 8 — Profile, then vectorise

**Profile first — do not guess.** Run one session through
`analyze_session_multi_run_by_id_date` (+ `run_all_metrics`) under `cProfile` (`snakeviz` to
view), then `line_profiler` on the top function. Use **local data** so I/O variance doesn't
dominate.

Likely finding: **data loading** (harp/aeon `.bin` reads, timestamp interpolation, `concat`)
dominates rather than the classification loops — in which case optimise I/O batching, fewer
`concat`s and vectorised event-window math, not sequential event logic for its own sake. The
Phase 5 measurements already point that way: a cold `derivatives.find_session` costs 14.6 s
against 29 ms to compute every metric for the session it found.

**Risk:** med — vectorisation can produce *almost* (not byte-) identical floats, so expect
some intended RED; the per-column diff localises it and you decide tolerance per case. Note
`DECISIONS.md` §1: summation order is part of several metrics' values.

---

## Phase 9 — Validation with clear errors

Currently **0 asserts** in classification/metrics. Add checks that "function X succeeded before
Y starts", with messages that aid troubleshooting.

Prefer explicit **`raise ValueError(msg)`** for production preconditions — bare `assert` is
stripped under `python -O`. Reserve `assert` for internal invariants. Optionally later: swap
`print`/`vprint` for the `logging` module with levels.

**Risk:** low (additive). **Effort:** low–med, spread throughout.

---

## Phase 10 — Modularise `visualization_utils.py`  *(proposed, after the restructure)*

**4a did not de-bloat `visualization/`.** Measured, it went 16,627 → 15,124 lines, and almost
all of that is one wholesale move (`compute_speed_analysis`). `visualization_utils.py` itself
lost 1.9%, and `metric_analysis` gained what it lost — the metrics moved, they did not vanish.
4a's value is **one definition per metric**, not line count; the plan overpromised a de-bloat
here.

So `visualization_utils.py` (6,690 lines) needs the same split 4b gave `metrics_utils.py`:
plumbing apart from plotters, then per-plot-family modules. Pure function moves, so low risk —
and `qc/plot_regression.py` gates it exactly, since it resolves each case across `MODULES` and
a move is invisible to the diff.

**Two traps:** add any new plotter module to the gate's `MODULES` list or its cases silently
read as "not found"; and moving a plotter between modules changes `file`/`chain` in saved-figure
provenance (`DECISIONS.md` §9).

**Phase 5 left two things here, deliberately.**

1. **The nested loader plumbing.** `valve_poke_plots.plot_valve_and_poke_events` is one 620-line
   function with six loaders nested inside it (`parse_exp_ts_to_uk`, `_safe_concat`,
   `_apply_offset_and_localize`, `_slice`, `_load_register_files`, `_try_load` — `:74`-`:284`),
   and `visualization_utils._session_hr_odors` (`:5189`) is nested in one plotter. They are pure
   loader plumbing and belong in `io/loaders.py`. Phase 5 moved only the loaders that were
   imported *across* modules, because those were the ones breaking layering; these break none.
   **`valve_poke_plots` has no `plot_regression` case**, so anyone extracting them must add one
   first or the move is unverifiable.
2. **The invariant to preserve: zero plotter-to-plotter imports.** Phase 5 moved 15 shared names
   into `prep.py` and `panels.py` so no plotter module imports from a sibling
   (`DECISIONS.md` §13). A split that reintroduces one puts the tangle straight back.
   `grep -n "^from hypnose_behavior.visualization" src/hypnose_behavior/visualization/**/*.py |
   grep -vE "primitives|prep|panels"` should stay empty.

**Not scheduled.**

---

## Parallel track — time-base audit for ephys/movement alignment

Keep this as a note — it will be checked later, as part of it lives within `sleap-hypnose`
(where tracking is done).

Ensure every saved event carries a **canonical, documented timestamp** suitable for aligning
with electrophysiology and movement data. The pipeline already does harp timestamp
interpolation plus a real-time (UK tz) offset — now `io/loaders.compute_real_time_offset`, one
definition since 4a. Audit that (a) the time base is consistent and documented, (b) it is
ideally tied to a hardware sync signal, (c) saved outputs expose it.

Unblocks multi-modal alignment. Independent of the rest — can run in parallel.
**Progress:** ~40%.

---

## Cross-cutting

**Unit tests** (`tests/`, or adjacent to `src/hypnose_behavior/qc/`): fast, mount-free tests for
the outcome classifier, FR/FA buckets, `_get_single_reward_info`, `_parse_date_input`,
`validate_subject`. **Build the outcome-classifier tests before Phase 6.**
`hypnose-helpers/tests/test_layout.py` (20 tests, mount-free, runs without pytest) is the
pattern.

**Lightweight CI** (optional): `check_imports` + unit tests in GitHub Actions on PRs. The
regression stays local — CI can't reach the data.

---

## Afterthought — cross-repo API  *(TODO, done later)*

Previously planned as a facade (`hypnose.behavior.accuracy(subjid, date)`). **Demoted**: the
repos do largely independent work and don't obviously need to call each other's analysis.

What they *do* need is **well-defined tidy DataFrame loaders** — the
`hypnose_somnotate.io.load_scores()` pattern: forgiving selectors in, one tidy DataFrame out,
identifier columns prepended, fast enough to call across a cohort, downstream computation left
to the caller. If this repo grows an equivalent `load_trials(...)` / `load_metrics(...)` with
the same shape, that probably covers the real cross-repo need without a facade. 4b's registry
is most of `load_metrics(...)` already.

Revisit only if a concrete consumer appears.

---

## Suggested order

```
Phase 0   decisions + collapse loaders/readers            DONE
Phase 1   rename                                          DONE
Phase 2   extract hypnose-helpers, session API, provenance DONE
Phase 3   re-baseline QC                                  DONE (superseded by 0)
Phase 4   metrics single source of truth (4a then 4b)     DONE
Phase 5   visualization primitives + thin plotters        DONE
Phase 6a  split the 4 long functions + unify outcome      <- next; regression GREEN
Phase 6b  poke_source + 0 ms positions                    intended output change
Phase 7   manifest provenance, schema & formats           couples with Phase 6
Phase 8   profile, then vectorise                         evidence-led, optional
Phase 9   validation                                      woven throughout, optional
Phase 10  modularise visualization_utils.py               proposed, not scheduled
∥         time-base audit                                 parallelisable
```

After each step, run the gate that can **see** the change (see §1 "gate on reachability").
GREEN ⇒ commit. Intended change ⇒ regenerate fixtures in the same commit and confirm via the
+/−/~ diff. Tag the finished round **v2.0.0**.
