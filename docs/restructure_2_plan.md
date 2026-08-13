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
  - **`-u` is not enough if you pipe through `grep`** — grep block-buffers too, and
    `regression.py` emits only ~20 short lines, well under grep's 4 KB buffer, so the whole
    run looks silent until it exits. Use `grep --line-buffered`. (Cost 2026-08-10: a green
    run that was indistinguishable from a hang for 12 minutes.)
  - **Runtime varies a lot** and is dominated by mount latency, not by the code. Measured on
    the same 9 sessions: ~2-3 min on a warm local-feeling mount, **~12 min** on 2026-08-10
    after a long session of repeated reads. Treat a slow run as normal until `ps` says
    otherwise: `ps -eo pid,etime,stat,command | grep regression.py` shows elapsed time, and
    state `U` just means it is waiting on mount I/O.

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
| 6a split the 4 long functions + unify the outcome rule | **done** 2026-08-10 | `97a01ad`..`5f25e4a` | **The four monoliths are gone**: 3,005 → 977 lines (`classify_trials` 1226→419, `abortion_classification` 736→162, `analyze_response_times` 580→263, `detect_trials` 462→133); the file 3,703 → 3,023. **`regression.py` GREEN on every commit, including the unification** — nothing in this phase moved a value. Two new leaves: `windows.py` (poke/valve primitives) and `outcome.py` (`classify_completed_trial` + `latency_label`), both importing nothing from the package, so `io/save_results.py` reaches the shared rule without a cycle (§3). **3 outcome sites → 1, 3 latency-bucket copies → 1.** The three valve-edge builders and the two poke summaries were measured and **kept apart** under names saying how they differ. 106 lines of dead nested defs dropped. **Measured before merging (§13): 1,731 trials, the rule never conflicts — `DECISIONS.md` §14.** New guard `qc/verbose_diff.py` (stdout, which `regression.py` never sees) and `qc/outcome_agreement.py` |

| 6b `poke_source` + the unpoked positions | **done** 2026-08-10 | `HEAD` | **Intended output change, fixtures regenerated in the same commit.** Every valve position is now written with `poke_source` ∈ `poke` (4620) / `grace` (91) / `outside_grace` (83). **The brief's rule was measured and not followed as written** — all 83 unpoked positions have the port OUT for the whole window with zero DIPort0 transitions, and 75 of them trail an *aborted* trial, where `abortion_classification` independently puts `last_odor_position` at the last **poked** position on 74/74. So a completed trial counts every presented position (the rig advanced through them all) while an aborted one stops the *sequence* at the last real poke (`_trim_unsampled_tail`) — but still **records** the position, so the blobs stay complete and only the credit shrinks. The trim is gated on AwaitReward, verified: `is_aborted` moved on 0 trials and every shrunken trial has null supply/AwaitReward/reward-poke. `_odourdisc_await_window` extracted so that gate has one definition feeding both the trim and the scoring branch. **Acceptance test 1 passed: `outcome_agreement.py` 1 conflict → 0**, trial 277 `false_response` → `rewarded` (§14). **Acceptance test 2 failed as briefed and was not forced: the §15 fallback still fires 20/20** — it lives in `analyze_response_times`, which builds its own position map from valve events; `poke_source` makes the proper anchor *available* but nothing selects it. Left for a follow-up rather than smuggled in. Sampling metrics filter to `poke_source == 'poke'` via `_real_pokes`, which no-ops on pre-6b sessions (§2). Fixture change audited cell-by-cell against HEAD: 18 of 1731 trials move a non-blob column, 83 gain a position, **0 pre-existing position values changed, 0 removed, 0 unexplained** |
| 6c split `classification_utils.py` | **done** 2026-08-11 | `HEAD` | **`classification_utils.py` is gone — 3,138 lines, 57 definitions, no facade.** Seven flat modules (`detect_trials`, `classify_trials`, `response_times`, `aborted_trials`, `hidden_rule`, `params`, `index`) plus the two existing leaves. **Pure move, and proved so rather than assumed: `qc/ast_move_check.py` reports 57/57 byte-identical**, 0 changed / missing / duplicated — and the checker was itself validated against injected drift, a whitespace-only edit, a deletion and a duplication first. Layering is a three-layer DAG with **zero worker-to-worker imports** (§17), which is what keeps the two position rules from being merged by proximity; the do-not-merge note now sits in `classify_trials.py`'s docstring and **the settling count was deliberately not taken here**. Two functions the plan's table did not place: the per-run orchestrator `classify_and_analyze_with_response_times` → `run.py`, and the trio `_next_after` / `_recording_end` / `_odourdisc_reward_window_end` → `windows.py` (shared by two workers ⇒ the shared thing becomes a leaf). Three dead `*_SCHEMA_PATH` constants dropped — duplicates of `io/loaders.py`'s live ones. **All five gates GREEN, none regenerated:** `regression.py`, `verify_scripts.py`, `check_imports.py` (62 modules), `verbose_diff.py` (**16,944 lines of stdout identical across 9 sessions**), `plot_regression.py`. **Trap: a dropped mount makes `verbose_diff.py` print GREEN off a 1-line error string — read the line count (§17).** Notebook star-imports left to the user to fix in one pass |
| ↳ 6b's second acceptance test | **closed, no work owed** | | The §15 fallback still fires 20/20, and that is **not a defect**: those trials were fixed in Phase 11 and are correct. The brief expected `poke_source` to retire the fallback; measured, `analyze_response_times` never reads it, and repointing the target would be cosmetic while risking `search_start` (hence outcomes) and requiring two deliberately different position rules to be reconciled. **Decided: leave it.** `DECISIONS.md` §15 |
| 6 close-out — one position rule, `sequence_depth`, latency rename | **done** 2026-08-12 | `HEAD` | **Phase 6's three deferred items, measured then closed in one commit with one fixture regeneration.** (1) **Three** position rules → one (`windows.positions_by_odor`); the third, in `aborted_trials._abort_positioned_events`, was uncatalogued by §13/§15 and matched the *old* classify rule, so merging only the two named ones left the tree self-inconsistent on the very trial at issue. Measured before merging: divergence needs a non-consecutive odor repeat — **1 of 1,731** fixture trials, **0 of 46,112** across subs 056-066 (263 sessions, read-only). (2) `sequence_depth` stated as one expression — completed `max(presentations)`, aborted `max(poke_source=='poke')` + `last_odor_position` fallback; the two single-meaning alternatives were measured at **84** and **32** moved trials and rejected. (3) latency rename, 46 replacements, **no value moved** (8 sessions: added/removed columns, zero changed). **Exactly one trial moved overall** — `sub-040 20251124` t44, an *experiment* fault (no `InitiationSequence` for 97.9 s ⇒ three F→A runs recorded as one trial); `detect_trials` deliberately unchanged, the rig itself emitted one `ChooseRandomSequence`. Two `RuntimeWarning`s added, silent on sound data. **Trap: an md5 RED says nothing about scale** — the abort-rule inconsistency showed up only as `last_odor_position` *not* moving in a cell-level diff. **Trap: a two-tree diff cannot see both sides being equally broken** — 2 `plot_regression` cases are now *vacuously* green until the server derivatives are re-analysed at the end of 7b. `DECISIONS.md` §18 |
| 7a manifest provenance | **done** 2026-08-12 | `HEAD` | `manifest.json` carries `commit` + `version` from `hypnose_helpers.provenance.provenance()`, alongside the existing `created_at`. **Manifest only** — the regression never reads it, so a per-run commit stamp cannot cause a spurious RED. **Both `provenance()` arguments passed explicitly**: `anchor=__file__` (a helpers-resolved anchor would stamp every repo with the *helpers* commit — the 2a/2b silent-resolution failure) and `call={"module": __name__}` (frame capture resolves to `__main__` from a notebook ⇒ `version=None`, and passing it sidesteps the §9 wrapper hazard). Both keys always written, `null` included, so absent-key ≠ unresolved-commit (§2). `lru_cache`d — it shells out to `git` and describes the code as *imported*. Measured `{'commit': 'b3f2497-dirty', 'version': '1.0.0'}`. **All five gates GREEN, none regenerated**; `regression.py` output byte-identical to the clean-tree baseline, `verbose_diff` 16,944 lines across 9 sessions, `plot_regression` 35/35. `DECISIONS.md` §19 |
| 7b.0 protocol-mode guard | **done** 2026-08-12 | `HEAD` | Enabling piece for 7b.1's mode-dependent record. `io/protocol_schema.py` (a leaf, standard library only) owns the three mode names and `resolve_mode`, which **raises `ConflictingProtocolError`** when a run is flagged both odour-discrimination and single-reward. The flags come from independent sources — the stage's protocol name and the schema's `isSingleRewardProtocol` — so nothing in the code makes them exclusive; the *experiment* does (odour-disc presents 1 position, single-reward needs ≥2). Raises rather than warns because `batch_analyze_sessions` catches per session: the broken session names itself, writes no derivative, and the batch completes, whereas a warning would write a `trial_data` silently missing the four determinacy columns. **Proved rather than assumed (§17), and the probe caught two defects the gates never would**: the first patch point was wrong (`run.py` resolves `_get_single_reward_info` itself and passes it in, so patching `classify_trials`' copy left the flag `False` — a *false pass*), and once it did fire, `run.py`'s per-run handler swallowed it, since that handler is `vprint`-gated and the batch default is `verbose=False` — the user saw only "No runs analyzed". Fixed by re-raising ahead of the generic handler: the condition is session-level, so skipping to the next run can only re-raise. Named `protocol_schema.py`, not `schema.py`, because "schema" already means the *task* schema here. **Gates: `check_imports` PASS, `regression` 9/9 GREEN, `verbose_diff` 16,944 lines identical.** `verify_scripts`/`plot_regression` deliberately skipped — no CLI wiring touched, and `plot_regression` reads saved derivatives and never re-runs classification |
| 7b.1 typed `TrialRecord` | **done** 2026-08-12 | `HEAD` | **Intended output change: 8 columns, fixtures regenerated in the same commit.** `classify_trials` builds a `@dataclass(slots=True)` record per trial, not a dict. **Three classes because the modes are exclusive** (§20): standard 43 fields / single-reward 55 / odour-disc 48, i.e. 61 / 73 / 66 saved columns. Measured before writing code: one *uniform* record adds **26** columns (13-19 per session), per-mode adds **8** (one per session for eight of nine). `SingleRewardTrialRecord` extends the standard one because single-reward uses *both* scorers; odour-disc does not, so `slots` makes `poke_window_end` unassignable there. The 8 are all-null and irreducible — data-determined, not mode-determined. `save_results`' two hand-maintained column lists collapse into one declaration-driven conform; mode threaded `classify_trials` → `merge` → `manifest["protocol_mode"]` for 7b.2 to read back. **Four traps, all caught before the gate:** (1) declaring `run_id` would make `merge._with_run_id` mint a phantom all-null `run_id_original` on every merged session; (2) `ASSEMBLED_COLUMNS` must stay out of the conform — `run_id`'s fallback is guarded on *absence*, `global_trial_id` would move off the front; (3) `protocol_mode` read before assignment in the manifest — a `NameError` on every save; (4) **the `datetime64`→`object` concat**: an all-`None` column carries no type, so one empty run turns the merged column `object` and `to_csv` writes `…806000` for `…806` — **154 + 135 cells on `sub-040 20251124`, values unchanged**, invisible on single-run sessions. Fixed by `DATETIME_FIELDS` (15, measured from the reference tree's parquet dtypes) cast losslessly, never `errors="coerce"`. **Gates: `regression` only `+ added` lines, zero `~ changed`, zero `- removed`, all 9 metrics md5s identical — cell-complete, since a per-column md5 moves if one cell does; `verbose_diff` 16,944 lines identical; `verify_scripts` GREEN; `check_imports` PASS.** `DECISIONS.md` §21 |
| 7b.2 loader schema check | **done** 2026-08-12 | `HEAD` | `load_results_dir` compares a saved `trial_data` against the current declaration and warns on what is missing — **the §19 stamp catches changed values, this catches a changed schema**. A tagged file is checked against `trial_data_columns(mode)`; an **untagged one is checked too, not skipped**, against `mode_independent_columns()` (60 columns, verified a strict subset of all three modes, so it cannot false-alarm). **That is the whole point**: the renamed latency columns are *merged*, hence mode-independent, so the server's `sub-040 20251124` now reports `fa_window_latency_ms`, `fa_response_time_ms`, `completed_window_latency_ms` — where the plan's original `TrialRecord.__dataclass_fields__` form reports only `fallback_reason`, a column nothing reads. Raised through `warnings` (stderr), so it cannot disturb the stdout `verbose_diff`/`plot_regression` compare. **Proved on four paths (§17):** real pre-7b files warn and name the columns; a session written by current code warns not at all; dropping two columns from a tagged file names exactly those two; an unknown mode says the schema was not checked. **Gates: `regression` GREEN 9/9 no regeneration, `plot_regression` GREEN 35/35, `check_imports` PASS.** **New trap, §22: `plot_regression`'s banner is not its result** — a dropped mount printed `GREEN: 35 plotters` off a run that attempted **31**, the movement plotters vanishing first; count the cases, as `verbose_diff` needs its line count read. And `REGRESSION RED: N mismatch(es)` with no `[RED]` lines is a mount, not a regression. `DECISIONS.md` §22 |
| 7b.3 `save_csv`, default off | **done** 2026-08-12 | `HEAD` | `save_session_analysis_results(..., save_csv=False)`, threaded through `analyze_session_multi_run_by_id_date` → `batch_analyze_sessions` → both CLI scripts (`--save-csv`). **The flag could not gate CSV alone**: `loaders._load_table_with_trial_data` read the three `non_initiated_*` tables from **CSV only**, so the default would have made them return an empty frame *with no error*. Parquet is now written for **every** table and the reader prefers it, so `save_csv` is purely additive. Verified both ways on `sub-053`: with `save_csv=False` no CSV exists at all and the three tables still load 195/1/1 rows, identical to `save_csv=True` — a CSV-only reader could not have. `.schema.json` follows the CSV, not the parquet. **All four QC entry points pass `save_csv=True` explicitly** (`_common`, `verify_scripts` ×2 scripts, `outcome_agreement`) so a later default change cannot silently break the gate. **Gates: `regression` GREEN 9/9 no regeneration, `verify_scripts` GREEN (the only gate exercising the new CLI flag), `check_imports` PASS.** `DECISIONS.md` §23 |
| 7b.4 pre — promote `frames.py` | **done** 2026-08-13 | `HEAD` | **Enabling piece for 7b.4a, and a pure move.** `metric_analysis/frames.py` → `hypnose_behavior/frames.py`, a leaf at the package root. 7b.4a needs `build_position_data` in `io/save_results.py`, which `trial_classification/run.py` imports — so importing it from `metric_analysis` would have given trial classification a `metric_analysis` dependency, exactly what `load_results.py`'s docstring said had been deliberately avoided. **The edge was removed rather than doubled: `io/` no longer imports `metric_analysis` at all.** This is the fix §3 asked for on 2026-08-06 (*"promoting `frames.py` to a schema layer below both is the honest fix — revisit only if it grows"*). **The whole file moved**, because `build_position_data` shares four helpers with `sequence_depth`/`reached_counts`/`sampled_positions`. **Not `schema/`** — §20 already ruled that word out, since "schema" means the *task* schema here. Eight in-repo imports rewritten and the old module **deleted, no shim**, so nothing can import the stale path by accident; two notebooks are knowingly left to raise `ImportError`. **Gates: `ast_move_check` 15/15 byte-identical PASS, `git diff -M --stat` 0 insertions 0 deletions, `check_imports` PASS (63 modules), `regression` GREEN 9/9 no regeneration.** `verify_scripts` skipped (no CLI wiring touched), `plot_regression`/`verbose_diff` skipped (a proven-identical move writes identical output). `DECISIONS.md` §3 |
| 7b.4a `position_data.parquet` | **done** 2026-08-13 | `HEAD` | **Additive: the blobs stay and the loader still derives, so no column and no metric value moves.** `save_results` writes one row per `trial x position`, with `poke_source` and §2's three provenance flags as real typed columns. **Built from the in-memory frame, deliberately** — the blobs are JSON-encoded into `trial_data.parquet`, so a *load*-time derivation gets its five timestamp fields back as ISO strings while this one carries real `datetime64`. That is the whole "typed columns" win. **Measured across all 9 sessions (4,791 rows): the two frames agree on every cell of every column once parsed, differing only in the dtype of `poke_odor_start`/`poke_odor_end`/`poke_first_in`/`valve_start`/`valve_end`** — same instants, no precision change — and all 14 evaluable `position_data` metrics return identical values, since each normalises through `_tz_naive` (`pd.to_datetime`). **Also measured: the projection is lossless for every field any reader consumes** — 25 of 26 `(blob, key)` pairs carried with equal values on all 4,791 occurrences, `differs` **zero everywhere** (the merge precedence never discards a value), the one exception being `position_valve_times.prior_presentations`, read only in memory by `classify_trials` and already persisted as `non_initiated_odor1_attempts`. **Trap recorded: no gate watches this file** — `_common` fingerprints `trial_data`'s CSV and the metrics dict, and the four metrics touching its timestamp columns are all unreported, so the GREEN is *additivity*, not coverage. **Gates: `regression` GREEN 9/9 no regeneration, `check_imports` PASS.** `verify_scripts` skipped (no CLI wiring), `plot_regression`/`verbose_diff` skipped (nothing changes at load time yet). `DECISIONS.md` §24 |
| 7b.5 per-grain metric tables | **done** 2026-08-13 | `HEAD` | `metrics_by_trial.parquet` (`global_trial_id`, 10 columns) and `metrics_by_poke.parquet` (`global_trial_id` + position, both outcome classes), written by `run_all_metrics` beside the JSON. **The plan's grain for `metrics_by_poke` did not exist**: `poke_durations` returned `["position","odor_name","poke_time_ms"]` with **no trial identifier**, so its rows were anonymous and could not be joined back. It now carries `global_trial_id` — a deliberate metric-output change, chosen over shipping an unjoinable table or rebuilding the frame in the writer (a second derivation of one quantity, §14). Safe because all four consumers select by name; the one concat site (`visualization_utils.py:1651`) is covered by `plot_sampling_times_analysis`. **Two parameterised metrics are still saveable**: `aborted` partitions *outcome classes* not figure options (both saved, 629+45 on `sub-057`), and `fa_types=None` means *unfiltered*. The `window`/`fa_types` variants remain unsaveable — §5's "save everything and only load" stays unreachable. **§21's trap in a new place**: `hr_abort_poke_gap` returns shape (0,4) **all-`object`** on a no-hidden-rule session, so every value column is forced numeric (`errors="raise"`, never `"coerce"`) — verified `sub-057` and `sub-040 20251124` yield the same 10 columns and dtypes. Indexed on every trial so the file left-joins 1:1. `save_tables` passed **explicitly** by `_common`/`verify_scripts` (§23) and by the two pooled calls, whose merged `global_trial_id` space would be ambiguous. No CLI flag added. **Gates: `regression` GREEN 9/9 no regeneration, `plot_regression` GREEN with 35 cases COUNTED (§22) and all four movement plotters present, `verify_scripts` GREEN (the `batch_process` path writes the tables), `check_imports` PASS.** `DECISIONS.md` §25 |
| 7b.6 re-analyse + extend the gate | **done** 2026-08-13 | `HEAD` | Nine fixture sessions re-analysed on the server by the phase owner, then **verified against the fixtures**: all 9 carry every required file plus both 7b.5 tables, every manifest stamps `commit=3094c40` / `version=1.0.0`, `protocol_mode` resolves correctly across all three modes (6 standard / 2 odour-disc / 1 single-reward), and `trial_data` + `metrics` **MATCH on all 9**. Then `regression.py` grew from **2 fingerprints to 6**: `position_data`, `metrics_by_trial`, `metrics_by_poke` (all from the **written file**, so the save path is covered; a missing table hashes `"ABSENT"` rather than being skipped) and `unreported_metrics` (16 of the 18 registered-but-unreported metrics). **This closed a real gap** — 7b.4a and 7b.5 both went GREEN on *additivity*, not coverage. **16 not 18**: `rolling_reward_fraction`/`rolling_hr_reward_fraction` take `window` positionally with no default, so fingerprinting them would invent a figure choice (§5). **Populated before blessed** (14 on 9/9, `fa_latency_from_pokeout` 7/9, `hr_abort_poke_gap` 3/9, none empty everywhere), and prior validation traced through the 4a history: 12 are plotter-backed and `plot_regression`-gated at extraction, 1 (`presentation_counts_by_odor`) is inside the metrics md5 via three REPORTED callers, 3 have a default variant no figure draws (one `if` from the validated path). **Regeneration purely additive: +8 keys per fixture, 0 removed, 0 changed**, verified before *and* after. **Gates: `regression` GREEN 54/54 (9 × 6 keys), `check_imports` PASS.** **Two operational traps recorded (§26): never run two mount-heavy jobs on the SMB share concurrently — the overlap exhausts the client handle pool (`Errno 24`, process wedged in `U`, unkillable) — and a shallow directory listing cannot clear a wedged mount.** `DECISIONS.md` §26 |
| 7b.4b prep — the uncarried-field guard | **done** 2026-08-13 | `HEAD` | **Enabling piece for 7b.4b, and the answer to "how do I know my new field got saved".** `build_position_data` copies a **whitelist** of blob fields; anything else is dropped silently, which is harmless while the blobs remain in `trial_data` and is **data loss with no signal** once 7b.4b removes them. Now enforced twice: `build_position_data(..., strict=True)` — passed **only** by `save_results` — raises `UncarriedPositionFieldError` naming the field and both remedies, while every read path warns instead, because the same function runs over sessions saved before these lists existed (§2) and refusing to read them would be the worse failure. Plus `qc/position_data_lossless.py`, which asserts the precondition for the drop — **not** "identical" (§24 — impossible), but *every blob field recoverable with an equal value except a named allow-list*. **Measured 9/9 GREEN: 25 of 26 `(blob,key)` pairs `differs:0 absent:0` on 4,791 occurrences each; the 26th allow-listed.** `KNOWN_UNCARRIED_FIELDS` is declared once in `frames.py` and **imported** by the gate, so guard and gate cannot disagree. **Proved before trusted (§17)**: emptying the allow-list REDs, corrupting **one cell of 9,770** REDs with the trial/position/values named, dropping a column REDs. **And the guard walked into its own trap**: its first whitelist included `_PRES_FIELDS`, which the builder never reads — so a field added there would have *passed the guard and still been dropped*. Fixed with `_PRES_ONLY_FIELDS`, read by both the copy loop and the check. **Gates: `regression` GREEN 54/54 (twice — with `strict=True` live, and after the fix), `position_data_lossless` GREEN 9/9, `check_imports` PASS.** `DECISIONS.md` §27 |
| 7b.4b step 1 — measure the gate's reach, then extend it | **done** 2026-08-13 | `HEAD` | **Measured before porting anything, and the measurement changed the plan.** Patching `Series.get` to record the caller's `file:line` on the three blob keys, then running `plot_regression`'s whole case list in one process: **it executed 4 of the 11 live per-trial blob reads; 7 were at zero** — so "step 3 GREEN" would have meant green on 4 of 11. Three of the seven are `pred_seq_utils` plotters simply absent from `CASES` (`last_odor_poke_time`, `first_odor_poke_duration`, `poke_time_all_pos`); **added, 35 → 38 cases**, each verified to *draw real data first* (482/40/116 points) and to register reads at exactly the site it was added for — a case drawing nothing would go green in both trees, section 26's trap in a new place. **`compute_speed_analysis` cannot be a case at all: it writes `speed_analysis.parquet` into `results_dir`, i.e. into the read-only share** (as does `plot_traces_with_speed_threshold` on a session lacking the file); the existing movement cases are safe only because their two sessions already carry it. Its control will be a one-off equality probe against a local copy. `debug.py`'s two helpers left unguarded by decision. **Also found, and not in the phase brief: nine sites pass an already-loaded trial frame to `build_position_data`**, which returns an *empty frame with no error* once the blobs go — they read through `io/loaders._load_trial_views`, not `load_results_dir`, so step 4 must switch **both** loaders, and `sing_rew._session_reward_rts` passes a *filtered* subset so a file read is not a substitution there. **Confirmed post-7b.6: the two formerly vacuous cases are genuinely green** — `plot_response_times_completed_vs_fa` draws 12 points (exactly the 12 it had lost) and `FR_latency` 8. **Gate: `plot_regression` GREEN with 38 cases COUNTED, 0 RED / 0 MISSING / 0 both-raise, all five movement plotters present.** `DECISIONS.md` section 28 |
| 7b.4b step 2 — `sequence_depth` onto `position_data` | **done** 2026-08-13 | `HEAD` | **The intricate half of 7b.4b, and no value moved.** `frames.sequence_depth(trial)` → `sequence_depths(trials, position_data)` (an `Int64` Series over the frame) with `reached_counts(trials, position_data)` on top; the per-trial signature **retired**, since its only consumer was `reached_counts`' own loop and keeping it would mean 1,731 frame slices a session plus a caller contract to pass the right one. Three metrics become `frame="trials+position_data"` — `abortion_rate_positionX` and `fa_abortion_stats` (both **REPORTED**, inside the metrics md5) and `fa_rate_by_position` — with their two session wrappers and two `visualization_utils` call sites threaded. **The two precedences are opposite** (presented: `in_poke_times` then `in_presentations`; sampled: the reverse, which is also the reverse of `build_position_data`'s merge), so the reconstruction rests on section 24's `differs: 0` inventory — a fact about the data, so it was **measured, not argued**: **1,731/1,731 trials equal** against HEAD's `frames.py` extracted by `git show`, on the written parquet *and* the load-time derivation, `reached_counts` identical on all 9. **Sensitivity proved: forcing the two rules section 18 rejected moves 84 and 32 trials — exactly section 18's recorded counts.** Key is **`global_trial_id` alone** (`trial_data` has no `subjid`/`date`; `trial_id` restarts per run), so a per-row `.map` not a `groupby` — a groupby would collapse pooled sessions' colliding ids and change a *denominator*; ambiguity now **warns** instead. **Two `presentations` guards removed that never read the blob** — they would have made a reported metric and a figure silently empty at step 5. `sampled_positions` and `_max_poke_time_position` **deleted**, zero callers each. **Gates: `regression` GREEN 54/54 no regeneration** (as predicted: `trial_classification` never imports `frames`, so `trial_data` and all three side-table md5s were unreachable by this change), **`position_data_lossless` GREEN 9/9** (25 pairs `differs:0 absent:0`, 1 allow-listed), **`plot_regression` GREEN 38/38 COUNTED**, `check_imports` PASS. `DECISIONS.md` section 28 |
| 7b schema & formats | in progress | | intended output change |
| 8 profile, then vectorise | not started | | |
| 9 validation | not started | | |
| 10 modularise `visualization_utils.py` | **proposed**, after the restructure | | not scheduled |
| 11 latency semantics *(unplanned, arose from 6a)* | **done** 2026-08-10 | `934c868`..`HEAD` | Three measured output changes, fixtures regenerated. §15: the response-time anchor falls back to the last poke *before* the odor — 20 trials that had no response time now have one, rewarded/unrewarded reach 100% coverage. §16: every reward latency now exists as **(a) window-relative** (sets the label, unchanged) and **(b) movement** (`response_time_ms` re-anchored; three new columns). `fa_latency_from_pokeout` repointed at (b). **Labels, outcomes and `decision_accuracy` did not move.** Naming debt recorded above |
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

### Naming debt — the (a)/(b) latency pair  **SETTLED 2026-08-12**

Every reward-port latency exists twice. Both **end at the reward-port poke**; they differ only
in where they start — **(a)** at the window start (what the FA/FR labels bucket), **(b)** at the
animal's last cue-port exit before that poke (how fast it moved). `DECISIONS.md` §16.

The base names did not say which was which. Settled as a **pure rename — no value moved**:

| was | is | which |
|---|---|---|
| `fa_latency_ms` | `fa_window_latency_ms` | (a) |
| `fr_latency_ms` | `fr_window_latency_ms` | (a) |
| `fa_movement_latency_ms` | `fa_response_time_ms` | (b) |
| `fr_movement_latency_ms` | `fr_response_time_ms` | (b) |
| `completed_window_latency_ms` | *unchanged* | (a) |
| `response_time_ms` | *unchanged* | (b) |

Every (a) ends `_window_latency_ms`; every (b) is a `response_time`. It worked as a rename
**because `response_time_ms` was already (b) and `completed_window_latency_ms` already (a)** —
the completed pair needed nothing. `trial_data` values are untouched; the fixture md5s moved
only because column *names* are part of the canonical CSV.

`response_time_ms` was **not** repointed to (a) for symmetry: measured, that costs ~1063 changed
values and about a second on `avg_response_time`, where the rename costs none. The residual
asymmetry — completed's (b) has no `completed_` prefix — is the price, taken knowingly.

**Registry keys did not move.** `fa_latency_by_type` and `fa_latency_from_pokeout` keep their
names and read the renamed columns; a registry key is a key in `metrics_*.json`, so renaming one
would be an output change beyond a column rename.

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

> **Reassigned to Phase 10 (2026-08-11), and not done in Phase 6.** Measured after 6c: it reads
> the **stored** `response_time_category` rather than re-deriving the outcome from supply/poke
> counts, so it cannot disagree with the rule that 6a unified — what it duplicates is the
> *display* mapping (category + hr flag + `fa_label` → `trial_type`, linestyle, colour).
> `visualization/` imports nothing from `outcome.py` and does not need to. That makes it a
> plotter-tidying job, which is Phase 10's, not an outcome-correctness job.

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

### Phase 6c — split `classification_utils.py`  *(separate chat, after 6b)*

6a removed the re-export shims and took the four monoliths apart, so the seams are now visible
in the file itself. What remains is ~3,000 lines that belong in separate modules:

| module | contents |
|---|---|
| `detect_trials.py` | `detect_trials` + its attempt/fallback helpers |
| `classify_trials.py` | `classify_trials` + position assignment, outcome scoring, its summary printer |
| `response_times.py` | `analyze_response_times` + its helpers and printer |
| `aborted_trials.py` | `abortion_classification`, `classify_noninitiated_FA` + their printer |
| `hidden_rule.py` | the hidden-rule resolution shared by `classify_trials` and `response_times` |
| `params.py` | `get_experiment_parameters`, `_sampling_parameters_ms`, `_get_single_reward_info` |
| `index.py` | `build_classification_index` |
| existing leaves | `windows.py`, `outcome.py` |

*(Module names settled at implementation time, 2026-08-11: `detect_trials` / `classify_trials` /
`aborted_trials` / `params` rather than `detect` / `classify` / `abortion` / `schema`. A module
named for the function it carries is findable from a traceback; `schema.py` in particular would
have read as the device schemas, which are a different thing in this repo.)*

**Two functions the table above did not place, settled during the split.**

1. **`classify_and_analyze_with_response_times` → `run.py`.** The per-run orchestrator, and
   `run.py`'s only consumer. It calls into `classify_trials`, `response_times`,
   `aborted_trials`, `index` and `params`, so it belongs *above* all of them rather than inside
   any one, and it is the per-run counterpart of `run.py`'s per-session
   `analyze_session_multi_run_by_id_date`.
2. **`_next_after`, `_recording_end`, `_odourdisc_reward_window_end` → `windows.py`**, so that
   row reads "existing leaves", not "unchanged". All three are shared by `classify_trials` and
   `response_times`; leaving them in either would have made the other import from a peer, which
   is the §13 tangle Phase 5 spent its effort removing. They are pure functions of timestamps
   and pandas objects with no package imports — `windows.py`'s stated contract exactly.

**Flat modules, not a subpackage.** `trial_classification/` has ~9 modules today and would reach
~15 — still one screen of `ls`, and it keeps the §3 leaf discipline checkable at a glance. A
subdirectory earns its place only when an area grows its own cluster with a registry, the way
`metric_analysis/metrics/` did. Nothing here has that shape.

**Verify with the Phase 4b technique, not just the gate.** 4b split a 2,639-line module and
proved it with an AST pass requiring all 112 pre-split function bodies to be byte-identical.
That turns a 3,000-line carve from "regression is green, probably fine" into "provably a move",
and catches the one thing the fingerprint cannot: a body that drifted while the output happened
not to move on nine sessions. Write that pass first.

*(4b's pass was not kept. 6c wrote `qc/ast_move_check.py` and kept it this time: it reads the
pre-split file out of any git ref, so it generalises to Phase 10's `visualization_utils.py`
split with `--old`/`--new-dir`. It reports MISSING / DUPLICATED / CHANGED / ADDED and was
checked against injected drift, a whitespace-only edit, a deletion and a duplication before
being trusted — a checker that has never been seen to fail is not evidence.)*

**After 6b, not before.** 6b rewrites how `classify_trials` writes positions; a file split
landing in the same window makes that diff unreadable. Same reasoning that put 6a before 6b.

**`classification_utils.py` should not survive as a facade.** The re-exports were deleted in 6a
precisely so this split can end with the file *gone*, rather than turned into an index of
imports pointing at the new modules — which would be the same problem spread across more files.

**Named do-not-merge for this split: the two position-assignment rules.** `classify.py` will
carry `_assign_positions_to_valve_events` and `response_times.py` will carry
`windows.first_occurrence_positions`. They look like the same helper and are not:

| | rule | trial presenting A, B, A |
|---|---|---|
| `classify_trials` | collapses only *consecutive* repeats; a non-consecutive re-entry is a **new** position | 3 positions |
| `analyze_response_times` | each odor keeps its first position; a later repeat **overwrites** that position's event | 2 positions, position 1 holding the *second* A |

Once they sit in two files this is the §13 setup exactly — and `hidden_rule.py` in the table
above already establishes the precedent of pulling shared logic out of these two. **6c is a pure
move: do not merge them.**

**The measurement that would settle it has not been done.** They diverge only when an odor
re-appears after a different odor, so if no trial's valve-event list ever contains a
non-consecutive odor repeat, the two rules are provably interchangeable on this data. Count
trials with such a repeat across the 9 regression sessions. **Zero** ⇒ a later phase may merge
them cheaply, and §15's response-time follow-up becomes tractable too; **non-zero** ⇒ that count
is the number of trials a merge would silently move, and which rule is right becomes a
scientific question, not a refactoring one. Either way this is a *separate* decision from 6c —
do the count, record it here, do not act on it during the split.

### Phase 6 is closed  *(2026-08-12)*

6a, 6b and 6c landed GREEN; the three deferred items were then measured and closed together in
one commit with a single fixture regeneration. **`DECISIONS.md` §18** holds what they settled.

| # | item | outcome |
|---|---|---|
| 1 | the two position rules | **merged** onto `windows.positions_by_odor`. Measured first: they diverge only on a non-consecutive odor repeat — **1 of 1,731** fixture trials, **0 of 46,112** across subs 056-066. A **third**, uncatalogued rule was found in `aborted_trials._abort_positioned_events` and merged too |
| 2 | `sequence_depth`'s branch | **kept as one expression**, not collapsed onto `presentations`: completed → `max(presentations)`, aborted → `max(poke_source=='poke')` w/ `last_odor_position` fallback. The three candidates were measured — the unfiltered form moves **84** trials, the fully-filtered one **32**; this rule moves none |
| 3 | the (a)/(b) latency rename | **done**, 46 replacements. Pure rename, no value moved — 8 of 9 sessions showed added/removed columns and **zero** changed ones |

**One trial moved in total**, `sub-040 20251124` trial 44, and it is an experiment fault rather
than analysis (§18): the rig emitted no `InitiationSequence` for 97.9 s, so three F→A sampling
runs became one trial. `detect_trials` was **not** changed — it is faithful to the rig, which
also emitted a single `ChooseRandomSequence` for the whole period.

**Two standing warnings were added**, both silent on sound data and therefore meaningful when
they speak: a repeated odor within a trial, and an aborted trial whose depth resolves by neither
route.

> **Left deliberately undone: the server's derivatives are stale** and every plotter reading the
> renamed columns returns empty on them. Re-analyse at the end of 7b, not before — see the note
> in 7b, and the two `plot_regression` cases that are *vacuously* green until then.

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

## Phase 7 — Schema, save formats & provenance

**Revised 2026-08-12, with the phase's owner.** Phase 7 has *two* goals and they are the same
work seen from two sides: make the saved data **easy to query** (typed columns, tidy tables, no
JSON smuggled into cells) and **easy to trust** (know what produced a file, and be told when it
is too old to use). The declaration that gives you the first is what makes the second checkable.

### 7a. Manifest provenance  *(quick win, ~½ day)*

Add the **git commit** and the **package version** alongside the existing `created_at` date in
`manifest.json`. **Phase 2c already did the work:** call
`hypnose_helpers.provenance.provenance()` rather than writing a second implementation — it
handles the `-dirty` suffix and the fact that the import package and the distribution differ
here (`hypnose_behavior` ← `hypnose-behavior-analysis`), which makes the naive
underscore-to-hyphen guess return `None`.

Keep these **in the manifest only** — the regression already ignores it, so they never enter
the fingerprint and never cause spurious RED.

**What it is for: auditing.** "Which sessions were produced before commit X, and should I
re-run them?" It catches the case a schema check cannot — Phase 6's close-out moved a *value* on
one trial while adding and removing no column at all, so a field-set comparison would have been
silent on it and a commit stamp would not.

> **This does not re-open `DECISIONS.md` §5.** §5 rejected provenance as a *metrics-cache key*,
> because a commit stamp invalidates on every unrelated commit — a docstring fix would force
> re-analysing the whole server. Same word, different job: stamping for **audit** is this
> phase's job; stamping to decide whether to **trust a cached metric** stays rejected, and
> plotters keep computing through the registry.

**Risk:** low. **Progress:** ~40%. **Done:** manifest carries commit + version + date;
regression unaffected.

### 7b. Schema & save formats

**Do these in order** — each step is the previous one's enabling piece.

#### 1. Typed `@dataclass TrialRecord` — the declaration everything else hangs off

Replace the free-form per-trial dict with explicit typed fields and `.to_row()` for the
DataFrame. Measured 2026-08-12: `save_results` names **27** columns while `trial_data` has
**60** — the other 33 exist only because some line assigned them, so *nothing currently declares
what a trial is*.

Use **`@dataclass(slots=True)`**, so assigning a field that is not declared raises at the
assignment site instead of silently inventing a column of NaNs (`trial_dict['fr_laency_ms']`
is valid Python today).

> **Pure restructure: `regression.py` GREEN is a hard invariant.** Split the *conversion* from
> any **raising validation** in `__post_init__` — that is new failure behaviour and belongs with
> **Phase 9**. A green golden master proves none of those branches fire, so it cannot tell
> "validation is correct" from "validation is dead code". Landing both together repeats the
> mistake 6a/6b were split to avoid: a RED that cannot be attributed.

**One intended output change to expect:** a uniform record means every session carries the
`fr_*` family as NaN where the protocol does not produce them. Today they are written only for
single-reward sessions — `fr_window_latency_ms` is in `sub-057`'s fixture and in none of the
other eight. All-NaN columns are ignored downstream, so this is a column-set change only.

#### 2. Make the dataclass the loader's schema check

Once `TrialRecord` exists it *is* the single declaration of what `trial_data` should contain, so
`io/load_results.py` can say:

```python
missing = set(TrialRecord.__dataclass_fields__) - set(trial_data.columns)
if missing:
    warn(f"{session}: saved before {missing} existed -- re-run trial classification")
```

**Why this rather than 7a's version stamp:** a git SHA says *something* changed between the file
and now, not whether *this file* is affected — a one-line plotter fix and a trial-classification
restructure look identical to it. Comparing field sets answers the question actually asked, and
costs no maintenance because the dataclass is already the source of truth. The two are
complementary: **the stamp catches changed values, the field set catches changed schema.**

**The concrete case this exists for** is Phase 6's latency rename. Every derivative saved before
it carries `fa_latency_ms` etc., so `FA_avg_response_times` and `sing_rew`'s `FR_latency` find no
column, hit their `if col not in trials.columns` guard and return **empty — a blank figure with
no error**. Measured on the archive: `plot_regression`'s `FR_latency` lost all 35 lines and
`plot_response_times_completed_vs_fa` all 12. Silent staleness is the failure mode; this check is
what makes it speak. Deliberately **not** patched with a legacy-name map in Phase 6 — that would
have been a special case to unpick here.

#### 3. CSV becomes an option, defaulting off

Keep a human-readable CSV of `trial_data`, but behind `save_csv=` — `True` when investigating,
**`False` by default** once the parquet path is trusted.

**Audit the readers before flipping the default.** `load_results_dir` already prefers parquet
and falls back, but `qc/_common`'s canonical form and ad-hoc diff scripts read
`trial_data.csv` directly. **The QC harness must set `save_csv=True` explicitly** rather than
relying on the default, so a later default change cannot quietly break the gate.

#### 4. `position_data.parquet` — the measured per-position table

Flatten `position_valve_times` / `position_poke_times` / `presentations` into one tidy table,
**one row per `trial × position`**, with `poke_source` and §2's provenance flags
(`in_poke_times` / `in_presentations` / `in_valve_times`) as real columns.

Today these are **JSON strings inside a cell** — a table smuggled into a column, equally awkward
in CSV and parquet. Flattening removes the parse at load (`build_position_data` costs ~21.9 ms
per session, every time), gives typed filterable columns, and shrinks the parquet.

`frames.build_position_data` already derives exactly this shape, so this promotes an existing
derivation to a written artifact rather than inventing a format. **Carry its provenance flags
across — `DECISIONS.md` §2.** Parquet's native nested types were considered and rejected: you
would explode a list-of-structs to a long frame for nearly every query anyway, and it cannot
round-trip to the human-readable CSV.

**Phase it:** add the side-table additively, keep the blobs during transition, drop the blobs
last. `trial_data` then holds **only one row per trial, scalar columns**.

#### 5. Per-trial metric tables — separate files, keyed by grain

Nine registered metrics return tables rather than session values and are deliberately absent
from `metrics_*.json`. **They do not go in `position_data`** — that table is what was
*measured*, and these are *derived*; mixing them makes it impossible to tell one from the other.
They also do not share one grain, so name the files by grain:

| file | grain | metrics |
|---|---|---|
| `metrics_by_trial.parquet` | `global_trial_id` | `inter_trial_interval`, `trial_poke_span`, `trial_poke_total`, `hr_abort_poke_gap`, `valve_to_reward_latency`, `reward_delivery_latency`, `fa_latency_from_pokeout` |
| `metrics_by_poke.parquet` | trial + poke index | `poke_durations` (739 rows for one session) |

**Not everything unreported is savable**, and that is not a gap to close:

- **by-odor / by-position aggregates** (`poke_duration_by_odor`, `fa_rate_by_odor`,
  `presentation_counts_by_odor`, …) are session-level summaries — they belong in the JSON.
- **figure-parameterised metrics** cannot be saved at all: three take a `window` and two take an
  `fa_types` filter, which are properties of the **figure**, not of the session (§5). This is
  why "save everything and only load" is unreachable.

**Session-level metrics stay JSON.** They are ~25 scalars plus numerator/denominator
contributions — metadata-shaped, and parquet buys nothing for a flat dict.

> **Saved metrics are an export and a record, never an input.** `DECISIONS.md` §5 settled that
> plotters compute through the registry, and it caught a real defect: `decision_accuracy`,
> `avg_response_time` and `FA_avg_response_times` were each obtainable *both* ways, so two
> figures could show one quantity and disagree. Measured, the cache is worth **25 ms** against a
> mount walk costing **seconds** (14.6 s for a cold `find_session`) — and flattening the blobs
> makes computing cheaper still. **Do not "optimise" a plotter by reading these back.**

#### 6. Re-analyse the server's derivatives — at the **end** of 7b

7b is the last phase that changes the saved schema, so re-running earlier means running twice.

The two gates differ here and only one is affected: **`regression.py` recomputes from rawdata
into a temp dir and never reads the archive**, so it is unaffected; **`plot_regression.py` reads
the saved derivatives**, so it is.

> **Until the re-run, two `plot_regression` cases are *vacuously* green** — `FR_latency`
> (`sub-057 20260709`) and `plot_response_times_completed_vs_fa` (`sub-040 20251124`,
> `20251229`). Both trees look for the renamed columns, neither finds them, both draw nothing,
> so the diff is empty **because both sides are broken**. A two-tree diff cannot see that. Do
> not read those greens as coverage. **Phase 10 depends on `plot_regression`, so this must be
> cleared before it starts.**

**Standardise on parquet for tables, JSON for metadata. No pickle** for saved outputs
(version-fragile — the somnotate work is a live example of pickle/version coupling biting).

**Risk:** med (touches downstream readers). **Intended schema change ⇒ regenerate fixtures
deliberately, with the +/−/~ diff confirming only the intended fields moved.**
**Done:** no pickle outputs; `TrialRecord` is the schema and the loader checks against it;
`save_csv` optional; `position_data` and the two metric tables exist; blobs removed; manifest
carries provenance; the server's derivatives re-analysed.

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
`validate_subject`. `hypnose-helpers/tests/test_layout.py` (20 tests, mount-free, runs without
pytest) is the pattern.

> **The "build the outcome-classifier tests before Phase 6" prerequisite was dropped
> (2026-08-11), after Phase 6 shipped without it.** `regression.py` exercises
> `classify_completed_trial` across 1,731 trials on 9 sessions and runs in a few minutes, which
> is stronger coverage than a handful of hand-written cases — and 6a/6b/6c are all committed
> GREEN, so the risk the prerequisite guarded against is spent.
>
> **Where it would still earn its place is 7b**, not Phase 6: `TrialRecord.__post_init__` is new
> validation logic whose branches the 9 fixture sessions may never take, and a golden master
> cannot see an unexercised branch. Reconsider there, on the evidence, rather than as a standing
> obligation.

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
