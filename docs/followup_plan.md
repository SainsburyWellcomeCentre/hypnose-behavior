# follow-up plan — after restructure_2 (target: v2.0.0 release and beyond)

Successor to `docs/restructure_2_plan.md`, which is **closed**: phases 0-7 and 11 are done,
and what remained live has been carried into the seven items below. Same branch,
**`hypnose-restructure`**; same goal — tidier, faster, reusable across the repo family
**without accidentally changing analysis output**.

**Two live documents, and nothing else:**

| document | what it is |
|---|---|
| `docs/followup_plan.md` (this) | the spine: remaining work and the operating rules |
| `docs/DECISIONS.md` | settled rules and standing traps — **read at the start of every item** |

`docs/archive/` holds closed working documents, including the restructure_2 plan and its
per-phase Progress table. **Nothing live points into it** and no item needs to read it; it
is there if a historical "why was it done that way" ever needs answering, and
`DECISIONS.md` sections 1-30 already hold everything that still binds.

---

## 0. Context a fresh session needs

`hypnose_behavior` under `src/`, with `io/`, `trial_classification/`, `metric_analysis/`,
`visualization/`, `utils/`, `qc/`, `modelling/`, plus `frames.py` as a package-root leaf.
Terminal entry points in `scripts/`. No back-compat shims; all imports canonical.

**The repo family:**

| repo | package | role |
|---|---|---|
| **hypnose-behavior-analysis** | `hypnose_behavior` | behavioural analysis (this repo) |
| hypnose-somnotate | `hypnose_somnotate` | EEG sleep scoring (v1.0.0) |
| hypnose-eeg-analysis | `hypnose_eeg` | EEG analysis (coming) |
| neuropixel analysis | `hypnose_ephys` | ephys (planned) |
| **hypnose-helpers** | `hypnose_helpers` | shared, modality-agnostic utilities |

**What Phase 7 left the data looking like** — this is the precondition for items 5 and 7:
`trial_data` is **one row per trial, scalar typed columns only**; the per-position record
lives in `position_data.parquet` (one row per `trial x position`, with `poke_source` and
provenance flags); table-returning metrics live in `metrics_by_trial.parquet` and
`metrics_by_poke.parquet`; session-level metrics stay in `metrics_*.json`. Parquet is the
format, CSV a convenience that is **off by default**.

**Sizes after Item 1 (2026-08-18).** `visualization_utils.py` (6,785) and
`movement_analysis_utils.py` (3,604) no longer exist; they are 13 modules, largest 1,646:

| module | lines |
|---|---|
| `visualization/movement/traces.py` | 1,646 |
| `visualization/pred_seq_utils.py` | 1,630 |
| `visualization/sing_rew.py` | 1,363 |
| `visualization/false_alarm.py` | 1,358 |
| `visualization/hidden_rule.py` | 1,238 |
| `visualization/sampling.py` | 1,111 |
| `visualization/accuracy.py` | 895 |
| `visualization/movement/summary_stats.py` | 875 |
| `visualization/movement/speed.py` | 763 |
| the rest (`prep` 658, `valve_poke_plots` 635, `rewards` 576, `timing` 522, `overview` 487, `choice` 438, `sing_rew_movement` 428, `tortuosity` 325, `sequence` 290, `panels` 264, `primitives` 125) | 125–658 each |

**Out of scope (explicit):** do NOT change protocol detection — the
`"odourdiscrimination" in name` string matching stays as-is.

---

## 1. The QC safety net — use it after every change

`src/hypnose_behavior/qc/`, see `qc/README.md`:

- **`regression.py`** — golden master, **90 checks** (9 coverage sessions x 10 fingerprints:
  `trial_data`, `metrics`, `position_data`, `metrics_by_trial`, `metrics_by_poke`,
  `non_initiated_attempts`, `unreported_metrics`, plus `non_initiated_sequences` /
  `non_initiated_odor1_attempts` / `non_initiated_FA` as deletion guards that stay ABSENT).
  Reports added/removed/changed columns and keys, so an intended change is easy to confirm.
  `--generate` writes baselines. ~3-15 min depending on mount. Sections 26, 30.
- **`plot_regression.py`** — old-vs-new diff of what the plotters **draw**, **43 cases**,
  signing both the returned value and every figure left open. Two-tree, not a golden
  master. Reports the **number of values compared** (~2.83 M) and names any case that
  compared nothing. `DECISIONS.md` sections 7, 22, 33.
- **`figure_provenance.py`** — does a saved figure's provenance record still name the
  plotter that drew it? The only gate that calls `save_figure`, which `plot_regression`
  deliberately never does. 92 figures across 37 cases. Sections 9, 33.
- **`position_data_lossless.py`** — asserts every blob field is recoverable from
  `position_data` bar a named allow-list. 9/9. Section 27.
- **`verify_scripts.py`** — regression through the actual CLI scripts (arg wiring).
- **`verbose_diff.py`** — two-tree stdout diff (what `regression.py` never sees).
- **`ast_move_check.py`** — byte-identity of function bodies, for pure moves.
- **`check_imports.py`** — referenced-but-not-imported globals. Seconds.

**Operating rules**

- Run everything with `~/miniconda3/envs/hypnose-analysis-test/bin/python`. Fixtures are
  only valid in the env in `fixtures/env.json` (py3.12.13 / pandas 3.0.1 / numpy 1.26.4 +
  pyarrow).
- **Invoke by absolute path, with `-u`, and no `cd`:**

  ```bash
  PY=~/miniconda3/envs/hypnose-analysis-test/bin/python
  QC=~/repos/harris_lab/hypnose/hypnose-behavior-analysis/src/hypnose_behavior/qc
  $PY -u $QC/regression.py
  ```

  `cd <repo> && $PY src/...` gets stopped at the permission layer and looks like a hang;
  without `-u` a redirected run shows nothing until it exits; and `-u` is **not enough
  through `grep`** — use `grep --line-buffered`.

- **Never filter gate output.** Grepping for RED hides `[ERROR]`; grepping for the summary
  hides the `+/-/~` lines that say what moved.
- **`plot_regression`'s banner is not its result — COUNT THE CASES.** A full run lists
  **43**, and reports **~2.83 M drawn values**. A case count says a case *ran*, not that
  it drew anything.
- **`REGRESSION RED: N` with no `[RED]` lines above it is a dropped mount**, not a
  regression. `[ERROR]` and `[NO BASELINE]` count into the same total.
- **Never run two mount-heavy jobs at once** — the overlap exhausts the SMB client handle
  pool (`Errno 24`, processes wedged in `U`). And a shallow `iterdir()` will "pass" while
  the mount is still wedged: verify by walking subject trees and **opening files**.
- **Gate on reachability, not on how big the change looks.** For a **pure move**:
  `ast_move_check` + `git diff -M --stat` (0 insertions / 0 deletions) + `check_imports`,
  and skip `plot_regression` — but only if it really is a move. A move can still change
  behaviour through module-level side effects or `__file__`-derived state (Phase 2a hit
  both).
- **Byte-identical philosophy.** Pure refactors keep the gates GREEN. Intended output
  changes get fixtures **regenerated deliberately in the same commit**, with the diff
  confirming only the intended fields moved.

---

## 2. How to work through this plan

**One item per chat.** Long sessions degrade in exactly the way this work cannot tolerate:
dropped items, repeated tool mistakes, and narration that disagrees with the output. Commit
at every boundary so each new chat starts from a green, known state.

Each item below states **what it delivers**, **its gate**, and **its known trap**. Where a
measurement has already been taken it is recorded here so nobody re-derives it.

---

## Item 1 — Phase 10: split `visualization_utils.py`, and move the movement plotters

**Done 2026-08-18.** Both files are gone, carved into **13 modules** by behavioural
construct, none over 1,650 lines; `movement_analysis/` renamed **`movement/`** (the
movement *analysis* has lived in `metric_analysis/` since Phase 4a, so the old name
pointed elsewhere) and `metric_analysis/movement.py` became
`metric_analysis/movement/speed_analysis.py`. See `DECISIONS.md` section 33.

**The boundary was decided by a measurement, and it was not the one the brief implies.**
The 22 public plotters in `visualization_utils.py` reference each other **zero times**, so
"where do the plotters go" was never the constraint — the only question was where the 12
private helpers go, which section 3 answers. Six went to the `prep.py` leaf.

**The coverage measurement found a gate gap worth more than the split.** `plot_regression`
signed only what a plotter *returned*, and four cases return a dict — so
`plot_traces_with_speed_threshold` was gated by **3 values against the 37,347 on its
figures**, and three of the four were the movement plotters this item moves. Every open
figure is now signed, and the gate reports **how many values it compared** (~2.83 M across
43 cases) so a case that draws nothing cannot pass as one that does.

**Three things the item found that no gate was watching:**

- **Provenance was ungated entirely** — `plot_regression` runs `save=False`, so
  `save_figure` never ran under any gate and the record embedded in every saved PDF was
  built by unexercised code. New gate: `qc/figure_provenance.py`. It also corrects
  section 9: a pure move does **not** change `chain`.
- **`plot_cumulative_rewards_by_trial` raised on every multi-session call** and no case
  reached it. Fixed.
- **Section 28's claim that `plot_traces_with_speed_threshold` "recomputes and saves" was
  wrong** — it never wrote anything, and its in-memory recompute was a second derivation of
  a `metric_analysis` quantity that no gate could reach. Removed in favour of a warning;
  `scripts/run_speed_analysis.py` is now the one way to produce `speed_analysis.parquet`.

*Original brief below.*

**Delivers.** `visualization_utils.py` (6,785) and `movement_analysis_utils.py` (3,604) —
**10,389 of the 15,319 lines in `visualization/`** — broken into modules by behavioural
construct, with `movement_analysis_utils.py` moved under
`visualization/movement_analysis/` beside `sing_rew_movement.py`.

**Do the coverage measurement first, as its own commit.** `plot_regression` has 38 cases,
which is *not* 38 of the ~60 plotters in those two files. Measure which plotters the cases
actually execute (the Phase 7b.4b technique: patch the accessor, run the case list in one
process, record caller lines), add cases for the uncovered ones, and **verify each new case
draws something before adding it** — a case that draws nothing goes green in both trees.
Section 28.

**Gate.** Move commits: `ast_move_check` + `git diff -M --stat` + `check_imports`. Cleanup
commits: `plot_regression` (counted).

**Trap.** "A pure move cannot change output" is *almost* always true; the exceptions are
module-level side effects and `__file__`-derived state, which is why the move gate includes
`check_imports` and the diff stat.

---

## Item 2 — a curated public API surface (not `__init__` re-exports)

**Done 2026-08-19**, with items 7b/7c. `hypnose_behavior/api.py`, 25 hand-picked names,
and **nothing in it pulls matplotlib** — which is now a constraint on what may be added,
not an accident. `hypnose_behavior/__init__.py` forwards four names (`session`,
`sessions`, `Session`, `metric_names`) via PEP 562's lazy `__getattr__`, the form this
item permits. See `DECISIONS.md` section 34.

**The gate was "import-time cost measured before and after", and it is unchanged:**
`import hypnose_behavior` **39 modules / 0.003 s**, `hypnose_behavior.frames` **614**
with no matplotlib, harp, aeon or dotmap, `hypnose_behavior.api` 1,326 — paid only when
touched. `check_imports` **does not check `__init__.py`**, so the forwarder was exercised
by hand.

*Original brief below.*

**Delivers.** One explicit, hand-maintained module (`hypnose_behavior/api.py`) naming what
other repos may use. Implemented by item 7.

**Decided: no eager re-exports in the package `__init__.py` files.** They are docstring-only
today, deliberately — section 3: *"Every package `__init__.py` is docstring-only, so
importing a submodule triggers no package-level side effects."* That property is what keeps
`frames.py` a leaf and `io/` free of `metric_analysis`. Eager re-exports would make
`import hypnose_behavior.frames` pull in matplotlib, harp, aeon and dotmap, and would be
paid by every downstream repo — including the Python-3.9-pinned ones `frames.py` is kept
importable for. If flat within-repo paths are ever wanted, use lazy `__getattr__`
(PEP 562), not eager imports. **Within-repo ergonomics are not the priority; cross-repo
access is.**

**Gate.** `check_imports`; import-time cost measured before and after.

---

## Item 3 — `ses` / session-index selectors in `trial_classification` and `metric_analysis`

**Done 2026-08-18.** All six selectors accepted by `batch_analyze_sessions` (rawdata),
`batch_run_all_metrics_with_merge` (derivatives) and the three CLI scripts.
`batch_analyze_sessions` no longer slices a listing locally. See `DECISIONS.md` section 32.

**The item's real content was the rawdata/derivatives split, and it is worse than the
brief assumed.** `--index N` names a *different* session in the two trees on **7 of 8
subjects** — all 27 on sub-061 — because derivatives is a subset of rawdata but **not a
prefix** of it. `ses` is tree-stable (9/9 fixture sessions), so the six selectors split
into tree-stable (`ses`, `dates`, `date_range`, `ses_range`) and tree-relative (`index`,
`index_range`). `batch_process.py`, which chains both resolvers, **refuses the two
tree-relative ones**.

**Both named gates were blind, and one is blind structurally.** `regression.py` never
executes either batch function — it calls `analyze_session_multi_run_by_id_date` and
`run_all_metrics` directly — so it was skipped on reachability rather than run for a
meaningless GREEN. `verify_scripts._run_cli` hardcoded `--dates`, and was extended with
the flags in the same commit; each case asserts the fixture md5 **and** that exactly one
session directory was written, because a flag that is accepted and then ignored would
match the md5 anyway.

*Original brief below.*

**Delivers.** The six session selectors (`ses`, `index`, `date_range`, `ses_range`,
`index_range`, `dates`) accepted by trial classification and metric analysis, as
`visualization/`'s 44 session-selecting functions already accept them.

**Gate.** `verify_scripts` (this is CLI wiring), `regression`.

**Trap.** Section 8: **`session_index` selects, it does not position.** An index is only
defined against a listing, so it must resolve through the same
`derivatives.find_session` / `session_selectors` machinery, never by slicing a sorted list
locally. Note that `trial_classification` resolves *rawdata* while the rest resolve
*derivatives*.

---

## Item 4 — `parameters.py` beside `frames.py`

**Done 2026-08-18.** `hypnose_behavior/parameters.py`, a package-root leaf holding
`PRE_ODOR_GRACE_MS`, `LATE_LATENCY_WINDOW_MULTIPLIER` (the inline `3.0`, now named) and
`CACHE_MAX_ITEMS`; `scoring_parameters()` is stamped into `manifest.json` as
`analysis_parameters`. `regression` GREEN 90/90 with no regeneration — a pure extraction.
See `DECISIONS.md` section 31 for what it settled: why the stamp is manifest-only and not
merged into `summary.json`'s `params`, why the stamp is built default-in, and the two
measurements below.

**Two of the plan's premises did not survive measurement, and both are recorded:**

- **The stamp is invisible to every gate**, exactly as `DECISIONS.md` section 19 requires of
  the manifest — so "the values go into the manifest" does *not* by itself discharge this
  item's trap. Deliberately left ungated; section 31 states what that leaves uncaught and
  the condition for revisiting it.
- **The 3× multiplier is gated on 4 of the 9 coverage sessions, not 9.** Five carry
  `response_time_window_sec = 99999.0`, so every latency buckets `_time_in` and the knob
  decides nothing there. The `fr_label` side is exercised at its boundary by no session at
  all.

*Original brief below.*

**Delivers.** One home for the genuinely hardcoded scoring knobs, plus their values stamped
into `manifest.json` alongside the commit and version.

**Measured 2026-08-13 — the list is short, and that is by design.** Three categories, only
one of which belongs in the file:

| category | examples | verdict |
|---|---|---|
| unit conversions | `* 1000.0` (ms), everywhere | **not parameters**, leave alone |
| per-session schema values | `sample_offset_time`, `required_min_sampling_time_ms` per odor, `isSingleRewardProtocol` | **must not be centralised** — read per session from the task schema (`params.py`); they legitimately differ between sessions |
| genuinely hardcoded knobs | `PRE_ODOR_GRACE_MS = 25.0` (`trial_classification/windows.py`), the **3x response-window** multiplier for `FA_late` / `FR_late`, `CACHE_MAX_ITEMS = 40` | **this file** |

`modelling/switchpoint/`'s constants (`ACF_MAX_LAG`, `N_STARTS`, `SWITCH_THRESHOLD`, …) are
a self-contained cluster and stay with their model.

**Gate.** `regression` (GREEN — this is a pure extraction; if a value moves, that is a
separate intended change).

**Trap.** A file that makes a value easy to change makes it easy to change **silently**,
after which old and new derivatives are no longer comparable and nothing says so. That is
why the values go into the manifest: section 19 established the manifest as the audit
surface, and "what was this session scored with" must be answerable from the file itself.

---

## Item 5 — collapse the `non_initiated_*` files: 9 files -> 2

**Done 2026-08-18 — and it went further than 9 -> 2.** `non_initiated_FA` renamed
**`non_initiated_attempts`**, and it is now the *only* non-initiated table: both
`non_initiated_sequences` and `non_initiated_odor1_attempts` are contained in it **by
construction**, so neither is written. Delivered as **3 files -> 1 by default** and 9 -> 3
under `--save-csv` (section 23 had already removed six of the original nine, and `save_csv`
was deliberately left uniform across tables). See `DECISIONS.md` section 30 — the gate gap
it found, the by-construction proof for both inputs, and the section 27 allow-list
justification that had to be restated because it named a table that no longer exists.

**The plan's gate was wrong, and that was the item's first finding.** Neither `regression`
nor `verify_scripts` could see a deleted or renamed table; the gate was extended first, as
its own commit, before the change was made.

*Original brief below, kept for the measurements it records.*

**Delivers.** `non_initiated_sequences` **deleted**, `non_initiated_FA` **renamed**,
`non_initiated_odor1_attempts` kept, CSV and `.schema.json` dropped for all of them.

**Measured 2026-08-13 across all nine fixtures — `non_initiated_sequences` is fully
redundant:**

| relation | result |
|---|---|
| columns: `sequences` (12) subset of `FA` (23) | **True**, zero columns unique to `sequences` |
| rows (key `run_id` + `attempt_start` + `odor_name`) | **subset on all 8 sessions carrying both** |
| shared cells | **zero genuine differences** |

So this is a **deletion, not a merge** — no union schema, no `kind` column. (sub-048 has
none of the three tables: it simply had no non-initiated attempts, and they are not written
when empty.)

**Rename it.** `non_initiated_FA` is not the FA subset — it is *all* non-initiated attempts
annotated with FA outcome. Left as-is the name tells every future reader the opposite of
the truth. `non_initiated_attempts` or similar.

**Open question to settle first.** The 5-6 rows `FA` has beyond `sequences` carry
`fa_label` but **null `attempt_number` and null `failure_reason`** — so they are not failed
initiation attempts in the same sense. Confirm what they are: either a real distinction the
renamed table should document, or a latent defect.

**Confirm it is true by construction**, in `aborted_trials.py`, not merely true on nine
sessions — section 27's lesson.

**Gate.** `regression` (intended output change: removed tables ⇒ deliberate `--generate`),
`verify_scripts`.

**Trap, and it nearly landed.** The first containment check reported "78 shared-cell
differences". 74 were `1` vs `1.0` — a dtype artefact from comparing as strings, caused by
the extra NaN rows forcing `attempt_number` to float; the other 4 vanished under proper
NaN/datetime handling. **A subset check says nothing about cell agreement, and a crude cell
check says nothing about values.**

---

## Item 6 — disposition of the old Phases 8 and 9

**Phase 8 (profile, then vectorise): closed, measured, not worth doing.** Section 5
measured a cold `find_session` at **14.6 s** against **29 ms** to compute every metric for
the session it found — and `build_position_data`, which was 22 of those 29 ms, is now a
file read (Phase 7b.4b). I/O dominates; vectorising the remainder is not where the time is.
Recorded so nobody re-derives the question.

**Phase 9 (validation with clear errors): narrowed, not urgent.** What landed in Phase 7 is
*schema* validation — the loader's missing-column warning, `ConflictingProtocolError`,
`UncarriedPositionFieldError`, and the no-position-source warning. Phase 9's original
target was clear errors on **bad input data**, which is a different thing and largely
untouched. Keep as a note to explore later; not required for the v2.0.0 release.

---

## Item 7 — parquet peek tool and the cross-repo accessors

**7a. A parquet peek tool.** ~~With CSV off by default, "what is in this file" needs an
answer that is not a notebook. Small, no design risk, do it first.~~
**Done 2026-08-18** — `io/parquet_peek.py` + `scripts/parquet_peek.py`, three narrowing
views (session inventory -> one line per column -> one column with values). See
`DECISIONS.md` section 29 for what it settled, including why it is not a `qc/` tool and
why it does not read the `.schema.json` sidecar.

**7b/7c. Done 2026-08-19** — `hypnose_behavior/accessors.py`, reached through `api.py`
(item 2, above). The handle, `sessions([...])` for a cohort, and `pooled` /
`pooled_metrics`; the README gained a task-oriented "what do I need → what do I call"
section. See `DECISIONS.md` section 34.

**Two of this item's premises did not survive measurement, and both are recorded there.**
Section 5's "three metrics take a `window` and two take an `fa_types`" is **swapped** —
two take a `window`, both required with no default, and section 26's boundary
(*required parameter and no wrapper*) is the one that holds. And "validate requested
column names against `trial_data_columns(mode)`" is wrong **in both directions**: measured
on the server, every saved session both carries columns the declaration does not name and
lacks ones it does, so validation is against *this session's frame* and the declaration is
used only to word the error.

**The gate was reachability, measured before writing code.** Seven of the eight QC tools
cannot see a module nothing imports (section 29). `plot_regression` became reachable —
and mandatory — only because `prep._computed_metrics` was repointed onto the new shared
`run.metric_value`: **GREEN, 43 cases, 2,828,307 drawn values**. The rest is
`check_imports` PASS plus a **79/79 probe** against two sessions analysed by current code
into a scratchpad root, whose load-bearing checks are that loaded md5 == saved md5 and
that `s.metrics(REPORT)` equals the `metrics_*.json` the same run exported, 25/25.

*Original brief below.*

**7b. A session handle and the measured-data accessors.**

```python
s = hypnose_behavior.session(57, 20260709)   # resolves ONCE
s.trial_data(columns=None)                   # None = all
s.position_data()
```

**Resolve the session once, not per call.** Every `get(subjid, date, …)` pays
`derivatives.find_session` — **14.6 s cold**, against 29 ms to compute every metric. Five
separate module-level functions means a caller wanting `trial_data` plus two metrics pays
it three times, and in a loop over sessions it dominates everything. Thin one-shot
functions may wrap the handle for casual use; the expensive path should be opt-out, not
mandatory.

**7c. The metric accessor — computes, does not load.**

```python
s.metrics(["decision_accuracy", "poke_durations"])
```

This is the best part of the design: it *feels* like loading, and is section 5 compliant.
`MetricSpec.call(results)` already does the work and the registry already knows each
metric's frame, grain and adapter — so this is mostly a public name over existing plumbing.

- **Do not make the caller pick the grain.** One `metrics([...])` returning each metric in
  its own shape beats three grain-specific functions: the registry knows the grain, the
  caller should not have to.
- **Figure-parameterised metrics must raise, not guess.** Three metrics take a `window` and
  two take `fa_types`; those are properties of a *figure*, not of a session (section 5) —
  which is exactly why `regression.py` fingerprints 16 of 18. Asking for one without its
  parameter must say so.
- **Validate requested column names** against `trial_data_columns(mode)` so a typo raises
  instead of silently returning a narrower frame — the `fr_laency_ms` lesson, section 21.

**Gate.** `check_imports`, `verify_scripts`; `regression` if any shared code path moves.

**Trap.** The module docstring must state that it **computes and does not read
`metrics_*.json`**, and there must be no sibling that loads. Otherwise someone later
"optimises" it into a disk read — precisely the defect section 5 caught, where one quantity
was obtainable two ways and two figures disagreed.

---

## Suggested order

1. ~~**7a** parquet peek — cheap, immediately useful.~~ **Done 2026-08-18.**
2. ~~**5** the `non_initiated` collapse — measured, small, tidies the output.~~
   **Done 2026-08-18.**
3. ~~**4** `parameters.py` + manifest stamp.~~ **Done 2026-08-18.**
4. ~~**3** session selectors.~~ **Done 2026-08-18.**
5. ~~**1** Phase 10 — coverage measurement first, then the move, then the cleanup.~~
   **Done 2026-08-18.**
6. ~~**7b/7c + 2** the accessors and the curated API — last, once the module layout has
   stopped moving.~~ **Done 2026-08-19.**

Item 6 needs no work: it is a decision, recorded above.

---

## All seven items are closed

| item | state |
|---|---|
| 1 — split `visualization_utils.py` | done 2026-08-18, section 33 |
| 2 — curated public API | done 2026-08-19, section 34 |
| 3 — session selectors | done 2026-08-18, section 32 |
| 4 — `parameters.py` | done 2026-08-18, section 31 |
| 5 — collapse `non_initiated_*` | done 2026-08-18, section 30 |
| 6 — disposition of Phases 8/9 | a decision, recorded above; no work |
| 7a / 7b / 7c — peek tool, handle, metric accessor | done 2026-08-18 / 2026-08-19, sections 29 and 34 |

**What is open, and deliberately so** — each is recorded where it was measured, not
carried here as a task:

- **The two visualization notebooks are broken by design** (section 33): they do
  `from ...visualization_utils import *` against a module that no longer exists.
  `api.py` now exists as the thing to point them at, which was the open question — but
  they are unchanged, because nothing gates a notebook and the user's call was to leave
  them loud.
- **`plot_movement_trace` can be covered by no gate case** — it needs an ezTrack
  `add_timestamps_to_tracking` CSV that no coverage session has (section 33).
- **The manifest's `analysis_parameters` stamp is asserted by no gate** (section 31),
  with the condition for revisiting stated there: a third knob.
- **`run_all_metrics` is not routed through `metric_value`** (section 34), because its
  loop's stdout *is* the metrics `.txt`.
- **The server has not been re-analysed.** Every saved session predates the restructure —
  no `protocol_mode`, no provenance stamp, position blobs still in `trial_data`. The
  accessors read those files correctly, and say what is missing when asked for a column
  that is not there; but the fixture sessions are the only ones current with this code.
