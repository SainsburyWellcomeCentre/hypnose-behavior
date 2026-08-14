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
`DECISIONS.md` sections 1-28 already hold everything that still binds.

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

**Sizes, measured 2026-08-13:**

| module | lines |
|---|---|
| `visualization/visualization_utils.py` | 6,785 |
| `visualization/movement_analysis_utils.py` | 3,604 |
| `visualization/pred_seq_utils.py` | 1,630 |
| `visualization/sing_rew.py` | 1,363 |
| `visualization/` total | 15,319 |

**Out of scope (explicit):** do NOT change protocol detection — the
`"odourdiscrimination" in name` string matching stays as-is.

---

## 1. The QC safety net — use it after every change

`src/hypnose_behavior/qc/`, see `qc/README.md`:

- **`regression.py`** — golden master, **54 checks** (9 coverage sessions x 6 fingerprints:
  `trial_data`, `metrics`, `position_data`, `metrics_by_trial`, `metrics_by_poke`,
  `unreported_metrics`). Reports added/removed/changed columns and keys, so an intended
  change is easy to confirm. `--generate` writes baselines. ~3-15 min depending on mount.
- **`plot_regression.py`** — old-vs-new diff of what the plotters **draw**, **38 cases**.
  Two-tree, not a golden master. **Item 1 depends on it.** `DECISIONS.md` sections 7, 22.
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
- **`plot_regression`'s banner is not its result — COUNT THE CASES.** A full run lists 38.
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

**7a. A parquet peek tool.** With CSV off by default, "what is in this file" needs an
answer that is not a notebook. Small, no design risk, do it first.

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

1. **7a** parquet peek — cheap, immediately useful.
2. **5** the `non_initiated` collapse — measured, small, tidies the output.
3. **4** `parameters.py` + manifest stamp.
4. **3** session selectors.
5. **1** Phase 10 — coverage measurement first, then the move, then the cleanup.
6. **7b/7c + 2** the accessors and the curated API — last, once the module layout has
   stopped moving.

Item 6 needs no work: it is a decision, recorded above.
