# proposed changes — after v2.0.0

Successor to `followup_plan.md` (archived at `db1b3b0`, all seven items closed). Same
goal: tidier, faster, reusable across the repo family **without accidentally changing
analysis output**.

Everything below was measured on `2b2c588` (v2.0.0). The numbers are recorded so a fresh
session does not re-derive them.

---

## 0. Context a fresh session needs

**The package, measured.** 102 modules, 36,836 lines, 311 internal import edges.
**Zero cycles at module granularity** (Tarjan SCC, no SCC > 1, no self-loops) — the file
graph is a DAG and should stay one. Two cycles exist at *directory* granularity, and
items 3–5 are about those.

| dir | files | lines | fan-out | imported by |
|---|---|---|---|---|
| `visualization/` | 25 | 16,196 (44.0%) | 5 | nothing |
| `trial_classification/` | 15 | 6,207 | 3 | `io`, `qc` |
| `metric_analysis/` | 19 | 4,799 | 3 | root, `qc`, `visualization` |
| `qc/` | 13 | 3,139 | 5 | **nothing** (a true sink — keep it that way) |
| `io/` | 11 | 2,686 | 3 | everything |
| `modelling/` | 9 | 1,770 | 2 | `qc`, `visualization` |
| `api.py` + `accessors.py` | 2 | 735 | 4 | nothing |
| `frames.py` + `parameters.py` | 2 | 584 | **0** | 15 / 4 modules |
| `debug/` | 1 | 511 | 2 | nothing |
| `utils/` | 2 | 159 | 2 | six dirs |

**The root cause behind items 6–8.** `utils/helpers.py:139` `_filter_session_dirs` is a
"paths-only shim" that receives `SessionRef` objects — carrying `.path`, `.ses`,
`.session_index`, `.date` — and returns bare `Path`s. **32 call sites.** Because the
`SessionRef` is discarded, every caller rebuilds by hand what it already had:

| duplication | count |
|---|---|
| `results_dir = ses_dir / "saved_analysis_results"` | 44 sites, 26 files |
| the same literal inside a `qc/` glob | 3 sites, 3 files |
| `if not results_dir.exists(): continue` immediately after | 35 sites |
| `ses_dir.name.split("_date-")[-1]` to recover the date | 18 files |

`parse_session_dirname` is already re-exported at `io/layout.py:33` and is used in
exactly one place (`accessors.py:208`). The docstring one function above the shim, at
`helpers.py:130`, names the problem — *"the habit that produced 17 copies of this
lookup"* — and then the shim below re-creates it.

**Out of scope (explicit):** do NOT change protocol detection, do NOT re-analyse the
server, do NOT touch the drawing code inside the plotters.

---

## 1. The QC safety net — use it after every change

Unchanged from the previous plan. Eight gates, `src/hypnose_behavior/qc/`:

| gate | when |
|---|---|
| `regression.py` | any change that could move a value (9 coverage sessions, 10 fingerprints) |
| `plot_regression.py` | any change a plotter can see (44 cases) |
| `verify_scripts.py` | CLI wiring |
| `check_imports.py` | every commit — it is seconds and it catches moved names |
| `ast_move_check.py` | **every pure move** (items 3, 4, 6) — proves a move was a move |
| `figure_provenance.py` | anything touching `save_figure` |
| `outcome_agreement.py`, `position_data_lossless.py` | the specific invariants they name |

**Trap carried over:** seven of the eight cannot see a module nothing imports. Check
reachability *before* trusting a GREEN.

**One item per chat.** Long sessions degrade in exactly the way this work cannot
tolerate.

---

## Order

| # | item | gate | risk |
|---|---|---|---|
| 1 | ~~metadata and documentation truth~~ **done 2026-08-20** | none needed | none |
| 2 | ~~`qc/check_layering.py` — the gate, before the moves~~ **done 2026-08-20** | itself | none |
| 3 | selection helpers `utils/` → `io/layout.py` | `ast_move_check` | low |
| 4 | `detect_settings` / `detect_stage` → `io/` | `ast_move_check` | low |
| 5 | the `outcome.py` exception — decide and whitelist | `check_layering` | none |
| 6 | `results_dir()` / `table_path()` and the 47 sites | `regression`, `plot_regression` | low |
| 7 | retire `_filter_session_dirs` → `SessionRef` | `plot_regression`, `regression` | medium |
| 8 | `prep.iter_sessions()` and the plotter preambles | `plot_regression` (44) | medium |
| 9 | collapse `metric_value` / `run_all_metrics` | `regression` | low |

Item 9 is independent of 1–8 and can be taken at any point as a short session.

---

## Item 1 — metadata and documentation truth

**Done 2026-08-20** (`e82f15c`). `regression` GREEN 9/9, `check_imports` PASS.

Pure deletion and correction. No gate can fail because nothing executable changes.

**1a. Drop the Python 3.9 floor.** `pyproject.toml:10` declares `>=3.9,<3.13`, with the
rationale at lines 11–13 (a 3.9-pinned sibling installing the base helpers). That
rationale has migrated: `hypnose-helpers/pyproject.toml:9-10` now carries it verbatim,
and the dependency flows one way — this repo imports `hypnose_helpers` in 7 files and
nothing flows back.

Set **`requires-python = ">=3.12,<3.13"`** in `pyproject.toml:10` and
`environment.yml:6`. Not 3.10: `qc/fixtures/env.json` records `"python": "3.12.13"`, so
3.12 is the only version the gates have ever run on. Declaring an untested floor is how
`qc/figure_provenance.py:231` came to use a PEP 701 nested-quote f-string that cannot be
parsed by 3.9 — the single file of 102 that fails a 3.9 `ast.parse`.

Then delete the now-obsolete `pyproject.toml:11-13` comment and the `importlib.resources`
note at `pyproject.toml:47-49` (keep the `__init__.py` in `resources/device_schemas` —
it costs nothing).

**1b. Collapse the `[behavioral]` extra into `dependencies`.** The split exists for the
reason `README.md:42` gives (swc-aeon needs ≥3.11, somnotate is bound to 3.9), which 1a
retires. **Measured — a base-only install is already broken for its own entry point:**

| entry point | modules reached | needs `aeon` / `harp` |
|---|---|---|
| `hypnose_behavior.api` | 21 | **yes** (via `io.loaders`, `io.readers`) |
| `hypnose_behavior.accessors` | 20 | **yes** |
| `qc._common` (every real gate) | 33 | **yes** |
| `frames`, `io.paths`, `io.save` | 1–3 | no |

The only modules that survive base-only are the three the extras comment names as the
point of the split — and those helpers now live in `hypnose-helpers`. So the base install
serves nobody: `pip install -e .` today produces a package whose documented public
surface (`api.py`) cannot be imported.

Move `swc-aeon==0.1.0`, `harp-python`, `moviepy`, `opencv-python` into `dependencies`,
delete `[project.optional-dependencies]`, and update the four references:
`environment.yml:26`, `pyproject.toml:31-33`, `README.md:38`, `README.md:40-42`.

**1c. Declare the qc package data.** `pyproject.toml:47-52` declares only
`resources/device_schemas/*.yml`, but `regression.py:56,84` resolve `fixtures/` and
`sessions.yml` package-relative. Add:

```toml
"hypnose_behavior.qc" = ["fixtures/*.json", "sessions.yml"]
```

Latent today because everyone runs an editable install; confusing the first day someone
does not.

**1d. `qc/README.md` — the four undocumented tools.** The table at lines 9–17 lists 7 of
the 11 tools. Add rows for `ast_move_check.py`, `outcome_agreement.py`,
`position_data_lossless.py`, `verbose_diff.py`. Item 2 adds a fifth row.

**1e. `qc/regression.py:4-14` — the stale docstring.** It says "nine things" and
describes `non_initiated_sequences`, `non_initiated_FA` and `non_initiated_odor1_attempts`
as live tables "as written". It fingerprints **10** keys across **9** coverage sessions,
and those three are deletion guards per `_common.py:102-108` — deliberately kept
fingerprinting as md5-of-`"ABSENT"` so a retired table cannot silently return. Keep the
design; fix the prose, and name `non_initiated_attempts`, which the docstring omits
entirely.

**1f. Delete `device_schemas/` at the repo root.** Byte-identical to
`resources/device_schemas/` (md5 `16adb05d…` and `ba7d9236…`) and read by nothing —
`io/loaders.py:37` resolves the package copy via `importlib.resources`. Two identical
schema files is one file that can drift.

**1g. `README.md:89-113` — add `modelling/`** (9 files, 1,770 lines) to the structure
map. `debug/` is deliberately left out of this item; see "Deferred" below.

---

## Item 2 — `qc/check_layering.py`, written *before* the moves

**Done 2026-08-20.** `qc/check_layering.py`, `ast` only, zero pipeline imports — `qc/`
measures fan-in 0 with it in place. Baseline as it reports today: 102 modules (103 with
itself), 311 edges, module graph a DAG, and **three** cycles at directory granularity —
`io ↔ utils`, `io ↔ trial_classification`, and `io → trial_classification → utils → io`,
the third because `trial_classification/detect_trials.py:19` and `run.py:43` import
`utils.helpers` one-way. All three open when item 3 removes `utils/helpers.py:7`. Every
cycle prints with every edge; a declared edge is *marked*, never removed, and a `DECLARED`
entry that no longer matches the tree fails the gate. Simulated against items 3–5: one
cycle left, declared, RESULT PASS.

**Do the measurement first, as its own commit.** This is Item 1 of the previous plan's
technique, and it applies here: write the gate, watch it report today's two cycles as the
baseline, then let items 3 and 4 turn it green. A layering rule that arrives after the
moves proves nothing about them.

**Delivers.** A small AST tool, in the shape of `check_imports.py`, that walks
`src/hypnose_behavior`, builds the module→module edge set, and asserts two things:

1. **No cycles at directory granularity.** Today: `io ↔ utils` and
   `io ↔ trial_classification`.
2. **Declared exceptions only.** A short, explicit allow-list in the module, each entry
   carrying the reason — not a suppression, a decision (item 5).

Nothing checks layering today. `check_imports.py` resolves global names;
`ast_move_check.py` proves moves. Neither sees an import direction.

**Gate.** Itself, plus `check_imports`. It imports nothing from the pipeline, so `qc/`
stays a sink with fan-in 0.

**Trap.** Build it on `ast`, not on `importlib` — the point is to read the source without
importing the package, so it runs with no mount and no behavioural dependencies.

---

## Item 3 — the selection helpers move to `io/layout.py`

**Breaks `io ↔ utils`.** `utils/helpers.py:7` imports `filter_sessions`, `layout_for`,
`list_sessions` from `io.layout`, while `io/loaders.py:31`, `io/save_results.py:20` and
`io/tracking.py:24` import `utils.helpers`.

**Move** `_iter_subject_dirs` (`helpers.py:78`), `session_selectors` (`:87`),
`_filter_sessions` (`:106`) and `_filter_session_dirs` (`:139`) → `io/layout.py`. They
*are* layout knowledge, which is why they need the three names they import.

**What is left in `helpers.py`** — `CACHE`, `vprint`, `read_tracking_table`,
`find_tracking_file`, `_get_from_cache`, `_update_cache`, `clear_cache`,
`print_cache_keys` — imports `parameters` only. `utils/` becomes a true leaf tier, `io →
utils` survives one-way, and the selection helpers land next to where `results_dir()` goes
in item 6.

**Gate.** `ast_move_check` (this is a pure move) + `check_imports` + `check_layering`
(cycle 1 of 2 clears) + `regression`.

**Trap.** `session_selectors` has **119 references**. This is the one move with real
breadth; `ast_move_check` proves no body drifted, `check_imports` proves every call site
still resolves.

---

## Item 4 — `detect_settings.py` and `detect_stage.py` move to `io/`

**Removes 7 of the 9 `io ↔ trial_classification` edges.** Both files (377 and 254 lines)
import exactly one package module between them — `io.readers` — and nothing from
`trial_classification`. They parse settings and stage out of the raw tree. They are
readers, misfiled.

After the move, `io/loaders.py:25` becomes an intra-`io` import, and the five
`trial_classification` modules that use them (`run.py:21`, `params.py:18`,
`hidden_rule.py:16`, `aborted_trials.py:18`, `detect_trials.py:16`) point *downward*,
which is the correct direction.

**Gate.** `ast_move_check` + `check_imports` + `verify_scripts` (both files are reachable
from the CLIs) + `regression`.

**Trap.** Both are imported as modules (`import ... as detect_settings`), not by name, so
a missed rename fails at attribute access rather than at import. `check_imports`
disassembles function bodies and will catch it; a plain import test will not.

---

## Item 5 — the `outcome.py` exception: decide it, do not move it

**Decision: `outcome.py` stays in `trial_classification/`.** The earlier suggestion to
promote it to a package-root leaf beside `frames.py` and `parameters.py` is **withdrawn**.
It is an 84-line trial-classification rule with three sibling callers in that package;
the root tier is for things the whole repo reads, and `outcome.py` is not that.

**What makes the remaining edge safe is already designed and already documented** —
`outcome.py:5-8`:

> *"This module imports nothing from the package except the root-level leaves
> (`parameters.py`). That is what lets `io/save_results.py` reach it without turning
> `io -> trial_classification` into a cycle."*

That is why the module graph has zero cycles. After item 4, the only surviving upward
edge is `io/save_results.py:28` → `trial_classification.outcome`: **one import of one
documented leaf.**

**Delivers.** The property is currently a promise in prose that nothing checks. Convert
it into two assertions in `check_layering.py`:

1. `trial_classification.outcome` may import from the package **only** `parameters`
   (and `frames`), and
2. `io/` may import from `trial_classification` **only** `outcome`.

**Why not the third option.** Moving the derivation upstream so `io/` never reaches for
the rule was considered and rejected on measurement: `_derive_outcome` is applied at
`save_results.py:195`, inside `save_session_analysis_results` (lines 130–446), *after*
that function assembles `trial_df` by merging `aborted_sequences_detailed` and
`completed_sequences_with_response_times`. Moving the derivation means moving the
assembly — a refactor of a 316-line function, for one import. Not worth it; revisit only
if that function is being split for its own reasons.

**Gate.** `check_layering` goes green with two recorded exceptions.

---

## Item 6 — `results_dir()` / `table_path()`, and the 47 sites

**Delivers.** In `io/layout.py` (the module that already holds "the part only this repo
knows"):

```python
RESULTS_DIRNAME = "saved_analysis_results"

def results_dir(session) -> Path:
    """The analysis-output directory for a session dir or SessionRef."""
    return Path(getattr(session, "path", session)) / RESULTS_DIRNAME

def table_path(results_dir, name: str) -> Path:
    """Today: flat. The one place the layout of a results dir is decided."""
    return Path(results_dir) / name
```

Then repoint the 44 literal sites (26 files) and the 3 `qc/` glob sites
(`_common.py:284`, `outcome_agreement.py:124`, `verify_scripts.py:79`).

**`table_path` must exist from day one** even though it is a one-liner. It is the seam the
deferred subfolder split flips, and a seam introduced later is a seam that has to find its
call sites again.

**This takes the count from 44 to 2.** Item 7 is what stops the next path-shaped change
from re-scattering it.

**Gate.** `regression` + `plot_regression` (44 cases) + `verify_scripts`, all expected
GREEN because nothing about the output moves.

**Trap.** Three writers build the path independently — `io/save_results.py:121`,
`metric_analysis/run.py:334`, `metric_analysis/movement/speed_analysis.py:489`. All three
must go through the seam, or the subfolder flip writes to two layouts.

---

## Item 7 — retire `_filter_session_dirs`

**Delivers.** The 32 call sites take `SessionRef` from `_filter_sessions` and read
`ref.results` / `ref.date` instead of rebuilding both. This is the commit that pays: it
removes the 35 `exists()` guards and the 18 `_date-` splits along with the path rebuilds.

`SessionRef` lives in `hypnose_helpers` and is modality-agnostic — `saved_analysis_results`
is this repo's convention, so the accessor belongs in `hypnose_behavior/io/layout.py`
(item 6), **not** in the helpers repo.

**Gate.** `plot_regression` (44 cases) is the instrument — most call sites are plotters —
plus `regression` and `verify_scripts`.

**Trap.** `_filter_session_dirs` returns a list; `_filter_sessions` may be consumed once if
it becomes a generator. Several callers use `enumerate(ses_dirs, start=1)` and at least one
takes `len()`. Keep it returning a list.

---

## Item 8 — `prep.iter_sessions()` and the plotter preambles

**Do not split `visualization/` again.** Phase 10 already moved the boundary *between*
files (`3604 -> 4 modules`, `3efa16c`); doing it again redistributes 16,196 lines without
reducing anything. The 900-line `movement/traces.py:733` is not 900 lines of drawing — it
is a repeated ~15-line preamble wrapped around drawing.

**Delivers.** In `visualization/prep.py` (fan-in 12 — already the shared layer):

```python
def iter_sessions(subjid, dates=None, **select):
    """Yield one record per *analysed* session: .results_dir, .date_str, .views, .ref."""
```

On top of items 6 and 7 this collapses the
`_filter_session_dirs` → `results_dir` → `exists()` → `_load_trial_views` → `date_str`
quintuple that appears at all 32 sites. `_load_trial_views` is already single-sourced in
`io/loaders.py` and called from six `visualization` modules, so half the seam exists.

**Extract the preamble, never the plots.** The function-length numbers (42 functions over
150 lines, 72 over 100, of 912) are a consequence of one plotter = one function, which is
defensible for figure code. Let them shrink where this makes them shrink for free.

**Gate.** `plot_regression`, 44 cases — exactly the right instrument, because only what
*feeds* the drawing changes.

**Trap.** `prep.py` is imported by 12 modules and `prep._computed_metrics` is what makes
`plot_regression` reachable at all (section 34). A mistake here is invisible to
`regression.py`, which never sees a figure.

---

## Item 9 — one dispatch for a metric value

**Independent of items 1–8.** `metric_analysis/run.py:113` `metric_value` and the loop at
`run.py:376-384` are two spellings of the identical three-branch dispatch
(`fa_abortion_stats` special case → `session` wrapper + adapter → `spec.call`). This is
the section 5 shape — one quantity obtainable two ways — inside the module that defines
the rule. `run.py:128` records the reason as stdout: `metric_value` swallows it,
`run_all_metrics` needs it, because its loop's output *is*
`metrics_<subj>_<date>.txt`.

**Delivers.** One parameter dissolves the reason:

```python
def metric_value(spec, results, *, capture=True):
    ctx = contextlib.redirect_stdout(io.StringIO()) if capture else contextlib.nullcontext()
    with ctx:
        ...   # the one dispatch
```

`run_all_metrics`'s loop becomes `metric_value(spec, results, capture=False)` inside its
existing `redirect_stdout(buffer)`. **One file, ~15 lines.**

**Measured — no fixture churn.** `qc/_common.py:136` hashes the metrics *dict*, and no
gate reads `metrics_<subj>_<date>.txt` (no `.txt` reference anywhere in `qc/`). The `.txt`
is the only artefact at risk, and nothing watches it.

**Gate.** `regression` GREEN + `check_imports` PASS.

**Trap.** `nullcontext` still lets the wrapper print — that is the point — but the loop
must stay inside its own `redirect_stdout(buffer)`, or the `.txt` gains the lines that
used to be swallowed and loses the ones it had.

---

## Deferred, deliberately

- **Split `saved_analysis_results/` into subfolders.** Carried over from
  `followup_plan.md`. After item 6 this is **one file** — `table_path()` gains the
  name→subfolder mapping plus a flat fallback (section 2's rule, since every existing
  session is flat) — not the 44 sites it is today. **Do it before the server
  re-analysis**, or the tree is analysed twice. `movement/` cannot be done unilaterally:
  this repo writes no tracking file, so that folder needs the SLEAP repo to write into it.

- **The single-reward metrics are outside the registry.** `run.py:387-425` hardcodes the
  whole family, against 70 `@metric` / `@session_metric` registrations elsewhere. It is
  why `run_all_metrics` cannot simply *be* a loop over `REPORT`. Registering them is a
  real item, not a cleanup; noted, not scheduled.

- **`debug/`.** 511 lines, no `__init__.py`, imported by nothing, 395 of its lines
  tab-indented against a space-indented repo, absent from the README map, and recorded at
  `DECISIONS.md:1680` as deliberately unguarded. The user's call, later.

- **A test suite for the pure leaves.** Every gate but `check_qlearning.py` needs the
  server mount. `frames.py` (533 lines), `outcome.py` (84), `parameters.py` (51) and
  `io/protocol_schema.py` need no mount, and `hypnose-helpers` already has a pytest layer
  (`tests/test_layout.py`, `tests/test_provenance.py`) to mirror. `outcome.py` in
  particular is the one rule three call sites depend on (section 14). Item 2's
  `check_layering.py` is the first gate here that runs with no mount; a `tests/` directory
  is the natural next step.

- **The server has not been re-analysed.** Unchanged from `followup_plan.md`: every saved
  session predates the restructure, and the fixture sessions are the only ones current
  with this code.
