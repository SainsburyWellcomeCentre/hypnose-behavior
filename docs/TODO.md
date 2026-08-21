# TODO — hypnose-behavior

Work that is known, scoped and deliberately not scheduled. Each entry carries the
measurement that makes it actionable, so picking one up does not mean re-deriving it.

Settled rules live in `DECISIONS.md`; closed plans live in `archive/`. Addresses below
were re-measured on `8e5ee27`.

---

## The standing caveat

> **The server has not been re-analysed.** Every saved session on the derivatives tree
> predates the v2.0.0 restructure. The nine coverage sessions in
> `src/hypnose_behavior/qc/sessions.yml` are the only ones current with this code, and the
> gates re-derive those from rawdata rather than reading what is saved.

This outlives any one plan.

---

## Split `saved_analysis_results/` into subfolders

Carried over from `followup_plan.md`. After item 6 of the restructure this is **one file**
— `io/layout.table_path()` gains the name→subfolder mapping plus a flat fallback (section
2's rule, since every existing session is flat). **Do it before the server re-analysis**,
or the tree is analysed twice. `movement/` cannot be done unilaterally: this repo writes no
tracking file, so that folder needs the SLEAP repo to write into it.

**Four sites the mapping cannot reach, because they discover rather than look up.** A
name→path mapping answers "where does `trial_data.parquet` live"; it cannot answer "what is
in this directory", so each of these needs its own line at flip time:
`io/parquet_peek._parquet_files` (`parquet_peek.py:86`) lists a session's tables with a
non-recursive `glob("*.parquet")` at `:93` and needs `rglob` to see a nested one, and the
three `qc/` globs (`_common.py:285`, `outcome_agreement.py:125`, `verify_scripts.py:80`)
spell `{RESULTS_DIRNAME}/trial_data.csv` and break the moment `trial_data` moves down a
level. `rglob` is the right spelling for all four: it reads a flat session and a nested one
alike, which is also what a half-re-analysed tree needs.

---

## The single-reward metrics are outside the registry

`metric_analysis/run.py:395-431` hardcodes the whole family, against 70 `@metric` /
`@session_metric` registrations elsewhere. It is why `run_all_metrics` cannot simply *be* a
loop over `REPORT` — item 9 collapsed the registry dispatch to one call
(`DECISIONS.md` section 37) and this block is what remains beside it, inside the same
buffer. Registering them is a real item, not a cleanup; noted, not scheduled.

---

## `debug/`

`debug/debug.py`, 512 lines, no `__init__.py`, imported by nothing, 395 of its lines
tab-indented against a space-indented repo, absent from the README structure map, and
recorded at `DECISIONS.md:1680` as deliberately unguarded. The user's call, later.

---

## A test suite for the pure leaves

Every gate but `check_qlearning.py` needs the server mount. `frames.py` (533 lines),
`trial_classification/outcome.py` (84), `parameters.py` (51) and `io/protocol_schema.py`
(350) need no mount, and `hypnose-helpers` already has a pytest layer
(`tests/test_layout.py`, `tests/test_provenance.py`) to mirror. `outcome.py` in particular
is the one rule three call sites depend on (`DECISIONS.md` section 14). `qc/check_layering.py`
is the first gate here that runs with no mount; a `tests/` directory is the natural next
step.
