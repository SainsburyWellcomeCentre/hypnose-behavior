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

## `sleap-hypnose` still resolves the flat layout

**The one piece of this change that lives in another repo.** `sleap_utils.py` spells
`session_dir / "saved_analysis_results"` in five places and searches it non-recursively:
`_find_tracking_files` globs `sleap_tracking_video*.{parquet,csv}` and `_find_combined_file`
globs `*_combined_sleap_tracking_timestamps.{parquet,csv}`, both directly in the results
directory, and `results_dir.glob("*.slp")` finds the raw predictions the same way.

After a session is migrated, all three of those are in `movement_analysis/` and none of
those globs sees them. `rglob` reads a flat session and a grouped one alike, which is the
same fix this repo applied to `io/parquet_peek._parquet_files`. Until it lands, run the
SLEAP steps for a session **before** migrating it, or migrate that repo first.

---

## `debug_sleap.csv` has no owner

Left at the top level by the migration because no rule claims it: it does not start with
`sleap_`, and nothing in this repo writes or reads it. Either it belongs in
`movement_analysis/` and the SLEAP repo should say so, or it is a stray debugging artefact
and can go. A decision, not a task.

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
