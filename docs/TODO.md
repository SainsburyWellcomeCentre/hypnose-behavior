# TODO — hypnose-behavior

Work that is known, scoped and deliberately not scheduled. Each entry carries the
measurement that makes it actionable, so picking one up does not mean re-deriving it.

Settled rules live in `DECISIONS.md`; closed plans live in `archive/`. Addresses below
were re-measured on `8e5ee27`.

---

## Before merging `ab-learning-detection`: per-odour mode and cohort cache

`6199a90` adds the cohort cache and the per-odour mode; the working tree (2026-10-22,
uncommitted) gives every discrimination-learning plotter the per-odour mode, adds
`DECISIONS.md` section 43 on the cache (this branch's section 40 is now 41, the merge slot
for main's 40) and the notebook cells. It also fixes two crashes: `plot_session_profiles`
raised on a bin's negative Wilson error bar (pooled G/E included), and
`sigmoid._surface_peaks` raised on a tied maximum (G's per-odour rate fit).

Passed: `check_imports`, `check_layering`; the cache check (cached = fresh exactly, stale
file / stale code / `refresh=True` rebuild); the separate-mode figures rendered and looked at
(A/B sub-064, G/E sub-063); the new notebook cells run headless.

- **Gates left** (`plot_regression`'s `_CHILD` harness with colours, A/B and G/E, about
  30 min): pooled figures against `ec676b6` on the final code; the per-odour plotters of
  `6199a90` against HEAD. A run before the last edits was 54 of 56 green; the 2 RED are
  `plot_session_profiles` on G/E, which raised at `ec676b6` and now draws: compare those
  against `ec676b6` with only the error-bar clip applied. State-space cases need
  `PYTENSOR_FLAGS="cxx=,mode=NUMBA"` on the Mac, where pytensor's C build fails (clang reads
  `-ld64` as `-l d64`). The 2026-10-19 selection (`ses_range (1, 14)` for all four) raises
  now on sub-063 ses-001; use each animal's A/B range.
- **Trials with no odour.** 234 A/B trials (sub-060/061/062, ses 11-12) have an empty `odor`
  and no reward port but are scored, so the pooled analyses count them and no single odour does
  (measured 2026-10-19, again 2026-10-22). List the sessions and trials, then decide whether
  the pooled view keeps them.
- **Then:** commit.

---

## Select hidden-rule sessions automatically

`summary.json` already carries `params.hidden_rule_odors` / `_positions` for every saved
session, pre-restructure ones included, so this needs no manifest flag and no
re-analysis — measured, it separated all nine fixture sessions correctly and found 29 of
sub-040's 48. Promote the `_session_hr_odors` closure in
`visualization/sampling.py:764` to `io/loaders.py` beside `iter_sessions`: it is a loader
concern, and five of the callers sit outside `visualization/`, so `prep.py` cannot hold
it (section 36). That also retires the copies in `visualization/hidden_rule.py` and
`movement/traces.py`, which each re-read `summary.json` for the same lookup. Then give
the `hidden_rule.py` plotters a `plot_hr_dates` flag that fills `dates` from it when no
dates are passed, so a hidden-rule figure stops needing the dates looked up by hand.
`plot_regression`-gated, since it reaches those 44 cases.

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

---

## Hidden-rule odours' ports from the schema

`metric_analysis/metrics/hidden_rule.py`'s `hr_odor_associations` learns which port each
hidden-rule odour pays (the A or B segment it is rewarded in) by voting over the subject's
hidden-rule successes, reading the `'A'`/`'B'` letter in `first_supply_odor_identity`, and
scans `summary.json` session by session until every odour is resolved. `visualization/prep.py`
uses it for the hidden-rule colours. The schema could give the port directly, the way it gives
`reward_port_by_odor` (`DECISIONS.md` section 42). Correct today, because hidden-rule sessions
reward A/B; needed once a hidden-rule protocol rewards other odours.
