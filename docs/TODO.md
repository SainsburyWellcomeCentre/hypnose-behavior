# TODO — hypnose-behavior

Work that is known, scoped and deliberately not scheduled. Each entry carries the
measurement that makes it actionable, so picking one up does not mean re-deriving it.

Settled rules live in `DECISIONS.md`. Addresses below were re-measured on `8e5ee27`.


---

## Per-odour session regression raises on perfectly separated sessions

`log_regression.fit_session_models` raises when a kept session holds only correct or only
incorrect choices, which separates the logistic fit. Per odour that is common once an odour is
learned, so with `separate_odors` the four plotters built on it (`plot_gain_decomposition`,
`plot_edge_gains`, `plot_position_profile`, `plot_session_profiles`) raise on G/E and on A/B in
attempts mode. The pooled fits are not affected. Measured 2026-11-06, default `min_trials`, A/B
63 and 66 ses 2-10, 64 ses 2-14, 65 ses 2-11, G/E 63 from 20260930, 66 from 20260929, both to
20261021:

- A/B completed: none.
- A/B attempts: sub-064 ses-7, A (21/21 correct).
- G/E completed: sub-066 ses-54 (52/52) and ses-55 (56/56), G.
- G/E attempts: sub-063 ses-53 (38/38) and sub-066 ses-55 (102/102), G.

Raising `min_trials` above them, as the error suggests, drops most G/E sessions. Either leave
separated sessions out of the fit with a warning (the figures then lose exactly the perfect
sessions), or fit a penalised (Firth) logistic, which keeps them with finite estimates. A
modelling decision; `plot_regression` on the pooled cases gates it.

---

## Select sessions by protocol label, from a per-subject index

Every saved session carries its protocol labels (`is_hidden_rule`, `is_probe`,
`is_probe_hidden_rule`, `is_odour_discrimination`, `is_singrew`) in `manifest.session` and
`summary.session`, and its odours in `summary.params.hidden_rule_odors` / `probe_odors`
(`DECISIONS.md` section 40). The hidden-rule figures take their dates by hand, and the
lookups in `visualization/sampling.py:768`, `hidden_rule.py:535`, `false_alarm.py:995` and
`movement/traces.py:1307` each re-read a session's `summary.json` for `hidden_rule_odors`
alone, which cannot tell a probe session from a hidden-rule one: sub-045 20260203 and
sub-040 20251124 list A/B there.

- **One index per subject**, `session_labels.json` in the subject's derivatives directory:
  per `ses`, its date, the five labels, both odour lists, and the manifest's `created_at`.
- **Written on every save, checked on every read.** `save_session_analysis_results` writes
  its session's entry whenever it writes a manifest, at the first analysis and at every
  re-run, so a re-run cannot leave the index stale. Every read checks the index against one
  listing of the subject's derivatives directory. A date the index lacks (any session saved
  before the index existed) is read from that session's `summary.json` and added; a date no
  longer on disk is dropped. The index is then written back, so the next call reads one
  file per subject.
- **A cache, not a second source** (section 29). `summary.json` stays authoritative and
  deleting the index rebuilds it. A session without labels is unknown, not False
  (section 2): it is indexed as unknown, and a selection says how many it skipped.
- **Atomic writes** (temp file, then `os.replace`), since two runs of one subject can save at
  once. A read writes to the derivatives tree as well, which the gates'
  `HYPNOSE_DERIVATIVES_ROOT` redirect covers.
- **The index selects; the caller decides.** A `labels=` selection beside `iter_sessions` in
  `io/loaders.py`: a loader concern with callers outside `visualization/`, so not `prep.py`
  (section 36). Whether a probe session counts as a hidden-rule session is each caller's
  choice; for the switch-point model it does not. The `hidden_rule.py` plotters then get a `plot_hr_dates` flag that fills `dates`
  from it when none are passed, and the four lookups above read the index.

Gate: `regression.py` (the save path) and `plot_regression.py` (the plotters).

---

## Probe trials downstream of classification

On a probe-hidden-rule session the hidden-rule columns hold the C/F trials only. The probe
trials have their own columns (`hit_probe`, `probe_success`, `early_reward_type`; section
40), and nothing after classification reads them. Measured on sub-062 20261007, `main` at
`dbcff01`:

- **The switch-point model needs no change.** `modelling/switchpoint/data.py`
  (`prepare_subject`) keeps every completed trial and scores it SHORT on
  `hidden_rule_success`, so a probe trial is LONG, a probe success included: 14 of the 93
  completed trials, beside 10 SHORT. SHORT means the hidden rule was used. The model runs
  on hidden-rule and probe-hidden-rule sessions only, never on a probe-only session, where
  A/B are the hidden-rule odours.
- **`hidden_rule_mask`**: its non-HR group holds the probe trials.
- **The plotters** have not run on a probe-hidden-rule session, and `plot_regression.py` has
  none among its cases. Eleven modules outside classification read the hidden-rule columns
  or odours, with 27 of the reads in `metric_analysis/metrics/hidden_rule.py`.

Gate: `regression.py` for the metrics; `plot_regression.py` with sub-062 added to its cases.

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
