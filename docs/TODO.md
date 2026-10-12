# TODO — hypnose-behavior

Work that is known, scoped and deliberately not scheduled. Each entry carries the
measurement that makes it actionable, so picking one up does not mean re-deriving it.

Settled rules live in `DECISIONS.md`; closed plans live in `archive/`. Addresses below
were re-measured on `8e5ee27`.

---

## ~~`sleap-hypnose` still resolves the flat layout~~ — done 2026-08-26

`sleap_utils.py` spelled `session_dir / "saved_analysis_results"` in five places and
searched it non-recursively, so a migrated session's `movement_analysis/` files were
invisible to it. That file no longer exists: `sleap-hypnose` is now `hypnose-sleap`, a
package whose `io/layout.py` resolves both layouts with `rglob` — the same fix this repo
applied to `io/parquet_peek._parquet_files`.

The ordering constraint is lifted: SLEAP steps and migration can run in either order.
`hypnose-sleap`'s `qc/check_layout.py` asserts the two repos still agree on
`MOVEMENT_SUBFOLDER`, `RESULTS_DIRNAME` and `SUBJECT_PATTERN`, and calls this repo's
`find_tracking_file` against a file written at their `write_path`, so the agreement is
checked rather than assumed.

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

## Odours A and B are hardcoded downstream of classification

Classification detects odour discrimination from the schema, so G/E sessions (subjects 63
and 66 from 2026-09-29) classify as `odour_discrimination`. Much of what reads them still
assumes the two rewarded odours are A and B, or names the reward ports after them. Measured on
`main`, 2026-10-07:

- **Port letters stored as odour identities.** `classify_trials.py:429,438` tag port 1 `'A'`
  and port 2 `'B'`, and that letter is what `first_supply_odor_identity` /
  `first_reward_poke_odor_identity` hold: a G trial paid at port 1 reads `'A'`.
- **Still keyed on A/B.** `metric_analysis/metrics/hidden_rule.py:387`
  (`hr_odor_associations`, below).
- **The metrics report's FA port line.** `metric_analysis/metrics/false_alarm.py`,
  `fa_port_ratio_by_odor_session`, prints `OdorC: A=4, B=0, Bias ratio: 1.000`, naming the
  ports A/B. It only matters once a protocol with false alarms on non-rewarded odours (hidden
  rule, sequences) runs with G/E; odour discrimination has none, and no such protocol is planned.
  If one is: name the ports from `PortOdors` when the session has `reward_port_by_odor` saved,
  and keep `A=`/`B=` otherwise, because this printer runs inside the metrics pipeline, where a
  raise fails the whole session's metrics run (and pooled multi-session results may carry no
  runs). It changes `metric_analysis/`, so gate it with `qc/regression.py`.
- **`ab_learning`** (branch `ab-learning-detection`). `data.py` keeps runs whose stage name
  matches `odourdiscrimination…stageN` and scores `correct_port = port == odor` through
  `PORT_LETTERS = {1: "A", 2: "B"}`. G/E runs are therefore never loaded, and would score
  every port visit wrong if they were.

### What replaces it

Every run saves `reward_port_by_odor` in its `parameters` (`manifest.json` and `summary.json`,
`session.runs[]`), e.g. `{"OdorG": 1, "OdorE": 2}`. It comes from the schema: a reward
condition's `position` is its port, 0 → port 1 and 1 → port 2, in every protocol, checked
against the supply data on the A/B and G/E sessions. `io.load_results.reward_ports(results,
run_id=None)` and `Session.reward_ports()` read it. They raise for a session saved before it
existed, so re-run trial classification first. `reward_ports_by_run` gives every run's map
and `reward_port_of(ports_by_run, run_id, odor)` one trial's port, since a session can mix
protocols; `reward_ports_by_letter` collapses the runs to `{odor letter: port}`, and
`PortOdors` collects the odours each port pays over a figure's sessions for its labels. The
migration, one step at a time:

1. **Choice and correctness in port terms.** The choice is already a port number
   (`first_supply_port`, `first_reward_poke_port`, `fa_port`). The correct port is
   `reward_ports()[odor_name]`. Nothing new should read the `*_odor_identity` letter columns;
   they stay for now as stale duplicates of `*_port`.
2. **The port decides layout and colour.** Port 1 always goes up and port 2 always goes down
   (`plot_choice_history`), port positions are uniform in the movement plots, and each port
   keeps one colour, so A and G (both port 1) share it. Labels name a port after the odours it
   pays (`FA Ratio (G-E)/(G+E)`, `FA to port G`); no plot needs "(port 1)" in a label.
3. **The plotters and metrics listed above**, on `main`. Done: `plot_choice_history`,
   `plot_decision_accuracy_by_odor`, and the false-alarm family (`fa_port_number`,
   `fa_analysis`, the FA-ratio plotters in `false_alarm.py` and `hidden_rule.py`,
   `get_fa_ratio_a_stats`), the shared colour builder `prep._build_odor_colors`, sampling's
   `plot_poke_duration_by_odor`, `hidden_rule_and_false_alarm`, and the odour colours, order and
   rewarded-odour filter in `pred_seq_utils`. The predictive-sequence protocol's own sequence
   names and colours (`SEQUENCE_COLORS`, the G-C / G-F split) stay. The movement plotters
   (`movement/traces.py`, `speed.py`, `tortuosity.py`) are migrated in code only, never run.

   **Skipped: the single-reward protocol**, which is not in use and is unlikely to get a G/E
   variant. If it does, its A/B code is: `metric_analysis/sing_rew_metrics.py:90-111`
   (`_odor_to_identity` / `_port_to_identity`, the anticipatory hit compares the final odour's
   letter with `fa_port` mapped 1 → A, 2 → B), `visualization/sing_rew.py:69-79,320-530`
   (`_port_label`, the `split_AB` boxplots, "Port A"/"Port B" legend), and
   `visualization/movement/sing_rew_movement.py:70-130,249-268` (`GROUP_*` keyed A/B,
   `_port_letter`, `_ab_letter`, which also reads the `*_odor_identity` columns).

   **Skipped for now: the hidden-rule switch-point model.** `modelling/switchpoint/data.py:80-98`
   (`_ab_label`: each trial's "reward identity" from `first_supply_odor_identity`, falling back
   to `last_odor`, kept only if `"A"`/`"B"`; `AB_LETTERS`, `subset_by_ab`) and
   `visualization/modelling/switchpoint/plots.py:64-66,126-135,242-250` (`_AB_COLORS`,
   "reward A" labels). The migration would key trials on `first_supply_port` (else the port
   of the last odour), colour by port and label with `PortOdors`.
4. **`ab_learning`**, on its branch after merging `main`. It selects runs on the per-run
   `protocol_mode == "odour_discrimination"`. The stage regex can go, because stage-1
   (`skipSampling`) runs are no longer analysed. `correct_port` comes from `reward_ports()`.

Hidden-rule odours also have a port (the A or B segment they are rewarded in). The hidden-rule
plots infer it from rewards today; the schema could give it the same way, later and separately.
