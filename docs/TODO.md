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

## Odours A and B are hardcoded downstream of classification

Classification detects odour discrimination from the schema, so G/E sessions (subjects 63
and 66 from 2026-09-29) classify as `odour_discrimination`. Much of what reads them still
assumes the two rewarded odours are A and B, or names the reward ports after them. Measured on
`main`, 2026-10-07:

- **Port letters stored as odour identities.** `classify_trials.py:429,438` tag port 1 `'A'`
  and port 2 `'B'`, and that letter is what `first_supply_odor_identity` /
  `first_reward_poke_odor_identity` hold: a G trial paid at port 1 reads `'A'`.
- **Plotters keyed on A/B.** `visualization/movement/speed.py:524-643`,
  `movement/traces.py:461-1250`, `movement/tortuosity.py:145,158`,
  `modelling/switchpoint/plots.py:65` and `data.py:90` (`_ab_label`), and
  `metric_analysis/metrics/hidden_rule.py:387`. `fa_port_ratio_by_odor_session` prints the
  ports as `A=` / `B=` in the metrics report.
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
   names and colours (`SEQUENCE_COLORS`, the G-C / G-F split) stay.

   **Skipped: the single-reward protocol**, which is not in use and is unlikely to get a G/E
   variant. If it does, its A/B code is: `metric_analysis/sing_rew_metrics.py:90-111`
   (`_odor_to_identity` / `_port_to_identity`, the anticipatory hit compares the final odour's
   letter with `fa_port` mapped 1 → A, 2 → B), `visualization/sing_rew.py:69-79,320-530`
   (`_port_label`, the `split_AB` boxplots, "Port A"/"Port B" legend), and
   `visualization/movement/sing_rew_movement.py:70-130,249-268` (`GROUP_*` keyed A/B,
   `_port_letter`, `_ab_letter`, which also reads the `*_odor_identity` columns).
4. **`ab_learning`**, on its branch after merging `main`. It selects runs on the per-run
   `protocol_mode == "odour_discrimination"`. The stage regex can go, because stage-1
   (`skipSampling`) runs are no longer analysed. `correct_port` comes from `reward_ports()`.

Hidden-rule odours also have a port (the A or B segment they are rewarded in). The hidden-rule
plots infer it from rewards today; the schema could give it the same way, later and separately.
