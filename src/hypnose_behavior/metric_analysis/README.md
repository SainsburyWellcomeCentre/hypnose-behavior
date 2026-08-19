# Metric Analysis (`src/hypnose_behavior/metric_analysis/`)

Behavioural metric calculation. Metrics are computed from the trial classification
output (`trial_data.parquet` + the saved tables/manifest/summary) and saved back into
the session's `saved_analysis_results/` directory as a `metrics_<sub>_<date>.json`
(values) + `.txt` (printed summary) pair, with merged results for multi-session runs.

## Running metrics

```
python scripts/run_metrics_analysis.py --subjids 59 --dates 20260618
```

- `load_session_results(subjid, date)` — load one session's `trial_data` + tables + manifest/summary.
- `run_all_metrics(results, save_txt=True, save_json=True)` — compute every metric for one
  session (returns a dict; optionally writes the json/txt).
- `batch_run_all_metrics_with_merge(...)` — run across any combination of subjids/dates
  (optional `protocol` filter), saving per-session results and a merged result per subject.

To add a metric, declare it where you define it — `@metric(frame=...)` on the pure
`f(frame) -> value` core, `@session_metric(core)` on the printing wrapper. Registering
makes it discoverable; adding its key to `run.REPORT` is the separate decision to save
it. See [`registry.py`](registry.py) and §4 of `docs/DECISIONS.md`.

## Single-reward outcome metrics

For single-reward ("singrew") sessions only, `run_all_metrics` adds two extra entries to
the metrics dict (and prints them to the `.txt`):

- `sing_rew_categories` — every trial sorted into a signal-detection outcome, with the
  trial count and the list of `global_trial_id`s per (sub)category (see
  [`sing_rew_metrics.py`](sing_rew_metrics.py)).
- `sing_rew_metrics` — the rates / sensitivity metrics derived from those counts
  (documented below).

A session counts as single-reward via the schema-derived `session.is_singrew` flag in
`manifest.json` / `summary.json` (True iff at least one candidate sequence is not rewarded
at its final position — the reliable source). Older sessions saved before that flag fall
back to a `stage_name` "singrew" substring match. Default-protocol sessions get neither
entry, so their output is unchanged.

### Outcome categories

Each trial is sorted into exactly one mutually-exclusive category from the singrew
trial_data columns (`is_aborted`, `response_time_category`, `reward_determinacy`,
`determined_final_odor`, `fa_label`, `false_response`, `fa_port`):

| Category | Rule | Subcategories |
| --- | --- | --- |
| **Hit** | completed & response_time_category ∈ {rewarded, unrewarded}; or aborted & reward_determinacy=rewarded & fa_label≠nFA | `rewarded_hit`, `port_error_hit`, `anticipatory_hit`, `anticipatory_port_error` |
| **Miss** | completed & response_time_category=timeout_delayed; or aborted & reward_determinacy=rewarded & fa_label=nFA | `omission_miss`, `forfeit_miss` |
| **False Alarm** | completed & false_response=True; or aborted & reward_determinacy=nonrewarded & fa_label≠nFA | `completed_fa`, `aborted_fa` |
| **Correct Rejection** | completed & false_response=False; or aborted & reward_determinacy=nonrewarded & fa_label=nFA | `passive_cr`, `active_cr` |
| **premature_port_entry** | aborted & reward_determinacy=ambiguous & fa_label≠nFA | — |
| **premature_abort** | aborted & reward_determinacy=ambiguous & fa_label=nFA | — |

Anything matching no rule is collected under `uncategorized`, and a `validation` block
flags any `global_trial_id` that is in no category or in more than one.

### Denominators

- `n_go` = hit + miss — rewarded-determined trials; go-side denominator.
- `n_nogo` = fa + cr — nonrewarded-determined trials; no-go-side denominator.
- `n_amb` = premature_port_entry + premature_abort — ambiguous aborts (status not knowable); excluded from d′.
- `n_det` = n_go + n_nogo — trials where reward status was determinable.
- `n_tot` = n_det + n_amb — all classified trials; denominator for premature rates.

### Tier 1 — discrimination / learning curve

- `hit_rate` = hit / n_go — P(respond | rewarded); responsiveness to rewarded sequences, rises with learning.
- `fa_rate` = fa / n_nogo — P(respond | non-rewarded); failure to suppress, falls with learning. The core suppression signal.
- `headline_sensitivity` (d′) = Φ⁻¹(H′) − Φ⁻¹(F′), with H′ = (hit+0.5)/(n_go+1) and F′ = (fa+0.5)/(n_nogo+1) — bias-free discrimination sensitivity; headline learning metric. 0 = chance, rises with learning.
- `criterion` = −½·(Φ⁻¹(H′) + Φ⁻¹(F′)) — response bias independent of sensitivity. >0 conservative (tends to withhold), <0 liberal (tends to respond). Separates "discriminates" from "just got cautious".
- `balanced_accuracy` = ½·(H + (1−F)) — single intuitive accuracy weighting both classes equally; robust to the 25/75 imbalance. 0.5 = chance, 1 = perfect.

### Tier 2 

- `earned_reward_rate` = rewarded_hit / n_go — fraction of rewarded trials that actually collected reward; the behavioral bottom line. Gap vs hit_rate = non-productive responding (wrong-port + anticipatory).
- `port_accuracy` = rewarded_hit / (rewarded_hit + port_error_hit) — A-vs-B identity discrimination among completed responded rewarded trials; separate axis from go/no-go.

### Tier 3 — strategy / efficiency

- `efficient_rejection_rate` = active_cr / n_nogo — P(bail early & withhold | non-rewarded); how often the animal uses the optimal early-abort rejection.
- `early_rejection_index` = active_cr / CR — of all correct rejections, the fraction achieved by early abort vs sitting through. Rises as the predict-and-bail strategy matures.
- `anticipatory_rate` = (anticipatory_hit + anticipatory_port_error) / n_go — rewarded trials where the animal aborted but still went to a port (jumped the gun). Knowledge-vs-policy signal; expected non-monotonic over learning.
- `forfeit_rate` = forfeit_miss / n_go — rewarded trials abandoned by aborting with no port entry; gave up a knowable reward (over-eager bailing).
- `omission_rate` = omission_miss / n_go — completed rewarded trials with no response; lapses/disengagement, distinct from forfeit.
- `impulsivity_rate` = premature_port_entry / n_tot — port entry before reward status was knowable; acting without information.
- `impatience_rate` = premature_abort / n_tot — leaving before status was knowable; bailing without information.

### Empty denominators

Every rate returns `NaN` (never a misleading 0) when its denominator is 0:

- `hit_rate`, `dprime`, `criterion`, `balanced_accuracy`, `earned_reward_rate`,
  `anticipatory_rate`, `forfeit_rate`, `omission_rate` → NaN when `n_go == 0`.
- `fa_rate`, `efficient_rejection_rate` → NaN when `n_nogo == 0`.
- `port_accuracy` → NaN when `rewarded_hit + port_error_hit == 0`.
- `early_rejection_index` → NaN when `CR == 0`.
- `impulsivity_rate`, `impatience_rate` → NaN when `n_tot == 0`.

The log-linear correction (the +0.5 / +1 in H′ and F′) keeps `dprime` / `criterion`
finite when hit or fa is 0 or maxed out, but it does **not** cover an empty go/no-go side
— `(hit+0.5)/(n_go+1)` is a valid 0.5 even at `n_go == 0` — so H′ and F′ are guarded to
NaN on `n_go == 0` / `n_nogo == 0` separately. H′ and F′ are persisted as their own
columns (`H_prime`, `F_prime`) alongside `dprime` / `criterion`.
