# Phase 5 brief — the work list the Phase 4 audit left behind

Extracted **2026-08-07** from the Phase 4 metric audit, which is now closed, so that Phase 5
never has to open it. Everything load-bearing for `visualization/` is here; what remains there
is evidence for why the *metric* values are what they are, not a task list.

Line numbers are from the tree at **`cbc7059`** (4b complete) and were re-measured against it,
not copied from the audit — the audit's numbers predate ~1,500 lines of deletions.

Read with `docs/DECISIONS.md`. This file says *what to move*; that file says *what not to break*.

> **Status 2026-08-07 (`2ce3dcc`) — Phase 5 is CLOSED. This brief is now history.**
>
> **Two of its sections were measurably wrong about the code, and were not followed.** Read
> `DECISIONS.md` §13 before believing anything below about duplication.
>
> | section | state |
> |---|---|
> | 1 — FETCH, the metrics-JSON readers | ✅ **done**. `_ensure_metrics_json` deleted, all 6 call sites compute, trap discharged |
> | 1 — FETCH that genuinely relocates | ✅ **done** as part of the layering fix — the shared loaders now sit in `visualization/prep.py` |
> | 2 — PREP, the shared prep module | ⚠️ **premise false.** Every helper in the table had exactly **one** definition; nothing was duplicated. The real defect was 15 names imported *between sibling plotter modules*, fixed by `prep.py` + `panels.py` |
> | 2 — finding 10 (trajectory prep) | ⚠️ **premise false.** "Every row has a twin" is wrong for 5 of 7 rows: `_infer_port` variants disagree on **63.8%** of trials, `_last_poke_out` on 1.4%, and `_port_letter` has no twin. Two were genuinely shared; the rest were **renamed, not merged** |
> | 2 — finding 14 | ✅ **respected.** The `OdorG-C`/`OdorG-F` relabelling stayed a display relabelling, and `movement_analysis_utils._odor_letter` now wraps `frames.odor_letter` with only its `"Unknown"` label kept local |
> | 3 — DISPLAY-AGG, the 20 mean ± SEM sites | ✅ **done** — `primitives.mean_sem`, 18 sites. The remaining `.std(ddof=1)` sites are **SD, not SEM** |
> | 3 — the other DISPLAY-AGG patterns (cumsum, per-day means, violins) | ➖ **not done, and not proposed again.** Same test as `style_axis`: adjacent one-liners are not duplication |
> | 4 — rolling means | ✅ **done, differently.** `rolling_mean` fits none of the three sites; they share the *windowing*, so `primitives.rolling_windows` is the primitive. switchpoint keeps `np.convolve` (66% ULP divergence, ungated file) |
>
> Also done and not in this brief: both 4a defects, `position_data` laziness, `load_results_dir`,
> the six session selectors on all 44 session-selecting functions, and **three new gate cases**
> (32 → 35) covering the movement plotters this brief's section 2 sent us into unguarded.
>
> The ULP worry in section 3 **did not materialise** — the three SEM idioms are bit-identical on
> clean data; it is NaN handling that differs. See `DECISIONS.md` §12.

---

## 0. The destination rule — where the audit was overruled

**The audit proposed `visualization/io/metric_loader.py` for every `FETCH`. 4a rejected that
destination**, and the rejection generalises:

> `_load_tracking_and_behavior` went to **`io/tracking.py`**, not `visualization/io/`, because
> `metric_analysis.movement` consumes it and **`metric_analysis` must never import
> `visualization`**.

So the home is decided by the consumer, not by where the function currently sits:

| consumers | home |
|---|---|
| anything outside `visualization/` | top-level `io/` |
| `visualization/` only | a module under `visualization/` |

**Where 4a/4b deviated from the audit, the deviation wins.** The audit was written against
`f72d201`; two phases have landed since.

---

## 1. FETCH — but most of it is a deletion, not a move

**Read `docs/DECISIONS.md` → "Load vs compute" first.** `metrics_*.json` stops being a plotting
input, so the two functions the audit called "already the correct 4a shape" are the ones this
phase *removes*.

### The metrics-JSON readers — replace with registry computes

| function | site |
|---|---|
| `_extract_metric_value` — dot-path lookup into the metrics dict | `visualization_utils.py:199` |
| `_ensure_metrics_json` — reads `metrics_*.json`, else runs `run_all_metrics` | `visualization_utils.py:241` |

Six call sites, five plotters:

| plotter | line |
|---|---|
| `plot_behavior_metrics` | 406 |
| `hidden_rule_and_false_alarm` | 880 |
| `plot_decision_accuracy_by_odor` | 1215 |
| `plot_abortion_and_fa_rates` | 2035 |
| `plot_response_times_completed_vs_fa` | 2445, 2461 |

`movement_analysis_utils.py` imports both (`:39`, `:41`) with **no call site** — the import goes
with them.

`REGISTRY[name].call(results)` returns the same shape the saved key holds; that is what makes
this newly possible (4b). **`plot_abortion_and_fa_rates` carries a trap — see DECISIONS.md.**

### FETCH that genuinely relocates

| function | site | note |
|---|---|---|
| `load_tracking_with_behavior` | `visualization_utils.py:117` | tracking + behaviour join, `in_trial` labelling |
| `_load_protocol_from_summary` | `visualization_utils.py:228` | |
| `_session_hr_odors` | `visualization_utils.py:5156` | nested |
| `_collect_sessions`, `_load_trial_data`, `_load_sorted_session` | `pred_seq_utils.py:57, 75, 374` | |
| `parse_exp_ts_to_uk`, `_safe_concat`, `_apply_offset_and_localize`, `_slice`, `_load_register_files`, `_try_load` | `valve_poke_plots.py:74, 139, 151, 181, 190, 284` | all **nested inside** the one 620-line function — this is why that file is one function. Loader plumbing → `io/loaders.py` |

**Do not re-fix `valve_poke_plots._compute_real_time_offset:220`.** Audit finding 15 is closed:
4a moved the computation to `io/loaders.compute_real_time_offset` and what remains is a thin
wrapper that calls it. It looks like a duplicate and is not.

**Already the correct shape — leave alone:** the four `movement_analysis_utils` plotters that
read `speed_analysis.parquet` (`plot_epoch_speeds_by_condition`,
`plot_traces_with_speed_threshold`, `plot_tortuosity_lines_overlay`,
`plot_movement_analysis_statistics`), and `modelling/switchpoint/plots.py`, which consumes
fitted model artifacts and re-derives nothing. That file is the shape this phase is aiming for.

---

## 2. PREP — the shared prep module

Nothing here is metric math; all of it is duplicated across files, which is why it is a Phase 5
job rather than a Phase 4 one.

| block | sites | note |
|---|---|---|
| JSON / label parsing: `_parse_json_value`, `_normalize_date`, `_sequence_label`, `_sequence_len_ok`, `_normalize_odor_name`, `_last_position_entry`, `_extract_position_entry`, `_ordered_position_entries` | `pred_seq_utils.py:36-165` | shared in practice with `sing_rew.py` and `movement_analysis/sing_rew_movement.py` |
| ordering + colour: `_order_sequence_labels`, `_order_odor_labels`, `_darken`, `_resolve_color`, `_canonical_odor`, `_build_odor_filter`, `_ordered_groups` | `pred_seq_utils.py:295-390` | **`_ordered_groups:363` is also one of the two defects this phase fixes** |
| marker sizing / legend: `_count_to_marker_size`, `_nice_round`, `_add_size_legend` | `pred_seq_utils.py:168-236` | |
| `sing_rew.py` prep: `_normalize_fr_types`, `_port_label`, `_fr_mask`, `_trim_leading_empty`, `_subject_color_map`, `_pretty_metric`, `_isnan`, `_size_legend_handles`, `_trim_timeline_to_singrew` | | **`_fr_mask` stays** — it *selects* false-response trials, it computes nothing |

### Finding 10 — trajectory prep, duplicated 2-4× each

The worst duplication left in `visualization/`. Current sites:

| helper | `movement_analysis_utils.py` | `movement_analysis/sing_rew_movement.py` |
|---|---|---|
| `_odor_letter` | 866 (nested) | 84 |
| `_infer_port` | 872, 2062, 2566 (nested ×3) | — |
| `_smooth_tracking` | 916, 2128 (nested ×2) | 205 |
| `_extract_segment` | 929 (nested) | 215 |
| `_last_poke_out` | 940, 2029 (nested ×2) | 144 |
| `_resample_trace` | 964 (nested) | 228 |
| `_port_letter` | — | 76 |

**De-duplicate the two files in one pass, not two** — every row here has a twin.

`_odor_letter` is a fifth copy of the odor-token normaliser; the canonical one is
**`metric_analysis.frames.odor_letter`**, which 4a already put in place and
`visualization_utils` already imports (`:20`). Point these at it — but see finding 14 below
before assuming it is a drop-in.

### Finding 14 — context-dependent `OdorG` relabelling, written twice

| site | function |
|---|---|
| `pred_seq_utils.py:941-945` | `poke_time_all_pos` |
| `pred_seq_utils.py:1200-1204` | `fa_analysis` |

Both relabel `OdorG` by the odor that preceded it — `OdorG-C` / `OdorG-F` — because G means a
different thing in each sequence (`ODOR_ORDER`, `:336`, carries both labels).

**The trap:** this silently changes the grouping key of any per-odor quantity computed here
relative to one computed in `metric_analysis`, where `frames.odor_letter` knows nothing about
context. When it becomes a shared helper it must stay a **display** relabelling. Pushing it into
`metric_analysis` would give every per-odor metric a context-dependent key and change values.

---

## 3. DISPLAY-AGG — the primitives

**Standing rule from the audit, still in force:** taking the mean±SEM of a metric across the
subjects or sessions on a plot is a property of the **figure**, not of the data. These stay in
`visualization/`; they were never a Phase 4 move. They are the single biggest source of Phase 5
primitives.

### The cross-session/subject mean ± SEM, written longhand 20 times

Measured 2026-08-07:

| file | sites |
|---|---|
| `visualization_utils.py` | `.sem()` at 2201, 2232, 2261, 2291, 2321; `std(ddof=1)/sqrt(n)` at 2504, 4151, 4412, 4656, 6502, 6533 — **11** |
| `movement_analysis_utils.py` | 1536, 1537 (the trajectory band), 3150, 3183, 3216, 3249, 3282 — **7** |
| `pred_seq_utils.py` | 1316 |
| `sing_rew.py` | 332 |

One `mean_sem(values)` plus one band helper (`ax.fill_between` of mean ± SEM) replaces all of
them. Note the two idioms are **not identical** — `Series.sem()` and
`np.std(ddof=1)/sqrt(len)` agree in value but not necessarily in the last ULP, and
`movement_analysis_utils:1536` uses `nanstd` over an axis. Pick one, expect the gate to show
which drawn values moved, and look at them.

### The other DISPLAY-AGG patterns

- **`cumsum` panels** — `plot_cumulative_rewards_by_trial`, `_plot_metric_over_sessions`,
  `sing_rew._plot_cumulative_hit_cr`, `sing_rew._partition_total`.
- **per-day means of values the metrics produced** — throughout `visualization_utils`.
- **mean ± SD per violin** — `pred_seq_utils._plot_violins_with_stats:254`.
- **mean trace across resampled trajectories** —
  `movement_analysis_utils.plot_trial_traces_by_mode` (mean ± SEM with a normal-direction band)
  and `sing_rew_movement._plot_category` (`nanmean`).
- **min-max normalisation of session means** — `plot_movement_analysis_statistics`.

A closing 4a sweep read all **80** aggregation sites across the seven files and confirmed every
one that is not listed as a metric above is `DISPLAY-AGG`. That sweep is the reason this list
can be trusted as complete.

---

## 4. Rolling means — three implementations, one primitive

| implementation | site | form |
|---|---|---|
| `_rolling_median_iqr` | `visualization_utils.py:3303` | rolling median + IQR band, window/step |
| `_plot_summary_rolling` | `pred_seq_utils.py:445` | rolling mean over window/step |
| `_rolling_pts` | `pred_seq_utils.py:1609` (inside `_plot_performance_rolling:1590`) | rolling mean over window/step, but see below |
| `_rolling_mean` | `modelling/switchpoint/plots.py:95` | centred moving average via `np.convolve` |

Four sites, three shapes. The plan names `rolling_mean(series, window)` plus a SEM/CI band
helper as the target.

**The one that is not just a display primitive:** `_rolling_pts` carries the *windowing rule*
of audit finding 12. Rolling a **rate** is `sum(numerator)/sum(denominator)` over the window,
never the mean of per-trial values — the latter silently divides by the window size. That is
now `over_windows(decision_accuracy, …)` in `metric_analysis/resolvers.py`, and it must stay
there. A display primitive may roll *values*; it must not re-absorb the rate reduction. See
DECISIONS.md → "Store contributions, never a per-trial value".

**Not a stat primitive:** the six `X`/`Y` tracking smoothers
(`.rolling(window, center=True, min_periods=1).mean()` at `movement_analysis_utils.py:132-133,
227-228, 403-406, 925-926, 2131-2132` and `sing_rew_movement.py:210-211`). Those are `PREP` —
they belong with finding 10, not with the rolling primitive.

---

## 5. Already done in Phase 4 — do not redo

- Every metric `visualization/` used to compute now has exactly one definition in
  `metric_analysis`; **no metric math remains in `visualization/`**. The "lose no metric"
  checklist is closed.
- All four of finding 5's poke-time extractors are deleted, along with the dormant
  `poke_ms > 0` filter.
- `_compute_real_time_offset` → `io/loaders.compute_real_time_offset` (finding 15).
- `_kw_mwu_by_group` → `metric_analysis/stats/kw_mwu.py`; `_hr_odor_associations` →
  `metric_analysis/metrics/hidden_rule.hr_odor_associations`.
- Q5 definitions B and C are deleted: `plot_position_completion_rate` is the complement of
  `abortion_rate_positionX`, and `plot_false_alarm_rate_by_position` calls
  `fa_rate_by_position`. Both denominators are `reached_counts`.
- **The 10× outlier rule stays in the plotters** (`pred_seq_utils`, 2 sites) — judgement call 4:
  metrics raw, filtering is display. Do not move it into `metric_analysis`.
