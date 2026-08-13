# DECISIONS — settled rules and standing traps

**Read this at the start of every phase.** These are the decisions and traps that outlive the
phase that produced them: each one either prevents a silently wrong number or constrains a
choice a later phase would otherwise make freely. Each entry carries the measurement it rests
on, so you never need the narrative — that is in `git log`.

Nothing here is optional reading because the failure mode is silent. A deleted trap costs a
wrong result that no gate reports.

---

## 1. Metric shape

### D0 — four tiers, not one signature *(4a, 2026-08-05)*

Every metric is a pure `f(frame) -> value` core plus a thin `f(results)` wrapper. The core is
what the registry and the resolvers (`by_group`, `over_windows`) consume. But the signature is
**not uniform** — there are four tiers, and treating them as one is how a variant gets
mis-resolved:

| tier | shape |
|---|---|
| 1 — trial-reducible (13) | `f(trials)`; every resolver works |
| 2 — grouping key inside a JSON blob (8) | `f(trials)` + `f(position_data)` for per-position/odor grouping |
| 3 — normalised by a whole-frame quantity (3) | `f(trials, *, reference=None)`; no `reference` on a rolling call means each window normalises by *itself* |
| 4 — non-initiated trials (4) | **removed, not ported** — dropped from the metric set in 4a |

### Store numerator/denominator contributions, never a per-trial value

**The rule that is easiest to get wrong, and the reason two rolling accuracies disagreed for
years.** A rate is not a per-trial quantity:

```python
num = (rtc == "rewarded")                    # numerator contribution
den = rtc.isin(["rewarded", "unrewarded"])   # denominator contribution
value_over(sl) = num[sl].sum() / den[sl].sum()
```

Storing one number per trial and taking a rolling mean gives `rewarded / window_size` — a
denominator silently containing timeouts and aborts. Any consumer that collects a metric's
contributions must reduce them the way the metric does: `metrics.common.reduce_rate` is public
for exactly that. Mean-type metrics store `(value, included)` and reduce to `sum/count`.

### Three tier-2 traps that change values invisibly

1. **Summation style is part of the metric.** The two pooled `avg_sampling_time_*` metrics
   accumulate `total += x` left to right; `avg_sampling_time_odor_x` calls `np.mean`, which sums
   pairwise. They disagree in the last ULP over a few hundred values — enough to move the
   metrics md5. `_sequential_mean` exists to reproduce the first. **Do not tidy either onto
   `Series.mean()`**, and do not tidy `_mean_sd_by` back onto the pandas reductions (the same
   trap, resolved to `np.mean`/`np.std` so panels 1-4 of `plot_sampling_times_analysis` stay
   byte-identical).
2. **`last_event_index` vs `is_last_event`.** `avg_sampling_time_aborted_sequence` excludes the
   entry whose `index_in_trial` equals `last_event_index`. `presentations` also carries an
   `is_last_event` flag which **agrees on all 9 fixture sessions** — but it is a *different
   rule*, and the code reproduces today's values rather than a rule that happens to match.
3. **`np.int64` counts serialise to JSON as strings.** `json.dumps(..., default=str)` writes
   them as `"3"`, not `3`. Any count returned from a metric must be cast to `int`, or the
   fingerprint moves for a reason that looks like nothing.

---

## 2. `position_data` — filter on the provenance flag matching your blob

`build_position_data` (in `metric_analysis/frames.py`) builds one row per `trial × position`
from the **union** of `position_poke_times`, `presentations` and `position_valve_times` —
because the three do not carry the same positions:

- on a **completed** trial, `position_valve_times` holds every position with a valve activation,
  including ones whose poke registered as ~0 ms, which the other two and `num_odors` all drop;
- on an **aborted** trial, all three are restricted to positions with a poke.

So every row records which blobs it came from: **`in_poke_times` / `in_presentations` /
`in_valve_times`**.

> **Every per-position metric must filter on the flag matching the blob it reads today.**
> Without it, `manual_vs_auto_stop_preference` — which counts valve durations — gains the 0 ms
> positions and changes value.

`poke_source` is deliberately **not** synthesised. Its absence is how `sampled_positions` knows
to omit the `only_true_pokes` variants rather than return the unfiltered value. Treating "no
marker" as "all real pokes" would make old and new sessions look comparable when they are not.

---

## 3. `frames.py` must stay a leaf — and it was promoted to one *(Phase 7b.4, 2026-08-13)*

**`hypnose_behavior/frames.py`, at the package root.** It is a leaf below both `io/` and
`metric_analysis/`, and the rule that made it safe there is the rule that keeps it safe here:

> **`frames.py` imports nothing from the package** — only `json`, `re`, `warnings`, `typing`
> and `pandas`. Every package `__init__.py` is docstring-only, so importing a submodule
> triggers no package-level side effects.

**The day anything in the package is imported into `frames.py`, every layer standing on it
inherits that dependency.** Keep its imports to the standard library and pandas.

### Why it moved, and what the move retired

It previously lived at `metric_analysis/frames.py`, and `io/load_results.py` imported it —
an `io → metric_analysis` edge that was safe only because of the leaf property above, and
that had to be justified at length every time someone read it. Phase 7b.4 needed a *second*
importer: `io/save_results.py`, which writes the `position_data` side-table, and which
`trial_classification/run.py` imports. That would have given trial classification a
`metric_analysis` dependency — the thing `load_results.py`'s docstring said had been
deliberately avoided.

So the edge was not doubled, it was removed. **`io/` no longer imports `metric_analysis`
at all.** The one-way edge that §3 used to defend does not exist any more; what remains is
`io/` and `metric_analysis/` both standing on a root-level leaf, which is the shape this
section asked for in 2026-08-06's note: *"Promoting `frames.py` to a schema layer below both
is the honest fix — revisit only if it grows."* It grew.

**The whole file moved, not just `build_position_data`** — that same note gives the reason:
the builder shares four helpers (`parse_json_column`, `_as_int`, `_is_aborted`,
`_entries_by_position`) with `sequence_depth` / `reached_counts` / `sampled_positions`, so
splitting it out would have duplicated them or made the two halves import each other.

**Not `schema/`, and not `io/`.** `io/` is the wrong home because `build_position_data`
performs no I/O — the 0.2 "knows the data vs knows the layout" test, unchanged. And a
top-level `schema/` package would re-introduce exactly the ambiguity §20 avoided when it
named its module `protocol_schema.py` rather than `schema.py`, *because "schema" already
means the **task** schema in this code-base*.

**Verified as a pure move**: `qc/ast_move_check.py` reports **15/15 definitions
byte-identical**, and `git diff -M --stat` reports the rename with **0 insertions, 0
deletions**. The eight in-repo import sites were rewritten and `metric_analysis/frames.py`
deleted outright — no re-export shim, so nothing can keep importing the old path by
accident. `check_imports` PASS (63 modules), `regression` GREEN 9/9, no regeneration.

> **Two notebooks still import `hypnose_behavior.metric_analysis.frames`**
> (`notebooks/trial_classification/metrics_analysis.ipynb`,
> `notebooks/visualization/behavior_visualization.ipynb`) and will raise `ImportError`
> until their import line is updated. Deliberately left: nothing gates notebooks, and an
> `ImportError` is loud rather than silent.

---

## 4. The 4b registry contract

`@metric(frame=…)` on the core, `@session_metric(core)` on the printing wrapper. 43 registered,
25 reported.

- **The frame is a decorator argument, not a file boundary** (`frame="trials" |
  "position_data" | "trials+position_data"`). Grouping under `metrics/` is by behavioural
  construct, which is why `fa_latency_from_pokeout` sits with the false alarms and not with the
  other latencies. `MetricSpec.call(results)` is what makes the declaration load-bearing rather
  than decorative.
- **`run.REPORT` holds the report order explicitly**, *because* registration order would make it
  a function of import order — i.e. of a file layout that has already changed twice. Being in
  `REGISTRY` makes a metric discoverable; being in `REPORT` is the separate decision to save it.
- **Only `f(frame) -> value` is registrable.** `fa_port_ratio(n_a, n_b)` and
  `get_fa_ratio_a_stats(subjid, dates)` are deliberately absent — useful functions, not metrics
  over a frame.
- **Re-registering the same function is a reload, not a clash.** The notebooks run under
  `%autoreload 2`, which re-executes a module body on every edit; raising there would make the
  registry unusable exactly where metrics get written. A *different* function claiming a
  registered name still raises, as does an unknown `frame=`.

---

## 5. Load vs compute — `metrics_*.json` is not a plotting input *(2026-08-07)*

**Decision: plotters compute through the registry. `metrics_*.json` / `.txt` stay as the export
and the record of an analysis run.** This deletes the staleness problem rather than managing it:
no provenance stamp, no invalidation rule, no backfill, and no way for two plots to disagree.

Measured before deciding (warm, sub-040 20251124):

| path | per session |
|---|---|
| read `trial_data.parquet` (205 KB) | 6.7 ms |
| `build_position_data` → 1022 rows | 21.9 ms |
| **compute total** | **29 ms** |
| read `metrics_*.json` (17 KB) | 4.2 ms |

7× the time and 12× the bytes — but **both paths already paid the expensive part**. The caller's
walk over the mount happened either way, and that is what costs seconds (14.6 s for one
`derivatives.find_session` on a cold mount, against 0.2 s on rawdata). The cache saves one small
file read and 25 ms of CPU.

**Why the end state "save everything and only load" is unreachable:** three unreported metrics
take a `window` and two take an `fa_types` filter — properties of the *figure*, not the session.
Nine more return per-trial or per-poke tables (`poke_durations` is 739 rows for one session)
that belong with Phase 7b's `position_data` side-table. Only `false_response_ratio` is a
genuinely missing parameter-free scalar.

**Why a provenance stamp was rejected:** stamping the JSON with the commit that wrote it
invalidates the cache on every unrelated commit — a docstring fix would force a re-analysis of
the whole server. The correct key is a hash of the *metric definitions* plus an mtime check
against `trial_data.parquet`. That works, and it is machinery in service of a cache worth 25 ms.

**The real defect it fixes:** `decision_accuracy`, `avg_response_time` and
`FA_avg_response_times` are each obtained **both** ways in `visualization/` today. Two plots can
show the same quantity and disagree.

**Items:**

1. ~~**Phase 5** — route the three dual-sourced quantities through one path. Pick compute.~~
   **Done** 2026-08-07 (`d35638e`).
2. ~~**Phase 5 or 7b** — make `position_data` lazy, then convert the remaining JSON readers.~~
   **Done** 2026-08-07 (`c56107f`, `c3e21d6`). **No plotter reads `metrics_*.json`.**
3. **Phase 7b** — decide where the nine per-trial tables live; they ride with `position_data`.
4. **Anytime** — `false_response_ratio` into `run.REPORT` if it should be saved (new key on
   every session ⇒ `--generate` in its own commit).

### How a plotter computes a metric *(settled in Phase 5 — do not re-derive)*

`visualization._computed_metrics(results_dir, keys)` evaluates
**`spec.adapter(spec.session(results))`** — deliberately the *same expression* `run.py` uses to
build the file, so what a plotter computes and what would have been saved cannot drift apart by
construction. Three things make that the right expression and not an obvious simpler one:

- **The wrapper, not the bare core.** Several cores take session configuration as keywords —
  `hidden_rule_counts_by_odor` needs `hr_odors`/`hr_positions`, which `_extract_hr_config` digs
  out of `manifest`/`summary`. `spec.call(results)` raises `TypeError` for those. Knowing how to
  find that config *is* the wrapper's job.
- **The adapter matters.** `spec.call()` alone is **not** the saved shape: `decision_accuracy_by_odor`
  returns a DataFrame whose saved form is `to_dict()`, and several wrappers return a Series
  where the key has always held a dict.
- **`fa_abortion_stats` is special-cased in both.** It reports three tables rather than a value,
  so it fits neither `wrapper -> adapter` nor `call`. Both `run.py` and `_computed_metrics` call
  `_report_fa_abortion_stats`.

Wrappers print; `_computed_metrics` suppresses that. A plotter is asking for a value, not a report.

**`load_results_dir(results_dir)` exists because the lookup is the expensive half.** It is
`load_session_results` minus `derivatives.find_session` — 14.6 s cold against 29 ms to compute
every metric. Plotters have already walked the tree, so routing them through the subject/date
resolver would re-pay that walk per session to arrive where they started.

### The trap is discharged — and what it was hiding

The legacy `fa_abortion_stats` string reader was safe to drop the moment
`plot_abortion_and_fa_rates` stopped reading the file, which is now. `_fa_stat_count` /
`_fa_stat_rate` still understand both forms; **their string branches are now unreachable** and
may be deleted whenever convenient.

Converting that plotter exposed a **second, latent bug the JSON path was hiding**. The block
commented *"Legacy abortion rate per position (if fa_abortion_stats missing)"* had no such
guard, and appended a duplicate set of position rows on every session. That stayed invisible
only because JSON stringifies dict keys: `int("1.0")` raised, and a bare `except ... : continue`
swallowed it. Computing yields real float keys, `int(1.0)` succeeds, and five duplicate rows per
session appear. Now guarded on `have_position_rates`, which is what the comment always claimed.

> **The general lesson, worth more than the instance:** a bare `except: continue` around a
> parse turned a duplicated-data bug into silence. The JSON round-trip was load-bearing by
> accident. Expect more of these wherever a reader is replaced by a compute.

**Caveat:** switching a plotter from load to compute *can* move a curve, for any session whose
saved JSON predates a metric change. That is the staleness surfacing, which is the point — so it
is a `plot_regression`-gated change with a deliberate look at the diffs.

### The trap inside it

**`plot_abortion_and_fa_rates` reads both the numeric and the legacy string form of
`fa_abortion_stats`.** 4b made the metric numeric (counts `int`, rates `float`, positions `int`);
**every `metrics_*.json` on the server still holds the legacy `"3/10 (0.30)"` form**, and that
plotter reads those files directly.

> Tidying the legacy reader away before the tree is re-analysed makes that plot draw **nothing**
> for every session — silently, because the plotter skips what it cannot parse.

The reader may go only once the plotter no longer reads the JSON at all (item 1 above) or the
whole derivatives tree has been re-analysed. Not before, and not as a cleanup.

---

## 6. One truthiness rule

There is exactly one: **`metrics/common._is_truthy`**, widened in 4b so a string that parses as a
non-zero number is truthy. It previously accepted the float `1.0` and rejected its string form
`"1.0"`, while `hr_odor_associations` accepted both — a latent divergence reachable only through
the CSV fallback, where a float column renders `True` as `"1.0"` (measured: 0 disagreements on
all 9 fixture sessions, because both flag columns arrive as native `bool` through parquet).

**This matters the moment anyone adds a flag column.** Use `_is_truthy`; do not write a second
rule, and do not narrow it — widening strictly cannot lose a row, narrowing silently can.

---

## 7. Figures are gated by `qc/plot_regression.py`, not by `regression.py`

`regression.py` fingerprints `trial_data` + the metrics dict and **never sees a figure**. Every
change inside `visualization/` is invisible to it — a plotting refactor can be silently wrong and
stay GREEN.

`qc/plot_regression.py` (added 2026-08-06) runs 32 plotter cases under Agg against a git revision
*and* the working tree, then diffs every line's xy data, collection offsets, patch geometry, axis
decoration and **stdout**. Deliberately a two-tree diff, not a golden master: figures are meant to
change, and the question is always whether *this* change moved a curve.

What it sees that `regression.py` cannot, demonstrated rather than claimed: a `pd.concat` over a
variable a refactor had deleted, swallowed by a bare `except Exception: continue` — every session
would have returned an empty frame silently. And a metric value that is *printed*, not drawn.

Three properties to know before relying on it:

- **It resolves each case's function across an ordered `MODULES` list**, so *moving* a plotter is
  invisible to the diff while a change in what it draws is not. Add new plotter modules to
  `MODULES` or their cases become "not found", which reads as untestable, not as green.
- **It seeds the global RNG (`np.random.seed(0)`) before each call**, because several plotters
  jitter points and never seed it. It also pins `PYTHONHASHSEED=0` and applies `use_style("nature")`
  in the child. A "both raise, unchanged" case is an ungated one, not a green one.
  *(Phase 5 fixed the two defects the last two were hiding — see sections 11 and 12 — so they now
  guard against a regression rather than mask a live bug. The RNG seeding is not in that
  category: the jitter is genuinely unseeded, and **that also makes the diff sensitive to how
  many points are drawn**, since one extra point shifts every subsequent draw. A change in point
  count therefore shows up as dozens of "changed" values, not just as "added" ones.)*
- **Two non-zero diffs are accepted and recorded:** a sub-nanosecond recovery the trial-timing
  metrics inherit from `e9516e4` (max rel 2.2e-07) and one ULP choice in
  `plot_sampling_times_analysis`.

---

## 8. `session_index` selects; it does not position

`session_index` is on every `SessionRef`, gap-free, and is also a selector
(`find_sessions(62, index_range=(1, 9))` — "this animal's first nine sessions", comparable across
cohorts recorded months apart, which `ses` cannot express).

**Do not make it a plot x-axis, and do not "finish the retrofit" in Phase 5.** The 8 plotters
count `enumerate(ses_dirs, 1)` *within the filtered selection*, so every plot's x starts at 1
whichever sessions were requested. `session_index` is the animal's full-history rank, so a
filtered call would plot at x=12,27,33 with a mostly empty axis. The premise that gaps in `ses`
break the x-axis did not apply — no plotter ever used `ses` as x.

*Selection and positioning are different jobs.* Same distinction as `sequence_depth` vs
`sampled_positions`, and it fails the same silent way if merged.

---

## 9. A `save_figure` wrapper must pass `skip_modules=(__name__,)`

Provenance capture walks the stack and stops at the **first** non-helpers frame — which, for a
repo that wraps `hypnose_helpers.viz.save_figure`, is the wrapper itself. Capturing *inside* the
wrapper does not fix it; `capture_call()` still returns the wrapper's own frame. Both consumer
repos pass `skip_modules=(__name__,)`, and there is a regression test for it.

**Two things reintroduce the hazard:**

- **Phase 5's plotting primitives.** Once `plot_accuracy` calls `line(ax, …)` which calls
  `save_figure`, the primitive's module needs skipping too — or pass `provenance=` explicitly,
  which overrides introspection and is the robust form.
- **The proposed Phase 10 `visualization_utils.py` split.** Moving a plotter between modules
  changes `file` and `chain` in every saved figure's provenance record.

Related: `function` is only ever "the nearest frame we did not skip", frequently a local closure
(`movement_analysis_utils` has four nested `_save_fig` helpers). **Read `chain` before
`function`.**

---

## 10. The unpoked positions and `poke_source` *(Phase 6b, 2026-08-10)*

**Intended output change, fixtures regenerated 2026-08-10.**

Every position whose valve opened is now written to `position_poke_times` and `presentations`,
and every entry carries **`poke_source`**:

| value | meaning | `poke_time_ms` | count, 9 sessions |
|---|---|---|---|
| `poke` | a genuine poke inside the odor window | measured | 4620 |
| `grace` | no poke in the window; last poke-out within `PRE_ODOR_GRACE_MS` of the valve opening | `GRACE − gap`, synthetic | 91 |
| `outside_grace` | no poke in the window and no grace credit | `0.0`, null timestamps | 83 |

The marker is the only reliable separator: animals genuinely poke for under 20 ms, so duration
cannot distinguish a grace entry, and the tell `poke_first_in == poke_odor_start` is also
satisfied by a real poke already in progress when the valve opened.

> Consumers must treat an **absent** `poke_source` as "unknown" and omit the filtered variant,
> never as "all real pokes" — sessions saved before 6b will never carry it (section 2).
> `metrics.sampling._real_pokes` implements exactly that: it filters to `poke` when the column
> is present and returns the rows untouched when it is not, so a pre-6b session keeps the
> sampling averages it has always had.

### The brief said "write them into `num_odors`". Measured, that is wrong for aborted trials

All 83 `outside_grace` positions are the same situation: **port OUT at valve open, and not one
DIPort0 transition during the window** — the animal was demonstrably away from the port for the
entire presentation. They are not 0 ms *measurements*, they are absences. And they are not
valve-switching artefacts either: median valve duration 332 ms (range 28–736), only 5 of 83
shorter than the odor's required minimum sampling time.

**75 of the 83 sit on aborted trials, all trailing.** Crediting those to `odor_sequence` /
`num_odors` would put an odor the animal never smelled at the end of the sequence and make it
`last_odor`. `abortion_classification` — a wholly independent pipeline — agrees: its
`last_odor_position` is the last **poked** position on **74 of 74** non-null cases, and 0 of 74
at the new maximum.

So the rule is split by what the rig did, not by what the entry looks like:

- **completed trial** (reached AwaitReward): every presented position counts. The rig advanced
  through all of them, including a final one our DIPort0 reconstruction scores as unpoked.
- **aborted trial**: the sequence stops at the last `poke` (`_trim_unsampled_tail`). The
  trailing entries are still **written** — only what the trial is *credited* with shrinks.

`_trim_unsampled_tail` returns a prefix and deletes nothing, so `position_poke_times` and
`presentations` always hold the full presented record and `last_event_index` marks where the
counted sequence ends. **Interior gaps are never trimmed** — a later valve opening proves the
sequence moved past them.

The trim is gated on AwaitReward, so **no rewarded trial can lose a position**. Verified:
`is_aborted` changed on 0 trials, and every trial whose sequence shrank has null
`await_reward_time`, `total_supply_count`, `first_supply_time` and `total_reward_pokes`.

### What moved

`sub-057` trial 277 is the headline: `['OdorE','OdorB']` → `['OdorG','OdorE','OdorB']`, a
rewarded triple, so `sequence_rewarded` flips False→True and a trial scored `false_response`
becomes `rewarded`. `qc/outcome_agreement.py` goes from 1 conflict to **0** (section 14).

The column list in `regression.py` is wide because **a column's md5 moves if one cell does**:
that single trial changes scoring branch, and the false-response and standard-scoring branches
write different column families, so 16 columns move by one cell each. Cell-level diff against
HEAD, sub-057 (339 trials): `position_poke_times` / `presentations` all 339 (the new field);
`num_odors` / `odor_sequence` **10**; `reward_determinacy` 7; `last_odor` 5;
`sequence_rewarded` **2**; everything else **1**. `last_event_index` reads 339 but is a dtype
flip int64→float64 from a single new null — only 10 cells differ numerically.

**Trailing `grace` entries are trimmed too**, not just `outside_grace` — the test is
`poke_source != 'poke'`. That removes 11 entries that HEAD did write, and it is exactly the
`presentations`-vs-`last_odor_position` disagreement this section previously measured at 10 of
1731 trials. The two sources now agree. One aborted trial (`sub-057` 235) consequently drops
`sequence_rewarded` True→False; it was never rewarded (no supply event, no AwaitReward), so no
outcome moves. One trial (`sub-057` 332) trims to an *empty* sequence: both its positions were
grace entries rescued by 0.36 ms and 0.104 ms of credit. Its positions are still recorded.

### `presentations` now means "presented", and one denominator moved with it

`presentation_counts_by_odor` — the denominator of `odorx_abortion_rate` — counts
`in_presentations` rows, which now include a trailing position the animal never poked. That is
deliberate and it is why the metric moved on 5 sessions: the odor **was** presented, so it
belongs in a count of presentations, while the numerator is `last_odor_name` (the last odor
actually *sampled*), so such a position contributes a presentation and no abortion. It is not
filtered on `poke_source`.

The general rule this establishes: **`presentations` answers "what did the rig deliver",
`poke_source` answers "what did the animal sample".** A metric that means the first counts rows;
a metric that means the second filters (`metrics.sampling._real_pokes`). Read a per-position
metric and decide which of the two it is before changing it.

### The two position helpers still must stay separate

`sequence_depth` ("how far the sequence got") is **never** filtered; `sampled_positions` ("was
this position sampled") **is**. A single filtered `reached_positions` produces physically
impossible sets: dropping a non-`poke` entry from the middle of a trial credits it with reaching
position 5 but not position 3, making any per-position denominator non-monotonic. A gap is
meaningless for *reached* and perfectly natural for *sampled*.

`sequence_depth` still branches: an **aborted** trial reads `last_odor_position`, a completed one
reads `max(position_poke_times)`. Both are now correct for the same reason — the aborted blob's
counted prefix and `last_odor_position` agree — so the branch may finally be collapsed onto
`presentations`. That is a *separate* intended change with its own regeneration; it is no longer
blocked, but it is not done here.

---

## 11. Group draw order must come from an ordered container *(Phase 5, 2026-08-07)*

`prep._ordered_groups(group_keys, preferred)` draws `preferred`'s labels in their
canonical order, then **every other label in the order `group_keys` yields them**. Three callers
accumulated into a bare `set()`, so those residual labels were drawn in **string-hash order**,
which varies between processes: two runs of the identical tree disagreed on 340 drawn values.

It went unnoticed for a reason worth remembering: `preferred` (`SEQUENCE_ORDER`) lists only the
four *3-odor* sequences, so on sub-040's 5-odor protocol **every one of its ~125 series** took
the hash path. A "preferred list" that covers none of the live data is not a partial ordering,
it is no ordering.

- The three callers now accumulate **first-seen**, matching `_plot_performance_daily`, which fed
  an ordered dict and was already deterministic — so that one did not move.
- `_ordered_groups` **sorts** an input that is a `set`/`frozenset`. A set has no order to
  preserve, so sorting is the only deterministic thing left to do with one. That is a guard
  against the defect returning, not the normal path.

Verified a **pure permutation**: the sorted multiset of every drawn point is identical old vs new
across all four affected figures. `_order_sequence_labels` has the same shape but is only ever
fed dicts, so it was never at risk.

---

## 12. Display arithmetic: `visualization/primitives.py` *(Phase 5, 2026-08-07)*

Taking the mean ± SEM of a metric across the subjects or sessions on a plot is a property of the
**figure**, not of the data. It lives in `visualization/primitives.py` and never in
`metric_analysis`. `mean_sem` replaced 18 longhand sites.

**These may roll or average *values*; they must never re-absorb a rate reduction** — that is
section 1's rule, and `over_windows` owns it.

**The NaN rule, which is why one primitive was worth having.** Three idioms were in use, and
they are bit-identical on clean data — measured, **0 disagreements in 20,000 random samples** on
the pinned pandas 3.0.1 / numpy 1.26.4, so the "last ULP" concern the Phase 5 brief inherited
does *not* materialise. They diverge the moment a NaN appears:

| idiom | denominator with a NaN present |
|---|---|
| `Series.sem()` | count of **finite** values — correct |
| `Series.std(ddof=1) / sqrt(len(s))` | full length — **understates the error** |
| `np.std(v, ddof=1) / sqrt(len(v))` | propagates `nan` |

`mean_sem` drops non-finite values first, so `n` is what actually contributed. It returns `nan`
for fewer than two finite values, where SEM is undefined; a caller wanting a zero-height error
bar says `0.0 if np.isnan(sem) else sem` rather than having the primitive invent one.

**Two `plot_regression` obligations for anything added here.** Register a new plotter module in
the gate's `MODULES` list or its cases read as "not found" — untestable, not green. And a
primitive that calls `save_figure` needs `skip_modules=(__name__,)` or an explicit
`provenance=`; see section 9.

### The three shared modules, and what each is for *(Phase 5, 2026-08-07)*

| module | holds |
|---|---|
| `visualization/primitives.py` | display **arithmetic** — `mean_sem`, `sem_band`, `rolling_windows`, `rolling_mean` |
| `visualization/prep.py` | shared **non-drawing** code — trajectory prep, JSON/label parsing, colour and marker sizing, session collection, the figure-level loaders |
| `visualization/panels.py` | the four shared helpers that **draw** |

### What the rolling call sites actually shared

The brief expected `rolling_mean` to absorb three call sites. **It fits none of them**, and the
reason generalises: they differ in the statistic (mean vs median + IQR), in the **anchor** (the
window's last element vs its centre) and in the NaN policy. What they share is the *windowing*,
so that is the primitive — `rolling_windows(n, window, step, partial)`.

- `partial` names a divergence that was silent while each site coerced its own window size: for
  a series **shorter than one window**, `_rolling_median_iqr` clamped and emitted one window,
  `_plot_summary_rolling` emitted none. Both behaviours are preserved, now explicitly. The
  windowing was verified identical to both original loops over 3,332 `(n, window, step)`
  combinations.
- **`switchpoint/plots._rolling_mean` deliberately keeps `np.convolve`.** A windowed `np.mean`
  disagrees with it in the last ULP on **66% of values** at `window=21`, measured on the binary
  0/1 series it actually rolls. That file is in neither `MODULES` nor any case, so the drift
  would have been caught by nothing. Section 1's rule: summation style is part of the quantity.

### `style_axis` was dropped, deliberately

The plan proposed it on the measurement that axis decoration is the largest repetition in
`visualization/` (53 legends, 55 axis labels). Re-measured: **54 collapsible runs, ~130 lines →
~54**, and *no correctness payoff at all* — three adjacent calls setting three independent
strings are not duplication, they cannot drift, and `style_axis(ax, xlabel="Session")` is not
clearer than `ax.set_xlabel("Session")`. Contrast `mean_sem`, which was worth having because the
three idioms it replaced genuinely disagreed on NaN. **Counting lines is not finding duplication.**

Two blind spots found while looking for somewhere to put it:

- **The gate records no spine state.** It captures title, xlabel, ylabel, xlim, ylim and tick
  labels. Any spine change is ungated.
- **`nature_style()` / `poster_style()` already set `axes.spines.top/right = False`**, so the 13
  explicit `spines[...].set_visible(False)` pairs are redundant *once `use_style()` has been
  called* — and Phase 2a deliberately stopped calling it at import. Deleting them would make
  those figures depend on a style the process may never have applied: section 7's
  `_style_log_yaxis` fragility in reverse. **They stay.**

---

## 13. Finding 10 is not duplication — do not merge those helpers *(Phase 5, 2026-08-07)*

The Phase 4 audit's finding 10 called the trajectory helpers in `movement_analysis_utils` ↔
`movement_analysis/sing_rew_movement` "the worst duplication left in `visualization/`", listed
seven of them duplicated 2-4× and said **"every row has a twin"**. Measured over 3,149 trials
across 15 sessions of sub-040, that is false for all but two. They are **different rules wearing
the same name**, and merging them changes what is plotted.

| helper | copies | what differs | measured disagreement |
|---|---|---|---|
| `_infer_port` | 3, one per plotter | the column search list: 7 cols / 9 + odor-number + odor-label / 11 + supply + identity | **63.8%** of trials (A vs C); 0.3% (A vs B) |
| `_last_poke_out` | 3 | last-by-position + `sequence_start` fallback / scan-back / **max** `poke_odor_end`, no fallback | **1.4%** — 44 trials |
| `_extract_segment` | 2 | one drops NaN X/Y, requires ≥2 rows and `end > start`; the other does none of it | structural |
| `_odor_letter` | 2 | one strips the `Odor` prefix (any letter); the other resolves **A/B only**, None otherwise | different contracts |
| `_port_letter` | 1 | — | has no twin at all |

The `_infer_port` figure has a concrete cause: variant C searches `first_supply_port` /
`first_reward_poke_port` early, and those are populated on most completed trials where the other
variant's seven columns are empty. Merging changes trace colour and grouping on two thirds of
trials.

`sing_rew_movement._last_poke_out`'s docstring already argued *against* the other rule — "never
to `sequence_start` … we return NaT so the trace is skipped rather than started at the wrong
place." A merge would have silently overruled a decision documented at the site.

**What was genuinely shared, and is now in `visualization/prep.py`:** `resample_trace` (identical
bar a redundant guard) and `smooth_xy` (identical wherever the three copies work today, and it
also survives a duplicated `X`/`Y` column, which two of them would have raised on). Five
definitions became two.

**The rest were renamed rather than merged** — `_infer_port_from_response` /
`_infer_port_with_odor_fallback` / `_infer_port_with_supply_identity`,
`_last_poke_out_by_position` / `_last_poke_out_scanning_back`, and
`sing_rew_movement._ab_letter` (an A/B *side* resolver, which is not what `_odor_letter` reads
as). A same-named function that behaves differently is the trap; a differently-named one is
documentation.

`movement_analysis_utils._odor_letter` now wraps the canonical
`metric_analysis.frames.odor_letter` and keeps only the `"Unknown"` label for a missing odor.
Measured over every odor value in 15 sessions the two agree on all of them except NaN, and
`"Unknown"` vs `"NAN"` only ever reaches a label, never a branch. The relabelling stays in the
plotter — finding 14's rule.

> **The general lesson:** "these look alike" is a hypothesis, not a finding. Three of the four
> pairs above would pass casual inspection. Checking cost two measurement scripts; not checking
> would have cost a 64% change in port assignment on a plotter that **no gate case covered**.

**The gate is now 35 cases.** `plot_trial_traces_by_mode`, `plot_tortuosity_lines_overlay` and
`plot_category_traces` — the three consumers of all of the above — were in no case until this
work added them, so the de-duplication would otherwise have been verified by nothing.

### What "the shared prep module" was actually pointing at

Brief section 2's first table lists ~17 helpers as "shared in practice" between `pred_seq_utils`,
`sing_rew` and `sing_rew_movement`. **Every one has exactly one definition** — there was no
duplication to remove. The real defect was the direction of the dependency: **15 names were
reached by importing from a sibling plotter module**, so `sing_rew` depended on `pred_seq_utils`
for `_parse_json_value` and `movement_analysis_utils` on `visualization_utils` for
`_clean_graph`.

Fixed by moving them into two leaves (`prep.py`, `panels.py`), the same shape as section 3's
`frames.py`: the shared thing becomes a leaf and every plotter depends on the leaf rather than on
a peer. **Zero plotter-to-plotter imports remain**, and that is the invariant to preserve — it is
cheap to check with one grep and it is what keeps `visualization/` splittable in Phase 10.

---

## 14. The three outcome derivations agree on the rule; they disagree on the *sequence* *(Phase 6a, 2026-08-10)*

Phase 6's premise was that rewarded/unrewarded/timeout is derived three times by code that
"shares no code and can drift". Measured with `qc/outcome_agreement.py` over **1,731 trials on
all 9 regression sessions** (1,243 completed), the premise is half right: the three do differ,
but not where the brief expected.

| pair | jointly defined | conflicts | coverage gap |
|---|---|---|---|
| A `classify_trials` vs B `analyze_response_times` | 1,050 | **1** (0.10%) | A defines 193 that B leaves null |
| A `classify_trials` vs C `save_results._derive_outcome` | 1,079 | **0** | A defines 164 that C leaves null |
| B `analyze_response_times` vs C `_derive_outcome` | 1,049 | **0** | C defines 30 that B leaves null; B defines 1 that C leaves null |

**The outcome rule itself never conflicts.** Wherever two of the three name a category, they
name the same one — with a single exception, and that exception is not about the outcome rule.

### The one conflict is the 0 ms positions bug, not a rule difference

`sub-057 20260709`, trial 277. A says `false_response`, B says `rewarded`.

```
position 1: OdorG  valve 506.0 ms | poke NO POKE RECORD -> position dropped
position 2: OdorE  valve 571.0 ms | poke 45.5 ms
position 3: OdorB  valve 556.0 ms | poke 26.7 ms
odor_sequence written = ['OdorE', 'OdorB']        reward_determinacy = off_protocol
```

The schema's rewarded sequences are `('OdorC','OdorF','OdorA')` and `('OdorG','OdorE','OdorB')`.
The animal saw the second one. `classify_trials` drops a position whose poke registered as
0 ms, so it wrote a two-odor sequence, found it in neither candidate, set
`sequence_rewarded=False`, and scored a completed rewarded trial as a false response.
`analyze_response_times` assigns positions from *all* valve events in the window, so it saw the
full triple, and its `rewarded` is the correct answer.

`reward_determinacy = off_protocol` is the tell: a presented prefix that starts no candidate
sequence means the sequence was **truncated**, not that the animal ran a no-go trial.

> **This is `DECISIONS.md` section 10 / Phase 6b, reached from a different direction.** The two
> functions do not disagree about what a reward is. They disagree about what the animal smelled,
> and the one that drops 0 ms positions is wrong. **Fixing it belongs to 6b**, and 6b will
> remove this conflict rather than the merge doing so.

### What this licenses, and what it does not

**A pure `classify_completed_trial(record) -> outcome` is safe** — provided the record carries
`sequence_rewarded` as an *input*, as the Phase 6a brief specifies, rather than recomputing the
sequence. All three sites then keep feeding it whatever sequence they resolve today, and the
zero conflicts above are preserved byte-for-byte.

**Closing the coverage gaps is NOT a refactor.** A defines 193 outcomes that B leaves null (163
`false_response`, 17 `rewarded`, 10 `timeout`, 3 `unrewarded`), and C defines 30 that B leaves
null. Those nulls are deliberate: `analyze_response_times` emits a category only when it could
also compute a response time, and it counts the rest in `failed_calculations`. Pointing all
three at one helper and letting it fill the gaps would move ~190 trials out of the null bucket
and into the accuracy denominators — an intended output change that needs its own decision and
its own fixture regeneration. **Unify the rule; do not unify the coverage.**

### What was done, 2026-08-10

`trial_classification/outcome.py` — a leaf, `classify_completed_trial(supply_count,
reward_poke_count, has_await_reward, sequence_rewarded)` — and all three sites call it. Each
keeps what genuinely differs: **its windows** (the caller counts and passes counts), **its
sequence** (`sequence_rewarded` is an input, never recomputed here) and **its coverage** (the
response-time pass still emits a category only when it has a response time). `regression.py`
stayed GREEN through the merge, which is the evidence that the rule was reproduced exactly.

`latency_label` lives there too, replacing the three hand-written copies of the
in / out / late arithmetic in `_score_false_response`, `_false_alarm` and
`classify_noninitiated_FA`.

> **6b removes the one conflict, and this merge does not hide it.** Because
> `sequence_rewarded` is an input, fixing the 0 ms positions fixes trial 277 at its source and
> all three sites follow. Re-run `qc/outcome_agreement.py` after 6b: the expected result is
> **zero conflicts**, and anything else means 6b changed a rule it should not have.

---

## 15. The response-time anchor falls back to the last poke *before* the odor *(Phase 11, 2026-08-10)*

**Intended output change, 20 trial cells, fixtures regenerated 2026-08-10.**

`analyze_response_times` produced no response time for **20 of 1243** completed trials on the 9
regression sessions — 17 scored `rewarded`, 3 `unrewarded`. All 20 failed at the same guard, and
the cause is the 0 ms positions again (§10), pulling the opposite way from §14:

- `classify_trials` **drops** 0 ms positions → truncated sequence → trial 277 mis-scored (§14);
- `analyze_response_times` **keeps** them → it anchors the response on the last *valve event*,
  which on these 20 trials is an odor the animal never poked.

`last_poke_out_before` scans inside that valve window for an IN→OUT transition. On all 20 the cue
port was already OUT when the valve opened (`port_in_at_odor_start=False`), and on 17 there were
no DIPort0 samples in the window at all — so there is no exit to find, and the trial was dropped
before any window logic ran.

**The fix:** when the scan finds nothing, fall back to `windows.last_poke_end_before(...)` — the
animal's last cue-port exit *before* the odor opened, which is the moment it was actually free to
move. It is the same primitive the pre-odor grace path uses. It fires only where the scan returns
`None`, so the other 1040 response times are byte-identical.

| outcome | before | after |
|---|---|---|
| rewarded | 953 / 970 (98.2%) | **970 / 970 (100%)** |
| unrewarded | 86 / 89 (96.6%) | **89 / 89 (100%)** |
| timeout | 10 / 20 | 10 / 20 (unchanged) |
| false_response | 1 / 164 | 1 / 164 (unchanged) |

Regression moved on exactly 5 sessions, one column (`response_time_ms`) and one metric
(`avg_response_time`). **`response_time_category` did not move** — `_derive_outcome` was already
filling those categories, and the response-time pass now independently arrives at the same label,
which is §14's zero-conflict result observed live.

### Do not use `poke_odor_end` as the anchor

The obvious-looking `max(poke_odor_end)` over the sampled positions is **wrong**. On a
grace-derived entry `poke_odor_end` is synthetic — `last_poke_end + PRE_ODOR_GRACE_MS` — so it
sits up to 25 ms *after* the animal actually left. Measured on `sub-048 20260306` trial 14, where
the real exit was 5.2 ms before OdorB opened and the grace entry reported 19.8 ms of poke:

```
anchor = max(poke_odor_end)      -> response time 2230.8 ms
anchor = real last poke-out      -> response time 2255.8 ms      (+25.0 ms, the grace padding)
```

A flat 25 ms bias on every trial where the fallback fires.

### The remaining 173 nulls are correct, not gaps

- **163 `false_response`** — `analyze_response_times` deliberately skips single-reward no-go
  completions so they stay out of the rewarded/unrewarded/timeout denominators. Their latency is
  recorded as `fr_window_latency_ms`, off a **different anchor** (`await_reward_time`, not the last
  poke-out), so the two are not interchangeable.
- **10 `timeout`** — no reward-port poke anywhere, so there is no response to time.

And measured for completeness: `fr_window_latency_ms` is present on **101/101** trials where a false
response occurred and absent on all 63 `nFR`, exactly 1:1 with the `false_response` flag. The
abort side is identical (`fa_window_latency_ms` 69/69 on FA, 0/44 on nFA). **Absence there means the
animal correctly withheld, not that a measurement failed.**

### 6b did NOT retire this fallback *(measured 2026-08-10)*

The Phase 6b brief predicted that once positions carried `poke_source` the anchor "can be
chosen properly instead of by fallback". Measured after 6b landed, **the fallback still fires
on all 20 trials**. Nothing regressed — the prediction was simply wrong about where the work
lives.

**The 20 trials are not broken — they were fixed in Phase 11 and remain fixed.** The rescue has
nothing to do with `poke_source`: that code has never read it. It keys off the scan inside the
target odor's window returning `None`, which it still does. Only completed trials reach here at
all — `analyze_response_times` emits `rewarded`/`unrewarded`/`timeout`, so no false alarm and no
aborted trial ever touches this path.

What the brief actually wanted was tidiness: choose the target as the last position with
`poke_source == 'poke'`, and the primary scan succeeds on its own, leaving the rescue branch dead.
Same trials, same response times — reached by rule rather than by rescue — and the reported
`target` would name an odor the animal actually sampled.

**Recommendation: leave it.** The gain is cosmetic and the risk is not. Repointing the target
changes `last_poke_out_time`, which feeds `search_start = max(last_poke_out_time,
await_reward_time)`; the new scan window can run past AwaitReward and pick up a *later* cue exit
if the animal returned to the port, moving `search_start`, the reward pokes found, and possibly
an outcome. It would also require reconciling two deliberately different position rules —
`first_occurrence_positions` here vs `_assign_positions_to_valve_events` in `classify_trials`,
which disagree whenever an odor re-appears after a different one.

**Do not assume the fallback is dead because `poke_source` exists** — and do not treat its firing
as a defect.

### No bespoke marker for "ended on an unpoked odor"

Tempting, and rejected. It is `poke_source[target] != "poke"` once §10 lands, so storing it
creates two things that must agree forever (§13). It would also have to freeze one definition of
"the odor that ended the trial", and `max_position` is the wrong one — a hidden-rule early-leave
ends before the last position, and the response-time target is the last *valve event*, which can
be neither. Consumers derive it at the point of use with the definition they need.

---

## 16. Every reward latency exists twice: window-relative and movement *(Phase 11, 2026-08-10)*

**Intended output change, fixtures regenerated 2026-08-10.**

A reward-port latency answers one of two different questions, and the codebase was answering
them inconsistently — sometimes one, sometimes the other, sometimes neither:

- **(a) window-relative** — from where the response window starts, i.e. from where the rig
  starts its counter. *Did the animal respond in time?* This is what the labels bucket.
- **(b) movement** — from the animal's **last cue-port exit before the reward poke**.
  *How fast did it move, once it left?*

They differ whenever the animal returns to the cue port after the window opens — resampling the
odor, or checking whether another one is coming. Measured on the 9 regression sessions, that is
**57% of false responses, 43.9% of false alarms and 3.7% of completed trials**. Charging that
resampling time to the response inflates it; on one abort by 284 seconds.

| family | (a) window-relative | (b) movement |
|---|---|---|
| completed | `completed_window_latency_ms` | **`response_time_ms`** |
| aborted | `fa_window_latency_ms` | `fa_response_time_ms` |
| no-go | `fr_window_latency_ms` | `fr_response_time_ms` |

**Every (a) ends `_window_latency_ms`; every (b) is a `response_time`.** The names below were
settled by the Phase 6 rename (2026-08-12) — `fa_latency_ms` / `fr_latency_ms` were (a) under a
name that did not say so, and `fa_movement_latency_ms` / `fr_movement_latency_ms` were (b).

### Labels stay on (a), and therefore do not move

`fa_label` / `fr_label` keep bucketing the **window-relative** time, so they are byte-identical
to before. That is deliberate. Bucketing the movement latency instead was measured and rejected:
it relabels **95 of 401 false alarms (23.7%)**, all toward `_time_in` (`FA_time_in` 263→350,
`FA_time_out` 58→29, `FA_late` 80→22), and it destroys what `FA_late` means — an animal that
aborted, resampled for four minutes, then poked 500 ms after finally leaving would become a
*fast* false alarm. "How promptly did it respond after giving up" is the question `FA_late`
answers, and only (a) can answer it.

The same reasoning keeps `unrewarded` / `timeout` on (a): the outcome window still opens at
AwaitReward, so **no outcome flips and `decision_accuracy` does not move**.

### The naming was inconsistent; the rename settled it *(2026-08-12)*

`response_time_ms` is (b) while `fa_latency_ms` / `fr_latency_ms` were (a) under names that did
not say which they were. Making `response_time_ms` mean (a) would have been consistent the other
way, but it repoints ~1063 values and shifts `avg_response_time` by about a second, against 39
values for keeping it as (b) — so **the semantics stayed and only the names moved**:
`fa_latency_ms` → `fa_window_latency_ms`, `fr_latency_ms` → `fr_window_latency_ms`,
`fa_movement_latency_ms` → `fa_response_time_ms`, `fr_movement_latency_ms` →
`fr_response_time_ms`. No `trial_data` value changed; the fixture md5s moved because column
*names* are part of the canonical CSV. **`avg_response_time` reads (b).**

The registry keys did **not** move: `fa_latency_by_type` and `fa_latency_from_pokeout` keep
their names and simply read the renamed columns, because a registry key is a key in
`metrics_*.json` and renaming one is an output change beyond a column rename.

> **What the labels do and do not read.** `fa_label` / `fr_label` bucket **(a)**, via
> `outcome.latency_label`. The **completed** outcome buckets neither:
> `classify_completed_trial` takes no latency at all — `rewarded` is a supply delivery,
> `unrewarded` is a reward-port poke without one, `timeout` is neither. Section 16's earlier
> phrasing ("keeps `unrewarded`/`timeout` on (a)") meant the *window the counts are taken in*,
> not a latency comparison, and read as if a latency decided the label. It does not.

### Consequence for the anchor primitive

(b) is computed with `windows.last_poke_end_before(cue_series, reward_poke)`, never from
`poke_odor_end` — see §15 for why that is 25 ms late on a grace-derived entry. The existing
metric `metric_analysis/metrics/false_alarm.fa_latency_from_pokeout` was an attempt at (b) that
anchored on `poke_odor_end`, so it carried exactly that bias **and** did not exclude resampling.

**Repointed 2026-08-10** — it now reads `fa_response_time_ms` and its frame drops from
`trials+position_data` to `trials`. Measured on the two `fa_analysis` gate sessions the row
*membership is identical* (90/90 and 15/15), so this is purely a value change, and a large one:

| | median, sub-040 20251124 |
|---|---|
| old, from `poke_odor_end` | 11216.0 ms |
| `fa_window_latency_ms` — (a), from the abortion | 7821.7 ms |
| new — (b) | **1822.3 ms** |

The old value exceeded even the abortion-anchored latency, because the last odor's poke-out
precedes the abortion: it was charging every later cue-port visit to the false alarm.

`regression.py` stays GREEN — the metric is registered but **not in `REPORT`**, so it is not in
the metrics fingerprint. The gate that sees it is `plot_regression.py`, whose `fa_analysis` case
goes RED by design. **A metric outside `REPORT` is invisible to the golden master; check whether
it is in `REPORT` before assuming a metric change is gated.**

Consumers with no `fa_response_time_ms` column get an empty Series rather than a fallback to
the old computation, by the §2 rule: a session saved before Phase 11 cannot be made to look
comparable to one saved after it.

---

## 17. `trial_classification/` is three layers, and the workers must not import each other *(Phase 6c, 2026-08-11)*

`classification_utils.py` (3,138 lines) is **gone** — not turned into a facade of re-exports.
Its 57 definitions now sit in a three-layer DAG:

| layer | modules | rule |
|---|---|---|
| **leaves** | `windows.py`, `outcome.py`, `params.py`, `hidden_rule.py`, `index.py` | import nothing from `trial_classification` |
| **workers** | `detect_trials.py`, `classify_trials.py`, `response_times.py`, `aborted_trials.py` | import only leaves |
| **top** | `run.py` | imports workers and leaves |

> **The invariant: zero worker-to-worker imports.** It is the §13 rule that keeps
> `visualization/` splittable, applied here. The moment `response_times.py` imports from
> `classify_trials.py`, the two position rules below are one `Cmd-click` apart and the next
> person merges them.

This is why `_next_after`, `_recording_end` and `_odourdisc_reward_window_end` went to
`windows.py` rather than staying with `classify_trials`: they are shared by two workers, and
the shared thing becomes a leaf. `hidden_rule.py` exists for exactly the same reason.

`classify_and_analyze_with_response_times` went to `run.py`. It orchestrates five of the new
modules, so it belongs above all of them, and it is the per-run counterpart of `run.py`'s
per-session `analyze_session_multi_run_by_id_date`.

### The two position rules are now in adjacent files — they are still not the same rule

`classify_trials._assign_positions_to_valve_events` and `windows.first_occurrence_positions`
disagree whenever an odor re-appears **after a different odor**. On a trial presenting A, B, A:
the first gives **3** positions (it collapses only *consecutive* repeats), the second gives **2**
(each odor keeps its first position; the later repeat overwrites that position's event). The
do-not-merge note lives in `classify_trials.py`'s module docstring, at the site.

**The count that would settle whether they are interchangeable on this data has still not been
taken** — trials whose valve-event list contains a non-consecutive odor repeat, across the 9
regression sessions. Zero ⇒ a merge is cheap and §15's follow-up becomes tractable; non-zero ⇒
that is the number of trials a merge would silently move. It is a separate decision from the
split and was deliberately not acted on during it.

### `qc/ast_move_check.py` — keep it, and prove it before trusting it

Phase 4b's byte-identity pass was not kept, so 6c had to write it again. This one is kept and
generalised: `--base` reads the pre-split file from any git ref, `--old`/`--new-dir` point it at
any carve, so **Phase 10's `visualization_utils.py` split can use it as-is**. It reports
MISSING / DUPLICATED / CHANGED / ADDED, and flags a whitespace-only change as such.

It proves the one thing the fingerprint cannot: a body that drifted in a branch the nine fixture
sessions never take. 6c ran **57/57 byte-identical**.

> **A checker that has never been seen to fail is not evidence.** Before trusting it, it was run
> against a copy with an injected statement, a whitespace-only edit, a deleted function and a
> duplicated module, and confirmed to report each. Do the same to any gate you write.

### Two traps this split walked into

- **A dropped ceph mount makes `verbose_diff.py` pass vacuously.** `_capture_all` turns any
  exception into a one-line `<<VERBOSE_DIFF_ERROR>>` string, so both trees "agree" and it prints
  GREEN. **Read the line count**: a real session prints 464–3,083 lines, and 6c's honest run
  compared 16,944 across the nine. `1 lines` means the mount, not the code.
- **`SCHEMA_DIR` / `BEHAVIOR_SCHEMA_PATH` / `OLFACTOMETER_SCHEMA_PATH` were dead** in
  `classification_utils.py` — no function read them, and `valve_poke_plots` imports them from
  `io/loaders.py`, which defines them identically. They were dropped with the file rather than
  carried into a new module. `ast_move_check.py` reports uncarried module constants for exactly
  this review.

---

## 18. One position rule, and the experiment fault it exposed *(Phase 6 follow-up, 2026-08-12)*

**Intended output change: one trial, fixtures regenerated 2026-08-12.**

There were **three** position rules, not the two §13/§15 named. The third,
`aborted_trials._abort_positioned_events`, was uncatalogued and happened to match the *old*
`classify_trials` rule — so unifying only the two named ones left the codebase disagreeing with
itself on exactly the trial the unification was meant to fix.

Measured before merging, as §13 requires: they diverge **only** when an odor re-appears after a
*different* odor. That is **1 of 1,731** fixture trials and **0 of 46,112** trials across
subjects 056-066 (263 sessions, read-only scan of saved `trial_data`).

> **The single rule is `windows.positions_by_odor`: one position per odor, and a later
> activation overwrites the position that odor already holds.** All three sites use it.
> Duplicates are *rejected*, not given a new position, because within one trial the rig runs one
> sequence and the schemas never repeat an odor — a duplicate means several sampling runs were
> merged, and overwriting keeps the **last** run, which is the run the outcome events belong to.

### The one divergent trial is an experiment fault, not a long sequence

`sub-040 20251124` run 3 trial 44. `InitiationSequence` fired at 15:34:24.806 and **not again
until 15:36:02.720** — 97.9 s — and `detect_trials` ends a trial at the next initiation, so
three separate F→A sampling runs became one trial with valve events F,A,F,A,F,A: six collapsed
positions, truncated to five by `max_positions`.

**`detect_trials` is faithful to the rig and was not changed.** The task software emitted one
`ChooseRandomSequence` for the whole 98 s, i.e. it also considered this one trial. The fault is
upstream of the analysis. Fixing it at the *position* layer resolves the trial to `F,A` from the
last run, and needs no change to trial detection.

Two segmentation approaches were tried and are **wrong** — do not revisit them:

- **`odour_led` is not a run marker.** It is the inverse of valve activity and also goes ON
  mid-run whenever a poke is too short (15:34:41.534, with F reopening 200 ms later).
- **Splitting the *collapsed* odor list puts the boundary ~10 s early**, merging `F(46.315)`
  with `F(56.024)` because they are consecutive after collapse, which would give position 1 a
  ten-second valve window. Any such split must run on the **raw** event list.

`classify_trials` raises a `RuntimeWarning` naming the repeated odors whenever this fires. It is
silent on all sound data, which is what makes it worth reading.

### `sequence_depth` is the **credited** depth, and neither single-meaning form

- **completed** -> `max(presentations)`, unfiltered: the rig advanced through every position.
- **aborted** -> `max(poke_source == 'poke')`, falling back to `last_odor_position`, warning
  when neither resolves (2 such trials exist: `sub-057` 332, trimmed to empty, and `sub-040` 82,
  zero positions — both already contributed to no denominator).

Measured over 1,731 trials, the alternatives are not substitutable: unfiltered
`max(presentations)` everywhere **moves 84 trials** (re-crediting what the abort trim removed);
`max(poke_source=='poke')` everywhere **moves 32** (dropping a completed trial's trailing
presented position). This rule reproduces the previous branch exactly.

The fallback's 486-of-486 agreement with `last_odor_position` is only meaningful **because the
abort pipeline now numbers positions the same way**. Before this change it did not, and that
agreement silently broke on trial 44 while every other position column moved — caught by a
cell-level diff, invisible to an md5.

> **`_presented_max` reads `position_poke_times` before `presentations` on purpose.** On
> sessions this pipeline writes, the two carry the same positions (maxima agree 1,730/1,730), so
> the order cannot matter. It matters for pre-6b files where they did *not* agree (§2): reading
> `position_poke_times` first reproduces exactly what the old code returned.

`frames._last_position` now adds `+1` to `last_event_index` — it is a **0-based index** into
`presentations` while `last_odor_position` is a **1-based position**, and the two were being
returned interchangeably. A latent off-by-one that never fired, because `last_odor_position` is
a column on every session this pipeline writes.

---

## 19. The manifest provenance stamp is for audit, and lives in the manifest only *(Phase 7a, 2026-08-12)*

`manifest.json` carries `commit` and `version` alongside `created_at`, from
`hypnose_helpers.provenance.provenance()` — the same implementation that stamps saved
figures (Phase 2c), so the two cannot drift.

**What it is for: auditing.** "Which sessions were produced before commit X, and should I
re-run them?" It catches what a schema check cannot — Phase 6's close-out moved a *value* on
one trial while adding and removing no column at all, so comparing field sets would have been
silent on it. The two are complementary: **the stamp catches changed values, §20's field set
catches changed schema.**

> **This does not re-open §5.** §5 rejected provenance as a *metrics-cache key*, because a
> commit stamp invalidates on every unrelated commit — a docstring fix would force
> re-analysing the whole server. Same word, different job. Stamping to decide whether to
> **trust a cached metric** stays rejected, and plotters keep computing through the registry.

**Manifest only, and that is load-bearing.** The regression fingerprints `trial_data` and the
metrics dict; it never reads the manifest, so a per-run commit stamp cannot cause a spurious
RED. Anything that enters the fingerprint and changes per commit would make the golden master
useless. Verified: all five gates GREEN with no regeneration.

### Both `provenance()` arguments are passed explicitly, and neither is optional here

`provenance(anchor=__file__, call={"module": __name__})`, rather than letting it inspect the
calling frame. Two distinct silent-resolution failures, each of which produces a
plausible-looking wrong answer rather than an error:

- **`anchor`** — `hypnose-helpers` is installed as a library, so an anchor resolved there
  stamps every repo with the *helpers* commit. The same failure `io/paths` hit in Phase 2a and
  `io/layout` in 2b.
- **`call`** — captured from the frame, a call from a notebook resolves to `__main__`, and
  `package_version` returns `None`. Passing `__name__` also sidesteps the §9 wrapper hazard
  entirely, because nothing is being captured.

`package_version` maps the import package to the distribution via `packages_distributions()`;
the naive underscore-to-hyphen guess returns `None` here, since `hypnose_behavior` ships from
`hypnose-behavior-analysis`. Measured: `{'commit': 'b3f2497-dirty', 'version': '1.0.0'}`.

**Both keys are always written, even as `null`.** A reader can then tell "written before
provenance existed" (no key) from "written by code whose commit could not be resolved"
(`None`) — §2's rule about absent markers, applied to the manifest. Cached per process with
`lru_cache`: it shells out to `git`, and the answer describes the code as *imported*, which
cannot change while the process runs.

---

## 20. The protocol mode is part of the saved schema, and the impossible one raises *(Phase 7b, 2026-08-12)*

`io/protocol_schema.py` owns the vocabulary: `STANDARD`, `SINGLE_REWARD`,
`ODOUR_DISCRIMINATION`, and `resolve_mode(...)` which returns one of them.

**Named `protocol_schema.py`, not `schema.py`.** "Schema" already means the *task* schema in
this code-base (`schema_settings`, `*_SCHEMA_PATH`, `resources/device_schemas/`,
`sequence_schema`). A module named `schema.py` meaning the *output* schema is a collision
waiting to mislead.

**A leaf — standard library only.** `trial_classification -> io.protocol_schema` and
`io.save_results -> io.protocol_schema` both hold only because of that, exactly as with
`frames.py` (§3). Every layer imports it; the day it imports back, they become real cycles.

### Why the mode is in the schema at all

A session's columns depend on which branch of `classify_trials` ran, and the three branches
write different families. Measured over the 9 fixture sessions: one uniform record adds **26
columns**, 13-19 per session. One record per mode adds **7**, 1 per session for eight of the
nine. The mode is therefore part of what the file *is*, and is written to `manifest.json` so a
reader checks against the right field set instead of guessing from the columns present.

### `ConflictingProtocolError` raises, and that is the safe choice

The two flags come from **independent sources** — `is_odour_discrimination` from the stage's
protocol name, `is_single_reward` from the schema's `isSingleRewardProtocol`
(`trial_classification/params.py`). Nothing in the code makes them exclusive; the experiment
does, by construction: odour discrimination presents a sequence of length 1, single-reward
needs ≥2 positions for a sequence to be rewarded-or-not at its end.

> Raising beats warning **because `batch_analyze_sessions` already catches per session**. The
> broken session names itself, writes no derivative, and the batch completes. A warning does
> the opposite: it writes a `trial_data` whose schema is undefined — today's control flow hits
> the odour-discrimination branch first and `continue`s past the false-response scoring, so
> the four determinacy columns would be **silently absent from a file that still looked
> complete** — and buries the notice in thousands of lines of batch output.

This is **not** Phase 9. Phase 9 validates data *values*; this is a contradiction between two
schema-derived flags that leaves the output schema undecidable.

### The probe found two defects the gates could not

Per §17 ("a checker that has never been seen to fail is not evidence"), the guard was forced
to fire before being trusted. Both failures were invisible to every gate, because the gates
only ever exercise the path where it does *not* fire:

1. **A false pass.** Patching `classify_trials._get_single_reward_info` had no effect: `run.py`
   resolves it itself and passes `single_reward_info=` in, so `classify_trials` only calls its
   own copy when the argument is `None`. The flag stayed `False`, nothing conflicted, and the
   probe reported success.
2. **The message was swallowed where it mattered.** `run.py`'s per-run handler is
   `vprint`-gated, and `batch_analyze_sessions` defaults to `verbose=False`, so a structurally
   broken session surfaced only as `No runs analyzed for subject=... date=...`.
   `ConflictingProtocolError` is now re-raised ahead of that handler: the condition is
   session-level, every run shares the task schema, so skipping to the next run can only
   re-raise.

Verified end to end: raises through the real pipeline, **0** derivative files written for the
broken session, and the batch loop prints the reason and carries on.

---

## 21. `trial_data` has one declaration, and it is mode-dependent *(Phase 7b.1, 2026-08-12)*

**Intended output change: 8 columns, fixtures regenerated 2026-08-12.**

`classify_trials` builds a `@dataclass(slots=True)` record per trial instead of a free-form
dict. `slots` makes `rec.fr_laency_ms = ...` an `AttributeError` at the assignment site,
where the dict silently invented a column of NaNs.

### Three classes, because the modes are mutually exclusive

`StandardTrialRecord` (43 fields), `SingleRewardTrialRecord` (55) and
`OdourDiscriminationTrialRecord` (48); with the merged columns, 61 / 73 / 66. A session is
exactly one mode -- odour discrimination presents a sequence of length 1, single-reward
needs >=2 positions -- and §20 raises if both flags ever hold.

> **One uniform record would have added 26 columns, 13-19 per session. Per-mode adds 8, one
> per session for eight of the nine.** Measured over the 9 fixtures before writing any code.

`SingleRewardTrialRecord` extends `StandardTrialRecord`, not the base: a single-reward
session uses *both* scorers, so its rewarded trials write `poke_window_end`. Odour
discrimination does not extend it -- that column belongs to the standard scorer's fixed
response deadline, which the protocol has no equivalent of. `slots` then enforces the mode
boundary: an odour-discrimination record physically cannot be given `poke_window_end`.

The 8 added columns are all-null and irreducible -- they are *data*-determined, not
mode-determined, so no honest mode gates them: `fallback_reason` (7 sessions; written only
on a pending-attempt fallback), `abort_reason` (2), and six reward-poke columns on sub-056
alone, where no odour-discrimination trial ever scored `unrewarded` while sub-061, same
mode, carries all six.

### What the record does **not** declare, and why it matters

`run_id`, `is_aborted` and `global_trial_id` are `trial_data` columns assigned during
assembly (`ASSEMBLED_COLUMNS`). Declaring them looks harmless and is not:

- **`run_id` -> a phantom `run_id_original` column.** `merge._with_run_id` copies any
  *existing* `run_id` to `run_id_original` before overwriting. Emitting `run_id` from every
  record would give every merged session an all-null column that exists on no session today.
- **They must also be excluded from `save_results`' conform.** `run_id`'s fallback is guarded
  on the column being *absent*, so pre-creating it as NaN leaves it null; and
  `global_trial_id` would be appended at the end rather than inserted at the front.

### The `datetime64` -> `object` trap, which only multi-run sessions show

A record declares every field, so a column no trial writes is all-`None` -- and a column of
nothing but `None` carries no type, so pandas infers `object`. Harmless in one frame. At
**`merge`'s concat it is not**: one run whose column is entirely empty turns the whole merged
column `object`, and `to_csv` then writes `str(Timestamp)` (`...806000`) where `datetime64`
wrote `...806`.

Measured on `sub-040 20251124`, a three-run session: **154 cells of `await_reward_time` and
135 of `first_supply_time`** changed representation while every value stayed the same
instant. The three single-run sessions checked at that point were clean, which is exactly how
it hid.

`DATETIME_FIELDS` (15 fields, **measured from the reference tree's parquet dtypes**, not
assumed) is cast in `_frame` by `_as_declared_datetime`. It is deliberately **not**
`pd.to_datetime(..., errors="coerce")`, which would turn anything unparseable into `NaT` --
destroying a value to fix a dtype. It converts only when provably lossless (column is
`object`, holds nothing but Timestamps and nulls) and returns it untouched otherwise. This
*restores* the dtype the old code got by accident, and keeps the parquet holding real
timestamps rather than Python objects.

### Evidence

`regression.py` reported **only `+ added column` lines on all 9 sessions -- zero `~ changed`,
zero `- removed`** -- and all 9 **metrics md5s identical**. A per-column md5 moves if one cell
moves, so that is cell-complete: no value changed anywhere. Cell-level diffs on 5 sessions
(including both the worst case and the multi-run one) independently showed 0 of 59-72 shared
columns differing. `verbose_diff` 16,944 lines identical; `verify_scripts` green.

---

## 22. The loader checks a saved file against the schema, and says so when it cannot *(Phase 7b.2, 2026-08-12)*

`load_results_dir` compares a saved `trial_data`'s columns against the current declaration and
warns on anything missing. **The stamp catches changed values (§19); this catches a changed
schema.** A git SHA says *something* changed between the file and now, not whether *this file*
is affected -- a one-line plotter fix and a trial-classification restructure look identical to
it.

**The case it exists for is silent today.** Every derivative saved before Phase 6's latency
rename carries the old names, so `FA_avg_response_times` and `sing_rew`'s `FR_latency` find no
column, hit their `if col not in trials.columns` guard, and return empty -- a blank figure
with no error.

### An unknown mode is checked, not skipped

Files written before 7b carry no `protocol_mode`, and inferring one from the columns present
would be circular. But `_TrialRecordBase`'s fields are common to all three modes, and the
merged and assembled columns do not depend on mode at all, so `mode_independent_columns()`
(60 columns, verified a strict subset of all three declarations) can be compared with no risk
of a false alarm.

> **That is not the weaker check where it counts.** The renamed columns are *merged* ones,
> hence mode-independent. Measured on the server's `sub-040 20251124`: this reports
> `fa_window_latency_ms`, `fa_response_time_ms` and `completed_window_latency_ms`, while
> comparing against the record's own fields alone -- the form the plan originally
> sketched -- reports only `fallback_reason`, a column nothing reads. A check that looks
> like it ran and found something trivial is worse than none.

Emitted via `warnings`, not `print`, so it lands on stderr and cannot disturb the stdout that
`verbose_diff.py` and `plot_regression.py` compare.

**Proved on all four paths, not assumed (§17):** real pre-7b server files warn and name the
three columns; a session written by the current code records `protocol_mode` and warns *not at
all*; dropping two columns from a *tagged* file names exactly those two; an unrecognised mode
reports that the schema was not checked.

### Trap: `plot_regression`'s banner is not its result -- count the cases

A dropped mount makes `plot_regression` print **`GREEN: 35 plotters draw identically`** off a
run that only attempted **31**. The sessions are not found, the cases are skipped, both trees
draw nothing, and the diff is empty because both sides are broken -- §18's "a two-tree diff
cannot see both sides being equally broken", in the gate §17 documented only for
`verbose_diff`.

> **Read the case count, the way `verbose_diff` needs its line count read.** A full run lists
> **35**. The four that vanish first are the movement plotters
> (`plot_epoch_speeds_by_condition`, `plot_tortuosity_lines_overlay`,
> `plot_traces_with_speed_threshold`, `plot_trial_traces_by_mode`), because they need tracking
> data. Caught here only by comparing against an earlier run's listing.

Corollary for `regression.py`: a mount failure shows as `[ERROR] ... No experiment runs found`,
which is counted in the same total as a real `[RED]` mismatch. **`REGRESSION RED: 9
mismatch(es)` with no `[RED]` lines above it is a mount, not a regression.** Never filter gate
output -- grepping for `RED` hides `[ERROR]`, and grepping for the summary hides the
`+/-/~` lines that say what actually moved.

---

## 23. Parquet is the format; CSV is a convenience, off by default *(Phase 7b.3, 2026-08-12)*

`save_session_analysis_results(..., save_csv=False)`. Parquet round-trips dtypes and blobs; a
second copy of every table is worth writing only when someone will read it by eye.

### The flag could not gate CSV without parquet for *every* table

`loaders._load_table_with_trial_data` could load the three `non_initiated_*` tables from **CSV
only** -- no parquet was ever written for them. Gating CSV alone would have made them return
an **empty frame with no error** for every session saved with the default: not "unreadable by
eye", but unreadable.

So parquet is now written for every table, the reader prefers parquet and falls back to CSV,
and `save_csv` is purely additive. Verified both ways on `sub-053 20260520`: with
`save_csv=False` **no CSV exists at all**, and `trial_data` / `non_initiated_sequences` /
`non_initiated_FA` still load 195 / 1 / 1 rows -- identical to `save_csv=True`. A CSV-only
reader could not have returned those, so the fallback is not dead code.

The `.schema.json` sidecar follows the **CSV**, not the parquet: it records which object
columns were JSON-encoded to survive flat text, which parquet does not need.

### Every QC entry point asks for CSV explicitly

`qc/_common.fingerprint_session`, `verify_scripts` (for both `run_trial_classification.py` and
`batch_process.py`, via `--save-csv`) and `outcome_agreement.py` all pass it, because all three
read `trial_data.csv` directly -- `_common` fingerprints the *canonical CSV*.

> **Never rely on the default in the harness.** The gate would then change meaning whenever
> the default did, and the failure is not a mismatch but a `FileNotFoundError` on a file
> nobody decided to stop writing.

---

## 24. `position_data` is a written table, and it is a *projection* of the blobs *(Phase 7b.4a, 2026-08-13)*

`position_data.parquet` is written beside `trial_data.parquet`: one row per
`trial x position`, with `poke_source` and section 2's three provenance flags
(`in_poke_times` / `in_presentations` / `in_valve_times`) as real typed columns rather than
a table smuggled into a JSON string inside a cell.

**Additive, deliberately.** The three blobs stay in `trial_data` and `load_results_dir`
still derives the frame; 7b.4b makes the loader read the file and only then drops them. So
7b.4a moves no existing column and no metric value -- `regression` GREEN 9/9, no
regeneration.

### It is built from the in-memory frame, and that is the point

The blobs are JSON-encoded on their way into `trial_data.parquet`, so a frame derived at
*load* time gets its five timestamp fields back as **ISO strings**. Built here, before
serialisation, they are real `datetime64[ns]`. That is the entire "typed columns" win, and
it is why the side-table is worth writing rather than just being a cached derivation.

**Measured, not assumed, across all nine fixture sessions (4,791 position rows):** the
load-derived and file-written frames agree on **every cell of every column** once parsed,
and differ **only** in the dtype of `poke_odor_start`, `poke_odor_end`, `poke_first_in`,
`valve_start` and `valve_end`. No value moves and no precision changes -- the same instants,
differently typed. All 14 evaluable `position_data` metrics return identical results either
way, because each reads those columns through `metrics.common._tz_naive`, i.e.
`pd.to_datetime`. (The 15th, `hidden_rule_counts_by_odor`, needs its wrapper for
`hr_odors`/`hr_positions` (section 5) so `spec.call` raises; it reads none of the five
columns, so it is *unmeasured by that probe* rather than verified.)

> **Neither `regression.py` nor any other gate watches this file.**
> `_common.fingerprint_session` fingerprints the canonical CSV of `trial_data` and the
> metrics dict -- `position_data` is neither, and the four metrics that touch its timestamp
> columns (`hr_abort_poke_gap`, `reward_delivery_latency`, `trial_poke_span`,
> `valve_to_reward_latency`) are all **unreported**, so they are not in the metrics dict
> either. 7b.4a goes GREEN because it is additive, *not* because anything checked the new
> table. Gating it means adding a third md5 to `_common`, as its own commit with its own
> `--generate`.

### The projection is lossless for every field anything reads -- with one exception

`build_position_data` is a **union with merge**, not an encoding: poke fields take
`position_poke_times` and fall back to `presentations`, valve fields take
`position_valve_times` and fall back to `presentations`, and any key outside its field lists
is dropped. So "the side-table is byte-identical to the blobs" is a sentence that cannot be
true, and 7b.4b's control must be an **inventory**, not a hash.

Measured exhaustively over the nine sessions -- every trial, every blob, every position,
every key -- **25 of 26 `(blob, key)` pairs are carried with an equal value on all 4,791
occurrences**, and `differs` is **zero everywhere**: the merge precedence never actually
discards a value, because the blobs agree on every shared field. The lossiness is by field
selection alone.

The one exception is **`position_valve_times.prior_presentations`** (1,730 occurrences, all
absent). It is written into position 1's valve entry by `classify_trials` and read exactly
once, *in memory*, by `classify_trials` itself, to build `non_initiated_odor1_attempts` --
which `save_results` already persists as its own table. Nothing reads it off a saved
`trial_data`. It also would not fit this table's grain: it is one row per *failed
position-1 attempt*, not per `trial x position`.

> So the honest statement, and the one 7b.4b may rely on: **`position_data` carries every
> blob field that any reader consumes.** The single dropped key is redundant with an
> existing saved table.

### What this does not license

Dropping the blobs. Measured separately: **15 live reads across 7 modules** read them off a
*loaded* `trial_data`, and `frames.sequence_depth` reads them off a **trial row** to feed
`reached_counts` -- the per-position denominator of `abortion_rate_positionX` and
`fa_abortion_stats`, both **reported** metrics. That work is 7b.4b, and it is sequenced
after 7b.6 because `plot_regression` cannot see the switch until the server carries both
representations: before the re-analysis no saved file has a `position_data.parquet`, so the
fallback fires on both trees and a green diff would mean nothing.

---

## 25. The per-grain metric tables, and the grain that had to be created *(Phase 7b.5, 2026-08-13)*

`run_all_metrics` writes two files beside `metrics_<subj>_<date>.json`, named by **grain**
because the table-returning metrics do not share one:

| file | grain | contents |
|---|---|---|
| `metrics_by_trial.parquet` | `global_trial_id` | 6 per-trial Series + `hr_abort_poke_gap`'s 3 value columns = 10 columns |
| `metrics_by_poke.parquet` | `global_trial_id` + position | `poke_durations` for **both** outcome classes, with an `aborted` column |

Session-level metrics stay in the JSON: ~25 scalars plus numerator/denominator
contributions, which is metadata-shaped, and parquet buys nothing for a flat dict.

> **An export and a record, never an input** (section 5). Nothing reads these back, and no
> plotter may be "optimised" by doing so -- the cache is worth 25 ms against a mount walk
> costing seconds, and two ways to obtain one quantity is how two figures come to disagree.

### The plan's grain for `metrics_by_poke` did not exist, and had to be made

The plan specified grain "trial + poke index". Measured, `poke_durations` returned
`["position", "odor_name", "poke_time_ms"]` -- **no trial identifier at all**, so its rows
were anonymous observations and that grain was not constructible.

`poke_durations` now carries `global_trial_id`. That is a deliberate change to a metric's
output, chosen over the two alternatives: saving the frame as-is ships a table nobody can
join back to a trial, which is most of the point of writing it; and rebuilding the table
from `position_data` in the writer would duplicate `_real_pokes` and the abort-event
exclusion, i.e. a second derivation of one quantity, which is what section 14 exists to
prevent.

**Safe because every consumer selects by name**, verified at all four call sites:
`_mean_sd_by` groups on `position` / `odor_name` and reads `poke_time_ms`;
`visualization_utils.py:5218` zips two named columns. The one site where the extra column
rides along is `visualization_utils.py:1651`, which renames, `.assign()`s and concatenates
-- and `plot_regression` covers it through `plot_sampling_times_analysis`. Emitted via
`reindex` so the column set is the same four whether or not the frame carries the id.

### `aborted` is not a figure parameter; `fa_types=None` is not a figure default

Two parameterised metrics are nonetheless saveable, and the distinction matters:

- **`poke_durations(aborted=)`** partitions **outcome classes**, not figure options. Both
  values are equally a record of the session, so the file carries both with an `aborted`
  column. Saving only the default would have dropped half the rows -- measured on
  `sub-057`, 629 completed against 45 aborted.
- **`fa_latency_from_pokeout(fa_types=None)`** means *unfiltered*, so the default is a
  well-defined value. The filtered variants stay figure-side.

That is the line: **three metrics taking a `window` and two taking an `fa_types` filter
have no single correct value to write**, and that is why "save everything and only load"
stays unreachable (section 5).

### The empty frame carries no type -- section 21's trap, in a new place

`hr_abort_poke_gap` returns **shape (0, 4) with all four dtypes `object`** on a session
with no hidden rule. Joined blind, that would make the file's schema a function of whether
the session happened to have a hidden rule -- exactly what 7b.1 spent its budget
eliminating. Every value column is therefore forced numeric, and `pd.to_numeric` is left at
its default `errors="raise"` rather than `"coerce"`, so an unexpected non-numeric is loud
instead of silently null. Verified: `sub-057` (no hidden rule) and `sub-040 20251124`
(hidden rule, multi-run) produce the **same 10 columns with the same dtypes**, differing
only in that three columns are all-null on the former.

`metrics_by_trial` is indexed on **every** trial in `trial_data`, not on the union of the
metrics' own keys, so the file left-joins one-to-one and an undefined trial reads as null
rather than as a missing row. The metrics are legitimately sparse -- on `sub-057`,
339 / 299 / 337 / 54 / 54 / 69 of 339 trials.

### The harness passes `save_tables` explicitly

`run_all_metrics(..., save_tables=True)`, folded into `need_output`. `qc/_common` and
`verify_scripts` pass `save_tables=False` **explicitly**, for section 23's reason: a gate
that relies on a default changes meaning whenever the default does. The two internal
`run_all_metrics(pooled_results, ...)` calls pass `False` too -- a pooled multi-session set
has no single `global_trial_id` space, so a per-trial table keyed on one would be ambiguous.

No CLI flag was added: the batch's per-session call picks up the default, which is what
7b.6 needs, and nobody asked for a switch.

**Gates:** `regression` GREEN 9/9 no regeneration; `plot_regression` GREEN, **35 cases
counted** (not the banner -- section 22) with all four movement plotters present and all
three `poke_durations` consumers exercised; `verify_scripts` GREEN, covering the
`batch_process` path that writes the tables; `check_imports` PASS.

---

## 26. The gate now covers what it writes, and the 16 metrics nothing watched *(Phase 7b.6, 2026-08-13)*

`regression.py` fingerprints **six** things per session, not two:

| key | what it covers |
|---|---|
| `trial_data` | the canonical CSV -- unchanged |
| `metrics` | the reported metrics dict (`run.REPORT`, 25 entries) -- unchanged |
| `position_data` | the side-table, **as written** |
| `metrics_by_trial` | the per-trial metric table, **as written** |
| `metrics_by_poke` | the per-poke metric table, **as written** |
| `unreported_metrics` | 16 registered metrics `REPORT` does not save |

**This closed a real gap, not a theoretical one.** 7b.4a and 7b.5 both landed GREEN, and
in both cases the GREEN measured *additivity* rather than coverage: the three tables were
written by code no gate read back, and 18 registered metrics were computed by code no gate
ran. A fingerprint that never looks at a file cannot tell "unchanged" from "unwatched".

The tables are fingerprinted from the **written file**, so the save path is covered; a
missing table records the md5 of `"ABSENT"` rather than being skipped, so a session that
silently stops writing one is a RED and not a shorter fingerprint (the section 2 rule).
`_common` therefore passes `save_tables=True` **explicitly** -- section 23's rule was
*pass it explicitly*, not *pass False*.

### Sixteen of eighteen, and the boundary is the section 5 line

`rolling_reward_fraction` and `rolling_hr_reward_fraction` take `window` **positionally,
with no default**, so fingerprinting them means inventing a figure choice. They are left
out. The other 16 are callable from their frames alone or have defaults that mean
something definite -- `fa_types=None` / `fr_types=None` are *unfiltered*, `aborted=False`
is *completed*. **Only that default variant is covered**; drift reachable only through a
non-default argument still is not.

### Populated before blessed, and already validated before that

`--generate` freezes whatever a metric returns, so a hash of an empty result is a canonised
bug. Measured across the nine sessions first: **14 populated on all 9**,
`fa_latency_from_pokeout` on 7 (it needs false alarms), `hr_abort_poke_gap` on 3 (it needs
a hidden rule). **None was empty everywhere.**

Populated is not the same as correct, so the prior validation was traced through the 4a
history rather than assumed. Three routes, and every one of the 16 sits on one of them:

- **12 -- the default variant is what an eyeballed plot draws.** 4a extracted these from
  `visualization/` and the conversion commits were `plot_regression`-gated: `1cafd76`
  (`inter_trial_interval`, `hr_abort_poke_gap`) **21/21 GREEN**, `70ec829` GREEN on all
  four, `604dd77` 18/19. So the chain is: eyeballed plot -> old inline computation ->
  gated byte-identical -> extracted core.
- **1 -- `presentation_counts_by_odor`** is in no plotter at all, but is consumed by
  `odorx_abortion_rate`, `hidden_rule_detection_rate` and `hidden_rule_counts_by_odor`,
  **all three in `REPORT`**, so it has been inside the metrics md5 since 4b.
- **3 -- `fa_rate_by_odor`, `fa_rate_by_position`, `false_response_ratio`**: the plotters
  always pass `fa_types=`/`odors=`, and `sing_rew` uses
  `false_response_ratio_contributions`, so the **default variant is not what any figure
  draws**. It differs from the validated path by a single `if fa_types is not None`
  branch -- same code, one conditional away. Recorded rather than blocked on.

Three deliberate deviations from "byte-identical" are on the record from 4a and are
representation changes, not value errors: `604dd77` moved 12 values by <=1 ULP (`np.mean`
vs pandas `mean`, chosen and confirmed at the time), `0012ae4` moved five metrics at the
0.999 ns `e9516e4` stopped truncating (max rel 2.2e-07), `de1a38d` moved one `FR_latency`
value by 18.8 ns.

**The regeneration was purely additive**: +8 keys per fixture, **0 removed, 0 changed**, on
all nine -- every pre-existing md5 byte-identical to the previous baseline, verified
*before* generating (the compare reported 18/18 green with 36 `[NO BASELINE]`) and again
after. Final compare **54/54 green**, 9/9 on each of the six keys.

### Two operational traps this cost real time to learn

1. **Never run two mount-heavy jobs against the SMB share at once.** An interrupted
   `--generate` overlapping with its own retry exhausted the SMB client's *handle pool*:
   `[Errno 24] Too many open files` on a recursive walk, and a process wedged in **U**
   state that `SIGKILL` could not reap until the I/O unwound. It is **not** a per-process
   fd limit -- `RLIMIT_NOFILE` was 1048576 and the system sat at 19,200 of 184,320. The
   fix is a force-unmount and remount, not a raised limit.
2. **A shallow probe cannot clear a mount.** Listing one directory of 62 entries "passed"
   while the mount was still wedged, and that false clear cost a second failed run. The
   walk that fails opens harp streams recursively, so the check must too: the honest probe
   walked both failing subject trees (2,476 dirs / 13,180 files), opened and read 526 real
   files, and watched the fd count stay flat at 4. Only that licenses "healthy".

> Also: `Bash(timeout=)` is capped at 600000 ms and **silently clamps**, so a longer value
> reads as a 10-minute kill on a job that needs more. Run the long gates in the background
> instead.

---

## 27. A blob field that nobody carries is a raise, not a silent drop *(Phase 7b.4b prep, 2026-08-13)*

`build_position_data` copies a **whitelist** of named fields out of the three per-trial
blobs. A key not on that list is dropped with no error, no empty column and no signal.
While the blobs remain in `trial_data` that is harmless -- the data is still there. Once
7b.4b removes them it is **data loss with nothing to notice it**, and the person who
loses the data is whoever adds a field to `classify_trials` and assumes it is saved.

So the whitelist is now enforced, in two complementary places.

### Strict at write, lenient at read -- and the split is the whole design

`build_position_data(trials, *, strict=False)`:

- **`save_results` passes `strict=True`.** The blobs were just produced by this run's
  classifier, so an unrecognised field means somebody added one and did not carry it.
  It raises `UncarriedPositionFieldError`, naming the field and both remedies. Raising
  is the safe failure in bulk for section 20's reason: `batch_analyze_sessions` catches
  per session, so the broken session names itself, writes nothing, and the batch
  completes.
- **Every read path leaves it `False`.** The same function runs over sessions saved long
  before these field lists existed, whose blobs are *not* today's (section 2). Refusing
  to read historical data because it carries a field we since stopped writing would be a
  far worse failure than dropping it. So the read path warns.

The check accumulates the field names seen across the whole frame and tests once, rather
than per entry: the answer is a property of the session, and a per-entry test would emit
the same warning thousands of times.

### One declaration, shared with the gate

`KNOWN_UNCARRIED_FIELDS` lives in `frames.py` and `qc/position_data_lossless.py`
**imports** it. A second copy in the gate is exactly the duplication that lets one be
updated and the other not, leaving a gate that blesses what the runtime forbids.

It holds one entry. `prior_presentations` is the failed Position-1 attempts preceding a
trial, read once *in memory* by `classify_trials` to build `non_initiated_odor1_attempts`
-- which is already persisted as its own table -- and it does not fit this grain anyway:
one row per failed *attempt*, not per `trial x position`. A field belongs on that list
only when the information is **not lost**; "nothing reads it today" is not a reason,
because the point is the reader that does not exist yet.

### `qc/position_data_lossless.py` asserts the precondition for the drop

**Not "the two are identical"** -- that can never be true (section 24), and a gate built
on it would be wrong rather than strict. It asserts the claim that actually licenses the
drop: *every field of every blob entry is recoverable from the matching `position_data`
row, with an equal value, except a named allow-list.* Per `(blob, key)` it reports
`equal` / `differs` / `absent`; `differs` always fails, `absent` fails unless
allow-listed.

**Measured, 9 sessions:** 25 of 26 pairs carried with **`differs: 0` and `absent: 0`**
on all 4,791 occurrences each; the 26th is the allow-listed one, 1,730 occurrences.

**Proved before trusted (section 17), on four paths:** emptying the allow-list reports
`prior_presentations` as NOT CARRIED; corrupting **one cell of 9,770** reports
`differs: 1` and names the trial, position and both values; dropping a carried column
reports NOT CARRIED on all 383; and the clean run is GREEN.

### The trap the guard itself walked into

The first version of the guard built its whitelist from `_POKE_FIELDS + _VALVE_FIELDS +
_PRES_FIELDS`. **`_PRES_FIELDS` is not read by the builder** -- it documents what that
blob happens to carry, while the copy loop hardcoded `("index_in_trial",
"is_last_event")` and sourced everything else from the poke/valve lists with
`presentations` as fallback. So a field added to `_PRES_FIELDS` alone would have
**passed the guard and still been dropped**: a false pass in precisely the scenario the
guard exists for.

Fixed by introducing `_PRES_ONLY_FIELDS`, read by *both* the copy loop and
`CARRIED_FIELDS`, so one declaration drives behaviour and check alike. The error message
now says explicitly not to use `_PRES_FIELDS`. `CARRIED_FIELDS` is unchanged at 13 names
-- `_PRES_FIELDS` was a strict subset -- so no output moved, confirmed by `regression`
GREEN 54/54.

> **The general rule: a guard whose whitelist is wider than the behaviour it guards
> reports a pass for something still broken.** Derive the check from what the code
> actually reads, never from a constant that merely looks authoritative.

### Adding a field later: two grains, no propagation between them

`TrialRecord` and the blob field lists are **separate declarations at different grains**,
and neither updates the other:

| adding… | declare in | automatic | still manual |
|---|---|---|---|
| a **trial-level** field (one value per trial) | the right `TrialRecord` class | `trial_data_columns()`, `save_results`' conform, the loader's schema check | assign it in `classify_trials`; add to `DATETIME_FIELDS` if it is a datetime (section 21) |
| a **per-position** field (one per `trial x position`) | `_POKE_FIELDS` / `_VALVE_FIELDS` / `_PRES_ONLY_FIELDS` | nothing | the list entry -- but `save_results` **raises** if you forget |

Not `_PRES_FIELDS`. Either way it is an intended output change: `regression` RED on
`+ added column` (in `trial_data_columns` or `position_data_columns`), regenerated
deliberately.

---

## 28. Measure which call sites the gate reaches *before* porting them *(Phase 7b.4b, 2026-08-13)*

Phase 7b.4b drops the three blobs, and the drop itself is three lines. The work is the
readers, and the first question is not "how do I port them" but **"which of them would a
green gate actually be talking about"** -- section 22 one level deeper. Measured before
writing any code, by patching `Series.get` / `DataFrame.get` to record the caller's
`file:line` whenever the key is one of the three blob columns, then running
`plot_regression`'s full case list in one process.

**`plot_regression` executed 4 of the 11 live per-trial blob reads. Seven were at zero.**

| covered | reads | by |
|---|---|---|
| `pred_seq_utils.fa_analysis` | 35 | `fa_analysis` |
| `movement_analysis_utils` (two `_last_poke_out*`) | 273 / 151 | `plot_trial_traces_by_mode`, `plot_traces_with_speed_threshold` |
| `sing_rew_movement._last_poke_out` | 155 | `plot_category_traces` |

| at zero | why |
|---|---|
| `pred_seq_utils.last_odor_poke_time` / `first_odor_poke_duration` / `poke_time_all_pos` | plotters simply not in `CASES` -- **now added**, 35 -> 38 |
| `metric_analysis.movement` (`_last_poke_out`, `_last_valve_start`) | see below |
| `debug.py` (two helpers, both reading a *loaded* `trial_data`) | not plotters; deliberately left unguarded |

Each of the three added cases was checked to **draw real data before being added** --
482 / 40 / 116 points, and each registers reads at exactly the site it was added for.
A case that draws nothing is the section 26 trap in a new place: it would have gone green
by drawing nothing in both trees.

### `compute_speed_analysis` cannot be gated by `plot_regression`, and the reason is a write

Its only caller is `run_speed_analysis_batch`, whose only caller in the repo is a
notebook -- so no gate reaches it. It cannot simply be added as a case: it **writes
`speed_analysis.parquet` into `results_dir`**, and `get_derivatives_root()` is
`/Volumes/harris/hypnose/derivatives`, which is strictly read-only. The same is true of
`plot_traces_with_speed_threshold` on a session *lacking* that file -- it recomputes and
saves. The existing movement cases are safe only because both their sessions
(`sub-040 20251124` / `20251229`, the only two fixtures that have it) already carry the
file, so they read it.

> **A gate case must not write into the data it reads.** Where that rules a gate out, the
> control is a one-off equality probe against a *local copy*, recorded with the commit --
> not a permanent gate, and not a silent gap either.

### What the `build_position_data` call sites hid

The inventory of "reads a blob column" misses a second, larger set: **nine sites pass an
already-loaded trial frame to `build_position_data`**, which after the drop returns an
**empty frame with no error**. They are well covered (4,374 reads from 11 cases), but they
reach `trial_data` through `io/loaders._load_trial_views` / `_load_sorted_session`, *not*
`io/load_results.load_results_dir` -- so "switch the loader to read the side-table" has to
mean **both** loaders or those nine break silently. One of them,
`sing_rew._session_reward_rts`, passes a *filtered* subset (rewarded trials only), so for
it a file read is not a substitution and needs an explicit filter back to the frame's own
trials.

### `sequence_depth` moved grain, and the rule did not *(step 2)*

`sequence_depth(trial)` read the blobs off a **trial row**; it is now
`sequence_depths(trials, position_data)`, returning an `Int64` Series over the frame, with
`reached_counts(trials, position_data)` on top. Sections 10 and 18 state the rule as a
property of *a trial* -- that is still exactly the rule, and this changes only where it
reads the per-position facts from.

**The per-trial signature was retired rather than kept.** Its only consumer was
`reached_counts`' own loop, so keeping it would have meant slicing the long frame once per
trial (1,731 slices a session) *and* handing every caller the job of passing the right
slice -- silent and wrong if they pass the wrong one.

**The two precedences are opposite, and that is the whole subtlety.** Presented reads
`in_poke_times` then `in_presentations`; sampled reads `in_presentations` then
`in_poke_times` -- which is also the opposite of the merge inside `build_position_data`,
where `position_poke_times` wins. Reconstructing the per-blob order from section 2's
provenance flags is exact **only because** section 24's inventory measured the blobs never
to disagree on a shared field. That is a measurement about the data, not a property of the
code, so it was measured directly rather than argued from:

- **1,731 of 1,731 trials equal** to the blob form, on the written `position_data.parquet`
  *and* on the load-time `build_position_data` derivation -- compared against HEAD's
  `frames.py` extracted with `git show`, not a re-typed copy of the old rule.
- `reached_counts` identical position-for-position on all nine sessions.
- **Sensitivity proved (section 17):** forcing the two single-meaning rules section 18
  rejected moves **84** and **32** trials -- exactly the counts section 18 recorded when it
  measured them off the blobs. A probe that cannot tell those apart would prove nothing.

### The key is `global_trial_id` alone, so counting is per row

`trial_data` carries no `subjid` / `date` column: measured on all nine fixtures, the only
ids the two frames share are `trial_id` and `global_trial_id`, and `trial_id` restarts per
run. So a pooled frame -- `merge.pool_results_dicts` concatenates sessions, and
`run_all_metrics` is called on one for `merged_metrics` -- has **no key that separates two
sessions' trials**.

> Hence a per-row `.map`, never a `groupby` on the trial frame. A groupby would collapse
> colliding ids into one row and **change the number of trials in a denominator**; a map
> yields one value per trial row whatever the ids do. Where ids do collide the depth is the
> maximum across them, and `sequence_depths` **warns saying so** rather than presenting a
> resolved-looking number -- section 2's "absent means unknown", applied to "ambiguous".

This is the same pooled ambiguity section 25 recorded for every other `position_data`
metric; it is not newly introduced here, and the fix is to compute per-position metrics one
session at a time.

### The readers yield entries and reduce nothing *(step 3)*

Thirteen call sites across six modules moved off the blobs, through one primitive:
`frames.position_entries_by_trial(position_data, flag)` -> `{global_trial_id: [entry, ...]}`
sorted by position. It **reduces nothing**, because the callers do not agree on what
"the last poke-out" is:

| caller | rule |
|---|---|
| `sing_rew_movement._last_poke_out` | the **maximum** `poke_odor_end`, explicitly not trusting position order |
| `metric_analysis.movement._last_poke_out`, `movement_analysis_utils._last_poke_out_scanning_back` | scan **back by position** to the first non-null |
| `movement_analysis_utils._last_poke_out_by_position` | the **last entry**, null accepted, falling through to row columns only when there are no entries at all |

Three rules, one shape of question -- sections 13 and 14 are about not merging helpers
that differ, so the primitive hands over the entries and each caller keeps its own.

**Every site passes the provenance flag matching the blob it used to read** (section 2):
`in_poke_times`, or `in_valve_times` for the two valve readers. `_last_valve_start` is the
one where it bites hardest -- it joins valve rows to poke rows by position, and
`position_valve_times` is a *superset*, so an unfiltered read would offer the join
positions the poke blob never had.

`io/loaders._load_position_data(results_dir, trials)` is the **single seam** between "where
the per-position facts live" and everything that reads them, so step 4 repoints one
function rather than nine call sites. It takes `trials`, not just the directory, because
callers do not all pass the whole session -- `sing_rew._session_reward_rts` passes the
rewarded trials only, and a reader of the saved table must filter back to the frame it was
handed or the metric silently widens.

Deleted with the port, all of them blob parsers with no remaining purpose:
`pred_seq_utils._extract_position_entry` / `_ordered_position_entries`,
`prep._last_position_entry` (imported by two modules, called by neither),
`debug._parse_like_mapping` / `_iter_position_entries`.

### `compute_speed_analysis` has no gate, so it got a probe -- and the probe needed proving

Four of the ported reads live in `compute_speed_analysis`, which no gate reaches and none
can (above). Its control: lift `_last_poke_out` and `_last_valve_start` out of **both**
trees -- they are nested functions, so by AST -- and run them over every trial of all nine
sessions. **1,731/1,731 identical for both helpers**, on the written parquet and the
load-time derivation alike.

> **The first sensitivity check reported the probe BLIND, and it was right to.** Perturbing
> a `poke_odor_end` cell changed nothing, because the cell chosen was not one either helper
> *reads* -- `_last_poke_out` returns the last non-null by position, and the perturbed row
> was interior. Re-aimed at the cell each helper actually returns, the probe discriminates:
> one cell -> `differs: 1`, every `poke_odor_end` +1 ms -> 272 of 273, every `valve_start`
> +1 ms -> 272 on `valve_start` and **0** on `poke_out`, and vice versa.

**A probe validated with the wrong perturbation proves nothing** -- section 27's "a guard
whose whitelist is wider than the behaviour it guards" in a second costume. Validate a
check by perturbing *the value the code under test returns*, not merely a value it can see.

### The saved table becomes the source, and the derivation the compatibility path *(step 4)*

`io/load_results.load_position_data(results_dir, trials)` is now the **one** place that
decides where the per-position facts come from: prefer `position_data.parquet`, fall back
to `build_position_data`. It lives in `load_results.py` -- the read side of
`save_results.py` -- and `io/loaders.py` **re-exports** it under the private name the
readers already import, the same arrangement (and reason) as the `readers.py` primitives:
one definition site, and `loaders`' heavy `harp`/`aeon` imports stay out of `load_results`.

**The fallback is not a formality.** Sections 2's rule is that an absent source means
*unknown*; sessions saved before 7b.4a have no such file and still carry the blobs, so the
derivation answers correctly for exactly the files the read is missing from.

**Filtered back to `trials`, always.** The saved table holds the whole session and callers
do not all pass the whole session -- `sing_rew._session_reward_rts` passes the rewarded
trials only. An unfiltered read would widen it from its subset to every trial in the
session: a changed metric, no error. When the two frames cannot be keyed on
`global_trial_id` the derivation is used instead, since it is defined by whatever frame it
is handed and therefore cannot make that mistake.

`SessionResults` keys off the `results_dir` **already in the mapping**, so laziness (the 22
of 29 ms, section 5) survives and there is no second attribute to keep in step through
`copy()`. `from_trials` sets no such key: a caller holding only a frame has no directory
to read, and the derivation is the right answer for it.

> **The case with neither source now speaks.** Blobs gone *and* no saved table -- a session
> that missed the re-analysis -- used to yield an empty frame, which every per-position
> metric reads as "this session had no positions". It now warns, naming the session and
> the remedy. That is section 27's failure one level up from the field lists: the silence
> was the bug, not the emptiness.

**Verified on all three paths before gating**, on `sub-040 20251124`: file present -> read
taken (`poke_odor_start` is `datetime64[ns]`, against the derivation's `str`, so the switch
is demonstrably real and not a no-op); file absent, blobs present -> fallback, 1,019 rows;
neither -> 0 rows and one `RuntimeWarning`. A 20-trial subset returns 59 rows spanning
exactly 20 trials, so the filter works.

**And the dtype change moved nothing drawn.** Section 24 measured that all 14 evaluable
`position_data` metrics normalise through `metrics.common._tz_naive`; the seven viz readers
ported in step 3 are *not* metrics and parse timestamps themselves, so they were the open
question here. `plot_regression` GREEN 38/38 answers it.

### Two `presentations` guards that never read the blob

`odorx_abortion_rate` and `plot_false_alarm_rate_by_position` both began
`if ... "presentations" not in columns: return empty` -- yet neither reads that blob; both
take their denominator from `position_data`. Left in place, the drop would have turned a
**reported metric** and a **figure** silently empty. Removed with the re-plumb, which is
also why they are worth naming: a guard on a column the function does not use is invisible
until the column goes.
