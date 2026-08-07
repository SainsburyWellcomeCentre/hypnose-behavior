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

## 3. `frames.py` must stay a leaf

`io/load_results.py` calls `build_position_data`, which lives in `metric_analysis/frames.py`.
That looks like `io/ → metric_analysis/` and therefore like a cycle. It is not, for one reason
only:

> **`frames.py` imports nothing from the package** — only `json`, `re`, `typing` and `pandas`.
> Both `io/__init__.py` and `metric_analysis/__init__.py` are docstring-only, so importing a
> submodule triggers no package-level side effects.

`io → metric_analysis.frames` is a one-way edge into a leaf. **The day a metric — or anything
else in the package — is imported into `frames.py`, `io/load_results.py` becomes a real cycle.**
Keep its imports to the standard library and pandas.

*(Settled 2026-08-06, do not re-open: `build_position_data` performs no I/O, so `io/` is the
wrong home by the 0.2 "knows the data vs knows the layout" test, and it shares four helpers with
`sequence_depth` / `reached_counts` / `sampled_positions`. Promoting `frames.py` to a schema
layer below both is the honest fix — revisit only if it grows.)*

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

## 10. Phase 7b TODO — the 0 ms positions and `poke_source`

Two data-writing bugs make the position record incomplete and ambiguous, and both surface as
per-position metrics that cannot be defined consistently.

1. **Write the 0 ms / no-poke positions.** A position whose poke registers as ~0 ms is currently
   omitted from `position_poke_times`, `presentations` *and* `num_odors`, even though the odor was
   presented and the sequence advanced through it. Write it with `poke_time_ms = 0` and null
   `poke_odor_start` / `poke_odor_end`.
2. **Add `poke_source`** to every position entry: `"poke"` for a genuine poke inside the odor
   window, `"grace"` for one synthesised by the `PRE_ODOR_GRACE_MS` path
   (`classification_utils:1281-1293`, where the poke ended *before* the valve opened), `"none"`
   for a 0 ms / no-poke position. Today a grace entry is indistinguishable from a real short poke
   except by the fragile tell `poke_first_in == poke_odor_start` — and animals genuinely poke for
   under 20 ms, so the marker is the only reliable separator. Direct measurement (grace set to 0)
   puts grace-derived entries at **~2-10 odors per session**.

Consumers must treat an **absent** `poke_source` as "unknown" and omit the filtered variant, never
as "all real pokes" — older sessions will never carry the field. Alters `trial_data` ⇒ deliberate
fixture regeneration with the diff confirming only the intended columns moved. The writing happens
in `classify_trials`, so it lands naturally with Phase 6's trial-loop cleanup.

### What it unblocks, and why `sequence_depth` looks wrong until then

`only_true_pokes` on the sampling metrics becomes computable, and `sequence_depth` collapses to a
one-line change.

**`sequence_depth` deliberately reproduces *today's* rule, not the `presentations`-sourced target.**
The target says the source is `presentations` and the set is `1..max(presented position)` for every
trial; the canonical metrics instead walk `1..last_odor_position` for an **aborted** trial. Measured
on the 9 fixture sessions, **10 of 1731 trials disagree**, moving `reached` counts on 3 sessions:

```
sub-048  today={1:181, 2:152, 3:117, 4:91, 5:65}   presentations={1:181, 2:153, 3:118, 4:92, 5:65}
sub-057  today={1:338, 2:283, 3:226}               presentations={1:339, 2:287, 3:227}
sub-059  today={1:221, 2:208, 3:139}               presentations={1:221, 2:209, 3:140}
```

The disagreeing trials are precisely the grace artifact. **Switching now would not be more correct
— it would bake that artifact into the denominators of `abortion_rate_positionX` and
`fa_abortion_stats`**, because nothing yet distinguishes a genuine short poke from a synthesised
one. Only after `poke_source` exists can the two sources agree, at which point the
aborted/completed branch collapses. The reasoning and the numbers are in the docstring.

### And why the two position helpers must stay separate

`sequence_depth` ("how far the sequence got") is **never** filtered; `sampled_positions` ("was this
position sampled") **is**. A single filtered `reached_positions` produces physically impossible
sets: dropping a non-`poke` entry from the middle of a trial credits it with reaching position 5
but not position 3, which makes any per-position denominator non-monotonic. A gap is meaningless
for *reached* and perfectly natural for *sampled*.

The contiguous `1..max` fill is doing real work until the fix lands — `sub-057 gid=108` has
`position_poke_times` keys `[2, 3]` and `num_odors=2`, but position 1 *was* presented and its 0 ms
poke was never written. `1..max` recovers it; plain membership loses a real position.

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
