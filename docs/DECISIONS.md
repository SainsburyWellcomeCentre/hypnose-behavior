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
  recorded as `fr_latency_ms`, off a **different anchor** (`await_reward_time`, not the last
  poke-out), so the two are not interchangeable.
- **10 `timeout`** — no reward-port poke anywhere, so there is no response to time.

And measured for completeness: `fr_latency_ms` is present on **101/101** trials where a false
response occurred and absent on all 63 `nFR`, exactly 1:1 with the `false_response` flag. The
abort side is identical (`fa_latency_ms` 69/69 on FA, 0/44 on nFA). **Absence there means the
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
| completed | `completed_window_latency_ms` *(new)* | **`response_time_ms`** *(re-anchored)* |
| aborted | **`fa_latency_ms`** *(unchanged)* | `fa_movement_latency_ms` *(new)* |
| no-go | **`fr_latency_ms`** *(unchanged)* | `fr_movement_latency_ms` *(new)* |

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

### The naming is knowingly inconsistent

`response_time_ms` is (b) while `fa_latency_ms` / `fr_latency_ms` are (a). Making
`response_time_ms` mean (a) would have been consistent, but it repoints ~1063 values and shifts
`avg_response_time` by about a second, against 39 values for keeping it as (b). The trade was
taken deliberately; the debt is recorded in the plan under Phase 6, to be settled as one
intended change rather than smuggled in here. **`avg_response_time` reads (b).**

### Consequence for the anchor primitive

(b) is computed with `windows.last_poke_end_before(cue_series, reward_poke)`, never from
`poke_odor_end` — see §15 for why that is 25 ms late on a grace-derived entry. The existing
metric `metric_analysis/metrics/false_alarm.fa_latency_from_pokeout` was an attempt at (b) that
anchored on `poke_odor_end`, so it carried exactly that bias **and** did not exclude resampling.

**Repointed 2026-08-10** — it now reads `fa_movement_latency_ms` and its frame drops from
`trials+position_data` to `trials`. Measured on the two `fa_analysis` gate sessions the row
*membership is identical* (90/90 and 15/15), so this is purely a value change, and a large one:

| | median, sub-040 20251124 |
|---|---|
| old, from `poke_odor_end` | 11216.0 ms |
| `fa_latency_ms` — (a), from the abortion | 7821.7 ms |
| new — (b) | **1822.3 ms** |

The old value exceeded even the abortion-anchored latency, because the last odor's poke-out
precedes the abortion: it was charging every later cue-port visit to the false alarm.

`regression.py` stays GREEN — the metric is registered but **not in `REPORT`**, so it is not in
the metrics fingerprint. The gate that sees it is `plot_regression.py`, whose `fa_analysis` case
goes RED by design. **A metric outside `REPORT` is invisible to the golden master; check whether
it is in `REPORT` before assuming a metric change is gated.**

Consumers with no `fa_movement_latency_ms` column get an empty Series rather than a fallback to
the old computation, by the §2 rule: a session saved before Phase 11 cannot be made to look
comparable to one saved after it.
