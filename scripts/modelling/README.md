# scripts/modelling

Model fits to behavioural trial sequences. Like the other `scripts/`, these are thin entry
points — this script holds only the `run_*` functions, the printed tables, and the CLI. All
the maths and figures live in `src/hypnose_behavior/`:

- **numeric core** — [`hypnose_behavior.modelling.switchpoint`](../../src/hypnose_behavior/modelling/switchpoint/),
  one module per role: `data` (build the sequence), `switch` (the switch-point model family),
  `qlearning`, `compare` (AIC/BIC), `permutation` (the sleep test), `autocorr` (the
  residual check), `bootstrap` (planned).
- **figures** — [`hypnose_behavior.visualization.modelling.switchpoint.plots`](../../src/hypnose_behavior/visualization/modelling/switchpoint/plots.py).

Run from the repo root in the project conda environment; the script adds `src/` to the path, so
no install is required.


| Script | What it does |
| --- | --- |
| `switchpoint_analysis.py` | Detects the LONG → SHORT strategy switch per animal, tests whether switches align with sleep, and provides two model diagnostics (logistic multi-start, residual autocorrelation) |


### The sequence being modelled

Each kept trial `i` becomes one binary outcome:

```
s[i] = 1  if hidden_rule_success == True   (SHORT: left early, used the hidden rule)
s[i] = 0  otherwise                        (LONG: waited out the full sequence)
```

### Reward identity (A / B)

Every trial also carries the identity of the reward it is associated with:


### The models

`s[i] ~ Bernoulli(p_i)`, with five competing descriptions of `p_i`:

| Model | `p_i` | Params | Meaning |
| --- | --- | --- | --- |
| `constant` | `p` | 1 | No strategy change |
| `switch` | `p1` if `i < tau` else `p2` | 3 | One abrupt change at trial `tau` |
| `logistic` | `lo + (hi - lo) · sigmoid(slope · (i - midpoint))` | 4 | A graded change |
| `switch2` | `p1` \| `p2` \| `p3` split by `tau1 < tau2`, gated `p1 ≤ p2 ≤ p3` | 5 | A change arriving in two stages (non-decreasing) |
| `qlearning` | `sigmoid(b · (Q_short − Q_long) + kappa · s_prev)`, values updated per trial | 5 | The **mechanistic null**: a gradual rise derived from value learning |

`tau` is the index of the **first trial of the post-switch regime** (so `1 ≤ tau ≤ n-1`; a
switch at 0 is just the constant model). The primary hypothesis is one directional switch,
`p1` low → `p2` high.

`switch2` maximizes **exhaustively** over all ordered pairs `tau1 < tau2`


The `logistic` model **nests** the `switch` model as `slope → ∞`, which is what makes
`slope` a direct read-out of **abruptness**: a large fitted slope is a step, a small one is
a slow drift.

#### `qlearning` — the null to be rejected

The other four models *describe* the P(SHORT) curve. This one *derives* it, from a
trial-by-trial value update. It is therefore fitted at its best (multi-start, below). 

Two options with **fixed** rewards `r_short = 1`, `r_long = 0`. Only the chosen option
updates, `Q[chosen] += alpha · (r[chosen] − Q[chosen])`, and the choice rule is

```
P(SHORT at t) = sigmoid(b · (Q_short − Q_long) + kappa · s_prev)
```

with `s_prev` = +1 after a SHORT trial, −1 after a LONG one, 0 at the first trial. Each
trial's probability uses the values held **before** that trial's update.

Three variants, all sharing `alpha ∈ [1e-4, 1]` and `b ∈ [1e-3, 50]`:

| Variant | Free params | `Q0` bounds | Meaning |
| --- | --- | --- | --- |
| `qlearn_free` | alpha, b, Q0_short, Q0_long | `[-10, 10]` | May start outside the experienceable range |
| `qlearn_constrained` | alpha, b, Q0_short, Q0_long | `[0, 1]` | Initial values lie within the range of experienceable outcomes |
| `qlearn_perseveration` | + kappa | `[-10, 10]`, kappa `[-10, 10]` | Plus a choice-history (stickiness) term |

### How the logistic is fitted (multi-start)

The logistic likelihood surface has **more than one basin**, and which one Nelder-Mead settles
in is decided mostly by the *initial slope*.

`fit_logistic` minimizes from **16 dispersed initial conditions** and keeps the best:

- the switch-point warm start (`midpoint = tau`, asymptotes at `p1`/`p2`), and
- a 5 × 3 grid — midpoints at the 10/30/50/70/90% quantiles of the trial axis, both asymptotes
  at the global SHORT rate, and initial slopes `0.05` / `0.5` / `5.0` (shallow-gradual through
  steep-near-step).


---

## Entry points

All are importable from a notebook and runnable from the terminal.

```python
from switchpoint_analysis import run_analysis, run_qlearning_sweep, run_permutation

results = run_analysis(
    subjids=[40, 45],
    date_ranges={40: (20251201, 20251231), 45: None},
    rewarded_only=True,
    likelihood_window=100,
    split_ab=False,      # True -> fit and plot the A- and B-reward trials separately
    show=False,          # keep the figures rather than displaying them
    qlearning_overlay=True,  # fit the Q-learning null and overlay it (default; False to skip)
)
results[40]["tau"], results[40]["hdi_width"], results[40]["figures"]["posterior"]
results[40]["qlearning"]["qlearn_free"]["alpha"]   # the three variant fits, or None if skipped

# with split_ab=True the per-subject value is nested by reward identity:
#   results[40]["A"]["tau"], results[40]["B"]["figures"]["strategy"]

sweeps = run_qlearning_sweep(       # standalone; three figures per animal, one per variant
    subjids=[40],
    date_ranges={40: None},
    rewarded_only=True,
    split_ab=True,
    show=False,
)
sweeps[40]["A"]["figures"]["qlearn_free"]

perm = run_permutation(
    subjids=[40, 45, 48, 50],       # may be a different set of animals
    date_ranges={s: None for s in (40, 45, 48, 50)},
    rewarded_only=True,
    inclusion="bic_switch_wins",
    n_permutations=10000,
    seed=0,                          # reproducible null
    show=False,
)
perm["observed_mean"], perm["p_value"], perm["null_means"], perm["n_pairs_dropped"]
```

`date_ranges` maps each subject to an inclusive `(start, end)` `YYYYMMDD` tuple, an explicit
date list, or `None` for all sessions. A `{subjid: date_range}` dict may be passed as
`subjids` on its own, matching the convention of the plotters in `hypnose_behavior.visualization`.

### `run_analysis` — figures per animal

Shown in this order, one animal at a time.

1. **Strategy** — SHORT/LONG per trial on the continuous trial axis, each trial coloured by
   its reward identity (A red, B teal, unresolved grey), with a blue dotted vertical line at
   each session end (sleep).
2. **Model comparison** — the data with every fitted model overlaid as **one fitted line each**
   (constant line, switch step, switch2 two-step, logistic curve), and the five-row AIC/BIC table 
   in-panel with the BIC winner marked, so *no switch / abrupt / gradual / two-stage* can be read 
   off directly. The printed table adds each model's loglik, the nesting check, and the winner's 
   fitted parameters.

   Unless `qlearning_overlay=False`, the three Q-learning variants are also drawn here, one
   **solid line each labelled `(null)`** — the variant's **one-step-ahead** curve, the quantity
   its AIC/BIC scores. That curve is conditioned on the animal's own choices, so it is *not* a
   prediction of the trajectory (with a large `kappa` it becomes a one-trial-lagged copy of the
   data); the honest, generative view is figure 4. 
3. **Posterior** — the switch-point posterior over *all* trials, plotted windowed to
   ±`likelihood_window` trials around its peak. `tau`, its session, and the HDI width are
   printed and annotated (HDI primary, FWHM secondary).
4. **Q-learning generative** (overlay only) — three stacked panels, one per variant, showing
   what each fitted null actually *predicts*: the model run forward on its own choices, drawn as
   the generative mean (thick), the 5–95% band, and a few individual simulated runs (faint). 


### `run_logistic_diagnostic` — is the logistic fit trustworthy?

A **standalone** diagnostic. Per animal it replays the shipped
start set and shows where each initial condition ends up, so you can see whether the starts
funnel into one optimum or split into basins:

- every converged sigmoid overlaid on the data, one colour per start — **winner bold**, the
  switch-point warm start **dashed**;
- in the margin above the data, each start's **initial** midpoint (▽) joined by a faint
  connector to where it **converged** (○), in that start's colour;
- a printed per-start table: initial midpoint → converged midpoint, converged slope, converged
  loglik, with `[best]` / `[warm]` tags, the number of distinct basins, and a note when a
  dispersed start beat the warm start.

```python
from switchpoint_analysis import run_logistic_diagnostic

diag = run_logistic_diagnostic([40], {40: (20251125, 20251231)}, rewarded_only=True,
                               split_ab=True, show=False)
diag[40]["A"]["best_label"], diag[40]["A"]["n_basins"], diag[40]["A"]["fig"]
```

### `run_permutation` — one two-panel figure

**Left**: two boxplots with the points overlaid — real `f` (one point per included animal) and
the span-guarded pool of shuffled `f` (one point per valid recipient × donor pair), annotated
with the observed mean and the p-value. **Right**: the paired-permutation null distribution of
the mean `f`, with the observed mean marked.

Returns `real_f`, `shuffled_f`, `null_means`, `observed_mean`, `p_value`, `n_permutations`,
`n_pairs_dropped`, `included_subjids`, `excluded_subjids` (no switch), `excluded_no_donor` (no
donor spans their `tau`), `per_subject`, and `fig`.

---

## Terminal usage

```bash
# per-animal switch-point fit and figures
python scripts/modelling/switchpoint_analysis.py analysis --subjids 40 --likelihood-window 100

# restrict to a date range, rewarded trials only
python scripts/modelling/switchpoint_analysis.py analysis --subjids 40 45 \
    --date-range 20251201 20251231 --rewarded-only

# fit the A- and B-reward trials separately
python scripts/modelling/switchpoint_analysis.py analysis --subjids 40 \
    --date-range 20251125 20251231 --rewarded-only --split-ab

# skip the Q-learning null fits (restores the pre-Q-learning figure and printout)
python scripts/modelling/switchpoint_analysis.py analysis --subjids 40 --no-qlearning

# Q-learning (alpha, b) parameter sweep -- three figures per animal, one per variant
python scripts/modelling/switchpoint_analysis.py qsweep --subjids 40 --rewarded-only --split-ab

# where does each logistic multi-start initial condition converge?
python scripts/modelling/switchpoint_analysis.py diagnostic --subjids 40 --rewarded-only --split-ab

# is the fitted model's residual serially independent? (the bootstrap's i.i.d. assumption)
python scripts/modelling/switchpoint_analysis.py autocorr --subjids 40 --rewarded-only

# do switches align with sleep?
python scripts/modelling/switchpoint_analysis.py permutation --subjids 40 45 48 50 --rewarded-only

# a looser inclusion rule, more permutations, a different seed
python scripts/modelling/switchpoint_analysis.py permutation --subjids 40 45 48 50 51 \
    --inclusion bic_beats_constant --n-permutations 50000 --seed 1
```

| Argument | Subcommand | Meaning |
| --- | --- | --- |
| `--subjids ID [ID ...]` | all | Subject id(s). Required. |
| `--dates D [D ...]` | all | Specific dates `YYYYMMDD`. |
| `--date-range START END` | all | Inclusive `YYYYMMDD` range (alternative to `--dates`). |
| `--rewarded-only` | all | Keep only rewarded trials; aborts are always dropped. |
| `--likelihood-window N` | `analysis` | Half-width of the posterior plot window (default 100). |
| `--split-ab` | `analysis`, `qsweep`, `diagnostic`, `autocorr` | Fit and plot the A- and B-reward trials separately. |
| `--no-qlearning` | `analysis` | Skip the Q-learning null fits and their overlay. |
| `--n-starts N` | `qsweep` | Random starting points per Q-learning fit, min 20 (default 32). |
| `--max-lag N` | `autocorr` | Largest lag reported, clamped to n-1 (default 50). |
| `--inclusion RULE` | `permutation` | Which animals count as having switched (default `bic_switch_wins`). |
| `--n-permutations N` | `permutation` | Permutations drawn for the null (default 10000). |
| `--seed N` | `qsweep`, `permutation` | RNG seed for a reproducible multi-start / null (default 0). |

`--dates` and `--date-range` are mutually exclusive; omit both for all dates. The CLI applies
one date range to every subject — for per-subject ranges, call the functions directly.
Subjects with no data are skipped via `hypnose_behavior.qc.validate.validate_subject`, as in the other
scripts.

## Deferred (planned follow-ups)

Not implemented yet; recorded here so the intent is not lost.

- **Transition-width comparison figure.** For every model *and* the empirical data, compute the
  transition width the same way — the number of trials it takes P(SHORT) to cross from 0.1 to
  0.9 — and show the per-model distribution of that width on one axis. The point is to put the
  near-instant descriptive fits (`switch` is a step, `logistic` steep) and the wide Q-learning
  *generative* spreads on a common ruler, so "how abrupt" becomes a single comparable number
  rather than a per-figure impression. For the Q-learning variants the width is measured on the
  simulated runs, **switched runs only** (`p2 - p1 >= switch_threshold`, as in
  `qlearning_generative_band`), and reported alongside `frac_switched` — a variant that seldom
  switches is failing differently from one that switches over a wide window, and the width alone
  hides that. Builds on the `crossing_widths` helper sketched for this (0.1→0.9 crossing on an
  arbitrary trajectory); wire it through `compare_models` outputs and the generative bands.

- **Interactive fit explorer (notebook only, not a paper figure).** An `ipywidgets` tool: pick
  an animal and a variant, get sliders for `alpha`, `b`, `Q0_short`, `Q0_long`, `kappa`, and
  redraw live over that animal's trial data — the one-step-ahead curve always, the generative
  mean and band on demand. Cheap enough to be interactive: `_trajectory_unchecked` is
  closed-form (no per-trial loop) and `simulate_qlearning` is vectorised across simulations, so
  a slider drag is a couple of array passes. For exploring how the fitted parameters shape the
  curve and sanity-checking a fit by hand, not for the manuscript.

## Caveats

- **The p-value's resolution is set by the number of animals, not by `n_permutations`.** With
  `N` recipients the paired null has at most `!N`-ish distinct donor assignments (9 for `N`=4,
  before the span guard removes some), so raising `n_permutations` only resamples the same
  handful of values — the null histogram is visibly discrete. `run_permutation` needs at least
  **two** included animals with a valid donor and raises otherwise, but a two- or three-animal
  test cannot produce a small `p` no matter how many permutations are drawn.
- Animals excluded by the span guard (`excluded_no_donor`) still act as donors for the others.
  They contribute no `f` of their own, so the observed statistic averages over fewer animals
  than were fitted.
- `f` is NaN when a `tau` precedes every session start. Session starts always begin at trial
  0 by construction, so this cannot occur for real or donated boundaries — it is a guard, not
  an expected case.



Quick explanations: 
tau = switch trial, printed with session 
p1, p2 = two regime rates. Probability of short-sequence solve before and after switch. 
95% HDI: uncertainty where switch happens. (how many trials cover 95% confidence)
FWHM: All trials whose posterior probability are within half of the width of the peak
AIC/BIC: both score model performance
