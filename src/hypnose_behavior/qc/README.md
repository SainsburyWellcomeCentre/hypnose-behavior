# qc — quality control

Tools to run **after major changes** to confirm the pipeline output is unchanged
(or to mark exactly what changed, if intended). Run them in the pinned conda env
recorded in [`fixtures/env.json`](fixtures/env.json). They read the real,
read-only `rawdata` and redirect all derivatives I/O to a throwaway temp dir
(`HYPNOSE_DERIVATIVES_ROOT`), so they never write to the server.

| Tool | Purpose |
| --- | --- |
| [`regression.py`](regression.py) | Golden-master value check: `trial_data` + metrics vs stored fixtures |
| [`verify_scripts.py`](verify_scripts.py) | Same, but through the actual `scripts/` CLIs (covers arg wiring) |
| [`plot_regression.py`](plot_regression.py) | Old-vs-new check of what the **plotters draw** — `regression.py` never sees a figure |
| [`figure_provenance.py`](figure_provenance.py) | Does a saved figure's provenance record still name the plotter that drew it? (`DECISIONS.md` §9) — the one gate that exercises `save_figure`, which `plot_regression` deliberately never calls |
| [`check_imports.py`](check_imports.py) | Static check: flag any referenced global that isn't imported |
| [`check_layering.py`](check_layering.py) | Static check: the module graph is a DAG, and every directory-level cycle is a declared decision — reads source with `ast`, so **no mount and no imports needed** |
| [`check_qlearning.py`](check_qlearning.py) | Structural self-check of the Q-learning null model — **synthetic data only, no mount needed** |
| [`ast_move_check.py`](ast_move_check.py) | AST byte-identity check for a pure move: every definition reappears with a byte-identical source segment |
| [`verbose_diff.py`](verbose_diff.py) | Old-vs-new diff of what `trial_classification` **prints** — `regression.py` never looks at stdout |
| [`outcome_agreement.py`](outcome_agreement.py) | How far the three independent rewarded/unrewarded/timeout derivations agree (`DECISIONS.md` §14) |
| [`position_data_lossless.py`](position_data_lossless.py) | Is `position_data` a lossless projection of the three per-trial blobs? (`DECISIONS.md` §24) |
| [`validate.py`](validate.py) | `validate_subject()` — pre-flight data-existence check used by the scripts |

## What `regression.py` checks

For each session in [`sessions.yml`](sessions.yml) it fingerprints the two
outputs that must not change:

- **`trial_data`** — md5 of the canonical CSV (sorted columns, no index; not
  parquet bytes, not the manifest/summary with their timestamps).
- **metrics** — md5 of the metrics dict returned by `run_all_metrics`.

Each fixture also stores a **per-column** md5 (trial_data) and a **per-metric-key**
md5 (metrics). The overall md5 is the pass/fail signal; on a mismatch these let
the report say exactly *what* changed:

```
[RED]  sub-053 20260528 metrics: expected 6ac8e236 got a1b2c3d4
      + added metric: false_response_rate
      ~ changed metric: choice_accuracy
```

## What `plot_regression.py` checks

`regression.py` fingerprints `trial_data` + the metrics dict, so every change
inside `visualization/` is invisible to it. `plot_regression.py` closes that gap:
it runs each plotter under the Agg backend against a git revision *and* the
working tree, then diffs every line's xy data, every collection's offsets, every
patch's geometry, the axis decoration and stdout.

```bash
$PY -u $QC/plot_regression.py                 # working tree vs HEAD
$PY -u $QC/plot_regression.py --ref f72d201   # ... vs any revision
$PY -u $QC/plot_regression.py --only plot_decision_accuracy
```

**Deliberately not a golden master.** Figures are meant to change as the plotters
evolve, so a stored fixture would be stale within a phase. The useful question is
always "did *this* change move a curve", which is a two-tree diff.

Several plotters jitter points with the global RNG and never seed it, so the
harness seeds it before every call; without that the comparison is noise.

## Standard workflow

```
1. (baseline) add/keep the sessions you care about in sessions.yml, then capture
   the current known-good output:
       python src/hypnose_behavior/qc/regression.py --generate

2. make your major change(s)

3. compare against the baseline:
       python src/hypnose_behavior/qc/regression.py        # exit 0 = GREEN, 1 = RED

4. GREEN  -> commit.
   RED    -> read the +/-/~ lines:
              * unintended change -> fix it.
              * intended change   -> regenerate the affected fixtures deliberately
                                     (step 1) in the SAME commit, so the new
                                     baseline and the change land together.
```

Run `verify_scripts.py` and `check_imports.py` as additional gates the same way.

## `check_qlearning.py` — the odd one out

Unlike the tools above it reads **no data at all**: it generates every sequence it uses, so it
runs anywhere, including with the mount disconnected. It is not a golden master either — there
is nothing to regenerate. It asserts the *structural* properties of
`hypnose_behavior.modelling.switchpoint.qlearning`, the things that must hold of the model whichever
animal it is fitted to, and that a plausible-looking fit would hide if they broke:

1. the closed-form value recursion equals the trial-by-trial simulation exactly;
2. parameters are recovered from simulated data — in the regime where they are identifiable,
   with the fast-learner regime asserted on the trajectory only, and *not* on `alpha`;
3. `qlearn_constrained` cannot hold P(SHORT) below 0.5 in steady state, and `qlearn_free` can;
4. the reward scale is unidentifiable (rewards `(1, 0)` and `(1, 1−d)` with `b` scaled by
   `1/d` give the same nll — pointwise, at the ML fit, and at an independent refit's optimum);
5. `kappa = 0` reduces the perseveration variant to `qlearn_free`, which it nests.

```
python src/hypnose_behavior/qc/check_qlearning.py            # exit 0 = PASS, 1 = FAIL
python src/hypnose_behavior/qc/check_qlearning.py --seed 7   # a different synthetic draw
```

Tolerances are the worst case over 10 synthetic seeds per setting, with headroom; tightening
them makes the check flaky rather than stricter.

## Adding / removing sessions

- **Add a session:** put `{subjid, date, label}` in `sessions.yml`, then generate
  its fixture. You can limit `--generate` to specific `subjid:date` keys so you
  don't recompute the others:
  ```
  python src/hypnose_behavior/qc/regression.py --generate 060:20260601
  ```
  Generate on a **known-good baseline** (before your changes) — `--generate`
  records whatever the current code produces.
- **Remove a session:** delete it from `sessions.yml`. Its fixture file is left in
  `fixtures/` but ignored (compare/generate only act on sessions in `sessions.yml`,
  intersected with any keys you pass). Delete the orphan fixture by hand if you like.

## Notes

- All pipeline imports live in one block in [`_common.py`](_common.py) — the single
  place to update if module locations change; the md5s must not change as a result.
- Fixtures are only valid in the environment recorded in `fixtures/env.json`
  (float formatting depends on pandas/numpy versions). Regenerate if you change env.
- `check_imports.py` skips the runnable qc tools in its default scan; pass a module
  or file path to check anything specifically.
