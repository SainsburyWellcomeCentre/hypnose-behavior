# Sudden vs. gradual learning in a 2AFC odor discrimination task

**Analysis plan.** Adapted from Rosenberg et al. (2021), *eLife* 10:e66175, "Mice in a labyrinth exhibit rapid learning, sudden insight, and efficient exploration."

---

## 0. Framing

### 0.1 Three variables

All three are in the data.

| Variable | Measured over | Tells us |
|---|---|---|
| P(successful initiation) | number of attempts | when the mouse learned to hold the poke |
| Accuracy | completed (= initiated) trials | when the mouse learned the odor→port association |
| Engagement | time | how quickly/often the mouse initiates — task engagement, thirst/motivation |

### 0.2 Initial plot

Cumulative rewards (y) over consecutive time (x). This combines two factors:

```
r(t) = λ(t) × p(t)
       trials/s  accuracy
```

Any bend in the raw cumulative reward curve could be due to increased engagement or increased accuracy. Everything below exists to differentiate them.

### 0.3 Hypothesis

Mice can experience moments of insight, reflected in sudden increases in performance. Consolidation of this insight should happen overnight. Thus changepoint(s) should fall at session boundaries rather than within a session.

### 0.4 Difference to Rosenberg

Rosenberg had events in continuous time (mouse exploring) and used an inhomogeneous Poisson process. Data here is structured in trials with binary outcomes (correct/incorrect), which requires Bernoulli modelling.

The same modelling approach nonetheless applies:

```
r(t) = r_i + ((r_f − r_i)/2) · [1 + erf((t − t_s)/w)]

erf(x) = (2/√π) ∫₀ˣ e^(−u²) du
```

Fit an S-shaped curve to the rate of rewards over time and determine a changepoint where the animal changed its behaviour to obtain more rewards. This change must be shown to be due to a change in **knowledge** (odor–port association) and not **engagement** (more trials per time, or initiating more trials).

### 0.5 Diagnostics

**D1 — Successful initiation rate.** Plot #successful initiations / #attempts per session.

Answers whether the animal first learns to initiate trials and then the A/B rule, or whether initiation learning is a confounding factor throughout the data.

**D2 — Accuracy in non-initiated vs completed trials.** Odor is delivered on short pokes, so port choice on a non-initiated trial is still a readout of the association (odor A delivered → went to port A). Plot both per session and compare across learning.

**D3 — Lose-shift check.** Because odors repeat until successful initiation, a mouse could in principle solve a trial by elimination: fail to initiate on odor A, visit port B, get nothing, then initiate on the repeated A and go to A — correct, without knowing the rule.

This requires a failed initiation with a port visit followed immediately by a successful initiation on the same odor, which should be rare and not systematic. **Count how often it actually occurs first.** If non-negligible, split completed trials and compare:

| Subset | Condition |
|---|---|
| A | no prior failed attempt on this odor |
| B | prior failure, no port visited |
| C | prior failure, incorrect port visited |

If accuracy in C exceeds A and B, elimination is doing real work.

### 0.6 Logistic regression

Models log-odds rather than probability, so the linear right-hand side cannot predict impossible values. Chance (p = 0.5) is logit 0

Asks 2 main questions: 
- Do sessions differ in their baseline performance --> ⍺_{sessions(k)}
- Is there a systematic improvement or deteriation within a session --> β

This fits a straight line to each session's log odds, meaning a sigmoid per session on the probability scale. The best fit is describing data best. 

**α — session as a factor.** One free number per session, not a slope. A linear term in session number would force a fixed daily improvement, which assumes gradual learning — the thing being tested. A factor lets each day land wherever it lands.

**β — within-session slope.** β > 0 = warm-up; β < 0 = satiation; β ≈ 0 = flat.

**Normalized trial position** for the per-session models:

```
x_{s,k} = (k − 1)/(K_s − 1)
```

x = 0 at the first trial of a session, x = 1 at the last, independent of whether the session has 20 or 120 trials.

#### Three nested models

| Model | Form | Says |
|---|---|---|
| M_a | logit p = α_s | all gain is between sessions |
| M_b | logit p = α_s + β·x_{s,k} | one shared within-session slope |
| M_c | logit p = α_s + β_s·x_{s,k} | per-session within-session slopes |

Hypothesis predicts **M_a is sufficient** — adding within-session slopes should not improve fit. All three are nested and nothing is unidentified under the null, so the **likelihood ratio test is valid here** (unlike the changepoint case in §2).

**The hypothesis test is M_c vs M_a (df = N), not M_b vs M_a.** M_b can hide a real effect through cancellation: positive within-session learning early and negative satiation late average toward β ≈ 0, so M_b vs M_a comes back non-significant while genuine within-session dynamics exist. M_b remains useful as a descriptive average, not as the test.

**The hypothesis is the null, so do not rest on p > 0.05.** Failing to reject may just mean low power with short sessions. Report W_s with confidence intervals and show they are small in absolute terms; the rigorous version is equivalence testing against a pre-specified smallest gain of interest. Lead instead with the positive quantitative claim from the decomposition below — *"overnight accounted for X% of total improvement"*.

#### Decomposition from M_c

```
L_start,s   = α_s
L_end,s     = α_s + β_s
W_s         = L_end,s − L_start,s          = β_s              (within-session gain)
O_s         = L_start,s+1 − L_end,s        = α_{s+1} − α_s − β_s   (overnight gain)
```

The decomposition is **exact and additive** — the β terms telescope:

```
total gain = (α_N + β_N) − α_1 = Σ W_s + Σ O_s
```

So report a single headline number: *X% of total improvement occurred overnight*. Plot W_s and O_s across sessions.

#### Cautions

- **β_s is a gain, not a rate.** Normalization means a 20-trial and a 120-trial session both sweep x from 0 to 1. Comparable as total gains, not as learning speed.
- **O_s has a larger SE than it looks.** It contrasts two edge predictions — the fitted end of session s and fitted start of s+1 — the least-constrained points of the regression. Get its SE from the full covariance matrix (`m.t_test()`), not by adding component SEs.
- **Short sessions give unstable β_s.** At ~20 trials that slope is nearly unidentified. Consider a minimum-trial cutoff or partial pooling.


```python
import statsmodels.formula.api as smf
m_a = smf.logit('correct ~ C(session)', data=df).fit()
m_b = smf.logit('correct ~ C(session) + x', data=df).fit()
m_c = smf.logit('correct ~ C(session) * x', data=df).fit()
```

Fit per animal. Pool with a per-animal random effect only for a group-level statement (for now, no pooling across animals).

### 0.7 Within- and across-session gain

- Accuracy over last N trials of session *j* vs first N trials of session *j+1* → **across-session gain**
- Accuracy over first N vs last N trials of session *j* → **within-session gain**
- Compare

At N = 20, SE on a proportion ≈ 0.10, so individual boundary comparisons are noisy. Aggregate across boundaries and animals; do not interpret single comparisons. This is the descriptive companion to 0.6, which carries the argument.

Additional Plot: pool all sessions per animal, and plot the accuracy binned by within-session position (1 plot non normalized, binning trials 0-10, 11-20 or similar, depending on window size, and one normalized, binning first 10% etc.)
This plot shows if there is a within session gain effect per animal, e.g., quick ramp-up in gain early in session etc. This can change interpretation of β values in 0.6 (non-zero beta might be just ramp-up early in session). 

---

## 1. Visualization

### 1.1 Panel Figure

One plot per animal, three stacked panels, shared x-axis `t_cumulative`.

| Panel | y | Allows reading |
|---|---|---|
| A | cumulative trial initiations | engagement / change in engagement |
| B | excess correct, plotted at each trial's timestamp | knowledge, chance-corrected |
| C | cumulative rewards | the product of A and B |

Session boundaries as dotted vertical lines.

**Excess correct:**

```
excess(k) = Σ_{j≤k} y_j − 0.5·k        i.e. np.cumsum(y − 0.5)
```

Correct steps up 0.5, incorrect steps down 0.5. Chance produces a flat line. Slope = p − 0.5.

Plotting excess(k) at timestamp t_k is a coordinate lookup — no model, no posterior.

---

### 1.2 Smith et al. (2004) state-space model

Optional replacement for moving average figure. trial-by-trial accuracy with credible intervals, without choosing window width. Adds citeable learning-trial criterion (what trial "learning" first appears --> first trial where lower credible bound on p_k exceeds 0.5). Assumption-free about chape, so run before sigmoid. Similar shape-check as excess correct figure in 1.1 B. 

*Smith, Frank, Wirth, Yanike, Hu, Kubota, Graybiel, Suzuki & Brown, J Neurosci 24:447.*

```
x_k = x_{k−1} + ε_k,     ε_k ~ N(0, σ²)
y_k ~ Bernoulli( logistic(x_k + μ) )
```

**Implementation:** `GaussianRandomWalk` latent + `Bernoulli` likelihood + NUTS. Use a **non-centered parameterization** or you'll hit funnel geometry. ~1000 latent states takes a few minutes to sample.

Data: the binary correct/incorrect sequence over trial index, per animal.

Could be done with: 
with pm.Model() as m:
    sigma = pm.HalfNormal('sigma', 0.5)
    z = pm.Normal('z', 0, 1, shape=K)              # non-centered
    x = pm.Deterministic('x', pt.cumsum(sigma * z))
    pm.Bernoulli('y', logit_p=x, observed=y)       # mu = 0 for 2AFC
and 
cert = (idata.posterior['x'].stack(s=('chain','draw')).values > 0).mean(axis=1)

## 2. Changepoint fitting

### 2.1 Model

Following Rosenberg's modelling, in Bernoulli form:

```
p(k) = p_i + ((p_f − p_i)/2) · [1 + erf((k − k_s)/w)]
```

| Term | Meaning |
|---|---|
| p_i | initial accuracy (0.5, chance) |
| p_f | asymptotic accuracy |
| k | current trial |
| k_s | trial at which learning happens — midpoint of the rate change |
| w | transition width in trials — the measure of suddenness |

**Interpreting w:**
- span [k_s − w, k_s + w] covers 84% of the change
- 10%→90% rise = 1.81·w
- w → 0 is a perfect step

### 2.2 Likelihood

For the binary Bernoulli sequence:

```
ln L = Σ_k [ y_k · ln p(k) + (1 − y_k) · ln(1 − p(k)) ]
```

### 2.3 Fitting

- Parameterize `logit(p_i)`, `logit(p_f)`, `k_s`, `log(w)` to enforce constraints.
- Multi-start, or a coarse grid over (k_s, w) — the surface is multimodal.
- Consider fixing p_i = 0.5 (3 free parameters) and comparing against the free-p_i fit.
- w below the spacing of the data is unresolvable and the likelihood goes flat. `ŵ ≈ 0` means "faster than the data can see," not "instantaneous."


**What this model cannot do:** exactly one bend, monotone, symmetric transition, flat at both ends. Two real changes are silently fit as one compromise — often with inflated w, which would be misread as evidence against sudden learning.

Note: if 0.6 finds non-zero β_s, there is within-session structure. This is not modelled by the sigmoid and will add variance (rather than bias). 

### 2.4 The step model (timing + uncertainty)

The sigmoid gives k_s, a single trial describing where the rate of change is maximal. But it carries no usable uncertainty on that estimate: w is entangled with k_s (a shift in k_s can be absorbed by a wider w), so the statistical precision on k_s cannot be read off directly (cannot say the switch is xxx±x trials).

The posterior SD (calculated below) describes how well localized the switch is given how much data there is.

A step model assumes an instantaneous transition, so it is only run on animals with sufficiently small w (Rosenberg et al.: w < 300 s; here: set an analogous threshold in trials).

#### Model

Identical to a standard Bernoulli switchpoint model, with k_s the first trial of the post-switch regime:

```
p(k) = p_i   for k < k_s
       p_f   for k >= k_s
```

#### Closed-form MLEs

Fix k_s and the data splits into two blocks, each with constant p.

MLE:

```
p_i_hat = c1/n1     p_f_hat = c2/n2
```

n1, c1 = trials and correct before k_s; n2, c2 from k_s onward.

Derivation for block 1: ln L1 = c1*ln p_i + (n1 - c1)*ln(1 - p_i); setting the derivative to zero gives c1(1 - p_i) = (n1 - c1)p_i, hence p_i = c1/n1.

#### Profile likelihood

Substituting the MLEs back:

```
l(k_s) = -[ n1*H(p_i_hat) + n2*H(p_f_hat) ]

H(p) = -p*ln p - (1-p)*ln(1-p)          (binary entropy, nats)
```

H is unpredictability: H(0.5) = ln 2 ~ 0.693 (maximum, coin flip), H(0) = H(1) = 0 (perfectly predictable).

Maximizing l means **minimizing total unpredictability** --> each side of the switchpoint is internally as consistent as possible. The maximum gives the point estimate k_s_hat.

#### Implementation

Vectorizes with cumulative sums; no loop over candidates:

```python
c1 = np.cumsum(y)[:-1]
n1 = np.arange(1, len(y))
c2, n2 = y.sum() - c1, len(y) - n1
```

p_hat, H and l are then elementwise.

- Convention: 0*ln 0 = 0 (arises whenever a block is all-correct or all-incorrect).
- Restrict the grid to exclude the first and last ~10 trials.

### 2.5 Posterior over the changepoint

Treat the normalized likelihood as a probability distribution over k_s (flat prior). Plot P(k_s) against trial index; its spread is the uncertainty in the estimate.

Marginalize p_i and p_f with Beta(1,1) priors rather than fixing them at their MLEs. Profiling treats the two accuracies as known exactly, which ignores their uncertainty and understates the posterior width — the wrong direction of error when the width *is* the deliverable.

```
ln P(k_s) = lnB(c1+1, n1-c1+1) + lnB(c2+1, n2-c2+1)
lnB(a,b)  = lnGamma(a) + lnGamma(b) - lnGamma(a+b)

P(k_s)    = exp(ln P(k_s)) / sum_k exp(ln P(k))
<k_s>     = sum_k P(k)*k
SD        = sqrt( sum_k P(k)*k^2 - <k_s>^2 )
```

**Always subtract the max before exponentiating**, or exp() overflows.

Deliverable: *k_s = trial X +/- Y trials*.

Note: check for multimodal posteriors --> multimodality means more than 1 switchpoint. Flag it and report. 

### 2.6 Sanity check

Fit a constant-accuracy model (M0: p(k) = p0, one parameter, same Bernoulli likelihood) and report the log-likelihood difference against the sigmoid.

Guards against interpreting k_s and w from an animal that never changed. A broad, flat P(k_s) in 2.5 says the same thing and is the primary evidence; this is the one number to quote alongside it.



## 3. GLM-HMM for states

Fit on choice, not accuracy. Pooled emissions across animals and per animal transition. 2 variants: recurrent states (**strategy model**: engaged/biased/disengaged) and 2 state knowledge (**knowledge model**: low odor weight vs high odor weight, possibly non-recurrent transition). 

*Ashwood, Roy, Stone, IBL, Urai, Churchland, Pouget & Pillow, Nat Neurosci 25:201 (2022).* Code: `github.com/zashwood/glm-hmm`; `ssm` from the Linderman lab.

**Fitting strategy: pooled emissions, per-animal transitions.** The *repertoire* of strategies is assumed shared across animals; which strategies each animal uses and when is individual.

1. Fit a global model on all animals' trials pooled → GLM weights defining each state.
2. Fix those weights; fit per-animal transition matrices and state sequences.

This handles heterogeneity directly — an animal that never disengages simply has ~0 posterior probability of that state throughout. It also makes ~1000 trials/animal workable; per-animal emission fitting would overfit at that size.

**Inputs.** An HMM consumes a sequence plus a per-trial design matrix. Model **choice**, not accuracy — a "biased left" state ignores the odor and goes left regardless, which is visible in choices and invisible in correct/incorrect.

- `y`, shape (n_trials,): choice, 0/1
- `X`, shape (n_trials, 4)
- session boundaries, so days are separate sequences

| Regressor | Coding | Weight means |
|---|---|---|
| odor identity | +1 A, −1 B | **sensitivity** — the key parameter |
| bias | constant 1 | side bias |
| previous choice | ±1 | perseveration / alternation |
| prev reward × prev choice | ±1 | win-stay / lose-shift |

**Critical:** previous-choice and previous-reward must refer to the previous **attempt**, including failed initiations — otherwise the model cannot see the elimination strategy from D3 and will attribute its effects to the odor weight, the very parameter being interpreted.

#### Two models, two hypotheses

| | 3a Strategy model | 3b Knowledge model |
|---|---|---|
| States | engaged / biased-L / biased-R / disengaged | naive (low odor weight) / expert (high) |
| Transitions | recurrent, both directions | forward-preferring |
| Learning appears as | growing **occupancy** of the engaged state | one-way **arrival** at a better state |
| Answers | does engagement fluctuate? | when was the rule acquired? |

**"Did the animal get better, or get better at paying attention?"** Ashwood found apparent gradual learning in IBL mice was substantially explained by rising engaged-state occupancy rather than improving performance within it. Your two models are the two hypotheses; compare them.

**Notes:**
- 2-state forward-only + Bernoulli emissions is nearly the same object as the §2.4 step model — a changepoint model in HMM clothing, with a soft geometric dwell time and free per-trial state posteriors.
- **Fit 3b unconstrained first.** If it never reverts, that is *stronger* evidence for stable learning than forbidding reversion by construction. If it does revert, that is a real finding — unstable knowledge — that the constraint would hide. Constrain only if the unconstrained fit is unstable.
- **Verify the two states differ in odor weight, not merely bias.** A 2-state fit easily splits into biased-left / biased-right, a strategy result masquerading as a knowledge result.
- **Boundary-constrained variant:** make the transition matrix time-varying — A = I at non-boundary trials, learnable A at boundaries. Compare against unconstrained. 

---

## 4. Full-data extension

Everything above uses completed trials. Once established, redo at **attempt level**.

**Justification for starting with completed trials:** the association can only be learned from a completed trial, so completed-trial index is arguably the correct *experience axis* for association learning, not merely a convenient approximation. Total attempts would be the wrong denominator.

**But:** because odor is delivered on short pokes, non-initiated trials carry real information about rule knowledge — hence D2 and D3.

Attempt-level analyses:

1. **Initiation learning.** Bernoulli sigmoid on `initiated / not initiated` over attempt index. Identical machinery, different binary column, near-zero marginal cost. This is the third variable in §0.1 and converts your complication into evidence.
2. **False alarms.** Failing to initiate but still running to a port = knows reward is available, doesn't yet know how to earn it. A fourth, mechanistically interesting measure.
3. **Extinction test.** If correct-choice-without-reward acts as a negative signal, animals with higher non-initiation rates should learn more slowly. One correlation across 8–10 animals — underpowered but free.
4. **Exposure axis.** Odor *exposures* per completed trial falls over training. If exposure drives learning, exposure count may be a better experience axis than completed-trial count. Compute both; check whether k̂_s moves.

## Order of execution

| Block | Content | Section | Answers | Blocking? |
|---|---|---|---|---|
| 1 | Cumulative rewards over time | 0.2 | is there any bend at all? | no |
| 2 | D1 initiation rate, D2 non-initiated accuracy, D3 lose-shift count | 0.5 | is completed-trials-only defensible? | **yes** — determines whether the primary DV needs restricting |
| 3 | Logistic regression M_a / M_b / M_c + W_s/O_s decomposition | 0.6 | overnight vs within-session; % of gain overnight | no, but this is the main hypothesis test |
| 4 | Within- and across-session gain, N trials | 0.7 | descriptive companion to block 3 | no |
| 5 | Three-panel figure per animal | 1.1 | engagement vs knowledge, visually; shape check | **yes** — if panel B shows two bends, block 7 is invalid |
| 6 | Smith et al. state-space | 1.2 | trial-by-trial accuracy with CIs; shape check | no — optional, but run before block 7 |
| 7 | Bernoulli sigmoid fit | 2.1–2.3 | **how sudden** — ŵ | core |
| 8 | Step model + posterior over k_s | 2.4–2.5 | **when, ± how much** | core; run only if ŵ small |
| 9 | Constant-model ΔlnL | 2.6 | did anything change at all? | one number, quote with block 8 |
| 10 | GLM-HMM: strategy model + knowledge model | 3 | occupancy of a good state vs arrival at a better one | later |
| 11 | Attempt-level re-run | 4 | initiation learning, false alarms, exposure axis | later |

Blocks 1–9 are the figure. 10–11 are the deeper account.