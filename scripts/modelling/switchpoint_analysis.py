#!/usr/bin/env python
"""Switch-point analysis of the LONG -> SHORT strategy change, per animal.

Entry points and CLI only -- the numeric models, stats and figures live in ``src/``:

- ``hypnose_behavior.modelling.switchpoint``                 -- data prep, model fits, comparison,
  permutation and autocorrelation maths (numpy in, dicts out).

- ``hypnose_behavior.visualization.modelling.switchpoint``   -- every figure.

Here: 

- ``run_analysis``                -- per-animal switch-point fit, posterior, and model comparison
  (with the three Q-learning null trajectories overlaid unless turned off).

- ``run_qlearning_sweep``         -- one ``(alpha, b)`` parameter-sweep figure per Q-learning
  variant (standalone; three figures per sequence).

- ``run_permutation``             -- do switches sit closer to *real* sleep boundaries than to
  other animals' donated ones?

- ``run_logistic_diagnostic``     -- where each of the logistic's multi-start initial conditions
  converges (standalone; nothing else depends on it).

- ``run_residual_autocorrelation``-- the i.i.d.-Bernoulli check behind the planned bootstrap.


Examples
--------
  python scripts/modelling/switchpoint_analysis.py analysis --subjids 40 --likelihood-window 100
  python scripts/modelling/switchpoint_analysis.py analysis --subjids 40 41 --date-range 20251201 20251231 --rewarded-only
  python scripts/modelling/switchpoint_analysis.py qsweep --subjids 40 --rewarded-only --split-ab
  python scripts/modelling/switchpoint_analysis.py permutation --subjids 40 41 42 --rewarded-only
  python scripts/modelling/switchpoint_analysis.py diagnostic --subjids 40 --rewarded-only --split-ab
  python scripts/modelling/switchpoint_analysis.py autocorr --subjids 40 --rewarded-only
"""
from __future__ import annotations

import sys
import argparse
from pathlib import Path
from typing import Iterable, Optional, Sequence, Union

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np
import matplotlib.pyplot as plt

from hypnose_behavior.io.save import nature_style, save_figure
from hypnose_behavior.qc.validate import validate_subject
from hypnose_behavior.modelling.switchpoint import (
    ACF_MATERIAL_THRESHOLD,
    ACF_MAX_LAG,
    AB_LETTERS,
    MODEL_ORDER,
    N_STARTS,
    QLEARN_DEFAULT_VARIANT,
    QLEARN_SWEEP_ALPHAS,
    QLEARN_SWEEP_BS,
    QLEARN_VARIANT_ORDER,
    SWITCH_THRESHOLD,
    WARM_START_LABEL,
    acf_bounds,
    compare_models,
    distance_to_session_start,
    fit_logistic_multistart,
    fit_qlearning_variants,
    fit_switchpoint,
    model_fitted_p,
    normalize_subjids_dates,
    pairwise_f,
    permutation_null_means,
    prepare_subject,
    qlearning_generative_band,
    qlearning_parameter_sweep,
    residual_acf,
    subject_label,
    subset_by_ab,
)
from hypnose_behavior.visualization.modelling.switchpoint.plots import (
    plot_model_comparison,
    plot_multistart,
    plot_permutation,
    plot_posterior,
    plot_qlearning_generative,
    plot_qlearning_sweep,
    plot_residual_autocorr,
    plot_strategy,
)

# Report a start beating the warm start, or a nesting violation, only when the loglik gain is
# worth reading.
_LOGLIK_REPORT_TOL = 1e-3

# Which animals count as "has a switch" in run_permutation. Keys are the `inclusion` values.

#NOTE: update inclusion rules when updating the permutation plot. Deferred. 
_INCLUSION_RULES = {
    # Strictest: the 3-parameter switch beats BOTH the constant and the 4-parameter
    # logistic on BIC, i.e. the change is real *and* abrupt rather than a slow drift.
    "bic_switch_wins": lambda c: c["best_bic"] == "switch",
    # The change is real, but a gradual (logistic) description may fit it even better.
    "bic_beats_constant": lambda c: c["switch"]["bic"] < c["constant"]["bic"],
    # Same as bic_switch_wins but under the more permissive AIC penalty.
    "aic_switch_wins": lambda c: c["best_aic"] == "switch",
    # No filtering; every animal contributes a tau.
    "all": lambda c: True,
}

__all__ = ["run_analysis", "run_qlearning_sweep", "run_permutation", "run_logistic_diagnostic",
           "run_residual_autocorrelation"]


# --- per-animal switch-point fit ----------------------------------------------------------


def _describe_fit(name: str, fit: dict) -> str:
    """The fitted parameters of one model, as a short human-readable line."""
    if name == "constant":
        return f"p = {fit['p']:.3f}"
    if name == "switch":
        return f"tau = {fit['tau']}, p1 = {fit['p1']:.3f} -> p2 = {fit['p2']:.3f}"
    if name == "switch2":
        return (f"tau1 = {fit['tau1']}, tau2 = {fit['tau2']}, "
                f"p1 = {fit['p1']:.3f} -> p2 = {fit['p2']:.3f} -> p3 = {fit['p3']:.3f}")
    if name == "logistic":
        return (f"start = {fit.get('start_label', '?')}, midpoint = {fit['midpoint']:.1f}, "
                f"slope = {fit['slope']:.4g}, lo = {fit['lo']:.3f}, hi = {fit['hi']:.3f}")
    if name == "qlearning":
        return (f"variant = {fit['variant']}, alpha = {fit['alpha']:.4g}, b = {fit['b']:.4g}, "
                f"Q0 = ({fit['q0_short']:.3f}, {fit['q0_long']:.3f})")
    return "not implemented"


def _print_qlearning_table(qlearning_fits: dict, qlearning_bands: Optional[dict] = None) -> None:
    """Print the three Q-learning variants' estimates and scores -- the null being tested.

    A ``boundary`` tag names any estimate that stopped within 1% of a bound: the optimizer hit
    the edge of the parameter space, so that number is a constraint artefact rather than an
    estimate, and the variant's fit should be read as "as good as the bounds allow".

    With ``qlearning_bands`` a continuation line reports what the fit actually *generates*: the
    fraction of simulated runs that reach the switch criterion, and the spread of switch trials
    over those runs only. Both numbers are needed -- a low fraction and a wide spread are two
    different ways of failing to reproduce an animal that switched abruptly at one trial, and
    the tau spread alone cannot tell them apart.
    """
    print(f"  {'q-learning variant':<22}{'alpha':>9}{'b':>9}{'Q0_short':>10}{'Q0_long':>9}"
          f"{'kappa':>8}{'nll':>10}{'AIC':>10}{'BIC':>10}  starts")
    for variant in QLEARN_VARIANT_ORDER:
        fit = qlearning_fits[variant]
        kappa = f"{fit['kappa']:>8.3f}" if variant == "qlearn_perseveration" else f"{'-':>8}"
        flags = f"{fit['n_starts_converged']}/{fit['n_starts']}"
        if fit["boundary_hit"]:
            flags += f"  boundary: {', '.join(fit['boundary_params'])}"
        if not fit["converged"]:
            flags += "  NO START CONVERGED"
        print(f"  {variant:<22}{fit['alpha']:>9.4g}{fit['b']:>9.4g}{fit['q0_short']:>10.3f}"
              f"{fit['q0_long']:>9.3f}{kappa}{fit['nll']:>10.3f}{fit['aic']:>10.1f}"
              f"{fit['bic']:>10.1f}  {flags}")
        band = (qlearning_bands or {}).get(variant)
        if band and band["n_sims"]:
            taus = band["switch_taus_switched"]
            spread = (f"switched-run tau: {taus.min()}-{taus.max()} "
                      f"(5-95% {np.percentile(taus, 5):.0f}-{np.percentile(taus, 95):.0f})"
                      if taus.size else "no switched run to take a tau spread from")
            print(f"  {'':<22}generative: {band['frac_switched']:.1%} of {band['n_sims']} runs "
                  f"reach p2 - p1 >= {SWITCH_THRESHOLD:g}; {spread}")


def _print_model_table(comparison: dict) -> None:
    """Print every model's loglik / AIC / BIC, the winner's parameters, and the nesting check.

    The nesting relations (``constant <= switch`` and ``switch <= logistic``) are a property of
    the models, not of the data: a violation means a fit failed to find its optimum, so it is
    called out rather than left to be spotted in the numbers. ``switch2`` is monotone-gated and
    does not nest the single switch, so it is not part of the check.
    """
    print(f"  {'model':<10}{'k':>3}{'loglik':>12}{'AIC':>11}{'BIC':>11}")
    for name in MODEL_ORDER:
        fit, score = comparison["fits"][name], comparison[name]
        if not fit.get("implemented", True):
            print(f"  {name:<10}{score['k_params']:>3}{'n/a':>12}{'n/a':>11}{'n/a':>11}"
                  f"  (not implemented)")
            continue
        tags = " ".join(t for t, on in (("<-AIC", name == comparison["best_aic"]),
                                        ("<-BIC", name == comparison["best_bic"])) if on)
        # For the logistic, name the multi-start that won, so the chosen fit is unambiguous.
        if name == "logistic" and fit.get("start_label"):
            tags = f"{tags}  (start {fit['start_label']})".lstrip()
        print(f"  {name:<10}{score['k_params']:>3}{score['loglik']:>12.3f}{score['aic']:>11.1f}"
              f"{score['bic']:>11.1f}  {tags}")

    ll = {name: comparison[name]["loglik"] for name in ("constant", "switch", "logistic")}
    violations = [msg for msg, ok in (
        ("constant <= switch", ll["constant"] <= ll["switch"] + _LOGLIK_REPORT_TOL),
        ("switch <= logistic", ll["switch"] <= ll["logistic"] + _LOGLIK_REPORT_TOL)) if not ok]
    if violations:
        print(f"  nesting: VIOLATED -> {'; '.join(violations)} (a fit failed to find its optimum)")
    else:
        print("  nesting: OK (constant <= switch <= logistic; switch2 monotone-gated)")

    best = comparison["best_bic"]
    print(f"  best model (BIC): {best} -- {_describe_fit(best, comparison['fits'][best])}")


def _analyse_sequence(prep: dict, rewarded_only: bool, likelihood_window: int,
                      qlearning_overlay: bool = True, defer_figures: bool = False,
                      around_switch: bool = False, plot_trials: int = 200) -> Optional[dict]:
    """Fit, print and plot one SHORT/LONG sequence -- a whole animal, or one A/B split of it.

    Figures are built in the order they should be read: strategy, model comparison, posterior.
    With ``qlearning_overlay`` the three Q-learning variants are fitted to the same sequence and
    tabulated; their one-step-ahead fits are drawn on the model-comparison figure, and a fourth
    ``generative`` figure shows what each variant actually predicts (its own simulated runs, mean
    and band) against the observed switch. Returns None (after a message) when the sequence is
    too short to fit.

    With ``defer_figures`` the maths and printing happen as usual but no figure is created; the
    returned ``figures`` dict is left empty and the caller builds the figures later via
    ``_build_sequence_figure`` (used by ``run_analysis`` to interleave the A and B splits by
    figure kind rather than emit all of A's figures before any of B's).
    """
    label = subject_label(prep)
    if prep["n_trials"] < 2:
        print(f"[switchpoint] {label}: {prep['n_trials']} kept trial(s); skipping.")
        return None

    fit = fit_switchpoint(prep["s"])
    # Fit the Q-learning variants FIRST, then hand compare_models the one it scores: otherwise
    # it would run a second, identical multi-start of that same variant on the same sequence.
    qlearning_fits = fit_qlearning_variants(prep["s"]) if qlearning_overlay else None
    comparison = compare_models(
        prep["s"], qlearning_fits[QLEARN_DEFAULT_VARIANT] if qlearning_fits else None)
    qlearning_bands = ({v: qlearning_generative_band(f, prep["n_trials"])
                        for v, f in qlearning_fits.items()} if qlearning_fits else None)
    tau = fit["tau"]
    global_tau = int(prep["global_ids"][tau])
    tau_session = prep["session_labels"][int(prep["session_index"][tau])]
    hdi_lo, hdi_hi = fit["hdi"]
    fwhm_lo, fwhm_hi = fit["fwhm"]
    hdi_width, fwhm_width = hdi_hi - hdi_lo + 1, fwhm_hi - fwhm_lo + 1

    print(f"\n[switchpoint] {label} ({prep['n_trials']} trials, "
          f"{len(prep['session_labels'])} sessions)")
    on_split = prep.get("ab_split") is not None
    axis = f"tau (trial index) = {tau}" + (f", global trial id {global_tau}" if on_split else "")
    print(f"  {axis}, in session {tau_session}")
    print(f"  p1 = {fit['p1']:.3f} -> p2 = {fit['p2']:.3f}")
    print(f"  95% HDI = [{hdi_lo}, {hdi_hi}], width = {hdi_width} trials")
    print(f"  FWHM    = [{fwhm_lo}, {fwhm_hi}], width = {fwhm_width} trials")
    _print_model_table(comparison)
    if qlearning_fits:
        _print_qlearning_table(qlearning_fits, qlearning_bands)

    result = {
        "tau": tau, "global_tau": global_tau, "tau_session": tau_session, "hdi": fit["hdi"],
        "hdi_width": hdi_width, "fwhm": fit["fwhm"], "fwhm_width": fwhm_width,
        "p1": fit["p1"], "p2": fit["p2"], "posterior": fit["posterior"], "comparison": comparison,
        "qlearning": qlearning_fits, "qlearning_bands": qlearning_bands,
        "session_ends": prep["session_ends"], "session_starts": prep["session_starts"],
        "session_labels": prep["session_labels"], "n_trials": prep["n_trials"],
        "ab_split": prep.get("ab_split"), "prep": prep, "fit": fit, "figures": {},
    }
    if not defer_figures:
        for kind in _FIGURE_KINDS:
            fig = _build_sequence_figure(result, kind, rewarded_only, likelihood_window,
                                         around_switch, plot_trials)
            if fig is not None:
                result["figures"][kind] = fig
    return result


# The figure kinds one sequence produces, in read order. `generative` is skipped when the
# Q-learning overlay is off (its inputs are None); see `_build_sequence_figure`.
_FIGURE_KINDS = ("strategy", "model_comparison", "posterior", "generative")

# Base file name per figure kind for `save=True`. The reward letter (A/B) is prefixed for a
# split_ab sequence, giving e.g. "A_all_models", "B_qlearning_generative"; `save_figure` then
# appends the subject/date tags. Short and informative.
_FIGURE_SAVE_NAMES = {
    "strategy": "strategy",
    "model_comparison": "all_models",
    "posterior": "switch_posterior",
    "generative": "qlearning_generative",
}


def _save_sequence_figures(result: dict, subjid: int, dates) -> list:
    """Save each of a sequence's figures as its own PDF under the subject's figures directory.

    File names are ``<letter>_<kind>`` (the letter present only for an A/B split), e.g.
    ``A_all_models`` or ``B_qlearning_generative``; ``save_figure`` appends the subject/date tags
    and resolves the path -- subject-level (``derivatives/sub-NNN_id-*/figures``) for the
    multi-date ranges this analysis uses. Returns the saved paths.
    """
    letter = result.get("ab_split")
    prefix = f"{letter}_" if letter else ""
    return [save_figure(fig, f"{prefix}{_FIGURE_SAVE_NAMES.get(kind, kind)}",
                        subjids=[subjid], dates=dates)
            for kind, fig in result["figures"].items()]


def _build_sequence_figure(result: dict, kind: str, rewarded_only: bool, likelihood_window: int,
                           around_switch: bool = False, plot_trials: int = 200):
    """Create one figure of the given kind for a sequence analysed by ``_analyse_sequence``.

    Kept separate from the maths so ``run_analysis`` can interleave the A and B splits by figure
    kind. Returns None for ``generative`` when the Q-learning overlay was off for this sequence.
    ``around_switch`` / ``plot_trials`` crop the model-comparison figure's x-axis to the switch.
    """
    prep = result["prep"]
    if kind == "strategy":
        return plot_strategy(prep, rewarded_only)
    if kind == "model_comparison":
        return plot_model_comparison(prep, result["comparison"], result["qlearning"],
                                     around_switch=around_switch, plot_trials=plot_trials)
    if kind == "posterior":
        return plot_posterior(prep, result["fit"], likelihood_window)
    if kind == "generative":
        if result["qlearning"] is None:
            return None
        # Show the observed switch's 95% HDI band only when the switch model wins BIC -- i.e. when
        # there is a real abrupt switch to localise. Without one the posterior is flat and the HDI
        # spans essentially every trial, so the band would shade the whole panel meaninglessly.
        show_hdi = result["comparison"]["best_bic"] == "switch"
        return plot_qlearning_generative(prep, result["qlearning"], result["qlearning_bands"],
                                         result["fit"], show_hdi=show_hdi)
    raise ValueError(f"unknown figure kind {kind!r}")


def _show_figures(show: bool) -> None:
    """Render the figures built so far, preserving the order they were created in.

    In an interactive notebook backend (ipympl ``widget`` / ``inline``) every figure created in
    the cell is auto-displayed at cell end, in creation order, by the backend's ``post_execute``
    flush hook -- so we must NOT also call ``plt.show()`` there. ipympl's ``show()`` eagerly
    displays only the *active* (last-created) figure mid-execution and drops it from the flush
    queue, which floats it above all the others -- so the last-built figure would appear
    first. For non-interactive/GUI backends (the CLI) nothing auto-displays, so
    ``plt.show()`` is required.
    """
    if not show:
        return
    backend = plt.get_backend().lower()
    notebook = any(tag in backend for tag in ("ipympl", "nbagg", "inline")) or backend == "widget"
    if notebook and plt.isinteractive():
        return  # the flush hook will display them in creation order
    plt.show()


def run_analysis(
    subjids: Union[int, Iterable[int], dict],
    date_ranges: Optional[dict] = None,
    rewarded_only: bool = False,
    likelihood_window: int = 100,
    split_ab: bool = False,
    show: bool = True,
    qlearning_overlay: bool = True,
    around_switch: bool = False,
    plot_trials: int = 200,
    save: bool = False,
) -> dict:
    """Fit and plot the strategy switch for each subject independently.

    Produces three figures per animal.

    Parameters
    ----------
    subjids : int | list[int] | dict
        Subject id(s). May also be a ``{subjid: date_range}`` dict as a shorthand, in which
        case ``date_ranges`` may be omitted.
    date_ranges : dict | None
        ``{subjid: date_range}``, each value an inclusive ``(start, end)`` ``YYYYMMDD``
        tuple, an explicit date list, or ``None`` for all sessions. A non-dict value is
        applied to every subject.
    rewarded_only : bool
        Keep only ``response_time_category == "rewarded"`` trials (always excludes aborts).
    likelihood_window : int
        Half-width, in trials, of the posterior plot's window around the peak.
    split_ab : bool
        Analyse the A- and B-reward trials separately: each subset gets its own contiguous
        trial axis, its own fits, and its own three figures.
    show : bool
        Display each animal's figures once built (default True).
    qlearning_overlay : bool
        Fit the three Q-learning variants (default True). 
    around_switch : bool
        Crop each model-comparison figure's x-axis to ``plot_trials`` trials either side of the
        switch tau (default False, showing the whole trial axis). Only the view changes; the fits
        are unaffected.
    plot_trials : int
        Half-width in trials of the ``around_switch`` crop (default 200).
    save : bool
        Save every figure to disk (default False). Each is written as its own PDF.

    Returns
    -------
    dict
        Keyed by subjid. Each value holds ``tau``, ``global_tau``, ``tau_session``, ``hdi``,
        ``hdi_width``, ``fwhm``, ``fwhm_width``, ``p1``, ``p2``, ``comparison``, ``qlearning``
        (the three variant fits, or None when the overlay is off), ``qlearning_bands`` (each
        variant's ``qlearning_generative_band``, or None likewise), ``session_ends``,
        ``session_starts``, ``session_labels``, ``n_trials``, ``ab_split``, ``prep``, ``fit``
        (the switch-point fit), and ``figures`` (``strategy``, ``model_comparison``,
        ``posterior``, and -- when the overlay is on -- ``generative``).

        With ``split_ab=True`` that value is instead nested one level deeper, keyed by reward
        identity: ``results[subjid]["A"]`` and ``results[subjid]["B"]``.
    """
    subjids, date_ranges, dates_for = normalize_subjids_dates(subjids, date_ranges)
    results = {}

    with plt.rc_context(nature_style()):
        for subjid in subjids:
            prep = prepare_subject(subjid, dates_for(subjid), rewarded_only)
            if split_ab:
                unresolved = int(np.sum(~np.isin(prep["ab"], list(AB_LETTERS))))
                if unresolved:
                    print(f"[switchpoint] Subject {subjid}: {unresolved} trial(s) of "
                          f"{prep['n_trials']} have no reward identity; excluded from the split.")
                splits = {letter: _analyse_sequence(subset_by_ab(prep, letter), rewarded_only,
                                                    likelihood_window, qlearning_overlay,
                                                    defer_figures=True)
                          for letter in AB_LETTERS}
                splits = {letter: r for letter, r in splits.items() if r is not None}
                if splits:
                    for kind in _FIGURE_KINDS:
                        for letter, r in splits.items():
                            fig = _build_sequence_figure(r, kind, rewarded_only, likelihood_window,
                                                         around_switch, plot_trials)
                            if fig is not None:
                                r["figures"][kind] = fig
                    results[subjid] = splits
            else:
                result = _analyse_sequence(prep, rewarded_only, likelihood_window,
                                           qlearning_overlay, around_switch=around_switch,
                                           plot_trials=plot_trials)
                if result is not None:
                    results[subjid] = result
            # Save this subject's figures (one PDF each) before display. A stored value is either a
            # single sequence (has a "figures" key) or a {letter: sequence} split.
            if save and subjid in results:
                stored = results[subjid]
                seqs = [stored] if "figures" in stored else list(stored.values())
                saved = [p for seq in seqs
                         for p in _save_sequence_figures(seq, subjid, dates_for(subjid))]
                if saved:
                    print(f"[switchpoint] Subject {subjid}: saved {len(saved)} figure(s) to "
                          f"{saved[0].parent}")
            # Per animal: build figures for this subject, then display. In the notebook the
            # flush hook renders them in creation order; see _show_figures.
            _show_figures(show)
    return results


# --- Q-learning parameter sweep (standalone) ----------------------------------------------
#
# The overlay in run_analysis shows only where each variant's likelihood peaked. This shows the
# surrounding neighbourhood: what a Q-learner of that variant looks like across a grid of
# learning rates and inverse temperatures, with everything else held at the ML fit. It is the
# visual form of the argument the null is there to make -- alpha controls how fast P(SHORT)
# rises and b how far it travels, so if no grid point produces a step, incremental value
# learning cannot account for an abrupt switch however it is tuned.


def _sweep_sequence(prep: dict, alphas: Sequence[float], bs: Sequence[float],
                    n_starts: int, seed: int) -> Optional[dict]:
    """Fit all three variants for one sequence and build one sweep figure per variant."""
    label = subject_label(prep)
    if prep["n_trials"] < 2:
        print(f"[qsweep] {label}: {prep['n_trials']} kept trial(s); skipping.")
        return None

    fits = fit_qlearning_variants(prep["s"], n_starts=n_starts, seed=seed)
    print(f"\n[qsweep] {label} ({prep['n_trials']} trials, {len(alphas)} x {len(bs)} "
          f"(alpha, b) grid per variant)")

    sweeps, bands, figures = {}, {}, {}
    for variant in QLEARN_VARIANT_ORDER:
        # generative=True: with kappa held at ML across the grid, one-step-ahead lines would all
        # step at the animal's switch trial regardless of alpha and b, which is the opposite of
        # what this figure is for.
        sweeps[variant] = qlearning_parameter_sweep(prep["s"], fits[variant], alphas, bs,
                                                    generative=True)
        bands[variant] = qlearning_generative_band(fits[variant], prep["n_trials"])
        figures[variant] = plot_qlearning_sweep(prep, variant, fits[variant], sweeps[variant],
                                                alphas, bs, bands[variant])
        # The ML fit must be at least as good as any grid point of the same variant -- the grid
        # lies inside the search space. A grid point beating it means the multi-start missed.
        best_grid = min(point["nll"] for point in sweeps[variant])
        if best_grid < fits[variant]["nll"] - _LOGLIK_REPORT_TOL:
            print(f"  NOTE: {variant}: a grid point reached nll {best_grid:.3f} < the ML fit's "
                  f"{fits[variant]['nll']:.3f} -- the multi-start missed the optimum.")

    _print_qlearning_table(fits, bands)

    return {"fits": fits, "sweeps": sweeps, "bands": bands, "alphas": tuple(alphas),
            "bs": tuple(bs), "ab_split": prep.get("ab_split"), "prep": prep, "figures": figures}


def run_qlearning_sweep(
    subjids: Union[int, Iterable[int], dict],
    date_ranges: Optional[dict] = None,
    rewarded_only: bool = False,
    split_ab: bool = False,
    alphas: Sequence[float] = QLEARN_SWEEP_ALPHAS,
    bs: Sequence[float] = QLEARN_SWEEP_BS,
    n_starts: int = N_STARTS,
    seed: int = 0,
    show: bool = True,
) -> dict:
    """Draw one Q-learning ``(alpha, b)`` parameter-sweep figure per variant, per animal.

    ``alpha`` is mapped to colour and ``b`` to linestyle, each with its own legend.

    Parameters
    ----------
    subjids, date_ranges, rewarded_only, split_ab, show
        As in ``run_analysis``. With ``split_ab`` the A- and B-reward trials are partitioned by
        the same rule and each subset gets its own fits and its own three figures -- never a
        pooled fit across the two.
    alphas, bs : sequence of float
        The sweep grid. Defaults are ``QLEARN_SWEEP_ALPHAS`` / ``QLEARN_SWEEP_BS`` (4 x 4);
        more than four of either makes the linestyle/colour mapping unreadable.
    n_starts : int
        Random starting points per fit (>= 20; see ``fit_qlearning``).
    seed : int
        Seed for the multi-start draw, so the fits are reproducible.

    Returns
    -------
    dict
        Keyed by subjid (nested by ``"A"`` / ``"B"`` when ``split_ab``): ``fits`` (per variant),
        ``sweeps`` (per variant, one entry per grid point with its ``nll`` and trajectory),
        ``alphas``, ``bs``, ``ab_split``, ``prep``, and ``figures`` (keyed by variant).
    """
    subjids, date_ranges, dates_for = normalize_subjids_dates(subjids, date_ranges)
    results = {}

    with plt.rc_context(nature_style()):
        for subjid in subjids:
            prep = prepare_subject(subjid, dates_for(subjid), rewarded_only)
            if split_ab:
                splits = {letter: _sweep_sequence(subset_by_ab(prep, letter), alphas, bs,
                                                  n_starts, seed)
                          for letter in AB_LETTERS}
                splits = {letter: r for letter, r in splits.items() if r is not None}
                if splits:
                    results[subjid] = splits
            else:
                result = _sweep_sequence(prep, alphas, bs, n_starts, seed)
                if result is not None:
                    results[subjid] = result
            _show_figures(show)
    return results


# --- logistic multi-start diagnostic (standalone) -----------------------------------------


def _diagnose_sequence(prep: dict) -> Optional[dict]:
    """Fit one sequence from every multi-start initial condition; print the table, plot it."""
    label = subject_label(prep)
    if prep["n_trials"] < 2:
        print(f"[multistart] {label}: {prep['n_trials']} kept trial(s); skipping.")
        return None

    fits = fit_logistic_multistart(prep["s"])
    best = int(np.argmax([fit["loglik"] for fit in fits]))
    warm = next(i for i, fit in enumerate(fits) if fit["label"] == WARM_START_LABEL)
    switch_loglik = fit_switchpoint(prep["s"])["loglik"]

    print(f"\n[multistart] {label} ({prep['n_trials']} trials, {len(fits)} initial conditions)")
    print(f"  switch-model loglik = {switch_loglik:.3f} (the logistic must reach at least this)")
    print(f"  {'start':<14}{'init mid':>10}{'->':^4}{'conv mid':>10}{'conv slope':>12}"
          f"{'loglik':>11}  {'':<4}")
    for i, fit in enumerate(fits):
        tag = " ".join(t for t, on in (("[best]", i == best), ("[warm]", i == warm)) if on)
        conv = "" if fit["converged"] else " (no conv.)"
        print(f"  {fit['label']:<14}{fit['initial_midpoint']:>10.1f}{'->':^4}{fit['midpoint']:>10.1f}"
              f"{fit['slope']:>12.4g}{fit['loglik']:>11.3f}  {tag}{conv}")

    basins = len({round(fit["midpoint"], 1) for fit in fits})
    spread = max(fit["loglik"] for fit in fits) - min(fit["loglik"] for fit in fits)
    print(f"  {basins} distinct converged midpoint(s); loglik spread across starts = {spread:.3f}")
    if fits[best]["loglik"] > fits[warm]["loglik"] + _LOGLIK_REPORT_TOL:
        print(f"  NOTE: a dispersed start beat the warm start by "
              f"{fits[best]['loglik'] - fits[warm]['loglik']:.3f} loglik -- "
              f"the warm start alone would have found a local optimum.")

    return {"fits": fits, "best": best, "best_label": fits[best]["label"],
            "warm": warm, "switch_loglik": switch_loglik, "n_basins": basins,
            "ab_split": prep.get("ab_split"), "prep": prep,
            "fig": plot_multistart(prep, fits, best, warm)}


def run_logistic_diagnostic(
    subjids: Union[int, Iterable[int], dict],
    date_ranges: Optional[dict] = None,
    rewarded_only: bool = False,
    split_ab: bool = False,
    show: bool = True,
) -> dict:
    """Show where every logistic multi-start initial condition converges, per animal.

    Per animal it prints a per-start table (initial midpoint -> converged midpoint, converged
    slope, converged loglik) and draws one figure: the raw SHORT/LONG trials with the
    empirical rolling P(SHORT), every converged sigmoid in its start's colour (winner bold,
    warm start dashed), and each start's initial and converged midpoint marked in the margin
    and joined -- so it is obvious whether the starts funnel to one optimum or split.

    Parameters
    ----------
    subjids, date_ranges, rewarded_only, split_ab, show
        As in ``run_analysis``. With ``split_ab`` the A and B trials are partitioned by the
        same rule ``run_analysis`` uses, and each gets its own table and figure.

    Returns
    -------
    dict
        Keyed by subjid (nested by ``"A"`` / ``"B"`` when ``split_ab``): ``fits`` (one entry
        per start), ``best`` / ``best_label``, ``warm``, ``switch_loglik``, ``n_basins``,
        ``ab_split``, ``prep``, and ``fig``.
    """
    subjids, date_ranges, dates_for = normalize_subjids_dates(subjids, date_ranges)
    results = {}

    with plt.rc_context(nature_style()):
        for subjid in subjids:
            prep = prepare_subject(subjid, dates_for(subjid), rewarded_only)
            if split_ab:
                splits = {letter: _diagnose_sequence(subset_by_ab(prep, letter))
                          for letter in AB_LETTERS}
                splits = {letter: r for letter, r in splits.items() if r is not None}
                if splits:
                    results[subjid] = splits
            else:
                result = _diagnose_sequence(prep)
                if result is not None:
                    results[subjid] = result
            _show_figures(show)
    return results


# --- residual autocorrelation diagnostic (standalone) -------------------------------------
#
# The planned parametric bootstrap null redraws each animal's sequence as independent Bernoulli
# trials from the fitted model. That is only valid if the *residuals* of the fitted model are
# serially independent -- if consecutive trials still co-vary once the strategy curve is
# removed, the bootstrap will understate the null's spread and inflate significance. This checks
# that assumption per animal. The maths is in ``hypnose_behavior.modelling.switchpoint.autocorr``.


def _autocorr_sequence(prep: dict, max_lag: int) -> Optional[dict]:
    """Residual-ACF diagnostic for one sequence: fit-best residuals, print, plot.

    Uses the BIC-best model's per-trial P(SHORT) as the fit, so the residuals carry whatever
    serial structure the *chosen* description leaves behind. Returns None (after a message)
    when the sequence is too short, or when the best model has no per-trial curve.
    """
    label = subject_label(prep)
    n = prep["n_trials"]
    if n < 3:
        print(f"[residual-acf] {label}: {n} kept trial(s); skipping.")
        return None

    comparison = compare_models(prep["s"])
    best = comparison["best_bic"]
    fitted = model_fitted_p(best, comparison["fits"][best], n)
    if fitted is None:
        print(f"[residual-acf] {label}: BIC-best model '{best}' has no per-trial curve; skipping.")
        return None
    resid = prep["s"].astype(float) - fitted

    lags = np.arange(1, int(min(max_lag, n - 1)) + 1)
    acf_full, npairs_full = residual_acf(resid, lags)
    acf_within, npairs_within = residual_acf(resid, lags, session_index=prep["session_index"])
    bound_full, bound_within = acf_bounds(npairs_full), acf_bounds(npairs_within)

    exceed_full = int(np.sum(np.isfinite(acf_full) & (np.abs(acf_full) > bound_full)))
    exceed_within = int(np.sum(np.isfinite(acf_within) & (np.abs(acf_within) > bound_within)))
    usable_within = int(np.sum(npairs_within > 0))
    r1_full, b1_full = acf_full[0], bound_full[0]
    r1_within, b1_within = acf_within[0], bound_within[0]

    sig_within = bool(np.isfinite(r1_within) and abs(r1_within) > b1_within)
    material = bool(sig_within and abs(r1_within) >= ACF_MATERIAL_THRESHOLD)

    print(f"\n[residual-acf] {label} ({n} trials, BIC-best model: {best})")
    print(f"  residuals = observed - fitted P(SHORT); mean = {resid.mean():+.4f}, "
          f"sd = {resid.std():.4f}")
    print(f"  full ACF          : lag-1 = {r1_full:+.3f} (band +/-{b1_full:.3f}); "
          f"{exceed_full} of {lags.size} lags exceed the band")
    if np.isfinite(r1_within):
        print(f"  within-session ACF: lag-1 = {r1_within:+.3f} (band +/-{b1_within:.3f}, "
              f"{npairs_within[0]} pairs); {exceed_within} of {usable_within} usable lags "
              f"exceed the band")
    else:
        print("  within-session ACF: no within-session lag-1 pair (sessions too short)")

    if material:
        verdict = (f"MATERIAL -- within-session lag-1 = {r1_within:+.3f} clears both the band "
                   f"(+/-{b1_within:.3f}) and |r| >= {ACF_MATERIAL_THRESHOLD:g}; the i.i.d. "
                   f"Bernoulli bootstrap null is questionable for this animal.")
    elif sig_within:
        verdict = (f"not material -- within-session lag-1 = {r1_within:+.3f} is significant but "
                   f"|r| < {ACF_MATERIAL_THRESHOLD:g}; i.i.d. bootstrap is broadly defensible.")
    elif np.isfinite(r1_within):
        verdict = (f"not material -- within-session lag-1 = {r1_within:+.3f} sits inside the "
                   f"+/-{b1_within:.3f} band; no evidence against i.i.d.")
    else:
        verdict = "inconclusive -- too little within-session data to judge trial-to-trial dependence."
    print(f"  VERDICT: {verdict}")

    fig = plot_residual_autocorr(prep, best, lags, acf_full, bound_full, acf_within, bound_within)
    return {
        "best_model": best, "residuals": resid, "fitted": fitted, "lags": lags,
        "acf_full": acf_full, "bound_full": bound_full, "n_pairs_full": npairs_full,
        "acf_within": acf_within, "bound_within": bound_within, "n_pairs_within": npairs_within,
        "lag1_full": float(r1_full),
        "lag1_within": float(r1_within) if np.isfinite(r1_within) else float("nan"),
        "n_exceed_full": exceed_full, "n_exceed_within": exceed_within,
        "sig_within": sig_within, "material": material,
        "ab_split": prep.get("ab_split"), "prep": prep, "fig": fig,
    }


def run_residual_autocorrelation(
    subjids: Union[int, Iterable[int], dict],
    date_ranges: Optional[dict] = None,
    rewarded_only: bool = False,
    max_lag: int = ACF_MAX_LAG,
    split_ab: bool = False,
    show: bool = True,
) -> dict:
    """Check the i.i.d.-Bernoulli assumption behind the planned parametric bootstrap null.

    Standalone -- ``run_analysis`` does not call it. For each animal it takes the BIC-best
    model's fitted per-trial P(SHORT), forms the residuals ``observed - fitted``, and reports
    their autocorrelation, so any serial dependence the fitted strategy curve fails to absorb
    is made visible. If residuals are serially correlated, redrawing trials independently from
    the fitted model would understate the null's spread.

    Per animal it prints the lag-1 residual autocorrelation and how many lags (1..``max_lag``)
    exceed the ``+/-1.96/sqrt(N)`` band, both over all trial pairs and over within-session
    pairs only, and a one-line verdict on whether lag-1 dependence is material. It draws one
    figure with two stacked ACF panels (all pairs; within-session pairs). The within-session
    version drops cross-session lag pairs, whose apparent correlation comes from the sleep gap
    and the strategy shift around it rather than from trial-to-trial dependence.

    Parameters
    ----------
    subjids, date_ranges, rewarded_only, split_ab, show
        As in ``run_analysis``. With ``split_ab`` the A- and B-reward trials are partitioned by
        the same rule and each gets its own residuals, table, verdict, and figure; titles and
        prints carry the reward tag.
    max_lag : int
        Largest lag reported (clamped to ``n - 1``). Default 50.

    Returns
    -------
    dict
        Keyed by subjid (nested by ``"A"`` / ``"B"`` when ``split_ab``): ``best_model``,
        ``residuals``, ``fitted``, ``lags``, ``acf_full`` / ``bound_full`` / ``n_pairs_full``,
        ``acf_within`` / ``bound_within`` / ``n_pairs_within``, ``lag1_full``, ``lag1_within``,
        ``n_exceed_full``, ``n_exceed_within``, ``sig_within``, ``material``, ``ab_split``,
        ``prep``, and ``fig``.
    """
    subjids, date_ranges, dates_for = normalize_subjids_dates(subjids, date_ranges)
    results = {}

    with plt.rc_context(nature_style()):
        for subjid in subjids:
            prep = prepare_subject(subjid, dates_for(subjid), rewarded_only)
            if split_ab:
                splits = {letter: _autocorr_sequence(subset_by_ab(prep, letter), max_lag)
                          for letter in AB_LETTERS}
                splits = {letter: r for letter, r in splits.items() if r is not None}
                if splits:
                    results[subjid] = splits
            else:
                result = _autocorr_sequence(prep, max_lag)
                if result is not None:
                    results[subjid] = result
            _show_figures(show)
    return results


# --- sleep-alignment permutation test -----------------------------------------------------


def run_permutation(
    subjids: Union[int, Iterable[int], dict],
    date_ranges: Optional[dict] = None,
    rewarded_only: bool = False,
    inclusion: str = "bic_switch_wins",
    n_permutations: int = 10000,
    seed: int = 0,
    show: bool = True,
) -> dict:
    """Permutation test of whether strategy switches sit closer to sleep than chance.

    Selects its own subjects and recomputes every fit, so it never depends on
    ``run_analysis`` having been called.

    Parameters
    ----------
    subjids, date_ranges, rewarded_only
        As in ``run_analysis``; the subject set may differ.
    inclusion : str
        Which animals count as having a switch:

        - ``"bic_switch_wins"`` (default) -- the switch model has the lowest BIC of the
          three, so the change is both real and abrupt rather than a gradual drift.
        - ``"bic_beats_constant"`` -- the switch model only has to beat the constant model.
        - ``"aic_switch_wins"`` -- as the default, under the milder AIC penalty.
        - ``"all"`` -- no filtering.
    n_permutations : int
        Permutations drawn for the null distribution.
    seed : int
        RNG seed, for a reproducible null.
    show : bool
        Call ``plt.show()``.

    Returns
    -------
    dict
        ``real_f`` (one value per included animal), ``shuffled_f`` (every span-valid
        recipient x donor pair, for the boxplot), ``null_means``, ``observed_mean``,
        ``p_value``, ``n_permutations``, ``n_pairs_dropped``, ``included_subjids``,
        ``excluded_subjids`` (no switch), ``excluded_no_donor`` (no donor spans their
        ``tau``), ``per_subject``, and ``fig``.

    Raises
    ------
    ValueError
        Unknown ``inclusion`` rule, or fewer than two animals left to compare.
    """
    if inclusion not in _INCLUSION_RULES:
        raise ValueError(f"inclusion must be one of {sorted(_INCLUSION_RULES)}, got {inclusion!r}")
    subjids, date_ranges, dates_for = normalize_subjids_dates(subjids, date_ranges)
    keep = _INCLUSION_RULES[inclusion]

    per_subject, excluded = {}, []
    for subjid in subjids:
        prep = prepare_subject(subjid, dates_for(subjid), rewarded_only)
        if prep["n_trials"] < 2:
            print(f"[permutation] Subject {subjid}: {prep['n_trials']} kept trial(s); excluded.")
            excluded.append(subjid)
            continue
        comparison = compare_models(prep["s"])
        if not keep(comparison):
            print(f"[permutation] Subject {subjid}: no switch under '{inclusion}' "
                  f"(BIC winner: {comparison['best_bic']}); excluded.")
            excluded.append(subjid)
            continue
        # The test keys on the SINGLE-switch tau, always. An animal may now be best described
        # by switch2, but pooling two f values per animal would weight it double and change
        # what the statistic means, so its first switch (tau1) is used and the choice is logged.
        switch2_best = comparison["best_bic"] == "switch2"
        tau = comparison["fits"]["switch"]["tau"]
        if switch2_best:
            fit2 = comparison["fits"]["switch2"]
            print(f"[permutation] Subject {subjid}: BIC winner is switch2 "
                  f"(tau1 = {fit2['tau1']}, tau2 = {fit2['tau2']}); using the single-switch "
                  f"tau = {tau} for f, and contributing one value only.")
        per_subject[subjid] = {
            "tau": tau, "session_starts": prep["session_starts"],
            "f": distance_to_session_start(tau, prep["session_starts"]),
            "comparison": comparison, "n_trials": prep["n_trials"],
            "last_trial": prep["n_trials"] - 1, "switch2_best": switch2_best,
        }

    candidates = list(per_subject)
    if len(candidates) < 2:
        raise ValueError(f"Need >= 2 included animals for the across-animal shuffle, "
                         f"got {len(candidates)} under inclusion='{inclusion}'")

    # Span guard: drop (recipient, donor) pairs whose donor axis is too short for the
    # recipient's tau. A recipient left with no valid donor cannot enter the test, but it
    # still donates its own boundaries to the others.
    f_matrix = pairwise_f(per_subject, candidates)
    valid_donors = {i: np.flatnonzero(np.isfinite(f_matrix[i])) for i in range(len(candidates))}
    recipients = [i for i in valid_donors if valid_donors[i].size]
    excluded_no_donor = [candidates[i] for i in valid_donors if not valid_donors[i].size]
    n_pairs_dropped = len(candidates) * (len(candidates) - 1) - int(np.isfinite(f_matrix).sum())

    if len(recipients) < 2:
        raise ValueError(f"Need >= 2 animals with at least one span-valid donor, got "
                         f"{len(recipients)}; {n_pairs_dropped} pair(s) failed the span guard")

    included = [candidates[i] for i in recipients]
    real_f = np.array([per_subject[s]["f"] for s in included], dtype=float)
    shuffled_f = f_matrix[np.isfinite(f_matrix)]
    observed_mean = float(np.mean(real_f))
    null_means = permutation_null_means(f_matrix, recipients, valid_donors, n_permutations, seed)
    p_value = float((1 + np.sum(null_means <= observed_mean)) / (n_permutations + 1))

    print(f"\n[permutation] included {len(included)} animals ({inclusion}): {included}")
    if excluded:
        print(f"[permutation] excluded {len(excluded)} animals (no switch): {excluded}")
    if excluded_no_donor:
        print(f"[permutation] excluded {len(excluded_no_donor)} animals (no span-valid donor): "
              f"{excluded_no_donor}")
    print(f"[permutation] dropped {n_pairs_dropped} of {len(candidates) * (len(candidates) - 1)} "
          f"(recipient, donor) pairs failing the span guard")
    print(f"  real f      : median {np.median(real_f):.1f}, mean {observed_mean:.1f}, n = {real_f.size}")
    print(f"  shuffled f  : median {np.median(shuffled_f):.1f}, mean {np.mean(shuffled_f):.1f}, "
          f"n = {shuffled_f.size}")
    print(f"  observed mean f = {observed_mean:.2f}, null mean = {null_means.mean():.2f} "
          f"({n_permutations} permutations, seed {seed})")
    print(f"  one-sided p (real f closer to sleep than chance) = {p_value:.4f}")

    with plt.rc_context(nature_style()):
        fig = plot_permutation(real_f, shuffled_f, null_means, observed_mean, p_value,
                               n_permutations, len(included), inclusion)
        _show_figures(show)

    return {"real_f": real_f, "shuffled_f": shuffled_f, "null_means": null_means,
            "observed_mean": observed_mean, "p_value": p_value, "n_permutations": n_permutations,
            "n_pairs_dropped": n_pairs_dropped, "included_subjids": included,
            "excluded_subjids": excluded, "excluded_no_donor": excluded_no_donor,
            "per_subject": per_subject, "fig": fig}


# --- terminal wrappers (parsing only; all logic stays in the functions above) -------------


def _resolve_dates(args) -> Optional[Union[tuple, list]]:
    """Turn the mutually exclusive --dates / --date-range args into a date_range value."""
    if args.date_range:
        return (args.date_range[0], args.date_range[1])
    if args.dates:
        return list(args.dates)
    return None


def _resolve_date_ranges(args) -> tuple[list[int], dict]:
    """Validate the requested subjects and map each to the same CLI-supplied date range."""
    dates = _resolve_dates(args)
    check_dates = list(args.dates) if args.dates else None
    subjids = [s for s in args.subjids if validate_subject(s, check_dates)["ok"]]
    return subjids, {s: dates for s in subjids}


def _add_shared_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--subjids", nargs="+", type=int, required=True, help="subject id(s)")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--dates", nargs="*", type=int, default=None, help="specific date(s) YYYYMMDD")
    group.add_argument("--date-range", nargs=2, type=int, metavar=("START", "END"),
                       help="inclusive YYYYMMDD range")
    parser.add_argument("--rewarded-only", action="store_true",
                        help="keep only rewarded trials (aborts are always dropped)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest="command", required=True)

    analysis = subparsers.add_parser("analysis", help="per-animal switch-point fit and figures")
    _add_shared_args(analysis)
    analysis.add_argument("--likelihood-window", type=int, default=100,
                          help="half-width in trials of the posterior plot window (default: 100)")
    analysis.add_argument("--split-ab", action="store_true",
                          help="fit and plot the A- and B-reward trials separately")
    analysis.add_argument("--no-qlearning", action="store_true",
                          help="skip the Q-learning null fits and their overlay on the "
                               "model-comparison figure")
    analysis.add_argument("--around-switch", action="store_true",
                          help="crop the model-comparison figure to trials near the switch")
    analysis.add_argument("--plot-trials", type=int, default=200,
                          help="half-width in trials of the --around-switch crop (default: 200)")
    analysis.add_argument("--save", action="store_true",
                          help="save each figure as a PDF in the subject's derivatives figures dir")

    qsweep = subparsers.add_parser("qsweep",
                                   help="Q-learning (alpha, b) parameter sweep, one figure per "
                                        "variant (standalone)")
    _add_shared_args(qsweep)
    qsweep.add_argument("--split-ab", action="store_true",
                        help="sweep the A- and B-reward trials separately")
    qsweep.add_argument("--n-starts", type=int, default=N_STARTS,
                        help=f"random starting points per fit, min 20 (default: {N_STARTS})")
    qsweep.add_argument("--seed", type=int, default=0,
                        help="RNG seed for the multi-start draw (default: 0)")

    diagnostic = subparsers.add_parser("diagnostic",
                                       help="logistic multi-start diagnostic (standalone)")
    _add_shared_args(diagnostic)
    diagnostic.add_argument("--split-ab", action="store_true",
                            help="diagnose the A- and B-reward trials separately")

    autocorr = subparsers.add_parser("autocorr",
                                     help="residual-autocorrelation check for the bootstrap null")
    _add_shared_args(autocorr)
    autocorr.add_argument("--max-lag", type=int, default=ACF_MAX_LAG,
                          help=f"largest lag reported, clamped to n-1 (default: {ACF_MAX_LAG})")
    autocorr.add_argument("--split-ab", action="store_true",
                          help="check the A- and B-reward trials separately")

    permutation = subparsers.add_parser("permutation", help="switch vs sleep-boundary alignment")
    _add_shared_args(permutation)
    permutation.add_argument("--inclusion", default="bic_switch_wins", choices=sorted(_INCLUSION_RULES),
                             help="which animals count as having a switch (default: bic_switch_wins)")
    permutation.add_argument("--n-permutations", type=int, default=10000,
                             help="permutations drawn for the null distribution (default: 10000)")
    permutation.add_argument("--seed", type=int, default=0,
                             help="RNG seed for the permutation null (default: 0)")

    args = parser.parse_args()
    subjids, date_ranges = _resolve_date_ranges(args)
    if not subjids:
        print("Nothing to run after validation.")
        return 1

    if args.command == "analysis":
        run_analysis(subjids, date_ranges, rewarded_only=args.rewarded_only,
                     likelihood_window=args.likelihood_window, split_ab=args.split_ab, show=True,
                     qlearning_overlay=not args.no_qlearning, around_switch=args.around_switch,
                     plot_trials=args.plot_trials, save=args.save)
    elif args.command == "qsweep":
        run_qlearning_sweep(subjids, date_ranges, rewarded_only=args.rewarded_only,
                            split_ab=args.split_ab, n_starts=args.n_starts, seed=args.seed,
                            show=True)
    elif args.command == "diagnostic":
        run_logistic_diagnostic(subjids, date_ranges, rewarded_only=args.rewarded_only,
                                split_ab=args.split_ab, show=True)
    elif args.command == "autocorr":
        run_residual_autocorrelation(subjids, date_ranges, rewarded_only=args.rewarded_only,
                                     max_lag=args.max_lag, split_ab=args.split_ab, show=True)
    else:
        run_permutation(subjids, date_ranges, rewarded_only=args.rewarded_only,
                        inclusion=args.inclusion, n_permutations=args.n_permutations,
                        seed=args.seed, show=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
