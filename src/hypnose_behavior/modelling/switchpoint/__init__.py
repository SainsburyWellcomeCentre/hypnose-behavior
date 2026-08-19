"""Switch-point analysis of the LONG -> SHORT strategy change.

The subsystem's numeric core, one module per role:

- ``data``        -- build the continuous SHORT/LONG sequence for an animal.
- ``switch``      -- the switch-point model family (constant / switch / switch2 / logistic).
- ``qlearning``   -- the mechanistic Q-learning account: the null to be rejected.
- ``compare``     -- fit all models and score them with AIC / BIC.
- ``permutation`` -- the sleep-alignment permutation test.
- ``autocorr``    -- residual-autocorrelation check for the bootstrap's i.i.d. assumption.
- ``bootstrap``   -- parametric bootstrap null (planned).

Figures live separately in ``hypnose_behavior.visualization.modelling.switchpoint``; the orchestration
and CLI live in ``scripts/modelling/switchpoint_analysis.py``.

The most-used names are re-exported here for convenience, e.g.
``from hypnose_behavior.modelling.switchpoint import fit_switchpoint, compare_models, prepare_subject``.
"""
from hypnose_behavior.modelling.switchpoint.data import (
    AB_LETTERS,
    normalize_subjids_dates,
    prepare_subject,
    subject_label,
    subset_by_ab,
)
from hypnose_behavior.modelling.switchpoint.switch import (
    WARM_START_LABEL,
    bernoulli_loglik,
    fit_constant,
    fit_logistic,
    fit_logistic_multistart,
    fit_switch2,
    fit_switchpoint,
    logistic_p,
    logistic_start_points,
    posterior_fwhm,
    posterior_hdi,
    switchpoint_loglik_profile,
    switchpoint_posterior,
)
from hypnose_behavior.modelling.switchpoint.qlearning import (
    GENERATIVE_QUANTILES,
    N_GENERATIVE_EXAMPLES,
    N_GENERATIVE_SIMS,
    N_STARTS,
    QLEARN_DEFAULT_VARIANT,
    QLEARN_SWEEP_ALPHAS,
    QLEARN_SWEEP_BS,
    QLEARN_VARIANTS,
    QLEARN_VARIANT_ORDER,
    R_LONG,
    R_SHORT,
    SWITCH_THRESHOLD,
    fit_qlearning,
    fit_qlearning_variants,
    qlearning_generative_band,
    qlearning_nll,
    qlearning_parameter_sweep,
    qlearning_trajectory,
    simulate_qlearning,
)
from hypnose_behavior.modelling.switchpoint.compare import MODEL_ORDER, compare_models, model_fitted_p
from hypnose_behavior.modelling.switchpoint.permutation import (
    distance_to_session_start,
    pairwise_f,
    permutation_null_means,
    sample_assignment,
)
from hypnose_behavior.modelling.switchpoint.autocorr import (
    ACF_MATERIAL_THRESHOLD,
    ACF_MAX_LAG,
    acf_bounds,
    residual_acf,
)

__all__ = [
    # data
    "AB_LETTERS", "normalize_subjids_dates", "prepare_subject", "subject_label", "subset_by_ab",
    # switch-point family
    "WARM_START_LABEL", "bernoulli_loglik", "fit_constant", "fit_logistic",
    "fit_logistic_multistart", "fit_switch2", "fit_switchpoint", "logistic_p",
    "logistic_start_points", "posterior_fwhm", "posterior_hdi", "switchpoint_loglik_profile",
    "switchpoint_posterior",
    # qlearning (the mechanistic null)
    "GENERATIVE_QUANTILES", "N_GENERATIVE_EXAMPLES", "N_GENERATIVE_SIMS", "N_STARTS",
    "QLEARN_DEFAULT_VARIANT",
    "QLEARN_SWEEP_ALPHAS", "QLEARN_SWEEP_BS", "QLEARN_VARIANTS", "QLEARN_VARIANT_ORDER",
    "R_LONG", "R_SHORT", "SWITCH_THRESHOLD", "fit_qlearning", "fit_qlearning_variants",
    "qlearning_generative_band", "qlearning_nll", "qlearning_parameter_sweep",
    "qlearning_trajectory", "simulate_qlearning",
    # comparison
    "MODEL_ORDER", "compare_models", "model_fitted_p",
    # permutation test
    "distance_to_session_start", "pairwise_f", "permutation_null_means", "sample_assignment",
    # autocorrelation diagnostic
    "ACF_MATERIAL_THRESHOLD", "ACF_MAX_LAG", "acf_bounds", "residual_acf",
]
