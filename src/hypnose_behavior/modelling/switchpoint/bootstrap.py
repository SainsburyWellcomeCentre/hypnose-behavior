"""Parametric bootstrap null for the model comparison -- NOT YET IMPLEMENTED.

Planned control: for each animal, redraw the sequence many times as independent Bernoulli
trials from the fitted (e.g. constant) model, refit the competing models to each surrogate,
and build the null distribution of the statistic of interest (a loglik gain, or a BIC margin)
so a switch can be called real rather than a fitting artefact.

The i.i.d.-Bernoulli assumption this null rests on is checked, per animal, by the residual-
autocorrelation diagnostic in ``autocorr.py`` -- run that before trusting a bootstrap p-value.

Implement ``bootstrap_null`` here; keep it pure numeric (numpy arrays and plain dicts), with
the subject selection, printing and plotting staying in the orchestration script.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

__all__ = ["bootstrap_null"]


def bootstrap_null(s: Sequence[int] | np.ndarray, *args, **kwargs) -> dict:
    """Parametric bootstrap null -- NOT IMPLEMENTED (see module docstring)."""
    raise NotImplementedError(
        "bootstrap_null is planned but not implemented yet; see the module docstring in "
        "hypnose_behavior.modelling.switchpoint.bootstrap for the intended design."
    )
