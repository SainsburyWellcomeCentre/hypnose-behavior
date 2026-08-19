"""Sleep-alignment permutation test: do switches sit closer to sleep boundaries than chance?

Pure numeric stats over the fitted per-subject results (numpy arrays and plain dicts in and
out). The statistic is the mean of ``f`` -- the number of trials from the start of the session
containing a switch to the switch itself -- and the null donates session boundaries across
animals. The orchestration that selects subjects, fits them, prints and plots is in
``scripts/modelling/switchpoint_analysis.py``; only the maths lives here.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

# Attempts to build a without-replacement donor assignment before allowing replacement.
_ASSIGNMENT_TRIES = 20

# TODO (deferred): nothing in this module needs changing, but the inclusion rules that decide
# which animals reach it (`_INCLUSION_RULES` in scripts/modelling/switchpoint_analysis.py) now
# run against a model set where `qlearning` is a real, eligible fit rather than a stub. An
# animal whose BIC winner is the mechanistic null is therefore dropped from the test without
# that being an explicit choice. See the TODO at `_INCLUSION_RULES`.

__all__ = [
    "distance_to_session_start",
    "pairwise_f",
    "sample_assignment",
    "permutation_null_means",
]


def distance_to_session_start(tau: float, boundaries: Sequence[float] | np.ndarray) -> float:
    """Trials from the start of the session containing ``tau`` to ``tau`` itself.

    ``boundaries`` are the ordered global trial ids on which sessions *start* (a session
    start is the trial after a sleep period). The containing session is the one with the
    greatest start at or before ``tau``, so ``f = tau - that_start``. Small ``f`` means
    the switch happened soon after sleep.

    ``tau`` at or after the last boundary falls in the final session and is handled by
    the same rule. ``tau`` before the first boundary belongs to no session and returns
    NaN, as does an empty ``boundaries`` -- both are undefined rather than zero, so a
    caller cannot silently average them in.

    Returns
    -------
    float
        Non-negative trial count, or NaN when ``tau`` precedes every session start.
    """
    starts = np.sort(np.asarray(boundaries, dtype=float).ravel())
    if starts.size == 0:
        return float("nan")
    index = int(np.searchsorted(starts, float(tau), side="right")) - 1
    if index < 0:
        return float("nan")
    return float(tau) - float(starts[index])


def pairwise_f(per_subject: dict, candidates: list) -> np.ndarray:
    """``f`` for every ordered (recipient, donor) pair, NaN where the pair is invalid.

    A pair is invalid on the diagonal, and when the recipient's ``tau`` falls beyond the
    donor's trial axis (``tau > donor last trial``). Scoring such a pair would measure ``f``
    from the donor's final session start -- an arbitrarily inflated value that biases the
    null -- so it is dropped. Note the donor's last *session start* is not the cutoff: a
    ``tau`` between it and the donor's last trial still lands inside a real donated session.
    """
    n = len(candidates)
    f = np.full((n, n), np.nan)
    for i, recipient in enumerate(candidates):
        tau = per_subject[recipient]["tau"]
        for j, donor in enumerate(candidates):
            if i == j or tau > per_subject[donor]["last_trial"]:
                continue
            f[i, j] = distance_to_session_start(tau, per_subject[donor]["session_starts"])
    return f


def sample_assignment(rng: np.random.Generator, recipients: list, valid_donors: dict) -> list:
    """Assign each recipient one span-valid donor, without replacement where possible.

    Recipients are filled in random order, each taking a donor not yet used. A greedy pass
    can strand a later recipient whose only valid donors are all taken, so the whole
    assignment is resampled; after ``_ASSIGNMENT_TRIES`` failures the sampler falls back to
    drawing with replacement, which always succeeds because every recipient here has at
    least one valid donor.
    """
    for _ in range(_ASSIGNMENT_TRIES):
        used, donors = set(), {}
        for i in rng.permutation(recipients):
            choices = [d for d in valid_donors[int(i)] if d not in used]
            if not choices:
                break
            donors[int(i)] = int(rng.choice(choices))
            used.add(donors[int(i)])
        if len(donors) == len(recipients):
            return [donors[i] for i in recipients]
    return [int(rng.choice(valid_donors[i])) for i in recipients]


def permutation_null_means(f_matrix: np.ndarray, recipients: list, valid_donors: dict,
                           n_permutations: int, seed: int) -> np.ndarray:
    """Null distribution of the mean ``f`` when every recipient gets one donated boundary set."""
    rng = np.random.default_rng(seed)
    null_means = np.empty(n_permutations, dtype=float)
    for k in range(n_permutations):
        donors = sample_assignment(rng, recipients, valid_donors)
        null_means[k] = np.mean([f_matrix[i, j] for i, j in zip(recipients, donors)])
    return null_means
