# Defers evaluation of PEP-604 annotations (`X | None`), keeping this module
# importable on Python 3.9 for repos pinned there (hypnose-eeg-preprocessing).
from __future__ import annotations

"""What a saved trial record *is* -- the single declaration of `trial_data`'s schema.

`trial_data` is written by `io/save_results.py`, read by `io/load_results.py`, and
built by `trial_classification/classify_trials.py`. Before restructure_2 Phase 7b
nothing declared its shape: `save_results` named 27 columns while the table carried
60, the other 33 existing only because some line assigned them. This module is that
declaration.

**A leaf.** It imports nothing from the package -- the standard library only. Both
`io/__init__.py` and `metric_analysis/__init__.py` are docstring-only, so
`trial_classification -> io.protocol_schema` triggers no package-level side effects and closes
no cycle. Keep it that way, for the same reason `frames.py` must stay a leaf
(`docs/DECISIONS.md` section 3): every layer here already imports it.

### The protocol mode decides the column set

A session's columns depend on which branch of `classify_trials` ran, and the three
branches write different families. Declaring one uniform record for all of them would
put ~26 all-NaN columns on the average session; declaring one per mode reproduces
what is on disk today to within 7 columns. So the mode is part of the schema, and
`resolve_mode` is where it is decided -- once, from the two flags, rather than
re-derived at each site that needs to know.
"""

__all__ = [
    "ConflictingProtocolError",
    "STANDARD",
    "SINGLE_REWARD",
    "ODOUR_DISCRIMINATION",
    "MODES",
    "resolve_mode",
]


class ConflictingProtocolError(Exception):
    """A session's two protocol flags disagree about which schema it follows.

    Named rather than a bare `ValueError` so it is greppable in a batch log and
    separately catchable: `batch_analyze_sessions` wraps each session in
    `except Exception`, where a `ValueError` is indistinguishable from pandas
    complaining about an index.
    """


# The three protocol modes. A session is exactly one of them, and the value is written
# to `manifest.json` so `io/load_results.py` can check the file against the right field
# set rather than guessing from the columns it happens to find.
STANDARD = "standard"
SINGLE_REWARD = "single_reward"
ODOUR_DISCRIMINATION = "odour_discrimination"

MODES = (STANDARD, SINGLE_REWARD, ODOUR_DISCRIMINATION)


def resolve_mode(*, is_odour_discrimination: bool, is_single_reward: bool) -> str:
    """Which of `MODES` this run follows. Raises `ConflictingProtocolError` on the impossible one.

    The two flags come from **independent sources** -- `is_odour_discrimination` from
    the protocol name in the detected stage, `is_single_reward` from the schema's
    `isSingleRewardProtocol` flag (`trial_classification/params.py`). Nothing in the
    code makes them exclusive; the *experiment* does, and by construction: odour
    discrimination presents a sequence of length 1, while the single-reward protocol
    needs at least 2 positions for a sequence to be rewarded-or-not at its end. There
    is no session that is both, and there cannot be one.

    **So this raises rather than warns.** Both flags true means the session was run
    with a structurally impossible configuration, which is a problem at the rig, not
    in the analysis -- and it needs fixing before any number off that session is used.
    Continuing would write a `trial_data` whose schema is undefined: today's control
    flow reaches the odour-discrimination branch first and `continue`s past the
    false-response scoring, so the four determinacy columns would be *silently absent*
    from a file that still looked complete.

    Raising is also the safer failure in bulk. `batch_analyze_sessions` catches per
    session, prints which subject and date failed, and carries on -- so one broken
    schema names itself and skips, writing no derivative, while the rest of the batch
    completes. A warning would do the opposite: write the malformed file and bury the
    notice in thousands of lines of batch output.

    This is **not** Phase 9. Phase 9 validates data *values*; this is a contradiction
    between two schema-derived flags that leaves the output schema undecidable, and it
    exists only because Phase 7b made the schema mode-dependent.
    """
    if is_odour_discrimination and is_single_reward:
        raise ConflictingProtocolError(
            "session is flagged as BOTH odour-discrimination and single-reward, which is "
            "impossible by design: odour discrimination presents a sequence of length 1 "
            "and the single-reward protocol requires at least 2 positions. The stage's "
            "protocol name contains 'odourdiscrimination' and the schema sets "
            "'isSingleRewardProtocol'. This is a structural fault in the session as it "
            "was run -- fix the task schema or the stage name before analysing it; the "
            "saved schema is undefined while both hold."
        )
    if is_odour_discrimination:
        return ODOUR_DISCRIMINATION
    if is_single_reward:
        return SINGLE_REWARD
    return STANDARD
