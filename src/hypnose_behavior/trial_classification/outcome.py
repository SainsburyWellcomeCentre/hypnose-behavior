"""The one rule for what a completed trial's outcome is.

Three sites call it -- ``classify_trials``, ``analyze_response_times`` and
``save_results._derive_outcome`` -- and each keeps what genuinely differs between them.

- **This module imports nothing from the package except the root-level leaves**
  (``parameters.py``). That is what lets ``io/save_results.py`` reach it without turning
  ``io -> trial_classification`` into a cycle. See DECISIONS.md section 3.
- ``sequence_rewarded`` is an **input**, never recomputed here: each caller resolves its
  own sequence.
- Each caller keeps its own **windows** (it counts and passes counts) and its own
  **coverage**. ``analyze_response_times`` names a category only when it can also compute
  a response time; folding that in would move ~190 trials into the accuracy denominators.
  See DECISIONS.md section 14.
"""
from __future__ import annotations

from hypnose_behavior.parameters import LATE_LATENCY_WINDOW_MULTIPLIER

REWARDED = 'rewarded'
UNREWARDED = 'unrewarded'
TIMEOUT = 'timeout'


def classify_completed_trial(*, supply_count, reward_poke_count, has_await_reward,
                             sequence_rewarded=None):
    """Outcome of one completed trial: ``'rewarded'``, ``'unrewarded'``, ``'timeout'`` or ``None``.

    Parameters
    ----------
    supply_count :
        Reward deliveries in the caller's reward window. Any pulse at all means the animal
        collected, so this is checked first and outranks everything below it.
    reward_poke_count :
        Reward-port poke onsets in the caller's response window. Going to a port without a
        delivery is an error, not a non-response.
    has_await_reward :
        Whether the trial reached AwaitReward. Without it there is no completion to score, and
        the answer is ``None`` rather than a timeout -- a trial that never got to the choice
        never had the chance to time out.
    sequence_rewarded :
        Single-reward protocol only. ``False`` means the animal completed a **no-go** sequence,
        which has no reward to collect: its outcome is carried by ``false_response`` /
        ``fr_label`` instead, so this returns ``None``. ``None`` means "not applicable", which
        is the default protocol and every non-single-reward session.

        This is an input, never recomputed here. The callers resolve the presented sequence
        differently and that difference is real -- see ``DECISIONS.md`` section 14.

    Returns
    -------
    str or None
        ``None`` means "this rule does not name an outcome for this trial", which is different
        from a timeout. Callers map ``TIMEOUT`` onto their own vocabulary where it differs
        (``analyze_response_times`` and the saved table call it ``timeout_delayed``).
    """
    if sequence_rewarded is False:
        return None
    if supply_count >= 1:
        return REWARDED
    if not has_await_reward:
        return None
    if reward_poke_count >= 1:
        return UNREWARDED
    return TIMEOUT


def latency_label(latency_ms, response_time_ms_window, prefix):
    """Bucket a reward-port latency into ``<prefix>_time_in`` / ``_time_out`` / ``_late``.

    One response window is "in", up to ``LATE_LATENCY_WINDOW_MULTIPLIER`` windows is "out",
    beyond that is "late" -- the window itself is the session's own, read from its task
    schema, and only the multiplier is hardcoded (``parameters.py``). Shared by the
    false-response labels on completed no-go trials (``FR``) and the false-alarm labels on
    aborted and non-initiated trials (``FA``).

    A ``None`` window means no threshold could be resolved, and everything falls to ``_late``.
    """
    if response_time_ms_window is not None and latency_ms <= response_time_ms_window:
        return f"{prefix}_time_in"
    if (response_time_ms_window is not None
            and latency_ms <= LATE_LATENCY_WINDOW_MULTIPLIER * response_time_ms_window):
        return f"{prefix}_time_out"
    return f"{prefix}_late"
