"""The one rule for what a completed trial's outcome is.

A leaf, in the sense of ``DECISIONS.md`` section 3: **this module imports nothing from the
package** -- only the standard library. That is what lets ``io/save_results.py`` reach it
without turning ``io -> trial_classification`` into a cycle. Keep it that way.

Before Phase 6a, rewarded/unrewarded/timeout was decided in three places that shared no code:
``classify_trials`` (which ``completed_sequence_*`` frame a trial is appended to),
``analyze_response_times`` (the ``response_time_category`` column) and
``save_results._derive_outcome`` (re-derived from the saved counts).

`qc/outcome_agreement.py` measured them against each other before they were merged, over 1,731
trials on all 9 regression sessions. **The rule never conflicted** -- see ``DECISIONS.md``
section 14. So this module is the rule, and the three call sites keep exactly the parts that
genuinely differ between them:

* **the windows**, which are protocol-specific -- the caller counts supply pulses and reward
  pokes over whatever span its protocol says, and passes counts;
* **the sequence**, which is resolved differently by each caller -- ``sequence_rewarded`` is an
  *input* here, never recomputed. The one measured conflict was a sequence difference caused by
  the 0 ms positions bug (section 10 / Phase 6b), not by the rule;
* **the coverage**, i.e. when a caller declines to name a category at all.
  ``analyze_response_times`` emits one only when it could also compute a response time and
  counts the rest in ``failed_calculations``; that stays at the call site, because folding it in
  would move ~190 trials into the accuracy denominators.
"""
from __future__ import annotations

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

    One response window is "in", up to three windows is "out", beyond that is "late". Shared by
    the false-response labels on completed no-go trials (``FR``) and the false-alarm labels on
    aborted and non-initiated trials (``FA``) -- the same arithmetic was written out three times
    before Phase 6a.

    A ``None`` window means no threshold could be resolved, and everything falls to ``_late``.
    """
    if response_time_ms_window is not None and latency_ms <= response_time_ms_window:
        return f"{prefix}_time_in"
    if response_time_ms_window is not None and latency_ms <= 3.0 * response_time_ms_window:
        return f"{prefix}_time_out"
    return f"{prefix}_late"
