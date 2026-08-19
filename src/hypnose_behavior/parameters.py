"""The genuinely hardcoded knobs, in one place, and stamped into every manifest.


- What belongs here: Hardcoded values that decide an output, but NOT unit conversions (e.g., s to ms),
  per session schema values (derived from session parameters), or modelling constants. 
"""
from __future__ import annotations

__all__ = [
    "PRE_ODOR_GRACE_MS",
    "LATE_LATENCY_WINDOW_MULTIPLIER",
    "CACHE_MAX_ITEMS",
    "scoring_parameters",
]


# Pre Odor Grace: How long after a poke-out that still led to a valve activation the valve activation is still counted. 
# Since the addition of poke_source, this only leads to a different label (poke_source == 'grace' vs 'outside_grace') and does not change any other output.
# Thus, this value is only changing this label and has no longer a meaningful effect, filtering poke_duration by PRE_ODOR_GRACE_MS is identical to filtering here. 
PRE_ODOR_GRACE_MS = 25.0

#: How many response windows a reward-port latency may span before it is ``_late``.
#:
#: One window is ``_time_in``, up to this many is ``_time_out``, beyond it is ``_late``.
LATE_LATENCY_WINDOW_MULTIPLIER = 3.0

#: Entries kept in ``utils.helpers.CACHE`` before the oldest is evicted.
CACHE_MAX_ITEMS = 40


#: Constants defined here that are **not** stamped into the manifest, each with the
#: reason it cannot answer "what was this session scored with".
_NOT_SCORING = {
    "CACHE_MAX_ITEMS": (
        "an LRU eviction bound on a read cache -- it changes how often a file is re-read, "
        "never what any value is, so it cannot answer 'what was this session scored with'"
    ),
}


def scoring_parameters() -> dict:
    """The knobs to record for this run, ``{name: value}``, sorted by name.

    Every public module-level constant here except those in ``_NOT_SCORING`` -- see this
    module's docstring for why the default is to include rather than to exclude.
    """
    return {
        name: value
        for name, value in sorted(globals().items())
        if name.isupper() and not name.startswith("_") and name not in _NOT_SCORING
    }
