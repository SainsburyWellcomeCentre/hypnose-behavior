"""The genuinely hardcoded knobs, in one place, and stamped into every manifest.

``scoring_parameters()`` is written to ``manifest.json`` as ``analysis_parameters``, so
"what was this session scored with" is answerable from the file itself.

- **This module imports nothing from the package** -- standard library only.
  ``trial_classification.windows`` / ``.outcome``, ``utils.helpers`` and
  ``io.save_results`` all stand on it, so the day it imports back they become real
  cycles. See DECISIONS.md sections 3 and 20.
- **What belongs here:** a value hardcoded in this code-base that decides an output.
  Not unit conversions (``* 1000.0`` is what the unit *is*), not per-session schema
  values (they legitimately differ per session and live in ``summary.json``'s
  ``params``), not ``modelling/switchpoint/``'s constants (they belong to one model),
  and not ``scripts/run_speed_analysis.py``'s flags (chosen per run, and applied after
  the manifest is written). See DECISIONS.md section 35.
- **The stamp is built default-in:** every public constant here is stamped unless named
  in ``_NOT_SCORING``. A hand-written list of what *to* stamp lets a new knob be applied
  to every session and recorded on none.
- **Edit this file; never assign to it at runtime.** Importers bind these names by value
  at import time, so an override would be *reported* by the stamp without being
  *applied* by code that already imported it.
- **No gate asserts the stamp.** ``qc/_common.fingerprint_session`` deliberately never
  reads the manifest. The knobs' *values* are gated through ``trial_data`` and
  ``position_data``; the block's continued existence is not. Revisit this the moment a
  third knob is added. See DECISIONS.md section 31.
"""
from __future__ import annotations

__all__ = [
    "PRE_ODOR_GRACE_MS",
    "LATE_LATENCY_WINDOW_MULTIPLIER",
    "CACHE_MAX_ITEMS",
    "scoring_parameters",
]


#: How long after a poke-out a still-credited poke may have ended, in milliseconds.
#:
#: The valve and the animal do not switch at the same instant, so a poke that ended
#: within this of an odor window opening is treated as overlapping it. Read by
#: ``trial_classification.windows.grace_poke_ms`` and nowhere else; it is what separates
#: ``poke_source == 'grace'`` from ``'outside_grace'`` (``DECISIONS.md`` section 10 --
#: 91 grace against 83 outside-grace entries on the nine coverage sessions).
#:
#: Changing it moves ``position_data`` and, through the aborted-sequence trim,
#: ``trial_data`` -- so ``qc/regression.py`` sees a change to this value.
PRE_ODOR_GRACE_MS = 25.0

#: How many response windows a reward-port latency may span before it is ``_late``.
#:
#: One window is ``_time_in``, up to this many is ``_time_out``, beyond it is ``_late``.
#: Read by ``trial_classification.outcome.latency_label`` and nowhere else, which is the
#: single rule behind ``fa_label`` and ``fr_label`` (``DECISIONS.md`` section 16). It
#: multiplies the session's *own* ``response_time_ms`` window, which is a per-session
#: schema value -- this is the multiplier only.
#:
#: The buckets it decides are ``trial_data`` columns and they are populated on the
#: coverage sessions (section 16 records ``FA_time_out`` 58 and ``FA_late`` 80), so
#: ``qc/regression.py`` sees a change to this value.
LATE_LATENCY_WINDOW_MULTIPLIER = 3.0

#: Entries kept in ``utils.helpers.CACHE`` before the oldest is evicted.
#:
#: In this file because it is hardcoded, excluded from the stamp because it is a cache
#: size: see ``_NOT_SCORING``.
CACHE_MAX_ITEMS = 40


#: Constants defined here that are **not** stamped into the manifest, each with the
#: reason it cannot answer "what was this session scored with".
#:
#: Keep this as small as the argument allows. A knob excluded from the stamp is a knob
#: whose value cannot be recovered from a saved session, so the bar is that changing it
#: **cannot change an output**, not merely that it looks incidental.
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
