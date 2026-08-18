"""The genuinely hardcoded knobs, in one place, and stamped into every manifest.

A leaf, in the sense of ``DECISIONS.md`` section 3 (``frames.py``) and section 20
(``io/protocol_schema.py``): **this module imports nothing from the package** -- standard
library only. Every layer stands on it -- ``trial_classification.windows``,
``trial_classification.outcome``, ``utils.helpers`` and ``io.save_results`` -- so the day
it imports back, ``trial_classification -> parameters`` and ``utils -> parameters`` become
real cycles. Keep it that way.

What belongs here
-----------------
A value that is **hardcoded in this code-base and decides an output**. There are two, and
the shortness of that list is the measurement, not an omission (follow-up plan, item 4).

What does not, and why each was left where it is
------------------------------------------------
* **Unit conversions** -- ``* 1000.0`` for milliseconds, everywhere. Not a parameter; it
  is what the unit *is*. Centralising it would invite changing it.
* **Per-session schema values** -- ``sample_offset_time_ms``,
  ``minimum_sampling_time_ms_by_odor``, ``response_time_window_sec``,
  ``isSingleRewardProtocol``. These are read per session from the task schema by
  ``trial_classification/params.py`` and recorded in ``summary.json``'s ``params`` block.
  They **legitimately differ between sessions**, so a single central value would be wrong
  for most of them. That block and this module answer different questions -- "what was
  this session configured with" against "what did the analysis code apply" -- and merging
  them would make the two indistinguishable to a reader.
* **``modelling/switchpoint/``'s constants** (``ACF_MAX_LAG``, ``N_STARTS``,
  ``SWITCH_THRESHOLD``, ...) -- a self-contained cluster belonging to one model, not to
  the scoring pipeline.

The stamp, and what it is for
-----------------------------
``scoring_parameters()`` is written to ``manifest.json`` as ``analysis_parameters``, so
"what was this session scored with" is answerable from the file itself -- section 19's
rule that the manifest is the audit surface. It sits in the manifest and **not** in
``summary.json``'s ``params``, for the reason above, and under its own key because
``manifest["session"]["runs"][].parameters`` already means the per-run *schema*
parameters.

The stamp is built **default-in**: every public module-level constant here is stamped
unless it is named in ``_NOT_SCORING``. That direction is deliberate. Default-out -- a
hand-written list of what to stamp -- is section 27's trap, where a declaration wider (or
here, narrower) than the behaviour it describes silently stops matching: a knob added to
this file would be applied to every session and recorded on none of them. Excluding one is
therefore an explicit act that carries its reason.

> **Edit this file; do not assign to it at runtime.** Importers bind these names by value
> at import time, so a runtime override of ``parameters.PRE_ODOR_GRACE_MS`` would be
> reported by the stamp without being applied by the code that already imported it. A
> stamp that disagrees with what ran is worse than no stamp.

> **The stamp is asserted by no gate.** ``qc/_common.fingerprint_session`` reads
> ``trial_data``, the metrics dict and the side tables, and deliberately never the
> manifest -- section 19 relies on that, so that a per-run commit stamp cannot cause a
> spurious RED. The knobs' *values* are gated indirectly (below); the stamp's continued
> existence is not. See ``DECISIONS.md`` section 31.
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
