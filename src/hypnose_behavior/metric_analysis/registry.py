# Defers evaluation of PEP-604 annotations (`X | None`), keeping this module
# importable on Python 3.9 for repos pinned there (hypnose-eeg-preprocessing).
from __future__ import annotations

"""What metrics exist, which frame each one reads, and how each is reported.

A metric declares itself where it is defined, so adding one does not touch the
orchestrator:

    @metric(frame="trials", title="Decision Accuracy")
    def decision_accuracy(trials): ...

    @session_metric(decision_accuracy)
    def decision_accuracy_session(results): ...

- **The frame is a decorator argument, not a file boundary.** Modules group by
  behavioural construct, so `sampling.py` holds both `trials` and `position_data`
  metrics, while `spec.call(results)` still knows which frame to hand each core.
- **Only `f(frame) -> value` is registrable.** `fa_port_ratio(n_a, n_b)` and
  `get_fa_ratio_a_stats(subjid, dates)` are deliberately absent: useful functions,
  not metrics over a frame.
- **The report order lives in `run.REPORT`, not here.** Deriving it from
  registration order would make it a function of import order. Registering makes a
  metric discoverable; naming it in `REPORT` is the separate decision to save it.
- **Re-registering the same function is a reload, not a clash** -- the notebooks run
  under `%autoreload 2`. A *different* function claiming a registered name still
  raises. See DECISIONS.md section 4.
"""

from dataclasses import dataclass
from typing import Callable, Optional

import pandas as pd

__all__ = ["MetricSpec", "REGISTRY", "metric", "session_metric", "as_dict"]

FRAMES = ("trials", "position_data", "trials+position_data")


@dataclass
class MetricSpec:
    """One registered metric: its core, its wrapper, and how it is reported."""

    name: str
    frame: str
    core: Callable
    key: str
    session: Optional[Callable] = None
    title: Optional[str] = None
    adapter: Optional[Callable] = None

    def call(self, results, **kwargs):
        """Evaluate the core against a `results` dict, using its declared frame.

        This is what `frame=` is *for*. `run_all_metrics` deliberately does not
        use it -- it goes through the `session` wrappers, which print, and whose
        stdout must stay byte-identical. Use this when you want the value:

            REGISTRY["decision_accuracy"].call(results)

        A core wanting session metadata beyond the frames (`hidden_rule_by_odor`
        wants `hr_odors` / `hr_positions`) takes it as a keyword; its wrapper is
        the thing that knows how to dig those out of `results`.
        """
        trials = results.get("trial_data")
        if trials is None:
            trials = pd.DataFrame()
        if self.frame == "trials":
            return self.core(trials, **kwargs)
        if self.frame == "position_data":
            return self.core(results.get("position_data"), **kwargs)
        return self.core(trials, results.get("position_data"), **kwargs)


REGISTRY: dict = {}


def _identity(fn):
    """What makes two function objects "the same metric" across a reload."""
    return (fn.__module__, fn.__qualname__)


def metric(*, frame, name=None, key=None, title=None, adapter=None):
    """Register a metric core.

    `name` defaults to the function's name and is how the registry and
    `run.REPORT` refer to it. `key` is the name it is saved under in
    `metrics_*.json` and defaults to `name` -- they differ in exactly one place,
    `hidden_rule_counts_by_odor`, which has always been saved as
    `hidden_rule_by_odor`. `adapter` converts the wrapper's return into the
    JSON-serialisable shape that key has always held.
    """
    if frame not in FRAMES:
        raise ValueError(f"frame must be one of {FRAMES}, got {frame!r}")

    def register(fn):
        metric_name = name or fn.__name__
        existing = REGISTRY.get(metric_name)
        if existing is not None and _identity(existing.core) != _identity(fn):
            clash = existing.core
            raise ValueError(f"metric {metric_name!r} already registered by "
                             f"{clash.__module__}.{clash.__name__}")
        # Re-registering the *same* function is a module reload, not a clash.
        # The notebooks run under `%autoreload 2`, which re-executes a module
        # body on every edit; raising there would make the registry unusable in
        # exactly the place metrics get written.
        REGISTRY[metric_name] = MetricSpec(
            name=metric_name, frame=frame, core=fn, key=key or metric_name,
            title=title, adapter=adapter,
        )
        fn.metric_name = metric_name
        return fn

    return register


def session_metric(core):
    """Attach the printing `f(results)` wrapper to an already-registered core."""
    def register(fn):
        metric_name = getattr(core, "metric_name", None)
        if metric_name is None:
            raise ValueError(f"{core.__name__} is not registered with @metric")
        REGISTRY[metric_name].session = fn
        return fn

    return register


def as_dict(value):
    """`.to_dict()` when the value has one, unchanged otherwise.

    The adapter for every metric whose wrapper returns a Series or DataFrame but
    whose saved form has always been a plain dict.
    """
    return value.to_dict() if hasattr(value, "to_dict") else value
