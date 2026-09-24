"""Running counts of engagement, knowledge and reward on one task clock per animal.

`cumulative_curves` returns, per animal, a running count that steps at each event's time
on the `data.task_time` clock (odour-discrimination runs laid end to end):

- ``initiations`` -- trials initiated, every trial row counted.
- ``attempts``    -- ``"attempts"`` mode only: initiations plus failed attempts that
  ended in a port visit, so it runs above ``initiations`` by the failed choice attempts.
- ``excess``      -- excess correct, ``cumsum(y - 0.5)`` over the choice rows of the
  mode: a correct choice steps up 0.5, an incorrect one down 0.5, and chance is flat.
- ``rewards``     -- rewarded completed trials, whatever the mode.

Rewards are the product of the other two: ``r(t) = rate(t) * p(t)``. Excess correct
removes chance from each choice, but on a time axis its slope is still ``choice rate *
(p - 0.5)``: a bend in ``excess`` that ``initiations`` shares at the same time is a change
in engagement, and only one that ``initiations`` lacks is a change in accuracy.

`excess_by_index` takes the rate out: excess correct over the mode's own row index, where
the slope is ``p - 0.5`` alone.
"""
from __future__ import annotations

import pandas as pd

from hypnose_behavior.modelling.ab_learning.data import choice_rows, choice_sequence, task_time

__all__ = ["SERIES", "cumulative_curves", "excess_by_index"]

# The running counts, in the order the figure stacks them.
SERIES = ("initiations", "attempts", "excess", "rewards")

_KEEP = ["subjid", "ses", "date", "session_idx"]


def _series(data: dict, frame: pd.DataFrame, column: str, name: str,
            step=None) -> pd.DataFrame:
    """One running count over ``frame``'s rows, in task-time order within an animal.

    ``step`` is each row's increment (1 when omitted). Rows without a task time -- a run
    with no recorded start or end -- are left out.
    """
    out = frame[_KEEP].assign(time=task_time(data, frame, column).to_numpy(),
                              step=1.0 if step is None else step)
    out = out[out["time"].notna()].sort_values(["subjid", "time"])
    out["value"] = out.groupby("subjid")["step"].cumsum()
    return out.drop(columns="step").assign(series=name)


def cumulative_curves(data: dict, mode: str = "completed") -> pd.DataFrame:
    """The running counts of the module docstring, one long frame over every animal.

        curves = cumulative_curves(load_ab_data(...))
        curves = cumulative_curves(ab, mode="attempts")

    ``mode`` sets the rows of ``excess`` (completed trials, or every choice attempt) and
    whether ``attempts`` is drawn up. Columns: the session identity, ``series``, ``time``
    (task hours) and ``value``, the count once that row's event is included.
    """
    trials, attempts = data["trials"], data["attempts"]
    choices = choice_rows(data, mode)
    parts = [
        _series(data, trials, "sequence_start", "initiations"),
        _series(data, choices, "time", "excess",
                step=choices["correct"].astype(float).to_numpy() - 0.5),
        _series(data, trials[trials["outcome"] == "rewarded"], "sequence_start", "rewards"),
    ]
    if mode == "attempts":
        visited = attempts[attempts["port_visit"].astype(bool)]
        both = pd.concat([trials[_KEEP + ["run_id"]].assign(time=trials["sequence_start"]),
                          visited[_KEEP + ["run_id"]].assign(time=visited["attempt_start"])],
                         ignore_index=True)
        parts.append(_series(data, both, "time", "attempts"))
    curves = pd.concat(parts, ignore_index=True).assign(mode=mode)
    return curves[_KEEP + ["mode", "series", "time", "value"]]


def excess_by_index(data: dict, mode: str = "completed") -> pd.DataFrame:
    """Excess correct over the mode's own row index, one row per choice.

        excess = excess_by_index(load_ab_data(...), mode="attempts")

    The `data.choice_sequence` rows (``k``, ``y``, ``hours``) plus ``excess``,
    ``cumsum(y - 0.5)`` within the animal once that row is included. Each row is one
    step, so the slope is ``p - 0.5`` and a bend is a change in accuracy only.
    """
    rows = choice_sequence(data, mode)
    rows["excess"] = (rows["y"] - 0.5).groupby(rows["subjid"]).cumsum()
    return rows.assign(mode=mode)
