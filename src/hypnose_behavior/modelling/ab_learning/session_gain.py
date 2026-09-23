"""Within- and across-session gain in raw accuracy, and accuracy by position in a session.

The descriptive companion to `log_regression`. Every function takes the fits of
`log_regression.fit_session_models`, so the rows, the position axis and the session floor
are the ones the regression used: a window counts completed trials in completed mode and
choice attempts in attempts mode, never a mix of the two.

- `edge_windows`     -- per session, accuracy over its first and its last N choices.
- `edge_gains`       -- first N to last N of a session (within), and last N of a session
  to first N of the next (across).
- `gain_summary`     -- the mean of each gain over sessions, per animal or pooled.
- `position_profile` -- accuracy binned by position in the session, pooled over an
  animal's sessions: in choices from the session start, or in fractions of the session.
- `session_profile`  -- the same binned by fraction, per session.

At N = 20 one window's accuracy carries a standard error near 0.1, so a single gain says
little; the means in `gain_summary` are what to read. A session's within gain and the
across gain after it share that session's last window with opposite signs, so the two
are negatively correlated and are compared side by side rather than differenced. Paired
that way they telescope, like W_s and O_s in `log_regression`: over sessions recorded back
to back they sum to the first window of the last session minus that of the first.

**Pooling by choice number mixes sessions.** Only long sessions reach the late bins of
the ``"choices"`` scale, so a late bin can differ from an early one by which sessions it
holds. Each profile row therefore carries two model predictions averaged over the same
rows: ``fitted_a`` (M_a, flat within every session) is that composition effect alone, and
``fitted_c`` (M_c) what the per-session slopes predict. The ``"fraction"`` scale gives
every session the same share of every bin, so M_a is flat there; a flat pooled profile
that ``fitted_c`` also reproduces means per-session slopes cancel in the pooling, not that
they are absent -- `session_profile` shows them one session at a time.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from hypnose_behavior.modelling.ab_learning.diagnostics import wilson_interval

__all__ = [
    "N_WINDOW",
    "SCALES",
    "edge_gains",
    "edge_windows",
    "gain_summary",
    "position_profile",
    "session_profile",
]

# Choices per edge window.
N_WINDOW = 20

# Position axes of a profile: choices from the session start, or fractions of the session.
SCALES = ("choices", "fraction")

_SESSION = ["subjid", "ses", "date", "session_idx"]
_GAIN_COLUMNS = ["value", "se", "lo", "hi"]


def _rows(fits: dict) -> pd.DataFrame:
    """Every fitted row in time order within its animal, with both models' predictions."""
    parts = []
    for subjid in sorted(fits):
        fit = fits[subjid]
        frame = fit["frame"]
        parts.append(frame.assign(
            mode=fit["mode"],
            fitted_a=np.asarray(fit["models"]["M_a"].predict(frame)),
            fitted_c=np.asarray(fit["models"]["M_c"].predict(frame))))
    return pd.concat(parts, ignore_index=True)


def edge_windows(fits: dict, n: int = N_WINDOW) -> pd.DataFrame:
    """Accuracy over each fitted session's first and last ``n`` choices.

    One row per session: ``n_session`` (its choices), ``first_k`` / ``first_n`` and
    ``last_k`` / ``last_n`` (correct of counted), ``first`` / ``last`` with their Wilson
    bounds (``_lo`` / ``_hi``), and ``disjoint`` (the session holds at least ``2 * n``
    choices, so the two windows share none). A session shorter than ``n`` gives both
    windows all of its choices.
    """
    rows = []
    for _, part in _rows(fits).groupby(["subjid", "session_idx"], sort=True):
        y = part["y"].to_numpy()
        first, last = y[:n], y[-n:]
        rows.append({**part.iloc[0][_SESSION + ["mode"]].to_dict(), "window": n,
                     "n_session": len(y), "first_k": int(first.sum()), "first_n": len(first),
                     "last_k": int(last.sum()), "last_n": len(last)})
    out = pd.DataFrame(rows)
    out["disjoint"] = out["n_session"] >= 2 * n
    for edge in ("first", "last"):
        out[edge] = out[f"{edge}_k"] / out[f"{edge}_n"]
        out[f"{edge}_lo"], out[f"{edge}_hi"] = wilson_interval(out[f"{edge}_k"],
                                                               out[f"{edge}_n"])
    return out


def _difference(k_from, n_from, k_to, n_to) -> dict:
    """``p_to - p_from`` of two independent proportions.

    Wald standard error, and Newcombe's hybrid score interval built from the two Wilson
    intervals, which unlike the Wald one stays open when a window is all correct.
    """
    k_from, n_from, k_to, n_to = (np.asarray(v, dtype=float) for v in (k_from, n_from, k_to,
                                                                        n_to))
    p_from, p_to = k_from / n_from, k_to / n_to
    lo_from, hi_from = wilson_interval(k_from, n_from)
    lo_to, hi_to = wilson_interval(k_to, n_to)
    value = p_to - p_from
    return {
        "value": value,
        "se": np.sqrt(p_from * (1 - p_from) / n_from + p_to * (1 - p_to) / n_to),
        "lo": value - np.sqrt((p_to - lo_to) ** 2 + (hi_from - p_from) ** 2),
        "hi": value + np.sqrt((hi_to - p_to) ** 2 + (p_from - lo_from) ** 2),
    }


def edge_gains(fits: dict, n: int = N_WINDOW) -> pd.DataFrame:
    """The change in accuracy inside each session and across each boundary.

    One row per animal and component:

    - ``within``     -- last ``n`` minus first ``n`` of session ``ses``; NaN unless the
      two windows are disjoint (see `edge_windows`).
    - ``across``     -- first ``n`` of ``ses_next`` minus last ``n`` of ``ses``, the two
      recorded back to back.
    - ``across_gap`` -- the same where a session between the two has no fitted rows
      (short, or without a choice of this mode), so it spans that session and a second
      night.

    ``value`` carries a Wald ``se`` and a Newcombe 95% interval (``lo`` / ``hi``).
    ``gap_days`` is the calendar distance across a boundary, ``sessions_skipped`` how
    many unfitted sessions fall inside it.
    """
    windows = edge_windows(fits, n)
    parts = []
    for _, animal in windows.groupby("subjid", sort=True):
        animal = animal.sort_values("session_idx").reset_index(drop=True)

        within = animal[_SESSION + ["mode", "window"]].assign(component="within")
        gain = _difference(animal["first_k"], animal["first_n"], animal["last_k"],
                           animal["last_n"])
        for column in _GAIN_COLUMNS:
            within[column] = np.where(animal["disjoint"], gain[column], np.nan)

        this = animal.iloc[:-1].reset_index(drop=True)
        after = animal.iloc[1:].reset_index(drop=True)
        skipped = (after["session_idx"] - this["session_idx"] - 1).to_numpy()
        dates = [pd.to_datetime(part["date"], format="%Y%m%d") for part in (this, after)]
        across = this[_SESSION + ["mode", "window"]].assign(
            component=np.where(skipped > 0, "across_gap", "across"),
            session_idx_next=after["session_idx"], ses_next=after["ses"],
            date_next=after["date"],
            gap_days=((dates[1] - dates[0]) / pd.Timedelta(days=1)).to_numpy(),
            sessions_skipped=skipped,
            **_difference(this["last_k"], this["last_n"], after["first_k"], after["first_n"]))
        parts.extend([within, across])
    columns = (_SESSION + ["mode", "window", "component", "session_idx_next", "ses_next",
                           "date_next", "gap_days", "sessions_skipped"] + _GAIN_COLUMNS)
    return pd.concat(parts, ignore_index=True).reindex(columns=columns)


def gain_summary(gains: pd.DataFrame, by=("subjid",)) -> pd.DataFrame:
    """The mean of each gain component over its sessions or boundaries.

    ``by`` groups further -- ``("subjid",)`` per animal, ``()`` pooled over animals. One
    row per group and component, NaN gains left out: ``m`` gains averaged, ``mean``,
    ``se`` (the gains' own standard errors combined, ``sqrt(sum(se**2)) / m``, i.e. the
    binomial noise of the windows) with its normal 95% interval ``lo`` / ``hi``, and
    ``sd``, the spread of the gains between sessions, which also holds real
    session-to-session variation.
    """
    kept = gains[gains["value"].notna()]
    keys = ["mode", *by, "component"]
    out = kept.groupby(keys).agg(m=("value", "size"), mean=("value", "mean"),
                                 sd=("value", "std"),
                                 var=("se", lambda s: float(np.sum(s ** 2))))
    out["se"] = np.sqrt(out.pop("var")) / out["m"]
    out["lo"] = out["mean"] - 1.96 * out["se"]
    out["hi"] = out["mean"] + 1.96 * out["se"]
    return out.reset_index()[keys + ["m", "mean", "se", "lo", "hi", "sd"]]


def _profile(fits: dict, keys: list, scale: str, bins: int, bin_width: int) -> pd.DataFrame:
    """Accuracy and both models' mean prediction per position bin, grouped by ``keys``."""
    if scale not in SCALES:
        raise ValueError(f"scale must be one of {SCALES}, got {scale!r}")
    rows = _rows(fits)
    grouped = rows.groupby(["subjid", "session_idx"])
    k = grouped.cumcount().to_numpy()
    if scale == "fraction":
        size = grouped["y"].transform("size").to_numpy()
        rows["bin"] = (k * bins) // size
        rows["start"], rows["end"] = rows["bin"] / bins, (rows["bin"] + 1) / bins
    else:
        rows["bin"] = k // bin_width
        # 1-based choice numbers, so the first bin reads 1 to `bin_width`.
        rows["start"], rows["end"] = rows["bin"] * bin_width + 1, (rows["bin"] + 1) * bin_width

    out = rows.groupby(["mode", *keys, "bin", "start", "end"], as_index=False).agg(
        n=("y", "size"), k=("y", "sum"), n_sessions=("session_idx", "nunique"),
        fitted_a=("fitted_a", "mean"), fitted_c=("fitted_c", "mean"))
    out["scale"] = scale
    out["center"] = (out["start"] + out["end"]) / 2
    out["accuracy"] = out["k"] / out["n"]
    out["accuracy_lo"], out["accuracy_hi"] = wilson_interval(out["k"], out["n"])
    return out[["mode", *keys, "scale", "bin", "start", "end", "center", "n", "k",
                "accuracy", "accuracy_lo", "accuracy_hi", "n_sessions", "fitted_a",
                "fitted_c"]]


def position_profile(fits: dict, scale: str = "fraction", *, bins: int = 10,
                     bin_width: int = 10) -> pd.DataFrame:
    """Accuracy by position in the session, pooled over each animal's fitted sessions.

    ``scale="fraction"`` splits every session into ``bins`` bins of equal count (the
    first tenth of its choices, and so on); ``scale="choices"`` counts ``bin_width``
    choices from the session start, so a long session reaches bins a short one never
    does.

    One row per animal and bin: ``start`` / ``end`` / ``center`` (fractions, or 1-based
    choice numbers), ``n`` / ``k``, ``accuracy`` with Wilson bounds, ``n_sessions``
    reaching the bin, and ``fitted_a`` / ``fitted_c``, M_a's and M_c's predictions
    averaged over the bin's rows.
    """
    return _profile(fits, ["subjid"], scale, bins, bin_width)


def session_profile(fits: dict, bins: int = 5) -> pd.DataFrame:
    """Accuracy per fraction of the session, one session at a time.

    The columns of `position_profile` on the ``"fraction"`` scale, plus the session
    identity; ``fitted_c`` is then the session's own M_c line averaged over each bin.
    """
    return _profile(fits, _SESSION, "fraction", bins, bins)
