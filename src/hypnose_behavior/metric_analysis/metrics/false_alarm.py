# Defers evaluation of PEP-604 annotations (`X | None`), keeping this module
# importable on Python 3.9 for repos pinned there (hypnose-eeg-preprocessing).
from __future__ import annotations

"""False alarms: their rates, their biases, their ports and their latencies.

Grouped by what is measured, not by what is returned --
``fa_latency_from_pokeout`` is here rather than in ``timing.py`` because it is a
property of the false alarm, and ``false_response_ratio`` is here because a
false response is the same construct under the single-reward schema's own
column (``false_response`` / ``fr_label``, not ``fa_label``; it is **not** the
single-reward ``fa_rate``).

The FA port count was the audit's finding 1 -- the same two-line count written
**eight** times. ``fa_port_counts`` states it once; ``fa_port_share_a`` is
rescaled from ``fa_port_ratio`` rather than recounted, per VARIANT resolutions 1
and 2.

``get_fa_ratio_a_stats`` is the odd one out: it lived in
``visualization/visualization_utils.py`` but contains no plotting at all, so 4a
moved it wholesale rather than repointing it.
"""

import numpy as np
import pandas as pd
from IPython.display import display

from hypnose_behavior.io.loaders import _load_trial_views
from hypnose_behavior.io.paths import get_derivatives_root
from hypnose_behavior.frames import (
    odor_letter,
    odor_sequence_tokens,
    reached_counts as _reached_counts,
)
from hypnose_behavior.metric_analysis.metrics.common import (
    _aborted_mask,
    _flag,
    _initiated,
    _latency_ms,
    _reduce_rate,
)
from hypnose_behavior.utils.helpers import _filter_session_dirs, _iter_subject_dirs
from hypnose_behavior.metric_analysis.registry import (
    as_dict,
    metric,
    session_metric,
)
# The report form of `fa_abortion_stats`. One-way edge into a leaf:
# `summary` imports nothing from `metrics/`.
from hypnose_behavior.metric_analysis.summary import format_fa_abortion_tables

__all__ = [
    "premature_response_rate_contributions", "premature_response_rate",
    "premature_response_rate_session",
    "response_contingent_FA_rate_contributions", "response_contingent_FA_rate",
    "response_contingent_FA_rate_session",
    "global_FA_rate_contributions", "global_FA_rate", "global_FA_rate_session",
    "FA_odor_bias", "FA_odor_bias_session",
    "FA_position_bias", "FA_position_bias_session",
    "FA_avg_response_times", "FA_avg_response_times_session",
    "fa_abortion_stats", "fa_abortion_stats_session",
    "fa_port_ratio_by_odor", "fa_port_ratio_by_odor_session",
    "fa_port_label", "fa_port_counts", "fa_port_ratio", "fa_port_share_a",
    "get_fa_ratio_a_stats",
    "fa_rate_by_odor", "fa_rate_by_position",
    "fa_latency_from_pokeout",
    "false_response_ratio_contributions", "false_response_ratio",
]


def _fa_port_payload(out):
    """`fa_port_ratio_by_odor`'s saved shape.

    One variant, not two: Phase 4a step 6 removed the non-initiated false
    alarms, so the `with_`/`without_non_initiated` wrapper this key used to carry
    no longer distinguishes anything.
    """
    return {
        'by_odor': as_dict(out['by_odor']),
        'counts': out['counts'],
        'total_fa_by_odor': out['total_fa_by_odor'],
    }


def premature_response_rate_contributions(trials):
    ab = _aborted_mask(trials)
    return ((ab & _flag(trials, "fa_label", "FA_time_in")).astype(int),
            ab.astype(int))


@metric(frame="trials", title="Premature Response Rate")
def premature_response_rate(trials):
    """FA_time_in among aborted / n aborted."""
    if trials.empty:
        return 0, 0, np.nan
    return _reduce_rate(*premature_response_rate_contributions(trials))


@session_metric(premature_response_rate)
def premature_response_rate_session(results):
    df = results.get("trial_data", pd.DataFrame())
    if df.empty:
        print("Premature Response Rate: no trial_data")
        return 0, 0, np.nan
    n_fa, n_total, rate = premature_response_rate(df)
    if n_total == 0:
        print("Premature Response Rate: no aborted trials")
        return 0, 0, np.nan
    print(f"Premature Response Rate: {n_fa}/{n_total} = {rate:.3f}")
    return n_fa, n_total, rate


def response_contingent_FA_rate_contributions(trials):
    num = (_aborted_mask(trials) & _flag(trials, "fa_label", "FA_time_in")).astype(int)
    rtc = trials["response_time_category"]
    return num, num + rtc.isin(["rewarded", "unrewarded"]).astype(int)


@metric(frame="trials", title="Response-Contingent False Alarm Rate")
def response_contingent_FA_rate(trials):
    """FA_time_in / (FA_time_in + rewarded + unrewarded)."""
    if trials.empty or "response_time_category" not in trials.columns:
        return 0, 0, np.nan
    return _reduce_rate(*response_contingent_FA_rate_contributions(trials))


@session_metric(response_contingent_FA_rate)
def response_contingent_FA_rate_session(results):
    df = results.get("trial_data", pd.DataFrame())
    if df.empty or "response_time_category" not in df.columns:
        print("Response-Contingent False Alarm Rate: missing trial_data/response_time_category")
        return 0, 0, np.nan
    n_fa, denom, rate = response_contingent_FA_rate(df)
    print(f"Response-Contingent False Alarm Rate: {n_fa}/{denom} = {rate:.3f}")
    return n_fa, denom, rate


def global_FA_rate_contributions(trials):
    return (_flag(trials, "fa_label", "FA_time_in").astype(int), _initiated(trials))


@metric(frame="trials", title="Global False Alarm Rate")
def global_FA_rate(trials):
    """FA_time_in / n initiated."""
    if trials.empty:
        return 0, 0, np.nan
    return _reduce_rate(*global_FA_rate_contributions(trials))


@session_metric(global_FA_rate)
def global_FA_rate_session(results):
    df = results.get("trial_data", pd.DataFrame())
    if df.empty:
        print("Global False Alarm Rate: no trial_data")
        return 0, 0, np.nan
    n_fa, n_ini, rate = global_FA_rate(df)
    print(f"Global False Alarm Rate: {n_fa}/{n_ini} = {rate:.3f}")
    return n_fa, n_ini, rate


@metric(frame="trials", title="FA Odor Bias", adapter=as_dict)
def FA_odor_bias(trials, *, reference=None):
    """Per-odor FA rate normalised by a baseline FA rate.

    `bias[odor] = (n_fa@odor / n_ab@odor) / reference`, with `reference`
    defaulting to this frame's own `total_fa / total_ab`. Passing it explicitly
    is what lets a rolling call keep a fixed session baseline instead of
    normalising each window by itself -- the plotters' `baseline="session"` vs
    `"window"` option, without any metric math moving back into `visualization/`.
    """
    empty = {'bias': {}, 'n_fa': {}, 'n_ab': {}, 'total_fa': 0, 'total_ab': 0}
    if trials.empty or "fa_label" not in trials.columns:
        return empty
    odor_col = "last_odor_name" if "last_odor_name" in trials.columns else "last_odor"
    if odor_col not in trials.columns:
        return empty
    aborted = trials[_aborted_mask(trials)]
    if aborted.empty:
        return empty

    fa_mask = aborted["fa_label"] == "FA_time_in"
    total_fa = int(fa_mask.sum())
    total_ab = len(aborted)
    ref = reference if reference is not None else (
        (total_fa / total_ab) if total_ab > 0 and total_fa > 0 else None)

    bias, n_fa, n_ab = {}, {}, {}
    for od in sorted(aborted[odor_col].dropna().unique()):
        at_od = aborted[odor_col] == od
        n_fa_od = int((fa_mask & at_od).sum())
        n_ab_od = int(at_od.sum())
        n_fa[od], n_ab[od] = n_fa_od, n_ab_od
        bias[od] = (n_fa_od / n_ab_od) / ref if n_ab_od > 0 and ref else np.nan
    return {'bias': bias, 'n_fa': n_fa, 'n_ab': n_ab,
            'total_fa': total_fa, 'total_ab': total_ab}


@session_metric(FA_odor_bias)
def FA_odor_bias_session(results):
    print("FA Odor Bias for FA Time In:")
    out = FA_odor_bias(results.get("trial_data", pd.DataFrame()))
    for od, bias in out['bias'].items():
        print(f"{od}: {out['n_fa'][od]}/{out['n_ab'][od]} FA, Bias: {bias:.3f}")
    return out


@metric(frame="trials", title="FA Position Bias", adapter=as_dict)
def FA_position_bias(trials, *, reference=None, with_counts=False):
    """`FA_odor_bias` by `last_odor_position`. See it for the `reference` rule."""
    if trials.empty or "fa_label" not in trials.columns:
        return ({}, {}, {}) if with_counts else pd.Series(dtype=float)
    position_col = "last_odor_position" if "last_odor_position" in trials.columns else "last_event_index"
    if position_col not in trials.columns:
        return ({}, {}, {}) if with_counts else pd.Series(dtype=float)
    aborted = trials[_aborted_mask(trials)]
    if aborted.empty:
        return ({}, {}, {}) if with_counts else pd.Series(dtype=float)

    fa_mask = aborted["fa_label"] == "FA_time_in"
    total_fa = int(fa_mask.sum())
    total_ab = len(aborted)
    ref = reference if reference is not None else (
        (total_fa / total_ab) if total_ab > 0 and total_fa > 0 else None)

    bias, n_fa, n_ab = {}, {}, {}
    for pos in sorted(aborted[position_col].dropna().unique()):
        at_pos = aborted[position_col] == pos
        n_fa_pos = int((fa_mask & at_pos).sum())
        n_ab_pos = int(at_pos.sum())
        key = int(pos) + 1 if position_col == "last_event_index" else int(pos)
        n_fa[key], n_ab[key] = n_fa_pos, n_ab_pos
        bias[key] = (n_fa_pos / n_ab_pos) / ref if n_ab_pos > 0 and ref else np.nan
    if with_counts:
        return bias, n_fa, n_ab
    return pd.Series(bias).sort_index()


@session_metric(FA_position_bias)
def FA_position_bias_session(results):
    print("FA Position Bias for FA Time In:")
    trials = results.get("trial_data", pd.DataFrame())
    parts = FA_position_bias(trials, with_counts=True)
    if not isinstance(parts, tuple):
        return parts
    bias, n_fa, n_ab = parts
    for pos in sorted(bias):
        print(f"Position {pos}: {n_fa[pos]}/{n_ab[pos]} FA, Bias: {bias[pos]:.3f}")
    return pd.Series(bias).sort_index()


@metric(frame="trials", title="FA Average Response Times")
def FA_avg_response_times(trials):
    """Mean `fa_window_latency_ms` per FA subtype."""
    out = {}
    if trials.empty or "fa_label" not in trials.columns or "fa_window_latency_ms" not in trials.columns:
        return out
    fa_df = trials[trials["fa_label"].notna()]
    for label, pretty in [("FA_time_in", "FA Time In"), ("FA_time_out", "FA Time Out"),
                          ("FA_late", "FA Late")]:
        s = pd.to_numeric(fa_df.loc[fa_df["fa_label"] == label, "fa_window_latency_ms"], errors="coerce").dropna()
        avg = s.mean() if not s.empty else np.nan
        out[pretty] = float(avg) if not np.isnan(avg) else np.nan
    return out


@session_metric(FA_avg_response_times)
def FA_avg_response_times_session(results):
    df = results.get("trial_data", pd.DataFrame())
    out = FA_avg_response_times(df)
    if not out:
        return out
    fa_df = df[df["fa_label"].notna()]
    for label, pretty in [("FA_time_in", "FA Time In"), ("FA_time_out", "FA Time Out"),
                          ("FA_late", "FA Late")]:
        n = len(pd.to_numeric(fa_df.loc[fa_df["fa_label"] == label, "fa_window_latency_ms"],
                              errors="coerce").dropna())
        avg = out[pretty]
        print(f"{pretty}: avg={avg:.1f} ms (n={n})" if not np.isnan(avg) else f"{pretty}: nan (n={n})")
    return out


def _fa_abortion_frames_missing(trials):
    """The guard `fa_abortion_stats` fails on, or None. Message is the caller's."""
    if trials.empty or "fa_label" not in trials.columns:
        return "No FA abortion data available."
    odor_col = "last_odor_name" if "last_odor_name" in trials.columns else "last_odor"
    if odor_col not in trials.columns:
        return "No FA abortion data available (missing odor column)."
    if "last_odor_position" not in trials.columns:
        return "No FA abortion data available (missing last_odor_position)."
    if not _aborted_mask(trials).any():
        return "No aborted trials found."
    return None


@metric(frame="trials+position_data", title="FA Abortion Stats")
def fa_abortion_stats(trials, position_data):
    """FA abortion breakdown by odor / position / odor x position.

    Returns three DataFrames, empty when the frame lacks what they need. Counts
    are `int`, rates `float`, positions `int`.

    **Numeric since Phase 4b** -- the audit's finding 3. These tables used to be
    built out of pre-formatted strings (`"3/10 (0.30)"`, `"2 (0.20)"`), so the
    saved `metrics['fa_abortion_stats']` was a table of prose and its one
    consumer, `plot_abortion_and_fa_rates`, parsed the numbers back out with
    `int(s.split()[0])`. `summary.format_fa_abortion_tables` renders the
    readable form for the txt report. The `"Abortion Rate Value"` column is gone
    from the metric: `"Abortion Rate"` *is* that value now.
    """
    df = trials
    empty = (pd.DataFrame(), pd.DataFrame(), pd.DataFrame())
    if _fa_abortion_frames_missing(df) is not None:
        return empty

    odor_col = "last_odor_name" if "last_odor_name" in df.columns else "last_odor"
    pos_col = "last_odor_position"

    aborted_all = df[_aborted_mask(df)]
    allowed_fa = {"FA_time_in", "FA_time_out", "FA_late"}

    subtype_labels = [
        ("FA_time_in", "FA Time In"),
        ("FA_time_out", "FA Time Out"),
        ("FA_late", "FA Late"),
    ]

    def _counts(sub_all):
        """The counts all three tables share, for one slice of the abortions."""
        sub_fa = sub_all[sub_all["fa_label"].isin(allowed_fa)]
        fa_labels = sub_fa["fa_label"].astype(str)
        n_total, n_fa = len(sub_all), len(sub_fa)
        row = {
            "Total Abortions": int(n_total),
            "FA Abortions": int(n_fa),
            "FA Abortion Rate": n_fa / n_total,
        }
        for subtype, pretty in subtype_labels:
            # int(): `.sum()` gives np.int64, which `json.dumps(default=str)`
            # writes as a *string* -- the trap the audit records for
            # `manual_vs_auto_stop_preference`.
            row[pretty] = int((fa_labels == subtype).sum())
        return row

    # Odor+Position table
    rows = []
    odors = sorted(aborted_all[odor_col].dropna().unique())
    positions = sorted(aborted_all[pos_col].dropna().unique())
    for odor in odors:
        for pos in positions:
            sub_all = aborted_all[(aborted_all[odor_col] == odor) & (aborted_all[pos_col] == pos)]
            if sub_all.empty:
                continue
            rows.append({"Odor": odor, "Position": int(pos), **_counts(sub_all)})
    df_out = pd.DataFrame(rows)

    # Per-odor table
    odor_rows = []
    for odor in odors:
        sub_all = aborted_all[aborted_all[odor_col] == odor]
        if sub_all.empty:
            continue
        odor_rows.append({"Odor": odor, **_counts(sub_all)})
    df_odor = pd.DataFrame(odor_rows)

    # Compute reached counts per position (denominator for overall abortion rate)
    reached = _reached_counts(df, position_data)

    # Per-position table (add overall abortion rate using reached counts)
    pos_rows = []
    for pos in positions:
        sub_all = aborted_all[aborted_all[pos_col] == pos]
        if sub_all.empty:
            continue
        counts = _counts(sub_all)
        n_total = counts.pop("Total Abortions")
        reached_pos = int(reached.get(int(pos), 0))
        pos_rows.append({
            "Position": int(pos),
            "Total Abortions": n_total,
            "Reached Trials": reached_pos,
            "Abortion Rate": (n_total / reached_pos) if reached_pos > 0 else np.nan,
            **counts,
        })
    df_pos = pd.DataFrame(pos_rows)

    return df_odor, df_pos, df_out


@session_metric(fa_abortion_stats)
def fa_abortion_stats_session(results, return_df=False):
    trials = results.get("trial_data", pd.DataFrame())
    missing = _fa_abortion_frames_missing(trials)
    if missing is not None:
        print(missing)
        return None if not return_df else (pd.DataFrame(), pd.DataFrame(), pd.DataFrame())

    df_odor, df_pos, df_out = fa_abortion_stats(trials, results.get("position_data"))

    if not return_df:
        # Readable form for a human at a notebook; the metric stays numeric.
        shown_odor, shown_pos, shown_out = format_fa_abortion_tables(df_odor, df_pos, df_out)
        if not df_odor.empty:
            print("=== By Odor ===")
            display(shown_odor)
        if not df_pos.empty:
            print("=== By Position ===")
            display(shown_pos)
        if not df_out.empty:
            print("=== By Odor+Position ===")
            display(shown_out)
        if df_odor.empty and df_pos.empty and df_out.empty:
            print("No FA abortions found.")
    return (df_odor, df_pos, df_out) if return_df else None


def _fa_type_mask(trials, fa_type):
    """`fa_label` mask for a single label, `"all"`, or a set of labels."""
    if "fa_label" not in trials.columns:
        return pd.Series(False, index=trials.index)
    labels = trials["fa_label"].astype(str)
    if isinstance(fa_type, str):
        if fa_type.lower() == 'all':
            return labels.str.startswith('FA_', na=False)
        return labels == fa_type
    return labels.isin({str(t) for t in fa_type})


@metric(frame="trials", title="FA Port Ratio by Odor", adapter=_fa_port_payload)
def fa_port_ratio_by_odor(trials, *, fa_type="FA_time_in"):
    """Signed FA port bias per odor: `(port A - port B) / (port A + port B)`.

    0 is no preference, positive a bias towards port A. A tier-1 core on
    `trial_data` since step 6 dropped the non-initiated FAs -- which is what makes
    it the canonical target for finding 1, the FA port ratio written eight times.

    `fa_type` takes a single label, `"all"` for every `FA_*`, or a set/list of
    labels; the set form is what lets the plotters' `fa_types` filters call this
    instead of re-counting.

    Returns `{'by_odor': Series, 'counts': {odor: {port_a, port_b}},
    'total_fa_by_odor': {odor: n}}`.
    """
    empty = {'by_odor': pd.Series(dtype=float), 'counts': {}, 'total_fa_by_odor': {}}
    if trials.empty:
        return empty

    fa_all = trials[_aborted_mask(trials) & _fa_type_mask(trials, fa_type)]
    if fa_all.empty or "fa_port" not in fa_all.columns or "last_odor_name" not in fa_all.columns:
        return empty

    ratios, counts, total_fa_by_odor = {}, {}, {}
    for odor in sorted(fa_all["last_odor_name"].dropna().unique()):
        n_a, n_b = fa_port_counts(fa_all[fa_all["last_odor_name"] == odor])
        n_total = n_a + n_b
        ratios[odor] = fa_port_ratio(n_a, n_b)
        counts[odor] = {'port_a': n_a, 'port_b': n_b} if n_total > 0 else {'port_a': 0, 'port_b': 0}
        total_fa_by_odor[odor] = n_total
    return {'by_odor': pd.Series(ratios).sort_index(), 'counts': counts,
            'total_fa_by_odor': total_fa_by_odor}


@session_metric(fa_port_ratio_by_odor)
def fa_port_ratio_by_odor_session(results):
    out = fa_port_ratio_by_odor(results.get("trial_data", pd.DataFrame()))
    print("FA Port Ratio by Odor (FA_time_in):")
    if not out['counts']:
        print("  No FA data with port and odor information found.")
        return out
    for odor, ratio in out['by_odor'].items():
        c = out['counts'][odor]
        if out['total_fa_by_odor'][odor] > 0:
            print(f"  {odor}: A={c['port_a']}, B={c['port_b']}, Bias ratio: {ratio:.3f}")
    return out


def _fa_filter_mask(frame, fa_types=None):
    """The `fa_label` mask the FA-rate metrics share.

    `fa_types=None` means "any labelled false alarm", spelled as the plotters
    spell it: not the literal `nFA`, and not null. A set selects subtypes and is
    matched case-insensitively.
    """
    if "fa_label" not in frame.columns:
        return pd.Series(False, index=frame.index)
    labels = frame["fa_label"]
    lower = labels.astype(str).str.lower()
    if fa_types is None:
        return lower.ne("nfa") & labels.notna()
    return lower.isin({str(s).strip().lower() for s in fa_types})


def fa_port_label(frame):
    """`fa_port` as `"A"` / `"B"` / None, one entry per row.

    The 1-is-A, 2-is-B mapping written out at every FA-port site (finding 1).
    `fa_port_counts` counts these labels, and `pred_seq_utils.fa_analysis` buckets
    its latencies by them, so the mapping itself is stated once.
    """
    if frame is None or len(frame) == 0 or "fa_port" not in frame.columns:
        return pd.Series(dtype=object, index=getattr(frame, "index", None))
    port = pd.to_numeric(frame["fa_port"], errors="coerce")
    return port.map({1: "A", 2: "B"}).where(port.isin([1, 2]))


@metric(frame="trials")
def fa_port_counts(frame):
    """`(n_port_a, n_port_b)` over `fa_port` -- 1 is port A, 2 is port B.

    The audit's finding 1: this two-line count was written **eight** times, in
    `fa_port_ratio_by_odor` plus seven independent recomputes across
    `visualization_utils.py`. They differed only in how the frame was sliced
    beforehand and which ratio was taken afterwards, so the counter takes an
    already-sliced frame and the slicing stays with the caller (or goes through
    `by_group`).
    """
    if frame is None or len(frame) == 0 or "fa_port" not in frame.columns:
        return 0, 0
    port = frame["fa_port"]
    return int((port == 1).sum()), int((port == 2).sum())


def fa_port_ratio(n_a, n_b):
    """Signed port bias `(A - B) / (A + B)`; NaN when neither port fired.

    0 is no preference, positive is a bias towards port A.
    """
    total = n_a + n_b
    return (n_a - n_b) / total if total > 0 else np.nan


def fa_port_share_a(n_a, n_b):
    """Port A's share of false alarms, on 0..1 rather than -1..1.

    Derived from `fa_port_ratio` rather than recounted, per VARIANT resolutions 1
    and 2: `A/(A+B) == (r+1)/2` exactly, and recounting from `fa_port` would
    reintroduce one of the duplicate implementations finding 1 exists to remove.
    (The rescale can land a ULP away from a direct `A/(A+B)`; these values are
    plotted, never fingerprinted.)
    """
    return (fa_port_ratio(n_a, n_b) + 1.0) / 2.0


def get_fa_ratio_a_stats(subjid, dates=None, odors=['C', 'F']):
    """Per-odor `A/(A+B)` false-alarm port share, one row per session per odor.

    Checklist 6 of Phase 4a, and the odd one out: it lived in
    `visualization/visualization_utils.py` but **contains no plotting at all**, so
    it moves here wholesale rather than being repointed. Its FA filter is every
    `FA_*` label, wider than `plot_fa_ratio_a_over_sessions`' single `fa_type`.

    The share is rescaled from `fa_port_ratio`, never recounted (VARIANT
    resolution 2).

    Returns
    -------
    DataFrame with columns date, session_num, odor, fa_ratio_a, n_fa_a, n_fa_b,
    n_total -- empty if the subject has no FA data for `odors`.
    """
    derivatives_dir = get_derivatives_root()

    rows = []

    for sid, subj_dir in _iter_subject_dirs(derivatives_dir, [subjid]):
        ses_dirs = _filter_session_dirs(subj_dir, dates)

        for session_num, ses in enumerate(ses_dirs, start=1):
            date_str = ses.name.split("_date-")[-1]
            results_dir = ses / "saved_analysis_results"

            if not results_dir.exists():
                continue

            ab_det = _load_trial_views(results_dir)["aborted_fa"]
            if not ab_det.empty:
                needed_cols = ['fa_label', 'last_odor_name', 'fa_port']
                ab_det = ab_det[[col for col in needed_cols if col in ab_det.columns]]

            if ab_det.empty or 'fa_label' not in ab_det.columns:
                continue

            # Same FA filter as plot_fa_ratio_a_over_sessions, widened to every FA_*.
            fa_all = ab_det[ab_det['fa_label'].astype(str).str.startswith('FA_', na=False)]
            if (fa_all.empty or 'fa_port' not in fa_all.columns
                    or 'last_odor_name' not in fa_all.columns):
                continue

            # Count over every odor present, then keep the requested ones.
            for odor in sorted(fa_all['last_odor_name'].dropna().unique()):
                if str(odor) not in [str(o) for o in odors]:
                    continue

                n_a, n_b = fa_port_counts(fa_all[fa_all['last_odor_name'] == odor])
                rows.append({
                    "date": int(date_str),
                    "session_num": session_num,
                    "odor": str(odor),
                    "fa_ratio_a": fa_port_share_a(n_a, n_b),
                    "n_fa_a": n_a,
                    "n_fa_b": n_b,
                    "n_total": n_a + n_b,
                })

    if not rows:
        print(f"No FA data found for subject {subjid} with odors {odors}")
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    print(f"\n{'='*70}")
    print(f"FA Ratio A/(A+B) Summary - Subject {str(subjid).zfill(3)}")
    print(f"{'='*70}")
    print(df.to_string(index=False))
    print(f"{'='*70}\n")

    return df


@metric(frame="trials")
def fa_rate_by_odor(trials, *, fa_types=None, odors=None):
    """FA aborts at an odor / (its passes in completed sequences + those aborts).

    Checklist 1. The denominator matches no canonical metric: not `FA_odor_bias`
    (aborts@odor) and not `odorx_abortion_rate` (presentations@odor). It counts
    how often the odor was sampled and passed, plus the times it was false-alarmed
    on -- so the rate answers "when this odor came up, how often did it draw a
    false alarm".

    `odors` fixes the index (and its order); by default every odor seen is
    reported. An odor with a zero denominator is omitted, not reported as 0.
    """
    if trials.empty:
        return pd.Series(dtype=float)
    aborted = _aborted_mask(trials)
    completed = trials[~aborted]
    ab = trials[aborted]
    ab_fa = ab[_fa_filter_mask(ab, fa_types)]

    completed_counts: dict = {}
    if "odor_sequence" in completed.columns:
        for seq in completed["odor_sequence"]:
            for tok in odor_sequence_tokens(seq):
                if tok is None or (isinstance(tok, float) and np.isnan(tok)):
                    continue
                letter = odor_letter(tok)
                completed_counts[letter] = completed_counts.get(letter, 0) + 1

    fa_counts: dict = {}
    if "last_odor_name" in ab_fa.columns:
        for last in ab_fa["last_odor_name"]:
            if last is None or (isinstance(last, float) and np.isnan(last)):
                continue
            letter = odor_letter(last)
            fa_counts[letter] = fa_counts.get(letter, 0) + 1

    keys = ([odor_letter(o) for o in odors] if odors is not None
            else sorted(set(completed_counts) | set(fa_counts)))
    rates = {}
    for od in keys:
        denom = completed_counts.get(od, 0) + fa_counts.get(od, 0)
        if denom > 0:
            rates[od] = fa_counts.get(od, 0) / denom
    return pd.Series(rates, dtype=float)


@metric(frame="trials+position_data")
def fa_rate_by_position(trials, position_data, *, fa_types=None):
    """FA aborts at position *p* / trials that reached *p*.

    Checklist 5. The denominator is `frames.reached_counts`, the package's single
    definition of "reached" (audit Q5). The plotter used to count the positions
    listed in each trial's `presentations` blob -- Q5's "definition C", now
    deleted -- so the drawn denominators change here even though no saved metric
    value does.
    """
    if trials.empty:
        return pd.Series(dtype=float)
    reached = _reached_counts(trials, position_data)
    aborted = trials[_aborted_mask(trials)]
    fa = aborted[_fa_filter_mask(aborted, fa_types)]
    fa_counts: dict = {}
    if "last_odor_position" in fa.columns:
        pos = pd.to_numeric(fa["last_odor_position"], errors="coerce").dropna().astype(int)
        fa_counts = {int(p): int(n) for p, n in pos.value_counts().items()}
    rates = {p: fa_counts.get(p, 0) / n for p, n in reached.items() if n > 0}
    return pd.Series(rates, dtype=float).sort_index()


@metric(frame="trials")
def fa_latency_from_pokeout(trials, *, fa_types=None):
    """`fa_time` minus the animal's last cue-port exit before it, in ms. Checklist 19.

    **Not** `trial_data.fa_window_latency_ms`, which is measured from the abortion timestamp and is
    what `fa_label` buckets (`DECISIONS.md` section 16 -- (a) vs (b)).

    Reads `fa_response_time_ms` rather than re-deriving the anchor. It used to compute it
    from `position_data.poke_odor_end`, which is wrong twice over: that timestamp is synthetic
    and 25 ms late whenever the pre-odor grace produced the entry (section 15), and it does not
    exclude the animal returning to the cue port between giving up and false-alarming, which
    happens on 44% of false alarms (section 16). Two independent derivations of one quantity is
    exactly what section 14 is about, so there is now one.

    Returns empty when the column is absent. Sessions saved before Phase 11 never carry it, and
    silently falling back to the old computation would make old and new sessions look
    comparable when they measure different things -- the section 2 rule for absent provenance.
    """
    if "fa_response_time_ms" not in trials.columns or "global_trial_id" not in trials.columns:
        return pd.Series(dtype=float)
    selected = trials[_fa_filter_mask(trials, fa_types)] if fa_types is not None else trials
    if selected.empty:
        return pd.Series(dtype=float)
    latencies = pd.to_numeric(selected["fa_response_time_ms"], errors="coerce")
    return pd.Series(latencies.to_numpy(),
                     index=selected["global_trial_id"].to_numpy(),
                     dtype=float).dropna()


def false_response_ratio_contributions(trials, *, fr_types=None):
    completed = ~_aborted_mask(trials)
    if "false_response" not in trials.columns:
        return pd.Series(0, index=trials.index), completed.astype(int)
    fr = trials["false_response"] == True  # noqa: E712 (element-wise, NaN-safe)
    if fr_types is not None and "fr_label" in trials.columns:
        wanted = {fr_types} if isinstance(fr_types, str) else set(fr_types)
        fr = fr & trials["fr_label"].isin(wanted)
    return (completed & fr).astype(int), completed.astype(int)


@metric(frame="trials")
def false_response_ratio(trials, *, fr_types=None):
    """False-response trials / completed trials. Checklist 22.

    **Not** the single-reward `fa_rate`, which is `false_alarm / n_nogo` off a
    different column (`fa_label`, not `fr_label`). `fr_types=None` counts every
    `false_response == True` trial whatever its label.
    """
    if trials.empty:
        return 0, 0, np.nan
    return _reduce_rate(*false_response_ratio_contributions(trials, fr_types=fr_types))
