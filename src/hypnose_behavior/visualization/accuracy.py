# Defers evaluation of PEP-604 annotations (`X | None`), keeping this module
# importable on Python 3.9 for repos pinned there (hypnose-eeg-preprocessing).
from __future__ import annotations

"""Decision-accuracy figures.

The three here answer the same question at three grains: per session across
animals, split by odor, and rolling within a session.
"""

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from hypnose_behavior.metric_analysis.metrics.accuracy import (
    decision_accuracy,
    global_choice_accuracy,
    rolling_reward_fraction,
)
from hypnose_behavior.metric_analysis.metrics.hidden_rule import hidden_rule_mask
from hypnose_behavior.metric_analysis.resolvers import by_group
from hypnose_behavior.io.layout import (
    _filter_session_dirs,
    _iter_subject_dirs,
    derivatives,
    normalize_subjid,
    session_selectors,
)
from hypnose_behavior.io.paths import (
    get_rawdata_root,
    get_derivatives_root,
    get_server_root,
)
import numpy as np
from hypnose_behavior.io.save import save_figure
from hypnose_behavior.visualization.panels import _clean_graph
from hypnose_behavior.io.loaders import (
    _load_table_with_trial_data,
    _load_trial_views,
)
from hypnose_behavior.visualization.prep import (
    _computed_metric,
    _series_line_widths,
)



def plot_decision_accuracy_by_odor(
    subjid,
    dates=None,
    figsize=(12, 6),
    plot_choice_acc=False,
    plot_AB=True,
    clean_graph=False,
    *,
    ses=None,
    index=None,
    date_range=None,
    ses_range=None,
    index_range=None,
    save=False,
    verbose=True,
):
    """
    Plot decision accuracy by odor (A, B) and total over dates.
    Optionally include global choice accuracy as a separate line.
    Fast version using pre-computed metrics with existing helper functions.
    
    Parameters:
    -----------
    subjid : int
        Subject ID
    dates : tuple, list, or None
        Date or date range. If None, plots all available dates.
    figsize : tuple, optional
        Figure size (default: (12, 6))
    plot_choice_acc : bool, optional
        If True, also plot global choice accuracy as a dark grey line (default: False)
    plot_AB : bool, optional
        If True, plot odor-specific accuracies for A and B (default: True). If False, omit A/B lines.
    clean_graph : bool, optional
        If True, print and clear title/labels/ticks/legend for external editing.
    
    Returns:
    --------
    fig, ax : matplotlib figure and axes

    ``ses`` / ``index`` / ``date_range`` / ``ses_range`` / ``index_range`` narrow the
    selection further; they intersect with ``dates`` and with each other, and ``index``
    is the subject's gap-free chronological rank (`io.layout.session_selectors`).
    """
    select = session_selectors(
        ses=ses, index=index, date_range=date_range,
        ses_range=ses_range, index_range=index_range,
    )
    rows = []
    base_path = get_rawdata_root()
    server_root = get_server_root()
    derivatives_dir = get_derivatives_root()
    
    def _normalize_odor_label(odor_raw):
        """Map assorted odor keys to canonical labels (A/B/Total/other)."""
        if isinstance(odor_raw, (int, float)) and not np.isnan(odor_raw):
            val = int(odor_raw)
            if val in (0, 1):
                return "A" if val == 0 else "B"
        if isinstance(odor_raw, str):
            raw = odor_raw.strip()
            lower = raw.lower()
            base = lower.replace("odor", "").replace("_", "").replace(" ", "")
            if base in {"a", "1", "01"}:
                return "A"
            if base in {"b", "2", "02"}:
                return "B"
            if lower in {"total", "overall"}:
                return "Total"
        return str(odor_raw)

    def _collect_odor_acc_rows(acc_block, date_int):
        """Handle both legacy flat dicts and new nested decision_accuracy_by_odor blocks."""
        collected = []

        def add_from_dict(dct):
            for odor, acc in dct.items():
                if isinstance(acc, (int, float)) and not np.isnan(acc):
                    collected.append({
                        "date": date_int,
                        "odor": _normalize_odor_label(odor),
                        "accuracy": float(acc)
                    })

        if not isinstance(acc_block, dict):
            return collected

        if "decision_accuracy_ab" in acc_block:
            add_from_dict(acc_block.get("decision_accuracy_ab", {}))
        if "decision_accuracy_total" in acc_block:
            add_from_dict(acc_block.get("decision_accuracy_total", {}))

        # If neither of the new-schema keys are present, assume legacy flat mapping
        if not collected:
            add_from_dict(acc_block)

        return collected

    for sid, subj_dir in _iter_subject_dirs(derivatives_dir, [subjid]):
        ses_dirs = _filter_session_dirs(subj_dir, dates, **select)
        for ses_dir in ses_dirs:
            date_str = ses_dir.name.split("_date-")[-1]
            results_dir = ses_dir / "saved_analysis_results"
            if not results_dir.exists():
                continue
            
            # Computed through the registry, not read from metrics_*.json --
            # `decision_accuracy` was one of the three quantities this repo
            # obtained both ways (`docs/DECISIONS.md` section 5). `_computed_metric`
            # returns the saved key's exact shape, so `_collect_odor_acc_rows`
            # below is unchanged.
            td = _load_trial_views(results_dir).get("trial_data", pd.DataFrame())
            if td.empty:
                continue

            # Add odor-specific accuracies (supports legacy flat dict and new nested schema)
            rows.extend(_collect_odor_acc_rows(
                _computed_metric(results_dir, "decision_accuracy_by_odor") or {}, int(date_str)))

            # Add total accuracy
            acc_total = decision_accuracy(td)[2]
            if isinstance(acc_total, (int, float)) and not np.isnan(acc_total):
                rows.append({
                    "date": int(date_str),
                    "odor": "Total",
                    "accuracy": float(acc_total)
                })

            # Add global choice accuracy if requested
            if plot_choice_acc:
                gca_value = global_choice_accuracy(td)[2]
                if isinstance(gca_value, (int, float)) and not np.isnan(gca_value):
                    rows.append({
                        "date": int(date_str),
                        "odor": "Global Choice Accuracy",
                        "accuracy": float(gca_value)
                    })
    
    if not rows:
        print("No data found")
        return None, None
    
    df = pd.DataFrame(rows)
    unique_dates = sorted(df["date"].unique())
    date_to_x = {d: i for i, d in enumerate(unique_dates)}
    df["x"] = df["date"].map(date_to_x)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    colors = {'A': '#FF6B6B', 'B': '#4ECDC4', 'Total': 'black', 'Global Choice Accuracy': 'darkgreen'}
    linewidths = {'A': 1.5, 'B': 1.5, 'Total': 4, 'Global Choice Accuracy': 3.5}
    markers = {'A': 'o', 'B': 'o', 'Total': 's', 'Global Choice Accuracy': '^'}
    linestyles = {'A': '-', 'B': '-', 'Total': '-', 'Global Choice Accuracy': '--'}
    
    # Determine which odors to plot (restricted set)
    unique_odors = set(df["odor"].unique())
    odors_to_plot = []
    if plot_AB:
        for base in ["A", "B"]:
            if base in unique_odors:
                odors_to_plot.append(base)
    if "Total" in unique_odors:
        odors_to_plot.append("Total")
    if plot_choice_acc and "Global Choice Accuracy" in unique_odors:
        odors_to_plot.append("Global Choice Accuracy")
    
    for odor in odors_to_plot:
        subset = df[df["odor"] == odor]
        if subset.empty:
            continue
        ax.plot(subset["x"].values, subset["accuracy"].values, 
                label=odor,
                color=colors.get(odor, '#999999'),
                linewidth=linewidths.get(odor, 1.5),
                linestyle=linestyles.get(odor, '-'),
                marker=markers.get(odor, 'o'),
                markersize=4 if odor not in ('Total', 'Global Choice Accuracy') else 6,
                alpha=0.7 if odor not in ('Total', 'Global Choice Accuracy') else 0.8,
                zorder=10 if odor in ('Total', 'Global Choice Accuracy') else 1)
    
    ax.set_xlabel('Day')
    ax.set_ylabel('Accuracy')
    ax.set_ylim([0, 1.05])
    ax.set_xlim([-0.1, len(unique_dates) + 0.1])
    ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=1, alpha=0.3)
    ax.legend(loc='best')
        
    # Shift tick positions left by 1 while keeping labels unchanged (day 1 plotted at x=0)
    orig_xticks = ax.get_xticks()
    # Use existing tick labels if present; otherwise derive from original tick values
    existing_labels = [lbl.get_text() for lbl in ax.get_xticklabels()]
    labels = existing_labels if any(existing_labels) else [str(int(tick)) if tick.is_integer() else f"{tick:g}" for tick in orig_xticks]
    ax.set_xticks(orig_xticks - 1)
    ax.set_xticklabels(labels)
    # Re-affirm limits so the left edge stays near 0 despite shifted ticks
    ax.set_xlim([-0.1, len(unique_dates) - 1 + 0.1])

    # Remove upper and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    title = f"Subject {str(subjid).zfill(3)} - Decision Accuracy by Odor"
    if plot_choice_acc:
        title += " (with Global Choice Accuracy)"
    ax.set_title(title)
    
    if clean_graph:
        _clean_graph(ax, xlabel="Day", ylabel="Accuracy")

    plt.tight_layout()
    
    if save:
        try:
            suffix = "with_AB" if plot_AB else "total_only"
            if plot_choice_acc:
                suffix += "_choice"
            save_name = f"decision_accuracy_by_odor_{suffix}"
            out_path = save_figure(
                fig,
                save_name,
                subjids=[subjid],
                dates=dates,
            )
            if verbose:
                print(
                    f"[plot_decision_accuracy_by_odor] Saved figure to {out_path}"
                )
        except Exception as exc:
            if verbose:
                print(
                    "[plot_decision_accuracy_by_odor] Failed to save figure: "
                    f"{exc}"
                )
    
    return fig, ax



def plot_decision_accuracy_rolling_average(
    subjid,
    dates=None,
    save=False,
    window_size=20.0,
    step_size=1.0,
    include_avg=False,
    hr_only=False,
    *,
    ses=None,
    index=None,
    date_range=None,
    ses_range=None,
    index_range=None,
):
    """
    Plot rolling decision accuracy for one subject across one or more sessions.

    Creates two figures:
    1) Completed trials only (is_aborted == False)
    2) All trials

    Decision accuracy is computed as:
    (# trials in numerator condition) / (# trials in window)

    Numerator condition:
    - hr_only=False: response_time_category == "rewarded"
    - hr_only=True: response_time_category == "rewarded" AND hidden_rule_success == True

    Rolling windows are computed within each session only (no cross-session
    sharing). The plotted line remains continuous over global trial index.

    Parameters
    ----------
    subjid : int
        Subject ID.
    dates : tuple, list, or None
        Date range tuple, explicit list of dates, or None for all sessions.
    save : bool, optional
        If True, save both figures via save_figure().
    window_size : float
        Rolling window size in trials. Converted to int and clamped to >= 1.
    step_size : float
        Step size in trials between consecutive windows. Converted to int and
        clamped to >= 1. A larger step size reduces the number of plotted points.
    include_avg : bool, optional
        If True, fill early windows of each session using session-average padding:
        rate = (sum(available_data) + missing * session_avg) / window_size.
        If False (default), windows are plotted only when a full in-session window
        is available.
    hr_only : bool, optional
        If True, numerator counts only trials that are both rewarded and
        hidden_rule_success == True. Denominator remains unchanged.

    Returns
    -------
    (fig_completed, ax_completed, fig_all, ax_all)
        Matplotlib figures and axes for completed-only and all-trials views.

    ``ses`` / ``index`` / ``date_range`` / ``ses_range`` / ``index_range`` narrow the
    selection further; they intersect with ``dates`` and with each other, and ``index``
    is the subject's gap-free chronological rank (`io.layout.session_selectors`).
    """
    select = session_selectors(
        ses=ses, index=index, date_range=date_range,
        ses_range=ses_range, index_range=index_range,
    )
    derivatives_dir = get_derivatives_root()
    window_n = max(1, int(window_size))
    step_n = max(1, int(step_size))

    # Collect per-session trial tables in chronological order.
    session_rows = []
    for sid, subj_dir in _iter_subject_dirs(derivatives_dir, [subjid]):
        ses_dirs = _filter_session_dirs(subj_dir, dates, **select)
        for ses_dir in ses_dirs:
            date_str = ses_dir.name.split("_date-")[-1]
            results_dir = ses_dir / "saved_analysis_results"
            if not results_dir.exists():
                continue
            td = _load_table_with_trial_data(results_dir, "trial_data")
            if td.empty:
                continue

            td = td.copy()
            td["is_aborted"] = td.get("is_aborted", False).fillna(False)
            td["response_time_category"] = td.get("response_time_category", "").astype(str)

            # Prefer true trial-time ordering when available.
            if "sequence_start" in td.columns:
                td["sequence_start"] = pd.to_datetime(td["sequence_start"], errors="coerce")
                td = td.sort_values("sequence_start", na_position="last")
            elif "timestamp" in td.columns:
                td["timestamp"] = pd.to_datetime(td["timestamp"], errors="coerce")
                td = td.sort_values("timestamp", na_position="last")

            td = td.reset_index(drop=True)
            td["date"] = int(date_str) if str(date_str).isdigit() else date_str
            td["_session_uid"] = str(ses_dir.name)
            session_rows.append(td)

    if not session_rows:
        print("No data found")
        return None, None, None, None

    def _build_plot_df(session_tables, completed_only: bool) -> pd.DataFrame:
        pieces = []
        global_x_counter = 0

        for ses_df in session_tables:
            df = ses_df.copy()
            if completed_only:
                df = df[df["is_aborted"] == False].copy()
            df = df.reset_index(drop=True)
            if df.empty:
                continue

            rewarded_mask = (df["response_time_category"] == "rewarded")
            if hr_only:
                hr_mask = df.get("hidden_rule_success", False)
                if isinstance(hr_mask, pd.Series):
                    hr_mask = hr_mask.fillna(False).astype(bool)
                else:
                    hr_mask = pd.Series(False, index=df.index)
                numerator_mask = rewarded_mask & hr_mask
            else:
                numerator_mask = rewarded_mask

            df["is_rewarded"] = numerator_mask.astype(int)
            n_trials = len(df)

            # The windowing rule is `rolling_reward_fraction`, whose denominator is
            # the window rather than rewarded+unrewarded -- deliberately not
            # `over_windows(decision_accuracy, ...)`, which draws a different curve.
            df["decision_accuracy"] = rolling_reward_fraction(
                df, window_n, step=step_n, include_avg=include_avg, hr_only=hr_only)

            if include_avg:
                # In include_avg mode, keep one x-unit per trial.
                x_local = np.arange(1, n_trials + 1)
            else:
                # Shift x so first valid full window of each session is at x=1.
                # Example window=30: point at trial 30 is displayed at session x=1.
                x_local = np.arange(1, n_trials + 1) - (window_n - 1)

            # Keep global trial index for debugging/reference.
            df["trial_idx"] = np.arange(1, n_trials + 1)

            # Display x-index used for plotting; session-wise compressed in standard mode.
            df["plot_x_idx"] = x_local + global_x_counter
            if include_avg:
                session_span = n_trials
            else:
                # Visual span equals number of possible full-window endpoints.
                session_span = max(1, n_trials - window_n + 1)
            global_x_counter += session_span
            pieces.append(df)

        if not pieces:
            return pd.DataFrame()
        return pd.concat(pieces, ignore_index=True)

    def _session_start_positions(plot_df: pd.DataFrame):
        if plot_df.empty or "_session_uid" not in plot_df.columns:
            return [], []
        valid = plot_df.dropna(subset=["decision_accuracy"])
        if valid.empty:
            return [], []
        starts = valid.groupby("_session_uid", sort=False)["plot_x_idx"].min().sort_values()
        return starts.values.tolist(), [str(s) for s in starts.index.tolist()]

    def _draw_plot(plot_df: pd.DataFrame, title: str):
        fig, ax = plt.subplots(figsize=(12, 6))

        if plot_df.empty:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
            return fig, ax

        # Plot each session separately so no line bridges session boundaries.
        if "_session_uid" in plot_df.columns:
            for _, ses_df in plot_df.groupby("_session_uid", sort=False):
                if ses_df.empty:
                    continue
                valid = ses_df.dropna(subset=["decision_accuracy"])
                if valid.empty:
                    continue
                ax.plot(
                    valid["plot_x_idx"].values,
                    valid["decision_accuracy"].values,
                    color="black",
                    linewidth=2.0,
                    alpha=0.9,
                )
        else:
            valid = plot_df.dropna(subset=["decision_accuracy"])
            if valid.empty:
                valid = plot_df
            ax.plot(
                valid["plot_x_idx"].values,
                valid["decision_accuracy"].values,
                color="black",
                linewidth=2.0,
                alpha=0.9,
            )

        start_x, _ = _session_start_positions(plot_df)
        if start_x:
            for i, x in enumerate(start_x):
                ax.axvline(
                    x=x,
                    color="#1f77b4",
                    linestyle=":",
                    linewidth=1.4,
                    alpha=0.9,
                    zorder=1,
                    label="Session start" if i == 0 else None,
                )

        ax.set_xlabel("Trials")
        ax.set_ylabel("Decision Accuracy")
        ax.set_ylim(0, 1.05)
        x_max = int(np.nanmax(plot_df["plot_x_idx"].values)) if not plot_df.empty else 1
        ax.set_xlim(1, max(x_max, 1))
        ax.set_title(title)
        ax.grid(False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if start_x:
            ax.legend(loc="lower right")

        plt.tight_layout()
        return fig, ax

    completed_df = _build_plot_df(session_rows, completed_only=True)
    all_df = _build_plot_df(session_rows, completed_only=False)

    mode_label = "include_avg" if include_avg else "standard"
    hr_label = "hr_only" if hr_only else "all_rewarded"

    fig_completed, ax_completed = _draw_plot(
        completed_df,
        f"Subject {str(subjid).zfill(3)} - Decision Accuracy Rolling Average (Completed Only, window={window_n}, step={step_n}, mode={mode_label}, numerator={hr_label})",
    )
    fig_all, ax_all = _draw_plot(
        all_df,
        f"Subject {str(subjid).zfill(3)} - Decision Accuracy Rolling Average (All Trials, window={window_n}, step={step_n}, mode={mode_label}, numerator={hr_label})",
    )

    if save:
        try:
            save_figure(
                fig_completed,
                f"decision_accuracy_rolling_average_completed_w{window_n}_s{step_n}_{mode_label}_{hr_label}",
                subjids=[subjid],
                dates=dates,
            )
        except Exception as exc:
            print(
                "[plot_decision_accuracy_rolling_average] Failed to save completed-only figure: "
                f"{exc}"
            )
        try:
            save_figure(
                fig_all,
                f"decision_accuracy_rolling_average_all_trials_w{window_n}_s{step_n}_{mode_label}_{hr_label}",
                subjids=[subjid],
                dates=dates,
            )
        except Exception as exc:
            print(
                "[plot_decision_accuracy_rolling_average] Failed to save all-trials figure: "
                f"{exc}"
            )

    return fig_completed, ax_completed, fig_all, ax_all



def plot_decision_accuracy(
    subjids,
    dates=None,
    figsize=(10, 7),
    title=None,
    *,
    ses=None,
    index=None,
    date_range=None,
    ses_range=None,
    index_range=None,
    save=False,
    verbose=True,
    show_title=True,
    show_legend=True,
    color_by_id=True,
    mean=False,
    show_criterion=False,
    criterion=0.8,
):
    """Decision accuracy over training days, one series per subject.

    Each animal is drawn as a colored line: its per-session decision accuracy
    (rewarded / (rewarded + unrewarded), same as the ``decision_accuracy`` metric,
    computed here from ``trial_data``) plotted against day index. Day 1 is each
    animal's first session with an A/B decision, so animals are aligned by
    training day rather than calendar date. Markers show each subject's value on
    each day, matching the subject-series style of :func:`plot_cumulative_rewards`.

    If ``mean=True``, a thicker black line shows the mean across animals at each
    day index. At a
    given day the mean uses only the animals that have data there, so as animals
    run out of sessions the mean is averaged over fewer of them.

    Hidden-rule split: if any session in the input contains hidden-rule trials
    (``hidden_rule_success == True``), decision accuracy is computed separately
    for non-HR and HR trials. The non-HR accuracy is drawn as the usual solid
    line; the HR accuracy is drawn as a dashed line in the same per-animal color
    (present only on days that have HR trials). The group mean is likewise split
    into a solid (non-HR) and dashed (HR) black line. If no session has HR trials,
    behavior is unchanged (single solid line = overall accuracy).

    Parameters
    ----------
    subjids : int | list[int] | dict
        Subject id(s). May also be a dict ``{subjid: date_range}`` as a
        convenience shorthand — in that case the dict is used as ``dates`` and the
        subjids are its keys.
    dates : list | tuple | dict | None
        Specific dates [YYYYMMDD, ...] or inclusive range (start, end). If a dict,
        must map ``subjid → date_range`` so each subject can use its own date
        window. Subjids not present as keys are skipped with a warning.
        ``None`` = all sessions for every subject.
    figsize : tuple
    title : str | None
    save : bool
    verbose : bool
    show_title : bool
        If False, no title is rendered (useful for poster-style figures).
    show_legend : bool
        If True, show a subject legend (default: True).
    color_by_id : bool
        If True, each animal's thin line is colored using the shared per-subject
        tab20 palette (:func:`plot_cumulative_rewards`). Colors are assigned by
        ascending subject id, so the same ids keep the same colors across plots.
    mean : bool
        If True, overlay the thick group-mean line. If False (default), no mean
        line is drawn and the per-animal lines are drawn thicker. Line widths are
        shared with :func:`plot_behavior_metrics` via ``_series_line_widths``.
    show_criterion : bool
        If True, draw a dashed horizontal criterion line (default: False).
    criterion : float
        Y-value for the criterion line (default: 0.8).

    Returns
    -------
    fig, ax

    ``ses`` / ``index`` / ``date_range`` / ``ses_range`` / ``index_range`` narrow the
    selection further; they intersect with ``dates`` and with each other, and ``index``
    is the subject's gap-free chronological rank (`io.layout.session_selectors`).
    """
    select = session_selectors(
        ses=ses, index=index, date_range=date_range,
        ses_range=ses_range, index_range=index_range,
    )
    # Mirror plot_behavior_metrics's input flexibility.
    if isinstance(subjids, dict):
        dates = subjids if not isinstance(dates, dict) or dates is None else dates
        subjids = list(subjids.keys())
    elif isinstance(dates, dict) and subjids is None:
        subjids = list(dates.keys())
    elif isinstance(subjids, set):
        subjids = sorted(subjids)
    elif not isinstance(subjids, (list, tuple)):
        subjids = [subjids]

    def _dates_for(subjid):
        if not isinstance(dates, dict):
            return dates
        if subjid in dates:
            return dates[subjid]
        try:
            int_key = int(subjid)
            if int_key in dates:
                return dates[int_key]
        except (TypeError, ValueError):
            pass
        str_key = str(subjid)
        if str_key in dates:
            return dates[str_key]
        return None

    def _dates_ok(date_range):
        """Reject malformed date tokens (must be 8-digit YYYYMMDD).

        A typo like ``2025118`` (7 digits) or ``202251120`` (9 digits) would
        otherwise be treated by ``_filter_session_dirs`` as a numeric range
        endpoint and silently match every real session, so we guard here.
        """
        def _ok(tok):
            s = str(tok)
            return s.isdigit() and len(s) == 8
        if date_range is None:
            return True
        if isinstance(date_range, tuple):
            return all(t is None or _ok(t) for t in date_range)
        if isinstance(date_range, (list, set)):
            return all(_ok(t) for t in date_range)
        return _ok(date_range)

    derivatives_dir = get_derivatives_root()

    def _decision_acc_split(td):
        """``(non_hr_accuracy, hr_accuracy)`` for one session.

        The HR / non-HR split is a *granularity* of `decision_accuracy`, not a metric
        of its own, so it is `by_group` over the canonical HR mask. A side with no
        trials is absent from the grouping and comes back as NaN, which is what the
        callers below test for.
        """
        acc = by_group(decision_accuracy, td, hidden_rule_mask(td)).reindex([False, True])
        return acc.iloc[0], acc.iloc[1]

    # Per-animal, day-aligned accuracy for non-HR ("main") and HR trials (day 1 =
    # first session with an A/B decision). HR splitting only matters if any
    # session actually has hidden-rule trials (hr_active).
    per_animal_series: dict = {}
    hr_active = False
    for subjid in subjids:
        subj_dates = _dates_for(subjid)
        if isinstance(dates, dict) and subj_dates is None:
            if verbose:
                print(f"Warning: No date range provided in dict for subject {subjid}, skipping")
            continue
        if not _dates_ok(subj_dates):
            if verbose:
                print(f"Warning: subject {subjid} has malformed date(s) {subj_dates!r} "
                      f"(expected 8-digit YYYYMMDD); skipping")
            continue

        subj_str = normalize_subjid(subjid)
        subj_dir = derivatives.subject_dir(subjid, missing_ok=True)
        if subj_dir is None:
            if verbose:
                print(f"Warning: No subject directory found for {subj_str}")
            continue

        main_vals, hr_vals = [], []
        for ses_dir in _filter_session_dirs(subj_dir, subj_dates, **select):
            results_dir = ses_dir / "saved_analysis_results"
            if not results_dir.exists():
                continue
            td = _load_trial_views(results_dir)["trial_data"]
            if td.empty or "response_time_category" not in td.columns:
                continue
            non_hr_acc, hr_acc = _decision_acc_split(td)
            # A session counts as a "day" only if it has an A/B decision.
            if np.isnan(non_hr_acc) and np.isnan(hr_acc):
                continue
            main_vals.append(non_hr_acc)
            hr_vals.append(hr_acc)
            if not np.isnan(hr_acc):
                hr_active = True

        if main_vals:
            per_animal_series[int(subjid)] = {"main": main_vals, "hr": hr_vals}

    if not per_animal_series:
        print("No data found")
        return None, None

    fig, ax = plt.subplots(figsize=figsize)

    # Per-subject color map (shared palette with plot_cumulative_rewards).
    # Sorted by ascending id so the same subject keeps its color across plots.
    subj_colors = {s: plt.cm.tab20(i % 20) for i, s in enumerate(sorted(subjids))}

    # Line widths shared with plot_behavior_metrics.
    per_series_lw, mean_lw = _series_line_widths(mean)

    # HR (dashed) lines are nudged up ~2.5 points and get small markers so they
    # stay visible when they exactly overlap the non-HR line, and so isolated HR
    # days (a gap on either side) still show up as a point.
    from matplotlib.transforms import offset_copy
    hr_offset = offset_copy(ax.transData, fig=fig, y=2.5, units="points")

    # Per-subject dash phase so overlapping HR (dashed) lines interleave — one
    # animal's dashes fall in another's gaps — instead of hiding each other
    # (probe accuracy is often a flat 1.0 for every animal, so they coincide).
    sorted_ids = sorted(subjids)
    n_ids = max(len(sorted_ids), 1)
    dash_on = dash_off = 6
    dash_period = dash_on + dash_off
    subj_dash_phase = {s: dash_period * i / n_ids for i, s in enumerate(sorted_ids)}

    # Line per animal, aligned so day 1 = first session with data. When HR trials
    # are present, the non-HR accuracy is the solid line and HR accuracy is a
    # dashed line in the same color (NaN days leave gaps).
    max_days = max(len(v["main"]) for v in per_animal_series.values())
    for subjid, series in per_animal_series.items():
        color = subj_colors[subjid] if color_by_id else "grey"
        alpha = 0.7 if color_by_id else 0.6
        main = np.array(series["main"], dtype=float)
        x = np.arange(1, len(main) + 1)
        ax.plot(
            x, main,
            color=color,
            linewidth=per_series_lw,
            alpha=alpha,
            marker="o",
            markersize=4,
            zorder=2,
        )
        if hr_active:
            hr = np.array(series["hr"], dtype=float)
            hr_ls = (subj_dash_phase.get(subjid, 0.0), (dash_on, dash_off))
            ax.plot(x, hr, color=color, linewidth=per_series_lw, alpha=alpha,
                    linestyle=hr_ls, marker="o", markersize=4,
                    transform=hr_offset, zorder=2.5)

    # Group mean at each day index, over whichever animals have data there.
    if mean:
        def _day_mean(key):
            mx, my = [], []
            for day in range(1, max_days + 1):
                vals = [s[key][day - 1] for s in per_animal_series.values()
                        if len(s[key]) >= day and not np.isnan(s[key][day - 1])]
                if vals:
                    mx.append(day)
                    my.append(float(np.mean(vals)))
            return mx, my

        mx, my = _day_mean("main")
        ax.plot(mx, my, color="black", linewidth=mean_lw, zorder=3)
        if hr_active:
            hx, hy = _day_mean("hr")
            ax.plot(hx, hy, color="black", linewidth=mean_lw, linestyle="--",
                    marker="o", markersize=5, transform=hr_offset, zorder=3.5)

    ax.set_xlabel("Day")
    ax.set_ylabel("Decision Accuracy")
    ax.set_xlim(0.8, max_days + 0.5)
    ax.set_ylim(0, 1.05)
    if show_criterion:
        ax.axhline(
            y=criterion,
            color="gray",
            linestyle="--",
            linewidth=1.2,
            alpha=0.8,
            zorder=1,
        )
    # Only ever tick whole days (never 1.5, 2.5, ...). Local import so autoreload
    # picks it up without a kernel restart.
    from matplotlib.ticker import MaxNLocator
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    subj_leg = None
    if show_legend and color_by_id:
        handles = [
            Line2D([0], [0], color=subj_colors[s], linewidth=1.5, label=f"Sub {str(s).zfill(3)}")
            for s in sorted(per_animal_series.keys())
        ]
        if handles:
            subj_leg = ax.legend(handles=handles, title="Subject", loc="best")

    if show_legend and hr_active:
        # Solid = non-HR, dashed = HR. Keep the subject legend too, if present.
        if subj_leg is not None:
            ax.add_artist(subj_leg)
        style_handles = [
            Line2D([0], [0], color="black", linestyle="-", linewidth=2.0, label="Non-HR"),
            Line2D([0], [0], color="black", linestyle="--", linewidth=2.0,
                   marker="o", markersize=5, label="HR"),
        ]
        ax.legend(handles=style_handles, title="Trial type", loc="lower right")

    if show_title:
        ax.set_title(title if title else "Decision Accuracy over Day")

    fig.tight_layout(pad=1.5)

    if save:
        try:
            if isinstance(dates, dict):
                save_dates = []
                for v in dates.values():
                    if isinstance(v, (list, tuple)):
                        save_dates.extend(v)
                    elif v is not None:
                        save_dates.append(v)
            else:
                save_dates = dates
            out_path = save_figure(
                fig, "decision_accuracy",
                subjids=list(subjids), dates=save_dates,
            )
            if verbose:
                print(f"[plot_decision_accuracy] Saved figure to {out_path}")
        except Exception as exc:
            if verbose:
                print(f"[plot_decision_accuracy] Failed to save figure: {exc}")

    plt.show()
    return fig, ax
