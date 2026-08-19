# Defers evaluation of PEP-604 annotations (`X | None`), keeping this module
# importable on Python 3.9 for repos pinned there (hypnose-eeg-preprocessing).
from __future__ import annotations

"""Sequence-position figures.

``presentations`` answers "what did the rig deliver" and
``poke_source`` answers "what did the animal sample". A per-position denominator
here counts the first.
"""

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from hypnose_behavior.metric_analysis.metrics.sequence import abortion_rate_positionX
from hypnose_behavior.utils.helpers import (
    _filter_session_dirs,
    session_selectors,
)
from hypnose_behavior.io.layout import (
    derivatives,
    normalize_subjid,
)
from hypnose_behavior.io.paths import get_derivatives_root
import numpy as np
from hypnose_behavior.io.save import save_figure
from hypnose_behavior.visualization.primitives import mean_sem
from hypnose_behavior.io.loaders import (
    _load_position_data,
    _load_trial_views,
)



def plot_position_completion_rate(
    subjids,
    dates=None,
    positions=(1, 2, 3, 4),
    figsize=(8, 6.8),
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
    color_by_id=False,
    avg_per_animal=False,
):
    """Per-position completion rate across sessions (dot plot with mean ± SD).

    For each session:
    - `max_pos = max(num_odors)` over all trials in the session.
    - Each completed trial (`is_aborted == False`) contributes one completed-count
      to every position 1..max_pos.
    - Each aborted trial (`is_aborted == True`) contributes one completed-count
      to positions 1..last_odor_position - 1 and one aborted-count at
      last_odor_position.
    - Completion rate at position p =
          completed[p] / (completed[p] + aborted[p]) * 100

    Each session yields one rate per requested position; rates are plotted as
    blue dots horizontally jittered around each x-tick, with a black mean line
    and SD error bars.

    Parameters
    ----------
    subjids : int | list[int] | dict
        Subject id(s). May also be a dict ``{subjid: date_range}`` as a convenience
        shorthand — in that case the dict is used as ``dates`` and the subjids are
        its keys.
    dates : list | tuple | dict | None
        Specific dates [YYYYMMDD, ...] or inclusive range (start, end). If a dict,
        must map ``subjid → date_range`` (each value itself a list/tuple/None
        passed through to ``_filter_session_dirs``); this lets each subject use
        its own date window — useful when animals are offset in calendar time.
        Subjids not present as keys are skipped with a warning. ``None`` = all
        sessions for every subject.
    positions : iterable[int]
        Positions to display on the x-axis (e.g. [1, 2, 3, 4]).
    figsize : tuple
    title : str | None
    save : bool
    verbose : bool
    show_title : bool
        If False, no title is rendered (useful for poster-style figures).
    color_by_id : bool
        If True, each animal's dots are colored consistently using the same
        per-subject tab20 palette as :func:`plot_cumulative_rewards`.
    avg_per_animal : bool
        If True, no individual session dots are drawn. Instead each animal's
        session rates at a position are shown as a small violin (one violin per
        animal per position, spread horizontally within the position slot). The
        black line then shows the mean ± SEM computed across animals (each animal
        contributing its mean of session rates). Violins are colored by subject
        when ``color_by_id`` is True.

    Returns
    -------
    fig, ax

    ``ses`` / ``index`` / ``date_range`` / ``ses_range`` / ``index_range`` narrow the
    selection further; they intersect with ``dates`` and with each other, and ``index``
    is the subject's gap-free chronological rank (`utils.helpers.session_selectors`).
    """
    select = session_selectors(
        ses=ses, index=index, date_range=date_range,
        ses_range=ses_range, index_range=index_range,
    )
    # Mirror plot_cumulative_rewards's input flexibility.
    if isinstance(subjids, dict):
        dates = subjids if not isinstance(dates, dict) or dates is None else dates
        subjids = list(subjids.keys())
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

    derivatives_dir = get_derivatives_root()
    positions = list(positions)
    rates_per_position: dict[int, list[float]] = {p: [] for p in positions}
    subj_per_position: dict[int, list] = {p: [] for p in positions}

    for subjid in subjids:
        subj_dates = _dates_for(subjid)
        if isinstance(dates, dict) and subj_dates is None:
            print(f"Warning: No date range provided in dict for subject {subjid}, skipping")
            continue

        subj_str = normalize_subjid(subjid)
        subj_dir = derivatives.subject_dir(subjid, missing_ok=True)
        if subj_dir is None:
            if verbose:
                print(f"Warning: No subject directory found for {subj_str}")
            continue

        ses_dirs = _filter_session_dirs(subj_dir, subj_dates, **select)
        for ses_dir in ses_dirs:
            results_dir = ses_dir / "saved_analysis_results"
            if not results_dir.exists():
                continue
            views = _load_trial_views(results_dir)
            td = views["trial_data"]
            if td.empty:
                continue

            abortion = abortion_rate_positionX(td, _load_position_data(results_dir, td))

            for p in positions:
                if p not in abortion.index:
                    continue
                rate = 1.0 - float(abortion.loc[p])
                if np.isnan(rate):
                    continue
                rates_per_position[p].append(rate)
                subj_per_position[p].append(subjid)

    fig, ax = plt.subplots(figsize=figsize)
    rng = np.random.default_rng(0)
    x_idx_array = np.arange(len(positions))
    halfwidth = 0.25  # horizontal extent of both mean line and dot jitter

    # Per-subject color map (shared palette with plot_cumulative_rewards).
    # Sorted by ascending id so the same subject keeps its color across plots.
    subj_colors = {s: plt.cm.tab20(i % 10) for i, s in enumerate(sorted(subjids))}

    for x_idx, p in enumerate(positions):
        rates = np.array(rates_per_position[p], dtype=float)
        if rates.size == 0:
            continue
        subj_ids = subj_per_position[p]

        if avg_per_animal:
            # One violin per animal (distribution of that animal's session
            # rates), spread horizontally within the position slot.
            per_animal: dict = {}
            for s, r in zip(subj_ids, rates):
                per_animal.setdefault(s, []).append(r)
            subj_order = [s for s in subjids if s in per_animal]
            n = len(subj_order)
            offsets = np.linspace(-halfwidth, halfwidth, n) if n > 1 else np.array([0.0])
            vwidth = (2 * halfwidth / max(n, 1)) * 0.8

            for subj, off in zip(subj_order, offsets):
                vals = np.array(per_animal[subj], dtype=float)
                color = subj_colors[subj] if color_by_id else "tab:blue"
                if vals.size >= 2:
                    parts = ax.violinplot([vals], positions=[x_idx + off],
                                          widths=vwidth, showextrema=False)
                    for body in parts["bodies"]:
                        body.set_facecolor(color)
                        body.set_edgecolor(color)
                        body.set_alpha(0.2)
                else:
                    # A single session can't form a violin; mark the point.
                    ax.scatter([x_idx + off], vals, color=color, alpha=0.7,
                               s=40, edgecolors="none", zorder=2)

            # Mean ± SEM across animals (each animal = mean of its session rates).
            animal_means = np.array([np.mean(per_animal[s]) for s in subj_order], dtype=float)
            mean, err = mean_sem(animal_means)
            err = 0.0 if np.isnan(err) else err
        else:
            jitter = rng.uniform(-halfwidth, halfwidth, size=rates.size)
            xs = np.full_like(jitter, x_idx) + jitter
            if color_by_id:
                pt_colors = [subj_colors[s] for s in subj_ids]
                ax.scatter(xs, rates, c=pt_colors, alpha=0.7, s=40,
                           edgecolors="none", zorder=2)
            else:
                ax.scatter(xs, rates, color="tab:blue", alpha=0.55, s=40,
                           edgecolors="none", zorder=2)
            mean = float(rates.mean())
            err = float(rates.std(ddof=1)) if rates.size > 1 else 0.0

        ax.hlines(mean, x_idx - halfwidth, x_idx + halfwidth,
                  colors="black", linewidth=2.0, zorder=3)
        ax.errorbar(x_idx, mean, yerr=err, color="black", linewidth=1.5,
                    capsize=6, capthick=1.5, fmt="none", zorder=3)

    ax.set_xticks(x_idx_array)
    ax.set_xticklabels([str(p) for p in positions])
    ax.set_xlabel("Sequence Position")
    ax.set_ylabel("Completion Rate")
    ax.set_xlim(-0.5, len(positions) - 0.5)
    ax.set_ylim(0, 1.05)

    if color_by_id:
        present = [s for s in subjids if any(s in subj_per_position[p] for p in positions)]
        handles = [
            Line2D([0], [0], marker="o", linestyle="none", color=subj_colors[s],
                   label=f"Sub {str(s).zfill(3)}")
            for s in present
        ]
        if handles:
            ax.legend(handles=handles, title="Subject", loc="best")

    if show_title:
        ax.set_title(title if title else "Position Completion Rate by Session")

    # Extra pad so the (often long, bold) y-axis label doesn't clip in the
    # notebook display. Saved figures use bbox_inches='tight' so they're
    # already safe, but the live preview honors the figsize bbox.
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
                fig, "position_completion_rate",
                subjids=list(subjids), dates=save_dates,
                boxplot=True,
            )
            if verbose:
                print(f"[plot_position_completion_rate] Saved figure to {out_path}")
        except Exception as exc:
            if verbose:
                print(f"[plot_position_completion_rate] Failed to save figure: {exc}")

    plt.show()
    return fig, ax
