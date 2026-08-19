from __future__ import annotations

"""Drawing helpers shared by more than one plotter module.

- **A helper two plotter modules need lives here, never in one of them.** Zero
  plotter-to-plotter imports is the invariant that keeps `visualization/` splittable.
  See DECISIONS.md sections 3 and 13.
- None of these calls `save_figure`, so the section 9 provenance hazard does not arise
  here. Anything added that does must pass `skip_modules=(__name__,)`.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from hypnose_behavior.visualization.prep import (
    _count_to_marker_size,
    _darken,
    _nice_round,
    _ordered_groups,
    _resolve_color,
)
from hypnose_behavior.visualization.primitives import rolling_windows

__all__ = ["_add_size_legend", "_plot_violins_with_stats", "_plot_summary_rolling",
           "_clean_graph"]


def _add_size_legend(ax, counts, *, loc="lower right", title="Trials per point"):
    """Add a secondary legend showing reference dot sizes for nicely rounded
    counts (min / mid / max of the counts actually plotted on the axes).
    Preserves any pre-existing legend.
    """
    from matplotlib.lines import Line2D
    if not counts:
        return
    cmin = min(counts)
    cmax = max(counts)
    if cmin == cmax:
        ref_values = [_nice_round(cmin)]
    else:
        cmid = (cmin + cmax) / 2.0
        ref_values = []
        seen = set()
        for v in (cmin, cmid, cmax):
            rv = _nice_round(v)
            if rv > 0 and rv not in seen:
                seen.add(rv)
                ref_values.append(rv)
        ref_values.sort()
    if not ref_values:
        return
    primary_legend = ax.get_legend()
    handles = []
    labels = []
    for v in ref_values:
        markersize = float(np.sqrt(_count_to_marker_size(v)))
        handles.append(
            Line2D(
                [], [],
                marker="o",
                linestyle="",
                color="#cccccc",
                markeredgecolor="#444444",
                markersize=markersize,
            )
        )
        labels.append(f"n={v}")
    ax.legend(
        handles,
        labels,
        loc=loc,
        title=title,
        labelspacing=1.2,
        borderpad=0.7,
        handletextpad=1.0,
        frameon=True,
    )
    if primary_legend is not None:
        ax.add_artist(primary_legend)

def _plot_violins_with_stats(ax, groups, y_label, x_label, *, color_map=None):
    """One violin per category (coloured by color_map, matching the summary
    line plots), with a dark SD whisker and a white mean marker on top."""
    labels = list(groups.keys())
    data = [groups[label] for label in labels]
    whisker_lw = 1.6
    cap_half_width = 0.06
    for i, values in enumerate(data, start=1):
        if not values:
            continue
        label = labels[i - 1]
        color = _resolve_color(label, color_map) if color_map is not None else "#4c72b0"
        edge = _darken(color)
        mean_val = float(np.mean(values))
        if len(values) > 1:
            parts = ax.violinplot(
                [values], positions=[i], widths=0.8,
                showextrema=False, showmeans=False, showmedians=False,
            )
            for body in parts["bodies"]:
                body.set_facecolor(color)
                body.set_edgecolor(edge)
                body.set_alpha(0.8)
                body.set_linewidth(1.2)
            err_val = float(np.std(values, ddof=1))
        else:
            # Single sample cannot form a violin; mark the point.
            ax.scatter([i], values, s=30, color=color, edgecolors=edge, linewidths=1.0, zorder=4)
            err_val = 0.0
        # SD whisker with small caps (dark neutral, reads on any fill colour).
        if err_val > 0:
            ax.vlines(i, mean_val - err_val, mean_val + err_val, colors="#2b2b2b", linewidth=whisker_lw, zorder=4)
            ax.hlines([mean_val - err_val, mean_val + err_val], i - cap_half_width, i + cap_half_width, colors="#2b2b2b", linewidth=whisker_lw, zorder=4)
        # Mean: white dot with a dark edge, clearly visible on any violin colour.
        ax.scatter([i], [mean_val], s=44, facecolor="white", edgecolors="#2b2b2b", linewidths=1.5, zorder=5)
    ax.set_xlim(0.5, len(labels) + 0.5)
    ax.set_ylabel(y_label)
    ax.set_xlabel(x_label)
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=45, ha="right")

def _plot_summary_rolling(session_data, *, color_map, group_order, ylabel, title, window_size, step_size, ylim_bottom=None):
    """
    Plot rolling mean within each session per group; X axis is continuous global trial id.
    Lines do not bridge session boundaries (gaps at day lines).
    """
    window_n = max(1, int(window_size))
    step_n = max(1, int(step_size))

    if not session_data:
        return None

    # First-seen order, not a `set` -- see `_ordered_groups`.
    all_groups = {}
    for s in session_data:
        for group in s["groups"]:
            all_groups.setdefault(group)
    if not all_groups:
        return None
    groups = _ordered_groups(all_groups, group_order)

    fig, ax = plt.subplots(figsize=(12, 6))

    # Width reserved on the X axis for a session that has no plottable rolling-window
    # points (e.g. fewer than window_size trials of any group). Keeps multi-day plots
    # from different categories visually aligned in day count.
    empty_session_span = 20

    global_offset = 0
    boundary_lines = []
    legend_done = set()

    for s in session_data:
        # Compute rolling-window points and per-session local bounds in one pass.
        session_points = {}
        min_x = None
        max_x = None
        for group in groups:
            entries = s["groups"].get(group, [])
            if not entries:
                continue
            entries_sorted = sorted(entries, key=lambda x: x[0])
            idxs = np.array([e[0] for e in entries_sorted])
            vals = np.array([e[1] for e in entries_sorted], dtype=float)
            pts = []
            # `partial=False`: a session shorter than one window plots nothing.
            for start, end in rolling_windows(len(vals), window_n, step=step_n):
                rate = float(np.nanmean(vals[start:end]))
                local_x = int(idxs[end - 1])
                pts.append((local_x, rate))
                min_x = local_x if min_x is None else min(min_x, local_x)
                max_x = local_x if max_x is None else max(max_x, local_x)
            if pts:
                session_points[group] = pts

        if global_offset > 0:
            boundary_lines.append(global_offset - 0.5)

        if min_x is None:
            global_offset += empty_session_span
            continue

        shift = global_offset - min_x
        for group, pts in session_points.items():
            xs = [lx + shift for lx, _ in pts]
            ys = [y for _, y in pts]
            color = _resolve_color(group, color_map)
            label = group if group not in legend_done else None
            legend_done.add(group)
            ax.plot(xs, ys, color=color, linewidth=2, alpha=0.9, label=label)

        global_offset += max_x - min_x + 1

    for x in boundary_lines:
        ax.axvline(
            x=x,
            color="#1f77b4",
            linestyle=":",
            linewidth=1.2,
            alpha=0.7,
            zorder=1,
        )

    ax.set_xlabel("Trials (adjusted)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if global_offset > 0:
        ax.set_xlim(left=0, right=global_offset - 1)
    else:
        ax.set_xlim(left=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if ylim_bottom is not None:
        ax.set_ylim(bottom=ylim_bottom)
    if legend_done:
        ax.legend(loc="best")
    fig.tight_layout()
    return fig

def _clean_graph(ax, *, xlabel: Optional[str] = None, ylabel: Optional[str] = None):
    """
    Hide title, axis labels, tick labels, and legend while printing them for external editing.

    - Leaves ticks/spines in place so layout remains.
    - Prints axis labels and tick values to stdout for reference.
    """
    try:
        title = ax.get_title()
        x_lab = xlabel if xlabel is not None else ax.get_xlabel()
        y_lab = ylabel if ylabel is not None else ax.get_ylabel()
        x_ticks = [round(float(t), 3) for t in ax.get_xticks()]
        y_ticks = [round(float(t), 3) for t in ax.get_yticks()]
        if title:
            print(f"[_clean_graph] title: {title}")
        if x_lab:
            print(f"[_clean_graph] x label: {x_lab}")
        if y_lab:
            print(f"[_clean_graph] y label: {y_lab}")
        print(f"[_clean_graph] x ticks: {x_ticks}")
        print(f"[_clean_graph] y ticks: {y_ticks}")
    except Exception:
        pass

    # Clear visible annotations but keep ticks present
    ax.set_title("")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    leg = ax.get_legend()
    if leg is not None:
        try:
            leg.remove()
        except Exception:
            pass
