"""Utilities for predicted sequence analysis."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Union
import ast
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from hypnose_behavior.io.layout import (
    _filter_session_dirs,
    _iter_subject_dirs,
    session_selectors,
)
from hypnose_behavior.frames import position_entries_by_trial
from hypnose_behavior.io.loaders import _load_position_data
from hypnose_behavior.visualization.prep import (
	_collect_sessions,
	_count_to_marker_size,
	_load_sorted_session,
	_ordered_groups,
	_parse_json_value,
	_resolve_color,
	_summary_save_suffix,
)
from hypnose_behavior.visualization.panels import (
	_add_size_legend,
	_plot_summary_rolling,
	_plot_violins_with_stats,
)
from hypnose_behavior.visualization.primitives import mean_sem, rolling_windows
from hypnose_behavior.metric_analysis.metrics.accuracy import decision_accuracy
from hypnose_behavior.metric_analysis.metrics.false_alarm import (
    fa_latency_from_pokeout,
    fa_port_label,
)
from hypnose_behavior.metric_analysis.metrics.sampling import (
    trial_poke_span,
    trial_poke_total,
)
from hypnose_behavior.metric_analysis.metrics.timing import (
    reward_delivery_latency,
    valve_to_reward_latency,
)
from hypnose_behavior.metric_analysis.resolvers import by_group, over_windows
from hypnose_behavior.io.paths import get_derivatives_root
from hypnose_behavior.io.save import save_figure, nice_x_locator
from matplotlib.patches import Patch


def _sequence_label(seq):
	if not isinstance(seq, (list, tuple)):
		return None
	parts = []
	for item in seq:
		text = str(item)
		if text.startswith("Odor"):
			parts.append(text.replace("Odor", "", 1))
		else:
			parts.append(text)
	return "-".join(parts) if parts else None


def _sequence_len_ok(seq, *, min_len: int = 3) -> bool:
	return isinstance(seq, (list, tuple)) and len(seq) >= min_len


def _normalize_odor_name(value):
	if value is None:
		return None
	text = str(value).strip()
	if not text:
		return None
	if text.startswith("Odor"):
		text = text.replace("Odor", "", 1)
	text = text.strip()
	return f"Odor{text}" if text else None


def _order_sequence_labels(groups):
	preferred = ["F-G-A", "E-D-A", "E-D-B", "C-G-B"]
	labels = []
	for name in preferred:
		if name in groups:
			labels.append(name)
	for name in groups:
		if name not in labels:
			labels.append(name)
	return labels


def _order_odor_labels(groups):
	def _key(name):
		text = str(name)
		if text.startswith("Odor"):
			text = text.replace("Odor", "", 1)
		return text
	return sorted(groups.keys(), key=_key)


SEQUENCE_COLORS = {
	"F-G-A": "#2ca02c",
	"C-G-B": "#d62728",
	"E-D-A": "#bfbfbf",
	"E-D-B": "#7f7f7f",
}

ODOR_COLORS = {
	"OdorA": "#2ca02c",
	"OdorB": "#d62728",
	"OdorF": "#7fbf7f",
	"OdorC": "#f08080",
	"OdorG": "#1f77b4",
	"OdorG-F": "#1f77b4",
	"OdorG-C": "#7fb3d5",
	"OdorE": "#a0a0a0",
	"OdorD": "#606060",
}

SEQUENCE_ORDER = ["F-G-A", "E-D-A", "E-D-B", "C-G-B"]
ODOR_ORDER = ["OdorA", "OdorB", "OdorC", "OdorD", "OdorE", "OdorF", "OdorG-F", "OdorG-C", "OdorG"]


def _canonical_odor(value) -> str:
	"""Bare upper-case odor letter: 'OdorC' / 'odor c' / 'C' -> 'C'."""
	t = str(value).strip()
	return t[4:].strip().upper() if t.lower().startswith("odor") else t.upper()


def _build_odor_filter(odor):
	"""Set of canonical odor letters to keep, or None for no filtering. Accepts
	a single odor or an iterable; 'A' and 'OdorA' both match odor A."""
	if odor is None:
		return None
	items = [odor] if isinstance(odor, str) else list(odor)
	return {_canonical_odor(o) for o in items}


def _plot_summary_daily(session_data, *, color_map, group_order, ylabel, title, ylim_bottom=None):
	"""
	Plot per-session mean for each group, connected by lines.

	session_data : list of dicts, one per session, each with keys:
		- "n_trials": int (full session length)
		- "groups": {group_label: [(trial_idx, value), ...]}
	X axis is Session (1, 2, ...).
	"""
	n_sessions = len(session_data)
	if n_sessions == 0:
		return None

	# First-seen order, not a `set` -- see `_ordered_groups`. Each session's
	# `groups` is itself built in trial order, so this is well defined.
	all_groups = {}
	for s in session_data:
		for group in s["groups"]:
			all_groups.setdefault(group)
	if not all_groups:
		return None
	groups = _ordered_groups(all_groups, group_order)

	fig, ax = plt.subplots(figsize=(10, 5))

	all_counts = []
	for group in groups:
		xs, ys, counts = [], [], []
		for i, s in enumerate(session_data):
			vals = [v for _, v in s["groups"].get(group, [])]
			if vals:
				xs.append(i + 1)
				ys.append(float(np.mean(vals)))
				counts.append(len(vals))
		if not xs:
			continue
		all_counts.extend(counts)
		color = _resolve_color(group, color_map)
		ax.plot(xs, ys, color=color, linewidth=2, label=group)
		sizes = [_count_to_marker_size(c) for c in counts]
		ax.scatter(xs, ys, s=sizes, color=color, zorder=3)

	ax.set_xlabel("Session")
	ax.set_ylabel(ylabel)
	ax.set_title(title)
	# Numeric session axis: a few nicely-rounded integer ticks (5, 10, 15,
	# ...) rather than one per session; matches the save-time x-tick cap.
	ax.xaxis.set_major_locator(nice_x_locator())
	ax.spines["top"].set_visible(False)
	ax.spines["right"].set_visible(False)
	if ylim_bottom is not None:
		ax.set_ylim(bottom=ylim_bottom)
	ax.legend(loc="best")
	_add_size_legend(ax, all_counts)
	fig.tight_layout()
	return fig


def _plot_summary(
	session_data,
	*,
	color_map,
	group_order,
	ylabel,
	title,
	moving_avg,
	window_size,
	step_size,
	ylim_bottom=None,
):
	"""Dispatcher: daily means (moving_avg=False) or rolling-within-session (moving_avg=True)."""
	if moving_avg:
		return _plot_summary_rolling(
			session_data,
			color_map=color_map,
			group_order=group_order,
			ylabel=ylabel,
			title=title,
			window_size=window_size,
			step_size=step_size,
			ylim_bottom=ylim_bottom,
		)
	return _plot_summary_daily(
		session_data,
		color_map=color_map,
		group_order=group_order,
		ylabel=ylabel,
		title=title,
		ylim_bottom=ylim_bottom,
	)


def _is_multi_session(date_vals):
	"""Summary plots are only meaningful when more than one session is loaded."""
	return date_vals is not None and len(date_vals) > 1


def _apply_shared_ylim(figs_to_share, *, bottom_zero=False):
	"""Set a common ylim across multiple figures so plots from a single call line up.

	The shared bottom is min of each figure's current bottom (or 0 if ``bottom_zero``),
	and the shared top is max of each figure's current top.
	"""
	if not figs_to_share:
		return
	axes = [fig.axes[0] for fig in figs_to_share if fig.axes]
	if not axes:
		return
	bottoms = [a.get_ylim()[0] for a in axes]
	tops = [a.get_ylim()[1] for a in axes]
	common_bottom = 0.0 if bottom_zero else min(bottoms)
	common_top = max(tops)
	for a in axes:
		a.set_ylim(common_bottom, common_top)


def last_odor_poke_time(
	subjids: Optional[Iterable[int]] = None,
	dates: Optional[Union[Iterable[Union[int, str]], tuple]] = None,
	*,
	ses=None,
	index=None,
	date_range=None,
	ses_range=None,
	index_range=None,
	save: bool = False,
	moving_avg: bool = False,
	window_size: int = 10,
	step_size: int = 1,
):
	"""
	Boxplots of last-odor poke time by odor sequence, separated by response category.

	When multiple sessions are loaded, additional summary plots are produced for
	rewarded and unrewarded categories (one each). With ``moving_avg=False`` a per-
	session mean is plotted (X axis: Session); with ``moving_avg=True`` a rolling
	mean of size ``window_size`` (step ``step_size``) is plotted within each session
	with X axis = continuous global trial id (no line continuation across days).

	``ses`` / ``index`` / ``date_range`` / ``ses_range`` / ``index_range`` narrow the
	selection further; they intersect with ``dates`` and with each other, and ``index``
	is the subject's gap-free chronological rank (`io.layout.session_selectors`).
	"""
	select = session_selectors(
		ses=ses, index=index, date_range=date_range,
		ses_range=ses_range, index_range=index_range,
	)
	figs = []
	categories = ["rewarded", "unrewarded", "timeout_delayed"]
	summary_cats = ["rewarded", "unrewarded"]
	for subjid, date_vals, results_dirs in _collect_sessions(subjids, dates, **select):
		per_cat_pooled = {cat: {} for cat in categories}
		per_cat_sessions = {cat: [] for cat in categories}

		for results_dir in results_dirs:
			df = _load_sorted_session(results_dir)
			if df.empty:
				for cat in categories:
					per_cat_sessions[cat].append({"n_trials": 0, "groups": {}})
				continue
			n_trials = len(df)
			completed = df[df.get("is_aborted") == False]
			# `in_poke_times` is the provenance flag matching the poke facts this reads;
			# an unfiltered view carries positions the poke blob never had -- section 2.
			pokes_by_trial = position_entries_by_trial(
				_load_position_data(results_dir, df), "in_poke_times")
			for cat in categories:
				session_groups = {}
				cat_df = completed[completed.get("response_time_category") == cat]
				for _, row in cat_df.iterrows():
					last_odor = row.get("last_odor")
					if last_odor not in {"OdorA", "OdorB"}:
						continue
					seq = _parse_json_value(row.get("odor_sequence"))
					if not _sequence_len_ok(seq):
						continue
					seq_label = _sequence_label(seq)
					if not seq_label:
						continue
					entries = pokes_by_trial.get(row.get("global_trial_id")) or []
					if not entries:
						continue
					last_entry = entries[-1]
					poke_ms = last_entry.get("poke_time_ms")
					if poke_ms is None:
						continue
					poke_val = float(poke_ms)
					trial_idx = int(row["_trial_idx"])
					session_groups.setdefault(seq_label, []).append((trial_idx, poke_val))
					per_cat_pooled[cat].setdefault(seq_label, []).append(poke_val)
				per_cat_sessions[cat].append({"n_trials": n_trials, "groups": session_groups})

		for cat in categories:
			pooled = per_cat_pooled[cat]
			if pooled:
				fig, ax = plt.subplots(figsize=(10, 5))
				ordered = {k: pooled[k] for k in _order_sequence_labels(pooled)}
				_plot_violins_with_stats(ax, ordered, "Last Odor Poke\nTime (ms)", "Odor Sequence", color_map=SEQUENCE_COLORS)
				ax.set_title(f"Subjid {subjid} {cat} last-odor poke time")
				ax.set_ylim(bottom=0)
				fig.tight_layout()
				figs.append(fig)
				if save:
					save_figure(fig, f"last_odor_poke_time_{cat}", subjids=[subjid], dates=date_vals)

		if _is_multi_session(date_vals):
			mode_label = "rolling" if moving_avg else "daily mean"
			for cat in summary_cats:
				summary_fig = _plot_summary(
					per_cat_sessions[cat],
					color_map=SEQUENCE_COLORS,
					group_order=SEQUENCE_ORDER,
					ylabel="Last Odor Poke\nTime (ms)",
					title=f"Subjid {subjid} {cat} last-odor poke time ({mode_label})",
					moving_avg=moving_avg,
					window_size=window_size,
					step_size=step_size,
					ylim_bottom=0,
				)
				if summary_fig is not None:
					figs.append(summary_fig)
					if save:
						suffix = _summary_save_suffix(moving_avg, window_size, step_size)
						save_figure(
							summary_fig,
							f"last_odor_poke_time_{cat}_summary_{suffix}",
							subjids=[subjid],
							dates=date_vals,
						)

	return figs


def trial_poke_duration(
	subjids: Optional[Iterable[int]] = None,
	dates: Optional[Union[Iterable[Union[int, str]], tuple]] = None,
	*,
	ses=None,
	index=None,
	date_range=None,
	ses_range=None,
	index_range=None,
	save: bool = False,
	moving_avg: bool = False,
	window_size: int = 10,
	step_size: int = 1,
):
	"""
	Boxplots of trial poke duration by odor sequence, separated by response category.

	Cross-date summary plots (one for rewarded, one for unrewarded) are produced
	in addition to the per-session pooled boxplots. See ``last_odor_poke_time`` for
	the meaning of ``moving_avg``, ``window_size``, ``step_size``.

	``ses`` / ``index`` / ``date_range`` / ``ses_range`` / ``index_range`` narrow the
	selection further; they intersect with ``dates`` and with each other, and ``index``
	is the subject's gap-free chronological rank (`io.layout.session_selectors`).
	"""
	select = session_selectors(
		ses=ses, index=index, date_range=date_range,
		ses_range=ses_range, index_range=index_range,
	)
	figs = []
	categories = ["rewarded", "unrewarded", "timeout_delayed"]
	summary_cats = ["rewarded", "unrewarded"]
	for subjid, date_vals, results_dirs in _collect_sessions(subjids, dates, **select):
		per_cat_pooled = {cat: {} for cat in categories}
		per_cat_sessions = {cat: [] for cat in categories}

		for results_dir in results_dirs:
			df = _load_sorted_session(results_dir)
			if df.empty:
				for cat in categories:
					per_cat_sessions[cat].append({"n_trials": 0, "groups": {}})
				continue
			n_trials = len(df)
			completed = df[df.get("is_aborted") == False]
			spans = trial_poke_span(_load_position_data(results_dir, df))
			for cat in categories:
				session_groups = {}
				cat_df = completed[completed.get("response_time_category") == cat]
				for _, row in cat_df.iterrows():
					seq = _parse_json_value(row.get("odor_sequence"))
					if not _sequence_len_ok(seq):
						continue
					seq_label = _sequence_label(seq)
					if not seq_label:
						continue
					dur_val = spans.get(row.get("global_trial_id"))
					if dur_val is None or pd.isna(dur_val):
						continue
					dur_val = float(dur_val)
					trial_idx = int(row["_trial_idx"])
					session_groups.setdefault(seq_label, []).append((trial_idx, dur_val))
					per_cat_pooled[cat].setdefault(seq_label, []).append(dur_val)
				per_cat_sessions[cat].append({"n_trials": n_trials, "groups": session_groups})

		for cat in categories:
			pooled = per_cat_pooled[cat]
			if pooled:
				fig, ax = plt.subplots(figsize=(10, 5))
				ordered = {k: pooled[k] for k in _order_sequence_labels(pooled)}
				_plot_violins_with_stats(ax, ordered, "Trial Poke Duration (ms)", "Odor Sequence", color_map=SEQUENCE_COLORS)
				ax.set_title(f"Subjid {subjid} {cat} trial poke duration")
				fig.tight_layout()
				figs.append(fig)
				if save:
					save_figure(fig, f"trial_poke_duration_{cat}", subjids=[subjid], dates=date_vals)

		summary_figs = []
		summary_save_specs = []
		if _is_multi_session(date_vals):
			mode_label = "rolling" if moving_avg else "daily mean"
			for cat in summary_cats:
				summary_fig = _plot_summary(
					per_cat_sessions[cat],
					color_map=SEQUENCE_COLORS,
					group_order=SEQUENCE_ORDER,
					ylabel="Trial Poke Duration (ms)",
					title=f"Subjid {subjid} {cat} trial poke duration ({mode_label})",
					moving_avg=moving_avg,
					window_size=window_size,
					step_size=step_size,
				)
				if summary_fig is not None:
					figs.append(summary_fig)
					summary_figs.append(summary_fig)
					if save:
						suffix = _summary_save_suffix(moving_avg, window_size, step_size)
						summary_save_specs.append(
							(summary_fig, f"trial_poke_duration_{cat}_summary_{suffix}")
						)

		_apply_shared_ylim(summary_figs)
		for fig, name in summary_save_specs:
			save_figure(fig, name, subjids=[subjid], dates=date_vals)

	return figs


def first_odor_poke_duration(
	subjids: Optional[Iterable[int]] = None,
	dates: Optional[Union[Iterable[Union[int, str]], tuple]] = None,
	*,
	ses=None,
	index=None,
	date_range=None,
	ses_range=None,
	index_range=None,
	save: bool = False,
	moving_avg: bool = False,
	window_size: int = 10,
	step_size: int = 1,
	odor: Optional[Union[str, Iterable[str]]] = None,
):
	"""
	Boxplot of first-odor poke duration grouped by odor name (completed trials only).

	``odor`` restricts which first-odor(s) are plotted (e.g. "A"/"OdorA" or
	["A", "B"]); trials whose first odor is not in the filter are dropped. Useful
	to exclude the occasional buggy trial that starts on an unexpected odor.

	A cross-date summary plot is added when multiple sessions are loaded
	(daily mean or rolling within session — see ``last_odor_poke_time``).

	``ses`` / ``index`` / ``date_range`` / ``ses_range`` / ``index_range`` narrow the
	selection further; they intersect with ``dates`` and with each other, and ``index``
	is the subject's gap-free chronological rank (`io.layout.session_selectors`).
	"""
	select = session_selectors(
		ses=ses, index=index, date_range=date_range,
		ses_range=ses_range, index_range=index_range,
	)
	figs = []
	odor_filter = _build_odor_filter(odor)
	for subjid, date_vals, results_dirs in _collect_sessions(subjids, dates, **select):
		pooled = {}
		session_records = []
		for results_dir in results_dirs:
			df = _load_sorted_session(results_dir)
			if df.empty:
				session_records.append({"n_trials": 0, "groups": {}})
				continue
			n_trials = len(df)
			completed = df[df.get("is_aborted") == False]
			pokes_by_trial = position_entries_by_trial(
				_load_position_data(results_dir, df), "in_poke_times")
			session_groups = {}
			for _, row in completed.iterrows():
				first_entry = next(
					(e for e in pokes_by_trial.get(row.get("global_trial_id")) or []
					 if e.get("position") == 1), None)
				if not first_entry:
					continue
				odor_name = first_entry.get("odor_name")
				poke_ms = first_entry.get("poke_time_ms")
				if odor_name is None or poke_ms is None:
					continue
				if odor_filter is not None and _canonical_odor(odor_name) not in odor_filter:
					continue
				key = str(odor_name)
				poke_val = float(poke_ms)
				trial_idx = int(row["_trial_idx"])
				session_groups.setdefault(key, []).append((trial_idx, poke_val))
				pooled.setdefault(key, []).append(poke_val)
			session_records.append({"n_trials": n_trials, "groups": session_groups})

		if pooled:
			fig, ax = plt.subplots(figsize=(10, 5))
			ordered = {k: pooled[k] for k in _order_odor_labels(pooled)}
			_plot_violins_with_stats(ax, ordered, "Poke Duration", "Odor", color_map=ODOR_COLORS)
			ax.set_title(f"Subjid {subjid} first-odor poke duration")
			ax.set_ylim(bottom=0)
			fig.tight_layout()
			figs.append(fig)
			if save:
				save_figure(fig, "first_odor_poke_duration", subjids=[subjid], dates=date_vals)

		if _is_multi_session(date_vals):
			mode_label = "rolling" if moving_avg else "daily mean"
			summary_fig = _plot_summary(
				session_records,
				color_map=ODOR_COLORS,
				group_order=ODOR_ORDER,
				ylabel="Poke Duration (ms)",
				title=f"Subjid {subjid} first-odor poke duration ({mode_label})",
				moving_avg=moving_avg,
				window_size=window_size,
				step_size=step_size,
				ylim_bottom=0,
			)
			if summary_fig is not None:
				figs.append(summary_fig)
				if save:
					suffix = _summary_save_suffix(moving_avg, window_size, step_size)
					save_figure(
						summary_fig,
						f"first_odor_poke_duration_summary_{suffix}",
						subjids=[subjid],
						dates=date_vals,
					)

	return figs


def poke_time_all_pos(
	subjids: Optional[Iterable[int]] = None,
	dates: Optional[Union[Iterable[Union[int, str]], tuple]] = None,
	*,
	ses=None,
	index=None,
	date_range=None,
	ses_range=None,
	index_range=None,
	save: bool = False,
	moving_avg: bool = False,
	window_size: int = 10,
	step_size: int = 1,
	odor: Optional[Union[str, Iterable[str]]] = None,
):
	"""
	Boxplots of poke duration pooled across positions, grouped by odor name.

	``odor`` restricts which odor(s) are plotted (e.g. "A"/"OdorA" or ["A", "B"]);
	entries whose odor is not in the filter are dropped (OdorG matches "G").

	Two plots are produced per subject: one for completed trials (is_aborted=False)
	and one for aborted trials (is_aborted=True), so poke durations can be compared
	between the two. Cross-date summary plots are added for each (daily mean per
	session, or rolling within session if ``moving_avg=True``).

	``ses`` / ``index`` / ``date_range`` / ``ses_range`` / ``index_range`` narrow the
	selection further; they intersect with ``dates`` and with each other, and ``index``
	is the subject's gap-free chronological rank (`io.layout.session_selectors`).
	"""
	select = session_selectors(
		ses=ses, index=index, date_range=date_range,
		ses_range=ses_range, index_range=index_range,
	)
	figs = []
	odor_filter = _build_odor_filter(odor)
	categories = [("completed", False), ("aborted", True)]
	for subjid, date_vals, results_dirs in _collect_sessions(subjids, dates, **select):
		per_cat_pooled = {name: {} for name, _ in categories}
		per_cat_sessions = {name: [] for name, _ in categories}

		for results_dir in results_dirs:
			df = _load_sorted_session(results_dir)
			if df.empty:
				for name, _ in categories:
					per_cat_sessions[name].append({"n_trials": 0, "groups": {}})
				continue
			n_trials = len(df)
			pokes_by_trial = position_entries_by_trial(
				_load_position_data(results_dir, df), "in_poke_times")
			for name, aborted_flag in categories:
				cat_df = df[df.get("is_aborted") == aborted_flag]
				session_groups = {}
				for _, row in cat_df.iterrows():
					ordered = pokes_by_trial.get(row.get("global_trial_id"))
					if not ordered:
						continue
					trial_idx = int(row["_trial_idx"])
					prev_odor = None
					for entry in ordered:
						odor_name = entry.get("odor_name")
						poke_ms = entry.get("poke_time_ms")
						odor_str = str(odor_name) if odor_name is not None else None
						if odor_name is not None and poke_ms is not None:
							key = odor_str
							if odor_str == "OdorG":
								if prev_odor == "OdorC":
									key = "OdorG-C"
								elif prev_odor == "OdorF":
									key = "OdorG-F"
								else:
									key = None
							if key is not None and (odor_filter is None or _canonical_odor(odor_name) in odor_filter):
								poke_val = float(poke_ms)
								session_groups.setdefault(key, []).append((trial_idx, poke_val))
								per_cat_pooled[name].setdefault(key, []).append(poke_val)
						if odor_str is not None:
							prev_odor = odor_str
				per_cat_sessions[name].append({"n_trials": n_trials, "groups": session_groups})

		for name, _ in categories:
			pooled = per_cat_pooled[name]
			if pooled:
				fig, ax = plt.subplots(figsize=(10, 5))
				ordered = {k: pooled[k] for k in _order_odor_labels(pooled)}
				_plot_violins_with_stats(ax, ordered, "Poke Duration (ms)", "Odor", color_map=ODOR_COLORS)
				ax.set_title(f"Subjid {subjid} poke duration (all positions, {name})")
				ax.set_ylim(bottom=0)
				fig.tight_layout()
				figs.append(fig)
				if save:
					save_figure(fig, f"poke_time_all_pos_{name}", subjids=[subjid], dates=date_vals)

		if _is_multi_session(date_vals):
			mode_label = "rolling" if moving_avg else "daily mean"
			for name, _ in categories:
				summary_fig = _plot_summary(
					per_cat_sessions[name],
					color_map=ODOR_COLORS,
					group_order=ODOR_ORDER,
					ylabel="Poke Duration (ms)",
					title=f"Subjid {subjid} poke duration (all positions, {name}) ({mode_label})",
					moving_avg=moving_avg,
					window_size=window_size,
					step_size=step_size,
					ylim_bottom=0,
				)
				if summary_fig is not None:
					figs.append(summary_fig)
					if save:
						suffix = _summary_save_suffix(moving_avg, window_size, step_size)
						save_figure(
							summary_fig,
							f"poke_time_all_pos_{name}_summary_{suffix}",
							subjids=[subjid],
							dates=date_vals,
						)

	return figs


def response_time(
	subjids: Optional[Iterable[int]] = None,
	dates: Optional[Union[Iterable[Union[int, str]], tuple]] = None,
	*,
	ses=None,
	index=None,
	date_range=None,
	ses_range=None,
	index_range=None,
	save: bool = False,
	moving_avg: bool = False,
	window_size: int = 10,
	step_size: int = 1,
):
	"""
	Boxplots of response time by odor sequence (completed, non-timeout trials).

	Cross-date summary plots are produced for rewarded and unrewarded trials
	(daily mean per session, or rolling within session if ``moving_avg=True``).

	``ses`` / ``index`` / ``date_range`` / ``ses_range`` / ``index_range`` narrow the
	selection further; they intersect with ``dates`` and with each other, and ``index``
	is the subject's gap-free chronological rank (`io.layout.session_selectors`).
	"""
	select = session_selectors(
		ses=ses, index=index, date_range=date_range,
		ses_range=ses_range, index_range=index_range,
	)
	figs = []
	summary_cats = ["rewarded", "unrewarded"]
	for subjid, date_vals, results_dirs in _collect_sessions(subjids, dates, **select):
		# Pass 1: collect raw records per session (no filtering yet) so we can
		# compute a group-wise outlier threshold from all sessions combined.
		sessions_raw = []
		for results_dir in results_dirs:
			df = _load_sorted_session(results_dir)
			if df.empty:
				sessions_raw.append({"n_trials": 0, "records": []})
				continue
			n_trials = len(df)
			completed = df[df.get("is_aborted") == False]
			completed = completed[completed.get("response_time_category") != "timeout_delayed"]
			# Raw: the 10x-group-mean outlier rule below is a display filter and
			# deliberately stays out of the metric.
			latencies = reward_delivery_latency(df, _load_position_data(results_dir, df))
			records = []
			for _, row in completed.iterrows():
				seq = _parse_json_value(row.get("odor_sequence"))
				if not _sequence_len_ok(seq):
					continue
				seq_label = _sequence_label(seq)
				if not seq_label:
					continue
				rt_ms = latencies.get(row.get("global_trial_id"))
				if rt_ms is None or pd.isna(rt_ms):
					continue
				rt_ms = float(rt_ms)
				trial_idx = int(row["_trial_idx"])
				cat = row.get("response_time_category")
				records.append((trial_idx, seq_label, cat, rt_ms))
			sessions_raw.append({"n_trials": n_trials, "records": records})

		# Compute per-sequence outlier threshold = 10 × group mean across all sessions.
		group_values = {}
		for s in sessions_raw:
			for _, seq_label, _, v in s["records"]:
				group_values.setdefault(seq_label, []).append(v)
		thresholds = {g: 10.0 * float(np.mean(vs)) for g, vs in group_values.items() if vs}

		# Filter and log exclusions.
		excluded_log = []
		for sess_idx, s in enumerate(sessions_raw):
			date_tag = date_vals[sess_idx] if sess_idx < len(date_vals) else "?"
			filtered = []
			for rec in s["records"]:
				_, seq_label, _, v = rec
				if v > thresholds.get(seq_label, np.inf):
					excluded_log.append((seq_label, date_tag, v))
					continue
				filtered.append(rec)
			s["records"] = filtered

		if excluded_log:
			print(
				f"[response_time] Subjid {subjid}: excluded {len(excluded_log)} "
				f"sample(s) > 10x group mean:"
			)
			for seq_label, date_tag, v in excluded_log:
				print(f"  {seq_label}: 1 sample ({v:.1f} ms) from {date_tag}")

		# Pass 2: build pooled + per-cat session structures from filtered records.
		pooled = {}
		per_cat_sessions = {cat: [] for cat in summary_cats}
		for s in sessions_raw:
			per_cat_session_groups = {cat: {} for cat in summary_cats}
			for trial_idx, seq_label, cat, v in s["records"]:
				pooled.setdefault(seq_label, []).append(v)
				if cat in per_cat_session_groups:
					per_cat_session_groups[cat].setdefault(seq_label, []).append((trial_idx, v))
			for cat in summary_cats:
				per_cat_sessions[cat].append({"n_trials": s["n_trials"], "groups": per_cat_session_groups[cat]})

		if pooled:
			fig, ax = plt.subplots(figsize=(10, 5))
			ordered = {k: pooled[k] for k in _order_sequence_labels(pooled)}
			_plot_violins_with_stats(ax, ordered, "Response Time (ms)", "Odor Sequence", color_map=SEQUENCE_COLORS)
			ax.set_title(f"Subjid {subjid} response time")
			fig.tight_layout()
			figs.append(fig)
			if save:
				save_figure(fig, "response_time", subjids=[subjid], dates=date_vals)

		summary_figs = []
		summary_save_specs = []
		if _is_multi_session(date_vals):
			mode_label = "rolling" if moving_avg else "daily mean"
			for cat in summary_cats:
				summary_fig = _plot_summary(
					per_cat_sessions[cat],
					color_map=SEQUENCE_COLORS,
					group_order=SEQUENCE_ORDER,
					ylabel="Response Time (ms)",
					title=f"Subjid {subjid} {cat} response time ({mode_label})",
					moving_avg=moving_avg,
					window_size=window_size,
					step_size=step_size,
				)
				if summary_fig is not None:
					figs.append(summary_fig)
					summary_figs.append(summary_fig)
					if save:
						suffix = _summary_save_suffix(moving_avg, window_size, step_size)
						summary_save_specs.append(
							(summary_fig, f"response_time_{cat}_summary_{suffix}")
						)

		_apply_shared_ylim(summary_figs)
		for fig, name in summary_save_specs:
			save_figure(fig, name, subjids=[subjid], dates=date_vals)

	return figs


def fa_analysis(
	subjids: Optional[Iterable[int]] = None,
	dates: Optional[Union[Iterable[Union[int, str]], tuple]] = None,
	fa_types: Optional[Iterable[str]] = None,
	*,
	ses=None,
	index=None,
	date_range=None,
	ses_range=None,
	index_range=None,
	save: bool = False,
	moving_avg: bool = False,
	window_size: int = 10,
	step_size: int = 1,
):
	"""
	FA trial analysis for aborted trials by last odor and FA port.

	Three cross-date summary plots are added (daily mean per session, or rolling
	within session if ``moving_avg=True``):
	  - FA poke time by odor.
	  - FA response time, FA→A, by odor.
	  - FA response time, FA→B, by odor.

	``ses`` / ``index`` / ``date_range`` / ``ses_range`` / ``index_range`` narrow the
	selection further; they intersect with ``dates`` and with each other, and ``index``
	is the subject's gap-free chronological rank (`io.layout.session_selectors`).
	"""
	select = session_selectors(
		ses=ses, index=index, date_range=date_range,
		ses_range=ses_range, index_range=index_range,
	)
	figs = []
	odor_whitelist = {"OdorC", "OdorD", "OdorE", "OdorF", "OdorG"}
	fa_labels = {"FA_time_in", "FA_time_out"}
	if fa_types is not None:
		fa_labels = set(fa_types)
	for subjid, date_vals, results_dirs in _collect_sessions(subjids, dates, **select):
		poke_groups = {}
		poke_session_records = []
		# Raw FA response-time records per session: list of (trial_idx, odor, port, value).
		# These are filtered for outliers below before being expanded into resp_groups
		# and resp_session_records.
		resp_raw_per_session = []
		session_n_trials = []

		for results_dir in results_dirs:
			df = _load_sorted_session(results_dir)
			if df.empty:
				poke_session_records.append({"n_trials": 0, "groups": {}})
				resp_raw_per_session.append([])
				session_n_trials.append(0)
				continue
			n_trials = len(df)
			session_n_trials.append(n_trials)
			aborted = df[df.get("is_aborted") == True]
			fa_df = aborted[aborted.get("fa_label").isin(fa_labels)]
			# Raw latencies; the 10x-group-mean rule below is a display filter.
			fa_latencies = fa_latency_from_pokeout(df)
			port_labels = fa_port_label(fa_df)
			pokes_by_trial = position_entries_by_trial(
				_load_position_data(results_dir, df), "in_poke_times")

			session_poke = {}
			session_resp_raw = []
			for _, row in fa_df.iterrows():
				last_odor = _normalize_odor_name(row.get("last_odor"))
				if last_odor not in odor_whitelist:
					continue

				ordered_entries = pokes_by_trial.get(row.get("global_trial_id"))
				if not ordered_entries:
					continue
				odor_entry = None
				preceding_odor = None
				prev = None
				for entry in ordered_entries:
					entry_odor = _normalize_odor_name(entry.get("odor_name"))
					if entry_odor == last_odor:
						odor_entry = entry
						preceding_odor = prev
						break
					prev = entry_odor
				if not odor_entry:
					continue

				last_odor_poke_label = last_odor
				if last_odor == "OdorG":
					if preceding_odor == "OdorC":
						last_odor_poke_label = "OdorG-C"
					elif preceding_odor == "OdorF":
						last_odor_poke_label = "OdorG-F"
					else:
						last_odor_poke_label = None

				poke_ms = odor_entry.get("poke_time_ms")
				if poke_ms is not None and last_odor_poke_label is not None:
					poke_val = float(poke_ms)
					trial_idx = int(row["_trial_idx"])
					poke_groups.setdefault(last_odor_poke_label, []).append(poke_val)
					session_poke.setdefault(last_odor_poke_label, []).append((trial_idx, poke_val))

				rt_ms = fa_latencies.get(row.get("global_trial_id"))
				if rt_ms is None or pd.isna(rt_ms):
					continue

				port_label = port_labels.get(row.name)
				if port_label is None or pd.isna(port_label):
					continue

				rt_val = float(rt_ms)
				trial_idx = int(row["_trial_idx"])
				session_resp_raw.append((trial_idx, last_odor, port_label, rt_val))

			poke_session_records.append({"n_trials": n_trials, "groups": session_poke})
			resp_raw_per_session.append(session_resp_raw)

		# Compute per-(odor, port) FA-response-time mean across all sessions and filter
		# any sample > 10 × group mean. Report exclusions.
		group_values = {}
		for sess_records in resp_raw_per_session:
			for _, odor, port_label, v in sess_records:
				group_values.setdefault((odor, port_label), []).append(v)
		thresholds = {
			key: 10.0 * float(np.mean(vs)) for key, vs in group_values.items() if vs
		}

		excluded_log = []
		for sess_idx, sess_records in enumerate(resp_raw_per_session):
			date_tag = date_vals[sess_idx] if sess_idx < len(date_vals) else "?"
			filtered = []
			for rec in sess_records:
				_, odor, port_label, v = rec
				if v > thresholds.get((odor, port_label), np.inf):
					excluded_log.append((odor, port_label, date_tag, v))
					continue
				filtered.append(rec)
			resp_raw_per_session[sess_idx] = filtered

		if excluded_log:
			print(
				f"[fa_analysis] Subjid {subjid}: excluded {len(excluded_log)} FA "
				f"response-time sample(s) > 10x group mean:"
			)
			for odor, port_label, date_tag, v in excluded_log:
				print(f"  {odor}→{port_label}: 1 sample ({v:.1f} ms) from {date_tag}")

		# Rebuild resp_groups (pooled per-odor, per-port lists) and resp_session_records
		# (per-session per-port per-odor (trial_idx, value) lists) from filtered records.
		resp_groups = {}
		resp_session_records = {"A": [], "B": []}
		for sess_idx, sess_records in enumerate(resp_raw_per_session):
			n_trials = session_n_trials[sess_idx]
			session_resp = {"A": {}, "B": {}}
			for trial_idx, odor, port_label, v in sess_records:
				resp_groups.setdefault(odor, {"A": [], "B": []})[port_label].append(v)
				session_resp[port_label].setdefault(odor, []).append((trial_idx, v))
			resp_session_records["A"].append({"n_trials": n_trials, "groups": session_resp["A"]})
			resp_session_records["B"].append({"n_trials": n_trials, "groups": session_resp["B"]})

		if poke_groups:
			fig, ax = plt.subplots(figsize=(10, 5))
			ordered = {k: poke_groups[k] for k in _order_odor_labels(poke_groups)}
			_plot_violins_with_stats(ax, ordered, "Poke Time (ms)", "Odor", color_map=ODOR_COLORS)
			ax.set_title(f"Subjid {subjid} FA poke time by odor")
			ax.set_ylim(bottom=0)
			fig.tight_layout()
			figs.append(fig)
			if save:
				save_figure(fig, "fa_poke_time", subjids=[subjid], dates=date_vals)

		if resp_groups:
			ordered_odors = _order_odor_labels(resp_groups)
			labels = []
			has_any = False
			fig, ax = plt.subplots(figsize=(12, 5))
			mean_half_width = 0.14
			cap_half_width = 0.08
			mean_lw = 2.2
			sem_lw = 1.4
			point_offset = 0.18
			jitter_half_width = 0.12
			rng = np.random.default_rng(0)

			for i, odor in enumerate(ordered_odors, start=1):
				odor_groups = resp_groups[odor]
				count_a = len(odor_groups.get("A", []))
				count_b = len(odor_groups.get("B", []))
				labels.append(f"{odor}\nRatio A/B: {count_a}/{count_b}")

				for port_label, color, offset in (
					("A", "red", -point_offset),
					("B", "green", point_offset),
				):
					values = odor_groups.get(port_label, [])
					if not values:
						continue
					has_any = True
					x_pos = i + offset
					xs = x_pos + rng.uniform(-jitter_half_width, jitter_half_width, size=len(values))
					ax.scatter(xs, values, s=18, color=color, alpha=0.4, zorder=2)
					mean_val, sem_val = mean_sem(values)
					sem_val = 0.0 if np.isnan(sem_val) else sem_val
					ax.hlines(
						mean_val,
						x_pos - mean_half_width,
						x_pos + mean_half_width,
						colors="black",
						linewidth=mean_lw,
						zorder=3,
					)
					ax.vlines(
						x_pos,
						mean_val - sem_val,
						mean_val + sem_val,
						colors="black",
						linewidth=sem_lw,
						zorder=3,
					)
					if sem_val > 0:
						ax.hlines(
							mean_val - sem_val,
							x_pos - cap_half_width,
							x_pos + cap_half_width,
							colors="black",
							linewidth=sem_lw,
							zorder=3,
						)
						ax.hlines(
							mean_val + sem_val,
							x_pos - cap_half_width,
							x_pos + cap_half_width,
							colors="black",
							linewidth=sem_lw,
							zorder=3,
						)

			if has_any:
				ax.set_xticks(range(1, len(labels) + 1))
				ax.set_xticklabels(labels, rotation=45, ha="right")
				ax.set_ylabel("FA Response Time (ms)")
				ax.set_xlabel("Odor")
				ax.set_title(f"Subjid {subjid} FA response time by port")
				ax.legend(
					handles=[
						Patch(facecolor="red", edgecolor="black", label="FA to port A"),
						Patch(facecolor="green", edgecolor="black", label="FA to port B"),
					],
					loc="upper right",
				)
				fig.tight_layout()
				figs.append(fig)
				if save:
					save_figure(fig, "fa_response_time", subjids=[subjid], dates=date_vals)
			else:
				plt.close(fig)

		resp_summary_figs = []
		resp_summary_save_specs = []
		if _is_multi_session(date_vals):
			mode_label = "rolling" if moving_avg else "daily mean"
			summary_fig = _plot_summary(
				poke_session_records,
				color_map=ODOR_COLORS,
				group_order=ODOR_ORDER,
				ylabel="FA Poke Time (ms)",
				title=f"Subjid {subjid} FA poke time by odor ({mode_label})",
				moving_avg=moving_avg,
				window_size=window_size,
				step_size=step_size,
				ylim_bottom=0,
			)
			if summary_fig is not None:
				figs.append(summary_fig)
				if save:
					suffix = _summary_save_suffix(moving_avg, window_size, step_size)
					save_figure(
						summary_fig,
						f"fa_poke_time_summary_{suffix}",
						subjids=[subjid],
						dates=date_vals,
					)

			for port_label in ("A", "B"):
				resp_summary_fig = _plot_summary(
					resp_session_records[port_label],
					color_map=ODOR_COLORS,
					group_order=ODOR_ORDER,
					ylabel="FA Response Time (ms)",
					title=f"Subjid {subjid} FA response time (FA→{port_label}) by odor ({mode_label})",
					moving_avg=moving_avg,
					window_size=window_size,
					step_size=step_size,
				)
				if resp_summary_fig is not None:
					figs.append(resp_summary_fig)
					resp_summary_figs.append(resp_summary_fig)
					if save:
						suffix = _summary_save_suffix(moving_avg, window_size, step_size)
						resp_summary_save_specs.append(
							(resp_summary_fig, f"fa_response_time_port{port_label}_summary_{suffix}")
						)

		_apply_shared_ylim(resp_summary_figs)
		for fig, name in resp_summary_save_specs:
			save_figure(fig, name, subjids=[subjid], dates=date_vals)

	return figs


def valve_to_reward(
	subjids: Optional[Iterable[int]] = None,
	dates: Optional[Union[Iterable[Union[int, str]], tuple]] = None,
	*,
	ses=None,
	index=None,
	date_range=None,
	ses_range=None,
	index_range=None,
	save: bool = False,
	moving_avg: bool = False,
	window_size: int = 10,
	step_size: int = 1,
):
	"""
	Boxplots of time from the last position's valve_start to first_supply_time,
	by odor sequence, split into rewarded and unrewarded trials.

	Cross-date summary plots (one for rewarded, one for unrewarded) are produced
	in addition to the per-session pooled boxplots. See ``last_odor_poke_time``
	for the meaning of ``moving_avg``, ``window_size``, ``step_size``.

	``ses`` / ``index`` / ``date_range`` / ``ses_range`` / ``index_range`` narrow the
	selection further; they intersect with ``dates`` and with each other, and ``index``
	is the subject's gap-free chronological rank (`io.layout.session_selectors`).
	"""
	select = session_selectors(
		ses=ses, index=index, date_range=date_range,
		ses_range=ses_range, index_range=index_range,
	)
	figs = []
	categories = ["rewarded", "unrewarded"]
	summary_cats = ["rewarded", "unrewarded"]
	for subjid, date_vals, results_dirs in _collect_sessions(subjids, dates, **select):
		per_cat_pooled = {cat: {} for cat in categories}
		per_cat_sessions = {cat: [] for cat in categories}

		for results_dir in results_dirs:
			df = _load_sorted_session(results_dir)
			if df.empty:
				for cat in categories:
					per_cat_sessions[cat].append({"n_trials": 0, "groups": {}})
				continue
			n_trials = len(df)
			completed = df[df.get("is_aborted") == False]
			valve_latencies = valve_to_reward_latency(df, _load_position_data(results_dir, df))
			for cat in categories:
				session_groups = {}
				cat_df = completed[completed.get("response_time_category") == cat]
				for _, row in cat_df.iterrows():
					seq = _parse_json_value(row.get("odor_sequence"))
					if not _sequence_len_ok(seq):
						continue
					seq_label = _sequence_label(seq)
					if not seq_label:
						continue
					dur_ms = valve_latencies.get(row.get("global_trial_id"))
					if dur_ms is None or pd.isna(dur_ms):
						continue
					dur_ms = float(dur_ms)
					trial_idx = int(row["_trial_idx"])
					session_groups.setdefault(seq_label, []).append((trial_idx, dur_ms))
					per_cat_pooled[cat].setdefault(seq_label, []).append(dur_ms)
				per_cat_sessions[cat].append({"n_trials": n_trials, "groups": session_groups})

		for cat in categories:
			pooled = per_cat_pooled[cat]
			if pooled:
				fig, ax = plt.subplots(figsize=(10, 5))
				ordered = {k: pooled[k] for k in _order_sequence_labels(pooled)}
				_plot_violins_with_stats(ax, ordered, "Valve→Reward Time (ms)", "Odor Sequence", color_map=SEQUENCE_COLORS)
				ax.set_title(f"Subjid {subjid} {cat} valve-to-reward time")
				ax.set_ylim(bottom=0)
				fig.tight_layout()
				figs.append(fig)
				if save:
					save_figure(fig, f"valve_to_reward_{cat}", subjids=[subjid], dates=date_vals)

		summary_figs = []
		summary_save_specs = []
		if _is_multi_session(date_vals):
			mode_label = "rolling" if moving_avg else "daily mean"
			for cat in summary_cats:
				summary_fig = _plot_summary(
					per_cat_sessions[cat],
					color_map=SEQUENCE_COLORS,
					group_order=SEQUENCE_ORDER,
					ylabel="Valve→Reward Time (ms)",
					title=f"Subjid {subjid} {cat} valve-to-reward time ({mode_label})",
					moving_avg=moving_avg,
					window_size=window_size,
					step_size=step_size,
					ylim_bottom=0,
				)
				if summary_fig is not None:
					figs.append(summary_fig)
					summary_figs.append(summary_fig)
					if save:
						suffix = _summary_save_suffix(moving_avg, window_size, step_size)
						summary_save_specs.append(
							(summary_fig, f"valve_to_reward_{cat}_summary_{suffix}")
						)

		_apply_shared_ylim(summary_figs)
		for fig, name in summary_save_specs:
			save_figure(fig, name, subjids=[subjid], dates=date_vals)

	return figs


def _plot_performance_daily(sessions_frames, subjid):
	n_sessions = len(sessions_frames)
	if n_sessions == 0:
		return None

	# Per sequence: list of (session_idx, pct, count). Plus pooled overall.
	# The percentage is scaled from the metric's own numerator and denominator
	# rather than from its rate, so the arithmetic is the `100.0 * r / t` this
	# replaces and not a second rounding of it.
	sequence_data = {}
	overall_data = []
	for i, sub in enumerate(sessions_frames):
		if sub is None or sub.empty:
			continue
		stats = by_group(decision_accuracy, sub, "sequence", values_only=False)
		# First-seen order, not `by_group`'s sorted index: it decides the insertion
		# order of `sequence_data`, and `_ordered_groups` draws any label outside
		# SEQUENCE_ORDER in exactly that order. See DECISIONS.md section 11.
		for seq_label in sub["sequence"].drop_duplicates():
			r, t, _ = stats[seq_label]
			if t > 0:
				sequence_data.setdefault(seq_label, []).append((i + 1, 100.0 * r / t, t))
		n_rewarded_all, n_total_all, _ = decision_accuracy(sub)
		if n_total_all > 0:
			overall_data.append((i + 1, 100.0 * n_rewarded_all / n_total_all, n_total_all))

	if not sequence_data and not overall_data:
		return None

	fig, ax = plt.subplots(figsize=(10, 5))

	all_counts = []
	ordered_seqs = _ordered_groups(sequence_data.keys(), SEQUENCE_ORDER)
	for seq_label in ordered_seqs:
		pts = sequence_data[seq_label]
		xs = [p[0] for p in pts]
		ys = [p[1] for p in pts]
		counts = [p[2] for p in pts]
		all_counts.extend(counts)
		color = _resolve_color(seq_label, SEQUENCE_COLORS)
		ax.plot(xs, ys, color=color, linewidth=2, label=seq_label)
		ax.scatter(xs, ys, s=[_count_to_marker_size(c) for c in counts], color=color, zorder=3)

	if overall_data:
		xs = [p[0] for p in overall_data]
		ys = [p[1] for p in overall_data]
		counts = [p[2] for p in overall_data]
		all_counts.extend(counts)
		ax.plot(xs, ys, color="black", linewidth=3, label="Overall")
		ax.scatter(xs, ys, s=[_count_to_marker_size(c) for c in counts], color="black", zorder=4)

	ax.set_xlabel("Session")
	ax.set_ylabel("Performance %")
	ax.set_title(f"Subjid {subjid} performance")
	# Numeric session axis: a few nicely-rounded integer ticks (5, 10, 15,
	# ...) rather than one per session; matches the save-time x-tick cap.
	ax.xaxis.set_major_locator(nice_x_locator())
	ax.set_ylim(0, 100)
	ax.spines["top"].set_visible(False)
	ax.spines["right"].set_visible(False)
	ax.legend(loc="best")
	_add_size_legend(ax, all_counts)
	fig.tight_layout()
	return fig


def _plot_performance_rolling(sessions_frames, window_size, step_size, subjid):
	window_n = max(1, int(window_size))
	step_n = max(1, int(step_size))
	if not sessions_frames:
		return None

	fig, ax = plt.subplots(figsize=(12, 6))

	empty_session_span = 20
	global_offset = 0
	boundary_lines = []
	legend_done = set()

	# First-seen order, not a `set` -- see `_ordered_groups`. This matches how
	# `_plot_performance_daily` below reaches the same set of labels.
	all_seq_labels = {}
	for sub in sessions_frames:
		if sub is not None and not sub.empty:
			for label in sub["sequence"]:
				all_seq_labels.setdefault(label)
	ordered_seqs = _ordered_groups(all_seq_labels, SEQUENCE_ORDER)

	def _rolling_pts(frame):
		"""(x, percentage) per trailing window -- `over_windows` on the metric core.

		x is the `_trial_idx` of the window's last trial. `over_windows` defaults to
		`min_periods=window`, i.e. no partial windows, and anchors the first window
		at position `window - 1`, which is where the loop this replaces emitted its
		first point for any `step`.
		"""
		frame = frame.sort_values("_trial_idx", kind="stable")
		windows = over_windows(decision_accuracy, frame, window_n, step=step_n)
		idxs = frame["_trial_idx"].to_numpy()
		return [(int(idxs[int(end)]), float(v) * 100.0)
			for end, v in zip(windows["end_index"], windows["value"])]

	for sub in sessions_frames:
		if sub is None or sub.empty:
			if global_offset > 0:
				boundary_lines.append(global_offset - 0.5)
			global_offset += empty_session_span
			continue

		seq_pts = {seq: _rolling_pts(part)
			for seq, part in sub.groupby("sequence", sort=False)}
		overall_pts = _rolling_pts(sub)

		min_x = None
		max_x = None
		for pts in list(seq_pts.values()) + [overall_pts]:
			for lx, _ in pts:
				min_x = lx if min_x is None else min(min_x, lx)
				max_x = lx if max_x is None else max(max_x, lx)

		if global_offset > 0:
			boundary_lines.append(global_offset - 0.5)

		if min_x is None:
			global_offset += empty_session_span
			continue

		shift = global_offset - min_x

		for seq_label in ordered_seqs:
			pts = seq_pts.get(seq_label, [])
			if not pts:
				continue
			xs = [lx + shift for lx, _ in pts]
			ys = [y for _, y in pts]
			color = _resolve_color(seq_label, SEQUENCE_COLORS)
			label = seq_label if seq_label not in legend_done else None
			legend_done.add(seq_label)
			ax.plot(xs, ys, color=color, linewidth=2, alpha=0.9, label=label)

		if overall_pts:
			xs = [lx + shift for lx, _ in overall_pts]
			ys = [y for _, y in overall_pts]
			label = "Overall" if "Overall" not in legend_done else None
			legend_done.add("Overall")
			ax.plot(xs, ys, color="black", linewidth=3, alpha=0.95, label=label)

		global_offset += max_x - min_x + 1

	for x in boundary_lines:
		ax.axvline(x=x, color="#1f77b4", linestyle=":", linewidth=1.2, alpha=0.7, zorder=1)

	ax.set_xlabel("Trials (adjusted)")
	ax.set_ylabel("Performance %")
	ax.set_title(f"Subjid {subjid} performance (rolling)")
	if global_offset > 0:
		ax.set_xlim(left=0, right=global_offset - 1)
	else:
		ax.set_xlim(left=0)
	ax.set_ylim(0, 100)
	ax.spines["top"].set_visible(False)
	ax.spines["right"].set_visible(False)
	if legend_done:
		ax.legend(loc="best")
	fig.tight_layout()
	return fig


def performance(
	subjids: Optional[Iterable[int]] = None,
	dates: Optional[Union[Iterable[Union[int, str]], tuple]] = None,
	*,
	ses=None,
	index=None,
	date_range=None,
	ses_range=None,
	index_range=None,
	save: bool = False,
	moving_avg: bool = False,
	window_size: int = 10,
	step_size: int = 1,
):
	"""
	Performance (% rewarded among completed trials) by odor sequence.

	Completed = ``is_aborted == False`` and ``response_time_category`` in
	{"rewarded", "unrewarded"} (timeout-delayed trials are excluded). For each
	sequence (F-G-A, C-G-B, E-D-A, E-D-B) the rate is rewarded / (rewarded +
	unrewarded). A thicker black "Overall" line is added showing the pooled
	rate across all sequences.

	With ``moving_avg=False`` (default) the X axis is Session and one point is
	plotted per session per sequence. With ``moving_avg=True`` the X axis is
	continuous global trial id and the rate is computed in a rolling window
	*per sequence* (e.g. last 10 occurrences of that sequence); the Overall
	line uses a rolling window over all completed trials in order.

	Unlike most other functions here, this works for a single session.

	``ses`` / ``index`` / ``date_range`` / ``ses_range`` / ``index_range`` narrow the
	selection further; they intersect with ``dates`` and with each other, and ``index``
	is the subject's gap-free chronological rank (`io.layout.session_selectors`).
	"""
	select = session_selectors(
		ses=ses, index=index, date_range=date_range,
		ses_range=ses_range, index_range=index_range,
	)
	figs = []
	for subjid, date_vals, results_dirs in _collect_sessions(subjids, dates, **select):
		# One frame per session, already reduced to the rows `decision_accuracy`
		# scores: completed, sequence-labelled, timeouts dropped. The denominator then
		# matches the canonical metric exactly, so the two helpers below are `by_group` /
		# `over_windows` calls rather than their own rewarded/(rewarded+unrewarded)
		# arithmetic.
		sessions_frames = []
		for results_dir in results_dirs:
			df = _load_sorted_session(results_dir)
			if df.empty:
				sessions_frames.append(pd.DataFrame())
				continue
			completed = df[df.get("is_aborted") == False]
			labels = completed["odor_sequence"].map(
				lambda v: _sequence_label(_parse_json_value(v))
				if _sequence_len_ok(_parse_json_value(v)) else None
			)
			sub = completed.assign(sequence=labels)
			sub = sub[sub["sequence"].notna()
				& sub["response_time_category"].isin(["rewarded", "unrewarded"])]
			sessions_frames.append(sub)

		if not any(len(f) for f in sessions_frames):
			continue

		if moving_avg:
			fig = _plot_performance_rolling(sessions_frames, window_size, step_size, subjid)
		else:
			fig = _plot_performance_daily(sessions_frames, subjid)

		if fig is not None:
			figs.append(fig)
			if save:
				suffix = _summary_save_suffix(moving_avg, window_size, step_size)
				save_figure(fig, f"performance_{suffix}", subjids=[subjid], dates=date_vals)

	return figs


def cummulative_poke_time(
	subjids: Optional[Iterable[int]] = None,
	dates: Optional[Union[Iterable[Union[int, str]], tuple]] = None,
	*,
	ses=None,
	index=None,
	date_range=None,
	ses_range=None,
	index_range=None,
	save: bool = False,
	moving_avg: bool = False,
	window_size: int = 10,
	step_size: int = 1,
):
	"""
	Boxplots of cumulative per-trial poke duration (sum of poke_time_ms across
	all positions), by odor sequence, split into rewarded and unrewarded.

	Cross-date summary plots (one for rewarded, one for unrewarded) are added
	when multiple sessions are loaded. See ``last_odor_poke_time`` for
	``moving_avg``, ``window_size``, ``step_size`` semantics.

	``ses`` / ``index`` / ``date_range`` / ``ses_range`` / ``index_range`` narrow the
	selection further; they intersect with ``dates`` and with each other, and ``index``
	is the subject's gap-free chronological rank (`io.layout.session_selectors`).
	"""
	select = session_selectors(
		ses=ses, index=index, date_range=date_range,
		ses_range=ses_range, index_range=index_range,
	)
	figs = []
	categories = ["rewarded", "unrewarded"]
	summary_cats = ["rewarded", "unrewarded"]
	for subjid, date_vals, results_dirs in _collect_sessions(subjids, dates, **select):
		per_cat_pooled = {cat: {} for cat in categories}
		per_cat_sessions = {cat: [] for cat in categories}

		for results_dir in results_dirs:
			df = _load_sorted_session(results_dir)
			if df.empty:
				for cat in categories:
					per_cat_sessions[cat].append({"n_trials": 0, "groups": {}})
				continue
			n_trials = len(df)
			completed = df[df.get("is_aborted") == False]
			poke_totals = trial_poke_total(_load_position_data(results_dir, df))
			for cat in categories:
				session_groups = {}
				cat_df = completed[completed.get("response_time_category") == cat]
				for _, row in cat_df.iterrows():
					seq = _parse_json_value(row.get("odor_sequence"))
					if not _sequence_len_ok(seq):
						continue
					seq_label = _sequence_label(seq)
					if not seq_label:
						continue
					total_ms = poke_totals.get(row.get("global_trial_id"))
					if total_ms is None or pd.isna(total_ms):
						continue
					total_ms = float(total_ms)
					trial_idx = int(row["_trial_idx"])
					session_groups.setdefault(seq_label, []).append((trial_idx, total_ms))
					per_cat_pooled[cat].setdefault(seq_label, []).append(total_ms)
				per_cat_sessions[cat].append({"n_trials": n_trials, "groups": session_groups})

		for cat in categories:
			pooled = per_cat_pooled[cat]
			if pooled:
				fig, ax = plt.subplots(figsize=(10, 5))
				ordered = {k: pooled[k] for k in _order_sequence_labels(pooled)}
				_plot_violins_with_stats(ax, ordered, "Cummulative Poke\nDuration (ms)", "Odor Sequence", color_map=SEQUENCE_COLORS)
				ax.set_title(f"Subjid {subjid} {cat} cummulative poke duration")
				ax.set_ylim(bottom=0)
				fig.tight_layout()
				figs.append(fig)
				if save:
					save_figure(fig, f"cummulative_poke_time_{cat}", subjids=[subjid], dates=date_vals)

		summary_figs = []
		summary_save_specs = []
		if _is_multi_session(date_vals):
			mode_label = "rolling" if moving_avg else "daily mean"
			for cat in summary_cats:
				summary_fig = _plot_summary(
					per_cat_sessions[cat],
					color_map=SEQUENCE_COLORS,
					group_order=SEQUENCE_ORDER,
					ylabel="Cummulative Poke\nDuration (ms)",
					title=f"Subjid {subjid} {cat} cummulative poke duration ({mode_label})",
					moving_avg=moving_avg,
					window_size=window_size,
					step_size=step_size,
					ylim_bottom=0,
				)
				if summary_fig is not None:
					figs.append(summary_fig)
					summary_figs.append(summary_fig)
					if save:
						suffix = _summary_save_suffix(moving_avg, window_size, step_size)
						summary_save_specs.append(
							(summary_fig, f"cummulative_poke_time_{cat}_summary_{suffix}")
						)

		_apply_shared_ylim(summary_figs)
		for fig, name in summary_save_specs:
			save_figure(fig, name, subjids=[subjid], dates=date_vals)

	return figs
