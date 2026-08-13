from __future__ import annotations

from itertools import product
from pathlib import Path
import ast
import json
import re

import pandas as pd

from hypnose_behavior.frames import position_entries_by_trial
from hypnose_behavior.io.loaders import (
    _load_position_data, load_all_streams, load_odor_mapping,
)
from hypnose_behavior.io.layout import derivatives, normalize_subjid, rawdata


_TIMESTAMP_DIR_RE = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}")


def _digits_only(value) -> str:
	digits = "".join(ch for ch in str(value) if ch.isdigit())
	if not digits:
		raise ValueError(f"Could not extract a numeric value from {value!r}")
	return digits


def _normalize_subjid(subjid) -> str:
	return normalize_subjid(_digits_only(subjid))


def _normalize_date(date) -> str:
	return _digits_only(date)


def _resolve_session_paths(subjid, date) -> tuple[Path, Path, Path, Path]:
	raw_session = rawdata.find_session(subjid, date=_normalize_date(date))
	raw_subject_dir = raw_session.subject_dir
	raw_session_dir = raw_session.path
	deriv_session_dir = derivatives.find_session(subjid, date=_normalize_date(date)).path
	results_dir = deriv_session_dir / "saved_analysis_results"
	if not results_dir.exists():
		raise FileNotFoundError(f"Results directory not found: {results_dir}")
	return raw_subject_dir, raw_session_dir, deriv_session_dir, results_dir


def _load_trial_data(results_dir: Path) -> pd.DataFrame:
	trial_parquet = results_dir / "trial_data.parquet"
	trial_csv = results_dir / "trial_data.csv"
	if trial_parquet.exists():
		trial_df = pd.read_parquet(trial_parquet)
	elif trial_csv.exists():
		trial_df = pd.read_csv(trial_csv)
	else:
		raise FileNotFoundError(f"Neither trial_data.parquet nor trial_data.csv exists in {results_dir}")

	trial_df = trial_df.copy()
	for col in [
		"sequence_start",
		"sequence_end",
		"timestamp",
		"initiation_sequence_time",
		"abortion_time",
		"fa_time",
		"await_reward_time",
		"first_supply_time",
		"poke_window_end",
		"reward_window_end",
	]:
		if col in trial_df.columns:
			trial_df[col] = pd.to_datetime(trial_df[col], errors="coerce")
	if "is_aborted" in trial_df.columns:
		trial_df["is_aborted"] = trial_df["is_aborted"].fillna(False).astype(bool)
	else:
		trial_df["is_aborted"] = False
	return trial_df


def _print_small_table(df: pd.DataFrame) -> None:
	if df.empty:
		print("<empty>")
	else:
		print(df.to_string(index=False))


def _is_within_any(ts: pd.Timestamp, windows: list[tuple[pd.Timestamp, pd.Timestamp]]) -> bool:
	for start, end in windows:
		if start <= ts <= end:
			return True
	return False


def _session_experiment_dirs(raw_session_dir: Path) -> list[Path]:
	behav_dir = raw_session_dir / "behav"
	if not behav_dir.exists():
		raise FileNotFoundError(f"No behav directory found in {raw_session_dir}")
	return sorted(
		[d for d in behav_dir.iterdir() if d.is_dir() and _TIMESTAMP_DIR_RE.fullmatch(d.name)],
		key=lambda path: path.name,
	)


def _load_raw_supply_times(raw_session_dir: Path) -> list[dict]:
	rows: list[dict] = []
	for experiment_dir in _session_experiment_dirs(raw_session_dir):
		try:
			data = load_all_streams(experiment_dir, verbose=False)
		except Exception as exc:
			print(f"[raw reward scan] Failed to load {experiment_dir.name}: {exc}")
			continue
		for port_key, port_num in (("pulse_supply_1", 1), ("pulse_supply_2", 2)):
			frame = data.get(port_key, pd.DataFrame())
			if not isinstance(frame, pd.DataFrame) or frame.empty:
				continue
			times = pd.Index(frame.index).dropna()
			for ts in times:
				rows.append(
					{
						"experiment": experiment_dir.name,
						"port": port_num,
						"time": pd.Timestamp(ts),
					}
				)
	return rows


def _load_purge_events_from_raw(raw_session_dir: Path) -> list[dict]:
	purge_events: list[dict] = []
	for experiment_dir in _session_experiment_dirs(raw_session_dir):
		try:
			data = load_all_streams(experiment_dir, verbose=False)
			odor_map = load_odor_mapping(experiment_dir, data=data, verbose=False)
		except Exception as exc:
			print(f"[purge scan] Failed to load {experiment_dir.name}: {exc}")
			continue

		valves = odor_map.get("olfactometer_valves", {}) or {}
		valve_to_odor = odor_map.get("valve_to_odor", {}) or {}
		odor_names_by_id = odor_map.get("odour_to_olfactometer_map") or odor_map.get("odor_to_olfactometer_map") or []

		def _resolve_name(olf_id: int, col_index: int, valve_key: str) -> str | None:
			odor_name = valve_to_odor.get(valve_key)
			if isinstance(odor_name, str):
				return odor_name
			if isinstance(odor_names_by_id, (list, tuple)) and len(odor_names_by_id) > olf_id:
				row = odor_names_by_id[olf_id]
				if isinstance(row, (list, tuple)) and 0 <= col_index < len(row):
					maybe = row[col_index]
					if isinstance(maybe, str):
						return maybe
			return None

		for olf_id, valve_df in valves.items():
			if not isinstance(valve_df, pd.DataFrame) or valve_df.empty:
				continue
			for col_index, col in enumerate(valve_df.columns):
				valve_key = f"{olf_id}{col_index}"
				odor_name = _resolve_name(int(olf_id), col_index, valve_key)
				if not isinstance(odor_name, str) or odor_name.lower() != "purge":
					continue
				series = valve_df[col].astype(bool)
				rises = series & ~series.shift(1, fill_value=False)
				falls = ~series & series.shift(1, fill_value=False)
				starts = list(series.index[rises])
				ends = list(series.index[falls])
				start_idx = 0
				end_idx = 0
				while start_idx < len(starts) and end_idx < len(ends):
					start_ts = starts[start_idx]
					end_ts = ends[end_idx]
					if end_ts <= start_ts:
						end_idx += 1
						continue
					purge_events.append(
						{
							"experiment": experiment_dir.name,
							"olf_id": int(olf_id),
							"col": str(col),
							"start": pd.Timestamp(start_ts),
							"end": pd.Timestamp(end_ts),
							"duration_ms": (pd.Timestamp(end_ts) - pd.Timestamp(start_ts)).total_seconds() * 1000.0,
						}
					)
					start_idx += 1
					end_idx += 1
	purge_events.sort(key=lambda row: row["start"])
	return purge_events


def check_supply_pulses_odour_discrimination(subjid, date):
	"""Print supply pulses that fall outside odour-discrimination reward windows."""
	_, raw_session_dir, _, results_dir = _resolve_session_paths(subjid, date)
	trial_df = _load_trial_data(results_dir)

	if "odourdiscrimination_mode" in trial_df.columns:
		mode_mask = trial_df["odourdiscrimination_mode"].fillna(False).astype(bool)
	else:
		mode_mask = pd.Series(True, index=trial_df.index)

	eligible = trial_df[mode_mask].copy()
	if eligible.empty:
		print(f"[{_normalize_subjid(subjid)} {_normalize_date(date)}] No odour-discrimination trials found.")
		return pd.DataFrame(columns=["experiment", "port", "time"])

	windows: list[tuple[pd.Timestamp, pd.Timestamp]] = []
	for _, row in eligible.iterrows():
		await_time = row.get("await_reward_time")
		window_end = row.get("reward_window_end")
		if pd.isna(await_time) or pd.isna(window_end):
			continue
		windows.append((pd.Timestamp(await_time), pd.Timestamp(window_end)))

	raw_pulses = pd.DataFrame(_load_raw_supply_times(raw_session_dir))
	if raw_pulses.empty:
		print(f"[{_normalize_subjid(subjid)} {_normalize_date(date)}] No supply pulses found in rawdata.")
		return raw_pulses

	raw_pulses = raw_pulses.sort_values(["time", "experiment", "port"], kind="stable").reset_index(drop=True)
	outside = raw_pulses[~raw_pulses["time"].apply(lambda ts: _is_within_any(pd.Timestamp(ts), windows))].copy()

	print(f"[{_normalize_subjid(subjid)} {_normalize_date(date)}] odour-discrimination trials: {len(eligible)}")
	print(f"Reward windows: {len(windows)}")
	print(f"Supply port events: {len(raw_pulses)}")
	print(f"Outside reward windows: {len(outside)}")
	if not outside.empty:
		_print_small_table(outside.head(20))

	return outside


def list_short_purges_between_odors(subjid, date, trial_id, *, threshold_ms: float = 200.0, onset_slack_ms: float = 50.0, verbose: bool = True):
	"""Print short purge events that occur between adjacent odors in one trial."""
	_, raw_session_dir, _, results_dir = _resolve_session_paths(subjid, date)
	trial_df = _load_trial_data(results_dir)
	if "trial_id" not in trial_df.columns:
		raise KeyError("trial_data does not contain trial_id")

	match = trial_df[trial_df["trial_id"].astype(str) == str(trial_id)].copy()
	if match.empty:
		match = trial_df[trial_df.index.astype(str) == str(trial_id)].copy()
	if match.empty:
		raise KeyError(f"No trial found for trial_id={trial_id!r}")

	row = match.iloc[0]
	if bool(row.get("is_aborted", False)):
		print(f"Trial {trial_id} is aborted; continuing with its stored odor timings.")

	# `in_valve_times` is the flag matching `position_valve_times`, the blob this read
	# before Phase 7b.4b -- and the one that matters most here, since that blob is a
	# *superset* of the other two (`DECISIONS.md` section 2).
	valves_by_trial = position_entries_by_trial(
		_load_position_data(results_dir, match), "in_valve_times")
	entries = [(info.get("position"), info)
	           for info in valves_by_trial.get(row.get("global_trial_id")) or []]
	if not entries:
		raise ValueError(f"No valve positions found for trial {trial_id}")

	def _position_sort_key(item):
		position = item[0]
		try:
			return int(position)
		except Exception:
			return str(position)

	entries = sorted(entries, key=_position_sort_key)
	purge_events = _load_purge_events_from_raw(raw_session_dir)

	windows: list[tuple[object, object, pd.Timestamp, pd.Timestamp]] = []
	for (current_pos, current_info), (next_pos, next_info) in zip(entries[:-1], entries[1:]):
		current_end = current_info.get("valve_end")
		next_start = next_info.get("valve_start")
		if current_end is None or next_start is None:
			continue
		current_end = pd.Timestamp(current_end)
		next_start = pd.Timestamp(next_start)
		if pd.isna(current_end) or pd.isna(next_start) or current_end >= next_start:
			continue
		windows.append(
			(
				current_pos,
				next_pos,
				current_end - pd.Timedelta(milliseconds=onset_slack_ms),
				next_start + pd.Timedelta(milliseconds=onset_slack_ms),
			)
		)

	matches = []
	for from_pos, to_pos, window_start, window_end in windows:
		for event in purge_events:
			if event["start"] < window_start or event["start"] > window_end:
				continue
			if event["duration_ms"] >= threshold_ms:
				continue
			matches.append(
				{
					"trial_id": row.get("trial_id", trial_id),
					"from_pos": from_pos,
					"to_pos": to_pos,
					"start": event["start"],
					"end": event["end"],
					"duration_ms": event["duration_ms"],
					"experiment": event["experiment"],
					"olf_id": event["olf_id"],
					"col": event["col"],
				}
			)

	result = pd.DataFrame(matches)
	if verbose:
		print(
			f"Short purge events (< {threshold_ms} ms) between odors for trial {trial_id} "
			f"(onset slack ±{onset_slack_ms} ms): {len(result)}"
		)
		if not result.empty:
			_print_small_table(result.sort_values(["start", "from_pos", "to_pos"], kind="stable"))
	return result


list_short_purge_between_odors = list_short_purges_between_odors


def poke_below_threshold(subjid, date, *, threshold_ms: float = 200.0):
	"""Return and print position poke events whose poke time is below a threshold."""
	_, _, _, results_dir = _resolve_session_paths(subjid, date)
	trial_df = _load_trial_data(results_dir)
	completed = trial_df[~trial_df["is_aborted"]].copy()

	# `in_poke_times` matches `position_poke_times`, the blob this read before 7b.4b.
	pokes_by_trial = position_entries_by_trial(
		_load_position_data(results_dir, completed), "in_poke_times")

	rows = []
	for _, row in completed.iterrows():
		trial_identifier = row.get("trial_id", row.name)
		for info in pokes_by_trial.get(row.get("global_trial_id")) or []:
			position = info.get("position")
			poke_ms = pd.to_numeric(info.get("poke_time_ms"), errors="coerce")
			if pd.isna(poke_ms) or poke_ms <= 0 or float(poke_ms) >= float(threshold_ms):
				continue
			rows.append(
				{
					"trial_id": trial_identifier,
					"position": position,
					"odor": info.get("odor_name"),
					"poke_time_ms": float(poke_ms),
					"poke_time": pd.to_datetime(info.get("poke_first_in"), errors="coerce"),
				}
			)

	hits = pd.DataFrame(rows).sort_values(["poke_time", "trial_id", "position"], kind="stable") if rows else pd.DataFrame(columns=["trial_id", "position", "odor", "poke_time_ms", "poke_time"])
	if hits.empty:
		print(f"No position poke times below {threshold_ms} ms found.")
	else:
		print(f"Found {len(hits)} sub-threshold poke events (< {threshold_ms} ms):")
		_print_small_table(hits)
	return hits


def count_raw_reward_events(subjid, date):
	"""Count total raw supply port events across all raw behav experiment folders."""
	_, raw_session_dir, _, _ = _resolve_session_paths(subjid, date)
	rows = _load_raw_supply_times(raw_session_dir)
	if not rows:
		print(f"[{_normalize_subjid(subjid)} {_normalize_date(date)}] No raw supply port events found.")
		return 0

	counts = pd.DataFrame(rows).groupby(["experiment", "port"], as_index=False).size().rename(columns={"size": "count"})
	total = int(counts["count"].sum())
	print(f"[{_normalize_subjid(subjid)} {_normalize_date(date)}] total raw supply port events: {total}")
	_print_small_table(counts.sort_values(["experiment", "port"], kind="stable"))
	return total


def high_speed_before_0(subjid, date, *, threshold: float = 200.0, t_min: float = -0.4, t_max: float = -0.001):
	"""Print trials where speed exceeds a threshold within a pre-zero time window."""
	_, _, _, results_dir = _resolve_session_paths(subjid, date)
	trial_df = _load_trial_data(results_dir)

	speed_path = results_dir / "speed_analysis.parquet"
	if not speed_path.exists():
		raise FileNotFoundError(f"speed_analysis.parquet not found at {speed_path}")
	speed_df = pd.read_parquet(speed_path)

	if "condition" in speed_df.columns:
		speed_df = speed_df[speed_df["condition"].astype(str).str.lower() == "rewarded"].copy()
	if "trial_index" not in speed_df.columns:
		raise KeyError("speed_analysis.parquet does not contain trial_index")

	if "trial_id" in trial_df.columns:
		trial_id_map = trial_df["trial_id"].to_dict()
	else:
		trial_id_map = {}

	response_col = trial_df.get("response_time_category")
	if response_col is None:
		reward_mask = pd.Series(True, index=trial_df.index)
	else:
		reward_mask = response_col.astype(str).str.lower() == "rewarded"
	rewarded_trials = trial_df[(~trial_df["is_aborted"]) & reward_mask].copy()

	rewarded_indices = set(rewarded_trials.index.tolist())
	speed_df = speed_df[speed_df["trial_index"].isin(rewarded_indices)].copy()
	if speed_df.empty:
		print("No rewarded trials with speed bins in the requested window.")
		return pd.DataFrame(columns=["trial_index", "trial_id", "bin_mid_s", "speed"])

	mask_window = speed_df["bin_mid_s"].between(t_min, t_max, inclusive="both")
	mask_speed = speed_df["speed"] > float(threshold)
	suspect = speed_df[mask_window & mask_speed].copy().sort_values(["trial_index", "bin_mid_s"], kind="stable")

	if suspect.empty:
		print(f"No bins exceed {threshold} in window [{t_min}, {t_max}] for rewarded trials.")
		return suspect

	suspect["trial_id"] = suspect["trial_index"].map(lambda idx: trial_id_map.get(idx, idx))
	print(
		f"Found {len(suspect)} speed bins above {threshold} in window [{t_min}, {t_max}] "
		f"across {suspect['trial_index'].nunique()} rewarded trials"
	)
	_print_small_table(suspect[["trial_index", "trial_id", "bin_mid_s", "speed"]])
	return suspect


def _normalize_pairs(subjids, dates):
	if isinstance(subjids, (list, tuple, set)):
		subj_list = list(subjids)
	else:
		subj_list = [subjids]
	if isinstance(dates, (list, tuple, set)):
		date_list = list(dates)
	else:
		date_list = [dates]

	if len(subj_list) == 1 and len(date_list) > 1:
		return [(subj_list[0], date_item) for date_item in date_list]
	if len(date_list) == 1 and len(subj_list) > 1:
		return [(subj_item, date_list[0]) for subj_item in subj_list]
	if len(subj_list) == len(date_list):
		return list(zip(subj_list, date_list))
	return list(product(subj_list, date_list))


def _available_session_dates(layout, subjid) -> list[str]:
	"""Return all session date tokens that exist on disk for a subject."""
	return sorted({session.date for session in layout.find_sessions(subjid)})


def _expand_date_range(subjid, start, end) -> list[str]:
	"""Expand an inclusive (start, end) range to the session dates that exist for a subject."""
	start_token = _normalize_date(start)
	end_token = _normalize_date(end)
	if start_token > end_token:
		start_token, end_token = end_token, start_token
	available = _available_session_dates(derivatives, subjid)
	return [date for date in available if start_token <= date <= end_token]


def check_pre_odor_grace_period(subjids, dates):
	"""Summarize num_odors across non-aborted trials for one or more sessions.

	``dates`` may be a single date, a list of discrete dates, or a 2-tuple
	``(start, end)`` which is treated as an inclusive date range and expanded
	to the sessions that exist on disk for each subject.
	"""
	if isinstance(dates, tuple) and len(dates) == 2:
		subj_list = list(subjids) if isinstance(subjids, (list, tuple, set)) else [subjids]
		pairs = [
			(subjid, date)
			for subjid in subj_list
			for date in _expand_date_range(subjid, dates[0], dates[1])
		]
		if not pairs:
			print(f"No sessions found in range {dates[0]}..{dates[1]} for {subj_list}.")
			return []
	else:
		pairs = _normalize_pairs(subjids, dates)
	summaries = []

	for subjid, date in sorted(pairs, key=lambda item: (_normalize_subjid(item[0]), _normalize_date(item[1]))):
		_, _, _, results_dir = _resolve_session_paths(subjid, date)
		trial_df = _load_trial_data(results_dir)
		trial_df = trial_df[~trial_df["is_aborted"]].copy()
		if trial_df.empty:
			print(f"[{_normalize_subjid(subjid)} {_normalize_date(date)}] No non-aborted trials found.")
			summaries.append({"subjid": subjid, "date": date, "num_odors_counts": {}})
			continue

		counts = trial_df["num_odors"].value_counts(dropna=False).sort_index()
		print(f"[{_normalize_subjid(subjid)} {_normalize_date(date)}] num_odors counts:")
		_print_small_table(counts.rename_axis("num_odors").reset_index(name="count"))

		unique_non_na = [value for value in counts.index.tolist() if pd.notna(value)]
		if len(set(unique_non_na)) > 1:
			max_num_odors = max(unique_non_na)
			below_max = trial_df[trial_df["num_odors"] != max_num_odors].copy()
			columns = [col for col in ["trial_id", "initiation_sequence_time", "odor_sequence", "num_odors"] if col in below_max.columns]
			sort_cols = [col for col in ["initiation_sequence_time", "trial_id"] if col in below_max.columns]
			detail = below_max.sort_values(sort_cols, kind="stable") if sort_cols else below_max
			detail = detail[columns] if columns else below_max
			print(f"Multiple num_odors values detected; trials with num_odors != {max_num_odors}:")
			_print_small_table(detail)

		summaries.append(
			{
				"subjid": subjid,
				"date": date,
				"num_odors_counts": counts.to_dict(),
			}
		)

	return summaries
