"""Plotting valve-state and poke events for a session/time-window.

A debugging view over the raw harp streams, not a derivative-tree figure: it reads
the session's rawdata directly.
"""
from __future__ import annotations

import re
import zoneinfo
from glob import glob
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import harp
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from IPython import get_ipython

from hypnose_behavior.io.loaders import (
    compute_real_time_offset,
    load, load_all_streams, load_experiment_events, load_odor_mapping, concat_digi_events,
    BEHAVIOR_SCHEMA_PATH, OLFACTOMETER_SCHEMA_PATH,
)

def plot_valve_and_poke_events(
    root,
    time_window=None,
    interactive=True,
    show=True,
    verbose=True,
):
    """
    Plot valve and poke events efficiently by:
      - Discovering experiment subfolders
      - If time_window is provided, selecting only subfolders whose heartbeat span overlaps the window
      - Loading only the required registers (valves, digital inputs, outputs, pulse supplies)
      - Applying the same real-time correction as load_all_streams (heartbeat + folder timestamp)
      - Concatenating and plotting

    time_window: tuple/list like (start_str, end_str or None) using 'HH:MM:SS[.ms]' in Europe/London.
                 If end is None, a 1-minute window starting at start is used.
    """
    import re
    import numpy as np
    import pandas as pd
    import harp
    from pathlib import Path
    from datetime import datetime, timezone
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import zoneinfo
    from glob import glob as _glob

    mpl.rcParams['timezone'] = 'Europe/London'
    uk_tz = zoneinfo.ZoneInfo("Europe/London")

    # --- Discover all experiment folders ---
    behav_dir = Path(root)
    if behav_dir.name != "behav":
        behav_dir = behav_dir.parent if behav_dir.parent.name == "behav" else behav_dir
    exp_dirs = [d for d in behav_dir.iterdir() if d.is_dir() and d.name.startswith("20") and "T" in d.name]
    exp_dirs.sort(key=lambda x: x.name)
    if verbose:
        print(f"Found {len(exp_dirs)} experiment files in {behav_dir}")

    if not exp_dirs:
        raise FileNotFoundError(f"No experiment directories found in: {behav_dir}")

    # Helpers to parse experiment folder timestamp and build a window
    def parse_exp_ts_to_uk(exp_dir: Path) -> datetime:
        # Name format 'YYYY-MM-DDTHH-MM-SS'
        m = re.search(r'\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}', exp_dir.name)
        if not m:
            # fallback: try parent
            m = re.search(r'\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}', exp_dir.parent.name)
        if not m:
            return None
        real_time_str = m.group(0)
        # Original pipeline: treat folder timestamp as UTC, then convert to Europe/London
        ref_utc = datetime.strptime(real_time_str, '%Y-%m-%dT%H-%M-%S').replace(tzinfo=timezone.utc)
        return ref_utc.astimezone(uk_tz)

    # Parse requested time window (Europe/London)
    # If user provided time strings only, anchor them to the date of the first exp_dir
    first_exp_dt_uk = parse_exp_ts_to_uk(exp_dirs[0])
    if first_exp_dt_uk is None:
        # fallback to today's date
        first_exp_dt_uk = datetime.now(uk_tz)
    if time_window is not None:
        start_str = time_window[0]
        end_str = time_window[1] if len(time_window) > 1 and time_window[1] else None

        date_str = first_exp_dt_uk.strftime('%Y-%m-%d')
        # Try with and without microseconds
        def _parse_hhmmss(s):
            for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"):
                try:
                    return pd.to_datetime(f"{date_str} {s}", format=fmt)
                except Exception:
                    continue
            # generic parser
            return pd.to_datetime(f"{date_str} {s}", errors='coerce')
        start_dt = _parse_hhmmss(start_str)
        if pd.isna(start_dt):
            raise ValueError(f"Could not parse start time: {start_str}")
        if start_dt.tzinfo is None:
            start_dt = start_dt.tz_localize(uk_tz)
        if end_str:
            end_dt = _parse_hhmmss(end_str)
            if pd.isna(end_dt):
                raise ValueError(f"Could not parse end time: {end_str}")
            if end_dt.tzinfo is None:
                end_dt = end_dt.tz_localize(uk_tz)
        else:
            end_dt = start_dt + pd.Timedelta(minutes=1)
    else:
        start_dt = None
        end_dt = None

    # --- Readers and minimal loader (only needed registers) ---
    behavior_reader = harp.create_reader(str(BEHAVIOR_SCHEMA_PATH), epoch=harp.REFERENCE_EPOCH)
    olf_reader = harp.create_reader(str(OLFACTOMETER_SCHEMA_PATH), epoch=harp.REFERENCE_EPOCH)

    registers = dict(
        odor_valve_state_0=("olfactometer_valves_0", olf_reader.OdorValveState, "Olfactometer0"),
        odor_valve_state_1=("olfactometer_valves_1", olf_reader.OdorValveState, "Olfactometer1"),
        digital_input=("digital_input_data", behavior_reader.DigitalInputState, "Behavior"),
        pulse_supply_1=("pulse_supply_1", behavior_reader.PulseSupplyPort1, "Behavior"),
        pulse_supply_2=("pulse_supply_2", behavior_reader.PulseSupplyPort2, "Behavior"),
        output_set=("output_set", behavior_reader.OutputSet, "Behavior"),
        output_clear=("output_clear", behavior_reader.OutputClear, "Behavior"),
        heartbeat=("heartbeat", behavior_reader.TimestampSeconds, "Behavior"),
    )

    def _safe_concat(dfs):
        dfs = [d for d in dfs if d is not None and isinstance(d, (pd.Series, pd.DataFrame)) and not d.empty]
        if not dfs:
            # Preserve type used downstream
            return pd.DataFrame()
        out = pd.concat(dfs, axis=0)
        try:
            out = out.sort_index()
        except Exception:
            pass
        return out

    def _apply_offset_and_localize(df, offset: pd.Timedelta):
        if df is None or (hasattr(df, "empty") and df.empty):
            return df
        idx = df.index
        # Ensure datetime index
        if not isinstance(idx, pd.DatetimeIndex):
            # If the reader produced a column 'Time', move it to index
            if 'Time' in df.columns:
                df = df.set_index('Time')
                idx = df.index
            else:
                # give up (unexpected)
                return df
        # Apply offset
        try:
            df.index = df.index + offset
        except Exception:
            pass
        # Ensure tz-aware Europe/London
        if df.index.tz is None:
            try:
                df.index = df.index.tz_localize(uk_tz)
            except Exception:
                # If index already tz-aware in another tz, try convert
                try:
                    df.index = df.index.tz_convert(uk_tz)
                except Exception:
                    pass
        return df

    def _slice(df, start, end):
        if df is None or (hasattr(df, "empty") and df.empty) or start is None or end is None:
            return df
        try:
            return df.loc[(df.index >= start) & (df.index <= end)]
        except Exception:
            return df

    # Per-file loader: enumerate all files for a register, skip empty/bad, concat the rest
    def _load_register_files(reg, folder: Path) -> pd.DataFrame | None:
        try:
            folder = Path(folder)
            if not folder.exists():
                return None
            pattern = f"{folder.joinpath(folder.name)}_{reg.register.address}_*.bin"
            files = sorted(_glob(pattern))
            if not files:
                return None
            chunks = []
            for f in files:
                try:
                    df = reg.read(f)
                    if df is None or (hasattr(df, "empty") and df.empty):
                        continue
                    chunks.append(df)
                except Exception:
                    # skip bad file
                    continue
            if not chunks:
                return None
            out = pd.concat(chunks, axis=0)
            try:
                out = out.sort_index()
            except Exception:
                pass
            return out
        except Exception:
            return None

    def _compute_real_time_offset(exp_dir: Path) -> tuple[pd.Timedelta, pd.Timestamp | None, pd.Timestamp | None]:
        """
        Compute the same real_time_offset used by load_all_streams.
        Also returns the heartbeat span (after offset) for quick overlap checks.
        """
        offset = pd.Timedelta(0)
        hb_start_uk = None
        hb_end_uk = None
        try:
            hb = load(registers['heartbeat'][1], exp_dir / registers['heartbeat'][2])
            if not hb.empty:
                hb = hb.reset_index()  # ensure 'Time' is a column
                hb['Time'] = pd.to_datetime(hb['Time'], errors='coerce')
            # The loader's own offset, never a second copy of it: a drifting copy
            # would shift every timestamp here relative to every other figure.
            offset = compute_real_time_offset(exp_dir, hb)
            # Heartbeat span (after offset)
            if 'Time' in hb.columns and not hb.empty:
                times = pd.to_datetime(hb['Time'], errors='coerce') + offset
                if times.dt.tz is None:
                    times = times.dt.tz_localize(uk_tz)
                else:
                    times = times.dt.tz_convert(uk_tz)
                hb_start_uk = times.min()
                hb_end_uk = times.max()
        except Exception as e:
            if verbose:
                print(f"[WARN] Heartbeat timing failed for {exp_dir.name}: {e}")
        return offset, hb_start_uk, hb_end_uk

    # If a time_window is provided, preselect only exp_dirs that overlap it using heartbeat span
    candidate_exp_dirs = []
    exp_dir_offsets = {}           # exp_dir -> offset
    exp_dir_first_times = []       # for session cut detection, earliest per exp
    for exp_dir in exp_dirs:
        offset, hb_start, hb_end = _compute_real_time_offset(exp_dir)
        exp_dir_offsets[exp_dir] = offset
        # Collect earliest known time (for later session cut markers)
        if hb_start is not None:
            exp_dir_first_times.append(hb_start)

        if start_dt is None or end_dt is None:
            candidate_exp_dirs.append(exp_dir)
            continue
        # Include only if heartbeat span overlaps the requested window
        if hb_start is None or hb_end is None:
            # Unknown span; conservatively include
            candidate_exp_dirs.append(exp_dir)
        else:
            if not (hb_end < start_dt or hb_start > end_dt):
                candidate_exp_dirs.append(exp_dir)

    if verbose:
        print(f"Selected {len(candidate_exp_dirs)} experiment(s) for loading based on time window overlap.")

    # --- Load and align required streams for each selected experiment ---
    valves0_list = []
    valves1_list = []
    digital_list = []
    pulse1_list = []
    pulse2_list = []
    output_set_list = []
    output_clear_list = []
    endinit_list = []

    def _try_load(label: str, reg, folder: Path) -> pd.DataFrame | None:
        try:
            if not folder.exists():
                return None
            # Load all matching files per register, skip empty/bad, concat good parts
            df = _load_register_files(reg, folder)
            if df is None or (hasattr(df, 'empty') and df.empty):
                return None
            return df
        except Exception as e:
            if verbose:
                print(f"[WARN] {label} load failed in {folder}: {e}")
            return None

    for exp_dir in candidate_exp_dirs:
        offset = exp_dir_offsets.get(exp_dir, pd.Timedelta(0))

        loaded = {
            key: _try_load(label, reg, exp_dir / subfolder)
            for key, (label, reg, subfolder) in registers.items()
            if key != "heartbeat"
        }

        # Apply offset and tz to each loaded stream
        for k in list(loaded.keys()):
            loaded[k] = _apply_offset_and_localize(loaded[k], offset)
            # If a time window is specified, slice to it now
            if time_window is not None:
                loaded[k] = _slice(loaded[k], start_dt, end_dt)

        # Add to accumulators
        if loaded.get('odor_valve_state_0') is not None and not loaded['odor_valve_state_0'].empty:
            valves0_list.append(loaded['odor_valve_state_0'])
        if loaded.get('odor_valve_state_1') is not None and not loaded['odor_valve_state_1'].empty:
            valves1_list.append(loaded['odor_valve_state_1'])
        if loaded.get('digital_input') is not None and not loaded['digital_input'].empty:
            digital_list.append(loaded['digital_input'])
        if loaded.get('pulse_supply_1') is not None and not loaded['pulse_supply_1'].empty:
            pulse1_list.append(loaded['pulse_supply_1'])
        if loaded.get('pulse_supply_2') is not None and not loaded['pulse_supply_2'].empty:
            pulse2_list.append(loaded['pulse_supply_2'])
        if loaded.get('output_set') is not None and not loaded['output_set'].empty:
            output_set_list.append(loaded['output_set'])
        if loaded.get('output_clear') is not None and not loaded['output_clear'].empty:
            output_clear_list.append(loaded['output_clear'])

        # EndInitiation events via the synced event loader (already uses the same offset logic internally)
        try:
            events = load_experiment_events(exp_dir, verbose=False)
            endinit_df = events.get('combined_end_initiation_df', pd.DataFrame())
            if isinstance(endinit_df, pd.DataFrame) and not endinit_df.empty:
                # Ensure index on Time and tz-aware
                if endinit_df.index.name != "Time" and "Time" in endinit_df.columns:
                    endinit_df = endinit_df.set_index("Time")
                if endinit_df.index.tz is None:
                    endinit_df.index = endinit_df.index.tz_localize(uk_tz)
                endinit_df = endinit_df.sort_index()
                # Slice to window if provided
                if time_window is not None:
                    endinit_df = _slice(endinit_df, start_dt, end_dt)
                # Keep only the EndInitiation flag
                if 'EndInitiation' not in endinit_df.columns and not endinit_df.empty:
                    endinit_df['EndInitiation'] = True
                endinit_list.append(endinit_df[['EndInitiation']])
        except Exception as e:
            if verbose:
                print(f"[WARN] Failed to load EndInitiation for {exp_dir.name}: {e}")

    # --- Concatenate all streams ---
    valves_0 = _safe_concat(valves0_list)
    valves_1 = _safe_concat(valves1_list)
    digital_input_data = _safe_concat(digital_list)
    pulse_supply_1 = _safe_concat(pulse1_list)
    pulse_supply_2 = _safe_concat(pulse2_list)
    output_set_all = _safe_concat(output_set_list)
    output_clear_all = _safe_concat(output_clear_list)
    endinit_df = _safe_concat(endinit_list)

    # Localize remaining naive indices (defensive)
    for df in [valves_0, valves_1, digital_input_data, pulse_supply_1, pulse_supply_2, endinit_df]:
        if isinstance(df, (pd.DataFrame, pd.Series)) and not df.empty:
            try:
                if df.index.tz is None:
                    df.index = df.index.tz_localize(uk_tz)
            except Exception:
                pass

    # Create odour_led (same logic as load_all_streams)
    if isinstance(output_set_all, pd.DataFrame) and not output_set_all.empty and \
       isinstance(output_clear_all, pd.DataFrame) and not output_clear_all.empty:
        try:
            if 'DOPort0' in output_clear_all and 'DOPort0' in output_set_all:
                odour_led = concat_digi_events(output_clear_all['DOPort0'].astype(bool),
                                               output_set_all['DOPort0'].astype(bool))
            else:
                odour_led = pd.Series(dtype=bool)
                if verbose:
                    print("[WARN] DOPort0 not found in outputs; odour_led unavailable.")
        except Exception as e:
            odour_led = pd.Series(dtype=bool)
            if verbose:
                print(f"[WARN] Could not create odour_led: {e}")
    else:
        odour_led = pd.Series(dtype=bool)
        if verbose:
            print("Could not create odour_led (missing output data).")

    # --- Detect session cuts (>5 min gap) using earliest per-experiment time (from heartbeat spans) ---
    session_cuts = []
    all_start_times = [t for t in exp_dir_first_times if t is not None]
    all_start_times = sorted(all_start_times)
    for i in range(1, len(all_start_times)):
        gap = (all_start_times[i] - all_start_times[i-1]).total_seconds()
        if gap > 300:
            session_cuts.append(all_start_times[i])
    if verbose and session_cuts:
        print(f"Session cuts detected at: {[t.strftime('%H:%M:%S') for t in session_cuts]}")

    # If a time_window was provided and we didn't slice earlier because of missing offset, do it now
    if time_window is not None:
        def restrict(df):
            if isinstance(df, (pd.DataFrame, pd.Series)) and not df.empty:
                try:
                    return df.loc[(df.index >= start_dt) & (df.index <= end_dt)]
                except Exception:
                    return df
            return df
        valves_0 = restrict(valves_0)
        valves_1 = restrict(valves_1)
        digital_input_data = restrict(digital_input_data)
        pulse_supply_1 = restrict(pulse_supply_1)
        pulse_supply_2 = restrict(pulse_supply_2)
        odour_led = restrict(odour_led) if isinstance(odour_led, pd.Series) else odour_led
        endinit_df = restrict(endinit_df)

    # --- Odor mapping (names for legend) ---
    odor_map = load_odor_mapping(exp_dirs[0], verbose=False)
    odour_to_olfactometer_map = odor_map.get('odour_to_olfactometer_map', [["A","B","C","D"],["E","F","G","Purge"]])

    # --- Plotting helpers ---
    def extend_to_window_end(df, end_dt_local):
        """Extend the last value of a DataFrame or Series to end_dt if needed."""
        if df is None or (hasattr(df, 'empty') and df.empty) or end_dt_local is None:
            return df
        try:
            if df.index[-1] >= end_dt_local:
                return df
        except Exception:
            return df
        if isinstance(df, pd.DataFrame):
            last_row = df.iloc[[-1]].copy()
            last_row.index = [end_dt_local]
            return pd.concat([df, last_row]).sort_index()
        elif isinstance(df, pd.Series):
            last_val = df.iloc[-1]
            extension = pd.Series([last_val], index=[end_dt_local])
            return pd.concat([df, extension]).sort_index()
        return df

    if interactive:
        try:
            get_ipython().run_line_magic('matplotlib', 'ipympl')
        except Exception:
            plt.ion()
    fig = plt.figure(figsize=(10, 5))
    ax = plt.gca()

    # Colors for valves
    valve_colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']

    # --- Extend signals to end_dt for clean step lines in a clipped window ---
    if time_window is not None:
        for name in [
            'odour_led',
            'digital_input_data',
            'valves_0',
            'valves_1',
            'pulse_supply_1',
            'pulse_supply_2',
            'endinit_df',
        ]:
            obj = locals()[name]
            if obj is not None and hasattr(obj, "empty") and not obj.empty:
                obj_extended = extend_to_window_end(obj, end_dt)
                locals()[name] = obj_extended

    # Plot odour LED and pokes
    if isinstance(odour_led, pd.Series) and not odour_led.empty:
        plt.step(odour_led.index, odour_led.astype(float) * 0.8, where='post', c='black', linewidth=2, label='Odour LED', alpha=0.7)
    if isinstance(digital_input_data, pd.DataFrame) and not digital_input_data.empty and 'DIPort0' in digital_input_data:
        plt.step(digital_input_data.index, digital_input_data['DIPort0'].astype(float) * 0.6, where='post', c='darkgray', linewidth=1, label='Odour Pokes')

    # Plot individual valves from olfactometer 0 and EndInitiation events and reward delivery
    valve_offset = -0.2
    if isinstance(valves_0, pd.DataFrame) and not valves_0.empty:
        for i, valve_col in enumerate(valves_0.columns):
            valve_data = valves_0[valve_col]
            color = valve_colors[i % len(valve_colors)]
            plt.step(valve_data.index, valve_data.astype(float) * 0.8 + valve_offset, where='post',
                    c=color, linewidth=1.5, label=f'{odour_to_olfactometer_map[0][i] if i < len(odour_to_olfactometer_map[0]) else valve_col} (Olfac1)', alpha=0.8)
            if isinstance(endinit_df, pd.DataFrame) and not endinit_df.empty and 'EndInitiation' in endinit_df:
                # Filter out NaT values from index before plotting
                valid_mask = endinit_df.index.notna()
                valid_endinit = endinit_df[valid_mask]
                if not valid_endinit.empty:
                    plt.scatter(valid_endinit.index, valid_endinit['EndInitiation'].astype(float) * 0.5 + valve_offset, s=5, c='k')
            if isinstance(pulse_supply_1, pd.DataFrame) and not pulse_supply_1.empty:
                valid_mask = pulse_supply_1.index.notna()
                valid_supply_1 = pulse_supply_1[valid_mask]
                if not valid_supply_1.empty:
                    plt.scatter(valid_supply_1.index, np.ones(len(valid_supply_1)) * (0.5 + valve_offset), s=5, c='r')
            if isinstance(pulse_supply_2, pd.DataFrame) and not pulse_supply_2.empty:
                valid_mask = pulse_supply_2.index.notna()
                valid_supply_2 = pulse_supply_2[valid_mask]
                if not valid_supply_2.empty:
                    plt.scatter(valid_supply_2.index, np.ones(len(valid_supply_2)) * (0.5 + valve_offset), s=5, c='r')
            valve_offset -= 0.3

    # Plot individual valves from olfactometer 1 and EndInitiation events and reward delivery
    if isinstance(valves_1, pd.DataFrame) and not valves_1.empty:
        base = len(valves_0.columns) if isinstance(valves_0, pd.DataFrame) and not valves_0.empty else 0
        for i, valve_col in enumerate(valves_1.columns):
            valve_data = valves_1[valve_col]
            color = valve_colors[(i + base) % len(valve_colors)]
            plt.step(valve_data.index, valve_data.astype(float) * 0.8 + valve_offset, where='post',
                    c=color, linewidth=1.5, label=f'{odour_to_olfactometer_map[1][i] if i < len(odour_to_olfactometer_map[1]) else valve_col} (Olfac2)', alpha=0.8, linestyle='--')
            if isinstance(endinit_df, pd.DataFrame) and not endinit_df.empty and 'EndInitiation' in endinit_df:
                # Filter out NaT values from index before plotting
                valid_mask = endinit_df.index.notna()
                valid_endinit = endinit_df[valid_mask]
                if not valid_endinit.empty:
                    plt.scatter(valid_endinit.index, valid_endinit['EndInitiation'].astype(float) * 0.5 + valve_offset, s=5, c='k')
            if isinstance(pulse_supply_1, pd.DataFrame) and not pulse_supply_1.empty:
                valid_mask = pulse_supply_1.index.notna()
                valid_supply_1 = pulse_supply_1[valid_mask]
                if not valid_supply_1.empty:
                    plt.scatter(valid_supply_1.index, np.ones(len(valid_supply_1)) * (0.5 + valve_offset), s=5, c='r')
            if isinstance(pulse_supply_2, pd.DataFrame) and not pulse_supply_2.empty:
                valid_mask = pulse_supply_2.index.notna()
                valid_supply_2 = pulse_supply_2[valid_mask]
                if not valid_supply_2.empty:
                    plt.scatter(valid_supply_2.index, np.ones(len(valid_supply_2)) * (0.5 + valve_offset), s=5, c='r')
            valve_offset -= 0.3

    # Plot reward pokes and reward delivery
    if isinstance(digital_input_data, pd.DataFrame) and not digital_input_data.empty:
        for di_col in digital_input_data.columns:
            if di_col == 'DIPort1':
                DIPort_data = digital_input_data[di_col]
                plt.step(DIPort_data.index, DIPort_data.astype(float) * 0.8 + valve_offset, where='post', c='orange', linewidth=1.5, label='Reward Pokes A')
                if isinstance(pulse_supply_1, pd.DataFrame) and not pulse_supply_1.empty:
                    valid_mask = pulse_supply_1.index.notna()
                    valid_supply_1 = pulse_supply_1[valid_mask]
                    if not valid_supply_1.empty:
                        plt.scatter(valid_supply_1.index, np.ones(len(valid_supply_1)) * (0.5 + valve_offset), s=5, c='r')
                if isinstance(pulse_supply_2, pd.DataFrame) and not pulse_supply_2.empty:
                    valid_mask = pulse_supply_2.index.notna()
                    valid_supply_2 = pulse_supply_2[valid_mask]
                    if not valid_supply_2.empty:
                        plt.scatter(valid_supply_2.index, np.ones(len(valid_supply_2)) * (0.5 + valve_offset), s=5, c='r')
                if isinstance(endinit_df, pd.DataFrame) and not endinit_df.empty and 'EndInitiation' in endinit_df:
                    valid_mask = endinit_df.index.notna()
                    valid_endinit = endinit_df[valid_mask]
                    if not valid_endinit.empty:
                        plt.scatter(valid_endinit.index, valid_endinit['EndInitiation'].astype(float) * 0.5 + valve_offset, s=5, c='k')
                valve_offset -= 0.3
            elif di_col == 'DIPort2':
                DIPort_data = digital_input_data[di_col]
                plt.step(DIPort_data.index, DIPort_data.astype(float) * 0.8 + valve_offset, where='post', c='cyan', linewidth=1.5, label='Reward Pokes B')
                if isinstance(pulse_supply_1, pd.DataFrame) and not pulse_supply_1.empty:
                    valid_mask = pulse_supply_1.index.notna()
                    valid_supply_1 = pulse_supply_1[valid_mask]
                    if not valid_supply_1.empty:
                        plt.scatter(valid_supply_1.index, np.ones(len(valid_supply_1)) * (0.5 + valve_offset), s=5, c='r')
                if isinstance(pulse_supply_2, pd.DataFrame) and not pulse_supply_2.empty:
                    valid_mask = pulse_supply_2.index.notna()
                    valid_supply_2 = pulse_supply_2[valid_mask]
                    if not valid_supply_2.empty:
                        plt.scatter(valid_supply_2.index, np.ones(len(valid_supply_2)) * (0.5 + valve_offset), s=5, c='r', label='Reward Delivery')
                if isinstance(endinit_df, pd.DataFrame) and not endinit_df.empty and 'EndInitiation' in endinit_df:
                    valid_mask = endinit_df.index.notna()
                    valid_endinit = endinit_df[valid_mask]
                    if not valid_endinit.empty:
                        plt.scatter(valid_endinit.index, valid_endinit['EndInitiation'].astype(float) * 0.5 + valve_offset, s=5, c='k', label='Trial End')
                valve_offset -= 0.3

    # --- Indicate session cuts ---
    for cut_time in session_cuts:
        plt.axvline(cut_time, color='gray', linestyle=':', linewidth=2, alpha=0.7)
        plt.text(cut_time, plt.ylim()[1], 'Session Cut', color='gray', rotation=90, va='top', ha='right', fontsize=8)

    # --- Adaptive/fine x-axis scale ---
    class AdaptiveTimeFormatter(mdates.DateFormatter):
        def __call__(self, x, pos=0):
            dt = mdates.num2date(x)
            return dt.strftime('%H:%M:%S.%f')[:-4]  # Show HH:MM:SS.ms

    def update_ticks(event):
        ax = event.canvas.figure.axes[0]
        xlim = ax.get_xlim()
        dt_start = mdates.num2date(xlim[0])
        dt_end = mdates.num2date(xlim[1])
        span = (dt_end - dt_start).total_seconds()
        # Adaptive major/minor ticks
        if span < 2:
            ax.xaxis.set_major_locator(mdates.SecondLocator(interval=1))
            ax.xaxis.set_minor_locator(mdates.MicrosecondLocator(interval=100000))  # 100ms
        elif span < 10:
            ax.xaxis.set_major_locator(mdates.SecondLocator(interval=2))
            ax.xaxis.set_minor_locator(mdates.SecondLocator(interval=1))
        elif span < 60:
            ax.xaxis.set_major_locator(mdates.SecondLocator(interval=10))
            ax.xaxis.set_minor_locator(mdates.SecondLocator(interval=1))
        elif span < 600:
            ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=1))
            ax.xaxis.set_minor_locator(mdates.SecondLocator(interval=10))
        elif span < 3600:
            ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=5))
            ax.xaxis.set_minor_locator(mdates.MinuteLocator(interval=1))
        elif span < 6*3600:
            ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
            ax.xaxis.set_minor_locator(mdates.MinuteLocator(interval=10))
        else:
            ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
            ax.xaxis.set_minor_locator(mdates.HourLocator(interval=1))
        ax.xaxis.set_major_formatter(AdaptiveTimeFormatter('%H:%M:%S.%f'))
        event.canvas.draw_idle()

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S', tz=uk_tz))

    if interactive:
        plt.gcf().canvas.mpl_connect('draw_event', update_ticks)
        # Trigger once
        try:
            update_ticks(type('event', (object,), {'canvas': plt.gcf().canvas})())
        except Exception:
            pass

    plt.xlabel('Time (Europe/London)')
    if time_window is not None:
        plt.xlim(start_dt, end_dt)
    plt.title('Individual Olfactometer Valve States vs Odour Pokes and LED')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.ylim(valve_offset - 0.3, 1.2)
    plt.yticks([])  # Removes y-axis tick marks and labels
    plt.tight_layout()

    if show:
        plt.show()

    return fig
