"""Loading Hypnose session data: the high-level loaders (load_experiment,
load_all_streams, load_experiment_events, load_odor_mapping) and the trial-data
table readers.

Extracted from trial_classification/classification_utils.py during the restructuring
(Phase 3). Pure move -- behaviour unchanged (verified by the regression harness).

The harp/aeon reader classes and file-reading primitives used to be duplicated here;
they now live once in io/readers.py and are re-exported below (restructure_2 Phase 0.3).
"""
from __future__ import annotations

import os
import re
import json
import zoneinfo
from glob import glob
from pathlib import Path
from datetime import datetime, timezone
from importlib.resources import files

import harp
import pandas as pd
from dotmap import DotMap
from aeon.io.reader import Reader, Csv
import aeon.io.api as api

import hypnose_behavior.trial_classification.detect_settings as detect_settings
from hypnose_behavior.io.paths import get_rawdata_root, get_derivatives_root, get_server_root
from hypnose_behavior.io.layout import rawdata
from hypnose_behavior.utils.helpers import vprint, _get_from_cache, _update_cache
# Reader classes and file-reading primitives are defined once, in io/readers.py. They are
# re-exported here because callers (classification_utils, notebooks) import them from this
# module; readers.py imports nothing from hypnose_behavior, so this direction stays acyclic.
from hypnose_behavior.io.readers import (  # noqa: F401
    SessionData, Video, TimestampedCsvReader,
    load, load_json, load_video, load_csv, concat_digi_events,
)

SCHEMA_DIR = files("hypnose_behavior.resources.device_schemas")
BEHAVIOR_SCHEMA_PATH = SCHEMA_DIR / "behavior.yml"
OLFACTOMETER_SCHEMA_PATH = SCHEMA_DIR / "olfactometer.yml"


def load_experiment(subjid, date, index=None):
    """
    Load experiment data with automatic session detection
    
    Parameters:
    -----------
    subjid : str or int
        Subject ID (e.g., '025' or 25)
    date : str or int  
        Date in format YYYYMMDD (e.g., '20250730' or 20250730)
    index : int, optional
        If multiple experiments exist, specify which one (0, 1, 2, etc.)
    
    Returns:
    --------
    Path object to experiment root, or None if selection needed
    """
    
    # One resolver for the whole family (restructure_2 Phase 2b). It reports the
    # available sessions on a miss, as this function used to do by hand, and raises on
    # an ambiguous subject or date rather than warning and taking the first match.
    session = rawdata.find_session(subjid, date=date)
    subject_dir = session.subject_dir
    print(f"Using subject directory: {subject_dir}")

    session_dir = session.path

    behav_dir = session_dir / "behav"
    if not behav_dir.exists():
        raise FileNotFoundError(f"No behav directory found in {session_dir}")
    
    # Find experiment folders (timestamp folders)
    experiment_dirs = [d for d in behav_dir.iterdir() 
                      if d.is_dir() and re.match(r'\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}', d.name)]
    
    if not experiment_dirs:
        # Better error reporting - show what directories actually exist
        all_dirs = [d.name for d in behav_dir.iterdir() if d.is_dir()]
        raise FileNotFoundError(f"No experiment directories found in {behav_dir}.\n"
                              f"Available directories: {all_dirs}")
    
    # Sort by timestamp (chronological order)
    experiment_dirs.sort(key=lambda x: x.name)
    
    # Handle multiple experiments
    if len(experiment_dirs) == 1:
        # Single experiment - return it directly
        root = experiment_dirs[0]
        print(f"Loaded experiment: {root}")
        return root
    
    elif index is None:
        # Multiple experiments, no index specified
        print(f"Multiple experiments detected for subject {session.subject} on {session.date}:")
        for i, exp_dir in enumerate(experiment_dirs):
            print(f"  Index {i}: {exp_dir.name}")
        print(f"\nPlease run again with index parameter:")
        print(f"root = load_experiment({subjid}, {date}, index=0)  # for first experiment")
        print(f"root = load_experiment({subjid}, {date}, index=1)  # for second experiment")
        return None
    
    else:
        # Index specified
        if index >= len(experiment_dirs) or index < 0:
            raise IndexError(f"Index {index} out of range. Available indices: 0-{len(experiment_dirs)-1}")
        
        root = experiment_dirs[index]
        print(f"Loaded experiment {index}: {root}")
        return root

#Helper function with shorter name 
def exp_data(subjid, date, index=None):
    """Alias for load_experiment with shorter name"""
    return load_experiment(subjid, date, index)


def compute_real_time_offset(root, heartbeat):
    """Offset from the harp hardware clock to UK wall-clock time.

    The session folder name carries the recording's UTC start
    (``YYYY-MM-DDTHH-MM-SS``) and the heartbeat register the hardware clock's;
    the difference is what ``load_all_streams`` adds to every stream's index.

    Written twice before Phase 4a: ``valve_poke_plots._compute_real_time_offset``
    opened with *"compute the same real_time_offset used by load_all_streams"*
    and then did exactly that. Drift between the two would have silently shifted
    every timestamp on that debugging plot relative to every other figure in the
    package -- the audit's finding 15. Returns ``Timedelta(0)`` when the folder
    name carries no timestamp or the heartbeat is unusable.
    """
    if heartbeat is None or heartbeat.empty or 'Time' not in heartbeat.columns or len(heartbeat) == 0:
        return pd.Timedelta(0)

    match = re.search(r'\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}', root.name)
    if not match:
        match = re.search(r'\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}', root.parent.name)
    if not match:
        return pd.Timedelta(0)

    real_time_ref_utc = datetime.strptime(match.group(0), '%Y-%m-%dT%H-%M-%S')
    real_time_ref_utc = real_time_ref_utc.replace(tzinfo=timezone.utc)
    uk_tz = zoneinfo.ZoneInfo("Europe/London")
    real_time_ref = real_time_ref_utc.astimezone(uk_tz)

    start_time_hardware = heartbeat['Time'].iloc[0]
    start_time_dt = (start_time_hardware.to_pydatetime()
                     if isinstance(start_time_hardware, pd.Timestamp) else start_time_hardware)
    if start_time_dt.tzinfo is None:
        start_time_dt = start_time_dt.replace(tzinfo=uk_tz)
    return real_time_ref - start_time_dt


def load_all_streams(root, apply_corrections = True, *args, verbose: bool = True, **kwargs):
    """
    Load all behavioral data streams with proper timestamp synchronization
    """
    vprint(verbose, f"Loading data streams from: {root}")
    
    # Create readers
    behavior_reader = harp.create_reader(str(BEHAVIOR_SCHEMA_PATH), epoch=harp.REFERENCE_EPOCH)
    olfactometer_reader = harp.create_reader(str(OLFACTOMETER_SCHEMA_PATH), epoch=harp.REFERENCE_EPOCH)
    
    data = {}
    
    # === TIMESTAMP SYNCHRONIZATION ===
    # Load heartbeat for timestamp conversion
    try:
        heartbeat = load(behavior_reader.TimestampSeconds, root/"Behavior")
        if not heartbeat.empty:
            heartbeat.reset_index(inplace=True)
        vprint(verbose, "Loaded heartbeat data")
    except Exception as e:
        print(f"Failed to load heartbeat: {e}")
        heartbeat = pd.DataFrame(columns=['Time', 'TimestampSeconds'])
    
    # Calculate real-time offset
    real_time_offset = pd.Timedelta(0)
    if not heartbeat.empty and 'Time' in heartbeat.columns and len(heartbeat) > 0:
        try:
            real_time_offset = compute_real_time_offset(root, heartbeat)
            if real_time_offset != pd.Timedelta(0):
                vprint(verbose, f"Calculated real-time offset: {real_time_offset}")
        except Exception as e:
            print(f"Error calculating real-time offset: {e}")

    # Create timestamp interpolation mapping
    timestamp_to_time = pd.Series()
    if not heartbeat.empty and 'Time' in heartbeat.columns and 'TimestampSeconds' in heartbeat.columns:
        heartbeat['Time'] = pd.to_datetime(heartbeat['Time'], errors='coerce')
        timestamp_to_time = pd.Series(data=heartbeat['Time'].values, index=heartbeat['TimestampSeconds'])
        vprint(verbose, "Created timestamp interpolation mapping")
    
    def interpolate_time(seconds):
        """Interpolate timestamps from seconds, with safety checks"""
        if timestamp_to_time.empty:
            return pd.NaT
        int_seconds = int(seconds)
        fractional_seconds = seconds % 1
        if int_seconds in timestamp_to_time.index:
            base_time = timestamp_to_time.loc[int_seconds]
            return base_time + pd.to_timedelta(fractional_seconds, unit='s')
        return pd.NaT
    
    # Store timing data
    data['heartbeat'] = heartbeat
    data['real_time_offset'] = real_time_offset
    data['timestamp_to_time'] = timestamp_to_time
    data['interpolate_time'] = interpolate_time
    
    # === LOAD ALL OTHER DATA STREAMS ===

    # Core behavioral data
    try:
        data['digital_input_data'] = load(behavior_reader.DigitalInputState, root/"Behavior")
        vprint(verbose, "Loaded digital_input_data")
    except Exception as e:
        print(f"Failed to load digital_input_data: {e}")
        data['digital_input_data'] = pd.DataFrame()
    
    try:
        data['output_set'] = load(behavior_reader.OutputSet, root/"Behavior")
        vprint(verbose, "Loaded output_set")
    except Exception as e:
        print(f"Failed to load output_set: {e}")
        data['output_set'] = pd.DataFrame()
    
    try:
        data['output_clear'] = load(behavior_reader.OutputClear, root/"Behavior")
        vprint(verbose, "Loaded output_clear")
    except Exception as e:
        print(f"Failed to load output_clear: {e}")
        data['output_clear'] = pd.DataFrame()
    
    # Olfactometer valve data
    try:
        data['olfactometer_valves_0'] = load(olfactometer_reader.OdorValveState, root/"Olfactometer0")
        vprint(verbose, "Loaded olfactometer_valves_0")
    except Exception as e:
        print(f"Failed to load olfactometer_valves_0: {e}")
        data['olfactometer_valves_0'] = pd.DataFrame()
    
    try:
        data['olfactometer_valves_1'] = load(olfactometer_reader.OdorValveState, root/"Olfactometer1")
        vprint(verbose, "Loaded olfactometer_valves_1")
    except Exception as e:
        print(f"Failed to load olfactometer_valves_1: {e}")
        data['olfactometer_valves_1'] = pd.DataFrame()
    
    # End valve states (commented in original but included for completeness)
    try:
        data['olfactometer_end_0'] = load(olfactometer_reader.EndValveState, root/"Olfactometer0")
        vprint(verbose, "Loaded olfactometer_end_0")
    except Exception as e:
        print(f"Failed to load olfactometer_end_0: {e}")
        data['olfactometer_end_0'] = pd.DataFrame()
    
    # Analog data
    try:
        data['analog_data'] = load(behavior_reader.AnalogData, root/"Behavior")
        vprint(verbose, "Loaded analog_data")
    except Exception as e:
        print(f"Failed to load analog_data: {e}")
        data['analog_data'] = pd.DataFrame()
    
    # Flow meter data
    try:
        data['flow_meter'] = load(olfactometer_reader.Flowmeter, root/"Olfactometer0")
        vprint(verbose, "Loaded flow_meter")
    except Exception as e:
        print(f"Failed to load flow_meter: {e}")
        data['flow_meter'] = pd.DataFrame()
    
    # Video data
    try:
        video_reader = Video()
        data['video_reader'] = video_reader
        data['video_data'] = load_video(video_reader, root/"VideoData")
        vprint(verbose, "Loaded video_data")
    except Exception as e:
        print(f"Failed to load video_data: {e}")
        data['video_reader'] = None
        data['video_data'] = pd.DataFrame()
    
    # Pulse supply (reward delivery)
    try:
        data['pulse_supply_1'] = load(behavior_reader.PulseSupplyPort1, root/"Behavior")
        vprint(verbose, "Loaded pulse_supply_1")
    except Exception as e:
        print(f"Failed to load pulse_supply_1: {e}")
        data['pulse_supply_1'] = pd.DataFrame()
    
    try:
        data['pulse_supply_2'] = load(behavior_reader.PulseSupplyPort2, root/"Behavior")
        vprint(verbose, "Loaded pulse_supply_2")
    except Exception as e:
        print(f"Failed to load pulse_supply_2: {e}")
        data['pulse_supply_2'] = pd.DataFrame()
    
    # Create combined odour LED signal
    try:
        if not data['output_clear'].empty and not data['output_set'].empty:
            data['odour_led'] = concat_digi_events(data['output_clear']['DOPort0'], data['output_set']['DOPort0'])
            vprint(verbose, "Created odour_led")
        else:
            data['odour_led'] = pd.Series()
            print("Could not create odour_led (missing output data)")
    except Exception as e:
        print(f"Failed to create odour_led: {e}")
        data['odour_led'] = pd.Series()
    
    # Store readers for later use
    data['behavior_reader'] = behavior_reader
    data['olfactometer_reader'] = olfactometer_reader
    
    if apply_corrections and real_time_offset != pd.Timedelta(0):
        vprint(verbose, "\nApplying time corrections to all data streams...")
        
        time_indexed_streams = [
            'digital_input_data', 'output_set', 'output_clear',
            'olfactometer_valves_0', 'olfactometer_valves_1', 
            'olfactometer_end_0', 'analog_data', 'flow_meter',
            'video_data', 'pulse_supply_1', 'pulse_supply_2', 
            'odour_led'
        ]
        
        
        for stream_name in time_indexed_streams:
            if stream_name in data and not data[stream_name].empty:
                try:
                    if isinstance(data[stream_name], pd.DataFrame):
                        # Check if index is datetime-like
                        if hasattr(data[stream_name].index, 'dtype') and pd.api.types.is_datetime64_any_dtype(data[stream_name].index):
                            data[stream_name].index = data[stream_name].index + real_time_offset
                            vprint(verbose, f"Applied correction to {stream_name}")
                        else:
                            print(f"Skipped {stream_name} (not datetime index)")
                            
                    elif isinstance(data[stream_name], pd.Series):
                        # Check if index is datetime-like
                        if hasattr(data[stream_name].index, 'dtype') and pd.api.types.is_datetime64_any_dtype(data[stream_name].index):
                            data[stream_name].index = data[stream_name].index + real_time_offset
                            vprint(verbose, f"Applied correction to {stream_name}")
                        else:
                            print(f"Skipped {stream_name} (not datetime index)")
                except Exception as e:
                    print(f"Failed to apply correction to {stream_name}: {e}")
    


    vprint(verbose, f"\nData loading complete! Loaded {len([k for k, v in data.items() if not (isinstance(v, pd.DataFrame) and v.empty) and not (isinstance(v, pd.Series) and v.empty)])} streams successfully.")
    
    return data

def load_experiment_events(root, *args, verbose: bool = True, **kwargs):
    """
    Load and process experiment events with automatic time synchronization
    matching load_all_streams() timing corrections
    """
    
    vprint(verbose, "Loading experiment events...")
    
    # === LOAD TIMING DATA ===
    try:
        behavior_reader = harp.create_reader(str(BEHAVIOR_SCHEMA_PATH), epoch=harp.REFERENCE_EPOCH)
        heartbeat = load(behavior_reader.TimestampSeconds, root/"Behavior")
        if not heartbeat.empty:
            heartbeat.reset_index(inplace=True)
        vprint(verbose, "Loaded heartbeat data for timing synchronization")
    except Exception as e:
        print(f"Failed to load heartbeat: {e}")
        heartbeat = pd.DataFrame(columns=['Time', 'TimestampSeconds'])
    
    # Calculate real-time offset (same logic as load_all_streams)
    real_time_offset = pd.Timedelta(0)
    if not heartbeat.empty and 'Time' in heartbeat.columns and len(heartbeat) > 0:
        try:
            # Extract timestamp from root folder name
            real_time_str = root.name
            match = re.search(r'\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}', real_time_str)
            if not match:
                real_time_str = root.parent.name
                match = re.search(r'\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}', real_time_str)
            
            if match:
                real_time_str = match.group(0)
                real_time_ref_utc = datetime.strptime(real_time_str, '%Y-%m-%dT%H-%M-%S')
                real_time_ref_utc = real_time_ref_utc.replace(tzinfo=timezone.utc)
                uk_tz = zoneinfo.ZoneInfo("Europe/London")
                real_time_ref = real_time_ref_utc.astimezone(uk_tz)
                
                start_time_hardware = heartbeat['Time'].iloc[0]
                start_time_dt = start_time_hardware.to_pydatetime()
                if start_time_dt.tzinfo is None:
                    start_time_dt = start_time_dt.replace(tzinfo=uk_tz)
                real_time_offset = real_time_ref - start_time_dt
                vprint(verbose, f"Calculated real-time offset: {real_time_offset}")
        except Exception as e:
            print(f"Error calculating real-time offset: {e}")
    
    # Create timestamp interpolation mapping (same as load_all_streams)
    timestamp_to_time = pd.Series()
    interpolate_time = None
    if not heartbeat.empty and 'Time' in heartbeat.columns and 'TimestampSeconds' in heartbeat.columns:
        heartbeat['Time'] = pd.to_datetime(heartbeat['Time'], errors='coerce')
        timestamp_to_time = pd.Series(data=heartbeat['Time'].values, index=heartbeat['TimestampSeconds'])
        
        def interpolate_time(seconds):
            """Interpolate timestamps from seconds, with safety checks"""
            if timestamp_to_time.empty:
                return pd.NaT
            int_seconds = int(seconds)
            fractional_seconds = seconds % 1
            if int_seconds in timestamp_to_time.index:
                base_time = timestamp_to_time.loc[int_seconds]
                return base_time + pd.to_timedelta(fractional_seconds, unit='s')
            return pd.NaT
        
        vprint(verbose, "Created timestamp interpolation mapping")
    
    # === LOAD EXPERIMENT EVENTS ===
    event_types = {
        'initiation_sequence': [],
        'end_initiation': [],
        'await_reward': [],
        'reset': [],
        'choose_random_sequence': [],
        'sample_reward_condition': []
    }
    
    experiment_events_dir = root / "ExperimentEvents"
    
    if not experiment_events_dir.exists():
        print("No ExperimentEvents directory found")
        return {f'combined_{event_type}_df': pd.DataFrame() for event_type in event_types.keys()}
    
    csv_files = list(experiment_events_dir.glob("*.csv"))
    vprint(verbose, f"Found {len(csv_files)} experiment event files")
    
    # Process each CSV file
    for csv_file in csv_files:
        try:
            ev_df = pd.read_csv(csv_file)
            vprint(verbose, f"Processing event file: {csv_file.name} with {len(ev_df)} rows")
            
            # Handle timestamp conversion (same logic as original notebook)
            if "Seconds" in ev_df.columns and interpolate_time is not None:
                ev_df = ev_df.sort_values("Seconds")
                ev_df["Time"] = ev_df["Seconds"].apply(interpolate_time)
                vprint(verbose, "Using Seconds column with interpolation")
            else:
                # Fallback: use seconds as relative time
                ev_df["Time"] = pd.to_datetime(ev_df["Seconds"], unit='s')
                vprint(verbose, "Using Seconds column as raw timestamp")
            
            # Apply real-time offset (CRITICAL for synchronization)
            if real_time_offset != pd.Timedelta(0):
                ev_df["Time"] = ev_df["Time"] + real_time_offset
                vprint(verbose, f"Applied real-time offset: {real_time_offset}")
            
            # Extract events
            if "Value" in ev_df.columns:
                vprint(verbose, f"Found Value column with values: {ev_df['Value'].unique()}")
                
                event_mappings = {
                    'EndInitiation': 'end_initiation',
                    'InitiationSequence': 'initiation_sequence', 
                    'Reset': 'reset',
                    'AwaitReward': 'await_reward',
                    'SampleRewardCondition': 'sample_reward_condition',
                    'ChooseRandomSequence': 'choose_random_sequence'
                }
                
                for event_value, event_key in event_mappings.items():
                    event_df = ev_df[ev_df["Value"] == event_value].copy()
                    if not event_df.empty:
                        vprint(verbose, f"Found {len(event_df)} {event_value} events")
                        event_df[event_value] = True
                        event_types[event_key].append(event_df[["Time", event_value]])
                        
        except Exception as e:
            print(f"Error processing event file {csv_file.name}: {e}")
    
    # Combine events into final DataFrames
    results = {}
    event_name_mapping = {
        'end_initiation': 'EndInitiation',
        'initiation_sequence': 'InitiationSequence',
        'reset': 'Reset', 
        'await_reward': 'AwaitReward',
        'sample_reward_condition': 'SampleRewardCondition',
        'choose_random_sequence': 'ChooseRandomSequence'
    }
    
    for event_key, frames_list in event_types.items():
        df_name = f'combined_{event_key}_df'
        column_name = event_name_mapping[event_key]
        
        if len(frames_list) > 0:
            combined_df = pd.concat(frames_list, ignore_index=True)
            combined_df.reset_index(drop=True, inplace=True)
            # Sort by time for proper chronological order
            combined_df = combined_df.sort_values('Time').reset_index(drop=True)
            results[df_name] = combined_df
            vprint(verbose, f"Combined {len(combined_df)} {column_name} events")
        else:
            results[df_name] = pd.DataFrame(columns=["Time", column_name])
            print(f"No {column_name} events found")
    
    print(f"Experiment events loading complete! All events synchronized with load_all_streams timing.")
    return results


def load_odor_mapping(root, *, data=None, verbose: bool = True, **kwargs):
    """
    Load odor mapping from session settings
    
    Parameters:
    -----------
    root : Path
        Experiment root directory
    data : dict, optional
        Data dictionary from load_all_streams() containing valve data
        If None, will load valve data internally
    
    Returns:
    --------
    dict containing odor mapping information
    """
    
    vprint(verbose, "Loading odor mapping from session settings...")
    
    # Get valve data
    if data is not None:
        olfactometer_valves_0 = data.get('olfactometer_valves_0', pd.DataFrame())
        olfactometer_valves_1 = data.get('olfactometer_valves_1', pd.DataFrame())
    else:
        # Load valve data if not provided
        try:
            olfactometer_reader = harp.create_reader(str(OLFACTOMETER_SCHEMA_PATH), epoch=harp.REFERENCE_EPOCH)
            olfactometer_valves_0 = load(olfactometer_reader.OdorValveState, root/"Olfactometer0")
            olfactometer_valves_1 = load(olfactometer_reader.OdorValveState, root/"Olfactometer1")
        except Exception as e:
            print(f"Could not load valve data: {e}")
            olfactometer_valves_0 = pd.DataFrame()
            olfactometer_valves_1 = pd.DataFrame()
    
    # Create valve data dictionary
    olfactometer_valves = {
        0: olfactometer_valves_0,
        1: olfactometer_valves_1,
    }
    
    try:
        # Load session settings (experiment-specific configuration)
        session_settings, session_schema = detect_settings.detect_settings(root)
        vprint(verbose, "Loaded session settings")
        
        # Extract valve configurations for each olfactometer
        olfactometer_commands = session_settings.metadata.iloc[0].olfactometerCommands
        olf_valves0 = [cmd.valvesOpenO0 for cmd in olfactometer_commands]
        olf_valves1 = [cmd.valvesOpenO1 for cmd in olfactometer_commands]
        
        vprint(verbose, f"Found {len(olf_valves0)} valve configurations for olfactometer 0")
        vprint(verbose, f"Found {len(olf_valves1)} valve configurations for olfactometer 1")
        
        # Create command index mapping (valve number -> command index)
        olf_command_idx = {}
        
        # Map olfactometer 0 valves (0-3) to command indices
        for val in range(4):
            try:
                cmd_idx = next(i for i, lst in enumerate(olf_valves0) if val in lst)
                olf_command_idx[f'0{val}'] = cmd_idx
            except StopIteration:
                print(f"Warning: Valve {val} not found in olfactometer 0 configuration")
        
        # Map olfactometer 1 valves (0-3) to command indices  
        for val in range(4):
            try:
                cmd_idx = next(i for i, lst in enumerate(olf_valves1) if val in lst)
                olf_command_idx[f'1{val}'] = cmd_idx
            except StopIteration:
                print(f"Warning: Valve {val} not found in olfactometer 1 configuration")
        
        vprint(verbose, f"Created valve-to-command mapping: {olf_command_idx}")
        
        # Create odor name mapping
        odour_to_olfactometer_map = [[] for _ in range(len(olfactometer_valves))]
        
        for valve_key, cmd_idx in olf_command_idx.items():
            olf_id = int(valve_key[0])  # Extract olfactometer ID (0 or 1)
            odor_name = olfactometer_commands[cmd_idx].name
            odour_to_olfactometer_map[olf_id].append(odor_name)
        
        vprint(verbose, f"Created odor mapping: {odour_to_olfactometer_map}")
        
        # Create reverse mapping: valve -> odor name
        valve_to_odor = {}
        for valve_key, cmd_idx in olf_command_idx.items():
            odor_name = olfactometer_commands[cmd_idx].name
            valve_to_odor[valve_key] = odor_name
        
        # Create olfactometer -> odor list mapping
        olfactometer_to_odors = {}
        for olf_id in range(len(olfactometer_valves)):
            olfactometer_to_odors[olf_id] = odour_to_olfactometer_map[olf_id]
        print("Odor mapping loaded successfully")

        return {
            'olfactometer_valves': olfactometer_valves,
            'session_settings': session_settings,
            'session_schema': session_schema,
            'olf_valves0': olf_valves0,
            'olf_valves1': olf_valves1,
            'olf_command_idx': olf_command_idx,
            'odour_to_olfactometer_map': odour_to_olfactometer_map,
            'valve_to_odor': valve_to_odor,
            'olfactometer_to_odors': olfactometer_to_odors
        }
        
    except Exception as e:
        print(f"Error loading odor mapping: {e}")
        return {
            'olfactometer_valves': olfactometer_valves,
            'session_settings': None,
            'session_schema': None,
            'olf_valves0': [],
            'olf_valves1': [],
            'olf_command_idx': {},
            'odour_to_olfactometer_map': [[], []],
            'valve_to_odor': {},
            'olfactometer_to_odors': {0: [], 1: []}
        }


# --- trial-data table loaders (moved from visualization/visualization_utils.py) -----------
# High-level readers over a session's ``saved_analysis_results`` directory. They live in io so
# the modelling and visualization layers each depend on io for their data, rather than on one
# another. ``_odor_to_letter`` decodes a stored odor token and travels with them because the
# same callers use it alongside the views.


def _load_table_with_trial_data(results_dir: Path, name: str) -> pd.DataFrame:
    """Load trial_data (parquet->csv) or a saved CSV table by name, using cache if available."""
    # Try cache for trial_data only
    if name == "trial_data":
        # Extract subjid and date from results_dir path
        # Expect path: .../sub-XXX_id-YYY/ses-*_date-ZZZZ/saved_analysis_results
        parts = results_dir.parts
        try:
            subj_part = [p for p in parts if p.startswith("sub-")][0]
            subjid = int(subj_part.split("-")[1])
            ses_part = [p for p in parts if p.startswith("ses-")][0]
            date_str = ses_part.split("_date-")[-1]
            date = int(date_str) if date_str.isdigit() else date_str
        except Exception:
            subjid = None
            date = None
        if subjid is not None and date is not None:
            cached_td = _get_from_cache(subjid, date, kind="trial_data")
            if cached_td is not None:
                print(f"[CACHE HIT] trial_data for {subjid}, {date}")
                return cached_td
        # Fallback to disk and cache once loaded
        df = None
        pq = results_dir / "trial_data.parquet"
        if pq.exists():
            try:
                df = pd.read_parquet(pq)
            except Exception:
                df = None
        if df is None:
            tcsv = results_dir / "trial_data.csv"
            if tcsv.exists():
                try:
                    df = pd.read_csv(tcsv)
                except Exception:
                    df = None
        if df is None:
            return pd.DataFrame()
        if subjid is not None and date is not None:
            _update_cache(subjid, [date], {date: df}, kind="trial_data")
        return df

    # Only these three non-initiated tables are loadable here. Parquet first, CSV second:
    # since Phase 7b.3 the CSV is written only when `save_csv=True`, so a CSV-only reader
    # would return an empty frame -- with no error -- for every session saved with the
    # default. Parquet is always written.
    allowed = {"non_initiated_sequences", "non_initiated_odor1_attempts", "non_initiated_FA"}
    if name in allowed:
        for path, reader in ((results_dir / f"{name}.parquet", pd.read_parquet),
                             (results_dir / f"{name}.csv", pd.read_csv)):
            if path.exists():
                try:
                    return reader(path)
                except Exception:
                    continue
    return pd.DataFrame()


def _load_trial_views(results_dir: Path) -> dict[str, pd.DataFrame]:
    """Load trial_data once and derive commonly used slices for plots.

    Returns keys:
      - trial_data: full table
      - completed: is_aborted == False
      - rewarded / unrewarded / timeout: completed filtered by response_time_category
      - aborted: is_aborted == True
      - aborted_fa: aborted with fa_label != nFA (case-insensitive)
      - aborted_hr: aborted with hit_hidden_rule == True
    """
    td = _load_table_with_trial_data(results_dir, "trial_data")
    if td.empty:
        return {
            "trial_data": pd.DataFrame(),
            "completed": pd.DataFrame(),
            "rewarded": pd.DataFrame(),
            "unrewarded": pd.DataFrame(),
            "timeout": pd.DataFrame(),
            "aborted": pd.DataFrame(),
            "aborted_fa": pd.DataFrame(),
            "aborted_hr": pd.DataFrame(),
        }

    td = td.copy()
    # Normalize datetime columns we rely on
    for col in ["sequence_start", "sequence_end", "timestamp", "initiation_sequence_time", "abortion_time", "fa_time", "await_reward_time", "first_supply_time", "poke_window_end"]:
        if col in td.columns:
            td[col] = pd.to_datetime(td[col], errors="coerce")

    td["is_aborted"] = td.get("is_aborted", False).fillna(False)
    td["response_time_category"] = td.get("response_time_category", "").astype(str)

    completed = td[~td["is_aborted"]].copy()
    aborted = td[td["is_aborted"]].copy()

    rewarded = completed[completed["response_time_category"] == "rewarded"].copy()
    unrewarded = completed[completed["response_time_category"] == "unrewarded"].copy()
    timeout = completed[completed["response_time_category"] == "timeout_delayed"].copy()

    fa_mask = aborted.get("fa_label").astype(str).str.lower().ne("nfa") if "fa_label" in aborted.columns else pd.Series(False, index=aborted.index)
    aborted_fa = aborted[fa_mask].copy()

    aborted_hr = aborted[aborted.get("hit_hidden_rule", False) == True].copy()

    return {
        "trial_data": td,
        "completed": completed,
        "rewarded": rewarded,
        "unrewarded": unrewarded,
        "timeout": timeout,
        "aborted": aborted,
        "aborted_fa": aborted_fa,
        "aborted_hr": aborted_hr,
    }


def _odor_to_letter(value) -> str:
    """Normalize a stored odor token ('OdorC' / '"OdorC"' / 'odor c' / 'C') to a
    bare upper-case letter."""
    s = str(value).strip().strip('[]"\'').strip()
    if s.lower().startswith("odor"):
        s = s[4:].strip()
    return s.upper()

