import re
import sys
import argparse
import datetime
import pandas as pd
import numpy as np
from pathlib import Path
import zoneinfo
import hypnose_behavior.io.readers as utils
import harp
import yaml
from functools import reduce

def detect_stage(root):
    """
    Extracts stage information from session metadata.
    Returns stage_name and hidden_rule_indices (can be multiple).
    Tries metadata first (flexible key matching), then Schema files as fallback.
    """
    
    path_root = Path(root)
    metadata_reader = utils.SessionData()
    
    try:
        # Try loading from SessionSettings first
        session_settings = utils.load_json(metadata_reader, path_root/"SessionSettings")
        
        # Handle nested structure: data might be in 'value' key or have nested 'metadata'
        if isinstance(session_settings, pd.DataFrame) and len(session_settings) > 0:
            row = session_settings.iloc[0]
            
            # Check if metadata is nested in 'value' column
            if 'value' in row:
                settings_obj = row['value']
                metadata = settings_obj.get('metadata') if isinstance(settings_obj, dict) else getattr(settings_obj, 'metadata', None)
            else:
                metadata = row.get('metadata') if isinstance(row, dict) else getattr(row, 'metadata', None)
        else:
            metadata = session_settings.iloc[0]['metadata']
        
        stage_name = None
        hidden_rule_indices = []
        
        # The metadata object itself contains another 'metadata' field with initialSequence
        # Try to access metadata.metadata.initialSequence
        inner_metadata = None
        for key in ['metadata', 'Metadata']:
            if hasattr(metadata, key):
                inner_metadata = getattr(metadata, key)
                break
            if isinstance(metadata, dict) and key in metadata:
                inner_metadata = metadata[key]
                break
        
        # Try to extract from initialSequence
        initial_sequence = None
        
        # First try inner_metadata if it exists
        if inner_metadata:
            for key in ['initialSequence', 'initial_sequence', 'InitialSequence']:
                if hasattr(inner_metadata, key):
                    initial_sequence = getattr(inner_metadata, key)
                    break
                if isinstance(inner_metadata, dict) and key in inner_metadata:
                    initial_sequence = inner_metadata[key]
                    break
        
        # Also try direct metadata (fallback)
        if not initial_sequence:
            for key in ['initialSequence', 'initial_sequence', 'InitialSequence']:
                if hasattr(metadata, key):
                    initial_sequence = getattr(metadata, key)
                    break
                if isinstance(metadata, dict) and key in metadata:
                    initial_sequence = metadata[key]
                    break
        
        if initial_sequence:
            # Extract filename
            filename = initial_sequence.split('/')[-1]
            stage_name = filename.replace('.yml', '').replace('.yaml', '')
            
            # Extract location indices from pattern: -location0123 → [0, 1, 2, 3]
            match = re.search(r'(?:[-_])?location(\d+)', filename, re.IGNORECASE)
            if match:
                location_str = match.group(1)
                # If it's a single digit or multi-digit number
                hidden_rule_indices = [int(d) for d in location_str]
            
            print(f"Detected stage from metadata: {stage_name}")
            if hidden_rule_indices:
                print(f"Hidden rule indices: {hidden_rule_indices}")
            else:
                print("No hidden rule indices found")
            
            return {
                'stage_name': stage_name,
                'hidden_rule_indices': hidden_rule_indices
            }
        
        # Fallback: Try to extract from sequences if initialSequence didn't work
        sequences = None
        for key in ['sequences', 'Sequences', 'sequence', 'Sequence']:
            if hasattr(metadata, key):
                sequences = getattr(metadata, key)
                break
            if isinstance(metadata, dict) and key in metadata:
                sequences = metadata[key]
                break
        
        if sequences and isinstance(sequences, list) and len(sequences) > 0:
            # Navigate through the nested structure to find the first 'name'
            for seq_group in sequences:
                if isinstance(seq_group, list):
                    for seq in seq_group:
                        if isinstance(seq, dict) and 'name' in seq:
                            stage_name = seq['name']
                            break
                elif isinstance(seq_group, dict) and 'name' in seq_group:
                    stage_name = seq_group['name']
                    break
                
                if stage_name:
                    break
            
            # Extract hidden rule indices from sequence name
            if stage_name:
                match = re.search(r'(?:[-_])?location(\d+)', stage_name, re.IGNORECASE)
                if match:
                    location_str = match.group(1)
                    hidden_rule_indices = [int(d) for d in location_str]
                
                print(f"Detected stage from sequences: {stage_name}")
                if hidden_rule_indices:
                    print(f"Hidden rule indices: {hidden_rule_indices}")
                else:
                    print("No hidden rule indices found")
                
                return {
                    'stage_name': stage_name,
                    'hidden_rule_indices': hidden_rule_indices
                }
        
        # If we get here, metadata didn't have what we need
        print("Could not extract stage from metadata, falling back to Schema files")
        
    except Exception as e:
        print(f"Warning: Could not load metadata ({e}), falling back to Schema files")
        import traceback
        traceback.print_exc()
    
    # ============ Fallback: Load from Schema files ============
    try:
        path_root = Path(root)
        
        # Try loading Schema JSON first
        try:
            metadata_reader = utils.SessionData()
            sequence_schema = utils.load_json(metadata_reader, path_root/"Schema") 
            sequence_metadata = sequence_schema['metadata'].iloc[0]
            sequences = sequence_metadata['sequences']
        except:
            # Fallback to YAML schema file
            session_settings = utils.load_json(metadata_reader, path_root/"SessionSettings")
            metadata = session_settings.iloc[0]['metadata']
            schema_filename = metadata.metadata.initialSequence.split("/")[-1]
            with open(path_root/"Schema"/schema_filename, 'r') as file:
                sequence_schema = yaml.load(file, Loader=yaml.SafeLoader)
                sequences = sequence_schema['sequences']
        
        stage_name = None
        hidden_rule_indices = []
        
        if isinstance(sequences, list) and len(sequences) > 0:
            for seq_group in sequences:
                if isinstance(seq_group, list):
                    for seq in seq_group:
                        if isinstance(seq, dict) and 'name' in seq:
                            stage_name = seq['name']
                            break
                elif isinstance(seq_group, dict) and 'name' in seq_group:
                    stage_name = seq_group['name']
                    break
                
                if stage_name:
                    break
        
        if stage_name:
            match = re.search(r'(?:[-_])?location(\d+)', stage_name, re.IGNORECASE)
            if match:
                location_str = match.group(1)
                hidden_rule_indices = [int(d) for d in location_str]
            
            print(f"Detected stage from Schema: {stage_name}")
            if hidden_rule_indices:
                print(f"Hidden rule indices: {hidden_rule_indices}")
            else:
                print("No hidden rule indices found")
            
            return {
                'stage_name': stage_name,
                'hidden_rule_indices': hidden_rule_indices
            }
        else:
            print("No stage name found in Schema")
            return {
                'stage_name': "Unknown",
                'hidden_rule_indices': []
            }
    
    except Exception as e:
        print(f"Error detecting stage from Schema: {e}")
        import traceback
        traceback.print_exc()
        return {
            'stage_name': "Unknown", 
            'hidden_rule_indices': []
        }

if __name__ == "__main__":
    # Deal with inputs
    parser = argparse.ArgumentParser(description="Get stage of a behavioral session")
    parser.add_argument("session_folder", help="Path to the session folder (e.g., sub-XXX/ses-YYY_date-YYYYMMDD)")
    
    args = parser.parse_args()
    
    # Check session folder structure
    session_folder = Path(args.session_folder)
    behav_folder = session_folder / "behav"
    
    if not behav_folder.exists():
        print(f"No behavior folder found at {behav_folder}")
        sys.exit(1)
    
    # Collect all session directories
    session_dirs = [d for d in behav_folder.iterdir() if d.is_dir()]
    print(f"Found {len(session_dirs)} behavioral sessions in {behav_folder}")

    # Get stage for each session directory
    for session_dir in sorted(session_dirs):
        print(f"\nProcessing session: {session_dir.name}")
        
         # Only process if session settings exist
        if not (session_dir / "SessionSettings").exists():
            print(f"No SessionSettings found in {session_dir.name}, skipping")
            continue

        try:
            stage_info = detect_stage(session_dir)
            print(f"Stage: {stage_info['stage_name']}")
            if stage_info['hidden_rule_index'] is not None:
                print(f"Hidden rule index: {stage_info['hidden_rule_index']}")
        except Exception as e:
            print(f"Something went wrong with {session_dir}: {e}. Continuing to next session.")
            continue