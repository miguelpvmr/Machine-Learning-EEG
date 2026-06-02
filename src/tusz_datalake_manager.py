import re
import shutil
import warnings
import numpy as np
import pandas as pd
import mne
from pathlib import Path
from tqdm.auto import tqdm

# --- GLOBAL CONFIGURATION ---
mne.set_log_level('ERROR')
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ==========================================
# --- PRIVATE HELPERS (INTERNAL LOGIC) ---
# ==========================================

def _get_clean_channel_list(raw_ch_names):
    """Standardizes channel names to verify 10-20 system requirements."""
    clean_mapping = [
        ch.replace('EEG ', '').replace('-REF', '').replace('-LE', '').strip().upper() 
        for ch in raw_ch_names
    ]
    synonyms = {'T7': 'T3', 'T8': 'T4', 'P7': 'T5', 'P8': 'T6'}
    return [synonyms.get(ch, ch) for ch in clean_mapping]

def _extract_file_metadata(edf_path, split_type):
    """Performs binary header reading and technical validation for EDF files."""
    try:
        raw = mne.io.read_raw_edf(edf_path, preload=False, verbose=False)
        clean_ch_names = _get_clean_channel_list(raw.ch_names)
        
        # Technical criteria: 19 channels + Midline sensors (Fz/Pz)
        has_min_channels = raw.info['nchan'] >= 19
        has_midline = 'FZ' in clean_ch_names and 'PZ' in clean_ch_names
        is_tech_ok = has_min_channels and has_midline

        filename_parts = edf_path.stem.split('_')
        p_id = filename_parts[0]
        s_id = filename_parts[1] if len(filename_parts) > 1 else "s000"
        t_id = filename_parts[2] if len(filename_parts) > 2 else "t000"

        with open(edf_path, 'rb') as f:
            header_bytes = f.read(256).decode('ascii', errors='ignore')
            patient_field = header_bytes[8:88].strip()

        age_match = re.search(r'Age:(\d+)', patient_field, re.IGNORECASE)
        age = int(age_match.group(1)) if age_match else None
        config = edf_path.parent.name

        metadata = {
            "patient_id": p_id,
            "original_split": split_type,
            "age_raw": age,
            "gender_raw": int(raw.info['subject_info']['sex']) if raw.info['subject_info'] and raw.info['subject_info']['sex'] else 0,
            "session_id": s_id,
            "partition_id": t_id,
            "config_type": config, 
            "file_name": edf_path.name,
            "sfreq": float(raw.info['sfreq']),
            "n_channels": int(raw.info['nchan']),
            "duration_sec": round(raw.n_times / raw.info['sfreq'], 2),
            "highpass": float(raw.info['highpass']),
            "lowpass": float(raw.info['lowpass']),
            "date": str(raw.info['meas_date']).split(' ')[0] if raw.info['meas_date'] else None,
            "is_tech_ok": is_tech_ok 
        }
        raw.close()
        return metadata
    except Exception:
        return None

def _consolidate_patient_data(df_raw):
    """Aggregates all scanned partitions into unique patient profiles."""
    patient_list = []
    for p_id, group in df_raw.groupby('patient_id'):
        df_valid = group[group['is_tech_ok'] == True]
        source = df_valid if not df_valid.empty else group
        
        final_age = source['age_raw'].mode()[0] if not source['age_raw'].mode().empty else None
        final_gender = source['gender_raw'].mode()[0] if not source['gender_raw'].mode().empty else 0
        orig_split = group['original_split'].iloc[0]
        
        patient_list.append({
            "patient_id": p_id,
            "original_split": orig_split,
            "age": final_age,
            "gender": final_gender,
            "n_sessions": df_valid['session_id'].nunique() if not df_valid.empty else 0,
            "total_duration_sec": round(df_valid['duration_sec'].sum(), 2) if not df_valid.empty else 0.0,
            "has_valid_data": not df_valid.empty
        })
    return pd.DataFrame(patient_list)

def _flatten_configuration_folders(base_path, pattern="*tcp_*"):
    """Eliminates intermediate configuration subdirectories (e.g., 01_tcp_ar)."""
    config_dirs = [d for d in base_path.rglob(pattern) if d.is_dir()]
    for config_dir in tqdm(config_dirs, desc="Flattening configs"):
        session_dir = config_dir.parent
        for item in config_dir.iterdir():
            if item.is_file():
                try:
                    shutil.move(str(item), str(session_dir / item.name))
                except Exception:
                    pass
        try:
            config_dir.rmdir()
        except OSError:
            pass

def _consolidate_and_rename_split(split_dir, base_path, id_mapping):
    """Moves and renames patient directories to their numeric identifiers."""
    patient_folders = [d for d in split_dir.iterdir() if d.is_dir()]
    for patient_dir in tqdm(patient_folders, desc=f"Consolidating {split_dir.name}"):
        old_name = patient_dir.name
        new_name = str(id_mapping.get(old_name, old_name))
        target_path = base_path / new_name
        
        if not target_path.exists():
            shutil.move(str(patient_dir), str(target_path))
        else:
            for session_dir in patient_dir.iterdir():
                if session_dir.is_dir():
                    target_session = target_path / session_dir.name
                    if not target_session.exists():
                        shutil.move(str(session_dir), str(target_session))
            try:
                shutil.rmtree(str(patient_dir))
            except Exception:
                pass

# ==========================================
# --- PUBLIC INTERFACE (EXECUTION) ---
# ==========================================

def create_metadata(base_path, metadata_output_path, train_ratio=0.7, seed=42):
    """
    Generates relational metadata with numeric IDs and hierarchical splits.
    Includes split_final in both Patient and Session tables for efficiency.
    """
    base_path = Path(base_path)
    metadata_output_path = Path(metadata_output_path)
    metadata_output_path.mkdir(parents=True, exist_ok=True)

    all_records = []
    for s in ["train", "dev", "eval"]:
        folder = base_path / s
        if not folder.exists(): continue
        files = list(folder.rglob("*.[eE][dD][fF]"))
        for f in tqdm(files, desc=f"Scanning {s.upper()}"):
            res = _extract_file_metadata(f, s)
            if res: all_records.append(res)

    if not all_records:
        return None, None

    df_master = pd.DataFrame(all_records)
    total_patients_scanned = df_master['patient_id'].nunique()
    df_patients = _consolidate_patient_data(df_master)
    
    # --- Split Logic ---
    np.random.seed(seed)
    valid_p_ids = df_patients[df_patients['has_valid_data']]['patient_id'].unique()
    np.random.shuffle(valid_p_ids)
    
    cutoff = int(len(valid_p_ids) * train_ratio)
    train_ids = set(valid_p_ids[:cutoff])
    
    def assign_split(row):
        if not row['has_valid_data']: return -1
        return 0 if row['patient_id'] in train_ids else 1

    df_patients['split_final'] = df_patients.apply(assign_split, axis=1)
    df_patients['patient_num_id'] = pd.factorize(df_patients['patient_id'])[0] + 1
    
    # --- Exports Preparation ---
    id_map = dict(zip(df_patients['patient_id'], df_patients['patient_num_id']))
    split_map = dict(zip(df_patients['patient_id'], df_patients['split_final']))
    
    # Table 1: Patients
    cols_p = ['patient_num_id', 'patient_id', 'original_split', 'age', 'gender', 'n_sessions', 'total_duration_sec', 'split_final']
    df_patients_export = df_patients[cols_p]

    # Table 2: Sessions (Including split_final column)
    df_sessions = df_master.copy()
    df_sessions['patient_num_id'] = df_sessions['patient_id'].map(id_map)
    
    def tag_session_split(row):
        if not row['is_tech_ok']: return -1
        return split_map.get(row['patient_id'], -1)

    df_sessions['split_final'] = df_sessions.apply(tag_session_split, axis=1)
    
    cols_s = ['patient_num_id', 'session_id', 'partition_id', 'config_type', 'file_name', 
              'sfreq', 'n_channels', 'duration_sec', 'highpass', 'lowpass', 'date', 'split_final']
    df_sessions_export = df_sessions[cols_s]

    # Save to Parquet
    df_patients_export.to_parquet(metadata_output_path / "metadata_patients.parquet", index=False)
    df_sessions_export.to_parquet(metadata_output_path / "metadata_sessions.parquet", index=False)

    # --- AUDIT REPORT ---
    train_stats = df_patients_export[df_patients_export['split_final'] == 0]
    test_stats = df_patients_export[df_patients_export['split_final'] == 1]
    excluded_partitions = len(df_master) - df_master['is_tech_ok'].sum()
    
    print("\n" + "="*60)
    print(f"TUSZ DATALAKE AUDIT REPORT (Seed: {seed} | Ratio: {train_ratio:.2f})")
    print("="*60)
    print(f"Total Partitions Scanned:      {len(df_master)}")
    print(f"Partitions Excluded (-1):      {excluded_partitions}")
    print(f"Partitions Retained (0 or 1):  {df_master['is_tech_ok'].sum()}")
    print("-" * 60)
    print(f"Total Unique Patients Found:   {total_patients_scanned}")
    print(f"Unique Patients Retained:      {len(df_patients_export[df_patients_export['split_final'] >= 0])}")
    print(f"Total Usable Signal Duration:  {df_patients_export['total_duration_sec'].sum()/3600:.2f} hours")
    print("-" * 60)
    print(f"SPLIT DISTRIBUTION ({int(train_ratio*100)}/{int((1-train_ratio)*100)})")
    print(f"TRAIN (0): {len(train_stats)} patients | {train_stats['total_duration_sec'].sum()/3600:.2f} hours")
    print(f"TEST  (1): {len(test_stats)} patients | {test_stats['total_duration_sec'].sum()/3600:.2f} hours")
    print("="*60 + "\n")

    return df_patients_export, df_sessions_export

def clean_bi_files(base_path):
    """Physically removes .csv_bi artifacts from the DataLake."""
    target_files = list(set(Path(base_path).rglob("*.csv_bi")))
    for f in tqdm(target_files, desc="Cleaning artifacts"):
        try:
            f.unlink()
        except Exception:
            pass

def restructure_directory(base_path, df_patients):
    """Consolidates the directory structure and applies numeric patient IDs."""
    base_path = Path(base_path)
    id_mapping = dict(zip(df_patients['patient_id'], df_patients['patient_num_id']))
    
    _flatten_configuration_folders(base_path)
    for split_name in ["train", "dev", "eval"]:
        split_dir = base_path / split_name
        if not split_dir.exists(): continue
        _consolidate_and_rename_split(split_dir, base_path, id_mapping)
        try:
            split_dir.rmdir()
        except OSError:
            pass
    print("Datalake physical restructuring completed.")