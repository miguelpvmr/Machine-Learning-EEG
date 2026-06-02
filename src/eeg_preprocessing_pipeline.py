import mne
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from tqdm.auto import tqdm
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings

# --- GLOBAL CONFIGURATION ---
mne.set_log_level('ERROR')
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ==========================================
# --- PRIVATE HELPERS (INTERNAL LOGIC) ---
# ==========================================

def _normalize_channels(raw, sfreq_target=250.0):
    """Standardizes channel names to 10-20 system and resamples the signal."""
    clean_map = {ch: ch.replace('EEG ', '').replace('-REF', '').replace('-LE', '').strip().upper() 
                 for ch in raw.ch_names}
    
    synonyms = {'T7': 'T3', 'T8': 'T4', 'P7': 'T5', 'P8': 'T6'}
    standard = {
        'FP1': 'Fp1', 'FP2': 'Fp2', 'F3': 'F3', 'F4': 'F4', 'C3': 'C3', 'C4': 'C4',
        'P3': 'P3', 'P4': 'P4', 'O1': 'O1', 'O2': 'O2', 'F7': 'F7', 'F8': 'F8',
        'T3': 'T3', 'T4': 'T4', 'T5': 'T5', 'T6': 'T6', 'FZ': 'Fz', 'CZ': 'Cz', 'PZ': 'Pz'
    }

    final_rename = {}
    for old, cln in clean_map.items():
        resolved = synonyms.get(cln, cln)
        fin = standard.get(resolved, resolved)
        if fin not in final_rename.values():
            final_rename[old] = fin
        else:
            final_rename[old] = f"DEL_{old}"

    raw.rename_channels(final_rename, verbose=False)
    existing = [ch for ch in standard.values() if ch in raw.ch_names]
    raw.pick(existing, verbose=False)
    
    if raw.info['sfreq'] != sfreq_target:
        raw.resample(sfreq_target, n_jobs=1, verbose=False)
    return raw

def _apply_clinical_filters(raw):
    """Applies zero-phase 60Hz Notch and 0.5-70Hz Band-pass Butterworth filters."""
    raw.notch_filter(freqs=60.0, notch_widths=1.0, method='iir', phase='zero', verbose=False)
    raw.filter(l_freq=0.5, h_freq=70.0, method='iir', 
               iir_params=dict(order=4, ftype='butter', output='sos'), 
               phase='zero', n_jobs=1, verbose=False)
    return raw

def _get_expected_montage_channels():
    """Returns the expected 18 bipolar derivations."""
    return [
        'Fp1-F7', 'F7-T3', 'T3-T5', 'T5-O1',
        'Fp2-F8', 'F8-T4', 'T4-T6', 'T6-O2',
        'Fp1-F3', 'F3-C3', 'C3-P3', 'P3-O1',
        'Fp2-F4', 'F4-C4', 'C4-P4', 'P4-O2',
        'Fz-Cz', 'Cz-Pz'
    ]

def _apply_bipolar_montage(raw):
    """Transforms signal to Longitudinal Bipolar Montage (Double Banana) with explicit validation."""
    pairs = [
        ('Fp1', 'F7'), ('F7', 'T3'), ('T3', 'T5'), ('T5', 'O1'),
        ('Fp2', 'F8'), ('F8', 'T4'), ('T4', 'T6'), ('T6', 'O2'),
        ('Fp1', 'F3'), ('F3', 'C3'), ('C3', 'P3'), ('P3', 'O1'),
        ('Fp2', 'F4'), ('F4', 'C4'), ('C4', 'P4'), ('P4', 'O2'),
        ('Fz', 'Cz'), ('Cz', 'Pz')
    ]
    
    current_channels = raw.ch_names
    missing = [ch for p in pairs for ch in p if ch not in current_channels]
    
    if missing:
        raise ValueError(f"Missing required electrodes for montage: {set(missing)}")

    anodes, cathodes = [p[0] for p in pairs], [p[1] for p in pairs]
    names = [f"{p[0]}-{p[1]}" for p in pairs]
    
    return mne.set_bipolar_reference(raw, anode=anodes, cathode=cathodes, 
                                     ch_name=names, drop_refs=True, copy=False, verbose=False)

def _save_to_parquet_optimized(raw, output_path):
    """Saves EEG data as Float32 Parquet with Snappy compression."""
    data = raw.get_data()
    df = pd.DataFrame(data.T, columns=raw.ch_names).astype('float32')
    table = pa.Table.from_pandas(df)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, output_path, compression='snappy')

def _process_single_partition(task, raw_root, clean_root):
    """
    Isolated worker function for parallel processing.
    """
    try:
        patient_dir = raw_root / str(task['patient_num_id'])
        search_results = list(patient_dir.rglob(task['file_name']))
        
        if not search_results:
            return 0
        
        input_path = search_results[0]
        output_path = clean_root / str(task['patient_num_id']) / task['session_id'] / f"{task['partition_id']}.parquet"

        raw = mne.io.read_raw_edf(input_path, preload=True, verbose=False)
        raw = _normalize_channels(raw, sfreq_target=250.0)
        raw = _apply_clinical_filters(raw)
        raw = _apply_bipolar_montage(raw)
        
        _save_to_parquet_optimized(raw, output_path)
        raw.close()
        
        return 1
    except Exception:
        return 0

def _verify_parquet_schema(file_path, expected_duration_sec, target_sfreq=250.0):
    """Lightweight metadata verification for Audit Report."""
    try:
        parquet_meta = pq.read_metadata(file_path)
        schema = parquet_meta.schema.to_arrow_schema()
        
        file_channels = schema.names
        expected_channels = _get_expected_montage_channels()
        channels_match = set(file_channels) == set(expected_channels)
        
        num_samples = parquet_meta.num_rows
        if expected_duration_sec > 0:
            calculated_sfreq = num_samples / expected_duration_sec
            sfreq_match = abs(calculated_sfreq - target_sfreq) < (target_sfreq * 0.01)
        else:
            sfreq_match = False
            
        return channels_match, sfreq_match
    except Exception:
        return False, False

# ==========================================
# --- PUBLIC INTERFACE (EXECUTION) ---
# ==========================================

def run_cleaning_pipeline(df_sessions, raw_root, processed_base_path, n_jobs=-1):
    """
    Multiprocessing ingestion pipeline with native tqdm integration.
    Supports scikit-learn style n_jobs parameters.
    """
    raw_root = Path(raw_root)
    clean_root = Path(processed_base_path) / "03_TUSZ_Clean"
    clean_root.mkdir(parents=True, exist_ok=True)
    
    df_work = df_sessions[df_sessions['split_final'] >= 0].copy()
    
    tasks = df_work.to_dict('records')
    total_cores = multiprocessing.cpu_count()
    
    if n_jobs < 0:
        workers = max(1, total_cores + 1 + n_jobs)
    elif n_jobs == 0:
        workers = 1
    else:
        workers = n_jobs
        
    print(f"Initiating parallel processing on {len(tasks)} files using {workers} core(s)...")
    
    success_count = 0
    
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(_process_single_partition, task, raw_root, clean_root) for task in tasks]
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Signal Cleaning"):
            success_count += future.result()
            
    failed_count = len(tasks) - success_count

    print("\n" + "-" * 50)
    print("Parallel Pipeline Execution Summary")
    print(f"Destination:            {clean_root}")
    print(f"Successfully Processed: {success_count}")
    print(f"Failed/Missing:         {failed_count}")
    print("-" * 50)


def verify_clean_datalake(df_sessions, processed_base_path):
    """
    Cross-references the processed DataLake against the original metadata.
    Ensures structural integrity (channels and sfreq).
    """
    clean_root = Path(processed_base_path) / "03_TUSZ_Clean"
    
    audit_results = {
        "valid_and_verified": 0,
        "missing_but_expected": [],
        "present_but_excluded": [],
        "invalid_channels": [],
        "invalid_sfreq": []
    }
    
    for _, row in tqdm(df_sessions.iterrows(), total=len(df_sessions), desc="Verifying Tensors"):
        expected_path = clean_root / str(row['patient_num_id']) / row['session_id'] / f"{row['partition_id']}.parquet"
        
        is_expected = row['split_final'] >= 0
        file_exists = expected_path.exists()
        
        if is_expected and not file_exists:
            audit_results["missing_but_expected"].append(row['file_name'])
            continue
        elif not is_expected and file_exists:
            audit_results["present_but_excluded"].append(row['file_name'])
            continue
        elif not is_expected and not file_exists:
            continue
            
        channels_ok, sfreq_ok = _verify_parquet_schema(expected_path, row['duration_sec'])
        
        if not channels_ok:
            audit_results["invalid_channels"].append(row['file_name'])
        if not sfreq_ok:
            audit_results["invalid_sfreq"].append(row['file_name'])
            
        if channels_ok and sfreq_ok:
            audit_results["valid_and_verified"] += 1

    print("\n" + "=" * 60)
    print("DATALAKE INTEGRITY AUDIT REPORT")
    print("=" * 60)
    print(f"Perfect Files Verified (18 Ch & 250Hz): {audit_results['valid_and_verified']}")
    print("-" * 60)
    print("ANOMALIES DETECTED:")
    print(f"Missing (Expected by Metadata):         {len(audit_results['missing_but_expected'])}")
    print(f"Rogue Files (Excluded by Metadata):     {len(audit_results['present_but_excluded'])}")
    print(f"Invalid Channel Structure:              {len(audit_results['invalid_channels'])}")
    print(f"Invalid Sampling Frequency:             {len(audit_results['invalid_sfreq'])}")
    print("=" * 60)
    
    return audit_results