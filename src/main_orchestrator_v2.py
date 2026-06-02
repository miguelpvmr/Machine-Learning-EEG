import os
import gc 
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import tqdm
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Tuple, Any, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed

# Optimization: Limit multi-threading in sub-processes
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

try:
    from feature_engineering import extract_comprehensive_features
except ImportError:
    raise ImportError("Critical Error: 'feature_engineering.py' module not found.")

class FeatureExtractionPipeline:
    """
    High-performance EEG Pipeline.
    
    Logic: 
    - 15% Consensus Threshold (~0.614s ictal activity required).
    - Duration-Weighted Voting (The specific label with the most time wins).
    - Stride: 1s for Ictal (Augmentation) / 4s for Background.
    """

    def __init__(self, root_path: str):
        self.root = Path(root_path).resolve()
        self.path_raw = self.root / "01_Raw_Consolidated"
        self.path_clean = self.root / "03_TUSZ_Clean"
        self.path_metadata = self.root / "02_Metadata" / "metadata_patients.parquet"
        # Updated output path to reflect the new threshold version
        self.path_output = self.root / "04_TUSZ_Features_ML" / "version=v2_augmented_labels"

        self.sampling_rate = 250
        self.window_samples = 1024 # 4.096 seconds
        self.stride_ictal = 1.0    # Fixed at 1s for ictal augmentation
        self.stride_bckg = 4.0

        self.patient_splits = self._load_metadata()
        self.annotation_index = self._build_annotation_index()

    def _load_metadata(self) -> Dict[str, int]:
        if not self.path_metadata.exists(): return {}
        try:
            df = pd.read_parquet(self.path_metadata)
            df['patient_num_id'] = df['patient_num_id'].astype(str)
            return dict(zip(df['patient_num_id'], df['split_final']))
        except Exception as e:
            print(f"Error loading metadata: {e}")
            return {}

    def _build_annotation_index(self) -> Dict[Tuple[str, str, str], Path]:
        print("Indexing raw annotations...")
        index = {}
        for csv_path in self.path_raw.rglob("*.csv"):
            name_parts = csv_path.stem.split('_')
            if len(name_parts) < 3: continue
            session_prefix = csv_path.parent.name.split('_')[0]
            index[(csv_path.parent.parent.name, session_prefix, name_parts[-1])] = csv_path
        return index

    def _load_ictal_intervals(self, csv_path: Path) -> List[Tuple[float, float, str]]:
        intervals = []
        try:
            with open(csv_path, 'r') as f:
                lines = f.readlines()
            data_start, col_map = -1, {}
            for i, line in enumerate(lines):
                if line.startswith('#') or not line.strip(): continue
                parts = [p.strip().lower() for p in line.split(',')]
                if 'start_time' in parts:
                    data_start, col_map = i + 1, {n: idx for idx, n in enumerate(parts)}
                    break
            if data_start == -1: return []
            for line in lines[data_start:]:
                parts = [p.strip() for p in line.split(',')]
                if len(parts) < 4: continue
                label = parts[col_map['label']].lower()
                if label != 'bckg':
                    intervals.append((float(parts[col_map['start_time']]), 
                                    float(parts[col_map['stop_time']]), label))
            return list(set(intervals))
        except: return []

    def _get_label_by_consensus(self, start: float, end: float, intervals: List[Tuple]) -> str:
        """
        Consensus Voting Logic:
        1. Accumulates duration for each label type within the window.
        2. Selects the label with the highest duration (The Winner).
        3. Applies a 15% threshold for final ictal assignment.
        """
        if not intervals: return 'bckg'
        
        label_durations = defaultdict(float)
        for i_s, i_e, label in intervals:
            overlap = min(end, i_e) - max(start, i_s)
            if overlap > 0:
                label_durations[label.lower()] += overlap
        
        if not label_durations: return 'bckg'
        
        # Winner-take-all based on cumulative duration
        winning_label = max(label_durations, key=label_durations.get)
        winning_duration = label_durations[winning_label]
        
        # Threshold calculation (15% of 4.096s)
        threshold = (end - start) * 0.15
        
        if winning_duration >= threshold:
            return winning_label
            
        return 'bckg'

    def run(self, mode: int = 3, max_workers: int = 4, target_patient: Optional[str] = None):
        all_tasks = self._discover_tasks()
        if not all_tasks: return
        mid = len(all_tasks) // 2
        
        if mode == 1:
            p_id = str(target_patient) if target_patient else all_tasks[0][0]
            tasks = [t for t in all_tasks if t[0] == p_id]
        elif mode == 2: tasks = all_tasks
        elif mode == 3: tasks = all_tasks[:mid]
        elif mode == 4: tasks = all_tasks[mid:]
        else: raise ValueError("Invalid mode (1-4).")

        print(f"Batch Processing | Mode: {mode} | Workers: {max_workers} | Total Sessions: {len(tasks)}")
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(self._process_session, t, (mode == 1)): t for t in tasks}
            for f in tqdm.tqdm(as_completed(futures), total=len(tasks), desc="Extracting Features"):
                f.result()

    def _discover_tasks(self) -> List[Tuple]:
        tasks = []
        if not self.path_clean.exists(): return tasks
        for p_dir in sorted(self.path_clean.iterdir()):
            if not p_dir.is_dir(): continue
            for s_dir in sorted(p_dir.iterdir()):
                files = list(s_dir.glob("*.parquet"))
                if files: tasks.append((p_dir.name, s_dir.name, files))
        return tasks

    def _process_session(self, task: Tuple, debug: bool = False) -> Dict[str, Any]:
        p_id, session_id, signal_files = task
        split_final = self.patient_splits.get(p_id, 0)
        
        out_dir = self.path_output / p_id
        out_dir.mkdir(parents=True, exist_ok=True)
        final_path, temp_path = out_dir / f"session_{session_id}.parquet", out_dir / f"session_{session_id}.tmp"
        
        if not debug and final_path.exists(): return {"status": "skipped"}
        if debug and final_path.exists(): os.remove(final_path)

        try:
            session_results = []
            for s_file in sorted(signal_files):
                partition_id = s_file.stem
                csv_path = self.annotation_index.get((p_id, session_id, partition_id))
                intervals = self._load_ictal_intervals(csv_path) if csv_path else []

                # Loading Parquet and selecting numeric data
                raw_df = pd.read_parquet(s_file).select_dtypes(include=[np.number])
                to_drop = [c for c in ['time', 'duration_sec'] if c in raw_df.columns]
                df_sig = raw_df.drop(columns=to_drop, errors='ignore').fillna(0)
                
                ch_names, signal = df_sig.columns.tolist(), df_sig.values.T.astype(np.float32)
                del raw_df, df_sig # RAM Guard
                
                idx = 0
                while idx + self.window_samples <= signal.shape[1]:
                    t_start = idx / self.sampling_rate
                    t_end = t_start + (self.window_samples / self.sampling_rate)
                    label = self._get_label_by_consensus(t_start, t_end, intervals)
                    
                    # Stride policy
                    stride = 4.0 if split_final == 1 else (self.stride_ictal if label != 'bckg' else self.stride_bckg)
                    
                    features, f_names = extract_comprehensive_features(
                        signal[:, idx : idx + self.window_samples], ch_names
                    )
                    
                    session_results.append({
                        'partition_id': partition_id,
                        'win_start_sec': np.float32(t_start),
                        'label': label,
                        **dict(zip(f_names, features.astype(np.float32)))
                    })
                    idx += int(stride * self.sampling_rate)
                del signal

            if session_results:
                pq.write_table(pa.Table.from_pandas(pd.DataFrame(session_results)), temp_path, compression='snappy')
                os.replace(temp_path, final_path)
            
            del session_results
            gc.collect() 
            return {"status": "success"}
        except Exception as e:
            if temp_path.exists(): os.remove(temp_path)
            if debug: print(f"Error in {session_id}: {e}")
            gc.collect()
            return {"status": "error"}

if __name__ == "__main__":
    DATALAKE_PATH = r"D:\TUSZ_project\TUSZ_DataLake"
    pipeline = FeatureExtractionPipeline(DATALAKE_PATH)
    # Mode 3: Process the first half of the dataset
    pipeline.run(mode=1, max_workers=1, target_patient = "209")