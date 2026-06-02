import mne
import pandas as pd
import numpy as np
import os
import warnings
import gc
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

# --- CONFIGURACIÓN DE SILENCIO TÉCNICO ---
mne.set_log_level('ERROR')
warnings.filterwarnings("ignore")

# --- CONFIGURACIÓN DE RUTAS ---
RAW_DATA_DIR = r"D:\TUSZ_DataLake\01_Raw_Consolidated"
METADATA_DIR = r"D:\TUSZ_DataLake\02_Metadata"
CLEAN_DATA_DIR = r"D:\TUSZ_DataLake\03_TUSZ_Clean"
SESSION_METADATA_FILE = Path(METADATA_DIR) / "metadata_sessions.parquet"

# --- MÓDULOS DE PROCESAMIENTO TÉCNICO ---

def normalize_channels(raw: mne.io.Raw, sfreq_target: float = 250.0) -> mne.io.Raw:
    """Estandariza nomenclatura, selecciona 10-20 y remuestrea."""
    # 1. Limpieza de etiquetas
    base_mapping = {ch: ch.replace('EEG ', '').replace('-REF', '').replace('-LE', '').strip() 
                    for ch in raw.ch_names}
    
    # 2. Diccionario de correcciones
    corrections = {'FP1': 'Fp1', 'FP2': 'Fp2', 'FZ': 'Fz', 'CZ': 'Cz', 'PZ': 'Pz'}
    final_mapping = {k: corrections.get(v.upper(), v) for k, v in base_mapping.items()}
    
    raw.rename_channels(final_mapping, verbose=False)
    
    # 3. Selección Sistema 10-20
    standard_1020 = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2',
                     'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'Fz', 'Cz', 'Pz']
    
    existing_channels = [ch for ch in standard_1020 if ch in raw.ch_names]
    raw.pick(existing_channels, verbose=False)
    
    if raw.info['sfreq'] != sfreq_target:
        raw.resample(sfreq_target, n_jobs=1, verbose=False)
    return raw

def apply_clinical_filters(raw: mne.io.Raw) -> mne.io.Raw:
    """Filtros de fase cero: Notch (60Hz) y Pasa-Banda (0.5-80Hz)."""
    raw.notch_filter(freqs=60.0, notch_widths=2.0, method='iir', 
                     phase='zero', n_jobs=1, verbose=False)
    
    filter_params = dict(order=8, ftype='butter', output='sos')
    raw.filter(l_freq=0.5, h_freq=80.0, method='iir', 
               iir_params=filter_params, phase='zero', n_jobs=1, verbose=False)
    return raw

def apply_longitudinal_bipolar_montage(raw: mne.io.Raw) -> mne.io.Raw:
    """Montaje Bipolar Longitudinal (Double Banana)."""
    pairs = [
        ('Fp1', 'F7'), ('F7', 'T3'), ('T3', 'T5'), ('T5', 'O1'),
        ('Fp2', 'F8'), ('F8', 'T4'), ('T4', 'T6'), ('T6', 'O2'),
        ('Fp1', 'F3'), ('F3', 'C3'), ('C3', 'P3'), ('P3', 'O1'),
        ('Fp2', 'F4'), ('F4', 'C4'), ('C4', 'P4'), ('P4', 'O2'),
        ('Fz', 'Cz'), ('Cz', 'Pz')
    ]
    
    current_channels = raw.ch_names
    valid_pairs = [(a, c) for a, c in pairs if a in current_channels and c in current_channels]
    
    anodes = [p[0] for p in valid_pairs]
    cathodes = [p[1] for p in valid_pairs]
    new_ch_names = [f"{p[0]}-{p[1]}" for p in valid_pairs]

    return mne.set_bipolar_reference(raw, anode=anodes, cathode=cathodes, 
                                     ch_name=new_ch_names, drop_refs=True, verbose=False)

def signal_preprocessing_worker(edf_path, session_metadata, output_base_dir):
    """Workflow atómico con limpieza explícita de RAM."""
    try:
        raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
        raw = normalize_channels(raw)
        raw = apply_clinical_filters(raw)
        raw_clean = apply_longitudinal_bipolar_montage(raw)
        
        df = raw_clean.to_data_frame()
        
        # Eliminar objetos pesados de MNE
        del raw
        del raw_clean
        
        # Enriquecimiento de datos
        df['patient_num_id'] = session_metadata['patient_num_id']
        df['duration_sec'] = session_metadata['duration_sec']
        df['date'] = session_metadata['date']
        
        # Metadata técnica
        df['filter_configuration'] = "Butterworth IIR 8th Order (Zero-Phase)"
        df['bandpass_range_hz'] = "0.5-80.0"
        df['montage_reference'] = "Longitudinal Bipolar (Double Banana)"
        
        # Guardado particionado
        p_id = session_metadata['patient_num_id']
        session_id = edf_path.parent.name
        dest_path = Path(output_base_dir) / f"patient={p_id}" / f"session={session_id}"
        dest_path.mkdir(parents=True, exist_ok=True)
        
        output_file = dest_path / f"{edf_path.stem}.parquet"
        df.to_parquet(output_file, engine='pyarrow', compression='snappy', index=False)
        
        # Liberar memoria final
        del df
        gc.collect() 
        return True
    except Exception as e:
        return f"Error en {edf_path.name}: {str(e)}"

def run_batch_preprocessing(source_dir, metadata_file, target_dir, workers=None):
    """Orquestador con lógica de concurrencia blindada."""
    print("📋 Cargando metadata de sesiones...")
    df_meta = pd.read_parquet(metadata_file)
    meta_lookup = df_meta.set_index('file_name')[['patient_num_id', 'duration_sec', 'date']].to_dict('index')

    recording_list = [p for p in Path(source_dir).rglob("*.edf") if p.name in meta_lookup]
    print(f"📊 Archivos para procesar: {len(recording_list)}")

    # Lógica de núcleos A PRUEBA DE BALAS
    if workers is not None:
        allocated_workers = workers
    else:
        # Intentar detectar núcleos, si falla usar 4 por defecto
        try:
            cores = os.cpu_count()
            allocated_workers = max(1, cores - 2) if cores is not None else 4
        except:
            allocated_workers = 4
        
    print(f"⚙️ Iniciando pool con {allocated_workers} núcleos...")

    error_logs = []
    with ProcessPoolExecutor(max_workers=allocated_workers) as executor:
        futures = {
            executor.submit(signal_preprocessing_worker, path, meta_lookup[path.name], target_dir): path.name 
            for path in recording_list
        }
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Signal Preprocessing"):
            try:
                res = future.result()
                if res is not True:
                    error_logs.append(res)
            except Exception as e:
                error_logs.append(f"Falla crítica: {str(e)}")

    if error_logs:
        with open("preprocessing_errors.log", "w") as f:
            f.write("\n".join(error_logs))
        print(f"⚠️ Se registraron {len(error_logs)} errores. Ver 'preprocessing_errors.log'.")
    
    print(f"✅ Proceso finalizado. Datos en: {target_dir}")

# --- ENTRADA PRINCIPAL ---
if __name__ == "__main__":
    # Asegúrate de que las carpetas existan o las rutas sean correctas
    run_batch_preprocessing(RAW_DATA_DIR, SESSION_METADATA_FILE, CLEAN_DATA_DIR, workers=4)