import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import pandas as pd
import numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed, TimeoutError
import tqdm
import json
import datetime
import pyarrow as pa
import pyarrow.parquet as pq
import os
import psutil  # NUEVO: Para monitorear la RAM
import time    # NUEVO: Para pausar el proceso si la RAM está llena

# Importación del motor matemático
try:
    from feature_engineering import extract_comprehensive_features
except ImportError:
    print("❌ Error: No se encontró 'feature_engineering.py'.")
    exit()

class FeatureExtractionPipeline:
    def __init__(self, root_path: str):
        self.root = Path(root_path).resolve()
        self.path_annotations = self.root / "01_Raw_Consolidated"
        self.path_metadata = self.root / "02_Metadata"
        self.path_signals = self.root / "03_TUSZ_Clean"
        self.path_output = self.root / "04_TUSZ_Features_ML" / "version=v1_raw_labels"
        
        metadata_file = self.path_metadata / "metadata_patients.parquet"
        if not metadata_file.exists():
            raise FileNotFoundError(f"❌ Metadata no encontrada.")
            
        df_meta = pd.read_parquet(metadata_file)
        self.patient_map = df_meta.set_index('patient_id')['patient_num_id'].to_dict()

        # 🔥 EL NUEVO INDEXADOR GLOBAL 🔥
        print("🔍 Escaneando archivos de anotaciones en Raw... (Tomará unos segundos)")
        self.annotation_map = {f.name: f for f in self.path_annotations.rglob('*.csv')}
        print(f"✅ ¡Se indexaron {len(self.annotation_map)} archivos de etiquetas!\n")

    def generate_lineage_report(self) -> None:
        """Genera el JSON con la configuración técnica para trazabilidad."""
        self.path_metadata.mkdir(parents=True, exist_ok=True)
        lineage = {
            "version": "v1_raw_labels",
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "config": {
                "fs_hz": 250, 
                "window_sec": 4.096, 
                "wavelet": "db4", 
                "precision": "float32",
                "compression": "snappy",
                "ram_cap": "90%"
            }
        }
        with open(self.path_metadata / "feature_lineage.json", "w") as f:
            json.dump(lineage, f, indent=4)

    def load_intervals(self, csv_path: Path) -> np.ndarray:
        try:
            # Ignora los metadatos de TUSZ que empiezan con '#'
            df = pd.read_csv(csv_path, comment='#')
            
            # BLINDAJE: Limpiamos espacios en blanco invisibles
            df.columns = df.columns.str.strip()
            df['label'] = df['label'].astype(str).str.strip()
            
            # Filtra solo las crisis y elimina canales duplicados en el mismo tiempo
            df_seizures = df[df['label'] != 'bckg']
            if df_seizures.empty:
                return np.empty((0, 3))
                
            intervals = df_seizures[['start_time', 'stop_time', 'label']].drop_duplicates()
            return intervals.values
        except Exception as e:
            print(f"\n⚠️ Error leyendo {csv_path.name}: {e}")
            return np.empty((0, 3))

    def get_window_label(self, start_t: float, end_t: float, intervals: np.ndarray) -> str:
        if intervals.size == 0: 
            return 'bckg'
            
        win_duration = end_t - start_t
        
        for s, e, lbl in intervals:
            # Calcula intersección real
            overlap_start = max(start_t, float(s))
            overlap_end = min(end_t, float(e))
            overlap_duration = max(0, overlap_end - overlap_start)
            
            # Si más del 50% de la ventana está en crisis, se etiqueta como crisis
            if overlap_duration >= (win_duration * 0.5):
                return str(lbl)
                
        return 'bckg'

    def process_session(self, task: tuple) -> dict:
        p_alpha, session_name, signal_files = task
        try:
            p_id_numeric = self.patient_map.get(p_alpha)
            if p_id_numeric is None: return {"status": "error", "p": p_alpha, "msg": "ID no en metadata"}
            
            out_dir = self.path_output / str(p_id_numeric)
            out_dir.mkdir(parents=True, exist_ok=True)
            
            file_path = out_dir / f"session_{session_name}.parquet"
            tmp_file_path = out_dir / f"session_{session_name}.tmp"
            
            # Checkpointing
            if file_path.exists():
                return {"status": "skipped", "session": session_name}

            # Limpieza atómica
            if tmp_file_path.exists():
                os.remove(tmp_file_path)

            print(f"⚙️ Procesando -> Paciente: {p_alpha} (ID:{p_id_numeric}) | Sesión: {session_name}", flush=True)

            fs, samples_win = 250, 1024
            win_sec = 4.096
            results = []

            for s_file in signal_files:
                
                # 🔥 WATCHDOG DE RAM AL 90% 🔥
                while psutil.virtual_memory().percent > 90.0:
                    print(f"⏳ RAM al {psutil.virtual_memory().percent}%. Pausando sesión {session_name} para evitar bloqueo...", flush=True)
                    time.sleep(5)

                df_raw = pd.read_parquet(s_file)
                cols_to_drop = ['time', 'patient_num_id', 'duration_sec', 'date', 
                               'filter_configuration', 'bandpass_range_hz', 'montage_reference']
                df_sig = df_raw.drop(columns=[c for c in cols_to_drop if c in df_raw.columns]).select_dtypes(include=[np.number]).fillna(0)
                
                ch_names = df_sig.columns.tolist()
                signal = df_sig.values.T.astype(np.float32) 

                # 🔥 BUSQUEDA GLOBAL INTELIGENTE 🔥
                ann_name = s_file.with_suffix('.csv').name
                ann_path = self.annotation_map.get(ann_name)
                
                if ann_path is None:
                    print(f"❌ ERROR DE RUTA: ¡No existe {ann_name} en ninguna subcarpeta de Raw!")
                    intervals = np.empty((0, 3))
                else:
                    intervals = self.load_intervals(ann_path)
                    # Quité el print de "BINGO" aquí para no saturar tu consola durante las miles de sesiones.
                    # Si quieres, puedes volver a agregarlo, pero en produccion es mejor mantener la consola limpia.

                idx = 0
                while idx + samples_win <= signal.shape[1]:
                    t_start = idx / fs
                    t_end = t_start + win_sec
                    label = self.get_window_label(t_start, t_end, intervals)
                    
                    # Stride adaptativo
                    stride = 1.0 if label != 'bckg' else 4.0
                    
                    f_vec, f_names = extract_comprehensive_features(signal[:, idx : idx + samples_win], ch_names)
                    
                    row = {'start_time': np.float32(t_start), 'end_time': np.float32(t_end), 'label': label}
                    row.update(dict(zip(f_names, f_vec.astype(np.float32))))
                    results.append(row)
                    
                    idx += int(stride * fs)

            if results:
                df_res = pd.DataFrame(results)
                cols = ['start_time', 'end_time', 'label'] + [c for c in df_res.columns if c not in ['start_time', 'end_time', 'label']]
                df_res = df_res[cols]
                
                table = pa.Table.from_pandas(df_res)
                custom_meta = {**table.schema.metadata, b'patient_num_id': str(p_id_numeric).encode()}
                table = table.replace_schema_metadata(custom_meta)
                
                # ESCRITURA ATÓMICA
                pq.write_table(table, tmp_file_path, compression='snappy')
                tmp_file_path.rename(file_path)

            return {"status": "success", "session": session_name}
        except Exception as e:
            return {"status": "error", "p": p_alpha, "msg": str(e)}

    def run(self, mode: int):
        self.generate_lineage_report()
        
        all_tasks = []
        for block in sorted(self.path_signals.iterdir()):
            if not block.is_dir(): continue
            for session in sorted(block.iterdir()):
                if not session.is_dir(): continue
                files = list(session.glob("*.parquet"))
                if files:
                    all_tasks.append((files[0].name.split('_')[0], session.name, files))

        total_tasks = len(all_tasks)
        if total_tasks == 0:
            print("⚠️ No hay tareas pendientes.")
            return

        midpoint = total_tasks // 2

        # SISTEMA DE MODOS
        if mode == 1:
            tasks = all_tasks[:1]
            print(f"🔬 MODO 1: Procesando SOLO EL PRIMER REGISTRO (1 sesión).")
        elif mode == 2:
            tasks = all_tasks
            print(f"🚀 MODO 2: Procesando TODO el dataset ({total_tasks} sesiones).")
        elif mode == 3:
            tasks = all_tasks[:midpoint]
            print(f"🌗 MODO 3: Procesando la PRIMERA MITAD (0 al {midpoint}).")
        elif mode == 4:
            tasks = all_tasks[midpoint:]
            print(f"🌓 MODO 4: Procesando la SEGUNDA MITAD ({midpoint} al {total_tasks}).")
        else:
            print("❌ Modo no válido. Elige 1, 2, 3 o 4.")
            return

        # 🔥 14 NÚCLEOS DE TUS 16 LÓGICOS 🔥
        with ProcessPoolExecutor(max_workers=8) as executor:
            futures = {executor.submit(self.process_session, t): t for t in tasks}
            processed, skipped, timeouts, errors = 0, 0, 0, 0
            
            for f in tqdm.tqdm(as_completed(futures, timeout=14400), total=len(tasks), desc="Extracción"):
                try:
                    res = f.result()
                    if res["status"] == "skipped": skipped += 1
                    elif res["status"] == "success": processed += 1
                    else: 
                        errors += 1
                        tqdm.tqdm.write(f"❌ Error en {res['p']}: {res['msg']}")
                except TimeoutError:
                    timeouts += 1
                    tqdm.tqdm.write("⚠️ TIMEOUT: Una sesión tardó más de 4 horas y fue cancelada.")
                except Exception as e:
                    errors += 1
                    tqdm.tqdm.write(f"🔥 Error crítico: {str(e)}")
        
        print("\n" + "="*50)
        print(f"✅ Lote finalizado.")
        print(f"   - Nuevos procesados  : {processed}")
        print(f"   - Saltados (listos)  : {skipped}")
        if errors > 0: print(f"   - Errores            : {errors}")
        if timeouts > 0: print(f"   - Cancelados (Lentos): {timeouts}")
        print("="*50)

if __name__ == "__main__":
    pipeline = FeatureExtractionPipeline("D:/TUSZ_DataLake")
    # 🔥 LISTO PARA EL PRIMER LOTE DE PRODUCCIÓN 🔥
    MODO_ELEGIDO = 4 
    pipeline.run(mode=MODO_ELEGIDO)