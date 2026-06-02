import os
import gc
import time
import random
import warnings
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.preprocessing import QuantileTransformer, RobustScaler, StandardScaler, FunctionTransformer
from sklearn.utils.class_weight import compute_sample_weight
from imblearn.over_sampling import SMOTE, ADASYN

try:
    import cupy as cp
except ImportError:
    cp = None

class OOFExtractor:
    """
    Framework Maestro de Extracción de Probabilidades Out-Of-Fold (OOF).
    Soporta arquitecturas multi-modelo concurrentes compartiendo transformaciones geométricas.
    Aplica una doble estratificación estricta (Patient-Wise + Class-Wise) sin fuga de datos.
    Genera un único archivo maestro Parquet alineado con compresión Snappy y retiene el Patient ID.
    """

    def __init__(self, df_map, train_set_paths, feature_cols, target_col, label_mapping=None):
        self.df_map = df_map if isinstance(df_map, pd.DataFrame) else self._parse_map(df_map)
        self.train_set_paths = train_set_paths
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.label_mapping = label_mapping
        self.patient_cache = {}

    def _parse_map(self, d):
        rows = []
        for f_id, data in d.items():
            for p_id in data.get('patients', []):
                rows.append({'fold_id': f_id, 'patient_num_id': str(p_id)})
        return pd.DataFrame(rows)

    def _set_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)

    def _get_scaler(self, scaler_type, n_samples, seed):
        valid = ['robust', 'standard', 'quantile_uniform', 'quantile_gaussian', 'none']
        if scaler_type not in valid:
            raise ValueError(f"scaler_type '{scaler_type}' no soportado.")
        
        if scaler_type == 'none': return FunctionTransformer(func=None)
        elif scaler_type == 'quantile_uniform': return QuantileTransformer(n_quantiles=min(1000, n_samples), output_distribution='uniform', random_state=seed)
        elif scaler_type == 'quantile_gaussian': return QuantileTransformer(n_quantiles=min(1000, n_samples), output_distribution='normal', random_state=seed)
        elif scaler_type == 'standard': return StandardScaler()
        return RobustScaler()

    def pre_cache_data(self):
        target_ids = set(self.df_map['patient_num_id'].unique())
        paths = sorted([p for p in self.train_set_paths if p.name in target_ids])
        cols = self.feature_cols + [self.target_col]
        
        pbar = tqdm(paths, desc="OOF Data Caching")
        for p_path in pbar:
            x_pts, y_pts = [], []
            for f in sorted(p_path.glob("*.parquet")):
                df = pd.read_parquet(f, columns=cols)
                if self.label_mapping:
                    df[self.target_col] = df[self.target_col].astype(str).str.lower().str.strip().map(self.label_mapping)
                df = df.dropna(subset=[self.target_col])
                if len(df) == 0: continue
                x_pts.append(df[self.feature_cols].values.astype(np.float32))
                y_pts.append(df[self.target_col].values.astype(np.int8))
            if x_pts:
                self.patient_cache[p_path.name] = (np.vstack(x_pts), np.concatenate(y_pts))
        gc.collect()

    def _stratified_subsample(self, ids, n, seed):
        """
        Submuestreo con doble capa de protección: proporcional por volumen de paciente
        y estrictamente estratificado por distribución de clases internas (crisis).
        """
        self._set_seed(seed)
        ids_s = sorted([str(i) for i in ids if str(i) in self.patient_cache])
        x_l, y_l = [], []
        
        if n is None:
            for p in ids_s:
                X, y = self.patient_cache[p]
                x_l.append(X); y_l.append(y)
            return np.vstack(x_l), np.concatenate(y_l)
        
        total_w = sum(len(self.patient_cache[i][1]) for i in ids_s)
        
        for p in ids_s:
            X_p, y_p = self.patient_cache[p]
            n_p = len(y_p)
            p_quota = max(1, int((n_p / total_w) * n))
            
            classes_p, counts_p = np.unique(y_p, return_counts=True)
            
            if len(classes_p) > 1:
                indices_p = []
                for cls, count in zip(classes_p, counts_p):
                    cls_quota = max(1, int((count / n_p) * p_quota))
                    available_idx = np.where(y_p == cls)[0]
                    chosen_cls_idx = np.random.choice(
                        available_idx, 
                        size=min(len(available_idx), cls_quota), 
                        replace=False
                    )
                    indices_p.append(chosen_cls_idx)
                final_idx_p = np.concatenate(indices_p)
            else:
                final_idx_p = np.random.choice(np.arange(n_p), size=min(n_p, p_quota), replace=False)
            
            x_l.append(X_p[final_idx_p])
            y_l.append(y_p[final_idx_p])
            
        return np.vstack(x_l), np.concatenate(y_l)

    def _get_val_data(self, ids):
        x, y, p_ids = [], [], []
        ids_s = sorted([str(i) for i in ids if str(i) in self.patient_cache])
        for i in ids_s:
            d = self.patient_cache[i]
            x.append(d[0]) 
            y.append(d[1]) 
            # --- MODIFICACIÓN CLAVE: Generamos un vector con el ID del paciente repetido N veces ---
            p_ids.extend([i] * len(d[1])) 
        return np.vstack(x), np.concatenate(y), np.array(p_ids)

    def _safe_predict_proba(self, model, X_val):
        if hasattr(model, "predict_proba"):
            return model.predict_proba(X_val)
        elif hasattr(model, "decision_function"):
            decisions = model.decision_function(X_val)
            if decisions.ndim == 1:
                probs = 1 / (1 + np.exp(-decisions))
                return np.vstack([1 - probs, probs]).T
            else:
                exp_d = np.exp(decisions - np.max(decisions, axis=1, keepdims=True))
                return exp_d / np.sum(exp_d, axis=1, keepdims=True)
        else:
            raise AttributeError(f"Estimador no soporta generación de probabilidades.")

    def extract(self, pipeline_meta_configs, export_dir, samples_per_fold=40000, seed=42):
        if not self.patient_cache: self.pre_cache_data()
        self._set_seed(seed)

        export_path = Path(export_dir)
        export_path.mkdir(parents=True, exist_ok=True)

        flat_configs = []
        for model_type, meta in pipeline_meta_configs.items():
            factory = meta["factory_call"]
            for cfg in meta["configs"]:
                extended_cfg = cfg.copy()
                extended_cfg["factory_call"] = factory
                extended_cfg["model_type"] = model_type
                flat_configs.append(extended_cfg)

        config_groups = {}
        for cfg in flat_configs:
            k = cfg['params'].get('pca__n_components', None)
            bal = cfg['params'].get('balancing', 'none')
            s_type = cfg.get('scaler', 'robust')
            adj_scaler = s_type if (s_type == 'none' or k is None) else ('quantile_uniform' if not s_type.startswith('quantile') else s_type)
            
            grp_key = (adj_scaler, k, bal)
            if grp_key not in config_groups:
                config_groups[grp_key] = []
            config_groups[grp_key].append(cfg)

        folds = sorted(self.df_map['fold_id'].unique())
        oof_accumulator = {cfg['name']: [] for cfg in flat_configs}
        oof_targets = []
        oof_patient_ids = [] # --- MODIFICACIÓN CLAVE: Acumulador de IDs ---

        pbar_folds = tqdm(folds, desc="OOF Master Loop (Folds)", position=0)

        for f_id in pbar_folds:
            train_ids = self.df_map[self.df_map['fold_id'] != f_id]['patient_num_id']
            val_ids = self.df_map[self.df_map['fold_id'] == f_id]['patient_num_id']

            X_tr_base, y_tr_base = self._stratified_subsample(train_ids, samples_per_fold, seed)
            
            # --- MODIFICACIÓN CLAVE: Recibimos el arreglo de IDs de validación ---
            X_vl_base, y_vl_base, p_ids_vl = self._get_val_data(val_ids)
            
            oof_targets.append(y_vl_base)
            oof_patient_ids.append(p_ids_vl)

            pbar_groups = tqdm(config_groups.items(), desc=f"Fold {f_id} - Compartiendo Transformaciones", position=1, leave=False)

            for (sc_type, pca_k, bal_type), group_configs in pbar_groups:
                
                sc = self._get_scaler(sc_type, len(X_tr_base), seed)
                X_tr_step = sc.fit_transform(X_tr_base)
                X_vl_step = sc.transform(X_vl_base)

                if pca_k is not None:
                    pca = PCA(n_components=pca_k, svd_solver='randomized', random_state=seed)
                    X_tr_step = pca.fit_transform(X_tr_step)
                    X_vl_step = pca.transform(X_vl_step)

                sw = None
                if bal_type == 'SMOTE': X_tr_bal, y_tr_bal = SMOTE(random_state=seed).fit_resample(X_tr_step, y_tr_base)
                elif bal_type == 'ADASYN': X_tr_bal, y_tr_bal = ADASYN(random_state=seed).fit_resample(X_tr_step, y_tr_base)
                elif bal_type == 'weight_balanced': sw = compute_sample_weight('balanced', y_tr_base); X_tr_bal, y_tr_bal = X_tr_step, y_tr_base
                else: X_tr_bal, y_tr_bal = X_tr_step, y_tr_base

                for cfg in group_configs:
                    params = cfg['params']
                    m_type = cfg['model_type']
                    
                    is_knn = (m_type == 'knn')
                    is_gnb = (m_type == 'gnb')
                    inf_params = ['decision_strategy', 'voting_weight'] if (is_gnb or is_knn) else []
                    if is_knn: inf_params.append('n_neighbors')
                    
                    fit_kwargs = {key.split('__')[-1]: v for key, v in params.items() if 'pca__' not in key and key != 'balancing' and key not in inf_params}
                    
                    model = cfg["factory_call"]().set_params(**fit_kwargs)

                    if cp is not None:
                        cp.get_default_memory_pool().free_all_blocks()
                        cp.get_default_pinned_memory_pool().free_all_blocks()

                    if is_knn: model.fit(X_tr_bal, y_tr_bal)
                    else: model.fit(X_tr_bal, y_tr_bal, sample_weight=sw)

                    y_prob = self._safe_predict_proba(model, X_vl_step)
                    oof_accumulator[cfg['name']].append(y_prob)

                del X_tr_step, X_vl_step, X_tr_bal, y_tr_bal
                gc.collect()
            
            del X_tr_base, y_tr_base, X_vl_base, y_vl_base
            gc.collect()

        print("\n[INFO] Construyendo DataFrame maestro consolidado...")
        y_meta_final = np.concatenate(oof_targets)
        p_ids_meta_final = np.concatenate(oof_patient_ids) # --- Acumulamos todos los IDs ---
        
        # --- MODIFICACIÓN CLAVE: Injectamos el ID en el DF ---
        df_master = pd.DataFrame({
            'patient_num_id': p_ids_meta_final, 
            'target_real': y_meta_final
        })
        
        for cfg in flat_configs:
            m_name = cfg['name']
            prob_list = oof_accumulator[m_name]
            m_matrix = np.vstack(prob_list)
            
            df_master[f'{m_name}_P_Background'] = m_matrix[:, 0].astype(np.float32)
            df_master[f'{m_name}_P_Focal'] = m_matrix[:, 1].astype(np.float32)
            df_master[f'{m_name}_P_Generalized'] = m_matrix[:, 2].astype(np.float32)
            
            del m_matrix
            gc.collect()
            
        output_file = export_path / "oof_stacking_level0_master.parquet"
        df_master.to_parquet(output_file, compression='snappy', index=False)
        
        del df_master, oof_accumulator, y_meta_final, p_ids_meta_final
        gc.collect()

        print(f"\n[OK] Pipeline completo finalizado. Archivo maestro guardado en:")
        print(f" -> {output_file}")
        return