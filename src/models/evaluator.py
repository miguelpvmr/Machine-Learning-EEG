import os
import gc
import random
import warnings
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm.auto import tqdm
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import QuantileTransformer, RobustScaler, StandardScaler, FunctionTransformer
from sklearn.decomposition import PCA
from sklearn.base import clone
from sklearn.utils.class_weight import compute_sample_weight
from imblearn.over_sampling import SMOTE, ADASYN

class Evaluator:
    def __init__(self, train_set_paths, test_set_paths, estimator, 
                 feature_cols, target_col, label_mapping=None):
        self.train_set_paths = train_set_paths
        self.test_set_paths = test_set_paths
        self.estimator = estimator
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.label_mapping = label_mapping
        self.pipeline_ = None
        # Almacén para los datos de calibración (40k)
        self.calibration_data_ = None 

    def _set_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)

    def _load_and_cache(self, folder_paths, desc="Cargando"):
        local_cache = {}
        cols = self.feature_cols + [self.target_col]
        for p_path in tqdm(folder_paths, desc=desc, leave=True):
            x_pts, y_pts = [], []
            for file in p_path.glob("*.parquet"):
                df = pd.read_parquet(file, columns=cols)
                if self.label_mapping:
                    df[self.target_col] = df[self.target_col].astype(str).str.lower().str.strip().map(self.label_mapping)
                df = df.dropna(subset=[self.target_col])
                if df.empty: continue
                x_pts.append(df[self.feature_cols].values.astype(np.float32))
                y_pts.append(df[self.target_col].values.astype(np.int8))
            if x_pts:
                local_cache[p_path.name] = (np.vstack(x_pts), np.concatenate(y_pts))
        return local_cache

    def _stratified_subsample(self, cache, target_n, seed):
        self._set_seed(seed)
        total_windows = sum(len(v[1]) for v in cache.values())
        x_list, y_list = [], []
        for p_id, (X_p, y_p) in cache.items():
            n_p = len(y_p)
            p_quota = max(1, int((n_p / total_windows) * target_n))
            classes_p, counts_p = np.unique(y_p, return_counts=True)
            if len(classes_p) > 1:
                indices_p = []
                for cls, count in zip(classes_p, counts_p):
                    cls_quota = max(1, int((count / n_p) * p_quota))
                    idx_cls = np.where(y_p == cls)[0]
                    sel = np.random.choice(idx_cls, size=min(len(idx_cls), cls_quota), replace=False)
                    indices_p.append(sel)
                final_idx_p = np.concatenate(indices_p)
            else:
                final_idx_p = np.random.choice(np.arange(n_p), size=min(n_p, p_quota), replace=False)
            x_list.append(X_p[final_idx_p])
            y_list.append(y_p[final_idx_p])
        return np.vstack(x_list), np.concatenate(y_list)

    def fit(self, best_params, scaler_type='robust', samples_for_train=40000, seed=42,
            export_path=None):
        """
        Entrena el modelo en la submuestra y guarda los 40k para calibrar umbrales.
        """
        self._set_seed(seed)
        train_cache = self._load_and_cache(self.train_set_paths, "Cargando para Entrenamiento")
        X_train_s, y_train_s = self._stratified_subsample(train_cache, samples_for_train, seed)
        del train_cache
        gc.collect()

        # --- Lógica de Preprocesamiento (Scaler + PCA) ---
        k_pca = best_params.get('pca__n_components', None)
        if k_pca is not None:
            scaler = QuantileTransformer(n_quantiles=min(1000, len(X_train_s)), output_distribution='uniform', random_state=seed)
            pca = PCA(n_components=k_pca, svd_solver='randomized', random_state=seed)
        else:
            scaler = RobustScaler()
            pca = None

        X_transformed = scaler.fit_transform(X_train_s)
        if pca: X_transformed = pca.fit_transform(X_transformed)

        # --- Configuración y Fit del Estimador ---
        model = clone(self.estimator)
        clean_params = {k.split('__')[-1]: v for k, v in best_params.items() if 'pca__' not in k and k != 'balancing'}
        model.set_params(**clean_params)

        bal = best_params.get('balancing', 'none')
        if bal == 'SMOTE':
            X_res, y_res = SMOTE(random_state=seed).fit_resample(X_transformed, y_train_s)
            model.fit(X_res, y_res)
        else:
            model.fit(X_transformed, y_train_s)

        # --- Construcción del Pipeline ---
        steps = [("scaler", scaler)]
        if pca: steps.append(("pca", pca))
        steps.append(("model", model))
        self.pipeline_ = Pipeline(steps)

        # Guardamos tripletas de probabilidad de los 40k para tu función de optimización
        self.calibration_data_ = {
            'y_true': y_train_s,
            'y_prob': self.pipeline_.predict_proba(X_train_s)
        }

        # Exportación opcional a Joblib
        if export_path:
            p = Path(export_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(self.pipeline_, p)
            print(f"[MLOps] Modelo exportado en: {export_path}")

        del X_train_s, y_train_s, X_transformed
        gc.collect()
        return self

    def get_full_probabilities(self, paths, desc="Inferencia"):
        """
        Devuelve las tripletas de probabilidad para UN set completo (Train o Test).
        """
        if self.pipeline_ is None:
            raise ValueError("Debe ejecutar .fit() primero.")
            
        cache = self._load_and_cache(paths, desc)
        X_full = np.vstack([v[0] for v in cache.values()])
        y_full = np.concatenate([v[1] for v in cache.values()])
        
        y_prob = self.pipeline_.predict_proba(X_full)
        
        del cache
        gc.collect()
        return {'y_true': y_full, 'y_prob': y_prob}