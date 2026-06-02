from sklearn.ensemble import RandomForestClassifier as SklearnRF
from .abstractModel import AbstractModel 
import numpy as np
import gc
import warnings

class RandomForest(AbstractModel):
    """
    Wrapper híbrido optimizado para investigación EEG.
    Resuelve la discrepancia de nombres entre Sklearn (criterion) 
    y cuML (split_criterion) durante búsquedas iterativas.
    """
    
    def _init_model(self):
        self.use_gpu = getattr(self, 'use_gpu', False)
        
        # Parámetros base unificados
        params = {
            'n_estimators': int(getattr(self, 'n_estimators', 100)),
            'max_depth': getattr(self, 'max_depth', None),
            'min_samples_split': int(getattr(self, 'min_samples_split', 2)),
            'min_samples_leaf': int(getattr(self, 'min_samples_leaf', 1)),
            'max_features': getattr(self, 'max_features', 'sqrt'),
            'bootstrap': getattr(self, 'bootstrap', True),
            'random_state': int(getattr(self, 'random_state', 42))
        }

        if self.use_gpu:
            from cuml.ensemble import RandomForestClassifier as CuMLRF
            # Mapeo inicial
            params['split_criterion'] = getattr(self, 'criterion', 'gini')
            params['n_bins'] = int(getattr(self, 'n_bins', 128))
            
            if params['max_depth'] is None:
                params['max_depth'] = 16 # Límite de seguridad en VRAM
                
            return CuMLRF(**params)
        else:
            params['criterion'] = getattr(self, 'criterion', 'gini')
            params['n_jobs'] = getattr(self, 'n_jobs', -1)
            return SklearnRF(**params)

    def set_params(self, **params):
        """
        Interceptor de parámetros para compatibilidad con GridSearch/Optuna.
        Traduce 'criterion' a 'split_criterion' si se detecta backend GPU.
        """
        # Actualizamos los atributos del wrapper
        for key, value in params.items():
            setattr(self, key, value)
            if key not in self._param_names:
                self._param_names.append(key)
        
        # Preparamos la traducción para el motor interno
        internal_params = params.copy()
        if getattr(self, 'use_gpu', False) and 'criterion' in internal_params:
            internal_params['split_criterion'] = internal_params.pop('criterion')
        
        # Sincronización con el motor
        if hasattr(self.model, 'set_params'):
            try:
                self.model.set_params(**internal_params)
            except Exception:
                self.model = self._init_model()
        else:
            self.model = self._init_model()
            
        return self

    def fit(self, X, y, sample_weight=None):
        """
        Implementación robusta de pesos. 
        Si el motor (especialmente en GPU) no soporta sample_weight, 
        se captura el error para no interrumpir el pipeline.
        """
        X_proc = np.ascontiguousarray(X, dtype=np.float32)
        y_proc = np.array(y, dtype=np.int32).ravel()
        
        try:
            if sample_weight is not None:
                sw_proc = np.ascontiguousarray(sample_weight, dtype=np.float32)
                self.model.fit(X_proc, y_proc, sample_weight=sw_proc)
            else:
                self.model.fit(X_proc, y_proc)
        except TypeError:
            # Esta captura es vital para cuML
            if sample_weight is not None:
                warnings.warn(f"El backend de {'GPU' if self.use_gpu else 'CPU'} no soporta sample_weight. "
                              "Se procede con entrenamiento estándar.")
            self.model.fit(X_proc, y_proc)
        
        del X_proc, y_proc
        gc.collect()
        return self

    def predict_proba(self, X):
        X_proc = np.ascontiguousarray(X, dtype=np.float32)
        return self.model.predict_proba(X_proc)
        
    def predict(self, X):
        X_proc = np.ascontiguousarray(X, dtype=np.float32)
        return self.model.predict(X_proc)