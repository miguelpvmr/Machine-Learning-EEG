from xgboost import XGBClassifier
import numpy as np
import gc
from .abstractModel import AbstractModel

class XGBoostModel(AbstractModel):
    """
    Implementación acelerada por GPU de XGBoost siguiendo la firma de AbstractModel.
    Optimizado para el manejo de desbalance de clases y grandes volúmenes de datos EEG.
    """
    def _init_model(self):
        """
        Instancia el motor de XGBoost con soporte para CUDA y parámetros planos.
        """
        # Extraemos los parámetros actuales; si no existen, usamos valores por defecto clínicos
        params = self.params
        
        # Forzamos parámetros de hardware para Colab Pro
        gpu_params = {
            'tree_method': 'hist', # Requerido para activar soporte GPU
            'device': 'cuda',      # Especifica el uso de núcleos CUDA
            'random_state': 42,
            'verbosity': 0
        }
        
        full_params = {**params, **gpu_params} # Combinamos con los parámetros de búsqueda (subsample, gamma, etc.)
        
        return XGBClassifier(**full_params)

    def fit(self, X, y, sample_weight=None):
        """
        Entrenamiento optimizado para VRAM. 
        Castea los datos a float32 para maximizar el rendimiento de los núcleos Tensor.
        """
        # Preparación de datos para el kernel de CUDA
        X_proc = np.ascontiguousarray(X, dtype=np.float32)
        y_proc = np.array(y, dtype=np.int32).ravel()
        
        # Manejo de pesos para el desbalance (80% Basal vs Crisis)
        sw_proc = None
        if sample_weight is not None:
            sw_proc = np.array(sample_weight, dtype=np.float32)

        
        self.model.fit(X_proc, y_proc, sample_weight=sw_proc) # Ejecución del entrenamiento en el servidor de Google
        del X_proc, y_proc, sw_proc # Limpieza agresiva de RAM en el CPU para evitar colapsos
        gc.collect()
        
        return self