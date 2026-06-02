import joblib
import numpy as np
import logging
from typing import Dict, Tuple, Any
from ...domain.interfaces.ml_model_interface import IClassificationModel

logger = logging.getLogger(__name__)

class SklearnModelWrapper(IClassificationModel):
    """
    Universal wrapper for Scikit-Learn pipelines exported via joblib.
    Dynamically handles binary and multiclass predictions, including custom wrappers like LogitModel.
    """
    
    def __init__(self, model_path: str):
        logger.info(f"Loading Scikit-Learn model artifact from: {model_path}")
        try:
            # 1. Carga del pipeline
            self.pipeline = joblib.load(model_path)
            last_estimator = self.pipeline.steps[-1][1]
            
            # 2. Detección Robusta de Clases (Tu lógica actual)
            if hasattr(self.pipeline, 'classes_'):
                self.classes = self.pipeline.classes_
            else:
                if hasattr(last_estimator, 'model') and hasattr(last_estimator.model, 'classes_'):
                    self.classes = last_estimator.model.classes_
                elif hasattr(last_estimator, 'classes_'):
                    self.classes = last_estimator.classes_
                else:
                    logger.warning("Atributo 'classes_' no detectado. Usando default detectado en logs [0, 1, 2].")
                    self.classes = np.array([0, 1, 2])

            # 3. EL PARCHE (SILENT PATCH): 
            # Inyectamos los atributos que Scikit-Learn busca para confirmar que está "fitted".
            # Esto elimina los FutureWarnings sin modificar el código de LogitModel.
            if not hasattr(last_estimator, 'classes_'):
                last_estimator.classes_ = self.classes
            
            # Algunos validadores de Pipeline buscan n_features_in_ o cualquier atributo con '_' al final
            if not hasattr(last_estimator, 'n_features_in_'):
                last_estimator.n_features_in_ = 1084 

            logger.info(f"Model loaded successfully. Detected classes: {self.classes}")
            
        except Exception as e:
            logger.error(f"Failed to load .joblib model: {e}")
            raise RuntimeError(f"Critical ML loading error: {e}")

    def predict(self, feature_vector: np.ndarray) -> Tuple[Any, Dict[str, float]]:
        """
        Runs the full pipeline (Scaler -> PCA -> Estimator).
        Returns the top prediction and a dictionary of all probabilities.
        """
        if feature_vector.ndim == 1: 
            feature_vector = feature_vector.reshape(1, -1)
            
        prediction = self.pipeline.predict(feature_vector)[0]
        probabilities = self.pipeline.predict_proba(feature_vector)[0]
        
        prob_dict = { 
            str(cls_name): float(prob) 
            for cls_name, prob in zip(self.classes, probabilities)
        }
        
        return prediction, prob_dict