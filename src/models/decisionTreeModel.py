from sklearn.tree import DecisionTreeClassifier
from .abstractModel import AbstractModel
import numpy as np
import gc

# ----------------------------------------------------------------
# SINGLE DECISION TREE WRAPPER (Optimized for EEG Data)
# ----------------------------------------------------------------
class DecisionTree(AbstractModel):
    """
    Professional wrapper for the Decision Tree Classifier.
    Interpretability-focused model for neurological event detection.
    Optimized for memory efficiency and Scikit-Learn GridSearch compatibility.
    """
    def _init_model(self):
        """
        Initializes the model using parameters passed via self.params.
        Defaults are overridden by the Trainer's GridSearch.
        """
        # Extraemos parámetros con valores por defecto seguros
        default_params = {
            'criterion': getattr(self, 'criterion', 'gini'),
            'max_depth': getattr(self, 'max_depth', None),
            'min_samples_split': int(getattr(self, 'min_samples_split', 2)),
            'min_samples_leaf': int(getattr(self, 'min_samples_leaf', 1)),
            'max_features': getattr(self, 'max_features', None), # Soporte explícito
            'random_state': 42
        }
        
        # Combinamos con cualquier otro parámetro en self.params (limpieza de prefijos clf__)
        return DecisionTreeClassifier(**default_params)

    def fit(self, X, y, sample_weight=None):
        """
        Standardizes input to float32/int32 to reduce memory footprint.
        Explicitly handles sample_weight for cost-sensitive learning experiments.
        """
        # Forzamos alineación de memoria y tipos de datos eficientes
        X_proc = np.ascontiguousarray(X, dtype=np.float32)
        y_proc = np.array(y, dtype=np.int32).ravel()
        
        self.model.fit(X_proc, y_proc, sample_weight=sample_weight)
        
        # Limpieza manual de referencias temporales
        del X_proc, y_proc
        gc.collect()
        return self

    def predict_proba(self, X):
        X_proc = np.ascontiguousarray(X, dtype=np.float32)
        return self.model.predict_proba(X_proc)

    def predict(self, X):
        X_proc = np.ascontiguousarray(X, dtype=np.float32)
        return self.model.predict(X_proc)