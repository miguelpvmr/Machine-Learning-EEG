from sklearn.naive_bayes import GaussianNB
from .abstractModel import AbstractModel
import numpy as np

class GaussianNBModel(AbstractModel):
    """
    Versión Bayesiana para Neuroingeniería.
    Intercepta parámetros personalizados para evitar conflictos con el 
    estimador base de scikit-learn.
    """
    def _init_model(self):
        # Recuperamos parámetros de suavizado para la inicialización
        self.var_smoothing = getattr(self, 'var_smoothing', 1e-9)
        self.decision_strategy = getattr(self, 'decision_strategy', 'map') 
        return GaussianNB(var_smoothing=self.var_smoothing)

    def set_params(self, **params):
        """
        Intercepta parámetros antes de que lleguen al GaussianNB interno.
        """
        # 1. Filtramos los parámetros que pertenecen al Wrapper y no a sklearn
        wrapper_params = ['use_weight_in_fit', 'decision_strategy']
        for p in wrapper_params:
            if p in params:
                # Los asignamos como atributos locales de la clase
                setattr(self, p, params.pop(p))
        
        # 2. Pasamos el resto (como var_smoothing) al AbstractModel/GaussianNB
        if params:
            super().set_params(**params)
        return self

    def fit(self, X, y, sample_weight=None):
        X_proc = np.ascontiguousarray(X, dtype=np.float32)
        y_proc = np.array(y, dtype=np.int8).ravel()

        # 1. Realidad Empírica para la Matriz de Costos (n / nj)
        classes, counts = np.unique(y_proc, return_counts=True)
        n_samples = len(y_proc)
        n_classes = len(classes)
        
        # Peso puro para el Riesgo Clínico
        empirical_weights = n_samples / counts

        self.cost_matrix = np.zeros((n_classes, n_classes), dtype=np.float32)
        for i in range(n_classes):
            for j in range(n_classes):
                if i != j:
                    self.cost_matrix[i, j] = empirical_weights[i]

        # 2. Configuración de Priors (Persistencia de la lógica del Trainer)
        if sample_weight is not None:
            sw_importance = np.array([sample_weight[y_proc == c].mean() for c in classes])
            new_priors = sw_importance / sw_importance.sum()
            self.model.set_params(priors=new_priors)
        else:
            self.model.set_params(priors=np.full(n_classes, 1.0 / n_classes))

        # 3. Decisión de Entrenamiento (Usa el parámetro filtrado en set_params)
        use_weight = getattr(self, 'use_weight_in_fit', False)

        if use_weight and sample_weight is not None:
            # Reciclamos empirical_weights ajustados al estándar n / (k * nj)
            fit_weights_class = empirical_weights * (1.0 / n_classes)
            weight_map = dict(zip(classes, fit_weights_class))
            emp_weight_vec = np.array([weight_map[c] for c in y_proc], dtype=np.float32)
            
            self.model.fit(X_proc, y_proc, sample_weight=emp_weight_vec)
        else:
            # Entrenamiento morfológicamente puro para señales EEG
            self.model.fit(X_proc, y_proc)
            
        return self

    def predict(self, X, decision_strategy=None):
        X_proc = np.ascontiguousarray(X, dtype=np.float32)
        # Prioridad: argumento directo > atributo de clase > valor por defecto
        strategy = decision_strategy or getattr(self, 'decision_strategy', 'map')
        
        if strategy == 'expected_risk' and self.cost_matrix is not None:
            probas = self.model.predict_proba(X_proc)
            # Aplicamos la Teoría de Decisión de Bayes: R(ai|x) = Σ L(cj, ai) P(cj|x)
            expected_risks = np.dot(probas, self.cost_matrix)
            return np.argmin(expected_risks, axis=1).astype(np.int8)
        
        return self.model.predict(X_proc).astype(np.int8)

    def predict_proba(self, X):
        X_proc = np.ascontiguousarray(X, dtype=np.float32)
        return self.model.predict_proba(X_proc)