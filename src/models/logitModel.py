import numpy as np
from sklearn.linear_model import LogisticRegression as SklearnLogit
from .abstractModel import AbstractModel

class LogitModel(AbstractModel):
    """
    Regresión Logística Híbrida (CPU/GPU) para señales EEG de alta dimensionalidad.
    Implementa interceptación de parámetros incompatibles para el motor cuML.
    """

    def _init_model(self):
        # Hiperparámetros base
        self.use_gpu = getattr(self, 'use_gpu', False)
        self.C = getattr(self, 'C', 0.1)
        self.penalty = getattr(self, 'penalty', 'l2')
        self.tol = getattr(self, 'tol', 1e-3)
        self.max_iter = getattr(self, 'max_iter', 2000)

        if self.use_gpu:
            try:
                from cuml.linear_model import LogisticRegression as CuMLLogit
                return CuMLLogit(
                    C=self.C,
                    penalty=self.penalty,
                    tol=self.tol,
                    max_iter=self.max_iter,
                    output_type='numpy'
                    # cuML usa 'qn' o 'lbfgs' por defecto, mucho más rápidos en GPU que 'saga'
                )
            except ImportError:
                print(" -> Advertencia: cuML no hallado. Revirtiendo a CPU.")
                self.use_gpu = False

        # Fallback a Sklearn (CPU pura)
        return SklearnLogit(
            C=self.C,
            penalty=self.penalty,
            solver='saga',  # 'saga' es ideal para datasets grandes en CPU
            tol=self.tol,
            max_iter=self.max_iter,
            n_jobs=-1,      # Usamos todos los núcleos disponibles
            random_state=42,
            warm_start=True
        )

    def set_params(self, **params):
        """
        Sobrescribe set_params para proteger a cuML de parámetros exclusivos de Sklearn.
        """
        # Extraemos variables de control del Wrapper
        if 'use_gpu' in params:
            self.use_gpu = params.pop('use_gpu')

        # Si el motor activo es cuML, eliminamos parámetros que harían crashear el C++ backend
        if self.use_gpu:
            incompatible_cuml_params = ['solver', 'n_jobs', 'random_state', 'warm_start', 'multi_class']
            for p in incompatible_cuml_params:
                params.pop(p, None)

        if params:
            super().set_params(**params)
            
        return self

    def fit(self, X, y, sample_weight=None):
        X_proc = np.ascontiguousarray(X, dtype=np.float32)
        
        # cuML prefiere int32 para las etiquetas de clasificación, int8 puede generar TypeErrors
        y_dtype = np.int32 if self.use_gpu else np.int8
        y_proc = np.array(y, dtype=y_dtype).ravel()

        if sample_weight is not None:
            self.model.fit(X_proc, y_proc, sample_weight=sample_weight)
        else:
            self.model.fit(X_proc, y_proc)
            
        return self

    def predict(self, X):
        X_proc = np.ascontiguousarray(X, dtype=np.float32)
        return self.model.predict(X_proc)

    def predict_proba(self, X):
        X_proc = np.ascontiguousarray(X, dtype=np.float32)
        return self.model.predict_proba(X_proc)