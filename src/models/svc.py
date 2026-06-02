from sklearn.svm import SVC as SklearnSVC
from .abstractModel import AbstractModel 
import numpy as np
import gc
import warnings

# ----------------------------------------------------------------
# HIBRID SVC WRAPPER (CPU Sklearn / GPU RAPIDS)
# ----------------------------------------------------------------
class SVC(AbstractModel):
    """
    Wrapper híbrido  para Máquinas de Vectores de Soporte.
    Soporta ejecución dinámica en CPU o GPU (RAPIDS cuML).
    Optimizado para kernels lineales y no lineales (RBF, Poly) en alta dimensionalidad.
    """

    def _init_model(self):
        """
        Inicializa el motor. Los parámetros entre Sklearn y cuML 
        son idénticos, lo que facilita el mapeo directo.
        """
        # --- LIMPIEZA PROFUNDA DE ESTADO (Mitigación RuntimeError C++) ---
        if hasattr(self, 'model'):
            del self.model
            gc.collect()
        # -----------------------------------------------------------------

        self.use_gpu = getattr(self, 'use_gpu', False)
        
        # Extracción unificada de hiperparámetros base
        C = float(getattr(self, 'C', 1.0))
        kernel = getattr(self, 'kernel', 'rbf')
        gamma = getattr(self, 'gamma', 'scale')
        degree = int(getattr(self, 'degree', 3))
        coef0 = float(getattr(self, 'coef0', 0.0))
        probability = getattr(self, 'probability', True)
        
        # Parámetros de rendimiento y estabilidad (Los "fusibles")
        cache_size = int(getattr(self, 'cache_size', 1000))
        max_iter = int(getattr(self, 'max_iter', -1))  # -1 es el default sin límite
        tol = float(getattr(self, 'tol', 1e-3))        # 1e-3 es el estándar
        
        # Scikit-Learn usa random_state, pero para homogeneizar usamos tu semilla
        random_seed = int(getattr(self, 'random_state', getattr(self, 'seed', 42)))

        if self.use_gpu:
            try:
                from cuml.svm import SVC as CuMLSVC
            except ImportError:
                raise ImportError(
                    "RAPIDS cuML no está instalado en este entorno. "
                    "Instálalo o configura 'use_gpu=False' para usar CPU."
                )
            
            # Instanciación Motor GPU con parámetros de control
            return CuMLSVC(
                C=C,
                kernel=kernel,
                gamma=gamma,
                degree=degree,
                coef0=coef0,
                probability=probability,
                cache_size=cache_size,
                max_iter=max_iter,    # Inyectado
                tol=tol,              # Inyectado
                random_state=random_seed
            )
        else:
            # Instanciación Motor CPU con parámetros de control
            return SklearnSVC(
                C=C,
                kernel=kernel,
                gamma=gamma,
                degree=degree,
                coef0=coef0,
                probability=probability,
                cache_size=cache_size,
                max_iter=max_iter,    # Inyectado
                tol=tol,              # Inyectado
                random_state=random_seed
            )

    def fit(self, X, y, sample_weight=None):
        """
        Estandariza los inputs a float32 contiguos.
        Usa la arquitectura try/except para garantizar tolerancia a fallos
        si alguna versión específica de hardware rechaza los pesos.
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
            if sample_weight is not None:
                warnings.warn(f"El backend {'GPU' if self.use_gpu else 'CPU'} de SVC ignoró sample_weight.")
            self.model.fit(X_proc, y_proc)
        
        del X_proc, y_proc
        gc.collect()
        return self

    def predict_proba(self, X):
        """
        Retorna las probabilidades si probability=True fue activado.
        """
        if not getattr(self, 'probability', True):
            raise ValueError("El SVC fue entrenado con probability=False. Actívalo para usar ROC-AUC.")
            
        X_proc = np.ascontiguousarray(X, dtype=np.float32)
        return self.model.predict_proba(X_proc)
        
    def predict(self, X):
        X_proc = np.ascontiguousarray(X, dtype=np.float32)
        return self.model.predict(X_proc)