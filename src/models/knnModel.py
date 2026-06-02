import numpy as np
from sklearn.neighbors import KNeighborsClassifier as SklearnKNN
from .abstractModel import AbstractModel
import gc

class KNNModel(AbstractModel):
    """
    KNN Híbrido acelerado por RAPIDS cuML para señales EEG.
    Implementa interceptación de parámetros para evitar errores en motores estrictos (cuML).
    """

    def _init_model(self):
        # Hiperparámetros base
        self.use_gpu = getattr(self, 'use_gpu', False)
        self.metric = getattr(self, 'metric', 'euclidean')
        self.n_neighbors = int(getattr(self, 'n_neighbors', 5))
        self.leaf_size = getattr(self, 'leaf_size', 30)
        
        # Parámetros de inferencia clínica (Wrapper)
        self.voting_weight = getattr(self, 'voting_weight', 'none') 
        self.decision_strategy = getattr(self, 'decision_strategy', 'map')
        self.batch_size = getattr(self, 'batch_size', 25_000)

        if self.use_gpu:
            try:
                from cuml.neighbors import KNeighborsClassifier as CuMLKNN
                return CuMLKNN(
                    n_neighbors=self.n_neighbors,
                    metric=self.metric,
                    output_type='numpy' 
                )
            except ImportError:
                print(" -> Advertencia: cuML no hallado. Revirtiendo a CPU.")
                self.use_gpu = False

        return SklearnKNN(
            n_neighbors=self.n_neighbors,
            metric=self.metric,
            leaf_size=self.leaf_size,
            weights='uniform', 
            algorithm='auto',
            n_jobs=-1
        )

    def set_params(self, **params):
        """
        Sobrescribe set_params para filtrar parámetros del Wrapper vs Motor.
        """
        # 1. Definimos parámetros exclusivos del Wrapper
        wrapper_params = ['decision_strategy', 'voting_weight', 'batch_size']
        
        # 2. Los extraemos para que no lleguen al .set_params() de cuML/Sklearn
        for p_name in wrapper_params:
            if p_name in params:
                setattr(self, p_name, params.pop(p_name))
        
        # 3. 'n_neighbors' es compartido (lo queremos en el wrapper y en el motor)
        if 'n_neighbors' in params:
            self.n_neighbors = int(params['n_neighbors'])

        # 4. El resto se envía al motor (self.model) a través de la clase base
        if params:
            super().set_params(**params)
            
        return self

    def fit(self, X, y, sample_weight=None):
        X_proc = np.ascontiguousarray(X, dtype=np.float32)
        y_proc = np.array(y, dtype=np.int32).ravel()
        
        self.y_train_ = y_proc 
        self.classes_ = np.unique(y_proc)
        
        classes, counts = np.unique(y_proc, return_counts=True)
        # Ponderación para Riesgo Esperado
        empirical_weights = len(y_proc) / (len(classes) * counts)
        
        self.cost_matrix_ = np.tile(empirical_weights[:, np.newaxis], (1, len(classes)))
        np.fill_diagonal(self.cost_matrix_, 0) 

        self.model.fit(X_proc, y_proc)
        return self

    def _compute_proba_logic(self, X_batch, k, voting_weight=None):
        dist, idx = self.model.kneighbors(X_batch, n_neighbors=k)
        
        n_queries = X_batch.shape[0]
        n_classes = len(self.classes_)
        w_type = voting_weight if voting_weight is not None else self.voting_weight
        
        neighbor_classes = self.y_train_[idx]
        
        if w_type == 'inverse_distance':
            weights = 1.0 / (dist + 1e-10)
        else:
            weights = np.ones_like(dist)

        probas = np.zeros((n_queries, n_classes), dtype=np.float32)
        for c_idx in range(n_classes):
            probas[:, c_idx] = np.sum(weights * (neighbor_classes == c_idx), axis=1)
        
        return probas / (probas.sum(axis=1, keepdims=True) + 1e-10)

    def predict_proba(self, X, n_neighbors=None, voting_weight=None):
        k = int(n_neighbors) if n_neighbors is not None else int(self.n_neighbors)
        X_proc = np.ascontiguousarray(X, dtype=np.float32)
        n_samples = X_proc.shape[0]
        
        if n_samples <= self.batch_size:
            return self._compute_proba_logic(X_proc, k, voting_weight=voting_weight)

        probas = []
        for i in range(0, n_samples, self.batch_size):
            batch = X_proc[i:i + self.batch_size]
            probas.append(self._compute_proba_logic(batch, k, voting_weight=voting_weight))
            gc.collect()
        
        return np.vstack(probas)

    def predict(self, X, decision_strategy=None, n_neighbors=None, voting_weight=None):
        strategy = decision_strategy or self.decision_strategy
        probas = self.predict_proba(X, n_neighbors=n_neighbors, voting_weight=voting_weight)
        
        if strategy == 'expected_risk' and hasattr(self, 'cost_matrix_'):
            expected_risks = np.matmul(probas, self.cost_matrix_.T)
            return np.argmin(expected_risks, axis=1).astype(np.int8)
        
        return np.argmax(probas, axis=1).astype(np.int8)