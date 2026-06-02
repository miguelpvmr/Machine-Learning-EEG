from abc import ABC, abstractmethod
from sklearn.base import BaseEstimator, ClassifierMixin
import copy

class AbstractModel(ABC, BaseEstimator, ClassifierMixin):
    def __init__(self, **params):
        """
        Clase base abstracta para modelos
        Mantenemos los parámetros planos para compatibilidad total con sklearn.
        """
        # Guardamos los parámetros directamente en el objeto
        for key, value in params.items():
            setattr(self, key, value)
        
        # Guardamos una referencia de los nombres de los parámetros
        self._param_names = list(params.keys())
        
        # Inicializamos el motor interno (XGBoost, DT, etc.)
        self.model = self._init_model()

    @property
    def params(self):
        """Atributo de conveniencia para acceder a los parámetros como diccionario."""
        return {name: getattr(self, name) for name in self._param_names}

    @abstractmethod
    def _init_model(self):
        """Instancia el motor subyacente."""
        pass

    def fit(self, X, y, sample_weight=None):
        """
        Entrenamiento con soporte para pesos (cost-sensitive learning).
        """
        # Si el modelo interno falla con sample_weight, lo capturamos
        try:
            if sample_weight is not None:
                self.model.fit(X, y, sample_weight=sample_weight)
            else:
                self.model.fit(X, y)
        except TypeError:
            # Si el modelo no soporta sample_weight (como algunos KNN), fit normal
            self.model.fit(X, y)
        
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def set_params(self, **params):
        """
        Actualiza los parámetros tanto del wrapper como del modelo interno.
        """
        for key, value in params.items():
            setattr(self, key, value)
            if key not in self._param_names:
                self._param_names.append(key)
        
        # Sincronizamos con el modelo interno si tiene el método
        if hasattr(self.model, 'set_params'):
            self.model.set_params(**params)
        else:
            # Si el modelo no es mutable (como algunos de PyTorch), 
            # lo reinicializamos con los nuevos parámetros
            self.model = self._init_model()
            
        return self

    def get_params(self, deep=True):
        """
        Crucial para sklearn.base.clone. 
        Debe retornar los parámetros del constructor.
        """
        return self.params