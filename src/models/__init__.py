# src/models/__init__.py

# Agregamos seguridad para que los imports relativos funcionen 
# incluso si se llama desde distintos puntos del proyecto
from .abstractModel import AbstractModel
from .decisionTreeModel import DecisionTree
from .logitModel import LogitModel
from .randomForest import RandomForest
from .gaussianNB import GaussianNBModel
from .XGboostModel import XGBoostModel
from .knnModel import KNNModel
from .svc import SVC

def get_model(model_name, **kwargs):
    """
    Factoría centralizada para los modelos de clasificación de bioseñales.
    
    Args:
        model_name (str): Identificador del modelo.
        **kwargs: Hiperparámetros que se pasarán al constructor del modelo.
    """
    _inventory = {
        'decision_tree': DecisionTree,
        'logit': LogitModel,
        'rf': RandomForest,
        'gaussian_nb': GaussianNBModel,
        'xgboost': XGBoostModel,
        'knn': KNNModel,
        'svc': SVC
    }
    
    model_class = _inventory.get(model_name.lower())
    
    if not model_class:
        raise ValueError(f"Modelo '{model_name}' no encontrado. "
                         f"Opciones válidas: {list(_inventory.keys())}")
    
    return model_class(**kwargs)

# Metadata del paquete
__all__ = [
    'AbstractModel',
    'DecisionTree',
    'LogitModel',
    'RandomForest',
    'GaussianNBModel',
    'XGBoostModel',
    'KNNModel',
    'SVC',
    'get_model'
]