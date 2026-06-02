from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, Tuple, Any

class IClassificationModel(ABC):
    """
    Abstract contract for Machine Learning classification models.
    Ensures the controller is agnostic to the specific ML framework used.
    """
    
    @abstractmethod
    def predict(self, feature_vector: np.ndarray) -> Tuple[Any, Dict[str, float]]:
        """
        Args:
            feature_vector: 2D array of shape (1, n_features).
            
        Returns:
            Tuple containing:
            - The predicted class label (e.g., 'seiz' or 1).
            - A dictionary mapping class names to their respective probabilities.
        """
        pass