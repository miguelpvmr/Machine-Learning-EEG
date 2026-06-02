from abc import ABC, abstractmethod
import numpy as np

class ISpatialTransformer(ABC):
    """
    Interface for spatial operations (Montage re-referencing).
    """
    @abstractmethod
    def transform(self, signal: np.ndarray) -> np.ndarray:
        pass