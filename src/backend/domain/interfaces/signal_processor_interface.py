from abc import ABC, abstractmethod
import numpy as np
from typing import Tuple

class ISignalProcessor(ABC):
    """
    Interface for temporal operations (Filters, Resampling).
    Returns the processed signal and the potentially updated sampling frequency.
    """
    @abstractmethod
    def process(self, signal: np.ndarray, fs: float) -> Tuple[np.ndarray, float]:
        pass
