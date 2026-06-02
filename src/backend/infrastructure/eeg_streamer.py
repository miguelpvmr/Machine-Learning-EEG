from abc import ABC, abstractmethod
import numpy as np
from typing import List, Tuple

class EEGStreamer(ABC):
    """
    Abstract Base Class defining the contract for any EEG hardware connection.
    
    Enforces Single Responsibility Principle (SRP) by isolating connection logic,
    polling, and hardware-specific configurations from the signal processing layer.
    """

    def __init__(self, target_fs: float = 250.0):
        """
        Args:
            target_fs: Expected sampling frequency in Hz (default: 250.0 based on TUSZ).
        """
        self.target_fs = target_fs
        self.hardware_fs: float = 0.0
        self.channel_names: List[str] = []

    @abstractmethod
    def connect(self) -> bool:
        """
        Establishes a connection with the hardware and initializes metadata.
        Must populate `self.hardware_fs` and `self.channel_names`.
        """
        pass

    @abstractmethod
    def fetch_chunk(self) -> Tuple[List[str], np.ndarray]:
        """
        Polls the hardware buffer.
        
        Returns:
            Tuple containing:
            - List of strings representing the channel names in this chunk.
            - np.ndarray: Matrix of shape (n_channels, n_samples).
        """
        pass

    @abstractmethod
    def disconnect(self) -> bool:
        """
        Gracefully closes the hardware connection and releases ports.
        """
        pass

    def validate_stream_parameters(self):
        """Validates that the hardware sampling frequency matches expectations."""
        if self.hardware_fs <= 0:
            raise RuntimeError("Hardware sampling frequency not detected.")