import numpy as np
from scipy import signal
from ..interfaces.signal_processor_interface import ISignalProcessor
from typing import Tuple

class NotchFilter(ISignalProcessor):
    """Zero-phase IIR Notch filter for powerline noise removal."""
    
    def __init__(self, freq: float = 60.0, quality_factor: float = 30.0):
        self.freq = freq
        self.q = quality_factor

    def process(self, data: np.ndarray, fs: float) -> Tuple[np.ndarray, float]:
        nyq = 0.5 * fs
        w0 = self.freq / nyq
        b, a = signal.iirnotch(w0, self.q)
        
        # Zero-phase filtering using b/a coefficients (safe for narrow notch)
        filtered_data = signal.filtfilt(b, a, data, axis=1)
        return filtered_data, fs