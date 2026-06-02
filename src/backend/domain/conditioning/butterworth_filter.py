import numpy as np
from scipy import signal
from ..interfaces.signal_processor_interface import ISignalProcessor
from typing import Tuple

class ButterworthFilter(ISignalProcessor):
    """Zero-phase IIR Butterworth bandpass filter using Second-Order Sections (SOS)."""
    
    def __init__(self, lowcut: float = 0.5, highcut: float = 70.0, order: int = 4):
        self.lowcut = lowcut
        self.highcut = highcut
        self.order = order

    def process(self, data: np.ndarray, fs: float) -> Tuple[np.ndarray, float]:
        # Design the SOS filter dynamically based on the input fs
        nyq = 0.5 * fs
        low = self.lowcut / nyq
        high = self.highcut / nyq
        sos = signal.butter(self.order, [low, high], btype='band', output='sos')
        
        # Apply zero-phase forward-backward filtering
        filtered_data = signal.sosfiltfilt(sos, data, axis=1)
        return filtered_data, fs