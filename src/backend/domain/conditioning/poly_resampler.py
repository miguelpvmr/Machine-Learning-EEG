import numpy as np
from scipy import signal
from ..interfaces.signal_processor_interface import ISignalProcessor
from typing import Tuple

class PolyResampler(ISignalProcessor):
    """
    High-performance polyphase resampler.
    Applies the resampling operation after the anti-aliasing Butterworth filter.
    """
    def __init__(self, target_fs: float = 250.0):
        self.target_fs = target_fs

    def process(self, data: np.ndarray, current_fs: float) -> Tuple[np.ndarray, float]:
        if current_fs == self.target_fs:
            return data, current_fs
            
        # Calculate up/down factors for polyphase resampling
        from math import gcd
        g = gcd(int(self.target_fs), int(current_fs))
        up = int(self.target_fs // g)
        down = int(current_fs // g)
        
        resampled_data = signal.resample_poly(data, up, down, axis=1)
        return resampled_data, self.target_fs