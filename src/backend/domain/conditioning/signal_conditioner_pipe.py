import numpy as np
import logging
from typing import List
from ..interfaces.signal_processor_interface import ISignalProcessor
from ..interfaces.spatial_transformer_interface import ISpatialTransformer

logger = logging.getLogger(__name__)

class SignalConditioner:
    """
    Coordinates the signal conditioning pipeline.
    Follows the Open/Closed Principle: you can inject any list of temporal filters.
    """
    
    def __init__(
        self, 
        temporal_pipeline: List[ISignalProcessor], 
        spatial_transformer: ISpatialTransformer,
        target_window_sec: float = 4.096,
        target_fs: float = 250.0
    ):
        self.temporal_pipeline = temporal_pipeline
        self.spatial_transformer = spatial_transformer
        self.target_window_sec = target_window_sec
        self.target_fs = target_fs
        
        # Calculate exactly how many samples we must output.
        # e.g., 4.096 sec * 250 Hz = 1024 samples exactly.
        self.expected_samples = int(self.target_window_sec * self.target_fs)

    def process(self, raw_8s_buffer: np.ndarray, original_fs: float) -> np.ndarray:
        """
        Executes the full conditioning sequence and extracts the central clean segment.
        """
        current_data = raw_8s_buffer
        current_fs = original_fs
        
        # 1. Apply Temporal Pipeline (Butterworth -> Notch -> PolyResampler)
        for processor in self.temporal_pipeline:
            current_data, current_fs = processor.process(current_data, current_fs)
            
        # 2. Apply Spatial Montage (X_bipolar = M * X_raw)
        bipolar_data = self.spatial_transformer.transform(current_data)
        
        # 3. Extract the central window to discard filter ringing
        final_segment = self._extract_center(bipolar_data, current_fs)
        
        # 4. Strict Validation (Fail-Fast Mechanism)
        self._validate_output(final_segment)
        
        return final_segment

    def _extract_center(self, data: np.ndarray, fs: float) -> np.ndarray:
        """
        Slices the array to keep only the central target window.
        """
        total_samples = data.shape[1]
        
        if total_samples < self.expected_samples:
            raise ValueError(f"Buffer too small: Have {total_samples} samples, need at least {self.expected_samples}.")
            
        start_idx = (total_samples - self.expected_samples) // 2
        end_idx = start_idx + self.expected_samples
        
        return data[:, start_idx:end_idx]
        
    def _validate_output(self, segment: np.ndarray):
        """
        Ensures the final segment is perfectly dimensioned for the downstream ML model.
        """
        actual_samples = segment.shape[1]
        if actual_samples != self.expected_samples:
            error_msg = (f"Dimension Error: Output length is {actual_samples} samples, "
                         f"but strictly expected {self.expected_samples} samples "
                         f"({self.target_window_sec}s at {self.target_fs}Hz).")
            logger.error(error_msg)
            raise RuntimeError(error_msg)