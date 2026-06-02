import time
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Optional
from .eeg_streamer import EEGStreamer

class VirtualStreamer(EEGStreamer):
    """
    A concrete implementation of a Virtual EEG hardware driver.
    
    Capabilities:
    1. Reads a historically recorded .parquet file and streams it chunk-by-chunk 
       in real-time, simulating a live patient. Loops automatically upon reaching EOF.
    2. Falls back to generating synthetic EEG-like noise if no file is provided.
    """

    def __init__(self, target_fs: float = 250.0, chunk_size_ms: int = 40, parquet_path: Optional[str] = None):
        super().__init__(target_fs)
        self.chunk_size_ms = chunk_size_ms
        self.parquet_path = Path(parquet_path) if parquet_path else None
        self.is_connected = False
        
        # Calculate exactly how many samples correspond to the requested milliseconds
        self.samples_per_chunk = int((self.chunk_size_ms / 1000.0) * self.target_fs)
        
        # Internal state for Parquet streaming
        self._data_matrix: Optional[np.ndarray] = None
        self._current_idx: int = 0
        
        # Default 10-20 channels in case we need to generate noise
        self.mock_channels = [
            'Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 
            'O1', 'O2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 
            'Fz', 'Cz', 'Pz'
        ]

    def connect(self) -> bool:
        print("VirtualStreamer: Initializing virtual hardware...")
        time.sleep(0.5) # Simulate hardware handshake latency
        
        if self.parquet_path and self.parquet_path.exists():
            print(f"VirtualStreamer: Mounting Parquet file '{self.parquet_path.name}'...")
            try:
                df = pd.read_parquet(self.parquet_path).select_dtypes(include=[np.number])
                
                # Assume the parquet columns are the channel names
                self.channel_names = df.columns.tolist()
                
                # Transpose to strictly match (n_channels, n_samples) format for DSP
                self._data_matrix = df.values.T.astype(np.float32)
                self.hardware_fs = self.target_fs
                print(f"VirtualStreamer: File loaded successfully. Shape: {self._data_matrix.shape}")
            except Exception as e:
                print(f"VirtualStreamer Error reading Parquet: {e}")
                self._fallback_to_noise()
        else:
            self._fallback_to_noise()
            
        self.is_connected = True
        self.validate_stream_parameters()
        return True

    def _fallback_to_noise(self):
        """Sets up the streamer to generate synthetic data if file loading fails."""
        print("VirtualStreamer: No valid Parquet file detected. Defaulting to Synthetic Noise Generator.")
        self.channel_names = self.mock_channels
        self.hardware_fs = self.target_fs
        self._data_matrix = None

    def fetch_chunk(self) -> Tuple[List[str], np.ndarray]:
        """
        Retrieves the next chunk of data, simulating network latency.
        """
        if not self.is_connected:
            return [], np.array([])
            
        # 1. Simulate real-time hardware polling delay
        time.sleep(self.chunk_size_ms / 1000.0)
        
        # 2. Extract Parquet Data
        if self._data_matrix is not None:
            end_idx = self._current_idx + self.samples_per_chunk
            
            # Automatic looping logic (Infinite stream)
            if end_idx > self._data_matrix.shape[1]:
                print("VirtualStreamer: End of file reached. Looping back to start.")
                self._current_idx = 0
                end_idx = self.samples_per_chunk
                
            chunk = self._data_matrix[:, self._current_idx:end_idx]
            self._current_idx = end_idx
            return self.channel_names, chunk
            
        # 3. Generate Synthetic Data
        else:
            noise_data = np.random.normal(
                loc=0.0, 
                scale=15.0, # Microvolts standard deviation
                size=(len(self.channel_names), self.samples_per_chunk)
            )
            return self.channel_names, noise_data.astype(np.float32)

    def disconnect(self) -> bool:
        print("VirtualStreamer: Powering down virtual hardware...")
        self.is_connected = False
        self._data_matrix = None # Free up RAM
        return True