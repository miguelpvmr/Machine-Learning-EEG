import numpy as np

class CircularBufferManager:
    """
    Manages a continuous, sliding-window memory buffer for real-time processing.
    
    Allocates static memory upfront and uses efficient array slicing to shift 
    data leftward as new samples arrive, maintaining the Overlap-Discard logic.
    """

    def __init__(self, n_channels: int, fs: int = 250, duration_sec: int = 8):
        """
        Args:
            n_channels: Number of validated channels (typically 19 for 10-20 system).
            fs: Sampling frequency in Hz.
            duration_sec: Total buffer length in seconds (includes pads + target window).
        """
        self.n_channels = n_channels
        self.fs = fs
        self.duration_sec = duration_sec
        self.max_samples = int(self.fs * self.duration_sec)
        
        # Pre-allocate static memory block (Float32 for performance)
        self.buffer = np.zeros((self.n_channels, self.max_samples), dtype=np.float32)
        self.current_samples = 0
        self._is_ready = False

    def update(self, new_data: np.ndarray):
        """
        Ingests a new chunk of data, sliding the buffer chronologically.
        
        Args:
            new_data: Matrix of shape (n_channels, n_new_samples).
        """
        n_new = new_data.shape[1]
        
        if n_new >= self.max_samples:
            # Edge case: Incoming chunk is larger than the entire buffer
            self.buffer = new_data[:, -self.max_samples:].copy()
            self.current_samples = self.max_samples
        else:
            # Standard sliding window logic: Shift left, insert right
            self.buffer[:, :-n_new] = self.buffer[:, n_new:]
            self.buffer[:, -n_new:] = new_data
            self.current_samples = min(self.max_samples, self.current_samples + n_new)
            
        if not self._is_ready and self.current_samples >= self.max_samples:
            self._is_ready = True

    def is_ready(self) -> bool:
        """Returns True if the initial 8-second warm-up is complete."""
        return self._is_ready

    def get_full_window(self) -> np.ndarray:
        """
        Retrieves the complete buffered data for downstream filtering.
        
        Returns:
            np.ndarray: A copy of the current 8-second memory state.
            
        Raises:
            RuntimeError: If queried before the initial buffer warm-up.
        """
        if not self._is_ready:
            raise RuntimeError("Buffer is warming up. 8 seconds of data not yet collected.")
        
        # Return a copy to prevent downstream filters from mutating the historical state
        return self.buffer.copy()