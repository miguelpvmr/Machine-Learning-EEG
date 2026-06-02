import numpy as np
from ..interfaces.spatial_transformer_interface import ISpatialTransformer

class LongitudinalBipolarMontage(ISpatialTransformer):
    """
    Applies the 'Double Banana' montage using strict O(1) Matrix Multiplication.
    Assumes the input signal has 19 channels ordered exactly as validated.
    """
    def __init__(self):
        # The 19 standard channels validated upstream
        self.channels = [
            'Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 
            'O1', 'O2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 
            'Fz', 'Cz', 'Pz'
        ]
        
        # The 18 bipolar derivations
        self.pairs = [
            ('Fp1', 'F7'), ('F7', 'T3'), ('T3', 'T5'), ('T5', 'O1'), # Left Temp
            ('Fp2', 'F8'), ('F8', 'T4'), ('T4', 'T6'), ('T6', 'O2'), # Right Temp
            ('Fp1', 'F3'), ('F3', 'C3'), ('C3', 'P3'), ('P3', 'O1'), # Left Para
            ('Fp2', 'F4'), ('F4', 'C4'), ('C4', 'P4'), ('P4', 'O2'), # Right Para
            ('Fz', 'Cz'), ('Cz', 'Pz')                               # Midline
        ]
        
        # Pre-compute the 18x19 Transformation Matrix M
        self.M = np.zeros((18, 19), dtype=np.float32)
        for row_idx, (anode, cathode) in enumerate(self.pairs):
            anode_idx = self.channels.index(anode)
            cathode_idx = self.channels.index(cathode)
            self.M[row_idx, anode_idx] = 1.0
            self.M[row_idx, cathode_idx] = -1.0

    def transform(self, data: np.ndarray) -> np.ndarray:
        """
        Calculates X_bipolar = M * X_raw.
        Returns matrix of shape (18, n_samples).
        """
        return self.M @ data