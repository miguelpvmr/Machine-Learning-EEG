import numpy as np

class SpatialDescriptorCalculator:
    """Calculates ASI, APG, and GFP using predefined index maps for O(1) speed."""
    
    def __init__(self):
        # Indices based on the 18 derivations from LongitudinalBipolarMontage
        self.idx_left = [0, 1, 2, 3, 8, 9, 10, 11]
        self.idx_right = [4, 5, 6, 7, 12, 13, 14, 15]
        self.idx_ant = [0, 1, 4, 5, 8, 9, 12, 13]  # Frontal derivations
        self.idx_post = [2, 3, 6, 7, 10, 11, 14, 15] # Parietal/Occipital derivations

    def _power(self, segment: np.ndarray, indices: list) -> float:
        return float(np.mean(segment[indices, :]**2)) if indices else 1e-9

    def calculate(self, scaled_segment: np.ndarray) -> np.ndarray:
        p_l = self._power(scaled_segment, self.idx_left)
        p_r = self._power(scaled_segment, self.idx_right)
        p_a = self._power(scaled_segment, self.idx_ant)
        p_p = self._power(scaled_segment, self.idx_post)
        
        asi = (p_l - p_r) / (p_l + p_r + 1e-9)
        apg = (p_a - p_p) / (p_a + p_p + 1e-9)
        
        # GFP: Standard deviation across channels at each time point, then averaged
        gfp = float(np.mean(np.std(scaled_segment, axis=0)))
        
        return np.array([asi, apg, gfp], dtype=np.float32)