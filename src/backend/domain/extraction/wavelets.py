import numpy as np
import pywt
from scipy.stats import skew, kurtosis, entropy

class WaveletAnalyzer:
    """Decomposes the signal and extracts non-linear/statistical features."""
    
    def __init__(self, wavelet: str = 'db4', level: int = 5):
        self.wavelet = wavelet
        self.level = level

    def _extract_band_features(self, coeffs: np.ndarray) -> list:
        # Intensity
        pwr = np.mean(coeffs**2)
        rms = np.sqrt(pwr)
        ll = np.mean(np.abs(np.diff(coeffs)))
        teo = np.mean(coeffs[1:-1]**2 - coeffs[0:-2] * coeffs[2:])
        
        # Statistics
        mean_val = np.mean(coeffs)
        std_val = np.std(coeffs)
        skew_val = float(skew(coeffs))
        kurt_val = float(kurtosis(coeffs))
        
        # Complexity
        counts, _ = np.histogram(coeffs, bins=10)
        prob = counts / (len(coeffs) + 1e-9)
        ent = float(entropy(prob + 1e-9))
        
        abs_diff = np.abs(np.diff(coeffs))
        L = np.sum(abs_diff)
        d = np.max(np.abs(coeffs - coeffs[0]))
        dk = np.log10(L + 1e-9) / np.log10(d + 1e-9) if d > 0 else 1.0
        
        return [rms, pwr, ll, teo, mean_val, std_val, skew_val, kurt_val, ent, dk]

    def analyze(self, scaled_segment: np.ndarray) -> np.ndarray:
        """
        Processes all channels and sub-bands.
        Returns a flat 1D array of all wavelet features.
        """
        all_features = []
        n_channels = scaled_segment.shape[0]
        
        for i in range(n_channels):
            # Mallat algorithm decomposition
            bands = pywt.wavedec(scaled_segment[i, :], self.wavelet, level=self.level)
            for b_coeffs in bands:
                all_features.extend(self._extract_band_features(b_coeffs))
                
        return np.array(all_features, dtype=np.float32)