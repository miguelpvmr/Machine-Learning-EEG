import numpy as np

class RobustScaler:
    """
    Applies robust scaling (Median and IQR) per channel to mitigate 
    high-amplitude artifacts.
    """
    def scale(self, segment: np.ndarray) -> np.ndarray:
        """
        Args:
            segment: Array of shape (channels, samples)
        Returns:
            np.ndarray: Scaled array.
        """
        # keepdims=True ensures the shapes align for broadcasting
        q1, q2, q3 = np.percentile(segment, [25, 50, 75], axis=1, keepdims=True)
        iqr = q3 - q1
        
        return (segment - q2) / (iqr + 1e-9)