import numpy as np

class FeatureExtractionEngine:
    """
    Coordinates the extraction pipeline:
    Scaling -> Spatial Descriptors -> Wavelet Features -> Concatenation.
    """
    def __init__(self, scaler, spatial_calc, wavelet_analyzer):
        self.scaler = scaler
        self.spatial_calc = spatial_calc
        self.wavelet_analyzer = wavelet_analyzer

    def extract(self, clean_segment: np.ndarray) -> np.ndarray:
        """
        Args:
            clean_segment: The 4.096s matrix from the SignalConditioner.
        Returns:
            np.ndarray: 1D feature vector ready for the Scikit-Learn Model.
        """
       
        scaled_segment = self.scaler.scale(clean_segment)  # 1. Local robust scaling
        spatial_features = self.spatial_calc.calculate(scaled_segment) # 2. Spatial Domain (ASI, APG, GFP)
        wavelet_features = self.wavelet_analyzer.analyze(scaled_segment) # 3. Time-Frequency Domain (DWT + Metrics)
        final_vector = np.concatenate([spatial_features, wavelet_features]) # 4. Construct Final Observation Vector
        return final_vector.reshape(1, -1) # Reshape to (1, 1083) as required by Scikit-Learn predict()