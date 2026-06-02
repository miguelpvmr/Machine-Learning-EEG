import numpy as np
import pywt
from scipy.stats import skew, kurtosis, entropy

def apply_robust_scaling(segment):
    """
    Applies robust scaling to the input EEG segment to mitigate the influence of 
    high-amplitude artifacts or ictal paroxysms.
    
    Parameters:
    segment (np.ndarray): EEG data of shape (channels, samples).
    
    Returns:
    np.ndarray: Scaled EEG data using local median and Interquartile Range (IQR).
    """
    q1, q2, q3 = np.percentile(segment, [25, 50, 75], axis=1, keepdims=True)
    iqr = q3 - q1
    # Add epsilon to prevent division by zero
    return (segment - q2) / (iqr + 1e-9)

def get_intensity_features(coeffs):
    """
    Computes intensity and morphological descriptors from wavelet coefficients.
    Includes RMS, Mean Power, Line Length, and Teager Energy Operator (TEO).
    
    Parameters:
    coeffs (np.ndarray): Array of wavelet coefficients for a specific band.
    
    Returns:
    list: [RMS, Power, Line Length, TEO]
    """
    pwr = np.mean(coeffs**2)
    rms = np.sqrt(pwr)
    line_length = np.mean(np.abs(np.diff(coeffs)))
    # Teager Energy Operator: sensitive to instantaneous frequency and amplitude changes
    teo = np.mean(coeffs[1:-1]**2 - coeffs[0:-2] * coeffs[2:])
    return [rms, pwr, line_length, teo]

def get_statistical_features(coeffs):
    """
    Calculates higher-order statistical moments to characterize the 
    distribution of the transform coefficients.
    
    Parameters:
    coeffs (np.ndarray): Array of wavelet coefficients.
    
    Returns:
    list: [Mean, Std Dev, Skewness, Kurtosis]
    """
    return [np.mean(coeffs), np.std(coeffs), skew(coeffs), kurtosis(coeffs)]

def get_complexity_features(coeffs):
    """
    Estimates non-linear dynamics and complexity through Shannon Entropy 
    and Katz Fractal Dimension.
    
    Parameters:
    coeffs (np.ndarray): Array of wavelet coefficients.
    
    Returns:
    list: [Shannon Entropy, Katz Fractal Dimension]
    """
    # Shannon Entropy based on 10-bin histogram probability distribution
    counts, _ = np.histogram(coeffs, bins=10)
    prob = counts / (len(coeffs) + 1e-9)
    ent = entropy(prob + 1e-9)
    
    # Katz Fractal Dimension (Dk)
    abs_diff = np.abs(np.diff(coeffs))
    L = np.sum(abs_diff) # Total length of the curve
    d = np.max(np.abs(coeffs - coeffs[0])) # Planar diameter
    
    # Dk calculation using log ratios
    dk = np.log10(L + 1e-9) / np.log10(d + 1e-9) if d > 0 else 1.0
    return [ent, dk]

def get_spatial_descriptors(scaled_window, ch_names):
    """
    Extracts spatial relationship features: Hemispheric Asymmetry, 
    Antero-Posterior Gradient, and Global Field Power.
    
    Parameters:
    scaled_window (np.ndarray): Scaled EEG window (channels, samples).
    ch_names (list): List of channel labels following the 10-20 system.
    
    Returns:
    list: [ASI, APG, GFP]
    """
    ch_map = {name: i for i, name in enumerate(ch_names)}
    
    # Define channel groups for spatial indexing
    left = [ch_map[c] for c in ch_names if any(x in c for x in ['1', '3', '5', '7', 'T3', 'T5'])]
    right = [ch_map[c] for c in ch_names if any(x in c for x in ['2', '4', '6', '8', 'T4', 'T6'])]
    ant = [ch_map[c] for c in ch_names if 'F' in c]
    post = [ch_map[c] for c in ch_names if 'P' in c or 'O' in c]
    
    def calculate_regional_power(idxs):
        return np.mean(scaled_window[idxs, :]**2) if idxs else 1e-9

    p_l, p_r = calculate_regional_power(left), calculate_regional_power(right)
    p_a, p_p = calculate_regional_power(ant), calculate_regional_power(post)
    
    # Asymmetry Index (ASI), Antero-Posterior Gradient (APG), Global Field Power (GFP)
    asi = (p_l - p_r) / (p_l + p_r + 1e-9)
    apg = (p_a - p_p) / (p_a + p_p + 1e-9)
    gfp = np.mean(np.std(scaled_window, axis=0))
    
    return [asi, apg, gfp]

def extract_comprehensive_features(raw_window, ch_names, wavelet='db4', level=5):
    """
    Main orchestration function to decompose EEG segments and aggregate 
    multi-domain features into a single observation vector.
    
    Parameters:
    raw_window (np.ndarray): Original EEG segment (channels, samples).
    ch_names (list): List of channel labels.
    wavelet (str): Mother wavelet (default 'db4').
    level (int): Decomposition depth (default 5).
    
    Returns:
    tuple: (feature_vector, feature_names)
    """
    # 1. Local Scaling
    window_scaled = apply_robust_scaling(raw_window)
    
    # 2. Spatial Domain Features
    spatial_data = get_spatial_descriptors(window_scaled, ch_names)
    spatial_labels = ['spatial_ASI', 'spatial_APG', 'spatial_GFP']
    
    # 3. Wavelet Domain Features (Time-Frequency)
    wavelet_data = []
    wavelet_labels = []
    
    bands = ['A5', 'D5', 'D4', 'D3', 'D2', 'D1']
    metrics = ['rms', 'pwr', 'll', 'teo', 'mean', 'std', 'skew', 'kurt', 'ent', 'dkatz']

    for i, channel in enumerate(ch_names):
        # Multiresolution decomposition using Mallat algorithm
        coeffs = pywt.wavedec(window_scaled[i, :], wavelet, level=level)
        
        for b_idx, b_coeffs in enumerate(coeffs):
            # Aggregate all atomic feature sets
            f1 = get_intensity_features(b_coeffs)
            f2 = get_statistical_features(b_coeffs)
            f3 = get_complexity_features(b_coeffs)
            
            wavelet_data.extend(f1 + f2 + f3)
            
            # Construct precise feature names for traceability
            for m_lab in metrics:
                wavelet_labels.append(f"{channel}_{bands[b_idx]}_{m_lab}")

    final_vector = np.concatenate([spatial_data, wavelet_data])
    final_names = spatial_labels + wavelet_labels
    
    return final_vector, final_names