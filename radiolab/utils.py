"""
Common utilities for RAMP.
"""

from typing import Optional, Tuple, Union

import numpy as np
from scipy import stats


def compute_rms(
    data: np.ndarray,
    method: str = 'mad',
    mask: Optional[np.ndarray] = None,
) -> float:
    """
    Compute RMS noise estimate from image data.
    
    Parameters
    ----------
    data : np.ndarray
        Image data.
    method : str
        Method to use:
        - 'mad': Median Absolute Deviation (robust to outliers)
        - 'std': Standard deviation after sigma-clipping
        - 'negative': RMS from negative pixels only
    mask : np.ndarray, optional
        Boolean mask where True = include pixel.
        
    Returns
    -------
    float
        Estimated RMS noise.
    """
    # Flatten and remove NaNs
    flat_data = data.flatten()
    flat_data = flat_data[~np.isnan(flat_data)]
    
    # Apply mask if provided
    if mask is not None:
        flat_mask = mask.flatten()
        flat_mask = flat_mask[~np.isnan(data.flatten())]
        flat_data = flat_data[flat_mask]
    
    if len(flat_data) == 0:
        return np.nan
    
    if method == 'mad':
        # MAD to sigma conversion: sigma = 1.4826 * MAD
        mad = np.median(np.abs(flat_data - np.median(flat_data)))
        return 1.4826 * mad
    
    elif method == 'std':
        # Sigma-clipped standard deviation
        for _ in range(3):  # 3 iterations of sigma clipping
            mean = np.mean(flat_data)
            std = np.std(flat_data)
            clip_mask = np.abs(flat_data - mean) < 3 * std
            flat_data = flat_data[clip_mask]
        return np.std(flat_data)
    
    elif method == 'negative':
        # Use negative pixels (source-free regions)
        neg_data = flat_data[flat_data < 0]
        if len(neg_data) == 0:
            # Fall back to MAD if no negative pixels
            return compute_rms(data, method='mad', mask=mask)
        return np.sqrt(np.mean(neg_data**2))
    
    else:
        raise ValueError(f"Unknown method: {method}")


def create_mask(
    data: np.ndarray,
    rms: float,
    sigma: float = 3.0,
    mode: str = 'above',
) -> np.ndarray:
    """
    Create a boolean mask based on signal threshold.
    
    Parameters
    ----------
    data : np.ndarray
        Image data.
    rms : float
        RMS noise level.
    sigma : float
        Sigma threshold.
    mode : str
        'above': mask pixels > sigma * rms
        'below': mask pixels < sigma * rms  
        'abs': mask pixels where |value| > sigma * rms
        
    Returns
    -------
    np.ndarray
        Boolean mask (True = signal detected).
    """
    threshold = sigma * rms
    
    if mode == 'above':
        return data > threshold
    elif mode == 'below':
        return data < threshold
    elif mode == 'abs':
        return np.abs(data) > threshold
    else:
        raise ValueError(f"Unknown mode: {mode}")


def weighted_mean(
    values: np.ndarray,
    weights: np.ndarray,
    axis: Optional[int] = None,
) -> Union[float, np.ndarray]:
    """
    Compute weighted mean.
    
    Parameters
    ----------
    values : np.ndarray
        Values to average.
    weights : np.ndarray
        Weights (e.g., 1/variance).
    axis : int, optional
        Axis along which to compute.
        
    Returns
    -------
    float or np.ndarray
        Weighted mean.
    """
    return np.sum(values * weights, axis=axis) / np.sum(weights, axis=axis)


def fwhm_to_sigma(fwhm: float) -> float:
    """Convert FWHM to Gaussian sigma."""
    return fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))


def sigma_to_fwhm(sigma: float) -> float:
    """Convert Gaussian sigma to FWHM."""
    return sigma * 2.0 * np.sqrt(2.0 * np.log(2.0))


def validate_wcs_compatible(
    headers: list,
    tolerance: float = 1e-6,
) -> bool:
    """
    Check if multiple headers have compatible WCS (same pixel grid).
    
    Parameters
    ----------
    headers : list of fits.Header
        Headers to compare.
    tolerance : float
        Tolerance for floating point comparison.
        
    Returns
    -------
    bool
        True if all WCS are compatible.
    """
    if len(headers) < 2:
        return True
    
    ref = headers[0]
    
    for h in headers[1:]:
        # Check dimensions
        if h.get('NAXIS1') != ref.get('NAXIS1'):
            return False
        if h.get('NAXIS2') != ref.get('NAXIS2'):
            return False
        
        # Check pixel scale
        if abs(h.get('CDELT1', 0) - ref.get('CDELT1', 0)) > tolerance:
            return False
        if abs(h.get('CDELT2', 0) - ref.get('CDELT2', 0)) > tolerance:
            return False
        
        # Check reference point
        if abs(h.get('CRVAL1', 0) - ref.get('CRVAL1', 0)) > tolerance:
            return False
        if abs(h.get('CRVAL2', 0) - ref.get('CRVAL2', 0)) > tolerance:
            return False
    
    return True
