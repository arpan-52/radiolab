"""
Spectral index and curvature fitting.

Fits power-law spectra with configurable polynomial order:
- order=1: spectral index (α)
- order=2: spectral index + curvature (α, β)
- order=n: n-th order polynomial in log-log space
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from scipy import optimize
from astropy.io import fits

from .io import load_images
from .beam import check_common_resolution, compute_common_beam, smooth_to_beam, get_beam, Beam
from .utils import compute_rms, create_mask


@dataclass
class SpectralFitResult:
    """
    Result of spectral index fitting.

    Attributes
    ----------
    coefficients : np.ndarray
        Fitted coefficients with shape (order+1, ny, nx).
        coefficients[0] = log(S0), the amplitude at reference frequency
        coefficients[1] = α, the spectral index
        coefficients[2] = β, the curvature (if order >= 2)
        etc.
    errors : np.ndarray
        1-sigma errors on each coefficient, same shape as coefficients.
    reference_freq : float
        Reference frequency (ν₀) used in fitting, in Hz.
    mask : np.ndarray
        Boolean mask of fitted pixels (True = fitted).
    chi2 : np.ndarray
        Reduced χ² per pixel.
    rms_used : np.ndarray
        RMS values used for thresholding, one per frequency.
    frequencies : np.ndarray
        Array of frequencies used in the fit.
    """
    coefficients: np.ndarray
    errors: np.ndarray
    reference_freq: float
    mask: np.ndarray
    chi2: np.ndarray
    rms_used: np.ndarray
    frequencies: np.ndarray
    
    @property
    def log_amplitude(self) -> np.ndarray:
        """log(S₀) at reference frequency."""
        return self.coefficients[0]
    
    @property
    def amplitude(self) -> np.ndarray:
        """S₀ at reference frequency."""
        return np.exp(self.coefficients[0])
    
    @property
    def spectral_index(self) -> np.ndarray:
        """Spectral index α (first order term)."""
        if self.coefficients.shape[0] > 1:
            return self.coefficients[1]
        return np.full(self.coefficients.shape[1:], np.nan)
    
    @property
    def spectral_index_error(self) -> np.ndarray:
        """Error on spectral index."""
        if self.errors.shape[0] > 1:
            return self.errors[1]
        return np.full(self.errors.shape[1:], np.nan)
    
    @property
    def curvature(self) -> np.ndarray:
        """Spectral curvature β (second order term)."""
        if self.coefficients.shape[0] > 2:
            return self.coefficients[2]
        return np.full(self.coefficients.shape[1:], np.nan)
    
    @property
    def curvature_error(self) -> np.ndarray:
        """Error on curvature."""
        if self.errors.shape[0] > 2:
            return self.errors[2]
        return np.full(self.errors.shape[1:], np.nan)
    
    def get_coefficient(self, order: int) -> np.ndarray:
        """Get coefficient of given order."""
        if order < self.coefficients.shape[0]:
            return self.coefficients[order]
        return np.full(self.coefficients.shape[1:], np.nan)
    
    def get_coefficient_error(self, order: int) -> np.ndarray:
        """Get error on coefficient of given order."""
        if order < self.errors.shape[0]:
            return self.errors[order]
        return np.full(self.errors.shape[1:], np.nan)


def fit_powerlaw(
    frequencies: np.ndarray,
    flux: np.ndarray,
    order: int = 1,
    flux_error: Optional[np.ndarray] = None,
    reference_freq: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Fit a power-law spectrum in log-log space.
    
    Model: log(S) = Σᵢ aᵢ * log(ν/ν₀)^i
    
    Parameters
    ----------
    frequencies : np.ndarray
        Frequencies in Hz.
    flux : np.ndarray
        Flux values (same units as data, e.g., Jy).
    order : int
        Polynomial order (1 = spectral index only, 2 = with curvature, etc.)
    flux_error : np.ndarray, optional
        Errors on flux values. If None, assumes uniform weights.
    reference_freq : float, optional
        Reference frequency. If None, uses geometric mean.
        
    Returns
    -------
    coefficients : np.ndarray
        Fitted coefficients [a₀, a₁, a₂, ...].
    errors : np.ndarray
        1-sigma errors on coefficients.
    chi2_reduced : float
        Reduced chi-squared of the fit.
    """
    # Filter valid (positive) flux values
    valid = (flux > 0) & np.isfinite(flux) & np.isfinite(frequencies)
    if np.sum(valid) < order + 1:
        # Not enough points
        return (np.full(order + 1, np.nan), 
                np.full(order + 1, np.nan), 
                np.nan)
    
    freq_valid = frequencies[valid]
    flux_valid = flux[valid]
    
    # Reference frequency
    if reference_freq is None:
        reference_freq = np.sqrt(freq_valid.min() * freq_valid.max())
    
    # Log transform
    log_nu = np.log(freq_valid / reference_freq)
    log_S = np.log(flux_valid)
    
    # Weights
    if flux_error is not None:
        flux_err_valid = flux_error[valid]
        # Propagate to log space: d(log S) = dS / S
        log_S_err = flux_err_valid / flux_valid
        weights = 1.0 / log_S_err**2
    else:
        weights = np.ones_like(log_S)
    
    # Build design matrix for polynomial
    # [1, log_nu, log_nu^2, ...]
    X = np.column_stack([log_nu**i for i in range(order + 1)])
    
    # Weighted least squares: (X^T W X)^-1 X^T W y
    W = np.diag(weights)
    try:
        XtWX = X.T @ W @ X
        XtWy = X.T @ W @ log_S
        
        # Solve and get covariance
        cov = np.linalg.inv(XtWX)
        coefficients = cov @ XtWy
        errors = np.sqrt(np.diag(cov))
        
        # Compute chi-squared
        residuals = log_S - X @ coefficients
        chi2 = np.sum(weights * residuals**2)
        dof = len(log_S) - (order + 1)
        chi2_reduced = chi2 / dof if dof > 0 else np.nan
        
    except np.linalg.LinAlgError:
        return (np.full(order + 1, np.nan),
                np.full(order + 1, np.nan),
                np.nan)
    
    return coefficients, errors, chi2_reduced


def fit_spectral_index(
    cube_or_images: Union[np.ndarray, Dict[float, str], str],
    frequencies: Optional[np.ndarray] = None,
    order: int = 1,
    rms: Optional[float] = None,
    sigma: float = 3.0,
    reference_freq: Optional[float] = None,
    target_beam: Optional[Beam] = None,
    mask: Optional[np.ndarray] = None,
) -> SpectralFitResult:
    """
    Perform pixel-by-pixel spectral fitting.
    
    Parameters
    ----------
    cube_or_images : np.ndarray or dict or str
        Either:
        - 3D cube with shape (nfreq, ny, nx)
        - Dictionary {frequency_hz: filepath}
        - Glob pattern string
    frequencies : np.ndarray, optional
        Frequencies in Hz. Required if cube_or_images is an ndarray.
    order : int
        Polynomial order for the fit.
        - 1: spectral index only (α)
        - 2: spectral index + curvature (α, β)
        - n: n-th order polynomial
    rms : float, optional
        RMS noise level. If None, computed from the data.
    sigma : float
        Sigma threshold for masking. Pixels with max flux < sigma*rms
        are not fitted.
    reference_freq : float, optional
        Reference frequency for the fit. If None, uses geometric mean.
    target_beam : Beam, optional
        Target beam to smooth all images to before fitting.
    mask : np.ndarray, optional
        Boolean mask of pixels to fit. If provided, overrides sigma threshold.
        
    Returns
    -------
    SpectralFitResult
        Dataclass containing coefficient maps, error maps, and metadata.
        
    Examples
    --------
    Fit spectral index only:
    
    >>> result = fit_spectral_index(cube, frequencies, order=1)
    >>> alpha = result.spectral_index
    
    Fit with curvature:
    
    >>> result = fit_spectral_index(cube, frequencies, order=2)
    >>> alpha = result.spectral_index
    >>> beta = result.curvature
    
    Using images directly:
    
    >>> images = {1.4e9: 'im1.fits', 1.5e9: 'im2.fits'}
    >>> result = fit_spectral_index(images, order=1, sigma=5)
    """
    # Load/prepare data
    cube, frequencies, header = _prepare_data(
        cube_or_images, frequencies, target_beam
    )
    
    nfreq, ny, nx = cube.shape
    
    if nfreq < order + 1:
        raise ValueError(
            f"Need at least {order + 1} frequency planes for order-{order} fit, "
            f"but only have {nfreq}"
        )
    
    # Reference frequency
    if reference_freq is None:
        reference_freq = np.sqrt(frequencies.min() * frequencies.max())
    
    print(f"Fitting order-{order} polynomial (α" +
          (", β" if order >= 2 else "") +
          (f", + {order-2} higher terms" if order > 2 else "") + ")")
    print(f"Reference frequency: {reference_freq/1e9:.3f} GHz")

    # Compute per-frequency RMS
    if rms is None:
        rms_array = np.array([compute_rms(cube[i]) for i in range(nfreq)])
        print(f"Computed per-frequency RMS:")
        for i, (freq, r) in enumerate(zip(frequencies, rms_array)):
            print(f"  {freq/1e9:.4f} GHz: {r:.3e}")
    elif np.isscalar(rms):
        # Single RMS provided, use for all frequencies
        rms_array = np.full(nfreq, rms)
        print(f"Using provided RMS: {rms:.3e}")
    else:
        rms_array = np.asarray(rms)
        print(f"Using provided per-frequency RMS")

    # Create mask based on per-frequency SNR
    if mask is None:
        # Require all frequencies to be above sigma*rms
        snr_mask = np.ones((ny, nx), dtype=bool)
        for i in range(nfreq):
            snr_mask &= (cube[i] > sigma * rms_array[i])
        mask = snr_mask
        print(f"Pixels above {sigma}σ at all frequencies: {np.sum(mask)} / {ny*nx}")
    
    # Initialize output arrays
    coefficients = np.full((order + 1, ny, nx), np.nan, dtype=np.float32)
    errors = np.full((order + 1, ny, nx), np.nan, dtype=np.float32)
    chi2 = np.full((ny, nx), np.nan, dtype=np.float32)
    
    # Pixel-by-pixel fitting
    print("Fitting spectra...")
    n_fitted = 0
    n_total = np.sum(mask)

    # Debug: print first few fitted pixels to verify fitting
    debug_count = 0
    debug_max = 5

    for iy in range(ny):
        for ix in range(nx):
            if not mask[iy, ix]:
                continue

            flux = cube[:, iy, ix]

            # Skip if not enough valid points
            valid = (flux > 0) & np.isfinite(flux)
            if np.sum(valid) < order + 1:
                continue

            # Fit with per-frequency RMS as flux errors for proper weighting
            coef, err, chi2_r = fit_powerlaw(
                frequencies, flux,
                order=order,
                flux_error=rms_array,
                reference_freq=reference_freq,
            )

            # Debug output for first few pixels
            if debug_count < debug_max and np.isfinite(coef[1]):
                freq_ghz = frequencies / 1e9
                print(f"  Debug pixel ({iy},{ix}):")
                print(f"    Freq (GHz): {freq_ghz}")
                print(f"    Flux: {flux}")
                print(f"    Ratio S_high/S_low: {flux[1]/flux[0]:.4f}")
                print(f"    Spectral index α: {coef[1]:.3f}")
                debug_count += 1

            coefficients[:, iy, ix] = coef
            errors[:, iy, ix] = err
            chi2[iy, ix] = chi2_r
            n_fitted += 1

        # Progress indicator
        if (iy + 1) % max(1, ny // 10) == 0:
            print(f"  Progress: {100*(iy+1)//ny}%")

    print(f"Fitted {n_fitted} / {n_total} masked pixels")

    return SpectralFitResult(
        coefficients=coefficients,
        errors=errors,
        reference_freq=reference_freq,
        mask=mask,
        chi2=chi2,
        rms_used=rms_array,
        frequencies=frequencies,
    )


def _prepare_data(
    cube_or_images: Union[np.ndarray, Dict[float, str], str],
    frequencies: Optional[np.ndarray],
    target_beam: Optional[Beam],
) -> Tuple[np.ndarray, np.ndarray, Optional[fits.Header]]:
    """Prepare data for spectral fitting."""
    
    if isinstance(cube_or_images, np.ndarray):
        # Direct cube input
        if frequencies is None:
            raise ValueError("frequencies required when providing a cube array")
        return cube_or_images, np.asarray(frequencies), None
    
    # Load from images
    data_dict, header_dict = load_images(cube_or_images, 
                                         list(frequencies) if frequencies else None)
    
    sorted_freqs = sorted(data_dict.keys())
    freq_array = np.array(sorted_freqs)
    
    # Check and smooth beams
    headers = [header_dict[f] for f in sorted_freqs]
    common, beams = check_common_resolution(headers)
    
    if not common or target_beam is not None:
        if target_beam is None:
            target_beam = compute_common_beam(beams)
            print(f"Smoothing to common beam: {target_beam}")
        
        for freq in sorted_freqs:
            data = data_dict[freq]
            header = header_dict[freq]
            smoothed, new_header = smooth_to_beam(data, header, target_beam)
            data_dict[freq] = smoothed
            header_dict[freq] = new_header
    
    # Stack into cube
    nfreq = len(sorted_freqs)
    sample_data = data_dict[sorted_freqs[0]]
    ny, nx = sample_data.shape[-2:]
    
    cube = np.zeros((nfreq, ny, nx), dtype=np.float32)
    for i, freq in enumerate(sorted_freqs):
        cube[i] = data_dict[freq]
    
    return cube, freq_array, header_dict[sorted_freqs[0]]


def save_spectral_fit(
    result: SpectralFitResult,
    output_prefix: str,
    header: Optional[fits.Header] = None,
) -> List[str]:
    """
    Save spectral fit results to FITS files.
    
    Parameters
    ----------
    result : SpectralFitResult
        Fit result from fit_spectral_index.
    output_prefix : str
        Prefix for output filenames.
    header : fits.Header, optional
        Reference header for WCS.
        
    Returns
    -------
    list of str
        List of created filenames.
    """
    from .io import save_fits
    
    created_files = []
    
    # Coefficient names
    coef_names = ['amplitude', 'spectral_index', 'curvature']
    coef_names.extend([f'coef_{i}' for i in range(3, result.coefficients.shape[0])])
    
    for i, name in enumerate(coef_names[:result.coefficients.shape[0]]):
        # Coefficient map
        coef_file = f"{output_prefix}_{name}.fits"
        save_fits(result.coefficients[i], header or fits.Header(), coef_file)
        created_files.append(coef_file)
        
        # Error map
        err_file = f"{output_prefix}_{name}_error.fits"
        save_fits(result.errors[i], header or fits.Header(), err_file)
        created_files.append(err_file)
    
    # Chi-squared map
    chi2_file = f"{output_prefix}_chi2.fits"
    save_fits(result.chi2, header or fits.Header(), chi2_file)
    created_files.append(chi2_file)
    
    # Mask
    mask_file = f"{output_prefix}_mask.fits"
    save_fits(result.mask.astype(np.float32), header or fits.Header(), mask_file)
    created_files.append(mask_file)
    
    return created_files
