"""
Beam handling and convolution kernel computation.

Provides tools for:
- Extracting beam information from FITS headers
- Computing common beams that enclose multiple input beams
- Computing convolution kernels to smooth to a target beam
- Smoothing images to a target resolution
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import numpy as np
from astropy.io import fits
from astropy.convolution import Gaussian2DKernel, convolve_fft
from scipy.ndimage import gaussian_filter


def _beam_to_covariance(sigma_maj: float, sigma_min: float, pa_rad: float) -> np.ndarray:
    """
    Convert beam parameters to 2D covariance matrix.
    
    Parameters
    ----------
    sigma_maj : float
        Major axis sigma (not FWHM).
    sigma_min : float
        Minor axis sigma (not FWHM).
    pa_rad : float
        Position angle in radians (astronomical convention: N through E).
        
    Returns
    -------
    np.ndarray
        2x2 covariance matrix.
    """
    # Convert astronomical PA (N through E = counterclockwise from +Y)
    # to math angle (counterclockwise from +X)
    # theta = 90° - PA (in image coordinates where Y is up)
    theta = np.pi / 2 - pa_rad
    
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    
    # Rotation matrix
    R = np.array([[cos_t, -sin_t],
                  [sin_t, cos_t]])
    
    # Diagonal variance matrix (major along first axis after rotation)
    D = np.array([[sigma_maj**2, 0],
                  [0, sigma_min**2]])
    
    # Covariance: Σ = R @ D @ R.T
    return R @ D @ R.T


def _covariance_to_beam(cov: np.ndarray) -> tuple:
    """
    Convert 2D covariance matrix to beam parameters.
    
    Parameters
    ----------
    cov : np.ndarray
        2x2 covariance matrix.
        
    Returns
    -------
    tuple
        (sigma_maj, sigma_min, pa_rad) where pa_rad is astronomical PA.
    """
    # Eigenvalue decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    
    # Sort by eigenvalue (largest first = major axis)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    sigma_maj = np.sqrt(eigenvalues[0])
    sigma_min = np.sqrt(eigenvalues[1])
    
    # Get angle of major axis eigenvector
    # eigenvectors[:, 0] is the major axis direction
    theta = np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0])
    
    # Convert math angle to astronomical PA
    # PA = 90° - theta
    pa_rad = np.pi / 2 - theta
    
    # Normalize PA to [0, 180) range
    while pa_rad < 0:
        pa_rad += np.pi
    while pa_rad >= np.pi:
        pa_rad -= np.pi
    
    return sigma_maj, sigma_min, pa_rad




@dataclass
class Beam:
    """
    Represents a Gaussian beam.
    
    Attributes
    ----------
    bmaj : float
        Major axis FWHM in degrees.
    bmin : float
        Minor axis FWHM in degrees.
    bpa : float
        Position angle in degrees (measured from North through East).
    """
    bmaj: float  # degrees
    bmin: float  # degrees
    bpa: float   # degrees
    
    @classmethod
    def from_header(cls, header: fits.Header) -> Optional['Beam']:
        """Create Beam from FITS header."""
        if 'BMAJ' in header and 'BMIN' in header:
            return cls(
                bmaj=float(header['BMAJ']),
                bmin=float(header['BMIN']),
                bpa=float(header.get('BPA', 0.0)),
            )
        return None
    
    @property
    def bmaj_arcsec(self) -> float:
        """Major axis in arcseconds."""
        return self.bmaj * 3600.0
    
    @property
    def bmin_arcsec(self) -> float:
        """Minor axis in arcseconds."""
        return self.bmin * 3600.0
    
    @property
    def area(self) -> float:
        """Beam solid angle in steradians."""
        # Gaussian beam area = pi * bmaj * bmin / (4 * ln(2))
        bmaj_rad = np.deg2rad(self.bmaj)
        bmin_rad = np.deg2rad(self.bmin)
        return np.pi * bmaj_rad * bmin_rad / (4 * np.log(2))
    
    @property 
    def area_arcsec2(self) -> float:
        """Beam area in square arcseconds."""
        return np.pi * self.bmaj_arcsec * self.bmin_arcsec / (4 * np.log(2))
    
    def __repr__(self) -> str:
        return (f"Beam(bmaj={self.bmaj_arcsec:.2f}\", "
                f"bmin={self.bmin_arcsec:.2f}\", "
                f"bpa={self.bpa:.1f}°)")
    
    def encloses(self, other: 'Beam', tolerance: float = 1e-6) -> bool:
        """Check if this beam can enclose another beam."""
        # Simple check: both axes must be >= other's axes
        # This is approximate; proper check would consider PA
        return (self.bmaj >= other.bmaj - tolerance and 
                self.bmin >= other.bmin - tolerance)
    
    def to_header_cards(self) -> dict:
        """Return dict of header keywords."""
        return {
            'BMAJ': self.bmaj,
            'BMIN': self.bmin,
            'BPA': self.bpa,
        }


def get_beam(header: fits.Header) -> Optional[Beam]:
    """
    Extract beam information from FITS header.
    
    Parameters
    ----------
    header : fits.Header
        FITS header.
        
    Returns
    -------
    Beam or None
        Beam object, or None if beam info not found.
    """
    return Beam.from_header(header)


def get_pixel_scale(header: fits.Header) -> Tuple[float, float]:
    """
    Get pixel scale in degrees from header.
    
    Returns
    -------
    (dx, dy) : tuple of float
        Pixel scale in degrees for x and y axes.
    """
    dx = abs(header.get('CDELT1', header.get('CD1_1', 1.0)))
    dy = abs(header.get('CDELT2', header.get('CD2_2', 1.0)))
    return dx, dy


def compute_common_beam(
    beams: List[Beam],
    target: Optional[Beam] = None,
) -> Beam:
    """
    Compute the smallest circular beam that encloses all input beams.
    
    Parameters
    ----------
    beams : list of Beam
        Input beams to enclose.
    target : Beam, optional
        If provided, use this as the target beam instead of computing.
        Will raise ValueError if target cannot enclose all input beams.
        
    Returns
    -------
    Beam
        Common beam that encloses all inputs.
        
    Raises
    ------
    ValueError
        If target beam is smaller than some input beams.
    """
    if not beams:
        raise ValueError("No beams provided")
    
    if target is not None:
        # Check that target can enclose all beams
        for beam in beams:
            if not target.encloses(beam):
                raise ValueError(
                    f"Target beam {target} cannot enclose input beam {beam}"
                )
        return target
    
    # Find maximum major axis
    max_bmaj = max(beam.bmaj for beam in beams)
    max_bmin = max(beam.bmin for beam in beams)
    
    # Use the larger of the two for a circular beam
    # Or take max of each axis for elliptical
    common_size = max(max_bmaj, max_bmin)
    
    # Add small padding to ensure we can always convolve
    common_size *= 1.01
    
    return Beam(bmaj=common_size, bmin=common_size, bpa=0.0)


def get_convolution_kernel(
    current_beam: Beam,
    target_beam: Beam,
    pixscale: float,
) -> Optional[Gaussian2DKernel]:
    """
    Compute Gaussian convolution kernel to go from current to target beam.
    
    Uses covariance matrix subtraction: Σ_target = Σ_current + Σ_kernel
    So: Σ_kernel = Σ_target - Σ_current
    
    This properly handles beams with different position angles.
    
    Parameters
    ----------
    current_beam : Beam
        Current beam of the image.
    target_beam : Beam
        Target beam to smooth to. The user specifies the desired PA via
        target_beam.bpa.
    pixscale : float
        Pixel scale in degrees.
        
    Returns
    -------
    Gaussian2DKernel or None
        Convolution kernel, or None if no smoothing needed
        (current already >= target).
        
    Raises
    ------
    ValueError
        If target beam is smaller than current beam (kernel would have
        negative variance).
    """
    # Convert FWHM to sigma: FWHM = 2 * sqrt(2 * ln(2)) * sigma
    fwhm_to_sigma = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    
    # Current beam sigmas in pixels
    current_sigma_maj = current_beam.bmaj * fwhm_to_sigma / pixscale
    current_sigma_min = current_beam.bmin * fwhm_to_sigma / pixscale
    current_pa_rad = np.deg2rad(current_beam.bpa)
    
    # Target beam sigmas in pixels
    target_sigma_maj = target_beam.bmaj * fwhm_to_sigma / pixscale
    target_sigma_min = target_beam.bmin * fwhm_to_sigma / pixscale
    target_pa_rad = np.deg2rad(target_beam.bpa)
    
    # Build covariance matrices
    cov_current = _beam_to_covariance(current_sigma_maj, current_sigma_min, current_pa_rad)
    cov_target = _beam_to_covariance(target_sigma_maj, target_sigma_min, target_pa_rad)
    
    # Kernel covariance: Σ_kernel = Σ_target - Σ_current
    cov_kernel = cov_target - cov_current
    
    # Check if kernel is positive semi-definite (smoothing is possible)
    eigenvalues = np.linalg.eigvalsh(cov_kernel)
    
    if np.any(eigenvalues < -1e-10):
        raise ValueError(
            f"Cannot smooth from {current_beam} to smaller beam {target_beam}. "
            f"Kernel covariance has negative eigenvalues: {eigenvalues}"
        )
    
    # If already at target resolution (within tolerance)
    # Check if kernel size is negligible (eigenvalues near zero)
    if np.all(eigenvalues < 0.01):
        return None
    
    # Clamp small negative eigenvalues to zero (numerical precision)
    eigenvalues = np.maximum(eigenvalues, 0)
    
    # Extract kernel beam parameters from covariance
    kernel_sigma_maj, kernel_sigma_min, kernel_pa_rad = _covariance_to_beam(cov_kernel)
    
    # Convert astronomical PA to theta for Gaussian2DKernel
    # Gaussian2DKernel theta: counterclockwise from +X axis
    # Astronomical PA: counterclockwise from +Y (North) through +X (East)
    # theta = 90° - PA
    kernel_theta = np.pi / 2 - kernel_pa_rad
    
    # Create kernel
    # Gaussian2DKernel: x_stddev is horizontal, y_stddev is vertical
    # We want major axis to be at angle theta from x-axis
    kernel = Gaussian2DKernel(
        x_stddev=kernel_sigma_min,  # minor axis
        y_stddev=kernel_sigma_maj,  # major axis
        theta=kernel_theta,
    )

    # Verify kernel is normalized (sums to 1) for flux conservation
    kernel_sum = np.sum(kernel.array)
    if abs(kernel_sum - 1.0) > 0.01:
        print(f"  WARNING: Kernel sum = {kernel_sum:.4f} (should be 1.0)")

    return kernel




def smooth_to_beam(
    data: np.ndarray,
    header: fits.Header,
    target_beam: Beam,
    inplace: bool = False,
) -> Tuple[np.ndarray, fits.Header]:
    """
    Smooth image to target beam resolution.
    
    Parameters
    ----------
    data : np.ndarray
        Image data (2D or with leading dimensions for freq/stokes).
    header : fits.Header
        FITS header with current beam info.
    target_beam : Beam
        Target beam to smooth to.
    inplace : bool
        If True, modify data in place (saves memory).
        
    Returns
    -------
    smoothed_data : np.ndarray
        Smoothed image data.
    new_header : fits.Header
        Updated header with new beam info.
    """
    current_beam = get_beam(header)
    
    if current_beam is None:
        raise ValueError("No beam information in header")
    
    dx, dy = get_pixel_scale(header)
    pixscale = np.sqrt(dx * dy)  # Geometric mean

    kernel = get_convolution_kernel(current_beam, target_beam, pixscale)

    if kernel is None:
        return data, header.copy()

    result = data if inplace else np.empty_like(data)

    nan_mask = np.isnan(data)
    data_filled = np.where(nan_mask, 0.0, data)

    # Jy/beam: convolution preserves pixel sum but beam area has grown, so
    # scale up to conserve integrated flux.
    beam_area_ratio = target_beam.area / current_beam.area

    if data.ndim == 2:
        smoothed = convolve_fft(data_filled, kernel,
                                normalize_kernel=True,
                                allow_huge=True)
        smoothed *= beam_area_ratio
        smoothed[nan_mask] = np.nan
        result[:] = smoothed
    else:
        for idx in np.ndindex(data.shape[:-2]):
            smoothed = convolve_fft(data_filled[idx], kernel,
                                    normalize_kernel=True,
                                    allow_huge=True)
            smoothed *= beam_area_ratio
            smoothed[nan_mask[idx]] = np.nan
            result[idx] = smoothed

    new_header = header.copy()
    new_header.update(target_beam.to_header_cards())
    
    return result, new_header


def check_common_resolution(headers: List[fits.Header]) -> Tuple[bool, List[Beam]]:
    """
    Check if all images have the same beam.
    
    Parameters
    ----------
    headers : list of fits.Header
        FITS headers to check.
        
    Returns
    -------
    common : bool
        True if all beams are identical (within tolerance).
    beams : list of Beam
        List of beam objects from each header.
    """
    beams = []
    for h in headers:
        beam = get_beam(h)
        if beam is None:
            raise ValueError("Missing beam information in one or more headers")
        beams.append(beam)
    
    if len(beams) <= 1:
        return True, beams
    
    # Compare all beams to the first one
    ref = beams[0]
    tolerance = 1e-8  # degrees
    
    for beam in beams[1:]:
        if (abs(beam.bmaj - ref.bmaj) > tolerance or
            abs(beam.bmin - ref.bmin) > tolerance or
            abs(beam.bpa - ref.bpa) > tolerance):
            return False, beams

    return True, beams


def check_pixel_scales(headers: List[fits.Header], tolerance: float = 1e-6) -> Tuple[bool, List[Tuple[float, float]]]:
    """
    Check if all images have the same pixel scale.

    Parameters
    ----------
    headers : list of fits.Header
        FITS headers to check.
    tolerance : float
        Relative tolerance for comparison.

    Returns
    -------
    common : bool
        True if all pixel scales are identical (within tolerance).
    scales : list of tuple
        List of (dx, dy) pixel scales in degrees.
    """
    scales = []
    for h in headers:
        dx, dy = get_pixel_scale(h)
        scales.append((dx, dy))

    if len(scales) <= 1:
        return True, scales

    ref_dx, ref_dy = scales[0]

    for dx, dy in scales[1:]:
        if (abs(dx - ref_dx) / ref_dx > tolerance or
            abs(dy - ref_dy) / ref_dy > tolerance):
            return False, scales

    return True, scales


def regrid_to_reference(
    data: np.ndarray,
    header: fits.Header,
    reference_header: fits.Header,
) -> Tuple[np.ndarray, fits.Header]:
    """
    Regrid image to match a reference WCS.

    Uses reproject for accurate flux-conserving interpolation.

    Parameters
    ----------
    data : np.ndarray
        Image data to regrid.
    header : fits.Header
        Header of input image.
    reference_header : fits.Header
        Header defining the target WCS grid.

    Returns
    -------
    regridded : np.ndarray
        Regridded image data.
    new_header : fits.Header
        Updated header with new WCS.
    """
    try:
        from reproject import reproject_interp
    except ImportError:
        raise ImportError(
            "The 'reproject' package is required for regridding. "
            "Install with: pip install reproject"
        )

    from astropy.wcs import WCS

    # Get input WCS (celestial only)
    input_wcs = WCS(header).celestial
    target_wcs = WCS(reference_header).celestial

    # Get target shape from reference header
    target_shape = (reference_header['NAXIS2'], reference_header['NAXIS1'])

    # Squeeze data if needed (remove degenerate axes)
    data_2d = np.squeeze(data)

    # Create input HDU-like object
    input_data = (data_2d, input_wcs)

    # Reproject
    regridded, footprint = reproject_interp(
        input_data,
        target_wcs,
        shape_out=target_shape,
        order='bilinear',
    )

    # Handle pixels outside footprint
    regridded[footprint == 0] = np.nan

    # Create new header with target WCS but preserve beam info
    new_header = reference_header.copy()

    # Preserve beam from original header
    for key in ['BMAJ', 'BMIN', 'BPA']:
        if key in header:
            new_header[key] = header[key]

    # Preserve frequency info from original header
    for key in ['FREQ', 'RESTFREQ', 'RESTFRQ', 'REFFREQ', 'CRVAL3', 'CRVAL4']:
        if key in header:
            new_header[key] = header[key]

    return regridded.astype(np.float32), new_header
