"""
Region-based spectral extraction.

Supports:
- Resolved spectra: spectrum per pixel within a region
- Integrated spectra: sum flux over region for a single spectrum
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from astropy.io import fits

try:
    from regions import Regions, PixCoord
    from regions import PolygonPixelRegion, CirclePixelRegion, EllipsePixelRegion
    HAS_REGIONS = True
except ImportError:
    HAS_REGIONS = False

from .beam import Beam, get_beam
from .spectral import fit_powerlaw, SpectralFitResult
from .utils import compute_rms


def load_region(
    region_file: Union[str, Path],
) -> 'Regions':
    """
    Load a DS9 or CRTF region file.
    
    Parameters
    ----------
    region_file : str or Path
        Path to region file.
        
    Returns
    -------
    Regions
        Astropy regions object.
        
    Raises
    ------
    ImportError
        If the regions package is not installed.
    ValueError
        If region file cannot be parsed.
    """
    if not HAS_REGIONS:
        raise ImportError(
            "The 'regions' package is required for region support. "
            "Install with: pip install regions"
        )
    
    region_file = Path(region_file)
    
    if not region_file.exists():
        raise FileNotFoundError(f"Region file not found: {region_file}")
    
    # Read and parse
    regions = Regions.read(region_file)
    
    return regions


def region_to_mask(
    region,
    shape: Tuple[int, int],
    header: Optional[fits.Header] = None,
) -> np.ndarray:
    """
    Convert a region to a boolean mask.
    
    Parameters
    ----------
    region : Region
        Astropy region object.
    shape : tuple
        (ny, nx) shape of the output mask.
    header : fits.Header, optional
        Header with WCS (required for sky regions).
        
    Returns
    -------
    np.ndarray
        Boolean mask where True = inside region.
    """
    if not HAS_REGIONS:
        raise ImportError("The 'regions' package is required")
    
    from astropy.wcs import WCS
    
    # Create WCS if header provided
    wcs = WCS(header).celestial if header is not None else None
    
    # Convert sky region to pixel region if needed
    if hasattr(region, 'to_pixel') and wcs is not None:
        region = region.to_pixel(wcs)
    
    # Create mask
    mask = region.to_mask(mode='center')
    
    # Apply to full image
    full_mask = np.zeros(shape, dtype=bool)
    
    if mask is not None:
        # Get bounding box
        bbox = mask.bbox
        slices = bbox.slices
        
        # Clip to image bounds
        y_slice = slice(max(0, slices[0].start), min(shape[0], slices[0].stop))
        x_slice = slice(max(0, slices[1].start), min(shape[1], slices[1].stop))
        
        # Get the corresponding part of the mask
        mask_data = mask.data
        
        # Adjust mask indices if we clipped
        my_start = max(0, -slices[0].start)
        my_stop = my_start + (y_slice.stop - y_slice.start)
        mx_start = max(0, -slices[1].start)
        mx_stop = mx_start + (x_slice.stop - x_slice.start)
        
        full_mask[y_slice, x_slice] = mask_data[my_start:my_stop, mx_start:mx_stop] > 0
    
    return full_mask


def create_polygon_mask(
    vertices: List[Tuple[float, float]],
    shape: Tuple[int, int],
) -> np.ndarray:
    """
    Create a polygon mask from vertices (no regions package needed).
    
    Parameters
    ----------
    vertices : list of tuple
        List of (x, y) pixel coordinates defining the polygon.
    shape : tuple
        (ny, nx) shape of the output mask.
        
    Returns
    -------
    np.ndarray
        Boolean mask.
    """
    from matplotlib.path import Path as MplPath
    
    ny, nx = shape
    
    # Create coordinate grid
    y, x = np.mgrid[:ny, :nx]
    points = np.column_stack([x.ravel(), y.ravel()])
    
    # Create path and check containment
    path = MplPath(vertices)
    mask = path.contains_points(points).reshape(shape)
    
    return mask


def extract_resolved_spectrum(
    cube: np.ndarray,
    frequencies: np.ndarray,
    region: Optional[np.ndarray] = None,
    region_file: Optional[str] = None,
    header: Optional[fits.Header] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract spectrum at each pixel within a region.
    
    Parameters
    ----------
    cube : np.ndarray
        Spectral cube with shape (nfreq, ny, nx).
    frequencies : np.ndarray
        Frequencies in Hz.
    region : np.ndarray, optional
        Boolean mask defining the region.
    region_file : str, optional
        Path to DS9/CRTF region file.
    header : fits.Header, optional
        Header with WCS (needed if using region_file with sky regions).
        
    Returns
    -------
    spectra : np.ndarray
        Spectra for each pixel, shape (n_pixels, nfreq).
    pixel_coords : np.ndarray
        Pixel coordinates (y, x) for each spectrum, shape (n_pixels, 2).
    """
    nfreq, ny, nx = cube.shape
    
    # Get mask
    if region is None and region_file is not None:
        regions = load_region(region_file)
        # Use first region if multiple
        region = region_to_mask(regions[0], (ny, nx), header)
    elif region is None:
        # Use all pixels
        region = np.ones((ny, nx), dtype=bool)
    
    # Extract pixels
    y_coords, x_coords = np.where(region)
    n_pixels = len(y_coords)
    
    spectra = np.zeros((n_pixels, nfreq), dtype=np.float32)
    for i in range(n_pixels):
        spectra[i] = cube[:, y_coords[i], x_coords[i]]
    
    pixel_coords = np.column_stack([y_coords, x_coords])
    
    return spectra, pixel_coords


def extract_integrated_spectrum(
    cube: np.ndarray,
    frequencies: np.ndarray,
    region: Optional[np.ndarray] = None,
    region_file: Optional[str] = None,
    header: Optional[fits.Header] = None,
    beam: Optional[Beam] = None,
    pixel_area: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Extract integrated spectrum by summing flux over a region.
    
    Parameters
    ----------
    cube : np.ndarray
        Spectral cube with shape (nfreq, ny, nx).
    frequencies : np.ndarray
        Frequencies in Hz.
    region : np.ndarray, optional
        Boolean mask defining the region.
    region_file : str, optional
        Path to DS9/CRTF region file.
    header : fits.Header, optional
        Header with WCS and beam info.
    beam : Beam, optional
        Beam for converting to flux density. If None, tries to read from header.
    pixel_area : float, optional
        Pixel area in steradians. If None, computed from header.
        
    Returns
    -------
    integrated_flux : np.ndarray
        Integrated flux at each frequency (Jy if properly calibrated).
    flux_error : np.ndarray
        Estimated error on integrated flux.
    n_pixels : int
        Number of pixels in the region.
    """
    nfreq, ny, nx = cube.shape
    
    # Get mask
    if region is None and region_file is not None:
        regions = load_region(region_file)
        region = region_to_mask(regions[0], (ny, nx), header)
    elif region is None:
        raise ValueError("Must provide either region mask or region_file")
    
    n_pixels = np.sum(region)
    
    # Get beam info for proper integration
    if beam is None and header is not None:
        beam = get_beam(header)
    
    # Get pixel area
    if pixel_area is None and header is not None:
        dx = abs(header.get('CDELT1', 1.0))
        dy = abs(header.get('CDELT2', 1.0))
        pixel_area = np.deg2rad(dx) * np.deg2rad(dy)  # steradians
    
    # Sum flux
    integrated_flux = np.zeros(nfreq, dtype=np.float64)
    for i in range(nfreq):
        plane = cube[i]
        integrated_flux[i] = np.nansum(plane[region])
    
    # Convert from sum(Jy/beam) to Jy
    # S_total = sum(S_pixel) * (pixel_area / beam_area)
    if beam is not None and pixel_area is not None:
        beam_area = beam.area  # steradians
        integrated_flux *= pixel_area / beam_area
    
    # Estimate error
    # Assuming uncorrelated noise: σ_total = σ_pixel * sqrt(N / N_beam)
    # where N_beam is the number of pixels per beam
    flux_error = np.zeros(nfreq, dtype=np.float64)
    for i in range(nfreq):
        plane = cube[i]
        rms = compute_rms(plane)
        
        if beam is not None and pixel_area is not None:
            # Effective independent measurements
            n_beams = n_pixels * pixel_area / beam.area
            flux_error[i] = rms * np.sqrt(n_beams) * pixel_area / beam.area
        else:
            flux_error[i] = rms * np.sqrt(n_pixels)
    
    return integrated_flux, flux_error, n_pixels


def fit_region_spectrum(
    cube: np.ndarray,
    frequencies: np.ndarray,
    region: Optional[np.ndarray] = None,
    region_file: Optional[str] = None,
    header: Optional[fits.Header] = None,
    mode: str = 'integrated',
    order: int = 1,
    reference_freq: Optional[float] = None,
    rms: Optional[float] = None,
    sigma: float = 3.0,
) -> Union[SpectralFitResult, Tuple[np.ndarray, np.ndarray, float]]:
    """
    Fit spectrum from a region.
    
    Parameters
    ----------
    cube : np.ndarray
        Spectral cube with shape (nfreq, ny, nx).
    frequencies : np.ndarray
        Frequencies in Hz.
    region : np.ndarray, optional
        Boolean mask defining the region.
    region_file : str, optional
        Path to DS9/CRTF region file.
    header : fits.Header, optional
        Header with WCS.
    mode : str
        'integrated': sum flux over region, fit single spectrum
        'resolved': fit spectrum for each pixel in region
    order : int
        Polynomial order for fitting.
    reference_freq : float, optional
        Reference frequency for fitting.
    rms : float, optional
        RMS for error estimation (for integrated mode).
    sigma : float
        Sigma threshold for resolved mode.
        
    Returns
    -------
    For 'integrated' mode:
        coefficients : np.ndarray
            Fitted coefficients.
        errors : np.ndarray
            Coefficient errors.
        chi2_reduced : float
            Reduced chi-squared.
            
    For 'resolved' mode:
        SpectralFitResult
            Full fit result object.
    """
    nfreq, ny, nx = cube.shape
    
    # Get mask
    if region is None and region_file is not None:
        regions = load_region(region_file)
        region = region_to_mask(regions[0], (ny, nx), header)
    elif region is None:
        raise ValueError("Must provide either region mask or region_file")
    
    if mode == 'integrated':
        # Extract integrated spectrum
        flux, flux_error, n_pixels = extract_integrated_spectrum(
            cube, frequencies, region=region, header=header
        )
        
        # Fit
        coefficients, errors, chi2_r = fit_powerlaw(
            frequencies, flux, 
            order=order,
            flux_error=flux_error,
            reference_freq=reference_freq,
        )
        
        print(f"Integrated over {n_pixels} pixels")
        print(f"Fitted spectrum: α = {coefficients[1] if len(coefficients) > 1 else 'N/A':.3f} "
              f"± {errors[1] if len(errors) > 1 else 'N/A':.3f}")
        
        return coefficients, errors, chi2_r
    
    elif mode == 'resolved':
        # Import here to avoid circular import
        from .spectral import fit_spectral_index
        
        # Create masked cube
        masked_cube = cube.copy()
        for i in range(nfreq):
            masked_cube[i][~region] = np.nan
        
        # Fit
        result = fit_spectral_index(
            masked_cube, frequencies,
            order=order,
            rms=rms,
            sigma=sigma,
            reference_freq=reference_freq,
            mask=region,
        )
        
        return result
    
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'integrated' or 'resolved'.")
