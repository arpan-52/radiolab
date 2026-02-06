"""
Spectral cube creation from images.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from astropy.io import fits

from .beam import (Beam, get_beam, compute_common_beam, smooth_to_beam,
                   check_common_resolution, check_pixel_scales, regrid_to_reference,
                   get_pixel_scale)
from .io import load_image, load_images, save_fits, crop_to_center


def crop_to_angular_size(
    data: np.ndarray,
    header: fits.Header,
    size_arcmin: float,
) -> Tuple[np.ndarray, fits.Header]:
    """
    Crop image to central region of given angular size.

    Parameters
    ----------
    data : np.ndarray
        Image data (2D or with leading dimensions).
    header : fits.Header
        FITS header with WCS.
    size_arcmin : float
        Size of the region in arcminutes (will be square).

    Returns
    -------
    cropped_data : np.ndarray
        Cropped image data.
    cropped_header : fits.Header
        Updated header with correct WCS.
    """
    # Get pixel scale in degrees
    dx, dy = get_pixel_scale(header)

    # Convert arcmin to degrees
    size_deg = size_arcmin / 60.0

    # Calculate number of pixels for this angular size
    npix_x = int(np.ceil(size_deg / dx))
    npix_y = int(np.ceil(size_deg / dy))

    # Use the smaller to ensure square and within bounds
    ny, nx = data.shape[-2:]
    npix = min(npix_x, npix_y, nx, ny)

    # Ensure even number for centering
    npix = npix - (npix % 2)

    print(f"  Cropping to {size_arcmin}' = {npix} pixels (pixel scale: {dx*3600:.2f}\")")

    # Use existing crop function
    return crop_to_center(data, header, npix)


def make_cube(
    source: Union[Dict[float, str], str, List[str]],
    frequencies: Optional[List[float]] = None,
    zoom: Optional[float] = None,
    target_beam: Optional[Beam] = None,
    output: Optional[str] = None,
) -> Tuple[np.ndarray, fits.Header, np.ndarray]:
    """
    Create a spectral cube from multiple frequency images.

    Pipeline order:
    1. Load images, sort by frequency
    2. Crop to angular size (if --zoom specified) - BEFORE regridding
    3. Regrid to common pixel scale (if different)
    4. Smooth to common beam
    5. Stack into cube

    Parameters
    ----------
    source : dict or str or list
        Either:
        - A dictionary {frequency_hz: filepath}
        - A glob pattern string (supports comma-separated patterns)
        - A list of file paths
    frequencies : list of float, optional
        Frequencies in Hz. If not provided, auto-detected from headers.
    zoom : float, optional
        If provided, crop all images to central region of this size in arcminutes.
        This is done BEFORE regridding for efficiency.
    target_beam : Beam, optional
        Target beam to smooth all images to. If None and images have different
        beams, automatically computes the common beam.
    output : str, optional
        If provided, save the cube to this path.

    Returns
    -------
    cube : np.ndarray
        3D spectral cube with shape (nfreq, ny, nx).
    header : fits.Header
        FITS header for the cube with proper WCS.
    freq_array : np.ndarray
        Array of frequencies in Hz corresponding to each plane.

    Examples
    --------
    Using a dictionary:

    >>> images = {
    ...     1.4e9: 'image_1400mhz.fits',
    ...     1.5e9: 'image_1500mhz.fits',
    ... }
    >>> cube, header, freqs = make_cube(images)

    Using a glob pattern with zoom in arcminutes:

    >>> cube, header, freqs = make_cube(
    ...     'images/*.fits',
    ...     zoom=10.0,  # 10 arcmin cutout
    ... )

    Multiple glob patterns:

    >>> cube, header, freqs = make_cube('band1/*.fits,band2/*.fits')
    """
    # Load images
    data_dict, header_dict = load_images(source, frequencies)

    if not data_dict:
        raise ValueError("No valid images loaded")

    # Sort by frequency
    sorted_freqs = sorted(data_dict.keys())
    freq_array = np.array(sorted_freqs)

    print(f"Loaded {len(sorted_freqs)} images")
    print(f"Frequency range: {freq_array.min()/1e9:.3f} - {freq_array.max()/1e9:.3f} GHz")

    # =========================================
    # STEP 1: Crop to angular size (BEFORE regridding)
    # =========================================
    if zoom is not None:
        print(f"\nCropping to central {zoom} arcmin:")
        for freq in sorted_freqs:
            data = data_dict[freq]
            header = header_dict[freq]
            cropped, new_header = crop_to_angular_size(data, header, zoom)
            data_dict[freq] = cropped
            header_dict[freq] = new_header

    # =========================================
    # STEP 2: Regrid to common pixel scale
    # =========================================
    headers = [header_dict[f] for f in sorted_freqs]
    common_pixscale, scales = check_pixel_scales(headers)

    print("\nPixel scales:")
    for freq, (dx, dy) in zip(sorted_freqs, scales):
        print(f"  {freq/1e9:.4f} GHz: {dx*3600:.3f}\" x {dy*3600:.3f}\"")

    if not common_pixscale:
        # Use first image as reference for WCS
        ref_freq = sorted_freqs[0]
        ref_header = header_dict[ref_freq]
        print(f"\nRegridding to reference WCS (from {ref_freq/1e9:.4f} GHz):")

        for freq in sorted_freqs:
            if freq == ref_freq:
                continue

            data = data_dict[freq]
            header = header_dict[freq]

            print(f"  Regridding {freq/1e9:.4f} GHz...")
            regridded, new_header = regrid_to_reference(data, header, ref_header)
            data_dict[freq] = regridded
            header_dict[freq] = new_header

        # Update headers list after regridding
        headers = [header_dict[f] for f in sorted_freqs]
    else:
        print("All images have same pixel scale")

    # =========================================
    # STEP 3: Smooth to common beam
    # =========================================
    common, beams = check_common_resolution(headers)

    print("\nBeams:")
    for freq, beam in zip(sorted_freqs, beams):
        print(f"  {freq/1e9:.4f} GHz: {beam}")

    if not common:
        print("Images have different beams - smoothing to common resolution")

        if target_beam is None:
            target_beam = compute_common_beam(beams)
        print(f"Target beam: {target_beam}")

        for freq in sorted_freqs:
            data = data_dict[freq]
            header = header_dict[freq]
            smoothed, new_header = smooth_to_beam(data, header, target_beam)
            data_dict[freq] = smoothed
            header_dict[freq] = new_header
    elif target_beam is not None:
        # User explicitly requested a target beam
        print(f"Smoothing to requested beam: {target_beam}")
        for freq in sorted_freqs:
            data = data_dict[freq]
            header = header_dict[freq]
            current_beam = get_beam(header)
            if current_beam and not target_beam.encloses(current_beam):
                raise ValueError(
                    f"Target beam {target_beam} cannot enclose beam at {freq/1e9:.3f} GHz"
                )
            smoothed, new_header = smooth_to_beam(data, header, target_beam)
            data_dict[freq] = smoothed
            header_dict[freq] = new_header

    # =========================================
    # STEP 4: Stack into cube
    # =========================================
    sample_data = data_dict[sorted_freqs[0]]
    ny, nx = sample_data.shape[-2:]
    nfreq = len(sorted_freqs)

    cube = np.zeros((nfreq, ny, nx), dtype=np.float32)
    for i, freq in enumerate(sorted_freqs):
        cube[i] = data_dict[freq]

    # Create header
    cube_header = _create_cube_header(
        header_dict[sorted_freqs[0]],
        freq_array,
        nfreq, ny, nx,
    )

    # Update beam if we smoothed
    if target_beam is not None:
        cube_header.update(target_beam.to_header_cards())
    elif beams:
        cube_header.update(beams[0].to_header_cards())

    # Save if requested
    if output is not None:
        print(f"\nSaving cube to: {output}")
        save_fits(cube, cube_header, output)

    print(f"Created cube with shape: {cube.shape}")
    return cube, cube_header, freq_array


def _create_cube_header(
    ref_header: fits.Header,
    frequencies: np.ndarray,
    nfreq: int,
    ny: int,
    nx: int,
) -> fits.Header:
    """Create FITS header for the spectral cube."""
    header = fits.Header()

    # Basic keywords
    header['SIMPLE'] = True
    header['BITPIX'] = -32
    header['NAXIS'] = 3
    header['NAXIS1'] = nx
    header['NAXIS2'] = ny
    header['NAXIS3'] = nfreq

    # Copy spatial WCS
    spatial_keys = [
        'CTYPE1', 'CRVAL1', 'CDELT1', 'CRPIX1', 'CUNIT1',
        'CTYPE2', 'CRVAL2', 'CDELT2', 'CRPIX2', 'CUNIT2',
        'EQUINOX', 'RADESYS',
    ]
    for key in spatial_keys:
        if key in ref_header:
            header[key] = ref_header[key]

    # Frequency axis
    header['CTYPE3'] = 'FREQ'
    header['CRVAL3'] = frequencies[0]  # Reference frequency
    header['CRPIX3'] = 1.0
    header['CUNIT3'] = 'Hz'

    # Calculate CDELT3 (frequency step)
    if nfreq > 1:
        # Use mean spacing (assumes roughly uniform)
        header['CDELT3'] = (frequencies[-1] - frequencies[0]) / (nfreq - 1)
    else:
        header['CDELT3'] = 1.0

    # Store actual frequencies in header (for non-uniform spacing)
    for i, freq in enumerate(frequencies):
        header[f'FREQ{i:04d}'] = (freq, f'Frequency of plane {i}')

    # Copy beam
    for key in ['BMAJ', 'BMIN', 'BPA']:
        if key in ref_header:
            header[key] = ref_header[key]

    # Copy other metadata
    for key in ['OBJECT', 'TELESCOP', 'INSTRUME', 'OBSERVER', 'BUNIT', 'BTYPE']:
        if key in ref_header:
            header[key] = ref_header[key]

    # Add history
    header['HISTORY'] = 'Created by RadioLab make_cube()'
    header['HISTORY'] = f'Number of frequency planes: {nfreq}'
    header['HISTORY'] = f'Frequency range: {frequencies.min()/1e9:.3f}-{frequencies.max()/1e9:.3f} GHz'

    return header


def validate_images(
    data_dict: Dict[float, np.ndarray],
    header_dict: Dict[float, fits.Header],
) -> bool:
    """
    Validate that all images have compatible WCS and dimensions.

    Parameters
    ----------
    data_dict : dict
        Dictionary mapping frequency -> image data.
    header_dict : dict
        Dictionary mapping frequency -> FITS header.

    Returns
    -------
    bool
        True if all images are compatible.

    Raises
    ------
    ValueError
        If images are incompatible.
    """
    if not data_dict:
        raise ValueError("No images to validate")

    freqs = list(data_dict.keys())
    ref_freq = freqs[0]
    ref_data = data_dict[ref_freq]
    ref_header = header_dict[ref_freq]

    ny_ref, nx_ref = ref_data.shape[-2:]

    for freq in freqs[1:]:
        data = data_dict[freq]
        ny, nx = data.shape[-2:]

        if nx != nx_ref or ny != ny_ref:
            raise ValueError(
                f"Image dimension mismatch: {freq/1e9:.3f} GHz has shape ({ny}, {nx}) "
                f"vs reference ({ny_ref}, {nx_ref})"
            )

        # Check WCS consistency
        header = header_dict[freq]
        for key in ['CDELT1', 'CDELT2', 'CRVAL1', 'CRVAL2']:
            if key in ref_header and key in header:
                if abs(header[key] - ref_header[key]) > 1e-10:
                    raise ValueError(
                        f"WCS mismatch for {key} at {freq/1e9:.3f} GHz"
                    )

    return True
