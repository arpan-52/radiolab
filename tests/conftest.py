"""
Pytest fixtures for RAMP tests.
"""

import numpy as np
import pytest
from astropy.io import fits
import tempfile
import os


@pytest.fixture
def sample_header():
    """Create a sample FITS header with WCS and beam."""
    header = fits.Header()
    header['SIMPLE'] = True
    header['BITPIX'] = -32
    header['NAXIS'] = 2
    header['NAXIS1'] = 100
    header['NAXIS2'] = 100
    
    # WCS
    header['CTYPE1'] = 'RA---SIN'
    header['CRVAL1'] = 180.0  # RA in degrees
    header['CDELT1'] = -1.0 / 3600  # 1 arcsec pixels
    header['CRPIX1'] = 50.0
    header['CUNIT1'] = 'deg'
    
    header['CTYPE2'] = 'DEC--SIN'
    header['CRVAL2'] = 45.0  # Dec in degrees
    header['CDELT2'] = 1.0 / 3600  # 1 arcsec pixels
    header['CRPIX2'] = 50.0
    header['CUNIT2'] = 'deg'
    
    # Beam (5" x 3" at PA=30)
    header['BMAJ'] = 5.0 / 3600  # degrees
    header['BMIN'] = 3.0 / 3600  # degrees
    header['BPA'] = 30.0  # degrees
    
    # Frequency
    header['FREQ'] = 1.4e9  # 1.4 GHz
    
    header['BUNIT'] = 'JY/BEAM'
    
    return header


@pytest.fixture
def sample_image(sample_header):
    """Create a sample 2D image with a point source."""
    np.random.seed(42)
    
    nx = sample_header['NAXIS1']
    ny = sample_header['NAXIS2']
    
    # Background noise
    image = np.random.normal(0, 1e-3, (ny, nx)).astype(np.float32)
    
    # Add a Gaussian source in the center
    y, x = np.mgrid[:ny, :nx]
    cx, cy = nx // 2, ny // 2
    sigma = 5  # pixels
    amplitude = 1.0  # Jy
    
    source = amplitude * np.exp(-((x - cx)**2 + (y - cy)**2) / (2 * sigma**2))
    image += source
    
    return image


@pytest.fixture
def sample_cube(sample_header):
    """Create a sample spectral cube with power-law spectrum."""
    np.random.seed(42)
    
    nx = sample_header['NAXIS1']
    ny = sample_header['NAXIS2']
    nfreq = 5
    
    frequencies = np.array([1.0e9, 1.2e9, 1.4e9, 1.6e9, 1.8e9])  # Hz
    ref_freq = np.sqrt(frequencies.min() * frequencies.max())
    
    # Spectral index map (varies across image)
    y, x = np.mgrid[:ny, :nx]
    cx, cy = nx // 2, ny // 2
    
    # Background alpha = -0.7, varying to -1.5 at center
    alpha_map = -0.7 + -0.8 * np.exp(-((x - cx)**2 + (y - cy)**2) / (2 * 10**2))
    
    # Amplitude map (Gaussian source)
    sigma = 5
    amp_map = 1.0 * np.exp(-((x - cx)**2 + (y - cy)**2) / (2 * sigma**2))
    
    # Create cube
    cube = np.zeros((nfreq, ny, nx), dtype=np.float32)
    
    for i, freq in enumerate(frequencies):
        # S = S0 * (nu/nu0)^alpha
        cube[i] = amp_map * (freq / ref_freq) ** alpha_map
        # Add noise
        cube[i] += np.random.normal(0, 1e-3, (ny, nx))
    
    return cube, frequencies, alpha_map


@pytest.fixture
def temp_fits_file(sample_image, sample_header):
    """Create a temporary FITS file."""
    with tempfile.NamedTemporaryFile(suffix='.fits', delete=False) as f:
        fits.writeto(f.name, sample_image, sample_header, overwrite=True)
        yield f.name
    
    os.unlink(f.name)


@pytest.fixture
def temp_fits_cube(sample_cube, sample_header):
    """Create temporary FITS files for a spectral cube."""
    cube, frequencies, alpha_map = sample_cube
    nfreq = len(frequencies)
    
    files = []
    headers = []
    
    with tempfile.TemporaryDirectory() as tmpdir:
        for i, freq in enumerate(frequencies):
            filepath = os.path.join(tmpdir, f'image_{int(freq/1e6)}MHz.fits')
            
            header = sample_header.copy()
            header['FREQ'] = freq
            header['CRVAL3'] = freq
            header['CTYPE3'] = 'FREQ'
            
            fits.writeto(filepath, cube[i], header, overwrite=True)
            files.append(filepath)
            headers.append(header)
        
        yield files, frequencies, headers, alpha_map
