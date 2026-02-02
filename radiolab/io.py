"""
FITS I/O utilities and header manipulation.
"""

import glob
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS


def load_image(path: Union[str, Path]) -> Tuple[np.ndarray, fits.Header]:
    """
    Load a single FITS image.
    
    Parameters
    ----------
    path : str or Path
        Path to the FITS file.
        
    Returns
    -------
    data : np.ndarray
        Image data (squeezed to 2D if possible).
    header : fits.Header
        FITS header.
    """
    path = Path(path)
    with fits.open(path) as hdul:
        data = hdul[0].data.copy()
        header = hdul[0].header.copy()
    
    # Squeeze to 2D if possible (remove degenerate axes)
    data = np.squeeze(data)
    
    return data, header


def load_images(
    source: Union[Dict[float, str], str, List[str]],
    frequencies: Optional[List[float]] = None,
    verbose: bool = True,
) -> Tuple[Dict[float, np.ndarray], Dict[float, fits.Header]]:
    """
    Load multiple FITS images.
    
    Parameters
    ----------
    source : dict or str or list
        Either:
        - A dictionary {frequency_hz: filepath}
        - A glob pattern string
        - A list of file paths
    frequencies : list of float, optional
        Frequencies in Hz. If not provided, will try to extract from headers
        or filenames automatically.
    verbose : bool
        Print information about loaded files.
        
    Returns
    -------
    data_dict : dict
        Dictionary mapping frequency -> image data.
    header_dict : dict
        Dictionary mapping frequency -> FITS header.
        
    Raises
    ------
    ValueError
        If frequencies cannot be determined.
    """
    data_dict = {}
    header_dict = {}
    
    if isinstance(source, dict):
        # Dictionary input: {freq: path}
        for freq, path in source.items():
            data, header = load_image(path)
            data_dict[float(freq)] = data
            header_dict[float(freq)] = header
        if verbose:
            print(f"Loaded {len(data_dict)} images from dictionary")
    elif isinstance(source, list):
        # List of files
        files = source
        
        if not files:
            raise ValueError("Empty file list provided")
        
        if verbose:
            print(f"Loading {len(files)} files from list")
        
        if frequencies is None:
            # Try to extract frequencies from headers first
            frequencies = []
            freq_sources = []
            
            for f in files:
                _, header = load_image(f)
                freq = freq_from_header(header)
                source_type = "header"
                
                if freq is None:
                    # Try filename as fallback
                    freq = parse_frequency_from_filename(Path(f).name)
                    source_type = "filename"
                
                if freq is None:
                    raise ValueError(
                        f"Could not extract frequency from {f}. "
                        "Please provide frequencies with -f option."
                    )
                
                frequencies.append(freq)
                freq_sources.append(source_type)
            
            if verbose:
                print("Auto-detected frequencies:")
                for f, freq, src in zip(files, frequencies, freq_sources):
                    print(f"  {Path(f).name}: {freq/1e9:.4f} GHz (from {src})")
        
        if len(files) != len(frequencies):
            raise ValueError(
                f"Number of files ({len(files)}) doesn't match "
                f"number of frequencies ({len(frequencies)})"
            )
        
        for freq, path in zip(frequencies, files):
            data, header = load_image(path)
            data_dict[float(freq)] = data
            header_dict[float(freq)] = header
    else:
        # Glob pattern string
        files = sorted(glob.glob(source))
        
        if not files:
            raise ValueError(f"No files found matching pattern: {source}")
        
        if verbose:
            print(f"Found {len(files)} files matching pattern")
        
        if frequencies is None:
            # Try to extract frequencies from headers first
            frequencies = []
            freq_sources = []
            
            for f in files:
                _, header = load_image(f)
                freq = freq_from_header(header)
                source_type = "header"
                
                if freq is None:
                    # Try filename as fallback
                    freq = parse_frequency_from_filename(Path(f).name)
                    source_type = "filename"
                
                if freq is None:
                    raise ValueError(
                        f"Could not extract frequency from {f}. "
                        "Please provide frequencies with -f option."
                    )
                
                frequencies.append(freq)
                freq_sources.append(source_type)
            
            if verbose:
                print("Auto-detected frequencies:")
                for f, freq, src in zip(files, frequencies, freq_sources):
                    print(f"  {Path(f).name}: {freq/1e9:.4f} GHz (from {src})")
        
        if len(files) != len(frequencies):
            raise ValueError(
                f"Number of files ({len(files)}) doesn't match "
                f"number of frequencies ({len(frequencies)})"
            )
        
        for freq, path in zip(frequencies, files):
            data, header = load_image(path)
            data_dict[float(freq)] = data
            header_dict[float(freq)] = header
    
    return data_dict, header_dict


def save_fits(
    data: np.ndarray,
    header: fits.Header,
    path: Union[str, Path],
    overwrite: bool = True,
) -> None:
    """
    Write data to a FITS file with proper header.
    
    Parameters
    ----------
    data : np.ndarray
        Image data to write.
    header : fits.Header
        FITS header.
    path : str or Path
        Output file path.
    overwrite : bool
        Whether to overwrite existing file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Update header dimensions
    header = header.copy()

    # Remove extra NAXISj keywords for higher dimensions
    for i in range(data.ndim + 1, 10):
        for key in [f'NAXIS{i}', f'CTYPE{i}', f'CRVAL{i}', f'CDELT{i}',
                    f'CRPIX{i}', f'CUNIT{i}', f'CROTA{i}']:
            if key in header:
                del header[key]

    # Set correct dimensions
    header['NAXIS'] = data.ndim
    for i, size in enumerate(data.shape[::-1], 1):
        header[f'NAXIS{i}'] = size

    fits.writeto(path, data.astype(np.float32), header, overwrite=overwrite)


def update_header(header: fits.Header, **kwargs) -> fits.Header:
    """
    Update header keywords.
    
    Parameters
    ----------
    header : fits.Header
        Original header.
    **kwargs
        Keyword-value pairs to update.
        
    Returns
    -------
    fits.Header
        Updated header copy.
    """
    header = header.copy()
    for key, value in kwargs.items():
        header[key.upper()] = value
    return header


def crop_to_center(
    data: np.ndarray,
    header: fits.Header,
    npix: int,
) -> Tuple[np.ndarray, fits.Header]:
    """
    Crop image to central npix × npix pixels and update WCS.
    
    Parameters
    ----------
    data : np.ndarray
        Image data (2D or higher).
    header : fits.Header
        FITS header with WCS.
    npix : int
        Size of the central region to keep.
        
    Returns
    -------
    cropped_data : np.ndarray
        Cropped image data.
    cropped_header : fits.Header
        Updated header with correct WCS.
        
    Raises
    ------
    ValueError
        If npix is larger than the image dimensions.
    """
    # Get spatial dimensions (last two axes)
    ny, nx = data.shape[-2:]
    
    if npix > min(nx, ny):
        raise ValueError(
            f"Requested crop size ({npix}) larger than image dimensions ({nx}×{ny})"
        )
    
    # Calculate crop boundaries
    x_start = (nx - npix) // 2
    y_start = (ny - npix) // 2
    x_end = x_start + npix
    y_end = y_start + npix
    
    # Crop data (handles any number of leading dimensions)
    cropped_data = data[..., y_start:y_end, x_start:x_end]
    
    # Update header
    header = header.copy()
    
    # Update NAXIS
    header['NAXIS1'] = npix
    header['NAXIS2'] = npix
    
    # Update CRPIX (reference pixel position)
    if 'CRPIX1' in header:
        header['CRPIX1'] = header['CRPIX1'] - x_start
    if 'CRPIX2' in header:
        header['CRPIX2'] = header['CRPIX2'] - y_start
    
    return cropped_data, header


def freq_from_header(header: fits.Header) -> Optional[float]:
    """
    Extract frequency from FITS header.
    
    Tries multiple common keywords and axis specifications,
    including WSClean and CASA output formats.
    
    Parameters
    ----------
    header : fits.Header
        FITS header.
        
    Returns
    -------
    float or None
        Frequency in Hz, or None if not found.
    """
    # Direct frequency keywords (most common)
    direct_keys = [
        'FREQ',       # Simple frequency
        'RESTFREQ',   # Rest frequency
        'RESTFRQ',    # Alternate spelling
        'REFFREQ',    # Reference frequency (WSClean)
        'OBSFREQ',    # Observed frequency
        'CRVAL3',     # Often frequency axis
        'CRVAL4',     # Sometimes frequency if Stokes is 3
    ]
    
    for key in direct_keys:
        if key in header:
            val = header[key]
            try:
                val = float(val)
                if val > 0:
                    return val
            except (TypeError, ValueError):
                continue
    
    # Check CTYPE axes to find frequency axis
    for axis in [3, 4, 5]:
        ctype_key = f'CTYPE{axis}'
        crval_key = f'CRVAL{axis}'
        
        if ctype_key in header:
            ctype = str(header[ctype_key]).upper()
            if 'FREQ' in ctype and crval_key in header:
                try:
                    val = float(header[crval_key])
                    if val > 0:
                        return val
                except (TypeError, ValueError):
                    continue
    
    # Check for WSClean MFS images with WSCNORMF keyword
    if 'WSCNORMF' in header:
        try:
            val = float(header['WSCNORMF'])
            if val > 0:
                return val
        except (TypeError, ValueError):
            pass
    
    return None


def parse_frequency_from_filename(
    filename: str,
    pattern: Optional[str] = None,
) -> Optional[float]:
    """
    Extract frequency from filename using regex pattern.
    
    Parameters
    ----------
    filename : str
        Filename (not full path).
    pattern : str, optional
        Regex pattern with a capturing group for the frequency.
        Default tries common patterns like '1400MHz', '1.4GHz', etc.
        
    Returns
    -------
    float or None
        Frequency in Hz, or None if not found.
    """
    if pattern:
        match = re.search(pattern, filename)
        if match:
            freq_str = match.group(1)
            # Try to parse as float
            try:
                return float(freq_str)
            except ValueError:
                return None
    
    # Default patterns
    patterns = [
        (r'(\d+(?:\.\d+)?)\s*[Gg][Hh][Zz]', 1e9),   # GHz
        (r'(\d+(?:\.\d+)?)\s*[Mm][Hh][Zz]', 1e6),   # MHz
        (r'(\d+(?:\.\d+)?)\s*[Kk][Hh][Zz]', 1e3),   # kHz
        (r'(\d+(?:\.\d+)?)\s*[Hh][Zz]', 1.0),       # Hz
    ]
    
    for pat, multiplier in patterns:
        match = re.search(pat, filename)
        if match:
            return float(match.group(1)) * multiplier
    
    return None
