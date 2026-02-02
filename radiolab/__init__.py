"""
RadioLab - Post-processing toolkit for radio astronomy images.

A Python library for:
- Spectral cube creation from multi-frequency images
- Spectral index and curvature fitting
- Beam handling and resolution matching
- Region-based spectral extraction
"""

__version__ = "0.1.0"

from .beam import Beam, get_beam, compute_common_beam, smooth_to_beam
from .cube import make_cube
from .spectral import fit_spectral_index, SpectralFitResult
from .regions import extract_resolved_spectrum, extract_integrated_spectrum
from .io import load_image, load_images, save_fits
from .utils import compute_rms, create_mask

__all__ = [
    # Version
    "__version__",
    # Beam
    "Beam",
    "get_beam",
    "compute_common_beam", 
    "smooth_to_beam",
    # Cube
    "make_cube",
    # Spectral
    "fit_spectral_index",
    "SpectralFitResult",
    # Regions
    "extract_resolved_spectrum",
    "extract_integrated_spectrum",
    # I/O
    "load_image",
    "load_images",
    "save_fits",
    # Utils
    "compute_rms",
    "create_mask",
]
