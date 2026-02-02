"""
Tests for cube creation module.
"""

import numpy as np
import pytest
import tempfile
import os

from radiolab.cube import make_cube, validate_images
from radiolab.beam import Beam


class TestMakeCube:
    """Tests for cube creation."""
    
    def test_make_cube_from_dict(self, temp_fits_cube):
        """Test creating cube from dictionary."""
        files, frequencies, headers, _ = temp_fits_cube
        
        # Create dict mapping
        images = {freq: path for freq, path in zip(frequencies, files)}
        
        cube, header, freqs = make_cube(images)
        
        assert cube.shape[0] == len(frequencies)
        assert np.allclose(freqs, frequencies)
    
    def test_make_cube_with_zoom(self, temp_fits_cube):
        """Test cube creation with zoom/crop."""
        files, frequencies, headers, _ = temp_fits_cube
        images = {freq: path for freq, path in zip(frequencies, files)}
        
        zoom = 50  # Central 50x50 pixels
        cube, header, freqs = make_cube(images, zoom=zoom)
        
        assert cube.shape[1] == zoom
        assert cube.shape[2] == zoom
        
        # WCS should be updated
        # CRPIX should reflect the crop
        original_crpix1 = headers[0]['CRPIX1']
        new_crpix1 = header['CRPIX1']
        assert new_crpix1 != original_crpix1
    
    def test_make_cube_header(self, temp_fits_cube):
        """Test that cube header is correct."""
        files, frequencies, headers, _ = temp_fits_cube
        images = {freq: path for freq, path in zip(frequencies, files)}
        
        cube, header, freqs = make_cube(images)
        
        # Check essential keywords
        assert header['NAXIS'] == 3
        assert header['NAXIS3'] == len(frequencies)
        assert header['CTYPE3'] == 'FREQ'
        assert header['CRVAL3'] == frequencies[0]
        
        # Check beam is copied
        assert 'BMAJ' in header
        assert 'BMIN' in header
    
    def test_make_cube_saves_file(self, temp_fits_cube):
        """Test saving cube to file."""
        files, frequencies, headers, _ = temp_fits_cube
        images = {freq: path for freq, path in zip(frequencies, files)}
        
        with tempfile.NamedTemporaryFile(suffix='.fits', delete=False) as f:
            output = f.name
        
        try:
            cube, header, freqs = make_cube(images, output=output)
            
            assert os.path.exists(output)
            
            # Check saved file
            from astropy.io import fits
            with fits.open(output) as hdul:
                assert hdul[0].data.shape == cube.shape
        finally:
            if os.path.exists(output):
                os.unlink(output)
    
    def test_make_cube_frequency_order(self, temp_fits_cube):
        """Test that cube is ordered by frequency."""
        files, frequencies, headers, _ = temp_fits_cube
        
        # Create dict in random order
        shuffled = list(zip(frequencies, files))
        np.random.shuffle(shuffled)
        images = {freq: path for freq, path in shuffled}
        
        cube, header, freqs = make_cube(images)
        
        # Frequencies should be sorted
        assert np.all(np.diff(freqs) > 0)


class TestValidateImages:
    """Tests for image validation."""
    
    def test_validate_compatible_images(self):
        """Test validation of compatible images."""
        from astropy.io import fits
        
        # Create compatible mock data
        data_dict = {
            1e9: np.random.randn(100, 100),
            2e9: np.random.randn(100, 100),
        }
        
        header = fits.Header()
        header['NAXIS1'] = 100
        header['NAXIS2'] = 100
        header['CDELT1'] = -1/3600
        header['CDELT2'] = 1/3600
        header['CRVAL1'] = 180.0
        header['CRVAL2'] = 45.0
        
        header_dict = {
            1e9: header.copy(),
            2e9: header.copy(),
        }
        
        # Should not raise
        result = validate_images(data_dict, header_dict)
        assert result is True
    
    def test_validate_incompatible_dimensions(self):
        """Test validation fails for incompatible dimensions."""
        from astropy.io import fits
        
        data_dict = {
            1e9: np.random.randn(100, 100),
            2e9: np.random.randn(50, 50),  # Different size!
        }
        
        header_dict = {
            1e9: fits.Header({'NAXIS1': 100, 'NAXIS2': 100}),
            2e9: fits.Header({'NAXIS1': 50, 'NAXIS2': 50}),
        }
        
        with pytest.raises(ValueError, match="dimension mismatch"):
            validate_images(data_dict, header_dict)
