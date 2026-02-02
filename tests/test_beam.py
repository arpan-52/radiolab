"""
Tests for beam handling module.
"""

import numpy as np
import pytest
from astropy.io import fits

from radiolab.beam import (
    Beam, get_beam, compute_common_beam, 
    get_convolution_kernel, smooth_to_beam,
    check_common_resolution,
)


class TestBeam:
    """Tests for Beam dataclass."""
    
    def test_beam_creation(self):
        """Test basic beam creation."""
        beam = Beam(bmaj=0.01, bmin=0.005, bpa=45.0)
        
        assert beam.bmaj == 0.01
        assert beam.bmin == 0.005
        assert beam.bpa == 45.0
    
    def test_beam_arcsec(self):
        """Test arcsecond conversion."""
        beam = Beam(bmaj=1/3600, bmin=0.5/3600, bpa=0)  # 1" x 0.5"
        
        assert abs(beam.bmaj_arcsec - 1.0) < 1e-6
        assert abs(beam.bmin_arcsec - 0.5) < 1e-6
    
    def test_beam_from_header(self, sample_header):
        """Test extracting beam from header."""
        beam = Beam.from_header(sample_header)
        
        assert beam is not None
        assert abs(beam.bmaj_arcsec - 5.0) < 0.01
        assert abs(beam.bmin_arcsec - 3.0) < 0.01
        assert beam.bpa == 30.0
    
    def test_beam_area(self):
        """Test beam area calculation."""
        # 10" circular beam
        beam = Beam(bmaj=10/3600, bmin=10/3600, bpa=0)
        
        # Area in arcsec^2 should be pi * a * b / (4 ln 2)
        expected = np.pi * 10 * 10 / (4 * np.log(2))
        assert abs(beam.area_arcsec2 - expected) < 0.01
    
    def test_beam_encloses(self):
        """Test beam enclosure check."""
        large = Beam(bmaj=0.01, bmin=0.008, bpa=0)
        small = Beam(bmaj=0.005, bmin=0.004, bpa=0)
        
        assert large.encloses(small)
        assert not small.encloses(large)


class TestConvolutionKernel:
    """Tests for convolution kernel computation."""
    
    def test_kernel_computation(self):
        """Test that kernel is computed correctly."""
        current = Beam(bmaj=5/3600, bmin=3/3600, bpa=0)
        target = Beam(bmaj=10/3600, bmin=8/3600, bpa=0)
        pixscale = 1/3600  # 1 arcsec
        
        kernel = get_convolution_kernel(current, target, pixscale)
        
        assert kernel is not None
        # Kernel should be reasonably sized
        assert kernel.shape[0] > 10
        assert kernel.shape[1] > 10
    
    def test_kernel_none_when_equal(self):
        """Test that no kernel is returned when beams are equal."""
        beam = Beam(bmaj=5/3600, bmin=3/3600, bpa=0)
        pixscale = 1/3600
        
        kernel = get_convolution_kernel(beam, beam, pixscale)
        
        assert kernel is None
    
    def test_kernel_error_when_smaller(self):
        """Test error when target beam is smaller."""
        current = Beam(bmaj=10/3600, bmin=8/3600, bpa=0)
        target = Beam(bmaj=5/3600, bmin=3/3600, bpa=0)
        pixscale = 1/3600
        
        with pytest.raises(ValueError):
            get_convolution_kernel(current, target, pixscale)
    
    def test_kernel_with_different_pa(self):
        """Test kernel computation with different position angles."""
        # Current beam with PA=45°
        current = Beam(bmaj=5/3600, bmin=3/3600, bpa=45)
        # Target beam with PA=0° (larger)
        target = Beam(bmaj=10/3600, bmin=8/3600, bpa=0)
        pixscale = 1/3600  # 1 arcsec
        
        kernel = get_convolution_kernel(current, target, pixscale)
        
        assert kernel is not None
        # Kernel should be 2D with reasonable size
        assert kernel.shape[0] > 5
        assert kernel.shape[1] > 5
        # Kernel should be normalized (sum to ~1)
        assert abs(kernel.array.sum() - 1.0) < 0.01


class TestCommonBeam:
    """Tests for common beam computation."""
    
    def test_common_beam_single(self):
        """Test common beam with single input."""
        beam = Beam(bmaj=0.01, bmin=0.005, bpa=0)
        common = compute_common_beam([beam])
        
        # Should return something that encloses the input
        assert common.encloses(beam)
    
    def test_common_beam_multiple(self):
        """Test common beam with multiple inputs."""
        beams = [
            Beam(bmaj=0.01, bmin=0.005, bpa=0),
            Beam(bmaj=0.008, bmin=0.006, bpa=45),
            Beam(bmaj=0.012, bmin=0.004, bpa=90),
        ]
        common = compute_common_beam(beams)
        
        # Common beam should enclose all inputs
        for beam in beams:
            assert common.encloses(beam)
    
    def test_common_beam_with_target(self):
        """Test that target beam is returned when valid."""
        beams = [
            Beam(bmaj=0.01, bmin=0.005, bpa=0),
            Beam(bmaj=0.008, bmin=0.006, bpa=45),
        ]
        target = Beam(bmaj=0.02, bmin=0.015, bpa=0)
        
        common = compute_common_beam(beams, target=target)
        
        assert common.bmaj == target.bmaj
        assert common.bmin == target.bmin


class TestSmoothing:
    """Tests for image smoothing."""
    
    def test_smooth_to_beam(self, sample_image, sample_header):
        """Test smoothing an image to a larger beam."""
        # Target beam larger than current
        target = Beam(bmaj=10/3600, bmin=8/3600, bpa=0)
        
        smoothed, new_header = smooth_to_beam(sample_image, sample_header, target)
        
        # Shape should be preserved
        assert smoothed.shape == sample_image.shape
        
        # Peak should be reduced (smoothing spreads flux)
        assert smoothed.max() < sample_image.max()
        
        # Header should have new beam
        assert abs(new_header['BMAJ'] - target.bmaj) < 1e-10


class TestCheckCommonResolution:
    """Tests for resolution checking."""
    
    def test_same_resolution(self, sample_header):
        """Test detection of common resolution."""
        headers = [sample_header.copy() for _ in range(3)]
        
        common, beams = check_common_resolution(headers)
        
        assert common is True
        assert len(beams) == 3
    
    def test_different_resolution(self, sample_header):
        """Test detection of different resolutions."""
        h1 = sample_header.copy()
        h2 = sample_header.copy()
        h2['BMAJ'] = sample_header['BMAJ'] * 2  # Different beam
        
        common, beams = check_common_resolution([h1, h2])
        
        assert common is False
