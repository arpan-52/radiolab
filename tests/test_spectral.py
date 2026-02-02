"""
Tests for spectral fitting module.
"""

import numpy as np
import pytest

from radiolab.spectral import fit_powerlaw, fit_spectral_index, SpectralFitResult


class TestFitPowerlaw:
    """Tests for single-pixel power-law fitting."""
    
    def test_fit_spectral_index_only(self):
        """Test fitting spectral index (order=1)."""
        frequencies = np.array([1.0e9, 1.2e9, 1.4e9, 1.6e9, 1.8e9])
        ref_freq = np.sqrt(frequencies.min() * frequencies.max())
        
        # Create power-law spectrum with alpha = -0.7
        alpha_true = -0.7
        S0 = 1.0  # Jy
        flux = S0 * (frequencies / ref_freq) ** alpha_true
        
        coeffs, errors, chi2 = fit_powerlaw(frequencies, flux, order=1)
        
        # Check spectral index
        assert len(coeffs) == 2  # [log(S0), alpha]
        assert abs(coeffs[1] - alpha_true) < 0.01
        assert chi2 < 0.1  # Should be very good fit
    
    def test_fit_with_curvature(self):
        """Test fitting spectral index + curvature (order=2)."""
        frequencies = np.array([1.0e9, 1.2e9, 1.4e9, 1.6e9, 1.8e9])
        ref_freq = np.sqrt(frequencies.min() * frequencies.max())
        
        # Create curved spectrum
        alpha_true = -0.7
        beta_true = -0.3
        S0 = 1.0
        
        log_nu = np.log(frequencies / ref_freq)
        log_S = np.log(S0) + alpha_true * log_nu + beta_true * log_nu**2
        flux = np.exp(log_S)
        
        coeffs, errors, chi2 = fit_powerlaw(frequencies, flux, order=2)
        
        assert len(coeffs) == 3  # [log(S0), alpha, beta]
        assert abs(coeffs[1] - alpha_true) < 0.01
        assert abs(coeffs[2] - beta_true) < 0.01
    
    def test_fit_with_noise(self):
        """Test fitting with noisy data."""
        np.random.seed(42)
        
        frequencies = np.array([1.0e9, 1.2e9, 1.4e9, 1.6e9, 1.8e9])
        ref_freq = np.sqrt(frequencies.min() * frequencies.max())
        
        alpha_true = -0.7
        S0 = 1.0
        flux = S0 * (frequencies / ref_freq) ** alpha_true
        
        # Add 5% noise
        flux_noisy = flux * (1 + np.random.normal(0, 0.05, len(flux)))
        
        coeffs, errors, chi2 = fit_powerlaw(frequencies, flux_noisy, order=1)
        
        # Should still recover alpha within errors
        assert abs(coeffs[1] - alpha_true) < 3 * errors[1]
    
    def test_fit_insufficient_points(self):
        """Test behavior with too few points."""
        frequencies = np.array([1.0e9])
        flux = np.array([1.0])
        
        coeffs, errors, chi2 = fit_powerlaw(frequencies, flux, order=1)
        
        # Should return NaN
        assert np.all(np.isnan(coeffs))
    
    def test_fit_negative_flux(self):
        """Test handling of negative flux values."""
        frequencies = np.array([1.0e9, 1.2e9, 1.4e9, -1.0, 1.8e9])  # One invalid
        flux = np.array([1.0, 0.9, 0.8, -0.1, 0.6])  # One negative
        
        coeffs, errors, chi2 = fit_powerlaw(frequencies, flux, order=1)
        
        # Should skip invalid points but still fit
        assert not np.all(np.isnan(coeffs))


class TestFitSpectralIndex:
    """Tests for pixel-by-pixel spectral fitting."""
    
    def test_fit_cube(self, sample_cube):
        """Test fitting a spectral cube."""
        cube, frequencies, alpha_true = sample_cube
        
        result = fit_spectral_index(cube, frequencies, order=1, sigma=3)
        
        assert isinstance(result, SpectralFitResult)
        assert result.spectral_index.shape == cube.shape[1:]
        assert result.spectral_index_error.shape == cube.shape[1:]
    
    def test_result_properties(self, sample_cube):
        """Test SpectralFitResult properties."""
        cube, frequencies, alpha_true = sample_cube
        
        result = fit_spectral_index(cube, frequencies, order=2, sigma=3)
        
        # Check properties exist
        assert result.spectral_index is not None
        assert result.curvature is not None
        assert result.amplitude is not None
        assert result.chi2 is not None
    
    def test_recovery_of_spectral_index(self, sample_cube):
        """Test that we recover the input spectral index."""
        cube, frequencies, alpha_true = sample_cube
        ny, nx = cube.shape[1:]
        
        result = fit_spectral_index(cube, frequencies, order=1, sigma=5)
        
        # Check center pixel where signal is strong
        cy, cx = ny // 2, nx // 2
        
        if result.mask[cy, cx]:
            fitted_alpha = result.spectral_index[cy, cx]
            true_alpha = alpha_true[cy, cx]
            
            # Should be within 10% or reasonable error
            assert abs(fitted_alpha - true_alpha) < 0.2 or \
                   abs(fitted_alpha - true_alpha) < 3 * result.spectral_index_error[cy, cx]
    
    def test_masking(self, sample_cube):
        """Test that low-signal pixels are masked."""
        cube, frequencies, _ = sample_cube
        
        # Use a high sigma threshold
        result = fit_spectral_index(cube, frequencies, order=1, sigma=10)
        
        # Not all pixels should be fitted
        assert np.sum(result.mask) < result.mask.size
        
        # Unmasked pixels should have NaN
        assert np.all(np.isnan(result.spectral_index[~result.mask]))
    
    def test_higher_order(self, sample_cube):
        """Test fitting with higher polynomial orders."""
        cube, frequencies, _ = sample_cube
        
        # Order 3
        result = fit_spectral_index(cube, frequencies, order=3, sigma=3)
        
        assert result.coefficients.shape[0] == 4  # 0, 1, 2, 3 coefficients
        assert result.get_coefficient(3) is not None


class TestSpectralFitResult:
    """Tests for SpectralFitResult dataclass."""
    
    def test_get_coefficient(self):
        """Test get_coefficient method."""
        coeffs = np.random.randn(3, 10, 10).astype(np.float32)
        errors = np.abs(np.random.randn(3, 10, 10)).astype(np.float32)
        
        result = SpectralFitResult(
            coefficients=coeffs,
            errors=errors,
            reference_freq=1.4e9,
            mask=np.ones((10, 10), dtype=bool),
            chi2=np.ones((10, 10), dtype=np.float32),
            rms_used=1e-3,
            frequencies=np.array([1e9, 1.5e9, 2e9]),
        )
        
        np.testing.assert_array_equal(result.get_coefficient(0), coeffs[0])
        np.testing.assert_array_equal(result.get_coefficient(1), coeffs[1])
        np.testing.assert_array_equal(result.get_coefficient(2), coeffs[2])
        
        # Out of range should return NaN
        assert np.all(np.isnan(result.get_coefficient(5)))
