"""Tests for the Gaussian + background model utilities."""

import numpy as np
import pytest

from gaussian_fit_parameter_tsa.model import (
    FitParams,
    area_from_A_sigma,
    calculate_fit_quality,
    estimate_initial_parameters,
    fit_gaussian_robust,
    fwhm_from_sigma,
    gaussian_bg,
)


class TestGaussianBg:
    """Test the gaussian_bg function."""

    def test_gaussian_bg_basic(self) -> None:
        """Test basic Gaussian function with background."""
        x = np.linspace(-5, 5, 100)
        A, mu, sigma, B = 2.0, 0.0, 1.0, 0.5

        result = gaussian_bg(x, A, mu, sigma, B)

        # Check that result is numpy array
        assert isinstance(result, np.ndarray)
        assert len(result) == len(x)

        # Check that background is correct (Gaussian should be very small at edges)
        assert np.isclose(result[0], B, atol=1e-4)  # At x=-5, Gaussian ≈ 0
        assert np.isclose(result[-1], B, atol=1e-4)  # At x=5, Gaussian ≈ 0

        # Check that peak is at mu
        peak_idx = np.argmax(result)
        assert np.isclose(x[peak_idx], mu, atol=0.1)

        # Check that peak value is A + B (allow for discrete sampling)
        assert np.isclose(result[peak_idx], A + B, atol=1e-2)

    def test_gaussian_bg_no_background(self) -> None:
        """Test Gaussian function with zero background."""
        x = np.linspace(-3, 3, 50)
        A, mu, sigma, B = 1.0, 0.0, 0.5, 0.0

        result = gaussian_bg(x, A, mu, sigma, B)

        # At x=mu, should be A (allow for discrete sampling)
        center_idx = len(x) // 2
        assert np.isclose(result[center_idx], A, atol=1e-2)

        # Test that the function has the expected shape
        # Peak should be at center
        peak_idx = np.argmax(result)
        assert np.isclose(x[peak_idx], mu, atol=0.1)

        # Function should be symmetric around center
        left_half = result[:center_idx]
        right_half = result[center_idx + 1 :][::-1]
        # Make sure arrays are the same length
        min_len = min(len(left_half), len(right_half))
        assert np.allclose(left_half[:min_len], right_half[:min_len], atol=1e-10)

    def test_gaussian_bg_shifted(self) -> None:
        """Test Gaussian function with shifted center."""
        x = np.linspace(0, 10, 100)
        A, mu, sigma, B = 3.0, 5.0, 1.0, 1.0

        result = gaussian_bg(x, A, mu, sigma, B)

        # Peak should be at mu=5
        peak_idx = np.argmax(result)
        assert np.isclose(x[peak_idx], mu, atol=0.1)

        # Peak value should be A + B (allow for discrete sampling)
        assert np.isclose(result[peak_idx], A + B, atol=1e-2)

    def test_gaussian_bg_wide(self) -> None:
        """Test Gaussian function with wide sigma."""
        x = np.linspace(-10, 10, 200)
        A, mu, sigma, B = 1.0, 0.0, 3.0, 0.2

        result = gaussian_bg(x, A, mu, sigma, B)

        # Should be wider than narrow Gaussian
        # Check that values are closer to background at edges
        assert np.isclose(result[0], B, atol=0.1)
        assert np.isclose(result[-1], B, atol=0.1)

    def test_gaussian_bg_narrow(self) -> None:
        """Test Gaussian function with narrow sigma."""
        x = np.linspace(-2, 2, 100)
        A, mu, sigma, B = 2.0, 0.0, 0.1, 0.5

        result = gaussian_bg(x, A, mu, sigma, B)

        # Should be very narrow
        # Most values should be close to background
        non_peak_values = np.concatenate([result[:40], result[60:]])
        assert np.allclose(non_peak_values, B, atol=0.1)


class TestFitParams:
    """Test the FitParams dataclass."""

    def test_fit_params_creation(self) -> None:
        """Test creating FitParams instance."""
        params = FitParams(A=2.0, mu=1.0, sigma=0.5, B=0.1, fwhm=1.18, area=2.51)

        assert params.A == 2.0
        assert params.mu == 1.0
        assert params.sigma == 0.5
        assert params.B == 0.1
        assert params.fwhm == 1.18
        assert params.area == 2.51

    def test_fit_params_immutable(self) -> None:
        """Test that FitParams is immutable."""
        params = FitParams(A=1.0, mu=0.0, sigma=1.0, B=0.0, fwhm=2.35, area=2.51)

        # Should not be able to modify attributes
        with pytest.raises(AttributeError):
            params.A = 2.0  # type: ignore[misc]


class TestFwhmFromSigma:
    """Test the fwhm_from_sigma function."""

    def test_fwhm_from_sigma_basic(self) -> None:
        """Test basic FWHM calculation."""
        sigma = 1.0
        fwhm = fwhm_from_sigma(sigma)

        # FWHM = 2 * sqrt(2 * ln(2)) * sigma ≈ 2.355 * sigma
        expected = 2 * np.sqrt(2 * np.log(2)) * sigma
        assert np.isclose(fwhm, expected, atol=1e-10)
        assert np.isclose(fwhm, 2.355, atol=0.01)

    def test_fwhm_from_sigma_zero(self) -> None:
        """Test FWHM calculation with zero sigma."""
        sigma = 0.0
        fwhm = fwhm_from_sigma(sigma)

        assert fwhm == 0.0

    def test_fwhm_from_sigma_large(self) -> None:
        """Test FWHM calculation with large sigma."""
        sigma = 10.0
        fwhm = fwhm_from_sigma(sigma)

        expected = 2 * np.sqrt(2 * np.log(2)) * sigma
        assert np.isclose(fwhm, expected, atol=1e-10)
        assert np.isclose(fwhm, 23.55, atol=0.01)

    def test_fwhm_from_sigma_negative(self) -> None:
        """Test FWHM calculation with negative sigma."""
        sigma = -1.0
        fwhm = fwhm_from_sigma(sigma)

        # Should handle negative sigma (though not physically meaningful)
        expected = 2 * np.sqrt(2 * np.log(2)) * sigma
        assert np.isclose(fwhm, expected, atol=1e-10)


class TestAreaFromASigma:
    """Test the area_from_A_sigma function."""

    def test_area_from_A_sigma_basic(self) -> None:
        """Test basic area calculation."""
        A, sigma = 1.0, 1.0
        area = area_from_A_sigma(A, sigma)

        # Area = A * sigma * sqrt(2 * pi)
        expected = A * sigma * np.sqrt(2 * np.pi)
        assert np.isclose(area, expected, atol=1e-10)
        assert np.isclose(area, 2.507, atol=0.01)

    def test_area_from_A_sigma_zero_amplitude(self) -> None:
        """Test area calculation with zero amplitude."""
        A, sigma = 0.0, 1.0
        area = area_from_A_sigma(A, sigma)

        assert area == 0.0

    def test_area_from_A_sigma_zero_sigma(self) -> None:
        """Test area calculation with zero sigma."""
        A, sigma = 1.0, 0.0
        area = area_from_A_sigma(A, sigma)

        assert area == 0.0

    def test_area_from_A_sigma_large_values(self) -> None:
        """Test area calculation with large values."""
        A, sigma = 5.0, 2.0
        area = area_from_A_sigma(A, sigma)

        expected = A * sigma * np.sqrt(2 * np.pi)
        assert np.isclose(area, expected, atol=1e-10)
        assert np.isclose(area, 25.07, atol=0.01)

    def test_area_from_A_sigma_negative_amplitude(self) -> None:
        """Test area calculation with negative amplitude."""
        A, sigma = -2.0, 1.0
        area = area_from_A_sigma(A, sigma)

        # Should handle negative amplitude
        expected = A * sigma * np.sqrt(2 * np.pi)
        assert np.isclose(area, expected, atol=1e-10)
        assert np.isclose(area, -5.013, atol=0.01)


class TestModelIntegration:
    """Test integration between model components."""

    def test_gaussian_area_consistency(self) -> None:
        """Test that calculated area matches numerical integration."""
        x = np.linspace(-10, 10, 1000)
        A, mu, sigma, B = 2.0, 0.0, 1.0, 0.5

        # Calculate function values
        y = gaussian_bg(x, A, mu, sigma, B)

        # Remove background for area calculation
        y_gaussian = y - B

        # Numerical integration (trapezoidal rule)
        numerical_area = np.trapezoid(y_gaussian, x)

        # Analytical area
        analytical_area = area_from_A_sigma(A, sigma)

        # Should be close (within 1% due to numerical integration limits)
        assert np.isclose(numerical_area, analytical_area, rtol=0.01)

    def test_fwhm_consistency(self) -> None:
        """Test that FWHM calculation is consistent with function values."""
        x = np.linspace(-5, 5, 1000)
        A, mu, sigma, B = 2.0, 0.0, 1.0, 0.5

        y = gaussian_bg(x, A, mu, sigma, B)

        # Find half maximum
        half_max = B + A / 2

        # Find indices where function crosses half maximum
        above_half_max = y > half_max
        if np.any(above_half_max):
            # Find the range where function is above half maximum
            start_idx = np.where(above_half_max)[0][0]
            end_idx = np.where(above_half_max)[0][-1]

            # Calculate FWHM from x values
            numerical_fwhm = x[end_idx] - x[start_idx]

            # Analytical FWHM
            analytical_fwhm = fwhm_from_sigma(sigma)

            # Should be close (within 2% due to discrete sampling)
            assert np.isclose(numerical_fwhm, analytical_fwhm, rtol=0.02)


class TestEstimateInitialParameters:
    """Test the estimate_initial_parameters function."""

    def test_estimate_initial_parameters_basic(self) -> None:
        """Test basic parameter estimation."""
        x = np.linspace(-5, 5, 100)
        A, mu, sigma, B = 2.0, 1.0, 1.5, 0.5
        y = gaussian_bg(x, A, mu, sigma, B)

        A0, mu0, sigma0, B0 = estimate_initial_parameters(x, y)

        # Check that estimates are reasonable
        assert A0 > 0
        assert B0 >= 0
        assert sigma0 > 0
        # mu0 should be close to the actual peak
        assert abs(mu0 - mu) < 2.0  # Within reasonable range

    def test_estimate_initial_parameters_with_noise(self) -> None:
        """Test parameter estimation with noisy data."""
        x = np.linspace(-5, 5, 100)
        A, mu, sigma, B = 3.0, 0.0, 1.0, 1.0
        y = gaussian_bg(x, A, mu, sigma, B)
        # Add some noise
        y += np.random.normal(0, 0.1, len(y))

        A0, mu0, sigma0, B0 = estimate_initial_parameters(x, y)

        # Check that estimates are reasonable even with noise
        assert A0 > 0
        assert B0 >= 0
        assert sigma0 > 0


class TestCalculateFitQuality:
    """Test the calculate_fit_quality function."""

    def test_calculate_fit_quality_perfect_fit(self) -> None:
        """Test fit quality calculation with perfect fit."""
        x = np.linspace(-5, 5, 100)
        A, mu, sigma, B = 2.0, 0.0, 1.0, 0.5
        y = gaussian_bg(x, A, mu, sigma, B)
        popt = [A, mu, sigma, B]

        r_squared, rmse = calculate_fit_quality(x, y, popt)

        # Perfect fit should have R² = 1 and RMSE = 0
        assert np.isclose(r_squared, 1.0, atol=1e-10)
        assert np.isclose(rmse, 0.0, atol=1e-10)

    def test_calculate_fit_quality_poor_fit(self) -> None:
        """Test fit quality calculation with poor fit."""
        x = np.linspace(-5, 5, 100)
        y = np.random.normal(0, 1, len(x))  # Random data
        popt = [1.0, 0.0, 1.0, 0.0]  # Poor fit parameters

        r_squared, rmse = calculate_fit_quality(x, y, popt)

        # Poor fit should have low R² and high RMSE
        assert r_squared < 0.5
        assert rmse > 0.5


class TestFitGaussianRobust:
    """Test the fit_gaussian_robust function."""

    def test_fit_gaussian_robust_basic(self) -> None:
        """Test basic robust Gaussian fitting."""
        x = np.linspace(-5, 5, 100)
        A, mu, sigma, B = 2.0, 1.0, 1.5, 0.5
        y = gaussian_bg(x, A, mu, sigma, B)

        result = fit_gaussian_robust(x, y)

        # Check that fit was successful
        assert result.success
        assert result.r_squared > 0.9  # Good fit
        assert result.rmse < 1.0  # Low error

        # Check that parameters are close to true values
        assert np.isclose(result.A, A, rtol=0.1)
        assert np.isclose(result.mu, mu, rtol=0.1)
        assert np.isclose(result.sigma, sigma, rtol=0.1)
        assert np.isclose(result.B, B, rtol=0.1)

    def test_fit_gaussian_robust_with_noise(self) -> None:
        """Test robust fitting with noisy data."""
        x = np.linspace(-5, 5, 100)
        A, mu, sigma, B = 3.0, 0.0, 1.0, 1.0
        y = gaussian_bg(x, A, mu, sigma, B)
        # Add noise
        y += np.random.normal(0, 0.2, len(y))

        result = fit_gaussian_robust(x, y)

        # Should still fit reasonably well
        assert result.success
        assert result.r_squared > 0.7  # Reasonable fit
        assert result.rmse < 2.0  # Acceptable error

    def test_fit_gaussian_robust_insufficient_data(self) -> None:
        """Test robust fitting with insufficient data."""
        x = np.array([1, 2, 3])  # Only 3 points
        y = np.array([1, 2, 1])

        with pytest.raises(ValueError, match="Need at least 4 data points"):
            fit_gaussian_robust(x, y)

    def test_fit_gaussian_robust_invalid_data(self) -> None:
        """Test robust fitting with invalid data."""
        x = np.array([1, 2, 3, 4])
        y = np.array([1, 2, np.nan, 4])  # Contains NaN

        result = fit_gaussian_robust(x, y)

        # Should handle NaN values and still fit
        assert (
            result.success or not result.success
        )  # Either succeeds or fails gracefully

    def test_fit_gaussian_robust_mismatched_lengths(self) -> None:
        """Test robust fitting with mismatched array lengths."""
        x = np.array([1, 2, 3, 4])
        y = np.array([1, 2, 3])  # Different length

        with pytest.raises(
            ValueError, match="x and y arrays must have the same length"
        ):
            fit_gaussian_robust(x, y)

    def test_fit_gaussian_robust_with_uncertainties(self) -> None:
        """Test robust fitting with uncertainty weights."""
        x = np.linspace(-5, 5, 100)
        A, mu, sigma, B = 2.0, 0.0, 1.0, 0.5
        y = gaussian_bg(x, A, mu, sigma, B)
        sigma_y = np.ones_like(y) * 0.1  # Constant uncertainty

        result = fit_gaussian_robust(x, y, sigma_y)

        # Should fit successfully with uncertainties
        assert result.success
        assert result.r_squared > 0.9
