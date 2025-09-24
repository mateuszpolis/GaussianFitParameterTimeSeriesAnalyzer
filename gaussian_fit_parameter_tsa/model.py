"""Gaussian + background model utilities for time series analysis.

This module provides functions and data structures for fitting Gaussian functions
with a constant background to time series data.

The model function is: f(x) = A·exp(-0.5·((x−μ)/σ)^2) + B
Reported outputs: (A, μ, σ, B, FWHM, area)
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from scipy.optimize import curve_fit


def gaussian_bg(
    x: np.ndarray, A: float, mu: float, sigma: float, B: float
) -> np.ndarray:
    """Gaussian function with constant background.

    Model: f(x) = A·exp(-0.5·((x−μ)/σ)^2) + B

    Args:
        x: Input array of x values
        A: Amplitude (peak height above background)
        mu: Center position (mean)
        sigma: Standard deviation (width parameter)
        B: Background level (constant offset)

    Returns:
        Array of function values at x
    """
    return A * np.exp(-0.5 * ((x - mu) / sigma) ** 2) + B


@dataclass(frozen=True)
class FitParams:
    """Parameters from Gaussian + background fit.

    Contains the fitted parameters and derived quantities from a Gaussian
    function with constant background fit.

    Attributes:
        A: Amplitude (peak height above background)
        mu: Center position (mean)
        sigma: Standard deviation (width parameter)
        B: Background level (constant offset)
        fwhm: Full width at half maximum
        area: Total area under the Gaussian peak
        A_err: Uncertainty in amplitude
        mu_err: Uncertainty in center position
        sigma_err: Uncertainty in standard deviation
        B_err: Uncertainty in background level
        r_squared: R-squared value for fit quality
        rmse: Root mean square error
        success: Whether the fit was successful
    """

    A: float
    mu: float
    sigma: float
    B: float
    fwhm: float
    area: float
    A_err: float = 0.0
    mu_err: float = 0.0
    sigma_err: float = 0.0
    B_err: float = 0.0
    r_squared: float = 0.0
    rmse: float = 0.0
    success: bool = False


def fwhm_from_sigma(sigma: float) -> float:
    """Calculate full width at half maximum from sigma.

    FWHM = 2 * sqrt(2 * ln(2)) * sigma ≈ 2.355 * sigma

    Args:
        sigma: Standard deviation of the Gaussian

    Returns:
        Full width at half maximum
    """
    return float(2 * np.sqrt(2 * np.log(2)) * sigma)


def area_from_A_sigma(A: float, sigma: float) -> float:
    """Calculate total area under Gaussian from amplitude and sigma.

    Area = A * sigma * sqrt(2 * pi)

    Args:
        A: Amplitude of the Gaussian
        sigma: Standard deviation of the Gaussian

    Returns:
        Total area under the Gaussian peak
    """
    return float(A * sigma * np.sqrt(2 * np.pi))


def estimate_initial_parameters(
    x: np.ndarray, y: np.ndarray
) -> Tuple[float, float, float, float]:
    """Estimate initial parameters for Gaussian fit.

    Uses robust methods to estimate initial guesses for A, mu, sigma, B.

    Args:
        x: Input array of x values
        y: Input array of y values

    Returns:
        Tuple of (A0, mu0, sigma0, B0) initial parameter estimates
    """
    # Background guess from edges (median is robust)
    edge_size = max(3, len(y) // 10)
    edges = np.r_[y[:edge_size], y[-edge_size:]]
    B0 = np.median(edges)

    # Peak amplitude and center
    idx = np.argmax(y)
    mu0 = x[idx]
    A0 = max(y[idx] - B0, 1e-6)

    # Width guess (fallback: a fraction of x-range)
    # Try to estimate from half-maximum if possible:
    half = B0 + 0.5 * A0
    above = y > half
    if np.any(above):
        x_hm = x[above]
        if len(x_hm) >= 2:
            fwhm0 = x_hm.max() - x_hm.min()
            sigma0 = max(fwhm0 / (2 * np.sqrt(2 * np.log(2))), 1e-6)
        else:
            sigma0 = (x.max() - x.min()) / 10.0
    else:
        sigma0 = (x.max() - x.min()) / 10.0

    return A0, mu0, sigma0, B0


def calculate_fit_quality(
    x: np.ndarray, y: np.ndarray, popt: np.ndarray
) -> Tuple[float, float]:
    """Calculate fit quality metrics.

    Args:
        x: Input array of x values
        y: Observed y values
        popt: Fitted parameters [A, mu, sigma, B]

    Returns:
        Tuple of (r_squared, rmse)
    """
    y_pred = gaussian_bg(x, *popt)
    resid = y - y_pred
    ss_res = np.sum(resid**2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    rmse = np.sqrt(np.mean(resid**2))
    return r_squared, rmse


def fit_gaussian_robust(
    x: np.ndarray, y: np.ndarray, sigma_y: Optional[np.ndarray] = None
) -> FitParams:
    """Fit Gaussian with background using robust methods.

    Performs a robust Gaussian fit with proper initial parameter estimation,
    bounds checking, and quality assessment.

    Args:
        x: Input array of x values
        y: Input array of y values
        sigma_y: Optional array of y uncertainties for weighted fitting

    Returns:
        FitParams object with fitted parameters and quality metrics

    Raises:
        ValueError: If fitting fails or data is invalid
    """
    if len(x) != len(y):
        raise ValueError("x and y arrays must have the same length")

    if len(x) < 4:
        raise ValueError("Need at least 4 data points for fitting")

    # Remove any NaN or infinite values
    valid_mask = np.isfinite(x) & np.isfinite(y)
    if not np.any(valid_mask):
        raise ValueError("No valid data points found")

    x_clean = x[valid_mask]
    y_clean = y[valid_mask]

    if sigma_y is not None:
        sigma_y_clean = sigma_y[valid_mask]
    else:
        sigma_y_clean = None

    try:
        # Estimate initial parameters
        A0, mu0, sigma0, B0 = estimate_initial_parameters(x_clean, y_clean)

        # Set reasonable bounds
        x_range = x_clean.max() - x_clean.min()
        y_range = y_clean.max() - y_clean.min()

        lower_bounds = [
            0.0,  # A >= 0
            x_clean.min() - x_range,  # mu can be outside data range
            1e-6,  # sigma > 0
            y_clean.min() - y_range,  # B can be negative
        ]

        upper_bounds = [
            y_clean.max() * 2,  # A reasonable upper bound
            x_clean.max() + x_range,  # mu can be outside data range
            x_range,  # sigma reasonable upper bound
            y_clean.max() + y_range,  # B reasonable upper bound
        ]

        # Perform the fit
        popt, pcov = curve_fit(
            gaussian_bg,
            x_clean,
            y_clean,
            p0=[A0, mu0, sigma0, B0],
            bounds=(lower_bounds, upper_bounds),
            sigma=sigma_y_clean,
            absolute_sigma=True,
            maxfev=5000,
        )

        # Extract parameters and uncertainties
        A, mu, sigma, B = popt
        perr = np.sqrt(np.diag(pcov))
        A_err, mu_err, sigma_err, B_err = perr

        # Calculate derived quantities
        fwhm = fwhm_from_sigma(sigma)
        area = area_from_A_sigma(A, sigma)

        # Calculate fit quality
        r_squared, rmse = calculate_fit_quality(x_clean, y_clean, popt)

        return FitParams(
            A=A,
            mu=mu,
            sigma=sigma,
            B=B,
            fwhm=fwhm,
            area=area,
            A_err=A_err,
            mu_err=mu_err,
            sigma_err=sigma_err,
            B_err=B_err,
            r_squared=r_squared,
            rmse=rmse,
            success=True,
        )

    except Exception:
        # Return a failed fit result
        return FitParams(
            A=0.0,
            mu=0.0,
            sigma=0.0,
            B=0.0,
            fwhm=0.0,
            area=0.0,
            success=False,
        )
