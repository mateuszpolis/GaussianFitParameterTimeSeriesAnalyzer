"""Gaussian + background model utilities for time series analysis.

This module provides functions and data structures for fitting Gaussian functions
with a constant background to time series data.

The model function is: f(x) = A·exp(-0.5·((x−μ)/σ)^2) + B
Reported outputs: (A, μ, σ, B, FWHM, area)
"""

from dataclasses import dataclass

import numpy as np


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
    """

    A: float
    mu: float
    sigma: float
    B: float
    fwhm: float
    area: float


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
