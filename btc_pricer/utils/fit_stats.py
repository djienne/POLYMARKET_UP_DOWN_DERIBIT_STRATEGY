"""Shared fit statistics utilities.

This module provides common functions for calculating model fit statistics
used by multiple model fitters (SSVI, Heston, etc.).
"""

from dataclasses import dataclass
from typing import Tuple
import numpy as np


@dataclass
class FitStats:
    """Fit statistics for model calibration."""
    r_squared: float
    rmse: float
    max_residual: float


def calculate_fit_stats(model_iv: np.ndarray, market_iv: np.ndarray) -> FitStats:
    """Calculate common fit statistics between model and market IVs.

    Computes R², RMSE, and maximum absolute residual for evaluating
    model fit quality.

    Args:
        model_iv: Array of model-implied volatilities.
        market_iv: Array of market implied volatilities.

    Returns:
        FitStats containing r_squared, rmse, and max_residual.

    Raises:
        ValueError: If arrays have different lengths.
    """
    if len(model_iv) != len(market_iv):
        raise ValueError("model_iv and market_iv must have same length")

    residuals = model_iv - market_iv
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((market_iv - np.mean(market_iv)) ** 2)

    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    rmse = np.sqrt(np.mean(residuals ** 2))
    max_residual = np.max(np.abs(residuals))

    return FitStats(
        r_squared=r_squared,
        rmse=rmse,
        max_residual=max_residual
    )


def calculate_iv_error_stats(
    model_iv: np.ndarray,
    market_iv: np.ndarray
) -> Tuple[float, float]:
    """Calculate IV error statistics (mean and max relative error).

    Args:
        model_iv: Array of model-implied volatilities.
        market_iv: Array of market implied volatilities.

    Returns:
        Tuple of (mean_relative_error, max_relative_error).
    """
    residuals = model_iv - market_iv
    iv_error_mean = np.mean(np.abs(residuals / market_iv))
    iv_error_max = np.max(np.abs(residuals / market_iv))
    return iv_error_mean, iv_error_max
