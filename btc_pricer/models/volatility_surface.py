"""Volatility Surface Protocol.

Defines an abstract interface for volatility models (SSVI, Heston, etc.)
so that Breeden-Litzenberger RND extraction works uniformly with any model.
"""

from typing import Protocol, runtime_checkable
import numpy as np


@runtime_checkable
class VolatilitySurface(Protocol):
    """Protocol for volatility surface models.

    Any model implementing this protocol can be used with
    BreedenLitzenberger.extract_from_surface() for RND extraction.

    Both SSVIModel and HestonModel implement this protocol.
    """

    @property
    def ttm(self) -> float:
        """Time to maturity in years."""
        ...

    def implied_volatility(self, k: float) -> float:
        """Calculate implied volatility for log-moneyness k.

        Args:
            k: Log-moneyness ln(K/F).

        Returns:
            Implied volatility (annualized).
        """
        ...

    def implied_volatility_strike(self, strike: float, forward: float) -> float:
        """Calculate implied volatility for a given strike.

        Args:
            strike: Strike price.
            forward: Forward price.

        Returns:
            Implied volatility.
        """
        ...

    def implied_volatility_array(self, k_array: np.ndarray) -> np.ndarray:
        """Calculate implied volatility for array of log-moneyness values.

        Args:
            k_array: Array of log-moneyness values.

        Returns:
            Array of implied volatilities.
        """
        ...
