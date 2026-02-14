"""Breeden-Litzenberger Risk-Neutral Density extraction.

The Breeden-Litzenberger formula extracts the risk-neutral probability
density from option prices:

f(K) = e^(rT) * d²C/dK²

Where C is the call price as a function of strike K.

For forward prices (discount factor = 1):
f(K) = d²C_fwd/dK²
"""

import math
from dataclasses import dataclass
from typing import Callable, Optional, Tuple
import numpy as np
from scipy.interpolate import UnivariateSpline, CubicSpline
from scipy.integrate import simpson, cumulative_trapezoid, quad
import warnings

from typing import TYPE_CHECKING

from scipy.stats import norm as _norm_dist

from .ssvi import SSVIModel, SSVIParams
from .black_scholes import BlackScholes
from .volatility_surface import VolatilitySurface

if TYPE_CHECKING:
    from .heston import HestonParams


@dataclass
class RNDResult:
    """Result of RND extraction."""
    strikes: np.ndarray
    density: np.ndarray
    forward: float
    ttm: float

    # Statistics
    mean: float
    mode: float
    variance: float
    std_dev: float
    skewness: float
    kurtosis: float  # Excess kurtosis

    # Percentiles
    percentile_5: float
    percentile_25: float
    percentile_50: float  # Median
    percentile_75: float
    percentile_95: float

    # Diagnostics
    integral: float  # Should be ~1
    is_valid: bool
    warnings: list

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "forward": self.forward,
            "ttm": self.ttm,
            "mean": self.mean,
            "mode": self.mode,
            "variance": self.variance,
            "std_dev": self.std_dev,
            "skewness": self.skewness,
            "kurtosis": self.kurtosis,
            "percentiles": {
                "5": self.percentile_5,
                "25": self.percentile_25,
                "50": self.percentile_50,
                "75": self.percentile_75,
                "95": self.percentile_95,
            },
            "integral": self.integral,
            "is_valid": self.is_valid,
            "warnings": self.warnings,
        }


class BreedenLitzenberger:
    """Extract Risk-Neutral Density from option prices."""

    def __init__(
        self,
        strike_grid_points: int = 500,
        strike_range_std: float = 3.0,
        use_log_strikes: bool = False
    ):
        """Initialize the BL extractor.

        Args:
            strike_grid_points: Number of points in strike grid.
            strike_range_std: Number of std devs for strike range.
            use_log_strikes: If True, use log-moneyness spacing for better tails.
        """
        self.strike_grid_points = strike_grid_points
        self.strike_range_std = strike_range_std
        self.use_log_strikes = use_log_strikes

    def _create_strike_grid(
        self,
        forward: float,
        atm_std: float
    ) -> np.ndarray:
        """Create strike grid for RND extraction.

        Args:
            forward: Forward price.
            atm_std: ATM standard deviation (vol * sqrt(ttm)).

        Returns:
            Array of strike prices.
        """
        k_min = -self.strike_range_std * atm_std
        k_max = self.strike_range_std * atm_std

        if self.use_log_strikes:
            # Log-moneyness spacing for better tail capture
            log_k = np.linspace(k_min, k_max, self.strike_grid_points)
            return forward * np.exp(log_k)
        else:
            # Linear spacing in strike space
            strike_min = forward * math.exp(k_min)
            strike_max = forward * math.exp(k_max)
            return np.linspace(strike_min, strike_max, self.strike_grid_points)

    def _extract_core(
        self,
        surface: VolatilitySurface,
        forward: float,
        initial_warnings: Optional[list] = None
    ) -> RNDResult:
        """Core RND extraction logic shared by all extraction methods.

        Args:
            surface: Volatility surface model implementing the protocol.
            forward: Forward price.
            initial_warnings: Optional list of warnings to include.

        Returns:
            RNDResult with density and statistics.
        """
        warn_list = initial_warnings if initial_warnings is not None else []

        ttm = surface.ttm

        # ATM volatility for range estimation
        atm_vol = surface.implied_volatility(0)
        atm_std = atm_vol * math.sqrt(ttm)

        # Create strike grid
        strikes = self._create_strike_grid(forward, atm_std)

        # Collect implied vols for all strikes (still per-strike through model)
        vols = np.array([
            surface.implied_volatility_strike(K, forward) for K in strikes
        ])

        # Vectorized Black-Scholes forward call pricing:
        #   C = F*N(d1) - K*N(d2)  with discount_factor=1
        # Algebraically identical to BlackScholes.forward_call_price per strike.
        sqrt_t = math.sqrt(ttm) if ttm > 0 else 0.0
        if sqrt_t > 0:
            safe_vols = np.maximum(vols, 1e-12)
            d1 = (np.log(forward / strikes) + 0.5 * safe_vols**2 * ttm) / (safe_vols * sqrt_t)
            d2 = d1 - safe_vols * sqrt_t
            call_prices = forward * _norm_dist.cdf(d1) - strikes * _norm_dist.cdf(d2)
            # Clamp: price must be in [0, forward] and >= intrinsic
            call_prices = np.maximum(call_prices, 0.0)
        else:
            call_prices = np.maximum(forward - strikes, 0.0)

        # Use cubic spline for smooth interpolation
        spline = CubicSpline(strikes, call_prices)

        # Second derivative gives density
        density = spline(strikes, 2)

        # Ensure non-negative
        neg_count = np.sum(density < 0)
        if neg_count > 0:
            pct_neg = neg_count / len(density) * 100
            if pct_neg > 5:
                warn_list.append(f"{pct_neg:.1f}% of density values were negative")
            density = np.maximum(density, 0)

        # Normalize
        raw_integral = simpson(density, x=strikes)
        if raw_integral > 0:
            density = density / raw_integral
        else:
            warn_list.append("Raw density integral was <= 0")
            return self._invalid_result(forward, ttm, strikes, warn_list)

        # Final integral check
        final_integral = simpson(density, x=strikes)

        # Calculate statistics
        stats = self._compute_statistics(strikes, density, forward)

        # Validation
        is_valid = True
        if abs(final_integral - 1.0) > 0.05:
            warn_list.append(f"Density integral = {final_integral:.4f}, expected ~1")
            is_valid = False

        if abs(stats["mean"] - forward) / forward > 0.1:
            warn_list.append(
                f"Mean ({stats['mean']:.0f}) differs from forward ({forward:.0f}) by >10%"
            )

        return RNDResult(
            strikes=strikes,
            density=density,
            forward=forward,
            ttm=ttm,
            mean=stats["mean"],
            mode=stats["mode"],
            variance=stats["variance"],
            std_dev=stats["std_dev"],
            skewness=stats["skewness"],
            kurtosis=stats["kurtosis"],
            percentile_5=stats["percentile_5"],
            percentile_25=stats["percentile_25"],
            percentile_50=stats["percentile_50"],
            percentile_75=stats["percentile_75"],
            percentile_95=stats["percentile_95"],
            integral=final_integral,
            is_valid=is_valid,
            warnings=warn_list
        )

    def extract_from_ssvi(
        self,
        ssvi_params: SSVIParams,
        forward: float
    ) -> RNDResult:
        """Extract RND from SSVI parameters.

        This is the preferred method as SSVI provides a smooth
        volatility surface, making differentiation stable.

        Args:
            ssvi_params: Fitted SSVI parameters.
            forward: Forward price.

        Returns:
            RNDResult with density and statistics.
        """
        initial_warnings = []

        # Check butterfly condition
        if not ssvi_params.butterfly_condition():
            initial_warnings.append("SSVI butterfly condition violated - may have arbitrage")

        ssvi_model = SSVIModel(ssvi_params)
        return self._extract_core(ssvi_model, forward, initial_warnings)

    def extract_from_surface(
        self,
        surface: VolatilitySurface,
        forward: float
    ) -> RNDResult:
        """Extract RND from any volatility surface model.

        This generic method works with any model implementing the
        VolatilitySurface protocol (SSVI, Heston, etc.).

        Args:
            surface: Volatility surface model implementing the protocol.
            forward: Forward price.

        Returns:
            RNDResult with density and statistics.
        """
        return self._extract_core(surface, forward)

    def extract_from_heston(
        self,
        params: "HestonParams",
        forward: float,
        use_quantlib: bool = True
    ) -> RNDResult:
        """Extract RND from Heston parameters.

        Convenience method that wraps extract_from_surface for Heston.

        Args:
            params: Heston parameters.
            forward: Forward price.
            use_quantlib: Use QuantLib for Heston pricing (default: True).

        Returns:
            RNDResult with density and statistics.
        """
        from .heston import HestonModel
        model = HestonModel(params, use_quantlib=use_quantlib)
        return self.extract_from_surface(model, forward)

    def extract_from_prices(
        self,
        strikes: np.ndarray,
        call_prices: np.ndarray,
        forward: float,
        ttm: float,
        smoothing: float = 0.0
    ) -> RNDResult:
        """Extract RND directly from call prices.

        This method requires smoothing before differentiation.

        Args:
            strikes: Array of strike prices.
            call_prices: Array of forward call prices (in USD).
            forward: Forward price.
            ttm: Time to maturity.
            smoothing: Smoothing parameter for spline (0 = interpolation).

        Returns:
            RNDResult with density and statistics.
        """
        warnings = []

        # Sort by strike
        sort_idx = np.argsort(strikes)
        strikes = strikes[sort_idx]
        call_prices = call_prices[sort_idx]

        # Fit smoothing spline
        spline = UnivariateSpline(strikes, call_prices, s=smoothing, k=3)

        # Create fine grid
        strike_fine = np.linspace(strikes.min(), strikes.max(), self.strike_grid_points)

        # Second derivative
        density = spline.derivative(2)(strike_fine)

        # Handle negative values
        neg_count = np.sum(density < 0)
        if neg_count > 0:
            pct_neg = neg_count / len(density) * 100
            if pct_neg > 5:
                warnings.append(f"{pct_neg:.1f}% of density values were negative")
            density = np.maximum(density, 0)

        # Normalize
        raw_integral = simpson(density, x=strike_fine)
        if raw_integral > 0:
            density = density / raw_integral
        else:
            warnings.append("Raw density integral was <= 0")
            return self._invalid_result(forward, ttm, strike_fine, warnings)

        final_integral = simpson(density, x=strike_fine)

        # Statistics
        stats = self._compute_statistics(strike_fine, density, forward)

        is_valid = True
        if abs(final_integral - 1.0) > 0.05:
            warnings.append(f"Density integral = {final_integral:.4f}")
            is_valid = False

        return RNDResult(
            strikes=strike_fine,
            density=density,
            forward=forward,
            ttm=ttm,
            mean=stats["mean"],
            mode=stats["mode"],
            variance=stats["variance"],
            std_dev=stats["std_dev"],
            skewness=stats["skewness"],
            kurtosis=stats["kurtosis"],
            percentile_5=stats["percentile_5"],
            percentile_25=stats["percentile_25"],
            percentile_50=stats["percentile_50"],
            percentile_75=stats["percentile_75"],
            percentile_95=stats["percentile_95"],
            integral=final_integral,
            is_valid=is_valid,
            warnings=warnings
        )

    def _compute_statistics(
        self,
        strikes: np.ndarray,
        density: np.ndarray,
        forward: float
    ) -> dict:
        """Compute statistics from the density.

        Args:
            strikes: Strike array.
            density: Probability density array.
            forward: Forward price.

        Returns:
            Dictionary of statistics.
        """
        # Mean
        mean = simpson(strikes * density, x=strikes)

        # Mode (maximum density point)
        mode_idx = np.argmax(density)
        mode = strikes[mode_idx]

        # Variance and higher moments
        centered = strikes - mean
        variance = simpson(centered ** 2 * density, x=strikes)
        std_dev = math.sqrt(max(0, variance))

        # Skewness
        if std_dev > 0:
            third_moment = simpson(centered ** 3 * density, x=strikes)
            skewness = third_moment / (std_dev ** 3)
        else:
            skewness = 0.0

        # Excess kurtosis
        if std_dev > 0:
            fourth_moment = simpson(centered ** 4 * density, x=strikes)
            kurtosis = fourth_moment / (std_dev ** 4) - 3
        else:
            kurtosis = 0.0

        # CDF for percentiles - use cumulative_trapezoid for O(n) complexity
        cdf = np.zeros(len(strikes))
        cdf[1:] = cumulative_trapezoid(density, x=strikes)
        # Normalize CDF to end at 1.0
        if cdf[-1] > 0:
            cdf = cdf / cdf[-1]

        # Interpolate percentiles
        def get_percentile(p: float) -> float:
            idx = np.searchsorted(cdf, p / 100)
            if idx >= len(strikes):
                return strikes[-1]
            if idx == 0:
                return strikes[0]
            # Linear interpolation
            frac = (p / 100 - cdf[idx-1]) / (cdf[idx] - cdf[idx-1] + 1e-10)
            return strikes[idx-1] + frac * (strikes[idx] - strikes[idx-1])

        return {
            "mean": mean,
            "mode": mode,
            "variance": variance,
            "std_dev": std_dev,
            "skewness": skewness,
            "kurtosis": kurtosis,
            "percentile_5": get_percentile(5),
            "percentile_25": get_percentile(25),
            "percentile_50": get_percentile(50),
            "percentile_75": get_percentile(75),
            "percentile_95": get_percentile(95),
        }

    def _invalid_result(
        self,
        forward: float,
        ttm: float,
        strikes: np.ndarray,
        warnings: list
    ) -> RNDResult:
        """Create an invalid result for error cases."""
        return RNDResult(
            strikes=strikes,
            density=np.zeros(len(strikes)),
            forward=forward,
            ttm=ttm,
            mean=forward,
            mode=forward,
            variance=0,
            std_dev=0,
            skewness=0,
            kurtosis=0,
            percentile_5=forward,
            percentile_25=forward,
            percentile_50=forward,
            percentile_75=forward,
            percentile_95=forward,
            integral=0,
            is_valid=False,
            warnings=warnings
        )

    def probability_between(
        self,
        rnd: RNDResult,
        lower: float,
        upper: float
    ) -> float:
        """Calculate probability that price falls between two values.

        Args:
            rnd: RND result.
            lower: Lower bound.
            upper: Upper bound.

        Returns:
            Probability in [0, 1].
        """
        mask = (rnd.strikes >= lower) & (rnd.strikes <= upper)
        if not np.any(mask):
            return 0.0

        return simpson(rnd.density[mask], x=rnd.strikes[mask])

    def probability_above(self, rnd: RNDResult, threshold: float) -> float:
        """Calculate probability that price is above threshold.

        Args:
            rnd: RND result.
            threshold: Price threshold.

        Returns:
            Probability in [0, 1].
        """
        mask = rnd.strikes >= threshold
        if not np.any(mask):
            return 0.0

        return simpson(rnd.density[mask], x=rnd.strikes[mask])

    def probability_below(self, rnd: RNDResult, threshold: float) -> float:
        """Calculate probability that price is below threshold.

        Args:
            rnd: RND result.
            threshold: Price threshold.

        Returns:
            Probability in [0, 1].
        """
        mask = rnd.strikes <= threshold
        if not np.any(mask):
            return 0.0

        return simpson(rnd.density[mask], x=rnd.strikes[mask])

    def probability_between_continuous(
        self,
        rnd: RNDResult,
        lower: float,
        upper: float
    ) -> float:
        """Calculate probability using continuous spline integration.

        More accurate than discrete grid method for narrow ranges.

        Args:
            rnd: RND result.
            lower: Lower bound.
            upper: Upper bound.

        Returns:
            Probability in [0, 1].
        """
        pdf_spline = CubicSpline(rnd.strikes, rnd.density, extrapolate=False)

        # Clamp to valid range
        lower = max(lower, rnd.strikes[0])
        upper = min(upper, rnd.strikes[-1])

        if lower >= upper:
            return 0.0

        # Suppress integration warnings for edge cases
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result, _ = quad(pdf_spline, lower, upper, limit=100)

        return max(0.0, min(1.0, result))

    def price_derivative(
        self,
        rnd: RNDResult,
        payoff_func: Callable[[np.ndarray], np.ndarray],
        discount_factor: float = 1.0
    ) -> float:
        """Price an arbitrary derivative using the RND.

        V = DF * E[payoff(S_T)] = DF * integral(payoff(K) * f(K) dK)

        Args:
            rnd: RND result.
            payoff_func: Function mapping strike array to payoff array.
            discount_factor: Discount factor for present value.

        Returns:
            Derivative price in USD.
        """
        payoffs = payoff_func(rnd.strikes)

        # Suppress integration warnings for edge cases
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            expected_payoff = simpson(payoffs * rnd.density, x=rnd.strikes)

        return discount_factor * expected_payoff

    def price_call(
        self,
        rnd: RNDResult,
        strike: float,
        discount_factor: float = 1.0
    ) -> float:
        """Price a call option using the RND.

        Args:
            rnd: RND result.
            strike: Strike price.
            discount_factor: Discount factor for present value.

        Returns:
            Call price in USD.
        """
        return self.price_derivative(
            rnd,
            lambda S: np.maximum(S - strike, 0),
            discount_factor
        )

    def price_put(
        self,
        rnd: RNDResult,
        strike: float,
        discount_factor: float = 1.0
    ) -> float:
        """Price a put option using the RND.

        Args:
            rnd: RND result.
            strike: Strike price.
            discount_factor: Discount factor for present value.

        Returns:
            Put price in USD.
        """
        return self.price_derivative(
            rnd,
            lambda S: np.maximum(strike - S, 0),
            discount_factor
        )

    def price_digital_call(
        self,
        rnd: RNDResult,
        strike: float,
        discount_factor: float = 1.0
    ) -> float:
        """Price a digital (binary) call option using the RND.

        Pays 1 if S_T > strike, 0 otherwise.

        Args:
            rnd: RND result.
            strike: Strike price.
            discount_factor: Discount factor for present value.

        Returns:
            Digital call price (probability * DF).
        """
        return discount_factor * self.probability_above(rnd, strike)

    def price_digital_put(
        self,
        rnd: RNDResult,
        strike: float,
        discount_factor: float = 1.0
    ) -> float:
        """Price a digital (binary) put option using the RND.

        Pays 1 if S_T < strike, 0 otherwise.

        Args:
            rnd: RND result.
            strike: Strike price.
            discount_factor: Discount factor for present value.

        Returns:
            Digital put price (probability * DF).
        """
        return discount_factor * self.probability_below(rnd, strike)
