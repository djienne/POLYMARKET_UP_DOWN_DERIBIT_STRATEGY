"""Black-Scholes pricing and implied volatility solver.

This module implements:
- Inverse (Deribit-style) call/put pricing in BTC
- Forward call/put pricing in USD
- Implied volatility solver using Brent's method

QuantLib Integration:
    When use_quantlib=True, pricing uses QuantLib's Black-76 formula
    for better numerical precision in edge cases.
"""

import math
from typing import Optional, Tuple
from scipy.stats import norm
from scipy.optimize import brentq
import numpy as np

# Try to import QuantLib
QUANTLIB_AVAILABLE = False
try:
    import QuantLib as ql
    QUANTLIB_AVAILABLE = True
except ImportError:
    pass


class BlackScholes:
    """Black-Scholes pricing and IV calculations for Bitcoin options."""

    # Default class-level settings (can be overridden with IVSolverConfig)
    IV_TOL = 1e-8
    IV_MIN = 0.01  # 1%
    IV_MAX = 5.0   # 500%
    MAX_ITER = 100

    # Module-level config that can be set by applications
    _config = None

    @classmethod
    def set_config(cls, config) -> None:
        """Set IV solver configuration.

        Args:
            config: IVSolverConfig instance.
        """
        cls._config = config
        if config is not None:
            cls.IV_TOL = config.tolerance
            cls.IV_MIN = config.min_iv
            cls.IV_MAX = config.max_iv
            cls.MAX_ITER = config.max_iterations

    @staticmethod
    def d1(forward: float, strike: float, vol: float, ttm: float) -> float:
        """Calculate d1 for Black-Scholes.

        Args:
            forward: Forward price.
            strike: Strike price.
            vol: Volatility (annualized).
            ttm: Time to maturity in years.

        Returns:
            d1 value.
        """
        if ttm <= 0 or vol <= 0:
            return 0.0
        sqrt_t = math.sqrt(ttm)
        return (math.log(forward / strike) + 0.5 * vol * vol * ttm) / (vol * sqrt_t)

    @staticmethod
    def d2(forward: float, strike: float, vol: float, ttm: float) -> float:
        """Calculate d2 for Black-Scholes.

        Args:
            forward: Forward price.
            strike: Strike price.
            vol: Volatility (annualized).
            ttm: Time to maturity in years.

        Returns:
            d2 value.
        """
        if ttm <= 0 or vol <= 0:
            return 0.0
        sqrt_t = math.sqrt(ttm)
        return BlackScholes.d1(forward, strike, vol, ttm) - vol * sqrt_t

    @classmethod
    def inverse_call_price_btc(
        cls,
        forward: float,
        strike: float,
        vol: float,
        ttm: float
    ) -> float:
        """Calculate inverse call price (Deribit convention) in BTC.

        Deribit quotes option prices in BTC, not USD.
        Formula: C_BTC = N(d1) - (K/F) * N(d2)

        Args:
            forward: Forward price in USD.
            strike: Strike price in USD.
            vol: Implied volatility.
            ttm: Time to maturity in years.

        Returns:
            Call price in BTC.
        """
        if ttm <= 0:
            # Expired option - intrinsic value
            return max(0, 1 - strike / forward) if forward > 0 else 0

        d1_val = cls.d1(forward, strike, vol, ttm)
        d2_val = cls.d2(forward, strike, vol, ttm)

        return norm.cdf(d1_val) - (strike / forward) * norm.cdf(d2_val)

    @classmethod
    def inverse_put_price_btc(
        cls,
        forward: float,
        strike: float,
        vol: float,
        ttm: float
    ) -> float:
        """Calculate inverse put price (Deribit convention) in BTC.

        Formula: P_BTC = (K/F) * N(-d2) - N(-d1)

        Args:
            forward: Forward price in USD.
            strike: Strike price in USD.
            vol: Implied volatility.
            ttm: Time to maturity in years.

        Returns:
            Put price in BTC.
        """
        if ttm <= 0:
            # Expired option - intrinsic value
            return max(0, strike / forward - 1) if forward > 0 else 0

        d1_val = cls.d1(forward, strike, vol, ttm)
        d2_val = cls.d2(forward, strike, vol, ttm)

        return (strike / forward) * norm.cdf(-d2_val) - norm.cdf(-d1_val)

    @classmethod
    def forward_call_price(
        cls,
        forward: float,
        strike: float,
        vol: float,
        ttm: float,
        discount_factor: float = 1.0
    ) -> float:
        """Calculate forward call price in USD.

        Formula: C_fwd = F * C_BTC (when discount_factor = 1)
        Standard Black formula: C = df * [F*N(d1) - K*N(d2)]

        Args:
            forward: Forward price in USD.
            strike: Strike price in USD.
            vol: Implied volatility.
            ttm: Time to maturity in years.
            discount_factor: Discount factor (default 1 for forward price).

        Returns:
            Call price in USD.
        """
        if ttm <= 0:
            return discount_factor * max(0, forward - strike)

        d1_val = cls.d1(forward, strike, vol, ttm)
        d2_val = cls.d2(forward, strike, vol, ttm)

        return discount_factor * (forward * norm.cdf(d1_val) - strike * norm.cdf(d2_val))

    @classmethod
    def forward_put_price(
        cls,
        forward: float,
        strike: float,
        vol: float,
        ttm: float,
        discount_factor: float = 1.0
    ) -> float:
        """Calculate forward put price in USD.

        Args:
            forward: Forward price in USD.
            strike: Strike price in USD.
            vol: Implied volatility.
            ttm: Time to maturity in years.
            discount_factor: Discount factor (default 1 for forward price).

        Returns:
            Put price in USD.
        """
        if ttm <= 0:
            return discount_factor * max(0, strike - forward)

        d1_val = cls.d1(forward, strike, vol, ttm)
        d2_val = cls.d2(forward, strike, vol, ttm)

        return discount_factor * (strike * norm.cdf(-d2_val) - forward * norm.cdf(-d1_val))

    @classmethod
    def implied_volatility(
        cls,
        price: float,
        forward: float,
        strike: float,
        ttm: float,
        option_type: str,
        is_btc_price: bool = True
    ) -> Optional[float]:
        """Solve for implied volatility using Brent's method.

        Args:
            price: Option price.
            forward: Forward price.
            strike: Strike price.
            ttm: Time to maturity in years.
            option_type: 'call' or 'put'.
            is_btc_price: If True, price is in BTC (Deribit convention).

        Returns:
            Implied volatility, or None if solver fails.
        """
        if ttm <= 0 or price <= 0:
            return None

        # Select pricing function
        if is_btc_price:
            if option_type == "call":
                price_func = cls.inverse_call_price_btc
            else:
                price_func = cls.inverse_put_price_btc
        else:
            if option_type == "call":
                price_func = lambda f, k, v, t: cls.forward_call_price(f, k, v, t)
            else:
                price_func = lambda f, k, v, t: cls.forward_put_price(f, k, v, t)

        # Objective function
        def objective(vol: float) -> float:
            return price_func(forward, strike, vol, ttm) - price

        # Check bounds
        try:
            low_val = objective(cls.IV_MIN)
            high_val = objective(cls.IV_MAX)

            # Check if solution exists in range
            if low_val * high_val > 0:
                # Same sign - no solution in range
                return None

            # Use Brent's method
            iv = brentq(
                objective,
                cls.IV_MIN,
                cls.IV_MAX,
                xtol=cls.IV_TOL,
                maxiter=cls.MAX_ITER
            )
            return iv

        except (ValueError, RuntimeError):
            return None

    @classmethod
    def vega_btc(
        cls,
        forward: float,
        strike: float,
        vol: float,
        ttm: float
    ) -> float:
        """Calculate vega for inverse option in BTC.

        Vega is the same for calls and puts.

        Args:
            forward: Forward price.
            strike: Strike price.
            vol: Implied volatility.
            ttm: Time to maturity in years.

        Returns:
            Vega (sensitivity to vol change).
        """
        if ttm <= 0 or vol <= 0:
            return 0.0

        d1_val = cls.d1(forward, strike, vol, ttm)
        sqrt_t = math.sqrt(ttm)

        # Standard vega divided by forward (for BTC denomination)
        return sqrt_t * norm.pdf(d1_val)

    @classmethod
    def delta_call_btc(
        cls,
        forward: float,
        strike: float,
        vol: float,
        ttm: float
    ) -> float:
        """Calculate delta for inverse call in BTC.

        Args:
            forward: Forward price.
            strike: Strike price.
            vol: Implied volatility.
            ttm: Time to maturity in years.

        Returns:
            Call delta.
        """
        if ttm <= 0:
            return 1.0 if forward > strike else 0.0

        d1_val = cls.d1(forward, strike, vol, ttm)
        return norm.cdf(d1_val)

    @classmethod
    def delta_put_btc(
        cls,
        forward: float,
        strike: float,
        vol: float,
        ttm: float
    ) -> float:
        """Calculate delta for inverse put in BTC.

        Args:
            forward: Forward price.
            strike: Strike price.
            vol: Implied volatility.
            ttm: Time to maturity in years.

        Returns:
            Put delta.
        """
        if ttm <= 0:
            return -1.0 if forward < strike else 0.0

        d1_val = cls.d1(forward, strike, vol, ttm)
        return norm.cdf(d1_val) - 1.0

    @classmethod
    def forward_call_price_quantlib(
        cls,
        forward: float,
        strike: float,
        vol: float,
        ttm: float,
        discount_factor: float = 1.0
    ) -> float:
        """Calculate forward call price using QuantLib's Black-76 formula.

        Falls back to native implementation if QuantLib is not available.

        Args:
            forward: Forward price in USD.
            strike: Strike price in USD.
            vol: Implied volatility.
            ttm: Time to maturity in years.
            discount_factor: Discount factor (default 1 for forward price).

        Returns:
            Call price in USD.
        """
        if not QUANTLIB_AVAILABLE:
            return cls.forward_call_price(forward, strike, vol, ttm, discount_factor)

        if ttm <= 0:
            return discount_factor * max(0, forward - strike)

        try:
            std_dev = vol * np.sqrt(ttm)
            return discount_factor * ql.blackFormula(
                ql.Option.Call, strike, forward, std_dev
            )
        except Exception:
            return cls.forward_call_price(forward, strike, vol, ttm, discount_factor)

    @classmethod
    def forward_put_price_quantlib(
        cls,
        forward: float,
        strike: float,
        vol: float,
        ttm: float,
        discount_factor: float = 1.0
    ) -> float:
        """Calculate forward put price using QuantLib's Black-76 formula.

        Falls back to native implementation if QuantLib is not available.

        Args:
            forward: Forward price in USD.
            strike: Strike price in USD.
            vol: Implied volatility.
            ttm: Time to maturity in years.
            discount_factor: Discount factor (default 1 for forward price).

        Returns:
            Put price in USD.
        """
        if not QUANTLIB_AVAILABLE:
            return cls.forward_put_price(forward, strike, vol, ttm, discount_factor)

        if ttm <= 0:
            return discount_factor * max(0, strike - forward)

        try:
            std_dev = vol * np.sqrt(ttm)
            return discount_factor * ql.blackFormula(
                ql.Option.Put, strike, forward, std_dev
            )
        except Exception:
            return cls.forward_put_price(forward, strike, vol, ttm, discount_factor)

    @classmethod
    def implied_volatility_quantlib(
        cls,
        price: float,
        forward: float,
        strike: float,
        ttm: float,
        option_type: str,
        discount_factor: float = 1.0
    ) -> Optional[float]:
        """Solve for implied volatility using QuantLib.

        Falls back to native implementation if QuantLib is not available.

        Args:
            price: Option price (forward price, not BTC).
            forward: Forward price.
            strike: Strike price.
            ttm: Time to maturity in years.
            option_type: 'call' or 'put'.
            discount_factor: Discount factor.

        Returns:
            Implied volatility, or None if solver fails.
        """
        if not QUANTLIB_AVAILABLE:
            return cls.implied_volatility(
                price, forward, strike, ttm, option_type, is_btc_price=False
            )

        if ttm <= 0 or price <= 0:
            return None

        try:
            opt_type = ql.Option.Call if option_type == "call" else ql.Option.Put
            return ql.blackFormulaImpliedStdDev(
                opt_type, strike, forward, price / discount_factor
            ) / np.sqrt(ttm)
        except Exception:
            return cls.implied_volatility(
                price, forward, strike, ttm, option_type, is_btc_price=False
            )


def is_quantlib_available() -> bool:
    """Check if QuantLib is available for use.

    Returns:
        True if QuantLib is installed and importable.
    """
    return QUANTLIB_AVAILABLE
