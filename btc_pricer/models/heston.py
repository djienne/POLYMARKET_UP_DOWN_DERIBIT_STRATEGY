"""Heston stochastic volatility model.

Implements the Heston (1993) model for option pricing with the
"Little Heston Trap" numerical stability fix from Albrecher et al. (2007).

The Heston model assumes:
    dS_t = S_t * sqrt(v_t) * dW_1
    dv_t = kappa * (theta - v_t) * dt + xi * sqrt(v_t) * dW_2
    corr(dW_1, dW_2) = rho

Parameters:
    v0: Initial variance
    kappa: Mean reversion speed
    theta: Long-term variance
    xi: Volatility of volatility
    rho: Correlation between asset and variance

QuantLib Integration:
    When use_quantlib=True and QuantLib is installed, pricing uses
    QuantLib's AnalyticHestonEngine for better numerical precision.
"""

import logging
import math
import os
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Callable, Optional, Tuple
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.integrate import quad

from .black_scholes import BlackScholes
from ..utils.fit_stats import calculate_fit_stats, calculate_iv_error_stats, FitStats

logger = logging.getLogger(__name__)

# Try to import QuantLib
QUANTLIB_AVAILABLE = False
try:
    import QuantLib as ql
    QUANTLIB_AVAILABLE = True
except ImportError:
    pass

# Optional Numba acceleration (fallback objective path)
NUMBA_AVAILABLE = False
try:
    from numba import njit
    NUMBA_AVAILABLE = True
except ImportError:
    njit = None


def _weighted_sse_python(
    model_iv: np.ndarray,
    market_iv: np.ndarray,
    weights: np.ndarray
) -> float:
    """Compute weighted sum of squared errors in pure Python/NumPy."""
    errors = model_iv - market_iv
    return float(np.sum(weights * errors * errors))


def _weighted_sse_relative_python(
    model_iv: np.ndarray,
    market_iv: np.ndarray,
    weights: np.ndarray
) -> float:
    """Weighted SSE with relative errors: sum(w * ((model-market)/market)²)."""
    rel_errors = (model_iv - market_iv) / (market_iv + 1e-10)
    return float(np.sum(weights * rel_errors * rel_errors))


if NUMBA_AVAILABLE:
    @njit(cache=True, fastmath=False)
    def _weighted_sse_numba(
        model_iv: np.ndarray,
        market_iv: np.ndarray,
        weights: np.ndarray
    ) -> float:
        """Compute weighted SSE using Numba (strict numerical mode)."""
        total = 0.0
        n = len(model_iv)
        for i in range(n):
            diff = model_iv[i] - market_iv[i]
            total += weights[i] * diff * diff
        return total

    @njit(cache=True, fastmath=False)
    def _weighted_sse_relative_numba(
        model_iv: np.ndarray,
        market_iv: np.ndarray,
        weights: np.ndarray
    ) -> float:
        """Compute weighted relative SSE using Numba (strict numerical mode)."""
        total = 0.0
        n = len(model_iv)
        for i in range(n):
            rel = (model_iv[i] - market_iv[i]) / (market_iv[i] + 1e-10)
            total += weights[i] * rel * rel
        return total


def _run_one_heston_de_start(
    log_moneyness, market_iv, ttm, forward, weights,
    bounds, x0, maxiter, atm_penalty_weight,
    use_quantlib, n_integration_points,
    early_termination_sse, seed, use_relative_error=True,
    popsize=15, level_bias_penalty=0.0
):
    """Run a single Heston DE start (top-level for ProcessPoolExecutor).

    Each worker creates its own objective and QuantLib objects so nothing
    unpicklable crosses process boundaries.

    Returns:
        Tuple of (params_array, sse, success).
    """
    atm_idx = int(np.argmin(np.abs(log_moneyness)))
    market_atm_iv = float(market_iv[atm_idx])
    strikes = forward * np.exp(log_moneyness)

    # Select SSE function based on relative error flag
    sse_fn = _weighted_sse_relative_python if use_relative_error else _weighted_sse_python

    obj = None
    if use_quantlib:
        try:
            import QuantLib as ql
            today = ql.Date.todaysDate()
            ql.Settings.instance().evaluationDate = today
            rate_handle = ql.YieldTermStructureHandle(
                ql.FlatForward(today, 0.0, ql.Actual365Fixed())
            )
            div_handle = ql.YieldTermStructureHandle(
                ql.FlatForward(today, 0.0, ql.Actual365Fixed())
            )

            def obj(x):
                v0, kappa, theta, xi, rho = x
                try:
                    spot = ql.QuoteHandle(ql.SimpleQuote(forward))
                    proc = ql.HestonProcess(
                        rate_handle, div_handle, spot,
                        v0, kappa, theta, xi, rho
                    )
                    model = ql.HestonModel(proc)
                    surface = ql.HestonBlackVolSurface(
                        ql.HestonModelHandle(model)
                    )
                    model_iv = np.array([
                        surface.blackVol(ttm, float(s)) for s in strikes
                    ])
                    sse = sse_fn(model_iv, market_iv, weights)
                    if atm_penalty_weight > 0:
                        atm_iv = surface.blackVol(ttm, forward)
                        atm_error = ((atm_iv - market_atm_iv) / (market_atm_iv + 1e-10)) ** 2 if use_relative_error else (atm_iv - market_atm_iv) ** 2
                        sse += atm_penalty_weight * len(log_moneyness) * atm_error
                    if level_bias_penalty > 0:
                        rel_errors = (model_iv - market_iv) / (market_iv + 1e-10)
                        mean_bias = float(np.mean(rel_errors))
                        sse += level_bias_penalty * len(log_moneyness) * mean_bias ** 2
                    return sse
                except (RuntimeError, ValueError, OverflowError):
                    return 1e10
        except ImportError:
            pass

    if obj is None:
        # Pure-Python fallback (no QuantLib)
        def obj(x):
            v0, kappa, theta, xi, rho = x
            try:
                params = HestonParams(
                    v0=v0, kappa=kappa, theta=theta,
                    xi=xi, rho=rho, ttm=ttm
                )
                model = HestonModel(params, n_integration_points, False)
                model_iv = np.array([
                    model.implied_volatility(k) for k in log_moneyness
                ])
                sse = sse_fn(model_iv, market_iv, weights)
                if atm_penalty_weight > 0:
                    atm_iv = model.implied_volatility(0.0)
                    atm_error = ((atm_iv - market_atm_iv) / (market_atm_iv + 1e-10)) ** 2 if use_relative_error else (atm_iv - market_atm_iv) ** 2
                    sse += atm_penalty_weight * len(log_moneyness) * atm_error
                if level_bias_penalty > 0:
                    rel_errors = (model_iv - market_iv) / (market_iv + 1e-10)
                    mean_bias = float(np.mean(rel_errors))
                    sse += level_bias_penalty * len(log_moneyness) * mean_bias ** 2
                return sse
            except (ValueError, RuntimeError, OverflowError):
                return 1e10

    de_callback = None
    if early_termination_sse is not None:
        def _cb(xk, convergence):
            return obj(xk) < early_termination_sse
        de_callback = _cb

    try:
        result = differential_evolution(
            obj, bounds, x0=x0,
            maxiter=maxiter, tol=1e-7, seed=seed,
            workers=1, polish=True, callback=de_callback,
            popsize=popsize,
        )
        return (np.array(result.x), float(result.fun), bool(result.success))
    except Exception:
        return (None, float('inf'), False)


@dataclass
class HestonParams:
    """Heston model parameters."""
    v0: float      # Initial variance, > 0
    kappa: float   # Mean reversion speed, > 0
    theta: float   # Long-term variance, > 0
    xi: float      # Vol-of-vol, > 0
    rho: float     # Correlation, in (-1, 1)
    ttm: float     # Time to maturity, > 0

    def __post_init__(self):
        """Validate parameters."""
        if self.v0 <= 0:
            raise ValueError(f"v0 must be positive, got {self.v0}")
        if self.kappa <= 0:
            raise ValueError(f"kappa must be positive, got {self.kappa}")
        if self.theta <= 0:
            raise ValueError(f"theta must be positive, got {self.theta}")
        if self.xi <= 0:
            raise ValueError(f"xi must be positive, got {self.xi}")
        if not -1 < self.rho < 1:
            raise ValueError(f"rho must be in (-1, 1), got {self.rho}")
        if self.ttm <= 0:
            raise ValueError(f"ttm must be positive, got {self.ttm}")

    def feller_condition(self) -> bool:
        """Check if Feller condition is satisfied.

        2*kappa*theta > xi^2 ensures variance process stays positive.

        Returns:
            True if Feller condition is satisfied.
        """
        return 2 * self.kappa * self.theta > self.xi ** 2

    def feller_ratio(self) -> float:
        """Calculate Feller ratio.

        Returns:
            2*kappa*theta / xi^2 (should be > 1 for Feller condition).
        """
        if self.xi == 0:
            return float('inf')
        return 2 * self.kappa * self.theta / (self.xi ** 2)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "v0": float(self.v0),
            "kappa": float(self.kappa),
            "theta": float(self.theta),
            "xi": float(self.xi),
            "rho": float(self.rho),
            "ttm": float(self.ttm),
            "feller_satisfied": bool(self.feller_condition()),
            "feller_ratio": float(self.feller_ratio())
        }


class HestonModel:
    """Heston stochastic volatility model with Little Heston Trap fix.

    The characteristic function implementation follows Albrecher et al. (2007)
    for numerical stability, especially for long maturities.

    When use_quantlib=True and QuantLib is installed, pricing uses
    QuantLib's AnalyticHestonEngine for better numerical precision.

    References:
        - Heston (1993): "A Closed-Form Solution for Options with Stochastic
          Volatility with Applications to Bond and Currency Options"
        - Albrecher et al. (2007): "The Little Heston Trap"
    """

    def __init__(
        self,
        params: HestonParams,
        n_integration_points: int = 256,
        use_quantlib: bool = True
    ):
        """Initialize the Heston model.

        Args:
            params: Heston parameters.
            n_integration_points: Number of points for numerical integration.
            use_quantlib: Use QuantLib for pricing when available (default: True).
        """
        self.params = params
        self.n_integration_points = n_integration_points
        self.use_quantlib = use_quantlib and QUANTLIB_AVAILABLE
        # Cache for IV computations
        self._iv_cache = {}

        # Initialize QuantLib model if requested
        if self.use_quantlib:
            self._setup_quantlib_model()

    @property
    def ttm(self) -> float:
        """Time to maturity (for VolatilitySurface protocol)."""
        return self.params.ttm

    def _setup_quantlib_model(self) -> None:
        """Initialize QuantLib Heston model components."""
        if not QUANTLIB_AVAILABLE:
            return

        # Set evaluation date
        today = ql.Date.todaysDate()
        ql.Settings.instance().evaluationDate = today

        # Calculate expiry date from TTM
        # Use ceil and ensure at least 1 day for very short-dated options
        days_to_expiry = max(1, int(np.ceil(self.params.ttm * 365)))
        self._ql_expiry = today + ql.Period(days_to_expiry, ql.Days)

        # Flat rate curves (zero for forward pricing)
        self._ql_rate_handle = ql.YieldTermStructureHandle(
            ql.FlatForward(today, 0.0, ql.Actual365Fixed())
        )
        self._ql_div_handle = ql.YieldTermStructureHandle(
            ql.FlatForward(today, 0.0, ql.Actual365Fixed())
        )


    def _get_ql_engine(self, forward: float) -> "ql.PricingEngine":
        """Get or build a cached QuantLib AnalyticHestonEngine for this forward.

        Reuses the HestonProcess/HestonModel/Engine across strikes to avoid
        rebuilding ~6 Python-level QuantLib objects per call.  Only the
        option payoff (strike) changes between calls.

        Args:
            forward: Forward price (used as spot since r=0).

        Returns:
            QuantLib AnalyticHestonEngine.
        """
        # Build once per forward price
        if (getattr(self, '_ql_cached_engine', None) is None
                or getattr(self, '_ql_cached_forward', None) != forward):
            spot_handle = ql.QuoteHandle(ql.SimpleQuote(forward))
            heston_process = ql.HestonProcess(
                self._ql_rate_handle,
                self._ql_div_handle,
                spot_handle,
                self.params.v0,
                self.params.kappa,
                self.params.theta,
                self.params.xi,
                self.params.rho,
            )
            heston_model = ql.HestonModel(heston_process)
            self._ql_cached_engine = ql.AnalyticHestonEngine(heston_model)
            self._ql_cached_forward = forward
        return self._ql_cached_engine

    def _price_call_quantlib(self, forward: float, strike: float) -> float:
        """Price call using QuantLib's AnalyticHestonEngine.

        Reuses a cached engine for the same forward, so only a new payoff
        and option object are created per call (instead of a full pricing
        graph with HestonProcess + HestonModel + Engine).

        Args:
            forward: Forward price.
            strike: Strike price.

        Returns:
            Forward call price in USD.
        """
        if not QUANTLIB_AVAILABLE:
            return self._price_call_native(forward, strike)

        try:
            engine = self._get_ql_engine(forward)

            payoff = ql.PlainVanillaPayoff(ql.Option.Call, strike)
            exercise = ql.EuropeanExercise(self._ql_expiry)
            option = ql.EuropeanOption(payoff, exercise)
            option.setPricingEngine(engine)

            return option.NPV()
        except (RuntimeError, ValueError, OverflowError) as e:
            # Fall back to native implementation on error
            logger.debug(f"QuantLib pricing failed, falling back to native: {e}")
            return self._price_call_native(forward, strike)

    def _price_call_native(self, forward: float, strike: float) -> float:
        """Price call using native characteristic function integration.

        Args:
            forward: Forward price.
            strike: Strike price.

        Returns:
            Forward call price in USD.
        """
        if strike <= 0 or forward <= 0:
            return 0.0

        def integrand(phi: float) -> float:
            if phi < 1e-10:
                return 0.0
            try:
                cf_phi_minus_i = self._heston_charfunc_with_forward(phi, forward, shift=-1)
                cf_phi = self._heston_charfunc_with_forward(phi, forward, shift=0)
                numerator = cf_phi_minus_i - strike * cf_phi
                denominator = 1j * phi * (strike ** (1j * phi))
                result = np.real(numerator / denominator)
                if np.isnan(result) or np.isinf(result):
                    return 0.0
                return result
            except (ValueError, RuntimeError, OverflowError, ZeroDivisionError):
                return 0.0

        # Numerical integration using rectangular method for stability
        P = 0.0
        umax = 100
        N = max(self.n_integration_points * 8, 2000)  # Scale up for native pricing
        dphi = umax / N

        for i in range(1, N):
            phi = dphi * (2 * i + 1) / 2
            P += dphi * integrand(phi)

        price = (forward - strike) / 2 + np.real(P) / np.pi
        price = max(0.0, price)
        price = min(forward, price)

        return price

    def _heston_charfunc(self, phi: complex) -> complex:
        """Heston characteristic function.

        Based on the reference implementation from the notebook.
        For forward pricing, r=0 and we work with forward F instead of spot S.

        Args:
            phi: Complex integration variable.

        Returns:
            Characteristic function value.
        """
        return self._heston_charfunc_core(phi)

    def call_price(self, forward: float, strike: float) -> float:
        """Calculate call option price.

        Uses QuantLib's AnalyticHestonEngine when use_quantlib=True,
        otherwise falls back to characteristic function integration.

        Args:
            forward: Forward price.
            strike: Strike price.

        Returns:
            Forward call price in USD.
        """
        if self.use_quantlib:
            return self._price_call_quantlib(forward, strike)
        return self._price_call_native(forward, strike)

    def call_price_native(self, forward: float, strike: float) -> float:
        """Calculate call via characteristic function (bypasses QuantLib).

        Useful for comparison or when QuantLib gives unexpected results.

        Args:
            forward: Forward price.
            strike: Strike price.

        Returns:
            Forward call price in USD.
        """
        return self._price_call_native(forward, strike)

    def _heston_charfunc_core(self, u: complex) -> complex:
        """Core Heston characteristic function computation.

        Computes the characteristic function value for a complex argument u.
        This is the shared mathematical core used by both _heston_charfunc
        and _heston_charfunc_with_forward.

        The formula follows Albrecher et al. (2007) "Little Heston Trap":
        phi = ((1-g*exp(d*tau))/(1-g))^(-2a/sigma^2)
              * exp(a*tau*(b-rspi+d)/sigma^2 + v0*(b-rspi+d)*((1-exp_d_tau)/(1-g*exp_d_tau))/sigma^2)

        where:
        - d = sqrt((rho*sigma*u*i - b)^2 + (u*i + u^2)*sigma^2)
        - g = (b - rspi + d) / (b - rspi - d)
        - a = kappa * theta
        - b = kappa (for forward pricing with lambd=0)
        - rspi = rho * sigma * u * i

        Args:
            u: Complex integration variable.

        Returns:
            Characteristic function value (without forward price factor).
        """
        v0 = self.params.v0
        kappa = self.params.kappa
        theta = self.params.theta
        sigma = self.params.xi
        rho = self.params.rho
        tau = self.params.ttm

        a = kappa * theta
        b = kappa

        rspi = rho * sigma * u * 1j

        d = np.sqrt((rho * sigma * u * 1j - b) ** 2 + (u * 1j + u ** 2) * sigma ** 2)

        d_tau = d * tau

        # Taylor branch for small |d*tau| to avoid catastrophic cancellation
        # in (1 - exp(d*tau)) and (1 - g*exp(d*tau))/(1-g)
        if np.abs(d_tau) < 0.05:
            # exp(d*tau) - 1 ≈ d*tau + (d*tau)²/2 + (d*tau)³/6
            exp_d_tau_m1 = d_tau + 0.5 * d_tau**2 + d_tau**3 / 6.0

            gminus = b - rspi - d  # denominator of g
            # numer = (1 - g*exp(d*tau)) * gminus
            #       = gminus - (b-rspi+d)*exp(d*tau)
            #       = (b-rspi-d) - (b-rspi+d)*(1 + exp_d_tau_m1)
            #       = -2*d - (b-rspi+d)*exp_d_tau_m1
            numer = -2 * d - (b - rspi + d) * exp_d_tau_m1

            # (1-g*exp(d*tau))/(1-g) = numer / (-2*d)
            denom_ratio = numer / (-2 * d)
            term2 = denom_ratio ** (-2 * a / sigma ** 2)

            # v0 term: (b-rspi+d) * (1-exp(d*tau)) / (1-g*exp(d*tau)) / sigma^2
            #        = (b-rspi+d) * (-exp_d_tau_m1) * gminus / numer / sigma^2
            v0_term = v0 * (b - rspi + d) * (-exp_d_tau_m1) * gminus / numer / sigma**2

            exp2 = np.exp(a * tau * (b - rspi + d) / sigma**2 + v0_term)
            return term2 * exp2

        g = (b - rspi + d) / (b - rspi - d)

        exp_d_tau = np.exp(d_tau)

        # term2 = ((1 - g * exp_d_tau) / (1 - g)) ^ (-2*a / sigma^2)
        term2_base = (1 - g * exp_d_tau) / (1 - g)
        term2 = term2_base ** (-2 * a / sigma ** 2)

        # exp2
        exp2_arg = a * tau * (b - rspi + d) / sigma ** 2 + \
                   v0 * (b - rspi + d) * ((1 - exp_d_tau) / (1 - g * exp_d_tau)) / sigma ** 2
        exp2 = np.exp(exp2_arg)

        return term2 * exp2

    def _heston_charfunc_with_forward(self, phi: float, forward: float, shift: int = 0) -> complex:
        """Heston characteristic function including forward price.

        Args:
            phi: Real part of integration variable.
            forward: Forward price.
            shift: Imaginary shift (0 or -1).

        Returns:
            Characteristic function value with F^(i*(phi+shift*i)) factor.
        """
        u = complex(phi, shift)
        F_factor = forward ** (1j * u)
        return F_factor * self._heston_charfunc_core(u)

    def put_price(self, forward: float, strike: float) -> float:
        """Calculate put option price via put-call parity.

        P = C - (F - K)

        Args:
            forward: Forward price.
            strike: Strike price.

        Returns:
            Forward put price in USD.
        """
        call = self.call_price(forward, strike)
        return call - (forward - strike)

    def implied_volatility_strike(self, strike: float, forward: float) -> float:
        """Calculate implied volatility for a given strike.

        Prices the option with Heston, then inverts to get BS IV.

        Args:
            strike: Strike price.
            forward: Forward price.

        Returns:
            Black-Scholes implied volatility.
        """
        # Check cache
        cache_key = (strike, forward)
        if cache_key in self._iv_cache:
            return self._iv_cache[cache_key]

        call_price = self.call_price(forward, strike)

        # Intrinsic value for a call
        intrinsic = max(0, forward - strike)

        if call_price <= 1e-10:
            # Deep OTM call - use minimum IV
            iv = 0.01
        elif call_price >= forward * 0.95:
            # Price too close to forward - use maximum IV
            iv = 2.0
        else:
            # Solve for IV
            iv = BlackScholes.implied_volatility(
                call_price, forward, strike, self.params.ttm,
                "call", is_btc_price=False
            )
            if iv is None:
                # Fallback to sqrt(v0) as approximation
                iv = np.sqrt(self.params.v0)

        self._iv_cache[cache_key] = iv
        return iv

    def implied_volatility(self, k: float) -> float:
        """Calculate implied volatility for log-moneyness k.

        Args:
            k: Log-moneyness ln(K/F).

        Returns:
            Implied volatility.
        """
        # Use normalized forward = 1, strike = exp(k)
        return self.implied_volatility_strike(np.exp(k), 1.0)

    def implied_volatility_array(self, k_array: np.ndarray) -> np.ndarray:
        """Calculate implied volatility for array of log-moneyness values.

        Args:
            k_array: Array of log-moneyness values.

        Returns:
            Array of implied volatilities.
        """
        return np.array([self.implied_volatility(k) for k in k_array])

    def clear_cache(self):
        """Clear the IV cache."""
        self._iv_cache.clear()


@dataclass
class HestonFitResult:
    """Result of Heston model calibration."""
    params: Optional[HestonParams]
    success: bool
    r_squared: float
    rmse: float
    max_residual: float
    n_points: int
    message: str
    iv_error_mean: float = 0.0
    iv_error_max: float = 0.0


class HestonFitter:
    """Calibrate Heston model to market implied volatilities.

    Features TTM-adaptive parameter bounds and multi-start optimization
    for improved fitting of short-dated options.
    """

    def __init__(
        self,
        v0_bounds: Tuple[float, float] = (0.01, 4.0),
        kappa_bounds: Tuple[float, float] = (0.1, 10.0),
        theta_bounds: Tuple[float, float] = (0.01, 4.0),
        xi_bounds: Tuple[float, float] = (0.1, 5.0),
        rho_bounds: Tuple[float, float] = (-0.99, 0.99),
        optimizer: str = "differential_evolution",
        n_integration_points: int = 256,
        use_quantlib: bool = True,
        # Short-dated TTM thresholds and bounds
        short_dated_ttm_threshold: float = 0.10,
        short_dated_xi_bounds: Tuple[float, float] = (0.1, 10.0),
        short_dated_kappa_bounds: Tuple[float, float] = (0.01, 15.0),
        very_short_dated_ttm_threshold: float = 0.02,
        very_short_dated_xi_bounds: Tuple[float, float] = (0.1, 15.0),
        very_short_dated_kappa_bounds: Tuple[float, float] = (0.001, 20.0),
        # Ultra-short-dated TTM thresholds and bounds
        ultra_short_dated_ttm_threshold: float = 0.01,
        ultra_short_dated_xi_bounds: Tuple[float, float] = (0.1, 10.0),
        ultra_short_dated_kappa_bounds: Tuple[float, float] = (0.5, 5.0),
        ultra_short_dated_theta_factor: Tuple[float, float] = (0.5, 2.0),
        # Multi-start optimization
        use_multi_start: bool = True,
        n_starts: int = 5,
        # QuantLib objective implementation strategy
        quantlib_objective_impl: str = "optimized",
        # Optional Numba acceleration for fallback objective
        enable_numba_fallback: bool = True,
        numba_strict_mode: bool = True,
        # Early termination: skip remaining multi-starts when SSE below threshold
        early_termination_sse: Optional[float] = None,
        # Relative error objective
        use_relative_error: bool = True,
        # Gaussian near-ATM weighting for short TTM
        short_ttm_gaussian_weighting: bool = True,
        short_ttm_gaussian_sigma_base: float = 0.05,
        short_ttm_gaussian_sigma_ttm_scale: float = 2.0,
        short_ttm_gaussian_floor: float = 0.1,
        max_workers: int = 4,
    ):
        """Initialize the Heston fitter.

        Args:
            v0_bounds: Bounds for initial variance.
            kappa_bounds: Bounds for mean reversion speed.
            theta_bounds: Bounds for long-term variance.
            xi_bounds: Bounds for vol-of-vol.
            rho_bounds: Bounds for correlation.
            optimizer: Optimization method.
            n_integration_points: Points for numerical integration.
            use_quantlib: Use QuantLib for pricing when available (default: True).
            short_dated_ttm_threshold: TTM threshold for short-dated options (~36 days).
            short_dated_xi_bounds: Extended xi bounds for short-dated options.
            short_dated_kappa_bounds: Extended kappa bounds for short-dated options.
            very_short_dated_ttm_threshold: TTM threshold for very short-dated (~7 days).
            very_short_dated_xi_bounds: Extended xi bounds for very short-dated.
            very_short_dated_kappa_bounds: Extended kappa bounds for very short-dated.
            ultra_short_dated_ttm_threshold: TTM threshold for ultra-short-dated (~3.5 days).
            ultra_short_dated_xi_bounds: xi bounds for ultra-short-dated.
            ultra_short_dated_kappa_bounds: Tight kappa bounds for ultra-short-dated.
            ultra_short_dated_theta_factor: theta = factor * v0 range for ultra-short-dated.
            use_multi_start: Whether to use multi-start optimization.
            n_starts: Number of starting points for multi-start optimization.
            quantlib_objective_impl: QuantLib objective implementation mode:
                "optimized" (object reuse) or "legacy" (rebuild objects per call).
            enable_numba_fallback: Enable Numba acceleration on fallback objective.
            numba_strict_mode: If True, keep conservative numerical behavior.
            early_termination_sse: When set, skip remaining multi-starts once
                best SSE falls below this threshold. None = disabled.
            max_workers: Maximum number of parallel workers for multi-start optimization.
        """
        self.v0_bounds = v0_bounds
        self.kappa_bounds = kappa_bounds
        self.theta_bounds = theta_bounds
        self.xi_bounds = xi_bounds
        self.rho_bounds = rho_bounds
        self.optimizer = optimizer
        self.n_integration_points = n_integration_points
        self.use_quantlib = use_quantlib

        # Short-dated settings
        self.short_dated_ttm_threshold = short_dated_ttm_threshold
        self.short_dated_xi_bounds = short_dated_xi_bounds
        self.short_dated_kappa_bounds = short_dated_kappa_bounds
        self.very_short_dated_ttm_threshold = very_short_dated_ttm_threshold
        self.very_short_dated_xi_bounds = very_short_dated_xi_bounds
        self.very_short_dated_kappa_bounds = very_short_dated_kappa_bounds
        self.ultra_short_dated_ttm_threshold = ultra_short_dated_ttm_threshold
        self.ultra_short_dated_xi_bounds = ultra_short_dated_xi_bounds
        self.ultra_short_dated_kappa_bounds = ultra_short_dated_kappa_bounds
        self.ultra_short_dated_theta_factor = ultra_short_dated_theta_factor

        # Multi-start settings
        self.use_multi_start = use_multi_start
        self.n_starts = n_starts

        # QuantLib objective implementation
        self.quantlib_objective_impl = quantlib_objective_impl.lower().strip()
        if self.quantlib_objective_impl not in {"optimized", "legacy"}:
            raise ValueError(
                "quantlib_objective_impl must be 'optimized' or 'legacy'"
            )

        # Optional Numba acceleration for non-QuantLib fallback objective
        self.enable_numba_fallback = bool(enable_numba_fallback and NUMBA_AVAILABLE)
        self.numba_strict_mode = bool(numba_strict_mode)

        # Early termination
        self.early_termination_sse = early_termination_sse

        # Relative error objective
        self.use_relative_error = use_relative_error

        # Gaussian near-ATM weighting for short TTM
        self.short_ttm_gaussian_weighting = short_ttm_gaussian_weighting
        self.short_ttm_gaussian_sigma_base = short_ttm_gaussian_sigma_base
        self.short_ttm_gaussian_sigma_ttm_scale = short_ttm_gaussian_sigma_ttm_scale
        self.short_ttm_gaussian_floor = short_ttm_gaussian_floor

        # Parallel multi-start workers cap
        self.max_workers = max_workers

    def _initialize_from_bs(
        self,
        log_moneyness: np.ndarray,
        market_iv: np.ndarray,
        ttm: float
    ) -> dict:
        """Initialize Heston parameters from Black-Scholes.

        Uses ATM volatility and smile shape to estimate initial parameters.

        Args:
            log_moneyness: Array of log-moneyness values.
            market_iv: Array of market implied volatilities.
            ttm: Time to maturity.

        Returns:
            Dictionary of initial parameter estimates.
        """
        # Find ATM point
        atm_idx = np.argmin(np.abs(log_moneyness))
        atm_iv = market_iv[atm_idx]

        # v0 and theta from ATM variance
        v0 = atm_iv ** 2
        theta = np.mean(market_iv ** 2)

        # Estimate rho from skew
        # Negative skew -> negative rho (typical for equities/crypto)
        otm_puts = log_moneyness < -0.1
        otm_calls = log_moneyness > 0.1

        if np.any(otm_puts) and np.any(otm_calls):
            put_iv_mean = np.mean(market_iv[otm_puts])
            call_iv_mean = np.mean(market_iv[otm_calls])
            # Skew: if puts have higher IV, rho is negative
            skew = (put_iv_mean - call_iv_mean) / atm_iv
            rho = np.clip(-skew * 2, -0.9, 0.9)
        else:
            rho = -0.5  # Default negative correlation

        # Estimate xi from curvature (smile convexity)
        if len(market_iv) >= 5:
            # Simple curvature estimate from second derivative
            iv_center = market_iv[len(market_iv)//2]
            iv_left = market_iv[len(market_iv)//4]
            iv_right = market_iv[3*len(market_iv)//4]
            curvature = (iv_left + iv_right - 2 * iv_center) / (atm_iv + 1e-6)
            xi = np.clip(np.abs(curvature) * 2 + 0.3, 0.2, 2.0)
        else:
            xi = 0.5

        # kappa: moderate mean reversion
        kappa = 2.0

        return {
            'v0': np.clip(v0, self.v0_bounds[0], self.v0_bounds[1]),
            'kappa': np.clip(kappa, self.kappa_bounds[0], self.kappa_bounds[1]),
            'theta': np.clip(theta, self.theta_bounds[0], self.theta_bounds[1]),
            'xi': np.clip(xi, self.xi_bounds[0], self.xi_bounds[1]),
            'rho': np.clip(rho, self.rho_bounds[0], self.rho_bounds[1])
        }

    def _get_ttm_adjusted_bounds(self, ttm: float, atm_iv: float = None) -> dict:
        """Get TTM-adjusted parameter bounds.

        Short-dated options need wider bounds for xi (vol-of-vol) and kappa
        (mean reversion) to capture steep smiles and fast-decaying behavior.

        For very short-dated options, v0 bounds are tightened around ATM variance
        since v0 dominates the ATM IV level (ATM_IV ≈ sqrt(v0) for short TTM).

        Args:
            ttm: Time to maturity in years.
            atm_iv: ATM implied volatility (optional, used to anchor v0 for short-dated).

        Returns:
            Dictionary of bounds for each parameter.
        """
        if ttm < self.ultra_short_dated_ttm_threshold:
            logger.debug(f"Using ultra-short-dated bounds for TTM={ttm:.4f}")
            if atm_iv is not None:
                atm_var = atm_iv ** 2
                # Two sub-tiers within ultra-short:
                # Sub-2-day (TTM < 0.005): extreme U-shaped smiles, wide freedom
                # 2-3.5 day (0.005-0.01): normal skew, tighter bounds to avoid
                #   level inflation from high-kappa + high-theta solutions
                if ttm < 0.005:
                    xi_bounds = self.ultra_short_dated_xi_bounds
                    kappa_bounds = self.ultra_short_dated_kappa_bounds
                    v0_factor = (0.40, 0.95)
                    theta_factor = (0.1, 1.5)
                else:
                    xi_bounds = (0.1, 15.0)
                    kappa_bounds = (0.001, 20.0)
                    v0_factor = (0.55, 0.95)
                    theta_factor = (0.1, 0.85)
                v0_bounds = (max(self.v0_bounds[0], atm_var * v0_factor[0]),
                             min(self.v0_bounds[1], atm_var * v0_factor[1]))
                theta_lo = max(self.theta_bounds[0], atm_var * theta_factor[0])
                theta_hi = min(self.theta_bounds[1], atm_var * theta_factor[1])
                theta_bounds = (theta_lo, theta_hi)
                logger.debug(
                    f"Ultra-short: v0=[{v0_bounds[0]:.4f}, {v0_bounds[1]:.4f}], "
                    f"theta=[{theta_lo:.4f}, {theta_hi:.4f}]"
                )
                return {
                    'v0': v0_bounds, 'kappa': kappa_bounds,
                    'theta': theta_bounds, 'xi': xi_bounds, 'rho': self.rho_bounds
                }
            # No ATM IV available, fall through to very-short logic
        elif ttm < self.very_short_dated_ttm_threshold:
            xi_bounds = self.very_short_dated_xi_bounds
            kappa_bounds = self.very_short_dated_kappa_bounds
            logger.debug(f"Using very-short-dated bounds for TTM={ttm:.4f}")
        elif ttm < self.short_dated_ttm_threshold:
            xi_bounds = self.short_dated_xi_bounds
            kappa_bounds = self.short_dated_kappa_bounds
            logger.debug(f"Using short-dated bounds for TTM={ttm:.4f}")
        else:
            xi_bounds = self.xi_bounds
            kappa_bounds = self.kappa_bounds

        # For very short-dated, anchor v0 around ATM variance
        # Since ATM_IV ≈ sqrt(v0) for short TTM, v0 ≈ ATM_IV^2
        v0_bounds = self.v0_bounds
        if atm_iv is not None and ttm < self.short_dated_ttm_threshold:
            atm_var = atm_iv ** 2
            # Allow v0 to vary more freely around ATM variance
            # Use wider factor for very short-dated to avoid boundary issues
            if ttm < self.very_short_dated_ttm_threshold:
                v0_lower = max(self.v0_bounds[0], atm_var * 0.25)
                v0_upper = min(self.v0_bounds[1], atm_var * 4.0)
            else:
                v0_lower = max(self.v0_bounds[0], atm_var * 0.5)
                v0_upper = min(self.v0_bounds[1], atm_var * 2.0)
            v0_bounds = (v0_lower, v0_upper)
            logger.debug(f"Anchoring v0 bounds to ATM variance: [{v0_lower:.4f}, {v0_upper:.4f}]")

        return {
            'v0': v0_bounds,
            'kappa': kappa_bounds,
            'theta': self.theta_bounds,
            'xi': xi_bounds,
            'rho': self.rho_bounds
        }

    def _generate_initial_guesses(
        self,
        log_moneyness: np.ndarray,
        market_iv: np.ndarray,
        ttm: float,
        bounds_dict: dict
    ) -> list:
        """Generate multiple starting points for optimization.

        Creates diverse initial guesses to avoid local minima, especially
        important for short-dated options with steep smiles.

        Args:
            log_moneyness: Array of log-moneyness values.
            market_iv: Array of market implied volatilities.
            ttm: Time to maturity.
            bounds_dict: TTM-adjusted parameter bounds.

        Returns:
            List of initial parameter dictionaries.
        """
        base = self._initialize_from_bs(log_moneyness, market_iv, ttm)
        guesses = [base]  # BS-based guess

        # High vol-of-vol guess (for steep smiles)
        guesses.append({
            'v0': base['v0'],
            'kappa': max(bounds_dict['kappa'][0] * 2, base['kappa'] * 0.5),
            'theta': base['theta'],
            'xi': (bounds_dict['xi'][0] + bounds_dict['xi'][1]) / 2,  # Mid-range
            'rho': base['rho']
        })

        # Low mean-reversion guess (short-dated behavior)
        guesses.append({
            'v0': base['v0'],
            'kappa': bounds_dict['kappa'][0],
            'theta': base['theta'],
            'xi': min(base['xi'] * 1.5, bounds_dict['xi'][1]),
            'rho': base['rho']
        })

        # More negative rho guess (stronger skew)
        guesses.append({
            'v0': base['v0'],
            'kappa': base['kappa'],
            'theta': base['theta'],
            'xi': base['xi'],
            'rho': max(base['rho'] - 0.2, bounds_dict['rho'][0])
        })

        # For very short-dated, add aggressive guesses with varied v0
        if bounds_dict['xi'][1] > 10:
            # High xi, low kappa
            guesses.append({
                'v0': base['v0'],
                'kappa': bounds_dict['kappa'][0],
                'theta': base['theta'],
                'xi': bounds_dict['xi'][1] * 0.7,
                'rho': max(base['rho'] - 0.2, -0.95)
            })
            # Higher v0 guess (in case ATM level is underestimated)
            guesses.append({
                'v0': base['v0'] * 1.5,
                'kappa': bounds_dict['kappa'][0],
                'theta': base['theta'] * 1.5,
                'xi': bounds_dict['xi'][1] * 0.5,
                'rho': base['rho']
            })
            # Even higher v0 with moderate xi
            guesses.append({
                'v0': base['v0'] * 2.0,
                'kappa': base['kappa'],
                'theta': base['theta'] * 2.0,
                'xi': (bounds_dict['xi'][0] + bounds_dict['xi'][1]) / 3,
                'rho': base['rho']
            })
            # Near-symmetric smile guess (rho close to 0, very high xi)
            # Crypto short-dated smiles often have right-wing uptick that
            # needs less skew and more curvature to approximate
            guesses.append({
                'v0': base['v0'],
                'kappa': (bounds_dict['kappa'][0] + bounds_dict['kappa'][1]) / 2,
                'theta': base['theta'],
                'xi': bounds_dict['xi'][1] * 0.8,
                'rho': -0.2
            })
            # Moderate skew with high xi
            guesses.append({
                'v0': base['v0'],
                'kappa': bounds_dict['kappa'][0] * 2,
                'theta': base['theta'],
                'xi': bounds_dict['xi'][1] * 0.6,
                'rho': -0.4
            })

        # Clip all guesses to bounds (use all generated, not just n_starts,
        # so ultra-short gets the benefit of diverse starting points)
        n_use = max(self.n_starts, len(guesses)) if bounds_dict['xi'][1] > 10 else self.n_starts
        clipped_guesses = []
        for g in guesses[:n_use]:
            clipped = {
                'v0': np.clip(g['v0'], bounds_dict['v0'][0], bounds_dict['v0'][1]),
                'kappa': np.clip(g['kappa'], bounds_dict['kappa'][0], bounds_dict['kappa'][1]),
                'theta': np.clip(g['theta'], bounds_dict['theta'][0], bounds_dict['theta'][1]),
                'xi': np.clip(g['xi'], bounds_dict['xi'][0], bounds_dict['xi'][1]),
                'rho': np.clip(g['rho'], bounds_dict['rho'][0], bounds_dict['rho'][1])
            }
            clipped_guesses.append(clipped)

        return clipped_guesses

    def _compute_weighted_sse(
        self,
        model_iv: np.ndarray,
        market_iv: np.ndarray,
        weights: np.ndarray,
        allow_numba: bool = True
    ) -> float:
        """Compute weighted SSE with optional Numba acceleration.

        When use_relative_error=True, computes sum(w * ((model-market)/market)²)
        which normalizes error contribution across IV levels.
        """
        use_relative = getattr(self, 'use_relative_error', True)
        if allow_numba and self.enable_numba_fallback and NUMBA_AVAILABLE:
            if use_relative:
                return float(_weighted_sse_relative_numba(model_iv, market_iv, weights))
            return float(_weighted_sse_numba(model_iv, market_iv, weights))
        if use_relative:
            return _weighted_sse_relative_python(model_iv, market_iv, weights)
        return _weighted_sse_python(model_iv, market_iv, weights)

    def _create_quantlib_objective_legacy(
        self,
        log_moneyness: np.ndarray,
        market_iv: np.ndarray,
        ttm: float,
        forward: float,
        weights: np.ndarray,
        atm_penalty_weight: float = 0.0,
        level_bias_penalty: float = 0.0
    ) -> Optional[Callable[[np.ndarray], float]]:
        """Legacy QuantLib objective: rebuild model/surface at each call."""
        if not QUANTLIB_AVAILABLE:
            # Fall back to standard objective if QuantLib not available
            return None

        # Find ATM market IV for penalty term
        atm_idx = np.argmin(np.abs(log_moneyness))
        market_atm_iv = market_iv[atm_idx]

        # Capture relative error flag for closure
        use_rel = getattr(self, 'use_relative_error', True)

        # Pre-compute strikes for efficiency
        strikes = forward * np.exp(log_moneyness)

        # Set up QuantLib evaluation date
        today = ql.Date.todaysDate()
        ql.Settings.instance().evaluationDate = today

        # Flat rate curves (zero for forward pricing)
        rate_handle = ql.YieldTermStructureHandle(
            ql.FlatForward(today, 0.0, ql.Actual365Fixed())
        )
        div_handle = ql.YieldTermStructureHandle(
            ql.FlatForward(today, 0.0, ql.Actual365Fixed())
        )

        def objective(x: np.ndarray) -> float:
            v0, kappa, theta, xi, rho = x
            try:
                # Spot handle (using forward as spot since r=0)
                spot_handle = ql.QuoteHandle(ql.SimpleQuote(forward))

                # Heston process
                heston_process = ql.HestonProcess(
                    rate_handle, div_handle, spot_handle,
                    v0, kappa, theta, xi, rho
                )

                # Heston model and vol surface
                heston_model = ql.HestonModel(heston_process)
                heston_handle = ql.HestonModelHandle(heston_model)
                vol_surface = ql.HestonBlackVolSurface(heston_handle)

                # Get model IVs using HestonBlackVolSurface (fast, analytical)
                model_iv = np.array([
                    vol_surface.blackVol(ttm, strike)
                    for strike in strikes
                ])

                # Weighted SSE
                sse = self._compute_weighted_sse(
                    model_iv, market_iv, weights, allow_numba=False
                )

                # ATM-level penalty (helps short-dated convergence)
                if atm_penalty_weight > 0:
                    model_atm_iv = vol_surface.blackVol(ttm, forward)
                    if use_rel:
                        atm_error = ((model_atm_iv - market_atm_iv) / (market_atm_iv + 1e-10)) ** 2
                    else:
                        atm_error = (model_atm_iv - market_atm_iv) ** 2
                    sse += atm_penalty_weight * len(log_moneyness) * atm_error

                # Mean-bias penalty: penalize systematic level offset
                if level_bias_penalty > 0:
                    rel_errors = (model_iv - market_iv) / (market_iv + 1e-10)
                    mean_bias = float(np.mean(rel_errors))
                    sse += level_bias_penalty * len(log_moneyness) * mean_bias ** 2

                return sse
            except (RuntimeError, ValueError, OverflowError):
                return 1e10

        return objective

    def _create_quantlib_objective_optimized(
        self,
        log_moneyness: np.ndarray,
        market_iv: np.ndarray,
        ttm: float,
        forward: float,
        weights: np.ndarray,
        atm_penalty_weight: float = 0.0,
        level_bias_penalty: float = 0.0
    ) -> Optional[Callable[[np.ndarray], float]]:
        """Optimized QuantLib objective: reuse model/surface across calls."""
        if not QUANTLIB_AVAILABLE:
            return None

        # Find ATM market IV for penalty term
        atm_idx = np.argmin(np.abs(log_moneyness))
        market_atm_iv = market_iv[atm_idx]

        # Capture relative error flag for closure
        use_rel = getattr(self, 'use_relative_error', True)

        # Pre-compute strikes for efficiency
        strikes = forward * np.exp(log_moneyness)

        # Set up QuantLib evaluation date
        today = ql.Date.todaysDate()
        ql.Settings.instance().evaluationDate = today

        # Flat rate curves (zero for forward pricing)
        rate_handle = ql.YieldTermStructureHandle(
            ql.FlatForward(today, 0.0, ql.Actual365Fixed())
        )
        div_handle = ql.YieldTermStructureHandle(
            ql.FlatForward(today, 0.0, ql.Actual365Fixed())
        )

        # Reusable QuantLib graph
        spot_quote = ql.SimpleQuote(forward)
        spot_handle = ql.QuoteHandle(spot_quote)
        heston_process = ql.HestonProcess(
            rate_handle, div_handle, spot_handle,
            0.04, 2.0, 0.04, 0.3, -0.3
        )
        heston_model = ql.HestonModel(heston_process)
        heston_handle = ql.HestonModelHandle(heston_model)
        vol_surface = ql.HestonBlackVolSurface(heston_handle)

        # Reusable param array for setParams
        ql_params = ql.Array(5)

        def objective(x: np.ndarray) -> float:
            v0, kappa, theta, xi, rho = x
            try:
                # QuantLib HestonModel.setParams order:
                # [theta, kappa, sigma(xi), rho, v0]
                ql_params[0] = float(theta)
                ql_params[1] = float(kappa)
                ql_params[2] = float(xi)
                ql_params[3] = float(rho)
                ql_params[4] = float(v0)
                heston_model.setParams(ql_params)

                model_iv = np.empty(len(strikes), dtype=np.float64)
                for i, strike in enumerate(strikes):
                    model_iv[i] = vol_surface.blackVol(ttm, float(strike))

                sse = self._compute_weighted_sse(
                    model_iv, market_iv, weights, allow_numba=False
                )

                if atm_penalty_weight > 0:
                    model_atm_iv = vol_surface.blackVol(ttm, forward)
                    if use_rel:
                        atm_error = ((model_atm_iv - market_atm_iv) / (market_atm_iv + 1e-10)) ** 2
                    else:
                        atm_error = (model_atm_iv - market_atm_iv) ** 2
                    sse += atm_penalty_weight * len(log_moneyness) * atm_error

                # Mean-bias penalty: penalize systematic level offset
                if level_bias_penalty > 0:
                    rel_errors = (model_iv - market_iv) / (market_iv + 1e-10)
                    mean_bias = float(np.mean(rel_errors))
                    sse += level_bias_penalty * len(log_moneyness) * mean_bias ** 2

                return sse
            except (RuntimeError, ValueError, OverflowError):
                return 1e10

        return objective

    def _create_quantlib_objective(
        self,
        log_moneyness: np.ndarray,
        market_iv: np.ndarray,
        ttm: float,
        forward: float,
        weights: np.ndarray,
        atm_penalty_weight: float = 0.0,
        level_bias_penalty: float = 0.0
    ) -> Optional[Callable[[np.ndarray], float]]:
        """Select configured QuantLib objective implementation."""
        if not QUANTLIB_AVAILABLE:
            return None

        if self.quantlib_objective_impl == "legacy":
            return self._create_quantlib_objective_legacy(
                log_moneyness, market_iv, ttm, forward, weights,
                atm_penalty_weight, level_bias_penalty
            )

        try:
            return self._create_quantlib_objective_optimized(
                log_moneyness, market_iv, ttm, forward, weights,
                atm_penalty_weight, level_bias_penalty
            )
        except Exception as exc:
            logger.warning(
                "Optimized QuantLib objective initialization failed (%s); "
                "falling back to legacy objective.",
                exc
            )
            return self._create_quantlib_objective_legacy(
                log_moneyness, market_iv, ttm, forward, weights,
                atm_penalty_weight, level_bias_penalty
            )

    def fit(
        self,
        log_moneyness: np.ndarray,
        market_iv: np.ndarray,
        ttm: float,
        weights: Optional[np.ndarray] = None,
        forward: float = 1.0
    ) -> HestonFitResult:
        """Fit Heston model to market implied volatilities.

        Uses TTM-adaptive bounds and multi-start optimization for robust
        fitting, especially for short-dated options with steep smiles.

        Args:
            log_moneyness: Array of log-moneyness values ln(K/F).
            market_iv: Array of market implied volatilities.
            ttm: Time to maturity in years.
            weights: Optional weights for fitting (e.g., vega weights).
            forward: Forward price (used for strike computation).

        Returns:
            HestonFitResult with fitted parameters and diagnostics.
        """
        if len(log_moneyness) != len(market_iv):
            raise ValueError("log_moneyness and market_iv must have same length")

        if len(log_moneyness) < 5:
            return HestonFitResult(
                params=None,
                success=False,
                r_squared=0.0,
                rmse=float('inf'),
                max_residual=float('inf'),
                n_points=len(log_moneyness),
                message="Need at least 5 data points for Heston fitting"
            )

        if weights is None:
            weights = np.ones(len(log_moneyness))

        # Normalize weights
        weights = weights / np.sum(weights) * len(weights)

        # Find ATM IV (closest to log-moneyness = 0)
        atm_idx = np.argmin(np.abs(log_moneyness))
        atm_iv = market_iv[atm_idx]

        # Gaussian near-ATM weighting for sub-1d TTM only: down-weight noisy
        # deep OTM wings. For TTM >= 1d, wings carry important shape info.
        if (self.short_ttm_gaussian_weighting and ttm < 0.003):
            sigma = (self.short_ttm_gaussian_sigma_base
                     + self.short_ttm_gaussian_sigma_ttm_scale * ttm)
            gaussian_w = np.maximum(
                np.exp(-log_moneyness**2 / (2 * sigma**2)),
                self.short_ttm_gaussian_floor,
            )
            weights = weights * gaussian_w
            weights = weights / np.sum(weights) * len(weights)
            logger.debug(
                f"Gaussian weighting: sigma={sigma:.4f}, "
                f"min_w={gaussian_w.min():.3f}, max_w={gaussian_w.max():.3f}"
            )

        # Relative error objective (use_relative_error=True) normalizes each
        # strike's contribution by its IV level, so explicit inverse-variance
        # or vega weighting hacks are no longer needed for short-dated smiles.

        # Get TTM-adjusted bounds (with ATM anchoring for short-dated)
        bounds_dict = self._get_ttm_adjusted_bounds(ttm, atm_iv=atm_iv)

        bounds = [
            bounds_dict['v0'],
            bounds_dict['kappa'],
            bounds_dict['theta'],
            bounds_dict['xi'],
            bounds_dict['rho']
        ]

        # Try to use QuantLib-based objective for faster optimization
        # Use ATM penalty for short-dated to help with level convergence
        # Sub-2-day: inverse-variance weights handle level, moderate penalty
        # 2-3.5 day: uniform weights, strong ATM penalty to fix level
        if ttm < 0.005:
            atm_penalty = 0.0  # No penalty — let optimizer freely trade level for curvature
            level_bias_penalty = 0.0
        elif ttm < self.ultra_short_dated_ttm_threshold:
            atm_penalty = 0.0  # No penalty — let optimizer freely trade level for curvature
            level_bias_penalty = 0.0
        elif ttm < self.very_short_dated_ttm_threshold:
            atm_penalty = 50.0  # Strong penalty for very short-dated
            level_bias_penalty = 0.0
        elif ttm < self.short_dated_ttm_threshold:
            atm_penalty = 10.0
            level_bias_penalty = 0.0
        else:
            atm_penalty = 0.0
            level_bias_penalty = 0.0
        ql_objective = None
        if self.use_quantlib and QUANTLIB_AVAILABLE:
            ql_objective = self._create_quantlib_objective(
                log_moneyness, market_iv, ttm, forward, weights,
                atm_penalty_weight=atm_penalty,
                level_bias_penalty=level_bias_penalty
            )

        def fallback_objective(x: np.ndarray) -> float:
            """Weighted sum of squared IV errors (fallback when QuantLib unavailable)."""
            v0, kappa, theta, xi, rho = x

            try:
                params = HestonParams(
                    v0=v0, kappa=kappa, theta=theta,
                    xi=xi, rho=rho, ttm=ttm
                )
                model = HestonModel(params, self.n_integration_points, self.use_quantlib)

                # Compute model IVs
                model_iv = np.array([
                    model.implied_volatility(k) for k in log_moneyness
                ])

                # Weighted SSE
                sse = self._compute_weighted_sse(
                    model_iv, market_iv, weights, allow_numba=True
                )

                # ATM-level penalty for short-dated
                if atm_penalty > 0:
                    model_atm_iv = model.implied_volatility(0.0)  # k=0 is ATM
                    if getattr(self, 'use_relative_error', True):
                        atm_error = ((model_atm_iv - atm_iv) / (atm_iv + 1e-10)) ** 2
                    else:
                        atm_error = (model_atm_iv - atm_iv) ** 2
                    sse += atm_penalty * len(log_moneyness) * atm_error

                # Mean-bias penalty for ultra-short: penalize systematic
                # level offset across all strikes
                if level_bias_penalty > 0:
                    rel_errors = (model_iv - market_iv) / (market_iv + 1e-10)
                    mean_bias = float(np.mean(rel_errors))
                    sse += level_bias_penalty * len(log_moneyness) * mean_bias ** 2

                return sse

            except (ValueError, RuntimeError, OverflowError):
                return 1e10  # Large penalty for invalid parameters

        objective = ql_objective if ql_objective is not None else fallback_objective

        # Generate initial guesses for multi-start optimization
        if self.use_multi_start:
            guesses = self._generate_initial_guesses(
                log_moneyness, market_iv, ttm, bounds_dict
            )
        else:
            guesses = [self._initialize_from_bs(log_moneyness, market_iv, ttm)]

        # Multi-start optimization
        best_result = None
        best_error = float('inf')

        # Use more iterations for short-dated options
        if ttm < self.ultra_short_dated_ttm_threshold:
            maxiter = 400  # Ultra-short-dated: relative error landscape is smoother
        elif ttm < self.very_short_dated_ttm_threshold:
            maxiter = 300  # Very short-dated
        elif ttm < self.short_dated_ttm_threshold:
            maxiter = 200  # Short-dated
        else:
            maxiter = 100  # Normal

        # Larger DE population for short TTM to escape local minima
        de_popsize = 25 if ttm < self.ultra_short_dated_ttm_threshold else 15

        # Early termination threshold (None = disabled)
        et_sse = self.early_termination_sse

        # Prepare starting points
        start_x0s = [
            np.array([g['v0'], g['kappa'], g['theta'], g['xi'], g['rho']])
            for g in guesses
        ]

        # Try parallel multi-start for DE with multiple starting points.
        # Each start runs in its own process with a fresh objective, so
        # the first good result can cancel the rest.
        parallel_done = False
        n_cpus = os.cpu_count() or 1
        if (self.optimizer == "differential_evolution"
                and len(start_x0s) > 1 and n_cpus > 1):
            try:
                n_workers = min(len(start_x0s), n_cpus, self.max_workers)
                with ProcessPoolExecutor(max_workers=n_workers) as executor:
                    futures = {
                        executor.submit(
                            _run_one_heston_de_start,
                            log_moneyness, market_iv, ttm, forward, weights,
                            bounds, x0, maxiter, atm_penalty,
                            self.use_quantlib and QUANTLIB_AVAILABLE,
                            self.n_integration_points, et_sse, 42 + i,
                            getattr(self, 'use_relative_error', True),
                            de_popsize, level_bias_penalty
                        ): i
                        for i, x0 in enumerate(start_x0s)
                    }
                    for future in as_completed(futures):
                        idx = futures[future]
                        try:
                            x, fun, success = future.result()
                            if x is not None and fun < best_error:
                                best_result = SimpleNamespace(x=x, fun=fun, success=success)
                                best_error = fun
                                logger.debug(
                                    f"Multi-start {idx+1}/{len(start_x0s)}: "
                                    f"error={fun:.6f}"
                                )
                                if et_sse is not None and best_error < et_sse:
                                    logger.debug(
                                        f"Early termination after start {idx+1}: "
                                        f"SSE={best_error:.2e} < {et_sse:.2e}"
                                    )
                                    for f in futures:
                                        f.cancel()
                                    break
                        except Exception as e:
                            logger.debug(f"Multi-start {idx+1} failed: {e}")
                parallel_done = True
            except Exception as e:
                logger.debug(
                    "Parallel multi-start failed (%s), falling back to sequential", e
                )

        # Sequential fallback (or L-BFGS-B optimizer, or single start)
        if not parallel_done:
            for i, x0 in enumerate(start_x0s):
                try:
                    if self.optimizer == "differential_evolution":
                        de_callback = None
                        if et_sse is not None:
                            def _make_callback(threshold):
                                def _cb(xk, convergence):
                                    val = objective(xk)
                                    return val < threshold
                                return _cb
                            de_callback = _make_callback(et_sse)

                        result = differential_evolution(
                            objective,
                            bounds,
                            x0=x0,
                            maxiter=maxiter,
                            tol=1e-7,
                            seed=42 + i,
                            workers=1,
                            polish=True,
                            callback=de_callback,
                            popsize=de_popsize,
                        )
                    else:
                        result = minimize(
                            objective,
                            x0,
                            method='L-BFGS-B',
                            bounds=bounds,
                            options={'maxiter': 500}
                        )

                    if result.fun < best_error:
                        best_result = result
                        best_error = result.fun
                        logger.debug(f"Multi-start {i+1}/{len(start_x0s)}: error={result.fun:.6f}")

                    if et_sse is not None and best_error < et_sse:
                        logger.debug(
                            f"Early termination after start {i+1}: "
                            f"SSE={best_error:.2e} < {et_sse:.2e}"
                        )
                        break

                except Exception as e:
                    logger.debug(f"Multi-start {i+1} failed: {e}")
                    continue

        # For very short-dated, also try L-BFGS-B from best DE result as polishing
        if ttm < self.very_short_dated_ttm_threshold and best_result is not None:
            try:
                polished = minimize(
                    objective,
                    best_result.x,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 1000}
                )
                if polished.fun < best_error:
                    best_result = polished
                    best_error = polished.fun
                    logger.debug(f"L-BFGS-B polish improved: error={polished.fun:.6f}")
            except Exception as e:
                logger.debug(f"L-BFGS-B polish failed: {e}")

        # Two-phase calibration for ultra-short TTM (< 0.01):
        # Phase 1: Fix kappa=min, theta=v0 → optimize only (v0, xi, rho)
        # Phase 2: Free all 5 params starting from phase 1 result
        # This eliminates the kappa-theta pathology (high kappa + theta=bound).
        if (ttm < self.ultra_short_dated_ttm_threshold
                and best_result is not None):
            kappa_fixed = bounds[1][0]  # min kappa bound
            theta_fixed = np.clip(atm_iv ** 2, bounds[2][0], bounds[2][1])
            # Also try lower theta to compensate for xi-driven level uplift
            theta_low = np.clip(atm_iv ** 2 * 0.5, bounds[2][0], bounds[2][1])
            # Phase 1: 3-param optimization (v0, xi, rho)
            phase1_bounds = [bounds[0], bounds[3], bounds[4]]  # v0, xi, rho

            def make_phase1_obj(kf, tf):
                def phase1_obj(x3):
                    x5 = np.array([x3[0], kf, tf, x3[1], x3[2]])
                    return objective(x5)
                return phase1_obj

            phase1_obj = make_phase1_obj(kappa_fixed, theta_fixed)

            phase1_starts = [
                np.array([atm_iv**2, 8.0, -0.4]),
                np.array([atm_iv**2, 12.0, -0.2]),
                np.array([atm_iv**2 * 1.2, 6.0, -0.6]),
                np.array([atm_iv**2, 10.0, -0.8]),
                np.array([atm_iv**2, 14.0, -0.7]),
                np.array([atm_iv**2 * 0.95, 8.0, -0.9]),
                np.array([atm_iv**2 * 0.9, 5.0, -0.5]),
                # Low v0 + high xi: compensate level for curvature
                np.array([atm_iv**2 * 0.7, 10.0, -0.6]),
                np.array([atm_iv**2 * 0.6, 12.0, -0.5]),
                np.array([atm_iv**2 * 0.75, 8.0, -0.7]),
            ]
            best_phase1_x = None
            best_phase1_err = float('inf')
            for p1_x0 in phase1_starts:
                p1_x0 = np.array([
                    np.clip(p1_x0[0], phase1_bounds[0][0], phase1_bounds[0][1]),
                    np.clip(p1_x0[1], phase1_bounds[1][0], phase1_bounds[1][1]),
                    np.clip(p1_x0[2], phase1_bounds[2][0], phase1_bounds[2][1]),
                ])
                try:
                    p1_res = minimize(
                        phase1_obj, p1_x0, method='L-BFGS-B',
                        bounds=phase1_bounds, options={'maxiter': 500}
                    )
                    if p1_res.fun < best_phase1_err:
                        best_phase1_x = p1_res.x
                        best_phase1_err = p1_res.fun
                except Exception:
                    continue

            # Also try DE in phase 1 for more robust exploration
            try:
                p1_de = differential_evolution(
                    phase1_obj, phase1_bounds,
                    maxiter=200, tol=1e-7, seed=99,
                    workers=1, polish=True, popsize=20,
                )
                if p1_de.fun < best_phase1_err:
                    best_phase1_x = p1_de.x
                    best_phase1_err = p1_de.fun
            except Exception:
                pass

            if best_phase1_x is not None:
                logger.debug(
                    f"Two-phase P1: err={best_phase1_err:.6f} "
                    f"(v0={best_phase1_x[0]:.4f}, xi={best_phase1_x[1]:.2f}, "
                    f"rho={best_phase1_x[2]:.3f})"
                )
                # Phase 2: free all 5 params from phase 1 result
                x0_phase2 = np.array([
                    best_phase1_x[0], kappa_fixed, theta_fixed,
                    best_phase1_x[1], best_phase1_x[2]
                ])
                try:
                    p2_res = minimize(
                        objective, x0_phase2, method='L-BFGS-B',
                        bounds=bounds, options={'maxiter': 1000}
                    )
                    if p2_res.fun < best_error:
                        best_result = p2_res
                        best_error = p2_res.fun
                        logger.debug(
                            f"Two-phase P2 improved: err={p2_res.fun:.6f}"
                        )
                except Exception as e:
                    logger.debug(f"Two-phase P2 failed: {e}")
                # Also try phase 1 result directly (no mean reversion)
                if best_phase1_err < best_error:
                    x_p1_full = np.array([
                        best_phase1_x[0], kappa_fixed, theta_fixed,
                        best_phase1_x[1], best_phase1_x[2]
                    ])
                    best_result = SimpleNamespace(
                        x=x_p1_full, fun=best_phase1_err, success=True
                    )
                    best_error = best_phase1_err
                    logger.debug(
                        f"Two-phase P1 adopted directly: err={best_phase1_err:.6f}"
                    )

            # Also try phase 1 with lower theta (compensates xi level uplift)
            phase1_obj_low = make_phase1_obj(kappa_fixed, theta_low)
            low_theta_starts = [
                np.array([atm_iv**2 * 0.85, 8.0, -0.5]),
                np.array([atm_iv**2 * 0.80, 10.0, -0.4]),
                np.array([atm_iv**2 * 0.75, 12.0, -0.6]),
                np.array([atm_iv**2 * 0.90, 6.0, -0.7]),
            ]
            for p1_x0 in low_theta_starts:
                p1_x0 = np.array([
                    np.clip(p1_x0[0], phase1_bounds[0][0], phase1_bounds[0][1]),
                    np.clip(p1_x0[1], phase1_bounds[1][0], phase1_bounds[1][1]),
                    np.clip(p1_x0[2], phase1_bounds[2][0], phase1_bounds[2][1]),
                ])
                try:
                    p1_res = minimize(
                        phase1_obj_low, p1_x0, method='L-BFGS-B',
                        bounds=phase1_bounds, options={'maxiter': 500}
                    )
                    if p1_res.fun < best_error:
                        x_low = np.array([
                            p1_res.x[0], kappa_fixed, theta_low,
                            p1_res.x[1], p1_res.x[2]
                        ])
                        # Phase 2 from low-theta start
                        try:
                            p2_low = minimize(
                                objective, x_low, method='L-BFGS-B',
                                bounds=bounds, options={'maxiter': 1000}
                            )
                            if p2_low.fun < best_error:
                                best_result = p2_low
                                best_error = p2_low.fun
                                logger.debug(
                                    f"Low-theta P2 improved: err={p2_low.fun:.6f}"
                                )
                        except Exception:
                            pass
                        # Also try the phase 1 result directly
                        if p1_res.fun < best_error:
                            best_result = SimpleNamespace(
                                x=x_low, fun=p1_res.fun, success=True
                            )
                            best_error = p1_res.fun
                except Exception:
                    continue

        # For very short-dated, try multiple L-BFGS-B starts with ATM-matched v0
        # The differential_evolution often gets stuck in wrong local minima for short TTM
        if ttm < self.very_short_dated_ttm_threshold:
            atm_v0 = atm_iv ** 2  # v0 that matches ATM IV

            # Try several starting points with v0 near ATM variance
            atm_starts = [
                np.array([atm_v0, 5.0, atm_v0, 3.0, -0.3]),
                np.array([atm_v0 * 1.2, 3.0, atm_v0, 2.5, -0.4]),
                np.array([atm_v0 * 1.5, 2.0, atm_v0 * 1.2, 2.0, -0.3]),
                np.array([atm_v0, 1.0, atm_v0 * 0.8, 4.0, -0.5]),
                # Zero-mean-reversion starts: low kappa + high xi
                np.array([atm_v0, bounds[1][0], atm_v0, 8.0, -0.5]),
                np.array([atm_v0, bounds[1][0], atm_v0, 10.0, -0.3]),
                np.array([atm_v0, bounds[1][0], atm_v0, 6.0, -0.7]),
                np.array([atm_v0, 0.5, atm_v0, 12.0, -0.1]),
                # Low v0 + high xi: trade level for curvature
                np.array([atm_v0 * 0.80, bounds[1][0], atm_v0 * 0.7, 10.0, -0.5]),
                np.array([atm_v0 * 0.75, 3.0, atm_v0 * 0.6, 8.0, -0.6]),
                np.array([atm_v0 * 0.85, 5.0, atm_v0 * 0.5, 12.0, -0.4]),
                # Moderate kappa with lower theta for curvature without level
                np.array([atm_v0 * 0.90, 5.0, atm_v0 * 0.4, 8.0, -0.5]),
                np.array([atm_v0 * 0.85, 8.0, atm_v0 * 0.3, 10.0, -0.4]),
            ]

            for j, x0_atm in enumerate(atm_starts):
                # Clip to bounds
                x0_atm = np.array([
                    np.clip(x0_atm[0], bounds[0][0], bounds[0][1]),
                    np.clip(x0_atm[1], bounds[1][0], bounds[1][1]),
                    np.clip(x0_atm[2], bounds[2][0], bounds[2][1]),
                    np.clip(x0_atm[3], bounds[3][0], bounds[3][1]),
                    np.clip(x0_atm[4], bounds[4][0], bounds[4][1]),
                ])

                try:
                    result_atm = minimize(
                        objective,
                        x0_atm,
                        method='L-BFGS-B',
                        bounds=bounds,
                        options={'maxiter': 1000}
                    )
                    if result_atm.fun < best_error:
                        best_result = result_atm
                        best_error = result_atm.fun
                        logger.debug(f"ATM-start {j+1} improved: error={result_atm.fun:.6f}")
                except Exception as e:
                    logger.debug(f"ATM-start {j+1} failed: {e}")

        if best_result is None:
            return HestonFitResult(
                params=None,
                success=False,
                r_squared=0.0,
                rmse=float('inf'),
                max_residual=float('inf'),
                n_points=len(log_moneyness),
                message="All optimization attempts failed"
            )

        v0_fit, kappa_fit, theta_fit, xi_fit, rho_fit = best_result.x

        # Create parameters object
        try:
            params = HestonParams(
                v0=v0_fit,
                kappa=kappa_fit,
                theta=theta_fit,
                xi=xi_fit,
                rho=rho_fit,
                ttm=ttm
            )
        except ValueError as e:
            return HestonFitResult(
                params=None,
                success=False,
                r_squared=0.0,
                rmse=float('inf'),
                max_residual=float('inf'),
                n_points=len(log_moneyness),
                message=f"Invalid parameters: {e}"
            )

        # Calculate fit statistics
        model = HestonModel(params, self.n_integration_points, self.use_quantlib)
        model_iv = np.array([model.implied_volatility(k) for k in log_moneyness])

        fit_stats = calculate_fit_stats(model_iv, market_iv)
        iv_error_mean, iv_error_max = calculate_iv_error_stats(model_iv, market_iv)

        success = best_result.success or fit_stats.r_squared > 0.7

        return HestonFitResult(
            params=params,
            success=success,
            r_squared=fit_stats.r_squared,
            rmse=fit_stats.rmse,
            max_residual=fit_stats.max_residual,
            n_points=len(log_moneyness),
            message=best_result.message if hasattr(best_result, 'message') else "Optimization completed",
            iv_error_mean=iv_error_mean,
            iv_error_max=iv_error_max
        )

    def fit_with_vega_weights(
        self,
        log_moneyness: np.ndarray,
        market_iv: np.ndarray,
        ttm: float,
        vegas: np.ndarray,
        forward: float = 1.0
    ) -> HestonFitResult:
        """Fit Heston model with vega-weighted objective.

        Higher vega options (ATM) get more weight in the fit.

        Args:
            log_moneyness: Array of log-moneyness values.
            market_iv: Array of market IVs.
            ttm: Time to maturity.
            vegas: Array of option vegas.
            forward: Forward price.

        Returns:
            HestonFitResult.
        """
        weights = vegas / np.sum(vegas) * len(vegas)
        return self.fit(log_moneyness, market_iv, ttm, weights, forward)


def check_iv_consistency(
    heston_model: HestonModel,
    log_moneyness: np.ndarray,
    market_iv: np.ndarray,
    threshold: float = 0.10,
    ttm: float = None,
    relaxation: float = 0.15,
    ttm_cutoff: float = 0.05,
) -> Tuple[bool, float, np.ndarray]:
    """Check if Heston-implied IVs are consistent with market IVs.

    For short TTM, the threshold is relaxed linearly from ``threshold``
    at ``ttm_cutoff`` to ``threshold + relaxation`` at TTM=0.

    Args:
        heston_model: Calibrated Heston model.
        log_moneyness: Array of log-moneyness values.
        market_iv: Array of market implied volatilities.
        threshold: Maximum allowed relative IV error (default 10%).
        ttm: Time to maturity (optional). When provided and below
            ``ttm_cutoff``, the effective threshold is increased.
        relaxation: Extra tolerance added at TTM=0 (default 0.15).
        ttm_cutoff: TTM below which relaxation kicks in (default 0.05).

    Returns:
        Tuple of (is_consistent, max_error, error_array).
    """
    effective_threshold = threshold
    if ttm is not None and ttm_cutoff > 0 and ttm < ttm_cutoff:
        effective_threshold = threshold + relaxation * (ttm_cutoff - ttm) / ttm_cutoff

    model_iv = heston_model.implied_volatility_array(log_moneyness)
    relative_errors = np.abs(model_iv - market_iv) / market_iv
    max_error = np.max(relative_errors)
    is_consistent = max_error <= effective_threshold

    return is_consistent, max_error, relative_errors
