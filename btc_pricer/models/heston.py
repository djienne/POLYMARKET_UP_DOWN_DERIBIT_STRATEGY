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
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
from scipy.optimize import differential_evolution, minimize

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

    def _price_call_quantlib(self, forward: float, strike: float) -> float:
        """Price call using QuantLib's AnalyticHestonEngine.

        Args:
            forward: Forward price.
            strike: Strike price.

        Returns:
            Forward call price in USD.
        """
        if not QUANTLIB_AVAILABLE:
            return self._price_call_native(forward, strike)

        try:
            # Spot handle (using forward as spot since r=0)
            spot_handle = ql.QuoteHandle(ql.SimpleQuote(forward))

            # Heston process
            heston_process = ql.HestonProcess(
                self._ql_rate_handle,
                self._ql_div_handle,
                spot_handle,
                self.params.v0,
                self.params.kappa,
                self.params.theta,
                self.params.xi,
                self.params.rho
            )

            # Heston model and engine
            heston_model = ql.HestonModel(heston_process)
            engine = ql.AnalyticHestonEngine(heston_model)

            # Create option
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
        Used by _heston_charfunc_with_forward and native pricing.

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

        g = (b - rspi + d) / (b - rspi - d)

        exp_d_tau = np.exp(d * tau)

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
    model_iv: Optional[np.ndarray] = None


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
        # Multi-start optimization
        use_multi_start: bool = True,
        n_starts: int = 5,
        # Early termination
        early_termination_sse: Optional[float] = None
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
            use_multi_start: Whether to use multi-start optimization.
            n_starts: Number of starting points for multi-start optimization.
            early_termination_sse: SSE threshold for early termination of
                multi-start optimization. If the best SSE after any start is
                below this threshold, remaining starts are skipped.
                None disables early termination.
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

        # Multi-start settings
        self.use_multi_start = use_multi_start
        self.n_starts = n_starts
        self.early_termination_sse = early_termination_sse

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
        if ttm < self.very_short_dated_ttm_threshold:
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

        # Clip all guesses to bounds
        clipped_guesses = []
        for g in guesses[:self.n_starts]:
            clipped = {
                'v0': np.clip(g['v0'], bounds_dict['v0'][0], bounds_dict['v0'][1]),
                'kappa': np.clip(g['kappa'], bounds_dict['kappa'][0], bounds_dict['kappa'][1]),
                'theta': np.clip(g['theta'], bounds_dict['theta'][0], bounds_dict['theta'][1]),
                'xi': np.clip(g['xi'], bounds_dict['xi'][0], bounds_dict['xi'][1]),
                'rho': np.clip(g['rho'], bounds_dict['rho'][0], bounds_dict['rho'][1])
            }
            clipped_guesses.append(clipped)

        return clipped_guesses

    def _create_quantlib_objective(
        self,
        log_moneyness: np.ndarray,
        market_iv: np.ndarray,
        ttm: float,
        forward: float,
        weights: np.ndarray,
        atm_penalty_weight: float = 0.0
    ) -> callable:
        """Create objective function using QuantLib's HestonBlackVolSurface.

        Builds QuantLib objects once and uses HestonModel.setParams() to
        update parameters in-place. The QuantLib observer pattern propagates
        parameter changes to HestonBlackVolSurface automatically.

        Falls back to rebuilding only HestonProcess/Model if setParams is
        unavailable, while still reusing constant handles.

        Args:
            log_moneyness: Array of log-moneyness values.
            market_iv: Array of market implied volatilities.
            ttm: Time to maturity.
            forward: Forward price.
            weights: Fitting weights.
            atm_penalty_weight: Weight for ATM-level penalty (helps short-dated).

        Returns:
            Objective function for optimizer.
        """
        if not QUANTLIB_AVAILABLE:
            return None

        # Find ATM market IV for penalty term
        atm_idx = np.argmin(np.abs(log_moneyness))
        market_atm_iv = market_iv[atm_idx]

        # Pre-compute strikes for efficiency
        strikes = forward * np.exp(log_moneyness)

        # Set up QuantLib evaluation date and constant handles (built once)
        today = ql.Date.todaysDate()
        ql.Settings.instance().evaluationDate = today

        rate_handle = ql.YieldTermStructureHandle(
            ql.FlatForward(today, 0.0, ql.Actual365Fixed())
        )
        div_handle = ql.YieldTermStructureHandle(
            ql.FlatForward(today, 0.0, ql.Actual365Fixed())
        )
        spot_handle = ql.QuoteHandle(ql.SimpleQuote(forward))

        # Build full QuantLib object graph once
        heston_process = ql.HestonProcess(
            rate_handle, div_handle, spot_handle,
            0.1, 1.0, 0.1, 0.5, -0.5  # Dummy initial params
        )
        heston_model = ql.HestonModel(heston_process)
        heston_handle = ql.HestonModelHandle(heston_model)
        vol_surface = ql.HestonBlackVolSurface(heston_handle)

        # Probe setParams: if available, use fast path (reuse all objects)
        use_set_params = False
        try:
            test_arr = ql.Array(5)
            test_arr[0], test_arr[1], test_arr[2] = 0.1, 1.0, 0.1
            test_arr[3], test_arr[4] = 0.5, -0.5
            heston_model.setParams(test_arr)
            # Verify vol_surface still works after setParams
            vol_surface.blackVol(ttm, strikes[0])
            use_set_params = True
            logger.debug("QuantLib setParams fast path enabled")
        except (AttributeError, RuntimeError):
            logger.debug("QuantLib setParams unavailable, using handle-reuse fallback")

        if use_set_params:
            # Fast path: reuse all objects, update params via setParams
            def objective(x: np.ndarray) -> float:
                v0, kappa, theta, xi, rho = x
                try:
                    params_arr = ql.Array(5)
                    params_arr[0] = v0
                    params_arr[1] = kappa
                    params_arr[2] = theta
                    params_arr[3] = xi
                    params_arr[4] = rho
                    heston_model.setParams(params_arr)

                    model_iv = np.array([
                        vol_surface.blackVol(ttm, strike)
                        for strike in strikes
                    ])

                    errors = (model_iv - market_iv) ** 2
                    sse = float(np.sum(weights * errors))

                    if atm_penalty_weight > 0:
                        model_atm_iv = vol_surface.blackVol(ttm, forward)
                        atm_error = (model_atm_iv - market_atm_iv) ** 2
                        sse += atm_penalty_weight * len(log_moneyness) * atm_error

                    return sse
                except (RuntimeError, ValueError, OverflowError):
                    return 1e10
        else:
            # Fallback: reuse constant handles, rebuild process/model per call
            def objective(x: np.ndarray) -> float:
                v0, kappa, theta, xi, rho = x
                try:
                    hp = ql.HestonProcess(
                        rate_handle, div_handle, spot_handle,
                        v0, kappa, theta, xi, rho
                    )
                    hm = ql.HestonModel(hp)
                    hh = ql.HestonModelHandle(hm)
                    vs = ql.HestonBlackVolSurface(hh)

                    model_iv = np.array([
                        vs.blackVol(ttm, strike)
                        for strike in strikes
                    ])

                    errors = (model_iv - market_iv) ** 2
                    sse = float(np.sum(weights * errors))

                    if atm_penalty_weight > 0:
                        model_atm_iv = vs.blackVol(ttm, forward)
                        atm_error = (model_atm_iv - market_atm_iv) ** 2
                        sse += atm_penalty_weight * len(log_moneyness) * atm_error

                    return sse
                except (RuntimeError, ValueError, OverflowError):
                    return 1e10

        return objective

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
        # Very short-dated need much stronger ATM penalty to avoid wrong level
        if ttm < self.very_short_dated_ttm_threshold:
            atm_penalty = 50.0  # Strong penalty for very short-dated
        elif ttm < self.short_dated_ttm_threshold:
            atm_penalty = 10.0
        else:
            atm_penalty = 0.0
        ql_objective = None
        if self.use_quantlib and QUANTLIB_AVAILABLE:
            ql_objective = self._create_quantlib_objective(
                log_moneyness, market_iv, ttm, forward, weights,
                atm_penalty_weight=atm_penalty
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
                errors = (model_iv - market_iv) ** 2
                sse = float(np.sum(weights * errors))

                # ATM-level penalty for short-dated
                if atm_penalty > 0:
                    model_atm_iv = model.implied_volatility(0.0)  # k=0 is ATM
                    atm_error = (model_atm_iv - atm_iv) ** 2
                    sse += atm_penalty * len(log_moneyness) * atm_error

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
        if ttm < self.very_short_dated_ttm_threshold:
            maxiter = 250  # Very short-dated need more iterations
        elif ttm < self.short_dated_ttm_threshold:
            maxiter = 200  # Short-dated
        else:
            maxiter = 100  # Normal

        et_sse = self.early_termination_sse

        for i, init_params in enumerate(guesses):
            # Inter-start early termination: skip remaining starts if best SSE
            # is already below threshold
            if i > 0 and et_sse is not None and best_error < et_sse:
                logger.debug(
                    f"Early termination: best_error={best_error:.2e} < "
                    f"threshold={et_sse:.2e}, skipping starts {i+1}-{len(guesses)}"
                )
                break

            x0 = np.array([
                init_params['v0'], init_params['kappa'],
                init_params['theta'], init_params['xi'], init_params['rho']
            ])

            try:
                if self.optimizer == "differential_evolution":
                    # Intra-start early termination via callback
                    de_callback = None
                    if et_sse is not None:
                        # Mutable container to track best objective value within DE
                        best_in_de = [float('inf')]

                        def _make_callback(threshold, tracker):
                            def callback(xk, convergence):
                                val = objective(xk)
                                if val < tracker[0]:
                                    tracker[0] = val
                                if val < threshold:
                                    return True  # Stop DE
                                return False
                            return callback

                        de_callback = _make_callback(et_sse, best_in_de)

                    result = differential_evolution(
                        objective,
                        bounds,
                        x0=x0,  # Seed with initial guess
                        maxiter=maxiter,
                        tol=1e-7,
                        seed=42 + i,  # Different seed for each start
                        workers=1,  # Single-threaded for reproducibility
                        polish=True,  # Use L-BFGS-B to polish result
                        callback=de_callback,
                    )
                else:
                    # Fall back to L-BFGS-B
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
                    logger.debug(f"Multi-start {i+1}/{len(guesses)}: error={result.fun:.6f}")

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

        # Calculate fit statistics using fast HestonBlackVolSurface path
        model_iv = self._compute_model_iv_fast(
            params, log_moneyness, forward, ttm
        )

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
            iv_error_max=iv_error_max,
            model_iv=model_iv
        )

    def _compute_model_iv_fast(
        self,
        params: HestonParams,
        log_moneyness: np.ndarray,
        forward: float,
        ttm: float
    ) -> np.ndarray:
        """Compute model IVs using the fast HestonBlackVolSurface path.

        Falls back to the slow HestonModel.implied_volatility path if
        QuantLib is unavailable.

        Args:
            params: Fitted Heston parameters.
            log_moneyness: Array of log-moneyness values.
            forward: Forward price.
            ttm: Time to maturity.

        Returns:
            Array of model implied volatilities.
        """
        if self.use_quantlib and QUANTLIB_AVAILABLE:
            try:
                today = ql.Date.todaysDate()
                ql.Settings.instance().evaluationDate = today

                rate_handle = ql.YieldTermStructureHandle(
                    ql.FlatForward(today, 0.0, ql.Actual365Fixed())
                )
                div_handle = ql.YieldTermStructureHandle(
                    ql.FlatForward(today, 0.0, ql.Actual365Fixed())
                )
                spot_handle = ql.QuoteHandle(ql.SimpleQuote(forward))

                heston_process = ql.HestonProcess(
                    rate_handle, div_handle, spot_handle,
                    params.v0, params.kappa, params.theta,
                    params.xi, params.rho
                )
                heston_model = ql.HestonModel(heston_process)
                heston_handle = ql.HestonModelHandle(heston_model)
                vol_surface = ql.HestonBlackVolSurface(heston_handle)

                strikes = forward * np.exp(log_moneyness)
                model_iv = np.array([
                    vol_surface.blackVol(ttm, strike)
                    for strike in strikes
                ])
                return model_iv
            except (RuntimeError, ValueError, OverflowError):
                pass

        # Fallback: slow path via HestonModel
        model = HestonModel(params, self.n_integration_points, self.use_quantlib)
        return np.array([model.implied_volatility(k) for k in log_moneyness])

def check_iv_consistency(
    heston_model: HestonModel,
    log_moneyness: np.ndarray,
    market_iv: np.ndarray,
    threshold: float = 0.10
) -> Tuple[bool, float, np.ndarray]:
    """Check if Heston-implied IVs are consistent with market IVs.

    Args:
        heston_model: Calibrated Heston model.
        log_moneyness: Array of log-moneyness values.
        market_iv: Array of market implied volatilities.
        threshold: Maximum allowed relative IV error (default 10%).

    Returns:
        Tuple of (is_consistent, max_error, error_array).
    """
    model_iv = heston_model.implied_volatility_array(log_moneyness)
    relative_errors = np.abs(model_iv - market_iv) / market_iv
    max_error = np.max(relative_errors)
    is_consistent = max_error <= threshold

    return is_consistent, max_error, relative_errors


def check_iv_consistency_from_result(
    fit_result: HestonFitResult,
    market_iv: np.ndarray,
    threshold: float = 0.10
) -> Tuple[bool, float, np.ndarray]:
    """Check IV consistency using pre-computed model IVs from HestonFitResult.

    Avoids redundant IV recomputation when model_iv is already available
    from the fitting process.

    Args:
        fit_result: Result from HestonFitter.fit() containing model_iv.
        market_iv: Array of market implied volatilities.
        threshold: Maximum allowed relative IV error (default 10%).

    Returns:
        Tuple of (is_consistent, max_error, error_array).
    """
    if fit_result.model_iv is None:
        raise ValueError(
            "fit_result.model_iv is None; use check_iv_consistency instead"
        )

    relative_errors = np.abs(fit_result.model_iv - market_iv) / market_iv
    max_error = np.max(relative_errors)
    is_consistent = max_error <= threshold

    return is_consistent, max_error, relative_errors
