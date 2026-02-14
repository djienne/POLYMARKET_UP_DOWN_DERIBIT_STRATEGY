"""SSVI (Surface SVI) model fitting.

Implements the SSVI parametrization from Gatheral & Jacquier (2014):
"Arbitrage-free SVI volatility surfaces"

The SSVI formula for total implied variance is:
w(k, θ) = θ/2 * [1 + ρφk + sqrt((φk + ρ)² + 1 - ρ²)]

Where:
- k = log(K/F) is log-moneyness
- θ = ATM total variance (σ²T at k=0)
- ρ ∈ (-1, 1) controls skew
- φ > 0 controls curvature
"""

import logging
import math
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple
import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.optimize import minimize, differential_evolution, NonlinearConstraint

from ..utils.fit_stats import calculate_fit_stats, FitStats

logger = logging.getLogger(__name__)


@dataclass
class SSVIParams:
    """SSVI model parameters."""
    theta: float  # ATM total variance
    rho: float    # Skew parameter
    phi: float    # Curvature parameter
    ttm: float    # Time to maturity (for converting to IV)

    def __post_init__(self):
        """Validate parameters."""
        if self.theta <= 0:
            raise ValueError(f"theta must be positive, got {self.theta}")
        if not -1 < self.rho < 1:
            raise ValueError(f"rho must be in (-1, 1), got {self.rho}")
        if self.phi <= 0:
            raise ValueError(f"phi must be positive, got {self.phi}")
        if self.ttm <= 0:
            raise ValueError(f"ttm must be positive, got {self.ttm}")

    def butterfly_condition(self) -> bool:
        """Check if butterfly arbitrage conditions are satisfied.

        Gatheral-Jacquier (2014) Theorem 4.2 sufficient conditions:
            Condition 1: θφ(1 + |ρ|) ≤ 4
            Condition 2: θφ²(1 + |ρ|) ≤ 4

        Returns:
            True if both conditions are satisfied (no butterfly arbitrage).
        """
        factor = self.theta * (1 + abs(self.rho))
        cond1 = factor * self.phi <= 4
        cond2 = factor * self.phi ** 2 <= 4
        return cond1 and cond2

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        factor = self.theta * (1 + abs(self.rho))
        return {
            "theta": float(self.theta),
            "rho": float(self.rho),
            "phi": float(self.phi),
            "ttm": float(self.ttm),
            "butterfly_satisfied": bool(self.butterfly_condition()),
            "butterfly_cond1": float(factor * self.phi),
            "butterfly_cond2": float(factor * self.phi ** 2),
        }


class SSVIModel:
    """SSVI volatility surface model."""

    def __init__(self, params: SSVIParams):
        """Initialize the SSVI model.

        Args:
            params: SSVI parameters.
        """
        self.params = params

    @property
    def ttm(self) -> float:
        """Time to maturity (for VolatilitySurface protocol)."""
        return self.params.ttm

    def total_variance(self, k: float) -> float:
        """Calculate total implied variance w(k).

        w(k) = θ/2 * [1 + ρφk + sqrt((φk + ρ)² + 1 - ρ²)]

        Args:
            k: Log-moneyness ln(K/F).

        Returns:
            Total implied variance.
        """
        theta = self.params.theta
        rho = self.params.rho
        phi = self.params.phi

        term = phi * k + rho
        sqrt_term = math.sqrt(term * term + 1 - rho * rho)

        return 0.5 * theta * (1 + rho * phi * k + sqrt_term)

    def implied_variance(self, k: float) -> float:
        """Calculate implied variance σ²(k).

        σ²(k) = w(k) / T

        Args:
            k: Log-moneyness.

        Returns:
            Implied variance.
        """
        return self.total_variance(k) / self.params.ttm

    def implied_volatility(self, k: float) -> float:
        """Calculate implied volatility σ(k).

        Args:
            k: Log-moneyness.

        Returns:
            Implied volatility.
        """
        var = self.implied_variance(k)
        return math.sqrt(max(0, var))

    def implied_volatility_strike(self, strike: float, forward: float) -> float:
        """Calculate implied volatility for a given strike.

        Args:
            strike: Strike price.
            forward: Forward price.

        Returns:
            Implied volatility.
        """
        k = math.log(strike / forward)
        return self.implied_volatility(k)

    def total_variance_array(self, k_array: np.ndarray) -> np.ndarray:
        """Calculate total variance for array of log-moneyness values.

        Args:
            k_array: Array of log-moneyness values.

        Returns:
            Array of total variances.
        """
        theta = self.params.theta
        rho = self.params.rho
        phi = self.params.phi

        term = phi * k_array + rho
        sqrt_term = np.sqrt(term * term + 1 - rho * rho)

        return 0.5 * theta * (1 + rho * phi * k_array + sqrt_term)

    def implied_volatility_array(self, k_array: np.ndarray) -> np.ndarray:
        """Calculate implied volatility for array of log-moneyness values.

        Args:
            k_array: Array of log-moneyness values.

        Returns:
            Array of implied volatilities.
        """
        var = self.total_variance_array(k_array) / self.params.ttm
        return np.sqrt(np.maximum(0, var))


@dataclass
class SSVIFitResult:
    """Result of SSVI fitting."""
    params: SSVIParams
    success: bool
    r_squared: float
    rmse: float
    max_residual: float
    n_points: int
    message: str


class _SSVIObjective:
    """Picklable SSVI objective for parallel differential_evolution (workers=-1).

    Encapsulates all data as instance attributes so pickle can serialize it
    across process boundaries.
    """

    def __init__(self, log_moneyness, market_iv, ttm, weights,
                 phi_bounds, use_relative_error, regularization_lambda):
        self.log_moneyness = log_moneyness
        self.market_iv = market_iv
        self.ttm = ttm
        self.weights = weights
        self.market_total_var = market_iv ** 2 * ttm
        self.phi_center = (phi_bounds[0] + phi_bounds[1]) / 4
        self.phi_max_sq = phi_bounds[1] ** 2
        self.use_relative_error = use_relative_error
        self.regularization_lambda = regularization_lambda

    def __call__(self, params):
        theta, rho, phi = params
        term = phi * self.log_moneyness + rho
        sqrt_term = np.sqrt(np.maximum(term ** 2 + 1 - rho ** 2, 1e-10))
        model_var = 0.5 * theta * (1 + rho * phi * self.log_moneyness + sqrt_term)

        if self.use_relative_error:
            model_iv = np.sqrt(np.maximum(model_var / self.ttm, 1e-10))
            rel_errors = (model_iv - self.market_iv) / self.market_iv
            sse = np.sum(self.weights * rel_errors ** 2)
            reg = (self.regularization_lambda
                   * (phi - self.phi_center) ** 2 / self.phi_max_sq)
            return sse + reg
        else:
            residuals = (model_var - self.market_total_var) ** 2
            return np.sum(self.weights * residuals)


class SSVIFitter:
    """Fit SSVI model to market implied volatilities."""

    def __init__(
        self,
        rho_bounds: Tuple[float, float] = (-0.99, 0.99),
        phi_bounds: Tuple[float, float] = (0.001, 5.0),
        theta_bounds: Tuple[float, float] = (0.001, 10.0),
        optimizer: str = "L-BFGS-B",
        short_dated_ttm_threshold: float = 0.10,
        short_dated_phi_bounds: Tuple[float, float] = (0.001, 20.0),
        very_short_dated_ttm_threshold: float = 0.02,
        very_short_dated_phi_bounds: Tuple[float, float] = (0.001, 200.0),
        use_multi_start: bool = True,
        n_starts: int = 5,
        use_global_optimizer: bool = True,
        global_optimizer: str = "differential_evolution",
        use_relative_error: bool = True,
        regularization_lambda: float = 0.001
    ):
        """Initialize the SSVI fitter.

        Args:
            rho_bounds: Bounds for rho parameter.
            phi_bounds: Bounds for phi parameter.
            theta_bounds: Bounds for theta parameter.
            optimizer: Scipy optimizer to use.
            short_dated_ttm_threshold: TTM threshold for short-dated options (~36 days).
            short_dated_phi_bounds: Wider phi bounds for short-dated options.
            very_short_dated_ttm_threshold: TTM threshold for very short-dated options (~7 days).
            very_short_dated_phi_bounds: Even wider phi bounds for very short-dated options.
            use_multi_start: Whether to use multi-start optimization.
            n_starts: Number of starting points for multi-start optimization.
            use_global_optimizer: Whether to fall back to global optimizer if needed.
            global_optimizer: Global optimizer to use (differential_evolution).
            use_relative_error: Whether to use relative error in IV space.
            regularization_lambda: Regularization strength for extreme phi values.
        """
        self.rho_bounds = rho_bounds
        self.phi_bounds = phi_bounds
        self.theta_bounds = theta_bounds
        self.optimizer = optimizer
        self.short_dated_ttm_threshold = short_dated_ttm_threshold
        self.short_dated_phi_bounds = short_dated_phi_bounds
        self.very_short_dated_ttm_threshold = very_short_dated_ttm_threshold
        self.very_short_dated_phi_bounds = very_short_dated_phi_bounds
        self.use_multi_start = use_multi_start
        self.n_starts = n_starts
        self.use_global_optimizer = use_global_optimizer
        self.global_optimizer = global_optimizer
        self.use_relative_error = use_relative_error
        self.regularization_lambda = regularization_lambda

    def _generate_initial_guesses(
        self,
        log_moneyness: np.ndarray,
        market_iv: np.ndarray,
        ttm: float,
        phi_bounds: Tuple[float, float]
    ) -> List[np.ndarray]:
        """Generate multiple starting points for optimization.

        Args:
            log_moneyness: Array of log-moneyness values.
            market_iv: Array of market implied volatilities.
            ttm: Time to maturity.
            phi_bounds: Bounds for phi parameter.

        Returns:
            List of initial parameter arrays [theta, rho, phi].
        """
        market_total_var = market_iv ** 2 * ttm

        # Data-driven theta estimate from ATM variance
        atm_idx = np.argmin(np.abs(log_moneyness))
        theta_atm = market_total_var[atm_idx]

        # Estimate rho from left/right wing IV difference
        left_mask = log_moneyness < -0.05
        right_mask = log_moneyness > 0.05
        if np.any(left_mask) and np.any(right_mask):
            left_iv = np.mean(market_iv[left_mask])
            right_iv = np.mean(market_iv[right_mask])
            # Higher left IV -> more negative rho
            iv_diff = (left_iv - right_iv) / (left_iv + right_iv)
            rho_est = np.clip(-iv_diff * 2, -0.8, 0.3)
        else:
            rho_est = -0.3

        # Estimate phi from curvature (quadratic fit)
        if len(log_moneyness) >= 5:
            try:
                coeffs = np.polyfit(log_moneyness, market_iv, 2)
                curvature = coeffs[0]
                atm_iv = market_iv[atm_idx]
                # phi relates to curvature: higher curvature -> higher phi
                phi_est = np.clip(abs(curvature) / atm_iv * 10, phi_bounds[0], phi_bounds[1] / 2)
            except (np.linalg.LinAlgError, ValueError):
                phi_est = 1.0
        else:
            phi_est = 1.0

        # Generate variations - include high-phi guesses for short-dated options
        guesses = [
            np.array([theta_atm, rho_est, phi_est]),  # Data-driven
            np.array([theta_atm, -0.3, 0.5]),  # Conservative
            np.array([theta_atm, -0.5, 2.0]),  # Higher skew/curvature
            np.array([theta_atm, rho_est, phi_bounds[1] / 4]),  # Higher phi
            np.array([theta_atm * 0.8, rho_est, phi_est * 2]),  # Lower theta, higher phi
        ]

        # For very short-dated options, add more aggressive high-phi starting points
        # These help capture the steep smile curvature typical of short expiries
        if phi_bounds[1] > 20:
            guesses.extend([
                np.array([theta_atm, rho_est, phi_bounds[1] / 2]),  # Very high phi
                np.array([theta_atm, -0.2, phi_bounds[1] / 3]),  # High phi, moderate skew
                np.array([theta_atm * 1.2, rho_est, phi_bounds[1] / 2]),  # Higher theta + high phi
            ])

        return guesses[:max(self.n_starts, 8)]  # Allow more starts for short-dated

    def _compute_model_iv(
        self,
        params: np.ndarray,
        log_moneyness: np.ndarray,
        ttm: float
    ) -> np.ndarray:
        """Compute model implied volatility from parameters.

        Args:
            params: Array [theta, rho, phi].
            log_moneyness: Array of log-moneyness values.
            ttm: Time to maturity.

        Returns:
            Array of model implied volatilities.
        """
        theta, rho, phi = params
        term = phi * log_moneyness + rho
        sqrt_term = np.sqrt(np.maximum(term ** 2 + 1 - rho ** 2, 1e-10))
        model_var = 0.5 * theta * (1 + rho * phi * log_moneyness + sqrt_term)
        return np.sqrt(np.maximum(model_var / ttm, 1e-10))

    def _create_objective(
        self,
        log_moneyness: np.ndarray,
        market_iv: np.ndarray,
        ttm: float,
        weights: np.ndarray,
        phi_bounds: Tuple[float, float] = (0.001, 5.0)
    ) -> Callable[[np.ndarray], float]:
        """Create the objective function.

        Returns an _SSVIObjective instance which is picklable, allowing
        differential_evolution to use workers=-1 for parallel population
        evaluation.

        Args:
            log_moneyness: Array of log-moneyness values.
            market_iv: Array of market implied volatilities.
            ttm: Time to maturity.
            weights: Fitting weights.
            phi_bounds: Phi bounds (used to set regularization center).

        Returns:
            Picklable callable objective.
        """
        return _SSVIObjective(
            log_moneyness, market_iv, ttm, weights,
            phi_bounds, self.use_relative_error, self.regularization_lambda
        )

    def _run_local_optimization(
        self,
        objective,
        x0: np.ndarray,
        bounds: List[Tuple[float, float]],
        butterfly_constraint
    ) -> Tuple[np.ndarray, float, bool]:
        """Run local optimization from a starting point.

        Args:
            objective: Objective function.
            x0: Initial guess.
            bounds: Parameter bounds.
            butterfly_constraint: Butterfly constraint function.

        Returns:
            Tuple of (best_params, best_value, success).
        """
        constraints = [{"type": "ineq", "fun": butterfly_constraint}]

        result = minimize(
            objective,
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 500, "ftol": 1e-10}
        )

        if not result.success:
            # Try without constraints as fallback
            result = minimize(
                objective,
                x0,
                method=self.optimizer,
                bounds=bounds,
                options={"maxiter": 500}
            )

        return result.x, result.fun, result.success

    def _run_global_optimization(
        self,
        objective,
        bounds: List[Tuple[float, float]],
        butterfly_constraint
    ) -> Tuple[np.ndarray, float, bool]:
        """Run global optimization using differential evolution.

        Args:
            objective: Objective function.
            bounds: Parameter bounds.
            butterfly_constraint: Butterfly constraint function.

        Returns:
            Tuple of (best_params, best_value, success).
        """
        # Create nonlinear constraint for both butterfly conditions
        nlc = NonlinearConstraint(
            lambda x: np.array([
                4 - x[0] * (1 + abs(x[1])) * x[2],
                4 - x[0] * (1 + abs(x[1])) * x[2] ** 2,
            ]),
            0, np.inf
        )

        result = differential_evolution(
            objective,
            bounds=bounds,
            constraints=nlc,
            maxiter=200,
            polish=True,
            seed=42,
            workers=4,
            updating='deferred'
        )

        return result.x, result.fun, result.success

    def _calculate_fit_stats(
        self,
        params: SSVIParams,
        log_moneyness: np.ndarray,
        market_iv: np.ndarray
    ) -> FitStats:
        """Calculate fit statistics.

        Args:
            params: Fitted SSVI parameters.
            log_moneyness: Array of log-moneyness values.
            market_iv: Array of market implied volatilities.

        Returns:
            FitStats with r_squared, rmse, and max_residual.
        """
        model = SSVIModel(params)
        model_iv = model.implied_volatility_array(log_moneyness)
        return calculate_fit_stats(model_iv, market_iv)

    def fit(
        self,
        log_moneyness: np.ndarray,
        market_iv: np.ndarray,
        ttm: float,
        weights: Optional[np.ndarray] = None
    ) -> SSVIFitResult:
        """Fit SSVI model to market data.

        Args:
            log_moneyness: Array of log-moneyness values ln(K/F).
            market_iv: Array of market implied volatilities.
            ttm: Time to maturity in years.
            weights: Optional weights for fitting.

        Returns:
            SSVIFitResult with fitted parameters and diagnostics.
        """
        if len(log_moneyness) != len(market_iv):
            raise ValueError("log_moneyness and market_iv must have same length")

        if len(log_moneyness) < 3:
            return SSVIFitResult(
                params=None,
                success=False,
                r_squared=0.0,
                rmse=float('inf'),
                max_residual=float('inf'),
                n_points=len(log_moneyness),
                message="Need at least 3 data points for fitting"
            )

        if weights is None:
            weights = np.ones(len(log_moneyness))

        # Select phi bounds based on TTM (tiered for short-dated options)
        # Very short-dated options need extremely high phi to capture steep smiles
        ttm_days = ttm * 365
        if ttm < self.very_short_dated_ttm_threshold:
            phi_bounds = self.very_short_dated_phi_bounds
            logger.info(f"SSVI fit: very short-dated regime (TTM={ttm_days:.1f}d), phi_bounds={phi_bounds}")
        elif ttm < self.short_dated_ttm_threshold:
            phi_bounds = self.short_dated_phi_bounds
            logger.info(f"SSVI fit: short-dated regime (TTM={ttm_days:.1f}d), phi_bounds={phi_bounds}")
        else:
            phi_bounds = self.phi_bounds
            logger.info(f"SSVI fit: standard regime (TTM={ttm_days:.1f}d), phi_bounds={phi_bounds}")

        # Scale theta bounds by TTM relative to a reference (3 months)
        # This ensures short-dated options can have appropriately small theta values
        # θ (total variance) = σ² × T, so expected θ scales linearly with TTM
        reference_ttm = 0.25  # 3 months reference
        ttm_scale = max(ttm / reference_ttm, 0.01)  # Floor to prevent numerical issues
        theta_lower = self.theta_bounds[0] * ttm_scale
        theta_upper = self.theta_bounds[1] * ttm_scale
        scaled_theta_bounds = (theta_lower, theta_upper)

        # Set up bounds
        bounds = [
            scaled_theta_bounds,
            self.rho_bounds,
            phi_bounds
        ]

        # Butterfly constraint (min of both Gatheral-Jacquier conditions)
        def butterfly_constraint(params: np.ndarray) -> float:
            theta, rho, phi = params
            factor = theta * (1 + abs(rho))
            return min(4 - factor * phi, 4 - factor * phi ** 2)

        # Create objective function (pass phi_bounds for adaptive regularization)
        objective = self._create_objective(log_moneyness, market_iv, ttm, weights, phi_bounds)

        # Generate initial guesses
        initial_guesses = self._generate_initial_guesses(
            log_moneyness, market_iv, ttm, phi_bounds
        )

        best_params = None
        best_value = float('inf')
        best_success = False

        # Multi-start local optimization
        if self.use_multi_start:
            for x0 in initial_guesses:
                # Clip initial guess to scaled bounds
                x0[0] = np.clip(x0[0], scaled_theta_bounds[0], scaled_theta_bounds[1])
                x0[1] = np.clip(x0[1], self.rho_bounds[0], self.rho_bounds[1])
                x0[2] = np.clip(x0[2], phi_bounds[0], phi_bounds[1])

                try:
                    params_opt, value, success = self._run_local_optimization(
                        objective, x0, bounds, butterfly_constraint
                    )
                    if value < best_value:
                        best_params = params_opt
                        best_value = value
                        best_success = success
                except (ValueError, RuntimeError, np.linalg.LinAlgError) as e:
                    logger.debug(f"Local optimization failed for starting point: {e}")
                    continue
        else:
            # Single start
            x0 = initial_guesses[0]
            x0[0] = np.clip(x0[0], scaled_theta_bounds[0], scaled_theta_bounds[1])
            x0[1] = np.clip(x0[1], self.rho_bounds[0], self.rho_bounds[1])
            x0[2] = np.clip(x0[2], phi_bounds[0], phi_bounds[1])

            best_params, best_value, best_success = self._run_local_optimization(
                objective, x0, bounds, butterfly_constraint
            )

        # Check if we need global optimization fallback
        need_global = False
        if best_params is not None:
            try:
                temp_params = SSVIParams(
                    theta=best_params[0],
                    rho=best_params[1],
                    phi=best_params[2],
                    ttm=ttm
                )
                fit_stats = self._calculate_fit_stats(
                    temp_params, log_moneyness, market_iv
                )
                need_global = fit_stats.r_squared < 0.85
            except ValueError:
                need_global = True
        else:
            need_global = True

        # Global optimizer fallback
        if need_global and self.use_global_optimizer:
            try:
                global_params, global_value, global_success = self._run_global_optimization(
                    objective, bounds, butterfly_constraint
                )
                if global_value < best_value:
                    best_params = global_params
                    best_value = global_value
                    best_success = global_success
            except (ValueError, RuntimeError) as e:
                logger.debug(f"Global optimization failed, keeping best local result: {e}")

        if best_params is None:
            return SSVIFitResult(
                params=None,
                success=False,
                r_squared=0.0,
                rmse=float('inf'),
                max_residual=float('inf'),
                n_points=len(log_moneyness),
                message="Optimization failed for all starting points"
            )

        theta_fit, rho_fit, phi_fit = best_params

        # Create parameters object
        try:
            params = SSVIParams(
                theta=theta_fit,
                rho=rho_fit,
                phi=phi_fit,
                ttm=ttm
            )
        except ValueError as e:
            return SSVIFitResult(
                params=None,
                success=False,
                r_squared=0.0,
                rmse=float('inf'),
                max_residual=float('inf'),
                n_points=len(log_moneyness),
                message=f"Invalid parameters: {e}"
            )

        # Calculate fit statistics
        fit_stats = self._calculate_fit_stats(params, log_moneyness, market_iv)

        # Post-fit butterfly validation: reject arbitrageable params
        butterfly_ok = params.butterfly_condition()
        if not butterfly_ok:
            factor = params.theta * (1 + abs(params.rho))
            logger.warning(
                f"SSVI fit rejected: butterfly violated "
                f"(cond1={factor * params.phi:.4f}, cond2={factor * params.phi ** 2:.4f}, limit=4)"
            )

        success = (best_success or fit_stats.r_squared > 0.8) and butterfly_ok

        return SSVIFitResult(
            params=params,
            success=success,
            r_squared=fit_stats.r_squared,
            rmse=fit_stats.rmse,
            max_residual=fit_stats.max_residual,
            n_points=len(log_moneyness),
            message="Optimization completed" + (" (global optimizer used)" if need_global and self.use_global_optimizer else "")
                + ("" if butterfly_ok else " [butterfly violated]")
        )

    def fit_with_vega_weights(
        self,
        log_moneyness: np.ndarray,
        market_iv: np.ndarray,
        ttm: float,
        vegas: np.ndarray
    ) -> SSVIFitResult:
        """Fit SSVI with vega-weighted objective.

        Higher vega options (ATM) get more weight in the fit.

        Args:
            log_moneyness: Array of log-moneyness values.
            market_iv: Array of market IVs.
            ttm: Time to maturity.
            vegas: Array of option vegas.

        Returns:
            SSVIFitResult.
        """
        # Normalize vegas to weights
        weights = vegas / np.sum(vegas) * len(vegas)
        return self.fit(log_moneyness, market_iv, ttm, weights)


# =============================================================================
# SSVI Surface Fit (Gatheral & Jacquier 2014)
# =============================================================================
# Proper SSVI surface parametrization with:
# - Global ρ (skew) shared across all slices
# - φ(θ) = η·θ^(-λ) power-law curvature (global η, λ)
# - Monotone increasing θ(T) ATM total variance term structure
# - Per-slice butterfly constraints: θ·φ(θ)·(1+|ρ|) ≤ 4 AND θ·φ(θ)²·(1+|ρ|) ≤ 4


@dataclass
class SSVISliceData:
    """Market data for one expiry slice."""
    expiry_name: str
    ttm: float
    log_moneyness: np.ndarray
    market_iv: np.ndarray
    forward: float
    weights: Optional[np.ndarray] = None


@dataclass
class SSVISurfaceParams:
    """Global SSVI surface parameters (Gatheral & Jacquier 2014).

    The surface formula:
        w(k, T) = θ(T)/2 · [1 + ρ·φ(θ)·k + √((φ(θ)·k + ρ)² + 1 - ρ²)]
    where φ(θ) = η·θ^(-λ), 0 ≤ λ ≤ 0.5, η > 0.
    """
    rho: float                # Global skew
    eta: float                # Power-law scale: φ(θ) = η·θ^(-λ)
    lam: float                # Power-law exponent (0 ≤ λ ≤ 0.5)
    thetas: List[float]       # Per-expiry θ values, monotone increasing
    ttms: List[float]         # Corresponding TTMs, sorted ascending
    expiry_names: List[str]   # Labels

    def __post_init__(self):
        """Validate parameters."""
        if not -1 < self.rho < 1:
            raise ValueError(f"rho must be in (-1, 1), got {self.rho}")
        if self.eta <= 0:
            raise ValueError(f"eta must be positive, got {self.eta}")
        if not 0 <= self.lam <= 0.5:
            raise ValueError(f"lam must be in [0, 0.5], got {self.lam}")
        if len(self.thetas) != len(self.ttms):
            raise ValueError("thetas and ttms must have same length")
        if len(self.thetas) != len(self.expiry_names):
            raise ValueError("thetas and expiry_names must have same length")
        # Validate monotonicity
        for i in range(1, len(self.thetas)):
            if self.thetas[i] < self.thetas[i - 1]:
                raise ValueError(
                    f"thetas must be monotone increasing, but "
                    f"theta[{i}]={self.thetas[i]:.6f} < theta[{i-1}]={self.thetas[i-1]:.6f}"
                )

    def phi(self, theta: float) -> float:
        """Compute curvature φ(θ) = η·θ^(-λ)."""
        return self.eta * theta ** (-self.lam)

    def to_per_slice_params(self) -> List[SSVIParams]:
        """Decompose to per-slice SSVIParams for downstream compatibility."""
        result = []
        for theta, ttm in zip(self.thetas, self.ttms):
            result.append(SSVIParams(
                theta=theta,
                rho=self.rho,
                phi=self.phi(theta),
                ttm=ttm,
            ))
        return result

    def interpolate_theta(self, ttm: float) -> float:
        """PCHIP interpolation in log-TTM space for arbitrary TTM.

        Uses monotone cubic Hermite splines (C¹ smooth, no overshoot).
        Extrapolates flat beyond the fitted range.
        """
        if len(self.ttms) == 1:
            return self.thetas[0]

        log_ttms = np.log(self.ttms)
        log_target = np.log(ttm)

        # Clamp to range (flat extrapolation)
        if log_target <= log_ttms[0]:
            return self.thetas[0]
        if log_target >= log_ttms[-1]:
            return self.thetas[-1]

        pchip = PchipInterpolator(log_ttms, self.thetas, extrapolate=False)
        return float(pchip(log_target))

    def get_params_for_ttm(self, ttm: float) -> SSVIParams:
        """Get SSVIParams for an arbitrary TTM via interpolation."""
        theta = self.interpolate_theta(ttm)
        return SSVIParams(theta=theta, rho=self.rho, phi=self.phi(theta), ttm=ttm)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "rho": float(self.rho),
            "eta": float(self.eta),
            "lam": float(self.lam),
            "thetas": [float(t) for t in self.thetas],
            "ttms": [float(t) for t in self.ttms],
            "expiry_names": list(self.expiry_names),
            "per_slice": [p.to_dict() for p in self.to_per_slice_params()],
        }


@dataclass
class SSVISurfaceFitResult:
    """Result of SSVI surface fitting."""
    params: Optional[SSVISurfaceParams]
    per_slice_params: List[SSVIParams]  # Decomposed for downstream use
    success: bool
    aggregate_r_squared: float
    per_slice_r_squared: List[float]
    message: str


class _SSVISurfaceObjective:
    """Picklable objective for SSVI surface joint fitting.

    Optimizes x = [ρ, η, λ, δ₁, ..., δₙ] where θᵢ = Σⱼ₌₁ⁱ δⱼ².
    """

    def __init__(self, slices_data, use_relative_error):
        # Store per-slice data as plain arrays for pickle
        self.n_slices = len(slices_data)
        self.log_moneyness = [s.log_moneyness for s in slices_data]
        self.market_iv = [s.market_iv for s in slices_data]
        self.ttms = [s.ttm for s in slices_data]
        self.weights = [
            s.weights if s.weights is not None else np.ones(len(s.market_iv))
            for s in slices_data
        ]
        self.use_relative_error = use_relative_error

    def _unpack(self, x):
        """Unpack optimization vector into (rho, eta, lam, thetas)."""
        rho = x[0]
        eta = x[1]
        lam = x[2]
        deltas = x[3:]
        # Monotone reparametrization: θᵢ = Σⱼ₌₁ⁱ δⱼ²
        thetas = np.cumsum(deltas ** 2)
        return rho, eta, lam, thetas

    def __call__(self, x):
        rho, eta, lam, thetas = self._unpack(x)

        total_error = 0.0
        penalty = 0.0

        for i in range(self.n_slices):
            theta_i = thetas[i]
            if theta_i <= 0:
                return 1e10

            phi_i = eta * theta_i ** (-lam)
            k = self.log_moneyness[i]
            ttm_i = self.ttms[i]
            w = self.weights[i]
            mkt_iv = self.market_iv[i]

            # SSVI total variance
            term = phi_i * k + rho
            sqrt_term = np.sqrt(np.maximum(term ** 2 + 1 - rho ** 2, 1e-10))
            model_var = 0.5 * theta_i * (1 + rho * phi_i * k + sqrt_term)
            model_iv = np.sqrt(np.maximum(model_var / ttm_i, 1e-10))

            if self.use_relative_error:
                rel_errors = (model_iv - mkt_iv) / mkt_iv
                total_error += np.sum(w * rel_errors ** 2)
            else:
                total_error += np.sum(w * (model_iv - mkt_iv) ** 2)

            # Butterfly penalty per slice (both conditions)
            factor_i = theta_i * (1 + abs(rho))
            butterfly_val1 = factor_i * phi_i
            butterfly_val2 = factor_i * phi_i ** 2
            if butterfly_val1 > 4:
                penalty += 1000.0 * (butterfly_val1 - 4) ** 2
            if butterfly_val2 > 4:
                penalty += 1000.0 * (butterfly_val2 - 4) ** 2

        return total_error + penalty


class SSVISurfaceFitter:
    """Joint SSVI surface fitter (Gatheral & Jacquier 2014).

    Fits global (ρ, η, λ) and monotone θ(T) across multiple expiry slices.
    Uses per-slice SSVIFitter for warm-starting, then joint differential_evolution.
    Falls back to independent per-slice fits if joint fit degrades quality.
    """

    def __init__(
        self,
        per_slice_fitter: SSVIFitter,
        rho_bounds: Tuple[float, float] = (-0.99, 0.99),
        eta_bounds: Tuple[float, float] = (0.01, 10.0),
        lam_bounds: Tuple[float, float] = (0.0, 0.5),
        maxiter: int = 300,
        workers: int = 4,
        use_relative_error: bool = True,
        fallback_to_independent: bool = True,
    ):
        self.per_slice_fitter = per_slice_fitter
        self.rho_bounds = rho_bounds
        self.eta_bounds = eta_bounds
        self.lam_bounds = lam_bounds
        self.maxiter = maxiter
        self.workers = min(workers, 4)  # Cap at 4 per CLAUDE.md
        self.use_relative_error = use_relative_error
        self.fallback_to_independent = fallback_to_independent

    def _compute_slice_r_squared(
        self,
        params: SSVISurfaceParams,
        slices: List[SSVISliceData],
    ) -> List[float]:
        """Compute per-slice R² for surface params."""
        r_squareds = []
        per_slice = params.to_per_slice_params()
        for sp, sl in zip(per_slice, slices):
            model = SSVIModel(sp)
            model_iv = model.implied_volatility_array(sl.log_moneyness)
            ss_res = np.sum((model_iv - sl.market_iv) ** 2)
            ss_tot = np.sum((sl.market_iv - np.mean(sl.market_iv)) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
            r_squareds.append(r2)
        return r_squareds

    def fit(self, slices: List[SSVISliceData]) -> SSVISurfaceFitResult:
        """Fit SSVI surface jointly across expiry slices.

        Step 1: Per-slice warm start using existing SSVIFitter.
        Step 2: Construct initial x0 from per-slice results.
        Step 3: Joint differential_evolution.
        Step 4: Validate vs per-slice — fall back if aggregate R² drops.
        """
        if len(slices) < 2:
            return SSVISurfaceFitResult(
                params=None,
                per_slice_params=[],
                success=False,
                aggregate_r_squared=0.0,
                per_slice_r_squared=[],
                message=f"Need at least 2 slices, got {len(slices)}",
            )

        # Sort slices by TTM ascending
        slices = sorted(slices, key=lambda s: s.ttm)

        # Step 1: Per-slice warm start
        per_slice_results = []
        per_slice_params_indep = []
        for sl in slices:
            result = self.per_slice_fitter.fit(
                sl.log_moneyness, sl.market_iv, sl.ttm, sl.weights
            )
            per_slice_results.append(result)
            if result.success and result.params is not None:
                per_slice_params_indep.append(result.params)
            else:
                per_slice_params_indep.append(None)

        # Need at least 2 successful per-slice fits to warm start
        successful = [p for p in per_slice_params_indep if p is not None]
        if len(successful) < 2:
            return SSVISurfaceFitResult(
                params=None,
                per_slice_params=[p for p in per_slice_params_indep if p is not None],
                success=False,
                aggregate_r_squared=0.0,
                per_slice_r_squared=[
                    r.r_squared if r.success else 0.0 for r in per_slice_results
                ],
                message="Fewer than 2 per-slice fits succeeded",
            )

        # Compute independent per-slice aggregate R² for fallback comparison
        indep_r2s = [r.r_squared if r.success else 0.0 for r in per_slice_results]
        indep_n_points = [r.n_points for r in per_slice_results]
        total_points = sum(indep_n_points)
        indep_agg_r2 = sum(
            r2 * n for r2, n in zip(indep_r2s, indep_n_points)
        ) / total_points if total_points > 0 else 0.0

        # Step 2: Construct initial guess
        rho_init, eta_init, lam_init, delta_init = self._construct_initial_guess(
            slices, per_slice_params_indep
        )

        # Step 3: Joint optimization
        n = len(slices)
        # δ bounds: small positive lower bound to avoid zero thetas
        delta_bounds = [(0.001, 5.0)] * n
        bounds = [self.rho_bounds, self.eta_bounds, self.lam_bounds] + delta_bounds

        objective = _SSVISurfaceObjective(slices, self.use_relative_error)

        x0 = np.concatenate([[rho_init, eta_init, lam_init], delta_init])

        try:
            result = differential_evolution(
                objective,
                bounds=bounds,
                x0=x0,
                maxiter=self.maxiter,
                polish=True,
                seed=42,
                workers=self.workers,
                updating='deferred',
            )
            opt_success = result.success
            opt_x = result.x
            opt_msg = result.message
        except (ValueError, RuntimeError) as e:
            logger.warning(f"SSVI surface differential_evolution failed: {e}")
            opt_success = False
            opt_x = None
            opt_msg = str(e)

        if opt_x is None:
            # Fall back to independent fits
            return SSVISurfaceFitResult(
                params=None,
                per_slice_params=[p for p in per_slice_params_indep if p is not None],
                success=False,
                aggregate_r_squared=indep_agg_r2,
                per_slice_r_squared=indep_r2s,
                message=f"Joint optimization failed: {opt_msg}",
            )

        # Unpack result
        rho_opt = opt_x[0]
        eta_opt = opt_x[1]
        lam_opt = opt_x[2]
        deltas_opt = opt_x[3:]
        thetas_opt = list(np.cumsum(deltas_opt ** 2))

        try:
            surface_params = SSVISurfaceParams(
                rho=float(rho_opt),
                eta=float(eta_opt),
                lam=float(lam_opt),
                thetas=thetas_opt,
                ttms=[s.ttm for s in slices],
                expiry_names=[s.expiry_name for s in slices],
            )
        except ValueError as e:
            logger.warning(f"SSVI surface params validation failed: {e}")
            return SSVISurfaceFitResult(
                params=None,
                per_slice_params=[p for p in per_slice_params_indep if p is not None],
                success=False,
                aggregate_r_squared=indep_agg_r2,
                per_slice_r_squared=indep_r2s,
                message=f"Invalid surface params: {e}",
            )

        # Step 4: Compare surface R² vs independent
        surface_r2s = self._compute_slice_r_squared(surface_params, slices)
        surface_n_points = [len(s.log_moneyness) for s in slices]
        surface_total = sum(surface_n_points)
        surface_agg_r2 = sum(
            r2 * n for r2, n in zip(surface_r2s, surface_n_points)
        ) / surface_total if surface_total > 0 else 0.0

        # Fall back if surface fit is worse
        if self.fallback_to_independent and surface_agg_r2 < indep_agg_r2 - 0.02:
            logger.info(
                f"SSVI surface R²={surface_agg_r2:.4f} < independent R²={indep_agg_r2:.4f} "
                f"(threshold -0.02), falling back to independent fits"
            )
            return SSVISurfaceFitResult(
                params=None,
                per_slice_params=[p for p in per_slice_params_indep if p is not None],
                success=False,
                aggregate_r_squared=indep_agg_r2,
                per_slice_r_squared=indep_r2s,
                message=(
                    f"Surface R²={surface_agg_r2:.4f} worse than independent "
                    f"R²={indep_agg_r2:.4f}, fell back"
                ),
            )

        decomposed = surface_params.to_per_slice_params()

        logger.info(
            f"SSVI surface fit: ρ={rho_opt:.3f}, η={eta_opt:.3f}, λ={lam_opt:.3f}, "
            f"R²={surface_agg_r2:.4f} (independent: {indep_agg_r2:.4f})"
        )
        for i, sl in enumerate(slices):
            logger.info(
                f"  {sl.expiry_name}: θ={thetas_opt[i]:.6f}, "
                f"φ={surface_params.phi(thetas_opt[i]):.3f}, "
                f"R²={surface_r2s[i]:.4f} (indep: {indep_r2s[i]:.4f})"
            )

        return SSVISurfaceFitResult(
            params=surface_params,
            per_slice_params=decomposed,
            success=True,
            aggregate_r_squared=surface_agg_r2,
            per_slice_r_squared=surface_r2s,
            message="Surface fit successful",
        )

    def _construct_initial_guess(
        self,
        slices: List[SSVISliceData],
        per_slice_params: List[Optional[SSVIParams]],
    ) -> Tuple[float, float, float, np.ndarray]:
        """Construct initial guess for joint optimization from per-slice fits.

        Returns:
            (rho_init, eta_init, lam_init, delta_init)
        """
        n = len(slices)

        # Collect successful per-slice (theta, rho, phi) values
        thetas_raw = []
        rhos = []
        phis = []
        for i, p in enumerate(per_slice_params):
            if p is not None:
                thetas_raw.append(p.theta)
                rhos.append(p.rho)
                phis.append(p.phi)
            else:
                # Fill from neighbors or use a default
                thetas_raw.append(None)
                rhos.append(None)
                phis.append(None)

        # ρ₀ = weighted median of per-slice ρ values
        valid_rhos = [r for r in rhos if r is not None]
        rho_init = float(np.median(valid_rhos)) if valid_rhos else -0.3

        # Fill missing thetas with interpolation
        valid_idxs = [i for i, t in enumerate(thetas_raw) if t is not None]
        if len(valid_idxs) < n:
            valid_thetas = [thetas_raw[i] for i in valid_idxs]
            valid_ttms = [slices[i].ttm for i in valid_idxs]
            for i in range(n):
                if thetas_raw[i] is None:
                    # Simple linear interp in TTM space
                    thetas_raw[i] = float(np.interp(
                        slices[i].ttm, valid_ttms, valid_thetas
                    ))

        # Enforce monotonicity
        thetas_mono = list(thetas_raw)
        for i in range(1, n):
            if thetas_mono[i] <= thetas_mono[i - 1]:
                thetas_mono[i] = thetas_mono[i - 1] + 1e-6

        # Compute δᵢ from monotone θ sequence: θ₁ = δ₁², θᵢ = θᵢ₋₁ + δᵢ²
        deltas = np.zeros(n)
        deltas[0] = math.sqrt(thetas_mono[0])
        for i in range(1, n):
            deltas[i] = math.sqrt(thetas_mono[i] - thetas_mono[i - 1])

        # Fit η₀, λ₀ from per-slice (θ, φ) pairs via log-linear regression
        # log(φ) = log(η) - λ·log(θ)
        valid_theta_phi = [
            (thetas_raw[i], phis[i])
            for i in range(n)
            if thetas_raw[i] is not None and phis[i] is not None
                and thetas_raw[i] > 0 and phis[i] > 0
        ]

        if len(valid_theta_phi) >= 2:
            log_thetas = np.array([math.log(tp[0]) for tp in valid_theta_phi])
            log_phis = np.array([math.log(tp[1]) for tp in valid_theta_phi])
            # log(φ) = log(η) - λ·log(θ)
            # Fit: log_phis = a + b * log_thetas, where a = log(η), b = -λ
            coeffs = np.polyfit(log_thetas, log_phis, 1)
            lam_init = float(np.clip(-coeffs[0], 0.0, 0.5))
            eta_init = float(np.clip(math.exp(coeffs[1]), self.eta_bounds[0], self.eta_bounds[1]))
        else:
            eta_init = 1.0
            lam_init = 0.25

        return rho_init, eta_init, lam_init, deltas
