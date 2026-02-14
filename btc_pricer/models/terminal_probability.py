"""Terminal Probability Calculator.

Calculates the probability that BTC ends above or below a price level at a future time.
Uses Monte Carlo simulation with the calibrated Heston or SSVI model.

The terminal probability answers: P(S_T >= ref_price) or P(S_T <= ref_price) at time T.
This is the correct metric for Polymarket daily BTC contracts which pay based
on the expiry price only.

Known Limitations:
-----------------
1. SSVI Local Vol Approximation: The SSVI path simulation uses a "sticky-moneyness"
   heuristic where volatility at each step is the SSVI implied vol at the current
   log(S/F). This captures the key smile dynamics (spot down -> vol up) but is not
   a true Dupire local volatility model and is not dynamically arbitrage-consistent.
   This is a standard industry approximation with acceptable accuracy for practical use.
"""

import logging
import math
from dataclasses import dataclass
from typing import Tuple, Optional

import numpy as np
from scipy.stats import norm

from .heston import HestonParams
from .ssvi import SSVIParams, SSVIModel

logger = logging.getLogger(__name__)

# Optional Numba acceleration for MC simulation
NUMBA_AVAILABLE = False
try:
    from numba import njit, prange, config as numba_config
    import numba
    NUMBA_AVAILABLE = True
    # Limit parallel Numba kernels to 4 cores max
    numba.set_num_threads(min(numba.config.NUMBA_NUM_THREADS, 4))
    logger.debug("Numba parallel: %d threads", numba.get_num_threads())
except ImportError:
    njit = None
    prange = range


if NUMBA_AVAILABLE:
    @njit(parallel=True, cache=True)
    def _heston_mc_numba(
        spot, v0, kappa, theta, xi, rho,
        dt, sqrt_dt, n_paths, n_steps, base_seed
    ):
        """Parallel Numba Heston MC kernel with per-path RNG.

        Each path seeds its own RNG via base_seed + j, eliminating the
        need to pre-allocate giant random arrays.  The outer loop uses
        prange for automatic thread parallelism.

        Args:
            spot: Initial spot price.
            v0: Initial variance.
            kappa: Mean reversion speed.
            theta: Long-term variance.
            xi: Vol-of-vol.
            rho: Correlation.
            dt: Time step.
            sqrt_dt: sqrt(dt).
            n_paths: Number of paths.
            n_steps: Number of steps.
            base_seed: Base random seed; path j uses base_seed + j.

        Returns:
            Array of final prices, shape (n_paths,).
        """
        rho_comp = math.sqrt(1.0 - rho * rho)
        final_prices = np.empty(n_paths)

        for j in prange(n_paths):
            np.random.seed(base_seed + j)
            price = spot
            v = v0
            for i in range(n_steps):
                z1 = np.random.standard_normal()
                z2_raw = np.random.standard_normal()
                z2 = rho * z1 + rho_comp * z2_raw

                v_pos = v if v > 0.0 else 0.0
                v_sqrt = math.sqrt(v_pos)

                # CIR variance update
                v = v + kappa * (theta - v) * dt + xi * v_sqrt * sqrt_dt * z2
                v = v if v > 0.0 else 0.0

                # Log-Euler price update
                price = price * math.exp(
                    -0.5 * v * dt + v_sqrt * sqrt_dt * z1
                )

            final_prices[j] = price

        return final_prices


@dataclass
class MCResult:
    """Result of Monte Carlo terminal probability calculation."""
    reference_price: float      # Target price level
    spot: float                 # Current price
    ttm: float                  # Time to maturity in years
    terminal_probability: float # P(ends beyond reference_price)
    confidence_interval: Tuple[float, float]  # 95% CI
    n_simulations: int
    direction: str              # "down" or "up"
    method: str                 # "heston" or "ssvi_local_vol"

    @property
    def reference_price_distance_pct(self) -> float:
        """Distance to reference price as percentage of spot."""
        return (self.reference_price - self.spot) / self.spot * 100

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "reference_price": self.reference_price,
            "spot": self.spot,
            "ttm": self.ttm,
            "terminal_probability": self.terminal_probability,
            "confidence_interval": list(self.confidence_interval),
            "n_simulations": self.n_simulations,
            "direction": self.direction,
            "method": self.method,
            "reference_price_distance_pct": self.reference_price_distance_pct,
        }


class TerminalProbabilityCalculator:
    """Calculate terminal probabilities via Monte Carlo simulation.

    Uses Monte Carlo simulation with the Heston stochastic volatility model
    or SSVI local vol to estimate the probability that the price ends above
    or below a reference price level at a future time.
    """

    def __init__(
        self,
        n_simulations: int = 200000,
        n_steps_per_day: int = 1440,  # 1-minute steps
        confidence_level: float = 0.95,
        use_antithetic: bool = True,
        seed: Optional[int] = None
    ):
        """Initialize the calculator.

        Args:
            n_simulations: Number of Monte Carlo paths.
            n_steps_per_day: Time steps per day (288 = 5-min intervals).
            confidence_level: Confidence level for CI (default 0.95).
            use_antithetic: Use antithetic variates for variance reduction.
            seed: Random seed for reproducibility.
        """
        self.n_simulations = n_simulations
        self.n_steps_per_day = n_steps_per_day
        self.confidence_level = confidence_level
        self.use_antithetic = use_antithetic
        self.seed = seed

        if seed is not None:
            np.random.seed(seed)

    def simulate_heston_paths(
        self,
        heston_params: HestonParams,
        spot: float,
        n_paths: int,
        n_steps: int
    ) -> np.ndarray:
        """Simulate Heston price paths using Euler-Maruyama discretization.

        The Heston SDE:
            dS_t = S_t * sqrt(v_t) * dW_1
            dv_t = kappa * (theta - v_t) * dt + xi * sqrt(v_t) * dW_2
            corr(dW_1, dW_2) = rho

        Args:
            heston_params: Calibrated Heston parameters.
            spot: Current spot price.
            n_paths: Number of paths to simulate.
            n_steps: Number of time steps.

        Returns:
            Array of shape (n_paths, n_steps+1) with price paths.
        """
        dt = heston_params.ttm / n_steps
        sqrt_dt = np.sqrt(dt)

        # Extract parameters
        v0 = heston_params.v0
        kappa = heston_params.kappa
        theta = heston_params.theta
        xi = heston_params.xi
        rho = heston_params.rho

        # Initialize arrays
        paths = np.zeros((n_paths, n_steps + 1))
        paths[:, 0] = spot

        # Initialize variance
        V = np.full(n_paths, v0)

        # Correlation factor
        rho_comp = np.sqrt(1 - rho**2)

        for i in range(n_steps):
            # Generate correlated Brownian increments
            Z1 = np.random.standard_normal(n_paths)
            Z2 = rho * Z1 + rho_comp * np.random.standard_normal(n_paths)

            # Ensure variance is non-negative (reflection scheme)
            V_sqrt = np.sqrt(np.maximum(V, 0))

            # Update variance (CIR process)
            V_new = V + kappa * (theta - V) * dt + xi * V_sqrt * sqrt_dt * Z2
            V = np.maximum(V_new, 0)  # Reflection to keep positive

            # Update price (log-Euler for numerical stability)
            # dlog(S) = -0.5*V*dt + sqrt(V)*dW
            paths[:, i + 1] = paths[:, i] * np.exp(
                -0.5 * np.maximum(V, 0) * dt + V_sqrt * sqrt_dt * Z1
            )

        return paths

    def simulate_heston_paths_compact(
        self,
        heston_params: HestonParams,
        spot: float,
        n_paths: int,
        n_steps: int
    ) -> np.ndarray:
        """Simulate Heston paths, returning only final prices.

        Uses the same Euler-Maruyama discretization as simulate_heston_paths()
        but stores only a single vector of length n_paths instead of the full
        (n_paths, n_steps+1) matrix.

        When Numba is available, delegates to a JIT-compiled kernel for
        significant speedup.

        Args:
            heston_params: Calibrated Heston parameters.
            spot: Current spot price.
            n_paths: Number of paths to simulate.
            n_steps: Number of time steps.

        Returns:
            Array of final prices, shape (n_paths,).
        """
        dt = heston_params.ttm / n_steps
        sqrt_dt = np.sqrt(dt)

        # Extract parameters
        v0 = heston_params.v0
        kappa = heston_params.kappa
        theta = heston_params.theta
        xi = heston_params.xi
        rho = heston_params.rho

        # Fast path: parallel Numba JIT kernel (no pre-allocation needed)
        if NUMBA_AVAILABLE:
            base_seed = np.random.randint(0, 2**31)
            return _heston_mc_numba(
                spot, v0, kappa, theta, xi, rho,
                dt, sqrt_dt, n_paths, n_steps,
                base_seed,
            )

        # Fallback: vectorized NumPy — generate randoms per-step
        rho_comp = np.sqrt(1 - rho**2)
        prices = np.full(n_paths, spot)
        V = np.full(n_paths, v0)

        for i in range(n_steps):
            Z1 = np.random.standard_normal(n_paths)
            Z2 = rho * Z1 + rho_comp * np.random.standard_normal(n_paths)

            V_sqrt = np.sqrt(np.maximum(V, 0))

            V_new = V + kappa * (theta - V) * dt + xi * V_sqrt * sqrt_dt * Z2
            V = np.maximum(V_new, 0)

            prices = prices * np.exp(
                -0.5 * np.maximum(V, 0) * dt + V_sqrt * sqrt_dt * Z1
            )

        return prices

    def _calculate_terminal_probability(
        self,
        final_prices: np.ndarray,
        reference_price: float,
        direction: str
    ) -> Tuple[float, float]:
        """Calculate terminal probability from final prices.

        Args:
            final_prices: Final prices, shape (n_paths,).
            reference_price: Reference price level.
            direction: "down" or "up".

        Returns:
            Tuple of (terminal_prob, se).
        """
        n_paths = len(final_prices)

        if direction == "down":
            terminal = final_prices <= reference_price
        else:
            terminal = final_prices >= reference_price

        terminal_prob = np.mean(terminal)
        se = np.sqrt(terminal_prob * (1 - terminal_prob) / n_paths)

        return terminal_prob, se

    def _validate_reference_price(self, spot: float, reference_price: float, direction: str):
        """Log a warning when reference price is on the wrong side of spot."""
        if direction == "down" and reference_price >= spot:
            logger.warning(
                f"Down reference price {reference_price} >= spot {spot}, "
                "probability will be high"
            )
        elif direction == "up" and reference_price <= spot:
            logger.warning(
                f"Up reference price {reference_price} <= spot {spot}, "
                "probability will be high"
            )

    def _simulate_final_prices(self, simulate_fn, ttm: float) -> np.ndarray:
        """Run MC with antithetic handling, return final_prices array.

        Args:
            simulate_fn: Callable(n_paths, n_steps) -> np.ndarray of final prices.
            ttm: Time to maturity in years.

        Returns:
            Array of final prices.
        """
        days = ttm * 365
        n_steps = max(int(days * self.n_steps_per_day), 10)
        n_paths = self.n_simulations // 2 if self.use_antithetic else self.n_simulations

        final1 = simulate_fn(n_paths, n_steps)

        if self.use_antithetic:
            final2 = simulate_fn(n_paths, n_steps)
            return np.concatenate([final1, final2])
        return final1

    def _build_result(
        self,
        final_prices: np.ndarray,
        reference_price: float,
        direction: str,
        spot: float,
        ttm: float,
        method: str
    ) -> MCResult:
        """Calculate terminal prob + CI and return MCResult."""
        terminal_prob, se = self._calculate_terminal_probability(
            final_prices, reference_price, direction
        )
        z = norm.ppf(1 - (1 - self.confidence_level) / 2)
        return MCResult(
            reference_price=reference_price,
            spot=spot,
            ttm=ttm,
            terminal_probability=terminal_prob,
            confidence_interval=(max(0, terminal_prob - z * se), min(1, terminal_prob + z * se)),
            n_simulations=len(final_prices),
            direction=direction,
            method=method,
        )

    def terminal_probability_heston(
        self,
        heston_params: HestonParams,
        spot: float,
        reference_price: float,
        direction: str = "down"
    ) -> MCResult:
        """Calculate terminal probability using Heston MC.

        Args:
            heston_params: Calibrated Heston parameters.
            spot: Current spot price.
            reference_price: Target price level.
            direction: "down" (reference_price < spot) or "up" (reference_price > spot).

        Returns:
            MCResult with terminal probability and confidence interval.
        """
        self._validate_reference_price(spot, reference_price, direction)

        logger.info(
            f"Simulating {self.n_simulations} paths "
            f"({self.n_steps_per_day}/day)..."
        )

        def simulate_fn(n_paths, n_steps):
            return self.simulate_heston_paths_compact(heston_params, spot, n_paths, n_steps)

        final_prices = self._simulate_final_prices(simulate_fn, heston_params.ttm)
        return self._build_result(final_prices, reference_price, direction, spot, heston_params.ttm, "heston")

    def terminal_probability_heston_multiple(
        self,
        heston_params: HestonParams,
        spot: float,
        reference_prices: list,
        direction: str = "down"
    ) -> list:
        """Calculate terminal probabilities for multiple reference prices efficiently.

        Simulates paths once and evaluates all reference prices.

        Args:
            heston_params: Calibrated Heston parameters.
            spot: Current spot price.
            reference_prices: List of reference price levels.
            direction: "down" or "up".

        Returns:
            List of MCResult objects.
        """
        logger.info(
            f"Simulating {self.n_simulations} paths for {len(reference_prices)} reference prices..."
        )

        def simulate_fn(n_paths, n_steps):
            return self.simulate_heston_paths_compact(heston_params, spot, n_paths, n_steps)

        final_prices = self._simulate_final_prices(simulate_fn, heston_params.ttm)
        return [
            self._build_result(final_prices, rp, direction, spot, heston_params.ttm, "heston")
            for rp in reference_prices
        ]

    def simulate_local_vol_paths_compact(
        self,
        ssvi_params: SSVIParams,
        spot: float,
        forward: float,
        n_paths: int,
        n_steps: int
    ) -> np.ndarray:
        """Simulate SSVI local-vol paths, returning only final prices.

        Uses a "sticky-moneyness" approach where volatility at each step
        depends on current spot's log-moneyness relative to forward.

        Args:
            ssvi_params: SSVI parameters for the vol surface.
            spot: Current spot price.
            forward: Forward price (for log-moneyness calculation).
            n_paths: Number of paths to simulate.
            n_steps: Number of time steps.

        Returns:
            Array of final prices, shape (n_paths,).
        """
        dt = ssvi_params.ttm / n_steps
        sqrt_dt = np.sqrt(dt)

        ssvi_model = SSVIModel(ssvi_params)

        prices = np.full(n_paths, spot)

        for i in range(n_steps):
            log_moneyness = np.log(prices / forward)
            vol = ssvi_model.implied_volatility_array(log_moneyness)
            Z = np.random.standard_normal(n_paths)
            prices = prices * np.exp(
                -0.5 * vol**2 * dt + vol * sqrt_dt * Z
            )

        return prices

    def terminal_probability_ssvi(
        self,
        ssvi_params: SSVIParams,
        spot: float,
        forward: float,
        reference_price: float,
        direction: str = "down"
    ) -> MCResult:
        """Calculate terminal probability using SSVI Local Vol MC.

        Unlike constant vol, this captures the vol smile:
        - When spot drops, IV increases (negative skew)
        - When spot rises, IV decreases
        This gives more realistic terminal probabilities.

        Args:
            ssvi_params: Calibrated SSVI parameters.
            spot: Current spot price.
            forward: Forward price (for moneyness calculation).
            reference_price: Target price level.
            direction: "down" (reference_price < spot) or "up" (reference_price > spot).

        Returns:
            MCResult with terminal probability and confidence interval.
        """
        self._validate_reference_price(spot, reference_price, direction)

        logger.info(
            f"Simulating {self.n_simulations} Local Vol paths "
            f"({self.n_steps_per_day}/day)..."
        )

        def simulate_fn(n_paths, n_steps):
            return self.simulate_local_vol_paths_compact(ssvi_params, spot, forward, n_paths, n_steps)

        final_prices = self._simulate_final_prices(simulate_fn, ssvi_params.ttm)
        return self._build_result(final_prices, reference_price, direction, spot, ssvi_params.ttm, "ssvi_local_vol")

    def terminal_probability_ssvi_multiple(
        self,
        ssvi_params: SSVIParams,
        spot: float,
        forward: float,
        reference_prices: list,
        direction: str = "down"
    ) -> list:
        """Calculate terminal probabilities for multiple reference prices using SSVI Local Vol.

        Simulates paths once and evaluates all reference prices.

        Args:
            ssvi_params: Calibrated SSVI parameters.
            spot: Current spot price.
            forward: Forward price.
            reference_prices: List of reference price levels.
            direction: "down" or "up".

        Returns:
            List of MCResult objects.
        """
        logger.info(
            f"Simulating {self.n_simulations} Local Vol paths for {len(reference_prices)} reference prices..."
        )

        def simulate_fn(n_paths, n_steps):
            return self.simulate_local_vol_paths_compact(ssvi_params, spot, forward, n_paths, n_steps)

        final_prices = self._simulate_final_prices(simulate_fn, ssvi_params.ttm)
        return [
            self._build_result(final_prices, rp, direction, spot, ssvi_params.ttm, "ssvi_local_vol")
            for rp in reference_prices
        ]
