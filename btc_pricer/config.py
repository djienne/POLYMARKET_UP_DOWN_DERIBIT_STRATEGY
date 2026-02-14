"""Configuration management for BTC Price Forecaster."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple, Optional
import yaml


@dataclass
class APIConfig:
    """Deribit API configuration."""
    base_url: str = "https://www.deribit.com/api/v2/public/"
    timeout: int = 10
    max_retries: int = 3


@dataclass
class FilterConfig:
    """Data quality filter configuration."""
    min_open_interest: int = 0
    max_bid_ask_spread_pct: float = 0.20
    min_days_to_expiry: int = 0
    moneyness_range: Tuple[float, float] = (0.5, 1.5)
    min_surface_points: int = 5  # Minimum data points for surface fitting


@dataclass
class SSVIConfig:
    """SSVI model fitting configuration."""
    rho_bounds: Tuple[float, float] = (-0.99, 0.99)
    phi_bounds: Tuple[float, float] = (0.001, 5.0)
    theta_bounds: Tuple[float, float] = (0.001, 10.0)
    optimizer: str = "L-BFGS-B"
    # TTM-adaptive settings for short-dated options
    short_dated_ttm_threshold: float = 0.10  # ~36 days
    short_dated_phi_bounds: Tuple[float, float] = (0.001, 20.0)
    # Multi-start optimization
    use_multi_start: bool = True
    n_starts: int = 5
    # Global optimizer fallback
    use_global_optimizer: bool = True
    global_optimizer: str = "differential_evolution"
    # Objective function settings
    use_relative_error: bool = True
    regularization_lambda: float = 0.001


@dataclass
class HestonConfig:
    """Heston model fitting configuration."""
    v0_bounds: Tuple[float, float] = (0.01, 4.0)
    kappa_bounds: Tuple[float, float] = (0.1, 10.0)
    theta_bounds: Tuple[float, float] = (0.01, 4.0)
    xi_bounds: Tuple[float, float] = (0.1, 5.0)
    rho_bounds: Tuple[float, float] = (-0.99, 0.99)
    optimizer: str = "differential_evolution"
    n_integration_points: int = 512
    calibration_integration_points: int = 1024  # Higher precision for calibration
    use_quantlib: bool = True  # Use QuantLib for better numerical precision

    # Short-dated TTM thresholds and bounds
    short_dated_ttm_threshold: float = 0.10  # ~36 days
    short_dated_xi_bounds: Tuple[float, float] = (0.1, 10.0)
    short_dated_kappa_bounds: Tuple[float, float] = (0.01, 15.0)

    very_short_dated_ttm_threshold: float = 0.02  # ~7 days
    very_short_dated_xi_bounds: Tuple[float, float] = (0.1, 15.0)
    very_short_dated_kappa_bounds: Tuple[float, float] = (0.001, 20.0)

    ultra_short_dated_ttm_threshold: float = 0.01  # ~3.5 days
    ultra_short_dated_xi_bounds: Tuple[float, float] = (0.1, 20.0)
    ultra_short_dated_kappa_bounds: Tuple[float, float] = (0.01, 10.0)
    ultra_short_dated_theta_factor: Tuple[float, float] = (0.3, 2.0)

    # Gaussian near-ATM weighting for short TTM
    short_ttm_gaussian_weighting: bool = True
    short_ttm_gaussian_sigma_base: float = 0.05
    short_ttm_gaussian_sigma_ttm_scale: float = 2.0
    short_ttm_gaussian_floor: float = 0.1

    # Multi-start optimization
    use_multi_start: bool = True
    n_starts: int = 5
    max_workers: int = 4  # Max parallel workers for multi-start
    # QuantLib objective implementation strategy
    quantlib_objective_impl: str = "optimized"  # "optimized" or "legacy"
    # Optional Numba acceleration (fallback objective path)
    enable_numba_fallback: bool = True
    numba_strict_mode: bool = True
    # Use relative IV error ((model-market)/market)² in objective function
    use_relative_error: bool = True
    # Early termination: skip remaining multi-starts when SSE below threshold
    # None = disabled (all starts run), 1e-5 = conservative (R² > 0.999)
    early_termination_sse: Optional[float] = None


@dataclass
class ModelConfig:
    """Model selection configuration."""
    default_model: str = "ssvi"        # "heston" or "ssvi"
    fallback_to_ssvi: bool = True      # Fall back to SSVI if Heston fails
    min_calibration_ttm_days: float = 1.0  # Skip expiries with TTM below this (days) for calibration
    iv_consistency_threshold: float = 0.10  # 10% max IV error
    ssvi_preference_threshold: float = 0.06  # SSVI must be 6% better in R² to be selected
    iv_consistency_relaxation: float = 0.15  # Extra tolerance added at TTM=0
    iv_consistency_ttm_cutoff: float = 0.05  # TTM below which relaxation kicks in (~18 days)


@dataclass
class BreedenLitzenbergerConfig:
    """Breeden-Litzenberger RND extraction configuration."""
    strike_grid_points: int = 500
    strike_range_std: float = 3.0
    use_log_strikes: bool = False  # Use log-moneyness spacing for better tails


@dataclass
class IntradayConfig:
    """Intraday forecasting configuration."""
    use_drift: bool = False
    annual_drift: float = 0.0
    standard_horizons: List[int] = field(default_factory=lambda: [1, 2, 4, 6, 8, 12, 24, 48, 72])


@dataclass
class IVSolverConfig:
    """Implied volatility solver configuration."""
    tolerance: float = 1e-8
    min_iv: float = 0.01
    max_iv: float = 5.0
    max_iterations: int = 100


@dataclass
class ValidationConfig:
    """Validation thresholds configuration."""
    spot_price_min: float = 10_000
    spot_price_max: float = 1_000_000
    iv_valid_range: Tuple[float, float] = (0.05, 5.0)
    iv_normal_range: Tuple[float, float] = (0.10, 3.0)
    iv_typical_range: Tuple[float, float] = (0.20, 2.0)
    r_squared_excellent: float = 0.90
    r_squared_acceptable: float = 0.80
    r_squared_poor: float = 0.50
    integral_tolerance: float = 0.05
    mean_forward_tolerance: float = 0.10


@dataclass
class TerminalConfig:
    """Terminal probability calculation configuration."""
    n_simulations: int = 1000000       # Number of Monte Carlo paths
    n_steps_per_day: int = 1440        # 1-minute steps
    confidence_level: float = 0.95
    use_antithetic: bool = True        # Variance reduction


@dataclass
class SSVISurfaceConfig:
    """SSVI surface joint fitting configuration (Gatheral & Jacquier 2014)."""
    enabled: bool = True                            # Always on by default
    max_ttm_days: float = 4.0                      # Only include expiries within this TTM window
    min_expiries: int = 2                           # Need at least 2 slices
    eta_bounds: Tuple[float, float] = (0.01, 10.0)
    lam_bounds: Tuple[float, float] = (0.0, 0.5)   # Gatheral constraint: λ ∈ [0, 0.5]
    rho_bounds: Tuple[float, float] = (-0.99, 0.99)
    maxiter: int = 300
    workers: int = 4
    use_relative_error: bool = True
    fallback_to_independent: bool = True           # Fall back if joint fit is worse


@dataclass
class OutputConfig:
    """Output configuration."""
    save_json: bool = True
    save_plots: bool = True
    plot_format: str = "png"
    dpi: int = 150


@dataclass
class Config:
    """Main configuration container."""
    api: APIConfig = field(default_factory=APIConfig)
    filters: FilterConfig = field(default_factory=FilterConfig)
    ssvi: SSVIConfig = field(default_factory=SSVIConfig)
    heston: HestonConfig = field(default_factory=HestonConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    breeden_litzenberger: BreedenLitzenbergerConfig = field(
        default_factory=BreedenLitzenbergerConfig
    )
    intraday: IntradayConfig = field(default_factory=IntradayConfig)
    iv_solver: IVSolverConfig = field(default_factory=IVSolverConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    terminal: TerminalConfig = field(default_factory=TerminalConfig)
    ssvi_surface: SSVISurfaceConfig = field(default_factory=SSVISurfaceConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    @classmethod
    def from_yaml(cls, path: Path) -> "Config":
        """Load configuration from YAML file."""
        with open(path, "r") as f:
            data = yaml.safe_load(f)

        config = cls()

        if "api" in data:
            config.api = APIConfig(**data["api"])

        if "filters" in data:
            filters_data = data["filters"].copy()
            if "moneyness_range" in filters_data:
                filters_data["moneyness_range"] = tuple(filters_data["moneyness_range"])
            config.filters = FilterConfig(**filters_data)

        if "ssvi" in data:
            ssvi_data = data["ssvi"]
            for key in ["rho_bounds", "phi_bounds", "theta_bounds", "short_dated_phi_bounds"]:
                if key in ssvi_data:
                    ssvi_data[key] = tuple(ssvi_data[key])
            config.ssvi = SSVIConfig(**ssvi_data)

        if "heston" in data:
            heston_data = data["heston"]
            for key in ["v0_bounds", "kappa_bounds", "theta_bounds", "xi_bounds", "rho_bounds",
                        "short_dated_xi_bounds", "short_dated_kappa_bounds",
                        "very_short_dated_xi_bounds", "very_short_dated_kappa_bounds",
                        "ultra_short_dated_xi_bounds", "ultra_short_dated_kappa_bounds",
                        "ultra_short_dated_theta_factor"]:
                if key in heston_data:
                    heston_data[key] = tuple(heston_data[key])
            if "quantlib_objective_impl" in heston_data:
                impl = str(heston_data["quantlib_objective_impl"]).lower().strip()
                if impl not in {"optimized", "legacy"}:
                    raise ValueError(
                        "heston.quantlib_objective_impl must be 'optimized' or 'legacy'"
                    )
                heston_data["quantlib_objective_impl"] = impl
            config.heston = HestonConfig(**heston_data)

        if "model" in data:
            config.model = ModelConfig(**data["model"])

        if "breeden_litzenberger" in data:
            config.breeden_litzenberger = BreedenLitzenbergerConfig(
                **data["breeden_litzenberger"]
            )

        if "intraday" in data:
            config.intraday = IntradayConfig(**data["intraday"])

        if "iv_solver" in data:
            config.iv_solver = IVSolverConfig(**data["iv_solver"])

        if "validation" in data:
            validation_data = data["validation"].copy()
            for key in ["iv_valid_range", "iv_normal_range", "iv_typical_range"]:
                if key in validation_data:
                    validation_data[key] = tuple(validation_data[key])
            config.validation = ValidationConfig(**validation_data)

        if "terminal" in data:
            config.terminal = TerminalConfig(**data["terminal"])
        elif "barrier" in data:
            config.terminal = TerminalConfig(**data["barrier"])

        if "ssvi_surface" in data:
            surface_data = data["ssvi_surface"].copy()
            for key in ["eta_bounds", "lam_bounds", "rho_bounds"]:
                if key in surface_data:
                    surface_data[key] = tuple(surface_data[key])
            config.ssvi_surface = SSVISurfaceConfig(**surface_data)

        if "output" in data:
            config.output = OutputConfig(**data["output"])

        return config

    def to_dict(self) -> dict:
        """Convert configuration to dictionary."""
        return {
            "api": {
                "base_url": self.api.base_url,
                "timeout": self.api.timeout,
                "max_retries": self.api.max_retries,
            },
            "filters": {
                "min_open_interest": self.filters.min_open_interest,
                "max_bid_ask_spread_pct": self.filters.max_bid_ask_spread_pct,
                "min_days_to_expiry": self.filters.min_days_to_expiry,
                "moneyness_range": list(self.filters.moneyness_range),
                "min_surface_points": self.filters.min_surface_points,
            },
            "ssvi": {
                "rho_bounds": list(self.ssvi.rho_bounds),
                "phi_bounds": list(self.ssvi.phi_bounds),
                "theta_bounds": list(self.ssvi.theta_bounds),
                "optimizer": self.ssvi.optimizer,
                "short_dated_ttm_threshold": self.ssvi.short_dated_ttm_threshold,
                "short_dated_phi_bounds": list(self.ssvi.short_dated_phi_bounds),
                "use_multi_start": self.ssvi.use_multi_start,
                "n_starts": self.ssvi.n_starts,
                "use_global_optimizer": self.ssvi.use_global_optimizer,
                "global_optimizer": self.ssvi.global_optimizer,
                "use_relative_error": self.ssvi.use_relative_error,
                "regularization_lambda": self.ssvi.regularization_lambda,
            },
            "heston": {
                "v0_bounds": list(self.heston.v0_bounds),
                "kappa_bounds": list(self.heston.kappa_bounds),
                "theta_bounds": list(self.heston.theta_bounds),
                "xi_bounds": list(self.heston.xi_bounds),
                "rho_bounds": list(self.heston.rho_bounds),
                "optimizer": self.heston.optimizer,
                "n_integration_points": self.heston.n_integration_points,
                "calibration_integration_points": self.heston.calibration_integration_points,
                "use_quantlib": self.heston.use_quantlib,
                "short_dated_ttm_threshold": self.heston.short_dated_ttm_threshold,
                "short_dated_xi_bounds": list(self.heston.short_dated_xi_bounds),
                "short_dated_kappa_bounds": list(self.heston.short_dated_kappa_bounds),
                "very_short_dated_ttm_threshold": self.heston.very_short_dated_ttm_threshold,
                "very_short_dated_xi_bounds": list(self.heston.very_short_dated_xi_bounds),
                "very_short_dated_kappa_bounds": list(self.heston.very_short_dated_kappa_bounds),
                "ultra_short_dated_ttm_threshold": self.heston.ultra_short_dated_ttm_threshold,
                "ultra_short_dated_xi_bounds": list(self.heston.ultra_short_dated_xi_bounds),
                "ultra_short_dated_kappa_bounds": list(self.heston.ultra_short_dated_kappa_bounds),
                "ultra_short_dated_theta_factor": list(self.heston.ultra_short_dated_theta_factor),
                "short_ttm_gaussian_weighting": self.heston.short_ttm_gaussian_weighting,
                "short_ttm_gaussian_sigma_base": self.heston.short_ttm_gaussian_sigma_base,
                "short_ttm_gaussian_sigma_ttm_scale": self.heston.short_ttm_gaussian_sigma_ttm_scale,
                "short_ttm_gaussian_floor": self.heston.short_ttm_gaussian_floor,
                "use_multi_start": self.heston.use_multi_start,
                "n_starts": self.heston.n_starts,
                "quantlib_objective_impl": self.heston.quantlib_objective_impl,
                "enable_numba_fallback": self.heston.enable_numba_fallback,
                "numba_strict_mode": self.heston.numba_strict_mode,
                "use_relative_error": self.heston.use_relative_error,
            },
            "model": {
                "default_model": self.model.default_model,
                "fallback_to_ssvi": self.model.fallback_to_ssvi,
                "min_calibration_ttm_days": self.model.min_calibration_ttm_days,
                "iv_consistency_threshold": self.model.iv_consistency_threshold,
                "ssvi_preference_threshold": self.model.ssvi_preference_threshold,
                "iv_consistency_relaxation": self.model.iv_consistency_relaxation,
                "iv_consistency_ttm_cutoff": self.model.iv_consistency_ttm_cutoff,
            },
            "breeden_litzenberger": {
                "strike_grid_points": self.breeden_litzenberger.strike_grid_points,
                "strike_range_std": self.breeden_litzenberger.strike_range_std,
                "use_log_strikes": self.breeden_litzenberger.use_log_strikes,
            },
            "intraday": {
                "use_drift": self.intraday.use_drift,
                "annual_drift": self.intraday.annual_drift,
                "standard_horizons": self.intraday.standard_horizons,
            },
            "iv_solver": {
                "tolerance": self.iv_solver.tolerance,
                "min_iv": self.iv_solver.min_iv,
                "max_iv": self.iv_solver.max_iv,
                "max_iterations": self.iv_solver.max_iterations,
            },
            "validation": {
                "spot_price_min": self.validation.spot_price_min,
                "spot_price_max": self.validation.spot_price_max,
                "iv_valid_range": list(self.validation.iv_valid_range),
                "iv_normal_range": list(self.validation.iv_normal_range),
                "iv_typical_range": list(self.validation.iv_typical_range),
                "r_squared_excellent": self.validation.r_squared_excellent,
                "r_squared_acceptable": self.validation.r_squared_acceptable,
                "r_squared_poor": self.validation.r_squared_poor,
                "integral_tolerance": self.validation.integral_tolerance,
                "mean_forward_tolerance": self.validation.mean_forward_tolerance,
            },
            "terminal": {
                "n_simulations": self.terminal.n_simulations,
                "n_steps_per_day": self.terminal.n_steps_per_day,
                "confidence_level": self.terminal.confidence_level,
                "use_antithetic": self.terminal.use_antithetic,
            },
            "ssvi_surface": {
                "enabled": self.ssvi_surface.enabled,
                "max_ttm_days": self.ssvi_surface.max_ttm_days,
                "min_expiries": self.ssvi_surface.min_expiries,
                "eta_bounds": list(self.ssvi_surface.eta_bounds),
                "lam_bounds": list(self.ssvi_surface.lam_bounds),
                "rho_bounds": list(self.ssvi_surface.rho_bounds),
                "maxiter": self.ssvi_surface.maxiter,
                "workers": self.ssvi_surface.workers,
                "use_relative_error": self.ssvi_surface.use_relative_error,
                "fallback_to_independent": self.ssvi_surface.fallback_to_independent,
            },
            "output": {
                "save_json": self.output.save_json,
                "save_plots": self.output.save_plots,
                "plot_format": self.output.plot_format,
                "dpi": self.output.dpi,
            },
        }
