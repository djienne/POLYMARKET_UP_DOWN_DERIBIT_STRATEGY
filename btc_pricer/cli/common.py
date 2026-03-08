"""Common CLI utilities shared across all CLI scripts.

This module centralizes utilities that were previously duplicated across
cli.py, cli_terminal.py, cli_probability.py, and cli_intraday.py.
"""

import argparse
import logging
import sys
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path
from typing import Tuple, Optional, Callable, Any, List, Dict, Union

import numpy as np

from btc_pricer.config import Config
from btc_pricer.api.deribit import DeribitClient, DeribitAPIError
from btc_pricer.api.binance import fetch_spot_with_fallback
from btc_pricer.data.filters import DataFilter, FilteredOption
from btc_pricer.models.ssvi import SSVIFitter, SSVIModel, SSVISurfaceFitter, SSVISliceData
from btc_pricer.models.heston import (
    HestonFitter, HestonModel, HestonParams,
    check_iv_consistency, check_iv_consistency_from_result,
)
from btc_pricer.constants import (
    FAR_FUTURE_DATE,
    OPTION_EXPIRY_HOUR_UTC,
    RELAXED_MIN_POINTS,
)


# =============================================================================
# Logging Setup
# =============================================================================

def setup_logging(verbose: bool, datefmt: str = "%Y-%m-%d %H:%M:%S") -> None:
    """Configure logging.

    Args:
        verbose: If True, use DEBUG level; otherwise INFO.
        datefmt: Date format string for log timestamps.
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt=datefmt
    )


# =============================================================================
# Configuration
# =============================================================================

def load_config(config_path: Path, logger: logging.Logger) -> Config:
    """Load configuration from path or use defaults.

    Args:
        config_path: Path to YAML configuration file.
        logger: Logger instance for status messages.

    Returns:
        Config object (loaded from file or defaults).
    """
    if config_path.exists():
        logger.info(f"Loading configuration from {config_path}")
        return Config.from_yaml(config_path)
    else:
        logger.info("Using default configuration")
        return Config()


# =============================================================================
# Date/Time Parsing
# =============================================================================

def parse_expiry_date(expiry: str) -> datetime:
    """Parse expiry string (e.g., '28JAN26') to datetime for sorting.

    This function was previously duplicated in cli.py, cli_terminal.py,
    cli_probability.py, and test scripts.

    Args:
        expiry: Expiry string in Deribit format (DDMMMYY).

    Returns:
        Datetime object for sorting. Returns FAR_FUTURE_DATE if unparseable.
    """
    try:
        return datetime.strptime(expiry, "%d%b%y")
    except ValueError:
        # Fallback: return far future date so unparseable expiries sort last
        return FAR_FUTURE_DATE


def parse_expiry_to_utc(expiry: str) -> datetime:
    """Parse expiry string to UTC datetime (options expire at 08:00 UTC).

    Args:
        expiry: Expiry string in Deribit format (DDMMMYY).

    Returns:
        UTC datetime with hour set to expiry time (08:00 UTC).
    """
    try:
        dt = datetime.strptime(expiry, "%d%b%y")
        return dt.replace(
            hour=OPTION_EXPIRY_HOUR_UTC,
            minute=0,
            second=0,
            tzinfo=timezone.utc
        )
    except ValueError:
        return FAR_FUTURE_DATE.replace(
            hour=OPTION_EXPIRY_HOUR_UTC,
            minute=0,
            second=0,
            tzinfo=timezone.utc
        )


def format_current_time_multizone() -> str:
    """Format current time in multiple timezones (UTC, Paris, ET).

    This pattern was duplicated in cli_terminal.py format functions.

    Returns:
        Formatted string like "2026-01-28 14:30 UTC / 15:30 Paris / 09:30 ET"
    """
    from zoneinfo import ZoneInfo

    now_utc = datetime.now(timezone.utc)
    now_paris = now_utc.astimezone(ZoneInfo("Europe/Paris"))
    now_et = now_utc.astimezone(ZoneInfo("America/New_York"))

    return (
        f"{now_utc.strftime('%Y-%m-%d %H:%M')} UTC / "
        f"{now_paris.strftime('%H:%M')} Paris / "
        f"{now_et.strftime('%H:%M')} ET"
    )


# =============================================================================
# Model Fitter Creation
# =============================================================================

def create_ssvi_fitter(config: Config) -> SSVIFitter:
    """Create SSVIFitter from configuration.

    Includes defensive getattr() calls for optional TTM-adaptive settings.

    Args:
        config: Application configuration.

    Returns:
        Configured SSVIFitter instance.
    """
    return SSVIFitter(
        rho_bounds=config.ssvi.rho_bounds,
        phi_bounds=config.ssvi.phi_bounds,
        theta_bounds=config.ssvi.theta_bounds,
        optimizer=config.ssvi.optimizer,
        short_dated_ttm_threshold=config.ssvi.short_dated_ttm_threshold,
        short_dated_phi_bounds=config.ssvi.short_dated_phi_bounds,
        very_short_dated_ttm_threshold=getattr(
            config.ssvi, 'very_short_dated_ttm_threshold', 0.02
        ),
        very_short_dated_phi_bounds=getattr(
            config.ssvi, 'very_short_dated_phi_bounds', (0.001, 200.0)
        ),
        use_multi_start=config.ssvi.use_multi_start,
        n_starts=config.ssvi.n_starts,
        use_global_optimizer=config.ssvi.use_global_optimizer,
        global_optimizer=config.ssvi.global_optimizer,
        use_relative_error=config.ssvi.use_relative_error,
        regularization_lambda=config.ssvi.regularization_lambda
    )


def create_ssvi_surface_fitter(config: Config) -> SSVISurfaceFitter:
    """Create SSVISurfaceFitter from configuration.

    Args:
        config: Application configuration.

    Returns:
        Configured SSVISurfaceFitter instance.
    """
    per_slice_fitter = create_ssvi_fitter(config)
    return SSVISurfaceFitter(
        per_slice_fitter=per_slice_fitter,
        rho_bounds=config.ssvi_surface.rho_bounds,
        eta_bounds=config.ssvi_surface.eta_bounds,
        lam_bounds=config.ssvi_surface.lam_bounds,
        maxiter=config.ssvi_surface.maxiter,
        workers=config.ssvi_surface.workers,
        use_relative_error=config.ssvi_surface.use_relative_error,
        fallback_to_independent=config.ssvi_surface.fallback_to_independent,
    )


def create_heston_fitter(config: Config) -> HestonFitter:
    """Create HestonFitter from configuration.

    Args:
        config: Application configuration.

    Returns:
        Configured HestonFitter instance.
    """
    return HestonFitter(
        v0_bounds=config.heston.v0_bounds,
        kappa_bounds=config.heston.kappa_bounds,
        theta_bounds=config.heston.theta_bounds,
        xi_bounds=config.heston.xi_bounds,
        rho_bounds=config.heston.rho_bounds,
        optimizer=config.heston.optimizer,
        n_integration_points=config.heston.n_integration_points,
        use_quantlib=config.heston.use_quantlib,
        # Short-dated TTM settings
        short_dated_ttm_threshold=config.heston.short_dated_ttm_threshold,
        short_dated_xi_bounds=config.heston.short_dated_xi_bounds,
        short_dated_kappa_bounds=config.heston.short_dated_kappa_bounds,
        very_short_dated_ttm_threshold=config.heston.very_short_dated_ttm_threshold,
        very_short_dated_xi_bounds=config.heston.very_short_dated_xi_bounds,
        very_short_dated_kappa_bounds=config.heston.very_short_dated_kappa_bounds,
        # Multi-start optimization
        use_multi_start=config.heston.use_multi_start,
        n_starts=config.heston.n_starts,
        # Early termination
        early_termination_sse=config.heston.early_termination_sse,
    )


# =============================================================================
# Surface Data Extraction
# =============================================================================

def extract_surface_data(
    otm_surface: List[FilteredOption],
    min_points: int = RELAXED_MIN_POINTS,
    iv_valid_range: Tuple[float, float] = (0.05, 5.0)
) -> Optional[Tuple[float, float, float, np.ndarray, np.ndarray]]:
    """Extract and validate data from OTM surface.

    Args:
        otm_surface: List of filtered OTM options.
        min_points: Minimum number of valid points required.
        iv_valid_range: Valid range for implied volatility (min, max).

    Returns:
        Tuple of (forward, ttm, spot, log_moneyness, market_iv) or None if invalid.
    """
    if len(otm_surface) < min_points:
        return None

    forward = otm_surface[0].forward_price
    ttm = otm_surface[0].time_to_expiry
    spot = otm_surface[0].spot_price

    log_moneyness = np.array([opt.log_moneyness for opt in otm_surface])
    market_iv = np.array([opt.mark_iv for opt in otm_surface])

    # Filter valid IVs using configurable range
    iv_min, iv_max = iv_valid_range
    valid_mask = (market_iv > iv_min) & (market_iv < iv_max)
    if np.sum(valid_mask) < min_points:
        return None

    log_moneyness = log_moneyness[valid_mask]
    market_iv = market_iv[valid_mask]

    return forward, ttm, spot, log_moneyness, market_iv


def _prepare_expiry_surface(
    expiry: str,
    options: list,
    data_filter: DataFilter,
    config: Config,
    logger: logging.Logger,
) -> Optional[Tuple[str, float, float, np.ndarray, np.ndarray]]:
    """Filter options and extract surface data for a single expiry.

    Returns:
        (expiry, forward, ttm, log_moneyness, market_iv) or None if invalid.
    """
    filtered, stats = data_filter.filter_options(options, return_stats=True)

    if stats and stats.failed_spread > 0:
        logger.warning(
            f"{expiry}: {stats.failed_spread} options excluded due to wide bid-ask spread"
        )
    if stats and stats.failed_open_interest > 0 and stats.failed_open_interest > len(filtered):
        logger.warning(
            f"{expiry}: {stats.failed_open_interest} options excluded due to low open interest"
        )

    otm_surface = data_filter.build_otm_surface(filtered)
    surface_data = extract_surface_data(
        otm_surface,
        min_points=RELAXED_MIN_POINTS,
        iv_valid_range=config.validation.iv_valid_range,
    )

    if surface_data is None:
        logger.warning(
            f"Skipping {expiry}: insufficient valid options after filtering "
            f"(need at least {RELAXED_MIN_POINTS})"
        )
        return None

    forward, ttm, _, log_moneyness, market_iv = surface_data

    min_points = config.filters.min_surface_points
    if len(log_moneyness) < min_points:
        logger.warning(
            f"{expiry}: Low liquidity - only {len(log_moneyness)} options "
            f"(recommended: {min_points}+)"
        )

    return (expiry, forward, ttm, log_moneyness, market_iv)


def _check_heston_iv_consistency(
    result,
    market_iv: np.ndarray,
    log_moneyness: np.ndarray,
    config: Config,
) -> Tuple[bool, float]:
    """Check Heston IV consistency, using pre-computed model_iv when available.

    Returns:
        (is_consistent, max_iv_error)
    """
    if result.model_iv is not None:
        is_consistent, max_iv_error, _ = check_iv_consistency_from_result(
            result, market_iv, config.model.iv_consistency_threshold,
        )
    else:
        model = HestonModel(result.params, use_quantlib=config.heston.use_quantlib)
        is_consistent, max_iv_error, _ = check_iv_consistency(
            model, log_moneyness, market_iv, config.model.iv_consistency_threshold,
        )
    return is_consistent, max_iv_error


def _fit_heston_single(
    config: Config,
    expiry: str,
    forward: float,
    ttm: float,
    log_moneyness: np.ndarray,
    market_iv: np.ndarray,
) -> Optional[Tuple[str, Any, float, float, float]]:
    """Fit Heston to a single expiry. Pickle-friendly for ProcessPoolExecutor.

    Returns:
        (expiry, params, r_squared, forward, ttm) or None if fitting failed.
    """
    import time as _time
    logger = logging.getLogger(__name__)

    logger.info(f"[parallel] Calibrating Heston to {expiry} ({len(log_moneyness)} options)...")
    _t0 = _time.time()

    fitter = create_heston_fitter(config)
    result = fitter.fit(log_moneyness, market_iv, ttm, forward=forward)
    _elapsed = _time.time() - _t0

    if not result.success or result.params is None:
        logger.warning(
            f"[parallel] Heston fit failed for {expiry} in {_elapsed:.1f}s: {result.message}"
        )
        return None

    is_consistent, max_iv_error = _check_heston_iv_consistency(
        result, market_iv, log_moneyness, config,
    )

    if not is_consistent:
        logger.warning(
            f"[parallel] Heston IV consistency failed for {expiry} in {_elapsed:.1f}s: "
            f"max error {max_iv_error:.1%}"
        )
        return None

    logger.info(
        f"[parallel] Heston calibration successful for {expiry} in {_elapsed:.1f}s "
        f"(R²={result.r_squared:.3f}, IV_err={max_iv_error:.1%})"
    )
    return (expiry, result.params, result.r_squared, forward, ttm)


# =============================================================================
# Calibration Functions
# =============================================================================

# Type alias for calibration result
CalibrationResult = Tuple[Any, float, str, float, float, float, str]  # params, spot, expiry, fwd, r2, ttm, model
CalibrationResultBoth = Tuple[
    Optional[Tuple[Any, float]],  # heston_data: (params, r2) or None
    Optional[Tuple[Any, float]],  # ssvi_data: always None (vestigial, kept for API stability)
    float, str, float, float      # spot, expiry, forward, ttm
]


def calibrate_to_expiry(
    client: DeribitClient,
    config: Config,
    target_expiry: Optional[str] = None,
    return_both: bool = False
) -> Union[CalibrationResult, CalibrationResultBoth]:
    """Calibrate Heston model to options for a target or nearest expiry.

    Tries Heston across expiries (parallel when return_both=True).
    SSVI Surface fitting is done separately via fit_ssvi_surface_for_ttm().

    Args:
        client: Deribit API client.
        config: Configuration.
        target_expiry: Specific expiry to calibrate to (e.g., "31JAN26").
                      If None, uses nearest liquid expiry.
        return_both: If True, run parallel Heston across expiries and return
                     CalibrationResultBoth (ssvi_data slot is always None).

    Returns:
        If return_both=False:
            Tuple of (params, spot_price, expiry_name, forward_price, r_squared, ttm, model_type)
            where model_type is always 'heston'.
        If return_both=True:
            Tuple of (heston_result, ssvi_result, spot_price, expiry_name, forward_price, ttm)
            where heston_result is (params, r_squared) or None, ssvi_result is always None.

    Raises:
        DeribitAPIError: If no options data received or Heston could not be calibrated.
    """
    logger = logging.getLogger(__name__)

    # Fetch all options
    logger.info("Fetching options data from Deribit...")
    options_by_expiry = client.fetch_all_options("BTC")

    if not options_by_expiry:
        raise DeribitAPIError("No options data received")

    deribit_spot = list(options_by_expiry.values())[0][0].spot_price
    logger.info(f"Deribit spot price: ${deribit_spot:,.0f}")

    # Fetch spot price with Binance primary, Deribit fallback
    spot_price, _ = fetch_spot_with_fallback(deribit_spot)

    # Filter and sort expiries
    data_filter = DataFilter(config.filters)

    # Sort expiries by closeness to target (or by TTM if no target)
    if target_expiry:
        target_expiry_upper = target_expiry.upper()
        sorted_expiries = sorted(
            options_by_expiry.items(),
            key=lambda x: abs(
                (parse_expiry_date(x[0]) - parse_expiry_date(target_expiry_upper)).days
            )
        )
    else:
        sorted_expiries = sorted(
            options_by_expiry.items(),
            key=lambda x: x[1][0].time_to_expiry if x[1] else float('inf')
        )

    # Cross-expiry calibration: skip ultra-short TTM expiries (first pass)
    min_calibration_ttm = getattr(config.model, 'min_calibration_ttm_days', 1.0) / 365.0
    short_ttm_expiries = []  # Save for fallback second pass

    # =========================================================================
    # return_both=True: parallel Heston across expiries
    # =========================================================================
    if return_both:
        return _calibrate_both_parallel(
            config, sorted_expiries, data_filter,
            spot_price, target_expiry, min_calibration_ttm, short_ttm_expiries,
            logger,
        )

    # =========================================================================
    # return_both=False: sequential Heston-only calibration
    # =========================================================================
    heston_fitter = create_heston_fitter(config)
    for expiry, options in sorted_expiries:
        # When auto-selecting (no target), skip expiries below min TTM threshold
        if target_expiry is None and options:
            expiry_ttm = options[0].time_to_expiry
            expiry_ttm_days = expiry_ttm * 365
            if expiry_ttm < min_calibration_ttm:
                logger.info(
                    f"Skipping {expiry} (TTM={expiry_ttm_days:.1f}d < min "
                    f"{config.model.min_calibration_ttm_days:.1f}d), "
                    f"using longer expiry for calibration"
                )
                short_ttm_expiries.append((expiry, options))
                continue

        surface = _prepare_expiry_surface(expiry, options, data_filter, config, logger)
        if surface is None:
            continue

        expiry, forward, ttm, log_moneyness, market_iv = surface

        import time as _time
        logger.info(f"Calibrating Heston to {expiry} ({len(log_moneyness)} options)...")
        _t0 = _time.time()
        heston_result = heston_fitter.fit(log_moneyness, market_iv, ttm, forward=forward)
        _heston_elapsed = _time.time() - _t0

        if heston_result.success and heston_result.params is not None:
            is_consistent, max_iv_error = _check_heston_iv_consistency(
                heston_result, market_iv, log_moneyness, config,
            )

            if is_consistent:
                logger.info(
                    f"Heston calibration successful for {expiry} in {_heston_elapsed:.1f}s "
                    f"(R²={heston_result.r_squared:.3f}, IV_err={max_iv_error:.1%})"
                )
                return (heston_result.params, spot_price, expiry, forward,
                        heston_result.r_squared, ttm, "heston")
            else:
                logger.warning(
                    f"Heston IV consistency check failed for {expiry} in {_heston_elapsed:.1f}s: "
                    f"max error {max_iv_error:.1%}"
                )
        else:
            logger.warning(
                f"Heston fit failed for {expiry} in {_heston_elapsed:.1f}s: "
                f"{heston_result.message}"
            )

        logger.warning(f"Heston failed for {expiry}")
        # Continue to next expiry

    # Second pass: try short-TTM expiries that were skipped (graceful degradation)
    if short_ttm_expiries:
        logger.warning(
            "All longer expiries failed calibration, falling back to short-TTM expiries..."
        )
        for expiry, options in short_ttm_expiries:
            surface = _prepare_expiry_surface(expiry, options, data_filter, config, logger)
            if surface is None:
                continue

            expiry, forward, ttm, log_moneyness, market_iv = surface

            logger.info(f"Calibrating Heston to {expiry} (fallback, {len(log_moneyness)} options)...")
            heston_result = heston_fitter.fit(log_moneyness, market_iv, ttm, forward=forward)

            if heston_result.success and heston_result.params is not None:
                is_consistent, max_iv_error = _check_heston_iv_consistency(
                    heston_result, market_iv, log_moneyness, config,
                )
                if is_consistent:
                    return (heston_result.params, spot_price, expiry, forward,
                            heston_result.r_squared, ttm, "heston")

    raise DeribitAPIError("Could not calibrate Heston to any expiry")


def _calibrate_both_parallel(
    config: Config,
    sorted_expiries: list,
    data_filter: DataFilter,
    spot_price: float,
    target_expiry: Optional[str],
    min_calibration_ttm: float,
    short_ttm_expiries: list,
    logger: logging.Logger,
) -> CalibrationResultBoth:
    """Parallel Heston calibration across expiries for return_both=True.

    Per-slice SSVI is no longer fitted here — the caller falls through to
    fit_ssvi_surface_for_ttm() which is the real SSVI model.

    Returns:
        CalibrationResultBoth tuple.  When Heston fails all expiries the
        ssvi_data slot is None; the caller should use fit_ssvi_surface_for_ttm().
    """
    import time as _time
    from concurrent.futures import ProcessPoolExecutor, as_completed

    # First pass: collect all valid expiry surfaces, skipping short TTM
    expiry_surfaces = []  # [(expiry, forward, ttm, log_moneyness, market_iv)]

    for expiry, options in sorted_expiries:
        if target_expiry is None and options:
            expiry_ttm = options[0].time_to_expiry
            if expiry_ttm < min_calibration_ttm:
                expiry_ttm_days = expiry_ttm * 365
                logger.info(
                    f"Skipping {expiry} (TTM={expiry_ttm_days:.1f}d < min "
                    f"{config.model.min_calibration_ttm_days:.1f}d), "
                    f"using longer expiry for calibration"
                )
                short_ttm_expiries.append((expiry, options))
                continue

        surface = _prepare_expiry_surface(expiry, options, data_filter, config, logger)
        if surface is not None:
            expiry_surfaces.append(surface)

    # Also include short-TTM expiries as fallback candidates
    for expiry, options in short_ttm_expiries:
        surface = _prepare_expiry_surface(expiry, options, data_filter, config, logger)
        if surface is not None:
            expiry_surfaces.append(surface)

    if not expiry_surfaces:
        raise DeribitAPIError("Could not calibrate Heston to any expiry")

    # Use closest expiry metadata for fallback return
    closest = expiry_surfaces[0]
    closest_expiry, closest_fwd, closest_ttm, _, _ = closest

    # Parallel Heston fitting across up to 4 expiries
    n_workers = min(len(expiry_surfaces), 4)
    logger.info(
        f"Launching parallel Heston calibration across {len(expiry_surfaces)} expiries "
        f"({n_workers} workers)..."
    )
    _t0_parallel = _time.time()

    best_heston = None  # (expiry, params, r2, forward, ttm)

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {}
        for exp, fwd, ttm, lm, iv in expiry_surfaces:
            future = executor.submit(
                _fit_heston_single, config, exp, fwd, ttm, lm, iv
            )
            futures[future] = exp

        for future in as_completed(futures):
            exp = futures[future]
            try:
                result = future.result()
            except Exception as e:
                logger.warning(f"[parallel] Heston worker exception for {exp}: {e}")
                continue

            if result is None:
                continue

            r_expiry, r_params, r_r2, r_fwd, r_ttm = result
            # Pick the result with the smallest TTM (closest expiry)
            if best_heston is None or r_ttm < best_heston[4]:
                best_heston = result

    _parallel_elapsed = _time.time() - _t0_parallel
    if best_heston is not None:
        h_expiry, h_params, h_r2, h_fwd, h_ttm = best_heston
        logger.info(
            f"Parallel Heston completed in {_parallel_elapsed:.1f}s — "
            f"best result: {h_expiry} (R²={h_r2:.3f}, TTM={h_ttm*365:.1f}d)"
        )
        heston_data = (h_params, h_r2)
        return (heston_data, None, spot_price, h_expiry, h_fwd, h_ttm)
    else:
        logger.info(
            f"Parallel Heston completed in {_parallel_elapsed:.1f}s — "
            f"no successful fit on any expiry"
        )

    # Heston failed all expiries — return None for both models.
    # The caller (run_terminal) will fall through to fit_ssvi_surface_for_ttm().
    logger.info("Heston failed on all expiries, caller will use SSVI surface fit")
    return (None, None, spot_price, closest_expiry, closest_fwd, closest_ttm)


def find_closest_expiry_after(
    options_by_expiry: Dict[str, List[Any]],
    target_utc: datetime
) -> str:
    """Find the closest expiry that expires AFTER the target time.

    Options expire at 08:00 UTC on their expiry date.

    Args:
        options_by_expiry: Dictionary mapping expiry strings to option data.
        target_utc: Target UTC datetime.

    Returns:
        Expiry string (e.g., "31JAN26").

    Raises:
        ValueError: If no expiry found after target time.
    """
    valid_expiries = []

    for expiry_str in options_by_expiry.keys():
        expiry_utc = parse_expiry_to_utc(expiry_str)
        if expiry_utc > target_utc:
            valid_expiries.append((expiry_str, expiry_utc))

    if not valid_expiries:
        raise ValueError(
            f"No options expiry found after target time "
            f"({target_utc.strftime('%Y-%m-%d %H:%M UTC')})"
        )

    # Sort by expiry time and return the closest one
    valid_expiries.sort(key=lambda x: x[1])
    return valid_expiries[0][0]


# =============================================================================
# SSVI Surface Fitting
# =============================================================================

class SurfaceFitResult:
    """Result of SSVI surface fitting and interpolation to a target TTM."""
    __slots__ = ('params_at_ttm', 'surface_params', 'r2', 'info', 'elapsed')

    def __init__(self, params_at_ttm, surface_params, r2, info, elapsed):
        self.params_at_ttm = params_at_ttm
        self.surface_params = surface_params
        self.r2 = r2
        self.info = info
        self.elapsed = elapsed


def _collect_surface_slices(
    options_by_expiry: Dict,
    data_filter: DataFilter,
    config: Config,
    max_ttm_days: float,
) -> list:
    """Collect SSVISliceData from options within a TTM window.

    Returns:
        List of SSVISliceData for expiries within max_ttm_days.
    """
    slices = []
    for exp_name in sorted(options_by_expiry.keys(), key=parse_expiry_date):
        opts = options_by_expiry[exp_name]
        filtered_opts, _ = data_filter.filter_options(opts, return_stats=True)
        otm = data_filter.build_otm_surface(filtered_opts)
        sd = extract_surface_data(
            otm, min_points=RELAXED_MIN_POINTS,
            iv_valid_range=config.validation.iv_valid_range,
        )
        if sd is None:
            continue
        fwd_s, ttm_s, _, log_k_s, mkt_iv_s = sd
        if ttm_s * 365 > max_ttm_days:
            continue
        slices.append(SSVISliceData(
            expiry_name=exp_name, ttm=ttm_s,
            log_moneyness=log_k_s, market_iv=mkt_iv_s, forward=fwd_s,
        ))
    return slices


def fit_ssvi_surface_for_ttm(
    client: DeribitClient,
    config: Config,
    target_ttm: float,
    options_by_expiry: Optional[Dict] = None,
) -> Optional[SurfaceFitResult]:
    """Fit SSVI surface across nearby expiries and interpolate to target TTM.

    Args:
        client: Deribit API client.
        config: Application configuration.
        target_ttm: Target time-to-maturity in years for interpolation.
        options_by_expiry: Pre-fetched options data (fetched if None).

    Returns:
        SurfaceFitResult or None if fitting failed.
    """
    import time as _time
    logger = logging.getLogger(__name__)

    logger.info("Fitting SSVI surface across nearby expiries...")

    if options_by_expiry is None:
        options_by_expiry = client.fetch_all_options("BTC")

    if not options_by_expiry:
        return None

    data_filter_surf = DataFilter(config.filters)
    surface_slices = _collect_surface_slices(
        options_by_expiry, data_filter_surf, config,
        config.ssvi_surface.max_ttm_days,
    )

    if len(surface_slices) < config.ssvi_surface.min_expiries:
        logger.info(
            f"Not enough slices for surface fit "
            f"({len(surface_slices)} < {config.ssvi_surface.min_expiries})"
        )
        return None

    surface_fitter = create_ssvi_surface_fitter(config)

    def _try_fit(slices_list, label):
        for attempt_name, slices_to_try in slices_list:
            if len(slices_to_try) < config.ssvi_surface.min_expiries:
                continue
            _t0 = _time.time()
            surface_fit = surface_fitter.fit(slices_to_try)
            elapsed = _time.time() - _t0

            if surface_fit.success and surface_fit.params is not None:
                params_obj = surface_fit.params
                r2 = surface_fit.aggregate_r_squared
                params_at_ttm = params_obj.get_params_for_ttm(target_ttm)
                n_slices = len(slices_to_try)
                slice_names = [s.expiry_name for s in slices_to_try]
                info = (
                    f"Surface: rho={params_obj.rho:.3f}, "
                    f"eta={params_obj.eta:.3f}, lam={params_obj.lam:.3f} "
                    f"({n_slices} expiries: {', '.join(slice_names)})"
                )
                logger.info(
                    f"SSVI surface fit OK ({attempt_name}) in {elapsed:.1f}s: "
                    f"R2={r2:.4f}, {info}"
                )
                logger.info(
                    f"Surface interpolated to TTM={target_ttm*365:.3f}d: "
                    f"theta={params_at_ttm.theta:.6f}, "
                    f"phi={params_at_ttm.phi:.3f}, "
                    f"rho={params_at_ttm.rho:.4f}"
                )
                return SurfaceFitResult(params_at_ttm, params_obj, r2, info, elapsed)
            else:
                logger.warning(
                    f"SSVI surface fit failed ({attempt_name}): {surface_fit.message}"
                )
        return None

    # Try progressively: drop near expiries (often degraded), then far expiry
    n = len(surface_slices)
    min_exp = config.ssvi_surface.min_expiries
    attempts = [("all slices", surface_slices)]
    if n > min_exp:
        attempts.append(("without shortest expiry", surface_slices[1:]))
    if n > min_exp + 1:
        attempts.append(("without 2 shortest expiries", surface_slices[2:]))
    if n > min_exp:
        attempts.append(("without longest expiry", surface_slices[:-1]))

    result = _try_fit(attempts, "standard")
    if result is not None:
        return result

    # Try expanding max_ttm_days to 2x (adds longer-dated expiries)
    expanded_max = config.ssvi_surface.max_ttm_days * 2
    expanded_slices = _collect_surface_slices(
        options_by_expiry, data_filter_surf, config, expanded_max,
    )

    if len(expanded_slices) > len(surface_slices):
        n_exp = len(expanded_slices)
        expand_attempts = [(f"expanded {expanded_max:.0f}d", expanded_slices)]
        if n_exp > min_exp:
            expand_attempts.append(
                (f"expanded {expanded_max:.0f}d without shortest", expanded_slices[1:])
            )
        if n_exp > min_exp + 1:
            expand_attempts.append(
                (f"expanded {expanded_max:.0f}d without 2 shortest", expanded_slices[2:])
            )
        result = _try_fit(expand_attempts, "expanded")
        if result is not None:
            return result

    logger.warning("SSVI surface fit failed all retry attempts")
    return None


def get_atm_iv_from_nearest_expiry(
    client: DeribitClient,
    config: Config
) -> Tuple[float, float, str, float]:
    """Fetch options and extract ATM IV from the nearest liquid expiry.

    Returns:
        Tuple of (atm_iv, spot_price, expiry_name, forward_price)

    Raises:
        DeribitAPIError: If no suitable expiry found.
    """
    logger = logging.getLogger(__name__)

    logger.info("Fetching options data from Deribit...")
    options_by_expiry = client.fetch_all_options("BTC")

    if not options_by_expiry:
        raise DeribitAPIError("No options data received")

    spot_price = list(options_by_expiry.values())[0][0].spot_price
    logger.info(f"Deribit spot price: ${spot_price:,.0f}")

    data_filter = DataFilter(config.filters)
    fitter = create_ssvi_fitter(config)

    sorted_expiries = sorted(
        options_by_expiry.items(),
        key=lambda x: x[1][0].time_to_expiry if x[1] else float('inf')
    )

    min_points = config.filters.min_surface_points
    for expiry, options in sorted_expiries:
        filtered, _ = data_filter.filter_options(options)

        if len(filtered) < min_points:
            logger.debug(f"Skipping {expiry}: insufficient options ({len(filtered)})")
            continue

        otm_surface = data_filter.build_otm_surface(filtered)
        surface_data = extract_surface_data(
            otm_surface,
            min_points=min_points,
            iv_valid_range=config.validation.iv_valid_range
        )
        if surface_data is None:
            continue

        forward, ttm, _, log_moneyness, market_iv = surface_data

        fit_result = fitter.fit(log_moneyness, market_iv, ttm)

        if fit_result.success and fit_result.params is not None:
            ssvi_model = SSVIModel(fit_result.params)
            atm_iv = ssvi_model.implied_volatility(0)

            logger.info(f"Using expiry {expiry} (TTM: {ttm*365:.1f} days)")
            logger.info(f"ATM IV: {atm_iv*100:.1f}%")
            logger.info(f"Forward: ${forward:,.0f}")

            return atm_iv, spot_price, expiry, forward

    raise DeribitAPIError("Could not find suitable expiry for ATM IV extraction")


# =============================================================================
# CLI Exception Handling
# =============================================================================

def handle_cli_exceptions(func: Callable) -> Callable:
    """Decorator to handle common CLI exceptions.

    Catches DeribitAPIError, KeyboardInterrupt, and general exceptions,
    logs them appropriately, and exits with appropriate codes.

    Args:
        func: Function to wrap.

    Returns:
        Wrapped function with exception handling.
    """
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        logger = logging.getLogger(__name__)
        try:
            return func(*args, **kwargs)
        except DeribitAPIError as e:
            logger.error(f"API error: {e}")
            sys.exit(1)
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
            sys.exit(130)
        except Exception as e:
            logger.exception(f"Unexpected error: {e}")
            sys.exit(1)
    return wrapper


# =============================================================================
# CLI Argument Parsing
# =============================================================================

def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    """Add common CLI arguments.

    Adds --config and --verbose arguments that are used by all CLI scripts.

    Args:
        parser: ArgumentParser to add arguments to.
    """
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config.yaml"),
        help="Path to configuration file"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
