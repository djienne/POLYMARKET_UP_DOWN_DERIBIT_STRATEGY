"""Common CLI utilities shared across all CLI scripts.

This module centralizes utilities that were previously duplicated across
cli.py, cli_terminal.py, and cli_intraday.py.
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
from btc_pricer.models.ssvi import SSVIFitter, SSVISurfaceFitter
from btc_pricer.models.heston import HestonFitter, HestonModel, HestonParams, check_iv_consistency
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
    cli_terminal.py (--method rnd), and test scripts.

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
        # Ultra-short-dated TTM settings
        ultra_short_dated_ttm_threshold=getattr(
            config.heston, 'ultra_short_dated_ttm_threshold', 0.01
        ),
        ultra_short_dated_xi_bounds=getattr(
            config.heston, 'ultra_short_dated_xi_bounds', (0.1, 10.0)
        ),
        ultra_short_dated_kappa_bounds=getattr(
            config.heston, 'ultra_short_dated_kappa_bounds', (0.5, 5.0)
        ),
        ultra_short_dated_theta_factor=getattr(
            config.heston, 'ultra_short_dated_theta_factor', (0.5, 2.0)
        ),
        # Multi-start optimization
        use_multi_start=config.heston.use_multi_start,
        n_starts=config.heston.n_starts,
        # QuantLib objective implementation + optional fallback acceleration
        quantlib_objective_impl=config.heston.quantlib_objective_impl,
        enable_numba_fallback=config.heston.enable_numba_fallback,
        numba_strict_mode=config.heston.numba_strict_mode,
        # Early termination
        early_termination_sse=getattr(config.heston, 'early_termination_sse', None),
        # Relative error objective
        use_relative_error=getattr(config.heston, 'use_relative_error', True),
        # Gaussian near-ATM weighting for short TTM
        short_ttm_gaussian_weighting=getattr(
            config.heston, 'short_ttm_gaussian_weighting', True
        ),
        short_ttm_gaussian_sigma_base=getattr(
            config.heston, 'short_ttm_gaussian_sigma_base', 0.05
        ),
        short_ttm_gaussian_sigma_ttm_scale=getattr(
            config.heston, 'short_ttm_gaussian_sigma_ttm_scale', 2.0
        ),
        short_ttm_gaussian_floor=getattr(
            config.heston, 'short_ttm_gaussian_floor', 0.1
        ),
        max_workers=getattr(config.heston, 'max_workers', 4),
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


# =============================================================================
# Calibration Functions
# =============================================================================

# Type alias for calibration result
CalibrationResult = Tuple[Any, float, str, float, float, float, str]  # params, spot, expiry, fwd, r2, ttm, model
CalibrationResultBoth = Tuple[
    Optional[Tuple[Any, float]],  # heston_data: (params, r2) or None
    Optional[Tuple[Any, float]],  # ssvi_data: (params, r2) or None
    float, str, float, float      # spot, expiry, forward, ttm
]


def calibrate_to_expiry(
    client: DeribitClient,
    config: Config,
    target_expiry: Optional[str] = None,
    return_both: bool = False
) -> Union[CalibrationResult, CalibrationResultBoth]:
    """Calibrate volatility model to options for a target or nearest expiry.

    Tries Heston first (better for vol-of-vol dynamics), falls back to SSVI
    if Heston calibration fails or produces poor IV fits.

    This function was previously duplicated (~180 lines each) in cli_terminal.py
    and cli_terminal.py (--method rnd) with minor differences.

    Args:
        client: Deribit API client.
        config: Configuration.
        target_expiry: Specific expiry to calibrate to (e.g., "31JAN26").
                      If None, uses nearest liquid expiry.
        return_both: If True, return both Heston and SSVI results when available.

    Returns:
        If return_both=False:
            Tuple of (params, spot_price, expiry_name, forward_price, r_squared, ttm, model_type)
            where model_type is 'heston' or 'ssvi'
        If return_both=True:
            Tuple of (heston_result, ssvi_result, spot_price, expiry_name, forward_price, ttm)
            where each result is (params, r_squared) or None if failed

    Raises:
        DeribitAPIError: If no options data received or no model could be calibrated.
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
    skip_heston = config.model.default_model != "heston" and not return_both
    heston_fitter = None if skip_heston else create_heston_fitter(config)
    ssvi_fitter = create_ssvi_fitter(config)

    min_points = config.filters.min_surface_points

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

    # When return_both=True: save SSVI result if Heston fails, keep trying Heston on later expiries
    saved_ssvi = None
    saved_ssvi_context = None

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

        # Filter options and get stats for warnings
        filtered, stats = data_filter.filter_options(options, return_stats=True)

        # Show liquidity warnings
        if stats and stats.failed_spread > 0:
            logger.warning(
                f"{expiry}: {stats.failed_spread} options excluded due to wide bid-ask spread"
            )
        if stats and stats.failed_open_interest > 0 and stats.failed_open_interest > len(filtered):
            logger.warning(
                f"{expiry}: {stats.failed_open_interest} options excluded due to low open interest"
            )

        # Build OTM surface
        otm_surface = data_filter.build_otm_surface(filtered)

        # Use relaxed min_points - always try closest expiry
        surface_data = extract_surface_data(
            otm_surface,
            min_points=RELAXED_MIN_POINTS,
            iv_valid_range=config.validation.iv_valid_range
        )

        if surface_data is None:
            logger.warning(
                f"Skipping {expiry}: insufficient valid options after filtering "
                f"(need at least {RELAXED_MIN_POINTS})"
            )
            continue

        forward, ttm, _, log_moneyness, market_iv = surface_data

        # Warn if fewer options than ideal
        if len(log_moneyness) < min_points:
            logger.warning(
                f"{expiry}: Low liquidity - only {len(log_moneyness)} options "
                f"(recommended: {min_points}+)"
            )

        # Try Heston first (unless disabled by config)
        heston_valid = False
        heston_params = None
        heston_r2 = None

        if skip_heston:
            logger.info(
                f"Skipping Heston calibration for {expiry} "
                f"(default_model={config.model.default_model})"
            )
        else:
            import time as _time
            logger.info(f"Calibrating Heston to {expiry} ({len(log_moneyness)} options)...")
            _t0 = _time.time()
            heston_result = heston_fitter.fit(log_moneyness, market_iv, ttm, forward=forward)
            _heston_elapsed = _time.time() - _t0

            if heston_result.success and heston_result.params is not None:
                # Check IV consistency
                heston_model = HestonModel(
                    heston_result.params,
                    use_quantlib=config.heston.use_quantlib
                )
                is_consistent, max_iv_error, _ = check_iv_consistency(
                    heston_model, log_moneyness, market_iv,
                    config.model.iv_consistency_threshold,
                    ttm=ttm,
                    relaxation=getattr(config.model, 'iv_consistency_relaxation', 0.15),
                    ttm_cutoff=getattr(config.model, 'iv_consistency_ttm_cutoff', 0.05),
                )

                if is_consistent:
                    logger.info(
                        f"Heston calibration successful for {expiry} in {_heston_elapsed:.1f}s "
                        f"(R²={heston_result.r_squared:.3f}, IV_err={max_iv_error:.1%})"
                    )
                    heston_valid = True
                    heston_params = heston_result.params
                    heston_r2 = heston_result.r_squared
                else:
                    logger.warning(
                        f"Heston IV consistency check failed for {expiry} in {_heston_elapsed:.1f}s: "
                        f"max error {max_iv_error:.1%}, trying SSVI..."
                    )
            else:
                logger.warning(
                    f"Heston fit failed for {expiry} in {_heston_elapsed:.1f}s: "
                    f"{heston_result.message}, trying SSVI..."
                )

        # If we already have SSVI saved and just need Heston, skip SSVI fitting
        if return_both and saved_ssvi is not None:
            if heston_valid:
                logger.info(
                    f"Heston succeeded on {expiry} (different from SSVI expiry "
                    f"{saved_ssvi_context[1]}), returning combined result"
                )
                return ((heston_params, heston_r2), saved_ssvi, *saved_ssvi_context)
            else:
                # Heston still failed, try next expiry
                continue

        # Also fit SSVI (for comparison or fallback)
        import time as _time
        logger.info(f"Fitting SSVI for {expiry}...")
        _t0 = _time.time()
        ssvi_result = ssvi_fitter.fit(log_moneyness, market_iv, ttm)

        ssvi_valid = False
        ssvi_params = None
        ssvi_r2 = None

        _ssvi_elapsed = _time.time() - _t0

        if ssvi_result.success and ssvi_result.params is not None:
            logger.info(
                f"SSVI calibration successful for {expiry} in {_ssvi_elapsed:.1f}s "
                f"(R²={ssvi_result.r_squared:.3f})"
            )
            ssvi_valid = True
            ssvi_params = ssvi_result.params
            ssvi_r2 = ssvi_result.r_squared

        # Return both if requested
        if return_both:
            if heston_valid and ssvi_valid:
                # Both succeeded — return immediately
                return ((heston_params, heston_r2), (ssvi_params, ssvi_r2),
                        spot_price, expiry, forward, ttm)
            elif ssvi_valid and not heston_valid:
                # SSVI ok, Heston failed — save SSVI result, keep trying Heston
                # on subsequent expiries
                saved_ssvi = (ssvi_params, ssvi_r2)
                saved_ssvi_context = (spot_price, expiry, forward, ttm)
                logger.info(
                    f"SSVI succeeded for {expiry}, continuing to find Heston fit..."
                )
                continue
            elif heston_valid and not ssvi_valid:
                # Heston ok, SSVI failed — return Heston only
                return ((heston_params, heston_r2), None,
                        spot_price, expiry, forward, ttm)
            else:
                # Neither worked
                logger.warning(f"Both Heston and SSVI failed for {expiry}")
                continue

        # Standard return: prefer Heston, fallback to SSVI
        if heston_valid:
            return (heston_params, spot_price, expiry, forward, heston_r2, ttm, "heston")
        elif ssvi_valid:
            return (ssvi_params, spot_price, expiry, forward, ssvi_r2, ttm, "ssvi")

        logger.warning(f"Both Heston and SSVI failed for {expiry}")
        # Continue to next expiry

    # If SSVI succeeded but Heston failed on all first-pass expiries, return SSVI-only
    # (unless there are short-TTM expiries to try for Heston)
    if saved_ssvi is not None and not short_ttm_expiries:
        logger.info("Heston failed on all expiries, returning SSVI-only result")
        return (None, saved_ssvi, *saved_ssvi_context)

    # Second pass: try short-TTM expiries that were skipped (graceful degradation)
    if short_ttm_expiries:
        logger.warning(
            "All longer expiries failed calibration, falling back to short-TTM expiries..."
        )
        for expiry, options in short_ttm_expiries:
            filtered, stats = data_filter.filter_options(options, return_stats=True)

            if stats and stats.failed_spread > 0:
                logger.warning(
                    f"{expiry}: {stats.failed_spread} options excluded due to wide bid-ask spread"
                )

            otm_surface = data_filter.build_otm_surface(filtered)
            surface_data = extract_surface_data(
                otm_surface,
                min_points=RELAXED_MIN_POINTS,
                iv_valid_range=config.validation.iv_valid_range
            )

            if surface_data is None:
                logger.warning(f"Skipping {expiry}: insufficient valid options after filtering")
                continue

            forward, ttm, _, log_moneyness, market_iv = surface_data

            heston_valid = False
            heston_params = None
            heston_r2 = None

            if not skip_heston:
                logger.info(f"Calibrating Heston to {expiry} (fallback, {len(log_moneyness)} options)...")
                heston_result = heston_fitter.fit(log_moneyness, market_iv, ttm, forward=forward)

                if heston_result.success and heston_result.params is not None:
                    heston_model = HestonModel(
                        heston_result.params,
                        use_quantlib=config.heston.use_quantlib
                    )
                    is_consistent, max_iv_error, _ = check_iv_consistency(
                        heston_model, log_moneyness, market_iv,
                        config.model.iv_consistency_threshold,
                        ttm=ttm,
                        relaxation=getattr(config.model, 'iv_consistency_relaxation', 0.15),
                        ttm_cutoff=getattr(config.model, 'iv_consistency_ttm_cutoff', 0.05),
                    )
                    if is_consistent:
                        heston_valid = True
                        heston_params = heston_result.params
                        heston_r2 = heston_result.r_squared

            # If we already have SSVI saved and just need Heston, skip SSVI fitting
            if return_both and saved_ssvi is not None:
                if heston_valid:
                    logger.info(
                        f"Heston succeeded on {expiry} (fallback, different from SSVI expiry "
                        f"{saved_ssvi_context[1]}), returning combined result"
                    )
                    return ((heston_params, heston_r2), saved_ssvi, *saved_ssvi_context)
                else:
                    continue

            logger.info(f"Fitting SSVI for {expiry} (fallback)...")
            ssvi_result = ssvi_fitter.fit(log_moneyness, market_iv, ttm)
            ssvi_valid = False
            ssvi_params = None
            ssvi_r2 = None
            if ssvi_result.success and ssvi_result.params is not None:
                ssvi_valid = True
                ssvi_params = ssvi_result.params
                ssvi_r2 = ssvi_result.r_squared

            if return_both:
                if heston_valid and ssvi_valid:
                    return ((heston_params, heston_r2), (ssvi_params, ssvi_r2),
                            spot_price, expiry, forward, ttm)
                elif ssvi_valid and not heston_valid:
                    saved_ssvi = (ssvi_params, ssvi_r2)
                    saved_ssvi_context = (spot_price, expiry, forward, ttm)
                    logger.info(
                        f"SSVI succeeded for {expiry} (fallback), "
                        f"continuing to find Heston fit..."
                    )
                    continue
                elif heston_valid and not ssvi_valid:
                    return ((heston_params, heston_r2), None,
                            spot_price, expiry, forward, ttm)
                else:
                    continue

            if heston_valid:
                return (heston_params, spot_price, expiry, forward, heston_r2, ttm, "heston")
            elif ssvi_valid:
                return (ssvi_params, spot_price, expiry, forward, ssvi_r2, ttm, "ssvi")

    # If SSVI succeeded somewhere but Heston never did, return SSVI-only
    if saved_ssvi is not None:
        logger.info("Heston failed on all expiries, returning SSVI-only result")
        return (None, saved_ssvi, *saved_ssvi_context)

    raise DeribitAPIError("Could not calibrate any model to any expiry")


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
