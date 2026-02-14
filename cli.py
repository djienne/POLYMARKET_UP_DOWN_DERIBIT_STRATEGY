#!/usr/bin/env python3
"""
BTC Price Forecaster - CLI Entry Point

Extract Risk-Neutral Densities from Deribit Bitcoin options using
SSVI model fitting and the Breeden-Litzenberger formula.
"""

import argparse
import json
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from btc_pricer.config import Config
from btc_pricer.api.deribit import DeribitClient, DeribitAPIError
from btc_pricer.data.filters import DataFilter, FilteredOption
from btc_pricer.models.black_scholes import BlackScholes
from btc_pricer.models.ssvi import SSVIFitter, SSVIParams, SSVISliceData
from btc_pricer.models.heston import HestonFitter, HestonParams, HestonFitResult, check_iv_consistency
from btc_pricer.models.breeden_litzenberger import BreedenLitzenberger, RNDResult
from btc_pricer.statistics.rnd_stats import RNDStatistics, ExpiryStats
from btc_pricer.visualization.plots import RNDPlotter
from btc_pricer.utils.sanity_checks import SanityChecker, CheckStatus
from btc_pricer.api.binance import fetch_spot_with_fallback
from btc_pricer.cli.common import (
    setup_logging,
    load_config,
    create_ssvi_fitter,
    create_ssvi_surface_fitter,
    create_heston_fitter,
    extract_surface_data,
    handle_cli_exceptions,
    add_common_arguments,
    parse_expiry_date,
)


def process_expiry(
    expiry: str,
    options: List[FilteredOption],
    config: Config,
    checker: SanityChecker,
    model_override: Optional[str] = None
) -> Optional[tuple]:
    """Process a single expiry to extract RND.

    Fits both Heston and SSVI models, then selects the best based on R² comparison.
    Prefers Heston unless SSVI R² exceeds Heston R² by the configured threshold.

    Args:
        expiry: Expiry date string.
        options: Filtered options for this expiry.
        config: Configuration.
        checker: Sanity checker.
        model_override: Optional model override ("heston" or "ssvi").

    Returns:
        Tuple of (params, RNDResult, log_moneyness, market_iv, model_used, extra_info) or None.
        extra_info contains heston_r_squared, ssvi_r_squared, and selection_reason.
    """
    logger = logging.getLogger(__name__)

    min_points = config.filters.min_surface_points
    if len(options) < min_points:
        logger.warning(f"Skipping {expiry}: only {len(options)} options after filtering")
        return None

    # Build OTM surface
    data_filter = DataFilter(config.filters)
    otm_surface = data_filter.build_otm_surface(options)

    # Extract and validate surface data
    surface_data = extract_surface_data(
        otm_surface,
        min_points=min_points,
        iv_valid_range=config.validation.iv_valid_range
    )
    if surface_data is None:
        logger.warning(f"Skipping {expiry}: insufficient valid OTM options or IVs")
        return None

    forward, ttm, spot, log_moneyness, market_iv = surface_data

    # Check API data
    api_check = checker.check_api_data(spot, len(otm_surface), expiry)
    if api_check.has_critical():
        logger.error(f"Critical API data issue for {expiry}")
        return None

    # Initialize BL extractor
    bl = BreedenLitzenberger(
        strike_grid_points=config.breeden_litzenberger.strike_grid_points,
        strike_range_std=config.breeden_litzenberger.strike_range_std,
        use_log_strikes=config.breeden_litzenberger.use_log_strikes
    )

    # If model override is specified, use the old single-model logic
    if model_override:
        return _process_expiry_single_model(
            expiry, log_moneyness, market_iv, ttm, forward,
            config, checker, bl, model_override, logger
        )

    # Fit models (skip Heston when config says SSVI-only)
    skip_heston = config.model.default_model != "heston"
    ssvi_fitter = create_ssvi_fitter(config)

    if skip_heston:
        logger.info(f"Fitting SSVI for {expiry} ({len(log_moneyness)} points) [Heston disabled]")
        ssvi_result = ssvi_fitter.fit(log_moneyness, market_iv, ttm)
        heston_result = HestonFitResult(
            success=False, params=None, r_squared=0.0,
            rmse=0.0, max_residual=0.0, n_points=0,
            message="disabled by config"
        )
    else:
        logger.info(f"Fitting both models for {expiry} ({len(log_moneyness)} points)")
        heston_fitter = create_heston_fitter(config)
        with ThreadPoolExecutor(max_workers=2) as pool:
            heston_future = pool.submit(heston_fitter.fit, log_moneyness, market_iv, ttm, forward=forward)
            ssvi_future = pool.submit(ssvi_fitter.fit, log_moneyness, market_iv, ttm)
            heston_result = heston_future.result()
            ssvi_result = ssvi_future.result()

    # Get R² values (default to -inf if fit failed)
    heston_r2 = float('-inf')
    heston_valid = False
    if heston_result.success and heston_result.params is not None:
        # Check Heston params
        heston_check = checker.check_heston_params(
            heston_result.params,
            heston_result.r_squared,
            heston_result.max_residual,
            expiry
        )
        if not heston_check.has_critical():
            # Check IV consistency
            from btc_pricer.models.heston import HestonModel
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
                heston_r2 = heston_result.r_squared
                heston_valid = True

    ssvi_r2 = float('-inf')
    ssvi_valid = False
    if ssvi_result.success and ssvi_result.params is not None:
        # Check SSVI params
        ssvi_check = checker.check_ssvi_params(
            ssvi_result.params,
            ssvi_result.r_squared,
            ssvi_result.max_residual,
            expiry
        )
        if not ssvi_check.has_critical():
            ssvi_r2 = ssvi_result.r_squared
            ssvi_valid = True

    # Model selection: prefer Heston unless SSVI is significantly better
    threshold = config.model.ssvi_preference_threshold
    use_ssvi = ssvi_valid and (ssvi_r2 >= heston_r2 + threshold or not heston_valid)

    # Determine selection reason
    if not heston_valid and not ssvi_valid:
        logger.error(f"Both Heston and SSVI failed for {expiry}")
        return None
    elif not heston_valid:
        selection_reason = "heston_failed"
    elif not ssvi_valid:
        selection_reason = "ssvi_failed"
    elif use_ssvi:
        selection_reason = "ssvi_significantly_better"
    else:
        selection_reason = "heston_preferred"

    selected_model = "ssvi" if use_ssvi else "heston"

    # Log selection
    heston_r2_str = f"{heston_r2:.4f}" if heston_valid else "failed"
    ssvi_r2_str = f"{ssvi_r2:.4f}" if ssvi_valid else "failed"
    logger.info(
        f"{expiry}: Heston R²={heston_r2_str}, SSVI R²={ssvi_r2_str} → "
        f"Using {selected_model.upper()} ({selection_reason})"
    )

    # Extract RND using selected model
    if selected_model == "heston":
        params = heston_result.params
        logger.info(
            f"Heston params for {expiry}: v0={params.v0:.4f}, "
            f"kappa={params.kappa:.3f}, theta={params.theta:.4f}, "
            f"xi={params.xi:.3f}, rho={params.rho:.3f}"
        )
        rnd = bl.extract_from_heston(params, forward)
    else:
        params = ssvi_result.params
        logger.info(
            f"SSVI params for {expiry}: theta={params.theta:.4f}, "
            f"rho={params.rho:.3f}, phi={params.phi:.3f}"
        )
        rnd = bl.extract_from_ssvi(params, forward)

    # Check RND
    rnd_check = checker.check_rnd(rnd, expiry)

    if rnd_check.has_critical():
        logger.error(f"Critical RND issue for {expiry}")
        return None

    if rnd.warnings:
        for warning in rnd.warnings:
            logger.warning(f"RND {expiry}: {warning}")

    # Return extra info for tracking
    extra_info = {
        "heston_r_squared": heston_r2 if heston_valid else None,
        "ssvi_r_squared": ssvi_r2 if ssvi_valid else None,
        "selection_reason": selection_reason,
    }

    return params, rnd, log_moneyness, market_iv, selected_model, extra_info


def _process_expiry_single_model(
    expiry: str,
    log_moneyness: np.ndarray,
    market_iv: np.ndarray,
    ttm: float,
    forward: float,
    config: Config,
    checker: SanityChecker,
    bl: BreedenLitzenberger,
    model: str,
    logger: logging.Logger
) -> Optional[tuple]:
    """Process expiry with a single specified model (for model_override case).

    Args:
        expiry: Expiry date string.
        log_moneyness: Log moneyness values.
        market_iv: Market implied volatilities.
        ttm: Time to maturity.
        forward: Forward price.
        config: Configuration.
        checker: Sanity checker.
        bl: Breeden-Litzenberger extractor.
        model: Model to use ("heston" or "ssvi").
        logger: Logger instance.

    Returns:
        Tuple of (params, RNDResult, log_moneyness, market_iv, model_used, extra_info) or None.
    """
    if model == "heston":
        logger.info(f"Fitting Heston for {expiry} (override)")
        heston_fitter = create_heston_fitter(config)
        result = heston_fitter.fit(log_moneyness, market_iv, ttm, forward=forward)

        if not result.success or result.params is None:
            logger.warning(f"Heston fit failed for {expiry}: {result.message}")
            return None

        # Check Heston params
        heston_check = checker.check_heston_params(
            result.params, result.r_squared, result.max_residual, expiry
        )
        if heston_check.has_critical():
            logger.error(f"Critical Heston issue for {expiry}")
            return None

        # Check IV consistency
        from btc_pricer.models.heston import HestonModel
        heston_model = HestonModel(result.params, use_quantlib=config.heston.use_quantlib)
        is_consistent, max_iv_error, _ = check_iv_consistency(
            heston_model, log_moneyness, market_iv,
            config.model.iv_consistency_threshold,
            ttm=ttm,
            relaxation=getattr(config.model, 'iv_consistency_relaxation', 0.15),
            ttm_cutoff=getattr(config.model, 'iv_consistency_ttm_cutoff', 0.05),
        )
        if not is_consistent:
            logger.error(
                f"Heston IV consistency check failed for {expiry}: "
                f"max error {max_iv_error:.2%}"
            )
            return None

        logger.info(
            f"Heston fit for {expiry}: v0={result.params.v0:.4f}, "
            f"kappa={result.params.kappa:.3f}, theta={result.params.theta:.4f}, "
            f"xi={result.params.xi:.3f}, rho={result.params.rho:.3f}, "
            f"R²={result.r_squared:.4f}, IV_err={max_iv_error:.2%}"
        )
        rnd = bl.extract_from_heston(result.params, forward)
        params = result.params
        r2 = result.r_squared
    else:
        logger.info(f"Fitting SSVI for {expiry} (override)")
        ssvi_fitter = create_ssvi_fitter(config)
        result = ssvi_fitter.fit(log_moneyness, market_iv, ttm)

        if not result.success or result.params is None:
            logger.warning(f"SSVI fit failed for {expiry}: {result.message}")
            return None

        # Check SSVI params
        ssvi_check = checker.check_ssvi_params(
            result.params, result.r_squared, result.max_residual, expiry
        )
        if ssvi_check.has_critical():
            logger.error(f"Critical SSVI issue for {expiry}")
            return None

        logger.info(
            f"SSVI fit for {expiry}: theta={result.params.theta:.4f}, "
            f"rho={result.params.rho:.3f}, phi={result.params.phi:.3f}, "
            f"R²={result.r_squared:.4f}"
        )
        rnd = bl.extract_from_ssvi(result.params, forward)
        params = result.params
        r2 = result.r_squared

    # Check RND
    rnd_check = checker.check_rnd(rnd, expiry)
    if rnd_check.has_critical():
        logger.error(f"Critical RND issue for {expiry}")
        return None

    if rnd.warnings:
        for warning in rnd.warnings:
            logger.warning(f"RND {expiry}: {warning}")

    extra_info = {
        "heston_r_squared": r2 if model == "heston" else None,
        "ssvi_r_squared": r2 if model == "ssvi" else None,
        "selection_reason": f"{model}_override",
    }

    return params, rnd, log_moneyness, market_iv, model, extra_info


@handle_cli_exceptions
def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="BTC Price Forecaster - Extract Risk-Neutral Densities from Deribit options"
    )
    add_common_arguments(parser)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results"),
        help="Output directory for results"
    )
    parser.add_argument(
        "--expiries",
        nargs="+",
        help="Specific expiries to process (e.g., 27DEC24 28MAR25)"
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip generating plots"
    )
    parser.add_argument(
        "--no-sanity-checks",
        action="store_true",
        help="Skip sanity checks (not recommended)"
    )
    parser.add_argument(
        "--model",
        choices=["heston", "ssvi"],
        default=None,
        help="Model to use for IV fitting (default: from config, typically heston)"
    )
    args = parser.parse_args()

    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

    # Load configuration
    config = load_config(args.config, logger)

    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)

    # Initialize components
    client = DeribitClient(config.api, config.validation)
    data_filter = DataFilter(config.filters)
    checker = SanityChecker(strict=False, validation_config=config.validation)
    stats_calc = RNDStatistics()
    plotter = RNDPlotter(dpi=config.output.dpi, heston_config=config.heston)

    # Fetch data
    logger.info("Fetching options data from Deribit...")
    options_by_expiry = client.fetch_all_options("BTC")

    if not options_by_expiry:
        logger.error("No options data received")
        sys.exit(1)

    deribit_spot = list(options_by_expiry.values())[0][0].spot_price
    logger.info(f"Deribit spot price: ${deribit_spot:,.0f}")

    # Use Binance spot price by default, fallback to Deribit
    spot_price, _ = fetch_spot_with_fallback(deribit_spot)

    logger.info(f"Found {len(options_by_expiry)} expiries")

    # Filter expiries if specified
    if args.expiries:
        options_by_expiry = {
            k: v for k, v in options_by_expiry.items()
            if k in args.expiries
        }
        logger.info(f"Processing {len(options_by_expiry)} selected expiries")

    # Process each expiry
    results: Dict[str, dict] = {}
    rnds: Dict[str, RNDResult] = {}
    model_params: Dict[str, dict] = {}  # Store params for any model type
    ssvi_params: Dict[str, SSVIParams] = {}  # Keep for backwards compatibility
    all_stats: List[ExpiryStats] = []
    models_used: Dict[str, str] = {}

    # Pre-filter expiries sequentially (fast, <100ms per expiry)
    sorted_expiries = sorted(options_by_expiry.keys(), key=parse_expiry_date)
    expiry_filtered = {}
    for expiry in sorted_expiries:
        options = options_by_expiry[expiry]
        filtered, filter_stats = data_filter.filter_options(options, return_stats=True)
        if filter_stats:
            logger.debug(
                f"Filtered {expiry}: {filter_stats.passed_filters}/{filter_stats.total_options} passed"
            )
        expiry_filtered[expiry] = filtered

    # Process expiries in parallel using threads
    # GIL-released Rust Heston calibrations run truly concurrently;
    # Python-bound work (SSVI, BL, sanity checks) is serialized by the GIL
    # which also protects non-thread-safe QuantLib SWIG wrappers.
    max_workers = min(len(expiry_filtered), os.cpu_count() or 4)
    logger.info(f"Processing {len(expiry_filtered)} expiries with {max_workers} workers")

    # Submit all expiry processing tasks
    expiry_results = {}
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(process_expiry, expiry, filtered, config, checker, args.model): expiry
            for expiry, filtered in expiry_filtered.items()
        }
        for future in as_completed(futures):
            expiry = futures[future]
            try:
                expiry_results[expiry] = future.result()
            except Exception:
                logger.exception(f"Exception processing {expiry}")
                expiry_results[expiry] = None

    # SSVI Surface refinement (opt-in post-processing)
    surface_result_info = None
    if config.ssvi_surface.enabled:
        logger.info("SSVI surface fit enabled, gathering slices...")
        surface_slices = []
        surface_expiry_data = {}  # expiry -> (log_k, mkt_iv, forward, ttm)

        for expiry in sorted_expiries:
            result = expiry_results.get(expiry)
            if result is None:
                continue
            params, rnd, log_k, mkt_iv, model_used, extra_info = result
            ttm_days = params.ttm * 365 if hasattr(params, 'ttm') else rnd.ttm * 365
            if ttm_days <= config.ssvi_surface.max_ttm_days:
                ttm = params.ttm if hasattr(params, 'ttm') else rnd.ttm
                forward = rnd.forward if hasattr(rnd, 'forward') else 0.0
                surface_slices.append(SSVISliceData(
                    expiry_name=expiry,
                    ttm=ttm,
                    log_moneyness=log_k,
                    market_iv=mkt_iv,
                    forward=forward,
                ))
                surface_expiry_data[expiry] = (log_k, mkt_iv)

        if len(surface_slices) >= config.ssvi_surface.min_expiries:
            logger.info(
                f"Fitting SSVI surface across {len(surface_slices)} slices "
                f"(TTM ≤ {config.ssvi_surface.max_ttm_days}d): "
                f"{[s.expiry_name for s in surface_slices]}"
            )
            surface_fitter = create_ssvi_surface_fitter(config)
            surface_result = surface_fitter.fit(surface_slices)
            surface_result_info = surface_result

            if surface_result.success and surface_result.params is not None:
                logger.info(
                    f"SSVI surface fit successful (R²={surface_result.aggregate_r_squared:.4f}), "
                    f"replacing per-slice SSVI params for affected expiries"
                )
                # Replace per-slice params with surface-consistent params
                for sp in surface_result.per_slice_params:
                    expiry_name = None
                    for sl in surface_slices:
                        if abs(sl.ttm - sp.ttm) < 1e-10:
                            expiry_name = sl.expiry_name
                            break
                    if expiry_name and expiry_name in surface_expiry_data:
                        ssvi_params[expiry_name] = sp
                        # Re-extract BL RND for this expiry
                        log_k_s, mkt_iv_s = surface_expiry_data[expiry_name]
                        bl_s = BreedenLitzenberger(
                            strike_grid_points=config.breeden_litzenberger.strike_grid_points,
                            strike_range_std=config.breeden_litzenberger.strike_range_std,
                            use_log_strikes=config.breeden_litzenberger.use_log_strikes,
                        )
                        old_result = expiry_results.get(expiry_name)
                        if old_result:
                            old_params, old_rnd, _, _, old_model, old_extra = old_result
                            forward_price = old_rnd.forward if hasattr(old_rnd, 'forward') else sp.theta  # fallback
                            new_rnd = bl_s.extract_from_ssvi(sp, forward_price)
                            # Update the expiry result with surface-refined params
                            expiry_results[expiry_name] = (
                                sp, new_rnd, log_k_s, mkt_iv_s, "ssvi_surface", {
                                    "heston_r_squared": old_extra.get("heston_r_squared"),
                                    "ssvi_r_squared": surface_result.per_slice_r_squared[
                                        [s.expiry_name for s in surface_slices].index(expiry_name)
                                    ],
                                    "selection_reason": "ssvi_surface",
                                }
                            )
            else:
                logger.warning(f"SSVI surface fit failed: {surface_result.message}")
        else:
            logger.info(
                f"Not enough slices for surface fit ({len(surface_slices)} < "
                f"{config.ssvi_surface.min_expiries})"
            )

    # Collect results in sorted order for deterministic output
    for expiry in sorted_expiries:
        result = expiry_results.get(expiry)

        if result is None:
            logger.warning(f"Skipping {expiry} due to processing failure")
            continue

        params, rnd, log_k, mkt_iv, model_used, extra_info = result

        # Store results
        model_params[expiry] = params.to_dict()
        models_used[expiry] = model_used
        rnds[expiry] = rnd

        # Keep ssvi_params for backwards compatibility with plotting
        if model_used in ("ssvi", "ssvi_surface"):
            ssvi_params[expiry] = params

        # Compute statistics
        expiry_stats = stats_calc.compute_stats(rnd, expiry, spot_price)
        all_stats.append(expiry_stats)

        # Store for JSON output
        results[expiry] = {
            "model_type": model_used,
            "model_params": params.to_dict(),
            "rnd_stats": rnd.to_dict(),
            "statistics": stats_calc.to_json(expiry_stats),
            "heston_r_squared": extra_info.get("heston_r_squared"),
            "ssvi_r_squared": extra_info.get("ssvi_r_squared"),
            "selection_reason": extra_info.get("selection_reason"),
        }

        # Print summary
        print(f"[{model_used.upper()}] ", end="")
        print(stats_calc.format_summary(expiry_stats))
        print()

        # Generate plots if requested
        if not args.no_plots and config.output.save_plots:
            # IV smile (include model name in filename)
            plotter.plot_iv_smile(
                log_k, mkt_iv, params, expiry,
                save_path=args.output / f"iv_smile_{model_used}_{expiry}.{config.output.plot_format}"
            )

            # Single density (include model name in filename)
            plotter.plot_single_density(
                rnd, expiry, spot_price,
                save_path=args.output / f"density_{model_used}_{expiry}.{config.output.plot_format}"
            )

    # Print comparison table
    if all_stats:
        print("\n" + stats_calc.format_table(all_stats))

    # Generate multi-expiry plots
    if not args.no_plots and config.output.save_plots and len(rnds) > 1:
        plotter.plot_multiple_densities(
            rnds, spot_price,
            save_path=args.output / f"densities_all.{config.output.plot_format}"
        )

        plotter.plot_fan_chart(
            rnds, spot_price,
            save_path=args.output / f"fan_chart.{config.output.plot_format}"
        )

        plotter.plot_term_structure(
            ssvi_params,
            save_path=args.output / f"term_structure.{config.output.plot_format}"
        )

    # Save JSON results
    if config.output.save_json:
        # Main results
        with open(args.output / "rnd_results.json", "w") as f:
            json.dump(results, f, indent=2, default=str)

        # SSVI parameters
        ssvi_json = {k: v.to_dict() for k, v in ssvi_params.items()}
        with open(args.output / "ssvi_params.json", "w") as f:
            json.dump(ssvi_json, f, indent=2)

        # Model parameters
        with open(args.output / "model_params.json", "w") as f:
            json.dump(model_params, f, indent=2)

        # Summary
        summary = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "spot_price": spot_price,
            "default_model": config.model.default_model,
            "models_used": models_used,
            "expiries_processed": list(rnds.keys()),
            "expiries_skipped": [
                k for k in options_by_expiry.keys() if k not in rnds
            ],
        }
        with open(args.output / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        logger.info(f"\nResults saved to {args.output}")

    # Print sanity check summary
    if not args.no_sanity_checks:
        checker.print_summary()

    # Close plots
    plotter.close_all()


if __name__ == "__main__":
    main()
