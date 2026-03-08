#!/usr/bin/env python3
"""
BTC Terminal Probability Calculator (consolidated CLI)

Modes:
  terminal (default) — P(S_T >= K at expiry) using MC + B-L
  barrier            — P(S_t >= K at any t <= T) using MC first-passage
  --bl-only          — B-L extraction only, skip MC and surface fitting (~5s)
  --intraday         — ATM IV sqrt(T) scaling forecasts for short horizons
"""

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from btc_pricer.config import Config
from btc_pricer.api.deribit import DeribitClient, DeribitAPIError
from btc_pricer.api.binance import fetch_spot_with_fallback
from btc_pricer.models.heston import HestonParams
from btc_pricer.models.terminal_probability import TerminalProbabilityCalculator, MCResult
from btc_pricer.models.breeden_litzenberger import BreedenLitzenberger
from btc_pricer.cli.common import (
    setup_logging,
    load_config,
    handle_cli_exceptions,
    add_common_arguments,
    find_closest_expiry_after,
    calibrate_to_expiry,
    extract_surface_data,
    format_current_time_multizone,
    parse_expiry_date,
    fit_ssvi_surface_for_ttm,
    get_atm_iv_from_nearest_expiry,
)
from btc_pricer.utils.time_parser import (
    parse_datetime_with_timezone,
    calculate_ttm_to_target,
    format_target_time,
    TimeParseError,
)


def decompose_probs(result: MCResult) -> Tuple[float, float]:
    """Return (prob_above, prob_below) regardless of direction."""
    if result.direction == "down":
        return 1 - result.terminal_probability, result.terminal_probability
    return result.terminal_probability, 1 - result.terminal_probability


def format_terminal_result(
    result: MCResult,
    params,
    r_squared: float,
    model_type: str = "heston",
    expiry_str: Optional[str] = None,
    target_utc: Optional[datetime] = None,
    target_tz_display: Optional[str] = None
) -> str:
    """Format terminal probability result for display."""
    ci_low, ci_high = result.confidence_interval
    ci_width = (ci_high - ci_low) / 2 * 100

    model_display = model_type.upper()
    if model_type == "ssvi":
        model_display = "SSVI Local Vol MC"

    # Format target time if provided (--until was used)
    target_display = ""
    if target_utc is not None:
        target_str = format_target_time(target_utc, target_tz_display or "UTC")
        target_display = f" Target Time:       {target_str}"

    # Format calibration expiry time
    expiry_display = ""
    if expiry_str:
        try:
            expiry_dt = datetime.strptime(expiry_str, "%d%b%y").replace(hour=8)  # 08:00 UTC
            label = "Calibration Expiry:" if target_utc else "Expiry:            "
            expiry_display = f" {label} {expiry_dt.strftime('%d %b %Y')} 08:00 UTC"
        except ValueError:
            label = "Calibration Expiry:" if target_utc else "Expiry:            "
            expiry_display = f" {label} {expiry_str}"

    # Current time in multiple timezones
    current_time_str = format_current_time_multizone()

    # Calculate terminal probabilities for both directions
    term_prob_above, term_prob_below = decompose_probs(result)

    lines = [
        "",
        "=" * 65,
        "          TERMINAL PROBABILITY ANALYSIS",
        "=" * 65,
        f" Current Time:      {current_time_str}",
        f" Spot Price:        ${result.spot:,.2f} (Binance)",
        f" Target Price:      ${result.target_price:,.0f} ({result.direction.upper()}, {result.target_distance_pct:+.1f}%)",
        f" Time Horizon:      {result.ttm * 365:.3f} days ({result.ttm * 365 * 24:.2f} hours)",
    ]
    if target_display:
        lines.append(target_display)
    if expiry_display:
        lines.append(expiry_display)
    lines.extend([
        "-" * 65,
        f" P(>${result.target_price:,.0f}): {term_prob_above * 100:.1f}% +/- {ci_width:.1f}%",
        f" P(<${result.target_price:,.0f}): {term_prob_below * 100:.1f}%",
        "-" * 65,
        f" Model: {model_display} (R²={r_squared:.3f})",
    ])

    if model_type == "heston":
        lines.append(
            f" Parameters: v0={params.v0:.3f}, kappa={params.kappa:.2f}, "
            f"theta={params.theta:.3f}, xi={params.xi:.2f}, rho={params.rho:.2f}"
        )
    else:  # ssvi
        lines.append(
            f" Parameters: theta={params.theta:.4f}, rho={params.rho:.3f}, phi={params.phi:.3f}"
        )

    lines.extend([
        f" Simulations: {result.n_simulations:,}",
        "=" * 65,
        "",
    ])
    return "\n".join(lines)


def format_comparison_result(
    heston_result: Optional[MCResult],
    ssvi_result: Optional[MCResult],
    heston_params,
    ssvi_params,
    heston_r2: Optional[float],
    ssvi_r2: Optional[float],
    spot: float,
    target_price: float,
    direction: str,
    ttm: float,
    expiry_str: Optional[str] = None,
    target_utc: Optional[datetime] = None,
    target_tz_display: Optional[str] = None,
    surface_bl_above: Optional[float] = None,
    surface_bl_below: Optional[float] = None,
    surface_r2: Optional[float] = None,
    surface_info: Optional[str] = None,
    ssvi_is_surface: bool = False,
) -> str:
    """Format comparison of Heston, SSVI, and optionally SSVI Surface results."""
    target_dist = (target_price - spot) / spot * 100

    # Format target time if provided
    target_display = ""
    if target_utc is not None:
        target_str = format_target_time(target_utc, target_tz_display or "UTC")
        target_display = f" Target Time:       {target_str}"

    # Format calibration expiry
    expiry_display = ""
    if expiry_str:
        try:
            expiry_dt = datetime.strptime(expiry_str, "%d%b%y").replace(hour=8)
            label = "Calibration Expiry:" if target_utc else "Expiry:            "
            expiry_display = f" {label} {expiry_dt.strftime('%d %b %Y')} 08:00 UTC"
        except ValueError:
            label = "Calibration Expiry:" if target_utc else "Expiry:            "
            expiry_display = f" {label} {expiry_str}"

    # Current time in multiple timezones
    current_time_str = format_current_time_multizone()

    lines = [
        "",
        "=" * 70,
        "               TERMINAL PROBABILITY COMPARISON",
        "=" * 70,
        f" Current Time:      {current_time_str}",
        f" Spot Price:        ${spot:,.2f} (Binance)",
        f" Target Price:      ${target_price:,.0f} ({direction.upper()}, {target_dist:+.1f}%)",
        f" Time Horizon:      {ttm * 365:.3f} days ({ttm * 365 * 24:.2f} hours)",
    ]

    if target_display:
        lines.append(target_display)
    if expiry_display:
        lines.append(expiry_display)

    lines.append("-" * 70)
    lines.append(f" {'Model':<20} {'R²':>8} {'P(>target)':>16} {'P(<target)':>16}")
    lines.append("-" * 70)

    # Heston row
    if heston_result is not None:
        ci_low, ci_high = heston_result.confidence_interval
        ci_width = (ci_high - ci_low) / 2 * 100
        p_above, p_below = decompose_probs(heston_result)
        above_str = f"{p_above*100:.1f}% ±{ci_width:.1f}%"
        lines.append(f" {'Heston MC':<20} {heston_r2:>7.3f} {above_str:>16} {p_below*100:>15.1f}%")
    else:
        lines.append(f" {'Heston MC':<20} {'failed':>8} {'-':>16} {'-':>16}")

    # SSVI row (MC uses surface params when available)
    ssvi_mc_label = "SSVI Surface MC" if ssvi_is_surface else "SSVI Local Vol MC"
    if ssvi_result is not None:
        ci_low, ci_high = ssvi_result.confidence_interval
        ci_width = (ci_high - ci_low) / 2 * 100
        p_above, p_below = decompose_probs(ssvi_result)
        above_str = f"{p_above*100:.1f}% ±{ci_width:.1f}%"
        lines.append(f" {ssvi_mc_label:<20} {ssvi_r2:>7.3f} {above_str:>16} {p_below*100:>15.1f}%")
    else:
        lines.append(f" {ssvi_mc_label:<20} {'failed':>8} {'-':>16} {'-':>16}")

    # SSVI Surface B-L row
    if surface_bl_above is not None:
        above_str = f"{surface_bl_above*100:.1f}%"
        lines.append(f" {'SSVI Surface B-L':<20} {surface_r2:>7.3f} {above_str:>16} {surface_bl_below*100:>15.1f}%")

    lines.append("-" * 70)

    # Show parameters for each model
    if heston_params is not None:
        lines.append(
            f" Heston: v0={heston_params.v0:.3f}, kappa={heston_params.kappa:.2f}, "
            f"theta={heston_params.theta:.3f}, xi={heston_params.xi:.2f}, rho={heston_params.rho:.2f}"
        )
    if ssvi_params is not None:
        lines.append(
            f" SSVI:   theta={ssvi_params.theta:.4f}, rho={ssvi_params.rho:.3f}, phi={ssvi_params.phi:.3f}"
        )
    if surface_info:
        lines.append(f" {surface_info}")

    n_sims = heston_result.n_simulations if heston_result else (ssvi_result.n_simulations if ssvi_result else 0)
    lines.append(f" Simulations: {n_sims:,}")
    lines.append("=" * 70)
    lines.append("")

    return "\n".join(lines)


def build_json_output(
    heston_r2: Optional[float],
    spot: float,
    target_price: float,
    direction: str,
    ttm: float,
    heston_params=None,
    bl_prob_above: Optional[float] = None,
    bl_prob_below: Optional[float] = None,
    surface_bl_above: Optional[float] = None,
    surface_bl_below: Optional[float] = None,
    surface_mc_above: Optional[float] = None,
    surface_mc_below: Optional[float] = None,
    surface_r2: Optional[float] = None,
    surface_params_at_ttm=None,
    surface_params_obj=None,
    timing: Optional[dict] = None,
    trading_model: str = "ssvi_surface",
) -> dict:
    """Build JSON output structure for terminal probability results."""
    output = {
        "spot_price": spot,
        "target_price": target_price,
        "direction": direction,
        "ttm_days": ttm * 365,
        "heston": None,
        "preferred_model": None,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    # Add Heston results (B-L only -- MC is skipped)
    if heston_params is not None:
        output["heston"] = {
            "prob_above": bl_prob_above,
            "prob_below": bl_prob_below,
            "r_squared": heston_r2,
            "params": heston_params.to_dict() if heston_params else None,
        }

    # Add SSVI Surface data
    if surface_bl_above is not None:
        output["ssvi_surface"] = {
            "prob_above": surface_bl_above,
            "prob_below": surface_bl_below,
            "r_squared": surface_r2,
            "params": surface_params_at_ttm.to_dict() if surface_params_at_ttm else None,
            "surface_params": surface_params_obj.to_dict() if surface_params_obj else None,
        }
        if surface_mc_above is not None:
            output["ssvi_surface"]["mc_prob_above"] = surface_mc_above
            output["ssvi_surface"]["mc_prob_below"] = surface_mc_below

    # Include timing data for downstream consumers
    output["timing"] = timing or {}

    # Determine preferred model from config (trading_model).
    # Both models are always calibrated; the non-trading model is saved for
    # backtesting comparison.
    if trading_model == "heston" and bl_prob_above is not None:
        output["preferred_model"] = "heston"
    elif surface_bl_above is not None:
        output["preferred_model"] = "ssvi_surface"

    # Add Breeden-Litzenberger terminal probabilities (Heston B-L)
    output["bl_prob_above"] = bl_prob_above
    output["bl_prob_below"] = bl_prob_below

    # avg_prob logic — uses the preferred (trading) model only
    preferred = output["preferred_model"]
    if preferred == "ssvi_surface":
        if surface_mc_above is not None and surface_bl_above is not None:
            output["avg_prob_above"] = (surface_mc_above + surface_bl_above) / 2
            output["avg_prob_below"] = (surface_mc_below + surface_bl_below) / 2
        elif surface_bl_above is not None:
            output["avg_prob_above"] = surface_bl_above
            output["avg_prob_below"] = surface_bl_below
        else:
            output["avg_prob_above"] = surface_mc_above
            output["avg_prob_below"] = surface_mc_below
    elif preferred == "heston":
        # Heston B-L only (MC skipped for Heston due to Feller violations)
        output["avg_prob_above"] = bl_prob_above
        output["avg_prob_below"] = bl_prob_below
    else:
        output["avg_prob_above"] = None
        output["avg_prob_below"] = None

    # Track divergence for diagnostics
    if surface_mc_above is not None and surface_bl_above is not None:
        output["bl_mc_divergence"] = abs(surface_mc_above - surface_bl_above)
    else:
        output["bl_mc_divergence"] = None

    return output


# =========================================================================
# Intraday mode
# =========================================================================

def run_intraday(args, config, logger):
    """Run intraday forecast mode."""
    from btc_pricer.models.intraday_forecast import (
        IntradayForecaster,
        format_intraday_forecast,
        format_intraday_table,
    )

    client = DeribitClient(config.api, config.validation)

    # Get ATM IV (from market or override)
    if args.atm_iv is not None and args.spot is not None:
        atm_iv = args.atm_iv
        spot_price = args.spot
        source_expiry = "manual_override"
        logger.info(f"Using manual ATM IV: {atm_iv*100:.1f}%")
        logger.info(f"Using manual spot: ${spot_price:,.0f}")
    else:
        atm_iv, deribit_spot, source_expiry, _ = get_atm_iv_from_nearest_expiry(
            client, config
        )
        spot_price, _ = fetch_spot_with_fallback(deribit_spot)

        if args.atm_iv is not None:
            atm_iv = args.atm_iv
            logger.info(f"Overriding ATM IV to: {atm_iv*100:.1f}%")
        if args.spot is not None:
            spot_price = args.spot
            logger.info(f"Overriding spot to: ${spot_price:,.0f}")

    forecaster = IntradayForecaster(
        use_drift=config.intraday.use_drift,
        annual_drift=config.intraday.annual_drift
    )

    if args.hours:
        series = forecaster.forecast_series(
            spot_price, atm_iv, args.hours, source_expiry
        )
    else:
        series = forecaster.forecast_series(
            spot_price, atm_iv, config.intraday.standard_horizons, source_expiry
        )

    print("\n" + format_intraday_table(series))

    print("\n" + "=" * 60)
    print("DETAILED FORECASTS")
    print("=" * 60)
    for forecast in series.forecasts[:3]:
        print("\n" + format_intraday_forecast(forecast))

    if args.output:
        output_data = {
            "timestamp": datetime.now().isoformat(),
            "spot_price": spot_price,
            "atm_iv_annual": atm_iv,
            "atm_iv_annual_pct": atm_iv * 100,
            "source_expiry": source_expiry,
            "forecasts": series.to_dict()["forecasts"]
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        logger.info(f"Results saved to {args.output}")


# =========================================================================
# BL-only mode
# =========================================================================

def run_bl_only(args, config, logger):
    """Run BL-only fast mode (no MC, no surface fitting)."""
    client = DeribitClient(config.api, config.validation)

    targets = args.targets if args.targets else [args.target]

    # Calibrate single model (faster)
    params, spot_price, calibration_expiry, forward, r_squared, ttm, model_type = calibrate_to_expiry(
        client, config, args.expiry
    )

    if args.spot is not None:
        spot_price = args.spot
        logger.info(f"Overriding spot price to: ${spot_price:,.2f}")

    ttm_days = ttm * 365

    # Extract RND using Breeden-Litzenberger
    logger.info("Extracting Risk-Neutral Density...")
    bl = BreedenLitzenberger(
        strike_grid_points=500, strike_range_std=4.0, use_log_strikes=True
    )

    if model_type == "heston":
        rnd = bl.extract_from_heston(params, forward, use_quantlib=config.heston.use_quantlib)
    else:
        rnd = bl.extract_from_ssvi(params, forward)

    if not rnd.is_valid:
        logger.warning(f"RND extraction warnings: {rnd.warnings}")

    # Calculate probabilities
    results = []
    for target in targets:
        prob_above = bl.probability_above(rnd, target)
        prob_below = bl.probability_below(rnd, target)
        distance_pct = (target - spot_price) / spot_price * 100
        results.append({
            "target_price": target,
            "distance_pct": distance_pct,
            "prob_above": prob_above,
            "prob_below": prob_below,
        })

    # Display
    model_display = model_type.upper()

    lines = [
        "",
        "=" * 75,
        "                    TERMINAL PROBABILITY (B-L ONLY)",
        "=" * 75,
        f" Spot Price: ${spot_price:,.2f}",
        f" Expiry: {calibration_expiry} ({ttm_days:.1f} days) | Model: {model_display} (R²={r_squared:.3f})",
        "-" * 75,
        f" {'Target Price':>14} {'Distance':>10} {'P(Above)':>12} {'P(Below)':>12}",
        "-" * 75,
    ]

    for r in sorted(results, key=lambda x: x["target_price"]):
        lines.append(
            f" ${r['target_price']:>13,.2f} {r['distance_pct']:>+9.1f}% "
            f"{r['prob_above']*100:>11.1f}% {r['prob_below']*100:>11.1f}%"
        )
    lines.extend(["-" * 75, ""])
    print("\n".join(lines))

    # Save to JSON if requested
    if args.output:
        output_data = {
            "timestamp": datetime.now().isoformat(),
            "spot_price": spot_price,
            "forward_price": forward,
            "calibration_expiry": calibration_expiry,
            "ttm_days": ttm_days,
            "model_type": model_type,
            "model_params": params.to_dict(),
            "r_squared": r_squared,
            "rnd_statistics": rnd.to_dict(),
            "probabilities": results,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2, default=str)
        logger.info(f"Results saved to {args.output}")


# =========================================================================
# Barrier mode
# =========================================================================

def run_barrier(args, config, logger):
    """Run barrier (first-passage) probability mode."""
    from btc_pricer.models.barrier_probability import BarrierProbabilityCalculator

    import time as _time

    targets = args.targets if args.targets else [args.target]

    n_sims = args.sims
    steps_per_day = args.steps_per_day
    if hasattr(config, 'monte_carlo'):
        if n_sims is None:
            n_sims = getattr(config.monte_carlo, 'n_simulations', 200000)
        if steps_per_day is None:
            steps_per_day = getattr(config.monte_carlo, 'n_steps_per_day', 288)
    else:
        if n_sims is None:
            n_sims = 200000
        if steps_per_day is None:
            steps_per_day = 288

    client = DeribitClient(config.api, config.validation)

    # Calibrate
    _t0_cal = _time.time()
    heston_data, ssvi_data, spot_price, calibration_expiry, forward, ttm = calibrate_to_expiry(
        client, config, args.expiry, return_both=True
    )
    _calibration_elapsed = _time.time() - _t0_cal

    heston_params, heston_r2 = heston_data if heston_data else (None, None)

    if args.ttm is not None:
        ttm = args.ttm / 365.0
    if args.spot is not None:
        spot_price = args.spot

    # SSVI surface fit
    options_by_expiry = client.fetch_all_options("BTC")
    surface_result = fit_ssvi_surface_for_ttm(client, config, ttm, options_by_expiry)

    surface_params_at_ttm = surface_result.params_at_ttm if surface_result else None
    surface_params_obj = surface_result.surface_params if surface_result else None
    surface_r2 = surface_result.r2 if surface_result else None
    surface_info = surface_result.info if surface_result else None

    if heston_params is not None:
        heston_params = HestonParams(
            v0=heston_params.v0, kappa=heston_params.kappa,
            theta=heston_params.theta, xi=heston_params.xi,
            rho=heston_params.rho, ttm=ttm
        )

    calculator = BarrierProbabilityCalculator(
        n_simulations=n_sims,
        n_steps_per_day=steps_per_day,
        use_antithetic=True
    )

    for target in targets:
        barrier_type = "down" if target < spot_price else "up"

        # Run SSVI Surface barrier MC
        barrier_result = None
        if surface_params_at_ttm is not None:
            logger.info(f"Running barrier MC for ${target:,.0f}...")
            _t0 = _time.time()
            barrier_result = calculator.touch_probability_ssvi(
                surface_params_at_ttm, spot_price, forward, target, barrier_type
            )
            _mc_elapsed = _time.time() - _t0
            logger.info(f"Barrier MC completed in {_mc_elapsed:.1f}s")

        # B-L probabilities
        bl = BreedenLitzenberger(
            strike_grid_points=500, strike_range_std=4.0, use_log_strikes=True
        )
        bl_prob_above = None
        bl_prob_below = None
        surface_bl_above = None
        surface_bl_below = None

        if heston_params is not None:
            try:
                rnd = bl.extract_from_heston(heston_params, forward, use_quantlib=True)
                if rnd and rnd.is_valid:
                    bl_prob_above = float(bl.probability_above(rnd, target))
                    bl_prob_below = float(bl.probability_below(rnd, target))
            except Exception as e:
                logger.warning(f"Heston B-L failed: {e}")

        if surface_params_at_ttm is not None:
            try:
                rnd = bl.extract_from_ssvi(surface_params_at_ttm, forward)
                if rnd and rnd.is_valid:
                    surface_bl_above = float(bl.probability_above(rnd, target))
                    surface_bl_below = float(bl.probability_below(rnd, target))
            except Exception as e:
                logger.warning(f"SSVI Surface B-L failed: {e}")

        # Display barrier result
        barrier_dist = (target - spot_price) / spot_price * 100
        current_time_str = format_current_time_multizone()
        lines = [
            "",
            "=" * 70,
            "               BARRIER (FIRST-PASSAGE) PROBABILITY",
            "=" * 70,
            f" Current Time:      {current_time_str}",
            f" Spot Price:        ${spot_price:,.2f} (Binance)",
            f" Barrier:           ${target:,.0f} ({barrier_type.upper()}, {barrier_dist:+.1f}%)",
            f" Time Horizon:      {ttm * 365:.3f} days ({ttm * 365 * 24:.2f} hours)",
            f" Expiry:            {calibration_expiry}",
            "-" * 70,
        ]

        if barrier_result is not None:
            ci_low, ci_high = barrier_result.confidence_interval
            ci_width = (ci_high - ci_low) / 2 * 100
            lines.append(f" P(touch ${target:,.0f}): {barrier_result.touch_probability * 100:.1f}% +/- {ci_width:.1f}%")
            lines.append(f" P(terminal):        {barrier_result.terminal_probability * 100:.1f}%")
            lines.append(f" Touch/Terminal:      {barrier_result.touch_to_terminal_ratio:.2f}x")
        if surface_bl_above is not None:
            lines.append(f" P(terminal B-L):    >{surface_bl_above*100:.1f}%  <{surface_bl_below*100:.1f}%")

        if surface_info:
            lines.append(f" {surface_info}")
        lines.append(f" Simulations: {n_sims:,}")
        lines.append("=" * 70)
        lines.append("")
        print("\n".join(lines))

    # JSON output
    if args.output and targets:
        target = targets[-1]
        barrier_type = "down" if target < spot_price else "up"
        output = {
            "mode": "barrier",
            "spot_price": spot_price,
            "barrier": target,
            "barrier_type": barrier_type,
            "ttm_days": ttm * 365,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        if barrier_result is not None:
            output["touch_probability"] = barrier_result.touch_probability
            output["terminal_probability"] = barrier_result.terminal_probability
        if surface_bl_above is not None:
            output["surface_bl_above"] = surface_bl_above
            output["surface_bl_below"] = surface_bl_below
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(output, f, indent=2)
        logger.info(f"JSON output written to {args.output}")


# =========================================================================
# Terminal mode (default — existing behavior)
# =========================================================================

def run_terminal(args, config, logger):
    """Run terminal probability mode (default)."""
    import time as _time
    _calibration_elapsed = None
    _surface_elapsed = None
    _mc_elapsed = None
    _bl_elapsed = None
    _sbl_elapsed = None

    targets = args.targets if args.targets else [args.target]

    n_sims = args.sims
    steps_per_day = args.steps_per_day
    if hasattr(config, 'monte_carlo'):
        if n_sims is None:
            n_sims = getattr(config.monte_carlo, 'n_simulations', 200000)
        if steps_per_day is None:
            steps_per_day = getattr(config.monte_carlo, 'n_steps_per_day', 288)
    else:
        if n_sims is None:
            n_sims = 200000
        if steps_per_day is None:
            steps_per_day = 288

    client = DeribitClient(config.api, config.validation)

    target_utc = None
    target_tz_display = None
    options_by_expiry = None

    if args.until:
        try:
            target_utc, target_tz_display = parse_datetime_with_timezone(args.until)
            ttm_from_until = calculate_ttm_to_target(target_utc)
            logger.info(f"Target time: {target_utc.strftime('%Y-%m-%d %H:%M UTC')} ({args.until})")
            logger.info(f"TTM to target: {ttm_from_until * 365:.2f} days")
        except TimeParseError as e:
            raise SystemExit(str(e))

        logger.info("Fetching options data to find appropriate expiry...")
        options_by_expiry = client.fetch_all_options("BTC")
        if not options_by_expiry:
            raise DeribitAPIError("No options data received")

        auto_expiry = find_closest_expiry_after(options_by_expiry, target_utc)
        logger.info(f"Auto-selected expiry: {auto_expiry}")

        _t0_cal = _time.time()
        heston_data, ssvi_data, spot_price, calibration_expiry, forward, _ = calibrate_to_expiry(
            client, config, auto_expiry, return_both=True
        )
        _calibration_elapsed = _time.time() - _t0_cal
        ttm = ttm_from_until
    else:
        _t0_cal = _time.time()
        heston_data, ssvi_data, spot_price, calibration_expiry, forward, ttm = calibrate_to_expiry(
            client, config, args.expiry, return_both=True
        )
        _calibration_elapsed = _time.time() - _t0_cal

        if args.ttm is not None:
            ttm = args.ttm / 365.0
            logger.info(f"Using custom TTM: {args.ttm:.1f} days")

    heston_params, heston_r2 = heston_data if heston_data else (None, None)

    # SSVI Surface fit using extracted helper
    surface_result = fit_ssvi_surface_for_ttm(client, config, ttm, options_by_expiry)

    surface_params_at_ttm = None
    surface_params_obj = None
    surface_r2 = None
    surface_info = None

    if surface_result is not None:
        surface_params_at_ttm = surface_result.params_at_ttm
        surface_params_obj = surface_result.surface_params
        surface_r2 = surface_result.r2
        surface_info = surface_result.info
        _surface_elapsed = surface_result.elapsed

    # Update params with custom TTM
    if heston_params is not None:
        heston_params = HestonParams(
            v0=heston_params.v0, kappa=heston_params.kappa,
            theta=heston_params.theta, xi=heston_params.xi,
            rho=heston_params.rho, ttm=ttm
        )

    if args.spot is not None:
        spot_price = args.spot
        logger.info(f"Overriding spot price to: ${spot_price:,.0f}")

    # Auto-detect direction
    directions = {}
    for target in targets:
        detected_dir = "down" if target < spot_price else "up"
        if args.direction != detected_dir:
            logger.info(
                f"Auto-correcting direction for ${target:,.0f}: "
                f"'{args.direction}' -> '{detected_dir}' (target {'<' if target < spot_price else '>'} spot)"
            )
        directions[target] = detected_dir

    calculator = TerminalProbabilityCalculator(
        n_simulations=n_sims, n_steps_per_day=steps_per_day, use_antithetic=True
    )

    json_output = None

    logger.info("Calculating terminal probabilities with both models...")

    for target in targets:
        heston_result = None
        ssvi_result = None

        if surface_params_at_ttm is not None:
            logger.info(f"Running SSVI Surface Local Vol MC for target ${target:,.0f}...")
            _t0 = _time.time()
            ssvi_result = calculator.terminal_probability_ssvi(
                surface_params_at_ttm, spot_price, forward, target, directions[target]
            )
            _mc_elapsed = _time.time() - _t0
            logger.info(f"SSVI Surface MC completed in {_mc_elapsed:.1f}s")

        bl_prob_above = None
        bl_prob_below = None
        surface_bl_above = None
        surface_bl_below = None

        bl = BreedenLitzenberger(
            strike_grid_points=500, strike_range_std=4.0, use_log_strikes=True
        )

        if heston_params is not None:
            try:
                _t0 = _time.time()
                rnd = bl.extract_from_heston(heston_params, forward, use_quantlib=True)
                _bl_elapsed = _time.time() - _t0
                if rnd and rnd.is_valid:
                    bl_prob_above = float(bl.probability_above(rnd, target))
                    bl_prob_below = float(bl.probability_below(rnd, target))
                    logger.info(
                        f"Heston B-L in {_bl_elapsed:.1f}s: P(>{target:,.0f})={bl_prob_above:.4f}, "
                        f"P(<{target:,.0f})={bl_prob_below:.4f}"
                    )
                else:
                    logger.warning("Heston B-L RND extraction returned invalid result")
            except Exception as e:
                logger.warning(f"Heston B-L failed: {e}")

        if surface_params_at_ttm is not None:
            try:
                _t0 = _time.time()
                rnd = bl.extract_from_ssvi(surface_params_at_ttm, forward)
                _sbl_elapsed = _time.time() - _t0
                if rnd and rnd.is_valid:
                    surface_bl_above = float(bl.probability_above(rnd, target))
                    surface_bl_below = float(bl.probability_below(rnd, target))
                    logger.info(
                        f"SSVI Surface B-L in {_sbl_elapsed:.1f}s: P(>{target:,.0f})={surface_bl_above:.4f}, "
                        f"P(<{target:,.0f})={surface_bl_below:.4f}"
                    )
                else:
                    logger.warning("SSVI Surface B-L RND extraction returned invalid result")
            except Exception as e:
                logger.warning(f"SSVI Surface B-L failed: {e}")

        if heston_result is not None and ssvi_result is not None:
            print(format_comparison_result(
                heston_result, ssvi_result,
                heston_params, surface_params_at_ttm,
                heston_r2, surface_r2,
                spot_price, target, directions[target], ttm,
                calibration_expiry, target_utc, target_tz_display,
                surface_bl_above=surface_bl_above, surface_bl_below=surface_bl_below,
                surface_r2=surface_r2, surface_info=surface_info, ssvi_is_surface=True,
            ))
        elif heston_result is not None:
            print(format_terminal_result(
                heston_result, heston_params, heston_r2, "heston",
                calibration_expiry, target_utc, target_tz_display
            ))
        elif ssvi_result is not None:
            print(format_terminal_result(
                ssvi_result, surface_params_at_ttm, surface_r2, "ssvi",
                calibration_expiry, target_utc, target_tz_display
            ))

        s_mc_above, s_mc_below = None, None
        if surface_params_at_ttm is not None and ssvi_result is not None:
            s_mc_above, s_mc_below = decompose_probs(ssvi_result)

        if args.output:
            timing = {
                "calibration_s": _calibration_elapsed,
                "surface_fit_s": _surface_elapsed,
                "mc_s": _mc_elapsed,
                "bl_s": _bl_elapsed,
                "surface_bl_s": _sbl_elapsed,
            }
            json_output = build_json_output(
                heston_r2, spot_price, target, directions[target], ttm,
                heston_params=heston_params,
                bl_prob_above=bl_prob_above, bl_prob_below=bl_prob_below,
                surface_bl_above=surface_bl_above, surface_bl_below=surface_bl_below,
                surface_mc_above=s_mc_above, surface_mc_below=s_mc_below,
                surface_r2=surface_r2,
                surface_params_at_ttm=surface_params_at_ttm,
                surface_params_obj=surface_params_obj,
                timing=timing,
                trading_model=config.model.trading_model,
            )

    if args.output and json_output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(json_output, f, indent=2)
        logger.info(f"JSON output written to {args.output}")


# =========================================================================
# Main entry point
# =========================================================================

@handle_cli_exceptions
def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="BTC Probability Calculator — terminal, barrier, BL-only, or intraday modes"
    )
    add_common_arguments(parser)

    # Mode selection
    parser.add_argument(
        "--mode",
        choices=["terminal", "barrier"],
        default="terminal",
        help="Probability mode: terminal (default) or barrier (first-passage)"
    )
    parser.add_argument(
        "--bl-only",
        action="store_true",
        help="BL-only fast mode: skip MC and surface fitting (~5s)"
    )
    parser.add_argument(
        "--intraday",
        action="store_true",
        help="Intraday ATM IV sqrt(T) scaling forecasts"
    )

    # Target arguments
    parser.add_argument(
        "--target",
        type=float,
        help="Single target/barrier price level (e.g., 85000)"
    )
    parser.add_argument(
        "--targets",
        type=float,
        nargs="+",
        help="Multiple target/barrier price levels (e.g., 80000 85000 90000 95000)"
    )
    parser.add_argument(
        "--expiry",
        type=str,
        help="Target expiry date (e.g., 31JAN26). Uses nearest liquid expiry if not specified."
    )
    parser.add_argument(
        "--direction",
        choices=["down", "up"],
        default="down",
        help="Direction: 'down' for price dropping to target, 'up' for rising (default: down)"
    )
    parser.add_argument(
        "--sims",
        type=int,
        default=None,
        help="Number of Monte Carlo simulations (default: from config)"
    )
    parser.add_argument(
        "--steps-per-day",
        type=int,
        default=None,
        help="Time steps per day for simulation (default: from config)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON file path"
    )
    parser.add_argument(
        "--spot",
        type=float,
        default=None,
        help="Override spot price"
    )
    parser.add_argument(
        "--ttm",
        type=float,
        default=None,
        help="Override time to maturity in days (for custom horizons)"
    )
    parser.add_argument(
        "--until",
        type=str,
        default=None,
        help="Target end time with timezone (e.g., '11:59 PM ET', '2026-01-30 18:00 PST'). "
             "Auto-selects closest expiry after target."
    )

    # Intraday-specific arguments
    parser.add_argument(
        "--hours",
        type=float,
        nargs="+",
        default=None,
        help="Forecast horizons in hours for --intraday mode (e.g., 1 4 12 24)"
    )
    parser.add_argument(
        "--atm-iv",
        type=float,
        default=None,
        help="Override ATM IV for --intraday mode (as decimal, e.g., 0.65 for 65%%)"
    )

    args = parser.parse_args()

    # Validate mutual exclusivity
    exclusive_count = sum([args.intraday, args.bl_only, args.mode == "barrier"])
    if exclusive_count > 1:
        parser.error("--intraday, --bl-only, and --mode barrier are mutually exclusive")

    # Validate intraday doesn't mix with target-based args
    if args.intraday:
        if args.target or args.targets:
            parser.error("--intraday is mutually exclusive with --target/--targets")
    else:
        # All other modes require a target
        if not args.target and not args.targets:
            parser.error("Must specify either --target or --targets (or use --intraday)")

    # Validate --until exclusivity
    if args.until:
        if args.expiry:
            parser.error("--until and --expiry are mutually exclusive")
        if args.ttm:
            parser.error("--until and --ttm are mutually exclusive")

    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

    config = load_config(args.config, logger)

    # Dispatch to appropriate mode
    if args.intraday:
        run_intraday(args, config, logger)
    elif args.bl_only:
        run_bl_only(args, config, logger)
    elif args.mode == "barrier":
        run_barrier(args, config, logger)
    else:
        run_terminal(args, config, logger)


if __name__ == "__main__":
    main()
