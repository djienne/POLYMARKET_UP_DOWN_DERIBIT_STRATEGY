#!/usr/bin/env python3
"""
BTC Terminal Probability Calculator

Calculate the probability that BTC ends above or below a price level at expiry.

Two modes:
  --method mc  (default): SSVI surface fit, MC simulation, Breeden-Litzenberger validation.
  --method rnd:           Lightweight RND-only mode via Breeden-Litzenberger (no MC).
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
from btc_pricer.models.heston import HestonParams
from btc_pricer.models.ssvi import SSVISliceData, SSVISurfaceParams
from btc_pricer.models.terminal_probability import TerminalProbabilityCalculator, MCResult
from btc_pricer.models.breeden_litzenberger import BreedenLitzenberger, RNDResult
from btc_pricer.cli.common import (
    setup_logging,
    load_config,
    handle_cli_exceptions,
    add_common_arguments,
    find_closest_expiry_after,
    calibrate_to_expiry,
    create_ssvi_surface_fitter,
    extract_surface_data,
    format_current_time_multizone,
    parse_expiry_date,
)
from btc_pricer.data.filters import DataFilter
from btc_pricer.constants import RELAXED_MIN_POINTS
from btc_pricer.utils.time_parser import (
    parse_datetime_with_timezone,
    calculate_ttm_to_target,
    format_target_time,
    TimeParseError,
)


def format_single_probability(
    spot: float,
    target_price: float,
    prob_above: float,
    prob_below: float,
    expiry: str,
    ttm_days: float,
    r_squared: float,
    model_type: str = "heston"
) -> str:
    """Format single price probability result for display (RND mode)."""
    distance_pct = (target_price - spot) / spot * 100

    model_display = model_type.upper()
    if model_type == "ssvi":
        model_display = "SSVI [Heston fallback]"

    lines = [
        "",
        "=" * 65,
        "           TERMINAL PROBABILITY ANALYSIS",
        "=" * 65,
        f" Spot Price:      ${spot:,.2f}",
        f" Target Price:    ${target_price:,.2f} ({distance_pct:+.1f}% from spot)",
        f" Expiry:          {expiry} ({ttm_days:.1f} days)",
        "-" * 65,
        f" P(BTC > ${target_price:,.2f}):  {prob_above * 100:.1f}%",
        f" P(BTC < ${target_price:,.2f}):  {prob_below * 100:.1f}%",
        "-" * 65,
        f" Model: {model_display} (R²={r_squared:.3f})",
        f" Based on RND from {expiry} options",
        "=" * 65,
        "",
    ]
    return "\n".join(lines)


def format_multiple_probabilities_table(
    results: list,
    spot: float,
    expiry: str,
    ttm_days: float,
    r_squared: float,
    model_type: str = "heston"
) -> str:
    """Format multiple price probability results as a table (RND mode)."""
    model_display = model_type.upper()
    if model_type == "ssvi":
        model_display = "SSVI [Heston fallback]"

    lines = [
        "",
        "=" * 75,
        "                    TERMINAL PROBABILITY ANALYSIS",
        "=" * 75,
        f" Spot Price: ${spot:,.2f}",
        f" Expiry: {expiry} ({ttm_days:.1f} days) | Model: {model_display} (R²={r_squared:.3f})",
        "-" * 75,
        f" {'Target Price':>14} {'Distance':>10} {'P(Above)':>12} {'P(Below)':>12}",
        "-" * 75,
    ]

    for r in sorted(results, key=lambda x: x["target_price"]):
        target = r["target_price"]
        dist = r["distance_pct"]
        p_above = r["prob_above"] * 100
        p_below = r["prob_below"] * 100

        lines.append(
            f" ${target:>13,.2f} {dist:>+9.1f}% {p_above:>11.1f}% {p_below:>11.1f}%"
        )

    lines.extend(["-" * 75, ""])
    return "\n".join(lines)


def run_rnd_mode(args, config, client):
    """Run lightweight RND-only mode (equivalent to old cli_probability.py).

    Calibrates to a single expiry, extracts the Risk-Neutral Density via
    Breeden-Litzenberger, and computes terminal probabilities without MC.
    """
    logger = logging.getLogger(__name__)

    # Resolve target prices from --price/--prices shorthands or --reference-price/--reference-prices
    reference_price = args.reference_price or args.price_alias
    reference_prices = args.reference_prices or args.prices_alias
    if not reference_price and not reference_prices:
        raise SystemExit("Error: Must specify --price/--prices or --reference-price/--reference-prices")
    target_prices = reference_prices if reference_prices else [reference_price]

    # Calibrate model (Heston with SSVI fallback)
    params, spot_price, calibration_expiry, forward, r_squared, ttm, model_type = calibrate_to_expiry(
        client, config, args.expiry
    )

    # Override spot if specified
    if args.spot is not None:
        spot_price = args.spot
        logger.info(f"Overriding spot price to: ${spot_price:,.2f}")

    ttm_days = ttm * 365

    # Extract RND using Breeden-Litzenberger
    logger.info("Extracting Risk-Neutral Density...")
    bl = BreedenLitzenberger(
        strike_grid_points=500,
        strike_range_std=4.0,
        use_log_strikes=True
    )

    if model_type == "heston":
        rnd = bl.extract_from_heston(params, forward, use_quantlib=config.heston.use_quantlib)
    else:  # ssvi
        rnd = bl.extract_from_ssvi(params, forward)

    if not rnd.is_valid:
        logger.warning(f"RND extraction warnings: {rnd.warnings}")

    # Calculate probabilities for each target price
    results = []
    for target in target_prices:
        prob_above = bl.probability_above(rnd, target)
        prob_below = bl.probability_below(rnd, target)
        distance_pct = (target - spot_price) / spot_price * 100

        results.append({
            "target_price": target,
            "distance_pct": distance_pct,
            "prob_above": prob_above,
            "prob_below": prob_below,
        })

        logger.debug(f"P(>{target:,.2f}) = {prob_above:.3f}, P(<{target:,.2f}) = {prob_below:.3f}")

    # Display results
    if len(results) == 1:
        r = results[0]
        print(format_single_probability(
            spot_price,
            r["target_price"],
            r["prob_above"],
            r["prob_below"],
            calibration_expiry,
            ttm_days,
            r_squared,
            model_type
        ))
    else:
        print(format_multiple_probabilities_table(
            results, spot_price, calibration_expiry, ttm_days, r_squared, model_type
        ))

    # Save to JSON if requested
    if args.output:
        output_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "spot_price": spot_price,
            "forward_price": forward,
            "calibration_expiry": calibration_expiry,
            "ttm_days": ttm_days,
            "model_type": model_type,
            "model_params": params.to_dict(),
            "r_squared": r_squared,
            "rnd_statistics": rnd.to_dict(),
            "probabilities": results
        }

        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2, default=str)

        logger.info(f"Results saved to {args.output}")


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
    """Format terminal probability result for display.

    Args:
        result: Terminal probability calculation result
        params: Model parameters (Heston or SSVI)
        r_squared: Calibration R-squared
        model_type: "heston" or "ssvi"
        expiry_str: Expiry string for calibration (e.g., "31JAN26")
        target_utc: Target UTC datetime if using --until
        target_tz_display: Original timezone abbreviation if using --until
    """
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
        f" Reference Price:   ${result.reference_price:,.0f} ({result.direction.upper()}, {result.reference_price_distance_pct:+.1f}%)",
        f" Time Horizon:      {result.ttm * 365:.3f} days ({result.ttm * 365 * 24:.2f} hours)",
    ]
    if target_display:
        lines.append(target_display)
    if expiry_display:
        lines.append(expiry_display)
    lines.extend([
        "-" * 65,
        f" P(>${result.reference_price:,.0f}): {term_prob_above * 100:.1f}% +/- {ci_width:.1f}%",
        f" P(<${result.reference_price:,.0f}): {term_prob_below * 100:.1f}%",
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
    reference_price: float,
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
    ref_dist = (reference_price - spot) / spot * 100

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
        f" Reference Price:   ${reference_price:,.0f} ({direction.upper()}, {ref_dist:+.1f}%)",
        f" Time Horizon:      {ttm * 365:.3f} days ({ttm * 365 * 24:.2f} hours)",
    ]

    if target_display:
        lines.append(target_display)
    if expiry_display:
        lines.append(expiry_display)

    lines.append("-" * 70)
    lines.append(f" {'Model':<20} {'R²':>8} {'P(>ref)':>16} {'P(<ref)':>16}")
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
    reference_price: float,
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
) -> dict:
    """Build JSON output structure for terminal probability calculation results.

    Args:
        heston_r2: Heston calibration R-squared
        spot: Current spot price
        reference_price: Reference price level
        direction: "up" or "down"
        ttm: Time to maturity in years
        bl_prob_above: Heston Breeden-Litzenberger P(S_T > reference_price) (or None)
        bl_prob_below: Heston Breeden-Litzenberger P(S_T < reference_price) (or None)
        surface_bl_above: SSVI Surface B-L P(S_T > reference_price) (or None)
        surface_bl_below: SSVI Surface B-L P(S_T < reference_price) (or None)
        surface_mc_above: SSVI Surface MC P(S_T > reference_price) (or None)
        surface_mc_below: SSVI Surface MC P(S_T < reference_price) (or None)
        surface_r2: SSVI Surface calibration R-squared (or None)
        surface_params_at_ttm: SSVIParams interpolated to target TTM (or None)
        surface_params_obj: SSVISurfaceParams full surface (or None)
        timing: Per-step timing dict (seconds) for downstream consumers (or None)

    Returns:
        Dictionary with structured JSON output
    """
    output = {
        "spot_price": spot,
        "reference_price": reference_price,
        "direction": direction,
        "ttm_days": ttm * 365,
        "heston": None,
        "preferred_model": None,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    # Add Heston results (B-L only — MC is skipped)
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

    # Determine preferred model:
    #   1. SSVI Surface (when surface B-L is available)
    #   2. Heston (fallback when Heston B-L is available)
    if surface_bl_above is not None:
        output["preferred_model"] = "ssvi_surface"
    elif bl_prob_above is not None:
        output["preferred_model"] = "heston"

    # Add Breeden-Litzenberger terminal probabilities (Heston B-L)
    output["bl_prob_above"] = bl_prob_above
    output["bl_prob_below"] = bl_prob_below

    # Model-specific avg_prob logic:
    #   - SSVI Surface: avg(MC, B-L) — both methods reliable for SSVI
    #   - Heston: B-L only — MC unreliable due to Feller violations
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
        # B-L only for Heston (MC skipped — unreliable due to Feller violations)
        if bl_prob_above is not None:
            output["avg_prob_above"] = bl_prob_above
            output["avg_prob_below"] = bl_prob_below
        else:
            output["avg_prob_above"] = None
            output["avg_prob_below"] = None
    else:
        output["avg_prob_above"] = None
        output["avg_prob_below"] = None

    # Track divergence for diagnostics (surface MC vs surface B-L when available)
    if surface_mc_above is not None and surface_bl_above is not None:
        output["bl_mc_divergence"] = abs(surface_mc_above - surface_bl_above)
    else:
        output["bl_mc_divergence"] = None

    return output


@handle_cli_exceptions
def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="BTC Terminal Probability Calculator - Terminal probabilities using Heston/SSVI MC"
    )
    add_common_arguments(parser)

    parser.add_argument(
        "--method",
        choices=["mc", "rnd"],
        default="mc",
        help="Calculation method: 'mc' for full SSVI surface + MC pipeline (default), "
             "'rnd' for lightweight RND-only via Breeden-Litzenberger"
    )
    parser.add_argument(
        "--reference-price",
        type=float,
        help="Single reference price level (e.g., 85000)"
    )
    parser.add_argument(
        "--price",
        type=float,
        dest="price_alias",
        help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--reference-prices",
        type=float,
        nargs="+",
        help="Multiple reference price levels (e.g., 80000 85000 90000 95000)"
    )
    parser.add_argument(
        "--prices",
        type=float,
        nargs="+",
        dest="prices_alias",
        help=argparse.SUPPRESS
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
        help="Direction: 'down' for price dropping to reference price, 'up' for rising (default: down)"
    )
    parser.add_argument(
        "--sims",
        type=int,
        default=None,
        help="Number of Monte Carlo simulations (default: from config, typically 200000)"
    )
    parser.add_argument(
        "--steps-per-day",
        type=int,
        default=None,
        help="Time steps per day for simulation (default: from config, typically 288 = 5-min)"
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
    args = parser.parse_args()

    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

    # Lightweight RND-only mode (replaces old cli_probability.py)
    if args.method == "rnd":
        config = load_config(args.config, logger)
        client = DeribitClient(config.api, config.validation)
        return run_rnd_mode(args, config, client)

    # Resolve shorthand aliases
    reference_price = args.reference_price or args.price_alias
    reference_prices = args.reference_prices or args.prices_alias
    direction = args.direction

    # Validate arguments
    if not reference_price and not reference_prices:
        parser.error("Must specify either --reference-price or --reference-prices")

    # Check for mutually exclusive arguments
    if args.until:
        if args.expiry:
            parser.error("--until and --expiry are mutually exclusive")
        if args.ttm:
            parser.error("--until and --ttm are mutually exclusive")

    ref_prices = reference_prices if reference_prices else [reference_price]

    # Initialize timing variables (set when each step completes)
    import time as _time
    _calibration_elapsed = None
    _surface_elapsed = None
    _mc_elapsed = None
    _bl_elapsed = None
    _sbl_elapsed = None

    # Load configuration
    config = load_config(args.config, logger)

    # Get terminal config (with defaults)
    n_sims = args.sims
    steps_per_day = args.steps_per_day

    # Try to load from config if available
    if hasattr(config, 'terminal'):
        if n_sims is None:
            n_sims = getattr(config.terminal, 'n_simulations', 200000)
        if steps_per_day is None:
            steps_per_day = getattr(config.terminal, 'n_steps_per_day', 288)
    else:
        if n_sims is None:
            n_sims = 200000
        if steps_per_day is None:
            steps_per_day = 288

    # Initialize API client
    client = DeribitClient(config.api, config.validation)

    # Variables for --until feature
    target_utc = None
    target_tz_display = None
    options_by_expiry = None  # Will be populated in --until path or on demand for surface

    # Handle --until argument: parse time, calculate TTM, find expiry
    if args.until:
        try:
            target_utc, target_tz_display = parse_datetime_with_timezone(args.until)
            ttm_from_until = calculate_ttm_to_target(target_utc)
            logger.info(
                f"Target time: {target_utc.strftime('%Y-%m-%d %H:%M UTC')} "
                f"({args.until})"
            )
            logger.info(f"TTM to target: {ttm_from_until * 365:.2f} days")
        except TimeParseError as e:
            parser.error(str(e))

        # Fetch options to find closest expiry after target
        logger.info("Fetching options data to find appropriate expiry...")
        options_by_expiry = client.fetch_all_options("BTC")

        if not options_by_expiry:
            raise DeribitAPIError("No options data received")

        try:
            auto_expiry = find_closest_expiry_after(options_by_expiry, target_utc)
            logger.info(f"Auto-selected expiry: {auto_expiry}")
        except ValueError as e:
            parser.error(str(e))

        # Calibrate both models to the auto-selected expiry
        _t0_cal = _time.time()
        heston_data, ssvi_data, spot_price, calibration_expiry, forward, _ = calibrate_to_expiry(
            client, config, auto_expiry, return_both=True
        )
        _calibration_elapsed = _time.time() - _t0_cal

        # Override TTM with the calculated value from --until
        ttm = ttm_from_until

    else:
        # Standard flow: calibrate both models to specified or nearest expiry
        _t0_cal = _time.time()
        heston_data, ssvi_data, spot_price, calibration_expiry, forward, ttm = calibrate_to_expiry(
            client, config, args.expiry, return_both=True
        )
        _calibration_elapsed = _time.time() - _t0_cal

        # Override TTM if specified
        if args.ttm is not None:
            ttm = args.ttm / 365.0
            logger.info(f"Using custom TTM: {args.ttm:.1f} days")

    # Extract params and R² for each model
    heston_params, heston_r2 = heston_data if heston_data else (None, None)
    _ssvi_params, _ssvi_r2 = ssvi_data if ssvi_data else (None, None)  # unused; kept for calibrate_to_expiry interface

    # Note: spot_price from calibrate_to_expiry is already Binance (with Deribit fallback)

    # SSVI Surface fit: fit across multiple expiries, interpolate to target TTM
    surface_params_at_ttm = None  # SSVIParams interpolated to target TTM
    surface_params_obj = None     # SSVISurfaceParams (full surface)
    surface_r2 = None
    surface_info = None

    logger.info("Fitting SSVI surface across nearby expiries...")
    # Fetch options if not already available (standard path doesn't have them)
    if options_by_expiry is None:
        options_by_expiry = client.fetch_all_options("BTC")

    if options_by_expiry:
        data_filter_surf = DataFilter(config.filters)
        surface_slices = []

        for exp_name in sorted(options_by_expiry.keys(), key=parse_expiry_date):
            opts = options_by_expiry[exp_name]
            filtered_opts, _ = data_filter_surf.filter_options(opts, return_stats=True)
            otm = data_filter_surf.build_otm_surface(filtered_opts)
            sd = extract_surface_data(
                otm, min_points=RELAXED_MIN_POINTS,
                iv_valid_range=config.validation.iv_valid_range,
            )
            if sd is None:
                continue
            fwd_s, ttm_s, _, log_k_s, mkt_iv_s = sd
            if ttm_s * 365 > config.ssvi_surface.max_ttm_days:
                continue
            surface_slices.append(SSVISliceData(
                expiry_name=exp_name, ttm=ttm_s,
                log_moneyness=log_k_s, market_iv=mkt_iv_s, forward=fwd_s,
            ))

        if len(surface_slices) >= config.ssvi_surface.min_expiries:
            surface_fitter = create_ssvi_surface_fitter(config)

            # Try progressively: all slices -> drop shortest expiry
            attempts = [
                ("all slices", surface_slices),
            ]
            if len(surface_slices) > config.ssvi_surface.min_expiries:
                attempts.append(("without shortest expiry", surface_slices[1:]))

            for attempt_name, slices_to_try in attempts:
                if len(slices_to_try) < config.ssvi_surface.min_expiries:
                    continue
                _t0 = _time.time()
                surface_fit = surface_fitter.fit(slices_to_try)
                _surface_elapsed = _time.time() - _t0

                if surface_fit.success and surface_fit.params is not None:
                    surface_params_obj = surface_fit.params
                    surface_r2 = surface_fit.aggregate_r_squared
                    surface_params_at_ttm = surface_params_obj.get_params_for_ttm(ttm)
                    n_slices = len(slices_to_try)
                    slice_names = [s.expiry_name for s in slices_to_try]
                    surface_info = (
                        f"Surface: rho={surface_params_obj.rho:.3f}, "
                        f"eta={surface_params_obj.eta:.3f}, lam={surface_params_obj.lam:.3f} "
                        f"({n_slices} expiries: {', '.join(slice_names)})"
                    )
                    logger.info(
                        f"SSVI surface fit OK ({attempt_name}) in {_surface_elapsed:.1f}s: "
                        f"R2={surface_r2:.4f}, {surface_info}"
                    )
                    logger.info(
                        f"Surface interpolated to TTM={ttm*365:.3f}d: "
                        f"theta={surface_params_at_ttm.theta:.6f}, "
                        f"phi={surface_params_at_ttm.phi:.3f}, "
                        f"rho={surface_params_at_ttm.rho:.4f}"
                    )
                    break
                else:
                    logger.warning(
                        f"SSVI surface fit failed ({attempt_name}): {surface_fit.message}"
                    )

            # If all attempts failed, try expanding max_ttm_days to 2x
            if surface_params_at_ttm is None:
                expanded_max = config.ssvi_surface.max_ttm_days * 2
                expanded_slices = []
                for exp_name in sorted(options_by_expiry.keys(), key=parse_expiry_date):
                    opts = options_by_expiry[exp_name]
                    filtered_opts, _ = data_filter_surf.filter_options(opts, return_stats=True)
                    otm = data_filter_surf.build_otm_surface(filtered_opts)
                    sd = extract_surface_data(
                        otm, min_points=RELAXED_MIN_POINTS,
                        iv_valid_range=config.validation.iv_valid_range,
                    )
                    if sd is None:
                        continue
                    fwd_s, ttm_s, _, log_k_s, mkt_iv_s = sd
                    if ttm_s * 365 > expanded_max:
                        continue
                    expanded_slices.append(SSVISliceData(
                        expiry_name=exp_name, ttm=ttm_s,
                        log_moneyness=log_k_s, market_iv=mkt_iv_s, forward=fwd_s,
                    ))

                if len(expanded_slices) > len(surface_slices):
                    expand_attempts = [
                        (f"expanded {expanded_max:.0f}d", expanded_slices),
                    ]
                    if len(expanded_slices) > config.ssvi_surface.min_expiries:
                        expand_attempts.append(
                            (f"expanded {expanded_max:.0f}d without shortest", expanded_slices[1:])
                        )
                    for attempt_name, slices_to_try in expand_attempts:
                        if len(slices_to_try) < config.ssvi_surface.min_expiries:
                            continue
                        _t0 = _time.time()
                        surface_fit = surface_fitter.fit(slices_to_try)
                        _surface_elapsed = _time.time() - _t0

                        if surface_fit.success and surface_fit.params is not None:
                            surface_params_obj = surface_fit.params
                            surface_r2 = surface_fit.aggregate_r_squared
                            surface_params_at_ttm = surface_params_obj.get_params_for_ttm(ttm)
                            n_slices = len(slices_to_try)
                            slice_names = [s.expiry_name for s in slices_to_try]
                            surface_info = (
                                f"Surface: rho={surface_params_obj.rho:.3f}, "
                                f"eta={surface_params_obj.eta:.3f}, lam={surface_params_obj.lam:.3f} "
                                f"({n_slices} expiries: {', '.join(slice_names)})"
                            )
                            logger.info(
                                f"SSVI surface fit OK ({attempt_name}) in {_surface_elapsed:.1f}s: "
                                f"R2={surface_r2:.4f}, {surface_info}"
                            )
                            logger.info(
                                f"Surface interpolated to TTM={ttm*365:.3f}d: "
                                f"theta={surface_params_at_ttm.theta:.6f}, "
                                f"phi={surface_params_at_ttm.phi:.3f}, "
                                f"rho={surface_params_at_ttm.rho:.4f}"
                            )
                            break
                        else:
                            logger.warning(
                                f"SSVI surface fit failed ({attempt_name}): {surface_fit.message}"
                            )

            if surface_params_at_ttm is None:
                logger.warning("SSVI surface fit failed all retry attempts")
        else:
            logger.info(
                f"Not enough slices for surface fit "
                f"({len(surface_slices)} < {config.ssvi_surface.min_expiries})"
            )

    # Update params with custom TTM
    if heston_params is not None:
        heston_params = HestonParams(
            v0=heston_params.v0,
            kappa=heston_params.kappa,
            theta=heston_params.theta,
            xi=heston_params.xi,
            rho=heston_params.rho,
            ttm=ttm
        )
    # Override spot if specified
    if args.spot is not None:
        spot_price = args.spot
        logger.info(f"Overriding spot price to: ${spot_price:,.0f}")

    # Auto-detect direction for each reference price based on spot
    directions = {}
    for rp in ref_prices:
        if rp < spot_price:
            detected_dir = "down"
        else:
            detected_dir = "up"

        if direction != detected_dir:
            logger.info(
                f"Auto-correcting direction for ${rp:,.0f}: "
                f"'{direction}' -> '{detected_dir}' (reference price {'<' if rp < spot_price else '>'} spot)"
            )
        directions[rp] = detected_dir

    # Create calculator
    calculator = TerminalProbabilityCalculator(
        n_simulations=n_sims,
        n_steps_per_day=steps_per_day,
        use_antithetic=True
    )

    # Collect results for JSON output
    json_output = None

    # Run both models and compare
    logger.info("Calculating terminal probabilities with both models...")

    for rp in ref_prices:
        heston_result = None
        ssvi_result = None

        # Skip Heston MC — use Heston B-L only (MC unreliable due to Feller violations)
        # heston_result stays None; Heston B-L is computed below from heston_params

        # Run SSVI Surface Local Vol MC
        if surface_params_at_ttm is not None:
            logger.info(f"Running SSVI Surface Local Vol MC for reference price ${rp:,.0f}...")
            _t0 = _time.time()
            ssvi_result = calculator.terminal_probability_ssvi(
                surface_params_at_ttm, spot_price, forward, rp, directions[rp]
            )
            _mc_elapsed = _time.time() - _t0
            logger.info(f"SSVI Surface MC completed in {_mc_elapsed:.1f}s")

        # Compute Breeden-Litzenberger terminal probabilities
        # Heston B-L and SSVI Surface B-L run independently
        bl_prob_above = None
        bl_prob_below = None
        surface_bl_above = None
        surface_bl_below = None

        bl = BreedenLitzenberger(
            strike_grid_points=500, strike_range_std=4.0, use_log_strikes=True
        )

        # 1. Heston B-L
        if heston_params is not None:
            try:
                _t0 = _time.time()
                rnd = bl.extract_from_heston(heston_params, forward, use_quantlib=True)
                _bl_elapsed = _time.time() - _t0
                if rnd and rnd.is_valid:
                    bl_prob_above = float(bl.probability_above(rnd, rp))
                    bl_prob_below = float(bl.probability_below(rnd, rp))
                    logger.info(
                        f"Heston B-L in {_bl_elapsed:.1f}s: P(>{rp:,.0f})={bl_prob_above:.4f}, "
                        f"P(<{rp:,.0f})={bl_prob_below:.4f}"
                    )
                else:
                    logger.warning("Heston B-L RND extraction returned invalid result")
            except Exception as e:
                logger.warning(f"Heston B-L failed: {e}")

        # 2. SSVI Surface B-L
        if surface_params_at_ttm is not None:
            try:
                _t0 = _time.time()
                rnd = bl.extract_from_ssvi(surface_params_at_ttm, forward)
                _sbl_elapsed = _time.time() - _t0
                if rnd and rnd.is_valid:
                    surface_bl_above = float(bl.probability_above(rnd, rp))
                    surface_bl_below = float(bl.probability_below(rnd, rp))
                    logger.info(
                        f"SSVI Surface B-L in {_sbl_elapsed:.1f}s: P(>{rp:,.0f})={surface_bl_above:.4f}, "
                        f"P(<{rp:,.0f})={surface_bl_below:.4f}"
                    )
                else:
                    logger.warning("SSVI Surface B-L RND extraction returned invalid result")
            except Exception as e:
                logger.warning(f"SSVI Surface B-L failed: {e}")

        # Display comparison if both succeeded, otherwise single result
        if heston_result is not None and ssvi_result is not None:
            print(format_comparison_result(
                heston_result, ssvi_result,
                heston_params, surface_params_at_ttm,
                heston_r2, surface_r2,
                spot_price, rp, directions[rp], ttm,
                calibration_expiry, target_utc, target_tz_display,
                surface_bl_above=surface_bl_above,
                surface_bl_below=surface_bl_below,
                surface_r2=surface_r2,
                surface_info=surface_info,
                ssvi_is_surface=True,
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

        # Build JSON output (only for single reference price; last one wins for multiple)
        # Extract surface MC probabilities from ssvi_result when surface is active
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
                heston_r2,
                spot_price, rp, directions[rp], ttm,
                heston_params=heston_params,
                bl_prob_above=bl_prob_above, bl_prob_below=bl_prob_below,
                surface_bl_above=surface_bl_above, surface_bl_below=surface_bl_below,
                surface_mc_above=s_mc_above, surface_mc_below=s_mc_below,
                surface_r2=surface_r2,
                surface_params_at_ttm=surface_params_at_ttm,
                surface_params_obj=surface_params_obj,
                timing=timing,
            )

    # Write JSON output if requested
    if args.output and json_output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(json_output, f, indent=2)
        logger.info(f"JSON output written to {args.output}")


if __name__ == "__main__":
    main()
