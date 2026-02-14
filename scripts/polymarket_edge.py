#!/usr/bin/env python3
"""Find edge between Polymarket prices and model probabilities."""

import argparse
import json
import subprocess
import re
import sys
from pathlib import Path

# Add project root to path for btc_pricer imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from btc_pricer.edge import has_edge as _has_edge, required_model_prob

# Default JSON file paths
DEFAULT_POLYMARKET_JSON = Path(__file__).parent.parent / "results" / "polymarket_data.json"
DEFAULT_TERMINAL_JSON = Path(__file__).parent.parent / "results" / "barrier_data.json"


def _parse_polymarket_stdout(output: str) -> dict:
    """Parse Polymarket script stdout using regex (fallback method).

    Args:
        output: Raw stdout from polymarket_btc_daily.py

    Returns:
        Dictionary with parsed market data
    """
    # Parse: "Price to beat: $102,500.00"
    barrier_match = re.search(r"Price to beat: \$([0-9,]+\.?\d*)", output)
    barrier = float(barrier_match.group(1).replace(",", "")) if barrier_match else None

    # Parse: "Time remaining: 5h 23m"
    time_match = re.search(r"Time remaining: (\d+)h (\d+)m", output)
    hours = int(time_match.group(1)) if time_match else 0
    minutes = int(time_match.group(2)) if time_match else 0

    # Parse: "UP    65.0% ($0.650)"
    up_match = re.search(r"UP\s+([0-9.]+)%", output)
    prob_up = float(up_match.group(1)) / 100 if up_match else None

    # Parse: "DOWN  35.0% ($0.350)"
    down_match = re.search(r"DOWN\s+([0-9.]+)%", output)
    prob_down = float(down_match.group(1)) / 100 if down_match else None

    # Parse: "Current price: $103,150.00"
    current_match = re.search(r"Current price: \$([0-9,]+\.?\d*)", output)
    current_price = float(current_match.group(1).replace(",", "")) if current_match else None

    return {
        "barrier": barrier,
        "hours_remaining": hours + minutes / 60,
        "hours": hours,
        "minutes": minutes,
        "prob_up": prob_up,
        "prob_down": prob_down,
        "current_price": current_price,
        "raw_output": output
    }


def run_polymarket_script(verbose: bool = False, json_path: Path = None):
    """Run polymarket_btc_daily.py and parse output.

    Args:
        verbose: Print raw script output
        json_path: Path for JSON output file (uses default if None)

    Returns:
        Dictionary with market data
    """
    script_dir = Path(__file__).parent
    script_path = script_dir / "polymarket_btc_daily.py"
    json_file = json_path or DEFAULT_POLYMARKET_JSON

    # Run with --json flag
    result = subprocess.run(
        [sys.executable, str(script_path), "--json", str(json_file)],
        capture_output=True,
        text=True
    )
    output = result.stdout

    # Always forward subprocess stderr for log visibility
    if result.stderr:
        for line in result.stderr.strip().splitlines():
            print(f"  [polymarket] {line}")

    # Check for subprocess failure
    if result.returncode != 0:
        print(f"[ERROR] polymarket_btc_daily.py failed with code {result.returncode}")
        if not output:
            print(f"[ERROR] stdout was empty")
        # Return dict with None values to signal failure
        return {
            "barrier": None,
            "hours_remaining": 0,
            "hours": 0,
            "minutes": 0,
            "prob_up": None,
            "prob_down": None,
            "current_price": None,
            "raw_output": output,
            "error": result.stderr
        }

    if verbose:
        print(f"[DEBUG] polymarket_btc_daily.py completed (returncode=0)")
        print("=== Polymarket Script Output ===")
        print(output)
        print("================================\n")

    # Try to read JSON output first
    try:
        if json_file.exists():
            if verbose:
                print(f"[DEBUG] Reading JSON from {json_file}")
            with open(json_file, "r") as f:
                data = json.load(f)

            result_data = {
                "barrier": data.get("barrier"),
                "hours_remaining": data.get("hours_remaining"),
                "hours": data.get("hours"),
                "minutes": data.get("minutes"),
                "prob_up": data.get("prob_up"),
                "prob_down": data.get("prob_down"),
                "current_price": data.get("current_price"),
                "raw_output": output
            }
            if verbose:
                print(f"[DEBUG] Polymarket JSON parsed: barrier={result_data['barrier']}, "
                      f"prob_up={result_data['prob_up']}, prob_down={result_data['prob_down']}")
            return result_data
        else:
            print(f"[WARN] JSON file does not exist: {json_file}")
    except (json.JSONDecodeError, OSError) as e:
        print(f"[WARN] Failed to read JSON ({json_file}), falling back to regex: {e}")

    # Fallback to regex parsing
    if verbose:
        print(f"[DEBUG] Using regex fallback for Polymarket parsing")
    fallback_result = _parse_polymarket_stdout(output)
    if fallback_result["barrier"] is None:
        print(f"[ERROR] Regex fallback also failed - could not parse Polymarket data from stdout")
        print(f"[ERROR] stdout preview: {output[:500] if output else 'EMPTY'}")
    elif verbose:
        print(f"[DEBUG] Regex parsed: barrier={fallback_result['barrier']}, "
              f"prob_up={fallback_result['prob_up']}, prob_down={fallback_result['prob_down']}")
    return fallback_result


def _parse_terminal_stdout(output: str) -> dict:
    """Parse terminal script stdout using regex (fallback method).

    Args:
        output: Raw stdout from cli_terminal.py

    Returns:
        Dictionary with parsed model probabilities
    """
    # Parse SSVI row from table format:
    # " SSVI Local Vol MC      0.992    98.4% ±0.0%            1.6%"
    # Columns: model, R², P(>barrier) ±ci, P(<barrier)
    ssvi_match = re.search(
        r"SSVI Local Vol MC\s+[\d.]+\s+([\d.]+)%\s*[±\+\-][\d.]+%\s+([\d.]+)%",
        output
    )
    if ssvi_match:
        prob_above = float(ssvi_match.group(1)) / 100
        prob_below = float(ssvi_match.group(2)) / 100
    else:
        # Fallback to Heston row if SSVI not found
        heston_match = re.search(
            r"Heston MC\s+[\d.]+\s+([\d.]+)%\s*[±\+\-][\d.]+%\s+([\d.]+)%",
            output
        )
        if heston_match:
            prob_above = float(heston_match.group(1)) / 100
            prob_below = float(heston_match.group(2)) / 100
        else:
            prob_above = None
            prob_below = None

    # Parse spot price: "Spot Price:        $103,150.00 (Binance)"
    spot_match = re.search(r"Spot Price:\s+\$([0-9,]+\.?\d*)", output)
    spot_price = float(spot_match.group(1).replace(",", "")) if spot_match else None

    return {
        "prob_above": prob_above,
        "prob_below": prob_below,
        "spot_price": spot_price,
        "raw_output": output
    }


def run_terminal_script(barrier: float, hours: float, verbose: bool = False, json_path: Path = None):
    """Run cli_terminal.py and parse output.

    Args:
        barrier: Barrier price level
        hours: Time remaining in hours
        verbose: Print raw script output
        json_path: Path for JSON output file (uses default if None)

    Returns:
        Dictionary with model probabilities
    """
    script_dir = Path(__file__).parent.parent
    script_path = script_dir / "cli_terminal.py"
    json_file = json_path or DEFAULT_TERMINAL_JSON

    # Convert hours to days for --ttm argument
    days = hours / 24.0

    # Run with --output flag for JSON
    result = subprocess.run(
        [sys.executable, str(script_path),
         "--reference-price", str(barrier),
         "--ttm", str(days),
         "--output", str(json_file)],
        capture_output=True,
        text=True
    )
    output = result.stdout

    # Always forward subprocess stderr for log visibility
    if result.stderr:
        for line in result.stderr.strip().splitlines():
            print(f"  [cli_terminal] {line}")

    # Check for subprocess failure
    if result.returncode != 0:
        print(f"[ERROR] cli_terminal.py failed with code {result.returncode}")
        if not output:
            print(f"[ERROR] stdout was empty")
        # Return dict with None values to signal failure
        return {
            "prob_above": None,
            "prob_below": None,
            "spot_price": None,
            "raw_output": output,
            "error": result.stderr
        }

    if verbose:
        print(f"[DEBUG] cli_terminal.py completed (returncode=0)")
        print("=== Terminal Script Output ===")
        print(output)
        print("==============================\n")

    # Try to read JSON output first
    try:
        if json_file.exists():
            if verbose:
                print(f"[DEBUG] Reading terminal JSON from {json_file}")
            with open(json_file, "r") as f:
                data = json.load(f)

            # Prefer SSVI model if available, otherwise Heston
            preferred = data.get("preferred_model", "ssvi")
            model_data = data.get(preferred) or data.get("ssvi") or data.get("heston")

            if model_data:
                # Determine which model was actually used
                if data.get(preferred):
                    used_model = preferred
                elif data.get("ssvi"):
                    used_model = "ssvi"
                else:
                    used_model = "heston"

                # Extract alternate model data (for comparison display)
                alt_model = "ssvi" if used_model == "heston" else "heston"
                alt_data = data.get(alt_model) or {}

                result_data = {
                    "prob_above": model_data.get("prob_above"),
                    "prob_below": model_data.get("prob_below"),
                    "spot_price": data.get("spot_price"),
                    "raw_output": output,
                    "model_used": used_model,
                    "r_squared": model_data.get("r_squared"),
                    "model_params": model_data.get("params"),
                    "heston_r_squared": (data.get("heston") or {}).get("r_squared"),
                    "ssvi_r_squared": (data.get("ssvi") or {}).get("r_squared"),
                    "alt_model": alt_model if alt_data else None,
                    "alt_prob_above": alt_data.get("prob_above"),
                    "alt_prob_below": alt_data.get("prob_below"),
                    "alt_r_squared": alt_data.get("r_squared"),
                    "alt_params": alt_data.get("params"),
                    "bl_prob_above": data.get("bl_prob_above"),
                    "bl_prob_below": data.get("bl_prob_below"),
                    "bl_mc_divergence": data.get("bl_mc_divergence"),
                    # Per-model probabilities for detailed logging
                    "surface_mc_above": (data.get("ssvi_surface") or {}).get("mc_prob_above"),
                    "surface_mc_below": (data.get("ssvi_surface") or {}).get("mc_prob_below"),
                    "surface_bl_above": (data.get("ssvi_surface") or {}).get("prob_above"),
                    "surface_bl_below": (data.get("ssvi_surface") or {}).get("prob_below"),
                    "surface_r_squared": (data.get("ssvi_surface") or {}).get("r_squared"),
                    "heston_bl_above": (data.get("heston") or {}).get("prob_above"),
                    "heston_bl_below": (data.get("heston") or {}).get("prob_below"),
                    "timing": data.get("timing", {}),
                }

                # Use B-L probability as primary model probability (more reliable than MC)
                avg_above = data.get("avg_prob_above")
                avg_below = data.get("avg_prob_below")
                if avg_above is not None:
                    result_data["mc_prob_above"] = result_data["prob_above"]
                    result_data["mc_prob_below"] = result_data["prob_below"]
                    result_data["prob_above"] = avg_above
                    result_data["prob_below"] = avg_below
                if verbose:
                    print(f"[DEBUG] Terminal JSON parsed (model={used_model}): "
                          f"prob_above={result_data['prob_above']}, prob_below={result_data['prob_below']}")
                return result_data
            else:
                print(f"[WARN] No model data found in JSON (preferred={preferred}, keys={list(data.keys())})")
        else:
            print(f"[WARN] Terminal JSON file does not exist: {json_file}")
    except (json.JSONDecodeError, OSError) as e:
        print(f"[WARN] Failed to read terminal JSON ({json_file}), falling back to regex: {e}")

    # Fallback to regex parsing
    if verbose:
        print(f"[DEBUG] Using regex fallback for terminal parsing")
    fallback_result = _parse_terminal_stdout(output)
    if fallback_result["prob_above"] is None:
        print(f"[ERROR] Regex fallback also failed - could not parse model probabilities from stdout")
        print(f"[ERROR] stdout preview: {output[:500] if output else 'EMPTY'}")
    elif verbose:
        print(f"[DEBUG] Regex parsed: prob_above={fallback_result['prob_above']}, prob_below={fallback_result['prob_below']}")
    return fallback_result


def find_opportunities(poly_data: dict, model_data: dict,
                       alpha_up: float = 1.5, alpha_down: float = 1.5,
                       floor_up: float = 0.65, floor_down: float = 0.65,
                       # Legacy parameters (ignored when alpha/floor are used)
                       min_edge_up: float = 2.0, min_edge_down: float = 1.25,
                       min_model_prob: float = 0.6):
    """Find trading opportunities with sufficient edge.

    Uses the continuous edge function: required = max(floor, 1 - (1 - market_prob)^alpha).
    Alpha/floor are specified per-direction so UP and DOWN can have different curve shapes.
    The edge ratio is still computed for display but the entry decision uses the continuous function.
    """
    opportunities = []

    # Check UP opportunity (model says price ends above barrier)
    if poly_data["prob_up"] and model_data["prob_above"]:
        market_p = poly_data["prob_up"]
        model_p = model_data["prob_above"]
        edge_up = model_p / market_p if market_p > 0 else 0
        req = required_model_prob(market_p, alpha_up, floor_up)
        opportunities.append({
            "direction": "UP",
            "polymarket_prob": market_p,
            "model_prob": model_p,
            "edge": edge_up,
            "has_edge": _has_edge(model_p, market_p, alpha_up, floor_up),
            "required_prob": req,
            "market_entry": market_p,
            "take_profit": market_p * 1.20,
        })

    # Check DOWN opportunity (model says price ends below barrier)
    if poly_data["prob_down"] and model_data["prob_below"]:
        market_p = poly_data["prob_down"]
        model_p = model_data["prob_below"]
        edge_down = model_p / market_p if market_p > 0 else 0
        req = required_model_prob(market_p, alpha_down, floor_down)
        opportunities.append({
            "direction": "DOWN",
            "polymarket_prob": market_p,
            "model_prob": model_p,
            "edge": edge_down,
            "has_edge": _has_edge(model_p, market_p, alpha_down, floor_down),
            "required_prob": req,
            "market_entry": market_p,
            "take_profit": market_p * 1.20,
        })

    return opportunities


def display_results(poly_data: dict, model_data: dict, opportunities: list,
                    alpha_up: float = 1.5, alpha_down: float = 1.5,
                    floor_up: float = 0.65, floor_down: float = 0.65,
                    min_edge_up: float = 2.0, min_edge_down: float = 1.25):
    """Display comparison and opportunities."""
    barrier = poly_data["barrier"]
    hours = poly_data["hours"]
    minutes = poly_data["minutes"]
    spot = model_data["spot_price"] or poly_data["current_price"]

    print()
    print("=" * 60)
    print("           POLYMARKET EDGE FINDER")
    print("=" * 60)
    print(f" Barrier: ${barrier:,.2f} (price to beat)")
    print(f" Time remaining: {hours}h {minutes}m")
    if spot:
        print(f" Current BTC: ${spot:,.2f}")
    if alpha_up == alpha_down and floor_up == floor_down:
        print(f" Edge curve: alpha={alpha_up}, floor={floor_up:.0%}")
    else:
        print(f" Edge curve: UP alpha={alpha_up} floor={floor_up:.0%} | DOWN alpha={alpha_down} floor={floor_down:.0%}")
    print("-" * 60)

    # Comparison table
    print(f" {'Direction':<18} {'Polymarket':>12} {'Model':>10} {'Required':>10} {'Edge':>10}")
    print("-" * 60)

    for opp in opportunities:
        direction = opp["direction"]
        if direction == "UP":
            label = f"UP (>${barrier:,.0f})"
        else:
            label = f"DOWN (<${barrier:,.0f})"

        poly_pct = f"{opp['polymarket_prob'] * 100:.1f}%"
        model_pct = f"{opp['model_prob'] * 100:.1f}%"
        req_pct = f"{opp['required_prob'] * 100:.1f}%"
        edge_str = f"{opp['edge']:.2f}x"

        print(f" {label:<18} {poly_pct:>12} {model_pct:>10} {req_pct:>10} {edge_str:>10}")

    print("-" * 60)
    print()

    # Check for opportunities
    edge_opportunities = [o for o in opportunities if o["has_edge"]]

    if not edge_opportunities:
        if alpha_up == alpha_down and floor_up == floor_down:
            print(f" No edge found (alpha={alpha_up}, floor={floor_up:.0%}).")
        else:
            print(f" No edge found.")
        print()
        print("=" * 60)
        return

    # Display each opportunity
    for opp in edge_opportunities:
        print(f" >>> OPPORTUNITY: BUY {opp['direction']} <<<")
        print()
        print(f" Polymarket says: {opp['polymarket_prob'] * 100:.1f}%")
        print(f" Model says:      {opp['model_prob'] * 100:.1f}%")
        print(f" Required:        {opp['required_prob'] * 100:.1f}%")
        print(f" Edge ratio:      {opp['edge']:.2f}x")
        print()
        print(" Entry Prices:")
        print(f"   Market buy:      ${opp['market_entry']:.3f}")
        print()
        print(f" Take Profit:       ${opp['take_profit']:.3f} (20% gain)")
        print()

    print("=" * 60)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Find edge between Polymarket prices and model probabilities"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Show raw output from underlying scripts"
    )
    parser.add_argument(
        "--alpha-up",
        type=float,
        default=1.5,
        help="Edge curve exponent for UP bets (default: 1.5). Higher = more edge required at high confidence."
    )
    parser.add_argument(
        "--alpha-down",
        type=float,
        default=1.5,
        help="Edge curve exponent for DOWN bets (default: 1.5)."
    )
    parser.add_argument(
        "--floor-up",
        type=float,
        default=0.65,
        help="Minimum model probability floor for UP bets (default: 0.65)"
    )
    parser.add_argument(
        "--floor-down",
        type=float,
        default=0.65,
        help="Minimum model probability floor for DOWN bets (default: 0.65)"
    )

    args = parser.parse_args()

    print("Fetching Polymarket data...")
    poly_data = run_polymarket_script(verbose=args.verbose)

    if poly_data["barrier"] is None:
        print("Error: Could not parse Polymarket data")
        sys.exit(1)

    if poly_data["hours_remaining"] <= 0:
        print("Error: Market has expired or no time remaining")
        sys.exit(1)

    print(f"Running model calibration for ${poly_data['barrier']:,.0f} barrier...")
    print(f"Time horizon: {poly_data['hours']}h {poly_data['minutes']}m")
    print()

    model_data = run_terminal_script(
        poly_data["barrier"],
        poly_data["hours_remaining"],
        verbose=args.verbose
    )

    if model_data["prob_above"] is None or model_data["prob_below"] is None:
        print("Error: Could not parse model probabilities")
        if args.verbose:
            print("Model output:")
            print(model_data["raw_output"])
        sys.exit(1)

    opportunities = find_opportunities(
        poly_data, model_data,
        alpha_up=args.alpha_up, alpha_down=args.alpha_down,
        floor_up=args.floor_up, floor_down=args.floor_down,
    )
    display_results(
        poly_data, model_data, opportunities,
        alpha_up=args.alpha_up, alpha_down=args.alpha_down,
        floor_up=args.floor_up, floor_down=args.floor_down,
    )


if __name__ == "__main__":
    main()
