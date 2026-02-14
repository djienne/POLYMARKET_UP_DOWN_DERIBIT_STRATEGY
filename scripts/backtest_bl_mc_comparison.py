#!/usr/bin/env python3
"""Run backtest 3 times: BL-only, MC-only, Mean — using live probabilities.csv.

Only the live trading CSV (results/probabilities.csv) has BL data in the extra
unnamed columns. This script:
  1. Reads the 14-col rows, back-derives MC from avg and BL.
  2. Writes 3 temporary CSVs with proper named columns, substituting
     model_prob_up/down with BL, MC, or mean values respectively.
  3. Shells out to backtest.py for each.
  4. Compares the results.

Usage:
    python scripts/backtest_bl_mc_comparison.py
    python scripts/backtest_bl_mc_comparison.py --alpha-up 2.4 --alpha-down 1.6 --floor-up 0.60 --floor-down 0.35
"""

import argparse
import csv
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

CSV_PATH = Path(__file__).resolve().parent.parent / "results" / "probabilities.csv"
BACKTEST_SCRIPT = Path(__file__).resolve().parent / "backtest.py"

# Header for rewritten CSVs (proper named columns)
HEADER = [
    "timestamp", "barrier_price", "time_remaining_hours", "spot_price",
    "model_prob_up", "model_prob_down", "poly_prob_up", "poly_prob_down",
    "edge_up", "edge_down", "bl_prob_up", "bl_prob_down",
    "avg_prob_up", "avg_prob_down",
]


def load_bl_rows(csv_path: Path) -> list[list[str]]:
    """Load only 14-column rows (those with BL data)."""
    rows = []
    with open(csv_path) as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for raw in reader:
            if len(raw) >= 14 and raw[10] != "" and raw[12] != "":
                rows.append(raw[:14])
    return rows


def write_regime_csv(rows: list[list[str]], regime: str, out_path: Path):
    """Write a CSV with model_prob_up/down replaced according to regime.

    Columns [4],[5] = model_prob_up, model_prob_down (used by backtest for edge)
    Columns [10],[11] = bl_prob_up, bl_prob_down
    Columns [12],[13] = avg_prob_up, avg_prob_down
    Back-derived MC = 2*avg - BL
    """
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(HEADER)
        for raw in rows:
            bl_up = float(raw[10])
            bl_down = float(raw[11])
            avg_up = float(raw[12])
            avg_down = float(raw[13])
            mc_up = 2 * avg_up - bl_up
            mc_down = 2 * avg_down - bl_down

            row = list(raw)
            if regime == "bl":
                row[4] = str(bl_up)
                row[5] = str(bl_down)
            elif regime == "mc":
                row[4] = str(mc_up)
                row[5] = str(mc_down)
            else:  # mean (current)
                row[4] = str(avg_up)
                row[5] = str(avg_down)

            # Also set avg columns to match so backtest's avg_prob path uses same value
            row[12] = row[4]
            row[13] = row[5]

            writer.writerow(row)


def run_backtest(csv_path: Path, extra_args: list[str]) -> str:
    """Run backtest.py and capture output."""
    cmd = [
        sys.executable, str(BACKTEST_SCRIPT),
        "--csv", str(csv_path),
        "--no-orderbook",
    ] + extra_args
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(BACKTEST_SCRIPT.parent.parent))
    return result.stdout + result.stderr


def extract_summary(output: str) -> dict:
    """Extract key metrics from backtest output."""
    metrics = {}

    # Total P&L — handles "+$0.21", "$-9.85", "-$10.18"
    m = re.search(r"Total P&L:\s+([+-]?)\$?([+-]?[\d.]+)", output)
    if m:
        sign = m.group(1)
        val = float(m.group(2))
        metrics["pnl"] = -val if sign == "-" else val

    # Return %
    m = re.search(r"Total P&L:.*?\(([+-]?[\d.]+)%\)", output)
    if m:
        metrics["return_pct"] = float(m.group(1))

    # Trades
    m = re.search(r"Total trades:\s+(\d+)", output)
    if m:
        metrics["trades"] = int(m.group(1))

    # Win rate
    m = re.search(r"Win rate:\s+([-\d.]+)%", output)
    if m:
        metrics["win_rate"] = float(m.group(1))

    # Sharpe
    m = re.search(r"Sharpe.*?:\s+([-\d.]+|inf|-inf|nan)", output)
    if m:
        try:
            metrics["sharpe"] = float(m.group(1))
        except ValueError:
            metrics["sharpe"] = 0.0

    # Profit factor
    m = re.search(r"Profit factor:\s+([-\d.]+|inf|-inf|nan)", output)
    if m:
        try:
            metrics["profit_factor"] = float(m.group(1))
        except ValueError:
            metrics["profit_factor"] = 0.0

    # Markets
    m = re.search(r"Markets traded:\s+(\d+)", output)
    if m:
        metrics["markets"] = int(m.group(1))

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Compare backtest: BL vs MC vs Mean")
    parser.add_argument("--csv", type=Path, default=CSV_PATH)
    parser.add_argument("--alpha-up", type=float, default=2.40)
    parser.add_argument("--alpha-down", type=float, default=1.60)
    parser.add_argument("--floor-up", type=float, default=0.60)
    parser.add_argument("--floor-down", type=float, default=0.35)
    parser.add_argument("--tp", type=float, default=0.30)
    parser.add_argument("--trail-activation", type=float, default=0.15)
    parser.add_argument("--trail-distance", type=float, default=0.10)
    parser.add_argument("--verbose", action="store_true", help="Show full backtest output")
    args = parser.parse_args()

    rows = load_bl_rows(args.csv)
    if not rows:
        print("No rows with BL data found in CSV.")
        return

    n_markets = len(set(r[1] for r in rows))
    print(f"Loaded {len(rows)} rows with BL data across {n_markets} markets")
    print(f"Config: alpha_up={args.alpha_up} alpha_down={args.alpha_down} "
          f"floor_up={args.floor_up} floor_down={args.floor_down} "
          f"tp={args.tp} trail={args.trail_activation}/{args.trail_distance}")
    print()

    extra_args = [
        "--alpha-up", str(args.alpha_up),
        "--alpha-down", str(args.alpha_down),
        "--floor-up", str(args.floor_up),
        "--floor-down", str(args.floor_down),
        "--tp", str(args.tp),
        "--trail-activation", str(args.trail_activation),
        "--trail-distance", str(args.trail_distance),
    ]

    results = {}
    with tempfile.TemporaryDirectory() as tmpdir:
        for regime in ["mean", "mc", "bl"]:
            tmp_csv = Path(tmpdir) / f"prob_{regime}.csv"
            write_regime_csv(rows, regime, tmp_csv)

            label = {"mean": "Mean (current)", "mc": "MC only", "bl": "BL only"}[regime]
            print(f"Running backtest: {label}...")
            output = run_backtest(tmp_csv, extra_args)

            if args.verbose:
                print(f"\n{'='*60}")
                print(f"  {label}")
                print(f"{'='*60}")
                print(output)

            metrics = extract_summary(output)
            results[regime] = metrics

            if not metrics:
                print(f"  WARNING: Could not parse backtest output for {label}")
                if not args.verbose:
                    print(f"  Output: {output[:500]}")

    # Summary table
    print()
    print("=" * 85)
    print(f"{'Regime':<16} {'Trades':>7} {'Win%':>7} {'P&L':>10} {'Return':>9} {'Sharpe':>8} {'PF':>8}")
    print("-" * 85)
    for regime, label in [("mean", "Mean (current)"), ("mc", "MC only"), ("bl", "BL only")]:
        m = results.get(regime, {})
        trades = m.get("trades", 0)
        wr = m.get("win_rate", 0)
        pnl = m.get("pnl", 0)
        ret = m.get("return_pct", 0)
        sharpe = m.get("sharpe", 0)
        pf = m.get("profit_factor", 0)
        print(f"{label:<16} {trades:>7} {wr:>6.1f}% ${pnl:>9.2f} {ret:>8.1f}% {sharpe:>8.2f} {pf:>8.2f}")
    print("=" * 85)

    # Highlight differences
    mean_trades = results.get("mean", {}).get("trades", 0)
    mc_trades = results.get("mc", {}).get("trades", 0)
    bl_trades = results.get("bl", {}).get("trades", 0)

    print(f"\nTrade count delta vs Mean: MC {mc_trades - mean_trades:+d}, BL {bl_trades - mean_trades:+d}")

    mean_pnl = results.get("mean", {}).get("pnl", 0)
    mc_pnl = results.get("mc", {}).get("pnl", 0)
    bl_pnl = results.get("bl", {}).get("pnl", 0)
    print(f"P&L delta vs Mean:        MC ${mc_pnl - mean_pnl:+.2f}, BL ${bl_pnl - mean_pnl:+.2f}")


if __name__ == "__main__":
    main()
