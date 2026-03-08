#!/usr/bin/env python3
"""Grid sweep: compare BL-only vs MC-only vs Mean across parameter combos.

Imports run_backtest directly (no subprocess) for speed.
Only uses the ~385 live rows that have BL data.

Usage:
    python scripts/sweep_bl_mc.py
    python scripts/sweep_bl_mc.py --fine    # finer grid
"""

import argparse
import copy
import csv
import sys
from itertools import product
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.backtest import run_backtest, load_csv, Trade, TradeResult

CSV_PATH = Path(__file__).resolve().parent.parent / "results" / "probabilities.csv"


def load_bl_rows(csv_path: Path) -> list[list[str]]:
    """Load raw 14-column rows."""
    rows = []
    with open(csv_path) as f:
        reader = csv.reader(f)
        next(reader)
        for raw in reader:
            if len(raw) >= 14 and raw[10] != "" and raw[12] != "":
                rows.append(raw[:14])
    return rows


HEADER = [
    "timestamp", "barrier_price", "time_remaining_hours", "spot_price",
    "model_prob_up", "model_prob_down", "poly_prob_up", "poly_prob_down",
    "edge_up", "edge_down", "bl_prob_up", "bl_prob_down",
    "avg_prob_up", "avg_prob_down",
]


def build_df(raw_rows: list[list[str]], regime: str) -> list[dict]:
    """Build a list-of-dicts dataframe with model_prob swapped per regime."""
    df = []
    for raw in raw_rows:
        bl_up = float(raw[10])
        bl_down = float(raw[11])
        avg_up = float(raw[12])
        avg_down = float(raw[13])
        mc_up = 2 * avg_up - bl_up
        mc_down = 2 * avg_down - bl_down

        if regime == "bl":
            prob_up, prob_down = bl_up, bl_down
        elif regime == "mc":
            prob_up, prob_down = mc_up, mc_down
        else:
            prob_up, prob_down = avg_up, avg_down

        poly_up = float(raw[6])
        poly_down = float(raw[7])
        edge_up = prob_up / poly_up if poly_up > 0 else 0
        edge_down = prob_down / poly_down if poly_down > 0 else 0

        df.append({
            "timestamp": raw[0],
            "barrier_price": float(raw[1]),
            "time_remaining_hours": float(raw[2]),
            "spot_price": float(raw[3]),
            "model_prob_up": prob_up,
            "model_prob_down": prob_down,
            "poly_prob_up": poly_up,
            "poly_prob_down": poly_down,
            "edge_up": edge_up,
            "edge_down": edge_down,
            "bl_prob_up": bl_up,
            "bl_prob_down": bl_down,
            "avg_prob_up": prob_up,
            "avg_prob_down": prob_down,
        })
    return df


def summarise_trades(trades: list, final_capital: float, capital: float) -> dict:
    n = len(trades)
    wins = sum(1 for t in trades if t.result in (TradeResult.WIN_EXPIRY, TradeResult.TP_FILLED, TradeResult.TRAILING_STOP))
    pnl = final_capital - capital
    return {
        "trades": n,
        "wins": wins,
        "win_rate": wins / n * 100 if n else 0,
        "pnl": pnl,
        "return_pct": pnl / capital * 100,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=CSV_PATH)
    parser.add_argument("--fine", action="store_true", help="Finer grid (more combos)")
    args = parser.parse_args()

    raw_rows = load_bl_rows(args.csv)
    if not raw_rows:
        print("No BL data found.")
        return

    # Pre-build dataframes for each regime (avoids rebuilding per param combo)
    dfs = {}
    for regime in ["mean", "mc", "bl"]:
        dfs[regime] = build_df(raw_rows, regime)

    n_markets = len(set(r[1] for r in raw_rows))
    print(f"Data: {len(raw_rows)} rows, {n_markets} markets\n")

    # Parameter grid
    if args.fine:
        alphas_up = [1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0]
        alphas_down = [1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
        floors_up = [0.50, 0.55, 0.60, 0.65, 0.70]
        floors_down = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45]
    else:
        alphas_up = [1.8, 2.1, 2.4, 2.7, 3.0]
        alphas_down = [1.2, 1.4, 1.6, 1.8, 2.0]
        floors_up = [0.50, 0.55, 0.60, 0.65, 0.70]
        floors_down = [0.25, 0.30, 0.35, 0.40, 0.45]

    tp = 0.30
    trail_act = 0.15
    trail_dist = 0.10
    capital = 100.0
    combos = list(product(alphas_up, alphas_down, floors_up, floors_down))
    print(f"Grid: {len(combos)} parameter combos x 3 regimes = {len(combos)*3} backtests\n")

    all_results = []
    for i, (au, ad, fu, fd) in enumerate(combos):
        row = {"alpha_up": au, "alpha_down": ad, "floor_up": fu, "floor_down": fd}
        for regime in ["mean", "mc", "bl"]:
            trades, final_cap = run_backtest(
                dfs[regime],
                tp_pct=tp,
                friction=0.015,
                capital=capital,
                order_size_pct=0.10,
                latency_minutes=2.0,
                trail_activation=trail_act,
                trail_distance=trail_dist,
                alpha_up=au,
                alpha_down=ad,
                floor_up=fu,
                floor_down=fd,
            )
            s = summarise_trades(trades, final_cap, capital)
            row[f"{regime}_trades"] = s["trades"]
            row[f"{regime}_pnl"] = s["pnl"]
            row[f"{regime}_wr"] = s["win_rate"]
        all_results.append(row)

        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(combos)} done...")

    print(f"\nAll {len(combos)} combos complete.\n")

    # Aggregate statistics
    mc_better = 0
    bl_better = 0
    mean_better = 0
    mc_worse = 0
    bl_worse = 0
    mc_pnl_deltas = []
    bl_pnl_deltas = []
    mc_trade_deltas = []
    bl_trade_deltas = []

    for r in all_results:
        mc_d = r["mc_pnl"] - r["mean_pnl"]
        bl_d = r["bl_pnl"] - r["mean_pnl"]
        mc_pnl_deltas.append(mc_d)
        bl_pnl_deltas.append(bl_d)
        mc_trade_deltas.append(r["mc_trades"] - r["mean_trades"])
        bl_trade_deltas.append(r["bl_trades"] - r["mean_trades"])

        if mc_d > 0.01: mc_better += 1
        if mc_d < -0.01: mc_worse += 1
        if bl_d > 0.01: bl_better += 1
        if bl_d < -0.01: bl_worse += 1

    n = len(all_results)
    mc_same = n - mc_better - mc_worse
    bl_same = n - bl_better - bl_worse

    print("=" * 80)
    print("SUMMARY: How often does each regime beat/match/lose vs Mean?")
    print("=" * 80)
    print(f"{'':>20} {'Better':>10} {'Same':>10} {'Worse':>10}")
    print("-" * 80)
    print(f"{'MC only':>20} {mc_better:>10} ({mc_better/n*100:.0f}%) {mc_same:>5} ({mc_same/n*100:.0f}%) {mc_worse:>5} ({mc_worse/n*100:.0f}%)")
    print(f"{'BL only':>20} {bl_better:>10} ({bl_better/n*100:.0f}%) {bl_same:>5} ({bl_same/n*100:.0f}%) {bl_worse:>5} ({bl_worse/n*100:.0f}%)")
    print()
    print(f"MC P&L delta vs Mean:  mean={np.mean(mc_pnl_deltas):+.2f}  median={np.median(mc_pnl_deltas):+.2f}  "
          f"min={np.min(mc_pnl_deltas):+.2f}  max={np.max(mc_pnl_deltas):+.2f}")
    print(f"BL P&L delta vs Mean:  mean={np.mean(bl_pnl_deltas):+.2f}  median={np.median(bl_pnl_deltas):+.2f}  "
          f"min={np.min(bl_pnl_deltas):+.2f}  max={np.max(bl_pnl_deltas):+.2f}")
    print()
    print(f"MC trade count delta:  mean={np.mean(mc_trade_deltas):+.2f}  "
          f"range=[{np.min(mc_trade_deltas):+d}, {np.max(mc_trade_deltas):+d}]")
    print(f"BL trade count delta:  mean={np.mean(bl_trade_deltas):+.2f}  "
          f"range=[{np.min(bl_trade_deltas):+d}, {np.max(bl_trade_deltas):+d}]")

    # Distribution of P&L by regime
    mean_pnls = [r["mean_pnl"] for r in all_results]
    mc_pnls = [r["mc_pnl"] for r in all_results]
    bl_pnls = [r["bl_pnl"] for r in all_results]

    print()
    print("=" * 80)
    print("P&L DISTRIBUTION ACROSS ALL PARAMETER COMBOS")
    print("=" * 80)
    for label, pnls in [("Mean", mean_pnls), ("MC", mc_pnls), ("BL", bl_pnls)]:
        print(f"  {label:<6}  mean=${np.mean(pnls):+.2f}  median=${np.median(pnls):+.2f}  "
              f"std=${np.std(pnls):.2f}  min=${np.min(pnls):+.2f}  max=${np.max(pnls):+.2f}")

    # Show worst MC-only cases
    worst_mc = sorted(all_results, key=lambda r: r["mc_pnl"] - r["mean_pnl"])[:10]
    print()
    print("=" * 80)
    print("TOP 10 WORST MC-ONLY CASES (vs Mean)")
    print("=" * 80)
    print(f"{'a_up':>5} {'a_dn':>5} {'f_up':>5} {'f_dn':>5}  "
          f"{'Mean$':>8} {'MC$':>8} {'BL$':>8}  {'MC-Mean':>8} {'Trades M/MC/BL':>15}")
    print("-" * 80)
    for r in worst_mc:
        print(f"{r['alpha_up']:>5.1f} {r['alpha_down']:>5.1f} {r['floor_up']:>5.2f} {r['floor_down']:>5.2f}  "
              f"{r['mean_pnl']:>+8.2f} {r['mc_pnl']:>+8.2f} {r['bl_pnl']:>+8.2f}  "
              f"{r['mc_pnl']-r['mean_pnl']:>+8.2f} {r['mean_trades']:>4}/{r['mc_trades']:>2}/{r['bl_trades']:>2}")

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("P&L by Regime Across Parameter Combos", fontsize=13, fontweight="bold")

    # Histogram of P&L
    ax = axes[0]
    bins = np.linspace(min(min(mean_pnls), min(mc_pnls), min(bl_pnls)) - 1,
                       max(max(mean_pnls), max(mc_pnls), max(bl_pnls)) + 1, 30)
    ax.hist(mean_pnls, bins=bins, alpha=0.5, label="Mean", color="tab:green")
    ax.hist(mc_pnls, bins=bins, alpha=0.5, label="MC", color="tab:blue")
    ax.hist(bl_pnls, bins=bins, alpha=0.5, label="BL", color="tab:orange")
    ax.set_xlabel("P&L ($)")
    ax.set_ylabel("Count")
    ax.set_title("P&L Distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Scatter: MC delta vs BL delta
    ax = axes[1]
    ax.scatter(mc_pnl_deltas, bl_pnl_deltas, alpha=0.3, s=15, c="tab:purple")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.axvline(0, color="black", linewidth=0.5)
    ax.set_xlabel("MC P&L - Mean P&L ($)")
    ax.set_ylabel("BL P&L - Mean P&L ($)")
    ax.set_title("P&L Delta: MC vs BL (relative to Mean)")
    ax.grid(True, alpha=0.3)

    # Trade count comparison
    ax = axes[2]
    x = range(n)
    mean_t = [r["mean_trades"] for r in all_results]
    mc_t = [r["mc_trades"] for r in all_results]
    bl_t = [r["bl_trades"] for r in all_results]
    ax.plot(sorted(mean_t), label="Mean", color="tab:green", alpha=0.7)
    ax.plot(sorted(mc_t), label="MC", color="tab:blue", alpha=0.7)
    ax.plot(sorted(bl_t), label="BL", color="tab:orange", alpha=0.7)
    ax.set_xlabel("Parameter combo (sorted by trade count)")
    ax.set_ylabel("Trades")
    ax.set_title("Trade Count Distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = Path(args.csv).parent / "sweep_bl_mc.png"
    fig.savefig(out_path, dpi=150)
    print(f"\nPlot saved to {out_path}")


if __name__ == "__main__":
    main()
