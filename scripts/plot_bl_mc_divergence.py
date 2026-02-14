#!/usr/bin/env python3
"""Analyse and plot BL vs MC probability divergence from probabilities.csv.

The CSV has 10-column rows (legacy, no BL data) and 14-column rows (with BL).
For the 14-column rows the extra unnamed columns are:
    [10] bl_prob_up   (%)
    [11] bl_prob_down (%)
    [12] avg_prob_up  (%)
    [13] avg_prob_down(%)

avg = (MC + BL) / 2  =>  MC = 2*avg - BL

This script:
  1. Back-derives raw MC probabilities.
  2. Plots BL vs MC over time, coloured by market (barrier).
  3. Plots the divergence (MC − BL) over time.
  4. Prints summary statistics per market.
"""

import argparse
import csv
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

CSV_PATH = Path(__file__).resolve().parent.parent / "results" / "probabilities.csv"


def load(csv_path: Path) -> list[dict]:
    rows = []
    with open(csv_path) as f:
        reader = csv.reader(f)
        header = next(reader)
        for raw in reader:
            if len(raw) < 14:
                continue  # skip legacy rows without BL data
            bl_up = float(raw[10]) if raw[10] != "" else None
            bl_down = float(raw[11]) if raw[11] != "" else None
            avg_up = float(raw[12]) if raw[12] != "" else None
            avg_down = float(raw[13]) if raw[13] != "" else None
            if bl_up is None or avg_up is None:
                continue

            # Back-derive MC: avg = (MC + BL) / 2  =>  MC = 2*avg - BL
            mc_up = 2 * avg_up - bl_up
            mc_down = 2 * avg_down - bl_down

            rows.append({
                "timestamp": datetime.fromisoformat(raw[0]),
                "barrier": float(raw[1]),
                "hours_remaining": float(raw[2]),
                "spot": float(raw[3]),
                "model_up": float(raw[4]),   # avg used for edge (%)
                "model_down": float(raw[5]),
                "poly_up": float(raw[6]),
                "poly_down": float(raw[7]),
                "bl_up": bl_up,
                "bl_down": bl_down,
                "mc_up": mc_up,
                "mc_down": mc_down,
                "avg_up": avg_up,
                "avg_down": avg_down,
            })
    return rows


def split_by_market(rows: list[dict]) -> dict[float, list[dict]]:
    markets: dict[float, list[dict]] = {}
    for r in rows:
        markets.setdefault(r["barrier"], []).append(r)
    return markets


def print_summary(markets: dict[float, list[dict]]) -> None:
    print("=" * 90)
    print(f"{'Barrier':>12}  {'N':>5}  {'MC-BL mean':>10}  {'MC-BL std':>10}  "
          f"{'|div| mean':>10}  {'|div| max':>10}  {'Hours span':>11}")
    print("-" * 90)
    for barrier in sorted(markets.keys()):
        rows = markets[barrier]
        divs = [r["mc_up"] - r["bl_up"] for r in rows]
        abs_divs = [abs(d) for d in divs]
        hours = [r["hours_remaining"] for r in rows]
        print(f"${barrier:>11,.2f}  {len(rows):>5}  {np.mean(divs):>+10.2f}  "
              f"{np.std(divs):>10.2f}  {np.mean(abs_divs):>10.2f}  "
              f"{np.max(abs_divs):>10.2f}  {min(hours):>5.1f}-{max(hours):>4.1f}h")
    print("=" * 90)

    all_divs = [r["mc_up"] - r["bl_up"] for m in markets.values() for r in m]
    print(f"\nOverall: {len(all_divs)} observations")
    print(f"  MC - BL (pp):  mean={np.mean(all_divs):+.2f}  std={np.std(all_divs):.2f}  "
          f"median={np.median(all_divs):+.2f}")
    print(f"  |divergence|:  mean={np.mean(np.abs(all_divs)):.2f}  "
          f"max={np.max(np.abs(all_divs)):.2f}")


def plot(markets: dict[float, list[dict]], out_dir: Path) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=False)
    fig.suptitle("BL vs MC Probability Divergence", fontsize=14, fontweight="bold")
    cmap = plt.cm.tab10

    # --- Panel 1: BL and MC over time ---
    ax = axes[0]
    for i, barrier in enumerate(sorted(markets.keys())):
        rows = markets[barrier]
        ts = [r["timestamp"] for r in rows]
        bl = [r["bl_up"] for r in rows]
        mc = [r["mc_up"] for r in rows]
        c = cmap(i % 10)
        label = f"${barrier:,.0f}"
        ax.plot(ts, mc, color=c, alpha=0.7, linewidth=1.0, label=f"MC {label}")
        ax.plot(ts, bl, color=c, alpha=0.7, linewidth=1.0, linestyle="--", label=f"BL {label}")
    ax.set_ylabel("P(above barrier) %")
    ax.set_title("MC (solid) vs BL (dashed) — P(above barrier)")
    ax.legend(fontsize=7, ncol=4, loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d %H:%M"))
    fig.autofmt_xdate()

    # --- Panel 2: Divergence (MC − BL) over time ---
    ax = axes[1]
    for i, barrier in enumerate(sorted(markets.keys())):
        rows = markets[barrier]
        ts = [r["timestamp"] for r in rows]
        div = [r["mc_up"] - r["bl_up"] for r in rows]
        c = cmap(i % 10)
        ax.plot(ts, div, color=c, alpha=0.7, linewidth=1.0, label=f"${barrier:,.0f}")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_ylabel("MC − BL (pp)")
    ax.set_title("Divergence: MC − BL (percentage points)")
    ax.legend(fontsize=7, ncol=4, loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d %H:%M"))

    # --- Panel 3: Divergence vs hours remaining ---
    ax = axes[2]
    for i, barrier in enumerate(sorted(markets.keys())):
        rows = markets[barrier]
        hrs = [r["hours_remaining"] for r in rows]
        div = [abs(r["mc_up"] - r["bl_up"]) for r in rows]
        c = cmap(i % 10)
        ax.scatter(hrs, div, color=c, alpha=0.4, s=10, label=f"${barrier:,.0f}")
    ax.set_xlabel("Hours remaining to expiry")
    ax.set_ylabel("|MC − BL| (pp)")
    ax.set_title("|Divergence| vs Time Remaining")
    ax.legend(fontsize=7, ncol=4, loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.invert_xaxis()

    plt.tight_layout()
    out_path = out_dir / "bl_mc_divergence.png"
    fig.savefig(out_path, dpi=150)
    print(f"\nPlot saved to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyse BL vs MC divergence")
    parser.add_argument("--csv", type=Path, default=CSV_PATH, help="Path to probabilities.csv")
    parser.add_argument("--no-plot", action="store_true", help="Skip plotting, just print stats")
    args = parser.parse_args()

    rows = load(args.csv)
    if not rows:
        print("No rows with BL data found.")
        return

    markets = split_by_market(rows)
    print(f"\nLoaded {len(rows)} rows with BL data across {len(markets)} markets\n")
    print_summary(markets)

    if not args.no_plot:
        plot(markets, args.csv.parent)


if __name__ == "__main__":
    main()
