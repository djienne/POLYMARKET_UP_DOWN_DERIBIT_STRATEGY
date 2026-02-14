#!/usr/bin/env python3
"""Compare entry signals: BL-only vs MC-only vs Mean (current approach).

For each row with BL data, back-derive raw MC probability then replay the
edge function under three regimes to see which would have triggered entries
and which would not.

Usage:
    python scripts/plot_entry_comparison.py
    python scripts/plot_entry_comparison.py --no-plot
"""

import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

# Add project root so we can import the edge function
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from btc_pricer.edge import has_edge, required_model_prob

CSV_PATH = Path(__file__).resolve().parent.parent / "results" / "probabilities.csv"

# Live config from config_dry_run.json
ALPHA_UP = 2.40
ALPHA_DOWN = 1.60
FLOOR_UP = 0.60
FLOOR_DOWN = 0.35


def load(csv_path: Path) -> list[dict]:
    rows = []
    with open(csv_path) as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for raw in reader:
            if len(raw) < 14:
                continue
            bl_up = float(raw[10]) if raw[10] != "" else None
            bl_down = float(raw[11]) if raw[11] != "" else None
            avg_up = float(raw[12]) if raw[12] != "" else None
            avg_down = float(raw[13]) if raw[13] != "" else None
            if bl_up is None or avg_up is None:
                continue

            # All values in CSV are percentages — convert to 0-1 fractions
            mc_up = (2 * avg_up - bl_up) / 100
            mc_down = (2 * avg_down - bl_down) / 100
            bl_up_f = bl_up / 100
            bl_down_f = bl_down / 100
            avg_up_f = avg_up / 100
            avg_down_f = avg_down / 100
            poly_up = float(raw[6]) / 100
            poly_down = float(raw[7]) / 100

            rows.append({
                "timestamp": datetime.fromisoformat(raw[0]),
                "barrier": float(raw[1]),
                "hours": float(raw[2]),
                "spot": float(raw[3]),
                "poly_up": poly_up,
                "poly_down": poly_down,
                "mc_up": mc_up,
                "mc_down": mc_down,
                "bl_up": bl_up_f,
                "bl_down": bl_down_f,
                "avg_up": avg_up_f,
                "avg_down": avg_down_f,
            })
    return rows


def check_entry(model_up, model_down, poly_up, poly_down):
    """Return (direction, edge) or (None, 0) using the live config edge curve."""
    up_ok = has_edge(model_up, poly_up, ALPHA_UP, FLOOR_UP)
    down_ok = has_edge(model_down, poly_down, ALPHA_DOWN, FLOOR_DOWN)
    edge_up = model_up / poly_up if poly_up > 0 else 0
    edge_down = model_down / poly_down if poly_down > 0 else 0

    if up_ok and down_ok:
        if edge_up >= edge_down:
            return "UP", edge_up
        return "DOWN", edge_down
    if up_ok:
        return "UP", edge_up
    if down_ok:
        return "DOWN", edge_down
    return None, 0


def analyse(rows):
    """Run entry check under all three regimes for every row."""
    results = []
    for r in rows:
        mean_dir, mean_edge = check_entry(r["avg_up"], r["avg_down"], r["poly_up"], r["poly_down"])
        mc_dir, mc_edge = check_entry(r["mc_up"], r["mc_down"], r["poly_up"], r["poly_down"])
        bl_dir, bl_edge = check_entry(r["bl_up"], r["bl_down"], r["poly_up"], r["poly_down"])

        results.append({
            **r,
            "mean_dir": mean_dir, "mean_edge": mean_edge,
            "mc_dir": mc_dir, "mc_edge": mc_edge,
            "bl_dir": bl_dir, "bl_edge": bl_edge,
        })
    return results


def print_summary(results):
    total = len(results)
    mean_entries = [r for r in results if r["mean_dir"]]
    mc_entries = [r for r in results if r["mc_dir"]]
    bl_entries = [r for r in results if r["bl_dir"]]

    print(f"\nConfig: ALPHA_UP={ALPHA_UP} ALPHA_DOWN={ALPHA_DOWN} FLOOR_UP={FLOOR_UP} FLOOR_DOWN={FLOOR_DOWN}")
    print(f"Total observations: {total}\n")

    print("=" * 80)
    print(f"{'Regime':<12} {'Entries':>8} {'% obs':>8}  {'UP':>6} {'DOWN':>6}  {'Avg edge':>10}")
    print("-" * 80)
    for label, entries in [("Mean (cur)", mean_entries), ("MC only", mc_entries), ("BL only", bl_entries)]:
        n = len(entries)
        n_up = sum(1 for e in entries if e[f"{label.split()[0].lower()}_dir" if label != "Mean (cur)" else "mean_dir"] == "UP")
        key = "mean" if label == "Mean (cur)" else label.split()[0].lower()
        n_up = sum(1 for e in entries if e[f"{key}_dir"] == "UP")
        n_down = n - n_up
        avg_e = np.mean([e[f"{key}_edge"] for e in entries]) if entries else 0
        print(f"{label:<12} {n:>8} {n/total*100:>7.1f}%  {n_up:>6} {n_down:>6}  {avg_e:>10.2f}x")
    print("=" * 80)

    # Disagreements
    print("\n--- Disagreements ---\n")

    # Cases where mean triggers but MC/BL wouldn't
    mc_miss = [r for r in results if r["mean_dir"] and not r["mc_dir"]]
    bl_miss = [r for r in results if r["mean_dir"] and not r["bl_dir"]]
    mc_extra = [r for r in results if not r["mean_dir"] and r["mc_dir"]]
    bl_extra = [r for r in results if not r["mean_dir"] and r["bl_dir"]]
    mc_bl_disagree = [r for r in results if r["mc_dir"] != r["bl_dir"]]
    dir_disagree = [r for r in results if r["mc_dir"] and r["bl_dir"] and r["mc_dir"] != r["bl_dir"]]

    print(f"Mean triggers but MC would NOT:  {len(mc_miss):>4}  (mean catches, MC misses)")
    print(f"Mean triggers but BL would NOT:  {len(bl_miss):>4}  (mean catches, BL misses)")
    print(f"MC triggers but Mean would NOT:  {len(mc_extra):>4}  (MC extra entries)")
    print(f"BL triggers but Mean would NOT:  {len(bl_extra):>4}  (BL extra entries)")
    print(f"MC and BL disagree (any):        {len(mc_bl_disagree):>4}")
    print(f"MC and BL both trigger, opposite dir: {len(dir_disagree):>3}")

    # Detail the disagreements
    if mc_miss or bl_miss or mc_extra or bl_extra:
        print("\n--- Detailed disagreements (first 20) ---\n")
        print(f"{'Timestamp':<22} {'Barrier':>10} {'Hrs':>5} "
              f"{'Mean':>8} {'MC':>8} {'BL':>8} "
              f"{'MC-BL up':>9} {'Poly up':>8} {'Avg up':>8}")
        print("-" * 105)

        seen = set()
        all_disagree = []
        for r in sorted(mc_miss + bl_miss + mc_extra + bl_extra, key=lambda r: r["timestamp"]):
            key = (r["timestamp"], r["barrier"])
            if key not in seen:
                seen.add(key)
                all_disagree.append(r)
        for r in all_disagree[:20]:
            ts = r["timestamp"].strftime("%m/%d %H:%M")
            div = (r["mc_up"] - r["bl_up"]) * 100
            m_d = r["mean_dir"] or "-"
            mc_d = r["mc_dir"] or "-"
            bl_d = r["bl_dir"] or "-"
            print(f"{ts:<22} ${r['barrier']:>9,.0f} {r['hours']:>5.1f} "
                  f"{m_d:>8} {mc_d:>8} {bl_d:>8} "
                  f"{div:>+8.1f}pp {r['poly_up']*100:>7.1f}% {r['avg_up']*100:>7.1f}%")

    # Per-market breakdown
    markets = sorted(set(r["barrier"] for r in results))
    print("\n--- Per-market breakdown ---\n")
    print(f"{'Barrier':>12} {'N':>5}  {'Mean':>6} {'MC':>6} {'BL':>6}  "
          f"{'MC miss':>8} {'BL miss':>8} {'MC extra':>8} {'BL extra':>8}")
    print("-" * 95)
    for b in markets:
        mr = [r for r in results if r["barrier"] == b]
        n = len(mr)
        n_mean = sum(1 for r in mr if r["mean_dir"])
        n_mc = sum(1 for r in mr if r["mc_dir"])
        n_bl = sum(1 for r in mr if r["bl_dir"])
        mc_m = sum(1 for r in mr if r["mean_dir"] and not r["mc_dir"])
        bl_m = sum(1 for r in mr if r["mean_dir"] and not r["bl_dir"])
        mc_e = sum(1 for r in mr if not r["mean_dir"] and r["mc_dir"])
        bl_e = sum(1 for r in mr if not r["mean_dir"] and r["bl_dir"])
        print(f"${b:>11,.2f} {n:>5}  {n_mean:>6} {n_mc:>6} {n_bl:>6}  "
              f"{mc_m:>8} {bl_m:>8} {mc_e:>8} {bl_e:>8}")


def plot(results, out_dir: Path):
    markets = sorted(set(r["barrier"] for r in results))
    n_markets = len(markets)
    fig, axes = plt.subplots(n_markets, 1, figsize=(14, 4 * n_markets), squeeze=False)
    fig.suptitle("Entry Signals: Mean vs MC-only vs BL-only", fontsize=14, fontweight="bold")

    for idx, barrier in enumerate(markets):
        ax = axes[idx, 0]
        mr = [r for r in results if r["barrier"] == barrier]
        ts = [r["timestamp"] for r in mr]

        # Plot required prob curve (UP direction) for reference
        req_up = [required_model_prob(r["poly_up"], ALPHA_UP, FLOOR_UP) * 100 for r in mr]
        ax.plot(ts, req_up, color="black", linewidth=1.0, linestyle=":", alpha=0.5, label="Required (UP)")

        # Plot the three model probs (UP direction)
        ax.plot(ts, [r["avg_up"] * 100 for r in mr], color="tab:green", linewidth=1.2, label="Mean (UP)")
        ax.plot(ts, [r["mc_up"] * 100 for r in mr], color="tab:blue", linewidth=1.0, alpha=0.7, label="MC (UP)")
        ax.plot(ts, [r["bl_up"] * 100 for r in mr], color="tab:orange", linewidth=1.0, alpha=0.7, linestyle="--", label="BL (UP)")

        # Mark entry signals
        for r in mr:
            t = r["timestamp"]
            if r["mean_dir"] == "UP":
                ax.axvline(t, color="tab:green", alpha=0.15, linewidth=3)
            if r["mc_dir"] == "UP" and not r["mean_dir"]:
                ax.axvline(t, color="tab:blue", alpha=0.3, linewidth=2)
            if r["bl_dir"] == "UP" and not r["mean_dir"]:
                ax.axvline(t, color="tab:orange", alpha=0.3, linewidth=2)

        ax.set_ylabel("Probability %")
        ax.set_title(f"Barrier ${barrier:,.0f}")
        ax.legend(fontsize=7, ncol=5, loc="upper left")
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d %H:%M"))

    plt.tight_layout()
    out_path = out_dir / "entry_comparison.png"
    fig.savefig(out_path, dpi=150)
    print(f"\nPlot saved to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Compare entry signals: BL vs MC vs Mean")
    parser.add_argument("--csv", type=Path, default=CSV_PATH)
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    rows = load(args.csv)
    if not rows:
        print("No rows with BL data found.")
        return

    results = analyse(rows)
    print_summary(results)

    if not args.no_plot:
        plot(results, args.csv.parent)


if __name__ == "__main__":
    main()
