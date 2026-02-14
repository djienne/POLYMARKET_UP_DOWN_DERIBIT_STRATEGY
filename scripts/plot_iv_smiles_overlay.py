#!/usr/bin/env python3
"""
Plot IV smiles for multiple expiries overlaid on a single figure.

Fetches live Deribit options data, fits Heston/SSVI models for each expiry,
and plots all IV smiles together with different colors.

Usage:
    python scripts/plot_iv_smiles_overlay.py
    python scripts/plot_iv_smiles_overlay.py --expiries 31JAN26 6FEB26 13FEB26
    python scripts/plot_iv_smiles_overlay.py --model ssvi
    python scripts/plot_iv_smiles_overlay.py --output results/iv_smiles_combined.png
"""

import argparse
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from btc_pricer.config import Config
from btc_pricer.api.deribit import DeribitClient
from btc_pricer.data.filters import DataFilter
from btc_pricer.models.ssvi import SSVIModel, SSVIParams
from btc_pricer.models.heston import HestonModel, HestonParams
from btc_pricer.utils.sanity_checks import SanityChecker
from btc_pricer.cli.common import (
    setup_logging,
    load_config,
    extract_surface_data,
    parse_expiry_date,
)
from cli import process_expiry


def main():
    parser = argparse.ArgumentParser(
        description="Plot IV smiles for multiple expiries on a single figure"
    )
    parser.add_argument(
        "--config", type=Path, default=Path("config.yaml"), help="Config file"
    )
    parser.add_argument(
        "--expiries", nargs="+", help="Specific expiries to include"
    )
    parser.add_argument(
        "--model", choices=["heston", "ssvi"], default=None,
        help="Force a specific model (default: auto-select per config)"
    )
    parser.add_argument(
        "--output", type=Path, default=Path("results/iv_smiles_overlay.png"),
        help="Output file path"
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

    config = load_config(args.config, logger)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Fetch data
    client = DeribitClient(config.api, config.validation)
    data_filter = DataFilter(config.filters)
    checker = SanityChecker(strict=False, validation_config=config.validation)

    logger.info("Fetching options data from Deribit...")
    options_by_expiry = client.fetch_all_options("BTC")
    if not options_by_expiry:
        logger.error("No options data received")
        sys.exit(1)

    # Filter expiries
    if args.expiries:
        options_by_expiry = {
            k: v for k, v in options_by_expiry.items() if k in args.expiries
        }

    sorted_expiries = sorted(options_by_expiry.keys(), key=parse_expiry_date)
    logger.info(f"Processing {len(sorted_expiries)} expiries: {sorted_expiries}")

    # Process each expiry: fit model and collect IV data
    smile_data = []  # List of (expiry, log_moneyness, market_iv, model_iv, model_name, ttm)

    for expiry in sorted_expiries:
        options = options_by_expiry[expiry]
        filtered, _ = data_filter.filter_options(options, return_stats=True)

        result = process_expiry(expiry, filtered, config, checker, args.model)
        if result is None:
            logger.warning(f"Skipping {expiry}: processing failed")
            continue

        params, rnd, log_k, mkt_iv, model_used, extra_info = result

        # Compute model IV curve
        k_fine = np.linspace(log_k.min(), log_k.max(), 200)
        if isinstance(params, HestonParams):
            model = HestonModel(params, use_quantlib=config.heston.use_quantlib)
        else:
            model = SSVIModel(params)
        model_iv = model.implied_volatility_array(k_fine)

        ttm = params.ttm
        smile_data.append((expiry, log_k, mkt_iv, k_fine, model_iv, model_used, ttm))
        logger.info(f"Processed {expiry} ({model_used.upper()}, TTM={ttm*365:.1f}d)")

    if not smile_data:
        logger.error("No expiries processed successfully")
        sys.exit(1)

    # Sort by TTM
    smile_data.sort(key=lambda x: x[6])

    # Plot
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        plt.style.use("seaborn-whitegrid")

    fig, ax = plt.subplots(figsize=(14, 9))

    colors = plt.cm.turbo(np.linspace(0.1, 0.9, len(smile_data)))

    for i, (expiry, log_k, mkt_iv, k_fine, model_iv, model_used, ttm) in enumerate(smile_data):
        color = colors[i]
        ttm_days = ttm * 365
        label = f"{expiry} ({model_used.upper()}, {ttm_days:.0f}d)"

        # Model fit line
        ax.plot(k_fine * 100, model_iv * 100, color=color, linewidth=2, label=label)

        # Market data points
        ax.scatter(log_k * 100, mkt_iv * 100, color=color, s=30, alpha=0.6,
                   edgecolors='black', linewidths=0.5, zorder=5)

    ax.axvline(0, color='gray', linestyle='--', alpha=0.5, label='ATM')

    ax.set_xlabel("Log Moneyness (%)", fontsize=13)
    ax.set_ylabel("Implied Volatility (%)", fontsize=13)
    ax.set_title("IV Smiles Across Expiries", fontsize=15)
    ax.legend(loc='upper right', fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(args.output, dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f"Saved overlay plot to {args.output}")
    print(f"Plotted {len(smile_data)} expiries: {[s[0] for s in smile_data]}")


if __name__ == "__main__":
    main()
