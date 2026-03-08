#!/usr/bin/env python3
"""
Test script to fit Heston model to all available expiries and generate IV smile plots.

This script:
1. Fetches option data for all expiries from Deribit
2. Filters the data (OI, spread, moneyness)
3. Fits Heston model to each expiry using the new TTM-adaptive bounds
4. Generates IV smile fit plots comparing market vs model IVs
5. Saves all plots to a specific folder

Usage:
    python scripts/test_heston_all_expiries.py [--output-dir OUTPUT_DIR]
"""

__test__ = False  # Manual script, not a pytest test module.

import sys
from pathlib import Path
from typing import List

import numpy as np
import matplotlib.pyplot as plt

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from btc_pricer.config import Config
from btc_pricer.models.heston import (
    HestonFitter, HestonModel, HestonParams, HestonFitResult, QUANTLIB_AVAILABLE
)
from btc_pricer.cli.common import create_heston_fitter
from btc_pricer.constants import BOUNDARY_CHECK_TOLERANCE

from test_model_base import (
    TestHarness,
    FitResult,
    setup_logging,
    create_base_argument_parser,
)


class HestonTestHarness(TestHarness):
    """Test harness for Heston model fitting."""

    @property
    def model_name(self) -> str:
        return "Heston"

    def create_fitter(self, config: Config) -> HestonFitter:
        """Create Heston fitter from configuration."""
        return create_heston_fitter(config)

    def fit_model(
        self,
        log_moneyness: np.ndarray,
        market_iv: np.ndarray,
        ttm: float,
        forward: float
    ):
        """Fit Heston model to market data."""
        fit_result = self.fitter.fit(log_moneyness, market_iv, ttm, forward=forward)

        if not fit_result.success or fit_result.params is None:
            self.logger.warning(f"Fit failed - {fit_result.message}")
            return None, None

        params = fit_result.params
        self.logger.info(
            f"v0={params.v0:.4f}, kappa={params.kappa:.3f}, "
            f"theta={params.theta:.4f}, xi={params.xi:.3f}, rho={params.rho:.3f}, "
            f"R²={fit_result.r_squared:.4f}"
        )

        return params, fit_result

    def get_bounds_for_ttm(self, ttm: float) -> dict:
        """Get Heston parameter bounds for given TTM."""
        return self.fitter._get_ttm_adjusted_bounds(ttm)

    def check_boundary_issues(self, params: HestonParams, bounds: dict) -> List[str]:
        """Check if Heston parameters hit bounds."""
        issues = []
        tol = BOUNDARY_CHECK_TOLERANCE

        if abs(params.kappa - bounds['kappa'][0]) < tol:
            issues.append("kappa_lb")
        if abs(params.kappa - bounds['kappa'][1]) < tol:
            issues.append("kappa_ub")
        if abs(params.xi - bounds['xi'][0]) < tol:
            issues.append("xi_lb")
        if abs(params.xi - bounds['xi'][1]) < tol:
            issues.append("xi_ub")

        return issues

    def plot_fit(
        self,
        log_moneyness: np.ndarray,
        market_iv: np.ndarray,
        params: HestonParams,
        fit_result: HestonFitResult,
        expiry: str,
        ttm: float,
        forward: float,
        spot: float,
        save_path: Path,
        bounds: dict
    ) -> None:
        """Generate IV smile plot with Heston fit diagnostics."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Left plot: IV smile
        ax1.scatter(
            log_moneyness * 100,
            market_iv * 100,
            s=80,
            alpha=0.8,
            label='Market IV',
            zorder=5,
            edgecolors='black',
            linewidths=0.5
        )

        # Model fit curve
        k_fine = np.linspace(log_moneyness.min() * 1.1, log_moneyness.max() * 1.1, 200)
        model = HestonModel(params, use_quantlib=True)
        model_iv = model.implied_volatility_array(k_fine)

        ax1.plot(
            k_fine * 100,
            model_iv * 100,
            'r-',
            linewidth=2.5,
            label='Heston Fit'
        )

        # Compute residuals
        model_iv_at_points = model.implied_volatility_array(log_moneyness)
        residuals = (market_iv - model_iv_at_points) * 100

        ax1.axvline(0, color='gray', linestyle='--', alpha=0.5, label='ATM')
        ax1.set_xlabel("Log Moneyness (%)", fontsize=12)
        ax1.set_ylabel("Implied Volatility (%)", fontsize=12)

        # Check if parameters hit bounds
        boundary_warnings = []
        tol = BOUNDARY_CHECK_TOLERANCE
        if abs(params.kappa - bounds['kappa'][0]) < tol:
            boundary_warnings.append(f"kappa at lower bound ({bounds['kappa'][0]:.3f})")
        if abs(params.kappa - bounds['kappa'][1]) < tol:
            boundary_warnings.append(f"kappa at upper bound ({bounds['kappa'][1]:.1f})")
        if abs(params.xi - bounds['xi'][0]) < tol:
            boundary_warnings.append(f"xi at lower bound ({bounds['xi'][0]:.1f})")
        if abs(params.xi - bounds['xi'][1]) < tol:
            boundary_warnings.append(f"xi at upper bound ({bounds['xi'][1]:.1f})")

        title_color = 'red' if boundary_warnings else 'black'
        boundary_text = f"\n[!] {', '.join(boundary_warnings)}" if boundary_warnings else ""

        ax1.set_title(
            f"IV Smile: {expiry} (TTM: {ttm*365:.1f} days, {len(log_moneyness)} pts)\n"
            f"v0={params.v0:.4f}, kappa={params.kappa:.3f}, "
            f"theta={params.theta:.4f}, xi={params.xi:.3f}, rho={params.rho:.3f}"
            f"{boundary_text}",
            fontsize=11,
            color=title_color
        )
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)

        # Right plot: Residuals
        colors = ['green' if r > 0 else 'red' for r in residuals]
        ax2.bar(log_moneyness * 100, residuals, color=colors, alpha=0.7, width=1.5)
        ax2.axhline(0, color='black', linewidth=1)

        ax2.set_xlabel("Log Moneyness (%)", fontsize=12)
        ax2.set_ylabel("Residual (Market - Model) (%)", fontsize=12)
        ax2.set_title(
            f"Fit Residuals\n"
            f"R²={fit_result.r_squared:.4f}, RMSE={fit_result.rmse*100:.2f}%, "
            f"Max Error={fit_result.max_residual*100:.2f}%",
            fontsize=11
        )
        ax2.grid(True, alpha=0.3)

        # Add info text box
        info_text = (
            f"Forward: ${forward:,.0f}\n"
            f"Spot: ${spot:,.0f}\n"
            f"Feller: {'Y' if params.feller_condition() else 'N'} "
            f"({params.feller_ratio():.2f})"
        )
        ax2.text(
            0.02, 0.98, info_text,
            transform=ax2.transAxes,
            fontsize=9,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        )

        plt.tight_layout()
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

    def print_summary_header(self) -> None:
        """Print Heston summary table header."""
        print("\n" + "=" * 100)
        print("HESTON FIT SUMMARY")
        print("=" * 100)
        print(f"{'Expiry':<10} {'TTM':>8} {'Cat':>6} {'Pts':>4} {'v0':>8} {'kappa':>8} "
              f"{'theta':>8} {'xi':>6} {'rho':>7} {'R²':>7} {'Feller':>6} {'Issues'}")
        print("-" * 100)

    def print_summary_row(self, result: FitResult) -> None:
        """Print a single row of the Heston summary table."""
        params = result.params
        feller = 'Y' if params.feller_condition() else 'N'
        issues = ','.join(result.boundary_issues) if result.boundary_issues else '-'

        print(
            f"{result.expiry:<10} {result.ttm_days:>7.1f}d {result.ttm_category:>6} "
            f"{result.n_points:>4} {params.v0:>8.4f} {params.kappa:>8.3f} "
            f"{params.theta:>8.4f} {params.xi:>6.2f} {params.rho:>7.3f} "
            f"{result.r_squared:>7.4f} {feller:>6} {issues}"
        )


def main():
    parser = create_base_argument_parser("Test Heston fitting for all expiries")
    parser.set_defaults(output_dir="results/heston_test_fits")
    args = parser.parse_args()

    setup_logging(args.verbose)

    # Load config
    config = Config()
    print(f"QuantLib available: {QUANTLIB_AVAILABLE}")
    print(f"Multi-start enabled: {config.heston.use_multi_start}, n_starts={config.heston.n_starts}")
    print(f"Short-dated TTM threshold: {config.heston.short_dated_ttm_threshold}")

    # Run test harness
    harness = HestonTestHarness(
        output_dir=args.output_dir,
        min_points=args.min_points,
        verbose=args.verbose
    )

    try:
        harness.run(config)
        harness.print_summary()
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
