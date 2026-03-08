#!/usr/bin/env python3
"""
Test script to fit SSVI model to all available expiries and generate IV smile plots.

This script:
1. Fetches option data for all expiries from Deribit
2. Filters the data (OI, spread, moneyness)
3. Fits SSVI model to each expiry
4. Generates IV smile fit plots comparing market vs model IVs
5. Saves all plots to a specific folder

Usage:
    python scripts/test_ssvi_all_expiries.py [--output-dir OUTPUT_DIR]
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
from btc_pricer.models.ssvi import SSVIFitter, SSVIModel, SSVIParams, SSVIFitResult
from btc_pricer.cli.common import create_ssvi_fitter
from btc_pricer.constants import BOUNDARY_CHECK_TOLERANCE

from test_model_base import (
    TestHarness,
    FitResult,
    setup_logging,
    create_base_argument_parser,
)


class SSVITestHarness(TestHarness):
    """Test harness for SSVI model fitting."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = None  # Store config for TTM category lookups

    @property
    def model_name(self) -> str:
        return "SSVI"

    def create_fitter(self, config: Config) -> SSVIFitter:
        """Create SSVI fitter from configuration."""
        self.config = config  # Store for phi bounds lookup
        return create_ssvi_fitter(config)

    def fit_model(
        self,
        log_moneyness: np.ndarray,
        market_iv: np.ndarray,
        ttm: float,
        forward: float
    ):
        """Fit SSVI model to market data."""
        fit_result = self.fitter.fit(log_moneyness, market_iv, ttm)

        if not fit_result.success or fit_result.params is None:
            self.logger.warning(f"Fit failed - {fit_result.message}")
            return None, None

        params = fit_result.params
        self.logger.info(
            f"theta={params.theta:.6f}, rho={params.rho:.3f}, phi={params.phi:.3f}, "
            f"R²={fit_result.r_squared:.4f}"
        )

        return params, fit_result

    def get_bounds_for_ttm(self, ttm: float) -> dict:
        """Get SSVI parameter bounds for given TTM."""
        if self.config is None:
            return {'phi': (0.001, 5.0)}

        # Get phi bounds based on TTM
        very_short_ttm = getattr(self.config.ssvi, 'very_short_dated_ttm_threshold', 0.02)
        very_short_phi = getattr(self.config.ssvi, 'very_short_dated_phi_bounds', (0.001, 100.0))

        if ttm < very_short_ttm:
            phi_bounds = very_short_phi
        elif ttm < self.config.ssvi.short_dated_ttm_threshold:
            phi_bounds = self.config.ssvi.short_dated_phi_bounds
        else:
            phi_bounds = self.config.ssvi.phi_bounds

        return {'phi': phi_bounds}

    def get_ttm_category(self, ttm: float, config: Config) -> str:
        """Determine TTM category for SSVI."""
        very_short = getattr(config.ssvi, 'very_short_dated_ttm_threshold', 0.02)
        short = config.ssvi.short_dated_ttm_threshold

        if ttm < very_short:
            return "very-short"
        elif ttm < short:
            return "short"
        return "normal"

    def check_boundary_issues(self, params: SSVIParams, bounds: dict) -> List[str]:
        """Check if SSVI parameters hit bounds."""
        issues = []
        tol = BOUNDARY_CHECK_TOLERANCE
        phi_bounds = bounds.get('phi', (0.001, 5.0))

        if abs(params.phi - phi_bounds[0]) < tol:
            issues.append("phi_lb")
        if abs(params.phi - phi_bounds[1]) < tol:
            issues.append("phi_ub")

        return issues

    def plot_fit(
        self,
        log_moneyness: np.ndarray,
        market_iv: np.ndarray,
        params: SSVIParams,
        fit_result: SSVIFitResult,
        expiry: str,
        ttm: float,
        forward: float,
        spot: float,
        save_path: Path,
        bounds: dict
    ) -> None:
        """Generate IV smile plot with SSVI fit diagnostics."""
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
        model = SSVIModel(params)
        model_iv = model.implied_volatility_array(k_fine)

        ax1.plot(
            k_fine * 100,
            model_iv * 100,
            'b-',
            linewidth=2.5,
            label='SSVI Fit'
        )

        # Compute residuals
        model_iv_at_points = model.implied_volatility_array(log_moneyness)
        residuals = (market_iv - model_iv_at_points) * 100

        ax1.axvline(0, color='gray', linestyle='--', alpha=0.5, label='ATM')
        ax1.set_xlabel("Log Moneyness (%)", fontsize=12)
        ax1.set_ylabel("Implied Volatility (%)", fontsize=12)

        # Check if phi hit bounds
        boundary_warnings = []
        tol = BOUNDARY_CHECK_TOLERANCE
        phi_bounds = bounds.get('phi', (0.001, 5.0))

        if abs(params.phi - phi_bounds[0]) < tol:
            boundary_warnings.append(f"phi at lower bound ({phi_bounds[0]:.3f})")
        if abs(params.phi - phi_bounds[1]) < tol:
            boundary_warnings.append(f"phi at upper bound ({phi_bounds[1]:.1f})")

        title_color = 'red' if boundary_warnings else 'black'
        boundary_text = f"\n[!] {', '.join(boundary_warnings)}" if boundary_warnings else ""

        ax1.set_title(
            f"IV Smile (SSVI): {expiry} (TTM: {ttm*365:.1f} days, {len(log_moneyness)} pts)\n"
            f"theta={params.theta:.6f}, rho={params.rho:.3f}, phi={params.phi:.3f}"
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
            f"Butterfly: {'Y' if params.butterfly_condition() else 'N'}"
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
        """Print SSVI summary table header."""
        print("\n" + "=" * 90)
        print("SSVI FIT SUMMARY")
        print("=" * 90)
        print(f"{'Expiry':<10} {'TTM':>8} {'Cat':>6} {'Pts':>4} {'theta':>10} "
              f"{'rho':>7} {'phi':>8} {'R²':>7} {'Bfly':>5} {'Issues'}")
        print("-" * 90)

    def print_summary_row(self, result: FitResult) -> None:
        """Print a single row of the SSVI summary table."""
        params = result.params
        bfly = 'Y' if params.butterfly_condition() else 'N'
        issues = ','.join(result.boundary_issues) if result.boundary_issues else '-'

        print(
            f"{result.expiry:<10} {result.ttm_days:>7.1f}d {result.ttm_category:>6} "
            f"{result.n_points:>4} {params.theta:>10.6f} {params.rho:>7.3f} "
            f"{params.phi:>8.3f} {result.r_squared:>7.4f} {bfly:>5} {issues}"
        )


def main():
    parser = create_base_argument_parser("Test SSVI fitting for all expiries")
    parser.set_defaults(output_dir="results/ssvi_test_fits")
    args = parser.parse_args()

    setup_logging(args.verbose)

    # Load config
    config = Config()
    print(f"Multi-start enabled: {config.ssvi.use_multi_start}, n_starts={config.ssvi.n_starts}")
    print(f"Short-dated TTM threshold: {config.ssvi.short_dated_ttm_threshold}")

    # Run test harness
    harness = SSVITestHarness(
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
