#!/usr/bin/env python3
"""
Shared test harness for model fitting test scripts.

This module provides common functionality for test_heston_all_expiries.py
and test_ssvi_all_expiries.py, reducing code duplication (~70% overlap).

Usage:
    from test_model_base import (
        TestHarness,
        setup_logging,
        parse_expiry_date,
        create_relaxed_filter,
        print_summary_table,
    )
"""

__test__ = False  # Manual script support module, not a pytest test module.

import argparse
import logging
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np
import matplotlib.pyplot as plt

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from btc_pricer.config import Config, FilterConfig
from btc_pricer.api.deribit import DeribitClient, DeribitAPIError
from btc_pricer.api.binance import fetch_spot_with_fallback
from btc_pricer.data.filters import DataFilter
from btc_pricer.cli.common import (
    parse_expiry_date,
    extract_surface_data,
    create_heston_fitter,
    create_ssvi_fitter,
)
from btc_pricer.constants import BOUNDARY_CHECK_TOLERANCE


# Re-export commonly used functions
__all__ = [
    'TestHarness',
    'setup_logging',
    'parse_expiry_date',
    'create_relaxed_filter',
    'print_summary_header',
    'print_summary_footer',
    'FitResult',
]


def setup_logging(verbose: bool = False) -> None:
    """Set up logging configuration for test scripts.

    Uses a shorter time format (HH:MM:SS) suitable for test output.

    Args:
        verbose: If True, use DEBUG level; otherwise INFO.
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )


def create_relaxed_filter(config: Config) -> DataFilter:
    """Create a DataFilter with relaxed settings for testing.

    Uses filters_relaxed from config if available, otherwise defaults.

    Args:
        config: Application configuration.

    Returns:
        DataFilter with relaxed settings.
    """
    # Try to get relaxed config from config file
    if hasattr(config, 'filters_relaxed'):
        return DataFilter(config.filters_relaxed)

    # Default relaxed settings
    relaxed_filter_config = FilterConfig(
        min_open_interest=1,  # Very relaxed for testing
        max_bid_ask_spread_pct=0.50,  # 50% spread allowed
        min_days_to_expiry=0,
        moneyness_range=(0.5, 1.5),  # Wider moneyness range
        min_surface_points=5
    )
    return DataFilter(relaxed_filter_config)


@dataclass
class FitResult:
    """Result from model fitting for a single expiry."""
    expiry: str
    ttm_days: float
    ttm_category: str
    n_points: int
    r_squared: float
    rmse: float
    params: Any  # HestonParams or SSVIParams
    boundary_issues: List[str]

    # Model-specific fields
    extra_fields: Dict[str, Any] = None

    def __post_init__(self):
        if self.extra_fields is None:
            self.extra_fields = {}


class TestHarness(ABC):
    """Abstract base class for model fitting test scripts.

    Provides common functionality for data fetching, filtering, and output.
    Subclasses implement model-specific fitting and plotting.
    """

    def __init__(
        self,
        output_dir: str,
        min_points: int = 5,
        verbose: bool = False
    ):
        """Initialize test harness.

        Args:
            output_dir: Directory to save plots and results.
            min_points: Minimum data points per expiry.
            verbose: Enable verbose logging.
        """
        self.output_dir = Path(output_dir)
        self.min_points = min_points
        self.verbose = verbose
        self.logger = logging.getLogger(self.__class__.__name__)
        self.results: List[FitResult] = []

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return model name (e.g., 'Heston', 'SSVI')."""
        pass

    @abstractmethod
    def create_fitter(self, config: Config):
        """Create model fitter from configuration."""
        pass

    @abstractmethod
    def fit_model(
        self,
        log_moneyness: np.ndarray,
        market_iv: np.ndarray,
        ttm: float,
        forward: float
    ):
        """Fit model to market data. Returns (params, fit_result) or (None, None)."""
        pass

    @abstractmethod
    def get_bounds_for_ttm(self, ttm: float) -> dict:
        """Get parameter bounds for given TTM."""
        pass

    @abstractmethod
    def check_boundary_issues(self, params, bounds: dict) -> List[str]:
        """Check if parameters hit bounds."""
        pass

    @abstractmethod
    def plot_fit(
        self,
        log_moneyness: np.ndarray,
        market_iv: np.ndarray,
        params,
        fit_result,
        expiry: str,
        ttm: float,
        forward: float,
        spot: float,
        save_path: Path,
        bounds: dict
    ) -> None:
        """Generate IV smile plot with fit diagnostics."""
        pass

    @abstractmethod
    def print_summary_row(self, result: FitResult) -> None:
        """Print a single row of the summary table."""
        pass

    @abstractmethod
    def print_summary_header(self) -> None:
        """Print summary table header."""
        pass

    def get_ttm_category(self, ttm: float, config: Config) -> str:
        """Determine TTM category (very-short, short, normal).

        Args:
            ttm: Time to maturity in years.
            config: Configuration with TTM thresholds.

        Returns:
            Category string.
        """
        if hasattr(config, 'heston'):
            very_short = config.heston.very_short_dated_ttm_threshold
            short = config.heston.short_dated_ttm_threshold
        else:
            very_short = 0.02
            short = 0.10

        if ttm < very_short:
            return "very-short"
        elif ttm < short:
            return "short"
        return "normal"

    def fetch_data(self, config: Config) -> Dict[str, List]:
        """Fetch options data from Deribit.

        Args:
            config: Configuration.

        Returns:
            Dictionary mapping expiry strings to option data.

        Raises:
            DeribitAPIError: If fetching fails.
        """
        self.logger.info("Fetching options data from Deribit...")
        client = DeribitClient(config.api, config.validation)

        options_by_expiry = client.fetch_all_options("BTC")
        if not options_by_expiry:
            raise DeribitAPIError("No options data received")

        # Sort by date
        sorted_expiries = sorted(options_by_expiry.keys(), key=parse_expiry_date)
        self.logger.info(f"Found {len(sorted_expiries)} expiries: {', '.join(sorted_expiries)}")

        return options_by_expiry

    def get_spot_price(self, options_by_expiry: Dict[str, List]) -> float:
        """Get spot price (Binance with Deribit fallback).

        Args:
            options_by_expiry: Options data to extract Deribit spot from.

        Returns:
            Spot price.
        """
        # Get Deribit spot from first option
        deribit_spot = None
        for expiry in sorted(options_by_expiry.keys(), key=parse_expiry_date):
            if options_by_expiry[expiry]:
                deribit_spot = options_by_expiry[expiry][0].spot_price
                break

        if deribit_spot is None:
            raise ValueError("Could not determine spot price")

        self.logger.info(f"Deribit spot price: ${deribit_spot:,.0f}")

        # Use Binance with fallback
        spot_price, source = fetch_spot_with_fallback(deribit_spot)
        return spot_price

    def run(self, config: Config) -> List[FitResult]:
        """Run the test for all expiries.

        Args:
            config: Configuration.

        Returns:
            List of FitResult objects.
        """
        self.logger.info(f"Output directory: {self.output_dir.absolute()}")

        # Fetch data
        options_by_expiry = self.fetch_data(config)
        spot_price = self.get_spot_price(options_by_expiry)

        # Create filter and fitter
        data_filter = create_relaxed_filter(config)
        self.fitter = self.create_fitter(config)

        # Process each expiry
        sorted_expiries = sorted(options_by_expiry.keys(), key=parse_expiry_date)

        for expiry in sorted_expiries:
            raw_options = options_by_expiry[expiry]

            # Filter options
            filtered_options, filter_stats = data_filter.filter_options(
                raw_options, return_stats=True
            )

            if self.verbose and filter_stats:
                self.logger.debug(
                    f"{expiry}: total={filter_stats.total_options}, "
                    f"passed={filter_stats.passed_filters}"
                )

            if len(filtered_options) < self.min_points:
                self.logger.warning(
                    f"{expiry}: Skipping - only {len(filtered_options)} options after filtering"
                )
                continue

            # Build OTM surface
            otm_surface = data_filter.build_otm_surface(filtered_options)

            # Extract surface data
            surface_data = extract_surface_data(
                otm_surface,
                min_points=self.min_points,
                iv_valid_range=config.validation.iv_valid_range
            )

            if surface_data is None:
                self.logger.warning(f"{expiry}: Skipping - insufficient valid OTM options")
                continue

            forward, ttm, spot, log_moneyness, market_iv = surface_data
            ttm_category = self.get_ttm_category(ttm, config)
            bounds = self.get_bounds_for_ttm(ttm)

            self.logger.info(
                f"{expiry}: Fitting {self.model_name} ({len(log_moneyness)} pts, "
                f"TTM={ttm*365:.1f}d, {ttm_category}-dated)..."
            )

            # Fit model
            params, fit_result = self.fit_model(log_moneyness, market_iv, ttm, forward)

            if params is None:
                self.logger.warning(f"{expiry}: Fit failed")
                continue

            # Check boundary issues
            boundary_issues = self.check_boundary_issues(params, bounds)
            if boundary_issues:
                self.logger.warning(
                    f"{expiry}: Parameters at bounds: {', '.join(boundary_issues)}"
                )

            # Store result
            result = FitResult(
                expiry=expiry,
                ttm_days=ttm * 365,
                ttm_category=ttm_category,
                n_points=len(log_moneyness),
                r_squared=fit_result.r_squared,
                rmse=fit_result.rmse,
                params=params,
                boundary_issues=boundary_issues,
            )
            self.results.append(result)

            # Generate plot
            plot_path = self.output_dir / f"iv_smile_{self.model_name.lower()}_{expiry}.png"
            self.plot_fit(
                log_moneyness, market_iv,
                params, fit_result,
                expiry, ttm, forward, spot,
                plot_path, bounds
            )
            self.logger.info(f"{expiry}: Saved plot to {plot_path}")

        return self.results

    def print_summary(self) -> None:
        """Print summary table of all results."""
        self.print_summary_header()

        for result in self.results:
            self.print_summary_row(result)

        print("-" * 100)

        # Statistics
        if self.results:
            r_squared_values = [r.r_squared for r in self.results]
            boundary_count = sum(1 for r in self.results if r.boundary_issues)

            print(f"\nTotal expiries processed: {len(self.results)}")
            print(f"Average R²: {np.mean(r_squared_values):.4f}")
            print(f"Min R²: {np.min(r_squared_values):.4f}")
            print(f"Max R²: {np.max(r_squared_values):.4f}")
            print(f"Expiries with boundary issues: {boundary_count}/{len(self.results)}")
            print(f"\nPlots saved to: {self.output_dir.absolute()}")
        else:
            print("\nNo expiries could be processed.")


def create_base_argument_parser(description: str) -> argparse.ArgumentParser:
    """Create argument parser with common options.

    Args:
        description: Parser description.

    Returns:
        Configured ArgumentParser.
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/test_fits",
        help="Directory to save IV smile plots"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    parser.add_argument(
        "--min-points",
        type=int,
        default=5,
        help="Minimum number of data points per expiry (default: 5)"
    )
    return parser
