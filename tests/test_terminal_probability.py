"""Tests for terminal probability calculator."""

import pytest
import numpy as np

from btc_pricer.models.terminal_probability import (
    MCResult,
    TerminalProbabilityCalculator,
)
from btc_pricer.models.heston import HestonParams


class TestMCResult:
    """Test MCResult dataclass."""

    def test_basic_creation(self):
        """Test creating a MCResult."""
        result = MCResult(
            reference_price=85000,
            spot=90000,
            ttm=0.1,
            terminal_probability=0.10,
            confidence_interval=(0.09, 0.11),
            n_simulations=100000,
            direction="down",
            method="heston"
        )
        assert result.reference_price == 85000
        assert result.spot == 90000
        assert result.terminal_probability == 0.10

    def test_reference_price_distance_pct(self):
        """Test reference price distance percentage calculation."""
        result = MCResult(
            reference_price=85000,
            spot=90000,
            ttm=0.1,
            terminal_probability=0.10,
            confidence_interval=(0.09, 0.11),
            n_simulations=100000,
            direction="down",
            method="heston"
        )
        # (85000 - 90000) / 90000 * 100 = -5.56%
        assert abs(result.reference_price_distance_pct - (-5.556)) < 0.01

    def test_to_dict(self):
        """Test conversion to dictionary."""
        result = MCResult(
            reference_price=85000,
            spot=90000,
            ttm=0.1,
            terminal_probability=0.10,
            confidence_interval=(0.09, 0.11),
            n_simulations=100000,
            direction="down",
            method="heston"
        )
        d = result.to_dict()
        assert d["reference_price"] == 85000
        assert d["terminal_probability"] == 0.10
        assert d["confidence_interval"] == [0.09, 0.11]
        assert "reference_price_distance_pct" in d
        assert "barrier" not in d
        assert "barrier_type" not in d


class TestTerminalProbabilityCalculator:
    """Test TerminalProbabilityCalculator class."""

    def setup_method(self):
        """Set up test fixtures."""
        # Use fewer simulations for testing
        self.calculator = TerminalProbabilityCalculator(
            n_simulations=10000,
            n_steps_per_day=48,  # 30-min steps
            seed=42
        )
        # Standard Heston parameters
        self.heston_params = HestonParams(
            v0=0.04,      # 20% vol
            kappa=2.0,
            theta=0.04,
            xi=0.3,
            rho=-0.7,
            ttm=0.1       # ~36 days
        )

    def test_simulate_heston_paths_shape(self):
        """Test that simulated paths have correct shape."""
        n_paths = 1000
        n_steps = 100
        spot = 90000

        paths = self.calculator.simulate_heston_paths(
            self.heston_params, spot, n_paths, n_steps
        )

        assert paths.shape == (n_paths, n_steps + 1)

    def test_simulate_heston_paths_initial_value(self):
        """Test that paths start at spot price."""
        n_paths = 1000
        n_steps = 100
        spot = 90000

        paths = self.calculator.simulate_heston_paths(
            self.heston_params, spot, n_paths, n_steps
        )

        np.testing.assert_array_equal(paths[:, 0], spot)

    def test_simulate_heston_paths_positive(self):
        """Test that all prices remain positive."""
        n_paths = 1000
        n_steps = 100
        spot = 90000

        paths = self.calculator.simulate_heston_paths(
            self.heston_params, spot, n_paths, n_steps
        )

        assert np.all(paths > 0)

    def test_simulate_heston_paths_mean_reasonable(self):
        """Test that path mean at terminal is reasonable (forward)."""
        n_paths = 5000
        n_steps = 100
        spot = 90000

        paths = self.calculator.simulate_heston_paths(
            self.heston_params, spot, n_paths, n_steps
        )

        # For risk-neutral (drift=0), mean should be close to spot
        # Allow 10% deviation due to simulation variance
        terminal_mean = np.mean(paths[:, -1])
        assert abs(terminal_mean - spot) / spot < 0.10

    def test_terminal_probability_down(self):
        """Test down direction terminal probability."""
        spot = 90000
        reference_price = 85000  # 5.5% below

        result = self.calculator.terminal_probability_heston(
            self.heston_params, spot, reference_price, "down"
        )

        assert isinstance(result, MCResult)
        assert result.direction == "down"
        assert result.method == "heston"
        assert 0 <= result.terminal_probability <= 1

    def test_terminal_probability_up(self):
        """Test up direction terminal probability."""
        spot = 90000
        reference_price = 95000  # 5.5% above

        result = self.calculator.terminal_probability_heston(
            self.heston_params, spot, reference_price, "up"
        )

        assert isinstance(result, MCResult)
        assert result.direction == "up"
        assert 0 <= result.terminal_probability <= 1

    def test_terminal_probability_confidence_interval(self):
        """Test that confidence interval is valid."""
        spot = 90000
        reference_price = 85000

        result = self.calculator.terminal_probability_heston(
            self.heston_params, spot, reference_price, "down"
        )

        ci_low, ci_high = result.confidence_interval
        assert ci_low <= result.terminal_probability <= ci_high
        assert ci_low >= 0
        assert ci_high <= 1

    def test_terminal_probability_at_spot(self):
        """Test that reference price at spot gives ~50% terminal probability."""
        spot = 90000
        reference_price = 90000  # At spot

        result = self.calculator.terminal_probability_heston(
            self.heston_params, spot, reference_price, "down"
        )

        # Terminal probability of ending below spot should be ~50%
        assert 0.30 < result.terminal_probability < 0.70

    def test_terminal_probability_far_reference_price(self):
        """Test that very far reference price gives low probability."""
        spot = 90000
        reference_price = 50000  # 44% below

        result = self.calculator.terminal_probability_heston(
            self.heston_params, spot, reference_price, "down"
        )

        # Should be very low probability for short TTM
        assert result.terminal_probability < 0.05

    def test_compact_returns_array(self):
        """Test that compact simulation returns a 1D array."""
        n_paths = 1000
        n_steps = 100
        spot = 90000

        final_prices = self.calculator.simulate_heston_paths_compact(
            self.heston_params, spot, n_paths, n_steps
        )

        assert isinstance(final_prices, np.ndarray)
        assert final_prices.shape == (n_paths,)
        assert np.all(final_prices > 0)


class TestMultipleReferencePrices:
    """Test multiple reference price calculation."""

    def setup_method(self):
        """Set up test fixtures."""
        self.calculator = TerminalProbabilityCalculator(
            n_simulations=5000,
            n_steps_per_day=24,
            seed=42
        )
        self.heston_params = HestonParams(
            v0=0.04,
            kappa=2.0,
            theta=0.04,
            xi=0.3,
            rho=-0.7,
            ttm=0.1
        )

    def test_multiple_reference_prices_returns_list(self):
        """Test that multiple reference prices returns list of results."""
        spot = 90000
        reference_prices = [80000, 85000, 90000]

        results = self.calculator.terminal_probability_heston_multiple(
            self.heston_params, spot, reference_prices, "down"
        )

        assert isinstance(results, list)
        assert len(results) == len(reference_prices)

    def test_multiple_reference_prices_ordered(self):
        """Test that results maintain reference price ordering."""
        spot = 90000
        reference_prices = [85000, 80000, 87500]

        results = self.calculator.terminal_probability_heston_multiple(
            self.heston_params, spot, reference_prices, "down"
        )

        result_prices = [r.reference_price for r in results]
        assert result_prices == reference_prices

    def test_multiple_reference_prices_monotonic_down(self):
        """Test that terminal probability increases as reference price gets closer to spot."""
        spot = 90000
        reference_prices = [75000, 80000, 85000]  # Increasing toward spot

        results = self.calculator.terminal_probability_heston_multiple(
            self.heston_params, spot, reference_prices, "down"
        )

        probs = [r.terminal_probability for r in results]
        # Probability should generally increase as reference price approaches spot
        # Allow small violations due to MC noise
        assert probs[-1] >= probs[0] - 0.05

    def test_multiple_reference_prices_all_valid(self):
        """Test that all reference price results are valid."""
        spot = 90000
        reference_prices = [80000, 85000, 87500]

        results = self.calculator.terminal_probability_heston_multiple(
            self.heston_params, spot, reference_prices, "down"
        )

        for result in results:
            assert 0 <= result.terminal_probability <= 1
            ci_low, ci_high = result.confidence_interval
            assert ci_low <= result.terminal_probability <= ci_high


class TestAntitheticVariance:
    """Test antithetic variance reduction."""

    def test_antithetic_enabled_doubles_paths(self):
        """Test that antithetic doubles effective path count."""
        n_sims = 1000
        calc = TerminalProbabilityCalculator(
            n_simulations=n_sims,
            n_steps_per_day=24,
            use_antithetic=True,
            seed=42
        )

        heston_params = HestonParams(
            v0=0.04, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7, ttm=0.1
        )

        result = calc.terminal_probability_heston(heston_params, 90000, 85000, "down")

        # With antithetic, effective n_simulations should equal requested
        assert result.n_simulations == n_sims

    def test_antithetic_disabled(self):
        """Test without antithetic variance reduction."""
        n_sims = 1000
        calc = TerminalProbabilityCalculator(
            n_simulations=n_sims,
            n_steps_per_day=24,
            use_antithetic=False,
            seed=42
        )

        heston_params = HestonParams(
            v0=0.04, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7, ttm=0.1
        )

        result = calc.terminal_probability_heston(heston_params, 90000, 85000, "down")

        assert result.n_simulations == n_sims


class TestSSVILocalVolMC:
    """Test SSVI Local Volatility Monte Carlo."""

    def setup_method(self):
        """Set up test fixtures."""
        from btc_pricer.models.ssvi import SSVIParams
        self.calculator = TerminalProbabilityCalculator(
            n_simulations=10000,
            n_steps_per_day=48,
            seed=42
        )
        # Typical SSVI params with negative skew (rho < 0)
        self.ssvi_params = SSVIParams(
            theta=0.01,   # ATM total variance (low for short TTM)
            rho=-0.3,     # Negative skew (put > call IV)
            phi=0.5,      # Moderate curvature
            ttm=0.1       # ~36 days
        )
        self.spot = 90000
        self.forward = 90000  # Assume forward ~ spot

    def test_ssvi_terminal_probability_basic(self):
        """Test basic SSVI terminal probability calculation."""
        reference_price = 85000  # Down

        result = self.calculator.terminal_probability_ssvi(
            self.ssvi_params, self.spot, self.forward, reference_price, "down"
        )

        assert 0 < result.terminal_probability < 1
        assert result.method == "ssvi_local_vol"
        assert result.reference_price == reference_price
        assert result.n_simulations > 0

    def test_ssvi_multiple_reference_prices(self):
        """Test multiple reference prices with SSVI."""
        reference_prices = [80000, 85000, 88000]

        results = self.calculator.terminal_probability_ssvi_multiple(
            self.ssvi_params, self.spot, self.forward, reference_prices, "down"
        )

        assert len(results) == len(reference_prices)
        for result in results:
            assert 0 <= result.terminal_probability <= 1
            assert result.method == "ssvi_local_vol"

    def test_ssvi_monotonic_reference_prices(self):
        """Test closer reference prices have higher terminal probability."""
        reference_prices = [75000, 82000, 88000]  # Increasing toward spot

        results = self.calculator.terminal_probability_ssvi_multiple(
            self.ssvi_params, self.spot, self.forward, reference_prices, "down"
        )

        probs = [r.terminal_probability for r in results]
        # Probability should increase as reference price approaches spot
        # Allow some MC noise
        assert probs[-1] >= probs[0] - 0.05

    def test_ssvi_up_direction(self):
        """Test up direction with SSVI."""
        reference_price = 95000  # Up

        result = self.calculator.terminal_probability_ssvi(
            self.ssvi_params, self.spot, self.forward, reference_price, "up"
        )

        assert 0 < result.terminal_probability < 1
        assert result.direction == "up"

    def test_ssvi_compact_returns_array(self):
        """Test that SSVI compact simulation returns a 1D array."""
        n_paths = 1000
        n_steps = 100

        final_prices = self.calculator.simulate_local_vol_paths_compact(
            self.ssvi_params, self.spot, self.forward, n_paths, n_steps
        )

        assert isinstance(final_prices, np.ndarray)
        assert final_prices.shape == (n_paths,)
        assert np.all(final_prices > 0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
