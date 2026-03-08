"""Tests for terminal probability calculator."""

import pytest
import numpy as np
from scipy.stats import norm

from btc_pricer.models.terminal_probability import (
    MCResult,
    TerminalProbabilityCalculator,
)
from btc_pricer.models.heston import HestonParams


class TestMCResult:
    """Test MCResult dataclass."""

    def test_basic_creation(self):
        """Test creating an MCResult."""
        result = MCResult(
            target_price=85000,
            spot=90000,
            ttm=0.1,
            terminal_probability=0.10,
            confidence_interval=(0.09, 0.11),
            n_simulations=100000,
            direction="down",
            method="heston"
        )
        assert result.target_price == 85000
        assert result.spot == 90000
        assert result.terminal_probability == 0.10

    def test_target_distance_pct(self):
        """Test target distance percentage calculation."""
        result = MCResult(
            target_price=85000,
            spot=90000,
            ttm=0.1,
            terminal_probability=0.10,
            confidence_interval=(0.09, 0.11),
            n_simulations=100000,
            direction="down",
            method="heston"
        )
        # (85000 - 90000) / 90000 * 100 = -5.56%
        assert abs(result.target_distance_pct - (-5.556)) < 0.01

    def test_to_dict(self):
        """Test conversion to dictionary."""
        result = MCResult(
            target_price=85000,
            spot=90000,
            ttm=0.1,
            terminal_probability=0.10,
            confidence_interval=(0.09, 0.11),
            n_simulations=100000,
            direction="down",
            method="heston"
        )
        d = result.to_dict()
        assert d["target_price"] == 85000
        assert d["terminal_probability"] == 0.10
        assert d["confidence_interval"] == [0.09, 0.11]
        assert "target_distance_pct" in d


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
        """Test down terminal probability."""
        spot = 90000
        target = 85000  # 5.5% below

        result = self.calculator.terminal_probability_heston(
            self.heston_params, spot, target, "down"
        )

        assert isinstance(result, MCResult)
        assert result.direction == "down"
        assert result.method == "heston"
        assert 0 <= result.terminal_probability <= 1

    def test_terminal_probability_up(self):
        """Test up terminal probability."""
        spot = 90000
        target = 95000  # 5.5% above

        result = self.calculator.terminal_probability_heston(
            self.heston_params, spot, target, "up"
        )

        assert isinstance(result, MCResult)
        assert result.direction == "up"
        assert 0 <= result.terminal_probability <= 1

    def test_terminal_probability_confidence_interval(self):
        """Test that confidence interval is valid."""
        spot = 90000
        target = 85000

        result = self.calculator.terminal_probability_heston(
            self.heston_params, spot, target, "down"
        )

        ci_low, ci_high = result.confidence_interval
        assert ci_low <= result.terminal_probability <= ci_high
        assert ci_low >= 0
        assert ci_high <= 1

    def test_terminal_probability_at_spot(self):
        """Test that target at spot gives ~50% probability."""
        spot = 90000
        target = 90000  # At spot

        result = self.calculator.terminal_probability_heston(
            self.heston_params, spot, target, "down"
        )

        # Should be around 50% for at-the-money
        assert 0.30 < result.terminal_probability < 0.70

    def test_terminal_probability_far_target(self):
        """Test that very far target gives low probability."""
        spot = 90000
        target = 50000  # 44% below

        result = self.calculator.terminal_probability_heston(
            self.heston_params, spot, target, "down"
        )

        # Should be very low probability for short TTM
        assert result.terminal_probability < 0.10


class TestMultipleTargets:
    """Test multiple target calculation."""

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

    def test_multiple_targets_returns_list(self):
        """Test that multiple targets returns list of results."""
        spot = 90000
        targets = [80000, 85000, 90000]

        results = self.calculator.terminal_probability_heston_multiple(
            self.heston_params, spot, targets, "down"
        )

        assert isinstance(results, list)
        assert len(results) == len(targets)

    def test_multiple_targets_ordered(self):
        """Test that results maintain target ordering."""
        spot = 90000
        targets = [85000, 80000, 87500]

        results = self.calculator.terminal_probability_heston_multiple(
            self.heston_params, spot, targets, "down"
        )

        result_targets = [r.target_price for r in results]
        assert result_targets == targets

    def test_multiple_targets_monotonic_down(self):
        """Test that terminal probability increases as target gets closer to spot."""
        spot = 90000
        targets = [75000, 80000, 85000]  # Increasing toward spot

        results = self.calculator.terminal_probability_heston_multiple(
            self.heston_params, spot, targets, "down"
        )

        probs = [r.terminal_probability for r in results]
        # Probability should generally increase as target approaches spot
        # Allow small violations due to MC noise
        assert probs[-1] >= probs[0] - 0.05

    def test_multiple_targets_all_valid(self):
        """Test that all target results are valid."""
        spot = 90000
        targets = [80000, 85000, 87500]

        results = self.calculator.terminal_probability_heston_multiple(
            self.heston_params, spot, targets, "down"
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
        target = 85000  # Down target

        result = self.calculator.terminal_probability_ssvi(
            self.ssvi_params, self.spot, self.forward, target, "down"
        )

        assert 0 < result.terminal_probability < 1
        assert result.method == "ssvi_local_vol"
        assert result.target_price == target
        assert result.n_simulations > 0

    def test_ssvi_multiple_targets(self):
        """Test multiple targets with SSVI."""
        targets = [80000, 85000, 88000]

        results = self.calculator.terminal_probability_ssvi_multiple(
            self.ssvi_params, self.spot, self.forward, targets, "down"
        )

        assert len(results) == len(targets)
        for result in results:
            assert 0 <= result.terminal_probability <= 1
            assert result.method == "ssvi_local_vol"

    def test_ssvi_monotonic_targets(self):
        """Test closer targets have higher terminal probability."""
        targets = [75000, 82000, 88000]  # Increasing toward spot

        results = self.calculator.terminal_probability_ssvi_multiple(
            self.ssvi_params, self.spot, self.forward, targets, "down"
        )

        probs = [r.terminal_probability for r in results]
        # Probability should increase as target approaches spot
        # Allow some MC noise
        assert probs[-1] >= probs[0] - 0.05

    def test_ssvi_up_target(self):
        """Test up target with SSVI."""
        target = 95000  # Up target

        result = self.calculator.terminal_probability_ssvi(
            self.ssvi_params, self.spot, self.forward, target, "up"
        )

        assert 0 < result.terminal_probability < 1
        assert result.direction == "up"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
