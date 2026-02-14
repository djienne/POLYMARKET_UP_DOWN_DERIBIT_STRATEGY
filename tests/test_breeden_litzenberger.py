"""Tests for Breeden-Litzenberger RND extraction."""

import pytest
import numpy as np
import math
from scipy.stats import norm

from btc_pricer.models.breeden_litzenberger import BreedenLitzenberger, RNDResult
from btc_pricer.models.ssvi import SSVIParams, SSVIModel
from btc_pricer.models.black_scholes import BlackScholes


class TestBreedenLitzenberger:
    """Test Breeden-Litzenberger RND extraction."""

    def setup_method(self):
        """Set up test fixtures."""
        self.bl = BreedenLitzenberger(
            strike_grid_points=500,
            strike_range_std=3.0
        )

    def test_extract_from_ssvi_basic(self):
        """Test basic RND extraction from SSVI."""
        params = SSVIParams(theta=0.04, rho=-0.3, phi=0.5, ttm=0.25)
        forward = 100000

        rnd = self.bl.extract_from_ssvi(params, forward)

        # Basic validity checks
        assert rnd.is_valid or len(rnd.warnings) > 0
        assert len(rnd.strikes) == 500
        assert len(rnd.density) == 500

        # Density should be non-negative
        assert np.all(rnd.density >= 0)

        # Should integrate to approximately 1
        assert 0.95 < rnd.integral < 1.05

    def test_rnd_integrates_to_one(self):
        """Test that RND integrates to 1."""
        params = SSVIParams(theta=0.05, rho=-0.2, phi=0.4, ttm=0.5)
        forward = 100000

        rnd = self.bl.extract_from_ssvi(params, forward)

        assert abs(rnd.integral - 1.0) < 0.02

    def test_rnd_mean_approximates_forward(self):
        """Test that RND mean is close to forward price."""
        params = SSVIParams(theta=0.04, rho=-0.2, phi=0.3, ttm=0.25)
        forward = 100000

        rnd = self.bl.extract_from_ssvi(params, forward)

        # Mean should be within 5% of forward
        assert abs(rnd.mean - forward) / forward < 0.05

    def test_rnd_statistics_reasonable(self):
        """Test that RND statistics are reasonable."""
        params = SSVIParams(theta=0.04, rho=-0.3, phi=0.5, ttm=0.25)
        forward = 100000

        rnd = self.bl.extract_from_ssvi(params, forward)

        # Std dev should be reasonable (10-100% of forward)
        assert 0.05 * forward < rnd.std_dev < 1.0 * forward

        # Skewness should be moderate
        assert -3 < rnd.skewness < 3

        # Percentiles should be ordered
        assert rnd.percentile_5 < rnd.percentile_25
        assert rnd.percentile_25 < rnd.percentile_50
        assert rnd.percentile_50 < rnd.percentile_75
        assert rnd.percentile_75 < rnd.percentile_95

    def test_rnd_mode_reasonable(self):
        """Test that RND mode is in reasonable range."""
        params = SSVIParams(theta=0.04, rho=-0.3, phi=0.5, ttm=0.25)
        forward = 100000

        rnd = self.bl.extract_from_ssvi(params, forward)

        # Mode should be between 50% and 200% of forward
        assert 0.5 * forward < rnd.mode < 2.0 * forward

    def test_lognormal_recovery(self):
        """Test that flat vol (Black-Scholes) gives log-normal RND."""
        # With rho=0 and phi→0, SSVI approaches constant vol
        # This should give approximately log-normal density
        vol = 0.5
        ttm = 0.25
        theta = vol**2 * ttm

        # Use very small phi and zero rho for flat vol
        params = SSVIParams(theta=theta, rho=0.0, phi=0.01, ttm=ttm)
        forward = 100000

        rnd = self.bl.extract_from_ssvi(params, forward)

        # For log-normal, mean ≈ forward (under forward measure)
        assert abs(rnd.mean - forward) / forward < 0.1

        # Variance of log-normal: Var = F²(e^(σ²T) - 1)
        expected_var = forward**2 * (math.exp(vol**2 * ttm) - 1)
        assert abs(rnd.variance - expected_var) / expected_var < 0.2

    def test_probability_calculations(self):
        """Test probability calculations from RND."""
        params = SSVIParams(theta=0.04, rho=-0.3, phi=0.5, ttm=0.25)
        forward = 100000

        rnd = self.bl.extract_from_ssvi(params, forward)

        # Probability above median should be close to 50%
        prob_above_median = self.bl.probability_above(rnd, rnd.percentile_50)
        assert 0.45 < prob_above_median < 0.55

        # Probability below 5th percentile should be close to 5%
        prob_below_5 = self.bl.probability_below(rnd, rnd.percentile_5)
        assert 0.03 < prob_below_5 < 0.08

        # Probability between 25th and 75th should be close to 50%
        prob_middle = self.bl.probability_between(rnd, rnd.percentile_25, rnd.percentile_75)
        assert 0.45 < prob_middle < 0.55

        # Total probability should sum to 1
        prob_all = self.bl.probability_between(rnd, rnd.strikes[0], rnd.strikes[-1])
        assert 0.98 < prob_all < 1.02

    def test_butterfly_violation_warning(self):
        """Test that butterfly violation produces warning."""
        # Create params that violate butterfly condition
        params = SSVIParams(theta=2.0, rho=0.8, phi=1.5, ttm=0.25)
        forward = 100000

        # Should still work but with warning
        rnd = self.bl.extract_from_ssvi(params, forward)

        assert any("butterfly" in w.lower() for w in rnd.warnings)


class TestBreedenLitzenbergerFromPrices:
    """Test RND extraction directly from prices."""

    def setup_method(self):
        """Set up test fixtures."""
        self.bl = BreedenLitzenberger(strike_grid_points=200)

    def test_extract_from_bs_prices(self):
        """Test extraction from Black-Scholes prices."""
        forward = 100000
        vol = 0.6
        ttm = 0.25

        # Generate call prices
        strikes = np.linspace(60000, 150000, 50)
        call_prices = np.array([
            BlackScholes.forward_call_price(forward, K, vol, ttm)
            for K in strikes
        ])

        rnd = self.bl.extract_from_prices(strikes, call_prices, forward, ttm)

        # Should integrate to ~1
        assert 0.9 < rnd.integral < 1.1

        # Mean should be close to forward
        assert abs(rnd.mean - forward) / forward < 0.15


class TestRNDResult:
    """Test RNDResult dataclass."""

    def test_to_dict(self):
        """Test conversion to dictionary."""
        rnd = RNDResult(
            strikes=np.array([90000, 100000, 110000]),
            density=np.array([0.3, 0.4, 0.3]),
            forward=100000,
            ttm=0.25,
            mean=100000,
            mode=100000,
            variance=1e9,
            std_dev=31623,
            skewness=0.1,
            kurtosis=0.5,
            percentile_5=70000,
            percentile_25=85000,
            percentile_50=100000,
            percentile_75=115000,
            percentile_95=130000,
            integral=1.0,
            is_valid=True,
            warnings=[]
        )

        d = rnd.to_dict()

        assert d["forward"] == 100000
        assert d["mean"] == 100000
        assert d["is_valid"] == True
        assert "percentiles" in d
        assert d["percentiles"]["50"] == 100000


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
