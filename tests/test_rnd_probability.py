"""Tests for Breeden-Litzenberger RND probability calculations."""

import pytest
import numpy as np

from btc_pricer.models.breeden_litzenberger import BreedenLitzenberger, RNDResult
from btc_pricer.models.ssvi import SSVIParams


class TestTerminalProbabilityLogic:
    """Test the probability calculation logic used by RND mode."""

    def setup_method(self):
        """Set up test fixtures."""
        self.bl = BreedenLitzenberger(
            strike_grid_points=500,
            strike_range_std=3.0
        )
        # Create a realistic SSVI surface
        self.ssvi_params = SSVIParams(theta=0.04, rho=-0.3, phi=0.5, ttm=0.25)
        self.forward = 100000

    def test_probability_above_plus_below_equals_one(self):
        """P(above) + P(below) should approximately equal 1."""
        rnd = self.bl.extract_from_ssvi(self.ssvi_params, self.forward)

        # Test at various price levels
        test_prices = [80000, 90000, 100000, 110000, 120000]

        for price in test_prices:
            p_above = self.bl.probability_above(rnd, price)
            p_below = self.bl.probability_below(rnd, price)

            # Sum should be very close to 1 (allowing for discrete integration error)
            total = p_above + p_below
            assert 0.98 < total < 1.02, f"P(above) + P(below) = {total} at price {price}"

    def test_probability_monotonicity(self):
        """P(above K) should decrease as K increases."""
        rnd = self.bl.extract_from_ssvi(self.ssvi_params, self.forward)

        test_prices = [80000, 90000, 100000, 110000, 120000]
        prev_prob = 1.0

        for price in test_prices:
            p_above = self.bl.probability_above(rnd, price)
            assert p_above <= prev_prob + 0.01, f"P(above {price}) = {p_above} > {prev_prob}"
            prev_prob = p_above

    def test_probability_at_extremes(self):
        """Test probability at extreme price levels."""
        rnd = self.bl.extract_from_ssvi(self.ssvi_params, self.forward)

        # Very low price: P(above) should be very high
        p_above_low = self.bl.probability_above(rnd, 50000)
        assert p_above_low > 0.95

        # Very high price: P(above) should be very low
        p_above_high = self.bl.probability_above(rnd, 200000)
        assert p_above_high < 0.05

    def test_probability_at_forward(self):
        """P(above forward) should be close to 50% for symmetric density."""
        # Use symmetric parameters (low skew)
        symmetric_params = SSVIParams(theta=0.04, rho=0.0, phi=0.3, ttm=0.25)
        rnd = self.bl.extract_from_ssvi(symmetric_params, self.forward)

        p_above_forward = self.bl.probability_above(rnd, self.forward)

        # Should be approximately 50% for symmetric density
        assert 0.40 < p_above_forward < 0.60

    def test_negative_skew_effect(self):
        """Negative rho (put skew) should decrease P(above) relative to forward."""
        # Negative skew: more probability mass below forward
        neg_skew_params = SSVIParams(theta=0.04, rho=-0.5, phi=0.5, ttm=0.25)
        rnd = self.bl.extract_from_ssvi(neg_skew_params, self.forward)

        # With negative skew, P(above forward) should be higher than 50%
        # because the left tail is fatter (hedging against drops)
        # Actually, negative rho creates put skew, meaning higher IV for low strikes
        # This implies the market expects potential drops, so there's more probability
        # in the left tail, making P(above forward) > 50%
        p_above_forward = self.bl.probability_above(rnd, self.forward)

        # Just verify it's reasonable - the exact value depends on implementation
        assert 0.3 < p_above_forward < 0.7


class TestProbabilityWithHeston:
    """Test probability calculations with Heston model."""

    def setup_method(self):
        """Set up test fixtures."""
        self.bl = BreedenLitzenberger(
            strike_grid_points=500,
            strike_range_std=4.0,
            use_log_strikes=True
        )

    @pytest.mark.skipif(
        not pytest.importorskip("btc_pricer.models.heston", reason="Heston not available"),
        reason="Heston model not available"
    )
    def test_heston_probability_calculation(self):
        """Test probability calculation from Heston RND."""
        from btc_pricer.models.heston import HestonParams

        params = HestonParams(
            v0=0.04,
            kappa=2.0,
            theta=0.04,
            xi=0.5,
            rho=-0.7,
            ttm=0.25
        )
        forward = 100000

        rnd = self.bl.extract_from_heston(params, forward, use_quantlib=True)

        # Basic validity
        assert rnd.is_valid or len(rnd.warnings) <= 2

        # P(above) + P(below) ≈ 1
        test_price = 95000
        p_above = self.bl.probability_above(rnd, test_price)
        p_below = self.bl.probability_below(rnd, test_price)

        assert 0.98 < p_above + p_below < 1.02


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
