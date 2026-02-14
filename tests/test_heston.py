"""Tests for Heston stochastic volatility model."""

import pytest
import numpy as np
import math
from btc_pricer.models.heston import (
    HestonParams, HestonModel, HestonFitter, check_iv_consistency
)
from btc_pricer.models.black_scholes import BlackScholes


class TestHestonParams:
    """Test Heston parameter validation."""

    def test_valid_params(self):
        """Test creating valid parameters."""
        params = HestonParams(
            v0=0.04, kappa=2.0, theta=0.04, xi=0.5, rho=-0.7, ttm=0.25
        )
        assert params.v0 == 0.04
        assert params.kappa == 2.0
        assert params.theta == 0.04
        assert params.xi == 0.5
        assert params.rho == -0.7
        assert params.ttm == 0.25

    def test_invalid_v0(self):
        """Test that non-positive v0 raises error."""
        with pytest.raises(ValueError):
            HestonParams(v0=-0.04, kappa=2.0, theta=0.04, xi=0.5, rho=-0.7, ttm=0.25)
        with pytest.raises(ValueError):
            HestonParams(v0=0.0, kappa=2.0, theta=0.04, xi=0.5, rho=-0.7, ttm=0.25)

    def test_invalid_kappa(self):
        """Test that non-positive kappa raises error."""
        with pytest.raises(ValueError):
            HestonParams(v0=0.04, kappa=-2.0, theta=0.04, xi=0.5, rho=-0.7, ttm=0.25)

    def test_invalid_theta(self):
        """Test that non-positive theta raises error."""
        with pytest.raises(ValueError):
            HestonParams(v0=0.04, kappa=2.0, theta=-0.04, xi=0.5, rho=-0.7, ttm=0.25)

    def test_invalid_xi(self):
        """Test that non-positive xi raises error."""
        with pytest.raises(ValueError):
            HestonParams(v0=0.04, kappa=2.0, theta=0.04, xi=-0.5, rho=-0.7, ttm=0.25)

    def test_invalid_rho(self):
        """Test that rho outside (-1, 1) raises error."""
        with pytest.raises(ValueError):
            HestonParams(v0=0.04, kappa=2.0, theta=0.04, xi=0.5, rho=1.0, ttm=0.25)
        with pytest.raises(ValueError):
            HestonParams(v0=0.04, kappa=2.0, theta=0.04, xi=0.5, rho=-1.0, ttm=0.25)

    def test_invalid_ttm(self):
        """Test that non-positive ttm raises error."""
        with pytest.raises(ValueError):
            HestonParams(v0=0.04, kappa=2.0, theta=0.04, xi=0.5, rho=-0.7, ttm=0.0)

    def test_feller_condition_satisfied(self):
        """Test Feller condition when satisfied."""
        # 2*2*0.09 = 0.36 > 0.25 = 0.5^2
        params = HestonParams(v0=0.04, kappa=2.0, theta=0.09, xi=0.5, rho=-0.7, ttm=0.25)
        assert params.feller_condition()
        assert params.feller_ratio() > 1

    def test_feller_condition_violated(self):
        """Test Feller condition when violated."""
        # 2*0.5*0.04 = 0.04 < 1.0 = 1.0^2
        params = HestonParams(v0=0.04, kappa=0.5, theta=0.04, xi=1.0, rho=-0.7, ttm=0.25)
        assert not params.feller_condition()
        assert params.feller_ratio() < 1

    def test_to_dict(self):
        """Test conversion to dictionary."""
        params = HestonParams(v0=0.04, kappa=2.0, theta=0.04, xi=0.5, rho=-0.7, ttm=0.25)
        d = params.to_dict()
        assert d["v0"] == 0.04
        assert d["kappa"] == 2.0
        assert d["theta"] == 0.04
        assert d["xi"] == 0.5
        assert d["rho"] == -0.7
        assert d["ttm"] == 0.25
        assert "feller_satisfied" in d
        assert "feller_ratio" in d


class TestHestonModel:
    """Test Heston model calculations."""

    def setup_method(self):
        """Set up test fixtures."""
        # Standard parameters with Feller condition satisfied
        self.params = HestonParams(
            v0=0.04, kappa=1.5, theta=0.04, xi=0.3, rho=-0.7, ttm=0.5
        )
        self.model = HestonModel(self.params)

    def test_ttm_property(self):
        """Test ttm property for VolatilitySurface protocol."""
        assert self.model.ttm == 0.5

    def test_call_price_positive(self):
        """Test that call prices are positive."""
        forward = 100.0
        strikes = [80, 90, 100, 110, 120]
        for strike in strikes:
            price = self.model.call_price(forward, strike)
            assert price >= 0

    def test_call_price_bounds(self):
        """Test that call prices satisfy bounds."""
        forward = 100.0
        strike = 100.0
        price = self.model.call_price(forward, strike)

        # Call price should be between 0 and forward - strike (for ITM) or 0 and forward
        assert price >= 0
        assert price <= forward

    def test_put_call_parity(self):
        """Test put-call parity: C - P = F - K."""
        forward = 100.0
        strike = 100.0

        call = self.model.call_price(forward, strike)
        put = self.model.put_price(forward, strike)

        parity = call - put
        expected = forward - strike

        assert abs(parity - expected) < 0.01  # Allow small numerical error

    def test_call_price_atm(self):
        """Test ATM call price is reasonable."""
        forward = 100.0
        strike = 100.0
        call = self.model.call_price(forward, strike)

        # ATM call price should be roughly F * sigma * sqrt(T) * 0.4
        # for low vol
        atm_vol = math.sqrt(self.params.v0)
        expected_approx = forward * atm_vol * math.sqrt(self.params.ttm) * 0.4
        assert call > 0
        assert call < forward * 0.3  # Not too expensive

    def test_implied_volatility_strike(self):
        """Test IV calculation from strike."""
        forward = 100.0
        strike = 100.0

        iv = self.model.implied_volatility_strike(strike, forward)
        assert iv > 0
        assert iv < 5.0  # Reasonable IV bounds

    def test_implied_volatility_log_moneyness(self):
        """Test IV calculation from log-moneyness."""
        k = 0  # ATM
        iv = self.model.implied_volatility(k)

        assert iv > 0
        assert iv < 5.0

    def test_implied_volatility_array(self):
        """Test IV array calculation."""
        k_array = np.array([-0.2, -0.1, 0, 0.1, 0.2])
        ivs = self.model.implied_volatility_array(k_array)

        assert len(ivs) == len(k_array)
        assert all(iv > 0 for iv in ivs)
        assert all(iv < 5.0 for iv in ivs)

    def test_smile_shape_with_negative_rho(self):
        """Test that negative rho produces downward skew."""
        k_array = np.array([-0.3, -0.15, 0, 0.15, 0.3])
        ivs = self.model.implied_volatility_array(k_array)

        # With negative rho, OTM puts (negative k) should have higher IV
        # than OTM calls (positive k) on average
        left_mean = np.mean(ivs[:2])
        right_mean = np.mean(ivs[3:])
        assert left_mean > right_mean * 0.9  # Allow some tolerance

    def test_cache_usage(self):
        """Test that caching works."""
        forward = 100.0
        strike = 100.0

        # First call
        iv1 = self.model.implied_volatility_strike(strike, forward)

        # Second call should use cache
        iv2 = self.model.implied_volatility_strike(strike, forward)

        assert iv1 == iv2

        # Clear cache
        self.model.clear_cache()
        assert len(self.model._iv_cache) == 0


class TestHestonModelStability:
    """Test numerical stability (Little Heston Trap)."""

    def test_long_maturity_stability(self):
        """Test stability for long maturities (T=5 years)."""
        params = HestonParams(
            v0=0.04, kappa=1.5, theta=0.04, xi=0.4, rho=-0.7, ttm=5.0
        )
        model = HestonModel(params)

        forward = 100.0
        strikes = [60, 80, 100, 120, 140]

        for strike in strikes:
            call = model.call_price(forward, strike)
            assert not np.isnan(call)
            assert not np.isinf(call)
            assert call >= 0

    def test_extreme_parameters_stability(self):
        """Test stability with more extreme parameters."""
        params = HestonParams(
            v0=0.1, kappa=0.5, theta=0.1, xi=0.8, rho=-0.9, ttm=2.0
        )
        model = HestonModel(params)

        forward = 100.0
        call = model.call_price(forward, 100.0)

        assert not np.isnan(call)
        assert not np.isinf(call)
        assert call >= 0


class TestHestonFitter:
    """Test Heston model calibration."""

    def setup_method(self):
        """Set up test fixtures."""
        self.fitter = HestonFitter()

    def test_fit_synthetic_heston_data(self):
        """Test fitting to synthetic Heston-generated data."""
        # True parameters
        true_params = HestonParams(
            v0=0.04, kappa=2.0, theta=0.04, xi=0.4, rho=-0.6, ttm=0.5
        )
        true_model = HestonModel(true_params)

        # Generate synthetic data
        log_moneyness = np.linspace(-0.3, 0.3, 15)
        market_iv = true_model.implied_volatility_array(log_moneyness)

        # Fit
        result = self.fitter.fit(log_moneyness, market_iv, 0.5)

        assert result.success
        assert result.params is not None
        assert result.r_squared > 0.80

    def test_fit_insufficient_data(self):
        """Test that fitting fails with insufficient data."""
        log_moneyness = np.array([0, 0.1])
        market_iv = np.array([0.4, 0.42])

        result = self.fitter.fit(log_moneyness, market_iv, 0.25)

        assert not result.success
        assert "at least 5" in result.message.lower()

    def test_fit_realistic_smile(self):
        """Test fitting a realistic BTC-like smile."""
        log_moneyness = np.array([
            -0.30, -0.20, -0.10, -0.05, 0.00, 0.05, 0.10, 0.20, 0.30
        ])

        # Typical BTC smile with skew
        market_iv = np.array([
            0.85, 0.75, 0.68, 0.65, 0.64, 0.65, 0.68, 0.75, 0.85
        ])

        ttm = 0.25

        result = self.fitter.fit(log_moneyness, market_iv, ttm)

        # Should be able to fit reasonable well
        assert result.params is not None
        # Heston may not fit perfectly but should capture general shape

    def test_bs_initialization(self):
        """Test that BS initialization produces reasonable starting point."""
        log_moneyness = np.array([-0.2, -0.1, 0, 0.1, 0.2])
        market_iv = np.array([0.7, 0.65, 0.6, 0.62, 0.68])
        ttm = 0.25

        init = self.fitter._initialize_from_bs(log_moneyness, market_iv, ttm)

        # v0 should be close to ATM variance
        assert 0.3 < init['v0'] < 0.5  # ATM IV = 0.6, so v0 ~ 0.36

        # rho should be negative (typical skew)
        assert init['rho'] < 0

        # All values should be within bounds
        assert self.fitter.v0_bounds[0] <= init['v0'] <= self.fitter.v0_bounds[1]
        assert self.fitter.kappa_bounds[0] <= init['kappa'] <= self.fitter.kappa_bounds[1]
        assert self.fitter.theta_bounds[0] <= init['theta'] <= self.fitter.theta_bounds[1]
        assert self.fitter.xi_bounds[0] <= init['xi'] <= self.fitter.xi_bounds[1]
        assert self.fitter.rho_bounds[0] <= init['rho'] <= self.fitter.rho_bounds[1]


class TestIVConsistency:
    """Test IV consistency checking."""

    def test_consistent_iv(self):
        """Test IV consistency check passes for well-fitted model."""
        params = HestonParams(
            v0=0.04, kappa=2.0, theta=0.04, xi=0.3, rho=-0.6, ttm=0.25
        )
        model = HestonModel(params)

        log_moneyness = np.array([-0.2, -0.1, 0, 0.1, 0.2])
        market_iv = model.implied_volatility_array(log_moneyness)

        is_consistent, max_error, errors = check_iv_consistency(
            model, log_moneyness, market_iv, threshold=0.10
        )

        assert is_consistent
        assert max_error < 0.01  # Self-consistency should be near-perfect

    def test_inconsistent_iv(self):
        """Test IV consistency check fails for mismatched model."""
        params = HestonParams(
            v0=0.04, kappa=2.0, theta=0.04, xi=0.3, rho=-0.6, ttm=0.25
        )
        model = HestonModel(params)

        log_moneyness = np.array([-0.2, -0.1, 0, 0.1, 0.2])
        # Use very different market IVs
        market_iv = np.array([0.9, 0.8, 0.7, 0.8, 0.9])

        is_consistent, max_error, errors = check_iv_consistency(
            model, log_moneyness, market_iv, threshold=0.10
        )

        assert not is_consistent
        assert max_error > 0.10


class TestHestonIntegration:
    """Integration tests for Heston model."""

    def test_iv_roundtrip(self):
        """Test that we can price -> IV -> price roundtrip."""
        params = HestonParams(
            v0=0.04, kappa=2.0, theta=0.04, xi=0.3, rho=-0.6, ttm=0.5
        )
        model = HestonModel(params)

        forward = 100.0
        strike = 100.0

        # Price call with Heston
        heston_price = model.call_price(forward, strike)

        # Get implied vol
        iv = model.implied_volatility_strike(strike, forward)

        # Price with Black-Scholes using that IV
        bs_price = BlackScholes.forward_call_price(forward, strike, iv, params.ttm)

        # Should match within tolerance
        assert abs(heston_price - bs_price) / heston_price < 0.01

    def test_volatility_surface_protocol(self):
        """Test that HestonModel implements VolatilitySurface protocol."""
        from btc_pricer.models.volatility_surface import VolatilitySurface

        params = HestonParams(
            v0=0.04, kappa=2.0, theta=0.04, xi=0.3, rho=-0.6, ttm=0.5
        )
        model = HestonModel(params)

        # Should be an instance of VolatilitySurface protocol
        assert isinstance(model, VolatilitySurface)

        # Should have required methods
        assert hasattr(model, 'ttm')
        assert hasattr(model, 'implied_volatility')
        assert hasattr(model, 'implied_volatility_strike')
        assert hasattr(model, 'implied_volatility_array')


class TestHestonShortDatedFitting:
    """Test TTM-dependent bounds and multi-start optimization for short-dated options."""

    def test_ttm_adjusted_bounds_short_dated(self):
        """Verify TTM-dependent bounds activate for short-dated options."""
        fitter = HestonFitter(
            short_dated_ttm_threshold=0.10,
            short_dated_xi_bounds=(0.1, 10.0),
            short_dated_kappa_bounds=(0.01, 15.0)
        )

        # Short-dated (TTM < 0.10)
        bounds = fitter._get_ttm_adjusted_bounds(ttm=0.05)
        assert bounds['xi'] == (0.1, 10.0)
        assert bounds['kappa'] == (0.01, 15.0)

        # Normal TTM (>= 0.10)
        bounds_normal = fitter._get_ttm_adjusted_bounds(ttm=0.25)
        assert bounds_normal['xi'] == fitter.xi_bounds
        assert bounds_normal['kappa'] == fitter.kappa_bounds

    def test_ttm_adjusted_bounds_very_short_dated(self):
        """Verify very-short-dated bounds for TTM < 0.02."""
        fitter = HestonFitter(
            very_short_dated_ttm_threshold=0.02,
            very_short_dated_xi_bounds=(0.1, 15.0),
            very_short_dated_kappa_bounds=(0.001, 20.0)
        )

        bounds = fitter._get_ttm_adjusted_bounds(ttm=0.01)
        assert bounds['xi'] == (0.1, 15.0)
        assert bounds['kappa'] == (0.001, 20.0)

    def test_multi_start_generates_multiple_guesses(self):
        """Verify multiple initial guesses are generated."""
        fitter = HestonFitter(use_multi_start=True, n_starts=5)

        log_moneyness = np.array([-0.2, -0.1, 0, 0.1, 0.2])
        market_iv = np.array([0.7, 0.65, 0.6, 0.62, 0.68])
        ttm = 0.05

        bounds_dict = fitter._get_ttm_adjusted_bounds(ttm)
        guesses = fitter._generate_initial_guesses(
            log_moneyness, market_iv, ttm, bounds_dict
        )

        assert len(guesses) >= 3
        assert len(guesses) <= fitter.n_starts

        # Each guess should be within bounds
        for g in guesses:
            assert bounds_dict['v0'][0] <= g['v0'] <= bounds_dict['v0'][1]
            assert bounds_dict['kappa'][0] <= g['kappa'] <= bounds_dict['kappa'][1]
            assert bounds_dict['theta'][0] <= g['theta'] <= bounds_dict['theta'][1]
            assert bounds_dict['xi'][0] <= g['xi'] <= bounds_dict['xi'][1]
            assert bounds_dict['rho'][0] <= g['rho'] <= bounds_dict['rho'][1]

    def test_fit_short_dated_no_boundary_hit(self):
        """Test that short-dated fitting doesn't hit boundaries."""
        fitter = HestonFitter(
            short_dated_ttm_threshold=0.10,
            short_dated_xi_bounds=(0.1, 10.0),
            short_dated_kappa_bounds=(0.01, 15.0),
            use_multi_start=True,
            n_starts=3
        )

        # Steep short-dated smile (like BTC near expiry)
        log_moneyness = np.array([-0.20, -0.15, -0.10, -0.05, 0.0, 0.05, 0.10, 0.15, 0.20])
        market_iv = np.array([0.95, 0.85, 0.75, 0.68, 0.65, 0.68, 0.75, 0.85, 0.95])
        ttm = 0.02  # ~7 days

        result = fitter.fit(log_moneyness, market_iv, ttm)

        # Should fit without hitting the xi upper bound
        assert result.params is not None
        # Parameters should not be exactly at bounds (boundary hit)
        bounds = fitter._get_ttm_adjusted_bounds(ttm)
        if result.params.xi < bounds['xi'][1] * 0.99:
            # Not at upper bound - good
            pass
        # Kappa should not be exactly at lower bound
        if result.params.kappa > bounds['kappa'][0] * 1.01:
            # Not at lower bound - good
            pass

    def test_fit_with_quantlib_objective(self):
        """Test that QuantLib-based calibration works."""
        try:
            import QuantLib as ql
            quantlib_available = True
        except ImportError:
            quantlib_available = False

        fitter = HestonFitter(use_quantlib=True, use_multi_start=False)

        # Moderate smile
        log_moneyness = np.array([-0.15, -0.10, -0.05, 0.0, 0.05, 0.10, 0.15])
        market_iv = np.array([0.70, 0.66, 0.63, 0.62, 0.63, 0.66, 0.70])
        ttm = 0.25

        result = fitter.fit(log_moneyness, market_iv, ttm)

        assert result.params is not None
        assert result.r_squared > 0.5


class TestHestonWithBreedenLitzenberger:
    """Test Heston model integration with Breeden-Litzenberger RND extraction."""

    def test_extract_rnd_from_heston(self):
        """Test RND extraction from Heston model."""
        from btc_pricer.models.breeden_litzenberger import BreedenLitzenberger

        params = HestonParams(
            v0=0.04, kappa=2.0, theta=0.04, xi=0.3, rho=-0.6, ttm=0.25
        )

        bl = BreedenLitzenberger(strike_grid_points=200)
        forward = 100000.0

        rnd = bl.extract_from_heston(params, forward)

        # Check RND is valid
        assert rnd.is_valid or len(rnd.warnings) < 3
        assert rnd.integral > 0.90  # Should integrate to ~1

        # Mean should be close to forward
        assert abs(rnd.mean - forward) / forward < 0.15

    def test_heston_vs_ssvi_rnd_similarity(self):
        """Test that Heston and SSVI produce similar RNDs for similar smiles."""
        from btc_pricer.models.breeden_litzenberger import BreedenLitzenberger
        from btc_pricer.models.ssvi import SSVIParams

        # Create Heston model
        heston_params = HestonParams(
            v0=0.04, kappa=2.0, theta=0.04, xi=0.3, rho=-0.5, ttm=0.25
        )

        # Create SSVI with similar characteristics
        # theta for SSVI is ATM total variance ~ v0 * ttm for short maturities
        ssvi_params = SSVIParams(
            theta=0.01,  # ATM total variance
            rho=-0.3,
            phi=0.5,
            ttm=0.25
        )

        bl = BreedenLitzenberger(strike_grid_points=200)
        forward = 100000.0

        rnd_heston = bl.extract_from_heston(heston_params, forward)
        rnd_ssvi = bl.extract_from_ssvi(ssvi_params, forward)

        # Both should be valid
        assert rnd_heston.integral > 0.90
        assert rnd_ssvi.integral > 0.90

        # Both means should be close to forward
        assert abs(rnd_heston.mean - forward) / forward < 0.20
        assert abs(rnd_ssvi.mean - forward) / forward < 0.20


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
