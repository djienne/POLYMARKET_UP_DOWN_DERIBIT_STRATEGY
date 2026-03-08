"""Integration tests for the full pipeline."""

import pytest
import numpy as np
from unittest.mock import Mock, patch

from btc_pricer.config import Config
from btc_pricer.api.deribit import OptionData
from btc_pricer.data.filters import DataFilter, FilteredOption
from btc_pricer.models.black_scholes import BlackScholes
from btc_pricer.models.ssvi import SSVIFitter
from btc_pricer.models.breeden_litzenberger import BreedenLitzenberger
from btc_pricer.statistics.rnd_stats import RNDStatistics
from btc_pricer.utils.sanity_checks import SanityChecker, CheckStatus


class TestFullPipeline:
    """Test the full RND extraction pipeline."""

    def create_mock_options(self, forward=100000, vol=0.7, ttm=0.25, n_strikes=20):
        """Create mock option data for testing."""
        spot = forward  # Simplified
        strikes = np.linspace(forward * 0.6, forward * 1.5, n_strikes)

        options = []
        for strike in strikes:
            # Determine option type (OTM)
            if strike < forward:
                option_type = "put"
                price_btc = BlackScholes.inverse_put_price_btc(forward, strike, vol, ttm)
            else:
                option_type = "call"
                price_btc = BlackScholes.inverse_call_price_btc(forward, strike, vol, ttm)

            # Add small spread
            bid = price_btc * 0.95
            ask = price_btc * 1.05

            opt = OptionData(
                instrument_name=f"BTC-TEST-{int(strike)}-{'C' if option_type == 'call' else 'P'}",
                strike=strike,
                option_type=option_type,
                expiration_timestamp=int((ttm * 365 * 24 * 3600 + 1e9) * 1000),
                expiration_date="28MAR25",
                bid_price=bid,
                ask_price=ask,
                mark_price=price_btc,
                mark_iv=vol,  # Flat vol for simplicity
                bid_iv=None,
                ask_iv=None,
                open_interest=100,
                underlying_price=forward,
                spot_price=spot,
                time_to_expiry=ttm
            )
            options.append(opt)

        return options

    def test_filtering(self):
        """Test data filtering step."""
        options = self.create_mock_options()
        data_filter = DataFilter()

        filtered, stats = data_filter.filter_options(options, return_stats=True)

        assert stats is not None
        assert stats.passed_filters > 0
        assert len(filtered) > 0

    def test_otm_surface_construction(self):
        """Test OTM surface construction."""
        options = self.create_mock_options()
        data_filter = DataFilter()

        filtered, _ = data_filter.filter_options(options, return_stats=True)
        otm_surface = data_filter.build_otm_surface(filtered)

        # Check that we have both puts and calls
        has_puts = any(opt.option_type == "put" for opt in otm_surface)
        has_calls = any(opt.option_type == "call" for opt in otm_surface)

        assert has_puts
        assert has_calls

        # Check OTM property
        for opt in otm_surface:
            if opt.option_type == "put":
                assert opt.strike <= opt.forward_price
            else:
                assert opt.strike >= opt.forward_price

    def test_ssvi_fitting(self):
        """Test SSVI fitting on filtered data."""
        # Create options with a smile (vary vol with moneyness)
        forward = 100000
        ttm = 0.25
        n_strikes = 20

        strikes = np.linspace(forward * 0.6, forward * 1.5, n_strikes)
        options = []

        for strike in strikes:
            # Create a smile: higher vol for OTM options
            log_k = np.log(strike / forward)
            vol = 0.6 + 0.3 * abs(log_k)  # Smile shape

            if strike < forward:
                option_type = "put"
                price_btc = BlackScholes.inverse_put_price_btc(forward, strike, vol, ttm)
            else:
                option_type = "call"
                price_btc = BlackScholes.inverse_call_price_btc(forward, strike, vol, ttm)

            bid = price_btc * 0.95
            ask = price_btc * 1.05

            opt = OptionData(
                instrument_name=f"BTC-TEST-{int(strike)}-{'C' if option_type == 'call' else 'P'}",
                strike=strike,
                option_type=option_type,
                expiration_timestamp=int((ttm * 365 * 24 * 3600 + 1e9) * 1000),
                expiration_date="28MAR25",
                bid_price=bid,
                ask_price=ask,
                mark_price=price_btc,
                mark_iv=vol,
                bid_iv=None,
                ask_iv=None,
                open_interest=100,
                underlying_price=forward,
                spot_price=forward,
                time_to_expiry=ttm
            )
            options.append(opt)

        data_filter = DataFilter()
        filtered, _ = data_filter.filter_options(options)
        otm_surface = data_filter.build_otm_surface(filtered)

        # Extract data
        log_moneyness = np.array([opt.log_moneyness for opt in otm_surface])
        market_iv = np.array([opt.mark_iv for opt in otm_surface])
        ttm = otm_surface[0].time_to_expiry

        # Fit SSVI
        fitter = SSVIFitter()
        result = fitter.fit(log_moneyness, market_iv, ttm)

        assert result.success
        assert result.params is not None
        assert result.r_squared > 0.7  # Acceptable fit for synthetic smile

    def test_rnd_extraction(self):
        """Test RND extraction from fitted SSVI."""
        options = self.create_mock_options(vol=0.6)
        data_filter = DataFilter()

        filtered, _ = data_filter.filter_options(options)
        otm_surface = data_filter.build_otm_surface(filtered)

        log_moneyness = np.array([opt.log_moneyness for opt in otm_surface])
        market_iv = np.array([opt.mark_iv for opt in otm_surface])
        ttm = otm_surface[0].time_to_expiry
        forward = otm_surface[0].forward_price

        # Fit and extract
        fitter = SSVIFitter()
        fit_result = fitter.fit(log_moneyness, market_iv, ttm)

        bl = BreedenLitzenberger()
        rnd = bl.extract_from_ssvi(fit_result.params, forward)

        assert rnd.is_valid or len(rnd.warnings) < 3
        assert 0.9 < rnd.integral < 1.1
        assert rnd.percentile_5 < rnd.percentile_50 < rnd.percentile_95

    def test_statistics_computation(self):
        """Test statistics computation."""
        options = self.create_mock_options()
        data_filter = DataFilter()

        filtered, _ = data_filter.filter_options(options)
        otm_surface = data_filter.build_otm_surface(filtered)

        log_moneyness = np.array([opt.log_moneyness for opt in otm_surface])
        market_iv = np.array([opt.mark_iv for opt in otm_surface])
        ttm = otm_surface[0].time_to_expiry
        forward = otm_surface[0].forward_price
        spot = otm_surface[0].spot_price

        fitter = SSVIFitter()
        fit_result = fitter.fit(log_moneyness, market_iv, ttm)

        bl = BreedenLitzenberger()
        rnd = bl.extract_from_ssvi(fit_result.params, forward)

        stats_calc = RNDStatistics()
        stats = stats_calc.compute_stats(rnd, "TEST", spot)

        assert stats.mean > 0
        assert stats.std_dev > 0
        assert len(stats.scenarios) > 0

    def test_sanity_checks(self):
        """Test sanity checking system."""
        checker = SanityChecker()

        # Test API check
        api_check = checker.check_api_data(100000, 50, "TEST")
        assert api_check.overall_status == CheckStatus.PASS

        # Test with extreme spot
        bad_api_check = checker.check_api_data(100, 50, "TEST2")
        assert bad_api_check.overall_status == CheckStatus.CRITICAL

    def test_full_pipeline_with_smirk(self):
        """Test full pipeline with typical BTC smile/smirk."""
        # Create data with negative skew
        forward = 100000
        ttm = 0.25
        n_strikes = 25

        strikes = np.linspace(forward * 0.5, forward * 1.5, n_strikes)
        log_k = np.log(strikes / forward)

        # Create smirk: higher IV for lower strikes
        base_vol = 0.65
        skew = -0.3
        market_iv = base_vol + skew * log_k + 0.2 * log_k**2

        # Create filtered options directly
        filtered_options = []
        for i, (k, s, iv) in enumerate(zip(log_k, strikes, market_iv)):
            if s < forward:
                opt_type = "put"
                price = BlackScholes.inverse_put_price_btc(forward, s, iv, ttm)
            else:
                opt_type = "call"
                price = BlackScholes.inverse_call_price_btc(forward, s, iv, ttm)

            filtered_options.append(FilteredOption(
                instrument_name=f"TEST-{i}",
                strike=s,
                option_type=opt_type,
                time_to_expiry=ttm,
                forward_price=forward,
                spot_price=forward,
                bid_price_btc=price * 0.98,
                ask_price_btc=price * 1.02,
                mid_price_btc=price,
                mark_price_btc=price,
                mark_iv=iv,
                open_interest=100,
                moneyness=s / forward,
                log_moneyness=k,
                is_otm=(opt_type == "put" and s < forward) or (opt_type == "call" and s > forward)
            ))

        # Fit SSVI
        fitter = SSVIFitter()
        fit_result = fitter.fit(log_k, market_iv, ttm)

        assert fit_result.success
        assert fit_result.params.rho < 0  # Should capture negative skew

        # Extract RND
        bl = BreedenLitzenberger()
        rnd = bl.extract_from_ssvi(fit_result.params, forward)

        assert rnd.is_valid or len(rnd.warnings) < 3

        # With negative skew, should have negative skewness in RND
        # (actually depends on the shape - fat left tail gives negative skew)

        # Statistics
        stats_calc = RNDStatistics()
        stats = stats_calc.compute_stats(rnd, "TEST", forward)

        # Format output
        summary = stats_calc.format_summary(stats)
        assert "TEST" in summary
        assert "Mean" in summary


class TestConfigIntegration:
    """Test configuration integration."""

    def test_default_config(self):
        """Test default configuration works."""
        config = Config()

        assert config.api.base_url.startswith("https://")
        assert config.filters.min_open_interest >= 0
        assert config.ssvi.optimizer in ["L-BFGS-B", "SLSQP"]

    def test_config_to_dict(self):
        """Test config serialization."""
        config = Config()
        d = config.to_dict()

        assert "api" in d
        assert "filters" in d
        assert "ssvi" in d


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_insufficient_options(self):
        """Test handling of insufficient options."""
        data_filter = DataFilter()

        # Empty list
        filtered, stats = data_filter.filter_options([], return_stats=True)
        assert len(filtered) == 0

    def test_one_sided_smile(self):
        """Test handling of one-sided data."""
        data_filter = DataFilter()

        # Create options only above forward (all calls)
        forward = 100000
        ttm = 0.25
        vol = 0.7

        options = []
        for strike in [105000, 110000, 115000, 120000, 125000]:
            price_btc = BlackScholes.inverse_call_price_btc(forward, strike, vol, ttm)

            opt = OptionData(
                instrument_name=f"BTC-TEST-{strike}-C",
                strike=strike,
                option_type="call",
                expiration_timestamp=int(1e12),
                expiration_date="TEST",
                bid_price=price_btc * 0.95,
                ask_price=price_btc * 1.05,
                mark_price=price_btc,
                mark_iv=vol,
                bid_iv=None,
                ask_iv=None,
                open_interest=100,
                underlying_price=forward,
                spot_price=forward,
                time_to_expiry=ttm
            )
            options.append(opt)

        filtered, _ = data_filter.filter_options(options)

        # Validate coverage should warn
        is_valid, msg = data_filter.validate_surface_coverage(filtered)
        assert not is_valid
        assert "below" in msg.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
