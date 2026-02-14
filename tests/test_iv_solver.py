"""Tests for Black-Scholes IV solver."""

import pytest
import math
from btc_pricer.models.black_scholes import BlackScholes


class TestBlackScholes:
    """Test Black-Scholes pricing and IV solver."""

    def test_d1_d2_basic(self):
        """Test d1 and d2 calculations."""
        forward = 100000
        strike = 100000  # ATM
        vol = 0.80
        ttm = 0.25

        d1 = BlackScholes.d1(forward, strike, vol, ttm)
        d2 = BlackScholes.d2(forward, strike, vol, ttm)

        # For ATM, d1 = vol * sqrt(T) / 2
        expected_d1 = vol * math.sqrt(ttm) / 2
        assert abs(d1 - expected_d1) < 1e-10

        # d2 = d1 - vol * sqrt(T)
        assert abs(d2 - (d1 - vol * math.sqrt(ttm))) < 1e-10

    def test_inverse_call_price_atm(self):
        """Test inverse call price at ATM."""
        forward = 100000
        strike = 100000
        vol = 0.80
        ttm = 0.25

        price_btc = BlackScholes.inverse_call_price_btc(forward, strike, vol, ttm)

        # ATM call price should be approximately 0.4 * vol * sqrt(T) for ATM
        # This is a rough approximation
        assert 0.1 < price_btc < 0.3

    def test_inverse_put_price_atm(self):
        """Test inverse put price at ATM."""
        forward = 100000
        strike = 100000
        vol = 0.80
        ttm = 0.25

        price_btc = BlackScholes.inverse_put_price_btc(forward, strike, vol, ttm)

        # ATM put should be similar to call
        call_btc = BlackScholes.inverse_call_price_btc(forward, strike, vol, ttm)
        assert abs(price_btc - call_btc) < 0.01

    def test_put_call_parity_btc(self):
        """Test put-call parity for inverse options."""
        forward = 100000
        strike = 90000
        vol = 0.80
        ttm = 0.25

        call_btc = BlackScholes.inverse_call_price_btc(forward, strike, vol, ttm)
        put_btc = BlackScholes.inverse_put_price_btc(forward, strike, vol, ttm)

        # Put-call parity: C - P = 1 - K/F (for inverse options)
        parity_lhs = call_btc - put_btc
        parity_rhs = 1 - strike / forward

        assert abs(parity_lhs - parity_rhs) < 1e-10

    def test_forward_call_price(self):
        """Test forward call price in USD."""
        forward = 100000
        strike = 100000
        vol = 0.80
        ttm = 0.25

        price_usd = BlackScholes.forward_call_price(forward, strike, vol, ttm)

        # Should equal F * C_BTC
        price_btc = BlackScholes.inverse_call_price_btc(forward, strike, vol, ttm)
        assert abs(price_usd - forward * price_btc) < 1e-6

    def test_iv_solver_call(self):
        """Test IV solver recovers known volatility."""
        forward = 100000
        strike = 95000
        vol = 0.75
        ttm = 0.5

        # Price the option
        price_btc = BlackScholes.inverse_call_price_btc(forward, strike, vol, ttm)

        # Recover IV
        recovered_iv = BlackScholes.implied_volatility(
            price_btc, forward, strike, ttm, "call", is_btc_price=True
        )

        assert recovered_iv is not None
        assert abs(recovered_iv - vol) < 1e-6

    def test_iv_solver_put(self):
        """Test IV solver for puts."""
        forward = 100000
        strike = 105000
        vol = 0.85
        ttm = 0.25

        price_btc = BlackScholes.inverse_put_price_btc(forward, strike, vol, ttm)

        recovered_iv = BlackScholes.implied_volatility(
            price_btc, forward, strike, ttm, "put", is_btc_price=True
        )

        assert recovered_iv is not None
        assert abs(recovered_iv - vol) < 1e-6

    def test_iv_solver_otm_call(self):
        """Test IV solver for OTM call."""
        forward = 100000
        strike = 150000  # Far OTM
        vol = 1.0
        ttm = 0.25

        price_btc = BlackScholes.inverse_call_price_btc(forward, strike, vol, ttm)

        recovered_iv = BlackScholes.implied_volatility(
            price_btc, forward, strike, ttm, "call", is_btc_price=True
        )

        assert recovered_iv is not None
        assert abs(recovered_iv - vol) < 1e-5

    def test_iv_solver_otm_put(self):
        """Test IV solver for OTM put."""
        forward = 100000
        strike = 60000  # Far OTM
        vol = 0.90
        ttm = 0.5

        price_btc = BlackScholes.inverse_put_price_btc(forward, strike, vol, ttm)

        recovered_iv = BlackScholes.implied_volatility(
            price_btc, forward, strike, ttm, "put", is_btc_price=True
        )

        assert recovered_iv is not None
        assert abs(recovered_iv - vol) < 1e-5

    def test_iv_solver_fails_invalid_price(self):
        """Test IV solver returns None for invalid price."""
        forward = 100000
        strike = 100000
        ttm = 0.25

        # Price too high (impossible)
        iv = BlackScholes.implied_volatility(
            1.5, forward, strike, ttm, "call", is_btc_price=True
        )
        assert iv is None

        # Negative price
        iv = BlackScholes.implied_volatility(
            -0.1, forward, strike, ttm, "call", is_btc_price=True
        )
        assert iv is None

    def test_vega(self):
        """Test vega calculation."""
        forward = 100000
        strike = 100000
        vol = 0.80
        ttm = 0.25

        vega = BlackScholes.vega_btc(forward, strike, vol, ttm)

        # Vega should be positive
        assert vega > 0

        # ATM vega should be largest
        vega_otm = BlackScholes.vega_btc(forward, 150000, vol, ttm)
        assert vega > vega_otm

    def test_delta_call(self):
        """Test call delta."""
        forward = 100000
        vol = 0.80
        ttm = 0.25

        # ATM delta: N(d1) where d1 = vol*sqrt(T)/2 for ATM
        # For high vol, delta is shifted above 0.5
        delta_atm = BlackScholes.delta_call_btc(forward, forward, vol, ttm)
        assert 0.45 < delta_atm < 0.65  # Allow for vol effect

        # ITM delta should be high
        delta_itm = BlackScholes.delta_call_btc(forward, 50000, vol, ttm)
        assert delta_itm > 0.9

        # OTM delta should be low
        delta_otm = BlackScholes.delta_call_btc(forward, 200000, vol, ttm)
        assert delta_otm < 0.15

    def test_delta_put(self):
        """Test put delta."""
        forward = 100000
        vol = 0.80
        ttm = 0.25

        # ATM put delta: N(d1) - 1
        delta_atm = BlackScholes.delta_put_btc(forward, forward, vol, ttm)
        assert -0.55 < delta_atm < -0.35  # Allow for vol effect

        # Call-put delta relation
        delta_call = BlackScholes.delta_call_btc(forward, 90000, vol, ttm)
        delta_put = BlackScholes.delta_put_btc(forward, 90000, vol, ttm)
        assert abs(delta_call - delta_put - 1) < 1e-10

    def test_expired_options(self):
        """Test pricing of expired options."""
        forward = 100000
        vol = 0.80
        ttm = 0  # Expired

        # ITM call
        call_itm = BlackScholes.inverse_call_price_btc(forward, 90000, vol, ttm)
        assert abs(call_itm - (1 - 90000/forward)) < 1e-10

        # OTM call
        call_otm = BlackScholes.inverse_call_price_btc(forward, 110000, vol, ttm)
        assert call_otm == 0

        # ITM put
        put_itm = BlackScholes.inverse_put_price_btc(forward, 110000, vol, ttm)
        assert abs(put_itm - (110000/forward - 1)) < 1e-10

        # OTM put
        put_otm = BlackScholes.inverse_put_price_btc(forward, 90000, vol, ttm)
        assert put_otm == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
