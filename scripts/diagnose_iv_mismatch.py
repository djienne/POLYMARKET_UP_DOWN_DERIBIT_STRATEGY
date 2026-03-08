#!/usr/bin/env python3
"""Diagnose the IV computation mismatch between calibration and plotting."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

# Import QuantLib
try:
    import QuantLib as ql
    QUANTLIB_AVAILABLE = True
except ImportError:
    QUANTLIB_AVAILABLE = False
    print("QuantLib not available!")
    sys.exit(1)

from btc_pricer.models.heston import HestonModel, HestonParams


def get_iv_via_vol_surface(v0, kappa, theta, xi, rho, ttm, forward, log_moneyness):
    """Get IVs using HestonBlackVolSurface (used in calibration)."""
    today = ql.Date.todaysDate()
    ql.Settings.instance().evaluationDate = today
    day_counter = ql.Actual365Fixed()

    rate_ts = ql.YieldTermStructureHandle(ql.FlatForward(today, 0.0, day_counter))
    div_ts = ql.YieldTermStructureHandle(ql.FlatForward(today, 0.0, day_counter))
    spot_handle = ql.QuoteHandle(ql.SimpleQuote(forward))

    process = ql.HestonProcess(rate_ts, div_ts, spot_handle, v0, kappa, theta, xi, rho)
    model = ql.HestonModel(process)
    heston_handle = ql.HestonModelHandle(model)
    vol_surface = ql.HestonBlackVolSurface(heston_handle)

    ivs = []
    for k in log_moneyness:
        strike = forward * np.exp(k)
        iv = vol_surface.blackVol(ttm, strike)
        ivs.append(iv)

    return np.array(ivs)


def get_iv_via_option_pricing(v0, kappa, theta, xi, rho, ttm, forward, log_moneyness):
    """Get IVs using option pricing + BS inversion (used in plotting)."""
    params = HestonParams(v0=v0, kappa=kappa, theta=theta, xi=xi, rho=rho, ttm=ttm)
    model = HestonModel(params, use_quantlib=True)

    # This uses call_price() -> BS inversion
    ivs = model.implied_volatility_array(log_moneyness)
    return ivs


def main():
    # Test parameters - the "bad" solution that optimizer finds
    v0_bad = 0.034
    kappa_bad = 8.25
    theta_bad = 4.0
    xi_bad = 5.0
    rho_bad = -0.3

    # The "expected" solution based on ATM IV
    v0_good = 0.14  # ATM IV ~ 37%, so v0 ~ 0.37^2 ~ 0.14
    kappa_good = 2.0
    theta_good = 0.14
    xi_good = 3.0
    rho_good = -0.3

    ttm = 3.0 / 365  # ~3 days
    forward = 100000.0

    # Log-moneyness range
    log_moneyness = np.linspace(-0.2, 0.2, 21)

    print("=" * 80)
    print("IV COMPUTATION METHOD COMPARISON")
    print("=" * 80)

    print(f"\nTTM: {ttm:.4f} years ({ttm*365:.1f} days)")
    print(f"Forward: ${forward:,.0f}")

    print("\n" + "-" * 80)
    print("BAD SOLUTION (optimizer finds this)")
    print(f"v0={v0_bad}, kappa={kappa_bad}, theta={theta_bad}, xi={xi_bad}, rho={rho_bad}")
    print("-" * 80)

    iv_vol_surface_bad = get_iv_via_vol_surface(
        v0_bad, kappa_bad, theta_bad, xi_bad, rho_bad, ttm, forward, log_moneyness
    )
    iv_option_pricing_bad = get_iv_via_option_pricing(
        v0_bad, kappa_bad, theta_bad, xi_bad, rho_bad, ttm, forward, log_moneyness
    )

    print(f"\n{'k':>8} | {'Vol Surface':>12} | {'Option Pricing':>14} | {'Diff':>8}")
    print("-" * 50)
    for i, k in enumerate(log_moneyness[::4]):  # Show every 4th point
        idx = i * 4
        iv1 = iv_vol_surface_bad[idx] * 100
        iv2 = iv_option_pricing_bad[idx] * 100
        diff = (iv1 - iv2)
        print(f"{k:>8.3f} | {iv1:>11.2f}% | {iv2:>13.2f}% | {diff:>+7.2f}%")

    atm_idx = len(log_moneyness) // 2
    print(f"\nATM (k=0): VolSurface={iv_vol_surface_bad[atm_idx]*100:.2f}%, "
          f"OptionPricing={iv_option_pricing_bad[atm_idx]*100:.2f}%")

    print("\n" + "-" * 80)
    print("GOOD SOLUTION (expected based on ATM IV)")
    print(f"v0={v0_good}, kappa={kappa_good}, theta={theta_good}, xi={xi_good}, rho={rho_good}")
    print("-" * 80)

    iv_vol_surface_good = get_iv_via_vol_surface(
        v0_good, kappa_good, theta_good, xi_good, rho_good, ttm, forward, log_moneyness
    )
    iv_option_pricing_good = get_iv_via_option_pricing(
        v0_good, kappa_good, theta_good, xi_good, rho_good, ttm, forward, log_moneyness
    )

    print(f"\n{'k':>8} | {'Vol Surface':>12} | {'Option Pricing':>14} | {'Diff':>8}")
    print("-" * 50)
    for i, k in enumerate(log_moneyness[::4]):
        idx = i * 4
        iv1 = iv_vol_surface_good[idx] * 100
        iv2 = iv_option_pricing_good[idx] * 100
        diff = (iv1 - iv2)
        print(f"{k:>8.3f} | {iv1:>11.2f}% | {iv2:>13.2f}% | {diff:>+7.2f}%")

    print(f"\nATM (k=0): VolSurface={iv_vol_surface_good[atm_idx]*100:.2f}%, "
          f"OptionPricing={iv_option_pricing_good[atm_idx]*100:.2f}%")

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    bad_diff = np.abs(iv_vol_surface_bad - iv_option_pricing_bad) * 100
    good_diff = np.abs(iv_vol_surface_good - iv_option_pricing_good) * 100
    print(f"\nBAD solution max discrepancy: {bad_diff.max():.2f}%")
    print(f"GOOD solution max discrepancy: {good_diff.max():.2f}%")

    if bad_diff.max() > 1.0:
        print("\n[!] SIGNIFICANT MISMATCH DETECTED for BAD solution!")
        print("    This explains why the optimizer finds a solution that looks good")
        print("    during calibration (using VolSurface) but gives wrong IVs")
        print("    during plotting (using option pricing + BS inversion).")


if __name__ == "__main__":
    main()
