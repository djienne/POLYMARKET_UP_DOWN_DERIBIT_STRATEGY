#!/usr/bin/env python3
"""Diagnose the calibration bug causing ~1% model IV."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import QuantLib as ql

from btc_pricer.models.heston import HestonModel, HestonParams


def main():
    # Parameters from the failed fit
    v0 = 0.1682
    kappa = 0.391
    theta = 0.2814
    xi = 0.514
    rho = -0.330
    ttm = 0.9 / 365  # 0.9 days

    forward = 89388.0  # From the plot
    log_moneyness = np.array([-0.06, -0.04, -0.02, 0.0, 0.02])

    print("=" * 80)
    print("DIAGNOSING CALIBRATION BUG")
    print("=" * 80)
    print(f"\nParameters: v0={v0}, kappa={kappa}, theta={theta}, xi={xi}, rho={rho}")
    print(f"TTM: {ttm:.6f} years ({ttm*365:.2f} days)")
    print(f"Forward: ${forward:,.0f}")

    # Check what days_to_expiry becomes
    days_to_expiry = int(ttm * 365)
    print(f"\n[!] days_to_expiry = int({ttm} * 365) = {days_to_expiry}")

    if days_to_expiry == 0:
        print("[BUG CONFIRMED] TTM rounds to 0 days in QuantLib setup!")
        print("This causes option price = intrinsic value, leading to ~0% model IV")

    # Create model
    params = HestonParams(v0=v0, kappa=kappa, theta=theta, xi=xi, rho=rho, ttm=ttm)
    model = HestonModel(params, use_quantlib=True)

    # Test with normalized forward=1 (what plotting uses)
    print("\n" + "-" * 80)
    print("Method 1: implied_volatility(k) - normalized forward=1 (used in plotting)")
    print("-" * 80)
    for k in log_moneyness:
        iv = model.implied_volatility(k)
        print(f"k={k:+.4f}: IV={iv*100:.2f}%")

    # Test with actual forward/strikes (what my buggy objective used)
    print("\n" + "-" * 80)
    print("Method 2: implied_volatility_strike(strike, forward) - actual prices")
    print("-" * 80)
    strikes = forward * np.exp(log_moneyness)
    for i, k in enumerate(log_moneyness):
        iv = model.implied_volatility_strike(strikes[i], forward)
        call_price = model.call_price(forward, strikes[i])
        intrinsic = max(0, forward - strikes[i])
        print(f"k={k:+.4f}: strike=${strikes[i]:,.0f}, call=${call_price:.2f}, "
              f"intrinsic=${intrinsic:.2f}, IV={iv*100:.2f}%")

    # Check raw QuantLib setup
    print("\n" + "-" * 80)
    print("QuantLib expiry date check")
    print("-" * 80)
    today = ql.Date.todaysDate()
    print(f"Today: {today}")

    # What the model does
    days_int = int(ttm * 365)
    expiry_int = today + ql.Period(days_int, ql.Days)
    print(f"With int(TTM*365)={days_int}: expiry={expiry_int}")

    # What it should do
    days_ceil = max(1, int(np.ceil(ttm * 365)))
    expiry_ceil = today + ql.Period(days_ceil, ql.Days)
    print(f"With ceil(TTM*365)={days_ceil}: expiry={expiry_ceil}")

    # Expected ATM IV from v0
    print("\n" + "-" * 80)
    print("Expected values")
    print("-" * 80)
    expected_atm_iv = np.sqrt(v0)
    print(f"Expected ATM IV from sqrt(v0): {expected_atm_iv*100:.2f}%")
    print(f"Market ATM IV (from plot): ~40%")


if __name__ == "__main__":
    main()
