#!/usr/bin/env python3
"""Compare SSVI vs Heston calibration: timing and fit quality for closest TTM."""

import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from btc_pricer.config import Config
from btc_pricer.api.deribit import DeribitClient
from btc_pricer.api.binance import fetch_spot_with_fallback
from btc_pricer.data.filters import DataFilter
from btc_pricer.cli.common import (
    create_ssvi_fitter,
    create_heston_fitter,
    extract_surface_data,
    parse_expiry_date,
)

import numpy as np


def main():
    config = Config.from_yaml(Path("config.yaml"))

    # Fetch options
    client = DeribitClient(config.api, config.validation)
    print("Fetching options from Deribit...")
    options_by_expiry = client.fetch_all_options("BTC")

    if not options_by_expiry:
        print("No options data received")
        return

    deribit_spot = list(options_by_expiry.values())[0][0].spot_price
    spot_price, _ = fetch_spot_with_fallback(deribit_spot)
    print(f"Spot price: ${spot_price:,.0f}")

    # Find closest TTM expiry
    sorted_expiries = sorted(
        options_by_expiry.items(),
        key=lambda x: x[1][0].time_to_expiry if x[1] else float('inf')
    )
    expiry, options = sorted_expiries[0]
    ttm_days = options[0].time_to_expiry * 365
    print(f"\nClosest expiry: {expiry} (TTM: {ttm_days:.1f} days)")
    print(f"Raw options: {len(options)}")

    # Filter
    data_filter = DataFilter(config.filters)
    filtered, stats = data_filter.filter_options(options, return_stats=True)
    otm_surface = data_filter.build_otm_surface(filtered)

    surface_data = extract_surface_data(
        otm_surface,
        min_points=4,
        iv_valid_range=config.validation.iv_valid_range,
    )
    if surface_data is None:
        print("Insufficient data after filtering")
        return

    forward, ttm, _, log_moneyness, market_iv = surface_data
    print(f"Filtered OTM options: {len(log_moneyness)}")
    print(f"Forward: ${forward:,.0f}, TTM: {ttm:.6f} yr ({ttm*365:.2f} days)")
    print(f"Log-moneyness range: [{log_moneyness.min():.4f}, {log_moneyness.max():.4f}]")
    print(f"IV range: [{market_iv.min():.2%}, {market_iv.max():.2%}]")

    # ── SSVI Calibration ──
    print("\n" + "=" * 60)
    print("SSVI CALIBRATION")
    print("=" * 60)
    ssvi_fitter = create_ssvi_fitter(config)

    t0 = time.perf_counter()
    ssvi_result = ssvi_fitter.fit(log_moneyness, market_iv, ttm)
    ssvi_time = time.perf_counter() - t0

    if ssvi_result.success and ssvi_result.params is not None:
        p = ssvi_result.params
        print(f"  Status:       SUCCESS")
        print(f"  Time:         {ssvi_time:.3f}s")
        print(f"  R-squared:    {ssvi_result.r_squared:.6f}")
        print(f"  RMSE:         {ssvi_result.rmse:.6f}")
        print(f"  Max residual: {ssvi_result.max_residual:.6f}")
        print(f"  N points:     {ssvi_result.n_points}")
        print(f"  Params:       theta={p.theta:.6f}, rho={p.rho:.4f}, phi={p.phi:.4f}")

        # Compute fitted IVs for residual analysis (SSVI formula: w(k) = θ/2 * (1 + ρφk + sqrt((φk + ρ)² + (1-ρ²))))
        phi_k = p.phi * log_moneyness
        fitted_var = (p.theta / 2.0) * (1.0 + p.rho * phi_k + np.sqrt((phi_k + p.rho)**2 + (1 - p.rho**2)))
        fitted_iv = np.sqrt(np.maximum(fitted_var / ttm, 1e-12))
        residuals = market_iv - fitted_iv
        print(f"  Mean abs err: {np.mean(np.abs(residuals)):.6f}")
        print(f"  Mean rel err: {np.mean(np.abs(residuals / market_iv)):.2%}")
    else:
        print(f"  Status: FAILED - {ssvi_result.message}")
        ssvi_time = None

    # ── Heston Calibration ──
    print("\n" + "=" * 60)
    print("HESTON CALIBRATION")
    print("=" * 60)
    # Use production config but override optimizer for speed comparison
    # DE is the production default but takes 10+ minutes on ultra-short TTM
    heston_fitter = create_heston_fitter(config)
    heston_fitter.optimizer = "L-BFGS-B"  # Much faster, still good fits
    print(f"  Optimizer: {heston_fitter.optimizer}, n_starts: {heston_fitter.n_starts}, "
          f"max_workers: {heston_fitter.max_workers}, quantlib: {heston_fitter.use_quantlib}")

    t0 = time.perf_counter()
    heston_result = heston_fitter.fit(log_moneyness, market_iv, ttm, forward=forward)
    heston_time = time.perf_counter() - t0

    if heston_result.success and heston_result.params is not None:
        p = heston_result.params
        print(f"  Status:       SUCCESS")
        print(f"  Time:         {heston_time:.3f}s")
        print(f"  R-squared:    {heston_result.r_squared:.6f}")
        print(f"  RMSE:         {heston_result.rmse:.6f}")
        print(f"  Max residual: {heston_result.max_residual:.6f}")
        print(f"  N points:     {heston_result.n_points}")
        print(f"  Params:       v0={p.v0:.6f}, kappa={p.kappa:.4f}, theta={p.theta:.6f}, xi={p.xi:.4f}, rho={p.rho:.4f}")

        # Compute fitted IVs
        from btc_pricer.models.heston import HestonModel
        heston_model = HestonModel(p, use_quantlib=config.heston.use_quantlib)
        fitted_iv_h = []
        for lm in log_moneyness:
            try:
                iv = heston_model.implied_volatility(lm, ttm)
                fitted_iv_h.append(iv)
            except Exception:
                fitted_iv_h.append(np.nan)
        fitted_iv_h = np.array(fitted_iv_h)
        valid = ~np.isnan(fitted_iv_h)
        if valid.any():
            residuals_h = market_iv[valid] - fitted_iv_h[valid]
            print(f"  Mean abs err: {np.mean(np.abs(residuals_h)):.6f}")
            print(f"  Mean rel err: {np.mean(np.abs(residuals_h / market_iv[valid])):.2%}")
    else:
        print(f"  Status: FAILED - {heston_result.message}")
        heston_time = None

    # ── Comparison ──
    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)
    print(f"  {'Metric':<20} {'SSVI':>15} {'Heston':>15}")
    print(f"  {'-'*20} {'-'*15} {'-'*15}")

    if ssvi_result.success:
        ssvi_r2_str = f"{ssvi_result.r_squared:.6f}"
        ssvi_rmse_str = f"{ssvi_result.rmse:.6f}"
        ssvi_time_str = f"{ssvi_time:.3f}s"
    else:
        ssvi_r2_str = ssvi_rmse_str = ssvi_time_str = "FAILED"

    if heston_result.success:
        heston_r2_str = f"{heston_result.r_squared:.6f}"
        heston_rmse_str = f"{heston_result.rmse:.6f}"
        heston_time_str = f"{heston_time:.3f}s"
    else:
        heston_r2_str = heston_rmse_str = heston_time_str = "FAILED"

    print(f"  {'R-squared':<20} {ssvi_r2_str:>15} {heston_r2_str:>15}")
    print(f"  {'RMSE':<20} {ssvi_rmse_str:>15} {heston_rmse_str:>15}")
    print(f"  {'Time':<20} {ssvi_time_str:>15} {heston_time_str:>15}")

    if ssvi_time and heston_time:
        ratio = heston_time / ssvi_time
        print(f"\n  Heston is {ratio:.1f}x {'slower' if ratio > 1 else 'faster'} than SSVI")

    if ssvi_result.success and heston_result.success:
        r2_diff = heston_result.r_squared - ssvi_result.r_squared
        if r2_diff > 0:
            print(f"  Heston R-squared is {r2_diff:.6f} higher (better fit)")
        else:
            print(f"  SSVI R-squared is {-r2_diff:.6f} higher (better fit)")


if __name__ == "__main__":
    main()
