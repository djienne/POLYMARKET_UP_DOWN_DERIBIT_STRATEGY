#!/usr/bin/env python3
"""Compare Heston vs SSVI calibration on closest TTM expiry.

Fetches live Deribit data, calibrates both models, extracts RND,
and prints a side-by-side comparison of fit quality, timing, and probabilities.
Also fetches current Polymarket prices for context.
"""

import time
import logging
import sys
from pathlib import Path

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from btc_pricer.config import Config
from btc_pricer.api.deribit import DeribitClient
from btc_pricer.api.binance import fetch_spot_with_fallback
from btc_pricer.data.filters import DataFilter
from btc_pricer.models.breeden_litzenberger import BreedenLitzenberger
from btc_pricer.models.heston import HestonModel, check_iv_consistency
from btc_pricer.models.ssvi import SSVIModel
from btc_pricer.cli.common import (
    create_heston_fitter,
    create_ssvi_fitter,
    extract_surface_data,
    parse_expiry_date,
)


def fetch_polymarket_data(logger):
    """Fetch current Polymarket Bitcoin Up/Down market data.

    Returns dict with barrier, prob_up, prob_down, title, hours_remaining or None.
    """
    try:
        from scripts.polymarket_btc_daily import (
            search_btc_daily_markets,
            find_closest_active_market,
            market_to_dict,
        )

        data = search_btc_daily_markets()
        event = find_closest_active_market(data)
        if event is None:
            logger.warning("No active Polymarket Bitcoin market found")
            return None

        return market_to_dict(event)
    except Exception as e:
        logger.warning(f"Failed to fetch Polymarket data: {e}")
        return None


def compute_cdf(rnd, level):
    """Compute P(price <= level) from RND."""
    mask = rnd.strikes <= level
    if not mask.any():
        return 0.0
    return np.trapezoid(rnd.density[mask], rnd.strikes[mask])


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(__name__)

    # Load config
    config = Config.from_yaml("config.yaml")

    # Fetch Polymarket data (do this first, it's fast)
    logger.info("Fetching Polymarket data...")
    poly = fetch_polymarket_data(logger)

    # Fetch options data
    logger.info("Fetching options data from Deribit...")
    client = DeribitClient(config.api, config.validation)
    options_by_expiry = client.fetch_all_options("BTC")

    if not options_by_expiry:
        logger.error("No options data received")
        sys.exit(1)

    deribit_spot = list(options_by_expiry.values())[0][0].spot_price
    spot_price, _ = fetch_spot_with_fallback(deribit_spot)

    # Find closest TTM expiry
    sorted_expiries = sorted(
        options_by_expiry.items(),
        key=lambda x: x[1][0].time_to_expiry if x[1] else float('inf')
    )
    expiry, options = sorted_expiries[0]

    # Filter and build surface
    data_filter = DataFilter(config.filters)
    filtered, _ = data_filter.filter_options(options, return_stats=True)
    otm_surface = data_filter.build_otm_surface(filtered)

    surface_data = extract_surface_data(
        otm_surface,
        min_points=3,
        iv_valid_range=config.validation.iv_valid_range
    )

    if surface_data is None:
        logger.error(f"Insufficient data for {expiry}")
        sys.exit(1)

    forward, ttm, _, log_moneyness, market_iv = surface_data

    print("=" * 70)
    print(f"MODEL COMPARISON: {expiry}")
    print(f"  Spot: ${spot_price:,.0f}  |  Forward: ${forward:,.0f}  |  TTM: {ttm:.4f}y ({ttm*365:.1f}d)")
    print(f"  Data points: {len(log_moneyness)}")
    print("=" * 70)

    # ── Polymarket data ─────────────────────────────────────────────────
    if poly:
        print("\n--- POLYMARKET ---")
        print(f"  Market:  {poly.get('market_title', 'N/A')}")
        barrier = poly.get("barrier")
        if barrier:
            print(f"  Barrier: ${barrier:,.2f}")
        hrs = poly.get("hours_remaining")
        if hrs is not None:
            h = poly.get("hours", 0)
            m = poly.get("minutes", 0)
            print(f"  Time:    {h}h {m}m remaining")
        prob_up = poly.get("prob_up")
        prob_down = poly.get("prob_down")
        if prob_up is not None:
            print(f"  UP:      {prob_up*100:.1f}% (${prob_up:.3f})")
        if prob_down is not None:
            print(f"  DOWN:    {prob_down*100:.1f}% (${prob_down:.3f})")

    # ── Heston calibration ──────────────────────────────────────────────
    print("\n--- HESTON CALIBRATION ---")
    heston_fitter = create_heston_fitter(config)

    t0 = time.perf_counter()
    heston_result = heston_fitter.fit(log_moneyness, market_iv, ttm, forward=forward)
    heston_time = time.perf_counter() - t0

    heston_valid = False
    heston_iv_err = None
    if heston_result.success and heston_result.params is not None:
        heston_model = HestonModel(heston_result.params, use_quantlib=config.heston.use_quantlib)
        is_consistent, max_iv_error, _ = check_iv_consistency(
            heston_model, log_moneyness, market_iv,
            config.model.iv_consistency_threshold,
            ttm=ttm,
            relaxation=getattr(config.model, 'iv_consistency_relaxation', 0.15),
            ttm_cutoff=getattr(config.model, 'iv_consistency_ttm_cutoff', 0.05),
        )
        heston_iv_err = max_iv_error
        heston_valid = is_consistent

        p = heston_result.params
        print(f"  Status:  {'VALID' if heston_valid else 'FAILED IV CHECK'}")
        print(f"  Time:    {heston_time:.2f}s")
        print(f"  R²:      {heston_result.r_squared:.6f}")
        print(f"  RMSE:    {heston_result.rmse:.6f}")
        print(f"  Max res: {heston_result.max_residual:.6f}")
        print(f"  IV err:  {max_iv_error:.4%}")
        print(f"  Params:  v0={p.v0:.4f}  kappa={p.kappa:.3f}  theta={p.theta:.4f}  xi={p.xi:.3f}  rho={p.rho:.3f}")
        print(f"  Feller:  {'satisfied' if p.feller_condition() else 'VIOLATED'} (ratio={p.feller_ratio():.3f})")
    else:
        print(f"  Status:  FAILED - {heston_result.message}")
        print(f"  Time:    {heston_time:.2f}s")

    # ── SSVI calibration ────────────────────────────────────────────────
    print("\n--- SSVI CALIBRATION ---")
    ssvi_fitter = create_ssvi_fitter(config)

    t0 = time.perf_counter()
    ssvi_result = ssvi_fitter.fit(log_moneyness, market_iv, ttm)
    ssvi_time = time.perf_counter() - t0

    ssvi_valid = False
    if ssvi_result.success and ssvi_result.params is not None:
        ssvi_valid = True
        p = ssvi_result.params
        print(f"  Status:  VALID")
        print(f"  Time:    {ssvi_time:.2f}s")
        print(f"  R²:      {ssvi_result.r_squared:.6f}")
        print(f"  RMSE:    {ssvi_result.rmse:.6f}")
        print(f"  Max res: {ssvi_result.max_residual:.6f}")
        print(f"  Params:  theta={p.theta:.4f}  rho={p.rho:.3f}  phi={p.phi:.3f}")
        print(f"  Butterfly: {'satisfied' if p.butterfly_condition() else 'VIOLATED'}")
    else:
        print(f"  Status:  FAILED - {ssvi_result.message}")
        print(f"  Time:    {ssvi_time:.2f}s")

    # ── Extract RND from both ───────────────────────────────────────────
    bl = BreedenLitzenberger(
        strike_grid_points=config.breeden_litzenberger.strike_grid_points,
        strike_range_std=config.breeden_litzenberger.strike_range_std,
        use_log_strikes=config.breeden_litzenberger.use_log_strikes
    )

    rnd_heston = None
    rnd_ssvi = None

    if heston_valid:
        rnd_heston = bl.extract_from_heston(heston_result.params, forward)
    if ssvi_valid:
        rnd_ssvi = bl.extract_from_ssvi(ssvi_result.params, forward)

    # ── Side-by-side RND comparison ─────────────────────────────────────
    if rnd_heston or rnd_ssvi:
        print("\n" + "=" * 70)
        print("RND STATISTICS COMPARISON")
        print("=" * 70)

        header = f"{'Metric':<25} {'Heston':>20} {'SSVI':>20}"
        print(header)
        print("-" * 65)

        def row(label, h_val, s_val, fmt="${:,.0f}"):
            h_str = fmt.format(h_val) if h_val is not None else "N/A"
            s_str = fmt.format(s_val) if s_val is not None else "N/A"
            print(f"{label:<25} {h_str:>20} {s_str:>20}")

        h = rnd_heston
        s = rnd_ssvi

        row("Mean",
            h.mean if h else None,
            s.mean if s else None)
        row("Mode",
            h.mode if h else None,
            s.mode if s else None)
        row("Median (P50)",
            h.percentile_50 if h else None,
            s.percentile_50 if s else None)
        row("Std Dev",
            h.std_dev if h else None,
            s.std_dev if s else None)
        row("Skewness",
            h.skewness if h else None,
            s.skewness if s else None, fmt="{:.4f}")
        row("Kurtosis",
            h.kurtosis if h else None,
            s.kurtosis if s else None, fmt="{:.4f}")
        row("5th percentile",
            h.percentile_5 if h else None,
            s.percentile_5 if s else None)
        row("25th percentile",
            h.percentile_25 if h else None,
            s.percentile_25 if s else None)
        row("75th percentile",
            h.percentile_75 if h else None,
            s.percentile_75 if s else None)
        row("95th percentile",
            h.percentile_95 if h else None,
            s.percentile_95 if s else None)
        row("Density integral",
            h.integral if h else None,
            s.integral if s else None, fmt="{:.6f}")
        row("Valid",
            h.is_valid if h else None,
            s.is_valid if s else None, fmt="{}")

    # ── Polymarket barrier probability comparison ───────────────────────
    barrier = poly.get("barrier") if poly else None
    if barrier and (rnd_heston or rnd_ssvi):
        print("\n" + "=" * 70)
        print(f"POLYMARKET BARRIER COMPARISON  (barrier = ${barrier:,.2f})")
        print("=" * 70)

        prob_up_poly = poly.get("prob_up")
        prob_down_poly = poly.get("prob_down")

        # Model P(above barrier) = 1 - CDF(barrier)
        p_above_h = 1.0 - compute_cdf(rnd_heston, barrier) if rnd_heston else None
        p_above_s = 1.0 - compute_cdf(rnd_ssvi, barrier) if rnd_ssvi else None
        p_below_h = compute_cdf(rnd_heston, barrier) if rnd_heston else None
        p_below_s = compute_cdf(rnd_ssvi, barrier) if rnd_ssvi else None

        print(f"\n{'':30} {'Polymarket':>12} {'Heston':>12} {'SSVI':>12}")
        print("-" * 68)

        # UP row
        poly_up_str = f"{prob_up_poly*100:.1f}%" if prob_up_poly is not None else "N/A"
        h_up_str = f"{p_above_h*100:.1f}%" if p_above_h is not None else "N/A"
        s_up_str = f"{p_above_s*100:.1f}%" if p_above_s is not None else "N/A"
        print(f"{'P(above barrier) = UP':<30} {poly_up_str:>12} {h_up_str:>12} {s_up_str:>12}")

        # DOWN row
        poly_dn_str = f"{prob_down_poly*100:.1f}%" if prob_down_poly is not None else "N/A"
        h_dn_str = f"{p_below_h*100:.1f}%" if p_below_h is not None else "N/A"
        s_dn_str = f"{p_below_s*100:.1f}%" if p_below_s is not None else "N/A"
        print(f"{'P(below barrier) = DOWN':<30} {poly_dn_str:>12} {h_dn_str:>12} {s_dn_str:>12}")

        # Edge (model - polymarket)
        print()
        print("Edge (model - Polymarket):")
        if prob_up_poly is not None:
            if p_above_h is not None:
                edge_h_up = (p_above_h - prob_up_poly) * 100
                print(f"  Heston UP edge:   {edge_h_up:>+.1f}pp")
            if p_above_s is not None:
                edge_s_up = (p_above_s - prob_up_poly) * 100
                print(f"  SSVI   UP edge:   {edge_s_up:>+.1f}pp")
        if prob_down_poly is not None:
            if p_below_h is not None:
                edge_h_dn = (p_below_h - prob_down_poly) * 100
                print(f"  Heston DOWN edge: {edge_h_dn:>+.1f}pp")
            if p_below_s is not None:
                edge_s_dn = (p_below_s - prob_down_poly) * 100
                print(f"  SSVI   DOWN edge: {edge_s_dn:>+.1f}pp")

        # Spot vs barrier context
        diff = spot_price - barrier
        pct = (diff / barrier) * 100
        direction = "above" if diff >= 0 else "below"
        print(f"\n  Spot is {direction} barrier by ${abs(diff):,.2f} ({pct:+.2f}%)")

    # ── Probability comparison at key levels ────────────────────────────
    if rnd_heston and rnd_ssvi:
        print("\n" + "=" * 70)
        print("PROBABILITY COMPARISON AT KEY PRICE LEVELS")
        print("=" * 70)

        # Generate test levels around spot, include barrier if available
        levels = [
            spot_price * 0.90,
            spot_price * 0.95,
            spot_price * 0.98,
        ]
        if barrier and barrier < spot_price * 0.98:
            levels.append(barrier)
        levels.append(spot_price)
        if barrier and spot_price * 0.98 <= barrier <= spot_price * 1.02:
            levels.append(barrier)
        levels.extend([
            spot_price * 1.02,
        ])
        if barrier and barrier > spot_price * 1.02:
            levels.append(barrier)
        levels.extend([
            spot_price * 1.05,
            spot_price * 1.10,
        ])
        # Deduplicate and sort
        levels = sorted(set(levels))

        print(f"{'Price Level':>12}  {'P(below) Heston':>16}  {'P(below) SSVI':>16}  {'Diff (pp)':>10}  {'Note':>10}")
        print("-" * 70)

        for level in levels:
            p_h = compute_cdf(rnd_heston, level)
            p_s = compute_cdf(rnd_ssvi, level)

            diff_pp = (p_h - p_s) * 100
            note = ""
            if barrier and abs(level - barrier) < 1:
                note = "BARRIER"
            elif abs(level - spot_price) < 1:
                note = "SPOT"

            print(f"${level:>10,.0f}  {p_h:>15.2%}  {p_s:>15.2%}  {diff_pp:>+9.2f}pp  {note:>10}")

    # ── Fit quality summary ─────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("FIT QUALITY SUMMARY")
    print("=" * 70)

    print(f"{'':25} {'Heston':>20} {'SSVI':>20}")
    print("-" * 65)

    h_r2 = f"{heston_result.r_squared:.6f}" if heston_result.success else "FAILED"
    s_r2 = f"{ssvi_result.r_squared:.6f}" if ssvi_result.success else "FAILED"
    print(f"{'R²':<25} {h_r2:>20} {s_r2:>20}")

    h_rmse = f"{heston_result.rmse:.6f}" if heston_result.success else "N/A"
    s_rmse = f"{ssvi_result.rmse:.6f}" if ssvi_result.success else "N/A"
    print(f"{'RMSE':<25} {h_rmse:>20} {s_rmse:>20}")

    h_max = f"{heston_result.max_residual:.6f}" if heston_result.success else "N/A"
    s_max = f"{ssvi_result.max_residual:.6f}" if ssvi_result.success else "N/A"
    print(f"{'Max residual':<25} {h_max:>20} {s_max:>20}")

    print(f"{'Calibration time':<25} {heston_time:>19.2f}s {ssvi_time:>19.2f}s")
    print(f"{'Speed ratio':<25} {heston_time/ssvi_time if ssvi_time > 0 else float('inf'):>19.1f}x {'1.0x':>20}")

    # ── IV residuals per strike ─────────────────────────────────────────
    if heston_valid and ssvi_valid:
        print("\n" + "=" * 70)
        print("IV RESIDUALS PER STRIKE (model IV - market IV)")
        print("=" * 70)

        heston_model = HestonModel(heston_result.params, use_quantlib=config.heston.use_quantlib)
        ssvi_model = SSVIModel(ssvi_result.params)

        print(f"{'log(K/F)':>10}  {'Market IV':>10}  {'Heston IV':>10}  {'SSVI IV':>10}  {'H err':>8}  {'S err':>8}")
        print("-" * 65)

        for i in range(len(log_moneyness)):
            lm = log_moneyness[i]
            miv = market_iv[i]

            # Heston model IV
            strike = forward * np.exp(lm)
            try:
                h_iv = heston_model.implied_volatility(strike, forward, ttm)
            except Exception:
                h_iv = float('nan')

            # SSVI model IV
            s_iv = ssvi_model.implied_volatility(lm)

            h_err = h_iv - miv
            s_err = s_iv - miv

            print(f"{lm:>10.4f}  {miv:>10.4f}  {h_iv:>10.4f}  {s_iv:>10.4f}  {h_err:>+8.4f}  {s_err:>+8.4f}")

    print()


if __name__ == "__main__":
    main()
