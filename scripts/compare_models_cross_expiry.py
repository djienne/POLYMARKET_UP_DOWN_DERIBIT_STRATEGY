#!/usr/bin/env python3
"""Compare models: calibrate on NEXT expiry, evaluate at TODAY's TTM.

The idea: the closest expiry often has few data points and noisy IVs.
Calibrating on the next expiry (more liquid) and projecting to today's
shorter TTM may give better probability estimates.

For Heston: v0/kappa/theta/xi/rho describe the SDE, so they're TTM-agnostic.
We just swap the TTM field to the shorter horizon.

For SSVI: theta is ATM total variance (sigma^2 * T), so we rescale:
  theta_today = theta_next * (ttm_today / ttm_next)
"""

import time
import logging
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from btc_pricer.config import Config
from btc_pricer.api.deribit import DeribitClient
from btc_pricer.api.binance import fetch_spot_with_fallback
from btc_pricer.data.filters import DataFilter
from btc_pricer.models.breeden_litzenberger import BreedenLitzenberger
from btc_pricer.models.heston import HestonModel, HestonParams, check_iv_consistency
from btc_pricer.models.ssvi import SSVIModel, SSVIParams
from btc_pricer.cli.common import (
    create_heston_fitter,
    create_ssvi_fitter,
    extract_surface_data,
    parse_expiry_date,
)


def fetch_polymarket_data(logger):
    """Fetch current Polymarket Bitcoin Up/Down market data."""
    try:
        from scripts.polymarket_btc_daily import (
            search_btc_daily_markets,
            find_closest_active_market,
            market_to_dict,
        )
        data = search_btc_daily_markets()
        event = find_closest_active_market(data)
        if event is None:
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

    config = Config.from_yaml("config.yaml")

    # Fetch Polymarket
    logger.info("Fetching Polymarket data...")
    poly = fetch_polymarket_data(logger)

    # Fetch Deribit options
    logger.info("Fetching options data from Deribit...")
    client = DeribitClient(config.api, config.validation)
    options_by_expiry = client.fetch_all_options("BTC")

    if not options_by_expiry:
        logger.error("No options data received")
        sys.exit(1)

    deribit_spot = list(options_by_expiry.values())[0][0].spot_price
    spot_price, _ = fetch_spot_with_fallback(deribit_spot)

    # Sort expiries by TTM
    sorted_expiries = sorted(
        options_by_expiry.items(),
        key=lambda x: x[1][0].time_to_expiry if x[1] else float('inf')
    )

    if len(sorted_expiries) < 2:
        logger.error("Need at least 2 expiries")
        sys.exit(1)

    # Closest expiry (today's) — used for TTM target
    exp_today, opts_today = sorted_expiries[0]
    # Next expiry — used for calibration
    exp_next, opts_next = sorted_expiries[1]

    # Build surfaces for both
    data_filter = DataFilter(config.filters)

    filtered_today, _ = data_filter.filter_options(opts_today, return_stats=True)
    otm_today = data_filter.build_otm_surface(filtered_today)
    surf_today = extract_surface_data(otm_today, min_points=3, iv_valid_range=config.validation.iv_valid_range)

    filtered_next, _ = data_filter.filter_options(opts_next, return_stats=True)
    otm_next = data_filter.build_otm_surface(filtered_next)
    surf_next = extract_surface_data(otm_next, min_points=3, iv_valid_range=config.validation.iv_valid_range)

    if surf_today is None or surf_next is None:
        logger.error("Insufficient data for one of the expiries")
        sys.exit(1)

    fwd_today, ttm_today, _, lm_today, iv_today = surf_today
    fwd_next, ttm_next, _, lm_next, iv_next = surf_next

    barrier = poly.get("barrier") if poly else None

    print("=" * 75)
    print("CROSS-EXPIRY CALIBRATION: fit on NEXT expiry, evaluate at TODAY's TTM")
    print("=" * 75)
    print(f"  Spot:            ${spot_price:,.0f}")
    if barrier:
        print(f"  Barrier:         ${barrier:,.2f}")
    print(f"  Today expiry:    {exp_today}  TTM={ttm_today:.4f}y ({ttm_today*365:.1f}d)  {len(lm_today)} pts")
    print(f"  Next expiry:     {exp_next}  TTM={ttm_next:.4f}y ({ttm_next*365:.1f}d)  {len(lm_next)} pts")

    # ── Polymarket ──────────────────────────────────────────────────────
    if poly:
        print(f"\n--- POLYMARKET: {poly.get('market_title', 'N/A')} ---")
        h, m = poly.get("hours", 0), poly.get("minutes", 0)
        print(f"  Time remaining: {h}h {m}m")
        prob_up = poly.get("prob_up")
        prob_down = poly.get("prob_down")
        if prob_up is not None:
            print(f"  UP:   {prob_up*100:.1f}%  |  DOWN: {prob_down*100:.1f}%")

    # ═══════════════════════════════════════════════════════════════════
    # A) DIRECT calibration on today's expiry (baseline)
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 75)
    print("A) DIRECT: calibrate on TODAY's expiry")
    print("=" * 75)

    bl = BreedenLitzenberger(
        strike_grid_points=config.breeden_litzenberger.strike_grid_points,
        strike_range_std=config.breeden_litzenberger.strike_range_std,
        use_log_strikes=config.breeden_litzenberger.use_log_strikes,
    )

    # Heston on today
    heston_fitter = create_heston_fitter(config)
    t0 = time.perf_counter()
    h_res_today = heston_fitter.fit(lm_today, iv_today, ttm_today, forward=fwd_today)
    h_time_today = time.perf_counter() - t0

    h_valid_today = False
    if h_res_today.success and h_res_today.params is not None:
        hm = HestonModel(h_res_today.params, use_quantlib=config.heston.use_quantlib)
        ok, iv_err, _ = check_iv_consistency(
            hm, lm_today, iv_today, config.model.iv_consistency_threshold,
            ttm=ttm_today,
            relaxation=getattr(config.model, 'iv_consistency_relaxation', 0.15),
            ttm_cutoff=getattr(config.model, 'iv_consistency_ttm_cutoff', 0.05),
        )
        h_valid_today = ok

    # SSVI on today
    ssvi_fitter = create_ssvi_fitter(config)
    t0 = time.perf_counter()
    s_res_today = ssvi_fitter.fit(lm_today, iv_today, ttm_today)
    s_time_today = time.perf_counter() - t0
    s_valid_today = s_res_today.success and s_res_today.params is not None

    print(f"  Heston: R²={h_res_today.r_squared:.4f}  RMSE={h_res_today.rmse:.4f}  {'VALID' if h_valid_today else 'INVALID'}  ({h_time_today:.1f}s)")
    print(f"  SSVI:   R²={s_res_today.r_squared:.4f}  RMSE={s_res_today.rmse:.4f}  {'VALID' if s_valid_today else 'INVALID'}  ({s_time_today:.1f}s)")

    rnd_h_direct = bl.extract_from_heston(h_res_today.params, fwd_today) if h_valid_today else None
    rnd_s_direct = bl.extract_from_ssvi(s_res_today.params, fwd_today) if s_valid_today else None

    # ═══════════════════════════════════════════════════════════════════
    # B) CROSS-EXPIRY: calibrate on NEXT expiry, project to today's TTM
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 75)
    print("B) CROSS-EXPIRY: calibrate on NEXT expiry, project to TODAY's TTM")
    print("=" * 75)

    # Heston on next expiry
    t0 = time.perf_counter()
    h_res_next = heston_fitter.fit(lm_next, iv_next, ttm_next, forward=fwd_next)
    h_time_next = time.perf_counter() - t0

    h_valid_next = False
    if h_res_next.success and h_res_next.params is not None:
        hm = HestonModel(h_res_next.params, use_quantlib=config.heston.use_quantlib)
        ok, iv_err, _ = check_iv_consistency(
            hm, lm_next, iv_next, config.model.iv_consistency_threshold,
            ttm=ttm_next,
            relaxation=getattr(config.model, 'iv_consistency_relaxation', 0.15),
            ttm_cutoff=getattr(config.model, 'iv_consistency_ttm_cutoff', 0.05),
        )
        h_valid_next = ok

    # SSVI on next expiry
    t0 = time.perf_counter()
    s_res_next = ssvi_fitter.fit(lm_next, iv_next, ttm_next)
    s_time_next = time.perf_counter() - t0
    s_valid_next = s_res_next.success and s_res_next.params is not None

    print(f"  Calibration on {exp_next}:")
    print(f"  Heston: R²={h_res_next.r_squared:.4f}  RMSE={h_res_next.rmse:.4f}  {'VALID' if h_valid_next else 'INVALID'}  ({h_time_next:.1f}s)")
    print(f"  SSVI:   R²={s_res_next.r_squared:.4f}  RMSE={s_res_next.rmse:.4f}  {'VALID' if s_valid_next else 'INVALID'}  ({s_time_next:.1f}s)")

    # Project Heston to today's TTM (swap TTM, keep process params)
    rnd_h_cross = None
    if h_valid_next:
        hp = h_res_next.params
        h_params_projected = HestonParams(
            v0=hp.v0, kappa=hp.kappa, theta=hp.theta,
            xi=hp.xi, rho=hp.rho, ttm=ttm_today
        )
        print(f"\n  Heston projected: v0={hp.v0:.4f} kappa={hp.kappa:.3f} theta={hp.theta:.4f} xi={hp.xi:.3f} rho={hp.rho:.3f} TTM={ttm_today:.4f}")
        rnd_h_cross = bl.extract_from_heston(h_params_projected, fwd_today)

    # Project SSVI to today's TTM (rescale theta = ATM total variance)
    rnd_s_cross = None
    if s_valid_next:
        sp = s_res_next.params
        theta_projected = sp.theta * (ttm_today / ttm_next)
        s_params_projected = SSVIParams(
            theta=theta_projected, rho=sp.rho, phi=sp.phi, ttm=ttm_today
        )
        print(f"  SSVI projected:   theta={theta_projected:.6f} (from {sp.theta:.6f}, scaled by {ttm_today/ttm_next:.4f}) rho={sp.rho:.3f} phi={sp.phi:.3f}")
        rnd_s_cross = bl.extract_from_ssvi(s_params_projected, fwd_today)

    # ═══════════════════════════════════════════════════════════════════
    # COMPARISON TABLE
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 75)
    print("PROBABILITY COMPARISON AT BARRIER" + (f"  (${barrier:,.2f})" if barrier else ""))
    print("=" * 75)

    results = {}

    def add_result(label, rnd):
        if rnd is None:
            results[label] = (None, None)
            return
        p_below = compute_cdf(rnd, barrier) if barrier else None
        p_above = 1.0 - p_below if p_below is not None else None
        results[label] = (p_above, p_below)

    add_result("Heston (direct)", rnd_h_direct)
    add_result("SSVI (direct)", rnd_s_direct)
    add_result("Heston (cross)", rnd_h_cross)
    add_result("SSVI (cross)", rnd_s_cross)

    prob_up_poly = poly.get("prob_up") if poly else None
    prob_down_poly = poly.get("prob_down") if poly else None

    print(f"\n{'Method':<25} {'P(UP)':>10} {'P(DOWN)':>10} {'UP edge':>10} {'DOWN edge':>10}")
    print("-" * 67)

    if prob_up_poly is not None:
        print(f"{'Polymarket':<25} {prob_up_poly*100:>9.1f}% {prob_down_poly*100:>9.1f}%")
        print("-" * 67)

    for label, (p_up, p_dn) in results.items():
        if p_up is None:
            print(f"{label:<25} {'N/A':>10} {'N/A':>10}")
            continue
        up_edge = (p_up - prob_up_poly) * 100 if prob_up_poly is not None else None
        dn_edge = (p_dn - prob_down_poly) * 100 if prob_down_poly is not None else None
        up_e = f"{up_edge:>+9.1f}pp" if up_edge is not None else ""
        dn_e = f"{dn_edge:>+9.1f}pp" if dn_edge is not None else ""
        print(f"{label:<25} {p_up*100:>9.1f}% {p_dn*100:>9.1f}% {up_e:>10} {dn_e:>10}")

    # ═══════════════════════════════════════════════════════════════════
    # RND STATISTICS
    # ═══════════════════════════════════════════════════════════════════
    all_rnds = [
        ("Heston direct", rnd_h_direct),
        ("SSVI direct", rnd_s_direct),
        ("Heston cross", rnd_h_cross),
        ("SSVI cross", rnd_s_cross),
    ]
    valid_rnds = [(n, r) for n, r in all_rnds if r is not None]

    if valid_rnds:
        print("\n" + "=" * 75)
        print("RND STATISTICS")
        print("=" * 75)

        header = f"{'Metric':<16}" + "".join(f"{n:>16}" for n, _ in valid_rnds)
        print(header)
        print("-" * (16 + 16 * len(valid_rnds)))

        def stat_row(label, attr, fmt="${:,.0f}"):
            vals = []
            for _, r in valid_rnds:
                v = getattr(r, attr, None)
                vals.append(fmt.format(v) if v is not None else "N/A")
            print(f"{label:<16}" + "".join(f"{v:>16}" for v in vals))

        stat_row("Mean", "mean")
        stat_row("Mode", "mode")
        stat_row("Median", "percentile_50")
        stat_row("Std Dev", "std_dev")
        stat_row("Skewness", "skewness", fmt="{:.4f}")
        stat_row("Kurtosis", "kurtosis", fmt="{:.4f}")
        stat_row("5th pct", "percentile_5")
        stat_row("95th pct", "percentile_95")

    # ═══════════════════════════════════════════════════════════════════
    # FULL PROBABILITY TABLE
    # ═══════════════════════════════════════════════════════════════════
    if len(valid_rnds) >= 2:
        print("\n" + "=" * 75)
        print("TERMINAL PROBABILITIES AT KEY LEVELS")
        print("=" * 75)

        levels = sorted(set([
            spot_price * 0.95,
            spot_price * 0.98,
        ] + ([barrier] if barrier else []) + [
            spot_price,
            spot_price * 1.02,
            spot_price * 1.05,
        ]))

        header = f"{'Level':>12}" + "".join(f"{n:>16}" for n, _ in valid_rnds) + f"{'Note':>10}"
        print(header)
        print("-" * (12 + 16 * len(valid_rnds) + 10))

        for level in levels:
            vals = []
            for _, r in valid_rnds:
                p = compute_cdf(r, level)
                vals.append(f"{p*100:.1f}%")
            note = ""
            if barrier and abs(level - barrier) < 1:
                note = "BARRIER"
            elif abs(level - spot_price) < 1:
                note = "SPOT"
            print(f"${level:>10,.0f}" + "".join(f"{v:>16}" for v in vals) + f"{note:>10}")

    print()


if __name__ == "__main__":
    main()
