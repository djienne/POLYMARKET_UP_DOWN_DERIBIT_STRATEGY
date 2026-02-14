#!/usr/bin/env python3
"""Standalone data collector for Polymarket BTC daily markets.

Collects orderbook snapshots and model-vs-market probability edges
without any trading logic. Outputs to CSV files compatible with the
backtesting and analysis pipeline.

Usage:
    python -m data_collector.collector          # Run continuously
    python -m data_collector.collector --once   # Single iteration
"""

import csv
import json
import os
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from scripts.polymarket_btc_daily import (
    search_btc_daily_markets,
    find_closest_active_market,
    parse_reference_time,
    get_binance_price,
)
from scripts.polymarket_edge import (
    run_polymarket_script,
    run_barrier_script,
    find_opportunities,
)
from btc_pricer.api.deribit import DeribitClient

from .order_book import OrderBookClient, log_orderbook_to_csv


COLLECTOR_DIR = Path(__file__).parent

DERIBIT_SNAPSHOT_COLUMNS = [
    "snapshot_timestamp",
    "expiry_date",
    "instrument_name",
    "strike",
    "option_type",
    "bid_price_btc",
    "ask_price_btc",
    "mark_price_btc",
    "mark_iv",
    "bid_iv",
    "ask_iv",
    "open_interest",
    "underlying_price",
    "spot_price",
    "time_to_expiry_years",
]


def load_config() -> dict:
    """Load collector config from data_collector/config.json."""
    config_path = COLLECTOR_DIR / "config.json"
    with open(config_path) as f:
        return json.load(f)


def log(message: str) -> None:
    """Print a timestamped log message."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}")


def fetch_current_market() -> Optional[dict]:
    """Fetch the current active Polymarket BTC daily market.

    Returns dict with event_id, title, end_date, barrier_price,
    clob_token_ids, description — or None on failure.
    """
    try:
        data = search_btc_daily_markets()
        event = find_closest_active_market(data)

        if not event:
            log("No active market found")
            return None

        title = event.get("title", "")
        end_date_str = event.get("endDate")
        description = event.get("description", "")

        end_date = None
        if end_date_str:
            end_date = datetime.fromisoformat(end_date_str.replace("Z", "+00:00"))

        ref_time = parse_reference_time(description)
        barrier_price = get_binance_price(ref_time) if ref_time else None

        markets = event.get("markets", [])
        clob_token_ids = {}

        if markets:
            market = markets[0]
            outcomes = market.get("outcomes", [])
            clob_tokens = market.get("clobTokenIds", [])

            if isinstance(outcomes, str):
                outcomes = json.loads(outcomes)
            if isinstance(clob_tokens, str):
                clob_tokens = json.loads(clob_tokens)

            for i, outcome in enumerate(outcomes):
                if i < len(clob_tokens):
                    clob_token_ids[outcome.upper()] = clob_tokens[i]

        return {
            "event_id": event.get("id"),
            "title": title,
            "end_date": end_date.isoformat() if end_date else None,
            "barrier_price": barrier_price,
            "clob_token_ids": clob_token_ids,
            "description": description,
        }

    except Exception as e:
        log(f"[ERROR] Error fetching market: {e}")
        log(f"[ERROR] Traceback: {traceback.format_exc()}")
        return None


def get_time_remaining_hours(market: dict) -> float:
    """Get hours remaining until market expiry."""
    end_date_str = market.get("end_date")
    if not end_date_str:
        return 0.0
    try:
        end_date = datetime.fromisoformat(end_date_str)
        now = datetime.now(timezone.utc)
        remaining = (end_date - now).total_seconds() / 3600
        return max(0.0, remaining)
    except ValueError:
        return 0.0


def snapshot_orderbook(
    market: dict,
    order_book_client: OrderBookClient,
    csv_path: Path,
) -> None:
    """Fetch and log orderbook snapshots for all tokens in a market."""
    clob_tokens = market.get("clob_token_ids", {})

    for direction, token_id in clob_tokens.items():
        if not token_id or len(token_id) < 20:
            continue

        order_book = order_book_client.fetch_order_book(token_id)
        if not order_book:
            continue

        best_bid = order_book.best_bid
        best_ask = order_book.best_ask
        spread = best_ask - best_bid if (best_bid > 0 and best_ask > 0) else 0.0
        sanity_ok = best_bid < best_ask if (best_bid > 0 and best_ask > 0) else True

        log_orderbook_to_csv(
            token_id=f"{direction}:{token_id[:8]}",
            best_bid=best_bid,
            best_ask=best_ask,
            spread=spread,
            bid_depth=len(order_book.bids),
            ask_depth=len(order_book.asks),
            csv_path=csv_path,
            sanity_ok=sanity_ok,
            barrier_price=market.get("barrier_price"),
        )

    log(f"Orderbook snapshot logged ({len(clob_tokens)} tokens)")


def log_probabilities_to_csv(
    poly_data: dict,
    model_data: dict,
    csv_path: Path,
) -> None:
    """Append probability data to CSV file."""
    poly_up = poly_data.get("prob_up", 0)
    poly_down = poly_data.get("prob_down", 0)
    model_up = model_data.get("prob_above", 0)
    model_down = model_data.get("prob_below", 0)

    edge_up = (model_up / poly_up) if poly_up > 0 else 0
    edge_down = (model_down / poly_down) if poly_down > 0 else 0

    # Per-model probabilities
    preferred_model = model_data.get("model_used", "")
    bl_mc_div = model_data.get("bl_mc_divergence")

    # SSVI Surface probabilities
    surface_mc_above = model_data.get("surface_mc_above")
    surface_mc_below = model_data.get("surface_mc_below")
    surface_bl_above = model_data.get("surface_bl_above")
    surface_bl_below = model_data.get("surface_bl_below")

    # Heston B-L probabilities
    heston_bl_above = model_data.get("heston_bl_above") or model_data.get("bl_prob_above")
    heston_bl_below = model_data.get("heston_bl_below") or model_data.get("bl_prob_below")

    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "barrier_price": poly_data.get("barrier"),
        "time_remaining_hours": poly_data.get("hours_remaining"),
        "spot_price": model_data.get("spot_price"),
        "model_prob_up": model_up * 100,
        "model_prob_down": model_down * 100,
        "poly_prob_up": poly_up * 100,
        "poly_prob_down": poly_down * 100,
        "edge_up": edge_up,
        "edge_down": edge_down,
        "surface_mc_prob_up": surface_mc_above * 100 if surface_mc_above is not None else "",
        "surface_mc_prob_down": surface_mc_below * 100 if surface_mc_below is not None else "",
        "surface_bl_prob_up": surface_bl_above * 100 if surface_bl_above is not None else "",
        "surface_bl_prob_down": surface_bl_below * 100 if surface_bl_below is not None else "",
        "heston_bl_prob_up": heston_bl_above * 100 if heston_bl_above is not None else "",
        "heston_bl_prob_down": heston_bl_below * 100 if heston_bl_below is not None else "",
        "heston_r_squared": model_data.get("heston_r_squared", ""),
        "ssvi_r_squared": model_data.get("surface_r_squared") or model_data.get("ssvi_r_squared", ""),
        "preferred_model": preferred_model,
        "bl_mc_divergence": bl_mc_div * 100 if bl_mc_div is not None else "",
    }

    file_exists = csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def run_edge_check(probabilities_path: Path, verbose: bool = False) -> None:
    """Run a full edge check: Polymarket data + model calibration + CSV log."""
    check_start = datetime.now(timezone.utc)

    # Use separate JSON temp files so we don't overwrite the trading bot's files
    poly_json = COLLECTOR_DIR / "results" / "polymarket_data.json"
    barrier_json = COLLECTOR_DIR / "results" / "barrier_data.json"
    poly_json.parent.mkdir(parents=True, exist_ok=True)

    try:
        log("Step 1/4: Fetching Polymarket data...")
        poly_data = run_polymarket_script(verbose=verbose, json_path=poly_json)

        if poly_data.get("error"):
            log(f"[ERROR] Polymarket script error: {poly_data['error'][:200]}")

        if poly_data["barrier"] is None:
            elapsed = int((datetime.now(timezone.utc) - check_start).total_seconds())
            log(f"Edge check failed ({elapsed}s): Could not get Polymarket data")
            return

        log(f"Step 1/4: OK - Barrier=${poly_data['barrier']:,.0f}, "
            f"UP={poly_data.get('prob_up', 'N/A')}, DOWN={poly_data.get('prob_down', 'N/A')}, "
            f"Time={poly_data.get('hours', 0)}h {poly_data.get('minutes', 0)}m")

        log(f"Step 2/4: Running model calibration for ${poly_data['barrier']:,.0f} barrier, "
            f"{poly_data['hours_remaining']:.2f}h TTM...")
        model_data = run_barrier_script(
            poly_data["barrier"],
            poly_data["hours_remaining"],
            verbose=verbose,
            json_path=barrier_json,
        )

        if model_data.get("error"):
            log(f"[ERROR] Barrier script error: {model_data['error'][:200]}")

        if model_data["prob_above"] is None:
            elapsed = int((datetime.now(timezone.utc) - check_start).total_seconds())
            log(f"Edge check failed ({elapsed}s): Could not get model probabilities")
            return

        log(f"Step 2/4: OK - Model P(above)={model_data['prob_above']:.4f}, "
            f"P(below)={model_data['prob_below']:.4f}")

        # Re-fetch Polymarket prices after calibration (~45s) for freshest comparison
        log("Step 3/4: Re-fetching fresh Polymarket prices...")
        fresh_poly_data = run_polymarket_script(verbose=verbose, json_path=poly_json)
        if fresh_poly_data["prob_up"] is not None:
            poly_data["prob_up"] = fresh_poly_data["prob_up"]
            poly_data["prob_down"] = fresh_poly_data["prob_down"]
            log(f"Step 3/4: OK - Fresh UP={poly_data['prob_up']:.4f}, "
                f"DOWN={poly_data['prob_down']:.4f}")
        else:
            log("Step 3/4: WARN - Could not refresh prices, using original")

        log("Step 4/4: Calculating edges and logging to CSV...")
        opportunities = find_opportunities(poly_data, model_data)

        # Log to CSV
        log_probabilities_to_csv(poly_data, model_data, probabilities_path)

        # Print summary
        elapsed = int((datetime.now(timezone.utc) - check_start).total_seconds())
        elapsed_mins = elapsed // 60
        elapsed_secs = elapsed % 60

        model_up = model_data.get("prob_above", 0) * 100
        model_down = model_data.get("prob_below", 0) * 100
        poly_up = poly_data.get("prob_up", 0) * 100
        poly_down = poly_data.get("prob_down", 0) * 100
        up_edge = (model_up / poly_up) if poly_up > 0 else 0
        down_edge = (model_down / poly_down) if poly_down > 0 else 0

        model_used = model_data.get("model_used", "unknown").upper()
        r2 = model_data.get("r_squared")
        r2_str = f"{r2:.4f}" if r2 is not None else "N/A"
        heston_r2 = model_data.get("heston_r_squared")
        ssvi_r2 = model_data.get("surface_r_squared") or model_data.get("ssvi_r_squared")

        log(f"Edge check complete ({elapsed_mins}m {elapsed_secs}s)")
        log(f"  Fit: {model_used} R²={r2_str}")
        if heston_r2 is not None and ssvi_r2 is not None:
            log(f"  R²:  Heston={heston_r2:.4f} SSVI={ssvi_r2:.4f}")

        # Per-model probability breakdown
        surf_mc = model_data.get("surface_mc_above")
        surf_bl = model_data.get("surface_bl_above")
        hest_bl = model_data.get("heston_bl_above") or model_data.get("bl_prob_above")
        surf_mc_dn = model_data.get("surface_mc_below")
        surf_bl_dn = model_data.get("surface_bl_below")
        hest_bl_dn = model_data.get("heston_bl_below") or model_data.get("bl_prob_below")

        log(f"  --- Probabilities (UP / DOWN) ---")
        if surf_mc is not None:
            log(f"  SSVI Surface MC:   {surf_mc * 100:5.1f}% / {surf_mc_dn * 100:.1f}%")
        if surf_bl is not None:
            log(f"  SSVI Surface B-L:  {surf_bl * 100:5.1f}% / {surf_bl_dn * 100:.1f}%")
        log(f"  Used ({model_used}):  {model_up:5.1f}% / {model_down:.1f}%  << EDGE")
        if hest_bl is not None and model_used != "HESTON":
            log(f"  Heston B-L:        {hest_bl * 100:5.1f}% / {hest_bl_dn * 100:.1f}%  (reference)")
        elif hest_bl is not None:
            log(f"  Heston B-L:        {hest_bl * 100:5.1f}% / {hest_bl_dn * 100:.1f}%")
        log(f"  Polymarket:        {poly_up:5.1f}% / {poly_down:.1f}%")
        log(f"  Edge: UP {up_edge:.2f}x | DOWN {down_edge:.2f}x")

        edge_opportunities = [o for o in opportunities if o.get("has_edge")]
        if edge_opportunities:
            best = max(edge_opportunities, key=lambda x: x["edge"])
            log(f"  Best edge: {best['direction']} at {best['edge']:.2f}x")
        else:
            log(f"  No significant edge found")

    except Exception as e:
        elapsed = int((datetime.now(timezone.utc) - check_start).total_seconds())
        log(f"[ERROR] Edge check failed ({elapsed}s): {e}")
        log(f"[ERROR] Traceback: {traceback.format_exc()}")


def snapshot_deribit_options(csv_path: Path) -> None:
    """Fetch all Deribit options and append a snapshot to CSV."""
    try:
        client = DeribitClient()
        options_by_expiry = client.fetch_all_options("BTC")

        now_str = datetime.now(timezone.utc).isoformat()
        rows = []

        for expiry_str, options in options_by_expiry.items():
            for opt in options:
                rows.append({
                    "snapshot_timestamp": now_str,
                    "expiry_date": opt.expiration_date,
                    "instrument_name": opt.instrument_name,
                    "strike": opt.strike,
                    "option_type": opt.option_type,
                    "bid_price_btc": opt.bid_price,
                    "ask_price_btc": opt.ask_price,
                    "mark_price_btc": opt.mark_price,
                    "mark_iv": opt.mark_iv,
                    "bid_iv": opt.bid_iv,
                    "ask_iv": opt.ask_iv,
                    "open_interest": opt.open_interest,
                    "underlying_price": opt.underlying_price,
                    "spot_price": opt.spot_price,
                    "time_to_expiry_years": opt.time_to_expiry,
                })

        if not rows:
            log("Deribit snapshot: no options returned")
            return

        file_exists = csv_path.exists()
        with open(csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=DERIBIT_SNAPSHOT_COLUMNS)
            if not file_exists:
                writer.writeheader()
            writer.writerows(rows)

        log(f"Deribit snapshot: {len(rows)} options across {len(options_by_expiry)} expiries")

    except Exception as e:
        log(f"[ERROR] Deribit snapshot failed: {e}")
        log(f"[ERROR] Traceback: {traceback.format_exc()}")


def run(once: bool = False, verbose: bool = False) -> None:
    """Main collector loop."""
    config = load_config()

    edge_interval = config.get("edge_check_interval_seconds", 300)
    orderbook_interval = config.get("orderbook_interval_seconds", 60)
    snapshot_interval = config.get("deribit_snapshot_interval_seconds", 300)

    # Resolve paths relative to data_collector/ so files stay separate from the trading bot
    probabilities_path = COLLECTOR_DIR / config.get("probabilities_file", "results/probabilities.csv")
    orderbook_path = COLLECTOR_DIR / config.get("orderbook_file", "results/orderbook.csv")
    snapshot_path = COLLECTOR_DIR / config.get("deribit_snapshot_file", "results/deribit_options.csv")

    # Ensure output directories exist
    probabilities_path.parent.mkdir(parents=True, exist_ok=True)
    orderbook_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)

    order_book_client = OrderBookClient()

    last_orderbook: Optional[datetime] = None
    last_edge_check: Optional[datetime] = None
    last_snapshot: Optional[datetime] = None
    current_market: Optional[dict] = None

    log("Starting data collector")
    log(f"  Edge check interval:    {edge_interval}s")
    log(f"  Orderbook interval:     {orderbook_interval}s")
    log(f"  Deribit snap interval:  {snapshot_interval}s")
    log(f"  Probabilities file:     {probabilities_path}")
    log(f"  Orderbook file:         {orderbook_path}")
    log(f"  Deribit snapshot file:  {snapshot_path}")

    while True:
        now = datetime.now(timezone.utc)

        # 1. Fetch/refresh market (every edge check interval, or on startup)
        should_fetch_market = (
            current_market is None
            or last_edge_check is None
            or (now - last_edge_check).total_seconds() >= edge_interval
        )

        if should_fetch_market:
            market = fetch_current_market()
            if market:
                if current_market is None or current_market.get("event_id") != market.get("event_id"):
                    log(f"Market: {market.get('title')}")
                    log(f"  Barrier: ${market.get('barrier_price', 0):,.2f}")
                    hours_rem = get_time_remaining_hours(market)
                    hours = int(hours_rem)
                    minutes = int((hours_rem - hours) * 60)
                    log(f"  Time remaining: {hours}h {minutes}m")
                current_market = market

        if current_market is None:
            log("No market available, retrying in 60s...")
            if once:
                return
            time.sleep(60)
            continue

        # Check for market expiry
        if get_time_remaining_hours(current_market) <= 0:
            log("Market expired, fetching new market...")
            current_market = None
            if once:
                return
            time.sleep(10)
            continue

        # 2. Orderbook snapshot
        should_snapshot = (
            last_orderbook is None
            or (now - last_orderbook).total_seconds() >= orderbook_interval
        )

        if should_snapshot:
            try:
                snapshot_orderbook(current_market, order_book_client, orderbook_path)
                last_orderbook = datetime.now(timezone.utc)
            except Exception as e:
                log(f"[ERROR] Orderbook snapshot failed: {e}")

        # 3. Deribit options snapshot
        should_snapshot_deribit = (
            last_snapshot is None
            or (now - last_snapshot).total_seconds() >= snapshot_interval
        )

        if should_snapshot_deribit:
            snapshot_deribit_options(snapshot_path)
            last_snapshot = datetime.now(timezone.utc)

        # 4. Edge check (model calibration)
        should_edge_check = (
            last_edge_check is None
            or (now - last_edge_check).total_seconds() >= edge_interval
        )

        if should_edge_check:
            run_edge_check(probabilities_path, verbose=verbose)
            last_edge_check = datetime.now(timezone.utc)

        if once:
            return

        # 5. Sleep until next action
        now = datetime.now(timezone.utc)
        next_orderbook = orderbook_interval
        if last_orderbook:
            elapsed = (now - last_orderbook).total_seconds()
            next_orderbook = max(0, orderbook_interval - elapsed)

        next_edge = edge_interval
        if last_edge_check:
            elapsed = (now - last_edge_check).total_seconds()
            next_edge = max(0, edge_interval - elapsed)

        next_snapshot = snapshot_interval
        if last_snapshot:
            elapsed = (now - last_snapshot).total_seconds()
            next_snapshot = max(0, snapshot_interval - elapsed)

        sleep_time = max(1, min(next_orderbook, next_edge, next_snapshot))
        time.sleep(sleep_time)


def main() -> None:
    """Entry point with argument parsing."""
    import argparse

    parser = argparse.ArgumentParser(description="Standalone BTC data collector")
    parser.add_argument("--once", action="store_true", help="Run once and exit")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    args = parser.parse_args()

    run(once=args.once, verbose=args.verbose)


if __name__ == "__main__":
    main()
