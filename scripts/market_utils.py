"""Shared market utilities for Polymarket BTC daily markets."""

import json
from datetime import datetime, timezone
from typing import Optional

from scripts.polymarket_btc_daily import (
    search_btc_daily_markets,
    find_closest_active_market,
    parse_reference_time,
    get_binance_price,
)


def fetch_current_market(log_fn=None) -> Optional[dict]:
    """Fetch the current active Polymarket BTC daily market.

    Args:
        log_fn: Optional logging function (called with message string).

    Returns:
        Dict with event_id, title, end_date, reference_price,
        clob_token_ids, description — or None on failure.
    """
    try:
        data = search_btc_daily_markets()
        event = find_closest_active_market(data)

        if not event:
            if log_fn:
                log_fn("No active market found")
            return None

        title = event.get("title", "")
        end_date_str = event.get("endDate")
        description = event.get("description", "")

        end_date = None
        if end_date_str:
            end_date = datetime.fromisoformat(end_date_str.replace("Z", "+00:00"))

        ref_time = parse_reference_time(description)
        reference_price = get_binance_price(ref_time) if ref_time else None

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
            "reference_price": reference_price,
            "clob_token_ids": clob_token_ids,
            "description": description,
        }

    except Exception as e:
        if log_fn:
            log_fn(f"[ERROR] Error fetching market: {e}")
            import traceback
            log_fn(f"[ERROR] Traceback: {traceback.format_exc()}")
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
