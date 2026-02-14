#!/usr/bin/env python3
"""Fetch closest Bitcoin Up/Down daily market from Polymarket."""

import argparse
import json
import re
import requests
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

GAMMA_URL = "https://gamma-api.polymarket.com"
BINANCE_URL = "https://api.binance.com/api/v3"


def _parse_all_reference_times(description):
    """Extract all ET date/time references from the market description.

    Matches both formats:
      - "Feb 5 '26 12:00 in the ET timezone" (barrier/reference time)
      - "Feb 6 '26 12:00 ET" (resolution time)

    Returns list of UTC datetimes in order of appearance.
    """
    pattern = r"(\w{3}) (\d{1,2}) '(\d{2}) (\d{1,2}):(\d{2}) (?:in the )?ET"
    matches = re.findall(pattern, description)

    months = {
        "Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "Jun": 6,
        "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12
    }

    results = []
    et_tz = ZoneInfo("America/New_York")
    for month_str, day, year, hour, minute in matches:
        month = months.get(month_str)
        if not month:
            continue
        year_full = 2000 + int(year)
        et_dt = datetime(year_full, month, int(day), int(hour), int(minute), tzinfo=et_tz)
        results.append(et_dt.astimezone(timezone.utc))

    return results


def parse_reference_time(description):
    """Extract the barrier/reference candle time (first date) from the description."""
    times = _parse_all_reference_times(description)
    return times[0] if times else None


def parse_resolution_time(description):
    """Extract the resolution candle time (second date) from the description."""
    times = _parse_all_reference_times(description)
    return times[1] if len(times) >= 2 else times[0] if times else None


def get_binance_price(timestamp_utc):
    """Fetch BTC/USDT close price for a specific 1-minute candle from Binance."""
    url = f"{BINANCE_URL}/klines"
    start_time_ms = int(timestamp_utc.timestamp() * 1000)

    params = {
        "symbol": "BTCUSDT",
        "interval": "1m",
        "startTime": start_time_ms,
        "limit": 1
    }

    response = requests.get(url, params=params, timeout=10)
    response.raise_for_status()
    data = response.json()

    if data:
        # Kline format: [open_time, open, high, low, close, volume, ...]
        return float(data[0][4])
    return None


def get_current_btc_price():
    """Fetch current BTC/USDT price from Binance."""
    url = f"{BINANCE_URL}/ticker/price"
    params = {"symbol": "BTCUSDT"}

    response = requests.get(url, params=params, timeout=10)
    response.raise_for_status()
    data = response.json()

    return float(data.get("price", 0))


def search_btc_daily_markets():
    """Search for Bitcoin Up or Down daily markets (today or upcoming)."""
    url = f"{GAMMA_URL}/public-search"

    # Search broadly for daily markets - filter by date happens in find_closest_active_market
    params = {"q": "Bitcoin Up or Down on"}

    response = requests.get(url, params=params, timeout=10)
    response.raise_for_status()

    return response.json()


def find_closest_active_market(data):
    """Filter for closest non-expired market matching the daily pattern."""
    now = datetime.now(timezone.utc)

    # Patterns to match both title formats:
    # "Bitcoin Up or Down on January 28?" (daily)
    # "Bitcoin Up or Down - January 28, 8AM ET" (hourly)
    patterns = [
        re.compile(r"Bitcoin Up or Down on \w+ \d+\??", re.IGNORECASE),
        re.compile(r"Bitcoin Up or Down - \w+ \d+,", re.IGNORECASE),
    ]

    # The API returns events in a dict with 'events' key
    events = data if isinstance(data, list) else data.get("events", [])

    candidates = []
    for event in events:
        title = event.get("title", "")

        # Check if title matches any of the patterns
        if not any(p.search(title) for p in patterns):
            continue

        # Skip closed markets
        if event.get("closed", False):
            continue

        # Parse end date
        end_date_str = event.get("endDate")
        if not end_date_str:
            continue

        try:
            end_date = datetime.fromisoformat(end_date_str.replace("Z", "+00:00"))
        except ValueError:
            continue

        # Skip expired markets
        if end_date <= now:
            continue

        candidates.append((end_date, event))

    if not candidates:
        return None

    # Sort by end date ascending to get the closest one
    candidates.sort(key=lambda x: x[0])
    return candidates[0][1]


def market_to_dict(event):
    """Extract market data into a dictionary for JSON output.

    Args:
        event: Polymarket event data

    Returns:
        Dictionary with structured market data, or None if parsing fails
    """
    title = event.get("title", "Unknown Market")
    description = event.get("description", "")
    end_date_str = event.get("endDate")

    result = {
        "market_title": title,
        "barrier": None,
        "current_price": None,
        "hours_remaining": None,
        "hours": None,
        "minutes": None,
        "prob_up": None,
        "prob_down": None,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    # Calculate time remaining
    if end_date_str:
        try:
            end_date = datetime.fromisoformat(end_date_str.replace("Z", "+00:00"))
            now = datetime.now(timezone.utc)
            remaining = end_date - now
            total_seconds = int(remaining.total_seconds())

            if total_seconds > 0:
                hours, remainder = divmod(total_seconds, 3600)
                minutes, _ = divmod(remainder, 60)
                result["hours"] = hours
                result["minutes"] = minutes
                result["hours_remaining"] = hours + minutes / 60
        except ValueError:
            pass

    # Get reference price (barrier) from Binance
    ref_time = parse_reference_time(description)
    if ref_time:
        try:
            ref_price = get_binance_price(ref_time)
            current_price = get_current_btc_price()
            if ref_price:
                result["barrier"] = ref_price
            if current_price:
                result["current_price"] = current_price
        except Exception:
            pass

    # Extract market data from the markets array
    markets = event.get("markets", [])
    if not markets:
        return result

    market = markets[0]
    outcomes = market.get("outcomes", [])
    outcome_prices = market.get("outcomePrices", [])

    # API sometimes returns JSON strings instead of lists
    if isinstance(outcomes, str):
        try:
            outcomes = json.loads(outcomes)
        except json.JSONDecodeError:
            outcomes = ["Up", "Down"]

    if isinstance(outcome_prices, str):
        try:
            outcome_prices = json.loads(outcome_prices)
        except json.JSONDecodeError:
            outcome_prices = []

    # Extract probabilities
    for i, outcome in enumerate(outcomes):
        if i < len(outcome_prices):
            try:
                price = float(outcome_prices[i])
                outcome_lower = outcome.lower()
                if outcome_lower == "up":
                    result["prob_up"] = price
                elif outcome_lower == "down":
                    result["prob_down"] = price
            except (ValueError, TypeError):
                pass

    return result


def display_market_info(event):
    """Display probabilities and prices.

    Returns:
        Dictionary with market data for JSON output
    """
    title = event.get("title", "Unknown Market")
    description = event.get("description", "")
    end_date_str = event.get("endDate")

    print(title)

    # Calculate time remaining and show expiry times
    if end_date_str:
        try:
            end_date = datetime.fromisoformat(end_date_str.replace("Z", "+00:00"))
            now = datetime.now(timezone.utc)
            remaining = end_date - now

            # Display expiry time in ET, UTC, and Paris
            end_utc_str = end_date.strftime("%H:%M UTC")
            # ET is UTC-5 (EST) or UTC-4 (EDT) - using EST for winter
            end_et = end_date.replace(tzinfo=None) - timedelta(hours=5)
            end_et_str = end_et.strftime("%H:%M ET")
            # Paris is UTC+1 (CET) in winter
            end_paris = end_date.replace(tzinfo=None) + timedelta(hours=1)
            end_paris_str = end_paris.strftime("%H:%M Paris")
            print(f"Expires: {end_et_str} / {end_utc_str} / {end_paris_str}")

            total_seconds = int(remaining.total_seconds())
            if total_seconds > 0:
                hours, remainder = divmod(total_seconds, 3600)
                minutes, _ = divmod(remainder, 60)
                print(f"Time remaining: {hours}h {minutes}m")
            else:
                print("Market has expired")
        except ValueError:
            pass

    print("-" * 40)

    # Get reference price from Binance
    ref_time = parse_reference_time(description)
    if ref_time:
        try:
            ref_price = get_binance_price(ref_time)
            current_price = get_current_btc_price()
            if ref_price:
                print(f"Price to beat: ${ref_price:,.2f}")
            if current_price:
                print(f"Current price: ${current_price:,.2f}")
                if ref_price:
                    diff = current_price - ref_price
                    pct = (diff / ref_price) * 100
                    direction = "above" if diff >= 0 else "below"
                    print(f"Difference:    {direction} by ${abs(diff):,.2f} ({pct:+.2f}%)")
            print("-" * 40)
        except Exception:
            pass

    # Extract market data from the markets array
    markets = event.get("markets", [])

    if not markets:
        print("No market data available")
        return market_to_dict(event)

    # Usually there's one market per event for these binary markets
    market = markets[0]
    outcomes = market.get("outcomes", [])
    outcome_prices = market.get("outcomePrices", [])

    # API sometimes returns JSON strings instead of lists
    if isinstance(outcomes, str):
        try:
            outcomes = json.loads(outcomes)
        except json.JSONDecodeError:
            outcomes = ["Up", "Down"]

    if isinstance(outcome_prices, str):
        try:
            outcome_prices = json.loads(outcome_prices)
        except json.JSONDecodeError:
            outcome_prices = []

    # Display each outcome
    for i, outcome in enumerate(outcomes):
        if i < len(outcome_prices):
            try:
                price = float(outcome_prices[i])
                probability = price * 100
                outcome_upper = outcome.upper()
                print(f"{outcome_upper:5} {probability:5.1f}% (${price:.3f})")
            except (ValueError, TypeError):
                print(f"{outcome}: Price unavailable")
        else:
            print(f"{outcome}: Price unavailable")

    return market_to_dict(event)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Fetch closest Bitcoin Up/Down daily market from Polymarket"
    )
    parser.add_argument(
        "--json",
        type=Path,
        metavar="PATH",
        help="Output JSON file path for structured data"
    )
    args = parser.parse_args()

    try:
        print("Fetching Bitcoin Up/Down markets from Polymarket...\n")
        data = search_btc_daily_markets()
        closest = find_closest_active_market(data)

        if closest:
            market_data = display_market_info(closest)

            # Write JSON output if requested
            if args.json and market_data:
                args.json.parent.mkdir(parents=True, exist_ok=True)
                with open(args.json, "w") as f:
                    json.dump(market_data, f, indent=2)
                print(f"\nJSON data written to {args.json}")
        else:
            print("No active Bitcoin Up/Down market found")

    except requests.RequestException as e:
        print(f"Error fetching data: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")


if __name__ == "__main__":
    main()
