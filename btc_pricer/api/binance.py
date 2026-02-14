"""Binance API client for fetching BTC spot price."""
import logging
from typing import Tuple

import requests


def fetch_binance_spot_price(symbol: str = "BTCUSDT") -> float:
    """Fetch current spot price from Binance.

    Args:
        symbol: Trading pair symbol (default: BTCUSDT)

    Returns:
        Current spot price in USD

    Raises:
        Exception: If Binance API call fails
    """
    logger = logging.getLogger(__name__)
    url = f"https://api.binance.com/api/v3/ticker/price?symbol={symbol}"
    response = requests.get(url, timeout=5)
    response.raise_for_status()
    data = response.json()
    price = float(data["price"])
    logger.info(f"Binance spot price: ${price:,.2f}")
    return price


def fetch_spot_with_fallback(deribit_spot: float, symbol: str = "BTCUSDT") -> Tuple[float, str]:
    """Fetch spot price from Binance with Deribit fallback.

    This function centralizes the spot price fetching logic that was previously
    duplicated across cli.py, cli_terminal.py, and cli_intraday.py.

    Args:
        deribit_spot: Spot price from Deribit API (used as fallback).
        symbol: Binance trading pair symbol (default: BTCUSDT).

    Returns:
        Tuple of (spot_price, source) where source is 'binance' or 'deribit'.

    Example:
        >>> spot, source = fetch_spot_with_fallback(deribit_spot=100000.0)
        >>> print(f"Using {source} spot: ${spot:,.2f}")
    """
    logger = logging.getLogger(__name__)

    try:
        spot_price = fetch_binance_spot_price(symbol)
        diff_pct = (spot_price - deribit_spot) / deribit_spot * 100
        logger.info(
            f"Deribit spot: ${deribit_spot:,.2f}, using Binance (diff: {diff_pct:+.2f}%)"
        )
        return spot_price, "binance"
    except Exception as e:
        logger.warning(f"Binance failed: {e}, using Deribit spot")
        return deribit_spot, "deribit"
