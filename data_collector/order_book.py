"""Self-contained CLOB order book client for the data collector.

Decoupled from scripts/dry_run/order_book.py so that the dry-run trader
and data collector never share CSV writers or token-ID formats.
"""

import csv
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import requests

CLOB_URL = "https://clob.polymarket.com"


@dataclass
class OrderBookSnapshot:
    """Snapshot of order book at a point in time."""

    token_id: str
    best_bid: float
    best_ask: float
    bids: list[tuple[float, float]]  # (price, size) sorted desc by price
    asks: list[tuple[float, float]]  # (price, size) sorted asc by price
    timestamp: str


class OrderBookClient:
    """Client for fetching CLOB order book data."""

    def __init__(self, timeout: int = 10):
        self.timeout = timeout

    def fetch_order_book(self, token_id: str) -> Optional[OrderBookSnapshot]:
        """Fetch order book for a token.

        Args:
            token_id: The CLOB token ID.

        Returns:
            OrderBookSnapshot or None if fetch fails.
        """
        url = f"{CLOB_URL}/book"
        params = {"token_id": token_id}

        try:
            response = requests.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()

            bids = []
            for bid in data.get("bids", []):
                price = float(bid.get("price", 0))
                size = float(bid.get("size", 0))
                if price > 0 and size > 0:
                    bids.append((price, size))

            asks = []
            for ask in data.get("asks", []):
                price = float(ask.get("price", 0))
                size = float(ask.get("size", 0))
                if price > 0 and size > 0:
                    asks.append((price, size))

            bids.sort(key=lambda x: x[0], reverse=True)
            asks.sort(key=lambda x: x[0])

            best_bid = bids[0][0] if bids else 0.0
            best_ask = asks[0][0] if asks else 0.0

            return OrderBookSnapshot(
                token_id=token_id,
                best_bid=best_bid,
                best_ask=best_ask,
                bids=bids,
                asks=asks,
                timestamp=data.get("timestamp", ""),
            )

        except requests.RequestException as e:
            print(f"Error fetching order book: {e}")
            return None
        except (KeyError, ValueError) as e:
            print(f"Error parsing order book: {e}")
            return None


def log_orderbook_to_csv(
    token_id: str,
    best_bid: float,
    best_ask: float,
    spread: float,
    bid_depth: int,
    ask_depth: int,
    csv_path: Path,
    sanity_ok: bool,
    barrier_price: Optional[float] = None,
) -> None:
    """Append orderbook snapshot to CSV file."""
    file_exists = csv_path.exists()

    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "token_id": token_id,
        "best_bid": best_bid,
        "best_ask": best_ask,
        "spread": spread,
        "spread_pct": (spread / best_ask * 100) if best_ask > 0 else 0,
        "bid_depth": bid_depth,
        "ask_depth": ask_depth,
        "sanity_ok": sanity_ok,
        "barrier_price": barrier_price if barrier_price is not None else "",
    }

    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
