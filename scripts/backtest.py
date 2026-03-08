"""Backtest script for replay of paper/live trading strategy against historical data."""

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional

# Add project root to path for btc_pricer imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from btc_pricer.edge import has_edge as _has_edge


# Price source indicators for trade output
PRICE_SOURCE_ORDERBOOK = "OB"  # Orderbook price used
PRICE_SOURCE_FALLBACK = "FB"  # Fallback to probability price

# Minimum order size to avoid dust trades and division-by-zero
MIN_ORDER_SIZE = 1.0


class Direction(str, Enum):
    UP = "UP"
    DOWN = "DOWN"


class TradeResult(str, Enum):
    TP_FILLED = "TP_FILLED"
    TRAILING_STOP = "TRAILING_STOP"
    STOP_LOSS = "STOP_LOSS"
    WIN_EXPIRY = "WIN_EXPIRY"
    LOSS_EXPIRY = "LOSS_EXPIRY"


@dataclass
class Position:
    direction: Direction
    entry_price: float  # poly_prob at entry (0-1 scale, e.g., $0.35)
    shares: float
    cost_basis: float
    entry_timestamp: str
    reference_price: float
    entry_source: str = PRICE_SOURCE_FALLBACK  # Track price source
    peak_price: float = 0.0  # Highest price seen since entry (for trailing stop)
    model_prob: float = 0.0  # model probability at entry (0-1)
    market_prob: float = 0.0  # Polymarket probability at entry (0-1)
    edge: float = 0.0  # model_prob / market_prob ratio
    spot_at_entry: float = 0.0  # BTC spot price at entry
    time_remaining_hours: float = 0.0  # hours to expiry at entry


@dataclass
class Trade:
    direction: str
    entry_price: float
    exit_price: float
    shares: float
    cost_basis: float
    proceeds: float
    pnl: float
    pnl_pct: float
    result: str
    entry_timestamp: str
    exit_timestamp: str
    reference_price: float
    final_spot: Optional[float]
    entry_source: str = PRICE_SOURCE_FALLBACK  # OB = orderbook, FB = fallback
    exit_source: str = PRICE_SOURCE_FALLBACK
    model_prob: float = 0.0  # model probability at entry (0-1)
    market_prob: float = 0.0  # Polymarket probability at entry (0-1)
    edge: float = 0.0  # model_prob / market_prob ratio
    spot_at_entry: float = 0.0  # BTC spot price at entry
    time_remaining_hours: float = 0.0  # hours to expiry at entry


def load_csv(csv_path: str) -> list[dict]:
    """Load probabilities CSV into list of dicts."""
    rows = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Convert numeric fields
            # Support both old 'barrier_price' and new 'reference_price' column names
            if "reference_price" in row:
                row["reference_price"] = float(row["reference_price"])
            elif "barrier_price" in row:
                row["reference_price"] = float(row["barrier_price"])
            else:
                raise KeyError("CSV missing both 'reference_price' and 'barrier_price' columns")
            row["time_remaining_hours"] = float(row["time_remaining_hours"])
            row["spot_price"] = float(row["spot_price"])
            row["model_prob_up"] = float(row["model_prob_up"])
            row["model_prob_down"] = float(row["model_prob_down"])
            row["poly_prob_up"] = float(row["poly_prob_up"])
            row["poly_prob_down"] = float(row["poly_prob_down"])
            row["edge_up"] = float(row["edge_up"])
            row["edge_down"] = float(row["edge_down"])

            # B-L and averaged columns (backward compatible)
            bl_up = row.get("bl_prob_up", "")
            bl_down = row.get("bl_prob_down", "")
            row["bl_prob_up"] = float(bl_up) if bl_up else 0.0
            row["bl_prob_down"] = float(bl_down) if bl_down else 0.0

            avg_up = row.get("avg_prob_up", "")
            avg_down = row.get("avg_prob_down", "")
            row["avg_prob_up"] = float(avg_up) if avg_up else row["model_prob_up"]
            row["avg_prob_down"] = float(avg_down) if avg_down else row["model_prob_down"]

            rows.append(row)
    return rows


def load_orderbook_csv(csv_path: str) -> dict[tuple[float, str], list[dict]]:
    """
    Load orderbook CSV indexed by (reference_price, direction) and sorted by timestamp.

    Args:
        csv_path: Path to orderbook.csv

    Returns:
        {(reference_price, "UP"): [rows...], ...} sorted by timestamp ascending.
        Legacy data without reference_price uses sentinel key (0.0, direction).
    """
    orderbook: dict[tuple[float, str], list[dict]] = {}

    try:
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Parse token_id to get direction (e.g., "UP:36656454" -> "UP")
                token_id = row.get("token_id", "")
                if ":" in token_id:
                    direction = token_id.split(":")[0].upper()
                else:
                    continue

                if direction not in ("UP", "DOWN"):
                    continue

                # Parse reference_price if present; fallback to 0.0 for legacy data
                barrier_str = row.get("reference_price", "")
                try:
                    barrier = float(barrier_str) if barrier_str else 0.0
                except (ValueError, TypeError):
                    barrier = 0.0

                key = (barrier, direction)

                # Convert numeric fields
                parsed_row = {
                    "timestamp": row["timestamp"],
                    "best_bid": float(row["best_bid"]),
                    "best_ask": float(row["best_ask"]),
                    "spread": float(row["spread"]),
                }
                orderbook.setdefault(key, []).append(parsed_row)

        # Sort each group by timestamp
        for key in orderbook:
            orderbook[key].sort(key=lambda r: r["timestamp"])

    except FileNotFoundError:
        pass  # Expected — no orderbook file
    except Exception as e:
        print(f"Warning: Failed to parse orderbook CSV: {e}")

    return orderbook


def get_orderbook_prices(
    orderbook: dict[tuple[float, str], list[dict]] | None,
    timestamp: str,
    direction: str,
    reference_price: float = 0.0,
) -> tuple[float | None, float | None]:
    """
    Get bid/ask prices from orderbook for given timestamp, direction, and barrier.

    Uses the most recent orderbook snapshot with timestamp <= given timestamp
    to avoid forward-looking bias.

    Args:
        orderbook: Dict keyed by (reference_price, direction) with lists of orderbook rows
        timestamp: Target timestamp (ISO format)
        direction: "UP" or "DOWN"
        reference_price: Barrier price for this market (0.0 = legacy fallback)

    Returns:
        (bid, ask) or (None, None) if not available
    """
    if orderbook is None:
        return None, None

    dir_upper = direction.upper()
    # Try exact (reference_price, direction) first, then legacy (0.0, direction)
    rows = orderbook.get((reference_price, dir_upper))
    if not rows and reference_price != 0.0:
        rows = orderbook.get((0.0, dir_upper))
    if not rows:
        return None, None

    # Binary search for the last row with timestamp <= target
    # (rows are sorted by timestamp ascending)
    left, right = 0, len(rows) - 1
    result_idx = -1

    while left <= right:
        mid = (left + right) // 2
        if rows[mid]["timestamp"] <= timestamp:
            result_idx = mid
            left = mid + 1
        else:
            right = mid - 1

    if result_idx == -1:
        return None, None

    row = rows[result_idx]
    return row["best_bid"], row["best_ask"]


def get_orderbook_rows_in_range(
    orderbook: dict[tuple[float, str], list[dict]] | None,
    start_ts: str,
    end_ts: str,
    direction: str,
    reference_price: float = 0.0,
) -> list[dict]:
    """
    Return all orderbook rows with start_ts < timestamp <= end_ts.

    Uses two binary searches on the sorted rows list for O(log n) lookup.

    Args:
        orderbook: Dict keyed by (reference_price, direction) with sorted row lists
        start_ts: Exclusive lower bound timestamp (ISO format)
        end_ts: Inclusive upper bound timestamp (ISO format)
        direction: "UP" or "DOWN"
        reference_price: Barrier price for this market

    Returns:
        List of orderbook row dicts in chronological order
    """
    if orderbook is None:
        return []

    dir_upper = direction.upper()
    rows = orderbook.get((reference_price, dir_upper))
    if not rows and reference_price != 0.0:
        rows = orderbook.get((0.0, dir_upper))
    if not rows:
        return []

    # Binary search for first row with timestamp > start_ts
    lo, hi = 0, len(rows)
    while lo < hi:
        mid = (lo + hi) // 2
        if rows[mid]["timestamp"] <= start_ts:
            lo = mid + 1
        else:
            hi = mid
    first = lo

    # Binary search for last row with timestamp <= end_ts
    lo, hi = first, len(rows)
    while lo < hi:
        mid = (lo + hi) // 2
        if rows[mid]["timestamp"] <= end_ts:
            lo = mid + 1
        else:
            hi = mid
    last = lo

    return rows[first:last]


def get_entry_price(direction: Direction, row: dict, orderbook, reference_price: float):
    """Get entry price: orderbook ASK, fallback to poly_prob."""
    bid, ask = get_orderbook_prices(orderbook, row["timestamp"], direction.value, reference_price=reference_price)
    if ask is not None:
        return ask, PRICE_SOURCE_ORDERBOOK
    poly_field = "poly_prob_up" if direction == Direction.UP else "poly_prob_down"
    return row[poly_field] / 100, PRICE_SOURCE_FALLBACK


def try_open_position(
    direction: Direction,
    entry_price: float,
    entry_source: str,
    barrier: float,
    timestamp: str,
    current_capital: float,
    order_size_pct: float,
    friction: float,
    model_prob: float = 0.0,
    market_prob: float = 0.0,
    edge: float = 0.0,
    spot_at_entry: float = 0.0,
    time_remaining_hours: float = 0.0,
) -> tuple[Optional["Position"], float]:
    """Create position if order size is valid. Returns (Position, cost) or (None, 0)."""
    order_size = current_capital * order_size_pct
    if order_size < MIN_ORDER_SIZE or entry_price <= 0:
        return None, 0.0
    cost = order_size * (1 + friction)
    shares = order_size / entry_price
    return Position(
        direction=direction,
        entry_price=entry_price,
        shares=shares,
        cost_basis=cost,
        entry_timestamp=timestamp,
        reference_price=barrier,
        entry_source=entry_source,
        peak_price=entry_price,
        model_prob=model_prob,
        market_prob=market_prob,
        edge=edge,
        spot_at_entry=spot_at_entry,
        time_remaining_hours=time_remaining_hours,
    ), cost


def group_markets(df: list[dict]) -> dict[float, list[dict]]:
    """Group rows by reference_price, sorted by timestamp within each group."""
    markets: dict[float, list[dict]] = {}
    for row in df:
        barrier = row["reference_price"]
        markets.setdefault(barrier, []).append(row)
    for barrier in markets:
        markets[barrier].sort(key=lambda r: r["timestamp"])
    return markets


@dataclass
class PendingSignal:
    """A pending entry signal waiting for latency period."""
    direction: Direction
    signal_timestamp: str
    reference_price: float
    model_prob: float = 0.0
    market_prob: float = 0.0
    edge: float = 0.0
    spot_at_entry: float = 0.0
    time_remaining_hours: float = 0.0


def run_backtest(
    df: list[dict],
    edge_up: float = 2.0,
    edge_down: float = 1.5,
    tp_pct: float = 0.25,
    friction: float = 0.02,
    capital: float = 100.0,
    order_size_pct: float = 0.05,
    latency_minutes: float = 0.0,
    orderbook: dict[tuple[float, str], list[dict]] | None = None,
    trail_activation: float = 0.20,
    trail_distance: float = 0.15,
    min_time_remaining_hours: float = 2.0,
    min_model_prob: float = 0.0,
    alpha: float | None = None,
    floor: float | None = None,
    alpha_up: float | None = None,
    alpha_down: float | None = None,
    floor_up: float | None = None,
    floor_down: float | None = None,
    stop_loss_pct: float = 0.0,
) -> tuple[list[Trade], float]:
    """
    Main backtest loop.

    Args:
        df: List of row dicts from CSV
        edge_up: Min edge for UP entry (legacy, ignored when alpha is set)
        edge_down: Min edge for DOWN entry (legacy, ignored when alpha is set)
        tp_pct: Take profit percentage (0.25 = 25%)
        friction: Fee + spread per side (0.02 = 2%)
        capital: Starting capital
        order_size_pct: Fraction of capital per trade
        latency_minutes: Delay between signal and entry (default: 0)
        orderbook: Optional orderbook data for realistic pricing
        trail_activation: Profit % to activate trailing stop (0.20 = 20%)
        trail_distance: Trail distance in pp from peak (0.15 = 15pp)
        min_model_prob: Min model probability (0-1) to enter (legacy)
        alpha: Edge curve exponent (None = use legacy). Deprecated: use alpha_up/alpha_down.
        floor: Min model probability floor. Deprecated: use floor_up/floor_down.
        alpha_up: Per-direction alpha for UP (overrides alpha if set)
        alpha_down: Per-direction alpha for DOWN (overrides alpha if set)
        floor_up: Per-direction floor for UP (overrides floor if set)
        floor_down: Per-direction floor for DOWN (overrides floor if set)
        stop_loss_pct: Stop-loss percentage (0.20 = exit at 20% below entry, 0 = disabled)

    Returns:
        Tuple of (trades list, final capital)
    """
    # Resolve per-direction params: alpha_up/alpha_down take priority over alpha
    _alpha_up = alpha_up if alpha_up is not None else alpha
    _alpha_down = alpha_down if alpha_down is not None else alpha
    _floor_up = floor_up if floor_up is not None else (floor if floor is not None else 0.65)
    _floor_down = floor_down if floor_down is not None else (floor if floor is not None else 0.65)
    use_continuous_edge = _alpha_up is not None
    trades: list[Trade] = []
    current_capital = capital
    position: Optional[Position] = None
    pending_signal: Optional[PendingSignal] = None

    # Group rows by reference_price (each barrier = one market)
    markets = group_markets(df)

    # Process markets in order of first timestamp
    sorted_barriers = sorted(markets.keys(), key=lambda b: markets[b][0]["timestamp"])

    for barrier in sorted_barriers:
        market_df = markets[barrier]
        # Reset pending signal when market changes
        pending_signal = None
        last_ob_checked_ts = ""

        for i, row in enumerate(market_df):
            is_last = i == len(market_df) - 1
            row_ts = datetime.fromisoformat(row["timestamp"].replace("Z", "+00:00"))

            # Check if pending signal is ready to execute
            if pending_signal is not None and position is None:
                signal_ts = datetime.fromisoformat(pending_signal.signal_timestamp.replace("Z", "+00:00"))
                elapsed_minutes = (row_ts - signal_ts).total_seconds() / 60

                if elapsed_minutes >= latency_minutes and row["time_remaining_hours"] >= min_time_remaining_hours:
                    # Execute the pending entry at current price
                    entry_price, entry_source = get_entry_price(pending_signal.direction, row, orderbook, barrier)
                    pos, cost = try_open_position(
                        pending_signal.direction, entry_price, entry_source,
                        barrier, row["timestamp"], current_capital, order_size_pct, friction,
                        model_prob=pending_signal.model_prob,
                        market_prob=pending_signal.market_prob,
                        edge=pending_signal.edge,
                        spot_at_entry=pending_signal.spot_at_entry,
                        time_remaining_hours=pending_signal.time_remaining_hours,
                    )
                    if pos:
                        position = pos
                        current_capital -= cost
                        last_ob_checked_ts = row["timestamp"]
                    pending_signal = None

            # Entry logic (if no position and no pending signal)
            if position is None and pending_signal is None and row["time_remaining_hours"] >= min_time_remaining_hours:
                # Use averaged probabilities if available, fall back to model_prob
                avg_up = row.get("avg_prob_up", row["model_prob_up"])
                avg_down = row.get("avg_prob_down", row["model_prob_down"])

                # Recompute edge from averaged probabilities (when B-L data exists)
                if "avg_prob_up" in row:
                    if row["poly_prob_up"] > 0:
                        row["edge_up"] = avg_up / row["poly_prob_up"]
                    if row["poly_prob_down"] > 0:
                        row["edge_down"] = avg_down / row["poly_prob_down"]

                for direction, edge_field, edge_threshold, avg_val, poly_field, dir_alpha, dir_floor in [
                    (Direction.UP, "edge_up", edge_up, avg_up, "poly_prob_up", _alpha_up, _floor_up),
                    (Direction.DOWN, "edge_down", edge_down, avg_down, "poly_prob_down", _alpha_down, _floor_down),
                ]:
                    if use_continuous_edge:
                        model_p = avg_val / 100
                        market_p = row[poly_field] / 100
                        entry_signal = _has_edge(model_p, market_p, dir_alpha, dir_floor)
                    else:
                        entry_signal = row[edge_field] >= edge_threshold and avg_val / 100 >= min_model_prob
                    if entry_signal:
                        # Capture entry context from signal row
                        _model_p = avg_val / 100
                        _market_p = row[poly_field] / 100
                        _edge = _model_p / _market_p if _market_p > 0 else 0.0
                        _spot = row["spot_price"]
                        _time_rem = row["time_remaining_hours"]

                        if latency_minutes > 0:
                            pending_signal = PendingSignal(
                                direction=direction,
                                signal_timestamp=row["timestamp"],
                                reference_price=barrier,
                                model_prob=_model_p,
                                market_prob=_market_p,
                                edge=_edge,
                                spot_at_entry=_spot,
                                time_remaining_hours=_time_rem,
                            )
                        else:
                            entry_price, entry_source = get_entry_price(direction, row, orderbook, barrier)
                            pos, cost = try_open_position(
                                direction, entry_price, entry_source,
                                barrier, row["timestamp"], current_capital, order_size_pct, friction,
                                model_prob=_model_p,
                                market_prob=_market_p,
                                edge=_edge,
                                spot_at_entry=_spot,
                                time_remaining_hours=_time_rem,
                            )
                            if pos:
                                position = pos
                                current_capital -= cost
                                last_ob_checked_ts = row["timestamp"]
                        break  # Only enter one direction

            # Exit logic (if has position)
            elif position is not None:
                tp_price = position.entry_price * (1 + tp_pct)
                activation_price = position.entry_price * (1 + trail_activation)
                sl_price = position.entry_price * (1 - stop_loss_pct) if stop_loss_pct > 0 else 0.0
                exit_triggered = False
                exit_price = 0.0
                exit_source = PRICE_SOURCE_FALLBACK
                exit_timestamp = row["timestamp"]
                result = TradeResult.LOSS_EXPIRY
                final_spot: Optional[float] = None

                # --- Intra-bar orderbook scanning ---
                # Check OB snapshots between last checked timestamp and current row
                ob_rows = get_orderbook_rows_in_range(
                    orderbook, last_ob_checked_ts, row["timestamp"],
                    position.direction.value, reference_price=barrier,
                )
                for ob_row in ob_rows:
                    ob_bid = ob_row["best_bid"]
                    ob_ask = ob_row["best_ask"]
                    ob_mid = (ob_bid + ob_ask) / 2 if ob_ask is not None else ob_bid
                    # Update peak price from intra-bar data
                    position.peak_price = max(position.peak_price, ob_bid)

                    # TP check: limit order fills at tp_price
                    if ob_bid >= tp_price:
                        exit_price = tp_price
                        exit_source = PRICE_SOURCE_ORDERBOOK
                        exit_timestamp = ob_row["timestamp"]
                        result = TradeResult.TP_FILLED
                        exit_triggered = True
                        break

                    # Trail check: GTC sell fills at trail_level
                    if position.peak_price >= activation_price:
                        trail_level = position.peak_price - position.entry_price * trail_distance
                        if ob_bid <= trail_level:
                            exit_price = trail_level
                            exit_source = PRICE_SOURCE_ORDERBOOK
                            exit_timestamp = ob_row["timestamp"]
                            result = TradeResult.TRAILING_STOP
                            exit_triggered = True
                            break

                    # Stop-loss check: trigger on mid, fill at bid
                    if sl_price > 0 and ob_mid <= sl_price:
                        exit_price = ob_bid
                        exit_source = PRICE_SOURCE_ORDERBOOK
                        exit_timestamp = ob_row["timestamp"]
                        result = TradeResult.STOP_LOSS
                        exit_triggered = True
                        break

                # --- Probability-row checks (if no intra-bar exit) ---
                if not exit_triggered:
                    # Get current price from orderbook bid or fallback to poly_prob
                    bid, ask = get_orderbook_prices(orderbook, row["timestamp"], position.direction.value, reference_price=barrier)
                    if bid is not None:
                        current_price = bid
                        exit_source = PRICE_SOURCE_ORDERBOOK
                    elif position.direction == Direction.UP:
                        current_price = row["poly_prob_up"] / 100
                    else:
                        current_price = row["poly_prob_down"] / 100

                    # Update peak price tracking
                    position.peak_price = max(position.peak_price, current_price)

                    # 1. TP check
                    if current_price >= tp_price:
                        exit_price = tp_price  # Limit order fills at TP target
                        result = TradeResult.TP_FILLED
                        exit_triggered = True

                    # 2. Trailing stop check
                    if not exit_triggered:
                        if position.peak_price >= activation_price:
                            trail_level = position.peak_price - position.entry_price * trail_distance
                            if current_price <= trail_level:
                                exit_price = trail_level
                                result = TradeResult.TRAILING_STOP
                                exit_triggered = True

                    # 3. Stop-loss check (trigger on mid, fill at bid)
                    mid_price = (bid + ask) / 2 if bid is not None and ask is not None else current_price
                    if not exit_triggered and sl_price > 0 and mid_price <= sl_price:
                        exit_price = current_price
                        result = TradeResult.STOP_LOSS
                        exit_triggered = True

                    # 4. Expiry check (last row or time <= 0)
                    if not exit_triggered and (is_last or row["time_remaining_hours"] <= 0):
                        final_spot = row["spot_price"]
                        if position.direction == Direction.UP:
                            won = final_spot >= barrier
                        else:
                            won = final_spot < barrier

                        exit_price = 1.0 if won else 0.0
                        exit_source = PRICE_SOURCE_FALLBACK  # Expiry is always deterministic
                        result = TradeResult.WIN_EXPIRY if won else TradeResult.LOSS_EXPIRY
                        exit_triggered = True

                if exit_triggered:
                    if result in (TradeResult.TP_FILLED, TradeResult.TRAILING_STOP, TradeResult.STOP_LOSS):
                        proceeds = position.shares * exit_price * (1 - friction)
                    else:
                        # Expiry settlement (WIN/LOSS) has no trading friction
                        proceeds = position.shares * exit_price
                    pnl = proceeds - position.cost_basis
                    pnl_pct = (pnl / position.cost_basis) * 100 if position.cost_basis > 0 else 0.0

                    trade = Trade(
                        direction=position.direction.value,
                        entry_price=position.entry_price,
                        exit_price=exit_price,
                        shares=position.shares,
                        cost_basis=position.cost_basis,
                        proceeds=proceeds,
                        pnl=pnl,
                        pnl_pct=pnl_pct,
                        result=result.value,
                        entry_timestamp=position.entry_timestamp,
                        exit_timestamp=exit_timestamp,
                        reference_price=position.reference_price,
                        final_spot=final_spot,
                        entry_source=position.entry_source,
                        exit_source=exit_source,
                        model_prob=position.model_prob,
                        market_prob=position.market_prob,
                        edge=position.edge,
                        spot_at_entry=position.spot_at_entry,
                        time_remaining_hours=position.time_remaining_hours,
                    )
                    trades.append(trade)
                    current_capital += proceeds
                    position = None

                last_ob_checked_ts = row["timestamp"]

    return trades, current_capital


def print_trades_table(trades: list[Trade]) -> None:
    """Print a condensed table of trades."""
    print("\n" + "=" * 80)
    print("TRADES")
    print("=" * 80)

    if not trades:
        print("\n  No trades executed.")
        return

    # Header
    print(f"\n  {'#':<3} {'Dir':<5} {'Barrier':>12} {'Open Time':<17} {'Close Time':<17} {'Result':<12} {'P&L':>10}")
    print("  " + "-" * 78)

    RESULT_LABELS = {"TP_FILLED": "TP", "TRAILING_STOP": "TRAIL", "STOP_LOSS": "SL", "WIN_EXPIRY": "WIN", "LOSS_EXPIRY": "LOSS"}

    for i, t in enumerate(trades, 1):
        result_short = RESULT_LABELS.get(t.result, t.result)
        pnl_str = f"+${t.pnl:.2f}" if t.pnl >= 0 else f"-${abs(t.pnl):.2f}"
        open_time = t.entry_timestamp[5:16].replace("T", " ")  # MM-DD HH:MM
        close_time = t.exit_timestamp[5:16].replace("T", " ")

        print(f"  {i:<3} {t.direction:<5} ${t.reference_price:>10,.2f} {open_time:<17} {close_time:<17} {result_short:<12} {pnl_str:>10}")

    print()


def print_trade_log(trades: list[Trade]) -> None:
    """Print formatted detailed trade log."""
    print("\n" + "=" * 80)
    print("TRADE DETAILS")
    print("=" * 80)

    if not trades:
        print("\n  No trades executed.")
        return

    for i, t in enumerate(trades, 1):
        pnl_sign = "+" if t.pnl >= 0 else ""
        result_icon = "WIN" if t.pnl >= 0 else "LOSS"

        # Price source indicators
        entry_src = f"[{t.entry_source}]" if hasattr(t, 'entry_source') else ""
        exit_src = f"[{t.exit_source}]" if hasattr(t, 'exit_source') else ""

        print(f"\n  [{i}] {t.direction} @ barrier ${t.reference_price:,.2f}")
        print(f"      Entry:    ${t.entry_price:.4f} {entry_src}  ({t.entry_timestamp[:19].replace('T', ' ')})")
        print(f"      Exit:     ${t.exit_price:.4f} {exit_src}  ({t.exit_timestamp[:19].replace('T', ' ')})")
        print(f"      Shares:   {t.shares:.4f}")
        print(f"      Cost:     ${t.cost_basis:.2f}")
        print(f"      Proceeds: ${t.proceeds:.2f}")
        print(f"      P&L:      {pnl_sign}${t.pnl:.2f} ({pnl_sign}{t.pnl_pct:.1f}%)  [{result_icon}]")
        print(f"      Result:   {t.result}")
        if t.final_spot is not None:
            print(f"      Final BTC: ${t.final_spot:,.2f}")


def get_data_stats(df: list[dict]) -> dict:
    """Compute statistics about the data."""
    if not df:
        return {"start": None, "end": None, "duration_hours": 0, "num_markets": 0}

    timestamps = [row["timestamp"] for row in df]
    barriers = set(row["reference_price"] for row in df)

    start_ts = min(timestamps)
    end_ts = max(timestamps)

    # Parse timestamps to compute duration
    start_dt = datetime.fromisoformat(start_ts.replace("Z", "+00:00"))
    end_dt = datetime.fromisoformat(end_ts.replace("Z", "+00:00"))
    duration = (end_dt - start_dt).total_seconds() / 3600

    return {
        "start": start_ts,
        "end": end_ts,
        "duration_hours": duration,
        "num_markets": len(barriers),
    }


def print_summary(
    trades: list[Trade],
    initial_capital: float,
    final_capital: float,
    edge_up: float,
    edge_down: float,
    tp_pct: float,
    friction: float,
    data_stats: dict,
    latency_minutes: float = 0.0,
    trail_activation: float = 0.20,
    trail_distance: float = 0.15,
    min_model_prob: float = 0.0,
    alpha: float | None = None,
    floor: float | None = None,
    alpha_up: float | None = None,
    alpha_down: float | None = None,
    floor_up: float | None = None,
    floor_down: float | None = None,
    stop_loss_pct: float = 0.0,
) -> None:
    """Print summary statistics."""
    print("\n" + "=" * 80)
    print("BACKTEST SUMMARY")
    print("=" * 80)

    # Data time window
    print("\n  DATA")
    print("  " + "-" * 50)
    if data_stats["start"]:
        start_str = data_stats["start"][:19].replace("T", " ")
        end_str = data_stats["end"][:19].replace("T", " ")
        duration = data_stats["duration_hours"]
        print(f"  Start:       {start_str} UTC")
        print(f"  End:         {end_str} UTC")
        print(f"  Duration:    {duration:.1f} hours")
        print(f"  Markets:     {data_stats['num_markets']}")

    print("\n  PARAMETERS")
    print("  " + "-" * 50)
    # Resolve display values for per-direction params
    _a_up = alpha_up if alpha_up is not None else alpha
    _a_down = alpha_down if alpha_down is not None else alpha
    _f_up = floor_up if floor_up is not None else floor
    _f_down = floor_down if floor_down is not None else floor
    if _a_up is not None:
        if _a_up == _a_down and _f_up == _f_down:
            print(f"  Edge mode:   continuous (alpha={_a_up}, floor={_f_up:.0%})")
        else:
            print(f"  Edge mode:   continuous (UP: alpha={_a_up} floor={_f_up:.0%} | DOWN: alpha={_a_down} floor={_f_down:.0%})")
    else:
        print(f"  Edge UP:     >= {edge_up}x")
        print(f"  Edge DOWN:   >= {edge_down}x")
        if min_model_prob > 0:
            print(f"  Min Model P: >= {min_model_prob * 100:.0f}%")
    print(f"  Take Profit: {tp_pct * 100:.0f}%")
    print(f"  Trail Act:   {trail_activation * 100:.0f}% (arm trailing stop)")
    print(f"  Trail Dist:  {trail_distance * 100:.0f}pp (from peak)")
    if stop_loss_pct > 0:
        print(f"  Stop Loss:   {stop_loss_pct * 100:.0f}% below entry")
    print(f"  Friction:    {friction * 100:.1f}% per side")
    print(f"  Latency:     {latency_minutes:.0f} min")
    print(f"  Capital:     ${initial_capital:.2f}")

    if not trades:
        print("\n  No trades executed.")
        return

    total_pnl = final_capital - initial_capital
    total_return_pct = (total_pnl / initial_capital) * 100

    # Count results
    tp_count = sum(1 for t in trades if t.result == TradeResult.TP_FILLED.value)
    trail_count = sum(1 for t in trades if t.result == TradeResult.TRAILING_STOP.value)
    sl_count = sum(1 for t in trades if t.result == TradeResult.STOP_LOSS.value)
    win_expiry = sum(1 for t in trades if t.result == TradeResult.WIN_EXPIRY.value)
    loss_expiry = sum(1 for t in trades if t.result == TradeResult.LOSS_EXPIRY.value)
    wins = sum(1 for t in trades if t.pnl > 0)
    losses = sum(1 for t in trades if t.pnl <= 0)
    win_rate = (wins / len(trades)) * 100 if trades else 0

    # P&L stats
    pnls = [t.pnl for t in trades]
    avg_pnl = sum(pnls) / len(pnls)
    winning_pnls = [p for p in pnls if p > 0]
    losing_pnls = [p for p in pnls if p < 0]
    avg_win = sum(winning_pnls) / len(winning_pnls) if winning_pnls else 0
    avg_loss = sum(losing_pnls) / len(losing_pnls) if losing_pnls else 0

    print("\n  RESULTS")
    print("  " + "-" * 50)
    print(f"  Total trades:      {len(trades)}")
    print(f"  TP filled:         {tp_count}")
    print(f"  Trailing stop:     {trail_count}")
    if sl_count > 0:
        print(f"  Stop loss:         {sl_count}")
    print(f"  Win at expiry:     {win_expiry}")
    print(f"  Loss at expiry:    {loss_expiry}")
    print(f"  Win rate:          {win_rate:.1f}%")

    print("\n  P&L")
    print("  " + "-" * 50)
    pnl_sign = "+" if total_pnl >= 0 else ""
    print(f"  Total P&L:         {pnl_sign}${total_pnl:.2f} ({pnl_sign}{total_return_pct:.1f}%)")
    print(f"  Final capital:     ${final_capital:.2f}")
    print(f"  Avg P&L/trade:     ${avg_pnl:.2f}")
    if winning_pnls:
        print(f"  Avg win:           +${avg_win:.2f}")
    if losing_pnls:
        print(f"  Avg loss:          ${avg_loss:.2f}")

    print("\n" + "=" * 80)


def save_trades_csv(trades: list[Trade], output_path: str) -> None:
    """Save trades to CSV file."""
    with open(output_path, "w", newline="") as f:
        fieldnames = [
            "direction",
            "entry_price",
            "exit_price",
            "shares",
            "cost_basis",
            "proceeds",
            "pnl",
            "pnl_pct",
            "result",
            "entry_timestamp",
            "exit_timestamp",
            "reference_price",
            "final_spot",
            "entry_source",
            "exit_source",
            "model_prob",
            "market_prob",
            "edge",
            "spot_at_entry",
            "time_remaining_hours",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for t in trades:
            writer.writerow(
                {
                    "direction": t.direction,
                    "entry_price": f"{t.entry_price:.4f}",
                    "exit_price": f"{t.exit_price:.4f}",
                    "shares": f"{t.shares:.4f}",
                    "cost_basis": f"{t.cost_basis:.2f}",
                    "proceeds": f"{t.proceeds:.2f}",
                    "pnl": f"{t.pnl:.2f}",
                    "pnl_pct": f"{t.pnl_pct:.1f}",
                    "result": t.result,
                    "entry_timestamp": t.entry_timestamp,
                    "exit_timestamp": t.exit_timestamp,
                    "reference_price": f"{t.reference_price:.2f}",
                    "final_spot": f"{t.final_spot:.2f}" if t.final_spot else "",
                    "entry_source": t.entry_source,
                    "exit_source": t.exit_source,
                    "model_prob": f"{t.model_prob:.4f}",
                    "market_prob": f"{t.market_prob:.4f}",
                    "edge": f"{t.edge:.4f}",
                    "spot_at_entry": f"{t.spot_at_entry:.2f}",
                    "time_remaining_hours": f"{t.time_remaining_hours:.2f}",
                }
            )
    print(f"\nTrades saved to: {output_path}")


def load_config() -> dict:
    """Return empty config — trading config no longer used."""
    return {}


def main():
    # Load config for defaults
    config = load_config()

    parser = argparse.ArgumentParser(
        description="Backtest trading strategy against historical probabilities data"
    )
    parser.add_argument(
        "--csv",
        type=str,
        default="data_collector/results/probabilities.csv",
        help="Path to probabilities.csv (default: data_collector/results/probabilities.csv)",
    )
    parser.add_argument(
        "--edge-up",
        type=float,
        default=2.0,
        help="Min edge for UP entry (default: 2.0)",
    )
    parser.add_argument(
        "--edge-down",
        type=float,
        default=1.5,
        help="Min edge for DOWN entry (default: 1.5)",
    )
    parser.add_argument(
        "--tp",
        type=float,
        default=0.25,
        help="Take profit percentage (default: 0.25 = 25%%)",
    )
    parser.add_argument(
        "--friction",
        type=float,
        default=0.015,
        help="Fee + spread per side (default: 0.015 = 1.5%%)",
    )
    parser.add_argument(
        "--capital",
        type=float,
        default=100.0,
        help="Starting capital (default: 100.0)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to save trades CSV",
    )
    parser.add_argument(
        "--latency",
        type=float,
        default=2.0,
        help="Delay in minutes between signal and entry (default: 2)",
    )
    parser.add_argument(
        "--orderbook",
        type=str,
        default="data_collector/results/orderbook.csv",
        help="Path to orderbook.csv (default: data_collector/results/orderbook.csv)",
    )
    parser.add_argument(
        "--no-orderbook",
        action="store_true",
        help="Disable orderbook price lookup (use probability fallback only)",
    )
    parser.add_argument(
        "--trail-activation",
        type=float,
        default=0.20,
        help="Profit %% to activate trailing stop (default: 0.20 = 20%%)",
    )
    parser.add_argument(
        "--trail-distance",
        type=float,
        default=0.15,
        help="Trail distance in pp from peak (default: 0.15 = 15pp)",
    )
    parser.add_argument(
        "--min-time-remaining",
        type=float,
        default=config.get("min_time_remaining_hours", 2.0),
        help="Min hours before expiry to allow entry (default: 2.0)",
    )
    parser.add_argument(
        "--min-model-prob",
        type=float,
        default=config.get("min_model_prob", 0.0),
        help="Min model probability (0-1) to enter (default: 0.0, legacy mode)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=None,
        help="Edge curve exponent for both directions (shorthand, overridden by --alpha-up/--alpha-down).",
    )
    parser.add_argument(
        "--floor",
        type=float,
        default=None,
        help="Min model probability floor for both directions (shorthand, overridden by --floor-up/--floor-down).",
    )
    parser.add_argument(
        "--alpha-up",
        type=float,
        default=None,
        help="Edge curve exponent for UP bets. Overrides --alpha for UP.",
    )
    parser.add_argument(
        "--alpha-down",
        type=float,
        default=None,
        help="Edge curve exponent for DOWN bets. Overrides --alpha for DOWN.",
    )
    parser.add_argument(
        "--floor-up",
        type=float,
        default=None,
        help="Min model probability floor for UP bets. Overrides --floor for UP.",
    )
    parser.add_argument(
        "--floor-down",
        type=float,
        default=None,
        help="Min model probability floor for DOWN bets. Overrides --floor for DOWN.",
    )
    parser.add_argument(
        "--stop-loss",
        type=float,
        default=0.0,
        help="Stop-loss percentage below entry (0.20 = exit at 20%% loss, 0 = disabled)",
    )

    args = parser.parse_args()

    # Validate trailing stop params
    if args.trail_activation >= args.tp:
        print(f"Error: --trail-activation ({args.trail_activation}) must be less than --tp ({args.tp})")
        return 1
    if args.trail_distance >= args.trail_activation:
        print(f"Error: --trail-distance ({args.trail_distance}) must be less than --trail-activation ({args.trail_activation})")
        return 1

    # Resolve CSV path
    csv_path = Path(args.csv)
    if not csv_path.is_absolute():
        # Relative to project root
        project_root = Path(__file__).parent.parent
        csv_path = project_root / args.csv

    if not csv_path.exists():
        print(f"Error: CSV file not found: {csv_path}")
        return 1

    print(f"Loading data from: {csv_path}")
    df = load_csv(str(csv_path))
    print(f"Loaded {len(df)} rows")

    # Load orderbook if enabled
    orderbook = None
    if not args.no_orderbook:
        orderbook_path = Path(args.orderbook)
        if not orderbook_path.is_absolute():
            project_root = Path(__file__).parent.parent
            orderbook_path = project_root / args.orderbook

        if orderbook_path.exists():
            orderbook = load_orderbook_csv(str(orderbook_path))
            ob_count = sum(len(rows) for rows in orderbook.values())
            if ob_count > 0:
                up_count = sum(len(rows) for (b, d), rows in orderbook.items() if d == "UP")
                down_count = sum(len(rows) for (b, d), rows in orderbook.items() if d == "DOWN")
                print(f"Loaded orderbook: {ob_count} entries (UP: {up_count}, DOWN: {down_count})")
            else:
                print("Orderbook file empty or invalid, using probability fallback")
                orderbook = None
        else:
            print(f"Orderbook not found at {orderbook_path}, using probability fallback")

    # Run backtest
    trades, final_capital = run_backtest(
        df,
        edge_up=args.edge_up,
        edge_down=args.edge_down,
        tp_pct=args.tp,
        friction=args.friction,
        capital=args.capital,
        latency_minutes=args.latency,
        orderbook=orderbook,
        trail_activation=args.trail_activation,
        trail_distance=args.trail_distance,
        min_time_remaining_hours=args.min_time_remaining,
        min_model_prob=args.min_model_prob,
        alpha=args.alpha,
        floor=args.floor,
        alpha_up=args.alpha_up,
        alpha_down=args.alpha_down,
        floor_up=args.floor_up,
        floor_down=args.floor_down,
        stop_loss_pct=args.stop_loss,
    )

    # Get data statistics
    data_stats = get_data_stats(df)

    # Print results
    print_trades_table(trades)
    print_trade_log(trades)
    print_summary(
        trades,
        initial_capital=args.capital,
        final_capital=final_capital,
        edge_up=args.edge_up,
        edge_down=args.edge_down,
        tp_pct=args.tp,
        friction=args.friction,
        data_stats=data_stats,
        latency_minutes=args.latency,
        trail_activation=args.trail_activation,
        trail_distance=args.trail_distance,
        min_model_prob=args.min_model_prob,
        alpha=args.alpha,
        floor=args.floor,
        alpha_up=args.alpha_up,
        alpha_down=args.alpha_down,
        floor_up=args.floor_up,
        floor_down=args.floor_down,
        stop_loss_pct=args.stop_loss,
    )

    # Save trades if output specified
    if args.output:
        save_trades_csv(trades, args.output)

    return 0


if __name__ == "__main__":
    exit(main())
