"""Runtime-path tests for scripts.backtest.

These tests target the same `run_backtest()` execution path used by the
backtesting CLI, rather than re-implementing trade logic in tests.
"""

from scripts.backtest import (
    PRICE_SOURCE_FALLBACK,
    PRICE_SOURCE_ORDERBOOK,
    TradeResult,
    run_backtest,
)


def _row(
    timestamp: str,
    *,
    barrier: float = 100_000.0,
    time_remaining_hours: float = 4.0,
    spot_price: float = 100_000.0,
    edge_up: float = 0.0,
    edge_down: float = 0.0,
    model_prob_up: float = 60.0,
    model_prob_down: float = 40.0,
    poly_prob_up: float = 40.0,
    poly_prob_down: float = 60.0,
) -> dict:
    return {
        "timestamp": timestamp,
        "barrier_price": barrier,
        "time_remaining_hours": time_remaining_hours,
        "spot_price": spot_price,
        "model_prob_up": model_prob_up,
        "model_prob_down": model_prob_down,
        "poly_prob_up": poly_prob_up,
        "poly_prob_down": poly_prob_down,
        "edge_up": edge_up,
        "edge_down": edge_down,
    }


def _ob_row(timestamp: str, bid: float, ask: float) -> dict:
    return {
        "timestamp": timestamp,
        "best_bid": bid,
        "best_ask": ask,
        "spread": ask - bid,
    }


def test_run_backtest_executes_up_entry_and_tp_via_orderbook():
    rows = [
        _row("2026-01-01T00:00:00Z", edge_up=3.0, poly_prob_up=40.0),
        _row("2026-01-01T00:05:00Z", edge_up=0.0, poly_prob_up=41.0),
    ]
    barrier = 100_000.0
    orderbook = {
        (barrier, "UP"): [
            _ob_row("2026-01-01T00:00:00Z", bid=0.39, ask=0.40),
            _ob_row("2026-01-01T00:05:00Z", bid=0.60, ask=0.61),
        ]
    }

    trades, final_capital = run_backtest(
        rows,
        edge_up=2.0,
        edge_down=1.25,
        tp_pct=0.25,
        friction=0.0,
        capital=100.0,
        order_size_pct=0.10,
        orderbook=orderbook,
        trail_activation=0.8,
        trail_distance=0.2,
    )

    assert len(trades) == 1
    trade = trades[0]
    assert trade.direction == "UP"
    assert trade.result == TradeResult.TP_FILLED.value
    assert trade.entry_source == PRICE_SOURCE_ORDERBOOK
    assert trade.exit_source == PRICE_SOURCE_ORDERBOOK
    assert trade.entry_price == 0.40
    assert trade.exit_price == 0.50  # TP target fill
    assert round(trade.pnl, 2) == 2.50
    assert round(final_capital, 2) == 102.50


def test_run_backtest_trailing_stop_uses_orderbook_peak_and_drop():
    rows = [
        _row("2026-01-01T00:00:00Z", edge_up=3.0, poly_prob_up=40.0),
        _row("2026-01-01T00:03:00Z", edge_up=0.0, poly_prob_up=50.0),
        _row("2026-01-01T00:06:00Z", edge_up=0.0, poly_prob_up=46.0),
    ]
    barrier = 100_000.0
    orderbook = {
        (barrier, "UP"): [
            _ob_row("2026-01-01T00:00:00Z", bid=0.39, ask=0.40),
            _ob_row("2026-01-01T00:03:00Z", bid=0.52, ask=0.53),
            _ob_row("2026-01-01T00:06:00Z", bid=0.47, ask=0.48),
        ]
    }

    trades, final_capital = run_backtest(
        rows,
        edge_up=2.0,
        edge_down=1.25,
        tp_pct=0.60,  # keep TP far away so trailing stop drives the exit
        friction=0.0,
        capital=100.0,
        order_size_pct=0.10,
        orderbook=orderbook,
        trail_activation=0.20,
        trail_distance=0.10,
    )

    assert len(trades) == 1
    trade = trades[0]
    assert trade.result == TradeResult.TRAILING_STOP.value
    assert trade.exit_source == PRICE_SOURCE_ORDERBOOK
    assert round(trade.exit_price, 2) == 0.48  # 0.52 - (0.40 * 0.10)
    assert round(trade.pnl, 2) == 2.00
    assert round(final_capital, 2) == 102.00


def test_run_backtest_expiry_settlement_uses_terminal_spot():
    rows = [
        _row("2026-01-01T00:00:00Z", edge_up=3.0, poly_prob_up=40.0),
        _row(
            "2026-01-01T01:00:00Z",
            time_remaining_hours=0.0,
            spot_price=105_000.0,
            poly_prob_up=39.0,
        ),
    ]

    trades, final_capital = run_backtest(
        rows,
        edge_up=2.0,
        edge_down=1.25,
        tp_pct=0.60,
        friction=0.0,
        capital=100.0,
        order_size_pct=0.10,
        orderbook=None,
        trail_activation=0.20,
        trail_distance=0.10,
    )

    assert len(trades) == 1
    trade = trades[0]
    assert trade.result == TradeResult.WIN_EXPIRY.value
    assert trade.entry_source == PRICE_SOURCE_FALLBACK
    assert trade.exit_source == PRICE_SOURCE_FALLBACK
    assert trade.exit_price == 1.0
    assert trade.final_spot == 105_000.0
    assert round(trade.pnl, 2) == 15.00
    assert round(final_capital, 2) == 115.00


def test_run_backtest_latency_delays_entry_until_pending_signal_matures():
    rows = [
        _row("2026-01-01T00:00:00Z", edge_up=3.0, poly_prob_up=40.0),
        _row("2026-01-01T00:01:00Z", edge_up=0.0, poly_prob_up=45.0),
        _row("2026-01-01T00:03:00Z", edge_up=0.0, poly_prob_up=50.0),
        _row("2026-01-01T01:00:00Z", time_remaining_hours=0.0, spot_price=110_000.0, poly_prob_up=48.0),
    ]

    trades, _ = run_backtest(
        rows,
        edge_up=2.0,
        edge_down=1.25,
        tp_pct=0.60,
        friction=0.0,
        capital=100.0,
        order_size_pct=0.10,
        latency_minutes=2.0,
        orderbook=None,
        trail_activation=0.20,
        trail_distance=0.10,
    )

    assert len(trades) == 1
    trade = trades[0]
    assert trade.entry_timestamp == "2026-01-01T00:03:00Z"
    assert trade.entry_price == 0.50  # fallback to poly_prob_up at delayed entry row


def test_run_backtest_falls_back_when_orderbook_key_is_missing():
    rows = [
        _row("2026-01-01T00:00:00Z", edge_up=3.0, barrier=101_000.0, poly_prob_up=42.0),
        _row("2026-01-01T01:00:00Z", barrier=101_000.0, time_remaining_hours=0.0, spot_price=102_000.0, poly_prob_up=41.0),
    ]
    # Wrong barrier key on purpose: should force fallback pricing.
    orderbook = {
        (100_000.0, "UP"): [
            _ob_row("2026-01-01T00:00:00Z", bid=0.30, ask=0.31),
        ]
    }

    trades, _ = run_backtest(
        rows,
        edge_up=2.0,
        edge_down=1.25,
        tp_pct=0.60,
        friction=0.0,
        capital=100.0,
        order_size_pct=0.10,
        orderbook=orderbook,
        trail_activation=0.20,
        trail_distance=0.10,
    )

    assert len(trades) == 1
    assert trades[0].entry_source == PRICE_SOURCE_FALLBACK


def test_run_backtest_executes_down_path_and_loss_expiry():
    rows = [
        _row("2026-01-01T00:00:00Z", edge_down=2.0, poly_prob_down=45.0),
        _row(
            "2026-01-01T01:00:00Z",
            time_remaining_hours=0.0,
            spot_price=110_000.0,  # above barrier -> DOWN loses
            edge_down=0.0,
            poly_prob_down=44.0,
        ),
    ]

    trades, final_capital = run_backtest(
        rows,
        edge_up=2.0,
        edge_down=1.25,
        tp_pct=0.60,
        friction=0.0,
        capital=100.0,
        order_size_pct=0.10,
        orderbook=None,
        trail_activation=0.20,
        trail_distance=0.10,
    )

    assert len(trades) == 1
    trade = trades[0]
    assert trade.direction == "DOWN"
    assert trade.result == TradeResult.LOSS_EXPIRY.value
    assert trade.exit_price == 0.0
    assert round(trade.pnl, 2) == -10.00
    assert round(final_capital, 2) == 90.00
