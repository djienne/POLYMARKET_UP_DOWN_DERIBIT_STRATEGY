"""Optimization script to find best backtest parameters.

Includes robustness features to address sensitivity to small data changes:
- Bootstrap confidence intervals for P&L estimates
- Leave-one-out cross-validation by market
- Risk-adjusted metrics (Sharpe ratio, profit factor)
- Parameter stability analysis
"""

import argparse
import json
import multiprocessing
import random
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

from backtest import load_csv, load_orderbook_csv, run_backtest, get_data_stats, group_markets, get_orderbook_prices

sys.path.insert(0, str(Path(__file__).parent.parent))


@dataclass
class RobustResult:
    """Extended result with robustness metrics."""
    edge_up: float  # Legacy (unused when alpha is set)
    edge_down: float  # Legacy (unused when alpha is set)
    tp_pct: float
    trades: int
    wins: int
    losses: int
    win_rate: float
    pnl: float
    pnl_pct: float
    final_capital: float
    # Continuous edge parameters (per-direction)
    alpha_up: Optional[float] = None
    alpha_down: Optional[float] = None
    floor_up: Optional[float] = None
    floor_down: Optional[float] = None
    # Robustness metrics
    sharpe_ratio: Optional[float] = None
    profit_factor: Optional[float] = None
    avg_pnl_per_trade: Optional[float] = None
    pnl_std: Optional[float] = None
    # Bootstrap metrics
    bootstrap_mean: Optional[float] = None
    bootstrap_std: Optional[float] = None
    bootstrap_5th: Optional[float] = None
    bootstrap_95th: Optional[float] = None
    # Cross-validation metrics
    cv_mean_pnl: Optional[float] = None
    cv_std_pnl: Optional[float] = None
    cv_worst_pnl: Optional[float] = None
    cv_markets_profitable: Optional[int] = None
    cv_total_markets: Optional[int] = None


def _annualize_sharpe(raw_sharpe: float, n_obs: int, duration_days: float) -> float:
    """Scale a per-observation Sharpe ratio to annualised (365-day) Sharpe.

    annualised = raw * sqrt(n_obs * 365 / duration_days)
    """
    if duration_days <= 0:
        return raw_sharpe
    return raw_sharpe * (n_obs * 365 / duration_days) ** 0.5


def compute_sharpe_ratio(trade_pnls: list[float], risk_free_rate: float = 0.0,
                         duration_days: float = 0.0) -> Optional[float]:
    """Compute Sharpe ratio from trade P&Ls (annualised when duration_days > 0)."""
    if len(trade_pnls) < 2:
        return None
    mean_pnl = sum(trade_pnls) / len(trade_pnls)
    variance = sum((p - mean_pnl) ** 2 for p in trade_pnls) / (len(trade_pnls) - 1)
    std_pnl = variance ** 0.5
    if std_pnl == 0:
        return None
    raw = (mean_pnl - risk_free_rate) / std_pnl
    if duration_days > 0:
        return _annualize_sharpe(raw, len(trade_pnls), duration_days)
    return raw


def compute_mtm_sharpe(pnl_values: list[float], duration_days: float = 0.0) -> Optional[float]:
    """Compute Sharpe ratio from consecutive changes in an MTM equity curve.

    This captures path-dependent risk (unrealized drawdowns) that per-trade
    Sharpe misses.  Falls back to None when there are fewer than 3 observations.
    Returns annualised Sharpe when duration_days > 0.
    """
    if len(pnl_values) < 3:
        return None
    changes = [pnl_values[i + 1] - pnl_values[i] for i in range(len(pnl_values) - 1)]
    n = len(changes)
    mean_chg = sum(changes) / n
    variance = sum((c - mean_chg) ** 2 for c in changes) / (n - 1)
    std_chg = variance ** 0.5
    if std_chg == 0:
        return None
    raw = mean_chg / std_chg
    if duration_days > 0:
        return _annualize_sharpe(raw, n, duration_days)
    return raw


def compute_profit_factor(trade_pnls: list[float]) -> Optional[float]:
    """Compute profit factor (gross profit / gross loss)."""
    gross_profit = sum(p for p in trade_pnls if p > 0)
    gross_loss = abs(sum(p for p in trade_pnls if p < 0))
    if gross_loss == 0:
        return float('inf') if gross_profit > 0 else None
    return gross_profit / gross_loss


# --- Multiprocessing worker globals and helpers ---
_worker_df = None
_worker_orderbook = None
_worker_duration_days = 0.0
_worker_order_size_pct = 0.05


def _worker_init(df, orderbook, order_size_pct=0.05):
    """Pool initializer: store data in worker globals (called once per worker process)."""
    global _worker_df, _worker_orderbook, _worker_duration_days, _worker_order_size_pct
    _worker_df = df
    _worker_orderbook = orderbook
    _worker_order_size_pct = order_size_pct
    stats = get_data_stats(df)
    _worker_duration_days = stats["duration_hours"] / 24 if stats.get("duration_hours") else 0.0


def _run_single_combo(args):
    """Run a single parameter combo and return a plain dict of results.

    Must be a top-level function so it's picklable for multiprocessing.
    Metrics are computed here so only plain dicts cross the process boundary.

    Args tuple layout:
        (edge_up, edge_down, tp_pct, trail_activation, trail_distance,
         friction, capital, latency_minutes, min_time_remaining_hours,
         min_model_prob, alpha_up, alpha_down, floor_up, floor_down,
         stop_loss_pct)
    """
    (edge_up, edge_down, tp_pct, trail_activation, trail_distance,
     friction, capital, latency_minutes, min_time_remaining_hours,
     min_model_prob, alpha_up, alpha_down, floor_up_val, floor_down_val,
     stop_loss_pct) = args

    trades, final_capital = run_backtest(
        _worker_df,
        edge_up=edge_up,
        edge_down=edge_down,
        tp_pct=tp_pct,
        friction=friction,
        capital=capital,
        order_size_pct=_worker_order_size_pct,
        latency_minutes=latency_minutes,
        orderbook=_worker_orderbook,
        trail_activation=trail_activation,
        trail_distance=trail_distance,
        min_time_remaining_hours=min_time_remaining_hours,
        min_model_prob=min_model_prob,
        alpha_up=alpha_up,
        alpha_down=alpha_down,
        floor_up=floor_up_val,
        floor_down=floor_down_val,
        stop_loss_pct=stop_loss_pct,
    )

    total_pnl = final_capital - capital
    pnl_pct = (total_pnl / capital) * 100
    num_trades = len(trades)
    wins = sum(1 for t in trades if t.pnl > 0)
    losses = sum(1 for t in trades if t.pnl < 0)
    win_rate = (wins / num_trades * 100) if num_trades > 0 else 0

    trade_pnls = [t.pnl for t in trades]
    profit_factor = compute_profit_factor(trade_pnls) if trade_pnls else None
    avg_pnl = sum(trade_pnls) / len(trade_pnls) if trade_pnls else None
    pnl_std = None
    if len(trade_pnls) >= 2:
        mean_p = sum(trade_pnls) / len(trade_pnls)
        pnl_std = (sum((p - mean_p)**2 for p in trade_pnls) / (len(trade_pnls) - 1)) ** 0.5

    # Compute max drawdown and MTM Sharpe from equity curve
    max_dd_pct = 0.0
    sharpe = None
    if trades:
        _, mtm_pnl = build_equity_curve(trades, _worker_df, friction, _worker_orderbook)
        if mtm_pnl:
            dd_info = compute_max_drawdown(mtm_pnl, capital=capital)
            max_dd_pct = dd_info["max_dd_pct"]
            sharpe = compute_mtm_sharpe(mtm_pnl, _worker_duration_days)
    # Fallback to per-trade Sharpe when equity curve is too short
    if sharpe is None and trade_pnls:
        sharpe = compute_sharpe_ratio(trade_pnls, duration_days=_worker_duration_days)

    return {
        "edge_up": edge_up,
        "edge_down": edge_down,
        "tp_pct": tp_pct,
        "trail_activation": trail_activation,
        "trail_distance": trail_distance,
        "min_model_prob": min_model_prob,
        "alpha_up": alpha_up,
        "alpha_down": alpha_down,
        "floor_up": floor_up_val,
        "floor_down": floor_down_val,
        "stop_loss_pct": stop_loss_pct,
        "trades": num_trades,
        "wins": wins,
        "losses": losses,
        "win_rate": win_rate,
        "pnl": total_pnl,
        "pnl_pct": pnl_pct,
        "final_capital": final_capital,
        "sharpe_ratio": sharpe,
        "profit_factor": profit_factor,
        "avg_pnl_per_trade": avg_pnl,
        "pnl_std": pnl_std,
        "max_dd_pct": max_dd_pct,
    }


def bootstrap_pnl(
    df: list[dict],
    params: dict,
    n_bootstrap: int = 100,
    friction: float = 0.015,
    capital: float = 100.0,
    latency_minutes: float = 0.0,
    orderbook: dict[tuple[float, str], list[dict]] | None = None,
    trail_activation: float = 0.20,
    trail_distance: float = 0.15,
    min_time_remaining_hours: float = 2.0,
    min_model_prob: float = 0.0,
    order_size_pct: float = 0.05,
) -> dict:
    """
    Bootstrap confidence intervals using block bootstrap on per-market P&L.

    Runs backtest once per market to get per-market P&L, then resamples
    the P&L scalars with replacement. Much faster and avoids barrier
    collision issues from merging rows of different markets.

    Returns dict with mean, std, 5th, and 95th percentile P&L.
    """
    # Group by market (barrier_price)
    markets = group_markets(df)
    market_list = list(markets.values())
    n_markets = len(market_list)

    if n_markets < 2:
        return {"mean": None, "std": None, "p5": None, "p95": None}

    # Step 1: Run backtest once per market to get per-market P&L
    per_market_pnls = []
    for market_rows in market_list:
        trades, final_cap = run_backtest(
            market_rows,
            edge_up=params["edge_up"],
            edge_down=params["edge_down"],
            tp_pct=params["tp_pct"],
            friction=friction,
            capital=capital,
            order_size_pct=order_size_pct,
            latency_minutes=latency_minutes,
            orderbook=orderbook,
            trail_activation=trail_activation,
            trail_distance=trail_distance,
            min_time_remaining_hours=min_time_remaining_hours,
            min_model_prob=min_model_prob,
            alpha_up=params.get("alpha_up"),
            alpha_down=params.get("alpha_down"),
            floor_up=params.get("floor_up"),
            floor_down=params.get("floor_down"),
            stop_loss_pct=params.get("stop_loss_pct", 0.0),
        )
        per_market_pnls.append(final_cap - capital)

    # Step 2: Bootstrap by resampling per-market P&L scalars
    bootstrap_totals = []
    for _ in range(n_bootstrap):
        sampled_pnls = random.choices(per_market_pnls, k=n_markets)
        bootstrap_totals.append(sum(sampled_pnls))

    bootstrap_totals.sort()
    n = len(bootstrap_totals)
    mean = sum(bootstrap_totals) / n
    return {
        "mean": mean,
        "std": (sum((p - mean)**2 for p in bootstrap_totals) / (n-1)) ** 0.5,
        "p5": bootstrap_totals[int(n * 0.05)],
        "p95": bootstrap_totals[int(n * 0.95)],
    }


def per_market_analysis(
    df: list[dict],
    params: dict,
    friction: float = 0.015,
    capital: float = 100.0,
    latency_minutes: float = 0.0,
    orderbook: dict[tuple[float, str], list[dict]] | None = None,
    trail_activation: float = 0.20,
    trail_distance: float = 0.15,
    min_time_remaining_hours: float = 2.0,
    min_model_prob: float = 0.0,
    order_size_pct: float = 0.05,
) -> dict:
    """
    Per-market P&L analysis.

    Runs backtest on each market individually and reports P&L distribution.
    Returns dict with mean, std, worst P&L, and count of profitable markets.
    """
    # Group by market (barrier_price)
    markets = group_markets(df)
    market_barriers = list(markets.keys())
    n_markets = len(market_barriers)

    if n_markets < 2:
        return {"mean": None, "std": None, "worst": None, "profitable": None, "total": None}

    cv_pnls = []
    for held_out_barrier in market_barriers:
        # Create test set (held-out market only)
        test_df = markets[held_out_barrier]

        # Run backtest on held-out market
        trades, final_capital = run_backtest(
            test_df,
            edge_up=params["edge_up"],
            edge_down=params["edge_down"],
            tp_pct=params["tp_pct"],
            friction=friction,
            capital=capital,
            order_size_pct=order_size_pct,
            latency_minutes=latency_minutes,
            orderbook=orderbook,
            trail_activation=trail_activation,
            trail_distance=trail_distance,
            min_time_remaining_hours=min_time_remaining_hours,
            min_model_prob=min_model_prob,
            alpha_up=params.get("alpha_up"),
            alpha_down=params.get("alpha_down"),
            floor_up=params.get("floor_up"),
            floor_down=params.get("floor_down"),
            stop_loss_pct=params.get("stop_loss_pct", 0.0),
        )
        cv_pnls.append(final_capital - capital)

    n = len(cv_pnls)
    profitable = sum(1 for p in cv_pnls if p > 0)

    return {
        "mean": sum(cv_pnls) / n,
        "std": (sum((p - sum(cv_pnls)/n)**2 for p in cv_pnls) / max(n-1, 1)) ** 0.5,
        "worst": min(cv_pnls),
        "profitable": profitable,
        "total": n,
    }


def run_optimization(
    df: list[dict],
    friction: float = 0.01,
    capital: float = 100.0,
    edge_up_range: tuple = (1.1, 2.0, 0.1),
    edge_down_range: tuple = (1.1, 2.0, 0.1),
    tp_pct_range: tuple = (0.15, 0.35, 0.05),
    latency_minutes: float = 0.0,
    enable_bootstrap: bool = False,
    enable_cv: bool = False,
    n_bootstrap: int = 100,
    orderbook: dict[tuple[float, str], list[dict]] | None = None,
    trail_activation_values: list[float] | None = None,
    trail_distance_values: list[float] | None = None,
    workers: int = 4,
    min_time_remaining_hours: float = 2.0,
    alpha_up_values: list[float] | None = None,
    alpha_down_values: list[float] | None = None,
    floor_up_values: list[float] | None = None,
    floor_down_values: list[float] | None = None,
    stop_loss_values: list[float] | None = None,
    order_size_pct: float = 0.05,
) -> list[dict]:
    """
    Run optimization loop to find best thresholds.

    When alpha_up_values is provided, uses the continuous edge function and
    grid-searches over (alpha_up, alpha_down, floor_up, floor_down) instead
    of (edge_up, edge_down).

    Args:
        df: Data from CSV
        friction: Fee + spread per side
        capital: Starting capital
        edge_up_range: (min, max, step) for edge_up values (legacy mode)
        edge_down_range: (min, max, step) for edge_down values (legacy mode)
        tp_pct_range: (min, max, step) for tp_pct values
        latency_minutes: Delay between signal and entry
        enable_bootstrap: Run bootstrap confidence intervals (slower)
        enable_cv: Run leave-one-out cross-validation (slower)
        n_bootstrap: Number of bootstrap iterations
        orderbook: Optional orderbook data for realistic pricing
        trail_activation_values: List of trail activation values to grid search
        trail_distance_values: List of trail distance values to grid search
        alpha_up_values: List of alpha values for UP direction
        alpha_down_values: List of alpha values for DOWN direction
        floor_up_values: List of floor values for UP direction
        floor_down_values: List of floor values for DOWN direction

    Returns:
        List of results sorted by P&L descending
    """
    use_continuous = alpha_up_values is not None
    results = []

    # Generate parameter values from ranges
    def frange(start, stop, step):
        if step <= 0:
            raise ValueError(f"frange step must be positive, got {step}")
        values = []
        v = start
        while v <= stop + 0.0001:  # Small epsilon for float comparison
            values.append(round(v, 2))
            v += step
        return values

    tp_pct_values = frange(*tp_pct_range)
    act_values = trail_activation_values or [0.20]
    dist_values = trail_distance_values or [0.15]
    sl_values = stop_loss_values or [0.0]

    # Build list of all parameter combos (with validation filtering)
    all_args = []

    if use_continuous:
        au_values = alpha_up_values
        ad_values = alpha_down_values or alpha_up_values
        fu_values = floor_up_values or [0.65]
        fd_values = floor_down_values or [0.65]

        for alpha_up in au_values:
            for alpha_down in ad_values:
                for floor_up_val in fu_values:
                    for floor_down_val in fd_values:
                        for tp_pct in tp_pct_values:
                            for trail_act in act_values:
                                if trail_act >= tp_pct:
                                    continue
                                for trail_dist in dist_values:
                                    if trail_dist >= trail_act:
                                        continue
                                    for sl_pct in sl_values:
                                        all_args.append((
                                            0.0, 0.0, tp_pct, trail_act, trail_dist,
                                            friction, capital, latency_minutes, min_time_remaining_hours,
                                            0.0, alpha_up, alpha_down, floor_up_val, floor_down_val,
                                            sl_pct,
                                        ))

        total_combos = len(all_args)
        print(f"\nTesting {total_combos} parameter combinations (continuous edge mode)...")
        print(f"  Alpha UP:  {au_values}")
        print(f"  Alpha DN:  {ad_values}")
        print(f"  Floor UP:  {fu_values}")
        print(f"  Floor DN:  {fd_values}")
        if any(v > 0 for v in sl_values):
            print(f"  Stop Loss: {[f'{v*100:.0f}%' for v in sl_values]}")
    else:
        edge_up_values = frange(*edge_up_range)
        edge_down_values = frange(*edge_down_range)

        for edge_up in edge_up_values:
            for edge_down in edge_down_values:
                for tp_pct in tp_pct_values:
                    for trail_act in act_values:
                        if trail_act >= tp_pct:
                            continue  # Sanity: activation must be below TP
                        for trail_dist in dist_values:
                            if trail_dist >= trail_act:
                                continue  # Sanity: distance must be below activation
                            for sl_pct in sl_values:
                                all_args.append((
                                    edge_up, edge_down, tp_pct, trail_act, trail_dist,
                                    friction, capital, latency_minutes, min_time_remaining_hours,
                                    0.0, None, None, None, None,
                                    sl_pct,
                                ))

        total_combos = len(all_args)
        print(f"\nTesting {total_combos} parameter combinations (legacy edge mode)...")
        print(f"  Edge UP:   {edge_up_values}")
        print(f"  Edge DOWN: {edge_down_values}")

    print(f"  TP %:      {[f'{v*100:.0f}%' for v in tp_pct_values]}")
    print(f"  Trail Act: {[f'{v*100:.0f}%' for v in act_values]}")
    print(f"  Trail Dist:{[f'{v*100:.0f}pp' for v in dist_values]}")
    print(f"  Friction:  {friction*100:.1f}% per side")
    print(f"  Order Size:{order_size_pct*100:.0f}% of capital")
    print(f"  Latency:   {latency_minutes:.0f} min")

    if workers <= 1:
        # Sequential: set module globals directly, no multiprocessing overhead
        global _worker_df, _worker_orderbook, _worker_order_size_pct, _worker_duration_days
        _worker_df = df
        _worker_orderbook = orderbook
        _worker_order_size_pct = order_size_pct
        stats = get_data_stats(df)
        _worker_duration_days = stats["duration_hours"] / 24 if stats.get("duration_hours") else 0.0
        combo_num = 0
        for args in all_args:
            combo_num += 1
            if combo_num % 100 == 0:
                print(f"  Progress: {combo_num}/{total_combos}")
            results.append(_run_single_combo(args))
    else:
        # Parallel: distribute across worker processes
        chunksize = max(1, total_combos // (workers * 4))
        combo_num = 0
        with multiprocessing.Pool(
            workers, initializer=_worker_init, initargs=(df, orderbook, order_size_pct)
        ) as pool:
            for result in pool.imap_unordered(_run_single_combo, all_args, chunksize=chunksize):
                results.append(result)
                combo_num += 1
                if combo_num % 100 == 0:
                    print(f"  Progress: {combo_num}/{total_combos}")

    # Sort by P&L descending
    results.sort(key=lambda x: x["pnl"], reverse=True)

    # Compute bootstrap and CV for top results if enabled
    if enable_bootstrap or enable_cv:
        print("\nComputing robustness metrics for top results...")
        top_n_for_robust = min(20, len(results))

        for i, result in enumerate(results[:top_n_for_robust]):
            if result["trades"] == 0:
                continue

            params = {
                "edge_up": result["edge_up"],
                "edge_down": result["edge_down"],
                "tp_pct": result["tp_pct"],
                "alpha_up": result.get("alpha_up"),
                "alpha_down": result.get("alpha_down"),
                "floor_up": result.get("floor_up"),
                "floor_down": result.get("floor_down"),
                "stop_loss_pct": result.get("stop_loss_pct", 0.0),
            }

            if enable_bootstrap:
                bootstrap = bootstrap_pnl(
                    df, params, n_bootstrap=n_bootstrap,
                    friction=friction, capital=capital, latency_minutes=latency_minutes,
                    orderbook=orderbook,
                    trail_activation=result["trail_activation"],
                    trail_distance=result["trail_distance"],
                    min_time_remaining_hours=min_time_remaining_hours,
                    min_model_prob=result.get("min_model_prob", 0),
                    order_size_pct=order_size_pct,
                )
                result["bootstrap_mean"] = bootstrap["mean"]
                result["bootstrap_std"] = bootstrap["std"]
                result["bootstrap_5th"] = bootstrap["p5"]
                result["bootstrap_95th"] = bootstrap["p95"]

            if enable_cv:
                cv = per_market_analysis(
                    df, params,
                    friction=friction, capital=capital, latency_minutes=latency_minutes,
                    orderbook=orderbook,
                    trail_activation=result["trail_activation"],
                    trail_distance=result["trail_distance"],
                    min_time_remaining_hours=min_time_remaining_hours,
                    min_model_prob=result.get("min_model_prob", 0),
                    order_size_pct=order_size_pct,
                )
                result["cv_mean_pnl"] = cv["mean"]
                result["cv_std_pnl"] = cv["std"]
                result["cv_worst_pnl"] = cv["worst"]
                result["cv_markets_profitable"] = cv["profitable"]
                result["cv_total_markets"] = cv["total"]

            if (i + 1) % 5 == 0:
                print(f"  Processed {i + 1}/{top_n_for_robust}")

    return results


def analyze_parameter_stability(results: list[dict], min_trades: int = 3) -> dict:
    """
    Analyze which parameter ranges consistently perform well.

    Returns dict with stable ranges for each parameter.
    """
    # Filter to results with sufficient trades
    filtered = [r for r in results if r["trades"] >= min_trades]
    if not filtered:
        return {}

    # Get top 20% of results by P&L
    n_top = max(1, len(filtered) // 5)
    top_results = sorted(filtered, key=lambda x: x["pnl"], reverse=True)[:n_top]

    # Detect mode: continuous (alpha_up present) or legacy
    use_continuous = top_results[0].get("alpha_up") is not None

    tp_pcts = [r["tp_pct"] for r in top_results]
    trail_acts = [r["trail_activation"] for r in top_results]
    trail_dists = [r["trail_distance"] for r in top_results]

    def get_range_stats(values):
        if not values:
            return None
        return {
            "min": min(values),
            "max": max(values),
            "mean": sum(values) / len(values),
            "median": sorted(values)[len(values) // 2],
        }

    sl_pcts = [r.get("stop_loss_pct", 0.0) for r in top_results]

    result = {
        "tp_pct": get_range_stats(tp_pcts),
        "trail_activation": get_range_stats(trail_acts),
        "trail_distance": get_range_stats(trail_dists),
        "stop_loss_pct": get_range_stats(sl_pcts) if any(v > 0 for v in sl_pcts) else None,
        "n_top": n_top,
        "n_total": len(filtered),
        "use_continuous": use_continuous,
    }

    if use_continuous:
        result["alpha_up"] = get_range_stats([r["alpha_up"] for r in top_results])
        result["alpha_down"] = get_range_stats([r["alpha_down"] for r in top_results])
        result["floor_up"] = get_range_stats([r["floor_up"] for r in top_results])
        result["floor_down"] = get_range_stats([r["floor_down"] for r in top_results])
    else:
        result["edge_up"] = get_range_stats([r["edge_up"] for r in top_results])
        result["edge_down"] = get_range_stats([r["edge_down"] for r in top_results])

    return result


def print_stability_analysis(
    results: list[dict],
    min_trades: int = 3,
    df: list[dict] | None = None,
    friction: float = 0.015,
    capital: float = 100.0,
    latency_minutes: float = 0.0,
    orderbook: dict | None = None,
    min_time_remaining_hours: float = 2.0,
    order_size_pct: float = 0.05,
) -> None:
    """Print parameter stability analysis and backtest the suggested robust parameters."""
    stability = analyze_parameter_stability(results, min_trades)
    if not stability:
        return

    print("\n" + "=" * 80)
    print("PARAMETER STABILITY ANALYSIS")
    print("=" * 80)
    print(f"\n  Analyzing top {stability['n_top']} of {stability['n_total']} parameter combinations")
    print("  " + "-" * 60)

    use_continuous = stability.get("use_continuous", False)

    if use_continuous:
        for label, key in [("Alpha UP (curvature)", "alpha_up"), ("Alpha DOWN (curvature)", "alpha_down"),
                           ("Floor UP (min model prob)", "floor_up"), ("Floor DOWN (min model prob)", "floor_down")]:
            if stability.get(key):
                s = stability[key]
                print(f"\n  {label}:")
                print(f"    Range in top performers: {s['min']:.2f} - {s['max']:.2f}")
                print(f"    Mean: {s['mean']:.2f}, Median: {s['median']:.2f}")
                print(f"    Suggested: {s['median']:.2f}")
    else:
        if stability.get("edge_up"):
            s = stability["edge_up"]
            print(f"\n  Edge UP:")
            print(f"    Range in top performers: {s['min']:.2f} - {s['max']:.2f}")
            print(f"    Mean: {s['mean']:.2f}, Median: {s['median']:.2f}")
            print(f"    Suggested: {s['median']:.2f}")

        if stability.get("edge_down"):
            s = stability["edge_down"]
            print(f"\n  Edge DOWN:")
            print(f"    Range in top performers: {s['min']:.2f} - {s['max']:.2f}")
            print(f"    Mean: {s['mean']:.2f}, Median: {s['median']:.2f}")
            print(f"    Suggested: {s['median']:.2f}")

    if stability.get("tp_pct"):
        s = stability["tp_pct"]
        print(f"\n  Take Profit:")
        print(f"    Range in top performers: {s['min']*100:.0f}% - {s['max']*100:.0f}%")
        print(f"    Mean: {s['mean']*100:.0f}%, Median: {s['median']*100:.0f}%")
        print(f"    Suggested: {s['median']*100:.0f}%")

    if stability.get("trail_activation"):
        s = stability["trail_activation"]
        print(f"\n  Trail Activation:")
        print(f"    Range in top performers: {s['min']*100:.0f}% - {s['max']*100:.0f}%")
        print(f"    Mean: {s['mean']*100:.0f}%, Median: {s['median']*100:.0f}%")
        print(f"    Suggested: {s['median']*100:.0f}%")

    if stability.get("trail_distance"):
        s = stability["trail_distance"]
        print(f"\n  Trail Distance:")
        print(f"    Range in top performers: {s['min']*100:.0f}pp - {s['max']*100:.0f}pp")
        print(f"    Mean: {s['mean']*100:.0f}pp, Median: {s['median']*100:.0f}pp")
        print(f"    Suggested: {s['median']*100:.0f}pp")

    if stability.get("stop_loss_pct"):
        s = stability["stop_loss_pct"]
        print(f"\n  Stop Loss:")
        print(f"    Range in top performers: {s['min']*100:.0f}% - {s['max']*100:.0f}%")
        print(f"    Mean: {s['mean']*100:.0f}%, Median: {s['median']*100:.0f}%")
        print(f"    Suggested: {s['median']*100:.0f}%")

    # Suggest "robust" middle-of-the-road parameters
    print("\n  " + "-" * 60)
    print("  SUGGESTED ROBUST PARAMETERS (median of top performers):")
    robust_tp = stability['tp_pct']['median']
    robust_trail_act = stability['trail_activation']['median'] if stability.get("trail_activation") else 0.20
    robust_trail_dist = stability['trail_distance']['median'] if stability.get("trail_distance") else 0.15
    robust_sl = stability['stop_loss_pct']['median'] if stability.get("stop_loss_pct") else 0.0

    if use_continuous:
        robust_alpha_up = stability['alpha_up']['median']
        robust_alpha_down = stability['alpha_down']['median']
        robust_floor_up = stability['floor_up']['median']
        robust_floor_down = stability['floor_down']['median']
        cmd = (f"    python scripts/backtest.py --alpha-up {robust_alpha_up:.2f} "
               f"--alpha-down {robust_alpha_down:.2f} "
               f"--floor-up {robust_floor_up:.2f} --floor-down {robust_floor_down:.2f} "
               f"--tp {robust_tp:.2f}")
    else:
        robust_edge_up = stability['edge_up']['median']
        robust_edge_down = stability['edge_down']['median']
        cmd = (f"    python scripts/backtest.py --edge-up {robust_edge_up:.2f} "
               f"--edge-down {robust_edge_down:.2f} "
               f"--tp {robust_tp:.2f}")
    if stability.get("trail_activation") and stability.get("trail_distance"):
        cmd += (f" --trail-activation {robust_trail_act:.2f}"
                f" --trail-distance {robust_trail_dist:.2f}")
    if robust_sl > 0:
        cmd += f" --stop-loss {robust_sl:.2f}"
    print(cmd)

    # Run backtest with the robust parameters and show results
    if df is not None:
        bt_kwargs = dict(
            tp_pct=robust_tp,
            friction=friction,
            capital=capital,
            order_size_pct=order_size_pct,
            latency_minutes=latency_minutes,
            orderbook=orderbook,
            trail_activation=robust_trail_act,
            trail_distance=robust_trail_dist,
            min_time_remaining_hours=min_time_remaining_hours,
            stop_loss_pct=robust_sl,
        )
        if use_continuous:
            bt_kwargs["alpha_up"] = robust_alpha_up
            bt_kwargs["alpha_down"] = robust_alpha_down
            bt_kwargs["floor_up"] = robust_floor_up
            bt_kwargs["floor_down"] = robust_floor_down
        else:
            bt_kwargs["edge_up"] = robust_edge_up
            bt_kwargs["edge_down"] = robust_edge_down
        trades, final_cap = run_backtest(df, **bt_kwargs)
        total_pnl = final_cap - capital
        pnl_pct = (total_pnl / capital) * 100
        n_trades = len(trades)
        wins = sum(1 for t in trades if t.pnl > 0)
        losses = sum(1 for t in trades if t.pnl < 0)
        win_rate = (wins / n_trades * 100) if n_trades > 0 else 0
        trade_pnls = [t.pnl for t in trades]
        pf = compute_profit_factor(trade_pnls) if trade_pnls else None

        # Compute MTM Sharpe and max drawdown from equity curve
        sharpe = None
        robust_max_dd_pct = 0.0
        if trades:
            _, robust_mtm_pnl = build_equity_curve(trades, df, friction, orderbook)
            if robust_mtm_pnl:
                robust_max_dd_pct = compute_max_drawdown(robust_mtm_pnl, capital=capital)["max_dd_pct"]
                _dur = get_data_stats(df)["duration_hours"] / 24 if df else 0.0
                sharpe = compute_mtm_sharpe(robust_mtm_pnl, _dur)
        # Fallback to per-trade Sharpe when equity curve is too short
        if sharpe is None and trade_pnls:
            _dur = get_data_stats(df)["duration_hours"] / 24 if df else 0.0
            sharpe = compute_sharpe_ratio(trade_pnls, duration_days=_dur)

        sharpe_str = f"{sharpe:.1f}" if sharpe is not None else "N/A"
        pf_str = f"{pf:.1f}" if pf is not None and pf != float('inf') else ("inf" if pf == float('inf') else "N/A")

        print(f"\n  ROBUST BACKTEST RESULT:")
        print(f"    Trades: {n_trades}, Win rate: {win_rate:.1f}%, P&L: {pnl_pct:+.1f}%")
        print(f"    Sharpe(Y): {sharpe_str}, Profit Factor: {pf_str}, Max DD (MTM): {robust_max_dd_pct:.1f}%")
        if n_trades > 0:
            avg_win = sum(t.pnl for t in trades if t.pnl > 0) / wins if wins else 0
            avg_loss = sum(t.pnl for t in trades if t.pnl < 0) / losses if losses else 0
            tp_count = sum(1 for t in trades if t.result == "TP_FILLED")
            trail_count = sum(1 for t in trades if t.result == "TRAILING_STOP")
            sl_count = sum(1 for t in trades if t.result == "STOP_LOSS")
            loss_count = sum(1 for t in trades if t.result == "LOSS_EXPIRY")
            print(f"    Avg win: +${avg_win:.2f}, Avg loss: ${avg_loss:.2f}")
            exits_str = f"TP: {tp_count}, Trail: {trail_count}"
            if sl_count > 0:
                exits_str += f", SL: {sl_count}"
            exits_str += f", Loss@expiry: {loss_count}"
            print(f"    {exits_str}")
        if n_trades < 10:
            print(f"\n  WARNING: Only {n_trades} trades. Results may be unstable.")

    print()


def build_equity_curve(trades, df, friction, orderbook=None):
    """Build a continuous equity curve with MTM unrealized P&L during open positions.

    Returns (timestamps, pnl_values) lists for plotting.
    Between trades: flat line at current realized P&L.
    During trades: mark-to-market unrealized P&L snapshots.
    """
    if not trades:
        return [], []

    sorted_trades = sorted(trades, key=lambda t: t.entry_timestamp)
    markets = group_markets(df)

    # Build a flat list of all rows sorted by timestamp for start/end points
    all_timestamps = [row["timestamp"] for row in df]
    global_start = min(all_timestamps)
    global_end = max(all_timestamps)

    timestamps = []
    pnl_values = []
    realized_pnl = 0.0

    # Start point
    timestamps.append(datetime.fromisoformat(global_start.replace("Z", "+00:00")))
    pnl_values.append(0.0)

    for trade in sorted_trades:
        entry_dt = datetime.fromisoformat(trade.entry_timestamp.replace("Z", "+00:00"))
        exit_dt = datetime.fromisoformat(trade.exit_timestamp.replace("Z", "+00:00"))

        # Flat segment: from last point to trade entry
        timestamps.append(entry_dt)
        pnl_values.append(realized_pnl)

        # MTM snapshots during the trade
        barrier = trade.reference_price
        market_rows = markets.get(barrier, [])
        direction = trade.direction  # "UP" or "DOWN"

        for row in market_rows:
            row_ts = row["timestamp"]
            if row_ts <= trade.entry_timestamp or row_ts >= trade.exit_timestamp:
                continue

            # Get current bid price for MTM
            bid, _ = get_orderbook_prices(orderbook, row_ts, direction, reference_price=barrier)
            if bid is not None:
                current_price = bid
            else:
                poly_field = "poly_prob_up" if direction == "UP" else "poly_prob_down"
                current_price = row[poly_field] / 100

            # Unrealized P&L = what we'd get selling now minus cost
            unrealized = trade.shares * current_price * (1 - friction) - trade.cost_basis
            timestamps.append(datetime.fromisoformat(row_ts.replace("Z", "+00:00")))
            pnl_values.append(realized_pnl + unrealized)

        # Close point
        realized_pnl += trade.pnl
        timestamps.append(exit_dt)
        pnl_values.append(realized_pnl)

    # End point
    timestamps.append(datetime.fromisoformat(global_end.replace("Z", "+00:00")))
    pnl_values.append(realized_pnl)

    return timestamps, pnl_values


def compute_max_drawdown(pnl_values, timestamps=None, capital=100.0):
    """Compute max drawdown from a P&L series.

    Args:
        pnl_values: List of cumulative P&L values (equity = capital + pnl).
        timestamps: Optional list of timestamps for peak/trough location.
        capital: Starting capital, used to convert drawdown to percentage.

    Returns dict with:
        max_dd: max drawdown amount (negative)
        max_dd_pct: max drawdown as % of peak equity (negative)
        peak_idx/trough_idx: indices into the series
    If timestamps is provided, also returns peak_ts/trough_ts.
    """
    if len(pnl_values) < 2:
        return {"max_dd": 0.0, "max_dd_pct": 0.0}

    peak = pnl_values[0]
    peak_idx = 0
    max_dd = 0.0
    dd_peak_idx = 0
    dd_trough_idx = 0

    for i, val in enumerate(pnl_values):
        if val > peak:
            peak = val
            peak_idx = i
        dd = val - peak  # negative when in drawdown
        if dd < max_dd:
            max_dd = dd
            dd_peak_idx = peak_idx
            dd_trough_idx = i

    # Percentage relative to peak equity (capital + peak P&L)
    peak_equity = capital + pnl_values[dd_peak_idx]
    max_dd_pct = (max_dd / peak_equity * 100) if peak_equity > 0 else 0.0

    result = {
        "max_dd": max_dd,
        "max_dd_pct": max_dd_pct,
        "peak_idx": dd_peak_idx,
        "trough_idx": dd_trough_idx,
    }
    if timestamps:
        result["peak_ts"] = timestamps[dd_peak_idx]
        result["trough_ts"] = timestamps[dd_trough_idx]
    return result


def plot_best_pnl_curve(trades, params, capital, save_path="results/pnl_curve.png",
                        df=None, friction=0.015, orderbook=None, label=None):
    """Plot cumulative PnL over time for the best parameter combination.

    If df is provided, plots a continuous mark-to-market equity curve with
    unrealized P&L during open positions and flat lines between trades.
    Otherwise falls back to trade-close-only staircase plot.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not trades:
        print("  No trades to plot.")
        return

    # Sort by exit timestamp (ISO strings sort lexicographically)
    sorted_trades = sorted(trades, key=lambda t: t.exit_timestamp)

    # Compute cumulative P&L at each trade close (for scatter markers)
    close_timestamps = [datetime.fromisoformat(t.exit_timestamp.replace("Z", "+00:00")) for t in sorted_trades]
    cum_pnl = []
    running = 0.0
    for t in sorted_trades:
        running += t.pnl
        cum_pnl.append(running)

    wins = [(ts, pnl) for ts, t, pnl in zip(close_timestamps, sorted_trades, cum_pnl) if t.pnl > 0]
    losses = [(ts, pnl) for ts, t, pnl in zip(close_timestamps, sorted_trades, cum_pnl) if t.pnl <= 0]

    has_prob_data = df is not None and len(df) > 0 and all(
        c in df[0] for c in ("model_prob_up", "model_prob_down", "poly_prob_up", "poly_prob_down")
    )
    nrows = 3 if has_prob_data else 1
    fig, axes = plt.subplots(nrows, 1, figsize=(12, 4 + 2.5 * (nrows - 1)),
                             sharex=True,
                             gridspec_kw={"height_ratios": [3, 1, 1]} if nrows == 3 else None)
    ax = axes[0] if nrows > 1 else axes

    # Plot MTM equity curve if raw data is available, else fallback to close-only line
    max_dd_info = None
    if df is not None:
        mtm_ts, mtm_pnl = build_equity_curve(sorted_trades, df, friction, orderbook)
        if mtm_ts:
            ax.plot(mtm_ts, mtm_pnl, color="steelblue", linewidth=1.2, zorder=2, label="_nolegend_")
            # Compute and shade max drawdown from MTM curve
            dd = compute_max_drawdown(mtm_pnl, mtm_ts, capital=capital)
            if dd["max_dd"] < 0:
                max_dd_info = dd
                peak_ts = dd["peak_ts"]
                trough_ts = dd["trough_ts"]
                peak_val = mtm_pnl[dd["peak_idx"]]
                trough_val = mtm_pnl[dd["trough_idx"]]
                # Shade the drawdown region
                ax.fill_between(
                    [peak_ts, trough_ts], peak_val, trough_val,
                    alpha=0.15, color="red", zorder=1,
                )
                # Annotate
                mid_ts = peak_ts + (trough_ts - peak_ts) / 2
                mid_val = (peak_val + trough_val) / 2
                ax.annotate(
                    f"Max DD: {dd['max_dd_pct']:.1f}%",
                    xy=(trough_ts, trough_val), xytext=(mid_ts, mid_val),
                    fontsize=8, color="darkred", fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color="darkred", lw=0.8),
                    zorder=5,
                )
    else:
        ax.plot(close_timestamps, cum_pnl, color="steelblue", linewidth=1.5, zorder=2)

    if wins:
        ax.scatter([w[0] for w in wins], [w[1] for w in wins], color="green", s=30, zorder=3, label="Win")
    if losses:
        ax.scatter([l[0] for l in losses], [l[1] for l in losses], color="red", s=30, zorder=3, label="Loss")
    ax.axhline(0, color="grey", linewidth=0.8, linestyle="--", zorder=1)

    ax.set_ylabel("Cumulative PnL ($)")

    # --- Probabilities subplots (UP and DOWN) ---
    if has_prob_data:
        import numpy as np

        prob_ts = [datetime.fromisoformat(r["timestamp"].replace("Z", "+00:00")) for r in df]
        model_up = [float(r["model_prob_up"]) for r in df]
        poly_up = [float(r["poly_prob_up"]) for r in df]
        model_down = [float(r["model_prob_down"]) for r in df]
        poly_down = [float(r["poly_prob_down"]) for r in df]

        # Normalize: if values are 0-100 scale, convert to 0-1
        if max(model_up + poly_up + model_down + poly_down) > 1.5:
            model_up = [v / 100.0 for v in model_up]
            poly_up = [v / 100.0 for v in poly_up]
            model_down = [v / 100.0 for v in model_down]
            poly_down = [v / 100.0 for v in poly_down]

        # Insert NaN at daily market boundaries (gap > 1 hour) to break lines
        ts_plot, mu, pu, md, pd = [], [], [], [], []
        for i in range(len(prob_ts)):
            if i > 0 and (prob_ts[i] - prob_ts[i - 1]).total_seconds() > 3600:
                ts_plot.append(prob_ts[i - 1] + (prob_ts[i] - prob_ts[i - 1]) / 2)
                mu.append(float("nan"))
                pu.append(float("nan"))
                md.append(float("nan"))
                pd.append(float("nan"))
            ts_plot.append(prob_ts[i])
            mu.append(model_up[i])
            pu.append(poly_up[i])
            md.append(model_down[i])
            pd.append(poly_down[i])

        # Helper: add trade entry/exit markers to a probability subplot
        def _mark_trades(ax_prob, direction_label):
            for t in sorted_trades:
                if t.direction != direction_label:
                    continue
                entry_ts = datetime.fromisoformat(t.entry_timestamp.replace("Z", "+00:00"))
                exit_ts = datetime.fromisoformat(t.exit_timestamp.replace("Z", "+00:00"))
                is_win = t.pnl > 0
                # Shade holding period
                ax_prob.axvspan(entry_ts, exit_ts,
                                alpha=0.10, color="green" if is_win else "red", zorder=0)
                # Entry marker (triangle up)
                ax_prob.axvline(entry_ts, color="green", alpha=0.6, linewidth=0.7, linestyle="-")
                # Exit marker (color by outcome)
                ax_prob.axvline(exit_ts, color="green" if is_win else "red",
                                alpha=0.6, linewidth=0.7, linestyle="-")

        # UP subplot
        ax_up = axes[1]
        ax_up.plot(ts_plot, mu, color="tab:blue", linewidth=1.0, label="Model")
        ax_up.plot(ts_plot, pu, color="tab:blue", linewidth=1.0, alpha=0.4, linestyle="--", label="Poly")
        ax_up.fill_between(ts_plot, mu, pu, alpha=0.15, color="tab:blue")
        _mark_trades(ax_up, "UP")
        ax_up.set_ylabel("UP Prob")
        ax_up.set_ylim(0, 1)
        ax_up.legend(loc="upper right", fontsize=7)

        # DOWN subplot
        ax_dn = axes[2]
        ax_dn.plot(ts_plot, md, color="tab:orange", linewidth=1.0, label="Model")
        ax_dn.plot(ts_plot, pd, color="tab:orange", linewidth=1.0, alpha=0.4, linestyle="--", label="Poly")
        ax_dn.fill_between(ts_plot, md, pd, alpha=0.15, color="tab:orange")
        _mark_trades(ax_dn, "DOWN")
        ax_dn.set_ylabel("DOWN Prob")
        ax_dn.set_ylim(0, 1)
        ax_dn.legend(loc="upper right", fontsize=7)
        ax_dn.set_xlabel("Time")
    else:
        ax.set_xlabel("Time")

    # Build rich title with all parameters and key metrics
    p = params
    trail_act = p.get('trail_activation', 0)
    trail_dist = p.get('trail_distance', 0)
    use_continuous = p.get('alpha_up') is not None
    sl_pct = p.get('stop_loss_pct', 0)
    sl_label = f"  sl={sl_pct*100:.0f}%" if sl_pct > 0 else ""
    if use_continuous:
        title_line1 = (
            f"aUP={p['alpha_up']:.2f}  aDN={p['alpha_down']:.2f}  "
            f"fUP={p['floor_up']:.2f}  fDN={p['floor_down']:.2f}  "
            f"tp={p['tp_pct']*100:.0f}%  trail={trail_act*100:.0f}%/{trail_dist*100:.0f}pp{sl_label}"
        )
    else:
        title_line1 = (
            f"edge_up={p['edge_up']:.2f}  edge_down={p['edge_down']:.2f}  "
            f"tp={p['tp_pct']*100:.0f}%  trail={trail_act*100:.0f}%/{trail_dist*100:.0f}pp{sl_label}"
        )
    if label:
        title_line1 = f"[Best by {label}]  {title_line1}"
    n_trades = len(trades)
    wins_n = sum(1 for t in trades if t.pnl > 0)
    win_rate = (wins_n / n_trades * 100) if n_trades > 0 else 0
    pnl_pct = p.get('pnl_pct', 0)
    dd_pct = max_dd_info['max_dd_pct'] if max_dd_info else 0
    sharpe = p.get('sharpe_ratio')
    sharpe_str = f"{sharpe:.1f}" if sharpe is not None else "N/A"
    title_line2 = (
        f"Trades: {n_trades}  W/L: {wins_n}/{n_trades - wins_n}  "
        f"Win: {win_rate:.0f}%  P&L: {pnl_pct:+.1f}%  "
        f"MaxDD: {dd_pct:.1f}%  Sharpe(Y): {sharpe_str}"
    )
    suptitle_y = 0.995 if nrows > 1 else 0.99
    top_margin = 0.94 if nrows > 1 else 0.88
    fig.suptitle(title_line1, fontsize=11, fontweight="bold", y=suptitle_y)
    ax.set_title(title_line2, fontsize=9, color="grey")

    ax.legend(loc="best")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.subplots_adjust(top=top_margin)

    save = Path(save_path)
    if not save.is_absolute():
        save = Path(__file__).parent.parent / save_path
    save.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(save), dpi=150)
    plt.close(fig)
    print(f"  PnL curve saved to: {save}")
    if max_dd_info:
        print(f"  Max MTM drawdown: {max_dd_info['max_dd_pct']:.1f}%")


def _format_result_row(r: dict, rank_str: str, duration_days: float, use_continuous: bool = False) -> str:
    """Format a single result row for the optimization table."""
    pnl_str = f"{r['pnl_pct']:+.1f}%"
    wl_str = f"{r['wins']}/{r['losses']}"
    sharpe_str = f"{r.get('sharpe_ratio', 0):.1f}" if r.get('sharpe_ratio') is not None else "N/A"
    pf = r.get('profit_factor')
    pf_str = f"{pf:.1f}" if pf is not None and pf != float('inf') else ("inf" if pf == float('inf') else "N/A")
    if duration_days > 0:
        daily_pct = r['pnl_pct'] / duration_days
        daily_str = f"{daily_pct:+.2f}%"
        days_str = f"{duration_days:.1f}"
        trd_day_str = f"{r['trades'] / duration_days:.2f}"
    else:
        daily_str = days_str = trd_day_str = "N/A"
    ta_str = f"{r['trail_activation']*100:.0f}%"
    td_str = f"{r['trail_distance']*100:.0f}p"
    dd_pct = r.get('max_dd_pct', 0.0)
    dd_str = f"{dd_pct:.1f}%"
    sl_pct = r.get('stop_loss_pct', 0.0)
    sl_str = f"{sl_pct*100:.0f}%" if sl_pct > 0 else "off"

    if use_continuous:
        au_str = f"{r.get('alpha_up', 0):.2f}"
        ad_str = f"{r.get('alpha_down', 0):.2f}"
        fu_str = f"{r.get('floor_up', 0):.2f}"
        fd_str = f"{r.get('floor_down', 0):.2f}"
        return (f"  {rank_str:<5} {au_str:<6} {ad_str:<6} {fu_str:<6} {fd_str:<6} "
                f"{r['tp_pct']*100:<6.0f} {ta_str:>5} {td_str:>5} {sl_str:>5} {r['trades']:<8} "
                f"{wl_str:<8} {r['win_rate']:<8.1f} {pnl_str:>8} {dd_str:>8} {days_str:>6} "
                f"{trd_day_str:>6} {daily_str:>8} {sharpe_str:>9} {pf_str:>6}")
    else:
        return (f"  {rank_str:<5} {r['edge_up']:<9.2f} {r['edge_down']:<9.2f} "
                f"{r['tp_pct']*100:<6.0f} {ta_str:>5} {td_str:>5} {sl_str:>5} {r['trades']:<8} "
                f"{wl_str:<8} {r['win_rate']:<8.1f} {pnl_str:>8} {dd_str:>8} {days_str:>6} "
                f"{trd_day_str:>6} {daily_str:>8} {sharpe_str:>9} {pf_str:>6}")


def _print_ranked_table(filtered: list[dict], sort_key: str, sort_label: str,
                        top_n: int, duration_days: float, table_width: int) -> list[dict]:
    """Sort filtered results by sort_key and print a ranked table. Returns sorted list."""
    if sort_key == "sharpe_ratio":
        sorted_list = sorted(filtered, key=lambda x: x.get("sharpe_ratio") or -999, reverse=True)
    elif sort_key == "cv_mean_pnl":
        sorted_list = sorted(filtered, key=lambda x: x.get("cv_mean_pnl") or -999, reverse=True)
    else:
        sorted_list = sorted(filtered, key=lambda x: x["pnl"], reverse=True)

    # Detect mode from results
    use_continuous = filtered[0].get("alpha_up") is not None if filtered else False

    label = f"  --- Top {top_n} by {sort_label} "
    print(label + "-" * max(0, table_width - len(label)))
    if use_continuous:
        header = (f"  {'Rank':<5} {'aUP':<6} {'aDN':<6} {'fUP':<6} {'fDN':<6} {'TP%':<6} {'TrlA':>5} {'TrlD':>5} {'SL%':>5} "
                  f"{'Trades':<8} {'W/L':<8} {'Win%':<8} {'P&L%':>8} {'MaxDD%':>8} {'Days':>6} "
                  f"{'Trd/D':>6} {'Daily%':>8} {'Sharpe(Y)':>9} {'PF':>6}")
    else:
        header = (f"  {'Rank':<5} {'Edge UP':<9} {'Edge DN':<9} {'TP%':<6} {'TrlA':>5} {'TrlD':>5} {'SL%':>5} "
                  f"{'Trades':<8} {'W/L':<8} {'Win%':<8} {'P&L%':>8} {'MaxDD%':>8} {'Days':>6} "
                  f"{'Trd/D':>6} {'Daily%':>8} {'Sharpe(Y)':>9} {'PF':>6}")
    print(header)
    print("  " + "-" * (table_width - 2))

    for i, r in enumerate(sorted_list[:top_n], 1):
        print(_format_result_row(r, str(i), duration_days, use_continuous=use_continuous))

    return sorted_list


def print_optimization_results(
    results: list[dict],
    top_n: int = 20,
    min_trades: int = 0,
    show_robust: bool = False,
    sort_by: str = "pnl",
    duration_days: float = 0.0,
    data_stats: dict | None = None,
    friction: float = 0.015,
    latency_minutes: float = 2.0,
    order_size_pct: float = 0.05,
) -> None:
    """Print optimization results table."""
    TABLE_WIDTH = 137

    # Filter by min trades
    filtered = [r for r in results if r["trades"] >= min_trades]

    # Banner
    print("\n" + "=" * TABLE_WIDTH)
    min_label = f"min {min_trades} trades" if min_trades > 0 else ""
    print(f"OPTIMIZATION RESULTS{min_label:>{TABLE_WIDTH - 20}}")
    if data_stats and data_stats.get("start"):
        start_str = data_stats["start"][:16].replace("T", " ")
        end_str = data_stats["end"][:16].replace("T", " ")
        dur_days = data_stats["duration_hours"] / 24
        print(f"  Data: {start_str} -> {end_str} ({dur_days:.1f} days, {data_stats['num_markets']} markets)")
    print(f"  Fixed: friction={friction*100:.1f}%/side, order_size={order_size_pct*100:.0f}%, latency={latency_minutes:.0f}min")
    print("=" * TABLE_WIDTH)

    # Always show Sharpe table
    print()
    sharpe_sorted = _print_ranked_table(filtered, "sharpe_ratio", "Sharpe", top_n, duration_days, TABLE_WIDTH)

    # Always show P&L table
    print()
    pnl_sorted = _print_ranked_table(filtered, "pnl", "P&L", top_n, duration_days, TABLE_WIDTH)

    # Optional CV table when --sort-by cv
    has_cv = any(r.get("cv_mean_pnl") is not None for r in filtered)
    cv_sorted = None
    if sort_by == "cv" and has_cv:
        print()
        cv_sorted = _print_ranked_table(filtered, "cv_mean_pnl", "CV Mean", top_n, duration_days, TABLE_WIDTH)

    # Determine primary sort for robustness tables and BEST PARAMETERS
    sort_label = {"pnl": "P&L", "sharpe": "Sharpe", "cv": "CV Mean"}.get(sort_by, "P&L")
    if sort_by == "sharpe":
        primary_sorted = sharpe_sorted
    elif sort_by == "cv" and cv_sorted is not None:
        primary_sorted = cv_sorted
    else:
        primary_sorted = pnl_sorted

    # Bootstrap CI table (using primary sort order)
    if show_robust and any(r.get("bootstrap_mean") is not None for r in primary_sorted[:top_n]):
        print("\n  " + "-" * (TABLE_WIDTH - 2))
        print("  BOOTSTRAP CONFIDENCE INTERVALS (90%)")
        print("  " + "-" * (TABLE_WIDTH - 2))
        print(f"  {'Rank':<5} {'Edge UP':<9} {'Edge DN':<9} {'TP%':<6} {'P&L':>10} {'Boot Mean':>12} {'Boot Std':>10} {'5th %':>10} {'95th %':>10}")
        print("  " + "-" * (TABLE_WIDTH - 2))

        for i, r in enumerate(primary_sorted[:top_n], 1):
            if r.get("bootstrap_mean") is None:
                continue
            pnl_str = f"${r['pnl']:.2f}"
            mean_str = f"${r['bootstrap_mean']:.2f}"
            std_str = f"${r['bootstrap_std']:.2f}"
            p5_str = f"${r['bootstrap_5th']:.2f}"
            p95_str = f"${r['bootstrap_95th']:.2f}"
            print(f"  {i:<5} {r['edge_up']:<9.2f} {r['edge_down']:<9.2f} {r['tp_pct']*100:<6.0f} {pnl_str:>10} {mean_str:>12} {std_str:>10} {p5_str:>10} {p95_str:>10}")

    # Per-market P&L table (using primary sort order)
    if show_robust and any(r.get("cv_mean_pnl") is not None for r in primary_sorted[:top_n]):
        print("\n  " + "-" * (TABLE_WIDTH - 2))
        print("  PER-MARKET P&L ANALYSIS")
        print("  " + "-" * (TABLE_WIDTH - 2))
        print(f"  {'Rank':<5} {'Edge UP':<9} {'Edge DN':<9} {'TP%':<6} {'P&L':>10} {'CV Mean':>10} {'CV Std':>10} {'Worst':>10} {'Profitable':>12}")
        print("  " + "-" * (TABLE_WIDTH - 2))

        for i, r in enumerate(primary_sorted[:top_n], 1):
            if r.get("cv_mean_pnl") is None:
                continue
            pnl_str = f"${r['pnl']:.2f}"
            cv_mean = f"${r['cv_mean_pnl']:.2f}"
            cv_std = f"${r['cv_std_pnl']:.2f}"
            worst = f"${r['cv_worst_pnl']:.2f}"
            prof = f"{r['cv_markets_profitable']}/{r['cv_total_markets']}"
            print(f"  {i:<5} {r['edge_up']:<9.2f} {r['edge_down']:<9.2f} {r['tp_pct']*100:<6.0f} {pnl_str:>10} {cv_mean:>10} {cv_std:>10} {worst:>10} {prof:>12}")

    # Detect mode from results
    use_continuous = filtered[0].get("alpha_up") is not None if filtered else False

    # Worst performers
    worst_results = sorted([r for r in filtered if r["trades"] > 0], key=lambda x: x["pnl"])[:5]
    if worst_results:
        print(f"\n  Worst performers by P&L:")
        print("  " + "-" * (TABLE_WIDTH - 2))
        for r in worst_results:
            print(_format_result_row(r, "--", duration_days, use_continuous=use_continuous))

    # BEST PARAMETERS footer
    print("\n" + "=" * TABLE_WIDTH)

    if primary_sorted:
        best = primary_sorted[0]
        print(f"  BEST PARAMETERS (by {sort_label}):")
        print("  " + "-" * 70)
        trail_flag = f" --trail-activation {best['trail_activation']:.2f} --trail-distance {best['trail_distance']:.2f}"
        sl_flag = f" --stop-loss {best['stop_loss_pct']:.2f}" if best.get('stop_loss_pct', 0) > 0 else ""
        if use_continuous:
            print(f"    python scripts/backtest.py --alpha-up {best['alpha_up']:.2f} --alpha-down {best['alpha_down']:.2f} --floor-up {best['floor_up']:.2f} --floor-down {best['floor_down']:.2f} --tp {best['tp_pct']:.2f}{trail_flag}{sl_flag}")
        else:
            print(f"    python scripts/backtest.py --edge-up {best['edge_up']:.2f} --edge-down {best['edge_down']:.2f} --tp {best['tp_pct']:.2f}{trail_flag}{sl_flag}")
        print(f"    Trades: {best['trades']}, Win rate: {best['win_rate']:.1f}%, P&L: {best['pnl_pct']:+.1f}%")

        if best.get("sharpe_ratio") is not None:
            pf_val = best.get('profit_factor')
            pf_disp = f"{pf_val:.1f}" if pf_val is not None and pf_val != float('inf') else ("inf" if pf_val == float('inf') else "N/A")
            dd_disp = f"{best.get('max_dd_pct', 0):.1f}%"
            print(f"    Sharpe(Y): {best['sharpe_ratio']:.1f}, Profit Factor: {pf_disp}, Max DD (MTM): {dd_disp}")

        if best.get("bootstrap_5th") is not None:
            print(f"    Bootstrap 90% CI: [${best['bootstrap_5th']:.2f}, ${best['bootstrap_95th']:.2f}]")

        if best.get("cv_mean_pnl") is not None:
            print(f"    CV Mean P&L: ${best['cv_mean_pnl']:.2f}, Profitable markets: {best['cv_markets_profitable']}/{best['cv_total_markets']}")

        # Warnings about small sample size
        if best["trades"] < 10:
            print(f"\n  WARNING: Only {best['trades']} trades. Results may be unstable.")
        if best.get("cv_total_markets") and best["cv_total_markets"] < 10:
            print(f"  WARNING: Only {best['cv_total_markets']} markets. Cross-validation unreliable.")
        if best.get("bootstrap_5th") is not None and best["bootstrap_5th"] < 0:
            print(f"  WARNING: 5th percentile is negative (${best['bootstrap_5th']:.2f}). Risk of loss.")

        print()


def load_config() -> tuple[dict, dict]:
    """Load config_optimize.json for default parameters."""
    project_root = Path(__file__).parent.parent

    opt_config = {}
    opt_path = project_root / "config_optimize.json"
    if opt_path.exists():
        with open(opt_path, "r") as f:
            opt_config = json.load(f)

    return opt_config, {}


def main():
    # Load config for defaults
    opt_config, trader_config = load_config()

    parser = argparse.ArgumentParser(
        description="Optimize backtest parameters to find best thresholds"
    )
    parser.add_argument(
        "--csv",
        type=str,
        default="data_collector/results/probabilities.csv",
        help="Path to probabilities.csv",
    )
    parser.add_argument(
        "--friction",
        type=float,
        default=opt_config.get("friction", 0.015),
        help="Fee + spread per side (default: 0.015 = 1.5%%)",
    )
    parser.add_argument(
        "--capital",
        type=float,
        default=opt_config.get("capital", 100.0),
        help="Starting capital (default: 100.0)",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=opt_config.get("top", 20),
        help="Number of top results to show (default: 20)",
    )
    parser.add_argument(
        "--min-trades",
        type=int,
        default=opt_config.get("min_trades", 6),
        help="Minimum trades to include in results (default: 6)",
    )
    # Parameter ranges
    parser.add_argument(
        "--edge-up-min",
        type=float,
        default=opt_config.get("edge_up_min", 1.1),
        help="Min edge_up to test (default: 1.1)",
    )
    parser.add_argument(
        "--edge-up-max",
        type=float,
        default=opt_config.get("edge_up_max", 2.0),
        help="Max edge_up to test (default: 2.0)",
    )
    parser.add_argument(
        "--edge-down-min",
        type=float,
        default=opt_config.get("edge_down_min", 1.1),
        help="Min edge_down to test (default: 1.1)",
    )
    parser.add_argument(
        "--edge-down-max",
        type=float,
        default=opt_config.get("edge_down_max", 2.0),
        help="Max edge_down to test (default: 2.0)",
    )
    parser.add_argument(
        "--tp-min",
        type=float,
        default=opt_config.get("tp_min", 0.10),
        help="Min TP%% to test (default: 0.10)",
    )
    parser.add_argument(
        "--tp-max",
        type=float,
        default=opt_config.get("tp_max", 0.30),
        help="Max TP%% to test (default: 0.30)",
    )
    parser.add_argument(
        "--tp-step",
        type=float,
        default=opt_config.get("tp_step", 0.05),
        help="Step size for TP parameter (default: 0.05)",
    )
    parser.add_argument(
        "--step",
        type=float,
        default=opt_config.get("edge_step", 0.1),
        help="Step size for edge parameters (default: 0.1)",
    )
    parser.add_argument(
        "--latency",
        type=float,
        default=opt_config.get("latency", 2.0),
        help="Delay in minutes between signal and entry (default: 2)",
    )
    # Robustness options
    parser.add_argument(
        "--bootstrap",
        action="store_true",
        help="Enable bootstrap confidence intervals (slower)",
    )
    parser.add_argument(
        "--cv",
        action="store_true",
        help="Enable per-market P&L analysis (slower)",
    )
    parser.add_argument(
        "--robust",
        action="store_true",
        help="Enable all robustness features (bootstrap + CV + stability)",
    )
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=opt_config.get("n_bootstrap", 100),
        help="Number of bootstrap iterations (default: 100)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible bootstrap results (default: None)",
    )
    parser.add_argument(
        "--sort-by",
        type=str,
        choices=["pnl", "sharpe", "cv"],
        default=opt_config.get("sort_by", "sharpe"),
        help="Sort results by: pnl, sharpe, or cv (default: sharpe)",
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
    trail_act_default = opt_config.get("trail_activation", [0.15, 0.20, 0.25, 0.30])
    if isinstance(trail_act_default, (int, float)):
        trail_act_default = [trail_act_default]
    parser.add_argument(
        "--trail-activation",
        type=float,
        nargs="+",
        default=trail_act_default,
        help="Trail activation values to grid search (default: 0.15 0.20 0.25 0.30)",
    )
    trail_dist_default = opt_config.get("trail_distance", [0.10, 0.15, 0.20])
    if isinstance(trail_dist_default, (int, float)):
        trail_dist_default = [trail_dist_default]
    parser.add_argument(
        "--trail-distance",
        type=float,
        nargs="+",
        default=trail_dist_default,
        help="Trail distance values to grid search (default: 0.10 0.15 0.20)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=opt_config.get("workers", 4),
        help="Number of parallel workers (default: 4, use 1 for sequential)",
    )
    parser.add_argument(
        "--min-time-remaining",
        type=float,
        default=trader_config.get("min_time_remaining_hours", 2.0),
        help="Min hours before expiry to allow entry (default: 2.0)",
    )

    # Continuous edge mode parameters (per-direction)
    alpha_up_default = opt_config.get("alpha_up", None)
    if isinstance(alpha_up_default, (int, float)):
        alpha_up_default = [alpha_up_default]
    parser.add_argument(
        "--alpha-up",
        type=float,
        nargs="+",
        default=alpha_up_default,
        help="Alpha UP values to grid search (enables continuous edge mode). E.g.: 1.0 1.5 2.0",
    )
    alpha_down_default = opt_config.get("alpha_down", None)
    if isinstance(alpha_down_default, (int, float)):
        alpha_down_default = [alpha_down_default]
    parser.add_argument(
        "--alpha-down",
        type=float,
        nargs="+",
        default=alpha_down_default,
        help="Alpha DOWN values to grid search. Defaults to same as --alpha-up if not set.",
    )
    floor_up_default = opt_config.get("floor_up", [0.35])
    if isinstance(floor_up_default, (int, float)):
        floor_up_default = [floor_up_default]
    parser.add_argument(
        "--floor-up",
        type=float,
        nargs="+",
        default=floor_up_default,
        help="Floor UP values to grid search (default: 0.35)",
    )
    floor_down_default = opt_config.get("floor_down", [0.35])
    if isinstance(floor_down_default, (int, float)):
        floor_down_default = [floor_down_default]
    parser.add_argument(
        "--floor-down",
        type=float,
        nargs="+",
        default=floor_down_default,
        help="Floor DOWN values to grid search (default: 0.35)",
    )
    parser.add_argument(
        "--order-size",
        type=float,
        default=opt_config.get("order_size_pct", 0.05),
        help="Order size as fraction of capital (default: 0.05 = 5%%)",
    )
    sl_default = opt_config.get("stop_loss", [0.0])
    if isinstance(sl_default, (int, float)):
        sl_default = [sl_default]
    parser.add_argument(
        "--stop-loss",
        type=float,
        nargs="+",
        default=sl_default,
        help="Stop-loss %% values to grid search (0.20 = 20%% below entry, 0 = disabled)",
    )

    args = parser.parse_args()

    # --robust enables all robustness features
    if args.robust:
        args.bootstrap = True
        args.cv = True

    # Set random seed for reproducibility
    if args.seed is not None:
        random.seed(args.seed)

    # Resolve CSV path
    csv_path = Path(args.csv)
    if not csv_path.is_absolute():
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

    # Show data stats
    data_stats = get_data_stats(df)
    if data_stats["start"]:
        start_str = data_stats["start"][:19].replace("T", " ")
        end_str = data_stats["end"][:19].replace("T", " ")
        print(f"Data: {start_str} to {end_str} ({data_stats['duration_hours']:.1f}h, {data_stats['num_markets']} markets)")

    print(f"Workers: {args.workers}")

    # Normalize scalar to list for trail params
    trail_act_values = args.trail_activation if isinstance(args.trail_activation, list) else [args.trail_activation]
    trail_dist_values = args.trail_distance if isinstance(args.trail_distance, list) else [args.trail_distance]

    # Normalize per-direction alpha/floor
    alpha_up_values = args.alpha_up if isinstance(args.alpha_up, list) else ([args.alpha_up] if args.alpha_up is not None else None)
    alpha_down_values = args.alpha_down if isinstance(args.alpha_down, list) else ([args.alpha_down] if args.alpha_down is not None else None)
    # If alpha_down not specified but alpha_up is, default alpha_down to alpha_up values
    if alpha_up_values is not None and alpha_down_values is None:
        alpha_down_values = alpha_up_values
    floor_up_values = args.floor_up if isinstance(args.floor_up, list) else [args.floor_up]
    floor_down_values = args.floor_down if isinstance(args.floor_down, list) else [args.floor_down]
    stop_loss_values = args.stop_loss if isinstance(args.stop_loss, list) else [args.stop_loss]

    # Run optimization
    results = run_optimization(
        df,
        friction=args.friction,
        capital=args.capital,
        edge_up_range=(args.edge_up_min, args.edge_up_max, args.step),
        edge_down_range=(args.edge_down_min, args.edge_down_max, args.step),
        tp_pct_range=(args.tp_min, args.tp_max, args.tp_step),
        latency_minutes=args.latency,
        enable_bootstrap=args.bootstrap,
        enable_cv=args.cv,
        n_bootstrap=args.n_bootstrap,
        orderbook=orderbook,
        trail_activation_values=trail_act_values,
        trail_distance_values=trail_dist_values,
        workers=args.workers,
        min_time_remaining_hours=args.min_time_remaining,
        alpha_up_values=alpha_up_values,
        alpha_down_values=alpha_down_values,
        floor_up_values=floor_up_values,
        floor_down_values=floor_down_values,
        stop_loss_values=stop_loss_values,
        order_size_pct=args.order_size,
    )

    # Print results
    show_robust = args.bootstrap or args.cv
    duration_days = data_stats["duration_hours"] / 24 if data_stats.get("duration_hours") else 0.0
    print_optimization_results(
        results,
        top_n=args.top,
        min_trades=args.min_trades,
        show_robust=show_robust,
        sort_by=args.sort_by,
        duration_days=duration_days,
        data_stats=data_stats,
        friction=args.friction,
        latency_minutes=args.latency,
        order_size_pct=args.order_size,
    )

    # Print stability analysis (with robust backtest)
    print_stability_analysis(
        results,
        min_trades=max(args.min_trades, 3),
        df=df,
        friction=args.friction,
        capital=args.capital,
        latency_minutes=args.latency,
        orderbook=orderbook,
        min_time_remaining_hours=args.min_time_remaining,
        order_size_pct=args.order_size,
    )

    # Plot PnL curves for best by Sharpe and best by P&L
    filtered = [r for r in results if r["trades"] >= args.min_trades]
    if filtered:
        plots = []

        # Best by Sharpe
        sharpe_sorted = sorted(filtered, key=lambda x: x.get("sharpe_ratio") or -999, reverse=True)
        if sharpe_sorted[0].get("sharpe_ratio") is not None:
            plots.append((sharpe_sorted[0], "results/pnl_curve_sharpe.png", "Sharpe"))

        # Best by P&L
        pnl_sorted = sorted(filtered, key=lambda x: x["pnl"], reverse=True)
        plots.append((pnl_sorted[0], "results/pnl_curve_pnl.png", "P&L"))

        for best, save_path, plot_label in plots:
            best_trades, _ = run_backtest(
                df,
                edge_up=best["edge_up"],
                edge_down=best["edge_down"],
                tp_pct=best["tp_pct"],
                friction=args.friction,
                capital=args.capital,
                order_size_pct=args.order_size,
                latency_minutes=args.latency,
                orderbook=orderbook,
                trail_activation=best["trail_activation"],
                trail_distance=best["trail_distance"],
                min_time_remaining_hours=args.min_time_remaining,
                min_model_prob=best.get("min_model_prob", 0),
                alpha_up=best.get("alpha_up"),
                alpha_down=best.get("alpha_down"),
                floor_up=best.get("floor_up"),
                floor_down=best.get("floor_down"),
                stop_loss_pct=best.get("stop_loss_pct", 0.0),
            )
            plot_best_pnl_curve(
                best_trades,
                best,
                args.capital,
                save_path=save_path,
                df=df,
                friction=args.friction,
                orderbook=orderbook,
                label=plot_label,
            )

    return 0


if __name__ == "__main__":
    exit(main())
