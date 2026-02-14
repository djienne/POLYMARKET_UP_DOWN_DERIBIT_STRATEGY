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
    alpha: Optional[float] = None  # Deprecated: use alpha_up/alpha_down
    floor: Optional[float] = None  # Deprecated: use floor_up/floor_down
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


def compute_sharpe_ratio(trade_pnls: list[float], risk_free_rate: float = 0.0) -> Optional[float]:
    """Compute Sharpe ratio from trade P&Ls."""
    if len(trade_pnls) < 2:
        return None
    mean_pnl = sum(trade_pnls) / len(trade_pnls)
    variance = sum((p - mean_pnl) ** 2 for p in trade_pnls) / (len(trade_pnls) - 1)
    std_pnl = variance ** 0.5
    if std_pnl == 0:
        return None
    return (mean_pnl - risk_free_rate) / std_pnl


def compute_mtm_sharpe(pnl_values: list[float]) -> Optional[float]:
    """Compute Sharpe ratio from consecutive changes in an MTM equity curve.

    This captures path-dependent risk (unrealized drawdowns) that per-trade
    Sharpe misses.  Falls back to None when there are fewer than 3 observations.
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
    return mean_chg / std_chg


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


def _worker_init(df, orderbook):
    """Pool initializer: store data in worker globals (called once per worker process)."""
    global _worker_df, _worker_orderbook
    _worker_df = df
    _worker_orderbook = orderbook


def _run_single_combo(args):
    """Run a single parameter combo and return a plain dict of results.

    Must be a top-level function so it's picklable for multiprocessing.
    Metrics are computed here so only plain dicts cross the process boundary.

    Args tuple layout:
        (edge_up, edge_down, tp_pct, trail_activation, trail_distance,
         friction, capital, latency_minutes, min_time_remaining_hours,
         min_model_prob, alpha_up, alpha_down, floor_up, floor_down)
    """
    (edge_up, edge_down, tp_pct, trail_activation, trail_distance,
     friction, capital, latency_minutes, min_time_remaining_hours,
     min_model_prob, alpha_up, alpha_down, floor_up_val, floor_down_val) = args

    trades, final_capital = run_backtest(
        _worker_df,
        edge_up=edge_up,
        edge_down=edge_down,
        tp_pct=tp_pct,
        friction=friction,
        capital=capital,
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
            sharpe = compute_mtm_sharpe(mtm_pnl)
    # Fallback to per-trade Sharpe when equity curve is too short
    if sharpe is None and trade_pnls:
        sharpe = compute_sharpe_ratio(trade_pnls)

    return {
        "edge_up": edge_up,
        "edge_down": edge_down,
        "tp_pct": tp_pct,
        "trail_activation": trail_activation,
        "trail_distance": trail_distance,
        "min_model_prob": min_model_prob,
        "alpha": alpha_up,  # backward compat (same as alpha_up)
        "floor": floor_up_val,  # backward compat (same as floor_up)
        "alpha_up": alpha_up,
        "alpha_down": alpha_down,
        "floor_up": floor_up_val,
        "floor_down": floor_down_val,
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
    min_model_prob_values: list[float] | None = None,
    alpha_values: list[float] | None = None,
    floor_values: list[float] | None = None,
    alpha_up_values: list[float] | None = None,
    alpha_down_values: list[float] | None = None,
    floor_up_values: list[float] | None = None,
    floor_down_values: list[float] | None = None,
) -> list[dict]:
    """
    Run optimization loop to find best thresholds.

    When alpha_up_values (or alpha_values) is provided, uses the continuous edge
    function and grid-searches over (alpha_up, alpha_down, floor_up, floor_down)
    instead of (edge_up, edge_down).

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
        min_model_prob_values: List of min model prob values to grid search (legacy)
        alpha_values: List of alpha values for both directions (deprecated, use alpha_up/alpha_down)
        floor_values: List of floor values for both directions (deprecated, use floor_up/floor_down)
        alpha_up_values: List of alpha values for UP direction
        alpha_down_values: List of alpha values for DOWN direction
        floor_up_values: List of floor values for UP direction
        floor_down_values: List of floor values for DOWN direction

    Returns:
        List of results sorted by P&L descending
    """
    # Resolve per-direction lists: alpha_up/alpha_down take priority over alpha
    _alpha_up_values = alpha_up_values or alpha_values
    _alpha_down_values = alpha_down_values or alpha_values
    _floor_up_values = floor_up_values or floor_values
    _floor_down_values = floor_down_values or floor_values
    use_continuous = _alpha_up_values is not None
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

    # Build list of all parameter combos (with validation filtering)
    all_args = []

    if use_continuous:
        au_values = _alpha_up_values
        ad_values = _alpha_down_values
        fu_values = _floor_up_values or [0.65]
        fd_values = _floor_down_values or [0.65]

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
                                    all_args.append((
                                        0.0, 0.0, tp_pct, trail_act, trail_dist,
                                        friction, capital, latency_minutes, min_time_remaining_hours,
                                        0.0, alpha_up, alpha_down, floor_up_val, floor_down_val,
                                    ))

        total_combos = len(all_args)
        print(f"\nTesting {total_combos} parameter combinations (continuous edge mode)...")
        print(f"  Alpha UP:  {au_values}")
        print(f"  Alpha DN:  {ad_values}")
        print(f"  Floor UP:  {fu_values}")
        print(f"  Floor DN:  {fd_values}")
    else:
        edge_up_values = frange(*edge_up_range)
        edge_down_values = frange(*edge_down_range)
        mmp_values = min_model_prob_values or [0.0]

        for edge_up in edge_up_values:
            for edge_down in edge_down_values:
                for tp_pct in tp_pct_values:
                    for trail_act in act_values:
                        if trail_act >= tp_pct:
                            continue
                        for trail_dist in dist_values:
                            if trail_dist >= trail_act:
                                continue
                            for mmp in mmp_values:
                                all_args.append((
                                    edge_up, edge_down, tp_pct, trail_act, trail_dist,
                                    friction, capital, latency_minutes, min_time_remaining_hours,
                                    mmp, None, None, None, None,
                                ))

        total_combos = len(all_args)
        print(f"\nTesting {total_combos} parameter combinations (legacy edge mode)...")
        print(f"  Edge UP:   {edge_up_values}")
        print(f"  Edge DOWN: {edge_down_values}")
        print(f"  Min M Prob:{[f'{v:.2f}' for v in mmp_values]}")

    print(f"  TP %:      {[f'{v*100:.0f}%' for v in tp_pct_values]}")
    print(f"  Trail Act: {[f'{v*100:.0f}%' for v in act_values]}")
    print(f"  Trail Dist:{[f'{v*100:.0f}pp' for v in dist_values]}")

    if workers <= 1:
        # Sequential: set module globals directly, no multiprocessing overhead
        global _worker_df, _worker_orderbook
        _worker_df = df
        _worker_orderbook = orderbook
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
            workers, initializer=_worker_init, initargs=(df, orderbook)
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
            }

            if enable_bootstrap:
                bootstrap = bootstrap_pnl(
                    df, params, n_bootstrap=n_bootstrap,
                    friction=friction, capital=capital, latency_minutes=latency_minutes,
                    orderbook=orderbook,
                    trail_activation=result["trail_activation"],
                    trail_distance=result["trail_distance"],
                    min_time_remaining_hours=min_time_remaining_hours,
                    min_model_prob=result["min_model_prob"],
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
                    min_model_prob=result["min_model_prob"],
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

    result = {
        "tp_pct": get_range_stats(tp_pcts),
        "trail_activation": get_range_stats(trail_acts),
        "trail_distance": get_range_stats(trail_dists),
        "n_top": n_top,
        "n_total": len(filtered),
        "use_continuous": use_continuous,
    }

    if use_continuous:
        alphas_up = [r["alpha_up"] for r in top_results]
        alphas_down = [r["alpha_down"] for r in top_results]
        floors_up = [r["floor_up"] for r in top_results]
        floors_down = [r["floor_down"] for r in top_results]
        result["alpha_up"] = get_range_stats(alphas_up)
        result["alpha_down"] = get_range_stats(alphas_down)
        result["floor_up"] = get_range_stats(floors_up)
        result["floor_down"] = get_range_stats(floors_down)
    else:
        edge_ups = [r["edge_up"] for r in top_results]
        edge_downs = [r["edge_down"] for r in top_results]
        min_model_probs = [r["min_model_prob"] for r in top_results]
        result["edge_up"] = get_range_stats(edge_ups)
        result["edge_down"] = get_range_stats(edge_downs)
        result["min_model_prob"] = get_range_stats(min_model_probs)

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
        if stability.get("alpha_up"):
            s = stability["alpha_up"]
            print(f"\n  Alpha UP (curvature):")
            print(f"    Range in top performers: {s['min']:.2f} - {s['max']:.2f}")
            print(f"    Mean: {s['mean']:.2f}, Median: {s['median']:.2f}")
            print(f"    Suggested: {s['median']:.2f}")

        if stability.get("alpha_down"):
            s = stability["alpha_down"]
            print(f"\n  Alpha DOWN (curvature):")
            print(f"    Range in top performers: {s['min']:.2f} - {s['max']:.2f}")
            print(f"    Mean: {s['mean']:.2f}, Median: {s['median']:.2f}")
            print(f"    Suggested: {s['median']:.2f}")

        if stability.get("floor_up"):
            s = stability["floor_up"]
            print(f"\n  Floor UP (min model prob):")
            print(f"    Range in top performers: {s['min']:.2f} - {s['max']:.2f}")
            print(f"    Mean: {s['mean']:.2f}, Median: {s['median']:.2f}")
            print(f"    Suggested: {s['median']:.2f}")

        if stability.get("floor_down"):
            s = stability["floor_down"]
            print(f"\n  Floor DOWN (min model prob):")
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

        if stability.get("min_model_prob"):
            s = stability["min_model_prob"]
            print(f"\n  Min Model Prob:")
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

    # Suggest "robust" middle-of-the-road parameters
    print("\n  " + "-" * 60)
    print("  SUGGESTED ROBUST PARAMETERS (median of top performers):")
    robust_tp = stability['tp_pct']['median']
    robust_trail_act = stability['trail_activation']['median'] if stability.get("trail_activation") else 0.20
    robust_trail_dist = stability['trail_distance']['median'] if stability.get("trail_distance") else 0.15

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
        robust_mmp = stability['min_model_prob']['median'] if stability.get("min_model_prob") else 0.0
        cmd = (f"    python scripts/backtest.py --edge-up {robust_edge_up:.2f} "
               f"--edge-down {robust_edge_down:.2f} "
               f"--tp {robust_tp:.2f}")
        if robust_mmp > 0:
            cmd += f" --min-model-prob {robust_mmp:.2f}"
    if stability.get("trail_activation") and stability.get("trail_distance"):
        cmd += (f" --trail-activation {robust_trail_act:.2f}"
                f" --trail-distance {robust_trail_dist:.2f}")
    print(cmd)

    # Run backtest with the robust parameters and show results
    if df is not None:
        bt_kwargs = dict(
            tp_pct=robust_tp,
            friction=friction,
            capital=capital,
            latency_minutes=latency_minutes,
            orderbook=orderbook,
            trail_activation=robust_trail_act,
            trail_distance=robust_trail_dist,
            min_time_remaining_hours=min_time_remaining_hours,
        )
        if use_continuous:
            bt_kwargs["alpha_up"] = robust_alpha_up
            bt_kwargs["alpha_down"] = robust_alpha_down
            bt_kwargs["floor_up"] = robust_floor_up
            bt_kwargs["floor_down"] = robust_floor_down
        else:
            bt_kwargs["edge_up"] = robust_edge_up
            bt_kwargs["edge_down"] = robust_edge_down
            bt_kwargs["min_model_prob"] = robust_mmp
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
                sharpe = compute_mtm_sharpe(robust_mtm_pnl)
        # Fallback to per-trade Sharpe when equity curve is too short
        if sharpe is None and trade_pnls:
            sharpe = compute_sharpe_ratio(trade_pnls)

        if sharpe is not None and duration_days > 0:
            annual_sharpe = sharpe * (365 / duration_days) ** 0.5
            sharpe_str = f"{annual_sharpe:.2f}"
        elif sharpe is not None:
            sharpe_str = f"{sharpe:.2f}"
        else:
            sharpe_str = "N/A"
        pf_str = f"{pf:.1f}" if pf is not None and pf != float('inf') else ("inf" if pf == float('inf') else "N/A")

        print(f"\n  ROBUST BACKTEST RESULT:")
        print(f"    Trades: {n_trades}, Win rate: {win_rate:.1f}%, P&L: {pnl_pct:+.1f}%")
        print(f"    Sharpe(Y): {sharpe_str}, Profit Factor: {pf_str}, Max DD (MTM): {robust_max_dd_pct:.1f}%")
        if n_trades > 0:
            avg_win = sum(t.pnl for t in trades if t.pnl > 0) / wins if wins else 0
            avg_loss = sum(t.pnl for t in trades if t.pnl < 0) / losses if losses else 0
            tp_count = sum(1 for t in trades if t.result == "TP_FILLED")
            trail_count = sum(1 for t in trades if t.result == "TRAILING_STOP")
            loss_count = sum(1 for t in trades if t.result == "LOSS_EXPIRY")
            print(f"    Avg win: +${avg_win:.2f}, Avg loss: ${avg_loss:.2f}")
            print(f"    TP: {tp_count}, Trail: {trail_count}, Loss@expiry: {loss_count}")
        if n_trades < 10:
            print(f"\n  WARNING: Only {n_trades} trades. Results may be unstable.")

    print()


from plotting import build_equity_curve, compute_max_drawdown, plot_best_pnl_curve  # noqa: E402 (after backtest imports)


def _format_result_row(r: dict, rank_str: str, duration_days: float, use_continuous: bool = False) -> str:
    """Format a single result row for the optimization table."""
    pnl_str = f"{r['pnl_pct']:+.1f}%"
    wl_str = f"{r['wins']}/{r['losses']}"
    raw_sharpe = r.get('sharpe_ratio')
    if raw_sharpe is not None and duration_days > 0:
        annual_sharpe = raw_sharpe * (365 / duration_days) ** 0.5
        sharpe_str = f"{annual_sharpe:.2f}"
    elif raw_sharpe is not None:
        sharpe_str = f"{raw_sharpe:.2f}"
    else:
        sharpe_str = "N/A"
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

    if use_continuous:
        au_str = f"{r.get('alpha_up', 0):.2f}"
        ad_str = f"{r.get('alpha_down', 0):.2f}"
        fu_str = f"{r.get('floor_up', 0):.2f}"
        fd_str = f"{r.get('floor_down', 0):.2f}"
        return (f"  {rank_str:<5} {au_str:<6} {ad_str:<6} {fu_str:<6} {fd_str:<6} "
                f"{r['tp_pct']*100:<6.0f} {ta_str:>5} {td_str:>5} {r['trades']:<8} "
                f"{wl_str:<8} {r['win_rate']:<8.1f} {pnl_str:>8} {dd_str:>8} {days_str:>6} "
                f"{trd_day_str:>6} {daily_str:>8} {sharpe_str:>9} {pf_str:>6}")
    else:
        mmp_str = f"{r.get('min_model_prob', 0):.2f}"
        return (f"  {rank_str:<5} {r['edge_up']:<9.2f} {r['edge_down']:<9.2f} "
                f"{r['tp_pct']*100:<6.0f} {ta_str:>5} {td_str:>5} {mmp_str:>5} {r['trades']:<8} "
                f"{wl_str:<8} {r['win_rate']:<8.1f} {pnl_str:>8} {dd_str:>8} {days_str:>6} "
                f"{trd_day_str:>6} {daily_str:>8} {sharpe_str:>9} {pf_str:>6}")


def print_optimization_results(
    results: list[dict],
    top_n: int = 20,
    min_trades: int = 0,
    show_robust: bool = False,
    sort_by: str = "pnl",
    duration_days: float = 0.0,
) -> None:
    """Print optimization results table."""
    # Filter by min trades
    filtered = [r for r in results if r["trades"] >= min_trades]

    # Sort by specified metric
    if sort_by == "sharpe" and any(r.get("sharpe_ratio") is not None for r in filtered):
        filtered.sort(key=lambda x: x.get("sharpe_ratio") or -999, reverse=True)
    elif sort_by == "cv" and any(r.get("cv_mean_pnl") is not None for r in filtered):
        filtered.sort(key=lambda x: x.get("cv_mean_pnl") or -999, reverse=True)
    # else keep P&L sorting

    # Detect mode from results
    use_continuous = filtered[0].get("alpha_up") is not None if filtered else False

    print("\n" + "=" * 120)
    sort_label = {"pnl": "P&L", "sharpe": "Sharpe", "cv": "CV Mean"}.get(sort_by, "P&L")
    mode_label = "continuous edge" if use_continuous else "legacy edge"
    if min_trades > 0:
        print(f"OPTIMIZATION RESULTS (Top {top_n} by {sort_label}, min {min_trades} trades, {mode_label})")
    else:
        print(f"OPTIMIZATION RESULTS (Top {top_n} by {sort_label}, {mode_label})")
    print("=" * 120)

    # Header
    if use_continuous:
        print(f"\n  {'Rank':<5} {'aUP':<6} {'aDN':<6} {'fUP':<6} {'fDN':<6} {'TP%':<6} {'TrlA':>5} {'TrlD':>5} {'Trades':<8} {'W/L':<8} {'Win%':<8} {'P&L%':>8} {'MaxDD%':>8} {'Days':>6} {'Trd/D':>6} {'Daily%':>8} {'Sharpe(Y)':>9} {'PF':>6}")
    else:
        print(f"\n  {'Rank':<5} {'Edge UP':<9} {'Edge DN':<9} {'TP%':<6} {'TrlA':>5} {'TrlD':>5} {'MnMP':>5} {'Trades':<8} {'W/L':<8} {'Win%':<8} {'P&L%':>8} {'MaxDD%':>8} {'Days':>6} {'Trd/D':>6} {'Daily%':>8} {'Sharpe(Y)':>9} {'PF':>6}")
    print("  " + "-" * 140)

    for i, r in enumerate(filtered[:top_n], 1):
        print(_format_result_row(r, str(i), duration_days, use_continuous=use_continuous))

    # Show robustness metrics if available
    if show_robust and filtered and filtered[0].get("bootstrap_mean") is not None:
        print("\n  " + "-" * 115)
        print("  BOOTSTRAP CONFIDENCE INTERVALS (90%)")
        print("  " + "-" * 115)
        print(f"  {'Rank':<5} {'Edge UP':<9} {'Edge DN':<9} {'TP%':<6} {'P&L':>10} {'Boot Mean':>12} {'Boot Std':>10} {'5th %':>10} {'95th %':>10}")
        print("  " + "-" * 115)

        for i, r in enumerate(filtered[:top_n], 1):
            if r.get("bootstrap_mean") is None:
                continue
            pnl_str = f"${r['pnl']:.2f}"
            mean_str = f"${r['bootstrap_mean']:.2f}"
            std_str = f"${r['bootstrap_std']:.2f}"
            p5_str = f"${r['bootstrap_5th']:.2f}"
            p95_str = f"${r['bootstrap_95th']:.2f}"
            print(f"  {i:<5} {r['edge_up']:<9.2f} {r['edge_down']:<9.2f} {r['tp_pct']*100:<6.0f} {pnl_str:>10} {mean_str:>12} {std_str:>10} {p5_str:>10} {p95_str:>10}")

    if show_robust and filtered and filtered[0].get("cv_mean_pnl") is not None:
        print("\n  " + "-" * 115)
        print("  PER-MARKET P&L ANALYSIS")
        print("  " + "-" * 115)
        print(f"  {'Rank':<5} {'Edge UP':<9} {'Edge DN':<9} {'TP%':<6} {'P&L':>10} {'CV Mean':>10} {'CV Std':>10} {'Worst':>10} {'Profitable':>12}")
        print("  " + "-" * 115)

        for i, r in enumerate(filtered[:top_n], 1):
            if r.get("cv_mean_pnl") is None:
                continue
            pnl_str = f"${r['pnl']:.2f}"
            cv_mean = f"${r['cv_mean_pnl']:.2f}"
            cv_std = f"${r['cv_std_pnl']:.2f}"
            worst = f"${r['cv_worst_pnl']:.2f}"
            prof = f"{r['cv_markets_profitable']}/{r['cv_total_markets']}"
            print(f"  {i:<5} {r['edge_up']:<9.2f} {r['edge_down']:<9.2f} {r['tp_pct']*100:<6.0f} {pnl_str:>10} {cv_mean:>10} {cv_std:>10} {worst:>10} {prof:>12}")

    # Show worst results
    worst = sorted([r for r in filtered if r["trades"] > 0], key=lambda x: x["pnl"])[:5]
    if worst:
        print("\n  " + "-" * 140)
        print("  Worst performers by P&L:")
        print("  " + "-" * 140)
        for r in worst:
            print(_format_result_row(r, "--", duration_days, use_continuous=use_continuous))

    # Show secondary table ranked by P&L if primary sort is not P&L
    if sort_by != "pnl" and filtered:
        pnl_sorted = sorted(filtered, key=lambda x: x["pnl_pct"], reverse=True)
        print("\n" + "=" * 120)
        if min_trades > 0:
            print(f"OPTIMIZATION RESULTS (Top {top_n} by P&L, min {min_trades} trades, {mode_label})")
        else:
            print(f"OPTIMIZATION RESULTS (Top {top_n} by P&L, {mode_label})")
        print("=" * 120)
        if use_continuous:
            print(f"\n  {'Rank':<5} {'aUP':<6} {'aDN':<6} {'fUP':<6} {'fDN':<6} {'TP%':<6} {'TrlA':>5} {'TrlD':>5} {'Trades':<8} {'W/L':<8} {'Win%':<8} {'P&L%':>8} {'MaxDD%':>8} {'Days':>6} {'Trd/D':>6} {'Daily%':>8} {'Sharpe(Y)':>9} {'PF':>6}")
        else:
            print(f"\n  {'Rank':<5} {'Edge UP':<9} {'Edge DN':<9} {'TP%':<6} {'TrlA':>5} {'TrlD':>5} {'MnMP':>5} {'Trades':<8} {'W/L':<8} {'Win%':<8} {'P&L%':>8} {'MaxDD%':>8} {'Days':>6} {'Trd/D':>6} {'Daily%':>8} {'Sharpe(Y)':>9} {'PF':>6}")
        print("  " + "-" * 140)
        for i, r in enumerate(pnl_sorted[:top_n], 1):
            print(_format_result_row(r, str(i), duration_days, use_continuous=use_continuous))

    print("\n" + "=" * 144)

    # Print best parameters with warnings
    if filtered:
        best = filtered[0]
        print(f"\n  BEST PARAMETERS (by {sort_label}):")
        print(f"  " + "-" * 70)
        trail_flag = f" --trail-activation {best['trail_activation']:.2f} --trail-distance {best['trail_distance']:.2f}"
        if use_continuous:
            print(f"    python scripts/backtest.py --alpha-up {best['alpha_up']:.2f} --alpha-down {best['alpha_down']:.2f} --floor-up {best['floor_up']:.2f} --floor-down {best['floor_down']:.2f} --tp {best['tp_pct']:.2f}{trail_flag}")
        else:
            mmp_flag = f" --min-model-prob {best.get('min_model_prob', 0):.2f}" if best.get('min_model_prob', 0) > 0 else ""
            print(f"    python scripts/backtest.py --edge-up {best['edge_up']:.2f} --edge-down {best['edge_down']:.2f} --tp {best['tp_pct']:.2f}{trail_flag}{mmp_flag}")
        print(f"    Trades: {best['trades']}, Win rate: {best['win_rate']:.1f}%, P&L: {best['pnl_pct']:+.1f}%")

        if best.get("sharpe_ratio") is not None:
            pf_val = best.get('profit_factor')
            pf_disp = f"{pf_val:.1f}" if pf_val is not None and pf_val != float('inf') else ("inf" if pf_val == float('inf') else "N/A")
            dd_disp = f"{best.get('max_dd_pct', 0):.1f}%"
            best_sharpe = best['sharpe_ratio']
            if duration_days > 0:
                best_sharpe = best_sharpe * (365 / duration_days) ** 0.5
            print(f"    Sharpe(Y): {best_sharpe:.2f}, Profit Factor: {pf_disp}, Max DD (MTM): {dd_disp}")

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
    """Load config_optimize.json and config_dry_run.json for default parameters."""
    project_root = Path(__file__).parent.parent

    opt_config = {}
    opt_path = project_root / "config_optimize.json"
    if opt_path.exists():
        with open(opt_path, "r") as f:
            opt_config = json.load(f)

    dry_config = {}
    dry_path = project_root / "config_dry_run.json"
    if dry_path.exists():
        with open(dry_path, "r") as f:
            dry_config = json.load(f)

    return opt_config, dry_config


def main():
    # Load config for defaults
    opt_config, dry_config = load_config()

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
        default=dry_config.get("min_time_remaining_hours", 2.0),
        help="Min hours before expiry to allow entry (default: from config_dry_run.json)",
    )
    mmp_default = opt_config.get("min_model_prob", [0.0, 0.1, 0.2, 0.3, 0.4])
    if isinstance(mmp_default, (int, float)):
        mmp_default = [mmp_default]
    parser.add_argument(
        "--min-model-prob",
        type=float,
        nargs="+",
        default=mmp_default,
        help="Min model prob values to grid search (default: 0.0 0.1 0.2 0.3 0.4, legacy mode)",
    )

    # Continuous edge mode parameters (per-direction)
    alpha_up_default = opt_config.get("alpha_up", opt_config.get("alpha", None))
    if isinstance(alpha_up_default, (int, float)):
        alpha_up_default = [alpha_up_default]
    parser.add_argument(
        "--alpha-up",
        type=float,
        nargs="+",
        default=alpha_up_default,
        help="Alpha UP values to grid search (enables continuous edge mode). E.g.: 1.0 1.2 1.5 2.0",
    )
    alpha_down_default = opt_config.get("alpha_down", opt_config.get("alpha", None))
    if isinstance(alpha_down_default, (int, float)):
        alpha_down_default = [alpha_down_default]
    parser.add_argument(
        "--alpha-down",
        type=float,
        nargs="+",
        default=alpha_down_default,
        help="Alpha DOWN values to grid search. Defaults to same as --alpha-up if not set.",
    )
    floor_up_default = opt_config.get("floor_up", opt_config.get("floor", [0.50, 0.55, 0.60, 0.65, 0.70, 0.75]))
    if isinstance(floor_up_default, (int, float)):
        floor_up_default = [floor_up_default]
    parser.add_argument(
        "--floor-up",
        type=float,
        nargs="+",
        default=floor_up_default,
        help="Floor UP values to grid search (default: 0.50 0.55 0.60 0.65 0.70 0.75)",
    )
    floor_down_default = opt_config.get("floor_down", opt_config.get("floor", [0.50, 0.55, 0.60, 0.65, 0.70, 0.75]))
    if isinstance(floor_down_default, (int, float)):
        floor_down_default = [floor_down_default]
    parser.add_argument(
        "--floor-down",
        type=float,
        nargs="+",
        default=floor_down_default,
        help="Floor DOWN values to grid search (default: 0.50 0.55 0.60 0.65 0.70 0.75)",
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

    # Normalize scalar to list for trail params and min_model_prob
    trail_act_values = args.trail_activation if isinstance(args.trail_activation, list) else [args.trail_activation]
    trail_dist_values = args.trail_distance if isinstance(args.trail_distance, list) else [args.trail_distance]
    mmp_values = args.min_model_prob if isinstance(args.min_model_prob, list) else [args.min_model_prob]

    # Normalize per-direction alpha/floor
    alpha_up_values = args.alpha_up if isinstance(args.alpha_up, list) else ([args.alpha_up] if args.alpha_up is not None else None)
    alpha_down_values = args.alpha_down if isinstance(args.alpha_down, list) else ([args.alpha_down] if args.alpha_down is not None else None)
    # If alpha_down not specified but alpha_up is, default alpha_down to alpha_up values
    if alpha_up_values is not None and alpha_down_values is None:
        alpha_down_values = alpha_up_values
    floor_up_values = args.floor_up if isinstance(args.floor_up, list) else [args.floor_up]
    floor_down_values = args.floor_down if isinstance(args.floor_down, list) else [args.floor_down]

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
        min_model_prob_values=mmp_values,
        alpha_up_values=alpha_up_values,
        alpha_down_values=alpha_down_values,
        floor_up_values=floor_up_values,
        floor_down_values=floor_down_values,
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
    )

    # Plot PnL curve for rank 1 result
    filtered = [r for r in results if r["trades"] >= args.min_trades]
    if filtered:
        sort_by = args.sort_by
        if sort_by == "sharpe" and any(r.get("sharpe_ratio") is not None for r in filtered):
            filtered.sort(key=lambda x: x.get("sharpe_ratio") or -999, reverse=True)
        elif sort_by == "cv" and any(r.get("cv_mean_pnl") is not None for r in filtered):
            filtered.sort(key=lambda x: x.get("cv_mean_pnl") or -999, reverse=True)

        best = filtered[0]
        best_trades, _ = run_backtest(
            df,
            edge_up=best["edge_up"],
            edge_down=best["edge_down"],
            tp_pct=best["tp_pct"],
            friction=args.friction,
            capital=args.capital,
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
        )
        plot_best_pnl_curve(
            best_trades,
            best,
            args.capital,
            save_path="results/pnl_curve.png",
            df=df,
            friction=args.friction,
            orderbook=orderbook,
        )

    return 0


if __name__ == "__main__":
    exit(main())
