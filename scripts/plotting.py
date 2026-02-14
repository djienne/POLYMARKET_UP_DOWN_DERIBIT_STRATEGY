"""Shared plotting utilities for backtest and optimizer."""

from datetime import datetime, timezone
from pathlib import Path

from backtest import group_markets, get_orderbook_prices


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
        barrier = trade.barrier_price
        market_rows = markets.get(barrier, [])
        direction = trade.direction  # "UP" or "DOWN"

        for row in market_rows:
            row_ts = row["timestamp"]
            if row_ts <= trade.entry_timestamp or row_ts >= trade.exit_timestamp:
                continue

            # Get current bid price for MTM
            bid, _ = get_orderbook_prices(orderbook, row_ts, direction, barrier_price=barrier)
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


def _build_prob_segments(sorted_trades, df):
    """Build chronologically-ordered probability segments for ALL markets.

    Shows probability data for every barrier (not just traded ones) so the
    subplots have continuous coverage without gaps.

    Returns list of (ts_list, model_up, market_up, model_down, market_down, trade_list) tuples.
    Each segment contains both UP and DOWN probabilities for one barrier.
    """
    markets = group_markets(df)

    barrier_trades: dict[float, list] = {}
    for t in sorted_trades:
        barrier_trades.setdefault(t.barrier_price, []).append(t)

    segments = []
    for barrier in sorted(markets.keys(),
                          key=lambda b: markets[b][0]["timestamp"]):
        rows = markets[barrier]
        if not rows:
            continue
        trade_list = barrier_trades.get(barrier, [])

        ts_list = [datetime.fromisoformat(r["timestamp"].replace("Z", "+00:00")) for r in rows]
        model_up = [r.get("avg_prob_up", r.get("model_prob_up", 0)) for r in rows]
        market_up = [r["poly_prob_up"] for r in rows]
        model_down = [r.get("avg_prob_down", r.get("model_prob_down", 0)) for r in rows]
        market_down = [r["poly_prob_down"] for r in rows]
        segments.append((ts_list, model_up, market_up, model_down, market_down, trade_list))

    return segments


def _render_pnl_chart(sorted_trades, cum_pnl, close_timestamps, wins, losses,
                      params, capital, df, friction, orderbook,
                      save_path, xlim=None, title_suffix=""):
    """Core rendering logic for the PnL chart. Shared by full and zoom views.

    Args:
        xlim: Optional (xmin, xmax) datetime tuple to zoom the time axis.
        title_suffix: Optional string appended to the title (e.g., " (zoom)").
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    has_prob_data = (hasattr(sorted_trades[0], "model_prob")
                     and sorted_trades[0].model_prob > 0
                     and df is not None)
    n_rows = 3 if has_prob_data else 1
    height_ratios = [3, 1, 1] if has_prob_data else [1]

    fig, axes = plt.subplots(n_rows, 1, figsize=(12, 5 + (4 if has_prob_data else 0)),
                             sharex=True, gridspec_kw={"height_ratios": height_ratios})
    ax = axes[0] if has_prob_data else axes

    # Plot MTM equity curve if raw data is available, else fallback to close-only line
    max_dd_info = None
    if df is not None:
        mtm_ts, mtm_pnl = build_equity_curve(sorted_trades, df, friction, orderbook)
        if mtm_ts:
            ax.plot(mtm_ts, mtm_pnl, color="steelblue", linewidth=1.2, zorder=2, label="_nolegend_")
            dd = compute_max_drawdown(mtm_pnl, mtm_ts, capital=capital)
            if dd["max_dd"] < 0:
                max_dd_info = dd
                peak_ts = dd["peak_ts"]
                trough_ts = dd["trough_ts"]
                peak_val = mtm_pnl[dd["peak_idx"]]
                trough_val = mtm_pnl[dd["trough_idx"]]
                # Only draw DD shading if it falls within the view
                if xlim is None or (peak_ts >= xlim[0] and trough_ts <= xlim[1]):
                    ax.fill_between(
                        [peak_ts, trough_ts], peak_val, trough_val,
                        alpha=0.15, color="red", zorder=1,
                    )
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

    # Build title
    p = params
    trail_act = p.get('trail_activation', 0)
    trail_dist = p.get('trail_distance', 0)
    if p.get('alpha_up') is not None:
        title_line1 = (
            f"aUP={p['alpha_up']:.2f} aDN={p['alpha_down']:.2f} "
            f"fUP={p['floor_up']:.2f} fDN={p['floor_down']:.2f}  "
            f"tp={p['tp_pct']*100:.0f}%  trail={trail_act*100:.0f}%/{trail_dist*100:.0f}pp"
            f"{title_suffix}"
        )
    else:
        title_line1 = (
            f"edge_up={p['edge_up']:.2f}  edge_down={p['edge_down']:.2f}  "
            f"tp={p['tp_pct']*100:.0f}%  trail={trail_act*100:.0f}%/{trail_dist*100:.0f}pp"
            f"{title_suffix}"
        )
    n_trades = len(sorted_trades)
    wins_n = sum(1 for t in sorted_trades if t.pnl > 0)
    win_rate = (wins_n / n_trades * 100) if n_trades > 0 else 0
    pnl_pct = p.get('pnl_pct', 0)
    dd_pct = max_dd_info['max_dd_pct'] if max_dd_info else 0
    sharpe = p.get('sharpe_ratio')
    sharpe_str = f"{sharpe:.2f}" if sharpe is not None else "N/A"
    title_line2 = (
        f"Trades: {n_trades}  W/L: {wins_n}/{n_trades - wins_n}  "
        f"Win: {win_rate:.0f}%  P&L: {pnl_pct:+.1f}%  "
        f"MaxDD: {dd_pct:.1f}%  Sharpe: {sharpe_str}"
    )
    fig.suptitle(title_line1, fontsize=11, fontweight="bold", y=0.99)
    ax.set_title(title_line2, fontsize=9, color="grey")
    ax.legend(loc="best")

    # Bottom subplots: UP and DOWN probabilities in separate panels
    if has_prob_data:
        ax_up = axes[1]
        ax_dn = axes[2]
        segments = _build_prob_segments(sorted_trades, df)

        for i, (ts_list, model_up, market_up, model_down, market_down, trade_list) in enumerate(segments):
            lbl_model = "Model" if i == 0 else "_nolegend_"
            lbl_poly = "Polymarket" if i == 0 else "_nolegend_"

            ax_up.plot(ts_list, model_up, linewidth=1.0, color="#4CAF50", label=lbl_model, zorder=2)
            ax_up.plot(ts_list, market_up, linewidth=1.0, color="#FF9800", label=lbl_poly, zorder=2)

            ax_dn.plot(ts_list, model_down, linewidth=1.0, color="#4CAF50", label=lbl_model, zorder=2)
            ax_dn.plot(ts_list, market_down, linewidth=1.0, color="#FF9800", label=lbl_poly, zorder=2)

            # Mark trade entries on the subplot matching direction
            for t in trade_list:
                entry_dt = datetime.fromisoformat(t.entry_timestamp.replace("Z", "+00:00"))
                marker_color = "#4CAF50" if t.pnl > 0 else "#F44336"
                target_ax = ax_up if t.direction == "UP" else ax_dn
                target_ax.axvline(entry_dt, color=marker_color, alpha=0.25, linewidth=3, zorder=0)
                target_ax.scatter([entry_dt], [t.model_prob * 100], marker="D", s=40,
                                  color="#2E7D32", edgecolors="white", linewidths=0.5, zorder=4)
                target_ax.scatter([entry_dt], [t.market_prob * 100], marker="s", s=40,
                                  color="#E65100", edgecolors="white", linewidths=0.5, zorder=4)

        ax_up.set_ylabel("UP Prob (%)")
        ax_dn.set_ylabel("DOWN Prob (%)")
        ax_dn.set_xlabel("Time")
        ax_up.legend(loc="lower left", fontsize=7)
        ax_dn.legend(loc="lower left", fontsize=7)
        ax_up.grid(True, alpha=0.3)
        ax_dn.grid(True, alpha=0.3)

    # Apply zoom if requested
    if xlim is not None:
        ax.set_xlim(xlim)

    fig.autofmt_xdate()
    fig.tight_layout()
    fig.subplots_adjust(top=0.88)

    save = Path(save_path)
    if not save.is_absolute():
        save = Path(__file__).parent.parent / save_path
    save.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(save), dpi=150)
    plt.close(fig)
    print(f"  PnL curve saved to: {save}")
    if max_dd_info:
        print(f"  Max MTM drawdown: {max_dd_info['max_dd_pct']:.1f}%")


def _render_prob_chart(sorted_trades, df, save_path, xlim=None):
    """Standalone probability chart with model vs market probabilities."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    segments = _build_prob_segments(sorted_trades, df)
    if not segments:
        return

    fig, (ax_up, ax_dn) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    for i, (ts_list, model_up, market_up, model_down, market_down, trade_list) in enumerate(segments):
        lbl_model = "Model" if i == 0 else "_nolegend_"
        lbl_poly = "Polymarket" if i == 0 else "_nolegend_"

        # UP probabilities
        ax_up.plot(ts_list, model_up, linewidth=1.2, color="#4CAF50",
                   label=lbl_model, zorder=2)
        ax_up.plot(ts_list, market_up, linewidth=1.2, color="#FF9800",
                   label=lbl_poly, zorder=2)
        ax_up.fill_between(ts_list, model_up, market_up, alpha=0.08, color="#4CAF50", zorder=1)

        # DOWN probabilities
        ax_dn.plot(ts_list, model_down, linewidth=1.2, color="#4CAF50",
                   label=lbl_model, zorder=2)
        ax_dn.plot(ts_list, market_down, linewidth=1.2, color="#FF9800",
                   label=lbl_poly, zorder=2)
        ax_dn.fill_between(ts_list, model_down, market_down, alpha=0.08, color="#FF9800", zorder=1)

        # Dashed bridge to next segment
        if i + 1 < len(segments):
            next_ts, next_mu, next_pu, next_md, next_pd, _ = segments[i + 1]
            ax_up.plot([ts_list[-1], next_ts[0]], [model_up[-1], next_mu[0]],
                       linewidth=0.8, linestyle=":", color="#4CAF50", alpha=0.4, zorder=1)
            ax_up.plot([ts_list[-1], next_ts[0]], [market_up[-1], next_pu[0]],
                       linewidth=0.8, linestyle=":", color="#FF9800", alpha=0.4, zorder=1)
            ax_dn.plot([ts_list[-1], next_ts[0]], [model_down[-1], next_md[0]],
                       linewidth=0.8, linestyle=":", color="#4CAF50", alpha=0.4, zorder=1)
            ax_dn.plot([ts_list[-1], next_ts[0]], [market_down[-1], next_pd[0]],
                       linewidth=0.8, linestyle=":", color="#FF9800", alpha=0.4, zorder=1)

        # Mark trade entries with annotations on matching subplot
        for t in trade_list:
            entry_dt = datetime.fromisoformat(t.entry_timestamp.replace("Z", "+00:00"))
            marker_color = "#4CAF50" if t.pnl > 0 else "#F44336"
            result_short = {"TP_FILLED": "TP", "TRAILING_STOP": "TRAIL",
                            "WIN_EXPIRY": "WIN", "LOSS_EXPIRY": "LOSS"}.get(t.result, t.result)
            pnl_str = f"+${t.pnl:.2f}" if t.pnl >= 0 else f"-${abs(t.pnl):.2f}"

            target_ax = ax_up if t.direction == "UP" else ax_dn
            target_ax.axvline(entry_dt, color=marker_color, alpha=0.20, linewidth=6, zorder=0)
            target_ax.scatter([entry_dt], [t.model_prob * 100], marker="D", s=60,
                              color="#2E7D32", edgecolors="white", linewidths=0.8, zorder=4)
            target_ax.scatter([entry_dt], [t.market_prob * 100], marker="s", s=60,
                              color="#E65100", edgecolors="white", linewidths=0.8, zorder=4)

            # Annotation: alternate above/below based on trade index to avoid overlap
            ann_idx = len([tt for tt in sorted_trades
                          if tt.entry_timestamp <= t.entry_timestamp])
            above = (ann_idx % 2 == 1)
            y_anchor = t.model_prob * 100 if above else t.market_prob * 100
            y_offset = 14 if above else -14
            va = "bottom" if above else "top"

            edge_str = f"{t.edge:.2f}x"
            target_ax.annotate(
                f"{edge_str}  {result_short} {pnl_str}",
                xy=(entry_dt, y_anchor),
                xytext=(0, y_offset), textcoords="offset points",
                fontsize=7, fontweight="bold", color=marker_color,
                ha="center", va=va, zorder=5,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=marker_color, alpha=0.85),
            )

    if xlim is not None:
        ax_up.set_xlim(xlim)

    ax_up.set_ylabel("UP Prob (%)", fontsize=11)
    ax_dn.set_ylabel("DOWN Prob (%)", fontsize=11)
    ax_dn.set_xlabel("Time", fontsize=11)
    fig.suptitle("Model vs Polymarket Probabilities at Entry", fontsize=13, fontweight="bold")
    ax_up.legend(loc="lower left", fontsize=8)
    ax_dn.legend(loc="lower left", fontsize=8)
    ax_up.grid(True, alpha=0.3)
    ax_dn.grid(True, alpha=0.3)
    ax_dn.xaxis.set_major_formatter(mdates.DateFormatter("%b %d %H:%M"))
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.subplots_adjust(top=0.93)

    save = Path(save_path)
    if not save.is_absolute():
        save = Path(__file__).parent.parent / save_path
    save.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(save), dpi=150)
    plt.close(fig)
    print(f"  Probabilities chart saved to: {save}")


def plot_best_pnl_curve(trades, params, capital, save_path="results/pnl_curve.png",
                        df=None, friction=0.015, orderbook=None,
                        zoom_start: str | None = None):
    """Plot cumulative PnL over time for the best parameter combination.

    If df is provided, plots a continuous mark-to-market equity curve with
    unrealized P&L during open positions and flat lines between trades.
    Otherwise falls back to trade-close-only staircase plot.

    Args:
        zoom_start: Optional ISO date string (e.g., "2026-02-07") to also
            generate a zoomed-in version saved as *_zoom.png.
    """
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

    # Full chart
    _render_pnl_chart(sorted_trades, cum_pnl, close_timestamps, wins, losses,
                      params, capital, df, friction, orderbook, save_path)

    # Zoomed chart
    if zoom_start is not None:
        zoom_dt = datetime.fromisoformat(zoom_start).replace(tzinfo=timezone.utc)

        # Find the end of the data
        all_exit_dts = [datetime.fromisoformat(t.exit_timestamp.replace("Z", "+00:00"))
                        for t in sorted_trades]
        if df:
            last_data = max(r["timestamp"] for r in df)
            all_exit_dts.append(datetime.fromisoformat(last_data.replace("Z", "+00:00")))
        zoom_end = max(all_exit_dts)

        zoom_path = save_path.replace(".png", "_zoom.png")
        _render_pnl_chart(sorted_trades, cum_pnl, close_timestamps, wins, losses,
                          params, capital, df, friction, orderbook, zoom_path,
                          xlim=(zoom_dt, zoom_end), title_suffix="  (zoom)")

        # Standalone probabilities chart (zoomed range)
        has_prob = hasattr(sorted_trades[0], "model_prob") and sorted_trades[0].model_prob > 0
        if has_prob and df is not None:
            prob_path = save_path.replace(".png", "_probabilities.png")
            _render_prob_chart(sorted_trades, df, prob_path, xlim=(zoom_dt, zoom_end))
