#!/usr/bin/env python3
"""
Generate an HTML dashboard showing trading parameters, edge curves,
probability thresholds, and model vs Polymarket comparison.

Usage:
    python scripts/generate_dashboard.py
    python scripts/generate_dashboard.py --output results/dashboard.html
"""

import csv
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from btc_pricer.edge import required_model_prob


def load_config(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def load_state(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def load_probabilities_csv(path: Path, max_rows: int = 2500) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows[-max_rows:]


def generate_edge_curve_data(alpha: float, floor: float) -> list[dict]:
    """Generate edge curve data points for plotting."""
    points = []
    for i in range(101):
        p = i / 100.0
        req = required_model_prob(p, alpha, floor)
        edge_ratio = req / p if p > 0 else float("inf")
        points.append({"market_prob": p, "required_prob": req, "edge_ratio": edge_ratio})
    return points


def build_html(config: dict, opt_config: dict, state: dict, prob_rows: list[dict]) -> str:
    # Extract config values
    alpha_up = config.get("edge_alpha_up", 1.5)
    alpha_down = config.get("edge_alpha_down", 1.5)
    floor_up = config.get("edge_floor_up", 0.65)
    floor_down = config.get("edge_floor_down", 0.35)
    tp_pct = config.get("tp_percentage", 0.25)
    trail_act = config.get("trail_activation", 0.0)
    trail_dist = config.get("trail_distance", 0.0)
    order_size = config.get("order_size_pct", 0.10)
    starting_cap = config.get("starting_capital", 100.0)
    grace_hrs = config.get("grace_period_hours", 1.0)
    min_time = config.get("min_time_remaining_hours", 1.0)
    edge_interval = config.get("edge_check_interval_seconds", 300)
    check_interval = config.get("check_interval_seconds", 10)

    # State
    capital = state.get("capital", {})
    current_capital = capital.get("current", starting_cap)
    total_pnl = capital.get("total_pnl", 0.0)
    open_positions = state.get("open_positions", [])
    closed_positions = state.get("closed_positions", [])
    current_market = state.get("current_market", {})

    # Generate edge curve JSON for Chart.js
    up_curve = generate_edge_curve_data(alpha_up, floor_up)
    down_curve = generate_edge_curve_data(alpha_down, floor_down)

    up_curve_json = json.dumps([{"x": p["market_prob"] * 100, "y": p["required_prob"] * 100} for p in up_curve])
    down_curve_json = json.dumps([{"x": p["market_prob"] * 100, "y": p["required_prob"] * 100} for p in down_curve])

    # Edge ratio table: sample points
    sample_probs = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95]
    edge_table_rows = ""
    for p in sample_probs:
        req_up = required_model_prob(p, alpha_up, floor_up)
        req_down = required_model_prob(p, alpha_down, floor_down)
        ratio_up = req_up / p if p > 0 else 0
        ratio_down = req_down / p if p > 0 else 0
        edge_table_rows += f"""
        <tr>
            <td>{p*100:.0f}%</td>
            <td>{req_up*100:.1f}%</td>
            <td>{ratio_up:.2f}x</td>
            <td class="{'edge-pass' if ratio_up > 1 else 'edge-fail'}">{ratio_up > 1}</td>
            <td>{req_down*100:.1f}%</td>
            <td>{ratio_down:.2f}x</td>
            <td class="{'edge-pass' if ratio_down > 1 else 'edge-fail'}">{ratio_down > 1}</td>
        </tr>"""

    # Probability history for chart (last 200 points)
    recent_probs = prob_rows[-200:]
    prob_timestamps = json.dumps([r["timestamp"][:16] for r in recent_probs])
    model_up_data = json.dumps([float(r.get("model_prob_up", 0)) for r in recent_probs])
    model_down_data = json.dumps([float(r.get("model_prob_down", 0)) for r in recent_probs])
    poly_up_data = json.dumps([float(r.get("poly_prob_up", 0)) for r in recent_probs])
    poly_down_data = json.dumps([float(r.get("poly_prob_down", 0)) for r in recent_probs])
    edge_up_data = json.dumps([float(r.get("edge_up", 0)) for r in recent_probs])
    edge_down_data = json.dumps([float(r.get("edge_down", 0)) for r in recent_probs])
    spot_data = json.dumps([float(r.get("spot_price", 0)) for r in recent_probs])
    barrier_data = json.dumps([float(r.get("barrier_price", 0)) for r in recent_probs])

    # BL data if available
    has_bl = any(r.get("bl_prob_up", "") not in ("", "0") for r in recent_probs)
    bl_up_data = json.dumps([float(r.get("bl_prob_up", 0) or 0) for r in recent_probs])
    bl_down_data = json.dumps([float(r.get("bl_prob_down", 0) or 0) for r in recent_probs])

    # Closed trades table
    trades_rows = ""
    for t in closed_positions:
        pnl = t.get("pnl", 0)
        pnl_pct = t.get("pnl_pct", 0)
        if isinstance(pnl_pct, str):
            pnl_pct = float(pnl_pct.rstrip("%"))
        pnl_class = "pnl-positive" if pnl >= 0 else "pnl-negative"
        opened = t.get("opened_at", "")[:16]
        closed = t.get("closed_at", "")[:16]
        trades_rows += f"""
        <tr>
            <td>{t.get('direction', '')}</td>
            <td>${t.get('barrier_price', 0):,.2f}</td>
            <td>${t.get('entry_price', 0):.4f}</td>
            <td>${t.get('exit_price', 0):.4f}</td>
            <td>{t.get('shares', 0):.2f}</td>
            <td class="{pnl_class}">${pnl:+.2f}</td>
            <td class="{pnl_class}">{pnl_pct:+.1f}%</td>
            <td>{t.get('result', '')}</td>
            <td>{opened}</td>
            <td>{closed}</td>
        </tr>"""

    # Open positions
    open_rows = ""
    for p in open_positions:
        opened = p.get("opened_at", "")[:16]
        open_rows += f"""
        <tr>
            <td>{p.get('direction', '')}</td>
            <td>${p.get('barrier_price', 0):,.2f}</td>
            <td>${p.get('entry_price', 0):.4f}</td>
            <td>${p.get('tp_price', 0):.4f}</td>
            <td>{p.get('shares', 0):.2f}</td>
            <td>${p.get('cost_basis', 0):.2f}</td>
            <td>{opened}</td>
        </tr>"""

    # Optimizer grid ranges
    opt_alpha_up = opt_config.get("alpha_up", [])
    opt_alpha_down = opt_config.get("alpha_down", [])
    opt_floor_up = opt_config.get("floor_up", [])
    opt_floor_down = opt_config.get("floor_down", [])
    opt_tp = [opt_config.get("tp_min", 0.05), opt_config.get("tp_max", 0.40), opt_config.get("tp_step", 0.05)]
    opt_trail_act = opt_config.get("trail_activation", [])
    opt_trail_dist = opt_config.get("trail_distance", [])

    mode = "LIVE" if not config.get("dry_run", True) else "DRY-RUN"
    pnl_class = "pnl-positive" if total_pnl >= 0 else "pnl-negative"
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>BTC Pricer Trading Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<style>
  :root {{
    --bg: #0d1117; --bg2: #161b22; --bg3: #21262d;
    --border: #30363d; --text: #e6edf3; --text2: #8b949e;
    --blue: #58a6ff; --green: #3fb950; --red: #f85149;
    --orange: #d29922; --purple: #bc8cff; --cyan: #39d2c0;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
    background: var(--bg); color: var(--text); padding: 24px;
    line-height: 1.5;
  }}
  h1 {{ font-size: 1.8em; margin-bottom: 4px; }}
  h2 {{ font-size: 1.3em; margin: 24px 0 12px; color: var(--blue); }}
  h3 {{ font-size: 1.1em; margin: 16px 0 8px; color: var(--text2); }}
  .subtitle {{ color: var(--text2); font-size: 0.9em; margin-bottom: 20px; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 16px; margin-bottom: 24px; }}
  .card {{
    background: var(--bg2); border: 1px solid var(--border);
    border-radius: 8px; padding: 16px;
  }}
  .card-title {{ color: var(--text2); font-size: 0.85em; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 8px; }}
  .card-value {{ font-size: 1.6em; font-weight: 600; }}
  .card-detail {{ color: var(--text2); font-size: 0.85em; margin-top: 4px; }}
  .badge {{
    display: inline-block; padding: 2px 8px; border-radius: 12px;
    font-size: 0.75em; font-weight: 600;
  }}
  .badge-live {{ background: rgba(63, 185, 80, 0.2); color: var(--green); }}
  .badge-dry {{ background: rgba(210, 153, 34, 0.2); color: var(--orange); }}
  table {{
    width: 100%; border-collapse: collapse; background: var(--bg2);
    border: 1px solid var(--border); border-radius: 8px; overflow: hidden;
    margin-bottom: 16px; font-size: 0.9em;
  }}
  th {{
    background: var(--bg3); color: var(--text2); padding: 10px 12px;
    text-align: left; font-weight: 600; font-size: 0.85em;
    text-transform: uppercase; letter-spacing: 0.03em;
  }}
  td {{ padding: 8px 12px; border-top: 1px solid var(--border); }}
  tr:hover {{ background: rgba(88, 166, 255, 0.04); }}
  .pnl-positive {{ color: var(--green); font-weight: 600; }}
  .pnl-negative {{ color: var(--red); font-weight: 600; }}
  .edge-pass {{ color: var(--green); }}
  .edge-fail {{ color: var(--red); }}
  .chart-container {{ background: var(--bg2); border: 1px solid var(--border); border-radius: 8px; padding: 16px; margin-bottom: 16px; }}
  .two-col {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }}
  @media (max-width: 768px) {{ .two-col {{ grid-template-columns: 1fr; }} }}
  .param-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 8px; }}
  .param-item {{ background: var(--bg3); border-radius: 6px; padding: 10px 12px; }}
  .param-label {{ color: var(--text2); font-size: 0.8em; }}
  .param-value {{ font-size: 1.1em; font-weight: 600; margin-top: 2px; }}
  .formula {{ background: var(--bg3); border-left: 3px solid var(--blue); padding: 12px 16px; border-radius: 0 6px 6px 0; font-family: 'Fira Code', monospace; margin: 8px 0 16px; }}
  footer {{ text-align: center; color: var(--text2); font-size: 0.8em; margin-top: 32px; padding-top: 16px; border-top: 1px solid var(--border); }}
</style>
</head>
<body>

<h1>BTC Pricer Trading Dashboard
  <span class="badge {'badge-live' if mode == 'LIVE' else 'badge-dry'}">{mode}</span>
</h1>
<div class="subtitle">Generated: {now}</div>

<!-- Summary Cards -->
<div class="grid">
  <div class="card">
    <div class="card-title">Capital</div>
    <div class="card-value">${current_capital:.2f}</div>
    <div class="card-detail">Starting: ${capital.get('starting', starting_cap):.2f}</div>
  </div>
  <div class="card">
    <div class="card-title">Total P&L</div>
    <div class="card-value {pnl_class}">${total_pnl:+.2f}</div>
    <div class="card-detail">{len(closed_positions)} closed trades</div>
  </div>
  <div class="card">
    <div class="card-title">Open Positions</div>
    <div class="card-value">{len(open_positions)}</div>
    <div class="card-detail">Market: {current_market.get('title', 'None')[:40]}</div>
  </div>
  <div class="card">
    <div class="card-title">Current Barrier</div>
    <div class="card-value">${current_market.get('barrier_price', 0):,.2f}</div>
    <div class="card-detail">Edge check every {edge_interval}s</div>
  </div>
</div>

<!-- Edge Parameters -->
<h2>Edge Entry Parameters</h2>

<div class="formula">
  required_model_prob(p) = max(floor, 1 &minus; (1 &minus; p)<sup>&alpha;</sup>)
  &nbsp;&nbsp;|&nbsp;&nbsp; Entry when model_prob &ge; required
</div>

<div class="two-col">
  <div>
    <h3>UP Direction</h3>
    <div class="param-grid">
      <div class="param-item"><div class="param-label">Alpha (&alpha;)</div><div class="param-value">{alpha_up}</div></div>
      <div class="param-item"><div class="param-label">Floor</div><div class="param-value">{floor_up*100:.0f}%</div></div>
    </div>
  </div>
  <div>
    <h3>DOWN Direction</h3>
    <div class="param-grid">
      <div class="param-item"><div class="param-label">Alpha (&alpha;)</div><div class="param-value">{alpha_down}</div></div>
      <div class="param-item"><div class="param-label">Floor</div><div class="param-value">{floor_down*100:.0f}%</div></div>
    </div>
  </div>
</div>

<h3>Exit & Risk Management</h3>
<div class="param-grid">
  <div class="param-item"><div class="param-label">Take Profit</div><div class="param-value">{tp_pct*100:.0f}%</div></div>
  <div class="param-item"><div class="param-label">Trail Activation</div><div class="param-value">{trail_act*100:.0f}%</div></div>
  <div class="param-item"><div class="param-label">Trail Distance</div><div class="param-value">{trail_dist*100:.0f}pp</div></div>
  <div class="param-item"><div class="param-label">Order Size</div><div class="param-value">{order_size*100:.0f}% of capital</div></div>
  <div class="param-item"><div class="param-label">Grace Period</div><div class="param-value">{grace_hrs}h</div></div>
  <div class="param-item"><div class="param-label">Min Time Remaining</div><div class="param-value">{min_time}h</div></div>
</div>

<!-- Edge Curve Chart -->
<h2>Edge Curves</h2>
<div class="chart-container">
  <canvas id="edgeCurveChart" height="100"></canvas>
</div>

<!-- Edge Ratio Table -->
<h2>Required Probability & Edge Ratio Table</h2>
<table>
  <thead>
    <tr>
      <th rowspan="2">Poly Prob</th>
      <th colspan="3" style="text-align:center; border-left: 1px solid var(--border);">UP (&alpha;={alpha_up}, floor={floor_up*100:.0f}%)</th>
      <th colspan="3" style="text-align:center; border-left: 1px solid var(--border);">DOWN (&alpha;={alpha_down}, floor={floor_down*100:.0f}%)</th>
    </tr>
    <tr>
      <th style="border-left: 1px solid var(--border);">Required</th>
      <th>Edge Ratio</th>
      <th>Needs Edge</th>
      <th style="border-left: 1px solid var(--border);">Required</th>
      <th>Edge Ratio</th>
      <th>Needs Edge</th>
    </tr>
  </thead>
  <tbody>
    {edge_table_rows}
  </tbody>
</table>

<!-- Probability History Chart -->
<h2>Model vs Polymarket Probabilities (Recent History)</h2>
<div class="chart-container">
  <canvas id="probChart" height="90"></canvas>
</div>

<!-- Edge Ratio History -->
<h2>Edge Ratio Over Time</h2>
<div class="chart-container">
  <canvas id="edgeHistChart" height="80"></canvas>
</div>

<!-- Spot vs Barrier -->
<h2>Spot Price vs Barrier</h2>
<div class="chart-container">
  <canvas id="spotChart" height="80"></canvas>
</div>

<!-- Open Positions -->
<h2>Open Positions</h2>
{"<p style='color: var(--text2);'>No open positions.</p>" if not open_positions else ""}
{"" if not open_positions else f'''<table>
  <thead><tr><th>Direction</th><th>Barrier</th><th>Entry</th><th>TP</th><th>Shares</th><th>Cost</th><th>Opened</th></tr></thead>
  <tbody>{open_rows}</tbody>
</table>'''}

<!-- Trade History -->
<h2>Trade History</h2>
{"<p style='color: var(--text2);'>No closed trades.</p>" if not closed_positions else ""}
{"" if not closed_positions else f'''<table>
  <thead><tr><th>Dir</th><th>Barrier</th><th>Entry</th><th>Exit</th><th>Shares</th><th>P&L</th><th>P&L %</th><th>Result</th><th>Opened</th><th>Closed</th></tr></thead>
  <tbody>{trades_rows}</tbody>
</table>'''}

<!-- Optimizer Grid -->
<h2>Optimizer Search Grid</h2>
<div class="param-grid">
  <div class="param-item">
    <div class="param-label">Alpha UP range</div>
    <div class="param-value" style="font-size:0.85em;">{', '.join(str(x) for x in opt_alpha_up[:6])}{'...' if len(opt_alpha_up) > 6 else ''}</div>
  </div>
  <div class="param-item">
    <div class="param-label">Alpha DOWN range</div>
    <div class="param-value" style="font-size:0.85em;">{', '.join(str(x) for x in opt_alpha_down[:6])}{'...' if len(opt_alpha_down) > 6 else ''}</div>
  </div>
  <div class="param-item">
    <div class="param-label">Floor UP range</div>
    <div class="param-value" style="font-size:0.85em;">{', '.join(str(x) for x in opt_floor_up[:6])}{'...' if len(opt_floor_up) > 6 else ''}</div>
  </div>
  <div class="param-item">
    <div class="param-label">Floor DOWN range</div>
    <div class="param-value" style="font-size:0.85em;">{', '.join(str(x) for x in opt_floor_down[:6])}{'...' if len(opt_floor_down) > 6 else ''}</div>
  </div>
  <div class="param-item">
    <div class="param-label">TP range</div>
    <div class="param-value" style="font-size:0.85em;">{opt_tp[0]*100:.0f}% &ndash; {opt_tp[1]*100:.0f}% (step {opt_tp[2]*100:.0f}%)</div>
  </div>
  <div class="param-item">
    <div class="param-label">Trail activation</div>
    <div class="param-value" style="font-size:0.85em;">{', '.join(str(x) for x in opt_trail_act)}</div>
  </div>
  <div class="param-item">
    <div class="param-label">Trail distance</div>
    <div class="param-value" style="font-size:0.85em;">{', '.join(str(x) for x in opt_trail_dist)}</div>
  </div>
  <div class="param-item">
    <div class="param-label">Min trades / Friction</div>
    <div class="param-value" style="font-size:0.85em;">{opt_config.get('min_trades', 6)} / {opt_config.get('friction', 0.015)*100:.1f}%</div>
  </div>
</div>

<footer>BTC Pricer &mdash; Risk-Neutral Density Trading System</footer>

<script>
// Chart.js defaults
Chart.defaults.color = '#8b949e';
Chart.defaults.borderColor = '#30363d';

// Edge Curve Chart
new Chart(document.getElementById('edgeCurveChart'), {{
  type: 'line',
  data: {{
    datasets: [
      {{
        label: 'UP required (\\u03b1={alpha_up}, floor={floor_up*100:.0f}%)',
        data: {up_curve_json},
        borderColor: '#3fb950',
        backgroundColor: 'rgba(63,185,80,0.1)',
        fill: true,
        tension: 0.3,
        pointRadius: 0,
        borderWidth: 2,
      }},
      {{
        label: 'DOWN required (\\u03b1={alpha_down}, floor={floor_down*100:.0f}%)',
        data: {down_curve_json},
        borderColor: '#f85149',
        backgroundColor: 'rgba(248,81,73,0.1)',
        fill: true,
        tension: 0.3,
        pointRadius: 0,
        borderWidth: 2,
      }},
      {{
        label: 'No edge (model = market)',
        data: Array.from({{length:101}}, (_,i) => ({{x:i, y:i}})),
        borderColor: '#484f58',
        borderDash: [5,5],
        pointRadius: 0,
        borderWidth: 1,
      }}
    ]
  }},
  options: {{
    responsive: true,
    plugins: {{
      title: {{ display: true, text: 'Required Model Probability vs Market Probability', font: {{size:14}} }},
      legend: {{ position: 'bottom' }}
    }},
    scales: {{
      x: {{ type: 'linear', title: {{ display: true, text: 'Polymarket Probability (%)' }}, min: 0, max: 100 }},
      y: {{ type: 'linear', title: {{ display: true, text: 'Required Model Probability (%)' }}, min: 0, max: 100 }}
    }}
  }}
}});

// Probability History Chart
const probLabels = {prob_timestamps};
new Chart(document.getElementById('probChart'), {{
  type: 'line',
  data: {{
    labels: probLabels,
    datasets: [
      {{ label: 'Model UP %', data: {model_up_data}, borderColor: '#3fb950', borderWidth: 1.5, pointRadius: 0, tension: 0.2 }},
      {{ label: 'Poly UP %', data: {poly_up_data}, borderColor: '#3fb950', borderDash: [4,4], borderWidth: 1.5, pointRadius: 0, tension: 0.2 }},
      {{ label: 'Model DOWN %', data: {model_down_data}, borderColor: '#f85149', borderWidth: 1.5, pointRadius: 0, tension: 0.2 }},
      {{ label: 'Poly DOWN %', data: {poly_down_data}, borderColor: '#f85149', borderDash: [4,4], borderWidth: 1.5, pointRadius: 0, tension: 0.2 }},
      {"" if not has_bl else f"""
      {{ label: 'B-L UP %', data: {bl_up_data}, borderColor: '#58a6ff', borderWidth: 1, pointRadius: 0, tension: 0.2, borderDash: [2,2] }},
      {{ label: 'B-L DOWN %', data: {bl_down_data}, borderColor: '#bc8cff', borderWidth: 1, pointRadius: 0, tension: 0.2, borderDash: [2,2] }},
      """}
    ]
  }},
  options: {{
    responsive: true,
    plugins: {{ legend: {{ position: 'bottom' }} }},
    scales: {{
      x: {{ display: true, ticks: {{ maxTicksLimit: 12, maxRotation: 45 }} }},
      y: {{ title: {{ display: true, text: 'Probability (%)' }}, min: 0, max: 100 }}
    }}
  }}
}});

// Edge Ratio History
new Chart(document.getElementById('edgeHistChart'), {{
  type: 'line',
  data: {{
    labels: probLabels,
    datasets: [
      {{ label: 'Edge UP (ratio)', data: {edge_up_data}, borderColor: '#3fb950', borderWidth: 1.5, pointRadius: 0, tension: 0.2 }},
      {{ label: 'Edge DOWN (ratio)', data: {edge_down_data}, borderColor: '#f85149', borderWidth: 1.5, pointRadius: 0, tension: 0.2 }},
      {{ label: '1.0x (no edge)', data: Array(probLabels.length).fill(1.0), borderColor: '#484f58', borderDash: [5,5], borderWidth: 1, pointRadius: 0 }},
    ]
  }},
  options: {{
    responsive: true,
    plugins: {{ legend: {{ position: 'bottom' }} }},
    scales: {{
      x: {{ display: true, ticks: {{ maxTicksLimit: 12, maxRotation: 45 }} }},
      y: {{ title: {{ display: true, text: 'Edge Ratio' }}, min: 0, suggestedMax: 2 }}
    }}
  }}
}});

// Spot vs Barrier
new Chart(document.getElementById('spotChart'), {{
  type: 'line',
  data: {{
    labels: probLabels,
    datasets: [
      {{ label: 'Spot Price', data: {spot_data}, borderColor: '#58a6ff', borderWidth: 2, pointRadius: 0, tension: 0.2 }},
      {{ label: 'Barrier', data: {barrier_data}, borderColor: '#d29922', borderDash: [6,3], borderWidth: 2, pointRadius: 0, tension: 0 }},
    ]
  }},
  options: {{
    responsive: true,
    plugins: {{ legend: {{ position: 'bottom' }} }},
    scales: {{
      x: {{ display: true, ticks: {{ maxTicksLimit: 12, maxRotation: 45 }} }},
      y: {{ title: {{ display: true, text: 'BTC Price ($)' }} }}
    }}
  }}
}});
</script>
</body>
</html>"""

    return html


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate trading dashboard HTML")
    parser.add_argument("--output", type=Path, default=Path("results/dashboard.html"))
    parser.add_argument("--config", type=Path, default=Path("config_dry_run.json"))
    parser.add_argument("--opt-config", type=Path, default=Path("config_optimize.json"))
    parser.add_argument("--state", type=Path, default=None)
    parser.add_argument("--probabilities", type=Path, default=Path("results/probabilities.csv"))
    args = parser.parse_args()

    config = load_config(args.config)
    opt_config = load_config(args.opt_config)

    # Auto-detect state file
    if args.state:
        state = load_state(args.state)
    else:
        live_state = Path("results/live_state.json")
        dry_state = Path("results/dry_run_state.json")
        state = load_state(live_state) if live_state.exists() else load_state(dry_state)

    prob_rows = load_probabilities_csv(args.probabilities)

    html = build_html(config, opt_config, state, prob_rows)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"Dashboard saved to {args.output}")
    print(f"  Config: {args.config}")
    print(f"  Probability rows: {len(prob_rows)}")
    print(f"  Trades: {len(state.get('closed_positions', []))} closed, {len(state.get('open_positions', []))} open")


if __name__ == "__main__":
    main()
