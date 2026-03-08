#!/usr/bin/env python3
"""Generate a formatted HTML file showing Deribit BTC option prices."""

import argparse
import base64
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

# Add project root to path for btc_pricer imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import requests

from btc_pricer.api.deribit import DeribitClient

LOGO_URL = "https://www.deribit.com/favicon.ico"
_logo_cache = None
FALLBACK_LOGO_SVG = """<svg xmlns='http://www.w3.org/2000/svg' width='440' height='120' viewBox='0 0 440 120' role='img' aria-label='Deribit'>
  <defs>
    <linearGradient id='g' x1='0%' y1='0%' x2='100%' y2='100%'>
      <stop offset='0%' stop-color='#5fe9df'/>
      <stop offset='100%' stop-color='#11c9bb'/>
    </linearGradient>
  </defs>
  <path d='M30 28h24V14h18v14h20V14h18v14h7c20.8 0 38 17.2 38 38s-17.2 38-38 38h-7v14H92v-14H72v14H54v-14H30V86h35V46H30zM83 46v40h34c10.9 0 20-9.1 20-20s-9.1-20-20-20z' fill='url(#g)'/>
  <text x='176' y='84' fill='#ecf3f7' font-size='78' font-family='Segoe UI, Arial, sans-serif' font-weight='700'>Deribit</text>
</svg>"""


def get_logo_b64():
    """Download Deribit logo and return as base64 data URI."""
    global _logo_cache
    if _logo_cache is not None:
        return _logo_cache
    try:
        r = requests.get(LOGO_URL, timeout=10)
        r.raise_for_status()
        ctype = (r.headers.get("Content-Type") or "").split(";")[0].strip().lower()
        if ctype.startswith("image/") and r.content:
            b64 = base64.b64encode(r.content).decode()
            _logo_cache = f"data:{ctype};base64,{b64}"
            return _logo_cache
    except Exception:
        pass
    fallback_b64 = base64.b64encode(FALLBACK_LOGO_SVG.encode("utf-8")).decode()
    _logo_cache = f"data:image/svg+xml;base64,{fallback_b64}"
    return _logo_cache


def fetch_option_chain():
    """Fetch full option chain from Deribit, return (spot, options_by_expiry)."""
    client = DeribitClient()
    options_by_expiry = client.fetch_all_options("BTC")
    spot = client.get_index_price("BTC")
    return spot, options_by_expiry


def build_chain(options_by_expiry):
    """Reorganise into {expiry: {strike: {call: opt, put: opt}}}."""
    chain = {}
    for expiry, opts in options_by_expiry.items():
        rows = defaultdict(dict)
        ttm = opts[0].time_to_expiry if opts else 0
        fwd = opts[0].underlying_price if opts else 0
        for o in opts:
            rows[o.strike][o.option_type] = o
        chain[expiry] = {"rows": dict(sorted(rows.items())), "ttm": ttm, "fwd": fwd}
    # sort by TTM
    chain = dict(sorted(chain.items(), key=lambda kv: kv[1]["ttm"]))
    return chain


def fmt_usd(val, spot):
    """BTC price -> USD string."""
    if val is None:
        return "&mdash;"
    return f"${val * spot:,.0f}"


def fmt_iv(val):
    if val is None or val == 0:
        return "&mdash;"
    return f"{val * 100:.1f}%"


def fmt_oi(val):
    if val is None or val == 0:
        return "&mdash;"
    return f"{val:,.0f}"


def generate_html(spot, chain, expiry_filter=None, logo_b64=None):
    """Return an HTML string with option chain tables."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    logo_src = logo_b64 if logo_b64 else ""
    logo_tag = (
        f'<img src="{logo_src}" alt="Deribit" class="logo" '
        "onerror=\"this.style.display='none';this.nextElementSibling.style.display='inline-flex'\">"
        '<span class="logo-fallback" style="display:none;">D</span>'
    )

    rows_html = []
    expiry_count = 0
    for expiry, data in chain.items():
        if expiry_filter and expiry not in expiry_filter:
            continue
        expiry_count += 1
        ttm_days = data["ttm"] * 365.25
        fwd = data["fwd"]
        strikes = data["rows"]

        trs = []
        for strike, sides in strikes.items():
            c = sides.get("call")
            p = sides.get("put")
            atm = abs(strike - spot) / spot < 0.005
            cls = ' class="atm"' if atm else ""

            trs.append(f"""<tr{cls}>
  <td class="num call bid">{fmt_usd(c.bid_price, spot) if c else "&mdash;"}</td>
  <td class="num call ask">{fmt_usd(c.ask_price, spot) if c else "&mdash;"}</td>
  <td class="num call mark">{fmt_usd(c.mark_price, spot) if c else "&mdash;"}</td>
  <td class="num call iv">{fmt_iv(c.mark_iv) if c else "&mdash;"}</td>
  <td class="num call oi">{fmt_oi(c.open_interest) if c else "&mdash;"}</td>
  <td class="strike">${strike:,.0f}</td>
  <td class="num put oi">{fmt_oi(p.open_interest) if p else "&mdash;"}</td>
  <td class="num put iv">{fmt_iv(p.mark_iv) if p else "&mdash;"}</td>
  <td class="num put mark">{fmt_usd(p.mark_price, spot) if p else "&mdash;"}</td>
  <td class="num put bid">{fmt_usd(p.bid_price, spot) if p else "&mdash;"}</td>
  <td class="num put ask">{fmt_usd(p.ask_price, spot) if p else "&mdash;"}</td>
</tr>""")

        table = f"""
<div class="expiry-block">
  <div class="table-head">
    <div class="left">Options (BTC)</div>
    <div class="center">Underlying future: ${fwd:,.2f}</div>
    <div class="center">{expiry}</div>
    <div class="right">Time to Expiry: {ttm_days:.0f}d</div>
  </div>
  <div class="table-wrap">
  <table>
    <thead>
      <tr>
        <th colspan="5" class="side-header call-header">CALLS</th>
        <th class="strike-header"></th>
        <th colspan="5" class="side-header put-header">PUTS</th>
      </tr>
      <tr>
        <th>Bid</th><th>Ask</th><th>Mark</th><th>IV</th><th>OI</th>
        <th class="strike">Strike</th>
        <th>OI</th><th>IV</th><th>Mark</th><th>Bid</th><th>Ask</th>
      </tr>
    </thead>
    <tbody>
      {"".join(trs)}
    </tbody>
  </table>
  </div>
</div>"""
        rows_html.append(table)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>BTC Option Chain &mdash; Deribit</title>
<style>
  :root {{
    --bg: #090d12;
    --panel: #0f141c;
    --line: #1f2833;
    --head: #0e131a;
    --head-2: #121923;
    --text: #d4dce8;
    --muted: #8493a7;
    --teal: #13c7bb;
    --green: #0fce71;
    --red: #ff5f6a;
    --violet: #5c55d6;
    --btn: #18212e;
  }}
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{
    font-family: "IBM Plex Sans", "Segoe UI", Arial, sans-serif;
    background: var(--bg); color: var(--text);
    line-height: 1.4;
  }}
  .container {{
    width: 100%;
    max-width: 1700px;
    margin: 0 auto;
  }}
  .top-nav {{
    background: #0a0f15;
    border-bottom: 1px solid var(--line);
    padding: 10px 14px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
  }}
  .nav-left, .nav-right {{
    display: flex;
    align-items: center;
    gap: 8px;
  }}
  .brand {{
    display: inline-flex;
    align-items: center;
    gap: 8px;
    margin-right: 8px;
    color: #dff8f5;
    font-weight: 650;
    letter-spacing: .2px;
  }}
  .logo {{
    width: 220px;
    height: 60px;
    object-fit: contain;
    display: inline-block;
  }}
  .logo-fallback {{
    width: 220px;
    height: 60px;
    border-radius: 10px;
    align-items: center;
    justify-content: center;
    font-weight: 700;
    font-size: 1.5rem;
    color: #ecf3f7;
    background: linear-gradient(90deg, #0b1820, #0f1e2a);
    border: 1px solid #2b3b4d;
  }}
  .nav-tab {{
    color: #a4b2c4;
    padding: 8px 12px;
    border-radius: 8px;
    font-size: 0.95rem;
  }}
  .nav-tab.active {{
    color: #dff6f5;
    border: 1px solid #19514e;
    background: rgba(19, 199, 187, 0.14);
  }}
  .ticker {{
    color: #e3edf7;
    font-family: "JetBrains Mono", "Consolas", monospace;
    background: #101723;
    border: 1px solid #253244;
    border-radius: 8px;
    padding: 7px 10px;
    font-size: 0.82rem;
  }}
  .btn {{
    border: 1px solid #2a3647;
    background: var(--btn);
    color: #d5dfea;
    border-radius: 9px;
    padding: 8px 12px;
    font-weight: 600;
  }}
  .btn.primary {{
    color: #03201f;
    background: linear-gradient(180deg, #23d8cb, #13b2a7);
    border-color: #1abcb1;
  }}
  .sub-nav {{
    background: #0c121a;
    border-top: 1px solid #151d28;
    border-bottom: 1px solid #151d28;
    padding: 12px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 10px;
    flex-wrap: wrap;
  }}
  .sub-title {{
    font-weight: 650;
  }}
  .sub-controls {{
    display: flex;
    gap: 8px;
    align-items: center;
    flex-wrap: wrap;
  }}
  .chip {{
    border: 1px solid #263244;
    background: #111a27;
    border-radius: 8px;
    color: #d0d9e5;
    padding: 6px 10px;
    font-size: 0.86rem;
  }}
  .chip.active {{
    color: #112a29;
    border-color: #2fd1c5;
    background: #31d5c8;
  }}
  .content {{
    padding: 10px 12px 20px;
  }}
  .meta-line {{
    color: var(--muted);
    font-size: 0.82rem;
    margin: 2px 0 12px;
    font-family: "JetBrains Mono", "Consolas", monospace;
  }}
  .expiry-block {{
    border: 1px solid var(--line);
    background: var(--panel);
    margin-bottom: 14px;
  }}
  .table-head {{
    display: grid;
    grid-template-columns: 1.4fr 1.3fr 0.9fr 1fr;
    gap: 8px;
    padding: 8px 12px;
    background: #0c1118;
    border-bottom: 1px solid var(--line);
    font-size: 0.94rem;
    color: #cdd8e6;
  }}
  .table-head .left {{
    font-weight: 650;
  }}
  .table-head .center, .table-head .right {{
    color: #9aa8bb;
    font-family: "JetBrains Mono", "Consolas", monospace;
  }}
  .table-wrap {{
    overflow-x: auto;
  }}
  table {{
    width: 100%;
    min-width: 980px;
    border-collapse: collapse;
    font-size: 0.84rem;
  }}
  th, td {{
    padding: 6px 9px;
    white-space: nowrap;
  }}
  thead th {{
    background: var(--head-2);
    color: var(--muted);
    font-weight: 600;
    text-transform: uppercase;
    font-size: 0.73rem;
    border-bottom: 1px solid var(--line);
  }}
  .side-header {{
    font-size: 1.05rem;
    text-align: center;
    letter-spacing: 0.3px;
    padding: 9px 0;
    color: #e6eef8;
  }}
  .call-header {{
    background: linear-gradient(180deg, #162c22, #121922);
  }}
  .put-header {{
    background: linear-gradient(180deg, #2d161d, #121922);
  }}
  .strike-header {{
    background: var(--violet);
  }}
  td {{
    border-top: 1px solid #1a2230;
    font-family: "JetBrains Mono", "Consolas", monospace;
    font-variant-numeric: tabular-nums;
  }}
  tbody tr:nth-child(even) td {{
    background: #101722;
  }}
  .num {{
    text-align: right;
  }}
  .call {{
    background: linear-gradient(180deg, rgba(26, 60, 42, 0.35), rgba(20, 29, 24, 0.2));
  }}
  .put {{
    background: linear-gradient(180deg, rgba(62, 30, 38, 0.35), rgba(31, 22, 25, 0.2));
  }}
  .bid {{
    color: var(--green);
    font-weight: 600;
  }}
  .ask {{
    color: var(--red);
    font-weight: 600;
  }}
  .mark {{
    color: #e6edf8;
    font-weight: 700;
  }}
  .iv {{
    color: #b8c5d7;
  }}
  .oi {{
    color: #8fa0b7;
    font-size: 0.78rem;
  }}
  .strike {{
    text-align: center;
    background: linear-gradient(180deg, #655ee3, #4f49bf);
    color: #f1eeff;
    font-weight: 700;
    border-left: 1px solid #756ff0;
    border-right: 1px solid #756ff0;
  }}
  th.strike {{
    background: linear-gradient(180deg, #625be0, #4e48bf);
    color: #f5f3ff;
  }}
  tr.atm td {{
    box-shadow: inset 0 1px 0 rgba(56, 225, 210, 0.35), inset 0 -1px 0 rgba(56, 225, 210, 0.35);
  }}
  tr:hover td {{
    filter: brightness(1.14);
  }}
  @media (max-width: 980px) {{
    .nav-left .nav-tab:nth-child(n+4) {{
      display: none;
    }}
    .table-head {{
      grid-template-columns: 1fr 1fr;
      row-gap: 3px;
      font-size: 0.83rem;
    }}
    .ticker {{
      font-size: 0.74rem;
      padding: 6px 8px;
    }}
  }}
  @media (max-width: 720px) {{
    .top-nav {{
      padding: 8px;
    }}
    .btn {{
      padding: 7px 10px;
    }}
    .sub-nav {{
      padding: 10px 8px;
    }}
    .content {{
      padding: 8px;
    }}
  }}
</style>
</head>
<body>
<div class="container">
  <div class="top-nav">
    <div class="nav-left">
      <div class="brand">{logo_tag}</div>
      <div class="nav-tab">Spot</div>
      <div class="nav-tab">Futures</div>
      <div class="nav-tab active">Options</div>
      <div class="nav-tab">Strategy</div>
    </div>
    <div class="nav-right">
      <div class="ticker">BTC ${spot:,.0f}</div>
      <button class="btn">Log In</button>
      <button class="btn primary">Register</button>
    </div>
  </div>
  <div class="sub-nav">
    <div class="sub-title">Options (BTC)</div>
    <div class="sub-controls">
      <span class="chip">Expiry dates</span>
      <span class="chip">Columns</span>
      <span class="chip">Filter</span>
      <span class="chip active">USD</span>
      <span class="chip">Dist</span>
    </div>
  </div>
  <div class="content">
    <div class="meta-line">Generated {now} | Spot ${spot:,.2f} | Expiries {expiry_count}</div>
    {"".join(rows_html)}
  </div>
</div>
</body>
</html>"""


def main():
    parser = argparse.ArgumentParser(description="Generate Deribit BTC option chain HTML")
    parser.add_argument(
        "--expiries", nargs="*", metavar="EXP",
        help="Filter to specific expiries, e.g. 28FEB26 28MAR26 (default: all)",
    )
    parser.add_argument(
        "-o", "--output", default="results/option_chain.html",
        help="Output HTML file path (default: results/option_chain.html)",
    )
    args = parser.parse_args()

    print("Fetching Deribit option chain...")
    spot, options_by_expiry = fetch_option_chain()
    print(f"  Spot: ${spot:,.2f}  |  Expiries: {len(options_by_expiry)}")

    print("Downloading logo...")
    logo_b64 = get_logo_b64()

    chain = build_chain(options_by_expiry)

    expiry_filter = set(e.upper() for e in args.expiries) if args.expiries else None
    html = generate_html(spot, chain, expiry_filter, logo_b64)

    with open(args.output, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Written to {args.output}")


if __name__ == "__main__":
    main()
