# BTC_pricer and POLYMARKET_UP_DOWN_DERIBIT_STRATEGY

Terminal probability calculator for Bitcoin price. Computes the probability that BTC ends above or below a given price at a future time. Uses SSVI surface calibrated on Deribit BTC options, Breeden-Litzenberger density extraction, and Monte Carlo simulation. Heston stochastic volatility is used as a fallback when SSVI surface fit fails. Includes a backtester and optimizer for trading Polymarket daily BTC Up/Down markets.

[Associated YouTube video](https://youtu.be/eqsnUcE5hrs?si=m4QOzVA49RIwzBoP)

## Install

```bash
pip install -r requirements.txt
```

## CLI Usage

```bash
# Extract RND from all Deribit expiries
python cli.py

# Terminal probability (SSVI Surface + Breeden-Litzenberger, with Heston fallback)
python cli_terminal.py --reference-price 85000 --until "Jan 31 11:59 PM ET"
python cli_terminal.py --reference-prices 80000 85000 90000 --expiry 31JAN26 --direction down

# Terminal probability (lightweight RND-only, no MC)
# --price is shorthand for --reference-price
python cli_terminal.py --method rnd --price 88565.72 --expiry 29JAN26

# Intraday forecast
python cli_intraday.py
```

## Data Collection

Runs continuously in Docker, collecting three data streams from public APIs (no API keys needed):

- **Probabilities** (every 5 min): finds the active Polymarket BTC daily market, fetches the Polymarket price and runs the SSVI model against it to compute the model-vs-market edge. Saved to `probabilities.csv`.
- **Orderbook** (every 1 min): CLOB orderbook snapshots (bids/asks) for the active market's UP and DOWN tokens. Saved to `orderbook.csv`.
- **Deribit options** (every 5 min): full options chain snapshot (strikes, IVs, bid/ask/mark prices) across all BTC expiries. Saved to `deribit_options.csv`.

```bash
docker-compose up data-collector
# or without Docker:
python -m data_collector.collector
```

All output goes to `data_collector/results/`. These CSVs feed directly into the backtester.

## Backtesting & Optimization

```bash
# Run backtest with edge and trailing stop parameters
python scripts/backtest.py \
  --alpha-up 2.40 --alpha-down 1.80 \
  --floor-up 0.35 --floor-down 0.35 \
  --tp 0.40 \
  --trail-activation 0.20 --trail-distance 0.15

# Optimize parameters
python scripts/optimize.py --min-trades 5
python scripts/optimize.py --min-trades 5 --robust --sort-by sharpe
```

Outputs in `results/`: `backtest_report.json`, `pnl_curve.png`, `pnl_curve_zoom.png`, `pnl_curve_probabilities.png`.

