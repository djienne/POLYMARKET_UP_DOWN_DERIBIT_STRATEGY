# probability of BTC hitting 85000 within end of january 2026 in eastern time
python cli_terminal.py --mode barrier --target 85000 --until "Jan 31 11:59 PM ET"

# probability of being above 88,565.72 at end of January 28th ET
python cli_terminal.py --target 88565.72 --expiry 29JAN26

# probability of BTC ending above 88565.72 at 17:00 UTC (polymarket end)
python cli_terminal.py --mode barrier --target 88565.72 --until "17:00 UTC"
