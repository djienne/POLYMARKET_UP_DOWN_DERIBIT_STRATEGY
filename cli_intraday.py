#!/usr/bin/env python3
"""
BTC Intraday Price Forecaster

Generate short-term (1h to 72h) price forecasts using ATM implied volatility
from Deribit options, scaled by sqrt(T).
"""

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from btc_pricer.config import Config
from btc_pricer.api.deribit import DeribitClient, DeribitAPIError
from btc_pricer.api.binance import fetch_spot_with_fallback
from btc_pricer.data.filters import DataFilter
from btc_pricer.models.ssvi import SSVIModel
from btc_pricer.models.intraday_forecast import (
    IntradayForecaster,
    format_intraday_forecast,
    format_intraday_table,
)
from btc_pricer.cli.common import (
    setup_logging,
    load_config,
    create_ssvi_fitter,
    extract_surface_data,
    handle_cli_exceptions,
    add_common_arguments,
)


def get_atm_iv_from_nearest_expiry(
    client: DeribitClient,
    config: Config
) -> tuple:
    """Fetch options and extract ATM IV from the nearest liquid expiry.

    Returns:
        Tuple of (atm_iv, spot_price, expiry_name, forward_price)
    """
    logger = logging.getLogger(__name__)

    # Fetch all options
    logger.info("Fetching options data from Deribit...")
    options_by_expiry = client.fetch_all_options("BTC")

    if not options_by_expiry:
        raise DeribitAPIError("No options data received")

    spot_price = list(options_by_expiry.values())[0][0].spot_price
    logger.info(f"Deribit spot price: ${spot_price:,.0f}")

    # Filter and find nearest expiry with sufficient data
    data_filter = DataFilter(config.filters)
    fitter = create_ssvi_fitter(config)

    # Sort expiries by TTM
    sorted_expiries = sorted(
        options_by_expiry.items(),
        key=lambda x: x[1][0].time_to_expiry if x[1] else float('inf')
    )

    min_points = config.filters.min_surface_points
    for expiry, options in sorted_expiries:
        # Filter options
        filtered, _ = data_filter.filter_options(options)

        if len(filtered) < min_points:
            logger.debug(f"Skipping {expiry}: insufficient options ({len(filtered)})")
            continue

        # Build OTM surface
        otm_surface = data_filter.build_otm_surface(filtered)

        # Extract and validate surface data
        surface_data = extract_surface_data(
            otm_surface,
            min_points=min_points,
            iv_valid_range=config.validation.iv_valid_range
        )
        if surface_data is None:
            continue

        forward, ttm, _, log_moneyness, market_iv = surface_data

        # Fit SSVI to get smooth ATM IV
        fit_result = fitter.fit(log_moneyness, market_iv, ttm)

        if fit_result.success and fit_result.params is not None:
            # Extract ATM IV from SSVI
            ssvi_model = SSVIModel(fit_result.params)
            atm_iv = ssvi_model.implied_volatility(0)

            logger.info(f"Using expiry {expiry} (TTM: {ttm*365:.1f} days)")
            logger.info(f"ATM IV: {atm_iv*100:.1f}%")
            logger.info(f"Forward: ${forward:,.0f}")

            return atm_iv, spot_price, expiry, forward

    raise DeribitAPIError("Could not find suitable expiry for ATM IV extraction")


@handle_cli_exceptions
def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="BTC Intraday Price Forecaster - Short-term forecasts using ATM IV"
    )
    add_common_arguments(parser)
    parser.add_argument(
        "--hours",
        type=float,
        nargs="+",
        default=None,
        help="Specific forecast horizons in hours (e.g., 1 2 4 8 24)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON file path"
    )
    parser.add_argument(
        "--atm-iv",
        type=float,
        default=None,
        help="Override ATM IV (as decimal, e.g., 0.65 for 65%%)"
    )
    parser.add_argument(
        "--spot",
        type=float,
        default=None,
        help="Override spot price"
    )

    args = parser.parse_args()

    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

    # Load configuration
    config = load_config(args.config, logger)

    # Get ATM IV (from market or override)
    if args.atm_iv is not None and args.spot is not None:
        atm_iv = args.atm_iv
        spot_price = args.spot
        source_expiry = "manual_override"
        logger.info(f"Using manual ATM IV: {atm_iv*100:.1f}%")
        logger.info(f"Using manual spot: ${spot_price:,.0f}")
    else:
        client = DeribitClient(config.api, config.validation)
        atm_iv, deribit_spot, source_expiry, _ = get_atm_iv_from_nearest_expiry(
            client, config
        )

        # Use Binance spot price by default, fallback to Deribit
        spot_price, _ = fetch_spot_with_fallback(deribit_spot)

        # Allow overrides
        if args.atm_iv is not None:
            atm_iv = args.atm_iv
            logger.info(f"Overriding ATM IV to: {atm_iv*100:.1f}%")
        if args.spot is not None:
            spot_price = args.spot
            logger.info(f"Overriding spot to: ${spot_price:,.0f}")

    # Create forecaster using config
    forecaster = IntradayForecaster(
        use_drift=config.intraday.use_drift,
        annual_drift=config.intraday.annual_drift
    )

    # Generate forecasts
    if args.hours:
        series = forecaster.forecast_series(
            spot_price, atm_iv, args.hours, source_expiry
        )
    else:
        # Use standard horizons from config
        series = forecaster.forecast_series(
            spot_price, atm_iv, config.intraday.standard_horizons, source_expiry
        )

    # Print results
    print("\n" + format_intraday_table(series))

    # Print detailed view for first few horizons
    print("\n" + "=" * 60)
    print("DETAILED FORECASTS")
    print("=" * 60)
    for forecast in series.forecasts[:3]:
        print("\n" + format_intraday_forecast(forecast))

    # Save to JSON if requested
    if args.output:
        output_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "spot_price": spot_price,
            "atm_iv_annual": atm_iv,
            "atm_iv_annual_pct": atm_iv * 100,
            "source_expiry": source_expiry,
            "forecasts": series.to_dict()["forecasts"]
        }

        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)

        logger.info(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
