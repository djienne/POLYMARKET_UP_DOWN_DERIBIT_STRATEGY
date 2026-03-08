"""Intraday price forecasting using ATM IV scaling.

This module provides short-term (hours to days) price forecasts by:
1. Extracting ATM implied volatility from the nearest expiry options
2. Scaling volatility by sqrt(T) for the target time horizon
3. Generating log-normal price distributions

This is a practical approximation when options don't exist for the
exact time horizon needed.
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
from scipy.stats import norm, lognorm

from .ssvi import SSVIModel, SSVIParams


@dataclass
class IntradayForecast:
    """Forecast for a specific time horizon."""
    hours: float
    spot_price: float
    forward_price: float  # Adjusted for drift if applicable

    # Volatility
    atm_iv_annual: float  # Annualized ATM IV used
    scaled_vol: float     # Volatility for this time horizon

    # Price statistics
    mean: float
    median: float
    mode: float
    std_dev: float

    # Percentiles
    percentile_1: float
    percentile_5: float
    percentile_10: float
    percentile_25: float
    percentile_50: float
    percentile_75: float
    percentile_90: float
    percentile_95: float
    percentile_99: float

    # Probability of moves
    prob_up_1pct: float
    prob_up_5pct: float
    prob_up_10pct: float
    prob_down_1pct: float
    prob_down_5pct: float
    prob_down_10pct: float

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "hours": self.hours,
            "spot_price": self.spot_price,
            "forward_price": self.forward_price,
            "atm_iv_annual": self.atm_iv_annual,
            "atm_iv_annual_pct": self.atm_iv_annual * 100,
            "scaled_vol": self.scaled_vol,
            "scaled_vol_pct": self.scaled_vol * 100,
            "mean": self.mean,
            "median": self.median,
            "mode": self.mode,
            "std_dev": self.std_dev,
            "std_dev_pct": self.std_dev / self.spot_price * 100,
            "percentiles": {
                "1": self.percentile_1,
                "5": self.percentile_5,
                "10": self.percentile_10,
                "25": self.percentile_25,
                "50": self.percentile_50,
                "75": self.percentile_75,
                "90": self.percentile_90,
                "95": self.percentile_95,
                "99": self.percentile_99,
            },
            "move_probabilities": {
                "up_1pct": self.prob_up_1pct,
                "up_5pct": self.prob_up_5pct,
                "up_10pct": self.prob_up_10pct,
                "down_1pct": self.prob_down_1pct,
                "down_5pct": self.prob_down_5pct,
                "down_10pct": self.prob_down_10pct,
            }
        }


@dataclass
class IntradayForecastSeries:
    """Series of forecasts for multiple time horizons."""
    spot_price: float
    atm_iv_annual: float
    source_expiry: str
    forecasts: List[IntradayForecast]

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "spot_price": self.spot_price,
            "atm_iv_annual": self.atm_iv_annual,
            "atm_iv_annual_pct": self.atm_iv_annual * 100,
            "source_expiry": self.source_expiry,
            "forecasts": [f.to_dict() for f in self.forecasts]
        }


class IntradayForecaster:
    """Generate intraday price forecasts using ATM IV scaling."""

    # Hours in a year (for annualization)
    HOURS_PER_YEAR = 365.25 * 24  # 8766 hours

    def __init__(
        self,
        use_drift: bool = False,
        annual_drift: float = 0.0
    ):
        """Initialize the intraday forecaster.

        Args:
            use_drift: If True, adjust forward price for expected drift.
            annual_drift: Annualized drift rate (e.g., 0.10 for 10% annual).
                         Only used if use_drift=True.
        """
        self.use_drift = use_drift
        self.annual_drift = annual_drift

    def extract_atm_iv(
        self,
        ssvi_params: SSVIParams
    ) -> float:
        """Extract ATM implied volatility from SSVI parameters.

        Args:
            ssvi_params: Fitted SSVI parameters.

        Returns:
            ATM implied volatility (annualized).
        """
        model = SSVIModel(ssvi_params)
        # ATM is at log-moneyness k = 0
        return model.implied_volatility(0)

    def extract_atm_iv_from_options(
        self,
        options: List,
        forward: float
    ) -> float:
        """Extract ATM IV directly from option data.

        Args:
            options: List of FilteredOption objects.
            forward: Forward price.

        Returns:
            ATM implied volatility (annualized).
        """
        if not options:
            raise ValueError("No options provided")

        # Find options closest to ATM
        atm_options = sorted(
            options,
            key=lambda o: abs(o.strike - forward)
        )[:4]  # Take 4 closest

        # Average their IVs
        ivs = [o.mark_iv for o in atm_options if o.mark_iv > 0]

        if not ivs:
            raise ValueError("No valid IVs found near ATM")

        return sum(ivs) / len(ivs)

    def forecast_single(
        self,
        spot_price: float,
        atm_iv_annual: float,
        hours: float
    ) -> IntradayForecast:
        """Generate forecast for a single time horizon.

        Uses log-normal distribution with volatility scaled by sqrt(T).

        Args:
            spot_price: Current spot price.
            atm_iv_annual: Annualized ATM implied volatility.
            hours: Forecast horizon in hours.

        Returns:
            IntradayForecast with price distribution.
        """
        # Convert hours to years
        T = hours / self.HOURS_PER_YEAR

        # Scale volatility
        scaled_vol = atm_iv_annual * math.sqrt(T)

        # Calculate forward price (with optional drift)
        if self.use_drift:
            forward_price = spot_price * math.exp(self.annual_drift * T)
        else:
            forward_price = spot_price

        # Log-normal parameters
        # If S_T = S_0 * exp((mu - 0.5*sigma^2)*T + sigma*sqrt(T)*Z)
        # For risk-neutral (or zero drift): mu = 0
        # ln(S_T) ~ N(ln(S_0) + (mu - 0.5*sigma^2)*T, sigma^2*T)

        mu_log = math.log(forward_price) - 0.5 * scaled_vol**2
        sigma_log = scaled_vol

        # Statistics of log-normal distribution
        mean = math.exp(mu_log + 0.5 * sigma_log**2)
        median = math.exp(mu_log)
        mode = math.exp(mu_log - sigma_log**2) if sigma_log > 0 else forward_price
        variance = (math.exp(sigma_log**2) - 1) * math.exp(2*mu_log + sigma_log**2)
        std_dev = math.sqrt(variance)

        # Percentiles using inverse CDF
        def percentile(p: float) -> float:
            z = norm.ppf(p / 100)
            return math.exp(mu_log + sigma_log * z)

        # Move probabilities
        def prob_above(threshold: float) -> float:
            if threshold <= 0:
                return 1.0
            z = (math.log(threshold) - mu_log) / sigma_log if sigma_log > 0 else float('inf')
            return 1 - norm.cdf(z)

        def prob_below(threshold: float) -> float:
            if threshold <= 0:
                return 0.0
            z = (math.log(threshold) - mu_log) / sigma_log if sigma_log > 0 else float('-inf')
            return norm.cdf(z)

        return IntradayForecast(
            hours=hours,
            spot_price=spot_price,
            forward_price=forward_price,
            atm_iv_annual=atm_iv_annual,
            scaled_vol=scaled_vol,
            mean=mean,
            median=median,
            mode=mode,
            std_dev=std_dev,
            percentile_1=percentile(1),
            percentile_5=percentile(5),
            percentile_10=percentile(10),
            percentile_25=percentile(25),
            percentile_50=percentile(50),
            percentile_75=percentile(75),
            percentile_90=percentile(90),
            percentile_95=percentile(95),
            percentile_99=percentile(99),
            prob_up_1pct=prob_above(spot_price * 1.01),
            prob_up_5pct=prob_above(spot_price * 1.05),
            prob_up_10pct=prob_above(spot_price * 1.10),
            prob_down_1pct=prob_below(spot_price * 0.99),
            prob_down_5pct=prob_below(spot_price * 0.95),
            prob_down_10pct=prob_below(spot_price * 0.90),
        )

    def forecast_series(
        self,
        spot_price: float,
        atm_iv_annual: float,
        hours_list: List[float],
        source_expiry: str = "unknown"
    ) -> IntradayForecastSeries:
        """Generate forecasts for multiple time horizons.

        Args:
            spot_price: Current spot price.
            atm_iv_annual: Annualized ATM implied volatility.
            hours_list: List of forecast horizons in hours.
            source_expiry: Name of the expiry used for ATM IV.

        Returns:
            IntradayForecastSeries with all forecasts.
        """
        forecasts = [
            self.forecast_single(spot_price, atm_iv_annual, hours)
            for hours in sorted(hours_list)
        ]

        return IntradayForecastSeries(
            spot_price=spot_price,
            atm_iv_annual=atm_iv_annual,
            source_expiry=source_expiry,
            forecasts=forecasts
        )

    def forecast_standard_horizons(
        self,
        spot_price: float,
        atm_iv_annual: float,
        source_expiry: str = "unknown"
    ) -> IntradayForecastSeries:
        """Generate forecasts for standard intraday horizons.

        Horizons: 1h, 2h, 4h, 6h, 8h, 12h, 24h, 48h, 72h

        Args:
            spot_price: Current spot price.
            atm_iv_annual: Annualized ATM implied volatility.
            source_expiry: Name of the expiry used for ATM IV.

        Returns:
            IntradayForecastSeries with standard horizon forecasts.
        """
        standard_hours = [1, 2, 4, 6, 8, 12, 24, 48, 72]
        return self.forecast_series(
            spot_price, atm_iv_annual, standard_hours, source_expiry
        )

    def generate_price_path_samples(
        self,
        spot_price: float,
        atm_iv_annual: float,
        hours: float,
        n_samples: int = 10000
    ) -> np.ndarray:
        """Generate Monte Carlo samples of future prices.

        Args:
            spot_price: Current spot price.
            atm_iv_annual: Annualized ATM IV.
            hours: Forecast horizon.
            n_samples: Number of samples to generate.

        Returns:
            Array of sampled future prices.
        """
        T = hours / self.HOURS_PER_YEAR
        scaled_vol = atm_iv_annual * math.sqrt(T)

        if self.use_drift:
            forward = spot_price * math.exp(self.annual_drift * T)
        else:
            forward = spot_price

        mu_log = math.log(forward) - 0.5 * scaled_vol**2

        # Generate samples
        z = np.random.standard_normal(n_samples)
        prices = np.exp(mu_log + scaled_vol * z)

        return prices

    def probability_range(
        self,
        spot_price: float,
        atm_iv_annual: float,
        hours: float,
        lower: float,
        upper: float
    ) -> float:
        """Calculate probability of price falling within a range.

        Args:
            spot_price: Current spot price.
            atm_iv_annual: Annualized ATM IV.
            hours: Forecast horizon.
            lower: Lower price bound.
            upper: Upper price bound.

        Returns:
            Probability in [0, 1].
        """
        T = hours / self.HOURS_PER_YEAR
        scaled_vol = atm_iv_annual * math.sqrt(T)

        if self.use_drift:
            forward = spot_price * math.exp(self.annual_drift * T)
        else:
            forward = spot_price

        mu_log = math.log(forward) - 0.5 * scaled_vol**2

        if scaled_vol <= 0:
            return 1.0 if lower <= forward <= upper else 0.0

        z_lower = (math.log(lower) - mu_log) / scaled_vol if lower > 0 else float('-inf')
        z_upper = (math.log(upper) - mu_log) / scaled_vol if upper > 0 else float('inf')

        return norm.cdf(z_upper) - norm.cdf(z_lower)


def format_intraday_forecast(forecast: IntradayForecast) -> str:
    """Format a single forecast as human-readable string."""
    lines = [
        f"=== {forecast.hours}h Forecast ===",
        f"Spot: ${forecast.spot_price:,.0f}",
        f"ATM IV (annual): {forecast.atm_iv_annual*100:.1f}%",
        f"Scaled Vol ({forecast.hours}h): {forecast.scaled_vol*100:.2f}%",
        "",
        f"Expected Range:",
        f"  1%  - 99%:  ${forecast.percentile_1:,.0f} - ${forecast.percentile_99:,.0f}",
        f"  5%  - 95%:  ${forecast.percentile_5:,.0f} - ${forecast.percentile_95:,.0f}",
        f"  10% - 90%:  ${forecast.percentile_10:,.0f} - ${forecast.percentile_90:,.0f}",
        f"  25% - 75%:  ${forecast.percentile_25:,.0f} - ${forecast.percentile_75:,.0f}",
        "",
        f"Mean:   ${forecast.mean:,.0f}",
        f"Median: ${forecast.median:,.0f}",
        f"Std Dev: ${forecast.std_dev:,.0f} ({forecast.std_dev/forecast.spot_price*100:.2f}%)",
        "",
        f"Move Probabilities:",
        f"  P(up   > 1%): {forecast.prob_up_1pct*100:.1f}%",
        f"  P(up   > 5%): {forecast.prob_up_5pct*100:.1f}%",
        f"  P(up   >10%): {forecast.prob_up_10pct*100:.1f}%",
        f"  P(down > 1%): {forecast.prob_down_1pct*100:.1f}%",
        f"  P(down > 5%): {forecast.prob_down_5pct*100:.1f}%",
        f"  P(down >10%): {forecast.prob_down_10pct*100:.1f}%",
    ]
    return "\n".join(lines)


def format_intraday_table(series: IntradayForecastSeries) -> str:
    """Format forecast series as a table."""
    lines = [
        "=" * 100,
        f"Intraday Price Forecasts (ATM IV: {series.atm_iv_annual*100:.1f}% from {series.source_expiry})",
        f"Spot Price: ${series.spot_price:,.0f}",
        "=" * 100,
        "",
        f"{'Hours':>6} {'Vol%':>8} {'5th':>12} {'25th':>12} {'Median':>12} {'75th':>12} {'95th':>12} {'StdDev%':>10}",
        "-" * 100,
    ]

    for f in series.forecasts:
        lines.append(
            f"{f.hours:>6.0f} {f.scaled_vol*100:>7.2f}% "
            f"${f.percentile_5:>10,.0f} ${f.percentile_25:>10,.0f} "
            f"${f.percentile_50:>10,.0f} ${f.percentile_75:>10,.0f} "
            f"${f.percentile_95:>10,.0f} {f.std_dev/f.spot_price*100:>9.2f}%"
        )

    lines.append("-" * 100)
    lines.append("")
    lines.append("Move Probabilities:")
    lines.append(f"{'Hours':>6} {'P(>+1%)':>10} {'P(>+5%)':>10} {'P(>+10%)':>10} {'P(<-1%)':>10} {'P(<-5%)':>10} {'P(<-10%)':>10}")
    lines.append("-" * 100)

    for f in series.forecasts:
        lines.append(
            f"{f.hours:>6.0f} {f.prob_up_1pct*100:>9.1f}% {f.prob_up_5pct*100:>9.1f}% "
            f"{f.prob_up_10pct*100:>9.1f}% {f.prob_down_1pct*100:>9.1f}% "
            f"{f.prob_down_5pct*100:>9.1f}% {f.prob_down_10pct*100:>9.1f}%"
        )

    return "\n".join(lines)
