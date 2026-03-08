"""Statistical analysis of Risk-Neutral Densities."""

from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np
from scipy.integrate import simpson, cumulative_trapezoid

from ..models.breeden_litzenberger import RNDResult


@dataclass
class PriceScenario:
    """Price scenario with probability."""
    description: str
    lower: Optional[float]
    upper: Optional[float]
    probability: float


@dataclass
class ExpiryStats:
    """Statistics for a single expiry."""
    expiry: str
    ttm: float
    forward: float
    mean: float
    mode: float
    median: float
    std_dev: float
    skewness: float
    kurtosis: float
    percentiles: Dict[int, float]
    scenarios: List[PriceScenario]


class RNDStatistics:
    """Compute and format statistics from RND results."""

    def compute_stats(
        self,
        rnd: RNDResult,
        expiry: str,
        spot_price: float
    ) -> ExpiryStats:
        """Compute comprehensive statistics for an RND.

        Args:
            rnd: RND extraction result.
            expiry: Expiry date string.
            spot_price: Current spot price.

        Returns:
            ExpiryStats with all statistics.
        """
        # Standard percentiles
        percentiles = {
            5: rnd.percentile_5,
            10: self._get_percentile(rnd, 10),
            25: rnd.percentile_25,
            50: rnd.percentile_50,
            75: rnd.percentile_75,
            90: self._get_percentile(rnd, 90),
            95: rnd.percentile_95,
        }

        # Price scenarios
        scenarios = self._compute_scenarios(rnd, spot_price)

        return ExpiryStats(
            expiry=expiry,
            ttm=rnd.ttm,
            forward=rnd.forward,
            mean=rnd.mean,
            mode=rnd.mode,
            median=rnd.percentile_50,
            std_dev=rnd.std_dev,
            skewness=rnd.skewness,
            kurtosis=rnd.kurtosis,
            percentiles=percentiles,
            scenarios=scenarios
        )

    def _get_percentile(self, rnd: RNDResult, p: float) -> float:
        """Get a specific percentile from RND.

        Args:
            rnd: RND result.
            p: Percentile (0-100).

        Returns:
            Price at percentile.
        """
        # Compute CDF - use cumulative_trapezoid for O(n) complexity
        cdf = np.zeros(len(rnd.strikes))
        cdf[1:] = cumulative_trapezoid(rnd.density, x=rnd.strikes)
        # Normalize CDF to end at 1.0
        if cdf[-1] > 0:
            cdf = cdf / cdf[-1]

        # Find percentile
        target = p / 100
        idx = np.searchsorted(cdf, target)

        if idx >= len(rnd.strikes):
            return rnd.strikes[-1]
        if idx == 0:
            return rnd.strikes[0]

        # Linear interpolation
        frac = (target - cdf[idx-1]) / (cdf[idx] - cdf[idx-1] + 1e-10)
        return rnd.strikes[idx-1] + frac * (rnd.strikes[idx] - rnd.strikes[idx-1])

    def _compute_scenarios(
        self,
        rnd: RNDResult,
        spot_price: float
    ) -> List[PriceScenario]:
        """Compute probability of various price scenarios.

        Args:
            rnd: RND result.
            spot_price: Current spot price.

        Returns:
            List of price scenarios with probabilities.
        """
        scenarios = []

        # Price change scenarios
        changes = [
            ("Crash (> -30%)", None, spot_price * 0.7),
            ("Large drop (-30% to -15%)", spot_price * 0.7, spot_price * 0.85),
            ("Moderate drop (-15% to -5%)", spot_price * 0.85, spot_price * 0.95),
            ("Flat (-5% to +5%)", spot_price * 0.95, spot_price * 1.05),
            ("Moderate gain (+5% to +15%)", spot_price * 1.05, spot_price * 1.15),
            ("Large gain (+15% to +30%)", spot_price * 1.15, spot_price * 1.30),
            ("Moon (> +30%)", spot_price * 1.30, None),
        ]

        for desc, lower, upper in changes:
            prob = self._probability_range(rnd, lower, upper)
            scenarios.append(PriceScenario(
                description=desc,
                lower=lower,
                upper=upper,
                probability=prob
            ))

        return scenarios

    def _probability_range(
        self,
        rnd: RNDResult,
        lower: Optional[float],
        upper: Optional[float]
    ) -> float:
        """Calculate probability within a range.

        Args:
            rnd: RND result.
            lower: Lower bound (None for unbounded).
            upper: Upper bound (None for unbounded).

        Returns:
            Probability.
        """
        if lower is None:
            lower = rnd.strikes[0]
        if upper is None:
            upper = rnd.strikes[-1]

        mask = (rnd.strikes >= lower) & (rnd.strikes <= upper)
        if not np.any(mask):
            return 0.0

        return simpson(rnd.density[mask], x=rnd.strikes[mask])

    def format_summary(self, stats: ExpiryStats) -> str:
        """Format statistics as human-readable summary.

        Args:
            stats: Expiry statistics.

        Returns:
            Formatted string.
        """
        lines = [
            f"=== {stats.expiry} (TTM: {stats.ttm*365:.1f} days) ===",
            f"Forward: ${stats.forward:,.0f}",
            f"Mean:    ${stats.mean:,.0f}",
            f"Mode:    ${stats.mode:,.0f}",
            f"Median:  ${stats.median:,.0f}",
            f"Std Dev: ${stats.std_dev:,.0f} ({stats.std_dev/stats.forward*100:.1f}%)",
            f"Skew:    {stats.skewness:.3f}",
            f"Kurt:    {stats.kurtosis:.3f}",
            "",
            "Percentiles:",
        ]

        for p, val in sorted(stats.percentiles.items()):
            pct_change = (val - stats.forward) / stats.forward * 100
            lines.append(f"  {p:3d}%: ${val:,.0f} ({pct_change:+.1f}%)")

        lines.append("")
        lines.append("Scenarios:")
        for s in stats.scenarios:
            lines.append(f"  {s.description}: {s.probability*100:.1f}%")

        return "\n".join(lines)

    def format_table(self, all_stats: List[ExpiryStats]) -> str:
        """Format multiple expiries as a comparison table.

        Args:
            all_stats: List of statistics for all expiries.

        Returns:
            Formatted table string.
        """
        if not all_stats:
            return "No data"

        # Sort by TTM
        all_stats = sorted(all_stats, key=lambda x: x.ttm)

        # Header
        lines = [
            "=" * 80,
            "BTC Price Forecast Summary (Risk-Neutral)",
            "=" * 80,
            "",
            f"{'Expiry':<12} {'TTM':>8} {'Forward':>12} {'Mean':>12} {'StdDev':>10} {'Skew':>8}",
            "-" * 80,
        ]

        for s in all_stats:
            ttm_days = f"{s.ttm*365:.0f}d"
            lines.append(
                f"{s.expiry:<12} {ttm_days:>8} ${s.forward:>10,.0f} ${s.mean:>10,.0f} "
                f"{s.std_dev/s.forward*100:>8.1f}% {s.skewness:>8.2f}"
            )

        lines.append("-" * 80)
        lines.append("")
        lines.append("Percentile Forecasts:")
        lines.append(f"{'Expiry':<12} {'5%':>12} {'25%':>12} {'50%':>12} {'75%':>12} {'95%':>12}")
        lines.append("-" * 80)

        for s in all_stats:
            lines.append(
                f"{s.expiry:<12} ${s.percentiles[5]:>10,.0f} ${s.percentiles[25]:>10,.0f} "
                f"${s.percentiles[50]:>10,.0f} ${s.percentiles[75]:>10,.0f} ${s.percentiles[95]:>10,.0f}"
            )

        return "\n".join(lines)

    def to_json(self, stats: ExpiryStats) -> dict:
        """Convert statistics to JSON-serializable dict.

        Args:
            stats: Expiry statistics.

        Returns:
            Dictionary suitable for JSON output.
        """
        return {
            "expiry": stats.expiry,
            "ttm_days": stats.ttm * 365,
            "forward": stats.forward,
            "mean": stats.mean,
            "mode": stats.mode,
            "median": stats.median,
            "std_dev": stats.std_dev,
            "std_dev_pct": stats.std_dev / stats.forward * 100,
            "skewness": stats.skewness,
            "kurtosis": stats.kurtosis,
            "percentiles": {str(k): v for k, v in stats.percentiles.items()},
            "scenarios": [
                {
                    "description": s.description,
                    "lower": s.lower,
                    "upper": s.upper,
                    "probability": s.probability,
                    "probability_pct": s.probability * 100,
                }
                for s in stats.scenarios
            ]
        }
