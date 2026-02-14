"""Visualization for RND analysis."""

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from ..models.breeden_litzenberger import RNDResult
from ..models.ssvi import SSVIModel, SSVIParams
from ..models.heston import HestonModel, HestonParams
from ..config import HestonConfig


class RNDPlotter:
    """Create visualizations for RND analysis."""

    def __init__(
        self,
        figsize: Tuple[int, int] = (12, 8),
        dpi: int = 150,
        style: str = "seaborn-v0_8-whitegrid",
        heston_config: Optional[HestonConfig] = None
    ):
        """Initialize the plotter.

        Args:
            figsize: Default figure size.
            dpi: Resolution for saved figures.
            style: Matplotlib style.
            heston_config: Optional Heston config for model instantiation.
        """
        self.figsize = figsize
        self.dpi = dpi
        self.heston_config = heston_config
        try:
            plt.style.use(style)
        except OSError:
            # Fallback if style not available
            plt.style.use("seaborn-whitegrid")

    def plot_single_density(
        self,
        rnd: RNDResult,
        expiry: str,
        spot_price: float,
        show_percentiles: bool = True,
        save_path: Optional[Path] = None
    ) -> Figure:
        """Plot a single RND density curve.

        Args:
            rnd: RND result.
            expiry: Expiry date string.
            spot_price: Current spot price.
            show_percentiles: Show percentile lines.
            save_path: Path to save figure.

        Returns:
            Matplotlib figure.
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        # Plot density
        ax.fill_between(
            rnd.strikes / 1000,
            rnd.density * 1000,
            alpha=0.3,
            label="Probability Density"
        )
        ax.plot(rnd.strikes / 1000, rnd.density * 1000, linewidth=2)

        # Vertical lines
        ax.axvline(
            spot_price / 1000, color='green', linestyle='--',
            linewidth=1.5, label=f'Spot: ${spot_price:,.0f}'
        )
        ax.axvline(
            rnd.forward / 1000, color='blue', linestyle='--',
            linewidth=1.5, label=f'Forward: ${rnd.forward:,.0f}'
        )
        ax.axvline(
            rnd.mean / 1000, color='red', linestyle=':',
            linewidth=1.5, label=f'Mean: ${rnd.mean:,.0f}'
        )

        if show_percentiles:
            for pct, val, alpha in [
                (5, rnd.percentile_5, 0.3),
                (25, rnd.percentile_25, 0.5),
                (75, rnd.percentile_75, 0.5),
                (95, rnd.percentile_95, 0.3)
            ]:
                ax.axvline(
                    val / 1000, color='gray', linestyle=':',
                    linewidth=1, alpha=alpha
                )
                ax.text(
                    val / 1000, ax.get_ylim()[1] * 0.9,
                    f'{pct}%', fontsize=8, ha='center', alpha=0.7
                )

        ax.set_xlabel("BTC Price ($K)", fontsize=12)
        ax.set_ylabel("Probability Density", fontsize=12)
        ax.set_title(
            f"Risk-Neutral Density: {expiry}\n"
            f"(TTM: {rnd.ttm*365:.1f} days, Skew: {rnd.skewness:.2f}, Kurt: {rnd.kurtosis:.2f})",
            fontsize=14
        )
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

        # Set reasonable x-axis limits
        x_min = max(spot_price * 0.3, rnd.percentile_5 * 0.9) / 1000
        x_max = min(spot_price * 2.5, rnd.percentile_95 * 1.1) / 1000
        ax.set_xlim(x_min, x_max)

        # Add generation timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M UTC")
        fig.text(
            0.99, 0.01, f"Generated: {timestamp}",
            ha='right', va='bottom', fontsize=8, color='gray',
            transform=fig.transFigure
        )

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')

        return fig

    def plot_multiple_densities(
        self,
        rnds: Dict[str, RNDResult],
        spot_price: float,
        save_path: Optional[Path] = None
    ) -> Figure:
        """Plot multiple RND densities overlaid.

        Args:
            rnds: Dictionary mapping expiry to RND result.
            spot_price: Current spot price.
            save_path: Path to save figure.

        Returns:
            Matplotlib figure.
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        # Sort by TTM
        sorted_expiries = sorted(rnds.keys(), key=lambda x: rnds[x].ttm)

        # Color map
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(sorted_expiries)))

        for expiry, color in zip(sorted_expiries, colors):
            rnd = rnds[expiry]
            ttm_days = rnd.ttm * 365
            ax.plot(
                rnd.strikes / 1000,
                rnd.density * 1000,
                linewidth=2,
                color=color,
                label=f"{expiry} ({ttm_days:.0f}d)"
            )

        ax.axvline(
            spot_price / 1000, color='black', linestyle='--',
            linewidth=2, label=f'Spot: ${spot_price:,.0f}'
        )

        ax.set_xlabel("BTC Price ($K)", fontsize=12)
        ax.set_ylabel("Probability Density", fontsize=12)
        ax.set_title("Risk-Neutral Densities by Expiry", fontsize=14)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

        # Set x-limits based on all densities
        all_p5 = [rnd.percentile_5 for rnd in rnds.values()]
        all_p95 = [rnd.percentile_95 for rnd in rnds.values()]
        x_min = max(spot_price * 0.3, min(all_p5) * 0.9) / 1000
        x_max = min(spot_price * 3.0, max(all_p95) * 1.1) / 1000
        ax.set_xlim(x_min, x_max)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')

        return fig

    def plot_fan_chart(
        self,
        rnds: Dict[str, RNDResult],
        spot_price: float,
        save_path: Optional[Path] = None
    ) -> Figure:
        """Create a fan chart showing price confidence bands over time.

        Args:
            rnds: Dictionary mapping expiry to RND result.
            spot_price: Current spot price.
            save_path: Path to save figure.

        Returns:
            Matplotlib figure.
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        # Sort by TTM
        sorted_items = sorted(rnds.items(), key=lambda x: x[1].ttm)

        ttms = [0] + [rnd.ttm * 365 for _, rnd in sorted_items]
        p5s = [spot_price] + [rnd.percentile_5 for _, rnd in sorted_items]
        p25s = [spot_price] + [rnd.percentile_25 for _, rnd in sorted_items]
        p50s = [spot_price] + [rnd.percentile_50 for _, rnd in sorted_items]
        p75s = [spot_price] + [rnd.percentile_75 for _, rnd in sorted_items]
        p95s = [spot_price] + [rnd.percentile_95 for _, rnd in sorted_items]
        means = [spot_price] + [rnd.mean for _, rnd in sorted_items]

        # Convert to $K
        p5s = np.array(p5s) / 1000
        p25s = np.array(p25s) / 1000
        p50s = np.array(p50s) / 1000
        p75s = np.array(p75s) / 1000
        p95s = np.array(p95s) / 1000
        means = np.array(means) / 1000

        # Fill bands
        ax.fill_between(ttms, p5s, p95s, alpha=0.2, color='blue', label='5-95%')
        ax.fill_between(ttms, p25s, p75s, alpha=0.3, color='blue', label='25-75%')

        # Lines
        ax.plot(ttms, p50s, 'b-', linewidth=2, label='Median')
        ax.plot(ttms, means, 'r--', linewidth=1.5, label='Mean')

        # Add expiry labels
        for expiry, rnd in sorted_items:
            ttm_days = rnd.ttm * 365
            ax.annotate(
                expiry,
                (ttm_days, rnd.percentile_50 / 1000),
                textcoords="offset points",
                xytext=(0, 10),
                ha='center',
                fontsize=8
            )

        ax.axhline(
            spot_price / 1000, color='green', linestyle='--',
            linewidth=1.5, label=f'Current: ${spot_price:,.0f}'
        )

        ax.set_xlabel("Days to Expiry", fontsize=12)
        ax.set_ylabel("BTC Price ($K)", fontsize=12)
        ax.set_title("BTC Price Fan Chart (Risk-Neutral)", fontsize=14)
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')

        return fig

    def plot_iv_smile(
        self,
        log_moneyness: np.ndarray,
        market_iv: np.ndarray,
        params: Union[SSVIParams, HestonParams],
        expiry: str,
        save_path: Optional[Path] = None,
        ttm: Optional[float] = None
    ) -> Figure:
        """Plot IV smile with model fit.

        Args:
            log_moneyness: Array of log-moneyness values.
            market_iv: Array of market implied volatilities.
            params: Fitted model parameters (SSVI or Heston).
            expiry: Expiry date string.
            save_path: Path to save figure.
            ttm: Time to maturity in years (optional, extracted from params if available).

        Returns:
            Matplotlib figure.
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        # Market data
        ax.scatter(
            log_moneyness * 100,
            market_iv * 100,
            s=50,
            alpha=0.7,
            label='Market IV',
            zorder=5
        )

        # Model fit
        k_fine = np.linspace(log_moneyness.min(), log_moneyness.max(), 200)

        if isinstance(params, HestonParams):
            # Use config if available for n_integration_points and use_quantlib
            if self.heston_config is not None:
                model = HestonModel(
                    params,
                    n_integration_points=self.heston_config.n_integration_points,
                    use_quantlib=self.heston_config.use_quantlib
                )
            else:
                model = HestonModel(params)
            model_iv = model.implied_volatility_array(k_fine)
            model_name = "Heston"
            title_params = (
                f"(v0={params.v0:.4f}, κ={params.kappa:.2f}, "
                f"θ={params.theta:.4f}, ξ={params.xi:.2f}, ρ={params.rho:.3f})"
            )
            # Get TTM from params if not provided
            if ttm is None:
                ttm = params.ttm
        else:
            model = SSVIModel(params)
            model_iv = model.implied_volatility_array(k_fine)
            model_name = "SSVI"
            title_params = f"(θ={params.theta:.4f}, ρ={params.rho:.3f}, φ={params.phi:.3f})"
            # Get TTM from params if not provided
            if ttm is None:
                ttm = params.ttm

        ax.plot(
            k_fine * 100,
            model_iv * 100,
            'r-',
            linewidth=2,
            label=f'{model_name} Fit'
        )

        ax.axvline(0, color='gray', linestyle='--', alpha=0.5)

        ax.set_xlabel("Log Moneyness (%)", fontsize=12)
        ax.set_ylabel("Implied Volatility (%)", fontsize=12)

        # Include TTM in title if available
        ttm_str = f", TTM: {ttm*365:.1f} days" if ttm is not None else ""
        ax.set_title(
            f"IV Smile: {expiry}{ttm_str}\n{title_params}",
            fontsize=14
        )
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

        # Add generation timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M UTC")
        fig.text(
            0.99, 0.01, f"Generated: {timestamp}",
            ha='right', va='bottom', fontsize=8, color='gray',
            transform=fig.transFigure
        )

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')

        return fig

    def plot_term_structure(
        self,
        ssvi_params_by_expiry: Dict[str, SSVIParams],
        save_path: Optional[Path] = None
    ) -> Figure:
        """Plot ATM volatility term structure.

        Args:
            ssvi_params_by_expiry: SSVI params by expiry.
            save_path: Path to save figure.

        Returns:
            Matplotlib figure.
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        # Sort by TTM
        sorted_items = sorted(
            ssvi_params_by_expiry.items(),
            key=lambda x: x[1].ttm
        )

        expiries = [exp for exp, _ in sorted_items]
        ttms = [p.ttm * 365 for _, p in sorted_items]
        atm_vols = [np.sqrt(p.theta / p.ttm) * 100 for _, p in sorted_items]

        ax.plot(ttms, atm_vols, 'bo-', linewidth=2, markersize=8)

        for i, exp in enumerate(expiries):
            ax.annotate(
                exp,
                (ttms[i], atm_vols[i]),
                textcoords="offset points",
                xytext=(0, 10),
                ha='center',
                fontsize=9
            )

        ax.set_xlabel("Days to Expiry", fontsize=12)
        ax.set_ylabel("ATM Implied Volatility (%)", fontsize=12)
        ax.set_title("BTC ATM Volatility Term Structure", fontsize=14)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')

        return fig

    def close_all(self):
        """Close all open figures."""
        plt.close('all')

    def plot_intraday_ranges(
        self,
        forecasts: list,
        spot_price: float,
        atm_iv: float,
        save_path: Optional[Path] = None
    ) -> Figure:
        """Plot intraday price forecast ranges.

        Args:
            forecasts: List of IntradayForecast objects.
            spot_price: Current spot price.
            atm_iv: ATM implied volatility used.
            save_path: Path to save figure.

        Returns:
            Matplotlib figure.
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        hours = [f.hours for f in forecasts]
        p5 = [f.percentile_5 / 1000 for f in forecasts]
        p25 = [f.percentile_25 / 1000 for f in forecasts]
        p50 = [f.percentile_50 / 1000 for f in forecasts]
        p75 = [f.percentile_75 / 1000 for f in forecasts]
        p95 = [f.percentile_95 / 1000 for f in forecasts]

        # Add t=0 point
        hours = [0] + hours
        spot_k = spot_price / 1000
        p5 = [spot_k] + p5
        p25 = [spot_k] + p25
        p50 = [spot_k] + p50
        p75 = [spot_k] + p75
        p95 = [spot_k] + p95

        # Fill confidence bands
        ax.fill_between(hours, p5, p95, alpha=0.2, color='blue', label='5-95%')
        ax.fill_between(hours, p25, p75, alpha=0.3, color='blue', label='25-75%')

        # Median line
        ax.plot(hours, p50, 'b-', linewidth=2, label='Median')

        # Spot price line
        ax.axhline(spot_k, color='green', linestyle='--', linewidth=1.5,
                   label=f'Spot: ${spot_price:,.0f}')

        ax.set_xlabel("Hours Ahead", fontsize=12)
        ax.set_ylabel("BTC Price ($K)", fontsize=12)
        ax.set_title(
            f"Intraday Price Forecast (ATM IV: {atm_iv*100:.1f}%)\n"
            f"Spot: ${spot_price:,.0f}",
            fontsize=14
        )
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)

        # Set x-axis to show nice tick marks
        ax.set_xticks([0, 1, 2, 4, 6, 8, 12, 24, 48, 72])
        ax.set_xlim(0, max(hours) * 1.05)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')

        return fig

    def plot_intraday_move_probs(
        self,
        forecasts: list,
        save_path: Optional[Path] = None
    ) -> Figure:
        """Plot probability of price moves over time.

        Args:
            forecasts: List of IntradayForecast objects.
            save_path: Path to save figure.

        Returns:
            Matplotlib figure.
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        hours = [f.hours for f in forecasts]

        # Upside probabilities
        ax1.plot(hours, [f.prob_up_1pct * 100 for f in forecasts],
                 'g-', linewidth=2, marker='o', label='>+1%')
        ax1.plot(hours, [f.prob_up_5pct * 100 for f in forecasts],
                 'g--', linewidth=2, marker='s', label='>+5%')
        ax1.plot(hours, [f.prob_up_10pct * 100 for f in forecasts],
                 'g:', linewidth=2, marker='^', label='>+10%')

        ax1.set_xlabel("Hours Ahead", fontsize=12)
        ax1.set_ylabel("Probability (%)", fontsize=12)
        ax1.set_title("Probability of Upside Moves", fontsize=14)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 60)

        # Downside probabilities
        ax2.plot(hours, [f.prob_down_1pct * 100 for f in forecasts],
                 'r-', linewidth=2, marker='o', label='<-1%')
        ax2.plot(hours, [f.prob_down_5pct * 100 for f in forecasts],
                 'r--', linewidth=2, marker='s', label='<-5%')
        ax2.plot(hours, [f.prob_down_10pct * 100 for f in forecasts],
                 'r:', linewidth=2, marker='^', label='<-10%')

        ax2.set_xlabel("Hours Ahead", fontsize=12)
        ax2.set_ylabel("Probability (%)", fontsize=12)
        ax2.set_title("Probability of Downside Moves", fontsize=14)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 60)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')

        return fig
