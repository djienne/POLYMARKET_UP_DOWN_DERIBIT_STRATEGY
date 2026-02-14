"""Data quality filters for options data."""

from dataclasses import dataclass
from typing import List, Optional, Tuple
import logging

from ..api.deribit import OptionData
from ..config import FilterConfig


logger = logging.getLogger(__name__)


@dataclass
class FilteredOption:
    """Container for filtered option data with computed fields."""
    instrument_name: str
    strike: float
    option_type: str
    time_to_expiry: float
    forward_price: float
    spot_price: float
    bid_price_btc: float
    ask_price_btc: float
    mid_price_btc: float
    mark_price_btc: float
    mark_iv: float
    open_interest: float
    moneyness: float  # K / F
    log_moneyness: float  # ln(K / F)
    is_otm: bool  # True if out-of-the-money


@dataclass
class FilterStats:
    """Statistics from filtering process."""
    total_options: int
    passed_filters: int
    failed_open_interest: int
    failed_bid_ask_existence: int
    failed_spread: int
    failed_ttl: int
    failed_moneyness: int


class DataFilter:
    """Filter options data for quality and relevance."""

    def __init__(self, config: Optional[FilterConfig] = None):
        """Initialize the data filter.

        Args:
            config: Filter configuration. Uses defaults if not provided.
        """
        self.config = config or FilterConfig()

    def filter_options(
        self,
        options: List[OptionData],
        return_stats: bool = False
    ) -> Tuple[List[FilteredOption], Optional[FilterStats]]:
        """Apply all quality filters to options data.

        Args:
            options: List of raw option data.
            return_stats: If True, return filtering statistics.

        Returns:
            List of filtered options, optionally with statistics.
        """
        stats = FilterStats(
            total_options=len(options),
            passed_filters=0,
            failed_open_interest=0,
            failed_bid_ask_existence=0,
            failed_spread=0,
            failed_ttl=0,
            failed_moneyness=0
        )

        filtered = []

        for opt in options:
            # Check open interest
            if opt.open_interest < self.config.min_open_interest:
                stats.failed_open_interest += 1
                continue

            # Check bid/ask existence
            if opt.bid_price is None or opt.ask_price is None:
                stats.failed_bid_ask_existence += 1
                continue

            if opt.bid_price <= 0 or opt.ask_price <= 0:
                stats.failed_bid_ask_existence += 1
                continue

            # Check bid < ask
            if opt.bid_price >= opt.ask_price:
                stats.failed_bid_ask_existence += 1
                continue

            # Check bid-ask spread
            mid_price = (opt.bid_price + opt.ask_price) / 2
            spread = (opt.ask_price - opt.bid_price) / mid_price

            if spread > self.config.max_bid_ask_spread_pct:
                stats.failed_spread += 1
                continue

            # Check time to expiry
            days_to_expiry = opt.time_to_expiry * 365.25
            if days_to_expiry < self.config.min_days_to_expiry:
                stats.failed_ttl += 1
                continue

            # Check moneyness
            forward = opt.underlying_price
            moneyness = opt.strike / forward

            min_m, max_m = self.config.moneyness_range
            if not (min_m <= moneyness <= max_m):
                stats.failed_moneyness += 1
                continue

            # Compute log moneyness
            import math
            log_moneyness = math.log(opt.strike / forward)

            # Determine if OTM
            is_otm = (
                (opt.option_type == "put" and opt.strike < forward) or
                (opt.option_type == "call" and opt.strike > forward)
            )

            filtered_opt = FilteredOption(
                instrument_name=opt.instrument_name,
                strike=opt.strike,
                option_type=opt.option_type,
                time_to_expiry=opt.time_to_expiry,
                forward_price=forward,
                spot_price=opt.spot_price,
                bid_price_btc=opt.bid_price,
                ask_price_btc=opt.ask_price,
                mid_price_btc=mid_price,
                mark_price_btc=opt.mark_price,
                mark_iv=opt.mark_iv,
                open_interest=opt.open_interest,
                moneyness=moneyness,
                log_moneyness=log_moneyness,
                is_otm=is_otm
            )

            filtered.append(filtered_opt)

        stats.passed_filters = len(filtered)

        if return_stats:
            return filtered, stats
        return filtered, None

    def select_otm_options(
        self,
        options: List[FilteredOption]
    ) -> List[FilteredOption]:
        """Select only out-of-the-money options.

        For RND extraction, we use:
        - Puts for K < Forward
        - Calls for K > Forward

        Args:
            options: List of filtered options.

        Returns:
            List of OTM options only.
        """
        return [opt for opt in options if opt.is_otm]

    def build_otm_surface(
        self,
        options: List[FilteredOption]
    ) -> List[FilteredOption]:
        """Build a volatility surface using OTM options.

        At each strike, select the OTM option (put for K < F, call for K > F).
        For ATM, average the put and call if both exist.

        Args:
            options: List of filtered options.

        Returns:
            List of options forming the OTM surface.
        """
        # Group by strike
        by_strike = {}
        for opt in options:
            if opt.strike not in by_strike:
                by_strike[opt.strike] = {}
            by_strike[opt.strike][opt.option_type] = opt

        surface = []

        for strike in sorted(by_strike.keys()):
            opts_at_strike = by_strike[strike]

            # Get forward from any option at this strike
            if opts_at_strike:
                forward = list(opts_at_strike.values())[0].forward_price
            else:
                continue

            # Select OTM option
            if strike < forward and "put" in opts_at_strike:
                surface.append(opts_at_strike["put"])
            elif strike > forward and "call" in opts_at_strike:
                surface.append(opts_at_strike["call"])
            elif strike == forward:
                # ATM: prefer call if available
                if "call" in opts_at_strike:
                    surface.append(opts_at_strike["call"])
                elif "put" in opts_at_strike:
                    surface.append(opts_at_strike["put"])

        return surface

    def validate_surface_coverage(
        self,
        options: List[FilteredOption]
    ) -> Tuple[bool, str]:
        """Validate that the options cover both sides of ATM.

        Args:
            options: List of filtered options.

        Returns:
            Tuple of (is_valid, message).
        """
        if not options:
            return False, "No options available"

        forward = options[0].forward_price

        strikes_below = [opt.strike for opt in options if opt.strike < forward]
        strikes_above = [opt.strike for opt in options if opt.strike > forward]

        if not strikes_below:
            return False, "No strikes below forward price"

        if not strikes_above:
            return False, "No strikes above forward price"

        min_points = getattr(self.config, 'min_surface_points', 5)
        if len(options) < min_points:
            return False, f"Only {len(options)} options, need at least {min_points}"

        return True, "Surface coverage is adequate"
