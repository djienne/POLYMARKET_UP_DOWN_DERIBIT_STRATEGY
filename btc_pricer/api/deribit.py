"""Deribit API client for fetching Bitcoin options data."""

import re
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import requests

from ..config import APIConfig, ValidationConfig


@dataclass
class OptionData:
    """Container for option data."""
    instrument_name: str
    strike: float
    option_type: str  # 'call' or 'put'
    expiration_timestamp: int
    expiration_date: str
    bid_price: Optional[float]  # In BTC
    ask_price: Optional[float]  # In BTC
    mark_price: float  # In BTC
    mark_iv: float  # As decimal (e.g., 0.80 for 80%)
    bid_iv: Optional[float]
    ask_iv: Optional[float]
    open_interest: float
    underlying_price: float  # Forward price
    spot_price: float
    time_to_expiry: float  # In years


class DeribitAPIError(Exception):
    """Exception raised for Deribit API errors."""
    pass


class DeribitClient:
    """Client for Deribit public API."""

    def __init__(
        self,
        config: Optional[APIConfig] = None,
        validation_config: Optional[ValidationConfig] = None
    ):
        """Initialize the Deribit API client.

        Args:
            config: API configuration. Uses defaults if not provided.
            validation_config: Validation config for spot price bounds.
        """
        self.config = config or APIConfig()
        self.validation_config = validation_config or ValidationConfig()
        self.session = requests.Session()
        self._last_request_time = 0.0
        self._min_request_interval = 0.1  # 100ms between requests

    def _rate_limit(self) -> None:
        """Enforce rate limiting."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self._min_request_interval:
            time.sleep(self._min_request_interval - elapsed)
        self._last_request_time = time.time()

    def _request(self, endpoint: str, params: Optional[Dict] = None) -> dict:
        """Make a request to the Deribit API.

        Args:
            endpoint: API endpoint (without base URL).
            params: Query parameters.

        Returns:
            Response data from the API.

        Raises:
            DeribitAPIError: If the request fails.
        """
        self._rate_limit()
        url = f"{self.config.base_url}{endpoint}"

        for attempt in range(self.config.max_retries):
            try:
                response = self.session.get(
                    url,
                    params=params,
                    timeout=self.config.timeout
                )
                response.raise_for_status()
                data = response.json()

                if "error" in data:
                    raise DeribitAPIError(
                        f"API error: {data['error'].get('message', 'Unknown error')}"
                    )

                return data.get("result", data)

            except requests.RequestException as e:
                if attempt == self.config.max_retries - 1:
                    raise DeribitAPIError(f"Request failed after {self.config.max_retries} attempts: {e}")
                time.sleep(2 ** attempt)  # Exponential backoff

        raise DeribitAPIError("Request failed")

    def get_index_price(self, currency: str = "BTC") -> float:
        """Get the current index (spot) price.

        Args:
            currency: Currency symbol (default: BTC).

        Returns:
            Current index price in USD.
        """
        result = self._request("get_index_price", {"index_name": f"{currency.lower()}_usd"})
        return result["index_price"]

    def get_instruments(
        self,
        currency: str = "BTC",
        kind: str = "option",
        expired: bool = False
    ) -> List[dict]:
        """Get available instruments.

        Args:
            currency: Currency symbol.
            kind: Instrument type (option, future, etc.).
            expired: Include expired instruments.

        Returns:
            List of instrument data.
        """
        params = {
            "currency": currency,
            "kind": kind,
            "expired": str(expired).lower()
        }
        return self._request("get_instruments", params)

    def get_book_summary_by_currency(
        self,
        currency: str = "BTC",
        kind: str = "option"
    ) -> List[dict]:
        """Get book summary for all instruments of a currency.

        This is the most efficient way to get option data as it returns
        all options in a single request.

        Args:
            currency: Currency symbol.
            kind: Instrument type.

        Returns:
            List of book summary data for all instruments.
        """
        params = {
            "currency": currency,
            "kind": kind
        }
        return self._request("get_book_summary_by_currency", params)

    def get_order_book(
        self,
        instrument_name: str,
        depth: int = 1
    ) -> dict:
        """Get order book for a specific instrument.

        Args:
            instrument_name: Instrument name (e.g., "BTC-27DEC24-100000-C").
            depth: Order book depth.

        Returns:
            Order book data including greeks and IVs.
        """
        params = {
            "instrument_name": instrument_name,
            "depth": depth
        }
        return self._request("get_order_book", params)

    @staticmethod
    def parse_instrument_name(instrument_name: str) -> Tuple[str, str, float, str]:
        """Parse a Deribit instrument name.

        Args:
            instrument_name: Instrument name (e.g., "BTC-27DEC24-100000-C").

        Returns:
            Tuple of (currency, expiry_str, strike, option_type).

        Raises:
            ValueError: If the instrument name format is invalid.
        """
        pattern = r"^(\w+)-(\d{1,2}[A-Z]{3}\d{2})-(\d+)-([CP])$"
        match = re.match(pattern, instrument_name)

        if not match:
            raise ValueError(f"Invalid instrument name format: {instrument_name}")

        currency, expiry_str, strike_str, opt_type = match.groups()
        strike = float(strike_str)
        option_type = "call" if opt_type == "C" else "put"

        return currency, expiry_str, strike, option_type

    @staticmethod
    def parse_expiry_string(expiry_str: str) -> datetime:
        """Parse expiry string to datetime.

        Args:
            expiry_str: Expiry string (e.g., "27DEC24").

        Returns:
            Datetime object for the expiry.
        """
        return datetime.strptime(expiry_str, "%d%b%y")

    def fetch_all_options(self, currency: str = "BTC") -> Dict[str, List[OptionData]]:
        """Fetch all options data grouped by expiry.

        This uses the bulk endpoint for efficiency.

        Args:
            currency: Currency symbol.

        Returns:
            Dictionary mapping expiry dates to lists of OptionData.
        """
        # Get spot price
        spot_price = self.get_index_price(currency)

        # Validate spot price using config
        spot_min = self.validation_config.spot_price_min
        spot_max = self.validation_config.spot_price_max
        if not spot_min <= spot_price <= spot_max:
            raise DeribitAPIError(
                f"Spot price {spot_price} outside expected range [${spot_min:,.0f} - ${spot_max:,.0f}]"
            )

        # Get all option book summaries
        book_summaries = self.get_book_summary_by_currency(currency, "option")

        if not book_summaries:
            raise DeribitAPIError("No options data returned from API")

        # Get instrument details for expiration timestamps
        instruments = self.get_instruments(currency, "option", expired=False)
        instrument_map = {inst["instrument_name"]: inst for inst in instruments}

        # Process options and group by expiry
        options_by_expiry: Dict[str, List[OptionData]] = {}
        current_time = time.time()

        for summary in book_summaries:
            instrument_name = summary.get("instrument_name", "")

            # Skip if no instrument data
            if instrument_name not in instrument_map:
                continue

            inst_data = instrument_map[instrument_name]

            try:
                _, expiry_str, strike, option_type = self.parse_instrument_name(
                    instrument_name
                )
            except ValueError:
                continue

            # Calculate time to expiry
            expiration_ts = inst_data["expiration_timestamp"] / 1000  # ms to seconds
            time_to_expiry = (expiration_ts - current_time) / (365.25 * 24 * 3600)

            if time_to_expiry <= 0:
                continue

            # Get underlying (forward) price - use mark price's underlying
            underlying_price = summary.get("underlying_price", spot_price)

            # Extract prices (in BTC)
            bid_price = summary.get("bid_price")
            ask_price = summary.get("ask_price")
            mark_price = summary.get("mark_price", 0)

            # Extract IVs (convert from percentage to decimal)
            mark_iv = summary.get("mark_iv", 0) / 100

            # Open interest
            open_interest = summary.get("open_interest", 0)

            option_data = OptionData(
                instrument_name=instrument_name,
                strike=strike,
                option_type=option_type,
                expiration_timestamp=int(expiration_ts * 1000),
                expiration_date=expiry_str,
                bid_price=bid_price,
                ask_price=ask_price,
                mark_price=mark_price,
                mark_iv=mark_iv,
                bid_iv=None,  # Not available in book summary
                ask_iv=None,
                open_interest=open_interest,
                underlying_price=underlying_price,
                spot_price=spot_price,
                time_to_expiry=time_to_expiry
            )

            if expiry_str not in options_by_expiry:
                options_by_expiry[expiry_str] = []
            options_by_expiry[expiry_str].append(option_data)

        # Sort options by strike within each expiry
        for expiry in options_by_expiry:
            options_by_expiry[expiry].sort(key=lambda x: x.strike)

        return options_by_expiry

    def fetch_option_details(self, instrument_name: str) -> dict:
        """Fetch detailed data for a specific option including greeks.

        Args:
            instrument_name: Instrument name.

        Returns:
            Detailed option data from order book endpoint.
        """
        return self.get_order_book(instrument_name, depth=1)
