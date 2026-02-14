"""Time parsing utilities with timezone support.

Provides functions to parse human-readable time specifications with timezone
abbreviations and convert them to UTC datetimes for TTM calculations.
"""

import re
from datetime import datetime, timezone, timedelta
from typing import Tuple, Optional
from zoneinfo import ZoneInfo

from dateutil import parser as dateutil_parser
from dateutil.relativedelta import relativedelta, MO, TU, WE, TH, FR, SA, SU


# Map common US timezone abbreviations to IANA names
# Note: Ambiguous abbreviations (e.g., CST could be Central or China) default to US
TIMEZONE_ALIASES = {
    # Eastern
    "ET": "America/New_York",
    "EST": "America/New_York",
    "EDT": "America/New_York",
    # Central
    "CT": "America/Chicago",
    "CST": "America/Chicago",
    "CDT": "America/Chicago",
    # Mountain
    "MT": "America/Denver",
    "MST": "America/Denver",
    "MDT": "America/Denver",
    # Pacific
    "PT": "America/Los_Angeles",
    "PST": "America/Los_Angeles",
    "PDT": "America/Los_Angeles",
    # UTC/GMT
    "UTC": "UTC",
    "GMT": "UTC",
    "Z": "UTC",
}

# Day name mappings for relative date parsing
DAY_NAMES = {
    "monday": MO,
    "tuesday": TU,
    "wednesday": WE,
    "thursday": TH,
    "friday": FR,
    "saturday": SA,
    "sunday": SU,
    "mon": MO,
    "tue": TU,
    "wed": WE,
    "thu": TH,
    "fri": FR,
    "sat": SA,
    "sun": SU,
}


class TimeParseError(ValueError):
    """Raised when a time string cannot be parsed."""
    pass


def _extract_timezone(time_str: str) -> Tuple[str, Optional[ZoneInfo], str]:
    """Extract timezone from time string and return cleaned string.

    Returns:
        Tuple of (cleaned_time_str, timezone, original_tz_display)
    """
    # Check for timezone abbreviations at the end of the string
    words = time_str.strip().split()

    if not words:
        raise TimeParseError(f"Empty time string")

    # Check if last word is a timezone abbreviation
    last_word = words[-1].upper()
    if last_word in TIMEZONE_ALIASES:
        tz_name = TIMEZONE_ALIASES[last_word]
        tz = ZoneInfo(tz_name)
        cleaned = " ".join(words[:-1])
        return cleaned, tz, last_word

    # No recognized timezone - this is an error
    raise TimeParseError(
        f"No timezone specified in '{time_str}'. "
        f"Please include a timezone (e.g., 'ET', 'PT', 'UTC')"
    )


def _parse_relative_time(time_str: str, tz: ZoneInfo) -> datetime:
    """Parse relative time expressions like 'tomorrow 5pm' or 'Friday 4pm'.

    Args:
        time_str: Time string without timezone (e.g., "tomorrow 5pm")
        tz: Timezone to interpret the time in

    Returns:
        Datetime in the specified timezone
    """
    time_str_lower = time_str.lower().strip()
    now_local = datetime.now(tz)

    # Pattern: "tomorrow" with optional time
    if time_str_lower.startswith("tomorrow"):
        rest = time_str_lower.replace("tomorrow", "").strip()
        base_date = now_local + timedelta(days=1)

        if rest:
            # Parse the time part
            try:
                time_part = dateutil_parser.parse(rest, fuzzy=True)
                return base_date.replace(
                    hour=time_part.hour,
                    minute=time_part.minute,
                    second=0,
                    microsecond=0
                )
            except (ValueError, TypeError):
                raise TimeParseError(f"Could not parse time '{rest}' in '{time_str}'")
        else:
            # Default to end of day
            return base_date.replace(hour=23, minute=59, second=0, microsecond=0)

    # Pattern: day name with optional time (e.g., "Friday 4pm")
    for day_name, day_obj in DAY_NAMES.items():
        if time_str_lower.startswith(day_name):
            rest = time_str_lower.replace(day_name, "").strip()

            # Find next occurrence of this day
            base_date = now_local + relativedelta(weekday=day_obj(+1))

            if rest:
                try:
                    time_part = dateutil_parser.parse(rest, fuzzy=True)
                    return base_date.replace(
                        hour=time_part.hour,
                        minute=time_part.minute,
                        second=0,
                        microsecond=0
                    )
                except (ValueError, TypeError):
                    raise TimeParseError(f"Could not parse time '{rest}' in '{time_str}'")
            else:
                # Default to 5pm for day-only
                return base_date.replace(hour=17, minute=0, second=0, microsecond=0)

    return None  # Not a relative time expression


def _is_time_only(time_str: str) -> bool:
    """Check if string is just a time without date (e.g., '11:59 PM')."""
    # Remove common time patterns to see if there's a date component
    cleaned = time_str.strip()

    # Check for date indicators
    date_indicators = [
        r'\d{4}',  # Year like 2026
        r'\d{1,2}/\d{1,2}',  # Date like 1/30
        r'\d{1,2}-\d{1,2}',  # Date like 1-30
        r'jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec',  # Month names
        r'tomorrow|monday|tuesday|wednesday|thursday|friday|saturday|sunday',
        r'mon|tue|wed|thu|fri|sat|sun',
    ]

    for pattern in date_indicators:
        if re.search(pattern, cleaned, re.IGNORECASE):
            return False

    # If we only have time-like patterns, it's time-only
    time_pattern = r'^(\d{1,2}(:\d{2})?\s*(am|pm)?|\d{1,2}:\d{2})$'
    return bool(re.match(time_pattern, cleaned, re.IGNORECASE))


def parse_datetime_with_timezone(time_str: str) -> Tuple[datetime, str]:
    """Parse a datetime string with timezone to UTC datetime.

    Supports formats like:
    - "11:59 PM ET" - today (or tomorrow if past)
    - "tomorrow 5pm PT" - relative time
    - "Jan 30 11:59 PM EST" - specific date
    - "2026-01-30 23:59 PST" - ISO-like format
    - "Friday 4pm CT" - next occurrence of day

    Args:
        time_str: Human-readable time string with timezone

    Returns:
        Tuple of (utc_datetime, original_tz_display_string)

    Raises:
        TimeParseError: If the time string cannot be parsed
    """
    if not time_str or not time_str.strip():
        raise TimeParseError("Empty time string")

    # Extract timezone
    cleaned_str, tz, tz_display = _extract_timezone(time_str)

    if not cleaned_str:
        raise TimeParseError(f"No time specified in '{time_str}'")

    # Try relative time parsing first
    result = _parse_relative_time(cleaned_str, tz)
    if result is not None:
        # Convert to UTC
        utc_result = result.astimezone(timezone.utc)
        return utc_result, tz_display

    # Check if it's time-only (no date component)
    if _is_time_only(cleaned_str):
        now_local = datetime.now(tz)
        try:
            time_part = dateutil_parser.parse(cleaned_str, fuzzy=True)
            target = now_local.replace(
                hour=time_part.hour,
                minute=time_part.minute,
                second=0,
                microsecond=0
            )

            # If time has already passed today, assume tomorrow
            if target <= now_local:
                target += timedelta(days=1)

            utc_result = target.astimezone(timezone.utc)
            return utc_result, tz_display

        except (ValueError, TypeError) as e:
            raise TimeParseError(f"Could not parse time '{cleaned_str}': {e}")

    # Try standard dateutil parsing for full datetime
    try:
        # Parse without timezone info first
        parsed = dateutil_parser.parse(cleaned_str, fuzzy=True)

        # Apply the extracted timezone
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=tz)

        # Convert to UTC
        utc_result = parsed.astimezone(timezone.utc)
        return utc_result, tz_display

    except (ValueError, TypeError) as e:
        raise TimeParseError(f"Could not parse datetime '{time_str}': {e}")


def calculate_ttm_to_target(target_utc: datetime) -> float:
    """Calculate TTM in years from now to target UTC datetime.

    Args:
        target_utc: Target datetime in UTC

    Returns:
        Time to maturity in years (can be fractional)

    Raises:
        TimeParseError: If target time is in the past
    """
    now_utc = datetime.now(timezone.utc)
    delta = target_utc - now_utc

    if delta.total_seconds() <= 0:
        raise TimeParseError(
            f"Target time {target_utc.strftime('%Y-%m-%d %H:%M UTC')} is in the past"
        )

    # Convert to years (using 365.25 days/year)
    return delta.total_seconds() / (365.25 * 24 * 3600)


def format_target_time(target_utc: datetime, original_tz: str) -> str:
    """Format target time for display.

    Args:
        target_utc: Target datetime in UTC
        original_tz: Original timezone abbreviation

    Returns:
        Formatted string like "30 Jan 2026 04:59 UTC (11:59 PM ET)"
    """
    # Get the original timezone for local display
    local_str = ""
    if original_tz.upper() in TIMEZONE_ALIASES:
        tz = ZoneInfo(TIMEZONE_ALIASES[original_tz.upper()])
        local_time = target_utc.astimezone(tz)
        # Handle Windows strftime (doesn't support %-I)
        try:
            local_str = local_time.strftime("%-I:%M %p")
        except ValueError:
            # Windows fallback - remove leading zero manually
            local_str = local_time.strftime("%I:%M %p").lstrip("0")

    utc_str = target_utc.strftime("%d %b %Y %H:%M UTC")

    if local_str:
        return f"{utc_str} ({local_str} {original_tz})"
    return utc_str
