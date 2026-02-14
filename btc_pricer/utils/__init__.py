"""Utility modules."""

# fit_stats has no dependencies on models, safe to import at module level
from .fit_stats import FitStats, calculate_fit_stats, calculate_iv_error_stats

# Time parsing utilities
from .time_parser import (
    parse_datetime_with_timezone,
    calculate_ttm_to_target,
    format_target_time,
    TimeParseError,
)


def __getattr__(name):
    """Lazy import of sanity_checks to avoid circular imports."""
    if name == "SanityChecker":
        from .sanity_checks import SanityChecker
        return SanityChecker
    elif name == "CheckStatus":
        from .sanity_checks import CheckStatus
        return CheckStatus
    elif name == "SanityCheckResult":
        from .sanity_checks import SanityCheckResult
        return SanityCheckResult
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "SanityChecker",
    "CheckStatus",
    "SanityCheckResult",
    "FitStats",
    "calculate_fit_stats",
    "calculate_iv_error_stats",
    "parse_datetime_with_timezone",
    "calculate_ttm_to_target",
    "format_target_time",
    "TimeParseError",
]
