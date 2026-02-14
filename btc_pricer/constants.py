"""Centralized constants for BTC Pricer.

This module contains all magic numbers and hardcoded values that were
previously scattered across the codebase. Using named constants improves
readability and maintainability.
"""

from datetime import datetime
from typing import Tuple

# =============================================================================
# Numerical Thresholds
# =============================================================================

# Tolerance for boundary check comparisons (e.g., parameter at bound detection)
BOUNDARY_CHECK_TOLERANCE: float = 0.001

# Valid range for implied volatility (min, max) - used for filtering
DEFAULT_IV_VALID_RANGE: Tuple[float, float] = (0.05, 5.0)

# Normal/typical IV ranges for validation
IV_NORMAL_RANGE: Tuple[float, float] = (0.10, 3.0)
IV_TYPICAL_RANGE: Tuple[float, float] = (0.20, 2.0)

# Heston model integration limit
HESTON_INTEGRATION_LIMIT: int = 100

# IV bounds for Heston model
HESTON_IV_BOUNDS: Tuple[float, float] = (0.01, 2.0)

# =============================================================================
# Calibration Settings
# =============================================================================

# Minimum points required for model fitting (relaxed for calibration)
RELAXED_MIN_POINTS: int = 5

# ATM penalty weights for Heston fitting: (very_short, short, normal)
ATM_PENALTY_WEIGHTS: Tuple[float, float, float] = (50.0, 10.0, 0.0)

# =============================================================================
# Time/Date Constants
# =============================================================================

# Far future date used as fallback for unparseable expiries (sorts last)
FAR_FUTURE_DATE: datetime = datetime(2099, 12, 31)

# Options expiry time (Deribit expires at 08:00 UTC)
OPTION_EXPIRY_HOUR_UTC: int = 8

# Reference TTM for theta scaling (3 months)
REFERENCE_TTM_MONTHS: float = 0.25

# =============================================================================
# Warning/Error Thresholds
# =============================================================================

# RND/Breeden-Litzenberger warning thresholds (as fractions)
BL_INTEGRAL_WARNING_THRESHOLD: float = 0.05  # 5%
BL_MEAN_WARNING_THRESHOLD: float = 0.10  # 10%

# =============================================================================
# Default Values
# =============================================================================

# Default number of Monte Carlo simulations
DEFAULT_MC_SIMULATIONS: int = 200000

# Default time steps per day for simulation (5-minute steps)
DEFAULT_STEPS_PER_DAY: int = 288

# Default strike grid points for RND extraction
DEFAULT_STRIKE_GRID_POINTS: int = 500

# Default strike range in standard deviations
DEFAULT_STRIKE_RANGE_STD: float = 3.0
