"""Sanity checks for the RND extraction pipeline."""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Dict, TYPE_CHECKING
import numpy as np
import logging

from ..models.breeden_litzenberger import RNDResult
from ..config import ValidationConfig

# Use TYPE_CHECKING to avoid circular imports with models
if TYPE_CHECKING:
    from ..models.ssvi import SSVIParams
    from ..models.heston import HestonParams


logger = logging.getLogger(__name__)


class CheckStatus(Enum):
    """Status of a sanity check."""
    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"
    CRITICAL = "critical"


@dataclass
class SanityCheckResult:
    """Result of a single sanity check."""
    name: str
    status: CheckStatus
    value: Optional[float]
    expected: str
    message: str


@dataclass
class CheckSummary:
    """Summary of all checks for a step."""
    step: str
    results: List[SanityCheckResult]
    overall_status: CheckStatus

    def has_critical(self) -> bool:
        return any(r.status == CheckStatus.CRITICAL for r in self.results)

    def has_failures(self) -> bool:
        return any(r.status in (CheckStatus.FAIL, CheckStatus.CRITICAL) for r in self.results)


class SanityChecker:
    """Perform sanity checks at each step of the pipeline."""

    def __init__(
        self,
        strict: bool = False,
        validation_config: Optional[ValidationConfig] = None
    ):
        """Initialize the sanity checker.

        Args:
            strict: If True, treat warnings as failures.
            validation_config: Optional validation thresholds configuration.
        """
        self.strict = strict
        self.config = validation_config or ValidationConfig()
        self.all_results: Dict[str, CheckSummary] = {}

    def _check_range(
        self,
        step: str,
        name: str,
        value: float,
        normal_range: tuple,
        display_name: str,
        wide_range: Optional[tuple] = None,
        fail_status: CheckStatus = CheckStatus.WARN
    ) -> SanityCheckResult:
        """Check if a value is within expected ranges.

        Args:
            step: Step name for result tracking.
            name: Check name.
            value: Value to check.
            normal_range: Tuple of (low, high) for normal/pass range.
            display_name: Human-readable name for the value.
            wide_range: Optional wider range; if value is outside normal but
                       inside wide range, returns WARN instead of fail_status.
            fail_status: Status to return if value is outside all ranges.

        Returns:
            SanityCheckResult for this check.
        """
        low, high = normal_range
        expected_str = f"{low} - {high}"

        if low <= value <= high:
            return self._add_result(
                step, name, CheckStatus.PASS,
                value, expected_str,
                f"{display_name} = {value:.4f} is within normal range"
            )

        if wide_range is not None:
            wide_low, wide_high = wide_range
            if wide_low <= value <= wide_high:
                return self._add_result(
                    step, name, CheckStatus.WARN,
                    value, expected_str,
                    f"{display_name} = {value:.4f} is outside typical range"
                )

        return self._add_result(
            step, name, fail_status,
            value, expected_str if wide_range is None else f"{wide_range[0]} - {wide_range[1]}",
            f"{display_name} = {value:.4f} is outside valid range"
        )

    def _add_result(
        self,
        step: str,
        name: str,
        status: CheckStatus,
        value: Optional[float],
        expected: str,
        message: str
    ) -> SanityCheckResult:
        """Add a check result."""
        result = SanityCheckResult(
            name=name,
            status=status,
            value=value,
            expected=expected,
            message=message
        )

        if step not in self.all_results:
            self.all_results[step] = CheckSummary(
                step=step,
                results=[],
                overall_status=CheckStatus.PASS
            )

        self.all_results[step].results.append(result)

        # Update overall status
        if status == CheckStatus.CRITICAL:
            self.all_results[step].overall_status = CheckStatus.CRITICAL
        elif status == CheckStatus.FAIL and self.all_results[step].overall_status != CheckStatus.CRITICAL:
            self.all_results[step].overall_status = CheckStatus.FAIL
        elif status == CheckStatus.WARN and self.all_results[step].overall_status == CheckStatus.PASS:
            self.all_results[step].overall_status = CheckStatus.WARN

        # Log based on status
        if status == CheckStatus.CRITICAL:
            logger.error(f"CRITICAL: {name} - {message}")
        elif status == CheckStatus.FAIL:
            logger.error(f"FAIL: {name} - {message}")
        elif status == CheckStatus.WARN:
            logger.warning(f"WARN: {name} - {message}")
        else:
            logger.debug(f"PASS: {name} - {message}")

        return result

    def check_api_data(
        self,
        spot_price: float,
        options_count: int,
        expiry: str
    ) -> CheckSummary:
        """Check API data quality.

        Args:
            spot_price: Spot price from API.
            options_count: Number of options fetched.
            expiry: Expiry being processed.

        Returns:
            CheckSummary for this step.
        """
        step = f"api_data_{expiry}"

        # Spot price range using config
        spot_min = self.config.spot_price_min
        spot_max = self.config.spot_price_max
        if spot_min <= spot_price <= spot_max:
            self._add_result(
                step, "spot_price_range", CheckStatus.PASS,
                spot_price, f"${spot_min:,.0f} - ${spot_max:,.0f}",
                f"Spot price ${spot_price:,.0f} is within expected range"
            )
        else:
            self._add_result(
                step, "spot_price_range", CheckStatus.CRITICAL,
                spot_price, f"${spot_min:,.0f} - ${spot_max:,.0f}",
                f"Spot price ${spot_price:,.0f} is outside expected range - likely API error"
            )

        # Options count
        if options_count >= 10:
            self._add_result(
                step, "options_count", CheckStatus.PASS,
                options_count, ">=10",
                f"Found {options_count} options"
            )
        elif options_count >= 5:
            self._add_result(
                step, "options_count", CheckStatus.WARN,
                options_count, ">=10",
                f"Only {options_count} options found, results may be unreliable"
            )
        else:
            self._add_result(
                step, "options_count", CheckStatus.FAIL,
                options_count, ">=5",
                f"Only {options_count} options found, skipping expiry"
            )

        return self.all_results[step]

    def check_iv(
        self,
        iv: float,
        strike: float,
        forward: float,
        expiry: str
    ) -> CheckStatus:
        """Check if implied volatility is reasonable.

        Args:
            iv: Implied volatility.
            strike: Strike price.
            forward: Forward price.
            expiry: Expiry string.

        Returns:
            Check status.
        """
        step = f"iv_{expiry}"
        moneyness = strike / forward

        # Use config ranges
        iv_normal_min, iv_normal_max = self.config.iv_normal_range
        iv_typical_min, iv_typical_max = self.config.iv_typical_range

        # Normal range
        if iv_normal_min <= iv <= iv_normal_max:
            if iv_typical_min <= iv <= iv_typical_max:
                status = CheckStatus.PASS
                msg = f"IV {iv:.2%} at K/F={moneyness:.2f} is within normal range"
            else:
                status = CheckStatus.WARN
                msg = f"IV {iv:.2%} at K/F={moneyness:.2f} is outside typical range [{iv_typical_min:.0%}, {iv_typical_max:.0%}]"
        else:
            status = CheckStatus.FAIL
            msg = f"IV {iv:.2%} at K/F={moneyness:.2f} is outside valid range [{iv_normal_min:.0%}, {iv_normal_max:.0%}]"

        self._add_result(step, f"iv_at_strike_{strike:.0f}", status, iv, f"{iv_normal_min:.0%} - {iv_normal_max:.0%}", msg)
        return status

    def check_ssvi_params(
        self,
        params: "SSVIParams",
        fit_r2: float,
        max_residual: float,
        expiry: str
    ) -> CheckSummary:
        """Check SSVI parameters and fit quality.

        Args:
            params: Fitted SSVI parameters.
            fit_r2: R-squared of the fit.
            max_residual: Maximum residual in IV.
            expiry: Expiry string.

        Returns:
            CheckSummary for SSVI checks.
        """
        step = f"ssvi_{expiry}"

        # Theta range
        if 0.01 <= params.theta <= 5.0:
            self._add_result(
                step, "theta_range", CheckStatus.PASS,
                params.theta, "0.01 - 5.0",
                f"θ = {params.theta:.4f} is within normal range"
            )
        else:
            self._add_result(
                step, "theta_range", CheckStatus.WARN,
                params.theta, "0.01 - 5.0",
                f"θ = {params.theta:.4f} is outside typical range"
            )

        # Rho range
        if abs(params.rho) <= 0.8:
            self._add_result(
                step, "rho_range", CheckStatus.PASS,
                params.rho, "|ρ| <= 0.8",
                f"ρ = {params.rho:.4f} indicates moderate skew"
            )
        else:
            self._add_result(
                step, "rho_range", CheckStatus.WARN,
                params.rho, "|ρ| <= 0.8",
                f"ρ = {params.rho:.4f} indicates extreme skew"
            )

        # Phi range
        if 0.01 <= params.phi <= 2.0:
            self._add_result(
                step, "phi_range", CheckStatus.PASS,
                params.phi, "0.01 - 2.0",
                f"φ = {params.phi:.4f} is within normal range"
            )
        else:
            self._add_result(
                step, "phi_range", CheckStatus.WARN,
                params.phi, "0.01 - 2.0",
                f"φ = {params.phi:.4f} is outside typical range"
            )

        # Butterfly conditions (Gatheral-Jacquier Theorem 4.2)
        factor = params.theta * (1 + abs(params.rho))
        cond1_val = factor * params.phi
        cond2_val = factor * params.phi ** 2
        binding = "cond2" if cond2_val >= cond1_val else "cond1"
        if params.butterfly_condition():
            self._add_result(
                step, "butterfly", CheckStatus.PASS,
                max(cond1_val, cond2_val),
                "θφ(1+|ρ|) ≤ 4 AND θφ²(1+|ρ|) ≤ 4",
                f"Butterfly satisfied: cond1={cond1_val:.4f}, cond2={cond2_val:.4f} (binding: {binding})"
            )
        else:
            self._add_result(
                step, "butterfly", CheckStatus.CRITICAL,
                max(cond1_val, cond2_val),
                "θφ(1+|ρ|) ≤ 4 AND θφ²(1+|ρ|) ≤ 4",
                f"Butterfly VIOLATED: cond1={cond1_val:.4f}, cond2={cond2_val:.4f} (binding: {binding}) - potential arbitrage"
            )

        # R-squared using config thresholds
        r2_excellent = self.config.r_squared_excellent
        r2_acceptable = self.config.r_squared_acceptable
        r2_poor = self.config.r_squared_poor
        if fit_r2 >= r2_excellent:
            self._add_result(
                step, "fit_r2", CheckStatus.PASS,
                fit_r2, f"R² >= {r2_excellent:.2f}",
                f"Fit quality R² = {fit_r2:.4f} is excellent"
            )
        elif fit_r2 >= r2_acceptable:
            self._add_result(
                step, "fit_r2", CheckStatus.WARN,
                fit_r2, f"R² >= {r2_excellent:.2f}",
                f"Fit quality R² = {fit_r2:.4f} is acceptable"
            )
        elif fit_r2 >= r2_poor:
            self._add_result(
                step, "fit_r2", CheckStatus.FAIL,
                fit_r2, f"R² >= {r2_acceptable:.2f}",
                f"Fit quality R² = {fit_r2:.4f} is poor"
            )
        else:
            self._add_result(
                step, "fit_r2", CheckStatus.CRITICAL,
                fit_r2, f"R² >= {r2_poor:.2f}",
                f"Fit quality R² = {fit_r2:.4f} is unacceptable"
            )

        # Max residual
        if max_residual < 0.05:
            self._add_result(
                step, "max_residual", CheckStatus.PASS,
                max_residual, "< 5%",
                f"Max residual {max_residual:.2%} is small"
            )
        elif max_residual < 0.10:
            self._add_result(
                step, "max_residual", CheckStatus.WARN,
                max_residual, "< 5%",
                f"Max residual {max_residual:.2%} - some outliers may affect fit"
            )
        else:
            self._add_result(
                step, "max_residual", CheckStatus.FAIL,
                max_residual, "< 10%",
                f"Max residual {max_residual:.2%} - significant outliers present"
            )

        return self.all_results[step]

    def check_heston_params(
        self,
        params: "HestonParams",
        fit_r2: float,
        max_residual: float,
        expiry: str
    ) -> CheckSummary:
        """Check Heston parameters and fit quality.

        Args:
            params: Fitted Heston parameters.
            fit_r2: R-squared of the fit.
            max_residual: Maximum residual in IV.
            expiry: Expiry string.

        Returns:
            CheckSummary for Heston checks.
        """
        step = f"heston_{expiry}"

        # v0 range (initial variance)
        self._check_range(
            step, "v0_range", params.v0,
            normal_range=(0.01, 2.0),
            display_name="v0",
            wide_range=(0.005, 4.0),
            fail_status=CheckStatus.FAIL
        )

        # kappa range (mean reversion speed)
        self._check_range(
            step, "kappa_range", params.kappa,
            normal_range=(0.5, 5.0),
            display_name="kappa",
            wide_range=(0.1, 10.0),
            fail_status=CheckStatus.FAIL
        )

        # theta range (long-term variance)
        self._check_range(
            step, "theta_range", params.theta,
            normal_range=(0.01, 2.0),
            display_name="theta",
            wide_range=(0.005, 4.0),
            fail_status=CheckStatus.FAIL
        )

        # xi range (vol-of-vol)
        self._check_range(
            step, "xi_range", params.xi,
            normal_range=(0.2, 2.0),
            display_name="xi",
            wide_range=(0.1, 5.0),
            fail_status=CheckStatus.FAIL
        )

        # rho range (correlation) - special case with absolute value
        if abs(params.rho) <= 0.8:
            self._add_result(
                step, "rho_range", CheckStatus.PASS,
                params.rho, "|rho| <= 0.8",
                f"rho = {params.rho:.4f} indicates moderate correlation"
            )
        else:
            self._add_result(
                step, "rho_range", CheckStatus.WARN,
                params.rho, "|rho| <= 0.8",
                f"rho = {params.rho:.4f} indicates extreme correlation"
            )

        # Feller condition: 2*kappa*theta > xi^2
        feller_ratio = params.feller_ratio()
        if params.feller_condition():
            self._add_result(
                step, "feller_condition", CheckStatus.PASS,
                feller_ratio, "2*kappa*theta/xi^2 > 1",
                f"Feller condition satisfied: ratio = {feller_ratio:.4f} > 1"
            )
        else:
            self._add_result(
                step, "feller_condition", CheckStatus.WARN,
                feller_ratio, "2*kappa*theta/xi^2 > 1",
                f"Feller condition VIOLATED: ratio = {feller_ratio:.4f} <= 1 - variance may go negative"
            )

        # R-squared
        if fit_r2 >= 0.85:
            self._add_result(
                step, "fit_r2", CheckStatus.PASS,
                fit_r2, "R² >= 0.85",
                f"Fit quality R² = {fit_r2:.4f} is good"
            )
        elif fit_r2 >= 0.70:
            self._add_result(
                step, "fit_r2", CheckStatus.WARN,
                fit_r2, "R² >= 0.85",
                f"Fit quality R² = {fit_r2:.4f} is acceptable"
            )
        elif fit_r2 >= 0.50:
            self._add_result(
                step, "fit_r2", CheckStatus.FAIL,
                fit_r2, "R² >= 0.70",
                f"Fit quality R² = {fit_r2:.4f} is poor"
            )
        else:
            self._add_result(
                step, "fit_r2", CheckStatus.CRITICAL,
                fit_r2, "R² >= 0.50",
                f"Fit quality R² = {fit_r2:.4f} is unacceptable"
            )

        # Max residual
        if max_residual < 0.05:
            self._add_result(
                step, "max_residual", CheckStatus.PASS,
                max_residual, "< 5%",
                f"Max residual {max_residual:.2%} is small"
            )
        elif max_residual < 0.10:
            self._add_result(
                step, "max_residual", CheckStatus.WARN,
                max_residual, "< 5%",
                f"Max residual {max_residual:.2%} - some outliers may affect fit"
            )
        else:
            self._add_result(
                step, "max_residual", CheckStatus.FAIL,
                max_residual, "< 10%",
                f"Max residual {max_residual:.2%} - significant outliers present"
            )

        return self.all_results[step]

    def check_iv_consistency(
        self,
        model_iv: np.ndarray,
        market_iv: np.ndarray,
        threshold: float,
        expiry: str
    ) -> CheckSummary:
        """Check if model-implied IVs are consistent with market IVs.

        Args:
            model_iv: Array of model-implied volatilities.
            market_iv: Array of market implied volatilities.
            threshold: Maximum allowed relative IV error.
            expiry: Expiry string.

        Returns:
            CheckSummary for IV consistency check.
        """
        step = f"iv_consistency_{expiry}"

        relative_errors = np.abs(model_iv - market_iv) / market_iv
        max_error = np.max(relative_errors)
        mean_error = np.mean(relative_errors)

        if max_error <= threshold:
            self._add_result(
                step, "iv_max_error", CheckStatus.PASS,
                max_error, f"<= {threshold:.0%}",
                f"Max IV error {max_error:.2%} is within threshold"
            )
        elif max_error <= threshold * 1.5:
            self._add_result(
                step, "iv_max_error", CheckStatus.WARN,
                max_error, f"<= {threshold:.0%}",
                f"Max IV error {max_error:.2%} slightly exceeds threshold"
            )
        else:
            self._add_result(
                step, "iv_max_error", CheckStatus.FAIL,
                max_error, f"<= {threshold:.0%}",
                f"Max IV error {max_error:.2%} exceeds threshold"
            )

        if mean_error <= threshold / 2:
            self._add_result(
                step, "iv_mean_error", CheckStatus.PASS,
                mean_error, f"<= {threshold/2:.0%}",
                f"Mean IV error {mean_error:.2%} is good"
            )
        else:
            self._add_result(
                step, "iv_mean_error", CheckStatus.WARN,
                mean_error, f"<= {threshold/2:.0%}",
                f"Mean IV error {mean_error:.2%} is elevated"
            )

        return self.all_results[step]

    def check_rnd(
        self,
        rnd: RNDResult,
        expiry: str
    ) -> CheckSummary:
        """Check RND extraction quality.

        Args:
            rnd: RND result.
            expiry: Expiry string.

        Returns:
            CheckSummary for RND checks.
        """
        step = f"rnd_{expiry}"

        # Integral using config tolerance
        integral_tol = self.config.integral_tolerance
        if 1 - integral_tol/5 <= rnd.integral <= 1 + integral_tol/5:
            self._add_result(
                step, "integral", CheckStatus.PASS,
                rnd.integral, f"{1 - integral_tol/5:.2f} - {1 + integral_tol/5:.2f}",
                f"Density integrates to {rnd.integral:.4f}"
            )
        elif 1 - integral_tol <= rnd.integral <= 1 + integral_tol:
            self._add_result(
                step, "integral", CheckStatus.WARN,
                rnd.integral, f"{1 - integral_tol/5:.2f} - {1 + integral_tol/5:.2f}",
                f"Density integral {rnd.integral:.4f} slightly off"
            )
        else:
            self._add_result(
                step, "integral", CheckStatus.FAIL,
                rnd.integral, f"{1 - integral_tol:.2f} - {1 + integral_tol:.2f}",
                f"Density integral {rnd.integral:.4f} is problematic"
            )

        # Mean vs Forward using config tolerance
        mean_tol = self.config.mean_forward_tolerance
        mean_diff_pct = abs(rnd.mean - rnd.forward) / rnd.forward
        if mean_diff_pct <= mean_tol / 2:
            self._add_result(
                step, "mean_forward", CheckStatus.PASS,
                mean_diff_pct, f"<= {mean_tol/2:.0%}",
                f"Mean ({rnd.mean:.0f}) within {mean_tol/2:.0%} of forward ({rnd.forward:.0f})"
            )
        elif mean_diff_pct <= mean_tol:
            self._add_result(
                step, "mean_forward", CheckStatus.WARN,
                mean_diff_pct, f"<= {mean_tol/2:.0%}",
                f"Mean ({rnd.mean:.0f}) differs from forward ({rnd.forward:.0f}) by {mean_diff_pct:.1%}"
            )
        else:
            self._add_result(
                step, "mean_forward", CheckStatus.CRITICAL,
                mean_diff_pct, f"<= {mean_tol:.0%}",
                f"Mean ({rnd.mean:.0f}) differs from forward ({rnd.forward:.0f}) by {mean_diff_pct:.1%}"
            )

        # Mode range
        mode_ratio = rnd.mode / rnd.forward
        if 0.5 <= mode_ratio <= 2.0:
            self._add_result(
                step, "mode_range", CheckStatus.PASS,
                mode_ratio, "0.5F - 2F",
                f"Mode ${rnd.mode:,.0f} is reasonable"
            )
        else:
            self._add_result(
                step, "mode_range", CheckStatus.WARN,
                mode_ratio, "0.5F - 2F",
                f"Mode ${rnd.mode:,.0f} is far from forward - possible numerical issue"
            )

        # Std dev
        std_pct = rnd.std_dev / rnd.forward
        if 0.10 <= std_pct <= 1.0:
            self._add_result(
                step, "std_dev", CheckStatus.PASS,
                std_pct, "10% - 100% of F",
                f"Std dev {std_pct:.1%} of forward is reasonable"
            )
        else:
            self._add_result(
                step, "std_dev", CheckStatus.WARN,
                std_pct, "10% - 100% of F",
                f"Std dev {std_pct:.1%} of forward is unusual"
            )

        # Skewness
        if -2 <= rnd.skewness <= 2:
            self._add_result(
                step, "skewness", CheckStatus.PASS,
                rnd.skewness, "-2 to 2",
                f"Skewness {rnd.skewness:.3f} is moderate"
            )
        else:
            self._add_result(
                step, "skewness", CheckStatus.WARN,
                rnd.skewness, "-2 to 2",
                f"Skewness {rnd.skewness:.3f} is extreme"
            )

        # Kurtosis
        if -1 <= rnd.kurtosis <= 10:
            self._add_result(
                step, "kurtosis", CheckStatus.PASS,
                rnd.kurtosis, "-1 to 10",
                f"Excess kurtosis {rnd.kurtosis:.3f}"
            )
        else:
            self._add_result(
                step, "kurtosis", CheckStatus.WARN,
                rnd.kurtosis, "-1 to 10",
                f"Excess kurtosis {rnd.kurtosis:.3f} indicates very fat tails"
            )

        # Percentile order
        pcts_ordered = (
            rnd.percentile_5 < rnd.percentile_25 <
            rnd.percentile_50 < rnd.percentile_75 < rnd.percentile_95
        )
        if pcts_ordered:
            self._add_result(
                step, "percentile_order", CheckStatus.PASS,
                None, "5 < 25 < 50 < 75 < 95",
                "Percentiles are correctly ordered"
            )
        else:
            self._add_result(
                step, "percentile_order", CheckStatus.CRITICAL,
                None, "5 < 25 < 50 < 75 < 95",
                "Percentiles are NOT correctly ordered - computational error"
            )

        return self.all_results[step]

    def get_summary(self) -> Dict[str, dict]:
        """Get summary of all checks.

        Returns:
            Dictionary summarizing all check results.
        """
        summary = {}
        for step, check_summary in self.all_results.items():
            summary[step] = {
                "overall_status": check_summary.overall_status.value,
                "pass_count": sum(1 for r in check_summary.results if r.status == CheckStatus.PASS),
                "warn_count": sum(1 for r in check_summary.results if r.status == CheckStatus.WARN),
                "fail_count": sum(1 for r in check_summary.results if r.status == CheckStatus.FAIL),
                "critical_count": sum(1 for r in check_summary.results if r.status == CheckStatus.CRITICAL),
                "details": [
                    {
                        "name": r.name,
                        "status": r.status.value,
                        "value": r.value,
                        "expected": r.expected,
                        "message": r.message
                    }
                    for r in check_summary.results
                ]
            }
        return summary

    def print_summary(self) -> None:
        """Print a formatted summary of all checks."""
        print("\n" + "=" * 60)
        print("SANITY CHECK SUMMARY")
        print("=" * 60)

        for step, check_summary in self.all_results.items():
            status_str = check_summary.overall_status.value.upper()
            print(f"\n[{status_str}] {step}")

            for result in check_summary.results:
                icon = {
                    CheckStatus.PASS: "✓",
                    CheckStatus.WARN: "⚠",
                    CheckStatus.FAIL: "✗",
                    CheckStatus.CRITICAL: "⛔"
                }.get(result.status, "?")
                print(f"  {icon} {result.name}: {result.message}")
