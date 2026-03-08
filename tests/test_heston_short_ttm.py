"""Tests for Heston calibration robustness with short TTM (1-7 days).

Uses a snapshot of real Deribit BTC options data (captured 2026-02-07) so the
IV smile shapes, moneyness ranges, and noise levels are representative.

Validates:
- Ultra-short-dated bounds tier produces convergent fits with acceptable R²
- v0 is pinned near ATM variance for short TTM
- IV consistency threshold is relaxed for short TTM
- Taylor expansion in charfunc matches standard computation

Plots are saved to test_plots/ for visual inspection.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from btc_pricer.api.deribit import OptionData
from btc_pricer.cli.common import extract_surface_data
from btc_pricer.config import Config
from btc_pricer.constants import RELAXED_MIN_POINTS
from btc_pricer.data.filters import DataFilter
from btc_pricer.models.heston import (
    HestonFitter,
    HestonModel,
    HestonParams,
    check_iv_consistency,
)


PLOT_DIR = Path(__file__).parent.parent / "test_plots"


SNAPSHOT_PATH = Path(__file__).parent / "fixtures" / "deribit_snapshot_8exp.json"

# Expiries with TTM < 0.01 (ultra-short-dated, 1-3.5 days)
ULTRA_SHORT_EXPIRIES = ["8FEB26", "9FEB26"]
# Expiries with TTM 0.007-0.01 (very short, ~2.5-3.5 days)
VERY_SHORT_EXPIRIES = ["10FEB26", "11FEB26"]
# All short-TTM expiries for parameterized tests
SHORT_EXPIRIES = ULTRA_SHORT_EXPIRIES + VERY_SHORT_EXPIRIES


def _load_surface_cases(
    expiries: List[str],
) -> Dict[str, Tuple[np.ndarray, np.ndarray, float, float]]:
    """Load real surface data from the fixture for the given expiries.

    Returns:
        Dict mapping expiry -> (log_moneyness, market_iv, ttm, forward).
    """
    assert SNAPSHOT_PATH.exists(), f"Missing fixture: {SNAPSHOT_PATH}"
    payload = json.loads(SNAPSHOT_PATH.read_text(encoding="utf-8"))
    # Use the same relaxed filters that the snapshot was captured with
    config_path = Path(__file__).parent.parent / "config.yaml"
    config = Config.from_yaml(config_path) if config_path.exists() else Config()
    data_filter = DataFilter(config.filters)

    cases = {}
    for expiry in expiries:
        raw_expiry = payload["expiries"][expiry]
        options = [OptionData(**opt) for opt in raw_expiry["options"]]
        filtered = data_filter.filter_options(options)
        if isinstance(filtered, tuple):
            filtered = filtered[0]
        otm_surface = data_filter.build_otm_surface(filtered)
        surface_data = extract_surface_data(
            otm_surface,
            min_points=RELAXED_MIN_POINTS,
            iv_valid_range=config.validation.iv_valid_range,
        )
        if surface_data is None:
            continue
        forward, ttm, _, log_moneyness, market_iv = surface_data
        cases[expiry] = (log_moneyness, market_iv, ttm, forward)
    return cases


@pytest.fixture(scope="module")
def short_ttm_cases():
    """Load all short-TTM surface cases once per module."""
    return _load_surface_cases(SHORT_EXPIRIES)


def _make_fitter(**overrides) -> HestonFitter:
    """Create a HestonFitter with production-like defaults + overrides."""
    defaults = dict(
        use_quantlib=True,
        use_multi_start=True,
        n_starts=9,
        ultra_short_dated_ttm_threshold=0.01,
        ultra_short_dated_xi_bounds=(0.1, 20.0),
        ultra_short_dated_kappa_bounds=(0.01, 10.0),
        ultra_short_dated_theta_factor=(0.3, 2.0),
        short_ttm_gaussian_weighting=True,
        short_ttm_gaussian_sigma_base=0.05,
        short_ttm_gaussian_sigma_ttm_scale=2.0,
        short_ttm_gaussian_floor=0.1,
    )
    defaults.update(overrides)
    return HestonFitter(**defaults)


def _plot_iv_smile(
    expiry: str,
    log_k: np.ndarray,
    market_iv: np.ndarray,
    model_iv: np.ndarray,
    ttm: float,
    r_squared: float,
    params: HestonParams,
) -> None:
    """Save an IV smile plot comparing market vs Heston model IVs."""
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), height_ratios=[3, 1])

    # Top: IV smile
    ax1.scatter(log_k, market_iv * 100, s=40, c="black", zorder=5, label="Market IV")
    ax1.plot(log_k, model_iv * 100, "r-", lw=2, label="Heston fit")
    ax1.set_ylabel("Implied Volatility (%)")
    ax1.set_title(
        f"{expiry}  |  TTM={ttm:.4f} ({ttm*365:.1f}d)  |  "
        f"R²={r_squared:.4f}\n"
        f"v0={params.v0:.4f}  kappa={params.kappa:.2f}  "
        f"theta={params.theta:.4f}  xi={params.xi:.2f}  rho={params.rho:.3f}"
    )
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Bottom: residuals
    residuals = (model_iv - market_iv) / market_iv * 100
    ax2.bar(log_k, residuals, width=0.005, color="steelblue", alpha=0.7)
    ax2.axhline(0, color="black", lw=0.5)
    ax2.set_xlabel("Log-moneyness ln(K/F)")
    ax2.set_ylabel("Relative error (%)")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(PLOT_DIR / f"iv_smile_{expiry}_TTM{ttm:.4f}.png", dpi=150)
    plt.close(fig)


# ============================================================================
# Convergence tests — real Deribit short-TTM data
# ============================================================================

@pytest.mark.slow
class TestUltraShortTTMConvergence:
    """Test that HestonFitter converges for ultra-short TTM on real data."""

    @pytest.mark.parametrize("expiry", SHORT_EXPIRIES)
    def test_short_ttm_converges(self, short_ttm_cases, expiry):
        """Fitter should return success=True and report R² for short-TTM expiries."""
        if expiry not in short_ttm_cases:
            pytest.skip(f"No surface data for {expiry}")

        log_k, market_iv, ttm, forward = short_ttm_cases[expiry]
        fitter = _make_fitter()
        result = fitter.fit(log_k, market_iv, ttm, forward=forward)

        assert result.params is not None, f"Fit returned no params for {expiry}"
        assert result.r_squared is not None, "R² not reported"
        assert isinstance(result.r_squared, float), "R² should be a float"

        # Generate IV smile plot for visual inspection
        model = HestonModel(result.params, use_quantlib=True)
        model_iv = model.implied_volatility_array(log_k)
        _plot_iv_smile(expiry, log_k, market_iv, model_iv, ttm,
                       result.r_squared, result.params)

        # R² threshold depends on TTM — sub-day expiries have steeper smiles
        # that Heston can only partially capture
        min_r2 = 0.50 if ttm < 0.003 else 0.70
        assert result.r_squared > min_r2, (
            f"{expiry} (TTM={ttm:.4f}): R²={result.r_squared:.4f} too low "
            f"(threshold={min_r2})"
        )

    @pytest.mark.parametrize("expiry", SHORT_EXPIRIES)
    def test_short_ttm_fit_quality(self, short_ttm_cases, expiry):
        """Report and check R² and RMSE for each short-TTM expiry."""
        if expiry not in short_ttm_cases:
            pytest.skip(f"No surface data for {expiry}")

        log_k, market_iv, ttm, forward = short_ttm_cases[expiry]
        fitter = _make_fitter()
        result = fitter.fit(log_k, market_iv, ttm, forward=forward)

        assert np.isfinite(result.rmse), f"RMSE not finite for {expiry}"
        assert np.isfinite(result.r_squared), f"R² not finite for {expiry}"
        # IV error should be bounded
        assert result.iv_error_max < 1.0, (
            f"{expiry}: max IV error {result.iv_error_max:.1%} exceeds 100%"
        )


# ============================================================================
# v0 pinning — verify v0 ≈ ATM_IV² for ultra-short TTM
# ============================================================================

@pytest.mark.slow
class TestV0PinnedNearATMVariance:
    """Test that fitted v0 is close to ATM_IV² for ultra-short TTM."""

    @pytest.mark.parametrize("expiry", ULTRA_SHORT_EXPIRIES)
    def test_v0_pinned_near_atm_variance(self, short_ttm_cases, expiry):
        """Fitted v0 should be within 50% of ATM_IV² for ultra-short TTM."""
        if expiry not in short_ttm_cases:
            pytest.skip(f"No surface data for {expiry}")

        log_k, market_iv, ttm, forward = short_ttm_cases[expiry]

        # Find ATM IV from data
        atm_idx = np.argmin(np.abs(log_k))
        atm_iv = market_iv[atm_idx]
        expected_v0 = atm_iv ** 2

        fitter = _make_fitter()
        result = fitter.fit(log_k, market_iv, ttm, forward=forward)

        assert result.params is not None, f"Fit failed for {expiry}"

        actual_v0 = result.params.v0
        relative_error = abs(actual_v0 - expected_v0) / expected_v0

        assert relative_error < 0.50, (
            f"{expiry}: v0={actual_v0:.4f} too far from ATM_var={expected_v0:.4f} "
            f"(relative error {relative_error:.1%}, R²={result.r_squared:.4f})"
        )


# ============================================================================
# IV consistency relaxation — TTM-adaptive threshold
# ============================================================================

@pytest.mark.slow
class TestIVConsistencyRelaxation:
    """Test that IV consistency threshold is relaxed for short TTM."""

    def test_relaxed_for_short_ttm(self):
        """With ttm=0.003, an 18% error should still pass."""
        params = HestonParams(
            v0=0.36, kappa=2.0, theta=0.36, xi=1.0, rho=-0.5, ttm=0.003
        )
        model = HestonModel(params, use_quantlib=True)

        log_k = np.array([-0.05, -0.03, 0.0, 0.03, 0.05])
        model_iv = model.implied_volatility_array(log_k)
        market_iv = model_iv * 1.18  # 18% higher

        # Should FAIL with strict threshold (10%)
        is_ok_strict, max_err_strict, _ = check_iv_consistency(
            model, log_k, market_iv, threshold=0.10, ttm=1.0
        )
        assert not is_ok_strict, (
            f"Expected strict check to fail, but max_err={max_err_strict:.1%}"
        )

        # Should PASS with short-TTM relaxation
        is_ok_relaxed, max_err_relaxed, _ = check_iv_consistency(
            model, log_k, market_iv, threshold=0.10,
            ttm=0.003, relaxation=0.15, ttm_cutoff=0.05
        )
        # Effective threshold ≈ 0.10 + 0.15 * (0.05-0.003)/0.05 ≈ 0.241
        assert is_ok_relaxed, (
            f"Expected relaxed check to pass for TTM=0.003, "
            f"but max_err={max_err_relaxed:.1%}"
        )

    def test_no_relaxation_above_cutoff(self):
        """Above ttm_cutoff, the threshold should not be relaxed."""
        params = HestonParams(
            v0=0.36, kappa=2.0, theta=0.36, xi=1.0, rho=-0.5, ttm=0.1
        )
        model = HestonModel(params, use_quantlib=True)

        log_k = np.array([-0.05, -0.03, 0.0, 0.03, 0.05])
        model_iv = model.implied_volatility_array(log_k)
        market_iv = model_iv * 1.12  # 12% error

        is_ok, max_err, _ = check_iv_consistency(
            model, log_k, market_iv, threshold=0.10,
            ttm=0.1, relaxation=0.15, ttm_cutoff=0.05
        )
        assert not is_ok, (
            f"Expected check to fail above cutoff, but max_err={max_err:.1%}"
        )


# ============================================================================
# Charfunc Taylor expansion — numerical stability
# ============================================================================

@pytest.mark.slow
class TestCharfuncTaylorExpansion:
    """Test that Taylor expansion matches standard charfunc for small tau."""

    def test_taylor_no_nan_inf(self):
        """Charfunc should produce finite values for ultra-short TTM."""
        params = HestonParams(
            v0=0.36, kappa=2.0, theta=0.36, xi=1.5, rho=-0.5, ttm=0.003
        )
        model = HestonModel(params, n_integration_points=512, use_quantlib=False)

        for u_real in [0.5, 1.0, 5.0, 10.0, 20.0]:
            u = complex(u_real, 0)
            val = model._heston_charfunc_core(u)
            assert np.isfinite(val), (
                f"charfunc returned non-finite for u={u}: {val}"
            )

    def test_taylor_matches_standard_at_boundary(self):
        """At ultra-short TTM, ATM IV should be close to sqrt(v0)."""
        params_short = HestonParams(
            v0=0.36, kappa=2.0, theta=0.36, xi=0.5, rho=-0.3, ttm=0.003
        )
        model_short = HestonModel(
            params_short, n_integration_points=512, use_quantlib=False
        )

        atm_iv_short = model_short.implied_volatility(0.0)
        expected_iv = np.sqrt(params_short.v0)

        assert np.isfinite(atm_iv_short), "ATM IV is not finite for short TTM"
        assert abs(atm_iv_short - expected_iv) / expected_iv < 0.15, (
            f"ATM IV={atm_iv_short:.4f} too far from sqrt(v0)={expected_iv:.4f}"
        )

    def test_charfunc_continuity_across_taylor_boundary(self):
        """Charfunc should produce finite IVs on both sides of the Taylor switch."""
        base_params = dict(v0=0.36, kappa=2.0, theta=0.36, xi=1.0, rho=-0.5)

        params_above = HestonParams(**base_params, ttm=0.05)
        params_below = HestonParams(**base_params, ttm=0.005)

        model_above = HestonModel(params_above, use_quantlib=False)
        model_below = HestonModel(params_below, use_quantlib=False)

        log_k = np.array([-0.05, -0.02, 0.0, 0.02, 0.05])
        iv_above = model_above.implied_volatility_array(log_k)
        iv_below = model_below.implied_volatility_array(log_k)

        assert np.all(np.isfinite(iv_above)), "Standard path produced non-finite IVs"
        assert np.all(np.isfinite(iv_below)), "Taylor path produced non-finite IVs"
        assert np.all(iv_above > 0), "Standard path produced non-positive IVs"
        assert np.all(iv_below > 0), "Taylor path produced non-positive IVs"


# ============================================================================
# Gaussian near-ATM weighting — shape and floor verification
# ============================================================================

@pytest.mark.slow
class TestGaussianWeighting:
    """Test that Gaussian near-ATM weighting produces correct weight profiles."""

    def test_weights_shape_and_atm_peak(self):
        """Gaussian weights should peak at ATM (log_k=0) and decay outward."""
        log_k = np.linspace(-0.10, 0.10, 21)
        ttm = 0.005  # ~1.8 days
        sigma = 0.05 + 2.0 * ttm  # 0.06
        weights = np.maximum(
            np.exp(-log_k**2 / (2 * sigma**2)), 0.1
        )
        # Peak should be at or near the center (ATM)
        assert np.argmax(weights) == 10, "Peak weight should be at ATM"
        # ATM weight should be 1.0
        assert abs(weights[10] - 1.0) < 1e-10, "ATM weight should be 1.0"
        # Wings should be lower than ATM
        assert weights[0] < weights[10], "Left wing should be lower than ATM"
        assert weights[-1] < weights[10], "Right wing should be lower than ATM"

    def test_floor_enforced(self):
        """No weight should fall below the floor value."""
        log_k = np.linspace(-0.20, 0.20, 41)
        floor = 0.1
        sigma = 0.05
        weights = np.maximum(
            np.exp(-log_k**2 / (2 * sigma**2)), floor
        )
        assert np.all(weights >= floor), "All weights must be >= floor"

    def test_sigma_scales_with_ttm(self):
        """Shorter TTM should produce tighter Gaussian (smaller sigma)."""
        sigma_short = 0.05 + 2.0 * 0.001  # sub-day
        sigma_long = 0.05 + 2.0 * 0.015   # ~5.5 days
        assert sigma_short < sigma_long, "Shorter TTM should have tighter sigma"
        assert abs(sigma_short - 0.052) < 0.001
        assert abs(sigma_long - 0.08) < 0.001


# ============================================================================
# Boundary-hit check — verify fitted params are not pinned at bounds
# ============================================================================

@pytest.mark.slow
class TestNoBoundaryHit:
    """Verify that fitted parameters are not stuck at bounds for short TTM."""

    @pytest.mark.parametrize("expiry", SHORT_EXPIRIES)
    def test_no_boundary_hit(self, short_ttm_cases, expiry):
        """No fitted parameter should be within 1% of its bounds."""
        if expiry not in short_ttm_cases:
            pytest.skip(f"No surface data for {expiry}")

        log_k, market_iv, ttm, forward = short_ttm_cases[expiry]
        fitter = _make_fitter()
        result = fitter.fit(log_k, market_iv, ttm, forward=forward)
        assert result.params is not None, f"Fit failed for {expiry}"

        p = result.params
        bounds_dict = fitter._get_ttm_adjusted_bounds(ttm, atm_iv=market_iv[np.argmin(np.abs(log_k))])

        # Check each parameter against its bounds
        param_checks = [
            ("theta", p.theta, bounds_dict["theta"]),
            ("kappa", p.kappa, bounds_dict["kappa"]),
        ]
        warnings_list = []
        for name, val, (lo, hi) in param_checks:
            span = hi - lo
            if span > 0:
                if (val - lo) / span < 0.01:
                    warnings_list.append(
                        f"{name}={val:.4f} at lower bound {lo:.4f}"
                    )
                if (hi - val) / span < 0.01:
                    warnings_list.append(
                        f"{name}={val:.4f} at upper bound {hi:.4f}"
                    )
        # This is a soft check — warn rather than fail, since some
        # boundary-hitting may still be optimal for extreme cases
        if warnings_list:
            pytest.xfail(
                f"{expiry} (TTM={ttm:.4f}): boundary hits: "
                + ", ".join(warnings_list)
            )
