"""Regression tests for performance optimization phases.

Compares pipeline outputs against golden reference values generated before
any optimizations. Ensures that "safe" optimizations (vol surface caching,
vectorized BS, compact MC) produce bit-identical results, and that
"approximate" optimizations (early termination) stay within relaxed tolerances.

Run: pytest tests/test_optimization_regression.py -v
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from btc_pricer.api.deribit import OptionData
from btc_pricer.cli.common import extract_surface_data
from btc_pricer.config import Config
from btc_pricer.constants import RELAXED_MIN_POINTS
from btc_pricer.data.filters import DataFilter
from btc_pricer.models.breeden_litzenberger import BreedenLitzenberger
from btc_pricer.models.heston import HestonFitter, HestonModel


SNAPSHOT_PATH = Path(__file__).parent / "fixtures" / "deribit_snapshot_8exp.json"
GOLDEN_PATH = Path(__file__).parent / "fixtures" / "golden_reference.json"


def _build_fitter(config: Config) -> HestonFitter:
    """Build HestonFitter matching production configuration."""
    return HestonFitter(
        v0_bounds=config.heston.v0_bounds,
        kappa_bounds=config.heston.kappa_bounds,
        theta_bounds=config.heston.theta_bounds,
        xi_bounds=config.heston.xi_bounds,
        rho_bounds=config.heston.rho_bounds,
        optimizer=config.heston.optimizer,
        n_integration_points=config.heston.n_integration_points,
        use_quantlib=config.heston.use_quantlib,
        short_dated_ttm_threshold=config.heston.short_dated_ttm_threshold,
        short_dated_xi_bounds=config.heston.short_dated_xi_bounds,
        short_dated_kappa_bounds=config.heston.short_dated_kappa_bounds,
        very_short_dated_ttm_threshold=config.heston.very_short_dated_ttm_threshold,
        very_short_dated_xi_bounds=config.heston.very_short_dated_xi_bounds,
        very_short_dated_kappa_bounds=config.heston.very_short_dated_kappa_bounds,
        use_multi_start=config.heston.use_multi_start,
        n_starts=config.heston.n_starts,
        quantlib_objective_impl=config.heston.quantlib_objective_impl,
        enable_numba_fallback=config.heston.enable_numba_fallback,
        numba_strict_mode=config.heston.numba_strict_mode,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def golden_reference():
    """Load golden reference values generated before optimizations."""
    assert GOLDEN_PATH.exists(), (
        f"Golden reference not found: {GOLDEN_PATH}\n"
        "Run: python scripts/generate_golden_reference.py"
    )
    return json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def pipeline_results():
    """Run full pipeline on all 8 expiries, cache results for the module."""
    assert SNAPSHOT_PATH.exists(), f"Missing fixture: {SNAPSHOT_PATH}"

    payload = json.loads(SNAPSHOT_PATH.read_text(encoding="utf-8"))
    config_path = Path("config.yaml")
    config = Config.from_yaml(config_path) if config_path.exists() else Config()

    data_filter = DataFilter(config.filters)
    fitter = _build_fitter(config)
    bl = BreedenLitzenberger(
        strike_grid_points=config.breeden_litzenberger.strike_grid_points,
        strike_range_std=config.breeden_litzenberger.strike_range_std,
        use_log_strikes=config.breeden_litzenberger.use_log_strikes,
    )

    results = {}

    for expiry in payload["selected_expiries"]:
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

        forward, ttm, spot, log_moneyness, market_iv = surface_data

        # Calibrate
        fit_result = fitter.fit(log_moneyness, market_iv, ttm, forward=forward)
        if not fit_result.success or fit_result.params is None:
            continue

        params = fit_result.params

        # BL extraction
        rnd = bl.extract_from_heston(params, forward, use_quantlib=config.heston.use_quantlib)
        p_above = bl.probability_above(rnd, forward)
        p_below = bl.probability_below(rnd, forward)

        results[expiry] = {
            "v0": float(params.v0),
            "kappa": float(params.kappa),
            "theta": float(params.theta),
            "xi": float(params.xi),
            "rho": float(params.rho),
            "r_squared": float(fit_result.r_squared),
            "rmse": float(fit_result.rmse),
            "p_above": float(p_above),
            "p_below": float(p_below),
            "bl_mean": float(rnd.mean),
            "bl_std": float(rnd.std_dev),
            "bl_integral": float(rnd.integral),
        }

    return results


# ---------------------------------------------------------------------------
# Strict tolerance tests (Phases 1, 2, 4, 5 — safe optimizations)
# ---------------------------------------------------------------------------

@pytest.mark.slow
@pytest.mark.analytics
def test_heston_params_match_golden_reference(golden_reference, pipeline_results):
    """Heston parameters must be bit-identical (same DE seed/path)."""
    for expiry, ref in golden_reference.items():
        assert expiry in pipeline_results, f"{expiry} missing from pipeline results"
        res = pipeline_results[expiry]
        ref_params = ref["heston_params"]

        assert abs(res["v0"] - ref_params["v0"]) < 1e-10, (
            f"{expiry} v0 mismatch: {res['v0']} vs {ref_params['v0']}"
        )
        assert abs(res["kappa"] - ref_params["kappa"]) < 1e-10, (
            f"{expiry} kappa mismatch: {res['kappa']} vs {ref_params['kappa']}"
        )
        assert abs(res["theta"] - ref_params["theta"]) < 1e-10, (
            f"{expiry} theta mismatch: {res['theta']} vs {ref_params['theta']}"
        )
        assert abs(res["xi"] - ref_params["xi"]) < 1e-10, (
            f"{expiry} xi mismatch: {res['xi']} vs {ref_params['xi']}"
        )
        assert abs(res["rho"] - ref_params["rho"]) < 1e-10, (
            f"{expiry} rho mismatch: {res['rho']} vs {ref_params['rho']}"
        )


@pytest.mark.slow
@pytest.mark.analytics
def test_r_squared_match_golden_reference(golden_reference, pipeline_results):
    """R-squared must match to machine precision."""
    for expiry, ref in golden_reference.items():
        assert expiry in pipeline_results, f"{expiry} missing from pipeline results"
        res = pipeline_results[expiry]

        assert abs(res["r_squared"] - ref["r_squared"]) < 1e-12, (
            f"{expiry} R² mismatch: {res['r_squared']} vs {ref['r_squared']}"
        )


@pytest.mark.slow
@pytest.mark.analytics
def test_bl_probabilities_match_golden_reference(golden_reference, pipeline_results):
    """BL probabilities must match to high precision."""
    for expiry, ref in golden_reference.items():
        assert expiry in pipeline_results, f"{expiry} missing from pipeline results"
        res = pipeline_results[expiry]

        assert abs(res["p_above"] - ref["bl_probability_above_forward"]) < 1e-10, (
            f"{expiry} P(above) mismatch: {res['p_above']} vs {ref['bl_probability_above_forward']}"
        )
        assert abs(res["p_below"] - ref["bl_probability_below_forward"]) < 1e-10, (
            f"{expiry} P(below) mismatch: {res['p_below']} vs {ref['bl_probability_below_forward']}"
        )


@pytest.mark.slow
@pytest.mark.analytics
def test_bl_statistics_match_golden_reference(golden_reference, pipeline_results):
    """BL mean, std, and integral must match."""
    for expiry, ref in golden_reference.items():
        assert expiry in pipeline_results, f"{expiry} missing from pipeline results"
        res = pipeline_results[expiry]

        assert abs(res["bl_mean"] - ref["bl_mean"]) < 1e-6, (
            f"{expiry} BL mean mismatch: {res['bl_mean']} vs {ref['bl_mean']}"
        )
        assert abs(res["bl_std"] - ref["bl_std"]) < 1e-6, (
            f"{expiry} BL std mismatch: {res['bl_std']} vs {ref['bl_std']}"
        )
        assert abs(res["bl_integral"] - ref["bl_integral"]) < 1e-10, (
            f"{expiry} BL integral mismatch: {res['bl_integral']} vs {ref['bl_integral']}"
        )


@pytest.mark.slow
@pytest.mark.analytics
def test_all_golden_expiries_present(golden_reference, pipeline_results):
    """All golden reference expiries must be present in pipeline results."""
    for expiry in golden_reference:
        assert expiry in pipeline_results, (
            f"Golden expiry {expiry} missing from pipeline results"
        )


# ---------------------------------------------------------------------------
# Relaxed tolerance tests (Phase 3 — early termination)
# ---------------------------------------------------------------------------

@pytest.mark.slow
@pytest.mark.analytics
def test_bl_probabilities_relaxed_tolerance(golden_reference, pipeline_results):
    """Probabilities within 1e-3 (for early-termination configurations)."""
    for expiry, ref in golden_reference.items():
        assert expiry in pipeline_results, f"{expiry} missing from pipeline results"
        res = pipeline_results[expiry]

        assert abs(res["p_above"] - ref["bl_probability_above_forward"]) < 1e-3, (
            f"{expiry} P(above) relaxed mismatch: "
            f"{res['p_above']} vs {ref['bl_probability_above_forward']}"
        )
        assert abs(res["p_below"] - ref["bl_probability_below_forward"]) < 1e-3, (
            f"{expiry} P(below) relaxed mismatch: "
            f"{res['p_below']} vs {ref['bl_probability_below_forward']}"
        )
