"""Strict equivalence tests for Heston calibration on real Deribit snapshots."""

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
from btc_pricer.models.heston import HestonFitter, NUMBA_AVAILABLE


SNAPSHOT_PATH = Path(__file__).parent / "fixtures" / "deribit_snapshot_8exp.json"


def _build_fitter(
    config: Config,
    impl: str,
    enable_numba_fallback: bool,
    use_quantlib: bool = True
) -> HestonFitter:
    return HestonFitter(
        v0_bounds=config.heston.v0_bounds,
        kappa_bounds=config.heston.kappa_bounds,
        theta_bounds=config.heston.theta_bounds,
        xi_bounds=config.heston.xi_bounds,
        rho_bounds=config.heston.rho_bounds,
        optimizer=config.heston.optimizer,
        n_integration_points=config.heston.n_integration_points,
        use_quantlib=use_quantlib,
        short_dated_ttm_threshold=config.heston.short_dated_ttm_threshold,
        short_dated_xi_bounds=config.heston.short_dated_xi_bounds,
        short_dated_kappa_bounds=config.heston.short_dated_kappa_bounds,
        very_short_dated_ttm_threshold=config.heston.very_short_dated_ttm_threshold,
        very_short_dated_xi_bounds=config.heston.very_short_dated_xi_bounds,
        very_short_dated_kappa_bounds=config.heston.very_short_dated_kappa_bounds,
        use_multi_start=config.heston.use_multi_start,
        n_starts=config.heston.n_starts,
        quantlib_objective_impl=impl,
        enable_numba_fallback=enable_numba_fallback,
        numba_strict_mode=True,
    )


def _atm_penalty(ttm: float, fitter: HestonFitter) -> float:
    if ttm < fitter.very_short_dated_ttm_threshold:
        return 50.0
    if ttm < fitter.short_dated_ttm_threshold:
        return 10.0
    return 0.0


@pytest.fixture(scope="module")
def deribit_cases():
    assert SNAPSHOT_PATH.exists(), f"Missing fixture: {SNAPSHOT_PATH}"
    payload = json.loads(SNAPSHOT_PATH.read_text(encoding="utf-8"))

    config_path = Path("config.yaml")
    config = Config.from_yaml(config_path) if config_path.exists() else Config()
    data_filter = DataFilter(config.filters)

    cases = []
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
            iv_valid_range=config.validation.iv_valid_range
        )
        if surface_data is None:
            continue

        forward, ttm, _, log_moneyness, market_iv = surface_data

        # Keep deterministic subset for runtime while preserving real-market shape.
        if len(log_moneyness) > 21:
            idx = np.unique(
                np.linspace(0, len(log_moneyness) - 1, num=21, dtype=int)
            )
            log_moneyness = log_moneyness[idx]
            market_iv = market_iv[idx]

        cases.append({
            "expiry": expiry,
            "forward": float(forward),
            "ttm": float(ttm),
            "log_moneyness": log_moneyness,
            "market_iv": market_iv,
        })

    return payload, config, cases


@pytest.mark.slow
@pytest.mark.analytics
def test_snapshot_has_eight_real_deribit_expiries(deribit_cases):
    payload, _config, cases = deribit_cases
    assert payload["source"] == "deribit"
    assert payload["currency"] == "BTC"
    assert len(payload["selected_expiries"]) >= 8
    assert len(cases) >= 8


@pytest.mark.slow
@pytest.mark.analytics
def test_quantlib_objective_value_parity_on_real_deribit(deribit_cases):
    _payload, config, cases = deribit_cases

    legacy = _build_fitter(config, impl="legacy", enable_numba_fallback=True, use_quantlib=True)
    optimized = _build_fitter(config, impl="optimized", enable_numba_fallback=True, use_quantlib=True)

    for case in cases[:8]:
        k = case["log_moneyness"]
        iv = case["market_iv"]
        ttm = case["ttm"]
        forward = case["forward"]

        weights = np.ones(len(k), dtype=np.float64)
        weights = weights / np.sum(weights) * len(weights)
        penalty = _atm_penalty(ttm, legacy)

        legacy_obj = legacy._create_quantlib_objective_legacy(
            k, iv, ttm, forward, weights, atm_penalty_weight=penalty
        )
        opt_obj = optimized._create_quantlib_objective_optimized(
            k, iv, ttm, forward, weights, atm_penalty_weight=penalty
        )
        assert legacy_obj is not None
        assert opt_obj is not None

        init = legacy._initialize_from_bs(k, iv, ttm)
        bounds = legacy._get_ttm_adjusted_bounds(ttm, atm_iv=float(iv[np.argmin(np.abs(k))]))

        x_init = np.array(
            [init["v0"], init["kappa"], init["theta"], init["xi"], init["rho"]],
            dtype=np.float64
        )
        x_mid = np.array([
            0.5 * (bounds["v0"][0] + bounds["v0"][1]),
            0.5 * (bounds["kappa"][0] + bounds["kappa"][1]),
            0.5 * (bounds["theta"][0] + bounds["theta"][1]),
            0.5 * (bounds["xi"][0] + bounds["xi"][1]),
            0.5 * (bounds["rho"][0] + bounds["rho"][1]),
        ], dtype=np.float64)

        for x in (x_init, x_mid):
            legacy_value = float(legacy_obj(x))
            optimized_value = float(opt_obj(x))
            assert abs(legacy_value - optimized_value) <= 1e-12, (
                f"{case['expiry']} objective mismatch: "
                f"legacy={legacy_value}, optimized={optimized_value}"
            )


@pytest.mark.slow
@pytest.mark.analytics
def test_fit_and_probability_parity_on_real_deribit(deribit_cases):
    _payload, config, cases = deribit_cases

    legacy = _build_fitter(config, impl="legacy", enable_numba_fallback=True, use_quantlib=True)
    optimized = _build_fitter(config, impl="optimized", enable_numba_fallback=True, use_quantlib=True)
    bl = BreedenLitzenberger(strike_grid_points=120)

    for case in cases[:8]:
        k = case["log_moneyness"]
        iv = case["market_iv"]
        ttm = case["ttm"]
        forward = case["forward"]

        legacy_fit = legacy.fit(k, iv, ttm, forward=forward)
        optimized_fit = optimized.fit(k, iv, ttm, forward=forward)

        assert legacy_fit.success == optimized_fit.success, f"{case['expiry']} success mismatch"
        assert legacy_fit.params is not None
        assert optimized_fit.params is not None

        assert abs(legacy_fit.params.v0 - optimized_fit.params.v0) <= 1e-10
        assert abs(legacy_fit.params.kappa - optimized_fit.params.kappa) <= 1e-10
        assert abs(legacy_fit.params.theta - optimized_fit.params.theta) <= 1e-10
        assert abs(legacy_fit.params.xi - optimized_fit.params.xi) <= 1e-10
        assert abs(legacy_fit.params.rho - optimized_fit.params.rho) <= 1e-10
        assert abs(legacy_fit.r_squared - optimized_fit.r_squared) <= 1e-12

        rnd_legacy = bl.extract_from_heston(legacy_fit.params, forward, use_quantlib=True)
        rnd_optimized = bl.extract_from_heston(optimized_fit.params, forward, use_quantlib=True)

        p_above_legacy = bl.probability_above(rnd_legacy, forward)
        p_below_legacy = bl.probability_below(rnd_legacy, forward)
        p_above_optimized = bl.probability_above(rnd_optimized, forward)
        p_below_optimized = bl.probability_below(rnd_optimized, forward)

        assert abs(p_above_legacy - p_above_optimized) <= 1e-12
        assert abs(p_below_legacy - p_below_optimized) <= 1e-12


@pytest.mark.analytics
def test_numba_weighted_sse_parity_on_real_deribit_vectors(deribit_cases):
    if not NUMBA_AVAILABLE:
        pytest.skip("Numba is not installed")

    _payload, config, cases = deribit_cases
    fitter = _build_fitter(config, impl="optimized", enable_numba_fallback=True, use_quantlib=True)

    for case in cases[:8]:
        market_iv = case["market_iv"].astype(np.float64)
        model_iv = market_iv + np.linspace(-1e-5, 1e-5, num=len(market_iv), dtype=np.float64)
        weights = np.linspace(0.8, 1.2, num=len(market_iv), dtype=np.float64)
        weights = weights / np.sum(weights) * len(weights)

        sse_python = fitter._compute_weighted_sse(model_iv, market_iv, weights, allow_numba=False)
        sse_numba = fitter._compute_weighted_sse(model_iv, market_iv, weights, allow_numba=True)
        assert abs(sse_python - sse_numba) <= 1e-18
