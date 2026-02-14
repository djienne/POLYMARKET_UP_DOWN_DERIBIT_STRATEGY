"""Generate golden reference values for optimization regression testing.

Runs the full production pipeline on the 8-expiry Deribit fixture and saves
exact numerical values to a JSON file for regression testing. Uses the same
code paths as production: HestonFitter.fit() -> BreedenLitzenberger.extract_from_heston()
-> probability_above/below.

Usage:
    python scripts/generate_golden_reference.py
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

# Ensure project root is on sys.path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from btc_pricer.api.deribit import OptionData
from btc_pricer.cli.common import extract_surface_data
from btc_pricer.config import Config
from btc_pricer.constants import RELAXED_MIN_POINTS
from btc_pricer.data.filters import DataFilter
from btc_pricer.models.breeden_litzenberger import BreedenLitzenberger
from btc_pricer.models.heston import HestonFitter, HestonModel


SNAPSHOT_PATH = project_root / "tests" / "fixtures" / "deribit_snapshot_8exp.json"
OUTPUT_PATH = project_root / "tests" / "fixtures" / "golden_reference.json"


def build_fitter(config: Config) -> HestonFitter:
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


def main():
    assert SNAPSHOT_PATH.exists(), f"Missing fixture: {SNAPSHOT_PATH}"

    payload = json.loads(SNAPSHOT_PATH.read_text(encoding="utf-8"))
    config_path = project_root / "config.yaml"
    config = Config.from_yaml(config_path) if config_path.exists() else Config()

    data_filter = DataFilter(config.filters)
    fitter = build_fitter(config)
    bl = BreedenLitzenberger(
        strike_grid_points=config.breeden_litzenberger.strike_grid_points,
        strike_range_std=config.breeden_litzenberger.strike_range_std,
        use_log_strikes=config.breeden_litzenberger.use_log_strikes,
    )

    golden = {}
    total_calibration_time = 0.0
    total_bl_time = 0.0

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
            print(f"  {expiry}: skipped (insufficient data)")
            continue

        forward, ttm, spot, log_moneyness, market_iv = surface_data
        print(f"  {expiry}: TTM={ttm:.6f}, {len(log_moneyness)} points, forward={forward:.2f}")

        # Calibrate Heston
        t0 = time.perf_counter()
        fit_result = fitter.fit(log_moneyness, market_iv, ttm, forward=forward)
        t_calibration = time.perf_counter() - t0
        total_calibration_time += t_calibration

        if not fit_result.success or fit_result.params is None:
            print(f"    FAILED: {fit_result.message}")
            continue

        params = fit_result.params

        # Extract RND and compute probabilities
        t0 = time.perf_counter()
        rnd = bl.extract_from_heston(params, forward, use_quantlib=config.heston.use_quantlib)
        p_above = bl.probability_above(rnd, forward)
        p_below = bl.probability_below(rnd, forward)
        t_bl = time.perf_counter() - t0
        total_bl_time += t_bl

        # Compute SSE for reference
        model = HestonModel(params, use_quantlib=config.heston.use_quantlib)
        model_iv = np.array([model.implied_volatility(k) for k in log_moneyness])
        residuals = model_iv - market_iv
        sse = float(np.sum(residuals ** 2))

        golden[expiry] = {
            "expiry": expiry,
            "ttm": float(ttm),
            "forward": float(forward),
            "spot": float(spot),
            "n_points": int(len(log_moneyness)),
            "heston_params": {
                "v0": float(params.v0),
                "kappa": float(params.kappa),
                "theta": float(params.theta),
                "xi": float(params.xi),
                "rho": float(params.rho),
            },
            "r_squared": float(fit_result.r_squared),
            "rmse": float(fit_result.rmse),
            "sse": sse,
            "bl_probability_above_forward": float(p_above),
            "bl_probability_below_forward": float(p_below),
            "bl_mean": float(rnd.mean),
            "bl_std": float(rnd.std_dev),
            "bl_integral": float(rnd.integral),
            "bl_is_valid": bool(rnd.is_valid),
            "timing_calibration_s": round(t_calibration, 3),
            "timing_bl_extraction_s": round(t_bl, 3),
        }

        print(
            f"    R²={fit_result.r_squared:.6f}, SSE={sse:.2e}, "
            f"P(above)={p_above:.6f}, P(below)={p_below:.6f}"
        )
        print(
            f"    Calibration: {t_calibration:.1f}s, BL: {t_bl:.1f}s"
        )

    # Save golden reference
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(golden, indent=2), encoding="utf-8")

    print(f"\nGolden reference saved to {OUTPUT_PATH}")
    print(f"  Expiries: {len(golden)}")
    print(f"  Total calibration time: {total_calibration_time:.1f}s")
    print(f"  Total BL extraction time: {total_bl_time:.1f}s")
    print(f"  Total pipeline time: {total_calibration_time + total_bl_time:.1f}s")


if __name__ == "__main__":
    main()
