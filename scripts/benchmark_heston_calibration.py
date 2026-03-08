#!/usr/bin/env python3
"""Benchmark Heston calibration on replayed real-Deribit snapshots."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from btc_pricer.api.deribit import OptionData
from btc_pricer.cli.common import extract_surface_data
from btc_pricer.config import Config
from btc_pricer.constants import RELAXED_MIN_POINTS
from btc_pricer.data.filters import DataFilter
from btc_pricer.models.heston import HestonFitter


@dataclass
class SnapshotCase:
    expiry: str
    forward: float
    ttm: float
    log_moneyness: np.ndarray
    market_iv: np.ndarray


def _build_fitter(config: Config, mode: str, enable_numba: bool) -> HestonFitter:
    # "legacy"/"optimized" select the QuantLib objective implementation
    ql_impl = mode if mode in ("legacy", "optimized") else "optimized"
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
        quantlib_objective_impl=ql_impl,
        enable_numba_fallback=enable_numba,
        numba_strict_mode=config.heston.numba_strict_mode,
    )


def _load_cases(snapshot_path: Path, config: Config) -> list[SnapshotCase]:
    payload = json.loads(snapshot_path.read_text(encoding="utf-8"))
    data_filter = DataFilter(config.filters)
    cases: list[SnapshotCase] = []

    for expiry in payload["selected_expiries"]:
        raw_exp = payload["expiries"][expiry]
        options = [OptionData(**opt) for opt in raw_exp["options"]]
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
        forward, ttm, _spot, log_moneyness, market_iv = surface_data
        cases.append(
            SnapshotCase(
                expiry=expiry,
                forward=float(forward),
                ttm=float(ttm),
                log_moneyness=log_moneyness,
                market_iv=market_iv,
            )
        )
    return cases


def run_benchmark(
    snapshot_path: Path,
    config_path: Path,
    mode: str,
    enable_numba: bool,
    repeats: int
) -> dict[str, Any]:
    config = Config.from_yaml(config_path) if config_path.exists() else Config()
    fitter = _build_fitter(config, mode=mode, enable_numba=enable_numba)
    cases = _load_cases(snapshot_path, config)

    case_results = []
    all_times = []

    for case in cases:
        run_times = []
        run_r2 = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            result = fitter.fit(
                case.log_moneyness, case.market_iv, case.ttm, forward=case.forward
            )
            dt = time.perf_counter() - t0
            run_times.append(dt)
            run_r2.append(float(result.r_squared))
            all_times.append(dt)
        case_results.append({
            "expiry": case.expiry,
            "n_points": int(len(case.log_moneyness)),
            "ttm": case.ttm,
            "median_s": float(statistics.median(run_times)),
            "p95_s": float(np.percentile(run_times, 95)),
            "r2_values": run_r2,
        })

    return {
        "mode": mode,
        "enable_numba_fallback": bool(enable_numba),
        "snapshot_path": str(snapshot_path),
        "repeats": int(repeats),
        "cases": case_results,
        "aggregate": {
            "n_cases": len(case_results),
            "median_s": float(statistics.median(all_times)) if all_times else 0.0,
            "p95_s": float(np.percentile(all_times, 95)) if all_times else 0.0,
            "total_s": float(sum(all_times)),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark Heston calibration on replayed Deribit snapshots."
    )
    parser.add_argument(
        "--snapshot",
        type=Path,
        default=Path("tests/fixtures/deribit_snapshot_8exp.json"),
        help="Path to captured snapshot fixture"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config.yaml"),
        help="Path to config.yaml"
    )
    parser.add_argument(
        "--mode",
        choices=["legacy", "optimized"],
        default="optimized",
        help="Calibration mode: legacy or optimized (Python/QuantLib)"
    )
    parser.add_argument(
        "--numba-fallback",
        choices=["on", "off"],
        default="on",
        help="Enable Numba in fallback objective"
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="Repetitions per expiry (default: 1)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output JSON path"
    )
    args = parser.parse_args()

    result = run_benchmark(
        snapshot_path=args.snapshot,
        config_path=args.config,
        mode=args.mode,
        enable_numba=(args.numba_fallback == "on"),
        repeats=args.repeats
    )

    print(json.dumps(result, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
