#!/usr/bin/env python3
"""Capture real Deribit option-chain snapshots for deterministic replay tests."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from btc_pricer.api.deribit import DeribitClient
from btc_pricer.cli.common import extract_surface_data
from btc_pricer.config import Config
from btc_pricer.constants import RELAXED_MIN_POINTS
from btc_pricer.data.filters import DataFilter


def _sorted_expiries(options_by_expiry: dict[str, list]) -> list[tuple[str, list]]:
    """Sort expiries by nearest maturity."""
    return sorted(
        options_by_expiry.items(),
        key=lambda x: x[1][0].time_to_expiry if x[1] else float("inf")
    )


def capture_snapshot(
    config_path: Path,
    output_path: Path,
    n_expiries: int
) -> dict:
    """Fetch and serialize a real Deribit snapshot for several expiries."""
    config = Config.from_yaml(config_path) if config_path.exists() else Config()
    client = DeribitClient(config.api, config.validation)
    data_filter = DataFilter(config.filters)

    options_by_expiry = client.fetch_all_options("BTC")
    sorted_expiries = _sorted_expiries(options_by_expiry)

    selected: dict[str, dict] = {}
    skipped: dict[str, str] = {}

    for expiry, options in sorted_expiries:
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
            skipped[expiry] = "insufficient_surface_points"
            continue

        forward, ttm, spot, _, _ = surface_data
        selected[expiry] = {
            "spot_price": float(spot),
            "forward_price": float(forward),
            "ttm": float(ttm),
            "n_options_raw": int(len(options)),
            "n_options_filtered": int(len(filtered)),
            "options": [asdict(opt) for opt in options],
        }
        if len(selected) >= n_expiries:
            break

    if len(selected) < n_expiries:
        raise RuntimeError(
            f"Only captured {len(selected)} valid expiries, requested {n_expiries}. "
            f"Skipped: {skipped}"
        )

    payload = {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": "deribit",
        "currency": "BTC",
        "requested_expiries": int(n_expiries),
        "selected_expiries": list(selected.keys()),
        "filters": {
            "min_open_interest": config.filters.min_open_interest,
            "max_bid_ask_spread_pct": config.filters.max_bid_ask_spread_pct,
            "moneyness_range": list(config.filters.moneyness_range),
            "min_surface_points_relaxed": RELAXED_MIN_POINTS,
            "iv_valid_range": list(config.validation.iv_valid_range),
        },
        "expiries": selected,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Capture a real Deribit snapshot for replayable calibration tests."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config.yaml"),
        help="Path to config.yaml (default: config.yaml)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("tests/fixtures/deribit_snapshot_8exp.json"),
        help="Output JSON path (default: tests/fixtures/deribit_snapshot_8exp.json)"
    )
    parser.add_argument(
        "--expiries",
        type=int,
        default=8,
        help="Number of valid expiries to capture (default: 8)"
    )
    args = parser.parse_args()

    payload = capture_snapshot(args.config, args.output, args.expiries)
    print(
        f"Captured {len(payload['selected_expiries'])} expiries to {args.output}:\n"
        + "\n".join(f"  - {exp}" for exp in payload["selected_expiries"])
    )


if __name__ == "__main__":
    main()
