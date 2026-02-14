"""Unit tests for scripts/optimize.py helper functions."""

import pytest
import math
import sys
from pathlib import Path

# optimize.py uses `from backtest import ...` (relative to scripts/),
# so we add scripts/ to sys.path for the import to work.
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from optimize import (
    compute_sharpe_ratio,
    compute_profit_factor,
    compute_mtm_sharpe,
    compute_max_drawdown,
    analyze_parameter_stability,
)


class TestComputeSharpeRatio:
    """Tests for compute_sharpe_ratio()."""

    def test_basic_positive_sharpe(self):
        """Positive mean with some variance -> positive Sharpe."""
        pnls = [1.0, 2.0, 1.5, 2.5, 1.0]
        result = compute_sharpe_ratio(pnls)
        assert result is not None
        assert result > 0

    def test_all_equal_returns_none(self):
        """Zero std dev -> None (division by zero guard)."""
        pnls = [1.0, 1.0, 1.0]
        assert compute_sharpe_ratio(pnls) is None

    def test_single_trade_returns_none(self):
        """Fewer than 2 trades -> None."""
        assert compute_sharpe_ratio([5.0]) is None

    def test_empty_returns_none(self):
        """Empty list -> None."""
        assert compute_sharpe_ratio([]) is None

    def test_negative_mean(self):
        """Negative mean P&L -> negative Sharpe."""
        pnls = [-1.0, -2.0, -1.5]
        result = compute_sharpe_ratio(pnls)
        assert result is not None
        assert result < 0

    def test_known_value(self):
        """Verify against hand-calculated Sharpe."""
        # mean = 1.0, sample std = sqrt(((0-1)^2 + (2-1)^2) / 1) = sqrt(2)
        pnls = [0.0, 2.0]
        result = compute_sharpe_ratio(pnls)
        expected = 1.0 / math.sqrt(2)
        assert result == pytest.approx(expected, rel=1e-6)

    def test_risk_free_rate(self):
        """Risk-free rate shifts the numerator."""
        pnls = [2.0, 4.0]
        with_rf = compute_sharpe_ratio(pnls, risk_free_rate=1.0)
        without_rf = compute_sharpe_ratio(pnls, risk_free_rate=0.0)
        assert with_rf < without_rf


class TestComputeProfitFactor:
    """Tests for compute_profit_factor()."""

    def test_basic_profit_factor(self):
        """Gross profit > gross loss -> PF > 1."""
        pnls = [3.0, -1.0, 2.0, -0.5]
        result = compute_profit_factor(pnls)
        # gross_profit = 5.0, gross_loss = 1.5
        assert result == pytest.approx(5.0 / 1.5)

    def test_no_losses_returns_inf(self):
        """All wins, no losses -> inf."""
        pnls = [1.0, 2.0, 3.0]
        assert compute_profit_factor(pnls) == float('inf')

    def test_no_wins_returns_none(self):
        """All losses, no wins -> None (0/loss)."""
        # gross_profit = 0, gross_loss > 0 -> 0 / loss = 0.0
        pnls = [-1.0, -2.0]
        result = compute_profit_factor(pnls)
        assert result == pytest.approx(0.0)

    def test_empty_returns_none(self):
        """Empty list -> None."""
        assert compute_profit_factor([]) is None

    def test_all_zero_returns_none(self):
        """All zero P&L -> no profit, no loss -> None."""
        assert compute_profit_factor([0.0, 0.0]) is None


class TestComputeMtmSharpe:
    """Tests for compute_mtm_sharpe()."""

    def test_basic_upward_curve(self):
        """Monotonically increasing curve -> positive Sharpe."""
        values = [0.0, 1.0, 2.0, 3.0, 4.0]
        result = compute_mtm_sharpe(values)
        # All changes = 1.0, std = 0 -> None (constant changes)
        assert result is None  # std == 0

    def test_varying_curve(self):
        """Non-constant changes -> valid Sharpe."""
        values = [0.0, 2.0, 1.0, 4.0, 3.0]
        result = compute_mtm_sharpe(values)
        assert result is not None

    def test_too_few_returns_none(self):
        """Fewer than 3 values -> None."""
        assert compute_mtm_sharpe([0.0, 1.0]) is None
        assert compute_mtm_sharpe([]) is None


class TestComputeMaxDrawdown:
    """Tests for compute_max_drawdown()."""

    def test_no_drawdown(self):
        """Monotonically increasing -> drawdown = 0."""
        pnl = [0.0, 1.0, 2.0, 3.0]
        result = compute_max_drawdown(pnl)
        assert result["max_dd"] == 0.0
        assert result["max_dd_pct"] == 0.0

    def test_simple_drawdown(self):
        """Peak then drop then recover."""
        pnl = [0.0, 5.0, 2.0, 6.0]
        result = compute_max_drawdown(pnl, capital=100.0)
        assert result["max_dd"] == pytest.approx(-3.0)
        # Peak equity = 100 + 5 = 105, dd% = -3/105 * 100
        assert result["max_dd_pct"] == pytest.approx(-3.0 / 105.0 * 100)

    def test_all_negative(self):
        """Continuous decline from start."""
        pnl = [0.0, -1.0, -3.0, -5.0]
        result = compute_max_drawdown(pnl, capital=100.0)
        assert result["max_dd"] == pytest.approx(-5.0)

    def test_single_value(self):
        """Single value -> no drawdown."""
        result = compute_max_drawdown([0.0])
        assert result["max_dd"] == 0.0

    def test_peak_trough_indices(self):
        """Verify correct peak/trough index identification."""
        pnl = [0.0, 10.0, 3.0, 8.0, 1.0, 12.0]
        result = compute_max_drawdown(pnl)
        assert result["peak_idx"] == 1
        assert result["trough_idx"] == 4

    def test_timestamps_included(self):
        """When timestamps provided, peak_ts and trough_ts are in result."""
        from datetime import datetime
        pnl = [0.0, 5.0, 2.0]
        ts = [datetime(2025, 1, 1), datetime(2025, 1, 2), datetime(2025, 1, 3)]
        result = compute_max_drawdown(pnl, timestamps=ts)
        assert result["peak_ts"] == ts[1]
        assert result["trough_ts"] == ts[2]


class TestAnalyzeParameterStability:
    """Tests for analyze_parameter_stability()."""

    def _make_results(self, n=20):
        """Create synthetic optimization results."""
        results = []
        for i in range(n):
            results.append({
                "edge_up": 1.5 + i * 0.02,
                "edge_down": 1.2 + i * 0.01,
                "tp_pct": 0.20 + i * 0.005,
                "trail_activation": 0.15 + i * 0.005,
                "trail_distance": 0.10 + i * 0.002,
                "min_model_prob": 0.5 + i * 0.02,
                "trades": 10 + i,
                "pnl": 10.0 - i * 0.3,  # decreasing P&L
            })
        return results

    def test_returns_stats_for_all_params(self):
        """Should return stats for all parameter names."""
        results = self._make_results()
        stability = analyze_parameter_stability(results, min_trades=5)
        assert "edge_up" in stability
        assert "edge_down" in stability
        assert "tp_pct" in stability
        assert "trail_activation" in stability
        assert "trail_distance" in stability

    def test_stats_have_expected_keys(self):
        """Each param stat should have min, max, mean, median."""
        results = self._make_results()
        stability = analyze_parameter_stability(results, min_trades=5)
        for key in ("edge_up", "edge_down", "tp_pct"):
            s = stability[key]
            assert "min" in s
            assert "max" in s
            assert "mean" in s
            assert "median" in s

    def test_min_trades_filters(self):
        """Results with too few trades should be excluded."""
        results = [
            {"edge_up": 1.5, "edge_down": 1.2, "tp_pct": 0.20,
             "trail_activation": 0.15, "trail_distance": 0.10,
             "trades": 2, "pnl": 100.0},
        ]
        stability = analyze_parameter_stability(results, min_trades=5)
        assert stability == {}

    def test_empty_results(self):
        """Empty input -> empty dict."""
        assert analyze_parameter_stability([], min_trades=1) == {}
