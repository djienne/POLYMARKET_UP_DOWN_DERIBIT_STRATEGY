"""Unit tests for btc_pricer.edge — continuous edge function."""

import pytest
from btc_pricer.edge import required_model_prob, has_edge


class TestRequiredModelProb:
    """Tests for required_model_prob() with default alpha=1.5, floor=0.65."""

    def test_market_50_hits_floor(self):
        """At market_prob=0.50, raw ≈ 0.646 < floor 0.65 → floor applies."""
        assert required_model_prob(0.50) == pytest.approx(0.65)

    def test_market_80(self):
        """At market_prob=0.80, required ≈ 0.911."""
        assert required_model_prob(0.80) == pytest.approx(0.911, abs=0.002)

    def test_market_90(self):
        """At market_prob=0.90, required ≈ 0.968."""
        assert required_model_prob(0.90) == pytest.approx(0.968, abs=0.002)

    def test_market_95(self):
        """At market_prob=0.95, required ≈ 0.989."""
        assert required_model_prob(0.95) == pytest.approx(0.989, abs=0.002)

    def test_market_30_floor(self):
        """At market_prob=0.30, floor kicks in → 0.65."""
        assert required_model_prob(0.30) == 0.65

    def test_market_0_returns_floor(self):
        """Edge case: market_prob=0 → floor."""
        assert required_model_prob(0.0) == 0.65

    def test_market_1_returns_1(self):
        """Edge case: market_prob=1.0 → 1.0."""
        assert required_model_prob(1.0) == 1.0

    def test_alpha_1_linear(self):
        """With alpha=1.0, required = max(floor, market_prob) — identity on doubt."""
        # market_prob=0.80 → raw = 1 - (0.20)^1 = 0.80
        assert required_model_prob(0.80, alpha=1.0, floor=0.5) == pytest.approx(0.80)
        # market_prob=0.40 → raw = 0.40, but floor=0.5 → 0.5
        assert required_model_prob(0.40, alpha=1.0, floor=0.5) == pytest.approx(0.5)

    def test_custom_floor(self):
        """Custom floor=0.70 at low market prob."""
        assert required_model_prob(0.30, alpha=1.5, floor=0.70) == 0.70

    def test_monotonically_increasing(self):
        """Required prob should increase as market_prob increases."""
        probs = [required_model_prob(p / 100) for p in range(1, 100)]
        for i in range(len(probs) - 1):
            assert probs[i] <= probs[i + 1] + 1e-12


class TestHasEdge:
    """Tests for has_edge() with default alpha=1.5, floor=0.65."""

    def test_edge_at_90_true(self):
        """model=0.97 vs market=0.90 → required ≈ 0.968 → True."""
        assert has_edge(0.97, 0.90) is True

    def test_edge_at_90_false(self):
        """model=0.96 vs market=0.90 → required ≈ 0.968 → False."""
        assert has_edge(0.96, 0.90) is False

    def test_below_floor_false(self):
        """model=0.64 vs market=0.30 → floor 0.65 → False."""
        assert has_edge(0.64, 0.30) is False

    def test_above_floor_but_market_below_floor_false(self):
        """model=0.66 vs market=0.30 → market < floor 0.65 → False."""
        assert has_edge(0.66, 0.30) is False

    def test_exact_threshold(self):
        """model == required → True (>= comparison)."""
        req = required_model_prob(0.80)
        assert has_edge(req, 0.80) is True

    def test_high_confidence_natural(self):
        """model=0.999 vs market=0.94 should pass naturally (no hardcoded override needed)."""
        assert has_edge(0.999, 0.94) is True

    def test_live_bug_market_below_floor_rejected(self):
        """Regression: model=35.2% market=17.5% floor=35% → market < floor → False."""
        assert has_edge(0.352, 0.175, alpha=1.6, floor=0.35) is False

    def test_both_above_floor_with_edge(self):
        """model=80% market=40% floor=35% → both above floor, model has edge → True."""
        assert has_edge(0.80, 0.40, alpha=1.5, floor=0.35) is True

    def test_market_below_floor_rejected(self):
        """model=80% market=10% floor=35% → market below floor → False."""
        assert has_edge(0.80, 0.10, alpha=1.5, floor=0.35) is False
