"""Tests for JSON file-based communication between scripts."""

import json
import pytest
from datetime import datetime, timezone, timedelta
from pathlib import Path
from unittest.mock import patch, MagicMock


class TestPolymarketJsonOutput:
    """Tests for polymarket_btc_daily.py JSON output."""

    def test_market_to_dict_complete_event(self):
        """Test market_to_dict with a complete event structure."""
        from scripts.polymarket_btc_daily import market_to_dict

        # Mock event data similar to real API response
        event = {
            "title": "Bitcoin Up or Down on January 30?",
            "description": "This market will resolve based on the BTC/USD price at Jan 30 '26 12:00 in the ET timezone",
            "endDate": (datetime.now(timezone.utc) + timedelta(hours=5, minutes=23)).isoformat(),
            "markets": [{
                "outcomes": ["Up", "Down"],
                "outcomePrices": ["0.65", "0.35"]
            }]
        }

        # Mock Binance API calls
        with patch('scripts.polymarket_btc_daily.get_binance_price', return_value=102500.0), \
             patch('scripts.polymarket_btc_daily.get_current_btc_price', return_value=103150.0), \
             patch('scripts.polymarket_btc_daily.parse_reference_time', return_value=datetime.now(timezone.utc)):

            result = market_to_dict(event)

        assert result["market_title"] == "Bitcoin Up or Down on January 30?"
        assert result["barrier"] == 102500.0
        assert result["current_price"] == 103150.0
        assert result["prob_up"] == 0.65
        assert result["prob_down"] == 0.35
        assert result["hours"] == 5
        assert result["minutes"] in (22, 23)  # Allow 1 minute variance due to timing
        assert abs(result["hours_remaining"] - 5.38) < 0.03  # Allow small timing variance
        assert "timestamp" in result

    def test_market_to_dict_json_string_outcomes(self):
        """Test market_to_dict when API returns JSON strings instead of lists."""
        from scripts.polymarket_btc_daily import market_to_dict

        event = {
            "title": "Bitcoin Up or Down on January 30?",
            "description": "",
            "endDate": (datetime.now(timezone.utc) + timedelta(hours=2)).isoformat(),
            "markets": [{
                "outcomes": '["Up", "Down"]',  # JSON string
                "outcomePrices": '["0.70", "0.30"]'  # JSON string
            }]
        }

        with patch('scripts.polymarket_btc_daily.parse_reference_time', return_value=None):
            result = market_to_dict(event)

        assert result["prob_up"] == 0.70
        assert result["prob_down"] == 0.30

    def test_market_to_dict_no_markets(self):
        """Test market_to_dict when no markets array exists."""
        from scripts.polymarket_btc_daily import market_to_dict

        event = {
            "title": "Bitcoin Up or Down on January 30?",
            "description": "",
            "endDate": (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat(),
            "markets": []
        }

        with patch('scripts.polymarket_btc_daily.parse_reference_time', return_value=None):
            result = market_to_dict(event)

        assert result["prob_up"] is None
        assert result["prob_down"] is None

    def test_market_to_dict_expired_market(self):
        """Test market_to_dict with expired market (negative time remaining)."""
        from scripts.polymarket_btc_daily import market_to_dict

        event = {
            "title": "Bitcoin Up or Down on January 29?",
            "description": "",
            "endDate": (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat(),
            "markets": [{
                "outcomes": ["Up", "Down"],
                "outcomePrices": ["0.50", "0.50"]
            }]
        }

        with patch('scripts.polymarket_btc_daily.parse_reference_time', return_value=None):
            result = market_to_dict(event)

        # Time fields should be None for expired market
        assert result["hours"] is None
        assert result["minutes"] is None
        assert result["hours_remaining"] is None


class TestCliTerminalJsonOutput:
    """Tests for cli_terminal.py JSON output."""

    def test_build_json_output_heston_only(self):
        """Test build_json_output with Heston B-L results."""
        from cli_terminal import build_json_output
        from btc_pricer.models.heston import HestonParams

        heston_params = HestonParams(
            v0=0.04, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7, ttm=0.01
        )

        # With default trading_model="ssvi_surface", Heston-only → preferred_model=None
        result_default = build_json_output(
            heston_r2=0.990,
            spot=103150.0,
            target_price=102500.0,
            direction="down",
            ttm=0.01,
            heston_params=heston_params,
            bl_prob_above=0.986,
            bl_prob_below=0.014,
        )

        assert result_default["spot_price"] == 103150.0
        assert result_default["target_price"] == 102500.0
        assert result_default["direction"] == "down"
        assert abs(result_default["ttm_days"] - 3.65) < 0.01

        # Heston results stored but NOT used for trading
        assert result_default["heston"]["prob_above"] == 0.986
        assert result_default["heston"]["prob_below"] == 0.014
        assert result_default["heston"]["r_squared"] == 0.990
        assert result_default["preferred_model"] is None
        assert result_default["avg_prob_above"] is None
        assert "timestamp" in result_default

        # With trading_model="heston", Heston-only → preferred_model="heston"
        result_heston = build_json_output(
            heston_r2=0.990,
            spot=103150.0,
            target_price=102500.0,
            direction="down",
            ttm=0.01,
            heston_params=heston_params,
            bl_prob_above=0.986,
            bl_prob_below=0.014,
            trading_model="heston",
        )

        assert result_heston["preferred_model"] == "heston"
        assert result_heston["avg_prob_above"] == 0.986
        assert result_heston["avg_prob_below"] == 0.014

    def test_build_json_output_surface_preferred(self):
        """Test build_json_output prefers SSVI Surface when available."""
        from cli_terminal import build_json_output
        from btc_pricer.models.heston import HestonParams

        heston_params = HestonParams(
            v0=0.04, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7, ttm=0.01
        )

        result = build_json_output(
            heston_r2=0.990,
            spot=103150.0,
            target_price=102500.0,
            direction="down",
            ttm=0.01,
            heston_params=heston_params,
            bl_prob_above=0.986,
            bl_prob_below=0.014,
            surface_bl_above=0.984,
            surface_bl_below=0.016,
            surface_mc_above=0.982,
            surface_mc_below=0.018,
            surface_r2=0.992,
        )

        assert result["preferred_model"] == "ssvi_surface"
        assert result["ssvi_surface"]["prob_above"] == 0.984
        assert result["ssvi_surface"]["mc_prob_above"] == 0.982
        # avg_prob should average MC and BL for SSVI Surface
        assert abs(result["avg_prob_above"] - (0.984 + 0.982) / 2) < 0.001

    def test_build_json_output_no_models(self):
        """Test build_json_output when no models succeed."""
        from cli_terminal import build_json_output

        result = build_json_output(
            heston_r2=None,
            spot=103150.0,
            target_price=102500.0,
            direction="down",
            ttm=0.01,
        )

        assert result["heston"] is None
        assert result["preferred_model"] is None


class TestPolymarketEdgeJsonParsing:
    """Tests for polymarket_edge.py JSON and regex parsing."""

    def test_parse_polymarket_stdout(self):
        """Test regex fallback parsing of Polymarket stdout."""
        from scripts.polymarket_edge import _parse_polymarket_stdout

        stdout = """Fetching Bitcoin Up/Down markets from Polymarket...

Bitcoin Up or Down on January 30?
Expires: 12:00 ET / 17:00 UTC / 18:00 Paris
Time remaining: 5h 23m
----------------------------------------
Price to beat: $102,500.00
Current price: $103,150.00
Difference:    above by $650.00 (+0.63%)
----------------------------------------
UP    65.0% ($0.650)
DOWN  35.0% ($0.350)
"""

        result = _parse_polymarket_stdout(stdout)

        assert result["barrier"] == 102500.0
        assert result["current_price"] == 103150.0
        assert result["hours"] == 5
        assert result["minutes"] == 23
        assert abs(result["hours_remaining"] - 5.383) < 0.01
        assert result["prob_up"] == 0.65
        assert result["prob_down"] == 0.35

    def test_parse_polymarket_stdout_no_data(self):
        """Test regex fallback when output has no parseable data."""
        from scripts.polymarket_edge import _parse_polymarket_stdout

        stdout = "No active Bitcoin Up/Down market found"

        result = _parse_polymarket_stdout(stdout)

        assert result["barrier"] is None
        assert result["prob_up"] is None
        assert result["prob_down"] is None
        assert result["hours"] == 0
        assert result["minutes"] == 0

    def test_parse_terminal_stdout(self):
        """Test regex fallback parsing of terminal script stdout."""
        from scripts.polymarket_edge import _parse_terminal_stdout

        stdout = """
----------------------------------------------------------------------
 Model                      R²     P(>target)      P(<target)
----------------------------------------------------------------------
 Heston MC                 0.990    98.6% ±0.0%            1.4%
 SSVI Surface MC           0.992    98.4% ±0.0%            1.6%
----------------------------------------------------------------------
 Spot Price:        $103,150.00 (Binance)
"""

        result = _parse_terminal_stdout(stdout)

        assert result["spot_price"] == 103150.0
        # Should parse SSVI row (preferred)
        assert abs(result["prob_above"] - 0.984) < 0.001
        assert abs(result["prob_below"] - 0.016) < 0.001

    def test_parse_terminal_stdout_heston_only(self):
        """Test regex fallback when only Heston row exists."""
        from scripts.polymarket_edge import _parse_terminal_stdout

        stdout = """
----------------------------------------------------------------------
 Spot Price:        $103,150.00 (Binance)
----------------------------------------------------------------------
 Heston MC                 0.990    98.6% ±0.0%            1.4%
----------------------------------------------------------------------
"""

        result = _parse_terminal_stdout(stdout)

        assert abs(result["prob_above"] - 0.986) < 0.001
        assert abs(result["prob_below"] - 0.014) < 0.001

    def test_run_polymarket_script_json_success(self, tmp_path):
        """Test run_polymarket_script reads JSON file successfully."""
        from scripts.polymarket_edge import run_polymarket_script

        # Create test JSON file
        json_data = {
            "barrier": 102500.0,
            "current_price": 103150.0,
            "hours_remaining": 5.383,
            "hours": 5,
            "minutes": 23,
            "prob_up": 0.65,
            "prob_down": 0.35,
            "market_title": "Bitcoin Up or Down on January 30?",
            "timestamp": "2026-01-30T11:37:00+00:00"
        }
        json_file = tmp_path / "polymarket_data.json"
        with open(json_file, "w") as f:
            json.dump(json_data, f)

        # Mock subprocess to avoid actual script execution
        with patch('scripts.polymarket_edge.subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(stdout="Script output", stderr="", returncode=0)

            result = run_polymarket_script(verbose=False, json_path=json_file)

        assert result["barrier"] == 102500.0
        assert result["prob_up"] == 0.65
        assert result["prob_down"] == 0.35
        assert result["hours"] == 5
        assert result["minutes"] == 23

    def test_run_polymarket_script_json_fallback(self, tmp_path):
        """Test run_polymarket_script falls back to regex on JSON error."""
        from scripts.polymarket_edge import run_polymarket_script

        # Create invalid JSON file
        json_file = tmp_path / "polymarket_data.json"
        with open(json_file, "w") as f:
            f.write("not valid json{")

        stdout = """Time remaining: 3h 15m
Price to beat: $101,000.00
Current price: $102,000.00
UP    60.0% ($0.600)
DOWN  40.0% ($0.400)
"""

        with patch('scripts.polymarket_edge.subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(stdout=stdout, stderr="", returncode=0)

            result = run_polymarket_script(verbose=False, json_path=json_file)

        # Should fall back to regex parsing
        assert result["barrier"] == 101000.0
        assert result["prob_up"] == 0.60
        assert result["hours"] == 3
        assert result["minutes"] == 15

    def test_run_terminal_script_json_success(self, tmp_path):
        """Test run_terminal_script reads JSON file successfully."""
        from scripts.polymarket_edge import run_terminal_script

        # Create test JSON file
        json_data = {
            "spot_price": 103150.0,
            "target_price": 102500.0,
            "direction": "down",
            "ttm_days": 0.224,
            "heston": {
                "prob_above": 0.986,
                "prob_below": 0.014,
                "r_squared": 0.990
            },
            "ssvi_surface": {
                "prob_above": 0.984,
                "prob_below": 0.016,
                "r_squared": 0.992
            },
            "preferred_model": "ssvi_surface",
            "timestamp": "2026-01-30T11:37:00+00:00"
        }
        json_file = tmp_path / "terminal_data.json"
        with open(json_file, "w") as f:
            json.dump(json_data, f)

        with patch('scripts.polymarket_edge.subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(stdout="Script output", stderr="", returncode=0)

            result = run_terminal_script(102500.0, 5.38, verbose=False, json_path=json_file)

        # Should use SSVI (preferred model)
        assert result["spot_price"] == 103150.0
        assert result["prob_above"] == 0.984
        assert result["prob_below"] == 0.016

    def test_run_terminal_script_json_fallback(self, tmp_path):
        """Test run_terminal_script falls back to regex on JSON error."""
        from scripts.polymarket_edge import run_terminal_script

        # Create invalid JSON file
        json_file = tmp_path / "terminal_data.json"
        with open(json_file, "w") as f:
            f.write("{invalid")

        stdout = """
 Spot Price:        $100,000.00 (Binance)
 SSVI Surface MC           0.985    97.0% ±0.1%            3.0%
"""

        with patch('scripts.polymarket_edge.subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(stdout=stdout, stderr="", returncode=0)

            result = run_terminal_script(99000.0, 4.0, verbose=False, json_path=json_file)

        # Should fall back to regex parsing
        assert result["spot_price"] == 100000.0
        assert result["prob_above"] == 0.97
        assert result["prob_below"] == 0.03


class TestFindOpportunities:
    """Tests for polymarket_edge.find_opportunities()."""

    def test_up_edge_detected(self):
        """Model prob exceeds continuous edge curve -> has_edge = True."""
        from scripts.polymarket_edge import find_opportunities

        poly = {"prob_up": 0.70, "prob_down": 0.30}
        model = {"prob_above": 0.95, "prob_below": 0.05}
        opps = find_opportunities(poly, model, alpha_up=1.5, alpha_down=1.5)

        up_opp = next(o for o in opps if o["direction"] == "UP")
        assert up_opp["has_edge"] is True
        assert up_opp["edge"] == pytest.approx(0.95 / 0.70, rel=1e-3)

    def test_down_edge_detected(self):
        """Model prob exceeds continuous edge curve for DOWN -> has_edge = True."""
        from scripts.polymarket_edge import find_opportunities

        poly = {"prob_up": 0.30, "prob_down": 0.70}
        model = {"prob_above": 0.05, "prob_below": 0.95}
        opps = find_opportunities(poly, model, alpha_up=1.5, alpha_down=1.5)

        down_opp = next(o for o in opps if o["direction"] == "DOWN")
        assert down_opp["has_edge"] is True
        assert down_opp["edge"] == pytest.approx(0.95 / 0.70, rel=1e-3)

    def test_no_edge(self):
        """Edge below thresholds -> has_edge = False for both."""
        from scripts.polymarket_edge import find_opportunities

        poly = {"prob_up": 0.50, "prob_down": 0.50}
        model = {"prob_above": 0.55, "prob_below": 0.45}
        opps = find_opportunities(poly, model, alpha_up=1.5, alpha_down=1.5)

        for opp in opps:
            assert opp["has_edge"] is False

    def test_missing_probabilities_excluded(self):
        """None probabilities -> no opportunity for that direction."""
        from scripts.polymarket_edge import find_opportunities

        poly = {"prob_up": None, "prob_down": 0.50}
        model = {"prob_above": 0.80, "prob_below": 0.20}
        opps = find_opportunities(poly, model)

        directions = [o["direction"] for o in opps]
        assert "UP" not in directions
        assert "DOWN" in directions

    def test_both_directions_returned(self):
        """With valid data, both UP and DOWN opportunities are returned."""
        from scripts.polymarket_edge import find_opportunities

        poly = {"prob_up": 0.50, "prob_down": 0.50}
        model = {"prob_above": 0.50, "prob_below": 0.50}
        opps = find_opportunities(poly, model)

        assert len(opps) == 2
        directions = {o["direction"] for o in opps}
        assert directions == {"UP", "DOWN"}

    def test_opportunity_fields(self):
        """Each opportunity has expected fields."""
        from scripts.polymarket_edge import find_opportunities

        poly = {"prob_up": 0.40, "prob_down": 0.60}
        model = {"prob_above": 0.90, "prob_below": 0.10}
        opps = find_opportunities(poly, model)

        for opp in opps:
            assert "direction" in opp
            assert "polymarket_prob" in opp
            assert "model_prob" in opp
            assert "edge" in opp
            assert "has_edge" in opp
            assert "required_prob" in opp
            assert "market_entry" in opp
            assert "take_profit" in opp


class TestJsonFileSchemas:
    """Tests to verify JSON output matches expected schemas."""

    def test_polymarket_json_schema(self):
        """Verify Polymarket JSON has all required fields."""
        from scripts.polymarket_btc_daily import market_to_dict

        event = {
            "title": "Test Market",
            "description": "",
            "endDate": (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat(),
            "markets": [{
                "outcomes": ["Up", "Down"],
                "outcomePrices": ["0.50", "0.50"]
            }]
        }

        with patch('scripts.polymarket_btc_daily.parse_reference_time', return_value=None):
            result = market_to_dict(event)

        required_fields = [
            "barrier", "current_price", "hours_remaining", "hours", "minutes",
            "prob_up", "prob_down", "market_title", "timestamp"
        ]

        for field in required_fields:
            assert field in result, f"Missing required field: {field}"

    def test_terminal_json_schema(self):
        """Verify terminal JSON has all required fields."""
        from cli_terminal import build_json_output
        from btc_pricer.models.heston import HestonParams

        heston_params = HestonParams(
            v0=0.04, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7, ttm=0.01
        )

        output = build_json_output(
            heston_r2=0.99,
            spot=101000.0,
            target_price=100000.0,
            direction="down",
            ttm=0.01,
            heston_params=heston_params,
            bl_prob_above=0.97,
            bl_prob_below=0.03,
        )

        required_fields = [
            "spot_price", "target_price", "direction", "ttm_days",
            "heston", "preferred_model", "timestamp"
        ]

        for field in required_fields:
            assert field in output, f"Missing required field: {field}"

        # Check nested structure when model exists
        heston_fields = ["prob_above", "prob_below", "r_squared"]
        for field in heston_fields:
            assert field in output["heston"], f"Missing heston field: {field}"
