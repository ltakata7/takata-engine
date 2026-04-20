"""Unit tests for multi-timeframe architecture."""

import numpy as np
import pandas as pd
import pytest

from takata_core.data.mtf_manager import MultiTimeframeManager
from takata_core.utils.session import get_wdo_session_phase, get_mes_session_phase


def _make_trend_df(n=250, direction="up", volatility=0.008):
    """Generate synthetic OHLCV with controlled trend."""
    rng = np.random.default_rng(42)
    drift = 0.003 if direction == "up" else -0.003 if direction == "down" else 0.0
    close = 100 * np.exp(np.cumsum(drift + rng.normal(0, volatility, n)))
    high = close * (1 + np.abs(rng.normal(0, 0.005, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.005, n)))
    open_ = np.roll(close, 1); open_[0] = 100
    vol = (1e6 + rng.integers(-1e5, 1e5, n)).astype(float)
    idx = pd.date_range("2025-01-02", periods=n, freq="B")
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": vol}, index=idx)


class TestMultiTimeframeManager:
    def test_bias_both_up(self):
        mtf = MultiTimeframeManager("MES")
        mtf.timeframes["daily"] = _make_trend_df(60, "up")
        mtf.timeframes["60m"] = _make_trend_df(100, "up")
        mtf._compute_bias()
        assert "long" in mtf.bias

    def test_bias_both_down(self):
        mtf = MultiTimeframeManager("MES")
        mtf.timeframes["daily"] = _make_trend_df(60, "down")
        mtf.timeframes["60m"] = _make_trend_df(100, "down")
        mtf._compute_bias()
        assert "short" in mtf.bias

    def test_bias_mixed(self):
        mtf = MultiTimeframeManager("MES")
        mtf.timeframes["daily"] = _make_trend_df(60, "up")
        mtf.timeframes["60m"] = _make_trend_df(100, "down")
        mtf._compute_bias()
        # Should be preferred, not "only"
        assert mtf.bias in ("long_preferred", "short_preferred", "neutral")

    def test_bias_neutral(self):
        mtf = MultiTimeframeManager("MES")
        mtf.timeframes["daily"] = _make_trend_df(60, "flat", volatility=0.002)
        mtf.timeframes["60m"] = _make_trend_df(100, "flat", volatility=0.002)
        mtf._compute_bias()
        assert mtf.bias in ("neutral", "long_preferred", "short_preferred")

    def test_levels_computed(self):
        mtf = MultiTimeframeManager("MES")
        mtf.timeframes["daily"] = _make_trend_df(60, "up")
        mtf.timeframes["60m"] = _make_trend_df(100, "up")
        mtf._compute_levels()
        assert "daily_support" in mtf.levels
        assert "daily_resistance" in mtf.levels
        assert "fibonacci" in mtf.levels
        assert "prev_day_high" in mtf.levels

    def test_nearest_support(self):
        mtf = MultiTimeframeManager("MES")
        mtf._levels = {
            "daily_support": [95, 90, 85],
            "hourly_support": [97, 93],
            "prev_day_low": 96,
            "daily_resistance": [], "hourly_resistance": [],
        }
        assert mtf.nearest_support(100) == 97

    def test_nearest_resistance(self):
        mtf = MultiTimeframeManager("MES")
        mtf._levels = {
            "daily_resistance": [105, 110, 115],
            "hourly_resistance": [103, 108],
            "prev_day_high": 104,
            "daily_support": [], "hourly_support": [],
        }
        assert mtf.nearest_resistance(100) == 103

    def test_entry_timeframe_crisis(self):
        mtf = MultiTimeframeManager("MES")
        assert mtf.select_entry_timeframe("crisis", 40) is None

    def test_entry_timeframe_high_vol(self):
        mtf = MultiTimeframeManager("MES")
        assert mtf.select_entry_timeframe("high_vol_choppy", 40) == "3m"

    def test_entry_timeframe_low_vol(self):
        mtf = MultiTimeframeManager("MES")
        assert mtf.select_entry_timeframe("low_vol_trending", 15) == "15m"

    def test_entry_timeframe_normal(self):
        mtf = MultiTimeframeManager("MES")
        assert mtf.select_entry_timeframe("low_vol_trending", 25) == "5m"

    def test_to_dict(self):
        mtf = MultiTimeframeManager("MES")
        mtf.timeframes["daily"] = _make_trend_df(30, "up")
        mtf.timeframes["5m"] = _make_trend_df(100, "up")
        mtf._compute_bias()
        mtf._compute_levels()
        d = mtf.to_dict()
        assert "bias" in d
        assert "trends" in d
        assert "levels" in d
        assert "timeframes_loaded" in d

    def test_needs_full_fetch(self):
        mtf = MultiTimeframeManager("MES")
        assert mtf.needs_full_fetch() is True  # no data yet
        mtf._last_full_fetch = __import__("time").time()
        mtf.timeframes["daily"] = _make_trend_df(30, "up")  # has data now
        assert mtf.needs_full_fetch(300) is False  # recently fetched + has data


class TestSessionPhases:
    def test_wdo_pre_ptax(self):
        ts = pd.Timestamp("2026-04-02 09:30", tz="America/Sao_Paulo")
        assert get_wdo_session_phase(ts) == "pre_ptax"

    def test_wdo_window_1(self):
        ts = pd.Timestamp("2026-04-02 10:05", tz="America/Sao_Paulo")
        assert get_wdo_session_phase(ts) == "ptax_window_1"

    def test_wdo_window_4(self):
        ts = pd.Timestamp("2026-04-02 13:05", tz="America/Sao_Paulo")
        assert get_wdo_session_phase(ts) == "ptax_window_4"

    def test_wdo_formation(self):
        ts = pd.Timestamp("2026-04-02 11:30", tz="America/Sao_Paulo")
        assert get_wdo_session_phase(ts) == "ptax_formation"

    def test_wdo_post_ptax(self):
        ts = pd.Timestamp("2026-04-02 14:00", tz="America/Sao_Paulo")
        assert get_wdo_session_phase(ts) == "post_ptax"

    def test_mes_open(self):
        ts = pd.Timestamp("2026-04-02 09:35", tz="America/New_York")
        assert get_mes_session_phase(ts) == "open"

    def test_mes_morning(self):
        ts = pd.Timestamp("2026-04-02 10:30", tz="America/New_York")
        assert get_mes_session_phase(ts) == "morning"

    def test_mes_midday(self):
        ts = pd.Timestamp("2026-04-02 12:00", tz="America/New_York")
        assert get_mes_session_phase(ts) == "midday"

    def test_mes_afternoon(self):
        ts = pd.Timestamp("2026-04-02 14:30", tz="America/New_York")
        assert get_mes_session_phase(ts) == "afternoon"

    def test_mes_close(self):
        ts = pd.Timestamp("2026-04-02 15:45", tz="America/New_York")
        assert get_mes_session_phase(ts) == "close"
