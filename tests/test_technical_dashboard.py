"""Unit tests for technical dashboard components."""

import numpy as np
import pandas as pd
import pytest

from takata_core.technical.trend import classify_trend, trend_label_short
from takata_core.technical.levels import (
    fibonacci_retracement,
    fibonacci_extension,
    support_resistance,
    pivot_points,
)
from takata_core.technical.patterns import detect_double_top, detect_double_bottom


def _make_uptrend(n=250):
    """Generate synthetic uptrending OHLCV data."""
    rng = np.random.default_rng(42)
    close = 100 * np.exp(np.cumsum(0.003 + rng.normal(0, 0.008, n)))
    high = close * (1 + np.abs(rng.normal(0, 0.005, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.005, n)))
    open_ = np.roll(close, 1); open_[0] = 100
    vol = (1e6 + rng.integers(-1e5, 1e5, n)).astype(float)
    idx = pd.date_range("2024-01-02", periods=n, freq="B")
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": vol}, index=idx)


def _make_downtrend(n=250):
    rng = np.random.default_rng(42)
    close = 100 * np.exp(np.cumsum(-0.001 + rng.normal(0, 0.01, n)))
    high = close * (1 + np.abs(rng.normal(0, 0.005, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.005, n)))
    open_ = np.roll(close, 1); open_[0] = 100
    vol = (1e6 + rng.integers(-1e5, 1e5, n)).astype(float)
    idx = pd.date_range("2024-01-02", periods=n, freq="B")
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": vol}, index=idx)


class TestTrend:
    def test_uptrend_detected(self):
        df = _make_uptrend()
        trend = classify_trend(df)
        assert "uptrend" in trend

    def test_downtrend_detected(self):
        df = _make_downtrend()
        trend = classify_trend(df)
        assert "downtrend" in trend

    def test_trend_label_short(self):
        assert trend_label_short("strong_uptrend") == "STRONG UP"
        assert trend_label_short("sideways") == "SIDEWAYS"


class TestFibonacci:
    def test_retracement_levels(self):
        fib = fibonacci_retracement(200.0, 100.0)
        assert fib["0.0%"] == 200.0
        assert fib["100.0%"] == 100.0
        assert abs(fib["50.0%"] - 150.0) < 0.01
        assert abs(fib["61.8%"] - 138.2) < 0.1

    def test_extension_levels(self):
        ext = fibonacci_extension(200.0, 100.0, 138.2)
        assert abs(ext["100.0%"] - 238.2) < 0.1
        assert abs(ext["161.8%"] - 300.0) < 0.1


class TestSupportResistance:
    def test_returns_levels(self):
        df = _make_uptrend()
        sr = support_resistance(df, n_levels=3)
        assert "support" in sr
        assert "resistance" in sr

    def test_pivot_points_keys(self):
        df = _make_uptrend()
        pvt = pivot_points(df)
        assert "P" in pvt
        assert "R1" in pvt
        assert "S1" in pvt
        assert pvt["R1"] > pvt["P"] > pvt["S1"]


class TestPatterns:
    def test_double_top_on_synthetic(self):
        """Build data with two peaks at similar levels."""
        rng = np.random.default_rng(42)
        n = 100
        # Ramp up → peak → dip → peak → dip
        base = np.concatenate([
            np.linspace(100, 120, 25),
            np.linspace(120, 110, 15),
            np.linspace(110, 119.5, 15),
            np.linspace(119.5, 105, 20),
            np.linspace(105, 108, 25),
        ])
        noise = rng.normal(0, 0.3, n)
        close = base + noise
        high = close + np.abs(rng.normal(0, 0.5, n))
        low = close - np.abs(rng.normal(0, 0.5, n))
        idx = pd.date_range("2024-01-02", periods=n, freq="B")
        df = pd.DataFrame({"open": close, "high": high, "low": low, "close": close, "volume": np.ones(n) * 1e6}, index=idx)

        patterns = detect_double_top(df, tolerance=0.03)
        # Should detect at least one double top
        assert len(patterns) >= 1
        assert patterns[0].direction == "bearish"
