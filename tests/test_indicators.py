"""Unit tests for core indicators."""

import numpy as np
import pandas as pd
import pytest

from takata_core.indicators import ema, macd, rsi, bollinger, adx, atr, vwap


@pytest.fixture
def sample_ohlcv():
    """50 bars of deterministic OHLCV data."""
    rng = np.random.default_rng(123)
    n = 50
    close = 100 + np.cumsum(rng.standard_normal(n) * 0.5)
    high = close + np.abs(rng.standard_normal(n)) * 0.3
    low = close - np.abs(rng.standard_normal(n)) * 0.3
    open_ = np.roll(close, 1)
    open_[0] = 100
    volume = (1000 + rng.integers(-200, 200, n)).astype(float)

    idx = pd.date_range("2025-01-02 09:30", periods=n, freq="5min", tz="America/New_York")
    return pd.DataFrame({
        "open": open_, "high": high, "low": low, "close": close, "volume": volume,
    }, index=idx)


class TestEMA:
    def test_length(self, sample_ohlcv):
        result = ema(sample_ohlcv["close"], period=9)
        assert len(result) == len(sample_ohlcv)

    def test_first_value_equals_input(self, sample_ohlcv):
        """With adjust=False, the first EMA value equals the first input."""
        result = ema(sample_ohlcv["close"], period=9)
        assert result.iloc[0] == pytest.approx(sample_ohlcv["close"].iloc[0])

    def test_smoothing(self, sample_ohlcv):
        """EMA should be smoother than raw close."""
        result = ema(sample_ohlcv["close"], period=20)
        assert result.std() < sample_ohlcv["close"].std()


class TestMACD:
    def test_columns(self, sample_ohlcv):
        result = macd(sample_ohlcv["close"])
        assert list(result.columns) == ["macd", "signal", "histogram"]

    def test_histogram_is_macd_minus_signal(self, sample_ohlcv):
        result = macd(sample_ohlcv["close"])
        expected_hist = result["macd"] - result["signal"]
        pd.testing.assert_series_equal(result["histogram"], expected_hist, check_names=False)


class TestRSI:
    def test_range(self, sample_ohlcv):
        result = rsi(sample_ohlcv["close"], period=14)
        valid = result.dropna()
        assert (valid >= 0).all()
        assert (valid <= 100).all()

    def test_warmup_nans(self, sample_ohlcv):
        result = rsi(sample_ohlcv["close"], period=14)
        # First value should be NaN (no diff for first bar)
        assert pd.isna(result.iloc[0])


class TestBollinger:
    def test_columns(self, sample_ohlcv):
        result = bollinger(sample_ohlcv["close"])
        assert list(result.columns) == ["lower", "middle", "upper"]

    def test_ordering(self, sample_ohlcv):
        result = bollinger(sample_ohlcv["close"]).dropna()
        assert (result["lower"] <= result["middle"]).all()
        assert (result["middle"] <= result["upper"]).all()

    def test_middle_is_sma(self, sample_ohlcv):
        result = bollinger(sample_ohlcv["close"], period=20)
        expected_sma = sample_ohlcv["close"].rolling(20).mean()
        pd.testing.assert_series_equal(
            result["middle"], expected_sma, check_names=False
        )


class TestADX:
    def test_columns(self, sample_ohlcv):
        result = adx(sample_ohlcv)
        assert list(result.columns) == ["adx", "plus_di", "minus_di"]

    def test_positive(self, sample_ohlcv):
        result = adx(sample_ohlcv).dropna()
        assert (result["adx"] >= 0).all()


class TestATR:
    def test_positive(self, sample_ohlcv):
        result = atr(sample_ohlcv)
        valid = result.dropna()
        assert (valid >= 0).all()

    def test_length(self, sample_ohlcv):
        result = atr(sample_ohlcv)
        assert len(result) == len(sample_ohlcv)


class TestVWAP:
    def test_within_price_range(self, sample_ohlcv):
        result = vwap(sample_ohlcv).dropna()
        assert result.min() >= sample_ohlcv["low"].min() - 1
        assert result.max() <= sample_ohlcv["high"].max() + 1

    def test_length(self, sample_ohlcv):
        result = vwap(sample_ohlcv)
        assert len(result) == len(sample_ohlcv)
