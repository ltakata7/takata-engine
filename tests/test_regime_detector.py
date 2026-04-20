"""Unit tests for the regime detector pipeline."""

import numpy as np
import pandas as pd
import pytest

from takata_engine.regime import RegimeDetector, RegimeParams
from takata_engine.config import load_config


def _make_ohlcv(n: int, base: float, volatility: float, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic OHLCV data with controlled volatility."""
    rng = np.random.default_rng(seed)
    returns = rng.normal(0, volatility, n)
    close = base * np.exp(np.cumsum(returns))
    high = close * (1 + np.abs(rng.normal(0, volatility * 0.3, n)))
    low = close * (1 - np.abs(rng.normal(0, volatility * 0.3, n)))
    open_ = np.roll(close, 1)
    open_[0] = base
    volume = (1000 + rng.integers(-200, 200, n)).astype(float)

    idx = pd.date_range("2025-01-02 09:30", periods=n, freq="5min", tz="America/New_York")
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )


def _make_regime_shift_data() -> pd.DataFrame:
    """Create data with a clear regime shift: low vol → high vol."""
    low_vol = _make_ohlcv(200, base=100.0, volatility=0.001, seed=10)
    high_vol = _make_ohlcv(200, base=low_vol["close"].iloc[-1], volatility=0.01, seed=20)
    # Shift high_vol index to continue after low_vol
    high_vol.index = pd.date_range(
        low_vol.index[-1] + pd.Timedelta("5min"), periods=200, freq="5min", tz="America/New_York"
    )
    return pd.concat([low_vol, high_vol])


class TestRegimeDetector:
    def test_fit_and_detect(self):
        df = _make_regime_shift_data()
        detector = RegimeDetector(k=2, window_size=20, stride=5, random_state=42,
                                  labels=["calm", "volatile"])
        detector.fit(df)
        regime = detector.detect(df)
        assert regime in ("calm", "volatile")

    def test_label_history_shape(self):
        df = _make_regime_shift_data()
        detector = RegimeDetector(k=2, window_size=20, stride=5, random_state=42,
                                  labels=["calm", "volatile"])
        detector.fit(df)
        history = detector.label_history(df)
        # Length = len(df) - 1 (log returns drop first bar)
        assert len(history) == len(df) - 1
        assert "regime_label" in history.columns

    def test_detect_before_fit_raises(self):
        df = _make_regime_shift_data()
        detector = RegimeDetector(k=2, window_size=20, stride=5)
        with pytest.raises(ValueError, match="not been fit"):
            detector.detect(df)

    def test_quality_scores(self):
        df = _make_regime_shift_data()
        detector = RegimeDetector(k=2, window_size=20, stride=5, random_state=42,
                                  labels=["calm", "volatile"])
        detector.fit(df)
        scores = detector.quality_scores(df)
        assert "within_mmd" in scores
        assert "between_mmd" in scores

    def test_regime_shift_detected(self):
        """The detector should assign different regimes to low-vol and high-vol sections."""
        df = _make_regime_shift_data()
        detector = RegimeDetector(k=2, window_size=20, stride=5, random_state=42,
                                  labels=["calm", "volatile"])
        detector.fit(df)
        history = detector.label_history(df)

        # First 150 bars (well into calm period) vs last 150 bars (high vol)
        first_half = history.iloc[:150]["regime_label"]
        second_half = history.iloc[-150:]["regime_label"]

        # Most common label should differ between halves
        first_mode = first_half.mode().iloc[0]
        second_mode = second_half.mode().iloc[0]
        assert first_mode != second_mode, (
            f"Expected different regimes, got '{first_mode}' in both halves"
        )

    def test_from_config(self):
        cfg = load_config("MES")
        detector = RegimeDetector.from_config(cfg["regime"])
        assert detector.k == 3
        assert detector.window_size == 30


class TestRegimeParams:
    def test_get_params_applies_overrides(self):
        cfg = load_config("MES")
        params = RegimeParams(cfg)
        adapted = params.get_params("crisis")
        assert adapted["indicators"]["rsi"]["period"] == 7
        assert adapted["risk"]["stop_atr_mult"] == 0.5

    def test_get_params_unknown_regime_returns_base(self):
        cfg = load_config("MES")
        params = RegimeParams(cfg)
        adapted = params.get_params("nonexistent_regime")
        assert adapted["indicators"]["rsi"]["period"] == 14  # base config

    def test_available_regimes(self):
        cfg = load_config("MES")
        params = RegimeParams(cfg)
        assert "low_vol_trending" in params.available_regimes
        assert "high_vol_choppy" in params.available_regimes
        assert "crisis" in params.available_regimes
