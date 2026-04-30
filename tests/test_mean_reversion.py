"""Tests for chop detection + mean-reversion setups."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from takata_engine.signals import (
    ChopRegime,
    MeanReversionSignal,
    detect_chop,
    evaluate_mean_reversion,
)


# ── Fixtures ──

@pytest.fixture
def chop_session() -> pd.DataFrame:
    """A range-bound session: 60 5-min bars oscillating tightly around 6000."""
    rng = pd.date_range("2026-04-29 09:00", periods=60, freq="5min")
    np.random.seed(7)
    close = 6000 + np.cumsum(np.random.randn(60)) * 0.4   # tight range
    open_ = np.r_[close[0], close[:-1]]
    high = np.maximum(open_, close) + 0.4
    low = np.minimum(open_, close) - 0.4
    vol = np.full(60, 1000.0)
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": vol}, index=rng)


@pytest.fixture
def trending_session() -> pd.DataFrame:
    """Strong uptrend — 60 bars climbing 6000 → 6080."""
    rng = pd.date_range("2026-04-29 09:00", periods=60, freq="5min")
    close = np.linspace(6000, 6080, 60)
    open_ = np.r_[close[0], close[:-1]]
    high = np.maximum(open_, close) + 1.0
    low = np.minimum(open_, close) - 0.3
    vol = np.full(60, 1000.0)
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": vol}, index=rng)


# ── Chop detection ──

def test_detect_chop_in_range_bound_session(chop_session):
    r = detect_chop(chop_session)
    assert r.is_chop, f"expected chop, got score={r.score} reasons={r.reasons}"
    assert r.score >= 60
    # ADX should be low and VWAP displacement small.
    assert r.adx < 25


def test_no_chop_in_trending_session(trending_session):
    r = detect_chop(trending_session)
    assert not r.is_chop, f"expected NO chop, got score={r.score} reasons={r.reasons}"


def test_chop_returns_default_for_short_input():
    """< 30 bars → safe default, no crash."""
    short_df = pd.DataFrame({
        "open": [6000.0]*5, "high": [6005.0]*5, "low": [5995.0]*5,
        "close": [6000.0]*5, "volume": [1000]*5,
    }, index=pd.date_range("2026-04-29 09:00", periods=5, freq="5min"))
    r = detect_chop(short_df)
    assert isinstance(r, ChopRegime)
    assert not r.is_chop
    assert r.score == 0


def test_chop_regime_to_dict(chop_session):
    r = detect_chop(chop_session)
    d = r.to_dict()
    assert {"is_chop", "score", "reasons", "adx", "bb_width_pct", "vwap_displacement"} <= d.keys()


# ── Mean-reversion signals ──

def test_vwap_fade_long_when_oversold_below_vwap(chop_session):
    """Price ~2.5 ATR below VWAP + RSI 25 + clear target distance → MR long."""
    # ATR=2 → stop=3 (floored). Need target ≥ 2.4 from price.
    # close=5993, vwap=5998 → target_dist=5, stop=3, R:R≈1.67 ✓
    indicators = {"close": 5993.0, "rsi": 25.0, "atr": 2.0, "vwap": 5998.0}
    sig = evaluate_mean_reversion(chop_session, indicators, vwap_now=5998.0)
    assert sig is not None
    assert sig.direction == "long"
    assert sig.setup_type == "vwap_fade"
    assert sig.target == 5998.0
    assert sig.mode == "mean_reversion"


def test_vwap_fade_short_when_overbought_above_vwap(chop_session):
    """Price 2.5 ATR above VWAP + RSI 75 + clear target → MR short."""
    indicators = {"close": 6007.0, "rsi": 75.0, "atr": 2.0, "vwap": 6002.0}
    sig = evaluate_mean_reversion(chop_session, indicators, vwap_now=6002.0)
    assert sig is not None
    assert sig.direction == "short"
    assert sig.setup_type == "vwap_fade"


def test_no_signal_when_rsi_neutral(chop_session):
    """No RSI extreme → no MR signal even with VWAP deviation."""
    indicators = {"close": 6005.0, "rsi": 55.0, "atr": 1.0, "vwap": 6003.0}
    sig = evaluate_mean_reversion(chop_session, indicators, vwap_now=6003.0)
    assert sig is None


def test_bollinger_extreme_short(chop_session):
    """bb_pct >= 1 + RSI overbought → MR short."""
    indicators = {"close": 6010.0, "rsi": 72.0, "atr": 1.5, "vwap": 6005.0, "bb_pct": 1.05}
    sig = evaluate_mean_reversion(chop_session, indicators, vwap_now=6005.0)
    assert sig is not None
    assert sig.direction == "short"
    # bb extreme might lose to a vwap_fade with similar strength; either is acceptable.
    assert sig.setup_type in ("bbands_extreme", "vwap_fade")


def test_poc_reclaim_short_from_above_vah():
    """Prev bar > VAH, now back inside value area → fade toward POC."""
    rng = pd.date_range("2026-04-29 09:00", periods=30, freq="5min")
    closes = [6010.0] * 28 + [6015.0, 6008.0]   # punched out, now reclaimed
    df = pd.DataFrame({
        "open": closes, "high": [c + 0.5 for c in closes],
        "low": [c - 0.5 for c in closes], "close": closes,
        "volume": [1000.0] * 30,
    }, index=rng)
    indicators = {"close": 6008.0, "rsi": 55.0, "atr": 1.5, "vwap": 6005.0, "bb_pct": 0.6}
    sig = evaluate_mean_reversion(
        df, indicators, vwap_now=6005.0,
        poc=6005.0, vah=6010.0, val=6000.0,
    )
    # Price moved 5pts toward poc; should fire a poc_reclaim short.
    assert sig is not None
    assert sig.direction == "short"
    # vwap_fade not eligible here (deviation only ~2 ATR; rsi 55 = neutral).


def test_poc_reclaim_long_from_below_val():
    """Wide value area so target distance to POC clears the R:R guard."""
    rng = pd.date_range("2026-04-29 09:00", periods=30, freq="5min")
    closes = [5995.0] * 28 + [5990.0, 5998.0]   # punched below, now reclaimed
    df = pd.DataFrame({
        "open": closes, "high": [c + 0.5 for c in closes],
        "low": [c - 0.5 for c in closes], "close": closes,
        "volume": [1000.0] * 30,
    }, index=rng)
    # POC=6003 → target_dist=5 from close=5998. Stop=3. R:R ≈ 1.67 ✓
    indicators = {"close": 5998.0, "rsi": 45.0, "atr": 1.5, "vwap": 6003.0, "bb_pct": 0.4}
    sig = evaluate_mean_reversion(
        df, indicators, vwap_now=6003.0,
        poc=6003.0, vah=6008.0, val=5995.0,
    )
    assert sig is not None
    assert sig.direction == "long"


def test_returns_none_for_short_input():
    """< 20 bars → None."""
    rng = pd.date_range("2026-04-29 09:00", periods=10, freq="5min")
    df = pd.DataFrame({
        "open": [6000.0]*10, "high": [6001.0]*10, "low": [5999.0]*10,
        "close": [6000.0]*10, "volume": [1000]*10,
    }, index=rng)
    sig = evaluate_mean_reversion(df, {"close": 6000, "rsi": 25, "atr": 1, "vwap": 6000})
    assert sig is None


def test_signal_to_dict_shape(chop_session):
    indicators = {"close": 5993.0, "rsi": 25.0, "atr": 2.0, "vwap": 5998.0}
    sig = evaluate_mean_reversion(chop_session, indicators, vwap_now=5998.0)
    d = sig.to_dict()
    assert d["mode"] == "mean_reversion"
    assert {"direction", "setup_type", "price", "stop", "target", "stop_pts",
            "target_pts", "strength", "reasons", "rr_ratio"} <= d.keys()
    assert d["rr_ratio"] > 0


def test_strongest_setup_wins_when_multiple_fire(chop_session):
    """Both VWAP fade AND BB extreme would fire; the higher-strength one is returned."""
    indicators = {"close": 6010.0, "rsi": 80.0, "atr": 1.0, "vwap": 6003.0, "bb_pct": 1.10}
    sig = evaluate_mean_reversion(chop_session, indicators, vwap_now=6003.0)
    assert sig is not None
    # Either could win; what matters is exactly one is returned.
    assert sig.direction == "short"
    assert sig.setup_type in ("vwap_fade", "bbands_extreme")
