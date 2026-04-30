"""Tests for volume_profile and anchored_vwap and opening_range."""

from __future__ import annotations

from datetime import time

import numpy as np
import pandas as pd
import pytest

from takata_engine.indicators import (
    OpeningRange,
    VolumeProfile,
    anchored_vwap,
    anchored_vwap_from_prior_day_extreme,
    anchored_vwap_from_session_open,
    opening_range,
    session_volume_profile,
    volume_profile,
)


# ── Fixtures ──

@pytest.fixture
def trending_session() -> pd.DataFrame:
    """A clean uptrending intraday session — 60 5-min bars, 6000 → 6060."""
    idx = pd.date_range("2026-04-29 09:00", periods=60, freq="5min")
    closes = np.linspace(6000, 6060, 60)
    opens = np.r_[closes[0], closes[:-1]]
    highs = np.maximum(opens, closes) + 0.5
    lows = np.minimum(opens, closes) - 0.5
    vols = np.full(60, 1000.0)
    return pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes, "volume": vols},
        index=idx,
    )


@pytest.fixture
def two_day_session() -> pd.DataFrame:
    """Two consecutive days, ~50 bars each, with different price levels."""
    rng1 = pd.date_range("2026-04-28 09:00", periods=50, freq="5min")
    rng2 = pd.date_range("2026-04-29 09:00", periods=50, freq="5min")
    idx = rng1.append(rng2)
    np.random.seed(42)
    # Day 1 — oscillates around 6000
    d1_close = 6000 + np.cumsum(np.random.randn(50)) * 0.5
    # Day 2 — gaps up, oscillates around 6020
    d2_close = 6020 + np.cumsum(np.random.randn(50)) * 0.5
    closes = np.r_[d1_close, d2_close]
    opens = np.r_[closes[0], closes[:-1]]
    highs = np.maximum(opens, closes) + np.random.rand(100)
    lows = np.minimum(opens, closes) - np.random.rand(100)
    vols = np.full(100, 1000.0)
    return pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes, "volume": vols},
        index=idx,
    )


# ── Volume Profile ──

def test_volume_profile_basic(trending_session):
    vp = volume_profile(trending_session, bins=20)
    assert isinstance(vp, VolumeProfile)
    # Trending session: POC should be somewhere in the price range
    assert 6000 <= vp.poc <= 6060
    assert vp.val <= vp.poc <= vp.vah
    # Value area should contain ≥ 70% of total volume
    bin_in_va = (
        (vp.bin_edges[:-1] >= vp.val) & (vp.bin_edges[1:] <= vp.vah)
    )
    va_volume = vp.bin_volumes[bin_in_va].sum()
    assert va_volume / vp.total_volume >= 0.65  # tolerance: bin granularity


def test_volume_profile_concentrated_volume():
    """Tight-range bars near 6000 with extra heavy volume → POC at 6000, narrow VA.

    Setup: 8 wide-range exploration bars [5990, 6010] with low volume, then 4
    narrow-range bars centered at 6000 with high volume. Since the heavy bars
    are tightly clustered, POC must land at their cluster.
    """
    idx = pd.date_range("2026-04-29 09:00", periods=12, freq="5min")
    df = pd.DataFrame({
        "open":   [5990.0]*8 + [6000.0]*4,
        "high":   [6010.0]*8 + [6001.0]*4,
        "low":    [5990.0]*8 + [5999.0]*4,
        "close":  [6005.0]*8 + [6000.0]*4,
        "volume": [100.0]*8  + [10000.0]*4,
    }, index=idx)
    vp = volume_profile(df, bins=20)
    # Heavy volume concentrated in [5999, 6001] → POC must be there.
    assert 5999 <= vp.poc <= 6001, f"expected POC near 6000, got {vp.poc}"
    # VA should hug the heavy cluster, not span the full range.
    assert vp.vah - vp.val < 6.0, f"VA too wide: [{vp.val}, {vp.vah}]"


def test_volume_profile_empty():
    vp = volume_profile(pd.DataFrame(columns=["open", "high", "low", "close", "volume"]))
    assert vp.poc == 0.0
    assert vp.total_volume == 0.0


def test_volume_profile_to_dict(trending_session):
    vp = volume_profile(trending_session)
    d = vp.to_dict()
    assert {"poc", "vah", "val", "total_volume", "value_area_pct", "n_bins"} <= d.keys()


def test_session_volume_profile_per_day(two_day_session):
    profiles = session_volume_profile(two_day_session, bins=15)
    assert len(profiles) == 2
    poc_values = sorted(p.poc for p in profiles.values())
    # Day 2 anchored ~6020, day 1 ~6000 — POCs should reflect that.
    assert poc_values[0] < poc_values[1]
    assert abs(poc_values[0] - 6000) < 5
    assert abs(poc_values[1] - 6020) < 5


# ── Anchored VWAP ──

def test_anchored_vwap_from_position_zero(trending_session):
    avwap = anchored_vwap(trending_session, anchor=0)
    assert avwap.notna().all()
    # On a clean uptrend, AVWAP should be between min and max close.
    last = avwap.iloc[-1]
    assert 6000 <= last <= 6060


def test_anchored_vwap_partial(trending_session):
    """Anchor at row 30 — first 30 rows should be NaN."""
    avwap = anchored_vwap(trending_session, anchor=30)
    assert avwap.iloc[:30].isna().all()
    assert avwap.iloc[30:].notna().all()
    # AVWAP at the anchor bar = typical price of that single bar.
    typical_30 = (trending_session.iloc[30][["high", "low", "close"]].mean())
    assert abs(avwap.iloc[30] - typical_30) < 0.01


def test_anchored_vwap_by_timestamp(trending_session):
    anchor_ts = trending_session.index[20]
    avwap = anchored_vwap(trending_session, anchor=anchor_ts)
    assert avwap.iloc[:20].isna().all()
    assert avwap.iloc[20:].notna().all()


def test_anchored_vwap_session_open_helper(two_day_session):
    avwap = anchored_vwap_from_session_open(two_day_session)
    # Should be NaN for day 1, non-NaN for day 2.
    last_date_mask = two_day_session.index.date == two_day_session.index[-1].date()
    assert avwap[~last_date_mask].isna().all()
    assert avwap[last_date_mask].notna().all()


def test_anchored_vwap_prior_day_extreme(two_day_session):
    """Anchored at prior-day high — values exist from that bar onward."""
    avwap = anchored_vwap_from_prior_day_extreme(two_day_session, extreme="high")
    # First non-NaN should be on day 1 (prior to day 2 — the "current" session).
    first_valid = avwap.first_valid_index()
    assert first_valid is not None
    assert first_valid.date() == two_day_session.index[0].date()  # day 1


def test_anchored_vwap_empty():
    df = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    s = anchored_vwap(df, anchor=0)
    assert len(s) == 0


def test_anchored_vwap_anchor_past_end(trending_session):
    """Anchor in the future → all NaN, no crash."""
    future = trending_session.index[-1] + pd.Timedelta(days=10)
    s = anchored_vwap(trending_session, anchor=future)
    assert s.isna().all()


# ── Opening Range ──

def test_opening_range_locks_after_window(trending_session):
    """30-min OR over 60 5-min bars → locks after bar 6 (30 min)."""
    or_ = opening_range(trending_session, minutes=30)
    assert or_.is_locked
    assert or_.bars_used == 6  # bars 0..5 in [09:00, 09:30)
    assert or_.high < or_.low + 10  # narrow uptrend window


def test_opening_range_state_classification(trending_session):
    or_ = opening_range(trending_session, minutes=30)
    # Trending up: end-of-session price is way above OR-high.
    last_price = trending_session["close"].iloc[-1]
    assert or_.state(last_price) == "above"
    assert or_.state(or_.midpoint) == "inside"
    assert or_.state(or_.low - 1.0) == "below"


def test_opening_range_still_forming():
    """Session shorter than the OR window → not locked."""
    idx = pd.date_range("2026-04-29 09:00", periods=2, freq="5min")
    df = pd.DataFrame({
        "open": [6000.0, 6001.0], "high": [6002.0, 6003.0],
        "low": [5999.0, 6000.0], "close": [6001.0, 6002.0],
        "volume": [1000, 1000],
    }, index=idx)
    or_ = opening_range(df, minutes=30)
    assert not or_.is_locked
    assert or_.bars_used == 2


def test_opening_range_session_start_filter(trending_session):
    """When `session_start` is set, OR begins at the first bar at-or-after."""
    or_ = opening_range(trending_session, minutes=15, session_start=time(9, 30))
    assert or_.is_locked
    # bars 09:30, 09:35, 09:40 used (3 bars in the 15-min window).
    assert or_.bars_used == 3


def test_opening_range_to_dict(trending_session):
    or_ = opening_range(trending_session, minutes=30)
    d = or_.to_dict()
    assert {"high", "low", "open", "is_locked", "width", "midpoint", "bars_used"} <= d.keys()


def test_opening_range_empty():
    or_ = opening_range(pd.DataFrame())
    assert not or_.is_locked
    assert or_.bars_used == 0
