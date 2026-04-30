"""Tests for absorption + aggression-streak trackers (Phase 6 tape depth)."""

from __future__ import annotations

import time

import pytest

from takata_engine.pricing.absorption import (
    AbsorptionTracker,
    get_absorption_tracker,
)
from takata_engine.pricing.aggression_streak import (
    AggressionStreakTracker,
    get_aggression_streak_tracker,
    MIN_STREAK_LENGTH,
)


# ── Absorption ──

def test_absorption_silent_until_warmup():
    """< 20 bars → no absorption signal even on heavy/tight bar."""
    t = AbsorptionTracker()
    for i in range(5):
        t.update(price=6000 + i * 0.1, high=6005, low=5995, volume=100, agg_balance=0)
    sig = t.signal()
    assert not sig["active"]


def test_bull_absorption_fires_on_heavy_negative_agg_tight_range():
    """Sellers press hard but range stays tight → bullish absorption."""
    t = AbsorptionTracker()
    # Build 20 baseline bars with normal volume + range.
    base_ts = 1.0
    for i in range(20):
        t.update(
            price=6000.0, high=6010.0, low=5990.0, volume=1000.0,
            agg_balance=0, timestamp=base_ts + i,
        )
    # Trigger bar: 4x volume, half range, strong negative agg.
    t.update(
        price=6000.0, high=6002.0, low=5998.0, volume=4000.0,
        agg_balance=-3000.0, timestamp=base_ts + 30,
    )
    sig = t.signal()
    assert sig["active"]
    assert sig["direction"] == "bull"
    assert sig["score_adjustment"] > 0
    assert any("absorption_bull" in r for r in sig["reasons"])


def test_bear_absorption_on_heavy_positive_agg_tight_range():
    t = AbsorptionTracker()
    base_ts = 1.0
    for i in range(20):
        t.update(price=6000.0, high=6010.0, low=5990.0, volume=1000.0,
                 agg_balance=0, timestamp=base_ts + i)
    t.update(
        price=6000.0, high=6002.0, low=5998.0, volume=4000.0,
        agg_balance=+3000.0, timestamp=base_ts + 30,
    )
    sig = t.signal()
    assert sig["active"]
    assert sig["direction"] == "bear"


def test_no_absorption_when_range_normal():
    """Heavy volume + range as wide as the baseline → no absorption."""
    t = AbsorptionTracker()
    base_ts = 1.0
    for i in range(20):
        t.update(price=6000.0, high=6010.0, low=5990.0, volume=1000.0,
                 agg_balance=0, timestamp=base_ts + i)
    # Baseline range = 20. Trigger bar range = 30 (1.5× wider) so range_z is positive.
    t.update(
        price=6005.0, high=6020.0, low=5990.0, volume=4000.0,
        agg_balance=-3000.0, timestamp=base_ts + 30,
    )
    sig = t.signal()
    assert not sig["active"]


def test_absorption_decays_after_10_minutes():
    """Latest absorption should age out at the TTL."""
    t = AbsorptionTracker()
    base_ts = 1000.0
    for i in range(20):
        t.update(price=6000.0, high=6010.0, low=5990.0, volume=1000.0,
                 agg_balance=0, timestamp=base_ts + i)
    t.update(
        price=6000.0, high=6002.0, low=5998.0, volume=4000.0,
        agg_balance=-3000.0, timestamp=base_ts + 30,
    )
    assert t.signal()["active"]
    # Now add bars 11+ minutes later — old absorption should fall out.
    for i in range(5):
        t.update(price=6000.0, high=6010.0, low=5990.0, volume=1000.0,
                 agg_balance=0, timestamp=base_ts + 30 + 700 + i)
    assert not t.signal()["active"]


def test_absorption_singleton():
    a = get_absorption_tracker()
    b = get_absorption_tracker()
    assert a is b


# ── Aggression streaks ──

def test_streak_silent_below_min_length():
    t = AggressionStreakTracker()
    for i in range(MIN_STREAK_LENGTH - 1):
        t.update(price=6000 + i, agg_balance=200, volume=1000)
    sig = t.signal()
    assert not sig["active"]


def test_bull_streak_fires_at_min_length():
    t = AggressionStreakTracker()
    for i in range(MIN_STREAK_LENGTH):
        t.update(price=6000 + i, agg_balance=200, volume=1000)
    sig = t.signal()
    assert sig["active"]
    assert sig["direction"] == "bull"
    assert sig["length"] == MIN_STREAK_LENGTH
    assert sig["score_adjustment"] > 0


def test_bear_streak_fires():
    t = AggressionStreakTracker()
    for i in range(MIN_STREAK_LENGTH):
        t.update(price=6000 - i, agg_balance=-200, volume=1000)
    sig = t.signal()
    assert sig["active"]
    assert sig["direction"] == "bear"


def test_streak_score_caps_at_15pct():
    """Even a 100-bar streak shouldn't push the confluence score by more than 0.15."""
    t = AggressionStreakTracker()
    for i in range(20):
        t.update(price=6000 + i, agg_balance=200, volume=1000)
    sig = t.signal()
    assert sig["score_adjustment"] <= 0.15


def test_neutral_bar_does_not_break_streak():
    """A bar with agg_balance below MIN_BAR_AGG is neutral; doesn't extend or
    break the streak."""
    t = AggressionStreakTracker()
    for i in range(5):
        t.update(price=6000 + i, agg_balance=200, volume=1000)
    cur_len = t.signal()["length"]
    # Now a neutral bar (agg_balance=10).
    t.update(price=6005, agg_balance=10, volume=1000)
    assert t.signal()["length"] == cur_len
    # And another bull bar — should extend.
    t.update(price=6006, agg_balance=200, volume=1000)
    assert t.signal()["length"] == cur_len + 1


def test_reversal_breaks_streak_and_starts_new_one():
    t = AggressionStreakTracker()
    for i in range(5):
        t.update(price=6000 + i, agg_balance=200, volume=1000)
    # Now reverse with a single bear bar — streak should reset.
    t.update(price=6004, agg_balance=-300, volume=1100)
    sig = t.signal()
    assert sig["direction"] == "bear"
    assert sig["length"] == 1
    assert not sig["active"]   # not yet ≥ MIN_STREAK_LENGTH


def test_exhaustion_fires_on_heavy_counter_bar():
    """5-bar bull streak followed by a counter bar with notably bigger volume
    → exhaustion flag."""
    t = AggressionStreakTracker()
    for i in range(5):
        t.update(price=6000 + i, agg_balance=200, volume=1000)
    # Counter bar with 1.5x volume.
    t.update(price=6004, agg_balance=-300, volume=1600)
    sig = t.signal()
    assert sig["exhaustion_active"]
    assert sig["exhaustion_direction"] == "bull"


def test_exhaustion_does_not_fire_on_quiet_reversal():
    """5-bar streak then counter bar with similar volume → no exhaustion (just a normal turn)."""
    t = AggressionStreakTracker()
    for i in range(5):
        t.update(price=6000 + i, agg_balance=200, volume=1000)
    t.update(price=6004, agg_balance=-300, volume=1000)   # same volume, no thrust
    sig = t.signal()
    assert not sig["exhaustion_active"]


def test_streak_singleton():
    a = get_aggression_streak_tracker()
    b = get_aggression_streak_tracker()
    assert a is b
