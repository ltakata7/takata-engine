"""Tests for the Hurst / skew / kurtosis statistical indicators.

These features are added to the live scanner's signal snapshot
(takata-trading) and consumed by the ML training pipeline. Each test
captures a property the production code relies on — Hurst stays in
[0,1], degenerate inputs return safe defaults instead of NaN, etc.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from takata_engine.indicators.statistical import (
    hurst_exponent,
    latest_statistical_features,
    rolling_kurtosis,
    rolling_skewness,
)


# ── Hurst exponent ───────────────────────────────────────────────────

class TestHurstExponent:
    def test_random_walk_is_near_half(self):
        # Cumulative-sum gaussian noise = price as a random walk.
        # By construction H should sit close to 0.5.
        rng = np.random.default_rng(42)
        prices = pd.Series(100 + np.cumsum(rng.standard_normal(500)))
        h = hurst_exponent(prices, max_lag=50)
        assert 0.3 <= h <= 0.7, f"random walk H={h}, expected ~0.5"

    def test_strong_trend_is_above_half(self):
        # Pure deterministic trend gives near-perfect persistence.
        prices = pd.Series(100 + np.linspace(0, 50, 300))
        h = hurst_exponent(prices, max_lag=50)
        assert h >= 0.7, f"trending series H={h}, expected ≥ 0.7"

    def test_anti_persistent_is_below_half(self):
        # Mean-reverting series (alternating sign returns) should drop
        # H below 0.5 — that's the regime where momentum signals fail.
        rng = np.random.default_rng(7)
        # AR(1) with negative coefficient = anti-persistent.
        n = 400
        rets = np.zeros(n)
        prev = 0.0
        for i in range(n):
            rets[i] = -0.7 * prev + rng.standard_normal() * 0.3
            prev = rets[i]
        prices = pd.Series(100 + np.cumsum(rets))
        h = hurst_exponent(prices, max_lag=50)
        assert h < 0.5, f"anti-persistent H={h}, expected < 0.5"

    def test_short_input_returns_default(self):
        # Breakage: an off-by-one on the min-length check would let
        # a fresh-start scanner pass a 2-bar series and get NaN, which
        # the ML logger would then write to disk as JSON "NaN" (which
        # json.loads chokes on later).
        assert hurst_exponent(pd.Series([100, 101, 102])) == 0.5
        assert hurst_exponent(pd.Series([], dtype=float)) == 0.5
        assert hurst_exponent(None) == 0.5

    def test_clamped_to_unit_interval(self):
        # Some pathological inputs can drive the slope outside [0,1];
        # the function must clamp so the ML logger never sees H=2.7.
        rng = np.random.default_rng(0)
        prices = pd.Series(100 + np.cumsum(rng.standard_normal(200)))
        h = hurst_exponent(prices)
        assert 0.0 <= h <= 1.0


# ── Rolling skew / kurt ──────────────────────────────────────────────

class TestRollingMoments:
    def test_skew_left_skewed_negative(self):
        # Returns dominated by a few large drops have negative skew —
        # exactly the situation we want to flag.
        s = pd.Series([0.001] * 50 + [-0.05])
        sk = rolling_skewness(s, window=30).iloc[-1]
        assert sk < -0.5, f"left-skewed should be < -0.5, got {sk}"

    def test_kurt_high_for_jumps(self):
        # A gaussian return stream with two big outliers gives high
        # excess kurtosis. Watch the *trailing* window only, since
        # earlier windows don't include the outliers.
        rng = np.random.default_rng(11)
        normal = list(rng.standard_normal(100) * 0.001)
        spiky = normal[-15:] + [0.10, -0.08] + normal[:15]  # 32 values
        ku = rolling_kurtosis(pd.Series(spiky), window=30).iloc[-1]
        assert ku > 1.0, f"jumpy returns kurt={ku}, expected > 1"

    def test_empty_series_returns_empty(self):
        assert rolling_skewness(pd.Series(dtype=float), 30).empty
        assert rolling_kurtosis(pd.Series(dtype=float), 30).empty


# ── Aggregator ───────────────────────────────────────────────────────

class TestLatestStatisticalFeatures:
    def test_returns_three_named_floats(self):
        # The live scanner calls this on each cycle and dict-merges
        # the result into its indicators snapshot. Schema drift here
        # = a KeyError in the signal_logger when it tries to record
        # the features.
        rng = np.random.default_rng(1)
        prices = pd.Series(100 + np.cumsum(rng.standard_normal(200)))
        d = latest_statistical_features(prices)
        assert set(d.keys()) == {"hurst_50", "return_skew_30", "return_kurt_30"}
        for v in d.values():
            assert isinstance(v, float)
            assert np.isfinite(v)

    def test_safe_defaults_on_short_history(self):
        # Cold-boot / first-bar case must not raise.
        d = latest_statistical_features(pd.Series([100.0]))
        assert d["hurst_50"] == 0.5
        assert d["return_skew_30"] == 0.0
        assert d["return_kurt_30"] == 0.0

    def test_handles_none(self):
        d = latest_statistical_features(None)
        assert d["hurst_50"] == 0.5
        assert d["return_skew_30"] == 0.0
        assert d["return_kurt_30"] == 0.0
