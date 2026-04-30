"""Tests for the Phase 2/3/4 ML feature extraction in LiveSignalLogger.

Locks in the feature schema so the live ML filter and the retrain endpoint
stay in lockstep — silent feature_names mismatches hide ML failures.
"""

from __future__ import annotations

import pytest

from takata_engine.ml.signal_logger import LiveSignalLogger


# ── Fixtures ──

@pytest.fixture
def base_indicators():
    return {
        "ema_fast": 6010, "ema_slow": 6005, "rsi": 65, "adx": 22,
        "macd_hist": 1.2, "atr": 5.0, "vwap": 6008, "close": 6012,
    }


@pytest.fixture
def trend_signal():
    return {
        "direction": "long", "price": 6012, "stop": 6005, "target": 6022,
        "strength": 0.75, "reasons": ["ema_bullish", "above_vwap", "rsi_bullish"],
        "mode": "trend",
    }


@pytest.fixture
def mr_signal():
    return {
        "direction": "short", "price": 6020, "stop": 6024, "target": 6010,
        "strength": 0.65, "reasons": ["vwap_dev_2.0atr", "rsi_overbought_75"],
        "mode": "mean_reversion", "setup_type": "vwap_fade",
    }


@pytest.fixture
def day_trade_ta():
    return {
        "volume_profile": {"poc": 6010, "vah": 6020, "val": 6000, "n_bins": 24},
        "anchored_vwap": {"session_open": 6008, "prior_high": 6018, "prior_low": 5995},
        "opening_range": {"high": 6015, "low": 6005, "open": 6010,
                          "is_locked": True, "state": "above", "width": 10},
    }


@pytest.fixture
def confluence():
    return {
        "score": 75, "label": "strong",
        "agreed": [{"peer": "VIX", "delta_pct": +1.2, "weight": 2},
                   {"peer": "ESFUT", "delta_pct": -0.5, "weight": -2}],
        "disagreed": [{"peer": "NQFUT", "delta_pct": +0.3, "weight": -2}],
        "missing": [],
    }


@pytest.fixture
def chop_regime():
    return {"is_chop": True, "score": 75, "adx": 14.5,
            "bb_width_pct": 22, "vwap_displacement": 0.6,
            "reasons": ["adx_low", "vwap_pinned"]}


# ── Feature columns canonical list ──

def test_feature_columns_includes_phase_2_3_4():
    cols = LiveSignalLogger.feature_columns()
    # Phase 2
    for name in ("poc_distance_atr", "vah_distance_atr", "val_distance_atr",
                 "va_position", "avwap_session_distance_atr",
                 "or_state_above", "or_state_below", "or_state_inside",
                 "or_width_atr", "or_locked"):
        assert name in cols, f"missing Phase 2 feature: {name}"
    # Phase 3
    for name in ("cross_asset_score", "cross_asset_agreed_count", "cross_asset_disagreed_count"):
        assert name in cols, f"missing Phase 3 feature: {name}"
    # Phase 4
    for name in ("chop_score", "chop_adx", "chop_bb_width_pct", "chop_vwap_displacement",
                 "is_mean_reversion",
                 "mr_setup_vwap_fade", "mr_setup_bbands_extreme", "mr_setup_poc_reclaim"):
        assert name in cols, f"missing Phase 4 feature: {name}"


def test_feature_columns_includes_phase_6():
    cols = LiveSignalLogger.feature_columns()
    for name in ("absorption_active", "absorption_dir",
                 "absorption_volume_z", "absorption_range_z",
                 "agg_streak_active", "agg_streak_dir",
                 "agg_streak_length", "agg_streak_cumulative",
                 "agg_exhaustion_active"):
        assert name in cols, f"missing Phase 6 feature: {name}"


def test_extract_phase6_absorption_bull(base_indicators, trend_signal):
    absorption = {"active": True, "direction": "bull", "volume_z": 2.5, "range_z": -1.2,
                  "score_adjustment": 0.10}
    feats = LiveSignalLogger._extract_ml_features(
        trend_signal, base_indicators, absorption=absorption,
    )
    assert feats["absorption_active"] == 1.0
    assert feats["absorption_dir"] == 1.0
    assert feats["absorption_volume_z"] == 2.5
    assert feats["absorption_range_z"] == -1.2


def test_extract_phase6_streak_bear_with_exhaustion(base_indicators, trend_signal):
    streak = {"active": True, "direction": "bear", "length": 6,
              "cumulative_agg": -1500, "exhaustion_active": True,
              "score_adjustment": 0.12}
    feats = LiveSignalLogger._extract_ml_features(
        trend_signal, base_indicators, agg_streak=streak,
    )
    assert feats["agg_streak_active"] == 1.0
    assert feats["agg_streak_dir"] == -1.0
    assert feats["agg_streak_length"] == 6
    assert feats["agg_streak_cumulative"] == -1500
    assert feats["agg_exhaustion_active"] == 1.0


def test_extract_phase6_defaults_when_absent(base_indicators, trend_signal):
    feats = LiveSignalLogger._extract_ml_features(trend_signal, base_indicators)
    assert feats["absorption_active"] == 0.0
    assert feats["absorption_dir"] == 0.0
    assert feats["agg_streak_active"] == 0.0
    assert feats["agg_streak_dir"] == 0.0
    assert feats["agg_exhaustion_active"] == 0.0


def test_feature_columns_no_duplicates():
    cols = LiveSignalLogger.feature_columns()
    assert len(cols) == len(set(cols))


# ── Feature extraction ──

def test_extract_with_no_phase234_context(base_indicators, trend_signal):
    """When new context is None, Phase 2/3/4 features default to neutral values."""
    feats = LiveSignalLogger._extract_ml_features(trend_signal, base_indicators)
    # Distance features default to 0 (neutral / no anchor).
    assert feats["poc_distance_atr"] == 0
    assert feats["vah_distance_atr"] == 0
    assert feats["avwap_session_distance_atr"] == 0
    # OR state one-hots all 0.
    assert feats["or_state_above"] == 0
    assert feats["or_state_below"] == 0
    assert feats["or_state_inside"] == 0
    # Cross-asset score defaults to 50 (neutral, not 0 — important for retraining).
    assert feats["cross_asset_score"] == 50
    # Chop score defaults to 0 (trending).
    assert feats["chop_score"] == 0
    # Trend signal isn't MR.
    assert feats["is_mean_reversion"] == 0
    assert feats["mr_setup_vwap_fade"] == 0


def test_extract_phase2_distances_normalized_by_atr(base_indicators, trend_signal, day_trade_ta):
    """POC distance should be (close-poc)/atr."""
    feats = LiveSignalLogger._extract_ml_features(trend_signal, base_indicators, day_trade_ta=day_trade_ta)
    # close=6012, poc=6010, atr=5 → distance = 0.4
    assert abs(feats["poc_distance_atr"] - 0.4) < 1e-6
    # close=6012 inside [6000, 6020] → va_position = 0
    assert feats["va_position"] == 0
    # OR state="above" (close 6012 > 6005? wait, OR high=6015)
    # Actually we set state="above" in the fixture, so the one-hot reflects that.
    assert feats["or_state_above"] == 1
    assert feats["or_locked"] == 1


def test_extract_va_position_above(base_indicators, trend_signal):
    """Price above VAH → va_position=+1."""
    above_va_ta = {"volume_profile": {"poc": 6000, "vah": 6005, "val": 5995}}
    base_indicators["close"] = 6010   # above VAH=6005
    feats = LiveSignalLogger._extract_ml_features(trend_signal, base_indicators, day_trade_ta=above_va_ta)
    assert feats["va_position"] == 1.0


def test_extract_va_position_below(base_indicators, trend_signal):
    below_va_ta = {"volume_profile": {"poc": 6010, "vah": 6015, "val": 6005}}
    base_indicators["close"] = 6000   # below VAL=6005
    feats = LiveSignalLogger._extract_ml_features(trend_signal, base_indicators, day_trade_ta=below_va_ta)
    assert feats["va_position"] == -1.0


def test_extract_phase3_confluence(base_indicators, trend_signal, confluence):
    feats = LiveSignalLogger._extract_ml_features(trend_signal, base_indicators, confluence=confluence)
    assert feats["cross_asset_score"] == 75
    assert feats["cross_asset_agreed_count"] == 2
    assert feats["cross_asset_disagreed_count"] == 1


def test_extract_phase4_mr_signal(base_indicators, mr_signal, chop_regime):
    feats = LiveSignalLogger._extract_ml_features(mr_signal, base_indicators, chop_regime=chop_regime)
    assert feats["is_mean_reversion"] == 1
    assert feats["mr_setup_vwap_fade"] == 1
    assert feats["mr_setup_bbands_extreme"] == 0
    assert feats["mr_setup_poc_reclaim"] == 0
    assert feats["chop_score"] == 75
    assert feats["chop_adx"] == 14.5


def test_extract_handles_none_atr_gracefully(trend_signal):
    """ATR=0 must not produce inf/NaN — divisor is clamped to 1.0."""
    indicators = {"close": 6000, "atr": 0, "rsi": 50}
    ta = {"volume_profile": {"poc": 6010, "vah": 6020, "val": 6000}}
    feats = LiveSignalLogger._extract_ml_features(trend_signal, indicators, day_trade_ta=ta)
    # (close-poc)/1.0 = -10 — not inf or NaN.
    assert feats["poc_distance_atr"] == -10
    assert feats["poc_distance_atr"] == feats["poc_distance_atr"]  # not NaN


def test_extract_complete_feature_set_matches_columns(
    base_indicators, mr_signal, day_trade_ta, confluence, chop_regime
):
    """The dict returned by _extract_ml_features must include every column
    from feature_columns() — this is the guarantee the retrain pipeline relies
    on to avoid silent missing-feature errors."""
    feats = LiveSignalLogger._extract_ml_features(
        mr_signal, base_indicators, day_trade_ta, confluence, chop_regime
    )
    cols = LiveSignalLogger.feature_columns()
    for c in cols:
        assert c in feats, f"feature_columns() includes {c} but _extract_ml_features() did not produce it"
