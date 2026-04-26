"""Feature store — converts signals and trades into ML training data."""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd

from takata_engine.signals.position import Position
from takata_engine.signals.signal_generator import Signal


def signal_to_features(signal: Signal) -> Dict[str, float]:
    """Extract ML features from a Signal object."""
    ind = signal.indicators
    return {
        "ema_fast": ind.get("ema_fast", 0),
        "ema_slow": ind.get("ema_slow", 0),
        "ema_spread": ind.get("ema_fast", 0) - ind.get("ema_slow", 0),
        "rsi": ind.get("rsi", 50),
        "macd": ind.get("macd", 0),
        "macd_signal": ind.get("macd_signal", 0),
        "macd_hist": ind.get("macd_hist", 0),
        "adx": ind.get("adx", 0),
        "plus_di": ind.get("plus_di", 0),
        "minus_di": ind.get("minus_di", 0),
        "atr": ind.get("atr", 0),
        "bb_upper": ind.get("bb_upper", 0),
        "bb_middle": ind.get("bb_middle", 0),
        "bb_lower": ind.get("bb_lower", 0),
        "bb_position": (ind.get("close", 0) - ind.get("bb_lower", 0)) / max(ind.get("bb_upper", 1) - ind.get("bb_lower", 1), 0.001),
        "vwap": ind.get("vwap", 0),
        "price_vs_vwap": ind.get("close", 0) - ind.get("vwap", 0),
        "close": ind.get("close", 0),
        "strength": signal.strength,
        "direction": 1.0 if signal.direction == "long" else -1.0,
        "n_reasons": len(signal.reasons),
        # Microstructure features (Phase A)
        "orb_distance_15": ind.get("orb_distance_15", 0),
        "orb_range_15": ind.get("orb_range_15", 0),
        "orb_broke_dir": ind.get("orb_broke_dir", 0),
        "vwap_sigma_distance": ind.get("vwap_sigma_distance", 0),
        "cum_delta": ind.get("cum_delta", 0),
        "delta_change": ind.get("delta_change", 0),
        # Microstructure features (Phase B)
        "rv_iv_ratio_5m": ind.get("rv_iv_ratio_5m", 0),
        "spread_percentile": ind.get("spread_percentile", 0),
        "spread_z_score": ind.get("spread_z_score", 0),
        "vol_burst_ratio": ind.get("vol_burst_ratio", 0),
        # Microstructure features (Phase C)
        "xcorr_ind": ind.get("xcorr_ind", 0),
        "xcorr_es": ind.get("xcorr_es", 0),
        "leadlag_best_corr": ind.get("leadlag_best_corr", 0),
        "leadlag_best_lag": ind.get("leadlag_best_lag", 0),
        "risk_regime_score": ind.get("risk_regime_score", 0),
        # Statistical / multifractal features (Phase D)
        "hurst_50": ind.get("hurst_50", 0.5),
        "return_skew_30": ind.get("return_skew_30", 0.0),
        "return_kurt_30": ind.get("return_kurt_30", 0.0),
    }


def build_training_data(
    signals: List[Signal],
    positions: List[Position],
) -> pd.DataFrame:
    """Build a labeled dataset from signals and their trade outcomes.

    Each row = one signal, features from indicators, label = whether the trade was profitable.
    """
    # Match signals to positions by timestamp proximity
    pos_by_time = {}
    for p in positions:
        pos_by_time[p.entry_time] = p

    rows = []
    for sig in signals:
        features = signal_to_features(sig)

        # Find matching position
        pos = pos_by_time.get(sig.timestamp)
        if pos and pos.status == "closed":
            features["pnl"] = pos.pnl
            features["profitable"] = 1.0 if pos.pnl > 0 else 0.0
            features["exit_reason"] = pos.exit_reason
            features["regime"] = sig.regime
            rows.append(features)

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows)


FEATURE_COLUMNS = [
    # Classic indicators
    "ema_spread", "rsi", "macd", "macd_hist", "adx",
    "plus_di", "minus_di", "atr", "bb_position",
    "price_vs_vwap", "strength", "direction", "n_reasons",
    # Microstructure features (Phase A — position/flow)
    "orb_distance_15",       # distance from nearest 15m ORB boundary (pts)
    "orb_range_15",          # 15m ORB range width (pts)
    "orb_broke_dir",         # 1=broke above, -1=broke below, 0=no break
    "vwap_sigma_distance",   # sigma distance from VWAP (−2 to +2 typical)
    "cum_delta",             # cumulative delta (buy−sell flow)
    "delta_change",          # delta change over lookback window
    # Microstructure features (Phase B — liquidity/activity)
    "rv_iv_ratio_5m",        # realized vol / implied vol (5m window)
    "spread_percentile",     # spread percentile rank (0−100)
    "spread_z_score",        # spread z-score vs session history
    "vol_burst_ratio",       # current volume / median volume
    # Microstructure features (Phase C — inter-market)
    "xcorr_ind",             # WDO–IND correlation (live rolling)
    "xcorr_es",              # WDO–ES correlation (live rolling)
    "leadlag_best_corr",     # best leader symbol correlation
    "leadlag_best_lag",      # best leader lag in ticks
    "risk_regime_score",     # composite risk-on/off score (−100 to +100)
    # Statistical / multifractal features (Phase D — Bloch 2016 §2.1.4–5)
    "hurst_50",              # Hurst exp on 50 bars (>0.5 trending)
    "return_skew_30",        # skewness of trailing 30-bar log-returns
    "return_kurt_30",        # excess kurtosis (tail-fatness flag)
]
