"""Mean-reversion signal arm — fires when the trend logic stays quiet.

Trend signals dry up in chop. The current scanner needs (3+ confluence reasons,
ADX up, MACD aligned, EMA slope, etc.) all pointing the same way. On a
range-bound day none of that holds, and signal count drops to zero — the user's
core complaint.

This module fills the gap. When chop is detected (low ADX + price oscillating
around VWAP + Bollinger compression), look for fade setups instead:

1. **VWAP fade**: price stretched ≥ N std-dev from anchored VWAP, RSI extreme,
   low ADX → fade back toward VWAP.
2. **Bollinger touch**: price tags the upper/lower BB with RSI confirmation
   → mean-revert toward the middle band.
3. **POC reclaim**: price punched through VAH or VAL and is now coming back
   into the value area → fade toward POC.

All MR signals are tagged `mode: "mean_reversion"` so analytics can track
win rate independently of trend signals.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


# ── Chop regime detection ──

@dataclass
class ChopRegime:
    """Whether the market is in chop / range / mean-revertable conditions."""
    is_chop: bool
    score: int                          # 0-100, higher = more chop-like
    reasons: List[str] = field(default_factory=list)
    adx: float = 0.0
    bb_width_pct: float = 0.0
    vwap_displacement: float = 0.0      # |price - vwap| / atr — higher = trending

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_chop": self.is_chop,
            "score": self.score,
            "reasons": self.reasons,
            "adx": round(self.adx, 1),
            "bb_width_pct": round(self.bb_width_pct, 2),
            "vwap_displacement": round(self.vwap_displacement, 2),
        }


def detect_chop(
    df: pd.DataFrame,
    adx_threshold: float = 18.0,
    bb_compression_pct: float = 30.0,
    lookback: int = 20,
) -> ChopRegime:
    """Score how chop-like the recent market is. Returns is_chop=True iff
    multiple chop indicators agree.

    Inputs:
        df: OHLCV with DatetimeIndex, ≥ 30 bars recommended.
        adx_threshold: ADX below this = trendless (default 18).
        bb_compression_pct: BB width below this percentile (over lookback) = squeeze.
        lookback: bars used for BB-width percentile + VWAP displacement.

    Reasons added:
        - adx_low: ADX < threshold
        - bb_compressed: current BB width < bb_compression_pct percentile
        - vwap_pinned: price oscillating within 1 ATR of VWAP recently
    """
    if df is None or len(df) < 30:
        return ChopRegime(is_chop=False, score=0)

    try:
        from takata_engine.indicators import adx, atr, bollinger, vwap
    except ImportError:
        return ChopRegime(is_chop=False, score=0)

    reasons: List[str] = []
    score = 0

    # ADX
    try:
        adx_val = float(adx(df)["adx"].iloc[-1])
    except Exception:
        adx_val = 0.0
    if adx_val and adx_val < adx_threshold:
        reasons.append("adx_low")
        score += 35

    # Bollinger width compression — current width vs lookback percentile.
    bb_width_pct = 50.0
    try:
        bb_df = bollinger(df["close"])
        widths = (bb_df["upper"] - bb_df["lower"]).tail(lookback).dropna()
        cur_width = widths.iloc[-1]
        bb_width_pct = (widths < cur_width).sum() / len(widths) * 100
        if bb_width_pct < bb_compression_pct:
            reasons.append("bb_compressed")
            score += 30
    except Exception:
        pass

    # VWAP displacement — how far has price strayed from VWAP recently?
    # Low displacement (< 1 ATR avg) = mean-reverting around VWAP.
    vwap_disp = 0.0
    try:
        vwap_s = vwap(df)
        atr_s = atr(df)
        recent = df.tail(lookback)
        if len(recent) >= 5:
            disp = abs(recent["close"] - vwap_s.tail(lookback)).mean()
            atr_avg = atr_s.tail(lookback).mean()
            vwap_disp = float(disp / atr_avg) if atr_avg > 0 else 0.0
            if vwap_disp < 1.0:
                reasons.append("vwap_pinned")
                score += 35
    except Exception:
        pass

    is_chop = score >= 60  # at least 2 of 3 chop signals must trigger

    return ChopRegime(
        is_chop=is_chop, score=int(score), reasons=reasons,
        adx=adx_val, bb_width_pct=bb_width_pct, vwap_displacement=vwap_disp,
    )


# ── Mean-reversion signals ──

@dataclass
class MeanReversionSignal:
    """A fade setup. Tagged `mode = "mean_reversion"` for downstream analytics."""
    direction: str              # "long" | "short"
    setup_type: str             # "vwap_fade" | "bbands_extreme" | "poc_reclaim"
    price: float
    stop: float
    target: float
    stop_pts: float
    target_pts: float
    strength: float             # 0-1
    reasons: List[str]
    mode: str = "mean_reversion"

    @property
    def rr_ratio(self) -> float:
        if self.stop_pts == 0:
            return 0.0
        return round(self.target_pts / self.stop_pts, 2)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "direction": self.direction,
            "setup_type": self.setup_type,
            "price": round(float(self.price), 2),
            "stop": round(float(self.stop), 2),
            "target": round(float(self.target), 2),
            "stop_pts": round(float(self.stop_pts), 2),
            "target_pts": round(float(self.target_pts), 2),
            "strength": round(float(self.strength), 3),
            "reasons": list(self.reasons),
            "mode": self.mode,
            "rr_ratio": self.rr_ratio,
        }


def evaluate_mean_reversion(
    df: pd.DataFrame,
    indicators: Dict[str, Any],
    vwap_now: Optional[float] = None,
    poc: Optional[float] = None,
    vah: Optional[float] = None,
    val: Optional[float] = None,
    rsi_oversold: float = 30.0,
    rsi_overbought: float = 70.0,
    min_stop_pts: float = 3.0,
    max_stop_pts: float = 8.0,
    rr_target: float = 1.8,
) -> Optional[MeanReversionSignal]:
    """Evaluate the three MR setups and return the strongest one, or None.

    Inputs:
        df: OHLCV DataFrame, ≥ 30 bars.
        indicators: dict from the scanner (close, rsi, adx, atr, vwap, bb_pct).
        vwap_now: anchored-session VWAP last value. Falls back to indicators["vwap"].
        poc, vah, val: latest volume profile values (may be None).
        rsi_*: RSI thresholds for "extreme."
        min_stop_pts / max_stop_pts: stop sizing bounds (instrument-aware caller).
        rr_target: target distance as multiple of stop (1.8 = "give me 1.8R").

    Returns:
        Strongest MeanReversionSignal across the three setup types, or None.
    """
    if df is None or len(df) < 20:
        return None

    close = float(indicators.get("close", df["close"].iloc[-1]))
    rsi = float(indicators.get("rsi", 50.0))
    atr_v = float(indicators.get("atr", 0.0))
    bb_pct = indicators.get("bb_pct")
    if vwap_now is None:
        vwap_now = float(indicators.get("vwap", close))

    candidates: List[MeanReversionSignal] = []

    # Stop scaling: ~1 ATR (tighter than trend's 2 ATR), bounded by caller.
    stop_dist = max(min_stop_pts, min(max_stop_pts, atr_v * 1.0))

    # ── Setup 1: VWAP fade ──
    # Stretched ≥ 1.5 ATR from VWAP with RSI confirmation → fade back to VWAP.
    if atr_v > 0 and vwap_now:
        deviation = (close - vwap_now) / atr_v
        if deviation >= 1.5 and rsi >= rsi_overbought:
            target = vwap_now
            target_dist = abs(close - target)
            if target_dist >= stop_dist * 0.8:
                strength = min(1.0, 0.5 + 0.1 * (deviation - 1.5) + 0.01 * (rsi - rsi_overbought))
                candidates.append(MeanReversionSignal(
                    direction="short", setup_type="vwap_fade",
                    price=close, stop=close + stop_dist, target=target,
                    stop_pts=stop_dist, target_pts=target_dist,
                    strength=strength,
                    reasons=[f"vwap_dev_{deviation:.1f}atr", f"rsi_overbought_{rsi:.0f}"],
                ))
        elif deviation <= -1.5 and rsi <= rsi_oversold:
            target = vwap_now
            target_dist = abs(close - target)
            if target_dist >= stop_dist * 0.8:
                strength = min(1.0, 0.5 + 0.1 * (-deviation - 1.5) + 0.01 * (rsi_oversold - rsi))
                candidates.append(MeanReversionSignal(
                    direction="long", setup_type="vwap_fade",
                    price=close, stop=close - stop_dist, target=target,
                    stop_pts=stop_dist, target_pts=target_dist,
                    strength=strength,
                    reasons=[f"vwap_dev_{-deviation:.1f}atr_below", f"rsi_oversold_{rsi:.0f}"],
                ))

    # ── Setup 2: Bollinger extreme ──
    # bb_pct is %B = (price - lower) / (upper - lower). 1 = at upper band, 0 = at lower.
    if bb_pct is not None:
        try:
            bp = float(bb_pct)
            if bp >= 1.0 and rsi >= rsi_overbought:
                target = close - stop_dist * rr_target
                candidates.append(MeanReversionSignal(
                    direction="short", setup_type="bbands_extreme",
                    price=close, stop=close + stop_dist, target=target,
                    stop_pts=stop_dist, target_pts=stop_dist * rr_target,
                    strength=min(1.0, 0.5 + (bp - 1.0) * 0.5 + (rsi - rsi_overbought) * 0.01),
                    reasons=[f"bb_at_upper_{bp:.2f}", f"rsi_overbought_{rsi:.0f}"],
                ))
            elif bp <= 0.0 and rsi <= rsi_oversold:
                target = close + stop_dist * rr_target
                candidates.append(MeanReversionSignal(
                    direction="long", setup_type="bbands_extreme",
                    price=close, stop=close - stop_dist, target=target,
                    stop_pts=stop_dist, target_pts=stop_dist * rr_target,
                    strength=min(1.0, 0.5 + (-bp) * 0.5 + (rsi_oversold - rsi) * 0.01),
                    reasons=[f"bb_at_lower_{bp:.2f}", f"rsi_oversold_{rsi:.0f}"],
                ))
        except (TypeError, ValueError):
            pass

    # ── Setup 3: POC reclaim from VAH / VAL ──
    # Price punched out of value area and is coming back in → fade to POC.
    # Detect "reclaim" by looking at the recent path: prior bar above VAH,
    # current bar back inside value area (or vice-versa for VAL).
    if poc is not None and vah is not None and val is not None and len(df) >= 3:
        try:
            prev_close = float(df["close"].iloc[-2])
            # Reclaim from above (short setup): prev > VAH, now back inside.
            if prev_close > vah and val < close <= vah:
                target = poc
                target_dist = abs(close - target)
                if target_dist >= stop_dist * 0.8:
                    candidates.append(MeanReversionSignal(
                        direction="short", setup_type="poc_reclaim",
                        price=close, stop=max(close + stop_dist, vah + stop_dist * 0.5),
                        target=target,
                        stop_pts=max(stop_dist, abs(vah + stop_dist * 0.5 - close)),
                        target_pts=target_dist,
                        strength=0.6,
                        reasons=["reclaim_from_above_vah", f"target_poc_{poc:.1f}"],
                    ))
            # Reclaim from below (long setup): prev < VAL, now back inside.
            elif prev_close < val and val <= close < vah:
                target = poc
                target_dist = abs(close - target)
                if target_dist >= stop_dist * 0.8:
                    candidates.append(MeanReversionSignal(
                        direction="long", setup_type="poc_reclaim",
                        price=close, stop=min(close - stop_dist, val - stop_dist * 0.5),
                        target=target,
                        stop_pts=max(stop_dist, abs(close - (val - stop_dist * 0.5))),
                        target_pts=target_dist,
                        strength=0.6,
                        reasons=["reclaim_from_below_val", f"target_poc_{poc:.1f}"],
                    ))
        except (IndexError, ValueError):
            pass

    if not candidates:
        return None
    # Highest strength wins. Tie-breaker: vwap_fade preferred over bbands over poc_reclaim
    # (more conservative setup type wins ties).
    type_priority = {"vwap_fade": 3, "bbands_extreme": 2, "poc_reclaim": 1}
    candidates.sort(
        key=lambda s: (s.strength, type_priority.get(s.setup_type, 0)),
        reverse=True,
    )
    return candidates[0]
