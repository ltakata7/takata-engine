"""Chart pattern recognition on OHLCV data."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd

from takata_core.technical.levels import _find_swing_points


@dataclass
class PatternMatch:
    """A detected chart pattern."""

    name: str
    direction: str  # "bullish", "bearish", "neutral"
    start_idx: int
    end_idx: int
    confidence: float  # 0.0–1.0
    description: str = ""


def detect_double_top(df: pd.DataFrame, tolerance: float = 0.02) -> List[PatternMatch]:
    """Detect double top patterns (bearish reversal)."""
    highs, _ = _find_swing_points(df["high"], order=5)
    patterns = []

    for i in range(len(highs) - 1):
        idx1, p1 = highs[i]
        idx2, p2 = highs[i + 1]

        # Two peaks at similar price, separated by at least 10 bars
        if abs(p1 - p2) / max(p1, p2) < tolerance and idx2 - idx1 > 10:
            # Check for valley between peaks
            valley = df["low"].iloc[idx1:idx2].min()
            neckline_drop = (min(p1, p2) - valley) / min(p1, p2)

            if neckline_drop > 0.02:
                patterns.append(PatternMatch(
                    name="double_top",
                    direction="bearish",
                    start_idx=idx1,
                    end_idx=idx2,
                    confidence=min(0.5 + neckline_drop * 5, 0.9),
                    description=f"Peaks at {p1:.2f} and {p2:.2f}, neckline drop {neckline_drop:.1%}",
                ))
    return patterns


def detect_double_bottom(df: pd.DataFrame, tolerance: float = 0.02) -> List[PatternMatch]:
    """Detect double bottom patterns (bullish reversal)."""
    _, lows = _find_swing_points(df["low"], order=5)
    patterns = []

    for i in range(len(lows) - 1):
        idx1, p1 = lows[i]
        idx2, p2 = lows[i + 1]

        if abs(p1 - p2) / max(p1, p2) < tolerance and idx2 - idx1 > 10:
            peak = df["high"].iloc[idx1:idx2].max()
            neckline_rise = (peak - max(p1, p2)) / max(p1, p2)

            if neckline_rise > 0.02:
                patterns.append(PatternMatch(
                    name="double_bottom",
                    direction="bullish",
                    start_idx=idx1,
                    end_idx=idx2,
                    confidence=min(0.5 + neckline_rise * 5, 0.9),
                    description=f"Troughs at {p1:.2f} and {p2:.2f}, neckline rise {neckline_rise:.1%}",
                ))
    return patterns


def detect_breakout(df: pd.DataFrame, lookback: int = 20) -> List[PatternMatch]:
    """Detect recent price breakouts above/below consolidation range."""
    patterns = []
    if len(df) < lookback + 5:
        return patterns

    # Look at the last `lookback` bars before the most recent 5
    consolidation = df.iloc[-(lookback + 5):-5]
    recent = df.iloc[-5:]

    range_high = consolidation["high"].max()
    range_low = consolidation["low"].min()
    range_pct = (range_high - range_low) / range_low

    # Tight consolidation (< 5% range) followed by breakout
    if range_pct < 0.05:
        last_close = recent["close"].iloc[-1]
        if last_close > range_high:
            patterns.append(PatternMatch(
                name="breakout_up",
                direction="bullish",
                start_idx=len(df) - lookback - 5,
                end_idx=len(df) - 1,
                confidence=min(0.5 + (last_close - range_high) / range_high * 10, 0.9),
                description=f"Broke above {range_high:.2f} after {range_pct:.1%} consolidation",
            ))
        elif last_close < range_low:
            patterns.append(PatternMatch(
                name="breakout_down",
                direction="bearish",
                start_idx=len(df) - lookback - 5,
                end_idx=len(df) - 1,
                confidence=min(0.5 + (range_low - last_close) / range_low * 10, 0.9),
                description=f"Broke below {range_low:.2f} after {range_pct:.1%} consolidation",
            ))

    return patterns


def detect_patterns(df: pd.DataFrame) -> List[PatternMatch]:
    """Run all pattern detectors and return combined results."""
    patterns = []
    patterns.extend(detect_double_top(df))
    patterns.extend(detect_double_bottom(df))
    patterns.extend(detect_breakout(df))
    # Sort by most recent first
    patterns.sort(key=lambda p: p.end_idx, reverse=True)
    return patterns
