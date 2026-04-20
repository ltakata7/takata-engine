"""WDO Multi-Timeframe Manager — builds timeframes from bridge bar data.

Unlike MES (which uses IBKR), WDO timeframes are built from:
- 5-min bars collected by the bridge from Profit Ultra tick data
- Resampled to 15m/60m for higher-timeframe trend analysis
- Daily from current session open/high/low/close
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import os

import requests

logger = logging.getLogger(__name__)

WDO_BRIDGE = os.getenv("WDO_BRIDGE_URL", "http://127.0.0.1:8002")

BIAS_LABELS = {
    "long_only": "LONG BIAS",
    "short_only": "SHORT BIAS",
    "long_preferred": "LEAN LONG",
    "short_preferred": "LEAN SHORT",
    "neutral": "NEUTRAL",
}


class WDOMultiTimeframe:
    """Multi-timeframe analysis for WDO using Profit Ultra bridge data."""

    def __init__(self, bridge_url: str = WDO_BRIDGE):
        self.bridge_url = bridge_url
        self.bars_5m: pd.DataFrame = pd.DataFrame()
        self.bars_15m: pd.DataFrame = pd.DataFrame()
        self.bars_60m: pd.DataFrame = pd.DataFrame()
        self._last_fetch: float = 0
        self._bias: str = "neutral"
        self._trends: Dict[str, str] = {}

    def fetch(self, timeout: float = 3.0) -> bool:
        """Fetch 5m bars from bridge, resample to 15m/60m."""
        try:
            r = requests.get(f"{self.bridge_url}/bars/WDOFUT/ohlcv", timeout=timeout)
            data = r.json()
        except Exception as e:
            logger.debug("WDO MTF fetch failed: %s", e)
            return False

        if not data.get("timestamp") or len(data["timestamp"]) < 5:
            return False

        df = pd.DataFrame({
            "open": data["open"],
            "high": data["high"],
            "low": data["low"],
            "close": data["close"],
            "volume": data.get("volume", [0] * len(data["open"])),
        }, index=pd.to_datetime(data["timestamp"]))

        self.bars_5m = df
        self._last_fetch = time.time()

        # Resample to 15m and 60m
        if len(df) >= 3:
            self.bars_15m = self._resample(df, "15T")
        if len(df) >= 12:
            self.bars_60m = self._resample(df, "60T")

        self._compute_bias()
        self._compute_trends()
        return True

    def _resample(self, df: pd.DataFrame, freq: str) -> pd.DataFrame:
        """Resample OHLCV to lower frequency."""
        resampled = df.resample(freq).agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }).dropna()
        return resampled

    def needs_fetch(self, interval: int = 60) -> bool:
        """Check if we should fetch new data."""
        return time.time() - self._last_fetch > interval

    # ── Bias from 60m + 5m trends ──

    def _compute_bias(self) -> None:
        """Directional bias from 60m (if available) + 5m bars."""
        trend_60 = self._simple_trend(self.bars_60m) if len(self.bars_60m) >= 5 else "flat"
        trend_5 = self._simple_trend(self.bars_5m) if len(self.bars_5m) >= 10 else "flat"

        if "up" in trend_60 and "up" in trend_5:
            self._bias = "long_only"
        elif "down" in trend_60 and "down" in trend_5:
            self._bias = "short_only"
        elif "up" in trend_60:
            self._bias = "long_preferred"
        elif "down" in trend_60:
            self._bias = "short_preferred"
        elif "up" in trend_5:
            self._bias = "long_preferred"
        elif "down" in trend_5:
            self._bias = "short_preferred"
        else:
            self._bias = "neutral"

    def _simple_trend(self, df: pd.DataFrame) -> str:
        """Quick EMA-based trend classification."""
        if df.empty or len(df) < 5:
            return "flat"
        close = df["close"]
        ema_fast = close.ewm(span=8, adjust=False).mean()
        ema_slow = close.ewm(span=21, adjust=False).mean()

        # Last values
        f = float(ema_fast.iloc[-1])
        s = float(ema_slow.iloc[-1])
        p = float(close.iloc[-1])

        if p > f > s:
            return "strong_up"
        elif f > s:
            return "up"
        elif p < f < s:
            return "strong_down"
        elif f < s:
            return "down"
        return "flat"

    def _compute_trends(self) -> None:
        """Compute trend for each available timeframe."""
        self._trends = {}
        if len(self.bars_5m) >= 10:
            self._trends["5m"] = self._simple_trend(self.bars_5m)
        if len(self.bars_15m) >= 5:
            self._trends["15m"] = self._simple_trend(self.bars_15m)
        if len(self.bars_60m) >= 3:
            self._trends["60m"] = self._simple_trend(self.bars_60m)

    @property
    def bias(self) -> str:
        return self._bias

    @property
    def bias_label(self) -> str:
        return BIAS_LABELS.get(self._bias, "NEUTRAL")

    @property
    def trends(self) -> Dict[str, str]:
        return self._trends

    # ── Key levels from bars ──

    @property
    def levels(self) -> Dict[str, Any]:
        """Compute support/resistance from bar data."""
        result = {
            "session_high": 0, "session_low": 0, "session_open": 0,
            "vwap_zone": "neutral",
            "bar_count_5m": len(self.bars_5m),
            "bar_count_15m": len(self.bars_15m),
            "bar_count_60m": len(self.bars_60m),
        }

        if not self.bars_5m.empty:
            result["session_high"] = float(self.bars_5m["high"].max())
            result["session_low"] = float(self.bars_5m["low"].min())
            result["session_open"] = float(self.bars_5m["open"].iloc[0])

            # Pivot from session data
            h = result["session_high"]
            l = result["session_low"]
            c = float(self.bars_5m["close"].iloc[-1])
            pivot = (h + l + c) / 3
            result["pivot"] = round(pivot, 1)
            result["r1"] = round(2 * pivot - l, 1)
            result["s1"] = round(2 * pivot - h, 1)
            result["r2"] = round(pivot + (h - l), 1)
            result["s2"] = round(pivot - (h - l), 1)

        return result

    # ── Intraday Volatility from real bars ──

    def intraday_vol(self) -> Dict[str, float]:
        """Compute intraday volatility metrics from actual 5-min candles.

        Returns metrics based on how the market is ACTUALLY behaving today,
        not a daily ATR proxy. Used for stop/target sizing.
        """
        df = self.bars_5m
        if df.empty or len(df) < 3:
            return {"avg_range": 5.0, "avg_body": 3.0, "session_range": 40.0,
                    "recent_range": 5.0, "vol_expanding": False, "suggested_stop": 7.0}

        # Per-bar range (high - low)
        bar_ranges = df["high"] - df["low"]
        # Per-bar body (|close - open|)
        bar_bodies = (df["close"] - df["open"]).abs()

        # All bars average
        avg_range = float(bar_ranges.mean())
        avg_body = float(bar_bodies.mean())

        # Recent bars (last 10) — most relevant for current conditions
        recent = min(10, len(df))
        recent_range = float(bar_ranges.tail(recent).mean())
        recent_body = float(bar_bodies.tail(recent).mean())

        # Session range (today's high - low)
        session_range = float(df["high"].max() - df["low"].min())

        # Volatility expanding or contracting?
        # Compare recent 5 bars vs previous 10
        if len(df) >= 15:
            early_range = float(bar_ranges.iloc[-15:-5].mean())
            late_range = float(bar_ranges.tail(5).mean())
            vol_expanding = late_range > early_range * 1.2
            vol_contracting = late_range < early_range * 0.7
        else:
            vol_expanding = False
            vol_contracting = False

        # Max single bar range (for extreme move detection)
        max_bar_range = float(bar_ranges.max())

        # Suggested stop: 1.5x recent avg range, capped 5-12
        raw_stop = recent_range * 1.5
        # If vol is expanding, widen slightly
        if vol_expanding:
            raw_stop *= 1.2
        # If vol is contracting, tighten
        if vol_contracting:
            raw_stop *= 0.85
        # Round to tick and cap
        suggested_stop = round(max(5.0, min(12.0, raw_stop)) * 2) / 2  # tick-align 0.5

        # Suggested target multiplier based on trending vs choppy
        # Big bodies relative to range = trending = wider targets
        body_ratio = avg_body / avg_range if avg_range > 0 else 0.5
        if body_ratio > 0.6:
            target_mult = 2.5  # trending — bodies > 60% of range
        elif body_ratio > 0.4:
            target_mult = 2.0  # normal
        else:
            target_mult = 1.5  # choppy — small bodies, long wicks

        return {
            "avg_range": round(avg_range, 1),
            "avg_body": round(avg_body, 1),
            "recent_range": round(recent_range, 1),
            "recent_body": round(recent_body, 1),
            "session_range": round(session_range, 1),
            "max_bar_range": round(max_bar_range, 1),
            "bar_count": len(df),
            "body_ratio": round(body_ratio, 2),
            "vol_expanding": vol_expanding,
            "vol_contracting": vol_contracting if len(df) >= 15 else False,
            "suggested_stop": suggested_stop,
            "target_mult": target_mult,
            "market_type": "trending" if body_ratio > 0.6 else "normal" if body_ratio > 0.4 else "choppy",
        }

    # ── Serialization ──

    def to_dict(self) -> Dict[str, Any]:
        return {
            "bias": self.bias,
            "bias_label": self.bias_label,
            "trends": self.trends,
            "levels": self.levels,
            "intraday_vol": self.intraday_vol(),
            "bars_loaded": {
                "5m": len(self.bars_5m),
                "15m": len(self.bars_15m),
                "60m": len(self.bars_60m),
            },
            "last_fetch": self._last_fetch,
        }
