"""Multi-Timeframe Data Manager — fetches and maintains bar data across timeframes.

Top-down analysis: daily/60m for bias, 15m for confirmation, 5m for entry precision.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from takata_core.indicators.technical import adx, atr, ema, sma
from takata_core.technical.levels import (
    fibonacci_retracement,
    pivot_points,
    support_resistance,
)
from takata_core.technical.trend import classify_trend

logger = logging.getLogger(__name__)

# IBKR fetch specs per timeframe
TIMEFRAME_SPECS = {
    "daily": {"duration": "20 D", "bar_size": "1 day", "min_bars": 20},
    "60m":   {"duration": "5 D",  "bar_size": "1 hour", "min_bars": 30},
    "15m":   {"duration": "3 D",  "bar_size": "15 mins", "min_bars": 50},
    "5m":    {"duration": "2 D",  "bar_size": "5 mins", "min_bars": 100},
}

FAST_UPDATE_SPECS = {
    "5m":  {"duration": "900 S",  "bar_size": "5 mins"},
    "15m": {"duration": "3600 S", "bar_size": "15 mins"},
}

BIAS_LABELS = {
    "long_only": "LONG BIAS",
    "short_only": "SHORT BIAS",
    "long_preferred": "LEAN LONG",
    "short_preferred": "LEAN SHORT",
    "neutral": "NEUTRAL",
}


class MultiTimeframeManager:
    """Maintains bar data across multiple timeframes for one instrument.

    Parameters
    ----------
    instrument : str
        Instrument symbol (e.g. ``"MES"``).
    """

    def __init__(self, instrument: str = "MES"):
        self.instrument = instrument.upper()
        self.timeframes: Dict[str, pd.DataFrame] = {}
        self._last_full_fetch: float = 0
        self._bias: Optional[str] = None
        self._levels: Optional[Dict[str, Any]] = None

    def fetch_all(self, ib, contract) -> bool:
        """Fetch all timeframes via IBKR connection.

        Parameters
        ----------
        ib : IB
            Connected ib_insync IB instance.
        contract : Contract
            Qualified IBKR contract.

        Returns
        -------
        bool
            True if all fetches succeeded.
        """
        from ib_insync import util

        success = True
        for tf, spec in TIMEFRAME_SPECS.items():
            try:
                bars = ib.reqHistoricalData(
                    contract, endDateTime="",
                    durationStr=spec["duration"],
                    barSizeSetting=spec["bar_size"],
                    whatToShow="TRADES", useRTH=True, formatDate=1,
                )
                if bars:
                    df = util.df(bars)
                    df["timestamp"] = pd.to_datetime(df["date"])
                    df = df.set_index("timestamp")[["open", "high", "low", "close", "volume"]]
                    self.timeframes[tf] = df
                    logger.debug("%s %s: loaded %d bars", self.instrument, tf, len(df))
                else:
                    logger.warning("%s %s: no data returned", self.instrument, tf)
                    success = False
            except Exception as e:
                logger.warning("%s %s fetch failed: %s", self.instrument, tf, e)
                success = False

        self._last_full_fetch = time.time()
        self._compute_bias()
        self._compute_levels()
        return success

    def update_fast(self, ib, contract) -> None:
        """Quick update — only 5m and 15m bars (most recent)."""
        from ib_insync import util

        for tf, spec in FAST_UPDATE_SPECS.items():
            try:
                bars = ib.reqHistoricalData(
                    contract, endDateTime="",
                    durationStr=spec["duration"],
                    barSizeSetting=spec["bar_size"],
                    whatToShow="TRADES", useRTH=True, formatDate=1,
                )
                if bars:
                    new_df = util.df(bars)
                    new_df["timestamp"] = pd.to_datetime(new_df["date"])
                    new_df = new_df.set_index("timestamp")[["open", "high", "low", "close", "volume"]]

                    if tf in self.timeframes:
                        existing = self.timeframes[tf]
                        merged = pd.concat([
                            existing[~existing.index.isin(new_df.index)],
                            new_df
                        ]).sort_index().tail(500)
                        self.timeframes[tf] = merged
                    else:
                        self.timeframes[tf] = new_df
            except Exception as e:
                logger.debug("%s %s fast update failed: %s", self.instrument, tf, e)

    def needs_full_fetch(self, interval: int = 300) -> bool:
        """Check if a full fetch is needed (default every 5 minutes)."""
        return time.time() - self._last_full_fetch > interval or not self.timeframes

    # ── Bias Determination ──

    def _compute_bias(self) -> None:
        """Determine directional bias from daily + 60m timeframes."""
        daily = self.timeframes.get("daily")
        hourly = self.timeframes.get("60m")

        if daily is None or len(daily) < 20:
            self._bias = "neutral"
            return
        if hourly is None or len(hourly) < 20:
            self._bias = "neutral"
            return

        daily_trend = classify_trend(daily)
        hourly_trend = classify_trend(hourly)

        # Both aligned = strong bias
        if "up" in daily_trend and "up" in hourly_trend:
            self._bias = "long_only"
        elif "down" in daily_trend and "down" in hourly_trend:
            self._bias = "short_only"
        # One aligned = moderate bias
        elif "up" in daily_trend:
            self._bias = "long_preferred"
        elif "down" in daily_trend:
            self._bias = "short_preferred"
        elif "up" in hourly_trend:
            self._bias = "long_preferred"
        elif "down" in hourly_trend:
            self._bias = "short_preferred"
        else:
            self._bias = "neutral"

    @property
    def bias(self) -> str:
        return self._bias or "neutral"

    @property
    def bias_label(self) -> str:
        return BIAS_LABELS.get(self._bias or "neutral", "NEUTRAL")

    @property
    def trends(self) -> Dict[str, str]:
        """Return trend classification for each available timeframe."""
        result = {}
        for tf, df in self.timeframes.items():
            if df is not None and len(df) >= 20:
                result[tf] = classify_trend(df)
            else:
                result[tf] = "insufficient_data"
        return result

    # ── Institutional Levels ──

    def _compute_levels(self) -> None:
        """Detect key S/R levels from higher timeframes."""
        self._levels = {
            "daily_support": [], "daily_resistance": [],
            "hourly_support": [], "hourly_resistance": [],
            "fibonacci": {}, "prev_day_high": 0, "prev_day_low": 0,
            "daily_vwap": 0, "pivot": {},
        }

        daily = self.timeframes.get("daily")
        hourly = self.timeframes.get("60m")

        if daily is not None and len(daily) >= 10:
            try:
                sr = support_resistance(daily, n_levels=3, order=5)
                self._levels["daily_support"] = sr["support"]
                self._levels["daily_resistance"] = sr["resistance"]
            except:
                pass

            try:
                high_52 = float(daily["high"].max())
                low_52 = float(daily["low"].min())
                self._levels["fibonacci"] = fibonacci_retracement(high_52, low_52)
            except:
                pass

            if len(daily) >= 2:
                self._levels["prev_day_high"] = float(daily["high"].iloc[-2])
                self._levels["prev_day_low"] = float(daily["low"].iloc[-2])

            try:
                self._levels["pivot"] = pivot_points(daily)
            except:
                pass

        if hourly is not None and len(hourly) >= 10:
            try:
                sr = support_resistance(hourly, n_levels=3, order=3)
                self._levels["hourly_support"] = sr["support"]
                self._levels["hourly_resistance"] = sr["resistance"]
            except:
                pass

    @property
    def levels(self) -> Dict[str, Any]:
        return self._levels or {}

    def nearest_support(self, price: float) -> Optional[float]:
        """Find nearest support level below current price."""
        all_supports = (
            self._levels.get("daily_support", []) +
            self._levels.get("hourly_support", []) +
            [self._levels.get("prev_day_low", 0)]
        )
        below = [s for s in all_supports if s > 0 and s < price]
        return max(below) if below else None

    def nearest_resistance(self, price: float) -> Optional[float]:
        """Find nearest resistance level above current price."""
        all_resistance = (
            self._levels.get("daily_resistance", []) +
            self._levels.get("hourly_resistance", []) +
            [self._levels.get("prev_day_high", 0)]
        )
        above = [r for r in all_resistance if r > 0 and r > price]
        return min(above) if above else None

    # ── Entry Timeframe Selection ──

    def select_entry_timeframe(self, regime: str, adx_value: float = 0) -> Optional[str]:
        """Select the optimal entry timeframe based on volatility regime.

        Returns
        -------
        str or None
            ``"3m"``, ``"5m"``, ``"15m"``, or None (no trade).
        """
        if regime == "crisis":
            return None
        elif regime == "high_vol_choppy" or adx_value > 35:
            return "3m"
        elif regime == "low_vol_trending" and adx_value < 20:
            return "15m"
        else:
            return "5m"

    # ── Serialization ──

    def to_dict(self) -> Dict[str, Any]:
        """Serialize state for API response."""
        return {
            "instrument": self.instrument,
            "bias": self.bias,
            "bias_label": self.bias_label,
            "trends": self.trends,
            "levels": self.levels,
            "timeframes_loaded": {tf: len(df) for tf, df in self.timeframes.items()},
            "last_full_fetch": self._last_full_fetch,
        }
