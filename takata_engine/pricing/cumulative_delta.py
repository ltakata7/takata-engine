"""Cumulative Delta tracker — running aggressor balance with divergence detection.

Cumulative delta is the session-long running sum of (buyer-aggressive volume −
seller-aggressive volume). It's the cleanest read on who's pressing — bulls
or bears — independent of price action.

**Why cumulative delta matters for WDO 3-5min trading:**
- Price/delta divergence = high-probability reversal signal
  (price makes new high but delta doesn't = no conviction = fade)
- Delta trending with price = trend confirmation
- Large delta shift without price move = absorption (institutional defense)
- Session peak/trough delta = structural level (often acts as magnet)

Signal types generated:
- ``delta_div_bear``       — price makes new high but delta rolled over
- ``delta_div_bull``       — price makes new low but delta rolled over up
- ``delta_confirm_bull``   — price up + delta up = clean trend
- ``delta_confirm_bear``   — price down + delta down = clean trend
- ``delta_absorption_bull`` — strong negative delta with no price drop = bid absorbing
- ``delta_absorption_bear`` — strong positive delta with no price rise = offer absorbing
- ``delta_flip_bull``      — cumulative delta flipped from negative to positive
- ``delta_flip_bear``      — cumulative delta flipped from positive to negative
- ``delta_extreme_high``   — delta at session high, trending = buyers in control
- ``delta_extreme_low``    — delta at session low, trending = sellers in control

Call ``update()`` every scan cycle with price + agg_balance (net aggression
delta from the bridge tape data).
Call ``signal()`` to get the current delta signal dict.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Deque, Dict, List, Optional, Tuple

import pytz

logger = logging.getLogger(__name__)

BRT = pytz.timezone("America/Sao_Paulo")

SESSION_OPEN_H = 9
SESSION_OPEN_M = 0

# How far back to look for swing highs/lows on price (in ticks)
SWING_LOOKBACK = 30

# How many recent (price, delta) samples to keep for divergence analysis
DIVERGENCE_WINDOW = 60

# Minimum price move to count as a "new high/low" (WDO points)
NEW_EXTREME_THRESHOLD = 1.0

# Minimum delta swing to confirm direction (contracts)
MIN_DELTA_SWING = 100.0


@dataclass
class DeltaState:
    """Cumulative delta state — resets each day."""
    date: str = ""
    cum_delta: float = 0.0
    session_high_delta: float = 0.0
    session_low_delta: float = 0.0
    session_high_price: float = 0.0
    session_low_price: float = float("inf")
    # Rolling history of (timestamp, price, cum_delta) samples
    history: Deque[Tuple[float, float, float]] = field(default_factory=lambda: deque(maxlen=DIVERGENCE_WINDOW))
    # Sign of cum_delta on the previous tick (for flip detection)
    prev_sign: int = 0  # -1, 0, +1

    def reset(self, today: str) -> None:
        self.date = today
        self.cum_delta = 0.0
        self.session_high_delta = 0.0
        self.session_low_delta = 0.0
        self.session_high_price = 0.0
        self.session_low_price = float("inf")
        self.history = deque(maxlen=DIVERGENCE_WINDOW)
        self.prev_sign = 0


class CumulativeDeltaTracker:
    """Tracks running aggressor balance and generates divergence signals."""

    def __init__(self) -> None:
        self._state = DeltaState()

    def update(self, price: float, agg_balance_delta: float) -> None:
        """Update cumulative delta.

        Parameters
        ----------
        price : float
            Current trade price.
        agg_balance_delta : float
            Net aggression volume since last update. Positive = more
            buyer aggression, negative = more seller aggression. The bridge
            typically publishes this as ``agg_balance`` — we take the diff
            from the previous tick.
        """
        if price <= 0:
            return

        now = datetime.now(BRT)
        today_str = now.strftime("%Y-%m-%d")

        if self._state.date != today_str:
            self._state.reset(today_str)

        mins_since_open = (now.hour - SESSION_OPEN_H) * 60 + (now.minute - SESSION_OPEN_M)
        if mins_since_open < 0:
            return

        # Accumulate
        self._state.cum_delta += agg_balance_delta

        # Track session extremes
        if self._state.cum_delta > self._state.session_high_delta:
            self._state.session_high_delta = self._state.cum_delta
        if self._state.cum_delta < self._state.session_low_delta:
            self._state.session_low_delta = self._state.cum_delta
        if price > self._state.session_high_price:
            self._state.session_high_price = price
        if self._state.session_low_price == float("inf") or price < self._state.session_low_price:
            self._state.session_low_price = price

        # Record sample
        ts = now.timestamp()
        self._state.history.append((ts, price, self._state.cum_delta))

    def signal(self, current_price: float) -> Dict[str, Any]:
        """Generate cumulative delta signal dict."""
        result: Dict[str, Any] = {
            "cum_delta": round(self._state.cum_delta, 0),
            "session_high_delta": round(self._state.session_high_delta, 0),
            "session_low_delta": round(self._state.session_low_delta, 0),
            "signal": "neutral",
            "score_adjustment": 0.0,
            "reasons": [],
        }
        reasons: List[str] = result["reasons"]
        adj = 0.0

        if len(self._state.history) < 10:
            return result

        history = list(self._state.history)
        # Current price/delta, and a reference from SWING_LOOKBACK ticks ago
        _, cur_price, cur_delta = history[-1]
        lookback_idx = max(0, len(history) - SWING_LOOKBACK)
        _, ref_price, ref_delta = history[lookback_idx]

        delta_change = cur_delta - ref_delta
        price_change = cur_price - ref_price

        result["delta_change"] = round(delta_change, 0)
        result["price_change"] = round(price_change, 1)

        # 1. Divergence — price new extreme but delta doesn't confirm
        # Bearish divergence: price near session high, delta rolled over
        if cur_price >= self._state.session_high_price - NEW_EXTREME_THRESHOLD:
            if cur_delta < self._state.session_high_delta - MIN_DELTA_SWING:
                reasons.append("delta_div_bear")
                adj += 0.12
                result["signal"] = "bearish_divergence"

        # Bullish divergence: price near session low, delta pushed higher
        if cur_price <= self._state.session_low_price + NEW_EXTREME_THRESHOLD:
            if cur_delta > self._state.session_low_delta + MIN_DELTA_SWING:
                reasons.append("delta_div_bull")
                adj += 0.12
                result["signal"] = "bullish_divergence"

        # 2. Confirmation — price and delta moving same direction recently
        if abs(price_change) >= 2.0 and abs(delta_change) >= MIN_DELTA_SWING:
            if price_change > 0 and delta_change > 0:
                reasons.append("delta_confirm_bull")
                adj += 0.07
            elif price_change < 0 and delta_change < 0:
                reasons.append("delta_confirm_bear")
                adj += 0.07

        # 3. Absorption — strong delta but no price follow-through
        # Big positive delta push but price flat = offer absorbing = bearish
        if delta_change > MIN_DELTA_SWING * 2 and abs(price_change) < 1.0:
            reasons.append("delta_absorption_bear")
            adj += 0.09
            result["signal"] = "absorption_bear"
        # Big negative delta but price flat = bid absorbing = bullish
        elif delta_change < -MIN_DELTA_SWING * 2 and abs(price_change) < 1.0:
            reasons.append("delta_absorption_bull")
            adj += 0.09
            result["signal"] = "absorption_bull"

        # 4. Delta flip — cumulative crossed zero (regime change)
        cur_sign = 1 if cur_delta > MIN_DELTA_SWING else (-1 if cur_delta < -MIN_DELTA_SWING else 0)
        if cur_sign != 0 and self._state.prev_sign != 0 and cur_sign != self._state.prev_sign:
            if cur_sign > 0:
                reasons.append("delta_flip_bull")
                adj += 0.08
            else:
                reasons.append("delta_flip_bear")
                adj += 0.08
        if cur_sign != 0:
            self._state.prev_sign = cur_sign

        # 5. Extreme — at session high/low delta and still trending
        if cur_delta == self._state.session_high_delta and delta_change > 0:
            reasons.append("delta_extreme_high")
            adj += 0.05
        elif cur_delta == self._state.session_low_delta and delta_change < 0:
            reasons.append("delta_extreme_low")
            adj += 0.05

        result["score_adjustment"] = round(adj, 3)
        return result

    def to_dict(self) -> Dict[str, Any]:
        """Dump current state for API/frontend."""
        return {
            "date": self._state.date,
            "cum_delta": round(self._state.cum_delta, 0),
            "session_high_delta": round(self._state.session_high_delta, 0),
            "session_low_delta": round(self._state.session_low_delta, 0),
            "session_high_price": round(self._state.session_high_price, 1),
            "session_low_price": round(self._state.session_low_price, 1) if self._state.session_low_price != float("inf") else 0,
            "history_depth": len(self._state.history),
        }


# Per-instrument singleton — MES (from IBKR ticks) and WDO (from Profit
# bridge agg_balance) maintain independent CVD state so they don't contaminate
# each other.
_delta_tracker_instances: Dict[str, CumulativeDeltaTracker] = {}


def get_delta_tracker(instrument: str = "WDO") -> CumulativeDeltaTracker:
    """Per-instrument cumulative delta tracker."""
    if instrument not in _delta_tracker_instances:
        _delta_tracker_instances[instrument] = CumulativeDeltaTracker()
    return _delta_tracker_instances[instrument]
