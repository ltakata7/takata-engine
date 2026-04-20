"""Opening Range Breakout (ORB) tracker for WDO fast-timeframe trading.

Tracks the opening range at two durations (5min and 15min from 09:00 BRT)
and fires signals when price breaks out, retraces to retest, or fails to break.

**Why ORB matters for WDO:**
- Brazilian open (09:00) is high-volume, sets the day's first structural levels
- Most prop desks watch the 15-min ORB as the first committable directional signal
- Failed breakouts (break + reclaim) are high-probability reversal setups
- 3-5 min day traders use ORB as their primary anchor until VWAP develops

Signal types generated:
- ``orb_break_above_15`` — clean break above 15-min ORH with volume
- ``orb_break_below_15`` — clean break below 15-min ORL with volume
- ``orb_break_above_5``  — 5-min ORH break (early signal, lower conviction)
- ``orb_break_below_5``  — 5-min ORL break
- ``orb_retest_long``    — price retesting ORH from above = bullish continuation
- ``orb_retest_short``   — price retesting ORL from below = bearish continuation
- ``orb_failed_break_long``  — broke ORL but reclaimed = short-trap, bullish
- ``orb_failed_break_short`` — broke ORH but rejected = long-trap, bearish

Call ``update()`` every scan cycle with price + current volume.
Call ``signal()`` to get the current ORB signal dict.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Deque, Dict, List, Optional

import pytz

logger = logging.getLogger(__name__)

BRT = pytz.timezone("America/Sao_Paulo")

# Session start (BRT). WDO electronic session opens 09:00.
SESSION_OPEN_H = 9
SESSION_OPEN_M = 0

# Track two range durations: 5-min (scalper) and 15-min (swing)
ORB_DURATIONS_MIN = [5, 15]

# Minimum break distance to avoid noise (WDO ticks every 0.5 pt)
MIN_BREAK_POINTS = 1.0

# Retest tolerance — how close to the boundary to count as a retest
RETEST_TOLERANCE = 1.5


@dataclass
class ORBRange:
    """Opening range high/low for a given duration."""
    duration_min: int
    high: float = 0.0
    low: float = float("inf")
    start_price: float = 0.0
    end_price: float = 0.0
    volume_in_range: float = 0.0
    tick_count: int = 0
    locked: bool = False              # True once the duration has elapsed
    broke_above: bool = False         # Has price ever broken above high?
    broke_below: bool = False         # Has price ever broken below low?
    first_break: Optional[str] = None # "above" or "below", whichever came first


@dataclass
class ORBState:
    """Full ORB state — resets each day."""
    date: str = ""
    ranges: Dict[int, ORBRange] = field(default_factory=dict)
    # Recent prices for retest detection (last 60 ticks)
    recent_prices: Deque[float] = field(default_factory=lambda: deque(maxlen=60))

    def reset(self, today: str) -> None:
        self.date = today
        self.ranges = {d: ORBRange(duration_min=d) for d in ORB_DURATIONS_MIN}
        self.recent_prices = deque(maxlen=60)


class ORBTracker:
    """Tracks opening range breakouts for WDO intraday trading.

    Feeds signals into the scanner alongside PTAX, VXBR, DI dynamics.
    """

    def __init__(self) -> None:
        self._state = ORBState()

    def update(self, price: float, volume_delta: float = 0.0) -> None:
        """Update ORB state. Call every scan cycle (~2s).

        Parameters
        ----------
        price : float
            Current trade price.
        volume_delta : float
            Volume since last update (from bridge tape).
        """
        if price <= 0:
            return

        now = datetime.now(BRT)
        today_str = now.strftime("%Y-%m-%d")

        # Day reset
        if self._state.date != today_str:
            self._state.reset(today_str)

        # Only accumulate during/after session open
        mins_since_open = (now.hour - SESSION_OPEN_H) * 60 + (now.minute - SESSION_OPEN_M)
        if mins_since_open < 0:
            return  # pre-market

        self._state.recent_prices.append(price)

        for duration, rng in self._state.ranges.items():
            if mins_since_open <= duration:
                # Still inside the range-formation window
                if rng.tick_count == 0:
                    rng.start_price = price
                rng.high = max(rng.high, price)
                rng.low = min(rng.low, price) if rng.low != float("inf") else price
                rng.volume_in_range += max(volume_delta, 0)
                rng.tick_count += 1
                rng.end_price = price
            else:
                # Range is locked — track breaks
                if not rng.locked:
                    rng.locked = True
                    logger.debug(
                        "ORB-%dm locked: %.1f / %.1f (range %.1f pts)",
                        duration, rng.low, rng.high, rng.high - rng.low,
                    )

                if price > rng.high + MIN_BREAK_POINTS:
                    if not rng.broke_above:
                        rng.broke_above = True
                        if rng.first_break is None:
                            rng.first_break = "above"
                if price < rng.low - MIN_BREAK_POINTS:
                    if not rng.broke_below:
                        rng.broke_below = True
                        if rng.first_break is None:
                            rng.first_break = "below"

    def signal(self, current_price: float) -> Dict[str, Any]:
        """Generate ORB signal for the WDO scanner.

        Returns dict with ranges, break state, and reasons list.
        """
        result: Dict[str, Any] = {
            "ranges": {},
            "signal": "neutral",
            "score_adjustment": 0.0,
            "reasons": [],
        }
        reasons: List[str] = result["reasons"]
        adj = 0.0

        for duration, rng in self._state.ranges.items():
            if rng.tick_count == 0:
                continue
            range_pts = rng.high - rng.low if rng.low != float("inf") else 0
            result["ranges"][duration] = {
                "high": round(rng.high, 1),
                "low": round(rng.low, 1) if rng.low != float("inf") else 0.0,
                "range_pts": round(range_pts, 1),
                "locked": rng.locked,
                "broke_above": rng.broke_above,
                "broke_below": rng.broke_below,
                "first_break": rng.first_break,
            }

            if not rng.locked:
                continue

            # Breakout signals — only fire at the moment of break or on retest
            price_above_high = current_price > rng.high + MIN_BREAK_POINTS
            price_below_low = current_price < rng.low - MIN_BREAK_POINTS
            within_retest_above = abs(current_price - rng.high) <= RETEST_TOLERANCE
            within_retest_below = abs(current_price - rng.low) <= RETEST_TOLERANCE

            # Use 15m ORB as the primary; 5m is supplementary
            weight = 0.10 if duration == 15 else 0.05

            # 1. Fresh breakout (first few ticks after break)
            if price_above_high and rng.first_break == "above":
                reasons.append(f"orb_break_above_{duration}")
                adj += weight
                result["signal"] = "bullish_breakout"
            elif price_below_low and rng.first_break == "below":
                reasons.append(f"orb_break_below_{duration}")
                adj += weight
                result["signal"] = "bearish_breakdown"

            # 2. Retest — already broke and now retesting the boundary
            if rng.broke_above and within_retest_above and not price_below_low:
                reasons.append(f"orb_retest_long_{duration}")
                adj += weight * 0.8  # retest is higher-conviction than initial break
            if rng.broke_below and within_retest_below and not price_above_high:
                reasons.append(f"orb_retest_short_{duration}")
                adj += weight * 0.8

            # 3. Failed breakout — broke below, reclaimed above high (bullish trap reversal)
            if rng.broke_below and not price_below_low and current_price > rng.high:
                reasons.append(f"orb_failed_break_long_{duration}")
                adj += weight * 1.2  # failed breaks are very high-conviction
                result["signal"] = "bullish_reversal"
            # Failed breakout — broke above, reclaimed below low (bearish trap reversal)
            if rng.broke_above and not price_above_high and current_price < rng.low:
                reasons.append(f"orb_failed_break_short_{duration}")
                adj += weight * 1.2
                result["signal"] = "bearish_reversal"

        result["score_adjustment"] = round(adj, 3)
        return result

    def to_dict(self) -> Dict[str, Any]:
        """Dump current state for API/frontend."""
        return {
            "date": self._state.date,
            "ranges": {
                d: {
                    "high": round(r.high, 1) if r.high else 0,
                    "low": round(r.low, 1) if r.low != float("inf") else 0,
                    "range_pts": round(r.high - r.low, 1) if r.low != float("inf") else 0,
                    "tick_count": r.tick_count,
                    "volume": round(r.volume_in_range, 0),
                    "locked": r.locked,
                    "broke_above": r.broke_above,
                    "broke_below": r.broke_below,
                    "first_break": r.first_break,
                }
                for d, r in self._state.ranges.items()
            },
        }


# Singleton accessor
_orb_tracker_instance: Optional[ORBTracker] = None


def get_orb_tracker() -> ORBTracker:
    """Get the singleton ORB tracker."""
    global _orb_tracker_instance
    if _orb_tracker_instance is None:
        _orb_tracker_instance = ORBTracker()
    return _orb_tracker_instance
