"""Absorption detection — heavy volume meets tight price range.

When a 5-min bar prints anomalously high volume but the high/low range is
tight relative to recent bars, it means aggressive flow hit passive
liquidity and was absorbed without moving price. Signal interpretation:

- **Bullish absorption**: bar with strong negative delta (sellers pressing)
  but tight body and minimal price drop → buyers absorbed the offer at
  this level. Often marks short-term lows.
- **Bearish absorption**: bar with strong positive delta but tight body
  and minimal upside → sellers absorbing the bid. Often marks short-term
  highs.

This is bar-level absorption (works off OHLCV + agg balance). For
tick-level absorption (DOM-aware iceberg detection) you need Phase 6
DOM data, which the bridge doesn't expose yet.

Companion to `cumulative_delta.py` which detects absorption from CVD
divergence — this module focuses on the single-bar geometry signature
(volume z-score vs range z-score).
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Deque, Dict, List, Optional

logger = logging.getLogger(__name__)

# How many bars to use for the rolling z-scores. 20 = ~100 min on 5m bars.
LOOKBACK_BARS = 20

# Volume z-score threshold (current bar must be N std devs above the
# rolling mean to count as "heavy volume"). 1.5 catches obvious spikes
# without firing on every above-average bar.
VOLUME_Z_THRESHOLD = 1.5

# Range z-score threshold (current bar's range must be N std devs BELOW
# the rolling mean to count as "tight"). -0.5 = below-median range.
RANGE_Z_THRESHOLD = -0.5

# How many recent absorption events to surface in `to_dict()`.
HISTORY_LIMIT = 10


@dataclass
class Bar:
    timestamp: float
    open: float
    high: float
    low: float
    close: float
    volume: float
    agg_balance: float = 0.0   # buyer-agg minus seller-agg, optional


@dataclass
class AbsorptionEvent:
    timestamp: float
    direction: str         # "bull" | "bear" | "neutral"
    price: float
    volume: float
    volume_z: float
    range_z: float
    agg_balance: float
    note: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "iso": datetime.fromtimestamp(self.timestamp).isoformat() if self.timestamp else None,
            "direction": self.direction,
            "price": round(self.price, 2),
            "volume": round(self.volume, 0),
            "volume_z": round(self.volume_z, 2),
            "range_z": round(self.range_z, 2),
            "agg_balance": round(self.agg_balance, 0),
            "note": self.note,
        }


class AbsorptionTracker:
    """Detects bar-level absorption via volume-z vs range-z combination.

    Usage:
        t = AbsorptionTracker()
        t.update(price, high, low, volume, agg_balance)
        sig = t.signal()  # latest absorption state
    """

    def __init__(self) -> None:
        self._bars: Deque[Bar] = deque(maxlen=LOOKBACK_BARS + 5)
        self._events: Deque[AbsorptionEvent] = deque(maxlen=50)
        self._latest: Optional[AbsorptionEvent] = None

    def update(
        self,
        price: float, high: float, low: float, volume: float,
        agg_balance: float = 0.0,
        timestamp: Optional[float] = None,
    ) -> None:
        """Append a new bar and re-evaluate absorption."""
        if price <= 0 or volume <= 0 or high <= low:
            return
        ts = timestamp if timestamp is not None else datetime.now().timestamp()
        bar = Bar(timestamp=ts, open=price, high=high, low=low,
                  close=price, volume=volume, agg_balance=agg_balance)
        self._bars.append(bar)
        self._evaluate()

    def _evaluate(self) -> None:
        """Compute z-scores over the lookback window and tag absorption events."""
        if len(self._bars) < LOOKBACK_BARS:
            return
        bars = list(self._bars)
        cur = bars[-1]
        prior = bars[:-1][-LOOKBACK_BARS:]   # exclude current bar from baseline

        vols = [b.volume for b in prior]
        ranges = [b.high - b.low for b in prior]
        vol_mean = sum(vols) / len(vols)
        rng_mean = sum(ranges) / len(ranges)
        vol_var = sum((v - vol_mean) ** 2 for v in vols) / len(vols)
        rng_var = sum((r - rng_mean) ** 2 for r in ranges) / len(ranges)
        vol_std = vol_var ** 0.5 if vol_var > 0 else 1.0
        rng_std = rng_var ** 0.5 if rng_var > 0 else 1.0

        cur_range = cur.high - cur.low
        vol_z = (cur.volume - vol_mean) / vol_std if vol_std > 0 else 0.0
        rng_z = (cur_range - rng_mean) / rng_std if rng_std > 0 else 0.0

        # Heavy volume + tight range = absorption.
        if vol_z >= VOLUME_Z_THRESHOLD and rng_z <= RANGE_Z_THRESHOLD:
            # Direction inferred from agg_balance:
            #   strongly negative agg + price held up → bullish absorption
            #   strongly positive agg + price held down → bearish absorption
            if cur.agg_balance < 0:
                direction = "bull"
                note = (f"sellers pressed ({cur.agg_balance:.0f} agg) into tight range "
                        f"(vol z={vol_z:.1f}, range z={rng_z:.1f}) — bid absorbed")
            elif cur.agg_balance > 0:
                direction = "bear"
                note = (f"buyers pressed (+{cur.agg_balance:.0f} agg) into tight range "
                        f"(vol z={vol_z:.1f}, range z={rng_z:.1f}) — offer absorbed")
            else:
                direction = "neutral"
                note = (f"heavy volume in tight range with no net flow "
                        f"(vol z={vol_z:.1f}, range z={rng_z:.1f}) — two-sided absorption")

            event = AbsorptionEvent(
                timestamp=cur.timestamp, direction=direction,
                price=cur.close, volume=cur.volume,
                volume_z=vol_z, range_z=rng_z,
                agg_balance=cur.agg_balance, note=note,
            )
            self._events.append(event)
            self._latest = event
        # Decay latest if it ages out — avoid stale absorption hints.
        elif self._latest and (cur.timestamp - self._latest.timestamp) > 600:  # 10 min
            self._latest = None

    def signal(self) -> Dict[str, Any]:
        """Current absorption state for the signal payload."""
        if self._latest is None:
            return {
                "active": False,
                "direction": "none",
                "score_adjustment": 0.0,
                "reasons": [],
            }
        # Score adjustment: the freshest absorption is worth a +0.10 nudge
        # when it agrees with the trade direction. The scanner uses this to
        # tilt confluence when absorption confirms.
        return {
            "active": True,
            "direction": self._latest.direction,
            "price": round(self._latest.price, 2),
            "volume_z": round(self._latest.volume_z, 2),
            "range_z": round(self._latest.range_z, 2),
            "score_adjustment": 0.10 if self._latest.direction != "neutral" else 0.05,
            "reasons": [f"absorption_{self._latest.direction}"],
            "note": self._latest.note,
        }

    def to_dict(self) -> Dict[str, Any]:
        recent = list(self._events)[-HISTORY_LIMIT:]
        return {
            "bars_seen": len(self._bars),
            "events_total": len(self._events),
            "latest": self._latest.to_dict() if self._latest else None,
            "recent_events": [e.to_dict() for e in recent],
        }


_singletons: Dict[str, AbsorptionTracker] = {}


def get_absorption_tracker(instrument: str = "WDO") -> AbsorptionTracker:
    """Per-instrument singleton so MES and WDO don't contaminate each other."""
    if instrument not in _singletons:
        _singletons[instrument] = AbsorptionTracker()
    return _singletons[instrument]
